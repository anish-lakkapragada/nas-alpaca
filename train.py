"""Train one model on the Alpaca train split; select by dev answer loss.

Loss = mean cross-entropy (nats) over supervised answer tokens + answer EOS.
Writes <out>/log.jsonl, config.json, best.safetensors and samples.txt.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import data
from model import GPT, Config, count_parameters

SAMPLE_PROMPTS = [
    "Give three tips for staying healthy.",
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Explain why the sky is blue.",
    "Name three primary colors.",
]


def loss_terms(model, inp, tgt, seg, pos, wt):
    logits = model(inp, pos, seg)
    ce = nn.losses.cross_entropy(logits, tgt, reduction="none")
    return (ce * wt).sum(), wt.sum()


def evaluate(model, packed, batch_rows=8):
    model.eval()
    total, count = 0.0, 0.0
    inp, tgt, seg, pos, wt = packed
    for i in range(0, len(inp), batch_rows):
        s, n = loss_terms(model, *(mx.array(a[i:i + batch_rows]) for a in (inp, tgt, seg, pos, wt)))
        mx.eval(s, n)
        total += s.item()
        count += n.item()
    model.train()
    return total / count


def generate(model, tok, prompt, max_new=120, temperature=0.8, top_k=40, seed=0):
    model.eval()
    ids, _ = data.encode(tok, [{"role": "user", "content": prompt}], generation_prompt=True)
    start = len(ids)
    key = mx.random.key(seed)
    for _ in range(max_new):
        x = mx.array([ids])
        logits = model(x, mx.arange(len(ids))[None], mx.zeros((1, len(ids)), mx.int32))[0, -1]
        logits = logits / temperature
        kth = mx.sort(logits)[-top_k]
        logits = mx.where(logits < kth, -mx.inf, logits)
        key, sub = mx.random.split(key)
        nxt = mx.random.categorical(logits, key=sub).item()
        if nxt == data.EOS:
            break
        ids.append(nxt)
    model.train()
    return tok.decode(ids[start:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=0, help="default 4*d_model")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-rows", type=int, default=8, help="rows of 1024 packed tokens")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spec", default=None, help="architecture spec JSON (arch.py)")
    p.add_argument("--max-params", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    mx.random.seed(a.seed)
    tok = data.tokenizer()
    train_ex = data.encode_split(tok, data.load_split("train"))
    dev_packed = data.pack(data.encode_split(tok, data.load_split("dev")), rng=None)

    spec = None
    if a.spec:
        import arch
        spec = arch.load_spec(a.spec)
        model = arch.SpecGPT(spec, dropout=a.dropout)
        cfg = model.config
        if a.max_params and count_parameters(model) > a.max_params:
            raise SystemExit(f"{count_parameters(model):,} parameters exceed --max-params {a.max_params:,}")
    else:
        cfg = Config(n_layers=a.layers, d_model=a.d_model, n_heads=a.heads,
                     d_ff=a.d_ff or 4 * a.d_model, dropout=a.dropout)
        model = GPT(cfg)
    n_params = count_parameters(model)
    (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "spec": spec, "args": vars(a),
                                                 "parameters": n_params}, indent=2))
    print(f"parameters {n_params:,}", flush=True)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    opt = optim.AdamW(learning_rate=lr_at(0), weight_decay=a.weight_decay, betas=[0.9, 0.95])
    def loss_fn(m, *batch):
        s, n = loss_terms(m, *batch)
        return s / n

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(*batch):
        loss, grads = grad_fn(model, *batch)
        grads, gnorm = optim.clip_grad_norm(grads, a.clip)
        opt.update(model, grads)
        return loss, gnorm

    rng = random.Random(a.seed)
    epoch, buf, cursor = 0, None, 0
    best, log = float("inf"), open(out / "log.jsonl", "w")
    t0, seen_targets, window = time.time(), 0, []
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            dev = evaluate(model, dev_packed)
            rec = {"step": step, "epoch": epoch, "dev_loss": dev, "elapsed": time.time() - t0,
                   "train_loss": float(np.mean(window)) if window else None,
                   "answer_tokens_seen": seen_targets}
            if dev < best:
                best = dev
                model.save_weights(str(out / "best.safetensors"))
                rec["best"] = True
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            window = []
            if step == a.steps:
                break
        if buf is None or cursor + a.batch_rows > len(buf[0]):
            buf, cursor = data.pack(train_ex, rng), 0
            epoch += 1 if step else 0
        batch = [mx.array(x[cursor:cursor + a.batch_rows]) for x in buf]
        cursor += a.batch_rows
        opt.learning_rate = lr_at(step)
        loss, _ = step_fn(*batch)
        mx.eval(loss, state)
        window.append(loss.item())
        seen_targets += int(buf[4][cursor - a.batch_rows:cursor].sum())

    total = time.time() - t0
    model.load_weights(str(out / "best.safetensors"))
    samples = [f"### {q}\n{generate(model, tok, q, seed=i)}\n" for i, q in enumerate(SAMPLE_PROMPTS)]
    (out / "samples.txt").write_text("\n".join(samples))
    summary = {"parameters": n_params, "best_dev_loss": best, "train_seconds": total,
               "steps": a.steps, "epochs_started": epoch + 1}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print("\n".join(samples))


if __name__ == "__main__":
    main()
