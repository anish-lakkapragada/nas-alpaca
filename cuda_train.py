"""Train the fixed Alpaca answer-token objective with PyTorch/CUDA.

Default frame: 10 layers, width 512, 8 heads, FFN 2048, tied vocabulary,
context 1024. data.py remains the authority for encoding, split and packing.
Validation is intentionally unavailable from this training entry point.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
from cuda_model import Config, SpecGPT, count_parameters

FORMAT_VERSION = 1


class AdamWNoBiasCorrection(torch.optim.AdamW):
    """MLX AdamW's uncorrected update, using PyTorch's fused CUDA implementation.

    MLX's original trainer defaults to bias_correction=False. At step t, let
    b1=1-beta1**t and s2=sqrt(1-beta2**t). Passing lr*b1/s2, eps/s2 and
    weight_decay*s2/b1 into corrected AdamW cancels both bias corrections and
    retains exactly the original decoupled decay. Recipe hyperparameters are
    restored after the call, including in state_dict(). All parameters in a
    group must receive gradients together so their internal step counts agree.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, fused=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, fused=fused)
        for group in self.param_groups:
            group["uncorrected_step"] = 0

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("AdamWNoBiasCorrection does not support closures")
        original = []
        for group in self.param_groups:
            if any(p.grad is None for p in group["params"]):
                raise ValueError("uncorrected fused AdamW requires a gradient for every parameter")
            original.append((group["lr"], group["eps"], group["weight_decay"]))
        try:
            for group, (lr, eps, wd) in zip(self.param_groups, original):
                step = group["uncorrected_step"] + 1
                beta1, beta2 = group["betas"]
                b1, s2 = 1 - beta1 ** step, math.sqrt(1 - beta2 ** step)
                group["lr"], group["eps"], group["weight_decay"] = lr * b1 / s2, eps / s2, wd * s2 / b1
            result = super().step()
            for group in self.param_groups:
                group["uncorrected_step"] += 1
            return result
        finally:
            for group, (lr, eps, wd) in zip(self.param_groups, original):
                group["lr"], group["eps"], group["weight_decay"] = lr, eps, wd


def lr_at(step, steps=5000, lr=1e-3, warmup=100, min_lr_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def prepare_batch(packed, start, batch_rows, device):
    """Select supervised locations on CPU, avoiding CUDA nonzero synchronization."""
    inp, tgt, seg, pos, wt = [a[start:start + batch_rows] for a in packed]
    idx = np.flatnonzero(wt.reshape(-1))
    if not len(idx):
        raise ValueError("batch contains no supervised answer tokens")
    if not np.all(wt.reshape(-1)[idx] == 1):
        raise ValueError("the fixed data pipeline must produce binary answer weights")
    # Entire packed datasets fit CPU RAM; copies are small compared with compute.
    def tensor(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.long)
    return {"inp": tensor(inp), "tgt": tensor(tgt), "seg": tensor(seg), "pos": tensor(pos),
            "answer_indices": tensor(idx), "answer_targets": tensor(tgt.reshape(-1)[idx]),
            "n_targets": len(idx)}


def answer_loss(model, batch, precision="bf16", hidden_fn=None, chunk_size=0):
    """Token loss sum; project only positions with answer labels, exactly as masked CE.

    CE and its accumulation use FP32; the network and tied projection use autocast.
    Projection chunks cap transient logits without changing the objective.
    """
    fn = hidden_fn if hidden_fn is not None else model.hidden_states
    with autocast_context(batch["inp"].device, precision):
        hidden = fn(batch["inp"], batch["pos"], batch["seg"])
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, batch["answer_indices"])
        size = chunk_size or len(batch["answer_targets"])
        losses = []
        for start in range(0, len(batch["answer_targets"]), size):
            logits = F.linear(selected[start:start + size], model.tok.weight)
            losses.append(F.cross_entropy(logits.float(), batch["answer_targets"][start:start + size],
                                          reduction="sum"))
        return torch.stack(losses).sum()


def loss_terms(model, inp, tgt, seg, pos, wt, precision="fp32"):
    """Reference masked loss API matching train.py, useful for parity checks."""
    with autocast_context(inp.device, precision):
        logits = model(inp, pos, seg)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             tgt.long().reshape(-1), reduction="none").reshape(tgt.shape)
        return (ce * wt).sum(), wt.sum()


@torch.inference_mode()
def evaluate(model, packed, batch_rows=8, precision="bf16", chunk_size=0):
    previous_mode = model.training
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    try:
        for start in range(0, len(packed[0]), batch_rows):
            batch = prepare_batch(packed, start, batch_rows, device)
            total += answer_loss(model, batch, precision, chunk_size=chunk_size).double()
            count += batch["n_targets"]
        result = float((total / count).cpu())
        if not math.isfinite(result):
            raise FloatingPointError("nonfinite dev loss")
        return result
    finally:
        model.train(previous_mode)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def atomic_checkpoint(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def provenance():
    root = Path(__file__).parent
    hashes = {}
    for rel in ["cuda_train.py", "cuda_model.py", "data.py", "data/train.json", "data/tokenizer.json"]:
        path = root / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--spec", type=Path)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-rows", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-params", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true", help="compile fixed-shape training hidden states")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    p.add_argument("--projection-chunk-size", type=int, default=0)
    p.add_argument("--benchmark-steps", type=int, default=0, help="measure this many steps, then exit")
    p.add_argument("--benchmark-warmup", type=int, default=10)
    p.add_argument("--resume", type=Path, help="resume a trusted local latest.pt/final.pt")
    p.add_argument("--cpu-threads", type=int, default=4)
    return p


def main():
    a = make_parser().parse_args()
    if min(a.steps, a.batch_rows, a.eval_every, a.log_every, a.cpu_threads) < 1:
        raise SystemExit("steps, batch rows, eval/log intervals and CPU threads must be positive")
    if min(a.benchmark_steps, a.benchmark_warmup, a.projection_chunk_size, a.checkpoint_every) < 0:
        raise SystemExit("benchmark, projection and checkpoint settings cannot be negative")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; select --device cpu --precision fp32 only for smoke tests")
    if device.type == "cuda" and a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support BF16")
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        # FP32 runs retain IEEE matmul for numerical parity. BF16 autocast uses tensor cores.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg = Config(n_layers=a.layers, d_model=a.d_model, n_heads=a.heads, d_ff=a.d_ff, dropout=a.dropout)
    spec = json.loads(a.spec.read_text()) if a.spec else None
    model = SpecGPT(spec=spec, config=cfg).to(device)
    n_params = count_parameters(model)
    if a.max_params and n_params > a.max_params:
        raise SystemExit(f"{n_params:,} parameters exceed --max-params {a.max_params:,}")
    if a.resume:
        if a.benchmark_steps:
            raise SystemExit("benchmark runs cannot resume")
        if not a.out.is_dir():
            raise SystemExit("--resume requires the original existing --out directory")
        old = json.loads((a.out / "config.json").read_text())
        recipe_keys = ["steps", "batch_rows", "lr", "warmup", "min_lr_frac", "weight_decay", "clip", "seed", "precision"]
        if old["model"] != model.config.to_dict() or old["spec"] != model.spec or any(old["args"][k] != vars(a)[k] for k in recipe_keys):
            raise SystemExit("resume configuration differs from the saved model/recipe")
    else:
        a.out.mkdir(parents=True, exist_ok=False)

    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                "cpu_threads": a.cpu_threads, "pid": os.getpid()}
    metadata = {"format_version": FORMAT_VERSION, "backend": "pytorch", "model": model.config.to_dict(),
                "spec": model.spec, "parameters": n_params,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "hardware": hardware, "sha256": provenance(),
                "objective": "mean CE over answer tokens and answer EOS; dev selection only",
                "optimizer": {"name": "AdamWNoBiasCorrection", "bias_correction": False,
                              "betas": [0.9, 0.95], "eps": 1e-8,
                              "implementation": "fused AdamW with algebraic bias-correction cancellation"}}
    if not a.resume:
        atomic_json(a.out / "config.json", metadata)
    print(json.dumps({"event": "configured", "parameters": n_params, "hardware": hardware}), flush=True)

    prep_start = time.perf_counter()
    tok = data.tokenizer()
    train_ex = data.encode_split(tok, data.load_split("train"))
    dev_packed = data.pack(data.encode_split(tok, data.load_split("dev")), rng=None)
    if not train_ex:
        raise SystemExit("empty training data")
    opt = AdamWNoBiasCorrection(model.parameters(), lr=lr_at(0, a.steps, a.lr, a.warmup, a.min_lr_frac),
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay,
                               fused=device.type == "cuda")
    rng = random.Random(a.seed)
    epoch, cursor, buf, pack_rng_state = 0, 0, None, None
    start_step, best, seen_targets, previous_elapsed = 0, float("inf"), 0, 0.0
    if a.resume:
        # Full optimizer/RNG checkpoints are trusted files generated by this program.
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step, best, seen_targets = ckpt["step"], ckpt["best_dev_loss"], ckpt["answer_tokens_seen"]
        epoch, cursor, pack_rng_state = ckpt["epoch"], ckpt["cursor"], ckpt["pack_rng_state"]
        rng.setstate(pack_rng_state)
        buf = data.pack(train_ex, rng)
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        previous_elapsed = ckpt["elapsed_seconds"]
    hidden_fn = model.hidden_states
    if a.compile:
        hidden_fn = torch.compile(hidden_fn, dynamic=False, mode=a.compile_mode)
    prep_seconds = time.perf_counter() - prep_start
    model.train()
    synchronize(device)
    t0 = time.perf_counter()
    measured_start = None
    measured_seconds = None
    training_seconds, eval_seconds, checkpoint_seconds, pack_seconds = 0.0, 0.0, 0.0, 0.0
    train_batches, train_tokens = 0, 0
    window_loss, window_count = torch.zeros((), device=device), 0
    interval_start = time.perf_counter()
    interval_steps, interval_tokens = 0, 0
    end_step = a.benchmark_warmup + a.benchmark_steps if a.benchmark_steps else a.steps
    log = (a.out / "log.jsonl").open("a" if a.resume else "w", buffering=1)

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def checkpoint(step, full=False):
        payload = {"format_version": FORMAT_VERSION, "model": model.state_dict(),
                   "step": step, "model_config": model.config.to_dict(), "spec": model.spec,
                   "best_dev_loss": best, "answer_tokens_seen": seen_targets}
        if full:
            payload.update(optimizer=opt.state_dict(), epoch=epoch, cursor=cursor,
                           pack_rng_state=pack_rng_state, torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           elapsed_seconds=previous_elapsed + time.perf_counter() - t0)
        return payload

    try:
        for step in range(start_step, end_step + 1):
            # Benchmark includes one initial dev pass; excludes evaluation from timed training.
            do_eval = step == start_step or (not a.benchmark_steps and (step % a.eval_every == 0 or step == end_step))
            if do_eval:
                synchronize(device)
                ev_start = time.perf_counter()
                dev = evaluate(model, dev_packed, a.batch_rows, a.precision, a.projection_chunk_size)
                synchronize(device)
                ev_time = time.perf_counter() - ev_start
                eval_seconds += ev_time
                rec = {"event": "evaluation", "step": step, "epoch": epoch, "dev_loss": dev,
                       "elapsed": previous_elapsed + time.perf_counter() - t0,
                       "eval_seconds": ev_time, "answer_tokens_seen": seen_targets}
                if dev < best:
                    best = dev
                    ck_start = time.perf_counter()
                    atomic_checkpoint(a.out / "best.pt", checkpoint(step))
                    checkpoint_seconds += time.perf_counter() - ck_start
                    rec["best"] = True
                emit(rec)
                interval_start = time.perf_counter()
                interval_steps, interval_tokens = 0, 0
            if step == end_step:
                break
            if a.benchmark_steps and step == a.benchmark_warmup:
                synchronize(device)
                measured_start = time.perf_counter()
            if buf is None or cursor + a.batch_rows > len(buf[0]):
                pk_start = time.perf_counter()
                pack_rng_state = rng.getstate()
                buf, cursor = data.pack(train_ex, rng), 0
                if len(buf[0]) < a.batch_rows:
                    raise ValueError("packed training split has fewer rows than --batch-rows")
                epoch += 1 if step else 0
                pack_seconds += time.perf_counter() - pk_start
            batch = prepare_batch(buf, cursor, a.batch_rows, device)
            cursor += a.batch_rows
            rate = lr_at(step, a.steps, a.lr, a.warmup, a.min_lr_frac)
            for group in opt.param_groups:
                group["lr"] = rate
            opt.zero_grad(set_to_none=True)
            train_start = time.perf_counter()
            loss_sum = answer_loss(model, batch, a.precision, hidden_fn, a.projection_chunk_size)
            loss = loss_sum / batch["n_targets"]
            loss.backward()
            # Norm check synchronizes once each step and prevents nonfinite optimizer updates.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, error_if_nonfinite=True)
            opt.step()
            window_loss += loss_sum.detach()
            window_count += batch["n_targets"]
            seen_targets += batch["n_targets"]
            train_tokens += batch["n_targets"]
            train_batches += 1
            interval_steps += 1
            interval_tokens += batch["n_targets"]
            # Synchronization here yields real per-step timing and bounds asynchronous errors.
            synchronize(device)
            training_seconds += time.perf_counter() - train_start
            if a.benchmark_steps and step + 1 == end_step:
                measured_seconds = time.perf_counter() - measured_start
            if (step + 1) % a.log_every == 0 or step + 1 == end_step:
                interval = time.perf_counter() - interval_start
                value = float((window_loss / window_count).cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"nonfinite training loss at step {step + 1}")
                emit({"event": "training", "step": step + 1, "epoch": epoch, "lr": rate,
                      "train_loss": value, "grad_norm": float(grad_norm.detach().cpu()),
                      "answer_tokens_seen": seen_targets,
                      "seconds_per_step": interval / max(1, interval_steps),
                      "packed_tokens_per_second": interval_steps * a.batch_rows * data.SEQ_LEN / interval,
                      "answer_tokens_per_second": interval_tokens / interval,
                      "eta_training_seconds": max(0, a.steps - step - 1) * interval / max(1, interval_steps),
                      "elapsed": previous_elapsed + time.perf_counter() - t0})
                window_loss.zero_()
                window_count = 0
                interval_start = time.perf_counter()
                interval_steps, interval_tokens = 0, 0
            if not a.benchmark_steps and a.checkpoint_every and (step + 1) % a.checkpoint_every == 0:
                ck_start = time.perf_counter()
                atomic_checkpoint(a.out / "latest.pt", checkpoint(step + 1, full=True))
                checkpoint_seconds += time.perf_counter() - ck_start

        if not a.benchmark_steps:
            ck_start = time.perf_counter()
            atomic_checkpoint(a.out / "final.pt", checkpoint(end_step, full=True))
            checkpoint_seconds += time.perf_counter() - ck_start
        total = previous_elapsed + time.perf_counter() - t0
        summary = {"format_version": FORMAT_VERSION, "parameters": n_params, "best_dev_loss": best,
                   "steps": end_step, "schedule_steps": a.steps, "epochs_started": epoch + 1,
                   "train_seconds": total, "preparation_seconds": prep_seconds,
                   "training_compute_seconds": training_seconds, "evaluation_seconds": eval_seconds,
                   "checkpoint_seconds": checkpoint_seconds, "packing_seconds": pack_seconds,
                   "answer_tokens_seen": seen_targets, "hardware": hardware,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        if a.benchmark_steps:
            seconds_per_step = measured_seconds / a.benchmark_steps
            expected_evals = 1 + math.ceil(a.steps / a.eval_every)
            summary.update(benchmark=True, warmup_steps=a.benchmark_warmup,
                           measured_steps=a.benchmark_steps, measured_seconds=measured_seconds,
                           measured_seconds_per_step=seconds_per_step,
                           estimated_5000_training_seconds=seconds_per_step * 5000,
                           estimated_schedule_training_seconds=seconds_per_step * a.steps,
                           estimated_schedule_with_dev_seconds=seconds_per_step * a.steps + expected_evals * eval_seconds,
                           estimate_excludes="startup/compilation, epoch repacking variance and checkpoint overhead")
        atomic_json(a.out / "summary.json", summary)
        emit({"event": "summary", **summary})
    finally:
        log.close()


if __name__ == "__main__":
    main()
