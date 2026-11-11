"""Score saved checkpoints on a split. The final comparison uses --split validation,
which is run once per final model and never used for selection.

  .venv/bin/python evaluate.py --split validation runs/search/X/  runs/search/Y/ --out result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import arch
import data
from model import GPT, Config, count_parameters
from train import loss_terms


def load(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = arch.SpecGPT(cfg["spec"]) if cfg.get("spec") else GPT(Config(**cfg["model"]))
    model.load_weights(str(run_dir / "best.safetensors"))
    model.eval()
    return model, cfg


def per_conversation(model, examples):
    """Answer-token loss sum and count per conversation (each scored alone)."""
    sums, counts = [], []
    for ex in examples:
        packed = data.pack([ex], None, seq_len=len(ex[0]) - 1)
        s, n = loss_terms(model, *(mx.array(a) for a in packed))
        mx.eval(s, n)
        sums.append(s.item()); counts.append(n.item())
    return np.array(sums), np.array(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--split", choices=["dev", "validation"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; scores are written once")
    tok = data.tokenizer()
    examples = data.encode_split(tok, data.load_split(a.split))
    results, per = {}, {}
    for r in a.runs:
        model, cfg = load(r)
        s, n = per_conversation(model, examples)
        per[str(r)] = (s, n)
        results[str(r)] = {"loss": float(s.sum() / n.sum()), "parameters": count_parameters(model),
                           "conversations": len(n), "answer_tokens": int(n.sum()),
                           "spec": (cfg.get("spec") or {}).get("name", "gpt")}
        print(r, json.dumps(results[str(r)]), flush=True)
    if len(a.runs) == 2:   # paired conversation bootstrap of the loss difference (second - first)
        (s0, n0), (s1, n1) = per[str(a.runs[0])], per[str(a.runs[1])]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(n0), (a.bootstrap, len(n0)))
        diffs = s1[idx].sum(1) / n1[idx].sum(1) - s0[idx].sum(1) / n0[idx].sum(1)
        results["difference_second_minus_first"] = {
            "point": float(s1.sum() / n1.sum() - s0.sum() / n0.sum()),
            "bootstrap_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"split": a.split, "results": results}, indent=2))
    print(json.dumps(results.get("difference_second_minus_first", {})))


if __name__ == "__main__":
    main()
