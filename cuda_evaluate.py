"""Score CUDA checkpoints by conversation; validation is final-comparison only.

Example: python cuda_evaluate.py baseline/ winner/ --split validation --out final.json
Outputs are write-once; selection during search must use --split dev.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import data
from cuda_model import Config, SpecGPT, count_parameters
from cuda_train import answer_loss, prepare_batch


def load(run_dir: Path, device="cuda", checkpoint="best.pt"):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = SpecGPT(spec=cfg["spec"], config=Config(**cfg["model"])).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


@torch.inference_mode()
def per_conversation(model, examples, precision="bf16", chunk_size=0):
    """Keep conversation sums/counts for paired bootstrap; never average example means."""
    sums, counts = [], []
    device = next(model.parameters()).device
    for ex in examples:
        packed = data.pack([ex], None, seq_len=len(ex[0]) - 1)
        batch = prepare_batch(packed, 0, 1, device)
        value = float(answer_loss(model, batch, precision, chunk_size=chunk_size).cpu())
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite evaluation loss")
        sums.append(value)
        counts.append(batch["n_targets"])
    return np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def paired_bootstrap(s0, n0, s1, n1, bootstrap=2000, seed=0):
    if len(s0) != len(s1) or not np.array_equal(n0, n1):
        raise ValueError("paired evaluation must contain identical conversations and token counts")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    rng = np.random.default_rng(seed)
    diffs = []
    # Bound memory even when users request a large number of resamples.
    for start in range(0, bootstrap, 128):
        idx = rng.integers(0, len(n0), (min(128, bootstrap - start), len(n0)))
        diffs.extend(s1[idx].sum(1) / n1[idx].sum(1) - s0[idx].sum(1) / n0[idx].sum(1))
    return {"point": float(s1.sum() / n1.sum() - s0.sum() / n0.sum()),
            "bootstrap_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "bootstrap_samples": bootstrap, "bootstrap_seed": seed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--split", choices=["dev", "validation"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--projection-chunk-size", type=int, default=0)
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix(".conversations.npz").exists():
        raise SystemExit(f"{a.out} or its conversation archive exists; evaluation outputs are write-once")
    tok = data.tokenizer()
    examples = data.encode_split(tok, data.load_split(a.split))
    results, per, archive = {}, [], {}
    for i, run in enumerate(a.runs):
        model, cfg = load(run, a.device, a.checkpoint)
        sums, counts = per_conversation(model, examples, a.precision, a.projection_chunk_size)
        per.append((sums, counts))
        archive[f"run_{i}_sums"], archive[f"run_{i}_counts"] = sums, counts
        results[str(run)] = {"loss": float(sums.sum() / counts.sum()),
                             "parameters": count_parameters(model), "conversations": len(counts),
                             "answer_tokens": int(counts.sum()), "spec": cfg["spec"].get("name", "gpt"),
                             "training_seed": cfg["args"]["seed"], "checkpoint": a.checkpoint}
        print(json.dumps({"run": str(run), **results[str(run)]}), flush=True)
        del model
    if len(per) == 2:
        results["difference_second_minus_first"] = paired_bootstrap(*per[0], *per[1], a.bootstrap, a.bootstrap_seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement by concurrent final evaluations.
    with a.out.open("x") as handle:
        json.dump({"split": a.split, "precision": a.precision, "results": results}, handle, indent=2)
    np.savez_compressed(a.out.with_suffix(".conversations.npz"), **archive)
    print(json.dumps(results.get("difference_second_minus_first", {})), flush=True)


if __name__ == "__main__":
    main()
