"""Export synthetic MLX reference cases for test_cuda.py, never reading the data.

Run on the original Mac environment:
  ../nas/.venv/bin/python export_mlx_fixture.py --out runs/parity

The default CPU device avoids competing with an ongoing Metal training job.
Each fixture contains every weight, every logit, the answer-weighted loss, and
gradient samples spanning every parameter (all entries for vectors; 16 rows
plus input-token rows for matrices). Dropout is disabled on both backends.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

import arch
from model import count_parameters


def composite_spec():
    """Exercise all eight approved changes, including every activation."""
    return {
        "name": "parity-all-primitives",
        "layers": [
            {"attn": "A0", "ffn": "F0", "loop": 2, "inject": True},
            {"attn": "A1", "ffn": "F1"},
            {"attn": "A0", "ffn": "F2"},
            {"attn": None, "ffn": "F3"},
            {"attn": "A4", "ffn": "F0"},
            {"attn": "A5", "ffn": "F5"},
            {"attn": "A6", "ffn": None},
        ],
        "ffns": {
            "F0": {"act": "gelu"},
            "F1": {"act": "relu2", "transpose_tie": True, "hidden": 768},
            "F2": {"act": "silu", "hidden": 1024},
            "F3": {"act": "gelu_tanh"},
            "F5": {"act": "relu"},
        },
    }


def synthetic_batch():
    rng = np.random.default_rng(20261111)
    inp = rng.integers(6, 32768, size=(2, 12), dtype=np.int32)
    tgt = rng.integers(6, 32768, size=(2, 12), dtype=np.int32)
    seg = np.array([[0] * 6 + [1] * 4 + [-1] * 2,
                    [0] * 4 + [1] * 6 + [-1] * 2], dtype=np.int32)
    pos = np.array([list(range(6)) + list(range(4)) + [0, 0],
                    list(range(4)) + list(range(6)) + [0, 0]], dtype=np.int32)
    wt = np.array([[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=np.float32)
    inp[seg == -1] = 0
    tgt[seg == -1] = 0
    tgt[:, 9] = 2  # Include supervised answer EOS targets.
    return dict(inp=inp, tgt=tgt, seg=seg, pos=pos, wt=wt)


def export_one(out, name, spec, seed, include_no_inject=False):
    mx.random.seed(seed)
    model = arch.SpecGPT(copy.deepcopy(spec), dropout=0.0)
    model.eval()
    mx.eval(model.parameters())
    batch_np = synthetic_batch()
    batch = {k: mx.array(v) for k, v in batch_np.items()}
    arrays = dict(batch_np)
    weights = dict(tree_flatten(model.parameters()))
    arrays.update({"weight::" + k: np.array(v) for k, v in weights.items()})
    rows = {}
    rng = np.random.default_rng(seed)
    for key, weight in weights.items():
        if weight.ndim >= 2:
            selected = rng.choice(weight.shape[0], min(16, weight.shape[0]), replace=False)
            if key == "tok.weight":
                selected = np.concatenate([selected, np.unique(batch_np["inp"])])
            rows[key] = np.unique(np.concatenate([[0, weight.shape[0] - 1], selected])).astype(np.int64)
            arrays["grad_rows::" + key] = rows[key]

    cases = [{"name": name, "spec": copy.deepcopy(spec)}]
    if include_no_inject:
        uninject = copy.deepcopy(spec)
        uninject["layers"][0]["inject"] = False
        cases.append({"name": name + "-no-inject", "spec": uninject})

    for index, case in enumerate(cases):
        model.spec = case["spec"]

        def loss_fn(m):
            logits = m(batch["inp"], batch["pos"], batch["seg"])
            ce = nn.losses.cross_entropy(logits, batch["tgt"], reduction="none")
            return (ce * batch["wt"]).sum() / batch["wt"].sum()

        logits = model(batch["inp"], batch["pos"], batch["seg"])
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(logits, loss, grads)
        arrays[f"case.{index}.logits"] = np.array(logits)
        arrays[f"case.{index}.loss"] = np.array(loss)
        for key, grad in tree_flatten(grads):
            if key in rows:
                grad = grad[mx.array(rows[key].astype(np.int32))]
            arrays[f"case.{index}.grad::{key}"] = np.array(grad)
        print(json.dumps({"case": case["name"], "parameters": count_parameters(model),
                          "loss": loss.item(), "gradient_parameters": len(weights)}), flush=True)
        del logits, loss, grads

    metadata = {"format": 1, "seed": seed, "mlx_version": mx.__version__,
                "config": model.config.to_dict(), "spec": spec, "cases": cases,
                "parameters": count_parameters(model),
                "gradient_sampling": "all vector entries; 16 random matrix rows plus endpoints and input-token embedding rows"}
    arrays["metadata"] = np.array(json.dumps(metadata))
    path = out / (name + ".npz")
    np.savez(path, **arrays)
    del arrays, model, weights, batch
    gc.collect()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": path.name, "sha256": digest, "bytes": path.stat().st_size,
            "cases": [case["name"] for case in cases]}


def export_optimizer_reference(out):
    """Record the installed MLX optimizer's actual defaults and update results."""
    initial = {"matrix": [[0.2, -0.5, 1.2], [0.0, 0.3, -0.7]],
               "vector": [1.0, 0.0, -0.1]}
    params = {key: mx.array(value, dtype=mx.float32) for key, value in initial.items()}
    optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.1, betas=[0.9, 0.95])
    steps = []
    for index, lr in enumerate([1e-5, 2e-5, 1e-3, 1e-4]):
        gradients = {"matrix": [[0.7 * (index + 1), -0.02, 1e-9], [0.0, -0.1, 0.003]],
                     "vector": [0.0, (-1) ** index * 0.03, 1e-7]}
        grads = {key: mx.array(value, dtype=mx.float32) for key, value in gradients.items()}
        clipped, norm = optim.clip_grad_norm(grads, 0.25)
        optimizer.learning_rate = lr
        params = optimizer.apply_gradients(clipped, params)
        mx.eval(params, optimizer.state, norm)
        steps.append({"lr": lr, "gradients": gradients, "gradient_norm": norm.item(),
                      "parameters": {key: np.array(value).tolist() for key, value in params.items()},
                      "first_moments": {key: np.array(optimizer.state[key]["m"]).tolist() for key in params},
                      "second_moments": {key: np.array(optimizer.state[key]["v"]).tolist() for key in params}})
    result = {"mlx_version": mx.__version__, "initial": initial, "betas": [0.9, 0.95],
              "eps": 1e-8, "weight_decay": 0.1, "clip": 0.25,
              "bias_correction": optimizer.bias_correction, "steps": steps}
    path = out / "optimizer.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "cases": ["uncorrected_adamw_four_updates"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/parity"))
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--optimizer-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.optimizer_only)
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    if args.optimizer_only:
        print(json.dumps(export_optimizer_reference(args.out)), flush=True)
        return
    manifest = [export_one(args.out, "baseline", arch.baseline_spec(8), 1701),
                export_one(args.out, "composite", composite_spec(), 1702, include_no_inject=True),
                export_optimizer_reference(args.out)]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixtures": manifest, "device": args.device}), flush=True)


if __name__ == "__main__":
    main()
