"""CUDA-port verification on CPU or NVIDIA, with optional original MLX fixtures.

  python test_cuda.py --device cpu
  python test_cuda.py --device cuda --fixtures runs/parity

All examples are synthetic. The held-out validation data is never read.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cuda_model import Config, SpecGPT, baseline_spec, count_parameters, validate


DEVICE = "cpu"


def tiny_config():
    return Config(n_layers=4, d_model=16, n_heads=4, d_ff=64,
                  vocab_size=41, max_len=16, dropout=0.0)


def tiny_model(spec=None):
    torch.manual_seed(1701)
    return SpecGPT(spec=spec, config=tiny_config()).to(DEVICE).eval()


def packed_batch():
    inp = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0]], device=DEVICE)
    tgt = torch.tensor([[7, 8, 9, 10, 2, 12, 13, 14, 2, 0, 0, 0]], device=DEVICE)
    pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 0, 0]], device=DEVICE)
    seg = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1]], device=DEVICE)
    wt = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32, device=DEVICE)
    return inp, tgt, seg, pos, wt


def weighted_loss(model, batch):
    inp, tgt, seg, pos, wt = batch
    logits = model(inp, pos, seg)
    ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(wt)
    return (ce * wt).sum() / wt.sum()


def test_baseline_parameter_counts():
    for config, expected in [
        (Config(), 48_780_288),
        (Config(n_layers=8, d_model=384, n_heads=6, d_ff=1536), 27_144_960),
    ]:
        model = SpecGPT(config=config)
        assert count_parameters(model) == expected, (config, count_parameters(model))
        # A tied output head must not introduce an additional parameter table.
        assert sum(p.numel() for p in model.parameters()) == expected
        del model
        gc.collect()


def test_primitive_parameter_counts_and_shared_norms():
    cfg = tiny_config()
    baseline = count_parameters(tiny_model())
    attn = 4 * cfg.d_model ** 2 + 2 * cfg.d_model
    ffn = 2 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model
    cases = []
    for kind, reduction in [("attn", attn), ("ffn", ffn)]:
        spec = baseline_spec(cfg.n_layers)
        spec["layers"][2][kind] = spec["layers"][0][kind]
        cases.append((spec, reduction))
        spec = baseline_spec(cfg.n_layers)
        spec["layers"][2][kind] = None
        cases.append((spec, reduction))
    spec = baseline_spec(cfg.n_layers)
    spec["ffns"] = {"F0": {"transpose_tie": True}}
    cases.append((spec, cfg.d_model * cfg.d_ff))
    spec = baseline_spec(cfg.n_layers)
    spec["ffns"] = {"F0": {"hidden": 32}}
    cases.append((spec, 2 * cfg.d_model * (cfg.d_ff - 32)))
    spec = baseline_spec(2)
    for layer in spec["layers"]:
        layer.update(loop=2, inject=True)
    cases.append((spec, 2 * (attn + ffn)))
    for spec, reduction in cases:
        model = tiny_model(spec)
        assert count_parameters(model) == baseline - reduction, spec
        assert len(list(model.named_parameters())) == len({id(p) for p in model.parameters()})
        inp, _, seg, pos, _ = packed_batch()
        assert tuple(model(inp, pos, seg).shape) == (1, 12, cfg.vocab_size)


def test_transpose_tie_math_and_gradient():
    spec = baseline_spec(4)
    spec["ffns"] = {"F0": {"transpose_tie": True}}
    model = tiny_model(spec)
    ffn = model.ffns["F0"]
    x = torch.randn(2, 3, 16, device=DEVICE)
    expected = F.gelu(ffn.up(ffn.ln(x)), approximate="none") @ ffn.up.weight
    actual = ffn(x)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert "down.weight" not in dict(ffn.named_parameters())
    actual.square().sum().backward()
    assert ffn.up.weight.grad is not None
    assert torch.isfinite(ffn.up.weight.grad).all()


def test_packed_equals_individual_loss_and_gradients():
    model = tiny_model()
    other = copy.deepcopy(model)
    batch = packed_batch()
    packed_loss = weighted_loss(model, batch)
    packed_loss.backward()
    total = torch.zeros((), device=DEVICE)
    count = batch[4].sum()
    for start, end in [(0, 5), (5, 9)]:
        part = tuple(x[:, start:end] for x in batch)
        total = total + weighted_loss(other, part) * part[4].sum()
        with torch.no_grad():
            torch.testing.assert_close(model(batch[0], batch[3], batch[2])[:, start:end],
                                       other(part[0], part[3], part[2]), atol=2e-6, rtol=2e-5)
    individual_loss = total / count
    individual_loss.backward()
    torch.testing.assert_close(packed_loss, individual_loss, atol=1e-6, rtol=1e-6)
    for (name, param), (_, reference) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(param.grad, reference.grad, atol=2e-6, rtol=3e-5, msg=name)


def test_no_conversation_or_future_leakage():
    model = tiny_model()
    inp, _, seg, pos, _ = packed_batch()
    with torch.no_grad():
        original = model(inp, pos, seg)
        changed = inp.clone()
        changed[:, :5] = torch.tensor([20, 21, 22, 23, 24], device=DEVICE)
        torch.testing.assert_close(original[:, 5:9], model(changed, pos, seg)[:, 5:9], atol=0, rtol=0)
        changed = inp.clone()
        changed[:, 3:5] = torch.tensor([25, 26], device=DEVICE)
        torch.testing.assert_close(original[:, :3], model(changed, pos, seg)[:, :3], atol=0, rtol=0)

    embeddings = []

    def capture(_module, _args, output):
        output.retain_grad()
        embeddings.append(output)

    hook = model.tok.register_forward_hook(capture)
    hidden = model.hidden_states(inp, pos, seg)
    hook.remove()
    (hidden[:, 5:9] * torch.arange(16, device=DEVICE)).square().sum().backward()
    assert torch.count_nonzero(embeddings[0].grad[:, :5]).item() == 0
    assert torch.count_nonzero(embeddings[0].grad[:, 9:]).item() == 0
    assert torch.count_nonzero(embeddings[0].grad[:, 5:9]).item() > 0


def test_whitelist_rejects_invalid_specs():
    bad = []
    spec = baseline_spec(4); spec["layers"][1]["attn"] = "A0"; bad.append(spec)
    spec = baseline_spec(4); spec["layers"][0]["loop"] = 2; bad.append(spec)
    spec = baseline_spec(4); spec["ffns"] = {"F0": {"hidden": 65}}; bad.append(spec)
    spec = baseline_spec(4); spec["ffns"] = {"F0": {"act": "swiglu"}}; bad.append(spec)
    spec = baseline_spec(4); spec["layers"][0]["extra"] = 1; bad.append(spec)
    spec = baseline_spec(4); spec["layers"][0].update(attn=None, ffn=None); bad.append(spec)
    spec = baseline_spec(4); spec["layers"][0]["inject"] = True; bad.append(spec)
    spec = baseline_spec(4); spec["ffns"] = {"F0": {"bias": True}}; bad.append(spec)
    for spec in bad:
        try:
            validate(spec, config=tiny_config())
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"accepted invalid architecture {spec}")


def test_training_answer_only_loss_and_gradient():
    from cuda_train import answer_loss, loss_terms, prepare_batch
    batch = packed_batch()
    reference = tiny_model()
    target = weighted_loss(reference, batch)
    target.backward()
    packed = tuple(array.detach().cpu().numpy() for array in batch)
    prepared = prepare_batch(packed, 0, 1, DEVICE)
    assert prepared["n_targets"] == int(batch[4].sum().item())
    for chunk_size in [None, 0, 2]:
        model = tiny_model()
        if chunk_size is None:
            total, count = loss_terms(model, *batch, precision="fp32")
            loss = total / count
        else:
            loss = answer_loss(model, prepared, precision="fp32", chunk_size=chunk_size) / prepared["n_targets"]
        loss.backward()
        torch.testing.assert_close(loss, target, atol=1e-6, rtol=1e-6)
        for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(param.grad, ref.grad, atol=2e-6, rtol=3e-5,
                                       msg=f"chunk={chunk_size} {name}")


def test_training_lr_schedule():
    from cuda_train import lr_at
    assert abs(lr_at(0, steps=5000) - 1e-5) < 1e-12
    assert abs(lr_at(99, steps=5000) - 1e-3) < 1e-12
    assert abs(lr_at(100, steps=5000) - 1e-3) < 1e-12
    assert abs(lr_at(5000, steps=5000) - 1e-4) < 1e-12
    assert lr_at(2000, steps=5000) > lr_at(2000, steps=2000)


def check_mlx_fixture(path):
    with np.load(path, allow_pickle=False) as fixture:
        meta = json.loads(str(fixture["metadata"]))
        config = Config(**meta["config"])
        weights = {key[len("weight::"):]: torch.from_numpy(fixture[key].copy())
                   for key in fixture.files if key.startswith("weight::")}
        batch = tuple(torch.from_numpy(fixture[key].copy()).to(DEVICE, dtype=torch.float32 if key == "wt" else torch.long)
                      for key in ["inp", "tgt", "seg", "pos", "wt"])
        for index, case in enumerate(meta["cases"]):
            model = SpecGPT(spec=case["spec"], config=config).to(DEVICE).eval()
            model.load_state_dict(weights, strict=True)
            assert count_parameters(model) == meta["parameters"]
            logits = model(batch[0], batch[3], batch[2])
            expected_logits = torch.from_numpy(fixture[f"case.{index}.logits"].copy()).to(DEVICE)
            torch.testing.assert_close(logits, expected_logits, atol=2e-5, rtol=2e-4,
                                       msg=f"MLX logits: {case['name']}")
            max_logit_error = (logits - expected_logits).abs().max().item()
            loss = weighted_loss(model, batch)
            torch.testing.assert_close(loss, torch.tensor(float(fixture[f"case.{index}.loss"]), device=DEVICE),
                                       atol=3e-6, rtol=2e-6, msg=f"MLX loss: {case['name']}")
            loss.backward()
            max_gradient_error = 0.0
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                actual = param.grad
                if "grad_rows::" + name in fixture:
                    rows = torch.from_numpy(fixture["grad_rows::" + name].copy()).to(DEVICE)
                    actual = actual.index_select(0, rows)
                expected = torch.from_numpy(fixture[f"case.{index}.grad::{name}"].copy()).to(DEVICE)
                torch.testing.assert_close(actual, expected, atol=8e-6, rtol=5e-4,
                                           msg=f"MLX gradient: {case['name']} {name}")
                max_gradient_error = max(max_gradient_error, (actual - expected).abs().max().item())
            print(json.dumps({"test": "mlx_parity", "case": case["name"], "loss": loss.item(),
                              "max_logit_abs_error": max_logit_error,
                              "max_sampled_gradient_abs_error": max_gradient_error}), flush=True)
            del model, logits, loss, expected_logits
            gc.collect()


def check_mlx_optimizer(path):
    from cuda_train import AdamWNoBiasCorrection
    fixture = json.loads(path.read_text())
    assert fixture["bias_correction"] is False
    params = {name: torch.nn.Parameter(torch.tensor(value, device=DEVICE, dtype=torch.float32))
              for name, value in fixture["initial"].items()}
    optimizer = AdamWNoBiasCorrection(list(params.values()), lr=fixture["steps"][0]["lr"],
                                betas=tuple(fixture["betas"]), eps=fixture["eps"],
                                weight_decay=fixture["weight_decay"], fused=DEVICE == "cuda")
    max_parameter_error = 0.0
    for index, step in enumerate(fixture["steps"]):
        for name, param in params.items():
            param.grad = torch.tensor(step["gradients"][name], device=DEVICE, dtype=torch.float32)
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), fixture["clip"])
        torch.testing.assert_close(norm, torch.tensor(step["gradient_norm"], device=DEVICE), atol=1e-7, rtol=1e-6)
        optimizer.param_groups[0]["lr"] = step["lr"]
        optimizer.step()
        for name, param in params.items():
            expected = torch.tensor(step["parameters"][name], device=DEVICE)
            torch.testing.assert_close(param, expected, atol=2e-7, rtol=2e-6,
                                       msg=f"MLX optimizer step {index} parameter {name}")
            max_parameter_error = max(max_parameter_error, (param - expected).abs().max().item())
            state = optimizer.state[param]
            for torch_key, reference_key in [("exp_avg", "first_moments"), ("exp_avg_sq", "second_moments")]:
                expected = torch.tensor(step[reference_key][name], device=DEVICE)
                torch.testing.assert_close(state[torch_key], expected, atol=2e-9, rtol=3e-6,
                                           msg=f"MLX optimizer step {index} {name} {torch_key}")
        # Resume halfway through: the uncorrected-step count and restored
        # recipe hyperparameters must survive serialization along with moments.
        if index == 1:
            saved = copy.deepcopy(optimizer.state_dict())
            params = {name: torch.nn.Parameter(param.detach().clone()) for name, param in params.items()}
            optimizer = AdamWNoBiasCorrection(list(params.values()), lr=step["lr"],
                                              betas=tuple(fixture["betas"]), eps=fixture["eps"],
                                              weight_decay=fixture["weight_decay"], fused=DEVICE == "cuda")
            optimizer.load_state_dict(saved)
            assert optimizer.param_groups[0]["lr"] == step["lr"]
            assert optimizer.param_groups[0]["eps"] == fixture["eps"]
            assert optimizer.param_groups[0]["weight_decay"] == fixture["weight_decay"]
    print(json.dumps({"test": "mlx_optimizer_parity", "updates": len(fixture["steps"]),
                      "resumed_after_update": 2,
                      "max_parameter_abs_error": max_parameter_error}), flush=True)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    DEVICE = args.device
    # Parity checks use full float32 math, not TF32 approximation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    if args.fixtures:
        manifest_path = args.fixtures / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest:
                path = args.fixtures / entry["file"]
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                assert digest == entry["sha256"], f"fixture checksum mismatch: {path}"
        paths = sorted(args.fixtures.glob("*.npz"))
        assert paths, f"no fixtures in {args.fixtures}"
        for path in paths:
            check_mlx_fixture(path)
        optimizer_path = args.fixtures / "optimizer.json"
        if optimizer_path.exists():
            check_mlx_optimizer(optimizer_path)
    print("All CUDA-port checks passed.", flush=True)


if __name__ == "__main__":
    main()
