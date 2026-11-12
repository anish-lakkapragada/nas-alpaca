"""Focused checks for the expanded architecture space, using synthetic data only.

Run ``python test_cuda_model_expanded.py --device cuda --legacy-model PATH`` to
also compare the baseline exactly against the frozen pre-expansion model file.
No training, dev, or test dataset is opened by this suite.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from nas.cuda_model import (Config, SpecGPT, baseline_parameter_count, baseline_spec,
                        count_parameters, spec_parameter_count, validate)


DEVICE = "cpu"


def config(**kwargs):
    values = dict(vocab_size=41, max_len=16, n_layers=4, d_model=16,
                  n_heads=4, d_ff=64, dropout=0.)
    values.update(kwargs)
    return Config(**values)


def make_model(spec=None, cfg=None):
    torch.manual_seed(1701)
    return SpecGPT(spec, cfg or config()).to(DEVICE)


def batch():
    inputs = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]], device=DEVICE)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=DEVICE)
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=DEVICE)
    targets = torch.tensor([[7, 8, 9, 2, 11, 12, 13, 2]], device=DEVICE)
    weights = torch.tensor([[0, 0, 1, 1, 0, 1, 1, 1]], device=DEVICE)
    return inputs, positions, segments, targets, weights


def loss_from_logits(logits, targets, weights):
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    return (ce * weights.flatten()).sum() / weights.sum()


def assert_gradients_equal(a, b, atol=0, rtol=0):
    left, right = dict(a.named_parameters()), dict(b.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].grad is not None, name
        assert right[name].grad is not None, name
        torch.testing.assert_close(left[name].grad, right[name].grad,
                                   atol=atol, rtol=rtol, msg=name)


def assert_rejected(spec, cfg=None):
    try:
        validate(spec, cfg or config())
    except (TypeError, ValueError):
        return
    raise AssertionError(f"unexpectedly accepted {spec}")


def test_original_baseline_count_and_expanded_marker_identity():
    assert baseline_parameter_count() == 27_144_960
    # Meta construction verifies actual module/embedding/scalar registration
    # without allocating both full-size baseline parameter tables.
    with torch.device("meta"):
        assert count_parameters(SpecGPT()) == 27_144_960
    spec = baseline_spec(4)
    expanded = dict(spec, search_space="expanded")
    cfg = config(dropout=.1)
    legacy_model, expanded_model = make_model(spec, cfg), make_model(expanded, cfg)
    assert legacy_model.state_dict().keys() == expanded_model.state_dict().keys()
    for name, tensor in legacy_model.state_dict().items():
        torch.testing.assert_close(tensor, expanded_model.state_dict()[name], atol=0, rtol=0)
    inputs, positions, segments, targets, weights = batch()
    losses = []
    for model in (legacy_model, expanded_model):
        torch.manual_seed(819)
        loss = loss_from_logits(model(inputs, positions, segments), targets, weights)
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    assert_gradients_equal(legacy_model, expanded_model)


def test_grow_local_ffns_while_total_parameters_shrink():
    cfg = config()
    spec = baseline_spec(2)
    spec.update(search_space="expanded", ffns={"F0": {"hidden": 128},
                                               "F1": {"hidden": 96, "act": "relu2"}})
    model = make_model(spec, cfg)
    assert model.ffns["F0"].up.out_features > cfg.d_ff
    assert count_parameters(model) == spec_parameter_count(spec, cfg)
    assert count_parameters(model) < baseline_parameter_count(cfg)
    del spec["search_space"]
    assert_rejected(spec, cfg)


def test_depth_width_and_per_frame_ceiling_are_not_search_budget():
    cfg = config(n_layers=1, d_model=24, n_heads=3, d_ff=48)
    spec = baseline_spec(3)
    spec["search_space"] = "expanded"
    spec["ffns"] = {"F0": {"hidden": 96}}
    model = make_model(spec, cfg)
    assert model.config.d_model == model.tok.embedding_dim == 24
    assert model.n_applications == 3 > cfg.n_layers
    assert count_parameters(model) == spec_parameter_count(spec, cfg)
    # The caller must impose its absolute experiment budget. A narrow frame's
    # conventional baseline is deliberately NOT a cap on expanded candidates.
    assert count_parameters(model) > baseline_parameter_count(cfg)
    legacy = copy.deepcopy(spec)
    del legacy["search_space"]
    assert_rejected(legacy, cfg)
    cfg_wider = config(d_model=24, n_heads=3, n_layers=1, d_ff=48)
    wider_shallower = make_model(baseline_spec(1), cfg_wider)
    assert count_parameters(wider_shallower) < baseline_parameter_count(config())


def test_ffn_ffn_attention_explicit_order_and_gradients():
    spec = {"search_space": "expanded", "layers": [
        {"ffn": "F0", "ffn_scale": {"init": .5, "learned": True}},
        {"ffn": "F1"},
        {"attn": "A0", "attn_scale": .75},
    ], "ffns": {"F0": {"hidden": 96, "act": "relu2"},
                "F1": {"hidden": 32, "act": "silu", "transpose_tie": True}}}
    model = make_model(spec).double()
    reference = copy.deepcopy(model)
    assert count_parameters(model) == spec_parameter_count(spec, config())
    assert len(model.attns) == 1 and len(model.ffns) == 2
    assert set(model.branch_scales) == {"L0_ffn"}
    inputs, positions, segments, targets, weights = batch()
    x = reference.tok(inputs) + reference.pos(positions)
    x = x + reference.branch_scales["L0_ffn"] * reference.ffns["F0"](x)
    x = x + reference.ffns["F1"](x)
    x = x + .75 * reference.attns["A0"](x, reference.attention_mask(segments))
    expected = F.linear(reference.ln_f(x), reference.tok.weight)
    actual = model(inputs, positions, segments)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss_from_logits(actual, targets, weights).backward()
    loss_from_logits(expected, targets, weights).backward()
    assert_gradients_equal(model, reference)
    # These two packed conversations must remain isolated under the new order.
    with torch.no_grad():
        for start, end in ((0, 4), (4, 8)):
            separate = model(inputs[:, start:end], positions[:, start:end], segments[:, start:end])
            torch.testing.assert_close(actual[:, start:end], separate, atol=1e-12, rtol=1e-12)


def test_adjacent_shared_sublayers_match_loop_and_count_once():
    cfg = config(n_layers=1)
    unrolled = {"search_space": "expanded", "layers": [
        {"ffn": "F0"}, {"ffn": "F0"}, {"attn": "A0"}],
        "ffns": {"F0": {"hidden": 96}}}
    looped = copy.deepcopy(unrolled)
    looped["layers"] = [{"ffn": "F0", "loop": 2}, {"attn": "A0"}]
    a, b = make_model(unrolled, cfg), make_model(looped, cfg)
    assert a.n_applications == b.n_applications == 3
    assert len(a.ffns) == len(b.ffns) == 1
    assert count_parameters(a) == count_parameters(b) == spec_parameter_count(unrolled, cfg)
    inputs, positions, segments, targets, weights = batch()
    logits = [model(inputs, positions, segments) for model in (a, b)]
    torch.testing.assert_close(*logits, atol=0, rtol=0)
    for output in logits:
        loss_from_logits(output, targets, weights).backward()
    assert_gradients_equal(a, b)
    # Shared weights do not accidentally share the per-entry learned scales.
    scaled = copy.deepcopy(unrolled)
    for layer in scaled["layers"][:2]:
        layer["ffn_scale"] = {"init": 1., "learned": True}
    c = make_model(scaled, cfg)
    assert count_parameters(c) == count_parameters(a) + 2
    assert c.branch_scales["L0_ffn"] is not c.branch_scales["L1_ffn"]
    assert_rejected(dict(unrolled, search_space="legacy"), cfg)


def test_expanded_keeps_primitive_validation():
    base = dict(baseline_spec(4), search_space="expanded")
    for hidden in (0, -1, True, 1.5, "64"):
        spec = copy.deepcopy(base)
        spec["ffns"] = {"F0": {"hidden": hidden}}
        assert_rejected(spec)
    for marker in (None, True, "all", "expandd"):
        assert_rejected(dict(base, search_space=marker))
    for changes in ({"act": "swiglu"}, {"bias": True}, {"transpose_tie": 1}):
        spec = copy.deepcopy(base)
        spec["ffns"] = {"F0": changes}
        assert_rejected(spec)
    for changes in ({"attn": None, "ffn": None}, {"order": "parallel"},
                    {"loop": 0}, {"inject": True}, {"attn_scale": float("nan")}):
        spec = copy.deepcopy(base)
        spec["layers"][0].update(changes)
        assert_rejected(spec)


def test_fixed_learned_scales_match_bf16_rounding_and_device_conversion():
    fixed = dict(baseline_spec(3), search_space="expanded")
    values = (.5, -.25, 0., 1.25, .7, .1)
    for index, layer in enumerate(fixed["layers"]):
        layer["attn_scale"], layer["ffn_scale"] = values[2 * index:2 * index + 2]
    learned = copy.deepcopy(fixed)
    for layer in learned["layers"]:
        for kind in ("attn", "ffn"):
            layer[kind + "_scale"] = {"init": layer[kind + "_scale"], "learned": True}
    a, b = make_model(fixed), make_model(learned)
    assert count_parameters(b) == count_parameters(a) + 6
    assert all(not key.startswith("_fixed_scale_") for key in a.state_dict())
    inputs, positions, segments, _, _ = batch()
    for dtype in (torch.float32, torch.float64):
        a.to(dtype=dtype)
        b.to(dtype=dtype)
        for key, value in a.named_buffers():
            if key.startswith("_fixed_scale_"):
                assert value.dtype == dtype and value.device.type == DEVICE
        torch.testing.assert_close(a(inputs, positions, segments), b(inputs, positions, segments),
                                   atol=0, rtol=0)
    a.float()
    b.float()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        torch.testing.assert_close(a(inputs, positions, segments), b(inputs, positions, segments),
                                   atol=0, rtol=0)


def check_frozen_baseline(path):
    module_spec = importlib.util.spec_from_file_location("frozen_cuda_model", path)
    old = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = old
    module_spec.loader.exec_module(old)
    # Full original 27M architecture, with its actual dropout and dimensions.
    # Reset both initialization and forward RNGs to require exact equivalence.
    torch.manual_seed(1701)
    original = old.SpecGPT().to(DEVICE)
    torch.manual_seed(1701)
    current = SpecGPT().to(DEVICE)
    assert count_parameters(current) == old.count_parameters(original) == 27_144_960
    assert original.state_dict().keys() == current.state_dict().keys()
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, current.state_dict()[name], atol=0, rtol=0)
    inputs, positions, segments, targets, weights = batch()
    logits = []
    for model in (original, current):
        torch.manual_seed(819)
        output = model(inputs, positions, segments)
        logits.append(output)
        loss_from_logits(output, targets, weights).backward()
    torch.testing.assert_close(*logits, atol=0, rtol=0)
    assert_gradients_equal(original, current)
    print("ok frozen_baseline_exact_parameters_outputs_and_gradients", flush=True)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--legacy-model", type=Path)
    args = parser.parse_args()
    DEVICE = args.device
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name, flush=True)
    if args.legacy_model:
        check_frozen_baseline(args.legacy_model)
    print("Expanded architecture checks passed.", flush=True)


if __name__ == "__main__":
    main()
