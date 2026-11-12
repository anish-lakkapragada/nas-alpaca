"""Checks for arch.py. Run: .venv/bin/python test_arch.py"""
import copy

import mlx.core as mx
import numpy as np

from nas.mlx import arch
from nas.mlx.model import GPT, Config, count_parameters

BASE_PARAMS = 27_144_960
ATTN = 4 * 384 * 384 + 2 * 384          # weights + its LayerNorm
FFN = 2 * 384 * 1536 + 2 * 384


def batch():
    rng = np.random.default_rng(0)
    inp = mx.array(rng.integers(6, 32768, (2, 64)), dtype=mx.int32)
    pos = mx.broadcast_to(mx.arange(64), (2, 64))
    seg = mx.zeros((2, 64), mx.int32)
    return inp, pos, seg


def test_baseline_identical():
    mx.random.seed(0)
    ref = GPT(Config(n_layers=8))
    mx.random.seed(0)
    new = arch.SpecGPT(arch.baseline_spec(8))
    ref.eval(); new.eval()
    assert count_parameters(new) == count_parameters(ref) == BASE_PARAMS
    a, b = ref(*batch()), new(*batch())
    assert mx.array_equal(a, b).item(), float(mx.abs(a - b).max())


def spec_with(**edits):
    s = arch.baseline_spec(8)
    s.update(copy.deepcopy(edits))
    return s


def params(s):
    return count_parameters(arch.SpecGPT(s))


def test_primitive_counts():
    b = arch.baseline_spec(8)
    # 2/3 tie attention / FFN between layers 0 and 2
    s = copy.deepcopy(b); s["layers"][2]["attn"] = "A0"
    assert params(s) == BASE_PARAMS - ATTN
    s = copy.deepcopy(b); s["layers"][2]["ffn"] = "F0"
    assert params(s) == BASE_PARAMS - FFN
    # 5/6 delete
    s = copy.deepcopy(b); s["layers"][3]["attn"] = None
    assert params(s) == BASE_PARAMS - ATTN
    s = copy.deepcopy(b); s["layers"][3]["ffn"] = None
    assert params(s) == BASE_PARAMS - FFN
    # 7 transpose tie
    s = copy.deepcopy(b); s["ffns"] = {"F1": {"transpose_tie": True}}
    assert params(s) == BASE_PARAMS - 384 * 1536
    # 8 narrow
    s = copy.deepcopy(b); s["ffns"] = {"F1": {"hidden": 1024}}
    assert params(s) == BASE_PARAMS - 2 * 384 * 512
    # 4 activation: no change
    s = copy.deepcopy(b); s["ffns"] = {f"F{i}": {"act": "relu2"} for i in range(8)}
    assert params(s) == BASE_PARAMS
    # 1 loop: 4 layers x2 = 8 applications, half the blocks
    s = {"layers": [{"attn": f"A{i}", "ffn": f"F{i}", "loop": 2, "inject": True} for i in range(4)]}
    assert params(s) == BASE_PARAMS - 4 * (ATTN + FFN)
    m = arch.SpecGPT(s); m.eval()
    assert m(*batch()).shape == (2, 64, 32768)


def test_transpose_tie_math():
    s = arch.baseline_spec(8); s["ffns"] = {"F0": {"transpose_tie": True}}
    m = arch.SpecGPT(s)
    f = m.ffns["F0"]
    x = mx.random.normal((3, 384))
    import mlx.nn as nn
    h = nn.gelu(f.up(f.ln(x)))
    assert mx.allclose(f(x), h @ f.up.weight, atol=1e-5).item()
    assert f(x).shape == (3, 384)


def test_rejections():
    bad = []
    s = arch.baseline_spec(8); s["layers"][1]["attn"] = "A0"; bad.append(s)          # adjacent tie
    s = arch.baseline_spec(8); s["layers"][0]["loop"] = 2; bad.append(s)             # 9 applications
    s = arch.baseline_spec(8); s["ffns"] = {"F0": {"hidden": 2048}}; bad.append(s)   # widening
    s = arch.baseline_spec(8); s["ffns"] = {"F0": {"act": "swiglu"}}; bad.append(s)  # not elementwise
    s = arch.baseline_spec(8); s["layers"][0]["extra"] = 1; bad.append(s)            # unknown primitive
    s = arch.baseline_spec(8); s["layers"][0]["attn"] = None; s["layers"][0]["ffn"] = None; bad.append(s)
    for s in bad:
        try:
            arch.validate(s)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid spec {s}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
