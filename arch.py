"""Spec-driven decoder supporting exactly the eight whitelisted primitives.

A spec is JSON:
{
  "name": "...", "description": "...",
  "layers": [                                   # applied in order
    {"attn": "A0", "ffn": "F0"},                # ids; null deletes that sublayer
    {"attn": "A1", "ffn": "F1", "loop": 2, "inject": true},
    ...
  ],
  "ffns": {"F0": {"hidden": 1536, "act": "gelu", "transpose_tie": false}}
}

Primitives (and nothing else):
  1 loop a layer          "loop": r  (optional "inject": re-add the loop input each pass,
                                      Y_{t+1} = layer(Y_t + X), Yang et al. 2024)
  2 tie two attentions    same attention id at two non-adjacent layers
  3 tie two FFNs          same FFN id at two non-adjacent layers
  4 change activation     ffns[id].act in ACTIVATIONS
  5 delete attention      "attn": null
  6 delete FFN            "ffn": null
  7 transpose-tie an FFN  ffns[id].transpose_tie = true  (down = up^T)
  8 narrow an FFN         ffns[id].hidden < 1536

Fixed: width 384, 6 heads, tied 32k word table, learned positions, 1,024 context.
A sublayer's pre-LayerNorm belongs to it and is shared with it. Total layer
applications (sum of loops) may not exceed 8, and the parameter count may not
exceed the baseline's. With the 8-layer baseline spec, construction and
initialisation consume random numbers in the same order as model.GPT, so the
same seed gives the identical model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from model import Attention, Config

ACTIVATIONS = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu2": lambda x: mx.square(nn.relu(x)),
    "silu": nn.silu,
    "gelu_tanh": nn.gelu_approx,
}
BASE = Config(n_layers=8)          # the approved baseline frame
MAX_APPLICATIONS = 8


def baseline_spec(n_layers: int = 8) -> dict:
    return {"name": f"baseline-n{n_layers}", "description": "standard GPT-2-style decoder",
            "layers": [{"attn": f"A{i}", "ffn": f"F{i}"} for i in range(n_layers)], "ffns": {}}


def ffn_options(spec, fid):
    o = {"hidden": BASE.d_ff, "act": "gelu", "transpose_tie": False}
    o.update(spec.get("ffns", {}).get(fid, {}))
    return o


def validate(spec: dict) -> None:
    layers = spec["layers"]
    allowed_layer = {"attn", "ffn", "loop", "inject"}
    allowed_ffn = {"hidden", "act", "transpose_tie"}
    apps = 0
    for i, L in enumerate(layers):
        extra = set(L) - allowed_layer
        if extra:
            raise ValueError(f"layer {i}: unsupported keys {extra}")
        if L.get("attn") is None and L.get("ffn") is None:
            raise ValueError(f"layer {i}: deleting both sublayers removes the layer entirely; omit it instead")
        r = L.get("loop", 1)
        if not (isinstance(r, int) and r >= 1):
            raise ValueError(f"layer {i}: loop must be a positive integer")
        if L.get("inject") and r == 1:
            raise ValueError(f"layer {i}: inject only applies to looped layers")
        apps += r
    if apps > MAX_APPLICATIONS:
        raise ValueError(f"{apps} layer applications exceed the {MAX_APPLICATIONS}-layer frame")
    for kind in ("attn", "ffn"):
        for i in range(1, len(layers)):
            a, b = layers[i - 1].get(kind), layers[i].get(kind)
            if a is not None and a == b:
                raise ValueError(f"{kind} {a} tied at adjacent layers {i-1},{i}; express reuse there as a loop")
    for fid, o in spec.get("ffns", {}).items():
        extra = set(o) - allowed_ffn
        if extra:
            raise ValueError(f"ffn {fid}: unsupported keys {extra}")
        o = ffn_options(spec, fid)
        if o["act"] not in ACTIVATIONS:
            raise ValueError(f"ffn {fid}: unknown activation {o['act']}")
        if not (1 <= o["hidden"] <= BASE.d_ff):
            raise ValueError(f"ffn {fid}: hidden width must be in [1, {BASE.d_ff}] (narrowing only)")


class SharedAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln = nn.LayerNorm(c.d_model)
        self.attn = Attention(c)

    def __call__(self, x, mask):
        return self.attn(self.ln(x), mask)


class SharedFFN(nn.Module):
    def __init__(self, c: Config, hidden: int, act: str, transpose_tie: bool):
        super().__init__()
        self.ln = nn.LayerNorm(c.d_model)
        self.up = nn.Linear(c.d_model, hidden, bias=False)
        if not transpose_tie:
            self.down = nn.Linear(hidden, c.d_model, bias=False)
        self.transpose_tie = transpose_tie
        self.act_name = act

    def __call__(self, x):
        h = ACTIVATIONS[self.act_name](self.up(self.ln(x)))
        if self.transpose_tie:
            return h @ self.up.weight          # up.weight is [hidden, d]; this is up^T
        return self.down(h)


class SpecGPT(nn.Module):
    def __init__(self, spec: dict, dropout: float = BASE.dropout):
        super().__init__()
        validate(spec)
        c = Config(n_layers=sum(L.get("loop", 1) for L in spec["layers"]), dropout=dropout)
        self.config, self.spec = c, spec
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.attns, self.ffns = {}, {}
        for L in spec["layers"]:                       # first-appearance order
            a, f = L.get("attn"), L.get("ffn")
            if a is not None and a not in self.attns:
                self.attns[a] = SharedAttention(c)
            if f is not None and f not in self.ffns:
                o = ffn_options(spec, f)
                self.ffns[f] = SharedFFN(c, o["hidden"], o["act"], o["transpose_tie"])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(dropout)
        self._init()

    def _init(self):
        std, resid = 0.02, 0.02 / math.sqrt(2 * self.config.n_layers)
        self.tok.weight = mx.random.normal(self.tok.weight.shape) * std
        self.pos.weight = mx.random.normal(self.pos.weight.shape) * std
        done = set()
        for L in self.spec["layers"]:
            for kind, table in (("attn", self.attns), ("ffn", self.ffns)):
                k = L.get(kind)
                if k is None or (kind, k) in done:
                    continue
                done.add((kind, k))
                m = table[k]
                if kind == "attn":
                    m.attn.qkv.weight = mx.random.normal(m.attn.qkv.weight.shape) * std
                    m.attn.out.weight = mx.random.normal(m.attn.out.weight.shape) * resid
                else:
                    m.up.weight = mx.random.normal(m.up.weight.shape) * std
                    if not m.transpose_tie:
                        m.down.weight = mx.random.normal(m.down.weight.shape) * resid

    def _layer(self, L, x, mask):
        a, f = L.get("attn"), L.get("ffn")
        if a is not None:
            x = x + self.drop(self.attns[a](x, mask))
        if f is not None:
            x = x + self.drop(self.ffns[f](x))
        return x

    def __call__(self, inputs, positions, segments):
        T = inputs.shape[1]
        causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        mask = ((segments[:, :, None] == segments[:, None, :]) & causal)[:, None]
        x = self.drop(self.tok(inputs) + self.pos(positions))
        for L in self.spec["layers"]:
            r, inject = L.get("loop", 1), L.get("inject", False)
            x_in = x
            x = self._layer(L, x, mask)
            for _ in range(r - 1):
                x = self._layer(L, x + x_in if inject else x, mask)
        return self.tok.as_linear(self.ln_f(x))


def load_spec(path) -> dict:
    return json.loads(Path(path).read_text())


def describe(spec: dict) -> str:
    parts = []
    for L in spec["layers"]:
        s = f"{L.get('attn') or '-'}/{L.get('ffn') or '-'}"
        if L.get("loop", 1) > 1:
            s += f"x{L['loop']}" + ("+inj" if L.get("inject") else "")
        parts.append(s)
    opts = {k: ffn_options(spec, k) for k in sorted(spec.get("ffns", {}))}
    fmt = lambda o: f"{o['act']},{o['hidden']}" + (",T" if o["transpose_tie"] else "")
    used = {L.get("ffn") for L in spec["layers"]} - {None}
    if opts and set(opts) >= used and len({fmt(o) for o in opts.values()}) == 1:
        extras = ["all FFN: " + fmt(next(iter(opts.values())))]
    else:
        extras = [f"{k}:{fmt(o)}" for k, o in opts.items()]
    return " ".join(parts) + (("  [" + "; ".join(extras) + "]") if extras else "")
