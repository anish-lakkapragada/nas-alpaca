"""A plain GPT-style decoder in MLX: pre-LayerNorm, learned positions, GELU MLP,
tied input/output embeddings."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab_size: int = 32768
    max_len: int = 1024
    n_layers: int = 4
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def __call__(self, x, mask):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        h = lambda t: t.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3)
        q, k, v = h(q), h(k), h(v)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn, self.mlp = Attention(c), MLP(c)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.blocks = [Block(c) for _ in range(c.n_layers)]
        self.ln_f = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self._init()

    def _init(self):
        std, resid = 0.02, 0.02 / math.sqrt(2 * self.config.n_layers)
        self.tok.weight = mx.random.normal(self.tok.weight.shape) * std
        self.pos.weight = mx.random.normal(self.pos.weight.shape) * std
        for b in self.blocks:
            for lin, s in ((b.attn.qkv, std), (b.attn.out, resid), (b.mlp.up, std), (b.mlp.down, resid)):
                lin.weight = mx.random.normal(lin.weight.shape) * s

    def __call__(self, inputs, positions, segments):
        # Causal, block-diagonal mask: attend only within the same conversation.
        T = inputs.shape[1]
        causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        same = segments[:, :, None] == segments[:, None, :]
        mask = (same & causal)[:, None]
        x = self.drop(self.tok(inputs) + self.pos(positions))
        for b in self.blocks:
            x = b(x, mask)
        return self.tok.as_linear(self.ln_f(x))


def count_parameters(model) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))
