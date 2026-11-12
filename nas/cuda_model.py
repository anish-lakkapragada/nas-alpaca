"""PyTorch/CUDA architecture search model with opt-in expanded architecture space.

The default frame is the original 27,144,960-parameter GPT: 8 layers, width
384, 6 attention heads and FFN width 1536. Archived 50M experiments remain
loadable by explicitly selecting Config(n_layers=10, d_model=512, n_heads=8,
d_ff=2048).

Attention and FFN IDs identify the entire sublayer, including its LayerNorm.
Attention is causal and restricted to equal conversation segment IDs. In
particular, padding segment -1 is handled exactly like in the MLX model.

Two additional authorized primitives are layer-local residual branch scaling
and sublayer reversal. A layer's ``order`` is ``attn_ffn`` (default) or
``ffn_attn``. Its ``attn_scale``/``ffn_scale`` is a finite fixed number, or
``{"init": number, "learned": bool}``; omission means fixed 1.0. The branch
update is ``x + scale * dropout(sublayer(x))``. Scalars belong to layer list
entries, remain independent when sublayer weights are shared, and are reused
across that entry's loop repetitions. No per-loop scalar copies are created.
Each learned scalar is counted and is an explicit +1 exception to the original
parameter ceiling; fixed scalars add no parameters or persistent buffers.

The newly authorized search uses ``"search_space": "expanded"`` at the spec
root. In that mode depth and FFN hidden widths are unrestricted by the frame,
and adjacent shared sublayers are allowed. The experiment controller, rather
than this model, enforces its absolute total-parameter budget. Global embedding
width and compatible head count remain Config fields. Arbitrary sublayer
sequences use the existing nullable branches: e.g. ``[{"ffn": "F0"},
{"ffn": "F1", "attn": "A0", "order": "ffn_attn"}]`` executes FFN, FFN,
attention, with separate FFN weights. Repeating an ID shares its entire
sublayer; repeating with new IDs allocates new weights. Initialization always
uses the same actual-entry-application-count rule, including entry loops.
Specs without the marker retain the original validation restrictions and the
same computation, initialization, parameter names, and checkpoint format.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    max_len: int = 1024
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("vocab_size", "max_len", "n_layers", "d_model", "n_heads", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


ACTIVATIONS = {
    "gelu": lambda x: F.gelu(x, approximate="none"),
    "relu": F.relu,
    "relu2": lambda x: F.relu(x).square(),
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


def baseline_spec(n_layers: int = 8) -> dict:
    return {
        "name": f"baseline-n{n_layers}",
        "description": "standard GPT-2-style decoder",
        "layers": [{"attn": f"A{i}", "ffn": f"F{i}"} for i in range(n_layers)],
        "ffns": {},
    }


def ffn_options(spec: dict, fid: str, config: Config | None = None) -> dict:
    config = config if config is not None else Config()
    options = {"hidden": config.d_ff, "act": "gelu", "transpose_tie": False}
    options.update(spec.get("ffns", {}).get(fid, {}))
    return options


def baseline_parameter_count(config: Config | None = None) -> int:
    c = config if config is not None else Config()
    return (
        (c.vocab_size + c.max_len) * c.d_model
        + c.n_layers * (4 * c.d_model**2 + 2 * c.d_model * c.d_ff + 4 * c.d_model)
        + 2 * c.d_model
    )


def residual_scale_options(layer: dict, kind: str) -> dict:
    """Parse one branch scalar without constructing tensors or adding weights."""
    field = f"{kind}_scale"
    value = layer.get(field, 1.0)
    if isinstance(value, dict):
        if set(value) != {"init", "learned"}:
            raise ValueError(f"{field}: object must contain exactly 'init' and 'learned'")
        initial, learned = value["init"], value["learned"]
        if not isinstance(learned, bool):
            raise ValueError(f"{field}: learned must be boolean")
    else:
        initial, learned = value, False
    if (not isinstance(initial, (int, float)) or isinstance(initial, bool)
            or not math.isfinite(initial)):
        raise ValueError(f"{field}: initial scale must be a finite real number")
    return {"init": float(initial), "learned": learned}


def learned_scale_count(spec: dict) -> int:
    """One learned scalar per present branch/layer entry, shared across loops."""
    return sum(int(residual_scale_options(layer, kind)["learned"])
               for layer in spec["layers"] for kind in ("attn", "ffn")
               if layer.get(kind) is not None)


def spec_parameter_count(spec: dict, config: Config | None = None) -> int:
    """Count unique weights, shared norms, and every learned branch scalar."""
    c = config if config is not None else Config()
    count = (c.vocab_size + c.max_len) * c.d_model + 2 * c.d_model
    attentions = {layer.get("attn") for layer in spec["layers"]} - {None}
    ffns = {layer.get("ffn") for layer in spec["layers"]} - {None}
    count += len(attentions) * (4 * c.d_model**2 + 2 * c.d_model)
    for fid in ffns:
        options = ffn_options(spec, fid, c)
        matrices = 1 if options["transpose_tie"] else 2
        count += matrices * c.d_model * options["hidden"] + 2 * c.d_model
    return count + learned_scale_count(spec)


def validate(spec: dict, config: Config | None = None) -> None:
    """Validate authorized primitives; expanded experiments enforce budgets outside.

    Legacy specs retain the original frame restrictions. The expanded marker
    grants architectural freedom only and changes no training settings.
    """
    c = config if config is not None else Config()
    search_space = spec.get("search_space", "legacy")
    if search_space not in ("legacy", "expanded"):
        raise ValueError("search_space must be 'legacy' or 'expanded'")
    expanded = search_space == "expanded"
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers must be a nonempty list")
    if not isinstance(spec.get("ffns", {}), dict):
        raise ValueError("ffns must be a dictionary")
    applications = 0
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layer {i}: expected a dictionary")
        extra = set(layer) - {"attn", "ffn", "loop", "inject", "order", "attn_scale", "ffn_scale"}
        if extra:
            raise ValueError(f"layer {i}: unsupported keys {extra}")
        if layer.get("attn") is None and layer.get("ffn") is None:
            raise ValueError(f"layer {i}: deleting both sublayers removes the layer entirely; omit it instead")
        if layer.get("order", "attn_ffn") not in ("attn_ffn", "ffn_attn"):
            raise ValueError(f"layer {i}: order must be 'attn_ffn' or 'ffn_attn'")
        for kind in ("attn", "ffn"):
            key = layer.get(kind)
            if key is not None and (not isinstance(key, str) or not key or "." in key):
                raise ValueError(f"layer {i}: {kind} ID must be a nonempty string without '.'")
            if key is None and f"{kind}_scale" in layer:
                raise ValueError(f"layer {i}: cannot scale a deleted {kind} branch")
            if key is not None:
                residual_scale_options(layer, kind)
        repeats = layer.get("loop", 1)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise ValueError(f"layer {i}: loop must be a positive integer")
        if not isinstance(layer.get("inject", False), bool):
            raise ValueError(f"layer {i}: inject must be boolean")
        if layer.get("inject") and repeats == 1:
            raise ValueError(f"layer {i}: inject only applies to looped layers")
        applications += repeats
    if not expanded and applications > c.n_layers:
        raise ValueError(f"{applications} layer applications exceed the {c.n_layers}-layer frame")
    for kind in ("attn", "ffn"):
        for i in range(1, len(layers)):
            previous, current = layers[i - 1].get(kind), layers[i].get(kind)
            if not expanded and previous is not None and previous == current:
                raise ValueError(f"{kind} {current} tied at adjacent layers {i-1},{i}; express reuse there as a loop")
    for fid, given in spec.get("ffns", {}).items():
        if not isinstance(given, dict):
            raise ValueError(f"ffn {fid}: expected a dictionary")
        extra = set(given) - {"hidden", "act", "transpose_tie"}
        if extra:
            raise ValueError(f"ffn {fid}: unsupported keys {extra}")
        options = ffn_options(spec, fid, c)
        if options["act"] not in ACTIVATIONS:
            raise ValueError(f"ffn {fid}: unknown activation {options['act']}")
        hidden = options["hidden"]
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError(f"ffn {fid}: hidden width must be a positive integer")
        if not expanded and hidden > c.d_ff:
            raise ValueError(f"ffn {fid}: hidden width must be an integer in [1, {c.d_ff}] (narrowing only)")
        if not isinstance(options["transpose_tie"], bool):
            raise ValueError(f"ffn {fid}: transpose_tie must be boolean")
    parameters = spec_parameter_count(spec, c)
    ceiling = baseline_parameter_count(c) + learned_scale_count(spec)
    if not expanded and parameters > ceiling:
        raise ValueError(f"{parameters:,} parameters exceed baseline plus authorized scalar allowance {ceiling:,}")


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.n_heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.n_heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.n_heads, -1).transpose(1, 2)
        # SDPA boolean masks use True for visible keys, as in MLX. The
        # original model has no attention-probability dropout.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class SharedAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attn = Attention(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.attn(self.ln(x), mask)


class SharedFFN(nn.Module):
    def __init__(self, config: Config, hidden: int, act: str, transpose_tie: bool):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, eps=1e-5)
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        if not transpose_tie:
            self.down = nn.Linear(hidden, config.d_model, bias=False)
        self.transpose_tie = transpose_tie
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = ACTIVATIONS[self.act_name](self.up(self.ln(x)))
        if self.transpose_tie:
            # up.weight has shape [hidden, width]; this is the tied down map.
            return hidden @ self.up.weight
        return self.down(hidden)


class SpecGPT(nn.Module):
    def __init__(self, spec: dict | None = None, config: Config | None = None):
        super().__init__()
        self.config = copy.deepcopy(config if config is not None else Config())
        self.spec = copy.deepcopy(spec if spec is not None else baseline_spec(self.config.n_layers))
        validate(self.spec, self.config)
        c = self.config
        self.n_applications = sum(layer.get("loop", 1) for layer in self.spec["layers"])
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.attns = nn.ModuleDict()
        self.ffns = nn.ModuleDict()
        # First-appearance order mirrors arch.py, including initialization.
        for layer in self.spec["layers"]:
            attention_id, ffn_id = layer.get("attn"), layer.get("ffn")
            if attention_id is not None and attention_id not in self.attns:
                self.attns[attention_id] = SharedAttention(c)
            if ffn_id is not None and ffn_id not in self.ffns:
                self.ffns[ffn_id] = SharedFFN(c, **ffn_options(self.spec, ffn_id, c))
        self.ln_f = nn.LayerNorm(c.d_model, eps=1e-5)
        self.drop = nn.Dropout(c.dropout)
        self.branch_scales = nn.ParameterDict()
        self._fixed_branch_scales = {}
        for index, layer in enumerate(self.spec["layers"]):
            for kind in ("attn", "ffn"):
                if layer.get(kind) is None:
                    continue
                key = f"L{index}_{kind}"
                options = residual_scale_options(layer, kind)
                if options["learned"]:
                    self.branch_scales[key] = nn.Parameter(torch.tensor(options["init"]))
                else:
                    self._fixed_branch_scales[key] = options["init"]
                    if options["init"] != 1.0:
                        # A Python scalar and a device FP32 scalar do not
                        # necessarily round alike when multiplying BF16.
                        # Match learned-scalar arithmetic exactly, without a
                        # trainable parameter or checkpoint entry. Module
                        # device/dtype conversions also move these buffers.
                        self.register_buffer(f"_fixed_scale_{key}",
                                             torch.tensor(options["init"]), persistent=False)
        self.register_buffer("_causal_mask", torch.ones(c.max_len, c.max_len, dtype=torch.bool).tril(), persistent=False)
        self._init()
        assert count_parameters(self) == spec_parameter_count(self.spec, c)

    def _init(self) -> None:
        std = 0.02
        # MLX bases residual scaling on actual layer applications, not the
        # ceiling. Deleted whole layers therefore change this scale there too.
        residual_std = std / math.sqrt(2 * self.n_applications)
        nn.init.normal_(self.tok.weight, std=std)
        nn.init.normal_(self.pos.weight, std=std)
        initialized = set()
        for layer in self.spec["layers"]:
            for kind, table in (("attn", self.attns), ("ffn", self.ffns)):
                key = layer.get(kind)
                if key is None or (kind, key) in initialized:
                    continue
                initialized.add((kind, key))
                module = table[key]
                if kind == "attn":
                    nn.init.normal_(module.attn.qkv.weight, std=std)
                    nn.init.normal_(module.attn.out.weight, std=residual_std)
                else:
                    nn.init.normal_(module.up.weight, std=std)
                    if not module.transpose_tie:
                        nn.init.normal_(module.down.weight, std=residual_std)

    def attention_mask(self, segments: torch.Tensor) -> torch.Tensor:
        length = segments.shape[1]
        causal = self._causal_mask[:length, :length]
        same_conversation = segments[:, :, None] == segments[:, None, :]
        return (same_conversation & causal)[:, None, :, :]

    def _layer(self, layer: dict, x: torch.Tensor, mask: torch.Tensor, index: int) -> torch.Tensor:
        order = ("ffn", "attn") if layer.get("order", "attn_ffn") == "ffn_attn" else ("attn", "ffn")
        for kind in order:
            sublayer_id = layer.get(kind)
            if sublayer_id is None:
                continue
            branch = self.attns[sublayer_id](x, mask) if kind == "attn" else self.ffns[sublayer_id](x)
            branch = self.drop(branch)
            key = f"L{index}_{kind}"
            scale = self.branch_scales[key] if key in self.branch_scales else self._fixed_branch_scales[key]
            if isinstance(scale, float) and scale != 1.0:
                scale = getattr(self, f"_fixed_scale_{key}")
            # The omitted/default fixed-1 case preserves the original graph.
            # Learned-1 still multiplies so the new scalar receives a gradient.
            if not isinstance(scale, float) or scale != 1.0:
                branch = branch * scale
            x = x + branch
        return x

    def hidden_states(self, inputs: torch.Tensor, positions: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        """Final normalized states, before tied vocabulary projection.

        A trainer can select answer positions here before F.linear(states,
        self.tok.weight), avoiding the large unused prompt-token logits while
        preserving their gradients through the transformer and embeddings.
        """
        if inputs.ndim != 2 or inputs.shape != positions.shape or inputs.shape != segments.shape:
            raise ValueError("inputs, positions, and segments must have equal [batch, length] shapes")
        if inputs.shape[1] > self.config.max_len:
            raise ValueError("sequence length exceeds config.max_len")
        mask = self.attention_mask(segments)
        x = self.drop(self.tok(inputs) + self.pos(positions))
        for index, layer in enumerate(self.spec["layers"]):
            repeats, inject = layer.get("loop", 1), layer.get("inject", False)
            layer_input = x
            x = self._layer(layer, x, mask, index)
            for _ in range(repeats - 1):
                x = self._layer(layer, x + layer_input if inject else x, mask, index)
        return self.ln_f(x)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        return F.linear(self.hidden_states(inputs, positions, segments), self.tok.weight)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
