"""Render the exact baseline / candidate-C architectures as publication figures.

Run from any directory with the project's Python environment:
    .venv/bin/python scripts/render_architecture_comparison.py
    .venv/bin/python scripts/render_architecture_comparison.py --font-family Charter \
        --output-dir figures/font_variants --stem 03-charter

The source snapshot retains the original run configs without machine metadata.
Module IDs and execution order are read from those specs, never inferred from
the picture. Outputs are a 300-dpi PNG, editable SVG, and vector PDF.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nas-figure-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--font-family", default="Helvetica Neue")
parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
parser.add_argument("--stem", default="baseline_vs_codex_architecture")
parser.add_argument("--check-only", action="store_true", help="Validate layout without exporting files")
args = parser.parse_args()
if Path(args.stem).name != args.stem:
    parser.error("--stem must be a filename without directory components")
FONT_FAMILY = args.font_family
FONT_FILES = {weight: findfont(FontProperties(family=FONT_FAMILY, weight=weight), fallback_to_default=False)
              for weight in ("normal", "bold")}
OUT = args.output_dir.resolve()
DATA = json.loads((ROOT / "figures" / "architecture_comparison.data.json").read_text())
BASE, CODEX = DATA["baseline"], DATA["codex"]
SPEC = CODEX["spec"]
assert SPEC == json.loads((ROOT / DATA["provenance"]["candidate_spec"]).read_text())
assert BASE["parameters"] == 27_144_960 and CODEX["parameters"] == 11_458_944
assert len(BASE["spec"]["layers"]) == 8 and len(SPEC["layers"]) == 37
assert len({(r["attn"], r["ffn"]) for r in SPEC["layers"][1:-1]}) == 35
assert all(r["attn"] == f"Amiddle{k % 5}" and r["ffn"] == f"Fmiddle{k % 7}"
           for k, r in enumerate(SPEC["layers"][1:-1]))
assert [i + 1 for i, r in enumerate(SPEC["layers"]) if r["order"] == "attn_ffn"] == [32]

W = 2700
INK = "#17232B"
MUTED = "#45535F"
FAINT = "#D6E0E8"
PAPER = "#FFFFFF"
PURPLE = "#7040A5"
ATTN = ["#F04B52", "#E7BB16", "#35BD77", "#3496EB", "#B65ACB"]
FFN = ["#F04B52", "#FF9B35", "#F0D83A", "#35BD77", "#44ACF0", "#7775E7", "#C971DA"]
BOUNDARY = "#A9B4C0"
SPACE_GRAY = "#28282D"
CODEX_BLUE = "#1557B0"
TITLE_SIZE = 60
MODULE_FONT_SIZE = 36
WEIGHT_FONT_SIZE = 30
ROW_Y, ROW_STEP, MODULE_H, MODULE_W, MODULE_GAP = 550, 98, 78, 400, 120
STACK_END = ROW_Y + 36 * ROW_STEP + MODULE_H
FINAL_NORM_Y = STACK_END + 68
PROJECTION_Y = FINAL_NORM_Y + 110
SOFTMAX_Y = PROJECTION_Y + 167
H = SOFTMAX_Y + 90


def contrast_with_white(hex_color):
    values = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in values]
    luminance = sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))
    return 1.05 / (luminance + 0.05)


plt.rcParams.update({
    "font.family": [FONT_FAMILY, "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 18,
    "text.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(W / 100, H / 100), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
all_text = []
bounded_text = []
module_contrasts = []
shared_weight_labels = []


def txt(x, y, text, size=18, color=INK, weight="normal", ha="left", **kw):
    # Avoid adding multi-line leading to single-line labels in tall-metric fonts.
    spacing = kw.pop("linespacing", 1.3 if "\n" in text else 1.0)
    vertical_alignment = kw.pop("va", "center")
    artist = ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                     ha=ha, va=vertical_alignment, linespacing=spacing, **kw)
    all_text.append(artist)
    return artist


def arrow(x1, y1, x2, y2, color=MUTED, lw=1.35, scale=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=scale, linewidth=lw, color=color,
                                shrinkA=0, shrinkB=0, zorder=2))


def activation_wire(points):
    """A directed activation edge, with no branches or disconnected row rails."""
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.35, color=MUTED, zorder=2,
                                joinstyle="round", capstyle="round"))


def box(x, y, w, h, fill=PAPER, edge=FAINT, radius=7, lw=0.8, **kw):
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0,rounding_size={radius}",
                      facecolor=fill, edgecolor=edge, linewidth=lw, **kw)
    ax.add_patch(p)
    return p


def module_chip(x, y, w, h, label, accent, kind, size=MODULE_FONT_SIZE, weight="bold", monochrome=False, detail=None, shared_id=None):
    """Encode module type by fill/shape and shared parameter identity by hue/math.

    Attention: pale fill, rounded corners, colored outline and dark lettering.
    FFN: stronger fill, nearly square corners and dark lettering.
    Hue still identifies the shared parameter bank within C.
    """
    rgb = to_rgb(accent)
    foreground = INK
    if monochrome:
        assert kind in ("attn", "ffn")
        fill = "#FFFFFF" if kind == "attn" else SPACE_GRAY
        foreground = "#000000" if kind == "attn" else "#FFFFFF"
        box(x, y, w, h, fill=fill, edge="#000000" if kind == "attn" else SPACE_GRAY,
            radius=10 if kind == "attn" else 1.5, lw=1.5)
    elif kind == "attn":
        fill = to_hex(tuple(0.16 * c + 0.84 for c in rgb))
        box(x, y, w, h, fill=fill, edge=accent, radius=10, lw=2)
        # A saturated accent makes even pale attention fills easy to track.
        box(x + 5, y + 5, 9, h - 10, fill=accent, edge=accent, radius=3, lw=0)
    else:
        assert kind == "ffn"
        fill = to_hex(tuple(0.60 * c + 0.40 for c in rgb))
        box(x, y, w, h, fill=fill, edge=accent, radius=1.5, lw=0.8)
        box(x + 1, y + h - 5, w - 2, 4, fill=accent, edge=accent, radius=0, lw=0)
    ratios = [contrast_with_white(foreground), contrast_with_white(fill)]
    contrast = max(ratios) / min(ratios)
    module_contrasts.append(contrast)
    label_y = y + h * 0.23 if detail else y + h / 2
    main_width = w - 140 if shared_id is not None else w
    t = txt(x + main_width / 2, label_y, label, size=size, color=foreground,
            ha="center", weight=weight)
    bounded_text.append((t, x + 18, y + 4, main_width - 36, h - 8))
    if shared_id is not None:
        # Every occurrence of a module carries the same typeset weight symbol.
        family = "Attn" if kind == "attn" else "FFN"
        index = int(shared_id.split("middle")[1])
        weight_math = rf"$W_{{{index}}}^{{\mathrm{{{family}}}}}$"
        tag_x, tag_w = x + w - 126, 116
        box(tag_x, y + 7, tag_w, h - 14, fill=PAPER, edge=accent, radius=5, lw=0.8)
        tag = txt(tag_x + tag_w / 2, y + h / 2, weight_math,
                  size=WEIGHT_FONT_SIZE, color=INK, ha="center", fontfamily="STIXGeneral")
        bounded_text.append((tag, tag_x + 6, y + 8, tag_w - 12, h - 16))
        shared_weight_labels.append({"module": shared_id, "family": family,
                                     "index": index, "math": weight_math})
    if detail:
        t = txt(x + w / 2, y + h * 0.66, detail, size=22,
                color=foreground, ha="center")
        bounded_text.append((t, x + 18, y + 4, w - 36, h - 8))


def module_label(mid):
    prefix = "Attn" if mid.startswith("A") else "FFN"
    if "middle" in mid:
        suffix = mid.split("middle")[1]
    elif mid.endswith("first"):
        suffix = "in"
    elif mid.endswith("last"):
        suffix = "out"
    else:
        suffix = mid[1:]
    return f"{prefix} {suffix}"


def module_color(mid):
    if "middle" not in mid:
        return BOUNDARY
    return (ATTN if mid.startswith("A") else FFN)[int(mid.split("middle")[1])]


def input_output(x, width, dim, codex=False):
    center = x + width / 2
    fill, edge = ("#F4F7FA", "#D9E2EA") if codex else ("#FFFFFF", "#000000")
    foreground, secondary = (INK, MUTED) if codex else ("#000000", "#000000")
    box(x, 330, width, 116, fill=fill, edge=edge)
    txt(center, 366, "Token + position embeddings", ha="center", size=24, weight="bold", color=foreground)
    txt(center, 413, f"Embedding width: {dim}", ha="center", size=22, color=secondary)
    if codex:
        first_center = x + MODULE_W / 2
        last_center = x + MODULE_W + MODULE_GAP + MODULE_W / 2
        activation_wire([(center, 446), (center, 500), (first_center, 500), (first_center, ROW_Y)])
        turn_y = STACK_END + 30
        activation_wire([(last_center, STACK_END), (last_center, turn_y), (center, turn_y), (center, FINAL_NORM_Y)])
    else:
        arrow(center, 446, center, ROW_Y, color=secondary)
        arrow(center, STACK_END, center, FINAL_NORM_Y, color=secondary)
    box(x, FINAL_NORM_Y, width, 70, fill=fill, edge=edge)
    txt(center, FINAL_NORM_Y + 35, "Final LayerNorm", size=24, ha="center", color=foreground)
    arrow(center, FINAL_NORM_Y + 71, center, PROJECTION_Y, color=secondary)
    box(x, PROJECTION_Y, width, 90, fill=fill, edge=edge)
    txt(center, PROJECTION_Y + 30, r"Vocabulary projection ($E^{T}$)", size=23, ha="center", color=foreground)
    txt(center, PROJECTION_Y + 69, "Tied to the input token-embedding weights", size=18, color=secondary, ha="center")
    arrow(center, PROJECTION_Y + 91, center, SOFTMAX_Y - 37, color=secondary)
    txt(center, SOFTMAX_Y, "Softmax → next-token probabilities", size=21, color=secondary, ha="center")


# Two centered titles, with generous whitespace and no comparison-table rules.
bw = cw = 2 * MODULE_W + MODULE_GAP
bx, cx = W / 4 - bw / 2, 3 * W / 4 - cw / 2
heading_renderer = fig.canvas.get_renderer()
pixel_to_data = W / ax.get_window_extent(heading_renderer).width
# Measure the complete line once. Separately centering each colored fragment
# shifts "Codex" because it has no descender, unlike "-optimized Architecture".
probe = ax.text(0, 0, "Codex-optimized Architecture", fontsize=TITLE_SIZE,
                fontweight="bold", va="baseline", linespacing=1.0)
probe_bounds = probe.get_window_extent(heading_renderer)
probe_anchor_y = ax.transData.transform((0, 0))[1]
title_baseline_y = 105 + ((probe_bounds.y0 + probe_bounds.y1) / 2 - probe_anchor_y) * pixel_to_data
probe.remove()
codex_title_artists = []
for center, name, record in [(bx + bw / 2, "Baseline Architecture", BASE),
                             (cx + cw / 2, "Codex-optimized Architecture", CODEX)]:
    if record is BASE:
        headings = [(txt(center, title_baseline_y, name, size=TITLE_SIZE, weight="bold",
                         ha="center", va="baseline", color="#000000"), 105)]
    else:
        # Center the complete title while coloring only the word "Codex".
        spans = [("Codex", CODEX_BLUE), ("-optimized Architecture", "#000000")]
        props = FontProperties(family=plt.rcParams["font.family"], weight="bold", size=TITLE_SIZE)
        widths = [heading_renderer.get_text_width_height_descent(text, props, False)[0] * pixel_to_data
                  for text, _ in spans]
        left = center - sum(widths) / 2
        headings = []
        for (text, color), width in zip(spans, widths):
            artist = txt(left, title_baseline_y, text, size=TITLE_SIZE, weight="bold",
                         color=color, va="baseline")
            headings.append((artist, 105))
            codex_title_artists.append(artist)
            left += width
    headings.append((txt(center, 225, f"({record['parameters'] / 1e6:.2f}M params)",
                         size=TITLE_SIZE, weight="bold", ha="center",
                         color="#000000" if record is BASE else CODEX_BLUE), 225))
    for artist, y in headings:
        # Baseline-aligned fragments can have different ascender/descender boxes
        # while their visible letters sit on the same line.
        top, height = (30, 140) if y == 105 else (y - 58, 116)
        bounded_text.append((artist, center - W / 4 + 40, top, W / 2 - 80, height))
reduction = 100 * (1 - CODEX["parameters"] / BASE["parameters"])

# Baseline: independent modules, with no implied sharing between the rows.
input_output(bx, bw, BASE["model"]["d_model"])
for index, layer in enumerate(BASE["spec"]["layers"]):
    y = ROW_Y + index * 460
    txt(bx - 35, y + 193, f"{index + 1:02d}", size=22, color="#000000", ha="right")
    module_chip(bx + 28, y + 26, bw - 56, 84,
                module_label(layer['attn']), "#000000", "attn", monochrome=True)
    arrow(bx + bw / 2, y + 111, bx + bw / 2, y + 173, color="#000000")
    module_chip(bx + 28, y + 174, bw - 56, 184,
                module_label(layer['ffn']), "#000000", "ffn", monochrome=True,
                detail="384 → 1,536 → 384\n(GELU activation)")
    if index < 7:
        arrow(bx + bw / 2, y + 387, bx + bw / 2, y + 459, color="#000000")

# C: the full execution schedule, including both boundaries and the order flip.
input_output(cx, cw, CODEX["model"]["d_model"], codex=True)

row_records = []
flow_edges = []
for index, layer in enumerate(SPEC["layers"]):
    y = ROW_Y + index * ROW_STEP
    reverse = layer.get("order", "attn_ffn") == "attn_ffn"
    order = ("attn", "ffn") if reverse else ("ffn", "attn")
    if index in (0, 36):
        box(cx - 12, y - 5, cw + 24, MODULE_H + 10, fill="#F0F3F6", edge="none", radius=5)
    if reverse:
        box(cx - 12, y - 5, cw + 24, MODULE_H + 10, fill="#F1E9FA", edge=PURPLE, radius=5, lw=1.2)
    txt(cx - 38, y + MODULE_H / 2, f"{index + 1:02d}", size=22, color=PURPLE if reverse else MUTED,
        weight="bold" if reverse else "normal", ha="right")
    for kind, x in zip(order, (cx, cx + MODULE_W + MODULE_GAP)):
        mid = layer[kind]
        module_chip(x, y, MODULE_W, MODULE_H, module_label(mid), module_color(mid), kind,
                    shared_id=mid if "middle" in mid else None)
    arrow(cx + MODULE_W + 1, y + MODULE_H / 2,
          cx + MODULE_W + MODULE_GAP - 1, y + MODULE_H / 2,
          color=PURPLE if reverse else MUTED)
    flow_edges.append({"source": [index + 1, order[0]], "target": [index + 1, order[1]], "kind": "within_row"})
    if index < 36:
        next_layer = SPEC["layers"][index + 1]
        next_first = "attn" if next_layer.get("order", "attn_ffn") == "attn_ffn" else "ffn"
        second_center = cx + MODULE_W + MODULE_GAP + MODULE_W / 2
        first_center = cx + MODULE_W / 2
        gap_center = y + (MODULE_H + ROW_STEP) / 2
        activation_wire([(second_center, y + MODULE_H), (second_center, gap_center),
                         (first_center, gap_center), (first_center, y + ROW_STEP)])
        flow_edges.append({"source": [index + 1, order[1]], "target": [index + 2, next_first], "kind": "between_rows"})
    row_records.append({"block": index + 1, "execution_order": [layer[k] for k in order]})

# Validate text placement before exporting, and keep a machine-readable schedule.
assert min(module_contrasts) >= 7
assert len(flow_edges) == 73
assert sum(e["kind"] == "between_rows" for e in flow_edges) == 36
assert len(shared_weight_labels) == 70
assert bw == cw
assert contrast_with_white(CODEX_BLUE) >= 4.5
assert all(a.get_verticalalignment() == "baseline" for a in codex_title_artists)
assert len({a.get_position()[1] for a in codex_title_artists}) == 1
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas = ax.get_window_extent(renderer)
for artist in all_text:
    bounds = artist.get_window_extent(renderer)
    assert bounds.x0 >= canvas.x0 and bounds.x1 <= canvas.x1, f"Text outside horizontal canvas: {artist.get_text()}"
    assert bounds.y0 >= canvas.y0 and bounds.y1 <= canvas.y1, f"Text outside vertical canvas: {artist.get_text()}"
for artist, x, y, width, height in bounded_text:
    bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bounds.x0 >= x and bounds.x1 <= x + width, f"Text wider than chip: {artist.get_text()}"
    # The inverted y-axis reverses the transformed bounds' vertical ordering.
    assert min(bounds.y0, bounds.y1) >= y and max(bounds.y0, bounds.y1) <= y + height, f"Text taller than chip: {artist.get_text()}"

if args.check_only:
    print(json.dumps({"font_family": FONT_FAMILY, "text_within_canvas": True,
                      "capsule_labels_fit": True, "positions_verified": len(row_records)}))
    plt.close(fig)
    raise SystemExit(0)

OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / args.stem
metadata = {"Title": "Baseline and Codex-optimized Architecture", "Creator": "NAS architecture figure renderer"}
fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=PAPER)
fig.savefig(stem.with_suffix(".svg"), facecolor=PAPER, metadata=metadata)
fig.savefig(stem.with_suffix(".pdf"), facecolor=PAPER, metadata=metadata)
preview = Path(tempfile.gettempdir()) / f"nas-{args.stem}-preview.png"
fig.savefig(preview, dpi=90, facecolor=PAPER)
plt.close(fig)

verification = {
    "baseline_parameters": BASE["parameters"],
    "candidate_c_parameters": CODEX["parameters"],
    "parameter_reduction_percent": reduction,
    "baseline_blocks": len(BASE["spec"]["layers"]),
    "candidate_c_blocks": len(SPEC["layers"]),
    "attention_palette": dict(zip([f"Attn {i}" for i in range(5)], ATTN)),
    "ffn_palette": dict(zip([f"FFN {i}" for i in range(7)], FFN)),
    "module_type_styles": {
        "attn": "pale fill, rounded corners, colored outline, dark lettering",
        "ffn": "strong rainbow fill, nearly square corners, dark lettering",
        "scope": "candidate C; baseline uses white attention and dark space-gray FFN blocks",
    },
    "architecture_title_font_size": TITLE_SIZE,
    "parameter_count_font_size": TITLE_SIZE,
    "codex_accent_color": CODEX_BLUE,
    "codex_title_spans": [{"text": "Codex", "color": CODEX_BLUE},
                          {"text": "-optimized Architecture", "color": "#000000"}],
    "codex_parameter_count_color": CODEX_BLUE,
    "codex_title_alignment": "shared baseline",
    "codex_title_baseline_y": title_baseline_y,
    "codex_title_contrast_ratio": contrast_with_white(CODEX_BLUE),
    "module_label_font_size": MODULE_FONT_SIZE,
    "shared_weight_font_size": WEIGHT_FONT_SIZE,
    "architecture_body_widths": {"baseline": bw, "codex": cw},
    "architecture_column_widths": {"baseline": W / 2, "codex": W / 2},
    "legend_column": False,
    "order_swap_callout": False,
    "order_swap_row_highlight": True,
    "baseline_palette": {"attn": {"fill": "#FFFFFF", "text": "#000000"},
                         "ffn": {"fill": SPACE_GRAY, "text": "#FFFFFF"}},
    "shared_weight_labels": shared_weight_labels,
    "candidate_c_execution_order": row_records,
    "candidate_c_activation_edges": flow_edges,
    "candidate_c_entry": "embedding output to block 1 FFN in",
    "candidate_c_exit": "block 37 Attn out to final LayerNorm",
    "png_dimensions": [W * 3, H * 3],
    "font_family": FONT_FAMILY,
    "font_files": FONT_FILES,
    "weight_math_font": "STIXGeneral",
    "text_within_canvas": True,
    "capsule_labels_fit": True,
    "minimum_module_label_contrast_ratio": min(module_contrasts),
    "preview": str(preview),
}
verification_path = (OUT / "architecture_comparison.verification.json"
                     if OUT == ROOT / "figures" and args.stem == "baseline_vs_codex_architecture"
                     else stem.with_suffix(".verification.json"))
verification_path.write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps({"outputs": [str(stem.with_suffix(ext)) for ext in (".png", ".svg", ".pdf")],
                  "preview": str(preview), "positions_verified": len(row_records)}, indent=2))
