"""Export an exact architecture diagram and independently reconciled parameter table.

Default input is the qualified frozen primary in ROOT/selection.json. Before a
selection is frozen, both --spec SPEC.json and --frame FRAME.json are required.
Only ROOT/architecture-report is written; no model checkpoint or dataset is read.

python cuda_minparams_architecture_report.py --root runs/cuda/minparams-20261111
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def relative_link(path, label, output):
    if not path or not Path(path).exists():
        return "unavailable locally"
    return f"[{label}]({quote(os.path.relpath(path, output), safe='/.-_')})"


def local_spec(root, given, architecture_id):
    if given:
        path = Path(given)
        candidate = path if path.is_absolute() else root / path
        if candidate.resolve().is_relative_to(root) and candidate.exists():
            return candidate
    candidate = root / "specs" / (architecture_id + ".json")
    return candidate if candidate.exists() else None


def inputs(root, spec_path=None, frame_path=None):
    if bool(spec_path) != bool(frame_path):
        raise ValueError("both --spec and --frame are required for an explicit architecture")
    if spec_path:
        spec_path, frame_path = Path(spec_path).resolve(), Path(frame_path).resolve()
        frame = json.loads(frame_path.read_text())
        if isinstance(frame, dict) and "model" in frame:
            frame = frame["model"]
        elif isinstance(frame, dict) and "frame" in frame:
            frame = frame["frame"]
        return frame, json.loads(spec_path.read_text()), {
            "mode": "explicit_spec_and_frame", "spec_path": str(spec_path), "frame_path": str(frame_path),
            "source_sha256": {str(spec_path): sha256(spec_path), str(frame_path): sha256(frame_path)},
            "selection_used": False}
    selection_path = root / "selection.json"
    if not selection_path.exists():
        raise ValueError("no frozen selection exists; supply both --spec and --frame for an explicit inspection")
    selection = json.loads(selection_path.read_text())
    if (selection.get("status") != "qualified" or not selection.get("primary")
            or not selection.get("frozen_utc") or selection.get("validation_used") is not False):
        raise ValueError("selection.json must contain a qualified frozen architecture selected without test data")
    primary = selection["primary"]
    if selection.get("primary_architecture_id") != primary.get("architecture_id"):
        raise ValueError("frozen primary architecture IDs disagree")
    frame, spec = primary["frame"], primary["spec"]
    spec_path = local_spec(root, primary.get("spec_path"), primary["architecture_id"])
    from nas.cuda_minparams_search import architecture_hash
    if architecture_hash(frame, spec) != primary["architecture_id"]:
        raise ValueError("frozen architecture ID differs from its frame/specification")
    if spec_path and architecture_hash(frame, json.loads(spec_path.read_text())) != primary["architecture_id"]:
        raise ValueError("local specification differs from the frozen architecture")
    return frame, spec, {"mode": "frozen_primary", "selection_used": True, "selection_path": str(selection_path),
                         "frozen_utc": selection["frozen_utc"], "architecture_id": primary["architecture_id"],
                         "expected_parameters": primary["parameters"], "spec_path": str(spec_path) if spec_path else None,
                         "source_sha256": {str(selection_path): sha256(selection_path),
                                           **({str(spec_path): sha256(spec_path)} if spec_path else {})}}


def options(spec, key, frame):
    given = spec.get("ffns", {}).get(key, {})
    return {"hidden": given.get("hidden", frame["d_ff"]), "activation": given.get("act", "gelu"),
            "transpose_tie": given.get("transpose_tie", False)}


def scale_options(layer, kind, index):
    given = layer.get(kind + "_scale", 1.)
    if isinstance(given, dict):
        initial, learned = given["init"], given["learned"]
    else:
        initial, learned = given, False
    if type(learned) is not bool or isinstance(initial, bool) or not isinstance(initial, (int, float)) or not math.isfinite(initial):
        raise ValueError("residual scales must have a finite initial value and boolean learned flag")
    return {"key": f"L{index}_{kind}", "initial_value": float(initial), "learned": learned,
            "unique_parameters": int(learned)}


def describe(frame, spec):
    """Independent counting by unique tensors; repeated applications add no weights."""
    for key in ("vocab_size", "max_len", "n_layers", "d_model", "n_heads", "d_ff"):
        if type(frame.get(key)) is not int or frame[key] <= 0:
            raise ValueError(f"positive integer {key} is required")
    if frame["d_model"] % frame["n_heads"]:
        raise ValueError("model width must be divisible by attention head count")
    if not spec.get("layers"):
        raise ValueError("architecture must have at least one execution entry")
    d, rows, modules, execution, scalars = frame["d_model"], [], {"attention": {}, "ffn": {}}, [], []
    calls = {"attention": Counter(), "ffn": Counter()}

    def allocate(category, module, tensor, shape, count, note=""):
        rows.append({"category": category, "module": module, "tensor": tensor, "shape": shape,
                     "unique_parameters": count, "note": note})

    allocate("embedding", "E", "token/output embedding", [frame["vocab_size"], d], frame["vocab_size"] * d,
             "Output logits reuse E transposed; output projection has no additional parameters.")
    allocate("embedding", "P", "learned position embedding", [frame["max_len"], d], frame["max_len"] * d)
    allocate("final_norm", "final_LN", "weight + bias", [[d], [d]], 2 * d)
    for index, layer in enumerate(spec["layers"]):
        repeats = layer.get("loop", 1)
        if type(repeats) is not int or repeats < 1:
            raise ValueError("positive integer loop count required")
        inject = layer.get("inject", False)
        if type(inject) is not bool or (inject and repeats == 1):
            raise ValueError("input reinjection requires a repeated execution entry")
        order = layer.get("order", "attn_ffn")
        if order not in ("attn_ffn", "ffn_attn"):
            raise ValueError("unsupported sublayer execution order")
        branches = []
        for kind in (("ffn", "attn") if order == "ffn_attn" else ("attn", "ffn")):
            key = layer.get(kind)
            if key is None:
                continue
            if not isinstance(key, str) or not key or "." in key:
                raise ValueError("module IDs must be nonempty strings without periods")
            category = "attention" if kind == "attn" else "ffn"
            calls[category][key] += repeats
            scale = scale_options(layer, kind, index)
            scalars.append({**scale, "entry": index + 1, "branch": kind, "module_id": key})
            if key not in modules[category]:
                if kind == "attn":
                    modules[category][key] = {"id": key, "kind": "attention", "width": d,
                                             "heads": frame["n_heads"], "head_width": d // frame["n_heads"],
                                             "layernorm_parameters": 2 * d, "unique_parameters": 4 * d * d + 2 * d}
                    allocate(category, key, "shared pre-LayerNorm weight + bias", [[d], [d]], 2 * d)
                    allocate(category, key, "QKV weight", [3 * d, d], 3 * d * d, "No linear bias.")
                    allocate(category, key, "attention output weight", [d, d], d * d, "No linear bias.")
                else:
                    opts = options(spec, key, frame)
                    hidden, tied = opts["hidden"], opts["transpose_tie"]
                    if type(hidden) is not int or hidden < 1 or type(tied) is not bool:
                        raise ValueError("FFN hidden width and transpose-tie flag are invalid")
                    modules[category][key] = {"id": key, "kind": "ffn", "width": d, **opts,
                                             "layernorm_parameters": 2 * d,
                                             "unique_parameters": 2 * d + d * hidden * (1 if tied else 2)}
                    allocate(category, key, "shared pre-LayerNorm weight + bias", [[d], [d]], 2 * d)
                    allocate(category, key, "up-projection weight", [hidden, d], hidden * d, "No linear bias.")
                    allocate(category, key, "down-projection weight", [d, hidden], 0 if tied else hidden * d,
                             "Transpose of this FFN's up-projection; no additional weights." if tied else "No linear bias.")
            if scale["learned"]:
                allocate("residual_scalar", scale["key"], "learned branch scalar", [], 1,
                         f"Initial value {scale['initial_value']:g}; reused on all {repeats} repetitions of this entry.")
            branches.append({"kind": category, "module_id": key, "scale": scale})
        if not branches:
            raise ValueError("an execution entry cannot have both branches deleted")
        execution.append({"entry": index + 1, "loop": repeats, "reinject_original_entry_input": inject, "branches": branches})
    for kind in modules:
        for key, module in modules[kind].items():
            module["execution_calls"] = calls[kind][key]
            module["shared_weights_and_layernorm"] = calls[kind][key] > 1
    totals = Counter()
    for row in rows:
        totals[row["category"]] += row["unique_parameters"]
    total = sum(row["unique_parameters"] for row in rows)
    return {"name": spec.get("name", "unnamed"), "frame": frame, "spec": spec, "modules": modules,
            "execution": execution, "residual_scalars": scalars, "parameter_allocation": rows,
            "parameters_by_category": dict(totals), "total_unique_parameters": total,
            "execution_entries": len(execution), "entry_applications": sum(r["loop"] for r in execution),
            "attention_calls": sum(calls["attention"].values()), "ffn_calls": sum(calls["ffn"].values()),
            "layernorm_epsilon": 1e-5, "attention_probability_dropout": 0.,
            "token_embedding_tied_to_output": True, "output_projection_additional_parameters": 0}


def reconcile(description, expected=None):
    frame, spec, total = description["frame"], description["spec"], description["total_unique_parameters"]
    from nas.cuda_minparams_search import parameter_count, architecture_hash
    import nas.cuda_minparams_search as search_implementation
    spec_count = parameter_count(frame, spec)
    result = {"independent_allocation_sum": total, "scheduler_spec_count": spec_count,
              "expected_frozen_count": expected, "architecture_id": architecture_hash(frame, spec),
              "implementation_sha256": {"cuda_minparams_search.py": sha256(search_implementation.__file__)}}
    if total != spec_count or (expected is not None and total != expected):
        raise ValueError(f"parameter accounting disagreement: {result}")
    try:
        import torch
        import nas.cuda_model as model_implementation
        from nas.cuda_model import Config, SpecGPT, count_parameters, spec_parameter_count
    except ImportError as error:
        result.update(actual_module_count=None, actual_model_check="unavailable: " + str(error))
    else:
        config = Config(**frame)
        # Meta construction counts real module Parameter objects without allocating weights.
        with torch.device("meta"):
            model = SpecGPT(spec=spec, config=config)
        result.update(actual_module_count=count_parameters(model), model_spec_count=spec_parameter_count(spec, config),
                      actual_model_check="passed using actual modules on the meta device", torch_version=torch.__version__)
        result["implementation_sha256"]["cuda_model.py"] = sha256(model_implementation.__file__)
        if result["actual_module_count"] != total or result["model_spec_count"] != total:
            raise ValueError(f"actual model parameter accounting disagreement: {result}")
    result["passed"] = True
    return result


def ascii_diagram(description):
    frame, d = description["frame"], description["frame"]["d_model"]
    token_input = f"Tokens ----> E: token embedding [{frame['vocab_size']} x {d}]"
    position_input = f"Positions -> P: learned positions [{frame['max_len']} x {d}]"
    input_width = max(len(token_input), len(position_input))
    lines = [description["name"], "=" * len(description["name"]),
             f"Unique learned parameters: {description['total_unique_parameters']:,}",
             f"{description['execution_entries']} execution entries; {description['entry_applications']} entry applications; "
             f"{description['attention_calls']} attention calls; {description['ffn_calls']} FFN calls.", "",
             f"{token_input:<{input_width}} --+",
             f"{position_input:<{input_width}} --+--> add --> Dropout(p={frame['dropout']:g}) --> x", "",
             "Each named attention/FFN module owns its pre-LayerNorm. Reusing its name reuses both weights and LayerNorm.", ""]
    for entry in description["execution"]:
        loops = entry["loop"]
        lines.append(f"+-- Entry {entry['entry']:02d}: execute the ordered branches below {loops} time(s)")
        if entry["reinject_original_entry_input"]:
            lines.append("|   Save x_entry = x before iteration 1.")
            lines.append(f"|   Before each iteration 2..{loops}: x <- x + x_entry")
        for branch in entry["branches"]:
            key, kind, scale = branch["module_id"], branch["kind"], branch["scale"]
            if kind == "attention":
                operation = f"Attention[{key}]({frame['n_heads']} heads x {d//frame['n_heads']})"
            else:
                module = description["modules"]["ffn"][key]
                operation = f"FFN[{key}]({d} -> {module['hidden']} {module['activation']} -> {d})"
                if module["transpose_tie"]:
                    operation += " [down = up.T]"
            coefficient = f"s_{scale['key']} [learned, init={scale['initial_value']:g}]" if scale["learned"] else f"{scale['initial_value']:g}"
            lines.extend([f"|   x --> LN[{key}] --> {operation} --> Dropout(p={frame['dropout']:g})",
                          f"|   x <- x + ({coefficient}) * branch_output"])
        lines.append("+-- end entry\n")
    lines.extend([f"x --> final LayerNorm[{d}] --> logits = x @ E.T --> [{frame['vocab_size']} logits]",
                  "                                           ^",
                  "                          same token embedding E; zero new output weights", "",
                  "Learned scale values above are initializations; trained values are checkpoint-specific.",
                  "All linear projections have no bias. LayerNorm includes its learned weight and bias.",
                  "Attention uses causal, conversation-isolated masking; position indices restart for each conversation.", ""])
    return "\n".join(lines)


def markdown(description, diagram, provenance, output):
    total = description["total_unique_parameters"]
    lines = ["# Exact architecture and parameter allocation", "",
             f"**{description['name']}** contains **{total:,} unique trainable parameters**. "
             "Shared parameters are counted once, regardless of execution frequency. "
             "Fixed residual scalars, dropout and attention masks add no learned parameters.", "",
             "```text", diagram.rstrip(), "```", "", "## Unique modules", "",
             "| Kind | ID | Geometry / activation | Calls per forward pass | Unique parameters, including shared LayerNorm |",
             "|---|---|---|---:|---:|"]
    for kind, modules in description["modules"].items():
        for key, module in modules.items():
            if kind == "attention":
                geometry = f"width {module['width']}; {module['heads']} heads × {module['head_width']}"
            else:
                geometry = f"{module['width']} → {module['hidden']} → {module['width']}; {module['activation']}"
                if module["transpose_tie"]:
                    geometry += "; down = upᵀ"
            lines.append(f"| {kind} | `{key}` | {geometry} | {module['execution_calls']} | {module['unique_parameters']:,} |")
    lines += ["", "## Parameter allocation", "", "| Category | Module | Tensor | Shape | Unique parameters | Share of total | Notes |",
              "|---|---|---|---|---:|---:|---|"]
    for row in description["parameter_allocation"]:
        lines.append(f"| {row['category']} | `{row['module']}` | {row['tensor']} | `{json.dumps(row['shape'])}` | "
                     f"{row['unique_parameters']:,} | {row['unique_parameters']/total:.2%} | {row['note']} |")
    lines += [f"| **Total** | | | | **{total:,}** | **100%** | |", "", "## Accounting checks", ""]
    checks = description["reconciliation"]
    lines += [f"- Independent tensor-allocation sum: **{checks['independent_allocation_sum']:,}**.",
              f"- Scheduler specification count: **{checks['scheduler_spec_count']:,}**."]
    if checks.get("actual_module_count") is not None:
        lines.append(f"- Actual model Parameter objects on the meta device: **{checks['actual_module_count']:,}**.")
        lines.append(f"- Model specification count: **{checks['model_spec_count']:,}**.")
    else:
        lines.append("- Actual model construction was unavailable in this Python environment: " + checks["actual_model_check"])
    if checks.get("expected_frozen_count") is not None:
        lines.append(f"- Frozen selection count: **{checks['expected_frozen_count']:,}**.")
    lines += ["", "## Sources", "", "- Specification: " + relative_link(provenance.get("spec_path"), "spec JSON", output)]
    if provenance.get("selection_path"):
        lines.append("- Frozen selection: " + relative_link(provenance["selection_path"], "selection.json", output))
    if provenance.get("frame_path"):
        lines.append("- Frame: " + relative_link(provenance["frame_path"], "frame JSON", output))
    lines += ["- Machine-readable accounting: [ARCHITECTURE.json](ARCHITECTURE.json).",
              "- Plain ASCII diagram: [architecture.txt](architecture.txt).", "",
              "This report describes architecture and initialization settings. It does not load checkpoints, training examples, or held-out results.", ""]
    return "\n".join(lines)


def export(root, spec_path=None, frame_path=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("run root does not exist")
    frame, spec, provenance = inputs(root, spec_path, frame_path)
    description = describe(frame, spec)
    description["reconciliation"] = reconcile(description, provenance.get("expected_parameters"))
    description["provenance"] = provenance
    description["exported_utc"] = datetime.now(timezone.utc).isoformat()
    description["exporter_sha256"] = sha256(__file__)
    description["data_or_checkpoints_accessed"] = False
    for path, digest in provenance["source_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("an architecture input changed while the report was generated")
    diagram = ascii_diagram(description)
    # Make accidental Unicode in labels explicit; the structural diagram remains portable ASCII.
    diagram = diagram.encode("ascii", errors="backslashreplace").decode("ascii")
    output = root / "architecture-report"
    output.mkdir(exist_ok=True)
    atomic_text(output / "architecture.txt", diagram)
    atomic_text(output / "ARCHITECTURE.md", markdown(description, diagram, provenance, output))
    atomic_text(output / "ARCHITECTURE.json", json.dumps(description, indent=2, allow_nan=False) + "\n")
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--frame", type=Path)
    args = parser.parse_args()
    description = export(args.root, args.spec, args.frame)
    print(json.dumps({"status": "exported", "name": description["name"],
                      "parameters": description["total_unique_parameters"], "reconciliation": description["reconciliation"],
                      "output": str(args.root / "architecture-report")}), flush=True)


if __name__ == "__main__":
    main()
