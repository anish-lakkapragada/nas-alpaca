"""Read-only audit export for the minimum-parameter search.

Reads existing ledgers, run metrics, and ranking/freeze metadata. Never loads
models, checkpoints, datasets, or test results; never changes execution files.
Only RUN_LOG.md, ARCHITECTURES.md, SEARCH_SUMMARY.json, and search_progress
PNG/SVG are written, under --out (default: ROOT/audit-report).

python cuda_minparams_report.py --root runs/cuda/minparams-20261111
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from urllib.parse import quote


SEARCH_SEEDS = (0, 1, 2)
BASELINE_PARAMETERS = 27_144_960


def stamp():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "—"


def cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.captured_utc = stamp()
        self.input_sha256, self.warnings = {}, []

    def bytes(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        if not path.resolve().is_relative_to(self.root):
            self.warnings.append(f"Ignored a source path outside the run root: {path}")
            return None
        raw = path.read_bytes()
        self.input_sha256[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, path, default=None):
        raw = self.bytes(path)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            self.warnings.append(f"Could not parse {Path(path).name}: {error}")
            return default

    def jsonl(self, path):
        raw = self.bytes(path)
        if raw is None:
            return []
        lines, rows = raw.splitlines(), []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                rows.append(row)
            except (ValueError, UnicodeDecodeError) as error:
                tail = "possibly still being written" if index == len(lines) else "malformed interior record"
                self.warnings.append(f"{Path(path).name}:{index}: {tail}; ignored without modifying source ({error})")
        return rows

    def artifact(self, given):
        """Resolve artifacts safely, including downloaded logs with remote paths."""
        if not given:
            return None
        path = Path(given)
        candidate = path if path.is_absolute() else self.root / path
        if candidate.resolve().is_relative_to(self.root):
            return candidate
        for directory in ("attempts", "specs", "logs"):
            if directory in path.parts:
                index = len(path.parts) - 1 - list(reversed(path.parts)).index(directory)
                relocated = self.root.joinpath(*path.parts[index:])
                if relocated.resolve().is_relative_to(self.root):
                    return relocated
        return None


def link(path, label, output):
    if path is None or not Path(path).exists():
        return "—"
    relative = os.path.relpath(path, output)
    return f"[{cell(label)}]({quote(relative, safe='/.-_')})"


def current_runs(snapshot):
    records, history = {}, defaultdict(list)
    for row in snapshot.jsonl(snapshot.root / "run_registry.jsonl"):
        run_id = row.get("job", {}).get("run_id")
        if not run_id:
            snapshot.warnings.append("A run-ledger row without job.run_id was ignored")
            continue
        records[run_id] = {**records.get(run_id, {}), **row}
        if not history[run_id] or history[run_id][-1] != row.get("status"):
            history[run_id].append(row.get("status", "unknown"))
    rows = []
    for run_id, record in records.items():
        folder = snapshot.artifact(record.get("path")) or snapshot.root / "attempts" / run_id
        summary = record.get("summary") or snapshot.json(folder / "summary.json", {})
        metrics = record.get("metrics") or {}
        config = {}
        if not record.get("parameters") or not record.get("host") or not record.get("frame"):
            config = snapshot.json(folder / "config.json", {})
        best = {"step": summary.get("best_step", metrics.get("best_step")),
                "dev_loss": summary.get("best_dev_loss", metrics.get("best_dev_loss")),
                "train_eval_loss": summary.get("train_eval_loss_at_best_dev", metrics.get("train_eval_loss_at_best_dev"))}
        if not finite(best["dev_loss"]) or not finite(best["train_eval_loss"]):
            evaluations = [r for r in snapshot.jsonl(folder / "log.jsonl")
                           if r.get("event") == "evaluation" and finite(r.get("dev_loss"))]
            if evaluations:
                best = min(evaluations, key=lambda r: r["dev_loss"])
        host = record.get("host") or config.get("hardware", {})
        gpu = record.get("physical_gpu", {})
        if not isinstance(gpu, dict):
            gpu = {"physical_index": gpu}
        row = {"run_id": run_id, "architecture_id": record.get("job", {}).get("architecture_id", record.get("architecture_sha256")),
               "name": record.get("job", {}).get("spec", config.get("spec", {}).get("name")),
               "seed": record.get("job", {}).get("seed", config.get("args", {}).get("seed")),
               "phase": record.get("job", {}).get("phase"), "status": record.get("status", "unknown"),
               "status_history": history[run_id], "parameters": record.get("parameters", summary.get("parameters", config.get("parameters"))),
               "frame": record.get("frame", config.get("model")), "queued_utc": record.get("queued_utc"),
               "started_utc": record.get("started_utc"), "ended_utc": record.get("ended_utc"),
               "updated_utc": record.get("updated_utc"), "hostname": host.get("hostname"), "public_ip": host.get("public_ip"),
               "gpu": gpu.get("physical_index", config.get("hardware", {}).get("physical_gpu_ids")), "gpu_uuid": gpu.get("uuid"),
               "steps": summary.get("steps"), "best_step": best.get("step"), "best_dev_loss": best.get("dev_loss"),
               "train_diagnostic_ce_at_best_dev": best.get("train_eval_loss"),
               "train_seconds": summary.get("train_seconds"), "wall_seconds": record.get("wall_seconds"),
               "reason": record.get("failure_reason") or record.get("stop_reason"),
               "path": str(folder), "spec_path": record.get("spec_path"),
               "summary": summary}
        rows.append(row)
    return sorted(rows, key=lambda r: (r["queued_utc"] or r["started_utc"] or r["updated_utc"] or "", r["run_id"]))


def architecture_rows(snapshot):
    records = snapshot.jsonl(snapshot.root / "architecture_registry.jsonl")
    rows = []
    for index, record in enumerate(records, 1):
        row = dict(record)
        row["ledger_record"] = index
        row["status"] = row.get("status", "accepted")
        if row.get("parameters") is None and row["status"] == "rejected":
            # The scheduler sometimes reports the computed oversize total only in its reason.
            import re
            match = re.search(r"\btotal (\d+) must be strictly below", row.get("reason", ""))
            if match:
                row["parameters"] = int(match.group(1))
                row["parameter_count_source"] = "scheduler rejection message"
        rows.append(row)
    return rows


def measured_architectures(runs, architectures, snapshot):
    by_id = {a.get("architecture_id"): a for a in architectures if a.get("status") == "accepted"}
    grouped = defaultdict(dict)
    for run in runs:
        if (run["status"] != "completed" or run["seed"] not in SEARCH_SEEDS or run["phase"] == "final"
                or run["steps"] != 5000 or run["summary"].get("schedule_steps", 5000) != 5000
                or run["summary"].get("benchmark") or not finite(run["best_dev_loss"])):
            continue
        slots = grouped[run["architecture_id"]]
        if run["seed"] in slots:
            snapshot.warnings.append(f"Duplicate completed architecture/seed: {run['architecture_id']} seed {run['seed']}; plot uses first completed attempt")
            continue
        slots[run["seed"]] = run
    baseline_id = next((a.get("architecture_id") for a in architectures
                        if a.get("name") == "baseline" and a.get("parameters") == BASELINE_PARAMETERS), None)
    baseline = grouped.get(baseline_id, {})
    if baseline_id is None:
        baseline_id = next((r["architecture_id"] for r in runs
                            if r["name"] == "baseline" and r["parameters"] == BASELINE_PARAMETERS), None)
        baseline = grouped.get(baseline_id, {})
    measurements = []
    for digest, seed_runs in grouped.items():
        first = next(iter(seed_runs.values()))
        values = [seed_runs[s]["best_dev_loss"] for s in sorted(seed_runs)]
        complete = set(seed_runs) == set(SEARCH_SEEDS)
        pairs_complete = complete and set(baseline) == set(SEARCH_SEEDS)
        delta = statistics.mean(seed_runs[s]["best_dev_loss"] - baseline[s]["best_dev_loss"] for s in SEARCH_SEEDS) if pairs_complete else None
        architecture = by_id.get(digest, {})
        measurements.append({"architecture_id": digest, "name": architecture.get("name", first["name"]),
                             "parameters": architecture.get("parameters", first["parameters"]),
                             "is_baseline": digest == baseline_id, "seeds": sorted(seed_runs),
                             "dev_losses": {str(s): seed_runs[s]["best_dev_loss"] for s in sorted(seed_runs)},
                             "mean_dev_ce": statistics.mean(values), "std_dev_ce": statistics.stdev(values) if len(values) > 1 else None,
                             "three_seed_confirmed": complete, "mean_paired_delta": delta,
                             "qualified": bool(digest != baseline_id and pairs_complete and delta <= 0
                                               and first["parameters"] < BASELINE_PARAMETERS)})
    return sorted(measurements, key=lambda r: (r["parameters"], r["mean_dev_ce"]))


def run_log(runs, snapshot, output):
    lines = ["# Every recorded training attempt", "", f"Snapshot: **{snapshot.captured_utc}**. All timestamps are UTC.", "",
             "Every distinct run ID appears below, including failures, interruptions, queued jobs and jobs that were not admitted. "
             "Train CE is the fixed 256-conversation training diagnostic at the best recorded dev checkpoint. "
             "Dev CE uses all 1,000 dev conversations. Incomplete attempts can have partial metrics; their status and completed steps remain visible. "
             "Wall time includes process overhead; training time comes from the trainer summary. Final-confirmation runs are listed separately by phase and seed, without contributing to search rankings.", "",
             "| Run / architecture | Seed / phase | Parameters | UTC queued / started / ended | Host / GPU | Status / transitions | Completed steps / best step | Train diagnostic CE | Dev CE | Wall / training seconds | Spec / curves / logs | Failure or interruption reason |",
             "|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|"]
    for run in runs:
        folder = Path(run["path"])
        spec = snapshot.artifact(run["spec_path"])
        links = [link(spec, "spec", output), link(folder / "curves.png", "PNG", output), link(folder / "curves.svg", "SVG", output),
                 link(folder / "log.jsonl", "metrics", output), link(snapshot.root / "logs" / (run["run_id"] + ".log"), "stdout", output)]
        links = [value for value in links if value != "—"]
        times = "<br>".join(cell(run[k]) for k in ("queued_utc", "started_utc", "ended_utc"))
        host = cell(run["hostname"]) + (f" ({cell(run['public_ip'])})" if run["public_ip"] else "")
        host += "<br>GPU " + cell(run["gpu"])
        if run["gpu_uuid"]:
            host += "<br>" + cell(run["gpu_uuid"])
        identity = link(folder, run["run_id"], output) if folder.exists() else "`" + cell(run["run_id"]) + "`"
        identity += "<br>" + cell(run["name"]) + "<br>`" + cell(run["architecture_id"]) + "`"
        status = "**" + cell(run["status"]) + "**<br>" + " → ".join(map(cell, run["status_history"]))
        parameters = f"{run['parameters']:,}" if isinstance(run["parameters"], int) else "—"
        lines.append("| " + " | ".join([identity, f"{cell(run['seed'])}<br>{cell(run['phase'])}", parameters, times, host, status,
                     f"{cell(run['steps'])} / {cell(run['best_step'])}", fmt(run["train_diagnostic_ce_at_best_dev"]), fmt(run["best_dev_loss"]),
                     f"{fmt(run['wall_seconds'], 1)} / {fmt(run['train_seconds'], 1)}", " · ".join(links) or "—", cell(run["reason"])]) + " |")
    lines += ["", "Raw append-only ledger: " + link(snapshot.root / "run_registry.jsonl", "run_registry.jsonl", output), ""]
    return "\n".join(lines)


def architecture_log(architectures, measurements, snapshot, output):
    measured = {row["architecture_id"]: row for row in measurements}
    lines = ["# Every registered architecture", "", f"Snapshot: **{snapshot.captured_utc}**.", "",
             "Every accepted or rejected architecture-registry entry appears below. Depth is the number of executed layer-entry applications; "
             "entries may contain one sublayer, two sublayers, or shared repetitions. Parameter totals count unique trainable weights. "
             "A rejected malformed specification may have no meaningful parameter count. Seeds and per-seed dev losses are explicit; "
             "only completed search seeds 0, 1 and 2 can establish three-seed qualification.", "",
             "| Record / architecture ID | Name / status | Registered UTC | Parameters | Frame: depth / width / heads / default FFN | Parent IDs / operator | Search measurements | Specification / rejection reason |",
             "|---|---|---|---:|---|---|---|---|"]
    rejected = []
    for record in architectures:
        digest = record.get("architecture_id", "missing-ID")
        frame = record.get("frame") or {}
        if not isinstance(frame, dict):
            frame = {}
        params = record.get("parameters")
        params = f"{params:,}" if isinstance(params, int) else "unavailable"
        shape = " / ".join(cell(frame.get(k)) for k in ("n_layers", "d_model", "n_heads", "d_ff"))
        shape += "<br>vocab " + cell(frame.get("vocab_size")) + "; context " + cell(frame.get("max_len")) + "; dropout " + cell(frame.get("dropout"))
        parents = "<br>".join("`" + cell(p) + "`" for p in record.get("parents", [])) or "—"
        parents += "<br>" + cell(record.get("operator")) + f"; generation {cell(record.get('generation'))}"
        row = measured.get(digest)
        observation = "No completed 5,000-step search measurements"
        if row:
            observation = "; ".join(f"s{seed}: {value:.4f}" for seed, value in row["dev_losses"].items())
            if row["three_seed_confirmed"]:
                observation += f"<br>3-seed mean {row['mean_dev_ce']:.4f}, SD {row['std_dev_ce']:.4f}"
                if not row["is_baseline"]:
                    observation += "<br>" + ("**qualified**" if row["qualified"] else "not qualified")
            else:
                observation += f"<br>{len(row['seeds'])} seed(s); confirmation incomplete"
        specification = link(snapshot.artifact(record.get("spec_path")), "spec JSON", output)
        if record["status"] == "rejected":
            anchor = f"rejected-{record['ledger_record']}"
            specification = f"[Rejected specification](#{anchor})<br>" + cell(record.get("reason"))
            rejected.append((anchor, record))
        lines.append("| " + " | ".join([f"{record['ledger_record']}<br>`{cell(digest)}`",
                     cell(record.get("name")) + "<br>**" + cell(record["status"]) + "**", cell(record.get("created_utc")),
                     params, shape, parents, observation, specification]) + " |")
    for anchor, record in rejected:
        lines.extend(["", f'<a id="{anchor}"></a>', "", f"## Rejected: {cell(record.get('name'))}", "",
                      cell(record.get("reason")), "", "```json", json.dumps({"frame": record.get("frame"), "spec": record.get("spec")}, indent=2), "```"])
    lines += ["", "Raw append-only ledger: " + link(snapshot.root / "architecture_registry.jsonl", "architecture_registry.jsonl", output), ""]
    return "\n".join(lines)


def progress_plot(measurements, output, captured_utc):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"nas-report-matplotlib-{os.getuid()}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
    baseline = next((r for r in measurements if r["is_baseline"]), None)
    screening = [r for r in measurements if not r["is_baseline"] and "0" in r["dev_losses"]]
    confirmed = [r for r in measurements if r["three_seed_confirmed"] and not r["is_baseline"]]
    qualified = [r for r in confirmed if r["qualified"]]
    axes[0].scatter([r["parameters"] / 1e6 for r in screening], [r["dev_losses"]["0"] for r in screening],
                    s=25, alpha=.65, c="#667085", edgecolors="none", label=f"Seed 0 screening ({len(screening)})")
    screened_confirmed = [r for r in confirmed if "0" in r["dev_losses"]]
    axes[0].scatter([r["parameters"] / 1e6 for r in screened_confirmed], [r["dev_losses"]["0"] for r in screened_confirmed],
                    s=46, facecolors="none", edgecolors="#1677a5", linewidths=1.1, label="Also confirmed on seeds 1, 2")
    for rows, color, marker, label in (([r for r in confirmed if not r["qualified"]], "#c06b27", "o", "Above baseline mean"),
                                      (qualified, "#087f5b", "D", "Matches / beats baseline mean")):
        if rows:
            axes[1].errorbar([r["parameters"] / 1e6 for r in rows], [r["mean_dev_ce"] for r in rows],
                             yerr=[r["std_dev_ce"] for r in rows], fmt=marker, markersize=5,
                             capsize=3, elinewidth=1, alpha=.9, color=color, label=f"{label} ({len(rows)})")
    if baseline and "0" in baseline["dev_losses"]:
        value = baseline["dev_losses"]["0"]
        axes[0].axhline(value, color="#20262e", linestyle="--", linewidth=1.2, label=f"Baseline seed 0: {value:.4f}")
        axes[0].scatter([baseline["parameters"] / 1e6], [value], marker="*", color="#20262e", s=110, zorder=5)
    if baseline and baseline["three_seed_confirmed"]:
        axes[1].axhline(baseline["mean_dev_ce"], color="#20262e", linestyle="--", linewidth=1.2,
                        label=f"Baseline 3-seed mean: {baseline['mean_dev_ce']:.4f}")
        axes[1].errorbar([baseline["parameters"] / 1e6], [baseline["mean_dev_ce"]], yerr=[baseline["std_dev_ce"]],
                         fmt="*", markersize=10, color="#20262e", capsize=3, zorder=5)
    if qualified:
        smallest = min(qualified, key=lambda r: (r["parameters"], r["mean_dev_ce"]))
        reduction = 1 - smallest["parameters"] / BASELINE_PARAMETERS
        axes[1].annotate(f"Smallest qualifier\n{smallest['parameters']/1e6:.2f}M ({reduction:.1%} fewer)",
                         (smallest["parameters"] / 1e6, smallest["mean_dev_ce"]), xytext=(9, -35), textcoords="offset points",
                         fontsize=9, color="#087f5b", arrowprops={"arrowstyle": "-", "color": "#087f5b"})
    if not confirmed:
        axes[1].text(.5, .5, "Three-seed comparisons pending", ha="center", va="center", transform=axes[1].transAxes, color="#667085")
    if not screening:
        axes[0].text(.5, .5, "Completed screening runs pending", ha="center", va="center", transform=axes[0].transAxes, color="#667085")
    axes[0].set_title("Screening: one training seed per architecture")
    axes[1].set_title("Confirmation: mean ± SD over seeds 0, 1, 2")
    for axis in axes:
        axis.set_xlabel("Total trainable parameters (millions)")
        axis.grid(alpha=.18)
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axis.legend(loc="best", framealpha=.95, fontsize=8)
        axis.margins(x=.08, y=.12)
    axes[0].set_ylabel("Best dev answer-token cross-entropy (nats)")
    fig.suptitle("Smaller transformers at fixed data and 5,000 training steps", fontsize=14, y=.985)
    fig.text(.5, .025, "All points use complete 5,000-step runs and dev-selected checkpoints. Error bars: sample standard deviation. No test metrics used.",
             ha="center", fontsize=8, color="#475467")
    fig.text(.99, .005, "Snapshot " + captured_utc, ha="right", fontsize=7, color="#667085")
    fig.tight_layout(rect=(0, .055, 1, .955))
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        destination = output / ("search_progress." + suffix)
        temporary = output / (f".search_progress.{os.getpid()}.tmp." + suffix)
        fig.savefig(temporary, format=suffix, dpi=200, bbox_inches="tight", facecolor="white")
        temporary.replace(destination)
    plt.close(fig)


def export(root, output=None, plots=True):
    snapshot = Snapshot(root)
    if not snapshot.root.is_dir():
        raise ValueError("run root does not exist")
    output = Path(output).resolve() if output else snapshot.root / "audit-report"
    runs = current_runs(snapshot)
    architectures = architecture_rows(snapshot)
    measurements = measured_architectures(runs, architectures, snapshot)
    # Ranking/event metadata is corroborative; all reported measurements come from run records.
    rankings = snapshot.json(snapshot.root / "rankings.json", {}) or {}
    events = snapshot.jsonl(snapshot.root / "events.jsonl")
    selection = snapshot.json(snapshot.root / "selection.json", {}) or {}
    by_id = {m["architecture_id"]: m for m in measurements}
    discrepancies = []
    for ranked in rankings.get("all", []):
        measured = by_id.get(ranked.get("architecture_id"))
        if (measured and ranked.get("seeds") == measured["seeds"] and finite(ranked.get("mean_dev_loss"))
                and abs(ranked["mean_dev_loss"] - measured["mean_dev_ce"]) > 1e-10):
            discrepancies.append(ranked["architecture_id"])
    if discrepancies:
        snapshot.warnings.append(f"Rankings differ from the current completed-run ledger for {len(discrepancies)} architectures; export uses ledger metrics")
    baseline = next((m for m in measurements if m["is_baseline"]), None)
    qualifiers = [m for m in measurements if m["qualified"]]
    statuses = Counter(r["status"] for r in runs)
    summary = {"captured_utc": snapshot.captured_utc, "run_root": str(snapshot.root),
               "metric": "answer-token-weighted mean CE including answer EOS; best dev checkpoint within 5000 steps",
               "training_diagnostic_conversations": 256, "dev_conversations": 1000,
               "test_data_or_metrics_accessed": False, "training_attempts": len(runs), "attempt_statuses": dict(statuses),
               "completed_search_runs": sum(r["status"] == "completed" and r["seed"] in SEARCH_SEEDS and r["phase"] != "final" for r in runs),
               "completed_fresh_confirmation_runs": sum(r["status"] == "completed" and r["phase"] == "final" for r in runs),
               "architecture_registry_entries": len(architectures),
               "architecture_statuses": dict(Counter(a["status"] for a in architectures)),
               "measured_search_architectures": len(measurements),
               "three_seed_confirmed_candidates": sum(m["three_seed_confirmed"] and not m["is_baseline"] for m in measurements),
               "qualified_candidates": len(qualifiers), "baseline": baseline,
               "smallest_qualifying_candidates": qualifiers[:10],
               "search_architectures_without_seed0": [m["architecture_id"] for m in measurements if "0" not in m["dev_losses"]],
               "selection_status": selection.get("status"), "frozen_primary_architecture_id": selection.get("primary_architecture_id"),
               "last_search_event": events[-1] if events else None,
               "limitations": ["Screening points show seed 0 alone; confirmed points show a mean and sample SD over seeds 0, 1, 2.",
                               "Qualification is a search-dev comparison and does not establish held-out test performance.",
                               "Incomplete or failed runs remain in the audit table but do not enter architecture ranking or plots.",
                               "This is a live snapshot; ledgers and derived rankings may have been captured at slightly different instants."],
               "warnings": snapshot.warnings, "input_sha256": snapshot.input_sha256,
               "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "RUN_LOG.md", run_log(runs, snapshot, output))
    atomic_text(output / "ARCHITECTURES.md", architecture_log(architectures, measurements, snapshot, output))
    if plots:
        progress_plot(measurements, output, snapshot.captured_utc)
    atomic_text(output / "SEARCH_SUMMARY.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="write Markdown/JSON only")
    args = parser.parse_args()
    summary = export(args.root, args.out, plots=not args.no_plots)
    print(json.dumps({"status": "exported", "output": str(args.out or args.root / "audit-report"),
                      "training_attempts": summary["training_attempts"], "architecture_entries": summary["architecture_registry_entries"],
                      "qualified_candidates": summary["qualified_candidates"], "warnings": summary["warnings"]}), flush=True)


if __name__ == "__main__":
    main()
