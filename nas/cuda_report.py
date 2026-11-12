"""Render scientific train/dev curves and leaderboards from saved run logs only.

One run: python cuda_report.py runs/search27/example
Search:  python cuda_report.py --root runs/search27 --out reports/search27
Fast leaderboard refresh: add --leaderboard-only (retains existing per-run plots).
This module never loads a dataset or checkpoint and does not inspect validation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def read_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    records = []
    for line in (directory / "log.jsonl").read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently running trainer may have an unfinished last line.
            continue
    # Resumed runs can repeat an evaluation step; retain the last observation.
    evaluation = {r["step"]: r for r in records if r.get("event") == "evaluation"}
    training = {r["step"]: r for r in records if r.get("event") == "training"}
    evaluation = [evaluation[s] for s in sorted(evaluation)]
    training = [training[s] for s in sorted(training)]
    finite = [r for r in evaluation if math.isfinite(r.get("dev_loss", float("nan")))]
    best = min(finite, key=lambda r: r["dev_loss"]) if finite else {}
    last = evaluation[-1] if evaluation else {}
    args = config.get("args", {})
    name = config.get("spec", {}).get("name", directory.name)
    metrics = {"run": str(directory), "name": name, "seed": args.get("seed"),
               "complete": bool(summary), "benchmark": bool(summary.get("benchmark", args.get("benchmark_steps", 0))),
               "steps": summary.get("steps", last.get("step", 0)), "schedule_steps": args.get("steps"),
               "parameters": config.get("parameters"), "structural_parameters": config.get("structural_parameters", config.get("parameters")),
               "learned_scale_parameters": config.get("learned_scale_parameters", 0),
               "best_step": best.get("step"), "best_dev_loss": best.get("dev_loss"),
               "train_eval_loss_at_best_dev": best.get("train_eval_loss"),
               "generalization_gap_at_best_dev": best.get("generalization_gap"),
               "final_dev_loss": last.get("dev_loss"), "final_train_eval_loss": last.get("train_eval_loss"),
               "final_minibatch_train_loss": training[-1].get("train_loss") if training else None,
               "train_seconds": summary.get("train_seconds"),
               "preparation_seconds": summary.get("preparation_seconds"), "compile": args.get("compile", False),
               "train_evaluation": config.get("train_evaluation"),
               "model": config.get("model"), "precision": args.get("precision")}
    return {"directory": directory, "config": config, "summary": summary,
            "evaluation": evaluation, "training": training, "metrics": metrics}


def plotting():
    # Share the font cache between short report processes, including sandboxed Macs.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"nas-matplotlib-{os.getuid()}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    return plt


def save_figure(fig, base):
    for extension in ["png", "svg"]:
        destination = base.with_suffix("." + extension)
        temporary = destination.with_name(destination.stem + ".tmp." + extension)
        fig.savefig(temporary, dpi=160, bbox_inches="tight", format=extension)
        temporary.replace(destination)


def render_run(directory):
    """Write curves.png, curves.svg, metrics.json; return a serializable metric row."""
    run = read_run(directory)
    metrics = run["metrics"]
    plt = plotting()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    evaluation, training = run["evaluation"], run["training"]
    if training:
        axes[0].plot([r["step"] for r in training], [r["train_loss"] for r in training],
                     color="#aeb8c4", linewidth=0.9, alpha=0.75, label="Training batches (dropout on)")
    axes[0].plot([r["step"] for r in evaluation], [r["dev_loss"] for r in evaluation],
                 color="#1764ab", linewidth=1.9, label="Dev (dropout off)")
    comparable = [r for r in evaluation if r.get("train_eval_loss") is not None]
    if comparable:
        axes[0].plot([r["step"] for r in comparable], [r["train_eval_loss"] for r in comparable],
                     color="#d07119", linewidth=1.7, label="Fixed training subset (dropout off)")
        gaps = [r["dev_loss"] - r["train_eval_loss"] for r in comparable]
        axes[1].plot([r["step"] for r in comparable], gaps, color="#5d477f", linewidth=1.6)
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    if metrics["best_step"] is not None:
        axes[0].scatter([metrics["best_step"]], [metrics["best_dev_loss"]], color="#1764ab", marker="*", s=100, zorder=5)
        axes[0].annotate(f"Best dev {metrics['best_dev_loss']:.4f}",
                         (metrics["best_step"], metrics["best_dev_loss"]),
                         xytext=(-8, 15), textcoords="offset points", ha="right", fontsize=9)
    axes[0].set_ylabel("Answer-token cross-entropy (nats)")
    axes[1].set_ylabel("Dev − train")
    axes[1].set_xlabel("Optimizer step")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend(loc="best", frameon=True, facecolor="white", framealpha=0.95)
    params = metrics["parameters"]
    title = f"{metrics['name']} · seed {metrics['seed']} · {params:,} parameters"
    fig.suptitle(textwrap.fill(title, width=96), fontsize=12)
    subset = metrics.get("train_evaluation") or {}
    footer = (f"Training diagnostic: fixed {subset.get('conversations', '?')} conversations; "
              "all curves are weighted by supervised answer-token count. Dev selects checkpoints.")
    fig.text(0.5, 0.012, footer, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, run["directory"] / "curves")
    plt.close(fig)
    atomic_text(run["directory"] / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    return metrics


def render_leaderboard(directories, output, top=12, run_plots=True):
    """Write JSON/CSV leaderboard and consolidated train/dev comparison figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, skipped = [], []
    for directory in sorted(set(map(Path, directories))):
        try:
            run = read_run(directory)
            if run["metrics"]["benchmark"] or not run["metrics"]["complete"]:
                continue
            if run["metrics"]["best_dev_loss"] is None:
                continue
            if run_plots:
                render_run(directory)
            runs.append(run)
        except (OSError, ValueError, KeyError) as error:
            skipped.append({"run": str(directory), "reason": str(error)})
    runs.sort(key=lambda r: (r["metrics"]["best_dev_loss"], r["metrics"]["parameters"]))
    metrics = [r["metrics"] for r in runs]
    for rank, row in enumerate(metrics, 1):
        row["rank"] = rank
    atomic_text(output / "leaderboard.json", json.dumps({"metric": "dev answer-token mean CE; lower is better",
                "selection": "best dev checkpoint within each completed training run",
                "validation_used": False, "runs": metrics, "skipped": skipped}, indent=2) + "\n")
    columns = ["rank", "name", "seed", "parameters", "learned_scale_parameters", "steps", "best_step", "best_dev_loss",
               "train_eval_loss_at_best_dev", "generalization_gap_at_best_dev", "final_dev_loss", "final_train_eval_loss",
               "train_seconds", "run"]
    csv_data = io.StringIO()
    writer = csv.DictWriter(csv_data, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(metrics)
    atomic_text(output / "leaderboard.csv", csv_data.getvalue())
    if not runs:
        return metrics
    plt = plotting()
    selected = runs[:top]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.get_cmap("tab20")
    for i, run in enumerate(selected):
        row, evaluation = run["metrics"], run["evaluation"]
        color = colors(i % 20)
        label = f"{row['rank']}. {row['name'][:55]} / seed {row['seed']} ({row['best_dev_loss']:.3f})"
        # Exclude initialization from the consolidated view to show learned differences.
        evaluation = [r for r in evaluation if r["step"] > 0]
        axes[0].plot([r["step"] for r in evaluation], [r["dev_loss"] for r in evaluation],
                     color=color, linewidth=1.5, label=label)
        comparable = [r for r in evaluation if r.get("train_eval_loss") is not None]
        axes[1].plot([r["step"] for r in comparable], [r["train_eval_loss"] for r in comparable], color=color, linewidth=1.5)
    axes[0].set_title("Dev loss (dropout off)")
    axes[1].set_title("Fixed training-subset loss (dropout off)")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Answer-token cross-entropy (nats)")
        axis.grid(alpha=0.2)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Top {len(selected)} of {len(runs)} completed runs by dev loss", fontsize=13)
    fig.tight_layout(rect=(0, min(0.36, 0.035 * math.ceil(len(selected) / 2)), 1, 0.95))
    save_figure(fig, output / "comparison")
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--leaderboard-only", action="store_true")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    runs = list(args.runs)
    if args.root:
        runs.extend(path.parent for path in args.root.rglob("summary.json") if (path.parent / "config.json").exists())
    if not runs:
        parser.error("provide run directories or --root with completed runs")
    if args.out:
        rows = render_leaderboard(runs, args.out, args.top, run_plots=not args.leaderboard_only)
        print(json.dumps({"completed_runs": len(rows), "output": str(args.out)}))
    else:
        for run in runs:
            print(json.dumps(render_run(run)))


if __name__ == "__main__":
    main()
