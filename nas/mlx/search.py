"""Run one search attempt end to end and record it, success or failure.

  .venv/bin/python search.py <spec.json> --steps 2000|5000 [--seed 0]

Trains with train.py, then writes experiments/search/<id>/ containing the spec,
config, full loss log, summary, samples and loss.png (train + dev vs step, with
the baseline at the same K in grey), and appends a row to
experiments/search/LOG.md and results.jsonl. Weights stay in runs/search/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from nas.paths import PROJECT_ROOT, python_command
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nas.mlx import arch
ROOT = PROJECT_ROOT
EXP = ROOT / "experiments" / "search"
BASE_PARAMS = 27_144_960
REFERENCE = {  # baseline dev-loss logs at each K (same recipe, seed 0)
    5000: ROOT / "experiments/baseline-depth/n8/log.jsonl",
    2000: EXP / "s00-baseline-k2000/log.jsonl",
}


def read_log(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def plot(attempt_dir: Path, title: str, steps: int):
    rows = read_log(attempt_dir / "log.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    s = [r["step"] for r in rows if r["step"] > 0]
    ax.plot(s, [r["train_loss"] for r in rows if r["step"] > 0], "-o", ms=3, color="#d95f02", label="train (avg over interval)")
    ax.plot([r["step"] for r in rows], [r["dev_loss"] for r in rows], "-o", ms=3, color="#1b9e77", label="dev (1,000 held-out conversations)")
    ref = REFERENCE.get(steps)
    if ref and ref.exists() and ref.resolve() != (attempt_dir / "log.jsonl").resolve():
        rr = read_log(ref)
        ax.plot([r["step"] for r in rr], [r["dev_loss"] for r in rr], "--", color="grey", label="baseline dev (N=8)")
        ax.plot([r["step"] for r in rr if r["step"] > 0], [r["train_loss"] for r in rr if r["step"] > 0], ":", color="grey", label="baseline train")
    ax.set_ylim(2.2, 6.0)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("answer cross-entropy (nats/token)")
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(attempt_dir / "loss.png"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    a = p.parse_args()

    spec = arch.load_spec(a.spec)
    attempt = f"{spec['name']}-k{a.steps}" + (f"-s{a.seed}" if a.seed else "")
    out_dir = EXP / attempt
    run_dir = ROOT / "runs" / "search" / attempt
    if out_dir.exists() or run_dir.exists():
        sys.exit(f"{attempt} already exists; never overwrite an attempt")
    out_dir.mkdir(parents=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.spec, out_dir / "spec.json")

    cmd = [*python_command("train.py", ROOT), "--out", str(run_dir), "--spec", a.spec,
           "--steps", str(a.steps), "--eval-every", str(a.eval_every), "--seed", str(a.seed),
           "--max-params", str(BASE_PARAMS)]
    t0 = time.time()
    with open(out_dir / "train.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    wall = time.time() - t0

    row = {"attempt": attempt, "name": spec["name"], "steps": a.steps, "seed": a.seed,
           "layout": arch.describe(spec), "description": spec.get("description", ""),
           "returncode": rc, "wall_seconds": round(wall)}
    if rc == 0:
        for f in ("config.json", "log.jsonl", "summary.json", "samples.txt"):
            shutil.copy(run_dir / f, out_dir / f)
        summ = json.loads((run_dir / "summary.json").read_text())
        rows = read_log(run_dir / "log.jsonl")
        best = min(rows, key=lambda r: r["dev_loss"])
        row.update(parameters=summ["parameters"], best_dev=round(summ["best_dev_loss"], 4),
                   best_step=best["step"], final_train=round(rows[-1]["train_loss"], 4),
                   sec_per_step=round(summ["train_seconds"] / a.steps, 3))
        ref = REFERENCE.get(a.steps)
        if ref and ref.exists():
            rb = min(r["dev_loss"] for r in read_log(ref))
            row["delta_vs_baseline"] = round(summ["best_dev_loss"] - rb, 4)
        plot(out_dir, f"{attempt}: {row['layout']}\n{summ['parameters']:,} params, best dev {summ['best_dev_loss']:.4f}", a.steps)
    (out_dir / "result.json").write_text(json.dumps(row, indent=2))
    with open(EXP / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    write_log_md()
    print(json.dumps(row))


def write_log_md():
    rows = read_log(EXP / "results.jsonl")
    lines = ["# Architecture search log", "",
             "**Objective (set by the user):** (1) primary: lower dev loss than the N=8 baseline;",
             "(2) secondary: fewer parameters (always <= 27,144,960; some reduction required).",
             "Parameter cuts are taken only where they cost little or no loss.", "",
             "Every attempt, in order, including failures. Dev = 1,000 conversations held out from",
             "train.json; validation.json is untouched. Baseline = N=8 GPT, 27,144,960 parameters.",
             "Δ = best dev loss minus the baseline's at the same K (negative is better).", "",
             "| # | Attempt | K | Layout | Params | Best dev | Δ vs base | Best step | s/step | Plot |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        if r["returncode"] != 0:
            lines.append(f"| {i} | {r['attempt']} | {r['steps']} | `{r['layout']}` | — | FAILED (rc {r['returncode']}) | | | | [log]({r['attempt']}/train.log) |")
            continue
        d = r.get("delta_vs_baseline")
        lines.append(f"| {i} | {r['attempt']} | {r['steps']} | `{r['layout']}` | {r['parameters']:,} | "
                     f"{r['best_dev']:.4f} | {'' if d is None else f'{d:+.4f}'} | {r['best_step']} | "
                     f"{r['sec_per_step']} | [loss.png]({r['attempt']}/loss.png) |")
    lines += ["", "## Notes per attempt", ""]
    for r in rows:
        lines.append(f"- **{r['attempt']}** — {r['description']}")
    (EXP / "LOG.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
