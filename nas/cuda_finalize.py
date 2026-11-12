"""Evaluate the frozen overnight selection once, then write the final report.

python cuda_finalize.py --out runs/cuda/search-27m-20261111 \
    --deadline-utc 2026-11-11T16:10:00+00:00

No architecture selection or training occurs here. Validation is unavailable
until the selection, fresh-seed runs, provenance, and checkpoints pass preflight.
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

from nas.cuda_search import (BASELINE_PARAMETERS, FRAME, RECIPE, architecture_hash,
                         baseline_spec, file_hash, parameter_count, parse_utc)

ROOT = PROJECT_ROOT


def now():
    return datetime.now(timezone.utc)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def inside(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path escapes the search directory: {path}")
    return resolved


def inspect_checkpoint(path, config):
    """Load only trusted local tensor checkpoints, and verify their exact schema."""
    import torch
    from nas.cuda_model import Config, SpecGPT, count_parameters
    torch.set_num_threads(4)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_config") != config["model"] or state.get("spec") != config["spec"]:
        raise ValueError(f"checkpoint architecture differs from config: {path}")
    model = SpecGPT(spec=config["spec"], config=Config(**config["model"]))
    model.load_state_dict(state["model"], strict=True)
    if count_parameters(model) != config["parameters"]:
        raise ValueError(f"checkpoint parameter count differs from config: {path}")
    scales = {key.removeprefix("branch_scales."): float(value.item()) for key, value in state["model"].items()
              if key.startswith("branch_scales.")}
    return {"step": state["step"], "best_dev_loss": state["best_dev_loss"], "learned_residual_scalars": scales}


def validate_selection(root, project_root=ROOT, checkpoint_inspector=inspect_checkpoint):
    selection_path, freeze_path = root / "final_selection.json", root / "selection.json"
    selection_digest, freeze_digest = file_hash(selection_path), file_hash(freeze_path)
    selected, frozen = read_json(selection_path), read_json(freeze_path)
    if selected.get("status") != "complete":
        raise ValueError("final_selection.json is not complete")
    if selected.get("validation_accessed") is not False or selected.get("fresh_seeds_used_for_architecture_selection") is not False:
        raise ValueError("selection must precede validation and must not select on fresh confirmation seeds")
    if selected.get("seeds") != [10, 11, 12] or selected.get("selection_seeds") != [0, 1, 2]:
        raise ValueError("selection/confirmation seeds differ from the approved protocol")
    if selected.get("frozen_selection") != frozen or frozen.get("validation_used") is not False:
        raise ValueError("final selection differs from the pre-confirmation freeze")
    winner_hash = selected["winner_spec_hash"]
    if frozen.get("primary_architecture_id") != winner_hash or selected["winner_spec"]["hash"] != winner_hash:
        raise ValueError("winner architecture differs from the frozen primary")
    spec_path = inside(selected["winner_spec"]["path"], root)
    winner_spec = read_json(spec_path)
    if file_hash(spec_path) != selected["winner_spec"]["file_sha256"] or architecture_hash(winner_spec) != winner_hash:
        raise ValueError("winner spec file or canonical architecture hash changed")
    winner_parameters = parameter_count(winner_spec)
    if not 0 < winner_parameters < BASELINE_PARAMETERS:
        raise ValueError("the frozen winner must have strictly fewer total parameters, including learned scalars")
    if selected["winner_spec"]["parameters"] != winner_parameters or selected["baseline_parameters"] != BASELINE_PARAMETERS:
        raise ValueError("frozen parameter counts do not match the executable architecture")
    if selected["parameter_reduction"] != BASELINE_PARAMETERS - winner_parameters:
        raise ValueError("frozen parameter reduction is inconsistent")
    baselines, winners = selected["baseline_runs"], selected["winner_runs"]
    if len(baselines) != 3 or len(winners) != 3 or len(set(baselines + winners)) != 6:
        raise ValueError("exactly three distinct baseline and three distinct winner runs are required")
    registry = {}
    for line in (root / "run_registry.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("path"):
            registry[str(Path(row["path"]).resolve())] = row
    records, train_hashes = [], None
    baseline_hash = architecture_hash(baseline_spec())
    for index, given in enumerate(baselines + winners):
        path = inside(given, root)
        config, summary = read_json(path / "config.json"), read_json(path / "summary.json")
        expected_seed = [10, 11, 12][index % 3]
        expected_architecture = baseline_hash if index < 3 else winner_hash
        expected_parameters = BASELINE_PARAMETERS if index < 3 else winner_parameters
        args = config["args"]
        if config["model"] != FRAME or architecture_hash(config["spec"]) != expected_architecture:
            raise ValueError(f"run has a different frame or architecture: {path}")
        if args["seed"] != expected_seed or any(args.get(key) != value for key, value in RECIPE.items()):
            raise ValueError(f"run has a different seed or training recipe: {path}")
        if args.get("precision") != "bf16" or args.get("compile") is not True:
            raise ValueError(f"run must use the shared BF16 compiled training path: {path}")
        if (summary.get("steps") != 5000 or summary.get("schedule_steps") != 5000 or summary.get("benchmark")
                or config["parameters"] != expected_parameters or summary["parameters"] != expected_parameters
                or not math.isfinite(summary["best_dev_loss"])):
            raise ValueError(f"run is not a complete finite 5000-step confirmation: {path}")
        checkpoint = path / "best.pt"
        digest = file_hash(checkpoint)
        ledger = registry.get(str(path), {})
        saved = ledger.get("checkpoints", {}).get("best.pt", {})
        if ledger.get("status") != "completed" or saved.get("pruned") or saved.get("sha256") != digest:
            raise ValueError(f"best checkpoint does not match the completed run ledger: {path}")
        job = ledger.get("job", {})
        if job.get("phase") != "final" or job.get("seed") != expected_seed:
            raise ValueError(f"run is not a fresh-seed final confirmation: {path}")
        details = checkpoint_inspector(checkpoint, config)
        if details["step"] != summary["best_step"] or not math.isclose(details["best_dev_loss"], summary["best_dev_loss"], abs_tol=1e-7):
            raise ValueError(f"checkpoint is not the recorded best dev checkpoint: {path}")
        hashes = {name: config["sha256"][name] for name in [*PROVENANCE_SUPPORT, "data/train.json", "data/tokenizer.json", "cuda_model.py", "cuda_train.py"]}
        if train_hashes is not None and hashes != train_hashes:
            raise ValueError("confirmation runs used different data, tokenizer, model, or trainer source")
        train_hashes = hashes
        records.append({"run": str(path), "role": "baseline" if index < 3 else "winner", "seed": expected_seed,
                        "parameters": expected_parameters, "architecture_sha256": expected_architecture,
                        "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "best_step": details["step"],
                        "best_dev_loss": summary["best_dev_loss"],
                        "learned_residual_scalars": details.get("learned_residual_scalars", {}),
                        "train_eval_loss_at_best_dev": summary.get("train_eval_loss_at_best_dev"),
                        "train_seconds": summary.get("train_seconds"), "hardware": config.get("hardware", {}),
                        "config_sha256": file_hash(path / "config.json"), "summary_sha256": file_hash(path / "summary.json")})
    for name, digest in train_hashes.items():
        if file_hash(source_path(name, project_root)) != digest:
            raise ValueError(f"current data or executable source differs from the confirmation runs: {name}")
    if file_hash(selection_path) != selection_digest or file_hash(freeze_path) != freeze_digest:
        raise ValueError("selection changed during preflight")
    return {"selection": selected, "runs": records, "training_provenance": train_hashes,
            "selection_sha256": selection_digest, "freeze_sha256": freeze_digest,
            "completed_run_directories": [path for path, record in registry.items() if record.get("status") == "completed"],
            "winner_parameters": winner_parameters, "winner_architecture_sha256": winner_hash, "winner_spec": winner_spec}


def unchanged(root, verified):
    if file_hash(root / "final_selection.json") != verified["selection_sha256"] or file_hash(root / "selection.json") != verified["freeze_sha256"]:
        raise ValueError("frozen selection changed during finalization")
    for run in verified["runs"]:
        if file_hash(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError("a final checkpoint changed during finalization")


def validation_result(path, runs):
    import numpy as np
    result = read_json(path)
    expected = [run["run"] for run in runs]
    if result.get("split") != "validation" or result.get("baseline_count") != 3 or result.get("run_order") != expected:
        raise ValueError("validation output does not describe the exact frozen six-model comparison")
    comparison = result["results"]["difference_candidate_minus_baseline_across_seeds"]
    values = [result["results"][run]["loss"] for run in expected] + [comparison["point"], *comparison["bootstrap_95"]]
    if len(comparison["bootstrap_95"]) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("validation results or interval are incomplete/nonfinite")
    if comparison["bootstrap_95"][0] > comparison["bootstrap_95"][1] or comparison.get("training_seeds") != 3:
        raise ValueError("validation bootstrap interval or seed count is invalid")
    actual_difference = statistics.mean(result["results"][run]["loss"] for run in expected[3:]) - statistics.mean(result["results"][run]["loss"] for run in expected[:3])
    if not math.isclose(actual_difference, comparison["point"], abs_tol=1e-8):
        raise ValueError("validation comparison does not match the six model losses")
    with np.load(path.with_suffix(".conversations.npz"), allow_pickle=False) as archive:
        counts = archive["run_0_counts"]
        for i, run in enumerate(expected):
            sums, n = archive[f"run_{i}_sums"], archive[f"run_{i}_counts"]
            if not np.array_equal(n, counts) or len(sums) != len(n) or not np.isfinite(sums).all() or not (n > 0).all():
                raise ValueError("paired per-conversation archive is incomplete or inconsistent")
            if not math.isclose(float(sums.sum() / n.sum()), result["results"][run]["loss"], abs_tol=1e-8):
                raise ValueError("validation summary differs from its conversation archive")
    return result


def summarize(root, verified, evaluated, provenance):
    results = evaluated["results"]
    baseline, winner = verified["runs"][:3], verified["runs"][3:]
    def stats(rows):
        dev = [row["best_dev_loss"] for row in rows]
        validation = [results[row["run"]]["loss"] for row in rows]
        scalars = {}
        for key in rows[0]["learned_residual_scalars"]:
            values = [row["learned_residual_scalars"][key] for row in rows]
            scalars[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values),
                            "by_seed": {str(row["seed"]): value for row, value in zip(rows, values)}}
        return {"parameters": rows[0]["parameters"], "dev_mean": statistics.mean(dev), "dev_std": statistics.stdev(dev),
                "validation_mean": statistics.mean(validation), "validation_std": statistics.stdev(validation),
                "learned_residual_scalars": scalars,
                "runs": [{**row, "validation_loss": loss} for row, loss in zip(rows, validation)]}
    comparison = results["difference_candidate_minus_baseline_across_seeds"]
    selected = verified["selection"]
    statuses = Counter(row["status"] for row in selected.get("run_statuses", []))
    completed_architectures = {row["architecture_id"] for row in selected.get("run_statuses", []) if row["status"] == "completed"}
    reduction = BASELINE_PARAMETERS - verified["winner_parameters"]
    report = {"status": "complete", "completed_utc": now().isoformat(),
              "objective": "Lower validation answer-token loss with fewer total parameters",
              "winner_name": selected["winner_spec"]["name"], "winner_architecture_sha256": verified["winner_architecture_sha256"],
              "winner_spec": verified["winner_spec"],
              "baseline": stats(baseline), "winner": stats(winner),
              "parameter_reduction": reduction, "parameter_reduction_fraction": reduction / BASELINE_PARAMETERS,
              "validation_comparison": comparison, "validation_mean_improved": comparison["point"] < 0,
              "validation_interval_below_zero": comparison["bootstrap_95"][1] < 0,
              "fresh_seed_dev_differences": [w["best_dev_loss"] - b["best_dev_loss"] for b, w in zip(baseline, winner)],
              "run_records_total": sum(statuses.values()), "run_statuses": dict(statuses),
              "completed_training_runs": statuses.get("completed", 0),
              "architectures_with_completed_runs": len(completed_architectures),
              "attempt_count": sum(statuses.values()), "attempt_statuses": dict(statuses),
              "attempt_count_scope": "all registered run records, including queued or not-admitted jobs; completed runs are counted separately",
              "registered_architecture_count": len({r["architecture_id"] for r in selected.get("run_statuses", [])}),
              "selection_freeze_sha256": verified["freeze_sha256"], "final_selection_sha256": verified["selection_sha256"],
              "selection_rule": selected["selection_rule"], "data_and_source_provenance": provenance,
              "validation_used_for_selection": False, "reselection_after_validation": False,
              "artifacts": {"final_selection": str(root / "final_selection.json"), "selection_freeze": str(root / "selection.json"),
                            "validation": str(root / "final-validation.json"), "conversations": str(root / "final-validation.conversations.npz"),
                            "all_run_leaderboard": str(root / "final-report/leaderboard.json"),
                            "all_run_csv": str(root / "final-report/leaderboard.csv"),
                            "train_dev_comparison_png": str(root / "final-report/comparison.png"),
                            "train_dev_comparison_svg": str(root / "final-report/comparison.svg"),
                            "run_ledger": str(root / "run_registry.jsonl")},
              "limitations": ["The winner was selected using dev loss; dev comparisons therefore include search selection effects.",
                              "The paired validation interval resamples the three observed training seeds and conversations; three seeds only partially characterize training variability.",
                              "BF16 NVIDIA results are compared within this implementation; historical MLX losses use a different numerical execution path.",
                              "The frozen winner is reported even if validation shows no improvement; validation does not trigger reselection or retraining."]}
    return report


def finalize(out, deadline, project_root=ROOT, checkpoint_inspector=inspect_checkpoint, report_renderer=None):
    root, project_root = Path(out).resolve(), Path(project_root).resolve()
    state_path, lock_path = root / "finalization.json", root / ".finalization.lock"
    if not root.is_dir():
        raise ValueError("search output directory does not exist")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "started_utc": now().isoformat()}).encode())
    os.close(lock_fd)
    try:
        previous = read_json(state_path) if state_path.exists() else {}
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    state = {"status": "preflight", "attempt": previous.get("attempt", 0) + 1, "started_utc": now().isoformat(),
             "deadline_utc": deadline.isoformat(), "validation_started": False, "validation_completed": False,
             "pid": os.getpid(), "hostname": socket.gethostname(), "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES")}
    process = None
    def save(**updates):
        state.update(updates, updated_utc=now().isoformat())
        atomic_json(state_path, state)
        print(json.dumps({"event": "finalization", **state}), flush=True)
    def stop_process():
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
    def interrupted(signum, frame):
        stop_process()
        raise InterruptedError(f"finalizer interrupted by signal {signum}")
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    for number in handlers:
        signal.signal(number, interrupted)
    try:
        verified = validate_selection(root, project_root, checkpoint_inspector)
        save(selection_sha256=verified["selection_sha256"], selection_freeze_sha256=verified["freeze_sha256"],
             checkpoint_sha256={r["run"]: r["checkpoint_sha256"] for r in verified["runs"]})
        if previous.get("selection_sha256") not in (None, verified["selection_sha256"]):
            raise ValueError("selection changed since a previous finalization attempt; refusing validation")
        if previous.get("checkpoint_sha256") not in (None, state["checkpoint_sha256"]):
            raise ValueError("final checkpoints changed since a previous finalization attempt")
        output, archive = root / "final-validation.json", root / "final-validation.conversations.npz"
        if output.exists() or archive.exists():
            if not (output.exists() and archive.exists() and previous.get("validation_completed")):
                raise ValueError("partial or untracked validation output exists; inspect it explicitly; no files were overwritten")
            hashes = {"json": file_hash(output), "conversations": file_hash(archive)}
            if hashes != previous.get("validation_output_sha256"):
                raise ValueError("existing validation outputs differ from the recorded completed evaluation")
            evaluated = validation_result(output, verified["runs"])
            provenance = previous["data_and_source_provenance"]
            save(status="reporting", validation_started=True, validation_completed=True,
                 validation_output_sha256=hashes, data_and_source_provenance=provenance)
            if previous.get("status") == "complete" and (root / "FINAL_REPORT.json").exists():
                if file_hash(root / "FINAL_REPORT.json") != previous.get("final_report_sha256"):
                    raise ValueError("completed final report changed since its recorded completion")
                save(status="complete", reused_completed_result=True, final_report=str(root / "FINAL_REPORT.json"),
                     final_report_sha256=previous["final_report_sha256"])
                return read_json(root / "FINAL_REPORT.json")
        else:
            remaining = (deadline - now()).total_seconds()
            if remaining <= 90:
                raise TimeoutError("less than 90 seconds remain before the finalization deadline")
            unchanged(root, verified)
            # This is the first permitted access to held-out data, after all preflight checks.
            provenance = {**verified["training_provenance"],
                          "data/validation.json": file_hash(project_root / "data/validation.json"),
                          "cuda_evaluate.py": file_hash(source_path("cuda_evaluate.py", project_root)),
                          "cuda_finalize.py": file_hash(Path(__file__))}
            command = [*python_command("cuda_evaluate.py", project_root), *[r["run"] for r in verified["runs"]],
                       "--split", "validation", "--baseline-count", "3", "--cpu-threads", "4",
                       "--precision", "bf16", "--out", str(output)]
            save(status="evaluating", validation_started=True, command=command, data_and_source_provenance=provenance)
            env = dict(os.environ, OMP_NUM_THREADS="4")
            with (root / "final-evaluator.log").open("a") as log:
                process = subprocess.Popen(command, cwd=project_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=max(1, (deadline - now()).total_seconds() - 60))
                except subprocess.TimeoutExpired as error:
                    stop_process()
                    raise TimeoutError("final validation evaluator exceeded its deadline allowance") from error
            if returncode:
                raise RuntimeError(f"final validation evaluator exited {returncode}; see final-evaluator.log")
            evaluated = validation_result(output, verified["runs"])
            unchanged(root, verified)
            save(status="reporting", validation_completed=True,
                 validation_output_sha256={"json": file_hash(output), "conversations": file_hash(archive)})
        if now() >= deadline:
            raise TimeoutError("finalization deadline reached before report rendering")
        from nas.cuda_report import render_leaderboard, render_run
        completed = [inside(path, root) for path in verified["completed_run_directories"]]
        if report_renderer is None:
            for run in verified["runs"]:
                render_run(run["run"])
            render_leaderboard(completed, root / "final-report", top=12, run_plots=False)
        else:
            report_renderer(completed, root / "final-report")
        report = summarize(root, verified, evaluated, provenance)
        try:
            from nas.cuda_inventory import build_inventory, write_inventory
            inventory = build_inventory(project_root / "runs")
            inventory_paths = write_inventory(inventory, project_root / "runs/cuda")
            report["artifacts"]["global_run_index"] = [str(path) for path in inventory_paths]
        except (ImportError, OSError, ValueError) as error:
            report["limitations"].append(f"Global inventory refresh failed: {error}")
        unchanged(root, verified)
        if now() >= deadline:
            raise TimeoutError("finalization deadline reached before final report completion")
        atomic_json(root / "FINAL_REPORT.json", report)
        save(status="complete", final_report=str(root / "FINAL_REPORT.json"), final_report_sha256=file_hash(root / "FINAL_REPORT.json"))
        return report
    except BaseException as error:
        stop_process()
        save(status="interrupted" if isinstance(error, InterruptedError) else "failed", error=f"{type(error).__name__}: {error}",
             retry_requires_same_frozen_selection=True)
        raise
    finally:
        stop_process()
        for number, handler in handlers.items():
            signal.signal(number, handler)
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-utc", type=parse_utc, required=True)
    args = parser.parse_args()
    try:
        report = finalize(args.out, args.deadline_utc)
    except Exception as error:
        print(json.dumps({"event": "finalization_failed", "error": f"{type(error).__name__}: {error}"}), flush=True)
        raise SystemExit(1)
    print(json.dumps({"event": "final_report_ready", "winner": report["winner_name"],
                      "validation_difference": report["validation_comparison"]["point"],
                      "report": str(args.out / "FINAL_REPORT.json")}), flush=True)


if __name__ == "__main__":
    main()
