"""Package a completed, audited minimum-parameter experiment for delivery.

Reads saved main-experiment reports/ledgers and a metrics-free diagnostic receipt.
Never opens independent diagnostic REPORT files, conversation arrays, checkpoints,
or datasets. Writes only FINAL_DELIVERY.md and FINAL_DELIVERY.json under --root.
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
import statistics
import sys
import time
from urllib.parse import quote


SEEDS = [100, 101, 102, 103, 104]
BASELINE_PARAMETERS = 27_144_960


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def wait_for_postprocessing(root, deadline, poll_seconds=10):
    """Wait only for the completion record; build() retains all strict gates."""
    require(0 < poll_seconds <= 10, "postprocessing poll interval must be in (0, 10] seconds")
    path = root.resolve() / "postprocessing.json"
    print(json.dumps({"event": "waiting_for_postprocessing", "utc": datetime.now(timezone.utc).isoformat(),
                      "deadline_utc": deadline.isoformat()}), flush=True)
    while True:
        current = datetime.now(timezone.utc)
        if current >= deadline:
            raise TimeoutError(f"postprocessing did not become ready before {deadline.isoformat()}")
        if path.is_file():
            record = read_json(path)
            require(record.get("status") != "failed", "postprocessing recorded failure; delivery will not proceed")
            print(json.dumps({"event": "postprocessing_record_ready", "utc": datetime.now(timezone.utc).isoformat(),
                              "status": record.get("status")}), flush=True)
            return
        time.sleep(min(poll_seconds, (deadline - current).total_seconds()))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    temporary.write_text(text)
    temporary.replace(path)


def relative(path, root):
    return os.path.relpath(Path(path).resolve(), root)


def link(path, label, root):
    return f"[{label}]({quote(relative(path, root), safe='/.-_')})"


def receipt_path(root, explicit):
    if explicit:
        result = explicit.resolve()
    else:
        choices = [p for p in [root / "COMPLETION_RECEIPT.json",
                   *root.parent.glob("independent-heldout-*/COMPLETION_RECEIPT.json")] if p.is_file()]
        require(len(choices) == 1, "supply --diagnostic-receipt: exactly one metrics-free receipt was not found")
        result = choices[0].resolve()
    require(result.name == "COMPLETION_RECEIPT.json", "only the metrics-free COMPLETION_RECEIPT.json may be read")
    return result


def build(root, explicit_receipt=None):
    root = root.resolve()
    require(root.is_dir(), "run root does not exist")
    # Gate before reading any completed main-experiment result or diagnostic receipt.
    report = read_json(root / "FINAL_REPORT.json")
    require(report.get("status") == "complete", "requires completed baseline/winner confirmation; source report is incomplete or has no qualifying winner")
    report_hash = digest(root / "FINAL_REPORT.json")
    post = read_json(root / "postprocessing.json")
    require(post.get("status") == "complete" and post.get("final_report_sha256") == report_hash,
            "postprocessing must be complete for this exact final report")
    expected_scripts = {"cuda_minparams_architecture_report.py", "cuda_minparams_report.py", "cuda_minparams_audit.py"}
    require({row["script"] for row in post["commands"]} == expected_scripts
            and all(row["returncode"] == 0 for row in post["commands"]), "postprocessing did not complete all required exporters")

    audits = sorted((root / "final-audit").glob("audit-*.json"))
    require(bool(audits), "no final artifact audit exists")
    audit_path = audits[-1]
    audit = read_json(audit_path)
    # Never hide a later failed audit by choosing an older pass.
    require(audit.get("status") == "pass" and not audit.get("failures"), "latest artifact audit must pass")
    selection = read_json(root / "selection.json")
    selection_hash = digest(root / "selection.json")
    require(audit.get("final_report_sha256") == report_hash
            and audit.get("selection_sha256") == selection_hash
            and report.get("selection_freeze_sha256") == selection_hash,
            "report/audit do not match the exact frozen selection")
    require(selection.get("status") == "qualified" and selection.get("frozen_utc")
            and selection.get("validation_used") is False, "selection is not a qualified dev-only freeze")
    primary = selection["primary"]
    require(primary["architecture_id"] == report["winner_architecture_id"] == selection["primary_architecture_id"],
            "frozen winner identities disagree")
    require(report.get("selection_seeds") == [0, 1, 2] and report.get("fresh_seeds") == SEEDS
            and all(report.get(key) is False for key in ["fresh_seeds_used_for_architecture_selection",
                    "test_used_for_selection", "reselection_after_test"]), "final report changed the selection/confirmation protocol")

    architecture = read_json(root / "architecture-report/ARCHITECTURE.json")
    require(architecture["provenance"].get("mode") == "frozen_primary"
            and architecture["provenance"]["architecture_id"] == primary["architecture_id"]
            and selection_hash in architecture["provenance"]["source_sha256"].values()
            and architecture["total_unique_parameters"] == primary["parameters"] == report["winner"]["parameters"],
            "architecture report does not describe this frozen winner")
    search = read_json(root / "audit-report/SEARCH_SUMMARY.json")
    require(search.get("frozen_primary_architecture_id") == primary["architecture_id"]
            and search.get("completed_fresh_confirmation_runs") == 10, "search log export is stale")

    records = {}
    for line in (root / "run_registry.jsonl").read_text().splitlines():
        row = json.loads(line)
        records[row["job"]["run_id"]] = row
    statuses = dict(Counter(row["status"] for row in records.values()))
    require(statuses == search["attempt_statuses"], "run-log counts changed after export")
    require(not any(s in statuses for s in ["queued", "running"]), "run registry still has unfinished work")
    require(statuses.get("completed", 0) == report["completed_training_runs"] == audit["counts"]["completed_training_runs"],
            "completed-run counts disagree")
    require(audit["counts"].get("completed_test_evaluations") == 10, "audit does not cover all ten fresh test evaluations")

    artifacts = {}

    def artifact(key, path, required=True):
        path = Path(path)
        if not path.is_file():
            require(not required, f"required delivery artifact missing: {path}")
            return None
        artifacts[key] = {"path": relative(path, root), "sha256": digest(path), "bytes": path.stat().st_size}
        return path

    fixed_artifacts = {
        "final_report": "FINAL_REPORT.md", "final_report_json": "FINAL_REPORT.json",
        "selection": "selection.json", "postprocessing": "postprocessing.json",
        "run_ledger": "run_registry.jsonl", "architecture_ledger": "architecture_registry.jsonl",
        "evaluation_ledger": "evaluation_registry.jsonl", "source_manifest": "workflow_source_manifest.json",
        "run_log": "audit-report/RUN_LOG.md", "architecture_log": "audit-report/ARCHITECTURES.md",
        "search_summary": "audit-report/SEARCH_SUMMARY.json", "search_progress": "audit-report/search_progress.png",
        "architecture": "architecture-report/ARCHITECTURE.md", "architecture_json": "architecture-report/ARCHITECTURE.json",
        "architecture_ascii": "architecture-report/architecture.txt", "disclosure": "TEST_ACCESS_DISCLOSURE.md",
    }
    for key, name in fixed_artifacts.items():
        artifact(key, root / name)
    artifact("audit", audit_path.with_suffix(".md"))
    artifact("audit_json", audit_path)
    missing_optional = []
    for key, name in {"fresh_curves": "final-report/comparison.png", "all_run_curves": "all-run-report/comparison.png",
                      "fresh_curve_metrics": "final-report/leaderboard.json", "main_test_report": "final-test.json"}.items():
        if artifact(key, root / name, required=False) is None:
            missing_optional.append(name)

    runtime = {}
    for role in ("baseline", "winner"):
        rows = sorted(report[role]["runs"], key=lambda row: row["seed"])
        require([row["seed"] for row in rows] == SEEDS, f"{role} does not contain five exact fresh seeds")
        observations = []
        for row in rows:
            run_id = Path(row["run"]).name
            record = records[run_id]
            require(record["status"] == "completed" and record["job"]["phase"] == "final"
                    and record["job"]["seed"] == row["seed"] and record["job"]["architecture_id"] == row["architecture_id"],
                    f"fresh run ledger identity mismatch: {run_id}")
            summary_path = root / "attempts" / run_id / "summary.json"
            require(digest(summary_path) == row["summary_sha256"], f"fresh summary changed after final report: {run_id}")
            summary = read_json(summary_path)
            require(summary["steps"] == summary["schedule_steps"] == 5000, "fresh run did not finish exactly 5,000 steps")
            wall, inferred = record.get("wall_seconds"), False
            if wall is None:
                wall = (timestamp(record["ended_utc"]) - timestamp(record["started_utc"])).total_seconds()
                inferred = True
            values = {"wall_seconds": wall, "trainer_elapsed_seconds": summary["train_seconds"],
                      "training_compute_seconds": summary["training_compute_seconds"],
                      "dev_evaluation_seconds": summary["evaluation_seconds"],
                      "train_diagnostic_evaluation_seconds": summary["train_evaluation_seconds"]}
            require(all(finite(value) and value >= 0 for value in values.values()), "invalid fresh-run timing")
            observations.append({"run_id": run_id, "seed": row["seed"], **values,
                                 "wall_inferred_from_timestamps": inferred,
                                 "physical_gpu": record.get("physical_gpu"), "hardware": row.get("hardware", {})})
            artifact("summary_" + run_id, summary_path)
        runtime[role] = {"runs": observations, "means": {key: statistics.mean(row[key] for row in observations)
                        for key in values}, "wall_seconds_std": statistics.stdev(row["wall_seconds"] for row in observations)}
    baseline, winner = report["baseline"], report["winner"]
    require(baseline["parameters"] == BASELINE_PARAMETERS and 0 < winner["parameters"] < BASELINE_PARAMETERS,
            "parameter counts do not describe a smaller candidate and the original baseline")
    reduction = BASELINE_PARAMETERS - winner["parameters"]
    require(report["parameter_reduction"] == reduction, "parameter reduction disagrees")
    ratio = runtime["winner"]["means"]["wall_seconds"] / runtime["baseline"]["means"]["wall_seconds"]
    comparison = report["test_comparison"]
    require(all(finite(v) for v in [comparison["point"], *comparison["bootstrap_95"]]), "nonfinite test comparison")
    matched = comparison["point"] <= 0
    require(report["public_claim_supported_by_mean"] == report["test_mean_no_worse"] == matched,
            "observed test comparison and public-claim gate disagree")

    receipt_file = receipt_path(root, explicit_receipt)
    receipt = read_json(receipt_file)  # Only this exact metrics-free basename is permitted.
    require(receipt.get("no_performance_metrics_or_interpretations") is True
            and receipt.get("evaluation_finished") is True, "diagnostic receipt is not completed and metrics-free")
    coverage = receipt["coverage"]
    require(coverage["evaluated_checkpoints"] == coverage["planned_checkpoints"] == 12
            and coverage["conversations_per_checkpoint"] == 2599 and coverage["training_seeds"] == [0, 1, 2],
            "unexpected diagnostic receipt coverage")
    require(receipt["source_data_sha256"]["data/validation.json"] == report["data_and_source_provenance"]["data/validation.json"],
            "diagnostic and final report do not identify the same held-out split")
    artifact("diagnostic_receipt", receipt_file)
    disclosure_text = (root / "TEST_ACCESS_DISCLOSURE.md").read_text()
    require(bool(disclosure_text.strip()), "test-access disclosure is empty")
    require(timestamp(receipt["started_utc"]) <= timestamp(receipt["completed_utc"])
            < timestamp(selection["frozen_utc"]), "receipt does not establish the disclosed pre-freeze diagnostic timing")
    diagnostic_seconds = (timestamp(receipt["completed_utc"]) - timestamp(receipt["started_utc"])).total_seconds()
    diagnostic = {key: receipt[key] for key in ["coverage", "started_utc", "completed_utc", "source_manifest_sha256", "snapshot_sha256"]}
    diagnostic.update(receipt_path=relative(receipt_file, root), scores_read_by_this_renderer=False,
                      pooled_with_fresh_confirmation=False, elapsed_service_seconds=diagnostic_seconds,
                      local_backup=receipt.get("local_backup"), local_copy_scope=receipt.get("local_copy_scope"))
    timing = audit["recomputed"]["observed_timing"]
    manifest = {"status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "renderer_sha256": digest(__file__), "winner_architecture_id": primary["architecture_id"],
                "public_claim_supported_by_observed_test_mean": matched,
                "test_interval_upper_no_worse": comparison["bootstrap_95"][1] <= 0,
                "parameter_reduction": reduction, "parameter_reduction_fraction": reduction / BASELINE_PARAMETERS,
                "result": {"baseline": {key: baseline[key] for key in ["parameters", "dev_mean", "dev_std", "test_mean", "test_std", "train_diagnostic_mean", "train_diagnostic_std"]},
                           "winner": {key: winner[key] for key in ["parameters", "dev_mean", "dev_std", "test_mean", "test_std", "train_diagnostic_mean", "train_diagnostic_std"]},
                           "test_comparison": comparison},
                "protocol": {"steps": 5000, "train_conversations": 48375, "dev_conversations": 1000,
                             "test_conversations": 2599, "train_diagnostic_conversations": 256,
                             "selection_seeds": [0, 1, 2], "fresh_seeds": SEEDS,
                             "dataset_counts_source": "fixed experiment protocol; renderer does not reopen datasets",
                             "data_and_source_provenance": report["data_and_source_provenance"],
                             "checkpoint_selection": "best dev checkpoint within each complete 5000-step run",
                             "test_used_for_architecture_selection": False, "reselection_after_confirmation": False},
                "fresh_runtime": runtime, "winner_to_baseline_wall_ratio": ratio,
                "all_training_timing": timing, "run_statuses": statuses, "audit_counts": audit["counts"],
                "audit_warnings": audit["warnings"], "independent_diagnostic": diagnostic,
                "missing_optional_artifacts": missing_optional, "artifacts": artifacts}

    def asset(key, label):
        return link(root / artifacts[key]["path"], label, root) if key in artifacts else label + " (unavailable; see recorded limitations)"

    outcome = ("The frozen smaller model matched or beat the baseline's observed mean test CE across five fresh paired training seeds."
               if matched else "The frozen smaller model did not match baseline mean test CE across five fresh paired training seeds. The proposed smaller-without-worse-loss claim is not supported.")
    lines = ["# Final architecture-search results", "", outcome, "",
             f"**{reduction / BASELINE_PARAMETERS:.2%} fewer parameters**: {BASELINE_PARAMETERS:,} → {winner['parameters']:,}. "
             f"Winner: `{primary['name']}` (`{primary['architecture_id']}`).", "",
             "| Fresh-seed result | Original baseline | Frozen smaller model |", "|---|---:|---:|"]
    lines.append(f"| Unique learned parameters | {BASELINE_PARAMETERS:,} | {winner['parameters']:,} |")
    for title, key in [("Dev CE", "dev"), ("Test CE", "test"), ("Training diagnostic CE", "train_diagnostic")]:
        lines.append(f"| {title}, mean ± seed SD | {baseline[key+'_mean']:.6f} ± {baseline[key+'_std']:.6f} | {winner[key+'_mean']:.6f} ± {winner[key+'_std']:.6f} |")
    lines += ["", "CE is answer-token-weighted cross-entropy in nats, including answer EOS. "
              "Training diagnostic and dev values refer to the same selected checkpoint; the diagnostic uses 256 fixed training examples with dropout off.", "",
              f"Paired test difference (smaller − baseline): **{comparison['point']:+.6f}**; "
              f"paired bootstrap 95% interval **[{comparison['bootstrap_95'][0]:+.6f}, {comparison['bootstrap_95'][1]:+.6f}]**. "
              "The interval resamples five observed paired seeds and paired conversations. Matching an observed mean is weaker evidence than an interval wholly at or below zero, and does not establish performance on every seed or dataset.", "",
              "Every completed training run started from scratch and ran exactly **5,000 optimizer steps**, with fixed data, tokenizer, packing, optimizer and learning-rate schedule. "
              "The fixed split contains **48,375 training, 1,000 dev, and 2,599 test conversations**. "
              "The 256-conversation training diagnostic is for measurement; it is not the training dataset. "
              "Same-seed baseline/candidate runs consume paired training streams. Checkpoint selection uses the same best-dev rule within 5,000 steps.", "",
              "The controller froze one smallest architecture whose mean dev CE across search seeds 0, 1, 2 was no worse than the baseline. "
              "It then trained baseline and winner on fresh paired seeds 100–104. Neither fresh dev nor test results trigger architecture reselection.", "",
              "## Runtime tradeoff", "", "The following measurements cover the five fresh runs of each model on the allocated NVIDIA GPUs.", "",
              "| Mean time per 5,000-step run | Baseline | Smaller model |", "|---|---:|---:|"]
    for title, key in [("Process wall time, including startup/compilation/evaluation", "wall_seconds"),
                       ("Trainer elapsed time", "trainer_elapsed_seconds"), ("Training compute", "training_compute_seconds"),
                       ("Periodic dev evaluation", "dev_evaluation_seconds"),
                       ("Periodic training-diagnostic evaluation", "train_diagnostic_evaluation_seconds")]:
        lines.append(f"| {title} | {runtime['baseline']['means'][key]:.2f} s | {runtime['winner']['means'][key]:.2f} s |")
    lines += ["", f"The smaller architecture takes **{ratio:.2f}× the baseline process wall time** at the same 5,000 steps. "
              "Parameter reduction does not imply a proportional runtime reduction: repeated execution and sharing can trade fewer weights for more computation. "
              "Trainer elapsed time and process wall time have different boundaries; the process measurement also includes startup and controller polling.", ""]
    if any(row["wall_inferred_from_timestamps"] for role in runtime.values() for row in role["runs"]):
        lines.append("Some fresh process durations were inferred from retained UTC timestamps; the JSON manifest identifies each such run.")
    lines += [f"The ledger contains **{len(records):,} training attempts**: " + ", ".join(f"{n:,} {s.replace('_', ' ')}" for s, n in sorted(statuses.items())) + ". "
              f"The passing audit covers **{audit['counts']['architectures_with_completed_runs']:,} trained architectures**, "
              "ten fresh training runs, and ten completed final test evaluations. All attempts, including failures and interrupted work, remain logged.", "",
              f"Summed training-attempt GPU wall time: **{timing['training_attempt_gpu_wall_hours']:.3f} hours**; "
              f"completed training compute: **{timing['completed_training_compute_gpu_hours']:.3f} GPU-hours**. " + timing["notes"], "",
              "## Test-access disclosure", "",
              "This is the **same held-out split used in the earlier CUDA study**, not a newly untouched test set. "
              "It remains held out from training. At the user's request, a separate diagnostic evaluated the baseline and three snapshotted candidates using search seeds 0–2: "
              "**12 existing checkpoints × 2,599 conversations**, before the main architecture freeze. "
              "Its receipt confirms step-5,000 checkpoints. These are separate from the final best-dev checkpoints on fresh seeds 100–104.", "",
              "The recorded disclosure states that diagnostic scores, rankings, and qualitative outcomes were withheld from the search agents and controller while selection continued. "
              "This was procedural separation on a shared filesystem, not cryptographically enforced confidentiality. "
              "This renderer has read only the metrics-free receipt and disclosure; it has not opened diagnostic reports or arrays. "
              "No diagnostic score is pooled with the five-seed final comparison or used to choose another architecture.", "",
              f"The diagnostic completed at {receipt['completed_utc']} and took {diagnostic_seconds:.1f} seconds of service elapsed time on its recorded single GPU. "
              "It added evaluation work, not new training runs. The main artifact audit covers the main experiment; the separate receipt records the diagnostic's own integrity checks.", "",
              "Diagnostic reports were copied locally and hash-verified. Its full per-conversation arrays remain on the server: automatic approval review rejected their local download. "
              "That backup limitation does not change the completed main comparison or its audit; it must remain disclosed when describing where evidence is retained.", "",
              asset("disclosure", "Full test-access disclosure") + " · " + asset("diagnostic_receipt", "Metrics-free diagnostic completion receipt"), "",
              "## Artifacts and provenance", "",
              "- " + asset("final_report", "Final report") + " · " + asset("final_report_json", "Final report JSON") + " · " + asset("selection", "Frozen selection"),
              "- " + asset("fresh_curves", "Fresh train/dev curves") + " · " + asset("all_run_curves", "Dev-ranked comparison curves, up to 12 runs") + " · " + asset("search_progress", "Search progress"),
              "- " + asset("fresh_curve_metrics", "Fresh curve summary metrics") + " · " + asset("main_test_report", "Main five-seed test report"),
              "- " + asset("architecture", "Architecture and parameter allocation") + " · " + asset("architecture_ascii", "ASCII diagram") + " · " + asset("architecture_json", "Architecture JSON"),
              "- " + asset("run_log", "Every training attempt") + " · " + asset("architecture_log", "Every accepted/rejected architecture") + " · " + asset("search_summary", "Search summary"),
              "- " + asset("run_ledger", "Raw run ledger") + " · " + asset("architecture_ledger", "Raw architecture ledger") + " · " + asset("evaluation_ledger", "Final evaluation ledger"),
              "- " + asset("audit", "Passing artifact audit") + " · " + asset("audit_json", "Audit JSON") + " · " + asset("postprocessing", "Postprocessing completion") + " · " + asset("source_manifest", "Frozen source/data hashes"),
              "- [Delivery manifest and linked-artifact hashes](FINAL_DELIVERY.json)", "",
              f"The audit reports {len(audit['warnings'])} warnings; see the linked audit for retained partial artifacts, pruning, inferred timing, and other applicable limitations. "
              "The audit recomputes saved means and differences, but does not rerun bootstrap draws or independently establish access confidentiality. "
              "This delivery renderer packages existing evidence; it does not train, score, or select models.", ""]
    if missing_optional:
        lines += ["Unavailable optional artifacts: " + ", ".join(missing_optional) + ".", ""]
    # Check the linked inputs stayed fixed during rendering. No source report is rewritten.
    for value in artifacts.values():
        require(digest(root / value["path"]) == value["sha256"], "an input changed while rendering delivery")
    markdown = "\n".join(lines)
    manifest["delivery_markdown_sha256"] = hashlib.sha256(markdown.encode()).hexdigest()
    atomic_text(root / "FINAL_DELIVERY.md", markdown)
    atomic_text(root / "FINAL_DELIVERY.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed main experiment run directory")
    parser.add_argument("--diagnostic-receipt", type=Path, help="metrics-free COMPLETION_RECEIPT.json; defaults to the sole sibling independent-heldout receipt")
    parser.add_argument("--wait-until", type=timestamp, help="optional timezone-aware ISO deadline for waiting on postprocessing.json, normalized to UTC")
    args = parser.parse_args()
    try:
        if args.wait_until is not None:
            wait_for_postprocessing(args.root, args.wait_until)
        result = build(args.root, args.diagnostic_receipt)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"Delivery refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "markdown": str(args.root / "FINAL_DELIVERY.md"),
                      "manifest": str(args.root / "FINAL_DELIVERY.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
