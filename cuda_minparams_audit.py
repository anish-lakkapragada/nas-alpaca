"""Independently audit finished search artifacts without scoring any model.

Refuses to inspect test artifacts before FINAL_REPORT.json is complete. Reads
frozen evidence and writes only final-audit/audit-<UTC>.json and .md. Exit 0 is
an integrity pass, 1 is an integrity failure, and 2 is an incomplete experiment
or audit invocation error. Recorded checkpoint pruning is permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from cuda_minparams_search import (architecture_hash, parameter_count,
                                  baseline_spec, FRAME, BASELINE_PARAMETERS,
                                  RECIPE)

PROJECT = Path(__file__).resolve().parent
SEARCH_SEEDS = (0, 1, 2)
FRESH_SEEDS = (100, 101, 102, 103, 104)
TRAIN_FILES = ("cuda_train.py", "cuda_model.py", "data.py", "data/train.json", "data/tokenizer.json")
FIXED_ARGS = {**RECIPE, "log_every": 25, "train_eval_conversations": 256,
              "train_eval_seed": 20261111, "precision": "bf16", "compile": True,
              "compile_mode": "default", "projection_chunk_size": 0}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def parse_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return result.astimezone(timezone.utc)


def same_number(a, b, tolerance=1e-9):
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=0, abs_tol=tolerance))


class Audit:
    def __init__(self, root, project_root=PROJECT):
        self.root, self.project = Path(root).resolve(), Path(project_root).resolve()
        # This gate precedes all accesses to held-out outputs and all writes.
        self.report = read_json(self.root / "FINAL_REPORT.json")
        if self.report.get("status") not in ("complete", "no_qualified_candidate"):
            raise RuntimeError("final report is not complete; refusing held-out artifact access")
        self.initial_report_sha256 = digest(self.root / "FINAL_REPORT.json")
        self.failures, self.warnings = [], []
        self.counts = Counter()
        self.runs, self.architectures, self.run_rows, self.histories = {}, {}, {}, defaultdict(list)
        self.recomputed = {}
        self.baseline_id = architecture_hash(FRAME, baseline_spec())

    def check(self, condition, code, message, path=None):
        if not condition:
            self.failures.append({"code": code, "message": message, **({"path": str(path)} if path else {})})
        return bool(condition)

    def warn(self, code, message, path=None):
        self.warnings.append({"code": code, "message": message, **({"path": str(path)} if path else {})})

    def inside(self, value):
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"artifact path escapes experiment root: {path}")
        return path

    def ledger(self, name):
        path = self.root / name
        rows = []
        if not self.check(path.is_file(), "ledger_missing", f"missing {name}", path):
            return rows
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError) as error:
                self.check(False, "ledger_parse", f"line {number}: {error}", path)
        return rows

    def sources(self):
        path = self.root / "workflow_source_manifest.json"
        self.manifest = read_json(path)["sha256"]
        for name, expected in self.manifest.items():
            source = (self.project / name).resolve()
            if not source.is_relative_to(self.project):
                self.check(False, "source_path", "source manifest path escapes project", source)
                continue
            self.check(source.is_file() and digest(source) == expected,
                       "source_hash", f"current source/data differs from frozen manifest: {name}", source)
            archived = self.root / "source" / name
            if not name.startswith("data/"):
                self.check(archived.is_file() and digest(archived) == expected,
                           "archived_source_hash", f"archived source differs: {name}", archived)
        for name in TRAIN_FILES:
            self.check(name in self.manifest, "training_source_missing", f"training source absent from manifest: {name}")
        search_provenance = read_json(self.root / "provenance.json")
        for name, expected in search_provenance["source_sha256"].items():
            self.check(self.manifest.get(name) == expected, "search_source_hash", f"search provenance differs: {name}")
        self.counts["frozen_source_and_data_files"] = len(self.manifest)

    def architecture_ledger(self):
        rows = self.ledger("architecture_registry.jsonl")
        rejected = set()
        for row in rows:
            try:
                aid = row["architecture_id"]
                if row.get("status") == "rejected":
                    rejected.add(aid)
                    self.check(bool(row.get("reason")), "rejection_reason", "rejected architecture lacks a reason")
                    continue
                self.check(aid not in self.architectures, "duplicate_architecture", "architecture accepted more than once")
                path = self.inside(row["spec_path"])
                spec = read_json(path)
                frame = row["frame"]
                count = parameter_count(frame, spec)
                self.check(digest(path) == row["file_sha256"], "spec_file_hash", "spec file changed", path)
                self.check(architecture_hash(frame, spec) == aid, "architecture_hash", "frame/spec identity differs", path)
                self.check(count == row["parameters"], "architecture_parameters", "unique parameter count differs", path)
                self.check(all(frame.get(key) == FRAME[key] for key in ("vocab_size", "max_len", "dropout")),
                           "fixed_frame", "vocabulary/context/dropout changed", path)
                if aid == self.baseline_id:
                    self.check(frame == FRAME and count == BASELINE_PARAMETERS,
                               "baseline_identity", "original baseline frame/count changed", path)
                else:
                    self.check(count < BASELINE_PARAMETERS and count > 0,
                               "candidate_parameter_cap", "candidate does not strictly reduce total parameters", path)
                    self.check(spec.get("search_space") == "expanded", "candidate_scope", "candidate lacks expanded-space marker", path)
                self.architectures[aid] = {**row, "spec": spec}
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.check(False, "architecture_record", str(error))
        self.check(self.baseline_id in self.architectures, "baseline_missing", "original baseline absent from architecture ledger")
        logged = {self.inside(row["spec_path"]) for row in self.architectures.values()}
        for path in (self.root / "specs").glob("*.json"):
            self.check(path.resolve() in logged, "unlogged_spec", "spec exists without accepted architecture record", path)
        self.counts.update(accepted_architectures=len(self.architectures), rejected_architectures=len(rejected))

    def training_ledgers(self):
        for row in self.ledger("run_registry.jsonl"):
            try:
                run_id = row["job"]["run_id"]
                self.histories[run_id].append(row)
                self.run_rows[run_id] = row
            except (KeyError, TypeError):
                self.check(False, "run_record", "run ledger row lacks job.run_id")
        allowed = {"completed", "failed", "interrupted", "timed_out", "not_admitted"}
        for run_id, row in self.run_rows.items():
            try:
                job, status = row["job"], row["status"]
                aid = job["architecture_id"]
                self.check(aid in self.architectures, "run_architecture", "run cites unregistered architecture", run_id)
                self.check(status in allowed, "unfinished_run", f"terminal report retains run status {status}", run_id)
                self.check(job.get("steps") == 5000, "run_steps", "run ledger does not declare exactly 5000 steps", run_id)
                path = self.inside(row["path"])
                self.check(path == self.root / "attempts" / run_id, "run_directory", "run path/ID disagree", path)
                history = self.histories[run_id]
                launched = any(item.get("status") == "running" for item in history)
                self.counts["launched_training_attempts"] += int(launched)
                self.counts[status + "_attempts"] += 1
                if launched:
                    self.check((self.root / "logs" / (run_id + ".log")).is_file(),
                               "stdout_missing", "launched run has no retained process log", path)
                    self.check(bool(row.get("started_utc")), "run_start", "launched run lacks start timestamp", path)
                if status == "completed":
                    self.check(launched, "unlogged_launch", "completed run has no running ledger event", path)
                    self.completed_run(run_id, row, path)
                elif launched:
                    explained = bool(row.get("failure_reason") or row.get("stop_reason"))
                    if not explained and status == "timed_out" and job.get("phase") == "final":
                        # The frozen finalizer records this terminal status and
                        # duration, but does not add a separate reason string.
                        self.warn("writer_omitted_failure_reason",
                                  "Final-confirmation writer recorded status=timed_out without a reason string; status is preserved, not repaired.", path)
                    else:
                        self.check(explained, "failure_reason", "failed/interrupted run lacks explanation", path)
                    if path.is_dir():
                        self.warn("partial_run", f"{status} attempt retained; excluded from scientific comparisons", path)
                self.checkpoints(row, path)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.check(False, "run_record", f"{run_id}: {error}")
        for path in (self.root / "attempts").iterdir():
            if path.is_dir():
                self.check(path.name in self.run_rows, "unlogged_attempt_directory", "training attempt directory has no ledger record", path)
        for path in (self.root / "logs").glob("*.log"):
            self.check(path.stem in self.run_rows, "unlogged_process_log", "trainer log has no run ledger record", path)
        # The scheduler also emits launch events; match both independent ledgers.
        for event in self.ledger("events.jsonl"):
            if event.get("event") == "launched":
                self.check(event.get("run_id") in self.run_rows, "unlogged_launch_event", "launch event missing from run registry")
        duplicate_keys = Counter((r["architecture_id"], r["seed"]) for r in self.runs.values())
        for key, count in duplicate_keys.items():
            self.check(count == 1, "duplicate_completed_seed", f"architecture/seed has {count} completed attempts: {key}")
        # Every completed architecture must have consumed exactly the same stream
        # for a given seed, not merely the finalists.
        for seed in sorted({r["seed"] for r in self.runs.values()}):
            rows = [r for r in self.runs.values() if r["seed"] == seed]
            self.check(len({r["summary"]["answer_tokens_seen"] for r in rows}) == 1,
                       "paired_training_tokens", f"training token totals differ across architectures for seed {seed}")
            self.check(len({r["token_progression_sha256"] for r in rows}) == 1,
                       "paired_token_progression", f"per-window supervised token progression differs for seed {seed}")
            diagnostics = {json.dumps(r["config"]["train_evaluation"], sort_keys=True) for r in rows}
            self.check(len(diagnostics) == 1, "paired_diagnostic", f"training diagnostic differs for seed {seed}")
        self.counts["completed_training_runs"] = len(self.runs)
        self.counts["architectures_with_completed_runs"] = len({r["architecture_id"] for r in self.runs.values()})

    def completed_run(self, run_id, row, path):
        config, summary = read_json(path / "config.json"), read_json(path / "summary.json")
        job, args = row["job"], config["args"]
        architecture = self.architectures[job["architecture_id"]]
        self.check(config["model"] == architecture["frame"] and architecture_hash(config["model"], config["spec"]) == job["architecture_id"],
                   "run_model_hash", "run frame/spec differs from architecture registry", path)
        count = parameter_count(config["model"], config["spec"])
        self.check(config.get("parameters") == summary.get("parameters") == row.get("parameters") == count,
                   "run_parameter_count", "run/config/summary parameter counts disagree", path)
        self.check(args.get("seed") == job["seed"], "run_seed", "trainer seed differs from ledger", path)
        for name, value in FIXED_ARGS.items():
            self.check(args.get(name) == value, "fixed_recipe", f"{name}: expected {value!r}, got {args.get(name)!r}", path)
        self.check(not args.get("resume") and not args.get("benchmark_steps") and not summary.get("benchmark"),
                   "fresh_training", "run resumed weights or used benchmark mode", path)
        self.check(summary.get("steps") == summary.get("schedule_steps") == 5000,
                   "completed_steps", "completed run is not exactly 5000 optimizer steps", path)
        for name in TRAIN_FILES:
            self.check(config["sha256"].get(name) == self.manifest.get(name),
                       "run_source_hash", f"run source/data differs: {name}", path)
        for name, expected in row.get("source_hashes", {}).items():
            self.check(self.manifest.get(name) == expected, "ledger_source_hash", f"ledger source differs: {name}", path)
        optimizer = config.get("optimizer", {})
        self.check(optimizer.get("name") == "AdamWNoBiasCorrection" and optimizer.get("bias_correction") is False
                   and optimizer.get("betas") == [.9, .95] and optimizer.get("eps") == 1e-8,
                   "optimizer", "optimizer differs from fixed recipe", path)
        diagnostic = config.get("train_evaluation", {})
        self.check(diagnostic.get("conversations") == 256 and diagnostic.get("dropout") is False
                   and diagnostic.get("seed") == 20261111 and diagnostic.get("split") == "train",
                   "training_diagnostic", "fixed train diagnostic metadata differs", path)
        self.check(type(summary.get("answer_tokens_seen")) is int and summary["answer_tokens_seen"] > 0,
                   "training_token_count", "completed run lacks positive integer supervised token count", path)
        self.check(summary == row.get("summary"), "ledger_summary", "summary.json differs from final run ledger summary", path)
        gpu = row.get("physical_gpu", {}).get("physical_index")
        self.check(gpu in (0, 1, 2, 3) and config.get("hardware", {}).get("physical_gpu_ids") == str(gpu),
                   "physical_gpu", "physical GPU assignment differs between config and ledger", path)
        records = []
        for number, line in enumerate((path / "log.jsonl").read_text().splitlines(), 1):
            try:
                records.append(json.loads(line))
            except ValueError as error:
                self.check(False, "curve_log_parse", f"line {number}: {error}", path)
        evaluation = [r for r in records if r.get("event") == "evaluation"]
        training = [r for r in records if r.get("event") == "training"]
        self.check([r.get("step") for r in evaluation] == list(range(0, 5001, 250)),
                   "dev_curve_steps", "dev/train-diagnostic curve lacks exact scheduled evaluations", path)
        self.check([r.get("step") for r in training] == list(range(25, 5001, 25)),
                   "training_curve_steps", "minibatch training curve lacks exact scheduled points", path)
        if evaluation:
            finite = all(same_number(r.get("dev_loss"), r.get("dev_loss")) and
                         same_number(r.get("train_eval_loss"), r.get("train_eval_loss")) for r in evaluation)
            self.check(finite, "curve_nonfinite", "nonfinite/missing dev or training diagnostic CE", path)
            if finite:
                best = min(evaluation, key=lambda r: r["dev_loss"])
                for key, value in (("best_dev_loss", best["dev_loss"]), ("best_step", best["step"]),
                                   ("train_eval_loss_at_best_dev", best["train_eval_loss"]),
                                   ("final_dev_loss", evaluation[-1]["dev_loss"]),
                                   ("final_train_eval_loss", evaluation[-1]["train_eval_loss"])):
                    self.check(same_number(summary.get(key), value), "curve_summary", f"summary {key} differs from saved curve", path)
        self.check(all(same_number(r.get("train_loss"), r.get("train_loss")) for r in training),
                   "training_curve_nonfinite", "nonfinite minibatch CE", path)
        token_progression = [(r.get("step"), r.get("answer_tokens_seen"), r.get("train_loss_answer_tokens")) for r in training]
        if training:
            self.check(training[-1].get("answer_tokens_seen") == summary["answer_tokens_seen"]
                       and sum(r.get("train_loss_answer_tokens", 0) for r in training) == summary["answer_tokens_seen"],
                       "curve_training_tokens", "training token progression disagrees with final total", path)
        token_progression_sha = hashlib.sha256(json.dumps(token_progression).encode()).hexdigest()
        for name in ("curves.png", "curves.svg", "metrics.json"):
            self.check((path / name).is_file(), "curve_artifact_missing", f"missing saved {name}", path)
        if (path / "metrics.json").is_file():
            metrics = read_json(path / "metrics.json")
            self.check(same_number(metrics.get("best_dev_loss"), summary["best_dev_loss"])
                       and metrics.get("parameters") == count, "rendered_metrics", "rendered metrics disagree", path)
        self.runs[run_id] = {"architecture_id": job["architecture_id"], "seed": job["seed"], "phase": job["phase"],
                             "path": str(path), "config": config, "summary": summary, "ledger": row,
                             "token_progression_sha256": token_progression_sha}

    def checkpoints(self, row, path):
        recorded = row.get("checkpoints", {})
        for name, info in recorded.items():
            checkpoint = self.inside(info["path"])
            self.check(checkpoint == path / name, "checkpoint_path", "checkpoint is attached to the wrong run", checkpoint)
            if info.get("pruned"):
                self.counts["recorded_pruned_checkpoints"] += 1
                self.check(bool(info.get("sha256")) and info.get("bytes", 0) > 0,
                           "pruned_checkpoint_provenance", "pruned checkpoint lacks hash/size", checkpoint)
                if checkpoint.exists():
                    self.warn("pruned_but_present", "checkpoint marked pruned still exists", checkpoint)
                continue
            self.check(checkpoint.is_file(), "checkpoint_missing", "retained checkpoint is absent without pruning record", checkpoint)
            if checkpoint.is_file():
                self.check(checkpoint.stat().st_size == info["bytes"] and digest(checkpoint) == info["sha256"],
                           "checkpoint_hash", "retained checkpoint bytes differ from ledger", checkpoint)
                self.counts["verified_checkpoint_files"] += 1
                self.counts["verified_checkpoint_bytes"] += checkpoint.stat().st_size
        for checkpoint in path.glob("*.pt"):
            if checkpoint.name in recorded:
                continue
            if row.get("status") in ("failed", "interrupted", "timed_out"):
                size = checkpoint.stat().st_size
                self.counts["unverified_partial_checkpoint_files"] += 1
                self.counts["unverified_partial_checkpoint_bytes"] += size
                self.recomputed.setdefault("unverified_partial_artifacts", []).append({
                    "path": str(checkpoint), "bytes": size, "run_id": row["job"]["run_id"],
                    "status": row["status"], "failure_reason": row.get("failure_reason"),
                    "stop_reason": row.get("stop_reason"), "trusted_ledger_hash": None})
                self.warn("unverified_partial_checkpoint",
                          f"{row['status']} attempt retains {size} checkpoint bytes without a trusted ledger hash; excluded from all completed scientific evidence.", checkpoint)
            else:
                self.check(False, "unlogged_checkpoint", "checkpoint lacks ledger hash/size", checkpoint)
        if row.get("status") == "completed":
            self.check("best.pt" in recorded and "final.pt" in recorded,
                       "checkpoint_records_missing", "completed run lacks best/final checkpoint provenance", path)

    def selection(self):
        self.frozen = read_json(self.root / "selection.json")
        self.freeze_hash = digest(self.root / "selection.json")
        self.freeze_time = parse_time(self.frozen["frozen_utc"])
        self.check(self.frozen.get("selection_seeds") == list(SEARCH_SEEDS)
                   and self.frozen.get("fresh_seeds") == list(FRESH_SEEDS), "selection_seeds", "selection/fresh seed protocol changed")
        self.check(self.frozen.get("validation_used") is False, "selection_test_access", "selection does not exclude test use")
        self.check(self.report.get("selection_freeze_sha256") == self.freeze_hash, "freeze_hash", "final report references another selection")
        grouped = defaultdict(dict)
        for run in self.runs.values():
            if run["seed"] in SEARCH_SEEDS and run["phase"] != "final":
                grouped[run["architecture_id"]][run["seed"]] = run
                self.check(parse_time(run["ledger"]["ended_utc"]) <= self.freeze_time,
                           "search_after_freeze", "search training completed after architecture freeze", run["path"])
        baseline = grouped.get(self.baseline_id, {})
        if not self.check(set(baseline) == set(SEARCH_SEEDS), "selection_baseline_seeds", "baseline lacks exactly three search seeds"):
            return
        reference = statistics.mean(r["summary"]["best_dev_loss"] for r in baseline.values())
        qualified = []
        for aid, seeds in grouped.items():
            if aid == self.baseline_id or set(seeds) != set(SEARCH_SEEDS):
                continue
            mean = statistics.mean(r["summary"]["best_dev_loss"] for r in seeds.values())
            if mean <= reference:
                qualified.append({"architecture_id": aid, "parameters": self.architectures[aid]["parameters"], "mean_dev_loss": mean})
        qualified.sort(key=lambda r: (r["parameters"], r["mean_dev_loss"]))
        self.recomputed.update(search_baseline_mean_dev=reference, qualified_candidates=qualified)
        expected_status = "qualified" if qualified else "no_qualified_candidate"
        self.check(self.frozen.get("status") == expected_status, "qualification_status", "selection qualification differs from recomputed means")
        expected = qualified[0]["architecture_id"] if qualified else None
        self.check(self.frozen.get("primary_architecture_id") == expected,
                   "smallest_qualifier", "frozen winner is not the smallest three-seed mean qualifier")
        self.check({r["architecture_id"] for r in self.frozen.get("qualified_candidates", [])} == {r["architecture_id"] for r in qualified},
                   "qualification_inventory", "frozen qualifier inventory differs from all completed search runs")
        for role in ("baseline", "primary"):
            entry = self.frozen.get(role)
            if entry is None:
                continue
            aid = entry["architecture_id"]
            self.check(architecture_hash(entry["frame"], entry["spec"]) == aid,
                       "frozen_architecture", "frozen frame/spec do not hash to selected architecture")
            self.check(aid in grouped and set(grouped[aid]) == set(SEARCH_SEEDS),
                       "frozen_run_set", "frozen entry lacks three search runs")
            if aid in grouped:
                for seed, run in grouped[aid].items():
                    self.check(self.inside(entry["run_paths"][str(seed)]) == Path(run["path"]),
                               "frozen_run_path", "frozen entry points to another run")
                mean = statistics.mean(r["summary"]["best_dev_loss"] for r in grouped[aid].values())
                self.check(same_number(entry.get("mean_dev_loss"), mean), "frozen_dev_mean", "frozen dev mean differs from saved runs")
        for role, field in (("baseline", "search_baseline_mean_dev"), ("primary", "search_winner_mean_dev")):
            if self.report["status"] == "complete":
                self.check(same_number(self.report.get(field), self.frozen[role]["mean_dev_loss"]),
                           "report_search_mean", f"reported {field} differs from frozen selection")

    def confirmation(self):
        primary_id = self.frozen["primary_architecture_id"]
        final = [r for r in self.runs.values() if r["phase"] == "final"]
        expected = {(aid, seed) for aid in (self.baseline_id, primary_id) for seed in FRESH_SEEDS}
        self.check({(r["architecture_id"], r["seed"]) for r in final} == expected and len(final) == 10,
                   "fresh_pairs", "completed confirmation must be exactly baseline/winner x five fresh seeds")
        for run in final:
            self.check(parse_time(run["ledger"]["started_utc"]) >= self.freeze_time,
                       "fresh_before_freeze", "fresh-seed training began before architecture selection froze", run["path"])
            self.check(run["ledger"].get("selection_freeze_sha256") == self.freeze_hash,
                       "fresh_freeze_hash", "fresh run cites a different frozen selection", run["path"])
        self.confirmed = read_json(self.root / "final_confirmation.json")
        self.confirmation_time = parse_time(self.confirmed["frozen_utc"])
        self.check(self.confirmed.get("selection_freeze_sha256") == self.freeze_hash
                   and self.confirmed.get("test_accessed") is False
                   and self.confirmed.get("fresh_seeds_used_for_architecture_selection") is False,
                   "checkpoint_freeze", "fresh checkpoint freeze metadata differs")
        self.checkpoint_identities_verified = self.check(
            len(self.confirmed["runs"]) == 10 and {r["run"] for r in self.confirmed["runs"]} == {r["path"] for r in final},
            "checkpoint_freeze_runs", "frozen checkpoint inventory differs from ten fresh training runs")
        final_by_path = {r["path"]: r for r in final}
        completions = [parse_time(r["ledger"]["ended_utc"]) for r in final]
        self.all_fresh_completed_time = max(completions) if completions else None
        for entry in self.confirmed["runs"]:
            run = final_by_path.get(entry["run"])
            if run is None:
                continue
            path = Path(run["path"])
            expected_role = "baseline" if run["architecture_id"] == self.baseline_id else "winner"
            identity = self.check(entry.get("seed") == run["seed"] and entry.get("role") == expected_role,
                                  "frozen_checkpoint_role_seed", "checkpoint freeze role/seed differs from its training run", path)
            self.checkpoint_identities_verified = self.checkpoint_identities_verified and identity
            self.check(parse_time(run["ledger"]["ended_utc"]) <= self.confirmation_time,
                       "training_after_checkpoint_freeze", "fresh run ended after checkpoint freeze", path)
            for filename, field in (("config.json", "config_sha256"), ("summary.json", "summary_sha256"), ("best.pt", "checkpoint_sha256")):
                verified = self.check(digest(path / filename) == entry[field], "fresh_artifact_hash", f"frozen {filename} hash differs", path)
                self.checkpoint_identities_verified = self.checkpoint_identities_verified and verified
            verified = self.check(entry["checkpoint_sha256"] == run["ledger"]["checkpoints"]["best.pt"]["sha256"],
                                  "fresh_checkpoint_ledger", "frozen best checkpoint differs from ledger", path)
            self.checkpoint_identities_verified = self.checkpoint_identities_verified and verified
        self.final_runs = final_by_path
        self.check(self.report.get("fresh_seeds") == list(FRESH_SEEDS)
                   and self.report.get("fresh_seeds_used_for_architecture_selection") is False
                   and self.report.get("test_used_for_selection") is False
                   and self.report.get("reselection_after_test") is False,
                   "no_reselection", "final report does not preserve frozen selection protocol")
        self.check(self.report.get("winner_architecture_id") == primary_id, "reported_winner", "report selected a different winner")
        for name, expected in self.report.get("data_and_source_provenance", {}).items():
            self.check(self.manifest.get(name) == expected, "test_source_hash", f"test source/data differs: {name}")
        for role, aid in (("baseline", self.baseline_id), ("winner", primary_id)):
            actual = sorted([r for r in final if r["architecture_id"] == aid], key=lambda r: r["seed"])
            reported = self.report[role]
            self.check([r["run"] for r in reported["runs"]] == [r["path"] for r in actual],
                       "report_fresh_runs", f"{role} report lists different fresh runs")
            for name, values in (("dev", [r["summary"]["best_dev_loss"] for r in actual]),
                                 ("train_diagnostic", [r["summary"]["train_eval_loss_at_best_dev"] for r in actual])):
                if len(values) == 5:
                    self.check(same_number(reported.get(name + "_mean"), statistics.mean(values))
                               and same_number(reported.get(name + "_std"), statistics.stdev(values)),
                               "report_fresh_metric", f"reported {role} {name} mean/std differs")

    def test_artifacts(self):
        import numpy as np
        events = self.ledger("evaluation_registry.jsonl")
        latest, histories = {}, defaultdict(list)
        rewrite_warning_emitted = False
        for row in events:
            try:
                output = str(self.inside(row["output"]))
                latest[output] = row
                histories[output].append(row)
                started = parse_time(row["started_utc"])
                after_selection = self.check(started >= self.freeze_time,
                                             "test_before_selection", "test evaluation started before architecture selection froze", output)
                after_training = self.check(self.all_fresh_completed_time is not None and started >= self.all_fresh_completed_time,
                                            "test_before_fresh_completion", "test evaluation started before all ten fresh training runs completed", output)
                trained = self.final_runs.get(row.get("run"))
                evaluation_identity_matches = (trained is not None and row.get("checkpoint_sha256") ==
                                               trained["ledger"]["checkpoints"]["best.pt"]["sha256"])
                if started < self.confirmation_time:
                    if after_selection and after_training and self.checkpoint_identities_verified and evaluation_identity_matches:
                        if not rewrite_warning_emitted:
                            self.warn("checkpoint_freeze_timestamp_rewritten",
                                      "Finalizer resume rewrote the checkpoint-freeze timestamp after an earlier evaluation. Exact frozen architecture/checkpoint identities match, and all fresh training completed before every accepted test start; no reselection inferred.")
                            rewrite_warning_emitted = True
                    else:
                        self.check(False, "test_before_freeze", "test evaluation predates checkpoint freeze without complete unchanged-checkpoint evidence", output)
                self.check(row["run"] in self.final_runs, "unexpected_test_run", "test evaluated a nonfinal architecture/seed", output)
            except (KeyError, ValueError, TypeError) as error:
                self.check(False, "test_ledger_record", str(error))
        completed = [row for row in latest.values() if row.get("status") == "completed"]
        by_run = defaultdict(list)
        for row in completed:
            by_run[row["run"]].append(row)
        self.check(set(by_run) == set(self.final_runs), "test_run_inventory", "test outputs do not cover exactly ten frozen fresh runs")
        for run_path, rows in by_run.items():
            self.check(len(rows) == 1, "duplicate_test_evaluation", "multiple successful test evaluations of one frozen run", run_path)
        scores = {}
        arrays = {}
        for row in completed:
            try:
                output = self.inside(row["output"])
                value = read_json(output)
                run = self.final_runs[row["run"]]
                checkpoint_sha = run["ledger"]["checkpoints"]["best.pt"]["sha256"]
                self.check(value.get("run") == row["run"] and value.get("seed") == run["seed"]
                           and value.get("checkpoint_sha256") == row.get("checkpoint_sha256") == checkpoint_sha,
                           "test_checkpoint_identity", "test output references another checkpoint/seed", output)
                self.check(value.get("config_sha256") == digest(Path(run["path"]) / "config.json"),
                           "test_config_identity", "test output config hash differs", output)
                self.check(value.get("split") == "validation" and value.get("conversations") == 2599
                           and value.get("precision") == "bf16", "test_protocol", "test split/size/precision differs", output)
                self.check(digest(output) == row.get("json_sha256")
                           and digest(output.with_suffix(".conversations.npz")) == row.get("conversations_sha256"),
                           "test_output_hash", "immutable test output bytes differ from ledger", output)
                self.check(parse_time(value["completed_utc"]) >= parse_time(row["started_utc"])
                           and parse_time(value["completed_utc"]) <= parse_time(row["ended_utc"]),
                           "test_timestamps", "test completion timestamp lies outside recorded evaluation", output)
                with np.load(output.with_suffix(".conversations.npz"), allow_pickle=False) as archive:
                    sums, counts = archive["sums"], archive["counts"]
                    self.check(len(sums) == len(counts) == 2599 and bool(np.isfinite(sums).all())
                               and bool((counts > 0).all()) and int(counts.sum()) == value["answer_tokens"],
                               "test_conversation_arrays", "test conversation archive dimensions/counts differ", output)
                    score = float(sums.sum() / counts.sum())
                    self.check(same_number(score, value["loss"]), "test_score_recomputation", "JSON CE differs from conversation sums/counts", output)
                    arrays[row["run"]] = (sums.copy(), counts.copy())
                scores[row["run"]] = score
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
                self.check(False, "test_artifact", f"{row.get('output')}: {error}")
        self.counts["completed_test_evaluations"] = len(completed)
        self.counts["total_test_evaluation_attempts"] = len(latest)
        for path in (self.root / "final-evaluations").glob("*.json"):
            self.check(str(path.resolve()) in latest, "unlogged_test_output", "test output is absent from evaluation ledger", path)
        expected_runs = self.confirmed["runs"]
        if not self.check(len(scores) == 10 and set(scores) == {r["run"] for r in expected_runs},
                          "recomputed_test_inventory", "cannot recompute all five paired test comparisons"):
            return
        counts = [arrays[r["run"]][1] for r in expected_runs]
        self.check(all(np.array_equal(counts[0], value) for value in counts[1:]),
                   "paired_test_tokens", "test conversation token counts differ between models/seeds")
        by_role = {role: sorted([r for r in expected_runs if r["role"] == role], key=lambda r: r["seed"])
                   for role in ("baseline", "winner")}
        deltas = [scores[w["run"]] - scores[b["run"]] for b, w in zip(by_role["baseline"], by_role["winner"])]
        for role, rows in by_role.items():
            values = [scores[r["run"]] for r in rows]
            self.check(same_number(self.report[role].get("test_mean"), statistics.mean(values))
                       and same_number(self.report[role].get("test_std"), statistics.stdev(values)),
                       "reported_test_statistics", f"{role} reported test mean/std differs from saved conversation data")
            for run in self.report[role]["runs"]:
                self.check(same_number(run["test_loss"], scores[run["run"]]), "reported_seed_test", "reported seed test loss differs")
        delta = statistics.mean(deltas)
        self.check(same_number(delta, self.report["test_comparison"].get("point")),
                   "reported_test_difference", "reported mean paired test difference is incorrect")
        matched = delta <= 0
        self.check(self.report.get("test_mean_no_worse") is matched
                   and self.report.get("public_claim_supported_by_mean") is matched,
                   "claim_eligibility", "public claim eligibility disagrees with recomputed test means")
        baseline_params, candidate_params = BASELINE_PARAMETERS, self.frozen["primary"]["parameters"]
        self.check(self.report["baseline"]["parameters"] == baseline_params
                   and self.report["winner"]["parameters"] == candidate_params
                   and self.report.get("parameter_reduction") == baseline_params - candidate_params
                   and same_number(self.report.get("parameter_reduction_fraction"), (baseline_params - candidate_params) / baseline_params),
                   "reported_parameter_reduction", "reported parameter reduction differs from unique counts")
        combined = read_json(self.root / "final-test.json")
        self.check(combined.get("run_order") == [r["run"] for r in expected_runs],
                   "combined_test_order", "combined test archive run order differs")
        with np.load(self.root / "final-test.conversations.npz", allow_pickle=False) as archive:
            for index, run in enumerate(expected_runs):
                sums, counts = arrays[run["run"]]
                self.check(np.array_equal(archive[f"run_{index}_sums"], sums)
                           and np.array_equal(archive[f"run_{index}_counts"], counts),
                           "combined_test_arrays", "combined test archive differs from immutable per-run archive")
                self.check(same_number(combined["results"][run["run"]]["loss"], scores[run["run"]]),
                           "combined_test_score", "combined test JSON differs from per-run archive")
        self.recomputed.update(test_baseline_mean=statistics.mean(scores[r["run"]] for r in by_role["baseline"]),
                               test_winner_mean=statistics.mean(scores[r["run"]] for r in by_role["winner"]),
                               paired_test_differences=deltas, mean_paired_test_difference=delta,
                               public_claim_supported_by_mean=matched)
        self.warn("bootstrap_interval_not_recomputed", "Audit recomputed exact test means/differences; bootstrap interval draws were not rerun.")

    def time_totals(self):
        rows = list(self.run_rows.values())
        launched = [r for r in rows if r.get("started_utc")]
        starts, ends = [], []
        gpu = defaultdict(lambda: {"attempts": 0, "completed": 0, "wall_seconds": 0., "training_compute_seconds": 0.,
                                   "inferred_duration_attempts": 0, "inferred_wall_seconds": 0., "unknown_duration_attempts": 0})
        for row in launched:
            index = str(row.get("physical_gpu", {}).get("physical_index", "unknown"))
            gpu[index]["attempts"] += 1
            gpu[index]["completed"] += int(row["status"] == "completed")
            start, end = parse_time(row["started_utc"]), None
            if row.get("ended_utc"):
                end = parse_time(row["ended_utc"])
                self.check(end >= start, "run_timestamp_order", "run ended before it started", row["path"])
                starts.append(start); ends.append(end)
            duration = row.get("wall_seconds")
            if duration is not None and (not same_number(duration, duration) or duration < 0):
                self.check(False, "invalid_wall_duration", "recorded wall duration is negative or nonfinite", row["path"])
                duration = None
            if duration is None and end is not None and end >= start:
                duration = (end - start).total_seconds()
                gpu[index]["inferred_duration_attempts"] += 1
                gpu[index]["inferred_wall_seconds"] += duration
                self.warn("duration_inferred_from_timestamps",
                          f"Missing wall_seconds inferred as {duration:.3f}s from retained UTC start/end timestamps; this may include recovery/polling delay and is not measured GPU compute.", row["path"])
            if duration is None:
                gpu[index]["unknown_duration_attempts"] += 1
                self.warn("unknown_attempt_duration", "No usable wall duration or start/end pair; excluded from timing total rather than treated as a measured zero.", row["path"])
            else:
                gpu[index]["wall_seconds"] += duration
            if row["status"] == "completed":
                gpu[index]["training_compute_seconds"] += row["summary"].get("training_compute_seconds", 0) or 0
        total_wall = sum(g["wall_seconds"] for g in gpu.values())
        compute = sum(g["training_compute_seconds"] for g in gpu.values())
        self.recomputed["observed_timing"] = {
            "by_physical_gpu": dict(gpu), "training_attempt_gpu_wall_seconds": total_wall,
            "training_attempt_gpu_wall_hours": total_wall / 3600,
            "inferred_duration_attempts": sum(g["inferred_duration_attempts"] for g in gpu.values()),
            "inferred_wall_seconds": sum(g["inferred_wall_seconds"] for g in gpu.values()),
            "unknown_duration_attempts": sum(g["unknown_duration_attempts"] for g in gpu.values()),
            "completed_training_compute_seconds": compute, "completed_training_compute_gpu_hours": compute / 3600,
            "training_attempt_elapsed_span_seconds": (max(ends) - min(starts)).total_seconds() if starts else None,
            "notes": "GPU wall totals sum scheduler-observed trainer durations, including preparation/evaluation/compilation and polling delay. Missing durations are inferred from valid UTC timestamps and disclosed separately; unknown durations are excluded. Inferred intervals can include recovery delay. Compute totals include completed training steps only; evaluation-only worker time and cloud billing idle time are excluded."}

    def execute(self):
        stages = [("sources", self.sources), ("architectures", self.architecture_ledger),
                  ("training", self.training_ledgers), ("selection", self.selection)]
        if self.report["status"] == "complete":
            stages += [("confirmation", self.confirmation), ("test_artifacts", self.test_artifacts)]
        else:
            self.check(self.report.get("test_accessed") is False
                       and self.report.get("public_claim_supported_by_mean") is False,
                       "no_qualifier_claim", "no-qualifier report allows a test claim")
            if (self.root / "evaluation_registry.jsonl").exists():
                self.check(not (self.root / "evaluation_registry.jsonl").read_text().strip(),
                           "unexpected_test_access", "no-qualifier experiment contains test evaluations")
        for name, stage in stages:
            try:
                stage()
            except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, ZeroDivisionError) as error:
                self.check(False, "audit_stage_incomplete", f"{name}: {type(error).__name__}: {error}")
        try:
            self.time_totals()
        except (ValueError, KeyError, TypeError) as error:
            self.check(False, "timing_summary", str(error))
        if self.counts["recorded_pruned_checkpoints"]:
            self.warn("recorded_checkpoint_pruning", f"{self.counts['recorded_pruned_checkpoints']} checkpoint binaries were deliberately pruned; their hashes/sizes remain recorded.")
        if self.report["status"] == "complete":
            self.check(self.report.get("completed_training_runs") == self.counts["completed_training_runs"],
                       "reported_run_count", "final report completed-run count differs from ledger")
            self.check(self.report.get("architectures_with_completed_runs") == self.counts["architectures_with_completed_runs"],
                       "reported_architecture_count", "final report trained-architecture count differs from ledger")
        self.check(digest(self.root / "FINAL_REPORT.json") == self.initial_report_sha256,
                   "report_changed_during_audit", "final report changed while audit was reading artifacts")
        if hasattr(self, "freeze_hash"):
            self.check(digest(self.root / "selection.json") == self.freeze_hash,
                       "selection_changed_during_audit", "selection changed during audit")
        return {"status": "pass" if not self.failures else "fail", "audited_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(self.root), "project_root": str(self.project), "audit_source_sha256": digest(Path(__file__)),
                "final_report_sha256": digest(self.root / "FINAL_REPORT.json"),
                "selection_sha256": digest(self.root / "selection.json") if (self.root / "selection.json").exists() else None,
                "counts": dict(self.counts), "recomputed": self.recomputed,
                "failures": self.failures, "warnings": self.warnings,
                "scope": "Artifact integrity and saved-metric recomputation only; no model scoring, training, selection changes, or experimental artifact modifications.",
                "limits": ["Logs/hashes establish internal consistency, not an externally trusted timestamp or proof against coordinated evidence rewriting.",
                           "Matching supervised token totals and fixed source/data/seed controls supports paired input streams; no per-batch token transcript was saved."]}


def markdown(result):
    lines = [f"# Final artifact audit: {result['status'].upper()}", "", result["scope"], "",
             f"Audited: {result['audited_utc']}", "", "| Count | Value |", "|---|---:|"]
    lines.extend(f"| {key.replace('_', ' ')} | {value:,} |" for key, value in sorted(result["counts"].items()))
    lines += ["", "## Recomputed results", ""]
    for key, value in result["recomputed"].items():
        if key not in ("qualified_candidates", "observed_timing"):
            lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    timing = result["recomputed"].get("observed_timing", {})
    if timing:
        lines += ["", f"Summed training-attempt GPU wall time: **{timing['training_attempt_gpu_wall_hours']:.3f} hours**.",
                  f"Completed training compute: **{timing['completed_training_compute_gpu_hours']:.3f} GPU-hours**.", timing["notes"]]
    for heading, key in (("Failures", "failures"), ("Warnings", "warnings")):
        lines += ["", "## " + heading, ""]
        lines += [f"- **{row['code']}**: {row['message']}" + (f" (`{row['path']}`)" if row.get("path") else "") for row in result[key]] or ["None."]
    lines += ["", "## Limits", ""] + ["- " + value for value in result["limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    args = parser.parse_args()
    try:
        audit = Audit(args.root, args.project_root)
        result = audit.execute()
        folder = audit.root / "final-audit"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = folder / ("audit-" + stamp)
        with target.with_suffix(".json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with target.with_suffix(".md").open("x") as stream:
            stream.write(markdown(result))
        print(json.dumps({"status": result["status"], "failures": len(result["failures"]),
                          "warnings": len(result["warnings"]), "json": str(target.with_suffix('.json')),
                          "markdown": str(target.with_suffix('.md'))}))
        return 0 if result["status"] == "pass" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Audit refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
