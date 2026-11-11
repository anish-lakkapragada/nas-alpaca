"""Protocol tests run without GPU/PyTorch; no model-training or held-out access."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cuda_minparams_finalize as final
from cuda_search import baseline_spec


def identity(frame, spec):
    clean = {k: v for k, v in spec.items() if k not in ("name", "description")}
    return hashlib.sha256(json.dumps([frame, clean], sort_keys=True).encode()).hexdigest()


def parameter_count(frame, spec):
    return final.BASELINE_PARAMETERS if frame["d_model"] == 384 else 21_000_000


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "search"
        self.project = Path(self.temporary.name).resolve() / "project"
        self.root.mkdir()
        (self.root / "specs").mkdir()
        (self.root / "attempts").mkdir()
        (self.root / "logs").mkdir()
        for name in final.TRAIN_PROVENANCE:
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.provenance = {name: final.file_hash(self.project / name) for name in final.TRAIN_PROVENANCE}
        self.baseline = self.architecture("baseline", 384, 3.4)
        self.primary = self.architecture("small", 320, 3.39)
        self.frozen = {"status": "qualified", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                       "validation_used": False, "selection_seeds": [0, 1, 2],
                       "fresh_seeds": [100, 101, 102, 103, 104],
                       "baseline": self.baseline, "primary": self.primary,
                       "primary_architecture_id": self.primary["architecture_id"],
                       "qualified_candidates": [self.primary], "rule": "minimum parameters, mean dev <= baseline"}
        final.atomic_json(self.root / "selection.json", self.frozen)
        self.tools = patch.object(final, "spec_tools", return_value=(identity, parameter_count))
        self.tools.start()

    def tearDown(self):
        self.tools.stop()
        self.temporary.cleanup()

    def architecture(self, name, width, loss):
        frame = dict(final.BASELINE_FRAME, d_model=width)
        if width != 384:
            frame.update(n_heads=5, n_layers=12, d_ff=1024)
        spec = baseline_spec()
        spec["name"] = name
        digest = identity(frame, spec)
        spec_path = self.root / "specs" / (digest + ".json")
        final.atomic_json(spec_path, spec)
        architecture = {"architecture_id": digest, "name": name, "frame": frame, "spec": spec,
                        "spec_path": str(spec_path), "parameters": parameter_count(frame, spec),
                        "mean_dev_loss": loss, "dev_losses": [loss] * 3, "run_paths": {}}
        for seed in final.SEARCH_SEEDS:
            path = self.make_run(architecture, seed, loss)
            architecture["run_paths"][str(seed)] = str(path)
            final.append_json(self.root / "run_registry.jsonl", {
                "status": "completed", "path": str(path),
                "job": {"run_id": path.name, "seed": seed, "architecture_id": digest, "phase": "search"}})
        return architecture

    def make_run(self, architecture, seed, loss, name=None):
        path = self.root / "attempts" / (name or f"{architecture['name']}-s{seed}")
        path.mkdir()
        diagnostic = {"conversations": 256, "dropout": False, "indices_sha256": "fixed", "answer_tokens": 16000}
        config = {"model": architecture["frame"], "spec": architecture["spec"], "parameters": architecture["parameters"],
                  "args": {**final.FIXED_ARGS, "seed": seed, "resume": None, "benchmark_steps": 0},
                  "sha256": self.provenance, "train_evaluation": diagnostic,
                  "optimizer": {"name": "AdamWNoBiasCorrection", "bias_correction": False, "betas": [.9, .95], "eps": 1e-8}}
        summary = {"steps": 5000, "schedule_steps": 5000, "parameters": architecture["parameters"],
                   "best_dev_loss": loss, "best_step": 4750, "answer_tokens_seen": 27_000_000 + seed,
                   "train_eval_loss_at_best_dev": 2.5, "final_train_eval_loss": 2.4,
                   "final_dev_loss": loss + .01, "train_seconds": 80}
        final.atomic_json(path / "config.json", config)
        final.atomic_json(path / "summary.json", summary)
        (path / "best.pt").write_bytes(b"test-checkpoint")
        return path

    def change_config(self, architecture, seed, change):
        path = Path(architecture["run_paths"][str(seed)]) / "config.json"
        config = final.read_json(path)
        change(config)
        final.atomic_json(path, config)

    def test_different_candidate_width_and_depth_are_allowed(self):
        checked = final.validate_freeze(self.root, self.project)
        self.assertTrue(checked["qualified"])
        self.assertEqual(checked["freeze"]["primary"]["frame"]["d_model"], 320)

    def test_training_step_or_warmup_changes_are_rejected(self):
        for key, bad in (("steps", 10000), ("warmup", 200), ("batch_rows", 16), ("train_eval_seed", 5)):
            with self.subTest(key=key):
                self.change_config(self.primary, 0, lambda c: c["args"].update({key: bad}))
                with self.assertRaisesRegex(ValueError, "fixed training recipe"):
                    final.validate_freeze(self.root, self.project)
                self.change_config(self.primary, 0, lambda c: c["args"].update({key: final.FIXED_ARGS[key]}))

    def test_changed_data_hash_is_rejected(self):
        self.change_config(self.primary, 1, lambda c: c["sha256"].update({"data/train.json": "other"}))
        with self.assertRaisesRegex(ValueError, "different training source"):
            final.validate_freeze(self.root, self.project)

    def test_unpaired_tokens_are_rejected(self):
        path = Path(self.primary["run_paths"]["2"]) / "summary.json"
        summary = final.read_json(path)
        summary["answer_tokens_seen"] += 1
        final.atomic_json(path, summary)
        with self.assertRaisesRegex(ValueError, "answer_tokens_seen"):
            final.validate_freeze(self.root, self.project)

    def test_no_reselection_of_nonminimal_qualifier(self):
        self.frozen["qualified_candidates"].append({"parameters": 19_000_000, "mean_dev_loss": 3.4})
        final.atomic_json(self.root / "selection.json", self.frozen)
        with self.assertRaisesRegex(ValueError, "smallest qualifying"):
            final.validate_freeze(self.root, self.project)

    def test_above_baseline_search_loss_is_rejected(self):
        self.frozen["primary"]["mean_dev_loss"] = 3.5
        final.atomic_json(self.root / "selection.json", self.frozen)
        with self.assertRaises(ValueError):
            final.validate_freeze(self.root, self.project)

    def test_fresh_seed_mapping_is_exactly_five_pairs(self):
        verified = final.validate_freeze(self.root, self.project)
        driver = final.Confirmation(self.root, verified, [0, 1, 2, 3], datetime.now(timezone.utc) + timedelta(hours=1), self.project)
        jobs = driver.jobs()
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({j["job"]["run_id"] for j in jobs}), 10)
        for role in ("baseline", "winner"):
            self.assertEqual([j["job"]["seed"] for j in jobs if j["role"] == role], final.FRESH_SEEDS)
        self.assertTrue(all(j["job"]["steps"] == 5000 for j in jobs))

    def test_training_command_fixed_recipe_and_dynamic_architecture(self):
        command = final.training_command(self.project, self.root / "new", self.primary, 103)
        for key, value in (("--steps", "5000"), ("--warmup", "100"), ("--d-model", "320"),
                           ("--layers", "12"), ("--heads", "5"), ("--max-params", "27144959"),
                           ("--train-eval-conversations", "256"), ("--seed", "103")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--compile", command)

    def test_failed_attempt_is_logged_and_retry_gets_new_path(self):
        verified = final.validate_freeze(self.root, self.project)
        driver = final.Confirmation(self.root, verified, [0], datetime.now(timezone.utc) + timedelta(hours=1), self.project)
        old = driver.jobs()[0]
        old_id = old["job"]["run_id"]
        driver.record(old, status="interrupted")
        retry = driver.jobs()[0]
        self.assertIsNone(driver.previous_attempt(retry))
        self.assertNotEqual(retry["job"]["run_id"], old_id)
        self.assertEqual(retry["retry_of"], [old_id])
        self.assertEqual(driver.records[old_id]["status"], "interrupted")

    def test_completed_attempt_is_reused_without_retraining(self):
        verified = final.validate_freeze(self.root, self.project)
        driver = final.Confirmation(self.root, verified, [0], datetime.now(timezone.utc) + timedelta(hours=1), self.project)
        old = driver.jobs()[0]
        driver.record(old, status="completed")
        self.assertEqual(driver.previous_attempt(driver.jobs()[0])["job"]["run_id"], old["job"]["run_id"])

    def test_no_qualifier_does_not_access_test_data(self):
        self.frozen.update(status="no_qualified_candidate", primary=None)
        final.atomic_json(self.root / "selection.json", self.frozen)
        # There is deliberately no data/validation.json in this fake project.
        report = final.finalize(self.root, [0], datetime.now(timezone.utc) + timedelta(hours=1), self.project)
        self.assertEqual(report["status"], "no_qualified_candidate")
        self.assertFalse(report["test_accessed"])
        self.assertFalse(report["public_claim_supported_by_mean"])

    def test_test_failure_remains_reportable_but_never_supports_claim(self):
        verified = final.validate_freeze(self.root, self.project)
        runs, results = [], {}
        for role, architecture, loss in (("baseline", self.baseline, 3.3), ("winner", self.primary, 3.31)):
            for seed in final.FRESH_SEEDS:
                path = self.make_run(architecture, seed, architecture["mean_dev_loss"])
                row = final.validate_training_run(path, architecture, seed, self.provenance)
                row["role"] = role
                runs.append(row)
                results[str(path)] = {"loss": loss}
        results["difference_candidate_minus_baseline_across_seeds"] = {"point": .01, "bootstrap_95": [.002, .02]}
        report = final.summarize(self.root, verified, runs, {"results": results}, self.provenance)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["test_mean_no_worse"])
        self.assertFalse(report["public_claim_supported_by_mean"])
        self.assertEqual(report["winner_architecture_id"], self.primary["architecture_id"])
        self.assertIn("did not match", final.human_report(report))


if __name__ == "__main__":
    unittest.main()
