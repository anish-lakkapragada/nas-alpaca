"""Audit-export tests use synthetic ledgers only; no experiment data access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cuda_minparams_report as report


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)
    (root / "specs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    specs = [("base", "baseline", 27144960), ("good", "small-qualified", 18000000),
             ("bad", "too-small-high-loss", 12000000), ("pending", "lucky-one-seed", 9000000)]
    for digest, name, parameters in specs:
        (root / "specs" / (digest + ".json")).write_text(json.dumps({"name": name, "layers": [{"attn": "A0", "ffn": "F0"}]}))
        append(root / "architecture_registry.jsonl", {"architecture_id": digest, "name": name, "parameters": parameters,
               "status": "accepted", "created_utc": "2026-11-11T23:00:00+00:00", "generation": 0,
               "operator": "breadth", "parents": [], "frame": {"n_layers": 8, "d_model": 384, "n_heads": 6, "d_ff": 1536},
               "spec_path": f"/home/ubuntu/remote-search/specs/{digest}.json"})
    append(root / "architecture_registry.jsonl", {"architecture_id": "rejected", "name": "oversized", "status": "rejected",
           "created_utc": "2026-11-11T23:00:01+00:00", "frame": {"d_model": 768}, "spec": {"layers": []},
           "parents": ["base"], "operator": "widen", "generation": 1,
           "reason": "total 48000000 must be strictly below 27144960"})
    ordinal = 0
    for digest, name, parameters in specs:
        seeds = [0] if digest == "pending" else [0, 1, 2]
        for seed in seeds:
            ordinal += 1
            run_id = f"r{ordinal:05d}-{digest}-s{seed}"
            folder = root / "attempts" / run_id
            folder.mkdir()
            loss = {"base": 3.42, "good": 3.40, "bad": 3.55, "pending": 3.20}[digest] + seed * .005
            summary = {"steps": 5000, "schedule_steps": 5000, "best_dev_loss": loss,
                       "best_step": 4750, "train_eval_loss_at_best_dev": 2.4, "train_seconds": 87,
                       "parameters": parameters}
            (folder / "summary.json").write_text(json.dumps(summary))
            (folder / "log.jsonl").write_text(json.dumps({"event": "evaluation", "step": 4750, "dev_loss": loss, "train_eval_loss": 2.4}) + "\n")
            (root / "logs" / (run_id + ".log")).write_text("synthetic log\n")
            row = {"job": {"run_id": run_id, "architecture_id": digest, "spec": name, "seed": seed, "phase": "breadth"},
                   "parameters": parameters, "path": f"/home/ubuntu/remote-search/attempts/{run_id}",
                   "spec_path": f"/home/ubuntu/remote-search/specs/{digest}.json", "frame": {"d_model": 384},
                   "host": {"hostname": "synthetic-host", "public_ip": "192.0.2.1"}, "physical_gpu": {"physical_index": seed % 4},
                   "queued_utc": "2026-11-11T23:01:00+00:00", "started_utc": "2026-11-11T23:01:01+00:00",
                   "ended_utc": "2026-11-11T23:02:41+00:00", "wall_seconds": 100, "summary": summary}
            for status in ("queued", "running", "completed"):
                append(root / "run_registry.jsonl", {**row, "status": status})
    for offset, status in enumerate(("failed", "interrupted", "not_admitted", "queued", "running")):
        append(root / "run_registry.jsonl", {"job": {"run_id": f"edge-{status}", "architecture_id": "pending", "spec": "lucky-one-seed", "seed": 1,
               "phase": "confirmation"}, "status": status, "parameters": 9000000,
               "path": f"/home/ubuntu/remote-search/attempts/edge-{status}", "failure_reason": "synthetic | reason"})
    # Fresh confirmation must remain visible in the log but never affect search rankings.
    append(root / "run_registry.jsonl", {"job": {"run_id": "fresh", "architecture_id": "good", "spec": "small-qualified", "seed": 100,
           "phase": "final"}, "status": "completed", "parameters": 18000000, "path": "/home/ubuntu/remote-search/attempts/fresh",
           "summary": {"steps": 5000, "schedule_steps": 5000, "best_dev_loss": 9.0, "train_eval_loss_at_best_dev": 8.0}})
    (root / "rankings.json").write_text(json.dumps({"all": [{"architecture_id": "good", "seeds": [0], "mean_dev_loss": 999}]}))
    append(root / "events.jsonl", {"event": "heartbeat", "phase": "breadth", "completed": 10})
    return root


class ExportTests(unittest.TestCase):
    def test_full_audit_export_is_read_only_and_uses_live_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(Path(directory) / "run")
            original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            summary = report.export(root, plots=False)
            self.assertEqual(summary["training_attempts"], 16)
            self.assertEqual(summary["architecture_registry_entries"], 5)
            self.assertEqual(summary["qualified_candidates"], 1)
            self.assertEqual(summary["completed_fresh_confirmation_runs"], 1)
            self.assertEqual(summary["smallest_qualifying_candidates"][0]["architecture_id"], "good")
            self.assertAlmostEqual(summary["smallest_qualifying_candidates"][0]["mean_dev_ce"], 3.405)
            self.assertFalse(summary["test_data_or_metrics_accessed"])
            for path, digest in original.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            markdown = (root / "audit-report/RUN_LOG.md").read_text()
            for status in ("failed", "interrupted", "not_admitted", "queued", "running"):
                self.assertIn("edge-" + status, markdown)
            self.assertIn("queued → running → completed", markdown)
            self.assertIn("../specs/good.json", markdown)
            self.assertIn("../attempts/r00004-good-s0", markdown)
            self.assertNotIn("](file:", markdown)
            architectures = (root / "audit-report/ARCHITECTURES.md").read_text()
            self.assertIn("48,000,000", architectures)
            self.assertIn("Rejected specification", architectures)
            self.assertIn("confirmation incomplete", architectures)

    def test_live_partial_line_is_reported_without_repairing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(Path(directory) / "run")
            path = root / "run_registry.jsonl"
            with path.open("ab") as stream:
                stream.write(b'{"partial":')
            before = path.read_bytes()
            summary = report.export(root, plots=False)
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(any("possibly still being written" in warning for warning in summary["warnings"]))

    def test_duplicate_completed_seed_is_not_counted_as_replication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(Path(directory) / "run")
            append(root / "run_registry.jsonl", {"job": {"run_id": "duplicate", "architecture_id": "pending", "spec": "lucky-one-seed", "seed": 0,
                   "phase": "breadth"}, "status": "completed", "parameters": 9000000,
                   "summary": {"steps": 5000, "schedule_steps": 5000, "best_dev_loss": 3.0}})
            summary = report.export(root, plots=False)
            self.assertEqual(summary["qualified_candidates"], 1)
            self.assertTrue(any("Duplicate completed architecture/seed" in warning for warning in summary["warnings"]))

    def test_external_artifact_is_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(Path(directory) / "run")
            snapshot = report.Snapshot(root)
            self.assertIsNone(snapshot.artifact("/etc/passwd"))
            self.assertIsNone(snapshot.bytes(Path(directory) / "outside.json"))


if __name__ == "__main__":
    unittest.main()
