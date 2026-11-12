"""Synthetic inventory integration tests; standard library, no GPU or datasets."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nas.cuda_inventory import build_inventory, write_inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nas-inventory-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.cuda = self.root / "cuda"

    def put(self, relative, value, lines=False):
        path = self.cuda / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))
        return path

    @staticmethod
    def config(width=384, layers=8, steps=5000, **hardware):
        return {"model": {"d_model": width, "n_layers": layers, "d_ff": width * 4},
                "spec": {"name": "baseline", "layers": [{"attn": "A0", "ffn": "F0"}], "ffns": {}},
                "args": {"seed": 0, "steps": steps},
                "hardware": {"device": "cuda:0", "gpu": "H100", **hardware}}

    def test_mixed_inventory_and_source_preservation(self):
        old = "old50m/attempts/baseline-k5000-s0"
        self.put(old + "/config.json", self.config(512, 10))
        self.put(old + "/summary.json", {"steps": 5000, "parameters": 48780288, "best_dev_loss": 3.4})
        self.put(old + "/log.jsonl", [{"step": 4750, "dev_loss": 3.4}, {"step": 5000, "dev_loss": 3.45},
                                      {"step": 5000, "train_loss": 2.5}], lines=True)
        self.put("old50m/status.json", {"attempts": {"baseline-k5000-s0": {
            "job": {"name": "baseline-k5000-s0", "seed": 0, "steps": 5000, "phase": "baselines"},
            "gpu": 2, "status": "completed", "wall_seconds": 123.}}})
        self.put("manual27m/baseline-original-s3/config.json", self.config())
        self.put("manual27m/baseline-original-s3/summary.json", {"steps": 5000, "parameters": 27144960, "best_dev_loss": 3.3})
        self.put("manual27m/protocol.json", {"runs": [{"name": "baseline-original-s3", "gpu": 3,
                  "out": "/home/ubuntu/nas-cuda/runs/cuda/manual27m/baseline-original-s3"}]})
        self.put("manual27m/measurements.json", {"runs": [{"name": "baseline-original-s3", "wall_seconds": 98.1}]})
        self.put("engineering-smoke/config.json", self.config(16, 2, 1, hostname="saved-host", physical_gpu_ids="0"))
        benchmark = self.config()
        benchmark["args"]["benchmark_steps"] = 10
        self.put("throughput/config.json", benchmark)
        self.put("throughput/summary.json", {"benchmark": True, "steps": 20, "best_dev_loss": 10.4})
        self.put("newsearch/provenance.json", {"host": {"hostname": "gpu-host", "public_ip": "192.0.2.10"}})
        self.put("newsearch/plan.json", {"protocol": {"frame": {"n_layers": 8, "d_model": 384, "d_ff": 1536}}})
        spec = self.put("newsearch/specs/abc.json", {"name": "scaled", "layers": [{"attn": "A0", "ffn": "F0"}]})
        self.put("newsearch/run_registry.jsonl", [
            {"job": {"run_id": "rfailed", "seed": 4, "steps": 5000, "phase": "breadth"},
             "status": "running", "physical_gpu": {"physical_index": 1, "uuid": "GPU-demo"},
             "started_utc": "2026-11-11T09:00:00Z", "spec_path": str(spec), "architecture_sha256": "abc"},
            {"job": {"run_id": "rfailed", "seed": 4, "steps": 5000, "phase": "breadth"},
             "status": "failed", "returncode": 7, "failure_reason": "OOM", "ended_utc": "2026-11-11T09:00:01Z"},
            {"job": {"run_id": "rqueued", "seed": 0, "steps": 5000}, "status": "not_admitted"},
            {"status": "historical_reference", "path": "/old/runs/cuda/manual27m/baseline-original-s3",
             "summary": {"steps": 5000, "best_dev_loss": 3.3}},
        ], lines=True)
        broken = self.put("broken-config/config.json", {})
        broken.write_text("{")
        original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.cuda.rglob("*") if path.is_file()}
        result = build_inventory(self.root, "192.0.2.99")
        rows = {row["relative_path"]: row for row in result["runs"]}
        self.assertEqual(result["run_count"], 7)
        historical = rows["cuda/" + old]
        self.assertEqual((historical["scope"], historical["physical_gpu_ids"], historical["final_dev_loss"]),
                         ("historical_50m", [2], 3.45))
        manual = rows["cuda/manual27m/baseline-original-s3"]
        self.assertEqual((manual["physical_gpu_ids"], manual["wall_seconds"], manual["status"]), ([3], 98.1, "completed"))
        self.assertEqual(rows["cuda/engineering-smoke"]["scope"], "engineering_smoke")
        self.assertEqual(rows["cuda/engineering-smoke"]["hostname"], "saved-host")
        self.assertEqual(rows["cuda/throughput"]["scope"], "benchmark")
        self.assertIsNone(rows["cuda/throughput"]["physical_gpu_ids"])
        failed = rows["cuda/newsearch/attempts/rfailed"]
        self.assertEqual((failed["status"], failed["returncode"], failed["physical_gpu_ids"], failed["wall_seconds"]),
                         ("failed", 7, [1], 1.))
        self.assertEqual((failed["architecture_spec"]["name"], failed["public_ip"]), ("scaled", "192.0.2.10"))
        self.assertEqual(rows["cuda/newsearch/attempts/rqueued"]["status"], "not_admitted")
        self.assertTrue(rows["cuda/broken-config"]["metadata_errors"])
        outputs = write_inventory(result, self.cuda)
        with outputs[1].open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 7)
        self.assertEqual(build_inventory(self.root, "192.0.2.99")["runs"], result["runs"])
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original.items()))

    def test_identical_names_in_different_schedulers_do_not_alias(self):
        self.put("first/attempts/baseline/config.json", self.config())
        self.put("second/run_registry.jsonl", [{"job": {"name": "baseline", "seed": 1, "steps": 5000},
                                                "status": "failed", "returncode": 2}], lines=True)
        rows = {row["relative_path"]: row for row in build_inventory(self.root)["runs"]}
        self.assertEqual(set(rows), {"cuda/first/attempts/baseline", "cuda/second/attempts/baseline"})
        self.assertEqual(rows["cuda/first/attempts/baseline"]["status"], "incomplete")
        self.assertEqual(rows["cuda/second/attempts/baseline"]["status"], "failed")

    def test_empty_inventory_and_dataset_rejection(self):
        self.assertEqual(build_inventory(self.root)["run_count"], 0)
        with self.assertRaises(ValueError):
            build_inventory(self.root, metadata_paths=[self.root / "data" / "validation.json"])

    def test_launch_role_excludes_original_frame_smoke_from_ranking(self):
        name = "validation-27m-new-primitives"
        self.put(name + "/config.json", self.config(steps=2))
        self.put(name + "/summary.json", {"steps": 2, "parameters": 27144962, "best_dev_loss": 10.2})
        launch = self.put(name + ".launch.json", {
            "command": ["python", "cuda_train.py", "--out", "/home/ubuntu/nas-cuda/runs/cuda/" + name,
                        "--steps", "2"], "gpu": 3, "started_unix": 1794386830.744748,
            "run_role": "engineering_smoke", "eligible_for_ranking": False})
        self.put("another-smoke/.launch.json", {"run_role": "engineering_smoke", "gpu": 1})
        result = build_inventory(self.root)
        rows = {row["relative_path"]: row for row in result["runs"]}
        row = rows["cuda/" + name]
        self.assertEqual((row["scope"], row["run_role"], row["eligible_for_ranking"]),
                         ("engineering_smoke", "engineering_smoke", False))
        self.assertEqual((row["steps_requested"], row["steps_completed"], row["physical_gpu_ids"]), (2, 2, [3]))
        self.assertEqual(row["field_sources"]["scope"], str(launch.resolve()))
        self.assertEqual(row["field_sources"]["eligible_for_ranking"], str(launch.resolve()))
        self.assertEqual(row["started_utc"], "2026-11-11T08:47:10.744748+00:00")
        self.assertFalse(rows["cuda/another-smoke"]["eligible_for_ranking"])
        self.assertFalse(rows["cuda/another-smoke"]["config_present"])
        outputs = write_inventory(result, self.cuda)
        with outputs[1].open() as stream:
            saved = {row["relative_path"]: row for row in csv.DictReader(stream)}
        self.assertEqual(saved["cuda/" + name]["eligible_for_ranking"], "False")


if __name__ == "__main__":
    unittest.main()
