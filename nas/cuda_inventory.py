"""Collate every recorded CUDA attempt without modifying source run artifacts.

Only JSON metadata and training log JSONL are read; checkpoints and datasets
(including validation) are never opened. Unknown location/timing/GPU fields
remain null. A logical CUDA device such as cuda:0 is never treated as a physical
GPU index. Existing per-run and append-only scheduler logs remain authoritative.

  python cuda_inventory.py --runs-root /home/ubuntu/nas-cuda/runs \
      --host-public-ip 192.0.2.1
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = PROJECT_ROOT
METADATA_NAMES = {"status.json", "results.json", "protocol.json", "provenance.json", "plan.json",
                  "measurements.json", "measurement.json", "wall-times.json", "wall_times.json",
                  "timings.json", "timing.json"}
CSV_FIELDS = ["relative_path", "run_id", "scope", "run_role", "eligible_for_ranking", "phase", "status", "failure_reason", "returncode",
              "architecture_name", "architecture_sha256", "architecture_hash_kind", "model_config", "architecture_spec",
              "seed", "steps_requested", "steps_completed", "parameters", "structural_parameters", "learned_scale_parameters",
              "best_dev_loss", "best_dev_step", "final_dev_loss", "last_dev_loss", "last_train_loss", "last_train_eval_loss",
              "train_eval_loss_at_best_dev", "final_train_eval_loss", "generalization_gap_at_best_dev", "answer_tokens_seen",
              "hostname", "public_ip", "physical_gpu_ids", "gpu_uuid", "gpu_name", "logical_device",
              "started_utc", "ended_utc", "wall_seconds", "train_seconds", "last_observed_utc", "config_present",
              "summary_present", "metrics_available", "evidence_files", "metadata_errors", "field_sources"]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, errors):
    try:
        return json.loads(path.read_text(), parse_constant=lambda _: None)
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return None


def iter_jsonl(path, errors):
    try:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=lambda _: None)
                    if isinstance(value, dict):
                        yield value
                except ValueError as exc:
                    errors.append(f"{path}:{number}: {exc}")
    except OSError as exc:
        errors.append(f"{path}: {exc}")


def physical_ids(value):
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return physical_ids(value.get("physical_index", value.get("physical_gpu_ids")))
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(physical_ids(item) or [])
        return result or None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return [value]
    if isinstance(value, str):
        result = []
        for item in value.split(","):
            item = item.strip()
            if item.isdecimal():
                result.append(int(item))
            elif item.startswith(("GPU-", "MIG-")):
                result.append(item)
        return result or None
    return None


class Inventory:
    def __init__(self, runs_root, host_public_ip=None):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.cuda_root = self.runs_root / "cuda"
        self.host_public_ip = host_public_ip
        self.records, self.priorities, self.errors, self.documents = {}, {}, [], {}

    def local_path(self, value, base):
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if "/runs/cuda/" in value:
            path = self.cuda_root / value.split("/runs/cuda/", 1)[1]
        elif value.startswith("cuda/"):
            path = self.runs_root / path
        elif not path.is_absolute():
            path = base / path
        try:
            path = path.resolve()
            path.relative_to(self.cuda_root)
        except ValueError:
            return None
        return path

    def get(self, folder):
        key = str(folder.relative_to(self.runs_root))
        if key not in self.records:
            record = {field: None for field in CSV_FIELDS}
            record.update(relative_path=key, run_id=folder.name, config_present=False, summary_present=False,
                          evidence_files=[], metadata_errors=[], field_sources={})
            self.records[key], self.priorities[key] = record, {}
        return key

    def assign(self, key, field, value, source, priority=50, allow_none=False):
        if value is None and not allow_none:
            return
        if priority >= self.priorities[key].get(field, -1):
            self.records[key][field] = value
            self.priorities[key][field] = priority
            self.records[key]["field_sources"][field] = str(source)

    def evidence(self, key, source):
        text = str(source)
        if text not in self.records[key]["evidence_files"]:
            self.records[key]["evidence_files"].append(text)

    def identify(self, value, base, create=False, fallback_name=None):
        job = value.get("job") or {}
        config = value.get("config") or {}
        for candidate in (value.get("path"), value.get("out"), config.get("args", {}).get("out")):
            folder = self.local_path(candidate, base)
            if folder:
                key = str(folder.relative_to(self.runs_root))
                if key in self.records or create:
                    return self.get(folder)
        name = value.get("run_id") or job.get("run_id") or job.get("name") or value.get("name") or fallback_name
        if not isinstance(name, str) or not name:
            return None
        matches = [key for key, record in self.records.items() if record["run_id"] == name or Path(key).name == name]
        for folder in (base / "attempts" / name, base / name):
            local = self.local_path(str(folder), base)
            if local and str(local.relative_to(self.runs_root)) in self.records:
                return self.get(local)
        if create and (job or value.get("run_id") or fallback_name):
            folder = self.local_path(str(base / "attempts" / name), base)
            if folder:
                return self.get(folder)
        if len(matches) == 1:
            return matches[0]
        if create:
            folder = self.local_path(str(base / "attempts" / name), base)
            if folder:
                return self.get(folder)
        return None

    def merge_hardware(self, key, hardware, source, priority):
        if not isinstance(hardware, dict):
            return
        for field in ("hostname", "public_ip"):
            self.assign(key, field, hardware.get(field), source, priority)
        self.assign(key, "physical_gpu_ids", physical_ids(hardware.get("physical_gpu_ids")), source, priority)
        self.assign(key, "gpu_name", hardware.get("gpu"), source, priority)
        self.assign(key, "logical_device", hardware.get("device"), source, priority)

    def merge_config(self, key, config, source):
        if not isinstance(config, dict):
            return
        args = config.get("args") or {}
        model = config.get("model") or config.get("model_config")
        spec = config.get("spec")
        self.assign(key, "model_config", model, source, 100)
        self.assign(key, "architecture_spec", spec, source, 100)
        if isinstance(spec, dict):
            self.assign(key, "architecture_name", spec.get("name"), source, 100)
        for field in ("parameters", "structural_parameters", "learned_scale_parameters"):
            self.assign(key, field, config.get(field), source, 90)
        self.assign(key, "seed", args.get("seed"), source, 100)
        self.assign(key, "steps_requested", args.get("steps"), source, 100)
        self.merge_hardware(key, config.get("hardware"), source, 90)
        self.records[key]["_args"] = args
        self.evidence(key, source)

    def merge_summary(self, key, summary, source, priority=85):
        if not isinstance(summary, dict):
            return
        for field in ("parameters", "structural_parameters", "learned_scale_parameters", "best_dev_loss", "final_dev_loss",
                      "train_eval_loss_at_best_dev", "final_train_eval_loss", "generalization_gap_at_best_dev",
                      "answer_tokens_seen", "train_seconds"):
            self.assign(key, field, summary.get(field), source, priority)
        self.assign(key, "steps_completed", summary.get("steps"), source, priority)
        self.assign(key, "steps_requested", summary.get("schedule_steps"), source, priority)
        self.assign(key, "best_dev_step", summary.get("best_step", summary.get("best_dev_step")), source, priority)
        self.merge_hardware(key, summary.get("hardware"), source, priority - 5)
        if summary.get("benchmark"):
            self.records[key]["_benchmark"] = True
        self.assign(key, "status", "completed", source, priority)
        self.evidence(key, source)

    def merge_record(self, key, value, source, priority=95):
        self.evidence(key, source)
        job = value.get("job") or {}
        self.assign(key, "run_role", value.get("run_role", job.get("run_role")), source, priority)
        eligible = value.get("eligible_for_ranking", job.get("eligible_for_ranking"))
        if isinstance(eligible, bool):
            self.assign(key, "eligible_for_ranking", eligible, source, priority)
        for field in ("status", "returncode", "parameters", "started_utc", "ended_utc", "wall_seconds"):
            if field == "status" and value.get(field) == "historical_reference":
                # This is a ledger role, not the execution status of that run.
                continue
            self.assign(key, field, value.get(field), source, priority)
        if "failure_reason" in value or "error" in value or "stop_reason" in value:
            self.assign(key, "failure_reason", value.get("failure_reason", value.get("error", value.get("stop_reason"))),
                        source, priority, allow_none=True)
        self.assign(key, "last_observed_utc", value.get("updated_utc", value.get("time_utc")), source, priority)
        self.assign(key, "run_id", job.get("run_id", value.get("run_id", job.get("name"))), source, priority)
        self.assign(key, "phase", job.get("phase", value.get("phase")), source, priority)
        self.assign(key, "seed", job.get("seed", value.get("seed")), source, priority)
        self.assign(key, "steps_requested", job.get("steps", value.get("steps")), source, priority)
        if isinstance(job.get("spec"), str):
            self.assign(key, "architecture_name", job["spec"], source, 70)
        architecture_id = value.get("architecture_sha256", job.get("architecture_id"))
        if architecture_id:
            self.assign(key, "architecture_sha256", architecture_id, source, priority)
            self.assign(key, "architecture_hash_kind", "scheduler_canonical", source, priority)
        gpu = value.get("physical_gpu")
        self.assign(key, "physical_gpu_ids", physical_ids(gpu if gpu is not None else value.get("gpu")), source, priority)
        if isinstance(gpu, dict):
            self.assign(key, "gpu_uuid", gpu.get("uuid"), source, priority)
            self.assign(key, "gpu_name", gpu.get("name"), source, priority)
        host = value.get("host") or {}
        for field in ("hostname", "public_ip"):
            self.assign(key, field, host.get(field, value.get(field)), source, priority)
        self.merge_config(key, value.get("config"), source)
        self.merge_summary(key, value.get("summary"), source, priority - 10)
        metrics = value.get("metrics") or value
        for field in ("best_dev_loss", "best_dev_step", "final_dev_loss", "train_eval_loss_at_best_dev",
                      "final_train_eval_loss", "generalization_gap_at_best_dev"):
            self.assign(key, field, metrics.get(field), source, priority)
        spec_path = self.local_path(value.get("spec_path"), source.parent)
        if spec_path and spec_path.is_file() and self.records[key]["architecture_spec"] is None:
            spec = read_json(spec_path, self.errors)
            if isinstance(spec, dict):
                self.assign(key, "architecture_spec", spec, spec_path, 80)
                self.assign(key, "architecture_name", spec.get("name"), spec_path, 80)
                self.evidence(key, spec_path)

    def scan_configs(self):
        for path in sorted(self.cuda_root.rglob("config.json")):
            key = self.get(path.parent)
            self.records[key]["config_present"] = True
            self.evidence(key, path)
            errors = []
            config = read_json(path, errors)
            self.records[key]["metadata_errors"].extend(errors)
            self.merge_config(key, config, path)
            summary_path = path.parent / "summary.json"
            if summary_path.is_file():
                self.records[key]["summary_present"] = True
                self.merge_summary(key, read_json(summary_path, self.records[key]["metadata_errors"]), summary_path)
            self.scan_log(key, path.parent / "log.jsonl")

    def scan_log(self, key, path):
        if not path.is_file():
            return
        rows = list(iter_jsonl(path, self.records[key]["metadata_errors"]))
        self.evidence(key, path)
        training = [row for row in rows if row.get("train_loss") is not None]
        evaluations = [row for row in rows if isinstance(row.get("dev_loss"), (int, float))]
        steps = [row["step"] for row in rows if isinstance(row.get("step"), int)]
        if steps:
            self.assign(key, "steps_completed", max(steps), path, 40)
        if training:
            self.assign(key, "last_train_loss", training[-1]["train_loss"], path, 70)
            self.assign(key, "answer_tokens_seen", training[-1].get("answer_tokens_seen"), path, 40)
        if evaluations:
            best, last = min(evaluations, key=lambda row: row["dev_loss"]), evaluations[-1]
            self.assign(key, "best_dev_loss", best["dev_loss"], path, 60)
            self.assign(key, "best_dev_step", best.get("step"), path, 60)
            self.assign(key, "last_dev_loss", last["dev_loss"], path, 70)
            self.assign(key, "last_train_eval_loss", last.get("train_eval_loss"), path, 70)
            self.assign(key, "train_eval_loss_at_best_dev", best.get("train_eval_loss"), path, 60)
            self.assign(key, "generalization_gap_at_best_dev", best.get("generalization_gap"), path, 60)
            if self.records[key]["summary_present"] and last.get("step") == self.records[key]["steps_completed"]:
                self.assign(key, "final_dev_loss", last["dev_loss"], path, 60)
                self.assign(key, "final_train_eval_loss", last.get("train_eval_loss"), path, 60)

    def scan_metadata(self, paths):
        for path in paths:
            value = read_json(path, self.errors)
            if not isinstance(value, dict):
                continue
            self.documents[path] = value
            # Curated measurements enrich runs present in the selected root;
            # they must not invent absent directories in another inventory.
            create = self.local_path(str(path), path.parent) is not None
            attempts = value.get("attempts")
            if isinstance(attempts, dict):
                for name, record in attempts.items():
                    if isinstance(record, dict):
                        key = self.identify(record, path.parent, create=create, fallback_name=name)
                        if key:
                            self.merge_record(key, record, path, 90 if create else 50)
            for container in (value, value.get("protocol") or {}):
                runs = container.get("runs") or []
                if isinstance(runs, list):
                    for record in runs:
                        if isinstance(record, dict):
                            key = self.identify(record, path.parent, create=create)
                            if key:
                                expected = record.get("checkpoint_and_record_sha256", {}).get("config.json")
                                if not create and expected:
                                    config_path = self.runs_root / key / "config.json"
                                    try:
                                        if hashlib.sha256(config_path.read_bytes()).hexdigest() != expected:
                                            continue
                                    except OSError:
                                        continue
                                self.merge_record(key, record, path, 80 if create else 50)
            # Original scheduler status may contain only current GPU assignments.
            assignments = value.get("gpu_assignments") or {}
            if isinstance(assignments, dict):
                for gpu, name in assignments.items():
                    key = self.identify({"name": name}, path.parent)
                    if key:
                        self.assign(key, "physical_gpu_ids", physical_ids(gpu), path, 75)

    def scan_launches(self):
        """A launch sidecar owns the run role even if its name suggests a model frame."""
        for path in sorted(self.cuda_root.rglob("*.launch.json")):
            value = read_json(path, self.errors)
            if not isinstance(value, dict):
                continue
            command = value.get("command")
            out = value.get("out") or value.get("path")
            if not out and isinstance(command, list):
                for index, arg in enumerate(command):
                    if arg == "--out" and index + 1 < len(command):
                        out = command[index + 1]
                        break
                    if isinstance(arg, str) and arg.startswith("--out="):
                        out = arg.split("=", 1)[1]
                        break
            if not out:
                out = str(path.parent if path.name == ".launch.json"
                          else path.with_name(path.name.removesuffix(".launch.json")))
            folder = self.local_path(out, path.parent)
            if folder is None:
                continue
            key = self.get(folder)
            self.merge_record(key, value, path, 110)
            started = value.get("started_unix")
            if isinstance(started, (int, float)) and not isinstance(started, bool):
                try:
                    self.assign(key, "started_utc", datetime.fromtimestamp(started, timezone.utc).isoformat(), path, 80)
                except (OverflowError, OSError, ValueError):
                    self.records[key]["metadata_errors"].append(f"{path}: invalid started_unix")

    def scan_registries(self):
        for path in sorted(self.cuda_root.rglob("run_registry.jsonl")):
            for record in iter_jsonl(path, self.errors):
                key = self.identify(record, path.parent, create=True)
                if key:
                    self.merge_record(key, record, path, 100)
        # Events fill gaps in older runs whose registry format predates JSONL.
        for path in sorted(self.cuda_root.rglob("events.jsonl")):
            for event in iter_jsonl(path, self.errors):
                if event.get("event") not in ("launched", "finished", "not_admitted"):
                    continue
                key = self.identify(event, path.parent, create=True)
                if not key:
                    continue
                record = dict(event)
                if event["event"] == "launched":
                    record.update(status="running", started_utc=event.get("time_utc"))
                elif event["event"] == "finished":
                    record["ended_utc"] = event.get("time_utc")
                else:
                    record.update(status="not_admitted", failure_reason=event.get("reason"))
                self.merge_record(key, record, path, 60)

    def finish(self):
        for key, record in self.records.items():
            folder = self.runs_root / key
            for parent in reversed(folder.parents):
                if parent != self.cuda_root and self.cuda_root not in parent.parents:
                    continue
                for name in ("provenance.json", "plan.json", "protocol.json"):
                    path = parent / name
                    document = self.documents.get(path)
                    if not document:
                        continue
                    host = document.get("host") or {}
                    for field in ("hostname", "public_ip"):
                        self.assign(key, field, host.get(field), path, 30)
                    protocol = document.get("protocol") or document
                    model = protocol.get("frame") or protocol.get("architecture")
                    if isinstance(model, dict) and "d_model" in model:
                        self.assign(key, "model_config", model, path, 30)
                    self.evidence(key, path)
            self.assign(key, "public_ip", self.host_public_ip, "--host-public-ip fallback", 5)
            model, spec, args = record["model_config"] or {}, record["architecture_spec"], record.pop("_args", {})
            if not record["architecture_sha256"] and (model or spec):
                payload = {"model": model or None, "spec": spec}
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                self.assign(key, "architecture_sha256", digest, "derived canonical JSON of model and full spec", 1)
                self.assign(key, "architecture_hash_kind", "model_and_spec_json", "derived", 1)
            name = key.lower()
            benchmark = record.pop("_benchmark", False) or args.get("benchmark_steps", 0)
            explicit_scope = record["run_role"]
            if explicit_scope in ("engineering_smoke", "benchmark", "historical_50m", "current_27m"):
                scope = explicit_scope
                self.assign(key, "scope", scope, record["field_sources"]["run_role"], 110)
            elif benchmark:
                scope = "benchmark"
            elif any(token in name for token in ("smoke", "parity", "engineering", "tiny")) or (model.get("d_model") and model["d_model"] < 384):
                scope = "engineering_smoke"
            elif model.get("d_model") == 512 and model.get("n_layers") == 10:
                scope = "historical_50m"
            elif model.get("d_model") == 384 and model.get("n_layers") == 8:
                scope = "current_27m"
            else:
                scope = "unknown"
            record["scope"] = scope
            if scope in ("engineering_smoke", "benchmark") and record["eligible_for_ranking"] is not False:
                self.assign(key, "eligible_for_ranking", False, "excluded scope: " + scope, 110)
            if record["phase"] is None and (record["architecture_name"] or "").startswith("baseline"):
                record["phase"] = "baseline"
            if record["status"] is None:
                record["status"] = "incomplete"
            if record["wall_seconds"] is None and record["started_utc"] and record["ended_utc"]:
                try:
                    start = datetime.fromisoformat(record["started_utc"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(record["ended_utc"].replace("Z", "+00:00"))
                    if start.tzinfo and end.tzinfo and end >= start:
                        self.assign(key, "wall_seconds", (end - start).total_seconds(), "derived from recorded UTC timestamps", 1)
                except (ValueError, TypeError):
                    pass
            record["metrics_available"] = [field for field in ("best_dev_loss", "final_dev_loss", "last_train_loss",
                                              "last_train_eval_loss", "train_eval_loss_at_best_dev", "final_train_eval_loss")
                                           if record[field] is not None]
        rows = [self.records[key] for key in sorted(self.records)]
        return {"format_version": 1, "generated_utc": now_utc(), "runs_root": str(self.runs_root),
                "inventory_root": str(self.cuda_root), "validation_accessed": False,
                "run_count": len(rows), "runs": rows, "metadata_errors": self.errors,
                "notes": ["Null means unknown; physical GPUs are never inferred from logical CUDA device numbers.",
                          "Status reflects the last recorded evidence, not a live process inspection.",
                          "Queued/not-admitted/failed attempts from registries are included even without config.json.",
                          "Benchmark requested steps describe the LR schedule; completed steps include benchmark warmup.",
                          "Explicit run roles override scope heuristics; engineering smoke and benchmark runs are excluded from ranking.",
                          "Ranking eligibility is null unless explicitly recorded or excluded by scope.",
                          "Architecture hashes preserve scheduler canonical IDs where available; fallback JSON hashes have a distinct kind.",
                          "Per-run source artifacts and append-only registries remain unchanged."]}


def build_inventory(runs_root, host_public_ip=None, metadata_paths=()):
    inventory = Inventory(runs_root, host_public_ip)
    inventory.scan_configs()
    paths = [path for path in inventory.cuda_root.rglob("*.json") if path.name in METADATA_NAMES]
    paths += [ROOT / "experiments" / "cuda" / name for name in ("baselines-27m.json", "baselines-50m.json")
              if (ROOT / "experiments" / "cuda" / name).is_file()]
    for given in metadata_paths:
        path = Path(given).expanduser().resolve()
        candidates = path.glob("*.json") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.name in ("train.json", "validation.json", "tokenizer.json"):
                raise ValueError(f"dataset files are not inventory metadata: {candidate}")
            paths.append(candidate)
    inventory.scan_metadata(sorted(set(paths)))
    inventory.scan_registries()
    inventory.scan_launches()
    return inventory.finish()


def write_inventory(index, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "RUN_INDEX.json", output_dir / "RUN_INDEX.csv"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    temporary.replace(json_path)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in index["runs"]:
            writer.writerow({field: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for field in CSV_FIELDS for value in [record.get(field)]})
    temporary.replace(csv_path)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--host-public-ip", help="fallback for records without a saved public IP")
    parser.add_argument("--metadata", type=Path, action="append", default=[], help="additional protocol/measurement JSON file or directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to RUNS_ROOT/cuda")
    args = parser.parse_args()
    index = build_inventory(args.runs_root, args.host_public_ip, args.metadata)
    outputs = write_inventory(index, args.output_dir or args.runs_root / "cuda")
    print(json.dumps({"runs": index["run_count"], "outputs": [str(path) for path in outputs], "metadata_errors": len(index["metadata_errors"])}))


if __name__ == "__main__":
    main()
