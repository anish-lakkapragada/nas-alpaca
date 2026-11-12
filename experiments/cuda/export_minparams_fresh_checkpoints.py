"""Export metadata for exactly ten completed fresh best-dev checkpoints.

Writes ROOT/fresh_checkpoint_manifest.json. Requires the original run directory
and final_confirmation.json, but does not require test evaluation to finish.
Only JSON metadata and file stats are read; checkpoint contents, test reports,
conversation arrays, and datasets are never opened. Expected checkpoint hashes
must be verified against the copied bytes by the caller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import sys


SEEDS = [100, 101, 102, 103, 104]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamp lacks timezone")
    return result.astimezone(timezone.utc)


def contained(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    require(path.is_relative_to(root), "metadata path is outside the run root")
    require(".." not in path.relative_to(root).parts, "path traversal is not permitted")
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        require(not current.is_symlink(), f"symlink is not permitted: {current}")
    require(path.resolve().is_relative_to(root), "resolved path escapes the run root")
    return path.resolve()


def export(root):
    root = root.resolve()
    require(root.is_dir(), "run root does not exist")
    inputs = {}

    def metadata(path, expected_hash=None):
        path = contained(root, path)
        raw = path.read_bytes()
        fingerprint = sha256(raw)
        require(expected_hash is None or fingerprint == expected_hash, f"metadata hash differs: {path}")
        inputs[path.relative_to(root).as_posix()] = fingerprint
        return json.loads(raw)

    selection = metadata(root / "selection.json")
    selection_hash = inputs["selection.json"]
    require(selection.get("status") == "qualified" and selection.get("validation_used") is False
            and selection.get("selection_seeds") == [0, 1, 2] and selection.get("fresh_seeds") == SEEDS,
            "selection is not a qualified freeze using the expected seeds")
    selection_time = timestamp(selection["frozen_utc"])
    models = {"baseline": selection["baseline"], "winner": selection["primary"]}
    require(selection["primary_architecture_id"] == models["winner"]["architecture_id"], "frozen winner IDs differ")
    require(models["baseline"]["parameters"] == 27_144_960
            and 0 < models["winner"]["parameters"] < 27_144_960, "baseline/candidate parameter counts differ from protocol")
    require(all(SHA256.fullmatch(model["architecture_id"]) for model in models.values())
            and models["baseline"]["architecture_id"] != models["winner"]["architecture_id"], "invalid architecture identities")
    confirmation = metadata(root / "final_confirmation.json")
    require(confirmation.get("status") == "frozen_checkpoints"
            and confirmation.get("selection_freeze_sha256") == selection_hash
            and confirmation.get("fresh_seeds_used_for_architecture_selection") is False,
            "fresh checkpoint confirmation is missing or belongs to a different selection")
    confirmation_time = timestamp(confirmation["frozen_utc"])
    require(confirmation_time >= selection_time, "checkpoint freeze predates architecture selection")
    runs = confirmation["runs"]
    require(len(runs) == 10, "confirmation must contain exactly ten fresh runs")

    ledger_path = contained(root, root / "run_registry.jsonl")
    raw = ledger_path.read_bytes()
    inputs["run_registry.jsonl"] = sha256(raw)
    latest = {}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        latest[row["job"]["run_id"]] = row
    fresh = {key: row for key, row in latest.items() if row["job"].get("phase") == "final"}
    require(not any(row["status"] in ("running", "queued") for row in fresh.values()), "fresh training is still incomplete")
    completed = {key for key, row in fresh.items() if row["status"] == "completed"}
    require(len(completed) == 10, "ledger must contain exactly ten completed fresh runs")

    items, seen, seen_ids = [], set(), set()
    for run in runs:
        role, seed = run["role"], run["seed"]
        require(role in models and type(seed) is int and seed in SEEDS and (role, seed) not in seen,
                "confirmation does not contain unique paired baseline/winner fresh seeds")
        seen.add((role, seed))
        architecture = models[role]
        run_path = contained(root, run["run"])
        run_id = run_path.name
        require(run_path == root / "attempts" / run_id and run_id not in seen_ids and run_id in completed,
                "confirmation run path/identity is absent, duplicated, or noncanonical")
        seen_ids.add(run_id)
        record = fresh[run_id]
        job = record["job"]
        require(job.get("steps") == 5000 and job.get("seed") == seed
                and job.get("architecture_id") == run["architecture_id"] == architecture["architecture_id"]
                and record.get("role") == role and record.get("selection_freeze_sha256") == selection_hash
                and contained(root, record["path"]) == run_path,
                f"fresh ledger/confirmation identity differs: {run_id}")
        require(selection_time <= timestamp(record["started_utc"]) <= timestamp(record["ended_utc"]) <= confirmation_time,
                f"fresh run is not completed between selection and checkpoint freeze: {run_id}")
        config = metadata(run_path / "config.json", run["config_sha256"])
        summary = metadata(run_path / "summary.json", run["summary_sha256"])
        require(summary == record["summary"] and summary["steps"] == summary["schedule_steps"] == 5000
                and not summary.get("benchmark") and config["args"]["steps"] == 5000
                and config["args"]["seed"] == seed and not config["args"].get("resume")
                and not config["args"].get("benchmark_steps"), f"fresh run is not a complete from-scratch 5000-step run: {run_id}")
        require(config["model"] == record["frame"] == run["frame"] == architecture["frame"]
                and config["parameters"] == summary["parameters"] == record["parameters"] == run["parameters"] == architecture["parameters"],
                f"fresh model metadata differs: {run_id}")
        best_step = summary["best_step"]
        require(type(best_step) is int and 0 <= best_step <= 5000 and best_step == run["best_step"],
                f"best-dev checkpoint step differs: {run_id}")
        saved = record["checkpoints"]["best.pt"]
        checkpoint = contained(root, run["checkpoint"])
        require(checkpoint == run_path / "best.pt" == contained(root, saved["path"])
                and saved.get("pruned") is False, f"best checkpoint path is inconsistent or pruned: {run_id}")
        expected_hash = saved["sha256"]
        require(SHA256.fullmatch(expected_hash) and expected_hash == run["checkpoint_sha256"],
                f"ledger/confirmation checkpoint hashes differ: {run_id}")
        size = saved["bytes"]
        info = checkpoint.stat()  # Intentionally never open checkpoint bytes.
        require(type(size) is int and size > 0 and stat.S_ISREG(info.st_mode) and info.st_size == size,
                f"checkpoint is missing, not regular, or has a different byte size: {run_id}")
        items.append({"role": role, "seed": seed, "run_id": run_id,
                      "architecture_id": architecture["architecture_id"], "parameters": architecture["parameters"],
                      "path": str(checkpoint), "relative_path": checkpoint.relative_to(root).as_posix(),
                      "bytes": size, "sha256": expected_hash, "training_steps": 5000, "checkpoint_step": best_step})
    require(seen == {(role, seed) for role in models for seed in SEEDS} and seen_ids == completed,
            "checkpoint set does not exactly cover the ten completed fresh paired runs")
    items.sort(key=lambda row: (row["role"] != "baseline", row["seed"]))
    for name, expected_hash in inputs.items():
        require(sha256((root / name).read_bytes()) == expected_hash, "metadata changed during export; retry")
    result = {"schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_root": str(root), "selection_sha256": selection_hash,
              "checkpoint_count": len(items), "total_bytes": sum(row["bytes"] for row in items),
              "fresh_seeds": SEEDS, "steps_per_training_run": 5000,
              "checkpoint_policy": "best dev checkpoint within each completed 5000-step run",
              "checkpoint_contents_read": False, "checkpoint_content_hashes_verified": False,
              "hash_note": "SHA256 values are expected hashes agreed by the completed-run ledger and checkpoint freeze; verify copied file contents against them.",
              "metadata_sha256": inputs, "checkpoints": items}
    target = root / "fresh_checkpoint_manifest.json"
    temporary = root / ".fresh_checkpoint_manifest.pending"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return target, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        target, result = export(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Fresh checkpoint export refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "complete", "manifest": str(target),
                      "checkpoints": result["checkpoint_count"], "bytes": result["total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
