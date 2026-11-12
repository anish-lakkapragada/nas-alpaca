"""Run the authorized search and frozen-finalist confirmation without a laptop."""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

PROJECT = PROJECT_ROOT


def now():
    return datetime.now(timezone.utc)


def timestamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError("deadline must include a timezone")
    return result.astimezone(timezone.utc)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--deadline", type=parse_time, required=True)
    parser.add_argument("--training-cutoff", type=parse_time, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".workflow.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"started_utc": timestamp(now()), "pid": os.getpid(),
              "root": str(root), "deadline_utc": timestamp(args.deadline),
              "search_cutoff_utc": timestamp(args.training_cutoff),
              "gpus": args.gpus, "training_steps_per_run": 5000}

    def event(stage, **details):
        status.update(stage=stage, updated_utc=timestamp(now()), **details)
        atomic_json(root / "workflow.json", status)
        with (root / "workflow_events.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(status), flush=True)

    core = [*PROVENANCE_SUPPORT, "cuda_model.py", "cuda_train.py", "data.py", "cuda_minparams_search.py",
            "cuda_minparams_finalize.py", "cuda_minparams_workflow.py", "cuda_evaluate.py",
            "cuda_report.py", "cuda_search.py", "cuda_finalize.py",
            "data/train.json", "data/tokenizer.json", "data/validation.json"]
    hashes = {name: hashlib.sha256((source_path(name, PROJECT)).read_bytes()).hexdigest() for name in core}
    manifest = root / "workflow_source_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["sha256"] != hashes:
            raise RuntimeError("frozen workflow source/data changed; refusing an unrecorded restart")
    else:
        atomic_json(manifest, {"created_utc": timestamp(now()), "sha256": hashes})
        source = root / "source"
        source.mkdir(exist_ok=True)
        for name in core:
            if not name.startswith("data/"):
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_bytes((source_path(name, PROJECT)).read_bytes())

    report = root / "FINAL_REPORT.json"
    if report.exists() and json.loads(report.read_text()).get("status") in ("complete", "no_qualified_candidate"):
        event("completed", report=str(root / "FINAL_REPORT.json"), resumed_after_completion=True)
        return
    if now() >= args.deadline:
        event("deadline_exceeded", error="No final report existed at the deadline")
        raise SystemExit(2)

    def run_stage(stage, command, timeout):
        event(stage, command=command)
        with (root / (stage + ".stdout.log")).open("a", buffering=1) as stream:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=stream,
                                       stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                event(stage + "_timeout", controller_pid=process.pid)
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise SystemExit(2)
        if returncode:
            event(stage + "_failed", returncode=returncode)
            raise SystemExit(returncode)
        event(stage + "_completed")

    if not (root / "selection.json").exists():
        command = [*python_command("cuda_minparams_search.py", PROJECT),
                   "--root", str(root), "--gpus", args.gpus,
                   "--deadline", timestamp(args.deadline),
                   "--training-cutoff", timestamp(args.training_cutoff), "--resume"]
        if args.compile:
            command.append("--compile")
        run_stage("search", command, max(60, (args.deadline - now()).total_seconds() - 300))

    final_deadline = args.deadline - timedelta(minutes=5)
    command = [*python_command("cuda_minparams_finalize.py", PROJECT),
               "--root", str(root), "--gpus", args.gpus,
               "--deadline", timestamp(final_deadline)]
    if args.compile:
        command.append("--compile")
    run_stage("confirmation", command, max(1, (args.deadline - now()).total_seconds()))
    if not (root / "FINAL_REPORT.json").exists():
        event("missing_final_report")
        raise SystemExit(3)
    final_status = json.loads(report.read_text()).get("status")
    if final_status not in ("complete", "no_qualified_candidate"):
        event("incomplete_confirmation", report=str(report))
        raise SystemExit(3)
    event("completed", report=str(report))


if __name__ == "__main__":
    main()
