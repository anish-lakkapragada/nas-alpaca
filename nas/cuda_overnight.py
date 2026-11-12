"""Run the authorized search, then freeze and evaluate its selected comparison.

The separate local Lambda watchdog owns cloud shutdown. This controller never
reads credentials, and only the finalizer opens the reserved validation split.
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = PROJECT_ROOT


def now():
    return datetime.now(timezone.utc)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline requires a timezone")
    return parsed.astimezone(timezone.utc)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-utc", type=utc, required=True)
    parser.add_argument("--training-cutoff-utc", type=utc, required=True)
    parser.add_argument("--finalization-cutoff-utc", type=utc, required=True)
    parser.add_argument("scheduler_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not now() < args.training_cutoff_utc < args.finalization_cutoff_utc < args.deadline_utc:
        raise SystemExit("require future training < finalization < rental deadline")
    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        raise SystemExit("search output already exists")
    control = args.out.with_name(args.out.name + "-control")
    control.mkdir(parents=True, exist_ok=False)
    source = control / "source"
    source.mkdir()
    hashes = {}
    for name in (*PROVENANCE_SUPPORT, "cuda_model.py", "cuda_train.py", "cuda_search.py", "cuda_evaluate.py",
                 "cuda_report.py", "cuda_overnight.py", "data.py",
                 "test_cuda.py", "requirements-cuda.txt"):
        contents = (source_path(name, ROOT)).read_bytes()
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(contents)
        hashes[name] = hashlib.sha256(contents).hexdigest()
    save(control / "source_manifest.json", hashes)
    extra = args.scheduler_args[1:] if args.scheduler_args[:1] == ["--"] else args.scheduler_args
    for forbidden in ("--out", "--deadline-utc", "--training-cutoff-utc", "--hours", "--plan-only"):
        if any(value == forbidden or value.startswith(forbidden + "=") for value in extra):
            raise SystemExit(f"controller owns {forbidden}")
    command = [*python_command("cuda_search.py", ROOT), "--out", str(args.out),
               "--deadline-utc", args.deadline_utc.isoformat(),
               "--training-cutoff-utc", args.training_cutoff_utc.isoformat(), *extra]
    state = {"started_utc": now().isoformat(), "status": "search_starting", "search_root": str(args.out),
             "deadline_utc": args.deadline_utc.isoformat(), "training_cutoff_utc": args.training_cutoff_utc.isoformat(),
             "finalization_cutoff_utc": args.finalization_cutoff_utc.isoformat(), "command": command,
             "source_sha256": hashes}
    save(control / "workflow.json", state)
    interrupted = False
    active = None

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if active and active.poll() is None:
            active.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="8", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    with (control / "search.log").open("x") as log:
        active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="search_running", scheduler_pid=active.pid)
        save(control / "workflow.json", state)
        timeout = max(1, (args.training_cutoff_utc + timedelta(seconds=30) - now()).total_seconds())
        try:
            code = active.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            active.terminate()
            try:
                code = active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                active.kill()
                code = active.wait()
            state["search_timeout"] = True
    state.update(search_returncode=code, search_finished_utc=now().isoformat())
    save(control / "workflow.json", state)
    if interrupted or code != 0:
        state.update(status="interrupted" if interrupted else "search_failed", finished_utc=now().isoformat())
        save(control / "workflow.json", state)
        return code or 130
    # Final reporting tooling may be installed while the long search runs;
    # freeze its exact source before it can access the reserved split.
    finalization_hashes = {}
    for name in ("cuda_finalize.py", "cuda_inventory.py", "cuda_evaluate.py", "cuda_report.py"):
        if not (source_path(name, ROOT)).is_file():
            state.update(status="finalization_failed", error=f"Missing {name}", finished_utc=now().isoformat())
            save(control / "workflow.json", state)
            return 3
        contents = (source_path(name, ROOT)).read_bytes()
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(contents)
        finalization_hashes[name] = hashlib.sha256(contents).hexdigest()
    state["finalization_source_sha256"] = finalization_hashes
    command = [*python_command("cuda_finalize.py", ROOT), "--out", str(args.out),
               "--deadline-utc", args.finalization_cutoff_utc.isoformat()]
    state.update(status="finalization_running", finalization_command=command)
    save(control / "workflow.json", state)
    with (control / "finalization.log").open("x") as log:
        active = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES="0"), stdout=log, stderr=subprocess.STDOUT)
        try:
            code = active.wait(timeout=max(1, (args.finalization_cutoff_utc - now()).total_seconds()))
        except subprocess.TimeoutExpired:
            active.terminate()
            try:
                code = active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                active.kill()
                code = active.wait()
            state["finalization_timeout"] = True
    state.update(status="completed" if code == 0 and not interrupted else "finalization_failed",
                 finalization_returncode=code, finished_utc=now().isoformat())
    save(control / "workflow.json", state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
