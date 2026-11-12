"""Run independent final artifact exporters after the frozen workflow finishes."""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, python_command, source_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None or args.poll_seconds <= 0:
        parser.error("deadline must include a timezone and polling must be positive")
    destination = root / "postprocessing.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        return 0 if previous.get("status") == "complete" else 1
    report = root / "FINAL_REPORT.json"
    while utc() < deadline:
        if report.is_file():
            try:
                status = json.loads(report.read_text()).get("status")
            except json.JSONDecodeError:
                status = None
            if status in ("complete", "no_qualified_candidate"):
                break
        time.sleep(min(args.poll_seconds, max(0, (deadline - utc()).total_seconds())))
    else:
        print("Final report was not complete before postprocessing deadline.", flush=True)
        return 2
    project = PROJECT_ROOT
    scripts = ["cuda_minparams_report.py", "cuda_minparams_audit.py"]
    if status == "complete":
        scripts.insert(0, "cuda_minparams_architecture_report.py")
    result = {"started_utc": utc().isoformat(), "status": "running", "commands": [],
              "final_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
    with (root / "postprocessing.log").open("a", buffering=1) as log:
        for script in scripts:
            command = [*python_command(script, project), "--root", str(root)]
            log.write(json.dumps({"started_utc": utc().isoformat(), "command": command}) + "\n")
            completed = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
            result["commands"].append({"script": script, "returncode": completed.returncode,
                                       "source_sha256": hashlib.sha256((source_path(script, project)).read_bytes()).hexdigest()})
            if completed.returncode:
                result["status"] = "failed"
                break
        else:
            result["status"] = "complete"
    result["finished_utc"] = utc().isoformat()
    temporary = destination.with_suffix(".pending")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
