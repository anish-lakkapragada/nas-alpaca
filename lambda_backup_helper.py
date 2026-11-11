"""Prepare one task-specific Lambda backup helper; default is an offline plan.

Only --execute creates resources. It never deletes or terminates anything, and
never retries an ambiguous launch. The caller must install the remote backup
and termination services after launch. Credentials remain in lambda_cloud's
existing provider; only the fresh helper's SSH public key is uploaded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from lambda_cloud import request

WORKSPACE = "ceb65cae26634a6081d7de0546c1d236"
TASK_INSTANCE = "5c07904125d442d8a43d41a1fd085ef3"
NAME = "nas-backup-20261111"
REGION = "us-east-1"
INSTANCE_TYPE = "gpu_1x_a10"
MOUNT = f"/lambda/nfs/{NAME}"
MAX_CENTS_PER_HOUR = 129
PRIVATE_DIR = Path(__file__).resolve().parents[2] / ".lambda"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def plan(filesystem_id="<NEW_TASK_FILESYSTEM_ID>", public_key="<NEW_HELPER_PUBLIC_KEY>"):
    return {
        "ssh-key": {"endpoint": "ssh-keys", "body": {
            "name": NAME, "public_key": public_key, "workspace_id": WORKSPACE}},
        "filesystem": {"endpoint": "filesystems", "body": {
            "name": NAME, "region": REGION, "workspace_id": WORKSPACE}},
        "launch": {"endpoint": "instance-operations/launch", "body": {
            "region_name": REGION, "instance_type_name": INSTANCE_TYPE,
            "ssh_key_names": [NAME], "file_system_mounts": [{
                "file_system_id": filesystem_id, "mount_point": MOUNT}],
            "workspace_id": WORKSPACE, "name": NAME,
            "tags": [{"key": "nas-task-instance", "value": TASK_INSTANCE},
                     {"key": "nas-purpose", "value": "durable-backup-helper"}]}},
        "price_cap_cents_per_hour": MAX_CENTS_PER_HOUR,
        "storage_rate": "not exposed by public region/filesystem APIs",
        "private_key_path": str(PRIVATE_DIR / f"{NAME}_ed25519"),
        "state_path": str(PRIVATE_DIR / f"{NAME}.json"),
        "followup_required": "Verify actual NFS mount; install backup and API termination services.",
    }


def save(path, state):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def execute(private_dir=PRIVATE_DIR, api=request):
    private_dir = Path(private_dir)
    private_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_path = private_dir / f"{NAME}.json"
    key_path = private_dir / f"{NAME}_ed25519"
    with open(private_dir / f"{NAME}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "task_instance_id": TASK_INSTANCE, "workspace_id": WORKSPACE,
            "name": NAME, "region": REGION, "created_at": now(),
            "private_key_path": str(key_path), "mount_point": MOUNT}
        if state.get("task_instance_id") != TASK_INSTANCE or state.get("name") != NAME:
            raise RuntimeError("Existing state belongs to a different task")
        if state.get("launch_attempted_at"):
            raise RuntimeError("Launch already attempted; inspect saved instance_id/API before any retry")

        last_request = 0.0

        def call(endpoint, body=None):
            nonlocal last_request
            time.sleep(max(0.0, 1.1 - (time.monotonic() - last_request)))
            last_request = time.monotonic()
            return api(endpoint, body)

        target = call(f"instances/{TASK_INSTANCE}")
        if target.get("workspace_id") != WORKSPACE or target.get("status") != "active":
            raise RuntimeError("Task instance is not active in the expected workspace")
        types = call("instance-types")
        offer = types[INSTANCE_TYPE]
        cents = offer["instance_type"]["price_cents_per_hour"]
        if cents > MAX_CENTS_PER_HOUR or REGION not in {
                region["name"] for region in offer["regions_with_capacity_available"]}:
            raise RuntimeError("Selected helper is unavailable or exceeds its hourly cap")
        state["price_cents_per_hour"] = cents
        state["capacity_checked_at"] = now()

        existing_keys = call(f"ssh-keys?workspace_id={WORKSPACE}")
        existing_filesystems = call(f"filesystems?workspace_id={WORKSPACE}")
        if not state.get("ssh_key_id") and any(x["name"] == NAME for x in existing_keys):
            raise RuntimeError("Task SSH-key name already exists without saved ownership; inspect manually")
        if not state.get("filesystem_id") and any(x["name"] == NAME for x in existing_filesystems):
            raise RuntimeError("Task filesystem name already exists without saved ownership; inspect manually")
        save(state_path, state)

        if not key_path.exists():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", NAME,
                            "-f", str(key_path)], check=True, capture_output=True, text=True)
        if key_path.is_symlink() or key_path.stat().st_mode & 0o077:
            raise RuntimeError("Helper private key must be an owner-only regular file")
        public_key = key_path.with_suffix(".pub").read_text().strip()
        requests = plan(public_key=public_key)
        if not state.get("ssh_key_id"):
            state["ssh_key_create_attempted_at"] = now()
            save(state_path, state)
            entry = requests["ssh-key"]
            result = call(entry["endpoint"], entry["body"])
            state["ssh_key_id"] = result["id"]
            save(state_path, state)
        if not state.get("filesystem_id"):
            state["filesystem_create_attempted_at"] = now()
            save(state_path, state)
            entry = requests["filesystem"]
            result = call(entry["endpoint"], entry["body"])
            state["filesystem_id"] = result["id"]
            save(state_path, state)

        filesystem = next((x for x in existing_filesystems if x["id"] == state["filesystem_id"]), None)
        if filesystem is not None and (
                filesystem["name"] != NAME or filesystem["region"]["name"] != REGION
                or filesystem.get("workspace_id") != WORKSPACE or filesystem["is_in_use"]):
            raise RuntimeError("Saved task filesystem does not match the expected unused resource")
        entry = plan(filesystem_id=state["filesystem_id"])["launch"]
        state["launch_request"] = entry["body"]
        state["launch_attempted_at"] = now()
        save(state_path, state)
        result = call(entry["endpoint"], entry["body"])
        ids = result["instance_ids"]
        state["returned_instance_ids"] = ids
        state["launch_response_at"] = now()
        if len(ids) == 1:
            state["instance_id"] = ids[0]
        save(state_path, state)
        if len(ids) != 1:
            raise RuntimeError("Unexpected instance count; inspect saved IDs immediately")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create the reviewed filesystem, key and one helper")
    args = parser.parse_args()
    result = execute() if args.execute else plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
