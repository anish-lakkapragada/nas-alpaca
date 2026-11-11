"""Root watchdog for only this task's persistent-backup helper.

No deployment or credentials are embedded. ``--self-test`` makes no API calls.
``--config /root/nas/helper-watchdog.json --preflight`` only reads the exact
helper/main instances. The daemon can terminate ONLY the hardcoded helper ID.
Before the hard deadline it requires both the main instance to be terminated
or absent and a verified receipt with a matching manifest on the mounted NFS.
The absolute deadline is enforced even when backup evidence or API reads fail.

Root-owned 0600 JSON config: instance_id, main_instance_id, credential_file,
backup_root (/lambda/nfs/FILESYSTEM/MAIN_INSTANCE_ID), receipt_file, state_file;
receipt_file must be a private root-owned file under /root, separate from NFS
so root-squash on the persistent filesystem cannot prevent its publication;
optional deadline_utc (cannot extend hard deadline), poll_seconds (default15,
maximum30), api_timeout_seconds (default10, maximum20), retry_seconds(default10).
Deploy lambda_remote_watchdog.py alongside this file for generic private-file,
curl transport, timestamp, and redaction functions only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from lambda_remote_watchdog import (
    ApiError, api_request, atomic_private_json, cutoff_reached, private_read,
    redact, utc, wall_now,
)

HELPER_INSTANCE_ID = "9ed2a014d28347bdaa4816926f9f1f32"
MAIN_INSTANCE_ID = "5c07904125d442d8a43d41a1fd085ef3"
HARD_DEADLINE = datetime(2026, 11, 11, 16, 15, tzinfo=timezone.utc)
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def helper_payload(instance_id):
    if instance_id != HELPER_INSTANCE_ID:
        raise ValueError("helper watchdog cannot terminate this instance")
    return {"instance_ids": [HELPER_INSTANCE_ID]}


def emit(event, secrets=(), **details):
    record = {"event": event, "at_utc": wall_now().isoformat(),
              "instance_id": HELPER_INSTANCE_ID, "main_instance_id": MAIN_INSTANCE_ID, **details}
    print(redact(json.dumps(record, allow_nan=False), secrets), flush=True)


def validate_config(config):
    helper_payload(config["instance_id"])
    if config["main_instance_id"] != MAIN_INSTANCE_ID:
        raise ValueError("main instance differs from this task")
    deadline = utc(config.get("deadline_utc", HARD_DEADLINE.isoformat()))
    if deadline > HARD_DEADLINE:
        raise ValueError("configured cutoff exceeds hard deadline")
    result = dict(config, deadline=deadline)
    for name in ("credential_file", "backup_root", "receipt_file", "state_file"):
        path = Path(config[name])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("configuration paths must be absolute without traversal")
        result[name] = path
    root = result["backup_root"]
    if not re.fullmatch(r"/lambda/nfs/[a-zA-Z0-9_-]+/" + MAIN_INSTANCE_ID, str(root)):
        raise ValueError("backup root must be this task's persistent filesystem directory")
    if result["receipt_file"].parts[1] != "root":
        raise ValueError("receipt must be private under /root")
    if len({result[name] for name in ("credential_file", "receipt_file", "state_file")}) != 3:
        raise ValueError("credential, receipt, and state paths must be distinct")
    for name, default, maximum in (("poll_seconds", 15, 30), ("api_timeout_seconds", 10, 20),
                                   ("retry_seconds", 10, 30)):
        number = config.get(name, default)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 1 <= number <= maximum:
            raise ValueError("invalid polling/timeout limit")
        result[name] = float(number)
    return result


def conservative_deadline(deadline, wall, monotonic, prior=None, boot_id=None):
    cutoff = monotonic + max(0, (deadline - wall).total_seconds())
    if (prior and prior.get("instance_id") == HELPER_INSTANCE_ID
            and prior.get("boot_id") == boot_id):
        old = prior.get("monotonic_deadline")
        if isinstance(old, (int, float)) and math.isfinite(old):
            cutoff = min(cutoff, old)
    return cutoff


def instance_state(instance_id, key, timeout):
    if instance_id not in (MAIN_INSTANCE_ID, HELPER_INSTANCE_ID):
        raise ValueError("status reads are restricted to this task's two instances")
    data = api_request("instances/" + instance_id, key, timeout)
    if not isinstance(data, dict) or data.get("id") != instance_id:
        raise ApiError(category="instance_identity_mismatch")
    status = data.get("status")
    if not isinstance(status, str) or not re.fullmatch(r"[a-z_]{1,32}", status):
        raise ApiError(category="invalid_instance_state")
    return {"id": instance_id, "status": status}


def main_is_gone(status=None, http_status=None):
    return status == "terminated" or http_status == 404


def verified_backup(receipt, manifest, manifest_sha256, current_time):
    """Pure receipt/manifest predicate; periodic complete snapshots also qualify.

    Finalization need not have succeeded: after the main machine is gone, a
    verified partial-search snapshot still survives on the persistent filesystem.
    """
    try:
        if not isinstance(receipt, dict) or not isinstance(manifest, dict):
            return False
        if any(value.get("schema_version") != 1 or value.get("instance_id") != MAIN_INSTANCE_ID
               for value in (receipt, manifest)):
            return False
        for name in ("verified", "durable", "backup_complete", "metadata_complete", "retained_checkpoints_complete"):
            if receipt.get(name) is not True:
                return False
        if (not isinstance(manifest_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", manifest_sha256)
                or receipt.get("manifest_sha256") != manifest_sha256
                or receipt.get("manifest_object_key") != "manifests/" + manifest_sha256 + ".json"):
            return False
        destination = receipt.get("destination")
        if not isinstance(destination, str) or not destination or manifest.get("destination") != destination:
            return False
        if manifest.get("finalization_status") != receipt.get("finalization_status"):
            return False
        if manifest.get("final_report_sha256") != receipt.get("final_report_sha256"):
            return False
        finished, created = utc(receipt["completed_utc"]), utc(manifest["created_utc"])
        if not -5 <= (current_time - finished).total_seconds() <= 24 * 60 * 60:
            return False
        if not -5 <= (finished - created).total_seconds() <= 10 * 60:
            return False
        files = manifest.get("files")
        if not isinstance(files, dict) or not files or receipt.get("artifact_count") != len(files):
            return False
        kinds = set()
        report_present = False
        for name, item in files.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                return False
            digest, size = item.get("sha256"), item.get("bytes")
            if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                    or item.get("object_key") != "objects/" + digest
                    or isinstance(size, bool) or not isinstance(size, int) or size < 0):
                return False
            kinds.add(item.get("kind"))
            report_present |= name.endswith("/FINAL_REPORT.json") and digest == receipt.get("final_report_sha256")
        if not {"metadata", "checkpoint"}.issubset(kinds):
            return False
        if receipt.get("finalization_status") == "complete" and not report_present:
            return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def backup_evidence(config):
    try:
        root = config["backup_root"]
        if (root.resolve() != root or not os.path.ismount(root.parent)
                or root.parent.resolve() != root.parent):
            return False, None
        receipt = json.loads(private_read(config["receipt_file"]))
        digest = receipt.get("manifest_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            return False, None
        manifest_path = root / "manifests" / (digest + ".json")
        if manifest_path.resolve() != manifest_path or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            return False, None
        # A content-addressed manifest must have the exact receipt hash. The
        # backup adapter already verified every object's size and SHA256 before
        # publishing this receipt; avoid rehashing all checkpoints at cutoff.
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        actual = hashlib.sha256(raw).hexdigest()
        return verified_backup(receipt, manifest, actual, wall_now()), actual
    except (OSError, TypeError, ValueError, AttributeError):
        return False, None


def termination_reason(due, main_gone, backup_verified):
    if due:
        return "hard_deadline"
    return "main_terminated_and_durably_backed_up" if main_gone and backup_verified else None


def run(config, key):
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    previous = json.loads(private_read(config["state_file"])) if config["state_file"].exists() else {}
    if previous.get("instance_id") not in (None, HELPER_INSTANCE_ID):
        raise ValueError("state belongs to another instance")
    if previous.get("termination_confirmed") is True:
        emit("already_terminated", (key,))
        return
    cutoff = conservative_deadline(config["deadline"], wall_now(), time.monotonic(), previous, boot_id)
    state = {**previous, "instance_id": HELPER_INSTANCE_ID, "main_instance_id": MAIN_INSTANCE_ID,
             "boot_id": boot_id, "monotonic_deadline": cutoff, "deadline_utc": config["deadline"].isoformat(),
             "started_utc": wall_now().isoformat(), "pid": os.getpid()}

    def save(**updates):
        state.update(updates, updated_utc=wall_now().isoformat())
        atomic_private_json(config["state_file"], state)

    save(status="running")
    emit("armed", (key,), deadline_utc=config["deadline"].isoformat())
    while True:
        due = cutoff_reached(config["deadline"], cutoff, wall_now(), time.monotonic())
        main_gone, verified = False, False
        if not due and not state.get("termination_requested"):
            remaining = min(cutoff - time.monotonic(), (config["deadline"] - wall_now()).total_seconds())
            if remaining > 1:
                try:
                    current = instance_state(MAIN_INSTANCE_ID, key, min(config["api_timeout_seconds"], remaining - 1))
                    main_gone = main_is_gone(current["status"])
                    save(main_status=current["status"])
                except (ApiError, OSError) as error:
                    main_gone = main_is_gone(http_status=getattr(error, "status", None))
                    save(main_status="absent" if main_gone else "read_error")
                    if not main_gone:
                        emit("main_status_error", (key,), category=getattr(error, "category", type(error).__name__),
                             http_status=getattr(error, "status", None))
                if main_gone:
                    verified, digest = backup_evidence(config)
                    save(main_terminated=True, backup_verified=verified, manifest_sha256=digest)
                    emit("main_terminated", (key,), backup_verified=verified)
            due = cutoff_reached(config["deadline"], cutoff, wall_now(), time.monotonic())
        reason = termination_reason(due, main_gone, verified)
        if reason or state.get("termination_requested"):
            if due and not state.get("deadline_logged"):
                save(deadline_logged=True)
                emit("deadline_cutoff", (key,), backup_verified=state.get("backup_verified", False),
                     main_status=state.get("main_status", "unknown"))
            save(termination_attempted=True, termination_reason=reason or state.get("termination_reason"))
            try:
                # Never let a status read delay the first hard-deadline POST.
                api_request("instance-operations/terminate", key, config["api_timeout_seconds"],
                            helper_payload(config["instance_id"]))
                save(status="termination_requested", termination_requested=True)
                emit("termination_requested", (key,), reason=state["termination_reason"])
            except (ApiError, OSError) as error:
                save(status="termination_retry", termination_last_error=type(error).__name__)
                emit("termination_error", (key,), category=getattr(error, "category", type(error).__name__),
                     http_status=getattr(error, "status", None))
            try:
                current = instance_state(HELPER_INSTANCE_ID, key, config["api_timeout_seconds"])
                if current["status"] == "terminated":
                    save(status="termination_confirmed", termination_confirmed=True)
                    emit("termination_confirmed", (key,), status="terminated")
                    return
            except (ApiError, OSError) as error:
                if getattr(error, "status", None) == 404 and state.get("termination_attempted"):
                    save(status="termination_confirmed", termination_confirmed=True)
                    emit("termination_confirmed", (key,), status="absent_after_request")
                    return
                emit("termination_status_error", (key,), category=getattr(error, "category", type(error).__name__),
                     http_status=getattr(error, "status", None))
            time.sleep(config["retry_seconds"])
        else:
            remaining = min(cutoff - time.monotonic(), (config["deadline"] - wall_now()).total_seconds())
            time.sleep(max(0.01, min(config["poll_seconds"], remaining)))


def self_test():
    import unittest
    from datetime import timedelta
    from unittest.mock import patch

    class Tests(unittest.TestCase):
        def config(self):
            root = "/lambda/nfs/synthetic/" + MAIN_INSTANCE_ID
            return validate_config(dict(instance_id=HELPER_INSTANCE_ID, main_instance_id=MAIN_INSTANCE_ID,
                                        credential_file="/synthetic/key", state_file="/synthetic/state",
                                        backup_root=root, receipt_file="/root/nas-backup/receipt.json"))

        def evidence(self):
            stamp = HARD_DEADLINE - timedelta(hours=1)
            files = {"runs/config.json": dict(kind="metadata", object_key="objects/" + "a" * 64, sha256="a" * 64, bytes=1),
                     "runs/best.pt": dict(kind="checkpoint", object_key="objects/" + "b" * 64, sha256="b" * 64, bytes=2)}
            manifest = dict(schema_version=1, instance_id=MAIN_INSTANCE_ID, created_utc=stamp.isoformat(),
                            destination="ssh://synthetic/persistent", finalization_status="running", final_report_sha256=None,
                            files=files)
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            receipt = dict(schema_version=1, instance_id=MAIN_INSTANCE_ID, completed_utc=stamp.isoformat(),
                           destination=manifest["destination"], verified=True, durable=True, backup_complete=True,
                           metadata_complete=True, retained_checkpoints_complete=True, artifact_count=len(files),
                           finalization_status="running", final_report_sha256=None, manifest_sha256=digest,
                           manifest_object_key="manifests/" + digest + ".json")
            return receipt, manifest, digest, stamp

        def test_exact_helper_mutation_only(self):
            self.assertEqual(helper_payload(HELPER_INSTANCE_ID), {"instance_ids": [HELPER_INSTANCE_ID]})
            for invalid in (MAIN_INSTANCE_ID, "other", None, [HELPER_INSTANCE_ID]):
                with self.assertRaises(ValueError):
                    helper_payload(invalid)

        def test_deadline_and_clock_rollback(self):
            start = HARD_DEADLINE - timedelta(seconds=30)
            cutoff = conservative_deadline(HARD_DEADLINE, start, 100)
            self.assertEqual(cutoff, 130)
            previous = dict(instance_id=HELPER_INSTANCE_ID, boot_id="same", monotonic_deadline=130)
            self.assertEqual(conservative_deadline(HARD_DEADLINE, start - timedelta(hours=1), 110, previous, "same"), 130)
            self.assertTrue(cutoff_reached(HARD_DEADLINE, cutoff, start - timedelta(hours=1), 130))

        def test_config_fixed_ids_and_cutoff(self):
            config = self.config()
            for changes in ({"instance_id": MAIN_INSTANCE_ID}, {"main_instance_id": HELPER_INSTANCE_ID},
                            {"deadline_utc": (HARD_DEADLINE + timedelta(seconds=1)).isoformat()},
                            {"backup_root": "/tmp"}, {"receipt_file": "/tmp/receipt"}, {"poll_seconds": 31}):
                with self.assertRaises(ValueError):
                    validate_config({**config, **changes})

        def test_never_early_with_live_main_or_missing_backup(self):
            for main_gone, verified in ((False, False), (False, True), (True, False)):
                self.assertIsNone(termination_reason(False, main_gone, verified))
            self.assertEqual(termination_reason(False, True, True), "main_terminated_and_durably_backed_up")
            self.assertEqual(termination_reason(True, False, False), "hard_deadline")
            for status in ("active", "terminating", "unknown", None):
                self.assertFalse(main_is_gone(status))
            self.assertTrue(main_is_gone("terminated"))
            self.assertTrue(main_is_gone(http_status=404))
            for http in (401, 403, 429, 500, None):
                self.assertFalse(main_is_gone(http_status=http))

        def test_matching_durable_receipt_and_manifest(self):
            receipt, manifest, digest, stamp = self.evidence()
            self.assertTrue(verified_backup(receipt, manifest, digest, stamp))
            for bad in ({}, None, "malformed", {**receipt, "manifest_sha256": "c" * 64},
                        {**receipt, "instance_id": HELPER_INSTANCE_ID}, {**receipt, "artifact_count": 0},
                        {**receipt, "completed_utc": (stamp - timedelta(days=2)).isoformat()}):
                self.assertFalse(verified_backup(bad, manifest, digest, stamp))
            for field in ("verified", "durable", "backup_complete", "metadata_complete", "retained_checkpoints_complete"):
                self.assertFalse(verified_backup({**receipt, field: False}, manifest, digest, stamp))
            self.assertFalse(verified_backup(receipt, {**manifest, "files": {}}, digest, stamp))
            self.assertFalse(verified_backup(receipt, {**manifest, "instance_id": HELPER_INSTANCE_ID}, digest, stamp))

        def test_missing_and_malformed_evidence(self):
            self.assertEqual(backup_evidence(self.config()), (False, None))
            with patch("pathlib.Path.resolve", autospec=True, side_effect=lambda path: path), \
                    patch("os.path.ismount", return_value=True), patch(__name__ + ".private_read", return_value="{malformed"):
                self.assertEqual(backup_evidence(self.config()), (False, None))

        def test_readonly_exact_preflight(self):
            with patch(__name__ + ".api_request", return_value={"id": HELPER_INSTANCE_ID, "status": "active"}) as call:
                self.assertEqual(instance_state(HELPER_INSTANCE_ID, "synthetic", 1)["status"], "active")
                self.assertEqual(call.call_args.args, ("instances/" + HELPER_INSTANCE_ID, "synthetic", 1))
            with patch(__name__ + ".api_request", return_value={"id": MAIN_INSTANCE_ID, "status": "active"}):
                with self.assertRaises(ApiError):
                    instance_state(HELPER_INSTANCE_ID, "synthetic", 1)

        def test_at_deadline_posts_only_helper_before_any_reads(self):
            events = []
            def request(endpoint, key, timeout, body=None):
                events.append((endpoint, body))
                return {}
            def status(instance_id, key, timeout):
                events.append(("status", instance_id))
                return {"id": instance_id, "status": "terminated"}
            with patch(__name__ + ".wall_now", return_value=HARD_DEADLINE), patch("time.monotonic", return_value=100), \
                    patch(__name__ + ".atomic_private_json"), patch(__name__ + ".emit"), \
                    patch(__name__ + ".api_request", side_effect=request), patch(__name__ + ".instance_state", side_effect=status), \
                    patch(__name__ + ".backup_evidence") as evidence:
                run(self.config(), "synthetic-not-a-credential")
                evidence.assert_not_called()
            self.assertEqual(events, [("instance-operations/terminate", {"instance_ids": [HELPER_INSTANCE_ID]}),
                                      ("status", HELPER_INSTANCE_ID)])

        def test_live_main_with_valid_backup_never_mutates(self):
            with patch(__name__ + ".wall_now", return_value=HARD_DEADLINE - timedelta(hours=1)), \
                    patch("time.monotonic", return_value=100), patch(__name__ + ".atomic_private_json"), \
                    patch(__name__ + ".emit"), patch(__name__ + ".api_request") as request, \
                    patch(__name__ + ".instance_state", return_value={"id": MAIN_INSTANCE_ID, "status": "active"}), \
                    patch(__name__ + ".backup_evidence", return_value=(True, "a" * 64)) as evidence, \
                    patch("time.sleep", side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    run(self.config(), "synthetic-not-a-credential")
                request.assert_not_called()
                evidence.assert_not_called()

        def test_secret_redaction(self):
            secret = "secret_SYNTHETIC-helper.nevervalid"
            self.assertNotIn(secret, redact("Bearer " + secret, (secret,)))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight", action="store_true", help="read-only exact helper and main instance check")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if args.config is None:
        parser.error("--config is required")
    key = None
    try:
        if os.geteuid() != 0:
            raise PermissionError("deployment must run as root")
        config = validate_config(json.loads(private_read(args.config)))
        key = private_read(config["credential_file"]).strip()
        if not key or any(character.isspace() for character in key):
            raise ValueError("credential file must contain one raw API key")
        if args.preflight:
            for instance_id in (HELPER_INSTANCE_ID, MAIN_INSTANCE_ID):
                try:
                    current = instance_state(instance_id, key, config["api_timeout_seconds"])
                except ApiError as error:
                    if error.status != 404 or instance_id != MAIN_INSTANCE_ID:
                        raise
                    current = {"id": instance_id, "status": "absent"}
                emit("preflight_verified", (key,), id=current["id"], status=current["status"])
            return
        run(config, key)
    except Exception as error:
        emit("watchdog_error", (key,) if key else (), category=getattr(error, "category", type(error).__name__),
             http_status=getattr(error, "status", None))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
