"""Independent root watchdog for this task's one Lambda instance.

Deployment and credentials are deliberately external to this source file.
Run ``python lambda_remote_watchdog.py --self-test`` without credentials.
Run ``--config /root/nas/watchdog.json --preflight`` for a read-only exact-ID
credential check. Without --preflight, the daemon backs up periodically and
terminates only this task instance after verified final backup or at cutoff.

Private root-owned config (0600):
  instance_id, credential_file (one raw API key), search_root, receipt_file,
  backup_command (argv list, executable absolute), state_file,
  deadline_utc (optional; cannot exceed the hard deadline),
  backup_interval_seconds (default 120), backup_timeout_seconds (max 120),
  api_timeout_seconds (max 20), retry_seconds (default 10).

The backup command must atomically write a root-owned 0600 JSON receipt:
  schema_version=1, instance_id, completed_utc, destination (nonempty string),
  verified/durable/backup_complete/metadata_complete/retained_checkpoints_complete=true,
  finalization_status="complete", final_report_sha256=SHA256(FINAL_REPORT.json).
Nonfinal periodic receipts may use another finalization_status and a null hash.
Backup stdout/stderr are never copied into watchdog logs; the adapter owns any
private diagnostic log. No backup is allowed to delay the absolute cutoff.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time

ALLOWED_INSTANCE_ID = "5c07904125d442d8a43d41a1fd085ef3"
HARD_DEADLINE = datetime(2026, 11, 11, 16, 15, tzinfo=timezone.utc)
API = "https://cloud.lambda.com/api/v1/"


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(timezone.utc)


def wall_now():
    return datetime.now(timezone.utc)


def exact_instance(instance_id):
    if instance_id != ALLOWED_INSTANCE_ID:
        raise ValueError("instance ID is outside this watchdog's single-instance authorization")
    return instance_id


def termination_payload(instance_id):
    return {"instance_ids": [exact_instance(instance_id)]}


def redact(value, secrets=()):
    """Defense in depth; API/backup response bodies are never logged at all."""
    value = str(value)
    for secret in secrets:
        if secret:
            for variant in (secret, base64.b64encode((secret + ":").encode()).decode()):
                value = value.replace(variant, "[REDACTED]")
    value = re.sub(r"(?i)(Bearer|Basic)\s+[^\s\"']+", r"\1 [REDACTED]", value)
    value = re.sub(r"secret_[A-Za-z0-9_.-]+", "[REDACTED]", value)
    value = re.sub(r"(?i)(LAMBDA_API_KEY\s*=\s*)[^\s]+", r"\1[REDACTED]", value)
    return value


def emit(event, secrets=(), **details):
    record = {"event": event, "at_utc": wall_now().isoformat(), "instance_id": ALLOWED_INSTANCE_ID, **details}
    print(redact(json.dumps(record, allow_nan=False), secrets), flush=True)


def private_read(path):
    """Reject symlinks, other owners, and group/world-readable private files."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
            raise PermissionError("private file must be root-owned, regular, and mode 0600 or stricter")
        with os.fdopen(fd, "r") as stream:
            fd = None
            return stream.read()
    finally:
        if fd is not None:
            os.close(fd)


def atomic_private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def validate_config(config):
    exact_instance(config["instance_id"])
    deadline = utc(config.get("deadline_utc", HARD_DEADLINE.isoformat()))
    if deadline > HARD_DEADLINE:
        raise ValueError("configured cutoff exceeds the authorized hard deadline")
    result = dict(config, deadline=deadline)
    for key in ("credential_file", "search_root", "receipt_file", "state_file"):
        value = Path(config[key])
        if not value.is_absolute():
            raise ValueError("configuration paths must be absolute")
        result[key] = value
    command = config["backup_command"]
    if (not isinstance(command, list) or not command or not all(isinstance(v, str) and v and "\x00" not in v for v in command)
            or not Path(command[0]).is_absolute()):
        raise ValueError("backup_command must be an argv list with an absolute executable")
    for name, default, minimum, maximum in [("backup_interval_seconds", 120, 10, 600),
                                          ("backup_timeout_seconds", 120, 1, 120),
                                          ("api_timeout_seconds", 15, 1, 20),
                                          ("retry_seconds", 10, 1, 30)]:
        number = config.get(name, default)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not minimum <= number <= maximum:
            raise ValueError(f"invalid {name}")
        result[name] = float(number)
    if len({str(result[key]) for key in ("credential_file", "receipt_file", "state_file")}) != 3:
        raise ValueError("credential, receipt and state paths must be distinct")
    return result


def conservative_monotonic_deadline(deadline, wall, monotonic, prior=None, boot_id=None):
    cutoff = monotonic + max(0, (deadline - wall).total_seconds())
    if prior and prior.get("boot_id") == boot_id and prior.get("instance_id") == ALLOWED_INSTANCE_ID:
        previous = prior.get("monotonic_deadline")
        if isinstance(previous, (int, float)):
            cutoff = min(cutoff, previous)
    return cutoff


def cutoff_reached(deadline, monotonic_deadline, wall, monotonic):
    return wall >= deadline or monotonic >= monotonic_deadline


def verified_final_backup(receipt, finalization, report, report_sha256, backup_succeeded,
                          backup_started, current_time, max_age_seconds=180):
    """Pure early-shutdown predicate; missing/malformed evidence always denies it."""
    try:
        if backup_succeeded is not True or not all(isinstance(v, dict) for v in (receipt, finalization, report)):
            return False
        if receipt.get("schema_version") != 1 or receipt.get("instance_id") != ALLOWED_INSTANCE_ID:
            return False
        for key in ("verified", "durable", "backup_complete", "metadata_complete", "retained_checkpoints_complete"):
            if receipt.get(key) is not True:
                return False
        if (receipt.get("finalization_status") != "complete" or finalization.get("status") != "complete"
                or report.get("status") != "complete" or not receipt.get("destination")):
            return False
        if not isinstance(receipt["destination"], str):
            return False
        if not re.fullmatch(r"[a-f0-9]{64}", report_sha256):
            return False
        if receipt.get("final_report_sha256") != report_sha256 or finalization.get("final_report_sha256") != report_sha256:
            return False
        finished = utc(receipt["completed_utc"])
        return (-5 <= (current_time - finished).total_seconds() <= max_age_seconds
                and (finished - backup_started).total_seconds() >= -5)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def termination_reason(due, verified_backup):
    if due:
        return "hard_deadline"
    return "completed_and_durably_backed_up" if verified_backup else None


class ApiError(RuntimeError):
    def __init__(self, status=None, category="api_error"):
        self.status, self.category = status, category
        super().__init__(category)


def curl_configuration(endpoint, key, body=None):
    values = {"url": API + endpoint, "header": "Authorization: Bearer " + key}
    value = "".join(f"{name} = {json.dumps(item)}\n" for name, item in values.items())
    value += 'header = "Accept: application/json"\nheader = "Content-Type: application/json"\n'
    if body is not None:
        value += 'request = "POST"\ndata = ' + json.dumps(json.dumps(body)) + "\n"
    return value


def api_request(endpoint, key, timeout, body=None):
    # Auth is solely in stdin. API bodies and stderr are not exception messages.
    command = ["/usr/bin/curl", "--silent", "--show-error", "--proto", "=https", "--tlsv1.2",
               "--connect-timeout", str(min(5, timeout)), "--max-time", str(timeout),
               "--config", "-", "--write-out", "\n%{http_code}"]
    try:
        result = subprocess.run(command, input=curl_configuration(endpoint, key, body),
                                capture_output=True, text=True, timeout=timeout + 1)
    except subprocess.TimeoutExpired as error:
        raise ApiError(category="transport_timeout") from error
    text, _, status = result.stdout.rpartition("\n")
    http = int(status) if status.isdigit() else None
    if result.returncode:
        raise ApiError(http, "transport_failure")
    if http is None or not 200 <= http < 300:
        raise ApiError(http, "http_error")
    try:
        response = json.loads(text)
        if "error" in response or "data" not in response:
            raise ValueError("invalid response")
        return response["data"]
    except (ValueError, TypeError) as error:
        raise ApiError(http, "invalid_api_response") from error


def instance_state(key, timeout):
    data = api_request("instances/" + ALLOWED_INSTANCE_ID, key, timeout)
    if not isinstance(data, dict) or data.get("id") != ALLOWED_INSTANCE_ID:
        raise ApiError(category="instance_identity_mismatch")
    status = data.get("status")
    if not isinstance(status, str) or not re.fullmatch(r"[a-z_]{1,32}", status):
        raise ApiError(category="invalid_instance_state")
    return {"id": ALLOWED_INSTANCE_ID, "status": status}


def stop_backup_group(process):
    """Stop descendants even if the process-group leader has already exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    grace_deadline = time.monotonic() + 0.75
    while time.monotonic() < grace_deadline:
        # Reap the leader promptly; poll() does not tell us whether descendants
        # remain, so always check the original dedicated process group.
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.025, max(0, grace_deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def bounded_backup(config, seconds_available):
    """No shell; terminate the exact backup process group before cutoff."""
    if seconds_available <= 3:
        return False, "insufficient_time_before_cutoff"
    timeout = min(config["backup_timeout_seconds"], seconds_available - 2)
    env = dict(os.environ, NAS_BACKUP_INSTANCE_ID=ALLOWED_INSTANCE_ID,
               NAS_BACKUP_RECEIPT_FILE=str(config["receipt_file"]),
               NAS_BACKUP_DEADLINE_UTC=config["deadline"].isoformat())
    process = subprocess.Popen(config["backup_command"], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code != 0:
            # A CLI deadline can call os._exit while SSH/rsync children survive.
            # Killing the exact private group remains necessary after wait().
            stop_backup_group(process)
        return code == 0, "success" if code == 0 else "command_failure"
    except subprocess.TimeoutExpired:
        stop_backup_group(process)
        return False, "timeout"
    except BaseException:
        stop_backup_group(process)
        raise


def final_backup_evidence(config, succeeded, started):
    try:
        receipt = json.loads(private_read(config["receipt_file"]))
        search = config["search_root"]
        finalization = json.loads((search / "finalization.json").read_text())
        report_path = search / "FINAL_REPORT.json"
        report = json.loads(report_path.read_text())
        digest = file_hash(report_path)
        return verified_final_backup(receipt, finalization, report, digest, succeeded, started, wall_now()), receipt
    except (OSError, ValueError, TypeError):
        return False, None


def run(config, key):
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    previous = json.loads(private_read(config["state_file"])) if config["state_file"].exists() else {}
    if previous.get("instance_id") not in (None, ALLOWED_INSTANCE_ID):
        raise ValueError("watchdog state belongs to another instance")
    if previous.get("termination_confirmed") is True:
        emit("already_terminated", (key,))
        return
    cutoff = conservative_monotonic_deadline(config["deadline"], wall_now(), time.monotonic(), previous, boot_id)
    state = {**previous, "instance_id": ALLOWED_INSTANCE_ID, "boot_id": boot_id,
             "deadline_utc": config["deadline"].isoformat(), "monotonic_deadline": cutoff,
             "started_utc": wall_now().isoformat(), "pid": os.getpid()}
    def save(**updates):
        state.update(updates, updated_utc=wall_now().isoformat())
        atomic_private_json(config["state_file"], state)
    save(status="running")
    emit("armed", (key,), deadline_utc=config["deadline"].isoformat())
    next_backup = 0.0
    final_verified = False
    while True:
        mono, wall = time.monotonic(), wall_now()
        due = cutoff_reached(config["deadline"], cutoff, wall, mono)
        if not due and mono >= next_backup and not state.get("termination_requested"):
            started = wall_now()
            seconds_left = min(cutoff - time.monotonic(), (config["deadline"] - wall_now()).total_seconds())
            try:
                succeeded, outcome = bounded_backup(config, seconds_left)
            except OSError:
                succeeded, outcome = False, "command_start_failure"
            final_verified, receipt = final_backup_evidence(config, succeeded, started)
            save(last_backup_succeeded=succeeded, last_backup_outcome=outcome,
                 last_backup_utc=wall_now().isoformat(), final_backup_verified=final_verified,
                 receipt_sha256=file_hash(config["receipt_file"]) if receipt is not None else None)
            emit("backup", (key,), outcome=outcome, final_backup_verified=final_verified)
            next_backup = time.monotonic() + config["backup_interval_seconds"]
            due = cutoff_reached(config["deadline"], cutoff, wall_now(), time.monotonic())
        reason = termination_reason(due, final_verified)
        if reason or state.get("termination_requested"):
            if due and not state.get("deadline_logged"):
                save(deadline_logged=True)
                emit("deadline_cutoff", (key,), final_backup_verified=final_verified,
                     backup_failure_at_cutoff=not final_verified, last_backup_outcome=state.get("last_backup_outcome"))
            # Send the cutoff action first; a read-only status request must never
            # consume the network timeout before the first deadline POST.
            save(termination_attempted=True, termination_reason=reason or state.get("termination_reason"))
            try:
                api_request("instance-operations/terminate", key, config["api_timeout_seconds"],
                            termination_payload(config["instance_id"]))
                save(status="termination_requested", termination_requested=True,
                     termination_reason=reason or state.get("termination_reason"))
                emit("termination_requested", (key,), reason=state["termination_reason"])
            except (ApiError, OSError) as error:
                save(status="termination_retry", termination_last_error=type(error).__name__)
                emit("termination_error", (key,), category=getattr(error, "category", type(error).__name__),
                     http_status=getattr(error, "status", None))
            try:
                current = instance_state(key, config["api_timeout_seconds"])
                if current["status"] == "terminated":
                    save(status="termination_confirmed", termination_confirmed=True)
                    emit("termination_confirmed", (key,), status=current["status"])
                    return
            except ApiError as error:
                if error.status == 404 and state.get("termination_attempted"):
                    save(status="termination_confirmed", termination_confirmed=True)
                    emit("termination_confirmed", (key,), status="absent_after_request")
                    return
                emit("termination_status_error", (key,), category=error.category, http_status=error.status)
            time.sleep(config["retry_seconds"])
        else:
            remaining = min(cutoff - time.monotonic(), (config["deadline"] - wall_now()).total_seconds())
            time.sleep(max(0.01, min(1.0, next_backup - time.monotonic(), remaining)))


def self_test():
    import unittest
    from datetime import timedelta
    from unittest.mock import patch
    class Tests(unittest.TestCase):
        def config(self):
            return dict(instance_id=ALLOWED_INSTANCE_ID, credential_file="/synthetic/key", search_root="/synthetic/search",
                        receipt_file="/synthetic/receipt", state_file="/synthetic/state", backup_command=["/bin/false"])
        def evidence(self):
            stamp = HARD_DEADLINE - timedelta(hours=1)
            digest = "a" * 64
            receipt = dict(schema_version=1, instance_id=ALLOWED_INSTANCE_ID, completed_utc=stamp.isoformat(),
                           destination="s3://synthetic/backup", verified=True, durable=True, backup_complete=True,
                           metadata_complete=True, retained_checkpoints_complete=True,
                           finalization_status="complete", final_report_sha256=digest)
            return receipt, {"status": "complete", "final_report_sha256": digest}, {"status": "complete"}, digest, stamp
        def test_exact_instance(self):
            self.assertEqual(termination_payload(ALLOWED_INSTANCE_ID), {"instance_ids": [ALLOWED_INSTANCE_ID]})
            for bad in ["other", ALLOWED_INSTANCE_ID + "x", [ALLOWED_INSTANCE_ID], None]:
                with self.assertRaises(ValueError):
                    termination_payload(bad)
        def test_cutoff_and_clock_rollback(self):
            start = HARD_DEADLINE - timedelta(seconds=60)
            cutoff = conservative_monotonic_deadline(HARD_DEADLINE, start, 100)
            self.assertEqual(cutoff, 160)
            self.assertFalse(cutoff_reached(HARD_DEADLINE, cutoff, start, 159))
            self.assertTrue(cutoff_reached(HARD_DEADLINE, cutoff, start - timedelta(hours=1), 160))
            self.assertTrue(cutoff_reached(HARD_DEADLINE, cutoff, HARD_DEADLINE, 100))
            prior = dict(instance_id=ALLOWED_INSTANCE_ID, boot_id="same", monotonic_deadline=160)
            self.assertEqual(conservative_monotonic_deadline(HARD_DEADLINE, start - timedelta(hours=1), 120, prior, "same"), 160)
        def test_config_never_extends_deadline_or_backup_bound(self):
            self.assertEqual(validate_config(self.config())["deadline"], HARD_DEADLINE)
            for patch_values in [{"deadline_utc": (HARD_DEADLINE + timedelta(seconds=1)).isoformat()},
                                 {"instance_id": "another-instance"}, {"backup_timeout_seconds": 121},
                                 {"backup_command": "shell command"}, {"backup_command": ["relative-executable"]}]:
                with self.assertRaises(ValueError):
                    validate_config({**self.config(), **patch_values})
        def test_verified_receipt_only(self):
            receipt, state, report, digest, stamp = self.evidence()
            call = lambda r, success=True: verified_final_backup(r, state, report, digest, success, stamp, stamp)
            self.assertTrue(call(receipt))
            self.assertFalse(call(receipt, False))
            for bad in [None, {}, "bad", {**receipt, "durable": False}, {**receipt, "instance_id": "other"},
                        {**receipt, "final_report_sha256": "b" * 64}, {**receipt, "completed_utc": "invalid"},
                        {**receipt, "completed_utc": (stamp - timedelta(minutes=5)).isoformat()}]:
                self.assertFalse(call(bad))
            for field in ["verified", "backup_complete", "metadata_complete", "retained_checkpoints_complete"]:
                self.assertFalse(call({**receipt, field: False}))
            self.assertIsNone(termination_reason(False, False))
            self.assertEqual(termination_reason(False, True), "completed_and_durably_backed_up")
            self.assertEqual(termination_reason(True, False), "hard_deadline")
        def test_no_secret_in_arguments_or_errors(self):
            secret = "secret_SYNTHETIC-never-valid.123"
            encoded = base64.b64encode((secret + ":").encode()).decode()
            cleaned = redact(f"{secret} Bearer {secret} Basic {encoded} LAMBDA_API_KEY={secret}", (secret,))
            self.assertNotIn(secret, cleaned)
            self.assertNotIn(encoded, cleaned)
            fake = subprocess.CompletedProcess([], 0, 'sensitive untrusted response\n403', 'Authorization: Bearer ' + secret)
            with patch("subprocess.run", return_value=fake) as mocked:
                with self.assertRaises(ApiError) as caught:
                    api_request("instances/" + ALLOWED_INSTANCE_ID, secret, 1)
                self.assertNotIn(secret, str(caught.exception))
                self.assertNotIn("sensitive", str(caught.exception))
                self.assertNotIn(secret, " ".join(mocked.call_args.args[0]))
                self.assertIn(secret, mocked.call_args.kwargs["input"])
        def test_no_backup_past_cutoff(self):
            with patch("subprocess.Popen") as mocked:
                self.assertEqual(bounded_backup({}, 2), (False, "insufficient_time_before_cutoff"))
                mocked.assert_not_called()
        def test_failed_command_or_missing_malformed_receipt_never_ends_early(self):
            import sys
            config = validate_config(self.config())
            config["backup_command"] = [sys.executable, "-c", "raise SystemExit(7)"]
            succeeded, outcome = bounded_backup(config, 30)
            self.assertFalse(succeeded)
            self.assertEqual(outcome, "command_failure")
            receipt, state, report, digest, stamp = self.evidence()
            verified = verified_final_backup(receipt, state, report, digest, succeeded, stamp, stamp)
            self.assertIsNone(termination_reason(False, verified))
            self.assertEqual(final_backup_evidence(config, True, stamp), (False, None))
            with patch(__name__ + ".private_read", return_value="{malformed"):
                self.assertEqual(final_backup_evidence(config, True, stamp), (False, None))
        def test_abrupt_parent_exit_stops_actual_descendant_group(self):
            import sys
            import tempfile
            with tempfile.TemporaryDirectory() as temporary:
                heartbeat = Path(temporary) / "heartbeat"
                pidfile = Path(temporary) / "child.pid"
                child = ("import os,signal,time\nfrom pathlib import Path\n"
                         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                         f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
                         f"heartbeat=Path({str(heartbeat)!r})\n"
                         "while True:\n heartbeat.write_text(str(time.monotonic_ns()))\n time.sleep(.01)\n")
                parent = ("import os,subprocess,sys,time\nfrom pathlib import Path\n"
                          f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
                          f"while not Path({str(heartbeat)!r}).exists(): time.sleep(.01)\n"
                          "os._exit(124)\n")
                config = validate_config(self.config())
                config["backup_command"] = [sys.executable, "-c", parent]
                start = time.monotonic()
                try:
                    self.assertEqual(bounded_backup(config, 30), (False, "command_failure"))
                    self.assertLess(time.monotonic() - start, 2)
                    previous = heartbeat.read_text()
                    time.sleep(.1)
                    self.assertEqual(heartbeat.read_text(), previous)
                finally:
                    if pidfile.exists():
                        try:
                            os.kill(int(pidfile.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        def test_preflight_endpoint_is_read_only_and_identity_checked(self):
            with patch(__name__ + ".api_request", return_value={"id": ALLOWED_INSTANCE_ID, "status": "active"}) as request:
                self.assertEqual(instance_state("synthetic", 1), {"id": ALLOWED_INSTANCE_ID, "status": "active"})
                self.assertEqual(request.call_args.args, ("instances/" + ALLOWED_INSTANCE_ID, "synthetic", 1))
                self.assertFalse(request.call_args.kwargs)
            with patch(__name__ + ".api_request", return_value={"id": "different-instance", "status": "active"}):
                with self.assertRaises(ApiError):
                    instance_state("synthetic", 1)
        def test_deadline_post_precedes_status_and_backup_failure(self):
            events = []
            def post(endpoint, key, timeout, body=None):
                events.append((endpoint, body))
                return {}
            def status(key, timeout):
                events.append(("status", None))
                return {"id": ALLOWED_INSTANCE_ID, "status": "terminated"}
            with patch(__name__ + ".wall_now", return_value=HARD_DEADLINE), patch("time.monotonic", return_value=100), \
                 patch(__name__ + ".atomic_private_json"), patch(__name__ + ".emit"), \
                 patch(__name__ + ".api_request", side_effect=post), patch(__name__ + ".instance_state", side_effect=status), \
                 patch(__name__ + ".bounded_backup") as backup:
                run(validate_config(self.config()), "synthetic-not-a-credential")
                backup.assert_not_called()
            self.assertEqual(events[0], ("instance-operations/terminate", {"instance_ids": [ALLOWED_INSTANCE_ID]}))
            self.assertEqual(events[1][0], "status")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight", action="store_true", help="read-only exact-instance credential/API check")
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
        if any(key in value for value in config["backup_command"]):
            raise ValueError("credentials must never appear in backup command arguments")
        if args.preflight:
            current = instance_state(key, config["api_timeout_seconds"])
            emit("preflight_verified", (key,), id=current["id"], status=current["status"])
            return
        run(config, key)
    except Exception as error:
        emit("watchdog_error", (key,) if key else (), category=getattr(error, "category", type(error).__name__),
             http_status=getattr(error, "status", None))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
