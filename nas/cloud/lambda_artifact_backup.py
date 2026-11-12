"""Verified, resumable artifact backups to a Lambda filesystem S3 adapter.

Run as root: python lambda_artifact_backup.py --config /root/backup-config.json
Both the configuration and credential_file must be root-owned mode 0600 JSON.
Credentials contain only access_key_id and secret_access_key. No AWS credential
provider chain is used. A watchdog must also enforce a 120-second process limit.
The CLI exits itself after at most 115 seconds; verified progress is atomic.

Required config: endpoint_url, region, bucket, prefix (containing instance_id as
one path component), credential_file, runs_root, search_relative_path,
receipt_file, cache_file, instance_id. Optional: historical_checkpoint_paths
(at most eight paths relative to runs_root), source_root (defaults to the parent
of runs_root), workers (1..4). Source files are read from the root and the
explicit nas/, tests/, scripts/, and docs/ directories; secrets stay excluded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import tempfile
import threading
import time
from urllib.parse import urlsplit


METADATA_SUFFIXES = {'.json', '.jsonl', '.log', '.csv', '.png', '.svg', '.md', '.npz', '.py', '.txt'}
SOURCE_SUFFIXES = {'.py', '.md', '.txt', '.sh', '.toml'}
BLOCKED_DIRS = {'data', 'dataset', 'datasets', '.git', '.ssh', '.aws', '.lambda', '__pycache__'}
DATASET_NAMES = {'train.json', 'validation.json', 'tokenizer.json', 'train.parquet', 'validation.parquet'}
SECRET_NAME = re.compile(r'(credential|secret|password|private[-_]?key|access[-_]?key|api[-_]?key)', re.I)
SHA256 = re.compile(r'^[a-f0-9]{64}$')


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def private_json(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('private JSON must be a root-owned regular file with mode 0600')
    with path.open() as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError('private JSON must contain an object')
    return result


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('invalid artifact relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '.' in value.split('/') or '//' in value:
        raise ValueError('artifact path traversal is forbidden')
    return path


def permitted(path):
    return not any(part in BLOCKED_DIRS or part in DATASET_NAMES or part.startswith('.env') or SECRET_NAME.search(part) for part in path.parts)


def inside(root, value):
    rel = relative(value)
    if not permitted(rel):
        raise ValueError('artifact path names identify datasets or secrets')
    path = root.joinpath(*rel.parts)
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError('artifact symlinks are forbidden')
    path.resolve().relative_to(root.resolve())
    return path


def validate_config(config):
    required = {'endpoint_url', 'region', 'bucket', 'prefix', 'credential_file', 'runs_root',
                'search_relative_path', 'receipt_file', 'cache_file', 'instance_id'}
    if required - config.keys():
        raise ValueError('backup configuration is missing required fields')
    parsed = urlsplit(config['endpoint_url'])
    if (parsed.scheme != 'https' or not re.fullmatch(r'files\.[a-z0-9.-]+\.lambda\.ai', parsed.hostname or '')
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('endpoint must be an HTTPS files.*.lambda.ai origin')
    instance = config['instance_id']
    prefix = relative(config['prefix'])
    if not isinstance(instance, str) or not re.fullmatch(r'[a-zA-Z0-9-]+', instance) or instance not in prefix.parts:
        raise ValueError('backup prefix must contain the exact task instance ID as a path component')
    if not isinstance(config['bucket'], str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', config['bucket']):
        raise ValueError('invalid bucket')
    search = relative(config['search_relative_path'])
    if search.parts[0] != 'cuda' or len(search.parts) < 2 or not permitted(search):
        raise ValueError('search must be inside runs/cuda')
    if not 1 <= config.get('workers', 4) <= 4:
        raise ValueError('workers must be between 1 and 4')
    historical = config.get('historical_checkpoint_paths', [])
    if not isinstance(historical, list) or len(historical) > 8:
        raise ValueError('at most eight historical checkpoints may be configured')
    for name in historical:
        checkpoint_path(config, name)
    return config


def make_client(config):
    """Both explicit fields are required, so boto3 never discovers credentials."""
    credentials = private_json(config['credential_file'])
    if not all(isinstance(credentials.get(key), str) and credentials[key] for key in ('access_key_id', 'secret_access_key')):
        raise ValueError('credential file requires explicit access_key_id and secret_access_key')
    import boto3
    from botocore.config import Config
    return boto3.client('s3', endpoint_url=config['endpoint_url'], region_name=config['region'],
                        aws_access_key_id=credentials['access_key_id'], aws_secret_access_key=credentials['secret_access_key'],
                        aws_session_token=None,
                        config=Config(signature_version='s3v4', connect_timeout=3, read_timeout=15,
                                      retries={'total_max_attempts': 2}, s3={'addressing_style': 'path'},
                                      max_pool_connections=4))


def checkpoint_path(config, value):
    path = relative(value)
    if path.parts[0] != 'cuda' or path.suffix != '.pt':
        raise ValueError('checkpoint must be a relative cuda/*.pt path')
    return inside(Path(config['runs_root']).resolve(), value)


def fingerprint(path):
    return stat_fingerprint(path.stat(follow_symlinks=False))


def stat_fingerprint(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('artifact is not a regular file')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def copy_snapshot(item, output, check_time, allocate=None):
    """Copy stable files or a bounded append-log prefix; retry atomic replacements.

    The cache records the opened-file fingerprint, never a later fingerprint for
    bytes absent from the snapshot. Checkpoints require the original collected
    identity and manifest hash. Final receipts additionally recheck the complete
    collected artifact set, so a prefix snapshot cannot qualify as final.
    """
    path = item['path']
    checkpoint = item['kind'] == 'checkpoint'
    append_log = path.suffix in ('.jsonl', '.log') and not checkpoint
    for attempt in range(3):
        check_time()
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as source:
                before = stat_fingerprint(os.fstat(source.fileno()))
                if checkpoint and before != item['fingerprint']:
                    raise ValueError('retained checkpoint changed before snapshotting')
                if allocate:
                    allocate(before[2])
                output.seek(0)
                output.truncate()
                digest, size = hashlib.sha256(), 0
                while size < before[2]:
                    check_time()
                    block = source.read(min(8 * 1024 * 1024, before[2] - size))
                    if not block:
                        break
                    digest.update(block)
                    output.write(block)
                    size += len(block)
                after_fd = stat_fingerprint(os.fstat(source.fileno()))
            after_path = fingerprint(path)
            stable = before == after_fd == after_path and size == before[2]
            grew = (append_log and size == before[2] and before[:2] == after_fd[:2] == after_path[:2]
                    and before[2] <= after_fd[2] <= after_path[2] and after_path[2] > before[2])
            if not stable and not grew:
                if checkpoint:
                    raise ValueError('retained checkpoint changed while snapshotting')
                if attempt < 2:
                    continue
                raise ValueError('artifact changed during all three snapshot attempts')
            sha = digest.hexdigest()
            expected = item.get('expected')
            if expected and (sha, size) != expected:
                raise ValueError('checkpoint does not match retained manifest SHA256/size')
            return {'sha256': sha, 'bytes': size, 'fingerprint': before, 'prefix_snapshot': not stable}
        except FileNotFoundError:
            if checkpoint or attempt == 2:
                raise
    raise ValueError('snapshot retries exhausted')


def collect(config):
    root = Path(config['runs_root']).resolve()
    search = inside(root, config['search_relative_path'])
    manifest_path = inside(root, config['search_relative_path'] + '/retained_checkpoints.json')
    manifest = json.loads(manifest_path.read_text())
    items = manifest.get('checkpoints', [])
    expected = {}
    for item in items:
        name, digest, size = item['relative_path'], item['sha256'], item['bytes']
        checkpoint_path(config, name)
        if not isinstance(digest, str) or not SHA256.fullmatch(digest) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('invalid retained checkpoint hash or size')
        if name in expected:
            raise ValueError('duplicate retained checkpoint')
        expected[name] = (digest, size)
    selected = manifest.get('relative_paths', list(expected))
    if not isinstance(selected, list) or set(selected) != set(expected):
        raise ValueError('retained checkpoint lists disagree or hashes are absent')
    forbidden = {Path(config[key]).resolve() for key in ('credential_file', 'cache_file', 'receipt_file')}
    if config.get('_config_file'):
        forbidden.add(Path(config['_config_file']).resolve())
    artifacts = {}
    for directory, folders, files in os.walk(inside(root, 'cuda'), followlinks=False):
        folders[:] = sorted(name for name in folders if permitted(PurePosixPath(name)) and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.suffix not in METADATA_SUFFIXES or not permitted(PurePosixPath(rel)) or path.is_symlink() or path.resolve() in forbidden:
                continue
            artifacts['runs/' + rel] = {'path': path, 'kind': 'metadata', 'fingerprint': fingerprint(path)}
    for name in sorted(set(selected) | set(config.get('historical_checkpoint_paths', []))):
        path = checkpoint_path(config, name)
        if path.resolve() in forbidden:
            raise ValueError('checkpoint overlaps a private file')
        artifacts['runs/' + name] = {'path': path, 'kind': 'checkpoint', 'fingerprint': fingerprint(path),
                                     'expected': expected.get(name)}
    source = Path(config.get('source_root', root.parent)).resolve()
    source_files = list(source.iterdir())
    for folder in ('nas', 'tests', 'scripts', 'docs'):
        directory = source / folder
        if not directory.is_dir() or directory.is_symlink():
            continue
        for current, folders, files in os.walk(directory, followlinks=False):
            folders[:] = sorted(name for name in folders
                                if permitted(PurePosixPath(name)) and not (Path(current) / name).is_symlink())
            source_files.extend(Path(current) / name for name in files)
    for path in sorted(source_files):
        relative_source = path.relative_to(source).as_posix()
        if (path.suffix in SOURCE_SUFFIXES and path.is_file() and not path.is_symlink()
                and permitted(PurePosixPath(relative_source)) and path.resolve() not in forbidden):
            artifacts['source/' + relative_source] = {'path': path, 'kind': 'source', 'fingerprint': fingerprint(path)}
    return artifacts


def destination(config):
    return config['endpoint_url'].rstrip('/') + '/' + config['bucket'] + '/' + config['prefix'].strip('/')


class Backup:
    def __init__(self, config, client, max_seconds=110):
        self.config, self.client = validate_config(config), client
        self.end = time.monotonic() + max_seconds
        self.lock = threading.Lock()
        self.cache = {'schema_version': 1, 'destination': destination(config), 'files': {}}
        try:
            previous = json.loads(Path(config['cache_file']).read_text())
            if previous.get('schema_version') == 1 and previous.get('destination') == self.cache['destination']:
                self.cache = previous
        except (FileNotFoundError, ValueError):
            pass

    def check_time(self):
        if time.monotonic() >= self.end:
            raise TimeoutError('backup invocation time budget exhausted')

    def head_matches(self, item):
        self.check_time()
        try:
            head = self.client.head_object(Bucket=self.config['bucket'], Key=item['object_key'])
        except Exception:
            return False
        return (head.get('ContentLength') == item['bytes'] and head.get('ETag') == item.get('etag')
                and head.get('Metadata', {}).get('sha256') == item['sha256'])

    def upload(self, stream, digest, size, key):
        self.check_time()
        stream.seek(0)
        self.client.put_object(Bucket=self.config['bucket'], Key=key, Body=stream, ContentLength=size,
                               Metadata={'sha256': digest})
        self.check_time()
        received = self.client.get_object(Bucket=self.config['bucket'], Key=key)
        body, actual, count = received['Body'], hashlib.sha256(), 0
        try:
            while True:
                self.check_time()
                block = body.read(8 * 1024 * 1024)
                if not block:
                    break
                actual.update(block)
                count += len(block)
        finally:
            body.close()
        if count != size or actual.hexdigest() != digest or received.get('ContentLength') != size:
            raise ValueError('remote object failed SHA256/size verification')
        result = {'object_key': key, 'sha256': digest, 'bytes': size, 'etag': received.get('ETag'), 'verified_utc': utc()}
        if not result['etag'] or not self.head_matches(result):
            raise ValueError('remote object HEAD does not match verified bytes')
        return result

    def artifact(self, name, item):
        self.check_time()
        old = self.cache['files'].get(name)
        expected = item.get('expected')
        if old and old.get('fingerprint') == item['fingerprint'] and (not expected or (old['sha256'], old['bytes']) == expected):
            if fingerprint(item['path']) == item['fingerprint'] and self.head_matches(old):
                return dict(old, kind=item['kind'])
        with tempfile.TemporaryFile() as snapshot:
            copied = copy_snapshot(item, snapshot, self.check_time)
            key = self.config['prefix'].strip('/') + '/objects/' + copied['sha256']
            result = self.upload(snapshot, copied['sha256'], copied['bytes'], key)
        result.update(copied, kind=item['kind'])
        with self.lock:
            self.cache['files'][name] = result
            atomic_json(self.config['cache_file'], self.cache)
        return result

    def run(self):
        receipt = {'schema_version': 1, 'instance_id': self.config['instance_id'], 'started_utc': utc(),
                   'completed_utc': None, 'verified': False, 'durable': False, 'backup_complete': False,
                   'metadata_complete': False, 'retained_checkpoints_complete': False,
                   'destination': destination(self.config), 'finalization_status': 'running', 'final_report_sha256': None}
        atomic_json(self.config['receipt_file'], receipt)
        try:
            artifacts, verified = collect(self.config), {}
            with ThreadPoolExecutor(max_workers=self.config.get('workers', 4)) as executor:
                futures = {executor.submit(self.artifact, name, item): name for name, item in artifacts.items()}
                for future in as_completed(futures):
                    verified[futures[future]] = future.result()
            self.check_time()
            root = Path(self.config['runs_root']).resolve()
            search = inside(root, self.config['search_relative_path'])
            finalization, workflow = {}, {}
            if (search / 'finalization.json').is_file():
                finalization = json.loads((search / 'finalization.json').read_text())
            control = search.with_name(search.name + '-control') / 'workflow.json'
            if control.is_file():
                workflow = json.loads(control.read_text())
            complete = finalization.get('status') == 'complete' and workflow.get('status') == 'completed'
            if complete:
                latest = collect(self.config)
                if set(latest) != set(artifacts) or any(latest[name]['fingerprint'] != artifacts[name]['fingerprint'] for name in artifacts):
                    raise ValueError('final artifacts changed during backup; retry before termination')
                report_name = 'runs/' + self.config['search_relative_path'] + '/FINAL_REPORT.json'
                report = verified.get(report_name)
                if not report or report['sha256'] != finalization.get('final_report_sha256'):
                    raise ValueError('final report hash does not match completed finalization')
                if not any(item['kind'] == 'checkpoint' for item in verified.values()):
                    raise ValueError('completed search must retain checkpoints')
                receipt.update(finalization_status='complete', final_report_sha256=report['sha256'])
            manifest = {'schema_version': 1, 'instance_id': self.config['instance_id'], 'created_utc': utc(),
                        'destination': destination(self.config), 'finalization_status': receipt['finalization_status'],
                        'final_report_sha256': receipt['final_report_sha256'], 'files': dict(sorted(verified.items()))}
            encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
            sha = hashlib.sha256(encoded).hexdigest()
            with tempfile.TemporaryFile() as snapshot:
                snapshot.write(encoded)
                saved = self.upload(snapshot, sha, len(encoded), self.config['prefix'].strip('/') + '/manifests/' + sha + '.json')
            receipt.update(completed_utc=utc(), verified=True, durable=True, backup_complete=True,
                           metadata_complete=True, retained_checkpoints_complete=True,
                           manifest_object_key=saved['object_key'], manifest_sha256=sha,
                           artifact_count=len(verified), total_bytes=sum(item['bytes'] for item in verified.values()))
            atomic_json(self.config['receipt_file'], receipt)
            return receipt
        except Exception as error:
            receipt.update(completed_utc=utc(), error_type=type(error).__name__)
            atomic_json(self.config['receipt_file'], receipt)
            raise


def run_backup(config, client, max_seconds=110):
    return Backup(config, client, max_seconds).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=110)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 115:
        parser.error('--max-seconds must be in 1..115; watchdog limit is 120 seconds')
    if os.geteuid() != 0:
        parser.error('root execution is required for private configuration and receipts')
    def expired(signum, frame):
        os.write(2, b'backup invocation deadline exceeded; verified cache retained\n')
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        config = validate_config(private_json(args.config))
        config['_config_file'] = str(args.config.resolve())
        result = run_backup(config, make_client(config), args.max_seconds)
        print(json.dumps(result), flush=True)
    except Exception as error:
        # Do not print exception text from credential or HTTP libraries.
        print(json.dumps({'backup_complete': False, 'error_type': type(error).__name__}), flush=True)
        return 1
    finally:
        signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
