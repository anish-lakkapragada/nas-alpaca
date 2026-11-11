"""Root-only verified backup to this task's persistent Lambda filesystem over SSH.

python lambda_filesystem_backup.py --config /root/filesystem-backup.json

Required JSON config: instance_id, remote_host (IP), remote_user, remote_root
(/lambda/nfs/FILESYSTEM/INSTANCE_ID), ssh_key_file, known_hosts_file, runs_root,
search_relative_path, receipt_file, cache_file, staging_dir. Optional workers
(1..4), historical_checkpoint_paths (<=8), source_root, max_staging_bytes
(defaults to and cannot exceed 120 GiB). Config/key/known_hosts are root-owned
regular mode0600 files. The destination parent must actually be a mount point.
Only immutable selected artifact snapshots cross SSH; datasets and credentials
are excluded. Remote verification hashes every required object each invocation.
The supervising watchdog must impose its own 120-second timeout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import stat
import subprocess
import tempfile
import threading
import time

from lambda_artifact_backup import atomic_json, collect, copy_snapshot, fingerprint, inside, private_json, relative, utc


MAX_STAGING_BYTES = 120 * 1024 ** 3

# Fixed code, quoted as one SSH command argument; all variable inputs use JSON
# stdin. Never execute any command or code taken from config or artifact names.
REMOTE_PROGRAM = r'''
import hashlib,json,os,re,stat,sys,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
request=json.load(sys.stdin)
root=Path(request['root'])
if not root.is_absolute() or '..' in root.parts or root.parts[1:3] != ('lambda','nfs'):
    raise ValueError('unsafe destination')
if not os.path.ismount(root.parent) or root.parent.resolve()!=root.parent:
    raise ValueError('destination filesystem is not mounted')
if root.is_symlink():
    raise ValueError('destination must not be a symlink')
root.mkdir(mode=0o755,exist_ok=True)
for name in ('objects','manifests'):
    folder=root/name
    if folder.is_symlink(): raise ValueError('object directory is a symlink')
    folder.mkdir(mode=0o755,exist_ok=True)
if request['action']=='prepare':
    print(json.dumps({'mounted':True,'root':str(root)}))
    sys.exit(0)
if request['action']=='receipt':
    receipt=request['receipt']
    digest=receipt['manifest_sha256']
    if not re.fullmatch(r'[a-f0-9]{64}',digest): raise ValueError('invalid manifest hash')
    key='manifests/'+digest+'.json'
    if receipt.get('manifest_object_key')!=key or receipt.get('instance_id')!=root.name:
        raise ValueError('receipt task or manifest mismatch')
    if not all(receipt.get(field) is True for field in ('verified','durable','backup_complete','metadata_complete','retained_checkpoints_complete')):
        raise ValueError('incomplete receipt')
    stored=(root/key).read_bytes()
    if hashlib.sha256(stored).hexdigest()!=digest: raise ValueError('manifest hash mismatch')
    manifest=json.loads(stored)
    if manifest.get('instance_id')!=root.name or any(manifest.get(field)!=receipt.get(field) for field in ('finalization_status','final_report_sha256','destination')):
        raise ValueError('receipt differs from verified manifest')
    receipt_path=Path('/root/nas-backup/receipt.json')
    receipt_path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='receipt-',dir=receipt_path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(receipt,stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,receipt_path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    print(json.dumps({'receipt_saved':True}))
    sys.exit(0)
def check(item):
    key=item['object_key']
    if not re.fullmatch(r'(objects/[a-f0-9]{64}|manifests/[a-f0-9]{64}\.json)',key):
        raise ValueError('invalid content address')
    path=root/key
    if path.is_symlink() or not path.is_file(): return key
    digest=hashlib.sha256()
    size=0
    with path.open('rb') as stream:
        while True:
            data=stream.read(8*1024*1024)
            if not data:break
            digest.update(data);size+=len(data)
    return None if size==item['bytes'] and digest.hexdigest()==item['sha256'] else key
with ThreadPoolExecutor(max_workers=4) as pool:
    failures=[value for value in pool.map(check,request['objects']) if value]
if failures:
    print(json.dumps({'verified':False,'failures':failures}))
    sys.exit(1)
manifest=request['manifest']
encoded=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
digest=hashlib.sha256(encoded).hexdigest()
key='manifests/'+digest+'.json'
fd,temporary=tempfile.mkstemp(prefix='snapshot-',dir=root/'manifests')
try:
    with os.fdopen(fd,'wb') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    os.chmod(temporary,0o644)
    os.replace(temporary,root/key)
finally:
    if os.path.exists(temporary):os.unlink(temporary)
if check({'object_key':key,'bytes':len(encoded),'sha256':digest}):
    raise ValueError('manifest verification failed')
print(json.dumps({'verified':True,'durable':True,'mounted':True,
                  'manifest_object_key':key,'manifest_sha256':digest,'objects_verified':len(request['objects'])}))
'''


def validate_config(config):
    required = {'instance_id', 'remote_host', 'remote_user', 'remote_root', 'ssh_key_file', 'known_hosts_file',
                'runs_root', 'search_relative_path', 'receipt_file', 'cache_file', 'staging_dir'}
    if required - config.keys():
        raise ValueError('filesystem backup config is missing required fields')
    config = dict(config)
    ipaddress.ip_address(config['remote_host'])
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', config['remote_user']):
        raise ValueError('invalid SSH user')
    instance = config['instance_id']
    if not isinstance(instance, str) or not re.fullmatch(r'[a-zA-Z0-9-]+', instance):
        raise ValueError('invalid instance ID')
    remote = PurePosixPath(config['remote_root'])
    if (not re.fullmatch(r'/lambda/nfs/[a-zA-Z0-9_-]+/[a-zA-Z0-9-]+', str(remote))
            or remote.name != instance):
        raise ValueError('remote_root must be the exact task directory on /lambda/nfs/FILESYSTEM')
    if not 1 <= config.get('workers', 4) <= 4:
        raise ValueError('workers must be between one and four')
    limit = config.get('max_staging_bytes', MAX_STAGING_BYTES)
    if not isinstance(limit, int) or not 1 <= limit <= MAX_STAGING_BYTES:
        raise ValueError('staging limit must be between one byte and 120 GiB')
    for name in ('ssh_key_file', 'known_hosts_file', 'runs_root', 'receipt_file', 'cache_file', 'staging_dir'):
        if not Path(config[name]).is_absolute():
            raise ValueError('local configuration paths must be absolute')
    private_paths = [str(Path(config[name]).resolve()) for name in ('ssh_key_file', 'known_hosts_file', 'receipt_file', 'cache_file')]
    if len(set(private_paths)) != len(private_paths):
        raise ValueError('private key, hosts, receipt, and cache paths must be distinct')
    search = relative(config['search_relative_path'])
    if search.parts[0] != 'cuda' or len(search.parts) < 2:
        raise ValueError('search must be under runs/cuda')
    historical = config.get('historical_checkpoint_paths', [])
    if not isinstance(historical, list) or len(historical) > 8:
        raise ValueError('at most eight historical checkpoints may be configured')
    from lambda_artifact_backup import checkpoint_path
    config['credential_file'] = config['ssh_key_file']  # Shared selector excludes this exact private path.
    for name in historical:
        checkpoint_path(config, name)
    return config


def private_file(path):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('SSH key and known_hosts must be root-owned regular files with mode0600')


class SSHTransport:
    def __init__(self, config, deadline):
        self.config, self.deadline = config, deadline
        private_file(config['ssh_key_file'])
        private_file(config['known_hosts_file'])
        self.ssh = ['ssh', '-i', config['ssh_key_file'], '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + config['known_hosts_file'],
                    '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
                    '-o', 'ControlMaster=auto', '-o', 'ControlPersist=90',
                    '-o', 'ControlPath=' + str(Path(config['staging_dir']) / 'ssh-control')]
        self.host = config['remote_user'] + '@' + config['remote_host']

    def command(self, argv, data=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('filesystem backup time budget exhausted')
        result = subprocess.run(argv, input=data, text=True, capture_output=True, timeout=remaining)
        if result.returncode:
            raise RuntimeError('backup transport failed with exit code ' + str(result.returncode))
        return result.stdout

    def request(self, value):
        value = dict(value, root=self.config['remote_root'])
        command = ('sudo -n ' if value['action'] == 'receipt' else '') + 'python3 -c ' + shlex.quote(REMOTE_PROGRAM)
        return json.loads(self.command([*self.ssh, self.host, command], json.dumps(value)))

    def prepare(self):
        result = self.request({'action': 'prepare'})
        if result.get('mounted') is not True:
            raise ValueError('backup destination is not a mounted filesystem')

    def transfer(self, staging, names):
        listing = staging / 'transfer-files.txt'
        listing.write_text(''.join(name + '\n' for name in sorted(names)))
        listing.chmod(0o600)
        command = ['rsync', '-a', '--protect-args', '--no-owner', '--no-group', '--checksum', '--timeout=30',
                   '--chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r', '--files-from=' + str(listing), '-e', shlex.join(self.ssh),
                   str(staging) + '/', self.host + ':' + self.config['remote_root'] + '/']
        self.command(command)

    def verify(self, objects, manifest):
        result = self.request({'action': 'verify', 'objects': objects, 'manifest': manifest})
        if result.get('verified') is not True or result.get('durable') is not True or result.get('mounted') is not True:
            raise ValueError('remote artifact verification failed')
        return result

    def save_receipt(self, receipt):
        if self.request({'action': 'receipt', 'receipt': receipt}).get('receipt_saved') is not True:
            raise ValueError('helper receipt was not saved')


class FilesystemBackup:
    def __init__(self, config, transport_factory=SSHTransport, max_seconds=110):
        self.config = validate_config(config)
        self.deadline = time.monotonic() + max_seconds
        self.transport_factory = transport_factory
        self.lock = threading.Lock()
        self.staging = Path(config['staging_dir'])
        if self.staging.is_symlink():
            raise ValueError('staging directory must not be a symlink')
        self.staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.staging.chmod(0o700)
        (self.staging / 'objects').mkdir(mode=0o700, exist_ok=True)
        if (self.staging / 'objects').is_symlink():
            raise ValueError('staging objects must not be a symlink')
        self.used = sum(path.stat().st_size for path in self.staging.rglob('*') if path.is_file())
        self.limit = config.get('max_staging_bytes', MAX_STAGING_BYTES)
        if self.used > self.limit:
            raise ValueError('existing staging artifacts exceed the configured cap')
        self.destination = 'ssh://' + config['remote_user'] + '@' + config['remote_host'] + config['remote_root']
        self.cache = {'schema_version': 1, 'destination': self.destination, 'files': {}}
        try:
            previous = json.loads(Path(config['cache_file']).read_text())
            if previous.get('schema_version') == 1 and previous.get('destination') == self.destination:
                self.cache = previous
        except (FileNotFoundError, ValueError):
            pass

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('filesystem backup time budget exhausted')

    def snapshot(self, name, item):
        self.check_time()
        old = self.cache['files'].get(name)
        expected = item.get('expected')
        if old and old.get('fingerprint') == item['fingerprint'] and (not expected or (old['sha256'], old['bytes']) == expected):
            path = self.staging / old['object_key']
            if path.is_file() and not path.is_symlink() and path.stat().st_size == old['bytes']:
                actual = hashlib.sha256()
                with path.open('rb') as stream:
                    while True:
                        self.check_time()
                        block = stream.read(8 * 1024 * 1024)
                        if not block:
                            break
                        actual.update(block)
                if actual.hexdigest() == old['sha256'] and fingerprint(item['path']) == item['fingerprint']:
                    return dict(old, kind=item['kind'])
        reserved = 0
        def allocate(size):
            nonlocal reserved
            with self.lock:
                if self.used + size - reserved > self.limit:
                    raise ValueError('staging growth would exceed configured cap')
                self.used += size - reserved
                reserved = size
        fd, temporary = tempfile.mkstemp(prefix='snapshot-', dir=self.staging)
        try:
            with os.fdopen(fd, 'wb') as output:
                copied = copy_snapshot(item, output, self.check_time, allocate)
                output.flush()
                os.fsync(output.fileno())
            key = 'objects/' + copied['sha256']
            with self.lock:
                target = self.staging / key
                previous_size = target.stat().st_size if target.exists() else 0
                os.replace(temporary, target)
                target.chmod(0o400)
                self.used -= previous_size
                result = dict(copied, object_key=key, kind=item['kind'])
                self.cache['files'][name] = result
                atomic_json(self.config['cache_file'], self.cache)
            return result
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
                with self.lock:
                    self.used -= reserved

    def run(self):
        receipt = {'schema_version': 1, 'instance_id': self.config['instance_id'], 'started_utc': utc(), 'completed_utc': None,
                   'verified': False, 'durable': False, 'backup_complete': False, 'metadata_complete': False,
                   'retained_checkpoints_complete': False, 'destination': self.destination,
                   'finalization_status': 'running', 'final_report_sha256': None}
        atomic_json(self.config['receipt_file'], receipt)
        try:
            artifacts, snapshots = collect(self.config), {}
            with ThreadPoolExecutor(max_workers=self.config.get('workers', 4)) as executor:
                jobs = {executor.submit(self.snapshot, name, item): name for name, item in artifacts.items()}
                for job in as_completed(jobs):
                    snapshots[jobs[job]] = job.result()
            self.check_time()
            root = Path(self.config['runs_root']).resolve()
            search = inside(root, self.config['search_relative_path'])
            final_path, control_path = search / 'finalization.json', search.with_name(search.name + '-control') / 'workflow.json'
            final = json.loads(final_path.read_text()) if final_path.is_file() else {}
            control = json.loads(control_path.read_text()) if control_path.is_file() else {}
            complete = final.get('status') == 'complete' and control.get('status') == 'completed'
            if complete:
                report = snapshots.get('runs/' + self.config['search_relative_path'] + '/FINAL_REPORT.json')
                if not report or report['sha256'] != final.get('final_report_sha256'):
                    raise ValueError('completed final report hash does not match finalization')
                if not any(item['kind'] == 'checkpoint' for item in snapshots.values()):
                    raise ValueError('completed search has no retained checkpoint')
                receipt.update(finalization_status='complete', final_report_sha256=report['sha256'])
            manifest = {'schema_version': 1, 'instance_id': self.config['instance_id'], 'created_utc': utc(),
                        'destination': self.destination, 'finalization_status': receipt['finalization_status'],
                        'final_report_sha256': receipt['final_report_sha256'], 'files': dict(sorted(snapshots.items()))}
            transport = self.transport_factory(self.config, self.deadline)
            transport.prepare()
            unique = {item['object_key']: item for item in snapshots.values()}
            transport.transfer(self.staging, unique)
            verified = transport.verify(list(unique.values()), manifest)
            expected_manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if verified.get('manifest_sha256') != expected_manifest_hash or verified.get('objects_verified') != len(unique):
                raise ValueError('remote manifest verification does not match this snapshot')
            if complete:
                latest = collect(self.config)
                if set(latest) != set(artifacts) or any(latest[name]['fingerprint'] != artifacts[name]['fingerprint'] for name in artifacts):
                    raise ValueError('final artifacts changed during backup; retry before termination')
            self.check_time()
            receipt.update(completed_utc=utc(), verified=True, durable=True, backup_complete=True,
                           metadata_complete=True, retained_checkpoints_complete=True,
                           manifest_object_key=verified['manifest_object_key'], manifest_sha256=expected_manifest_hash,
                           artifact_count=len(snapshots), total_bytes=sum(item['bytes'] for item in unique.values()))
            transport.save_receipt(receipt)
            atomic_json(self.config['receipt_file'], receipt)
            return receipt
        except Exception as error:
            receipt.update(completed_utc=utc(), error_type=type(error).__name__)
            atomic_json(self.config['receipt_file'], receipt)
            raise


def run_backup(config, transport_factory=SSHTransport, max_seconds=110):
    return FilesystemBackup(config, transport_factory, max_seconds).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=110)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('root execution is required')
    if not 1 <= args.max_seconds <= 115:
        parser.error('--max-seconds must be in 1..115')
    def expired(signum, frame):
        os.write(2, b'backup deadline exceeded; staged snapshots retained\n')
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        config = private_json(args.config)
        config['_config_file'] = str(args.config.resolve())
        print(json.dumps(run_backup(config, max_seconds=args.max_seconds)), flush=True)
    except Exception as error:
        print(json.dumps({'backup_complete': False, 'error_type': type(error).__name__}), flush=True)
        return 1
    finally:
        signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
