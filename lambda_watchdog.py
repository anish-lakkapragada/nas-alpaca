"""Local, independent deadline enforcement and artifact retrieval for our rental.

Run under caffeinate on the Mac. Never uploads the Lambda API credential. The
state file may contain only instances provisioned for this task. A final rsync
is attempted before termination; periodic copies protect intermediate results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

from lambda_cloud import request


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def checkpoint_paths(state, destination):
    """Only copy explicitly retained checkpoints, never arbitrary remote paths."""
    paths = set(state.get('backup_checkpoint_paths', []))
    search_root = state.get('active_search_relative_path')
    if search_root:
        manifest = destination / search_root / 'retained_checkpoints.json'
        if manifest.is_file():
            paths.update(json.loads(manifest.read_text()).get('relative_paths', []))
    safe = []
    for name in sorted(paths):
        if not isinstance(name, str) or '\n' in name or '\r' in name:
            raise ValueError('invalid checkpoint path')
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.pt' or path.parts[0] != 'cuda':
            raise ValueError('checkpoint must be a relative CUDA run .pt path')
        safe.append(str(path))
    return safe


def sync_instance(instance, state_dir, state):
    if instance.get('role') == 'durable_backup_helper':
        # The helper receives cloud-to-cloud copies on its persistent filesystem.
        # Its ID remains covered by this fallback's termination loop.
        return True
    host = instance.get('ip')
    if not host:
        return False
    key = state_dir / 'nas_ed25519'
    known = state_dir / 'known_hosts'
    destination = state_dir / 'artifacts' / instance['id']
    destination.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-i', str(key), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
           '-o', f'UserKnownHostsFile={known}']
    remote = f'ubuntu@{host}:/home/ubuntu/nas-cuda/runs/'
    # Every run's records and plots are retained. Bulk optimizer snapshots are
    # excluded so the laptop cannot fill up during a large architecture search.
    includes = ['--include=*/']
    for suffix in ('json', 'jsonl', 'log', 'csv', 'png', 'svg', 'md', 'npz', 'py', 'txt'):
        includes.append(f'--include=*.{suffix}')
    result = subprocess.run(['rsync', '-az', '--timeout=30', *includes, '--exclude=*',
                             '-e', shlex.join(ssh), remote, str(destination) + '/'],
                            capture_output=True, text=True, timeout=45)
    print(json.dumps({'event': 'sync', 'instance_id': instance['id'],
                      'returncode': result.returncode, 'at': datetime.now(timezone.utc).isoformat()}), flush=True)
    success = result.returncode == 0
    retained = checkpoint_paths(state, destination)
    if retained:
        listing = state_dir / f"checkpoint-files-{instance['id']}.txt"
        listing.write_text('\n'.join(retained) + '\n')
        listing.chmod(0o600)
        result = subprocess.run(['rsync', '-az', '--partial', '--timeout=30',
                                 f'--files-from={listing}', '-e', shlex.join(ssh),
                                 remote, str(destination) + '/'],
                                capture_output=True, text=True, timeout=90)
        print(json.dumps({'event': 'checkpoint_sync', 'instance_id': instance['id'],
                          'files': len(retained), 'returncode': result.returncode,
                          'at': datetime.now(timezone.utc).isoformat()}), flush=True)
        success = success and result.returncode == 0
    return success


def completed_and_backed_up(state, state_dir):
    relative = state.get('active_search_relative_path')
    if not relative:
        return False
    for instance in state['instances']:
        if instance.get('terminated') or instance.get('role') == 'durable_backup_helper':
            continue
        destination = state_dir / 'artifacts' / instance['id']
        search = destination / relative
        workflow = search.with_name(search.name + '-control') / 'workflow.json'
        report = search / 'FINAL_REPORT.json'
        manifest = search / 'retained_checkpoints.json'
        if not all(path.is_file() for path in (workflow, report, manifest)):
            return False
        if json.loads(workflow.read_text()).get('status') != 'completed':
            return False
        retained = json.loads(manifest.read_text())
        if not retained.get('checkpoints'):
            return False
        safe = set(checkpoint_paths(state, destination))
        for item in retained.get('checkpoints', []):
            relative_path = item['relative_path']
            if relative_path not in safe:
                return False
            path = destination / relative_path
            if not path.is_file() or path.stat().st_size != item['bytes']:
                return False
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != item['sha256']:
                return False
    return True


def save_state(path, state):
    temporary = path.with_suffix('.watchdog.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    while True:
        state = json.loads(args.state.read_text())
        deadline = utc(state['deadline_utc'])
        instances = [item for item in state['instances'] if not item.get('terminated')]
        if not instances:
            print('No active task instances; watchdog exiting.', flush=True)
            return
        seconds_left = (deadline - datetime.now(timezone.utc)).total_seconds()
        due = seconds_left <= 0
        # Leave the final three minutes to the cloud backup service. Starting a
        # blocking local transfer here could delay the exact rental cutoff.
        sync_ok = seconds_left > 180
        for instance in instances if sync_ok else []:
            try:
                sync_ok = sync_instance(instance, args.state.parent, state) and sync_ok
            except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                sync_ok = False
                print(json.dumps({'event': 'sync_error', 'instance_id': instance['id'],
                                  'error_type': type(exc).__name__,
                                  'at': datetime.now(timezone.utc).isoformat()}), flush=True)
        finished = False
        if sync_ok and state.get('local_early_termination_enabled', False):
            try:
                finished = completed_and_backed_up(state, args.state.parent)
            except (OSError, ValueError, KeyError):
                pass
        if due or datetime.now(timezone.utc) >= deadline or finished:
            # Termination is restricted to exact task IDs in the local rental record.
            ids = [item['id'] for item in instances]
            try:
                requested = [item['id'] for item in instances if not item.get('termination_requested')]
                if requested:
                    request('instance-operations/terminate', {'instance_ids': requested})
                    print(json.dumps({'event': 'termination_requested', 'instance_ids': requested,
                                      'reason': 'completed_and_backed_up' if finished else 'deadline'}), flush=True)
                    state = json.loads(args.state.read_text())
                    for item in state['instances']:
                        if item['id'] in requested:
                            item['termination_requested'] = True
                    state['status'] = 'termination_requested'
                    save_state(args.state, state)
                current = {item['id']: item.get('status') for item in request('instances')}
                state = json.loads(args.state.read_text())
                for item in state['instances']:
                    if item['id'] in ids and (item['id'] not in current or current[item['id']] == 'terminated'):
                        item['terminated'] = True
                if all(item.get('terminated') for item in state['instances']):
                    state['status'] = 'termination_confirmed'
                    save_state(args.state, state)
                    print(json.dumps({'event': 'termination_confirmed', 'instance_ids': ids}), flush=True)
                    return
                save_state(args.state, state)
                time.sleep(15)
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                print(json.dumps({'event': 'termination_error', 'error': str(exc)}), flush=True)
                time.sleep(15)
        else:
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            time.sleep(min(120 if remaining > 180 else 15, max(1, remaining)))


if __name__ == '__main__':
    main()
