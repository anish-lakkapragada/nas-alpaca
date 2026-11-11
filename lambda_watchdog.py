"""Local, independent deadline enforcement and artifact retrieval for our rental.

Run under caffeinate on the Mac. Never uploads the Lambda API credential. The
state file may contain only instances provisioned for this task. A final rsync
is attempted before termination; periodic copies protect intermediate results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

from lambda_cloud import request


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def sync_instance(instance, state_dir):
    host = instance.get('ip')
    if not host:
        return
    key = state_dir / 'nas_ed25519'
    known = state_dir / 'known_hosts'
    destination = state_dir / 'artifacts' / instance['id']
    destination.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-i', str(key), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
           '-o', f'UserKnownHostsFile={known}']
    result = subprocess.run(['rsync', '-az', '--timeout=45', '-e', shlex.join(ssh),
                             f'ubuntu@{host}:/home/ubuntu/nas-cuda/runs/', str(destination) + '/'],
                            capture_output=True, text=True, timeout=150)
    print(json.dumps({'event': 'sync', 'instance_id': instance['id'],
                      'returncode': result.returncode, 'at': datetime.now(timezone.utc).isoformat()}), flush=True)


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
        due = (deadline - datetime.now(timezone.utc)).total_seconds() <= 180
        for instance in instances:
            try:
                sync_instance(instance, args.state.parent)
            except (subprocess.TimeoutExpired, OSError) as exc:
                print(json.dumps({'event': 'sync_error', 'instance_id': instance['id'],
                                  'error_type': type(exc).__name__}), flush=True)
        if due or datetime.now(timezone.utc) >= deadline:
            # Termination is restricted to exact task IDs in the local rental record.
            ids = [item['id'] for item in instances]
            try:
                result = request('instance-operations/terminate', {'instance_ids': ids})
                print(json.dumps({'event': 'termination_requested', 'instance_ids': ids, 'result': result}), flush=True)
                for item in state['instances']:
                    if item['id'] in ids:
                        item['terminated'] = True
                state['status'] = 'termination_requested'
                args.state.write_text(json.dumps(state, indent=2) + '\n')
                return
            except RuntimeError as exc:
                print(json.dumps({'event': 'termination_error', 'error': str(exc)}), flush=True)
                time.sleep(15)
        else:
            time.sleep(min(120, max(1, (deadline - datetime.now(timezone.utc)).total_seconds() - 180)))


if __name__ == '__main__':
    main()
