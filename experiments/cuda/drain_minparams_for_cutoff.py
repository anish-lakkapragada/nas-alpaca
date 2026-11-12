"""One-off, bounded operational restart; never edits frozen experiment sources.

Run as the same ubuntu user in an independent systemd service with
KillMode=process. Requires the workflow unit's ExecStart to already contain the
new cutoff. The helper itself never signals trainers or changes systemd units.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def process(pid):
    """Start ticks distinguish reused PIDs; cmdline is empty for zombies."""
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                    uid=folder.stat().st_uid,
                    argv=[x.decode() for x in (folder / 'cmdline').read_bytes().split(b'\0') if x])
    except FileNotFoundError:
        return None


def argument(argv, name):
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i+1]
        if item.startswith(name + '='):
            return item.split('=', 1)[1]
    return None


def validate_identity(info, script, root):
    if (not info or info['uid'] != os.getuid() or info['state'] == 'Z'
            or not any(Path(item).name == script or item == 'nas.' + Path(script).stem
                       for item in info['argv'])):
        raise RuntimeError(f'PID identity is not the expected live {script}')
    value = argument(info['argv'], '--root') or argument(info['argv'], '--out')
    if value is None or Path(value).resolve() != root:
        raise RuntimeError(f'{script} does not control the requested experiment root')


def identical(pid, ticks):
    current = process(pid)
    return current is not None and current['start_ticks'] == ticks and current['uid'] == os.getuid()


def continue_exact(pid, ticks):
    if identical(pid, ticks):
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def read_registry(root):
    latest = {}
    lines = (root / 'run_registry.jsonl').read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1:
                raise RuntimeError('registry tail is being written; retry later')
            raise
        latest[row['job']['run_id']] = row
    return latest


def children(pid):
    try:
        ids = (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()
        return [p for number in ids if (p := process(int(number))) is not None]
    except FileNotFoundError:
        return []


def rescue_child(parent_read_fd, parent_write_fd, scheduler_pid, ticks, seconds, path):
    """Survives main helper death when its independent service uses KillMode=process."""
    child = os.fork()
    if child:
        os.close(parent_read_fd)
        return child
    os.close(parent_write_fd)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        ready, _, _ = select.select([parent_read_fd], [], [], seconds)
        continue_exact(scheduler_pid, ticks)
        atomic_json(path, {'time_utc': utc(), 'action': 'SIGCONT rescue',
                          'reason': 'helper pipe closed' if ready else 'bounded rescue timer',
                          'scheduler_pid': scheduler_pid, 'scheduler_start_ticks': ticks})
    finally:
        os.close(parent_read_fd)
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scheduler-pid', type=int, required=True)
    parser.add_argument('--workflow-pid', type=int, required=True)
    parser.add_argument('--workflow-unit', default='nas-minparams-20261111.service')
    parser.add_argument('--new-training-cutoff', default='2026-11-12T06:00:00Z')
    parser.add_argument('--expected-final-deadline', default='2026-11-12T06:30:00Z')
    parser.add_argument('--wait-timeout', type=float, default=3600)
    parser.add_argument('--drain-timeout', type=float, default=1000)
    parser.add_argument('--execute', action='store_true', help='otherwise inspect and write an operational record only')
    args = parser.parse_args()
    if not 1 <= args.drain_timeout <= 1000 or not 1 <= args.wait_timeout <= 3600:
        parser.error('drain timeout must be <=1000s; pre-pause wait must be <=3600s')
    root = args.root.resolve()
    if (root / 'selection.json').exists():
        raise RuntimeError('selection already frozen; refusing restart')
    scheduler, workflow = process(args.scheduler_pid), process(args.workflow_pid)
    validate_identity(scheduler, 'cuda_minparams_search.py', root)
    validate_identity(workflow, 'cuda_minparams_workflow.py', root)
    if scheduler['ppid'] != args.workflow_pid:
        raise RuntimeError('scheduler is not a direct child of the expected workflow')
    unit = subprocess.check_output(['systemctl', 'show', args.workflow_unit, '--property=ExecStart', '--value'], text=True, timeout=10)
    if ('--training-cutoff ' + args.new_training_cutoff) not in unit and ('--training-cutoff=' + args.new_training_cutoff) not in unit:
        raise RuntimeError('workflow ExecStart does not contain the authorized new cutoff')
    if ('--deadline ' + args.expected_final_deadline) not in unit and ('--deadline=' + args.expected_final_deadline) not in unit:
        raise RuntimeError('workflow final deadline differs from the explicitly expected deadline')
    unit_pid = int(subprocess.check_output(['systemctl', 'show', args.workflow_unit, '--property=MainPID', '--value'], text=True, timeout=10).strip())
    if unit_pid != args.workflow_pid:
        raise RuntimeError('workflow unit MainPID differs from the verified workflow PID')
    operations = root / 'operations'; operations.mkdir(exist_ok=True)
    label = datetime.now(timezone.utc).strftime('cutoff-drain-%Y%m%dT%H%M%S%fZ')
    output = operations / (label + '.json')
    plan_bytes = (root / 'plan.json').read_bytes()
    state = dict(started_utc=utc(), helper_pid=os.getpid(), root=str(root), scheduler=scheduler,
                 workflow=workflow, workflow_unit=args.workflow_unit, new_exec_start=unit.strip(),
                 new_training_cutoff=args.new_training_cutoff,
                 expected_final_deadline=args.expected_final_deadline, old_plan=json.loads(plan_bytes),
                 old_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), timeline=[])
    def event(name, **details):
        state['timeline'].append(dict(time_utc=utc(), event=name, **details))
        state['status'] = name
        atomic_json(output, state)
        print(json.dumps(state['timeline'][-1]), flush=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'helper received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    event('identities_and_updated_unit_verified', execute=args.execute)
    if not args.execute:
        return 0
    paused = False
    rescue_write = None
    rescue_pid = None
    start_ticks = scheduler['start_ticks']
    try:
        wait_until = time.monotonic() + args.wait_timeout
        last_report = 0
        while True:
            if time.monotonic() >= wait_until:
                raise TimeoutError('queue did not reach a natural drain boundary before pre-pause timeout')
            if (root / 'selection.json').exists():
                raise RuntimeError('selection froze while waiting; refusing restart')
            current = process(args.scheduler_pid)
            validate_identity(current, 'cuda_minparams_search.py', root)
            if current['start_ticks'] != start_ticks:
                raise RuntimeError('scheduler PID was reused')
            try:
                registry = read_registry(root)
            except RuntimeError:
                time.sleep(.2); continue
            queued = [row['job']['run_id'] for row in registry.values() if row['status'] == 'queued']
            inbox = sorted(path.name for path in (root / 'inbox').glob('*.json'))
            if queued or inbox:
                if time.monotonic()-last_report >= 20:
                    event('waiting_for_empty_queue', queued_run_ids=queued, inbox=inbox)
                    last_report = time.monotonic()
                time.sleep(1); continue
            # Arm an independent CONT timer before the first STOP signal.
            read_fd, rescue_write = os.pipe()
            rescue_pid = rescue_child(read_fd, rescue_write, args.scheduler_pid, start_ticks,
                                      args.drain_timeout + 15, operations / (label + '.rescue.json'))
            paused = True
            pause_started = time.monotonic()
            os.kill(args.scheduler_pid, signal.SIGSTOP)
            deadline = time.monotonic()+5
            while time.monotonic()<deadline:
                current = process(args.scheduler_pid)
                if current and current['state'] in ('T','t'):break
                time.sleep(.05)
            else:raise RuntimeError('scheduler did not enter stopped state')
            # Reject a race in which a new queue was made just before STOP.
            registry = read_registry(root)
            queued = [r['job']['run_id'] for r in registry.values() if r['status']=='queued']
            if queued or list((root/'inbox').glob('*.json')):
                raise RuntimeError('new queued work appeared during STOP; safely resume and retry helper later')
            if (root/'selection.json').exists():
                raise RuntimeError('selection froze at pause boundary; safely resume')
            break
        direct = children(args.scheduler_pid)
        paths = {str(Path(row['path']).resolve()): row for row in registry.values()}
        active = []
        for child in direct:
            out = argument(child['argv'], '--out')
            if child['state']=='Z':
                row = next((r for r in registry.values() if r.get('pid')==child['pid']), None)
            else:
                if not any((Path(arg).name=='cuda_train.py' or arg=='nas.cuda_train') for arg in child['argv']) or out is None:
                    raise RuntimeError(f'unexpected direct child {child["pid"]}; refusing drain')
                row = paths.get(str(Path(out).resolve()))
            if row is None:
                raise RuntimeError(f'direct trainer child {child["pid"]} lacks a run ledger entry')
            active.append(dict(pid=child['pid'], start_ticks=child['start_ticks'], run_id=row['job']['run_id'], path=row['path']))
        event('scheduler_paused', direct_trainers=active, rescue_pid=rescue_pid)
        drain_until = pause_started+args.drain_timeout
        while True:
            live = [child for child in children(args.scheduler_pid) if child['state']!='Z']
            if not live:break
            if time.monotonic()>=drain_until:
                raise TimeoutError('trainer drain exceeded bounded timeout; resuming normal scheduler without TERM')
            if any(child['pid'] not in {r['pid'] for r in active} for child in live):
                raise RuntimeError('unexpected child appeared while scheduler was stopped')
            if time.monotonic()-last_report >= 20:
                event('waiting_for_trainers', live_pids=[c['pid'] for c in live]);last_report=time.monotonic()
            time.sleep(1)
        # Include already-reaped runs awaiting ledger collection at pause time.
        finishing = {r['job']['run_id']:r['path'] for r in registry.values() if r['status']=='running'}
        finishing.update({r['run_id']:r['path'] for r in active})
        for run_id, folder in finishing.items():
            summary = json.loads((Path(folder)/'summary.json').read_text())
            if summary.get('steps')!=5000 or summary.get('schedule_steps')!=5000 or summary.get('benchmark'):
                raise RuntimeError(f'{run_id} did not finish exactly 5000 steps; resuming scheduler without TERM')
        event('all_trainers_finished_5000_steps', run_ids=sorted(finishing))
        if not identical(args.scheduler_pid,start_ticks) or (root/'selection.json').exists():
            raise RuntimeError('scheduler identity or freeze state changed before TERM')
        os.kill(args.scheduler_pid,signal.SIGTERM)  # Queued while stopped.
        event('termination_queued_for_clean_collection')
        continue_exact(args.scheduler_pid,start_ticks)
        paused=False
        event('scheduler_continued_for_clean_restart', expected_exit=130)
        return 0
    except BaseException as error:
        event('helper_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # Even a logging failure or caught signal must never strand the scheduler.
        if paused:
            continue_exact(args.scheduler_pid,start_ticks)
        if rescue_write is not None:
            os.close(rescue_write)
        if rescue_pid is not None:
            try:os.waitpid(rescue_pid,os.WNOHANG)
            except ChildProcessError:pass


if __name__=='__main__':
    raise SystemExit(main())
