"""No-GPU lifecycle tests for the overnight scheduler and its audit contracts."""
from __future__ import annotations

import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import subprocess
import sys

import cuda_search as search

FAKE_WORKER = r'''
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cuda_search as search
p=argparse.ArgumentParser()
p.add_argument('root'); p.add_argument('--out'); p.add_argument('--spec'); p.add_argument('--seed',type=int)
p.add_argument('--behavior',default='ok'); p.add_argument('--trace')
a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
def trace(event):
 with open(a.trace,'a') as f:
  f.write(json.dumps({'event':event,'time':time.time(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid()})+'\n')
trace('start')
if a.behavior=='timeout': time.sleep(30)
if a.behavior=='failure':
 trace('end'); print('deliberate fake failure'); raise SystemExit(7)
time.sleep(.08)
spec=json.loads(Path(a.spec).read_text()); count=search.parameter_count(spec)
loss=(3.4 if spec['name']=='baseline' else 3.3)+a.seed*.001
summary={'steps':5000,'schedule_steps':5000,'parameters':count,'best_dev_loss':loss,
         'train_seconds':.08,'best_step':4750,'train_eval_loss_at_best_dev':loss-.2,
         'generalization_gap_at_best_dev':.2,'final_dev_loss':loss+.01,'final_train_eval_loss':loss-.2}
if a.behavior=='wrong': summary['steps']=4999
config={'model':search.FRAME,'spec':spec,'args':dict(search.RECIPE,seed=a.seed,precision='bf16'),'sha256':{}}
if a.behavior=='drift': config['sha256']['cuda_model.py']='0'*64
(out/'config.json').write_text(json.dumps(config))
(out/'summary.json').write_text(json.dumps(summary))
(out/'log.jsonl').write_text(json.dumps({'step':4750,'dev_loss':loss,'train_eval_loss':loss-.2,'generalization_gap':.2})+'\n')
for name in ['best.pt','latest.pt','final.pt']: (out/name).write_bytes(b'x'*131072)
trace('end')
'''


class FakeScheduler(search.Scheduler):
    def command(self, job):
        behavior = getattr(self, 'behaviors', {}).get(job.spec, 'ok')
        return [sys.executable, str(self.args.out.parent / 'worker.py'), str(search.ROOT),
                '--out', str(self.args.out / 'attempts' / job.run_id),
                '--spec', str(self.args.out / 'specs' / (job.architecture_id + '.json')),
                '--seed', str(job.seed), '--behavior', behavior,
                '--trace', str(self.args.out / 'trace.jsonl')]


def create_scheduler(root, name='search'):
    root = root.resolve()
    out = root / name
    (root / 'worker.py').write_text(FAKE_WORKER)
    args = search.make_parser().parse_args(['--out', str(out), '--hours', '1', '--gpus', '900,901',
             '--runs-root', str(root), '--no-plots', '--estimated-5000-seconds', '.1',
             '--poll-seconds', '.01', '--shutdown-buffer-seconds', '0', '--final-reserve-seconds', '0'])
    args.gpus = [900, 901]
    args.plan_only = True  # Schema checks run in their own torch process below.
    args.out.mkdir()
    for directory in ('attempts', 'logs', 'specs'):
        (args.out / directory).mkdir()
    return FakeScheduler(args, search.utc_now() + timedelta(hours=1))


def test_candidates_and_canonicalization():
    from cuda_model import Config, validate, spec_parameter_count, learned_scale_count
    candidates = search.candidates()
    assert len(candidates) >= 200
    assert len({search.architecture_hash(spec) for spec in candidates.values()}) == len(candidates)
    for spec in candidates.values():
        validate(spec, Config(**search.FRAME))
        assert spec_parameter_count(spec, Config(**search.FRAME)) == search.parameter_count(spec)
        assert learned_scale_count(spec) == search.learned_scale_count(spec)
    a = search.baseline_spec()
    b = copy.deepcopy(a)
    b['name'] = 'different label'
    b['ffns']['unused'] = {'act': 'silu'}
    for layer in b['layers']:
        layer.update(loop=1, inject=False, order='attn_ffn', attn_scale={'init':1.,'learned':False})
    assert search.architecture_hash(a) == search.architecture_hash(b)
    b['layers'][0]['attn_scale'] = {'init': 1., 'learned': True}
    assert search.architecture_hash(a) != search.architecture_hash(b)
    for parent in ('baseline', 'relu2-narrow1472-learned-ffn', 'relu2-loop4x2-inject'):
        for _, spec in search.mutations(candidates[parent], 1):
            validate(spec, Config(**search.FRAME))
            assert spec_parameter_count(spec, Config(**search.FRAME)) == search.parameter_count(spec)
    print('ok candidates, canonical uniqueness, all primitives, mutation validation')


def test_parallel_audit_retention_and_final_contract():
    with tempfile.TemporaryDirectory(prefix='nas-scheduler-') as directory:
        scheduler = create_scheduler(Path(directory))
        try:
            scheduler.acquire_gpus()
            specs = search.candidates()
            ids = []
            for name in ('baseline', 'relu2-narrow1472', 'relu2-narrow1280'):
                ids.append(scheduler.register(name, specs[name])[0])
            jobs = [scheduler.make_job(identifier, seed, 'confirmation' if seed else 'breadth')
                    for identifier in ids for seed in (0,1,2)]
            scheduler.run_stage(jobs, 'fake-confirmation')
            assert len(scheduler.completed) == 9
            assert scheduler.make_job(ids[0], 0, 'duplicate') is None
            trace = [json.loads(line) for line in (scheduler.args.out/'trace.jsonl').read_text().splitlines()]
            active, maximum = set(), 0
            for event in trace:
                gpu = event['gpu']
                if event['event'] == 'start':
                    assert gpu not in active
                    active.add(gpu); maximum=max(maximum,len(active))
                else:
                    active.remove(gpu)
            assert maximum == 2 and not active
            for record in scheduler.records.values():
                assert record['status'] == 'completed'
                assert record['source_sha256'] and record['physical_gpu']['physical_index'] in (900,901)
                assert record['started_utc'] and record['ended_utc']
                assert record['metrics']['train_eval_loss_at_best_dev'] is not None
                assert record['checkpoints']['best.pt']['sha256']
            total = sum(path.stat().st_size for path in scheduler.args.out.rglob('*') if path.is_file())
            scheduler.args.max_disk_gib = total * .75 / 1024**3
            scheduler.prune_if_needed()
            assert not scheduler.stop_requested
            assert any(cp['pruned'] for record in scheduler.records.values() for cp in record['checkpoints'].values())
            assert all((Path(item['path'])/'best.pt').is_file() for item in scheduler.completed.values())
            assert all((Path(item['path'])/'log.jsonl').is_file() for item in scheduler.completed.values())
            scheduler.args.max_disk_gib = 2
            scheduler.final_confirmation()
            final = json.loads((scheduler.args.out/'final_selection.json').read_text())
            assert final['status'] == 'complete'
            assert len(final['baseline_runs']) == len(final['winner_runs']) == 3
            assert final['parameter_reduction'] > 0
            assert final['fresh_seeds_used_for_architecture_selection'] is False
            assert final['validation_accessed'] is False
            for seed,b,w in zip([10,11,12],final['baseline_runs'],final['winner_runs']):
                assert json.loads((Path(b)/'config.json').read_text())['args']['seed'] == seed
                assert json.loads((Path(w)/'config.json').read_text())['args']['seed'] == seed
            retained = json.loads((scheduler.args.out/'retained_checkpoints.json').read_text())
            assert retained['relative_paths']
            assert all(not Path(path).is_absolute() and '..' not in Path(path).parts for path in retained['relative_paths'])
            ledger = [json.loads(line) for line in (scheduler.args.out/'run_registry.jsonl').read_text().splitlines()]
            assert any(row['status']=='running' for row in ledger)
            assert any(row['status']=='completed' for row in ledger)
            scheduler.stage='completed'
        finally:
            scheduler.close()
    print('ok parallel exclusivity, provenance, pruning, retention and frozen finalizer contract')


def test_failures_timeout_and_deferred_readmission():
    with tempfile.TemporaryDirectory(prefix='nas-scheduler-') as directory:
        scheduler = create_scheduler(Path(directory))
        try:
            specs=search.candidates()
            scheduler.behaviors={'relu2':'failure','relu2-narrow1472':'wrong','relu2-narrow1280':'drift'}
            ids={name:scheduler.register(name,specs[name])[0] for name in ['baseline','relu2','relu2-narrow1472','relu2-narrow1280']}
            jobs=[scheduler.make_job(ids[name],0,'failure-test') for name in scheduler.behaviors]
            scheduler.run_stage(jobs,'failure-test')
            assert all(scheduler.records[job.run_id]['status']=='failed' for job in jobs)
            assert any('source/data changed' in scheduler.records[job.run_id]['failure_reason'] for job in jobs)
            deferred=scheduler.make_job(ids['baseline'],1,'confirmation')
            scheduler.run_stage([deferred],'too-late',search.utc_now()-timedelta(seconds=1))
            assert scheduler.records[deferred.run_id]['status']=='not_admitted'
            readmitted=scheduler.make_job(ids['baseline'],1,'confirmation')
            assert readmitted==deferred
            scheduler.run_stage([readmitted],'readmitted')
            assert scheduler.records[deferred.run_id]['status']=='completed'
            scheduler.behaviors['baseline']='timeout'
            scheduler.args.job_timeout_seconds=.25
            timeout=scheduler.make_job(ids['baseline'],2,'timeout')
            scheduler.run_stage([timeout],'timeout')
            assert scheduler.records[timeout.run_id]['status']=='timed_out'
            scheduler.behaviors['baseline']='ok'
            scheduler.args.job_timeout_seconds=900
            interrupted=scheduler.make_job(ids['baseline'],3,'interrupted')
            scheduler.launch(900,interrupted)
            scheduler.stop_children('test interruption')
            assert scheduler.records[interrupted.run_id]['status']=='interrupted'
            assert not scheduler.running
        finally:
            scheduler.close()
    print('ok failed/malformed/drifted results, timeout, interruption and deferred readmission')


if __name__=='__main__':
    if '--candidates-only' in sys.argv:
        test_candidates_and_canonicalization()
        raise SystemExit(0)
    # macOS OpenMP can abort after fork from a torch-importing process. Keep
    # model validation isolated from the standard-library fake-worker tests.
    subprocess.run([sys.executable, __file__, '--candidates-only'], check=True)
    test_parallel_audit_retention_and_final_contract()
    test_failures_timeout_and_deferred_readmission()
    print('All scheduler tests passed; no training jobs or GPUs used.')
