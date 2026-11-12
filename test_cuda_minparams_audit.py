"""Synthetic evidence checks; never opens a live experiment or scores a model."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import numpy as np

from cuda_minparams_audit import Audit, TRAIN_FILES, FIXED_ARGS, digest
from cuda_minparams_search import FRAME, baseline_spec, standard, parameter_count, architecture_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def fixture(root, project):
    manifest = {}
    for name in (*TRAIN_FILES, 'cuda_minparams_search.py', 'cuda_evaluate.py', 'cuda_minparams_finalize.py', 'data/validation.json'):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic ' + name)
        manifest[name] = digest(path)
        if not name.startswith('data/'):
            (root / 'source').mkdir(exist_ok=True)
            (root / 'source' / name).write_bytes(path.read_bytes())
    write(root / 'workflow_source_manifest.json', {'sha256': manifest})
    write(root / 'provenance.json', {'source_sha256': {n: manifest[n] for n in TRAIN_FILES}})
    for name in ('architecture_registry.jsonl', 'run_registry.jsonl', 'events.jsonl', 'evaluation_registry.jsonl'):
        (root / name).touch()
    for name in ('attempts', 'logs', 'specs', 'final-evaluations'):
        (root / name).mkdir()
    architectures = []
    for name, pair in [('baseline', (FRAME, baseline_spec())), ('small', standard(8,256,1024)), ('larger', standard(8,320,1280))]:
        frame, spec = copy.deepcopy(pair)
        spec['name'] = name
        aid = architecture_hash(frame, spec)
        path = root / 'specs' / (aid + '.json')
        write(path, spec)
        row = dict(architecture_id=aid, name=name, frame=frame, spec=spec, parameters=parameter_count(frame,spec),
                   spec_path=str(path), file_sha256=digest(path), status='accepted')
        append(root / 'architecture_registry.jsonl', {k:v for k,v in row.items() if k!='spec'})
        architectures.append(row)
    diagnostic = dict(split='train', conversations=256, dropout=False, seed=20261111, answer_tokens=100, indices_sha256='fixed')
    run_rows = []
    def run(architecture, seed, phase):
        index = len(run_rows)+1
        rid = f'r{index:05d}-{architecture["architecture_id"][:10]}-s{seed}'
        path = root / 'attempts' / rid
        path.mkdir()
        (root / 'logs' / (rid+'.log')).write_text('synthetic trainer output')
        name=architecture['name']
        loss = {'baseline':3.4,'small':3.39,'larger':3.3}[name] + seed*.00001
        args = dict(FIXED_ARGS, seed=seed, resume=None, benchmark_steps=0)
        config = dict(model=architecture['frame'], spec=architecture['spec'], parameters=architecture['parameters'], args=args,
                      sha256={n:manifest[n] for n in TRAIN_FILES}, train_evaluation=diagnostic,
                      optimizer=dict(name='AdamWNoBiasCorrection', bias_correction=False, betas=[.9,.95],eps=1e-8),
                      hardware=dict(physical_gpu_ids=str(index%4)))
        write(path/'config.json', config)
        summary = dict(parameters=architecture['parameters'], steps=5000, schedule_steps=5000, best_step=5000,
                       best_dev_loss=loss, final_dev_loss=loss, train_eval_loss_at_best_dev=2.,final_train_eval_loss=2.,
                       answer_tokens_seen=(100+seed)*200, train_seconds=100.,training_compute_seconds=50.)
        write(path/'summary.json',summary)
        for step in range(0,5001,250):
            append(path/'log.jsonl',dict(event='evaluation',step=step,dev_loss=loss+(5000-step)/1e6,
                                       train_eval_loss=2.+(5000-step)/1e6))
        for step in range(25,5001,25):
            append(path/'log.jsonl',dict(event='training',step=step,train_loss=2.5,answer_tokens_seen=step//25*(100+seed),
                                       train_loss_answer_tokens=100+seed))
        for ext in ('png','svg'):(path/f'curves.{ext}').write_text('synthetic curve')
        write(path/'metrics.json',dict(best_dev_loss=loss,parameters=architecture['parameters']))
        checkpoints={}
        for name in ('best.pt','final.pt'):
            checkpoint=path/name;checkpoint.write_bytes(('synthetic '+rid+name).encode())
            checkpoints[name]=dict(path=str(checkpoint),sha256=digest(checkpoint),bytes=checkpoint.stat().st_size,pruned=False)
        row=dict(job=dict(run_id=rid,architecture_id=architecture['architecture_id'],spec=architecture['name'],seed=seed,phase=phase,steps=5000),
                 path=str(path),parameters=architecture['parameters'],summary=summary,source_hashes=config['sha256'],
                 physical_gpu=dict(physical_index=index%4),checkpoints=checkpoints,
                 started_utc='2026-11-11T23:20:00Z' if phase=='final' else '2026-11-11T23:00:00Z',
                 ended_utc='2026-11-11T23:22:00Z' if phase=='final' else '2026-11-11T23:02:00Z',wall_seconds=120)
        if phase=='final':row['selection_freeze_sha256']=digest(root/'selection.json')
        append(root/'run_registry.jsonl',{**row,'status':'running'})
        row['status']='completed';append(root/'run_registry.jsonl',row)
        append(root/'events.jsonl',dict(event='launched',run_id=rid))
        run_rows.append(row)
        return row
    for a in architectures:
        a['run_paths']={str(seed):run(a,seed,'search')['path'] for seed in (0,1,2)}
        a['mean_dev_loss']={'baseline':3.4,'small':3.39,'larger':3.3}[a['name']]+.00001
    baseline,winner,larger=architectures
    qualified=[{k:a[k] for k in ('architecture_id','parameters','mean_dev_loss')} for a in (winner,larger)]
    selection=dict(status='qualified',frozen_utc='2026-11-11T23:10:00Z',selection_seeds=[0,1,2],fresh_seeds=list(range(100,105)),
                   baseline=baseline,primary=winner,primary_architecture_id=winner['architecture_id'],qualified_candidates=qualified,validation_used=False)
    write(root/'selection.json',selection)
    fresh=[]
    for role,a in [('baseline',baseline),('winner',winner)]:
        for seed in range(100,105):
            row=run(a,seed,'final');path=Path(row['path'])
            fresh.append(dict(run=row['path'],role=role,seed=seed,config_sha256=digest(path/'config.json'),
                              summary_sha256=digest(path/'summary.json'),checkpoint_sha256=digest(path/'best.pt')))
    write(root/'final_confirmation.json',dict(frozen_utc='2026-11-11T23:30:00Z',selection_freeze_sha256=digest(root/'selection.json'),
          test_accessed=False,fresh_seeds_used_for_architecture_selection=False,runs=fresh))
    scores={};archives={}
    for index,row in enumerate(fresh):
        score=(3.32 if row['role']=='baseline' else 3.25)+row['seed']*.00001
        output=root/'final-evaluations'/(Path(row['run']).name+'.json')
        value=dict(run=row['run'],seed=row['seed'],checkpoint_sha256=row['checkpoint_sha256'],config_sha256=row['config_sha256'],
                   split='validation',conversations=2599,precision='bf16',answer_tokens=2599,loss=score,completed_utc='2026-11-11T23:41:00Z')
        write(output,value)
        sums=np.full(2599,score);counts=np.ones(2599,dtype=np.int64)
        np.savez_compressed(output.with_suffix('.conversations.npz'),sums=sums,counts=counts)
        event=dict(output=str(output),run=row['run'],status='completed',started_utc='2026-11-11T23:40:00Z',ended_utc='2026-11-11T23:42:00Z',
                   checkpoint_sha256=row['checkpoint_sha256'],json_sha256=digest(output),conversations_sha256=digest(output.with_suffix('.conversations.npz')))
        append(root/'evaluation_registry.jsonl',event)
        scores[row['run']]=value
        archives[f'run_{index}_sums']=sums;archives[f'run_{index}_counts']=counts
    write(root/'final-test.json',dict(run_order=[r['run'] for r in fresh],results=scores))
    np.savez_compressed(root/'final-test.conversations.npz',**archives)
    import statistics
    def stats(role,a):
        rows=[r for r in fresh if r['role']==role];vals=[scores[r['run']]['loss'] for r in rows]
        dev=[read_json(Path(r['run'])/'summary.json')['best_dev_loss'] for r in rows]
        return dict(parameters=a['parameters'],runs=[{**r,'test_loss':scores[r['run']]['loss']} for r in rows],
                    dev_mean=statistics.mean(dev),dev_std=statistics.stdev(dev),train_diagnostic_mean=2.,train_diagnostic_std=0.,
                    test_mean=statistics.mean(vals),test_std=statistics.stdev(vals))
    def read_json(path):return json.loads(path.read_text())
    reduction=baseline['parameters']-winner['parameters']
    report=dict(status='complete',selection_freeze_sha256=digest(root/'selection.json'),winner_architecture_id=winner['architecture_id'],
                baseline=stats('baseline',baseline),winner=stats('winner',winner),test_comparison=dict(point=-.07),
                test_mean_no_worse=True,public_claim_supported_by_mean=True,parameter_reduction=reduction,
                parameter_reduction_fraction=reduction/baseline['parameters'],search_baseline_mean_dev=baseline['mean_dev_loss'],
                search_winner_mean_dev=winner['mean_dev_loss'],fresh_seeds=list(range(100,105)),fresh_seeds_used_for_architecture_selection=False,
                test_used_for_selection=False,reselection_after_test=False,data_and_source_provenance=manifest,
                completed_training_runs=len(run_rows),architectures_with_completed_runs=3)
    write(root/'FINAL_REPORT.json',report)
    return run_rows


def main():
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary).resolve();root=base/'experiment';root.mkdir();project=base/'project';project.mkdir()
        rows=fixture(root,project)
        result=Audit(root,project).execute()
        assert result['status']=='pass',result['failures']
        assert result['counts']['completed_training_runs']==19
        assert abs(result['recomputed']['mean_paired_test_difference']+.07)<1e-9
        # Recorded pruning is allowed, while an unlogged attempt is an error.
        pruned=copy.deepcopy(rows[6])
        for metadata in pruned['checkpoints'].values():
            Path(metadata['path']).unlink();metadata['pruned']=True
        append(root/'run_registry.jsonl',pruned)
        result=Audit(root,project).execute()
        assert result['status']=='pass',result['failures']
        assert result['counts']['recorded_pruned_checkpoints']==2
        unknown=root/'attempts'/'unlogged';unknown.mkdir()
        result=Audit(root,project).execute()
        assert any(r['code']=='unlogged_attempt_directory' for r in result['failures'])
        unknown.rmdir()
        selection=json.loads((root/'selection.json').read_text())
        altered=copy.deepcopy(selection);altered['primary_architecture_id']=selection['qualified_candidates'][1]['architecture_id']
        write(root/'selection.json',altered)
        result=Audit(root,project).execute()
        assert any(r['code']=='smallest_qualifier' for r in result['failures'])
        write(root/'selection.json',selection)
        report=json.loads((root/'FINAL_REPORT.json').read_text())
        report['baseline']['test_mean']+=.01;write(root/'FINAL_REPORT.json',report)
        result=Audit(root,project).execute()
        assert any(r['code']=='reported_test_statistics' for r in result['failures'])
        report['status']='incomplete_confirmation';write(root/'FINAL_REPORT.json',report)
        try:Audit(root,project)
        except RuntimeError:pass
        else:raise AssertionError('incomplete experiment was not refused')
        assert not (root/'final-audit').exists(),'library audit unexpectedly wrote artifacts'
    # A legitimate finalizer resume can rewrite the checkpoint-freeze timestamp
    # while retaining exact already-completed checkpoint/evaluation identities.
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary).resolve();root=base/'experiment';root.mkdir();project=base/'project';project.mkdir()
        rows=fixture(root,project)
        path=root/'final_confirmation.json'
        confirmation=json.loads(path.read_text());confirmation['frozen_utc']='2026-11-11T23:50:00Z'
        write(path,confirmation)
        result=Audit(root,project).execute()
        assert result['status']=='pass',result['failures']
        assert any(w['code']=='checkpoint_freeze_timestamp_rewritten' for w in result['warnings'])
        # Timestamp relaxation cannot accept a different selected checkpoint.
        altered=copy.deepcopy(confirmation);altered['runs'][0]['checkpoint_sha256']='wrong'
        write(path,altered)
        result=Audit(root,project).execute()
        assert result['status']=='fail'
        assert any(f['code']=='fresh_artifact_hash' for f in result['failures'])
        assert any(f['code']=='test_before_freeze' for f in result['failures'])
        write(path,confirmation)
        original=(root/'evaluation_registry.jsonl').read_text()
        events=[json.loads(line) for line in original.splitlines()]
        for timestamp,code in [('2026-11-11T23:05:00Z','test_before_selection'),
                               ('2026-11-11T23:21:00Z','test_before_fresh_completion')]:
            changed=copy.deepcopy(events);changed[0]['started_utc']=timestamp
            (root/'evaluation_registry.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in changed))
            result=Audit(root,project).execute()
            assert result['status']=='fail'
            assert any(f['code']==code for f in result['failures'])
        (root/'evaluation_registry.jsonl').write_text(original)
    # Frozen finalizer timeout/interruption schemas omit some diagnostic fields.
    # Their untrusted partial weights must not become completed-run evidence.
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary).resolve();root=base/'experiment';root.mkdir();project=base/'project';project.mkdir()
        rows=fixture(root,project)
        before=Audit(root,project).execute()['recomputed']['observed_timing']['training_attempt_gpu_wall_seconds']
        for status in ('timed_out','interrupted'):
            source=next(r for r in rows if r['job']['phase']=='final')
            rid='partial-'+status;path=root/'attempts'/rid;path.mkdir()
            (root/'logs'/(rid+'.log')).write_text('synthetic stopped trainer')
            (path/'best.pt').write_bytes(b'untrusted partial checkpoint')
            row=dict(job={**source['job'],'run_id':rid},path=str(path),parameters=source['parameters'],
                     physical_gpu=source['physical_gpu'],selection_freeze_sha256=source['selection_freeze_sha256'],
                     started_utc='2026-11-11T23:15:00Z',ended_utc='2026-11-11T23:17:00Z')
            if status=='timed_out':
                row['wall_seconds']=120.
            else:
                row['failure_reason']='finalizer stopped before completion'
            append(root/'run_registry.jsonl',{**row,'status':'running'})
            append(root/'run_registry.jsonl',{**row,'status':status})
        result=Audit(root,project).execute()
        assert result['status']=='pass',result['failures']
        assert result['counts']['completed_training_runs']==19
        assert result['counts']['unverified_partial_checkpoint_files']==2
        assert len(result['recomputed']['unverified_partial_artifacts'])==2
        assert any(w['code']=='writer_omitted_failure_reason' for w in result['warnings'])
        assert any(w['code']=='duration_inferred_from_timestamps' for w in result['warnings'])
        timing=result['recomputed']['observed_timing']
        assert timing['inferred_duration_attempts']==1 and timing['inferred_wall_seconds']==120
        assert timing['training_attempt_gpu_wall_seconds']==before+240
        # A checkpoint missing trusted provenance in a completed run remains fatal.
        completed=rows[0];(Path(completed['path'])/'extra.pt').write_bytes(b'unlogged completed-run weight')
        result=Audit(root,project).execute()
        assert result['status']=='fail'
        assert any(f['code']=='unlogged_checkpoint' for f in result['failures'])
    print('Synthetic artifact audit passed, including preserved pruning, resume timestamps, exact checkpoint identity, '
          'pre-selection/pre-fresh-completion rejection, interrupted/timeout inventories and disclosed inferred durations.')


if __name__=='__main__':main()
