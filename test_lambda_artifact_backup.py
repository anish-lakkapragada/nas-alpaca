"""Offline S3 behavior tests; no cloud resources, credentials, datasets, or GPUs."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch

from lambda_artifact_backup import collect, copy_snapshot, fingerprint, private_json, run_backup, validate_config


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.lock = threading.Lock()
        self.corrupt_reads = False
        self.after_put = None

    def put_object(self, Bucket, Key, Body, ContentLength, Metadata):
        data = Body.read()
        assert len(data) == ContentLength
        with self.lock:
            self.calls.append(('put', Key))
            self.objects[Key] = {'data': data, 'metadata': Metadata.copy(), 'etag': '"' + hashlib.md5(data).hexdigest() + '"'}
        if self.after_put:
            self.after_put(Key)
        return {'ETag': self.objects[Key]['etag']}

    def head_object(self, Bucket, Key):
        with self.lock:
            self.calls.append(('head', Key))
            item = self.objects[Key]
            return {'ETag': item['etag'], 'ContentLength': len(item['data']), 'Metadata': item['metadata'].copy()}

    def get_object(self, Bucket, Key):
        with self.lock:
            self.calls.append(('get', Key))
            item = self.objects[Key]
            data = item['data'] + (b'corrupt' if self.corrupt_reads else b'')
            return {'Body': io.BytesIO(data), 'ETag': item['etag'], 'ContentLength': len(item['data'])}


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nas-backup-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.search = self.runs / 'cuda/search'
        self.search.mkdir(parents=True)
        self.config = {'endpoint_url': 'https://files.us-south-1.lambda.ai', 'region': 'us-south-1',
                       'bucket': 'test-filesystem', 'prefix': 'nas/task-instance', 'instance_id': 'task-instance',
                       'credential_file': str(self.root / 'credentials.json'), 'runs_root': str(self.runs),
                       'search_relative_path': 'cuda/search', 'receipt_file': str(self.root / 'receipt.json'),
                       'cache_file': str(self.root / 'cache.json'), 'workers': 2}
        self.checkpoint = self.search / 'attempts/r1/best.pt'
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'checkpoint bytes')
        self.manifest()
        (self.search / 'attempts/r1/config.json').write_text('{"steps":5000}')
        (self.search / 'attempts/r1/log.jsonl').write_text('{"step":5000,"dev_loss":3.1}\n')
        (self.root / 'cuda_train.py').write_text('print("source snapshot")\n')
        self.s3 = FakeS3()

    def manifest(self):
        data = self.checkpoint.read_bytes()
        name = self.checkpoint.relative_to(self.runs).as_posix()
        (self.search / 'retained_checkpoints.json').write_text(json.dumps({
            'relative_paths': [name], 'checkpoints': [{'relative_path': name,
                 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}]}))

    def complete(self):
        report = self.search / 'FINAL_REPORT.json'
        report.write_text('{"status":"complete"}')
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        (self.search / 'finalization.json').write_text(json.dumps({'status': 'complete', 'final_report_sha256': digest}))
        control = self.search.with_name('search-control')
        control.mkdir(exist_ok=True)
        (control / 'workflow.json').write_text('{"status":"completed"}')
        return digest

    def test_verified_final_receipt_incremental_and_changed_objects(self):
        digest = self.complete()
        first = run_backup(self.config, self.s3)
        self.assertTrue(all(first[field] for field in ('verified', 'durable', 'backup_complete',
                          'metadata_complete', 'retained_checkpoints_complete')))
        self.assertEqual((first['finalization_status'], first['final_report_sha256']), ('complete', digest))
        self.assertEqual(stat.S_IMODE(Path(self.config['receipt_file']).stat().st_mode), 0o600)
        manifest = json.loads(self.s3.objects[first['manifest_object_key']]['data'])
        self.assertIn('source/cuda_train.py', manifest['files'])
        self.assertIn('runs/cuda/search/attempts/r1/best.pt', manifest['files'])
        before = len(self.s3.calls)
        run_backup(self.config, self.s3)
        calls = self.s3.calls[before:]
        self.assertFalse(any(method in ('put', 'get') and '/objects/' in key for method, key in calls))
        self.assertGreaterEqual(sum(method == 'head' for method, key in calls), first['artifact_count'])
        log = self.search / 'attempts/r1/log.jsonl'
        log.write_text(log.read_text() + '{"step":5001}\n')
        before = len(self.s3.calls)
        run_backup(self.config, self.s3)
        self.assertEqual(sum(method == 'put' and '/objects/' in key for method, key in self.s3.calls[before:]), 1)

    def test_remote_corruption_is_detected_and_cache_resumes(self):
        first = run_backup(self.config, self.s3)
        cache = json.loads(Path(self.config['cache_file']).read_text())
        item = cache['files']['runs/cuda/search/attempts/r1/best.pt']
        self.s3.objects[item['object_key']]['etag'] = 'corrupted-etag'
        self.s3.objects[item['object_key']]['data'] = b'corrupt'
        before = len(self.s3.calls)
        run_backup(self.config, self.s3)
        self.assertEqual(self.s3.objects[item['object_key']]['data'], self.checkpoint.read_bytes())
        self.assertEqual(sum(method == 'put' and '/objects/' in key for method, key in self.s3.calls[before:]), 1)
        (self.search / 'new.json').write_text('{"new":true}')
        self.s3.corrupt_reads = True
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            run_backup(self.config, self.s3)
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['backup_complete'])
        self.s3.corrupt_reads = False
        before = len(self.s3.calls)
        result = run_backup(self.config, self.s3)
        self.assertTrue(result['backup_complete'])
        self.assertEqual(sum(method == 'put' and '/objects/' in key for method, key in self.s3.calls[before:]), 1)
        self.assertEqual(first['finalization_status'], 'running')

    def test_path_dataset_secret_and_unselected_checkpoint_exclusion(self):
        for relative_name in ('data/train.json', 'datasets/validation.json', 'credentials.json', '.env', '.env.json', 'private-key.txt'):
            path = self.search / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('must never upload')
        (self.search / 'unselected.pt').write_bytes(b'not retained')
        (self.search / 'linked.json').symlink_to(self.root / 'credentials.json')
        names = set(collect(self.config))
        self.assertFalse(any('credentials' in name or 'linked' in name or 'unselected' in name or '/data/' in name for name in names))
        for bad in ('../outside.pt', '/tmp/outside.pt', 'cuda/../../outside.pt', 'cuda/data/train.pt', 'cuda/secret.pt'):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                validate_config(dict(self.config, historical_checkpoint_paths=[bad]))
        (self.search / 'bad.pt').symlink_to(self.checkpoint)
        with self.assertRaises(ValueError):
            validate_config(dict(self.config, historical_checkpoint_paths=['cuda/search/bad.pt']))

    def test_retained_hash_mismatch_and_final_source_mutation_fail_closed(self):
        self.checkpoint.write_bytes(b'wrong checkpoint')
        with self.assertRaisesRegex(ValueError, 'retained manifest'):
            run_backup(self.config, self.s3)
        self.manifest()
        self.complete()
        marker = self.search / 'late.json'
        def mutate(key):
            if '/objects/' in key and not marker.exists():
                marker.write_text('{}')
        self.s3.after_put = mutate
        with self.assertRaisesRegex(ValueError, 'changed during backup'):
            run_backup(self.config, self.s3)
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['backup_complete'])
        self.s3.after_put = None
        self.assertEqual(run_backup(self.config, self.s3)['finalization_status'], 'complete')

    def test_endpoint_instance_private_config_and_deadline(self):
        for endpoint in ('http://files.us-south-1.lambda.ai', 'https://evil.example',
                         'https://files.us-south-1.lambda.ai.evil.example', 'https://key@files.us-south-1.lambda.ai',
                         'https://files.us-south-1.lambda.ai/path'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_config(dict(self.config, endpoint_url=endpoint))
        with self.assertRaises(ValueError):
            validate_config(dict(self.config, prefix='nas/another-instance'))
        private = self.root / 'private.json'
        private.write_text('{}')
        private.chmod(0o644)
        with self.assertRaises(ValueError):
            private_json(private)
        with self.assertRaises(TimeoutError):
            run_backup(self.config, self.s3, max_seconds=-1)
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['verified'])

    def test_growing_log_prefix_and_atomic_json_retry(self):
        path = self.search / 'live.jsonl'
        initial, appended = b'{"step":1}\n', b'{"step":2}\n'
        path.write_bytes(initial)
        item = {'path': path, 'kind': 'metadata', 'fingerprint': fingerprint(path)}
        class MutatingOutput(io.BytesIO):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback
            def write(self, data):
                result = super().write(data)
                callback, self.callback = self.callback, None
                if callback:
                    callback()
                return result
        def append():
            with path.open('ab') as stream:
                stream.write(appended)
        output = MutatingOutput(append)
        result = copy_snapshot(item, output, lambda: None)
        self.assertEqual(output.getvalue(), initial)
        self.assertTrue(result['prefix_snapshot'])
        self.assertEqual(result['fingerprint'], item['fingerprint'])
        output = io.BytesIO()
        fresh = copy_snapshot(item, output, lambda: None)
        self.assertEqual(output.getvalue(), initial + appended)
        self.assertFalse(fresh['prefix_snapshot'])
        path = self.search / 'live.json'
        path.write_bytes(b'{"generation":1}')
        item = {'path': path, 'kind': 'metadata', 'fingerprint': fingerprint(path)}
        def replace():
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(b'{"generation":2}')
            temporary.replace(path)
        output = MutatingOutput(replace)
        result = copy_snapshot(item, output, lambda: None)
        self.assertEqual(output.getvalue(), b'{"generation":2}')
        self.assertEqual(result['fingerprint'], fingerprint(path))

    def test_s3_growth_after_collection_succeeds_and_retained_change_stays_strict(self):
        path = self.search / 'live.log'
        path.write_bytes(b'first\n')
        real_collect = collect
        def changing_collect(config):
            result = real_collect(config)
            with path.open('ab') as stream:
                stream.write(b'second\n')
            return result
        with patch('lambda_artifact_backup.collect', side_effect=changing_collect):
            receipt = run_backup(self.config, self.s3)
        manifest = json.loads(self.s3.objects[receipt['manifest_object_key']]['data'])
        item = manifest['files']['runs/cuda/search/live.log']
        self.assertEqual(self.s3.objects[item['object_key']]['data'], b'first\nsecond\n')
        strict = {'path': self.checkpoint, 'kind': 'checkpoint', 'fingerprint': fingerprint(self.checkpoint)}
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'retained checkpoint changed'):
            copy_snapshot(strict, io.BytesIO(), lambda: None)


if __name__ == '__main__':
    unittest.main()
