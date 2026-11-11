"""Offline filesystem backup and real embedded verifier tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path, PosixPath
import tempfile
import unittest
from unittest.mock import patch

from lambda_filesystem_backup import run_backup, validate_config, REMOTE_PROGRAM
from lambda_artifact_backup import collect


class FakeTransport:
    def __init__(self):
        self.objects = {}
        self.corrupt = False
        self.mounted = True
        self.changed_bytes = 0
        self.after_verify = None
        self.receipt = None

    def factory(self, config, deadline):
        return self

    def prepare(self):
        if not self.mounted:
            raise ValueError('not mounted')

    def transfer(self, staging, names):
        for name in names:
            value = (staging / name).read_bytes()
            if self.objects.get(name) != value:
                self.changed_bytes += len(value)
                self.objects[name] = value

    def verify(self, objects, manifest):
        for item in objects:
            data = self.objects[item['object_key']] + (b'bad' if self.corrupt else b'')
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('remote corruption')
        data = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        sha = hashlib.sha256(data).hexdigest()
        key = 'manifests/' + sha + '.json'
        self.objects[key] = data
        if self.after_verify:
            self.after_verify()
        return {'verified': True, 'durable': True, 'mounted': True, 'manifest_object_key': key,
                'manifest_sha256': sha, 'objects_verified': len(objects)}

    def save_receipt(self, receipt):
        self.receipt = dict(receipt)


class FilesystemBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nas-filesystem-backup-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runs = self.root / 'runs'
        self.search = self.runs / 'cuda/search'
        self.search.mkdir(parents=True)
        self.checkpoint = self.search / 'best.pt'
        self.checkpoint.write_bytes(b'checkpoint')
        self.config = {'instance_id': 'task-instance', 'remote_host': '192.0.2.1', 'remote_user': 'ubuntu',
                       'remote_root': '/lambda/nfs/task-filesystem/task-instance', 'ssh_key_file': str(self.root / 'key'),
                       'known_hosts_file': str(self.root / 'known_hosts'), 'runs_root': str(self.runs),
                       'search_relative_path': 'cuda/search', 'receipt_file': str(self.root / 'receipt.json'),
                       'cache_file': str(self.root / 'cache.json'), 'staging_dir': str(self.root / 'stage')}
        (self.search / 'retained_checkpoints.json').write_text(json.dumps({
            'relative_paths': ['cuda/search/best.pt'], 'checkpoints': [{'relative_path': 'cuda/search/best.pt',
                'sha256': hashlib.sha256(b'checkpoint').hexdigest(), 'bytes': 10}]}))
        (self.search / 'log.jsonl').write_text('{"step":5000}\n')
        (self.root / 'cuda_model.py').write_text('# source\n')
        self.transport = FakeTransport()

    def run_backup(self, **changes):
        return run_backup(dict(self.config, **changes), self.transport.factory)

    def complete(self):
        report = self.search / 'FINAL_REPORT.json'
        report.write_text('{"status":"complete"}')
        sha = hashlib.sha256(report.read_bytes()).hexdigest()
        (self.search / 'finalization.json').write_text(json.dumps({'status': 'complete', 'final_report_sha256': sha}))
        control = self.search.with_name('search-control')
        control.mkdir()
        (control / 'workflow.json').write_text('{"status":"completed"}')
        return sha

    def test_final_backup_incremental_and_immutable_manifest(self):
        sha = self.complete()
        receipt = self.run_backup()
        self.assertTrue(receipt['verified'] and receipt['durable'] and receipt['backup_complete'])
        self.assertEqual((receipt['finalization_status'], receipt['final_report_sha256']), ('complete', sha))
        self.assertEqual(receipt, self.transport.receipt)
        before = self.transport.changed_bytes
        self.run_backup()
        self.assertEqual(self.transport.changed_bytes, before)
        manifest = json.loads(self.transport.objects[receipt['manifest_object_key']])
        self.assertIn('source/cuda_model.py', manifest['files'])
        self.assertIn('runs/cuda/search/best.pt', manifest['files'])
        (self.search / 'log.jsonl').write_text('{"step":5001}\n')
        self.run_backup()
        self.assertEqual(self.transport.changed_bytes - before, len('{"step":5001}\n'))

    def test_corruption_and_unmounted_filesystem_fail_closed_and_resume(self):
        self.transport.corrupt = True
        with self.assertRaisesRegex(ValueError, 'corruption'):
            self.run_backup()
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['verified'])
        self.transport.corrupt = False
        self.assertTrue(self.run_backup()['verified'])
        cache = json.loads(Path(self.config['cache_file']).read_text())
        staged = Path(self.config['staging_dir']) / cache['files']['runs/cuda/search/best.pt']['object_key']
        staged.chmod(0o600)
        staged.write_bytes(b'xxxxxxxxxx')
        self.assertTrue(self.run_backup()['verified'])
        self.assertEqual(staged.read_bytes(), b'checkpoint')
        self.transport.mounted = False
        with self.assertRaisesRegex(ValueError, 'mounted'):
            self.run_backup()
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['backup_complete'])

    def test_final_artifact_changes_reject_termination_receipt(self):
        self.complete()
        self.transport.after_verify = lambda: (self.search / 'log.jsonl').write_text('{"late":true}')
        with self.assertRaisesRegex(ValueError, 'changed during backup'):
            self.run_backup()
        self.assertFalse(json.loads(Path(self.config['receipt_file']).read_text())['backup_complete'])
        self.assertIsNone(self.transport.receipt)
        self.transport.after_verify = None
        self.assertEqual(self.run_backup()['finalization_status'], 'complete')

    def test_log_growing_after_collection_backs_up_current_prefix(self):
        path = self.search / 'log.jsonl'
        initial = path.read_bytes()
        def changing_collect(config):
            result = collect(config)
            with path.open('ab') as stream:
                stream.write(b'{"step":5001}\n')
            return result
        with patch('lambda_filesystem_backup.collect', side_effect=changing_collect):
            receipt = self.run_backup()
        manifest = json.loads(self.transport.objects[receipt['manifest_object_key']])
        item = manifest['files']['runs/cuda/search/log.jsonl']
        self.assertEqual(self.transport.objects[item['object_key']], initial + b'{"step":5001}\n')
        self.assertTrue(receipt['backup_complete'])
        self.assertEqual(receipt['finalization_status'], 'running')

    def test_paths_secrets_and_staging_limit(self):
        for field, value in [('remote_host', '192.0.2.1;cat /etc/passwd'), ('remote_user', 'ubuntu;whoami'),
                             ('remote_root', '/tmp/task-instance'), ('remote_root', '/lambda/nfs/task/other-instance'),
                             ('remote_root', '/lambda/nfs/task/../task-instance'), ('max_staging_bytes', 121 * 1024 ** 3)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_config(dict(self.config, **{field: value}))
        for name in ('train.json', 'validation.json', 'credentials.json', '.env.json'):
            (self.search / name).write_text('must never upload')
        self.run_backup()
        self.assertFalse(any(b'must never upload' in data for data in self.transport.objects.values()))
        with self.assertRaises(ValueError):
            self.run_backup(max_staging_bytes=1)

    def test_actual_remote_program_verifies_manifest_and_saves_receipt(self):
        """Execute unchanged program, virtualizing only its mounted root location."""
        remote = self.root / 'task-instance'
        remote.mkdir()
        (remote / 'objects').mkdir()
        data = b'remote content'
        sha = hashlib.sha256(data).hexdigest()
        (remote / 'objects' / sha).write_bytes(data)
        item = {'object_key': 'objects/' + sha, 'sha256': sha, 'bytes': len(data)}
        manifest = {'instance_id': 'task-instance', 'finalization_status': 'running',
                    'final_report_sha256': None, 'destination': 'test', 'files': {'a': item}}
        original_parts = PosixPath.parts
        def virtual_parts(path):
            parts = original_parts.__get__(path)
            return ('/', 'lambda', 'nfs', 'test', 'task-instance') if path == remote else parts
        def remote_call(request):
            output = io.StringIO()
            real_path = PosixPath
            def mapped_path(value):
                return self.root / 'private-receipt/receipt.json' if value == '/root/nas-backup/receipt.json' else real_path(value)
            with patch('pathlib.PosixPath.parts', property(virtual_parts)), patch('os.path.ismount', return_value=True), \
                 patch('pathlib.Path', side_effect=mapped_path), \
                 patch('sys.stdin', io.StringIO(json.dumps(dict(request, root=str(remote))))), contextlib.redirect_stdout(output):
                try:
                    exec(compile(REMOTE_PROGRAM, '<remote verifier>', 'exec'), {})
                except SystemExit as error:
                    if error.code:
                        raise
            return json.loads(output.getvalue())
        result = remote_call({'action': 'verify', 'objects': [item], 'manifest': manifest})
        self.assertTrue(result['verified'])
        self.assertTrue((remote / result['manifest_object_key']).is_file())
        receipt = dict(result, instance_id='task-instance', finalization_status='running', final_report_sha256=None,
                       destination='test', backup_complete=True, metadata_complete=True, retained_checkpoints_complete=True)
        self.assertTrue(remote_call({'action': 'receipt', 'receipt': receipt})['receipt_saved'])
        self.assertEqual(json.loads((self.root / 'private-receipt/receipt.json').read_text()), receipt)
        (remote / 'objects' / sha).write_bytes(b'corrupt')
        with self.assertRaises(SystemExit):
            remote_call({'action': 'verify', 'objects': [item], 'manifest': manifest})


if __name__ == '__main__':
    unittest.main()
