"""Offline restore integrity, traversal, symlink, and overwrite checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from lambda_restore_backup import restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nas-restore-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.backup = self.root / "backup"
        (self.backup / "objects").mkdir(parents=True)
        (self.backup / "manifests").mkdir()
        self.out = self.root / "restored"
        self.manifest = {"schema_version": 1, "instance_id": "task-instance", "files": {}}
        self.contents = {
            "runs/cuda/search/best.pt": b"\x00checkpoint\xff\x01",
            "runs/cuda/search/config.json": b'{"seed":10}\n',
            "source/cuda_model.py": b"# archived source\n",
            "runs/cuda/search/empty.log": b"",
        }
        for name, data in self.contents.items():
            self.add(name, data)
        self.path = self.write_manifest()

    def add(self, name, data):
        sha = hashlib.sha256(data).hexdigest()
        (self.backup / "objects" / sha).write_bytes(data)
        self.manifest["files"][name] = {"object_key": "objects/" + sha, "sha256": sha,
                                         "bytes": len(data), "prefix_snapshot": True}

    def write_manifest(self, raw=None):
        raw = raw if raw is not None else json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()
        sha = hashlib.sha256(raw).hexdigest()
        path = self.backup / "manifests" / (sha + ".json")
        path.write_bytes(raw)
        return path

    def run_restore(self, manifest=None, output=None, root=None):
        return restore(root or self.backup, manifest or self.path, output or self.out)

    def test_round_trip_binary_empty_shared_objects_and_relative_manifest(self):
        self.add("runs/cuda/another/best.pt", self.contents["runs/cuda/search/best.pt"])
        self.path = self.write_manifest()
        result = self.run_restore(manifest=self.path.relative_to(self.backup).as_posix())
        self.assertEqual((result["status"], result["files"], result["unique_objects"]), ("restored", 5, 4))
        for name, data in self.contents.items():
            self.assertEqual((self.out / name).read_bytes(), data)
        self.assertEqual((self.out / "runs/cuda/another/best.pt").read_bytes(), self.contents["runs/cuda/search/best.pt"])
        self.assertEqual({path.relative_to(self.out).as_posix() for path in self.out.rglob("*") if path.is_file()}, set(self.manifest["files"]))

    def test_manifest_checksum_before_json_or_output(self):
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            self.run_restore()
        self.assertFalse(self.out.exists())

    def test_object_hash_size_missing_all_fail_before_output(self):
        item = self.manifest["files"]["runs/cuda/search/best.pt"]
        path = self.backup / item["object_key"]
        original = path.read_bytes()
        for data in (b"x" * len(original), original + b"extra", None):
            with self.subTest(data=data):
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
                with self.assertRaises((ValueError, OSError)):
                    self.run_restore()
                self.assertFalse(self.out.exists())
                path.write_bytes(original)

    def test_unsafe_logical_paths_and_file_directory_conflicts(self):
        original = dict(self.manifest["files"])
        item = next(iter(original.values()))
        for name in ("../escaped", "/absolute", "runs/../escaped", "runs//bad", "runs/./bad",
                     "source\\bad", "runs/bad\nname", "other/file", "runs"):
            with self.subTest(name=name):
                self.manifest["files"] = {**original, name: item}
                self.path = self.write_manifest()
                with self.assertRaises(ValueError):
                    self.run_restore()
                self.assertFalse(self.out.exists())
        self.manifest["files"] = {"runs/item": item, "runs/item/child": item}
        self.path = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.run_restore()

    def test_bad_content_address_and_byte_count(self):
        item = self.manifest["files"]["runs/cuda/search/best.pt"]
        original = dict(item)
        for change in ({"object_key": "../outside"}, {"object_key": "objects/" + "a" * 64},
                       {"bytes": True}, {"bytes": -1}, {"sha256": "g" * 64}):
            with self.subTest(change=change):
                item.clear()
                item.update(original, **change)
                self.path = self.write_manifest()
                with self.assertRaises(ValueError):
                    self.run_restore()
                self.assertFalse(self.out.exists())

    def test_duplicate_json_keys_are_rejected(self):
        raw = b'{"schema_version":1,"schema_version":1,"files":{}}'
        self.path = self.write_manifest(raw)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_restore()

    def test_existing_output_directory_file_and_dangling_link_are_untouched(self):
        self.out.mkdir()
        sentinel = self.out / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertEqual(sentinel.read_text(), "keep")
        sentinel.unlink()
        self.out.rmdir()
        self.out.write_text("keep-file")
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertEqual(self.out.read_text(), "keep-file")
        self.out.unlink()
        self.out.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertTrue(self.out.is_symlink())

    def test_object_and_manifest_symlinks_are_rejected(self):
        item = self.manifest["files"]["runs/cuda/search/best.pt"]
        obj = self.backup / item["object_key"]
        real = self.root / "real-object"
        obj.rename(real)
        obj.symlink_to(real)
        with self.assertRaises(OSError):
            self.run_restore()
        self.assertFalse(self.out.exists())
        obj.unlink()
        real.rename(obj)
        manifest_real = self.root / "real-manifest"
        self.path.rename(manifest_real)
        self.path.symlink_to(manifest_real)
        with self.assertRaises(OSError):
            self.run_restore()
        self.assertFalse(self.out.exists())

    def test_special_object_is_rejected_without_waiting_for_a_writer(self):
        item = self.manifest["files"]["runs/cuda/search/best.pt"]
        obj = self.backup / item["object_key"]
        obj.unlink()
        os.mkfifo(obj)
        with self.assertRaisesRegex(ValueError, "file type"):
            self.run_restore()
        self.assertFalse(self.out.exists())

    def test_symlinked_roots_parent_directories_and_manifest_escape(self):
        alias = self.root / "backup-alias"
        alias.symlink_to(self.backup, target_is_directory=True)
        with self.assertRaises(OSError):
            self.run_restore(root=alias, manifest=self.path.relative_to(self.backup))
        outside = self.root / "outside"
        outside.mkdir()
        parent_alias = self.root / "output-alias"
        parent_alias.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            self.run_restore(output=parent_alias / "restored")
        self.assertEqual(list(outside.iterdir()), [])
        for manifest in (self.root / "elsewhere.json", "../escape.json"):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.run_restore(manifest=manifest)
        objects = self.backup / "objects"
        actual = self.root / "actual-objects"
        objects.rename(actual)
        objects.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(OSError):
            self.run_restore()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
