"""Restore a verified Lambda content-addressed backup using only the stdlib.

Example:
  python lambda_restore_backup.py --backup-root /lambda/nfs/NAME/INSTANCE_ID \
    --manifest manifests/SHA256.json --out /home/ubuntu/restored-nas

The output must not exist and its parent must already exist. All unique objects
are verified before output creation, then checked again while copying. A later
I/O failure may leave an incomplete output directory; retry into a new path.
No links, credentials, network operations, or executable metadata are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

SHA256 = re.compile(r"^[a-f0-9]{64}$")
MANIFEST_NAME = re.compile(r"^([a-f0-9]{64})\.json$")
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
BLOCK_BYTES = 4 * 1024 * 1024


def absolute_path(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("path traversal is forbidden")
    return path.absolute()


def components(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError("invalid relative artifact path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError("artifact path traversal is forbidden")
    return tuple(parts)


def open_under(directory_fd, parts, directory=False):
    """Walk with openat/O_NOFOLLOW, including every parent directory."""
    cursor = os.dup(directory_fd)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory or index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=cursor)
            os.close(cursor)
            cursor = child
        kind = os.fstat(cursor).st_mode
        if not (stat.S_ISDIR(kind) if directory else stat.S_ISREG(kind)):
            raise ValueError("backup entry has the wrong file type")
        result, cursor = cursor, None
        return result
    finally:
        if cursor is not None:
            os.close(cursor)


def open_directory(path):
    anchor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return open_under(anchor, path.parts[1:], directory=True)
    finally:
        os.close(anchor)


def unique_json(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON object key")
        result[name] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON value is forbidden")


def read_manifest(root_fd, parts):
    match = MANIFEST_NAME.fullmatch(parts[-1])
    if not match:
        raise ValueError("manifest basename must be its lowercase SHA256 plus .json")
    with os.fdopen(open_under(root_fd, parts), "rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds 64 MiB")
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_MANIFEST_BYTES or digest != match[1]:
        raise ValueError("manifest SHA256 does not match its basename")
    manifest = json.loads(raw, object_pairs_hook=unique_json, parse_constant=reject_constant)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1):
        raise ValueError("unsupported backup manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest has no artifact files")
    objects, names = {}, set(files)
    for name, item in files.items():
        parts = components(name)
        if len(parts) < 2 or parts[0] not in ("runs", "source"):
            raise ValueError("logical artifacts must be beneath runs/ or source/")
        if any("/".join(parts[:end]) in names for end in range(1, len(parts))):
            raise ValueError("artifact paths conflict as file and directory")
        if not isinstance(item, dict):
            raise ValueError("artifact metadata must be an object")
        sha, size = item.get("sha256"), item.get("bytes")
        if (not isinstance(sha, str) or not SHA256.fullmatch(sha)
                or isinstance(size, bool) or not isinstance(size, int) or size < 0
                or item.get("object_key") != "objects/" + sha):
            raise ValueError("invalid content address, SHA256, or byte count")
        if sha in objects and objects[sha] != size:
            raise ValueError("same object has conflicting byte counts")
        objects[sha] = size
    return manifest, digest, objects


def check_object(root_fd, digest, expected_bytes, output=None):
    actual, count = hashlib.sha256(), 0
    with os.fdopen(open_under(root_fd, ("objects", digest)), "rb") as source:
        if os.fstat(source.fileno()).st_size != expected_bytes:
            raise ValueError("backup object byte count does not match manifest")
        for block in iter(lambda: source.read(BLOCK_BYTES), b""):
            count += len(block)
            if count > expected_bytes:
                raise ValueError("backup object changed size during verification")
            actual.update(block)
            if output is not None:
                output.write(block)
    if count != expected_bytes or actual.hexdigest() != digest:
        raise ValueError("backup object SHA256 or byte count does not match manifest")


def output_parent(root_fd, parts):
    cursor = os.dup(root_fd)
    try:
        for part in parts:
            try:
                os.mkdir(part, 0o700, dir_fd=cursor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cursor)
            os.close(cursor)
            cursor = child
        result, cursor = cursor, None
        return result
    finally:
        if cursor is not None:
            os.close(cursor)


def restore(backup_root, manifest_path, out):
    root, output = absolute_path(backup_root), absolute_path(out)
    original_manifest = os.fspath(manifest_path)
    manifest_path = Path(manifest_path)
    if manifest_path.is_absolute():
        manifest_relative = manifest_path.relative_to(root).as_posix()
    else:
        manifest_relative = str(manifest_path)
    # Validate the original spelling as well, before pathlib normalizes dots.
    if not Path(manifest_path).is_absolute():
        components(original_manifest)
    if ".." in manifest_path.parts:
        raise ValueError("manifest traversal is forbidden")
    manifest_parts = components(manifest_relative)
    root_fd, parent_fd, output_fd = open_directory(root), None, None
    try:
        parent_fd = open_directory(output.parent)
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("restore output must not exist")
        manifest, digest, objects = read_manifest(root_fd, manifest_parts)
        for sha, size in sorted(objects.items()):
            check_object(root_fd, sha, size)
        os.mkdir(output.name, 0o700, dir_fd=parent_fd)
        output_fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        for name, item in sorted(manifest["files"].items()):
            parts = components(name)
            directory = output_parent(output_fd, parts[:-1])
            try:
                fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                with os.fdopen(fd, "wb") as stream:
                    check_object(root_fd, item["sha256"], item["bytes"], stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(directory)
        return {"status": "restored", "manifest_sha256": digest,
                "instance_id": manifest.get("instance_id"), "output": str(output),
                "files": len(manifest["files"]), "unique_objects": len(objects),
                "bytes": sum(item["bytes"] for item in manifest["files"].values())}
    finally:
        for fd in (output_fd, parent_fd, root_fd):
            if fd is not None:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = restore(args.backup_root, args.manifest, args.out)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Restore failed: {error}\nAn output directory created before an I/O failure may be incomplete.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
