"""Frozen pre-parallel inventory reference for differential CPU tests.

Extracted from source SHA-256 8c4dc55a21e469e04c3c0757af8d8be1b89399ccee0b438f1daf37935688f620.
Not an alternate production implementation.
"""

import hashlib
import os
from pathlib import Path
import stat


def _read_only_mount(path_or_fd):
    """Unknown mount status must retain byte reads, never authorize reuse."""
    try:
        status = os.fstatvfs(path_or_fd) if isinstance(path_or_fd, int) else os.statvfs(path_or_fd)
        return bool(status.f_flag & os.ST_RDONLY)
    except (OSError, AttributeError, NotImplementedError):
        return False


def _file_state(status):
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _tree_manifest(
    root,
    *,
    ignore_bytecode=False,
    exclude_root_names=(),
    allow_empty=False,
    reuse_readonly_aliases=False,
    provider_digests=None,
):
    """Logical names + bytes, including directory-symlink topology and contents."""
    root = Path(root).resolve(strict=True)
    records = []
    # Only toolchain callers opt in. SDK/package inventories retain every read.
    # Metadata alone is insufficient: an open shared mmap can change bytes with
    # unchanged times. Read-only mounts supply the premise for alias reuse.
    reuse_readonly_aliases = reuse_readonly_aliases and _read_only_mount(root)
    # Seed only from full native reads in this identity construction. Copy so
    # inventories cannot export their reads to other stages or later calls.
    file_digests = dict(provider_digests or {}) if reuse_readonly_aliases else {}

    # Reuse storage, never content: every cache miss still reads through EOF.
    # Local to this invocation so concurrent/reentrant inventories cannot alias it.
    read_buffer = bytearray(256 * 1024)
    read_view = memoryview(read_buffer)

    def visit(path, rel, ancestors):
        before = path.stat()
        inode = before.st_dev, before.st_ino
        if stat.S_ISDIR(before.st_mode):
            if inode in ancestors:
                records.append((rel, 0, ancestors[inode], "directory-back-edge"))
                return
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if not rel and child.name in exclude_root_names:
                    continue
                if ignore_bytecode and (child.name == "__pycache__" or child.suffix in (".pyc", ".pyo")):
                    continue
                visit(child, f"{rel}/{child.name}" if rel else child.name, ancestors | {inode: rel})
        elif stat.S_ISREG(before.st_mode):
            state = None
            with path.open("rb", buffering=0) as handle:
                # A symlink may escape a read-only root to a writable mount.
                # Check the opened file, and still open/check every alias.
                if reuse_readonly_aliases and _read_only_mount(handle.fileno()):
                    state = _file_state(before)
                    if state != _file_state(os.fstat(handle.fileno())):
                        raise RuntimeError(f"toolchain input changed before hashing: {path}")
                digest = file_digests.get(state) if state is not None else None
                if digest is None:
                    hasher = hashlib.sha256()
                    while True:
                        count = handle.readinto(read_buffer)
                        if count is None:
                            raise BlockingIOError(f"toolchain input read would block: {path}")
                        if count == 0:
                            break
                        hasher.update(read_view[:count])
                    digest = hasher.hexdigest()
                if state is not None and state != _file_state(os.fstat(handle.fileno())):
                    raise RuntimeError(f"toolchain input changed while hashing: {path}")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise RuntimeError(f"toolchain input changed while hashing: {path}")
            if state is not None:
                if state != _file_state(after):
                    raise RuntimeError(f"toolchain input changed while hashing: {path}")
                file_digests[state] = digest
            records.append((rel, before.st_size, digest, "file"))
        else:
            raise RuntimeError(f"unsupported toolchain input kind: {path}")

    visit(root, "", {})
    if not records and not allow_empty:
        raise RuntimeError(f"empty toolchain input tree: {root}")
    return records
