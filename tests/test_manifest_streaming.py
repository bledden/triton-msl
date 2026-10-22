"""Streaming storage can be reused; file bytes, EOF and refusals cannot be skipped."""

import hashlib
from pathlib import Path

import pytest

from triton_msl.backend import _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize("size", [0, 1, 262143, 262144, 262145, 524311])
def test_complete_digest_at_buffer_boundaries(tmp_path, size):
    payload = (bytes(range(256)) * ((size + 255) // 256))[:size]
    path = tmp_path / "file"
    path.write_bytes(payload)
    assert tc._tree_manifest(tmp_path) == [("file", size, hashlib.sha256(payload).hexdigest(), "file")]


def test_tail_and_empty_file_do_not_hash_prior_buffer_contents(tmp_path):
    files = {"a-long": b"a" * 600_013, "b-short": b"b" * 13, "c-empty": b"", "d-final": b"last"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    assert tc._tree_manifest(tmp_path) == [
        (name, len(data), hashlib.sha256(data).hexdigest(), "file") for name, data in files.items()
    ]


def wrap_reads(monkeypatch, target, action):
    real = Path.open

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def readinto(self, buffer):
            return action(self.handle, buffer)

    def opening(path, *args, **kw):
        handle = real(path, *args, **kw)
        return Reader(handle) if path == target else handle

    monkeypatch.setattr(Path, "open", opening)


@pytest.mark.parametrize("step", [1, 8191])
def test_short_read_is_not_eof(tmp_path, monkeypatch, step):
    payload = bytes(range(251)) * (1100 if step > 1 else 2)
    path = tmp_path / "file"
    path.write_bytes(payload)
    counts = []

    def read(handle, buffer):
        data = handle.read(min(step, len(buffer)))
        buffer[:len(data)] = data
        counts.append(len(data))
        return len(data)

    wrap_reads(monkeypatch, path, read)
    assert tc._tree_manifest(tmp_path) == [("file", len(payload), hashlib.sha256(payload).hexdigest(), "file")]
    assert len(counts) > 2 and counts[-1] == 0 and sum(counts) == len(payload)


@pytest.mark.parametrize("failure", ["would_block", "io_error"])
@pytest.mark.parametrize("after_partial", [False, True])
def test_read_failure_cannot_publish_identity(tmp_path, monkeypatch, failure, after_partial):
    path = tmp_path / "file"
    path.write_bytes(b"complete bytes must be read")
    first = True

    def read(handle, buffer):
        nonlocal first
        if first and after_partial:
            first = False
            data = handle.read(3)
            buffer[:len(data)] = data
            return len(data)
        if failure == "would_block":
            return None
        raise OSError("injected read failure")

    wrap_reads(monkeypatch, path, read)
    monkeypatch.setattr(tc, "_snapshot", None)
    monkeypatch.setattr(tc, "_resolve_toolchain", lambda **kw: ({"sdk": tmp_path}, {}))
    for key in tc._EXTERNAL_SEARCH:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MetalNonRecoverableError, match="would block|injected read failure"):
        tc.toolchain_identity()
    assert tc._snapshot is None


def test_reentrant_inventory_has_independent_storage(tmp_path, monkeypatch):
    outer, inner = tmp_path / "outer", tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    (outer / "a").write_bytes(b"outer payload")
    (inner / "b").write_bytes(b"inner payload must not overwrite outer buffer")
    real = hashlib.sha256
    entered = False

    class Hash:
        def __init__(self, *args, **kw):
            self.inner = real(*args, **kw)

        def update(self, value):
            nonlocal entered
            if not entered:
                entered = True
                assert tc._tree_manifest(inner) == [
                    ("b", 45, real(b"inner payload must not overwrite outer buffer").hexdigest(), "file")
                ]
            self.inner.update(value)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", Hash)
    assert tc._tree_manifest(outer) == [("a", 13, real(b"outer payload").hexdigest(), "file")]
