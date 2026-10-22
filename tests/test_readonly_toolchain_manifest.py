"""CPU controls for read-only toolchain alias reuse and its writable boundary."""

import hashlib
import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton_msl.backend._toolchain_contract import _tree_manifest as original
from triton_msl.backend import _toolchain_contract as tc


def candidate(root, **kwargs):
    return tc._tree_manifest(root, reuse_readonly_aliases=True, **kwargs)


@pytest.fixture(autouse=True)
def readonly_mount_model(monkeypatch):
    # Temporary fixtures are writable; model the mount property only in these
    # unit controls. Real-mount equality is validated separately in host evidence.
    monkeypatch.setattr(tc, "_read_only_mount", lambda path_or_fd: True)


def test_aliases_keep_all_records_but_hash_one_physical_file(tmp_path, monkeypatch):
    target = tmp_path / "a"
    target.write_bytes(b"content")
    (tmp_path / "b").symlink_to("a")
    os.link(target, tmp_path / "c")
    expected = original(tmp_path)
    real = hashlib.sha256
    hashed = []

    def track(*args, **kwargs):
        hashed.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", track)
    assert candidate(tmp_path) == expected
    assert len(hashed) == 1
    assert len(expected) == 3
    assert candidate(tmp_path) == expected
    assert len(hashed) == 2  # no reuse across inventories


def test_equal_bytes_in_different_files_are_not_an_alias(tmp_path, monkeypatch):
    (tmp_path / "a").write_bytes(b"same")
    (tmp_path / "b").write_bytes(b"same")
    real = hashlib.sha256
    calls = []

    def track(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", track)
    candidate(tmp_path)
    assert len(calls) == 2


def test_directory_aliases_cycles_and_exclusions_are_unchanged(tmp_path):
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "file").write_bytes(b"data")
    (folder / "back").symlink_to("..", target_is_directory=True)
    (tmp_path / "b").symlink_to("a", target_is_directory=True)
    (tmp_path / "omit").mkdir()
    (tmp_path / "omit" / "x").write_bytes(b"ignored")
    (tmp_path / "cached.pyc").write_bytes(b"ignored")
    for options in [{}, {"ignore_bytecode": True, "exclude_root_names": ("omit",)}]:
        assert candidate(tmp_path, **options) == original(tmp_path, **options)


@pytest.mark.parametrize("mutation", ["content", "replacement"])
def test_alias_changed_between_visits_is_rehashed(tmp_path, monkeypatch, mutation):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"old")
    b.symlink_to("a")
    real = Path.stat
    changed = False

    def observe(path, *args, **kwargs):
        nonlocal changed
        if path == b and not changed:
            changed = True
            if mutation == "replacement":
                temp = tmp_path / "replacement"
                temp.write_bytes(b"new")
                os.replace(temp, a)
            else:
                a.write_bytes(b"new")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", observe)
    rows = candidate(tmp_path)
    assert rows == [
        ("a", 3, hashlib.sha256(b"old").hexdigest(), "file"),
        ("b", 3, hashlib.sha256(b"new").hexdigest(), "file"),
    ]


def test_changed_while_reading_refuses(tmp_path, monkeypatch):
    path = tmp_path / "a"
    path.write_bytes(b"old")
    real = hashlib.sha256

    class MutatingHash:
        def __init__(self):
            self.inner = real()

        def update(self, value):
            self.inner.update(value)
            path.write_bytes(b"changed")

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", MutatingHash)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        candidate(tmp_path)


def test_cached_alias_still_requires_open_permission(tmp_path, monkeypatch):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"data")
    b.symlink_to("a")
    real = Path.open

    def observe(path, *args, **kwargs):
        if path == b:
            raise PermissionError("alias denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    with pytest.raises(PermissionError, match="alias denied"):
        candidate(tmp_path)


@pytest.mark.parametrize("kind", ["empty", "dangling", "fifo"])
def test_unproved_inputs_still_refuse(tmp_path, kind):
    if kind == "dangling":
        (tmp_path / "a").symlink_to("absent")
    elif kind == "fifo":
        os.mkfifo(tmp_path / "a")
    for implementation in [original, candidate]:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            implementation(tmp_path)


@pytest.mark.parametrize(
    "root_readonly,file_readonly,opt_in,expected_reads",
    [
        (False, False, True, 3),  # Inverted pin: writable roots read every alias.
        (False, True, True, 3),
        (True, False, True, 3),  # Symlink/submount escape must not inherit root trust.
        (True, True, False, 3),  # SDK/package callers do not opt in.
        (True, True, True, 1),
    ],
)
def test_mount_boundary_hash_counts(tmp_path, monkeypatch, root_readonly, file_readonly, opt_in, expected_reads):
    target = tmp_path / "a"
    target.write_bytes(b"content")
    (tmp_path / "b").symlink_to("a")
    os.link(target, tmp_path / "c")
    monkeypatch.setattr(
        tc, "_read_only_mount", lambda value: file_readonly if isinstance(value, int) else root_readonly
    )
    real = hashlib.sha256
    hashed = []

    def track(*args, **kwargs):
        hashed.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", track)
    rows = tc._tree_manifest(tmp_path, reuse_readonly_aliases=opt_in)
    assert len(rows) == 3 and len(hashed) == expected_reads


@pytest.mark.parametrize("escaped_file", [False, True])
def test_open_shared_mapping_is_reread_on_writable_mount(tmp_path, monkeypatch, escaped_file):
    """APFS can leave both times unchanged until the mapping closes.

    Keep the writer open across the inventory: timestamp-only alias reuse would
    return the old digest on APFS. No mutation is possible on admitted mounts.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"old-bytes")
    b.symlink_to("a")
    monkeypatch.setattr(tc, "_read_only_mount", lambda value: escaped_file and not isinstance(value, int))
    real_stat = Path.stat
    changed = False
    with a.open("r+b") as writer, mmap.mmap(writer.fileno(), 0, access=mmap.ACCESS_WRITE) as mapping:

        def observe(path, *args, **kwargs):
            nonlocal changed
            if path == b and not changed:
                mapping[:] = b"new-bytes"
                changed = True
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", observe)
        rows = candidate(tmp_path)
        assert rows == [
            ("a", 9, hashlib.sha256(b"old-bytes").hexdigest(), "file"),
            ("b", 9, hashlib.sha256(b"new-bytes").hexdigest(), "file"),
        ]


@pytest.mark.parametrize("failure", [OSError, AttributeError, NotImplementedError])
@pytest.mark.parametrize("descriptor", [False, True])
def test_unknown_mount_status_does_not_authorize_reuse(monkeypatch, failure, descriptor):
    # Recover the actual helper because the common fixture models readonly mounts.
    helper = _actual_read_only_mount

    def fail(value):
        raise failure("mount status unavailable")

    monkeypatch.setattr(os, "fstatvfs" if descriptor else "statvfs", fail)
    assert not helper(42 if descriptor else Path("/unknown"))


@pytest.mark.parametrize("flags,expected", [(0, False), (os.ST_RDONLY, True), (os.ST_NOSUID, False)])
@pytest.mark.parametrize("descriptor", [False, True])
def test_mount_gate_uses_readonly_bit(monkeypatch, flags, expected, descriptor):
    monkeypatch.setattr(os, "fstatvfs" if descriptor else "statvfs", lambda value: SimpleNamespace(f_flag=flags))
    assert _actual_read_only_mount(42 if descriptor else Path("/modeled")) is expected


@pytest.mark.parametrize("same_root", [False, True])
def test_identity_opts_in_only_toolchain_roles(tmp_path, monkeypatch, same_root):
    toolchain, sdk = tmp_path / "Toolchain", tmp_path / "SDK"
    toolchain.mkdir()
    sdk.mkdir()
    roots = {"metal": toolchain, "metallib": toolchain, "sdk": toolchain if same_root else sdk}
    monkeypatch.setattr(tc, "_snapshot", None)
    monkeypatch.setattr(tc, "_resolve_toolchain", lambda **kwargs: (roots, {}))
    for name in tc._EXTERNAL_SEARCH:
        monkeypatch.delenv(name, raising=False)
    calls = []

    def manifest(root, **options):
        calls.append((root, options))
        return [("file", 1, "content-hash", "file")]

    monkeypatch.setattr(tc, "_tree_manifest", manifest)
    tc.toolchain_identity()
    assert calls == [(toolchain.resolve(), {"reuse_readonly_aliases": True}), (roots["sdk"].resolve(), {})]
    tc.toolchain_identity()
    assert len(calls) == 2  # No mount/inventory work on warm identity.


_actual_read_only_mount = tc._read_only_mount
