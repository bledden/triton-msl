"""Threaded per-file hashing must be invisible in the manifest and fail closed."""

import hashlib
import os
from pathlib import Path
import threading
import time

import pytest

import importlib.util

from triton_msl.backend import _toolchain_contract as candidate

_reference = Path(__file__).parent / "fixtures/parallel_inventory/serial_reference.py"
_spec = importlib.util.spec_from_file_location("_serial_inventory_reference", _reference)
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)

tp = candidate


def build(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/z.bin").write_bytes(os.urandom(300_000))
    (tmp_path / "a/y.txt").write_bytes(b"y" * 13)
    (tmp_path / "b").mkdir()
    (tmp_path / "b/empty").write_bytes(b"")
    (tmp_path / "b/big").write_bytes(bytes(range(256)) * 4000)
    (tmp_path / "b/link-to-a").symlink_to(tmp_path / "a")  # directory symlink: topology + contents again
    (tmp_path / "b/loop").symlink_to(tmp_path)  # back edge
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/x.pyc").write_bytes(b"pyc")
    (tmp_path / "skip-root").mkdir()
    (tmp_path / "skip-root/f").write_bytes(b"f")
    return tmp_path


@pytest.mark.parametrize("workers", [None, 1, 2, 3, 8, 64])
def test_records_identical_to_baseline_for_every_worker_count(tmp_path, workers):
    root = build(tmp_path)
    for options in ({}, {"ignore_bytecode": True, "exclude_root_names": ("skip-root",)}):
        expected = baseline._tree_manifest(root, **options)
        got = tp._tree_manifest(root, workers=workers, **options)
        assert got == expected
        assert [r for r in got if r[3] == "directory-back-edge"]


def test_default_policy_is_bounded_and_positive():
    assert 1 <= tp._inventory_workers(None) <= tp._INVENTORY_WORKER_LIMIT
    assert tp._inventory_workers(3) == 3
    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(ValueError):
            tp._inventory_workers(bad)


def test_every_file_is_still_opened_once_and_no_thread_survives(tmp_path, monkeypatch):
    root = build(tmp_path)
    opened = []
    real = Path.open

    def opening(path, *args, **kw):
        opened.append(str(path))
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)
    before = threading.active_count()
    records = tp._tree_manifest(root, workers=4)
    assert threading.active_count() == before
    assert sorted(opened) == sorted(str(root / r[0]) for r in records if r[3] == "file")
    assert len(opened) == sum(r[3] == "file" for r in records)


def test_change_during_hash_in_worker_refuses_whole_manifest(tmp_path, monkeypatch):
    root = build(tmp_path)
    target = root / "b/big"
    real = Path.open

    def opening(path, *args, **kw):
        handle = real(path, *args, **kw)
        if path == target:
            # Mutate after the pre-stat, while the worker holds the descriptor.
            time.sleep(0.01)
            os.utime(target, ns=(1, 1))
        return handle

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        tp._tree_manifest(root, workers=4)


def test_first_failure_in_traversal_order_is_raised_and_pool_is_drained(tmp_path, monkeypatch):
    root = build(tmp_path)
    real = Path.open
    calls = []

    def opening(path, *args, **kw):
        calls.append(path.name)
        if path.name == "y.txt":
            raise PermissionError("y")
        if path.name == "big":
            raise IsADirectoryError("big")
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)
    before = threading.active_count()
    with pytest.raises(PermissionError):  # a/y.txt precedes b/big in traversal order
        tp._tree_manifest(root, workers=8)
    assert threading.active_count() == before


def test_would_block_and_short_reads_per_worker_buffer(tmp_path, monkeypatch):
    files = {f"f{i:02d}": bytes([i]) * (262144 * (i % 3) + i) for i in range(12)}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    assert tp._tree_manifest(tmp_path, workers=6) == [
        (name, len(data), hashlib.sha256(data).hexdigest(), "file") for name, data in sorted(files.items())
    ]


def test_readonly_alias_reuse_under_threads_matches_serial_and_opens_every_alias(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    payload = os.urandom(300_000)  # above the pool threshold: aliases are pooled
    (root / "real").write_bytes(payload)
    for i in range(6):
        (root / f"alias{i}").symlink_to(root / "real")
    monkeypatch.setattr(tp, "_read_only_mount", lambda target: True)
    monkeypatch.setattr(baseline, "_read_only_mount", lambda target: True)
    opened = []
    real = Path.open

    def opening(path, *args, **kw):
        opened.append(str(path))
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)
    serial = baseline._tree_manifest(root, reuse_readonly_aliases=True)
    opened.clear()
    threaded = tp._tree_manifest(root, reuse_readonly_aliases=True, workers=4)
    assert threaded == serial
    assert len(opened) == 7  # every alias opened, reuse or not
    assert len({r[2] for r in threaded}) == 1 and threaded[0][2] == hashlib.sha256(payload).hexdigest()


def test_seeded_provider_digests_are_copied_not_shared(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "f").write_bytes(b"data")
    monkeypatch.setattr(tp, "_read_only_mount", lambda target: True)
    seeds = {}
    tp._tree_manifest(root, reuse_readonly_aliases=True, provider_digests=seeds, workers=4)
    assert seeds == {}


def test_empty_tree_and_unsupported_kind_refuse(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        tp._tree_manifest(tmp_path, workers=4)
    assert tp._tree_manifest(tmp_path, workers=4, allow_empty=True) == []
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(RuntimeError, match="unsupported"):
        tp._tree_manifest(tmp_path, workers=4)


def test_reentrant_and_concurrent_inventories_do_not_share_buffers(tmp_path):
    roots = []
    for i in range(3):
        r = tmp_path / f"r{i}"
        r.mkdir()
        for j in range(5):
            (r / f"f{j}").write_bytes(bytes([i * 16 + j]) * (100_000 + j * 7919))
        roots.append(r)
    expected = [baseline._tree_manifest(r) for r in roots]
    results = [None] * 3

    def run(i):
        results[i] = tp._tree_manifest(roots[i], workers=4)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == expected


def test_small_files_hash_in_calling_thread_and_large_files_in_workers(tmp_path, monkeypatch):
    (tmp_path / "small").write_bytes(b"s" * (tp._INVENTORY_POOL_MIN_BYTES - 1))
    (tmp_path / "large").write_bytes(b"l" * tp._INVENTORY_POOL_MIN_BYTES)
    threads = {}
    real = Path.open

    def opening(path, *args, **kw):
        threads[path.name] = threading.current_thread().name
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)
    tp._tree_manifest(tmp_path, workers=4)
    assert threads["small"] == threading.current_thread().name
    assert threads["large"].startswith("triton-msl-inventory")
    threads.clear()
    tp._tree_manifest(tmp_path, workers=1)
    assert set(threads.values()) == {threading.current_thread().name}


def test_traversal_failure_cancels_pool_and_raises_original(tmp_path, monkeypatch):
    (tmp_path / "a-large").write_bytes(b"a" * tp._INVENTORY_POOL_MIN_BYTES)
    os.mkfifo(tmp_path / "z-fifo")
    before = threading.active_count()
    with pytest.raises(RuntimeError, match="unsupported"):
        tp._tree_manifest(tmp_path, workers=4)
    assert threading.active_count() == before


# ---- F1 precedence controls (Astra review 2026-09-29): earliest failure in traversal order ----

BIG = 262144


def _fail_opens(monkeypatch, failures, gates=None):
    real = Path.open
    gates = gates or {}

    def opening(path, *args, **kw):
        if path.name in gates:
            assert gates[path.name].wait(5)
        if path.name in failures:
            raise failures[path.name]
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)


def _refuses_with(func, expected):
    before = {t.ident for t in threading.enumerate()}
    try:
        func()
    except BaseException as observed:
        assert {t.ident for t in threading.enumerate()} == before
        assert observed is expected, f"raised {type(observed).__name__}: {observed}"
    else:
        pytest.fail("did not refuse")


def test_worker_failure_before_inline_failure(tmp_path, monkeypatch):
    (tmp_path / "a-large").write_bytes(b"a" * BIG)
    (tmp_path / "z-small").write_bytes(b"z")
    first, second = PermissionError("first"), OSError("second")
    _fail_opens(monkeypatch, {"a-large": first, "z-small": second})
    _refuses_with(lambda: baseline._tree_manifest(tmp_path), first)
    _refuses_with(lambda: tp._tree_manifest(tmp_path, workers=2), first)


def test_worker_failure_before_walk_failure(tmp_path, monkeypatch):
    (tmp_path / "a-large").write_bytes(b"a" * BIG)
    os.mkfifo(tmp_path / "z-fifo")
    first = PermissionError("first")
    _fail_opens(monkeypatch, {"a-large": first})
    _refuses_with(lambda: baseline._tree_manifest(tmp_path), first)
    _refuses_with(lambda: tp._tree_manifest(tmp_path, workers=2), first)


def test_two_workers_reverse_completion_exposes_traversal_order(tmp_path, monkeypatch):
    (tmp_path / "a-large").write_bytes(b"a" * BIG)
    (tmp_path / "b-large").write_bytes(b"b" * BIG)
    first, second = PermissionError("a"), OSError("b")
    release = threading.Event()
    real = Path.open

    def opening(path, *args, **kw):
        if path.name == "b-large":
            release.set()  # b fails first in time; a is held until then
            raise second
        if path.name == "a-large":
            assert release.wait(5)
            raise first
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "open", opening)
    # Traversal order still wins: a precedes b.
    _refuses_with(lambda: tp._tree_manifest(tmp_path, workers=2), first)


def test_successful_workers_before_inline_failure(tmp_path, monkeypatch):
    for name in ("a-large", "b-large"):
        (tmp_path / name).write_bytes(name.encode() * (BIG // 7 + 1))
    (tmp_path / "z-small").write_bytes(b"z")
    later = OSError("inline")
    _fail_opens(monkeypatch, {"z-small": later})
    _refuses_with(lambda: baseline._tree_manifest(tmp_path), later)
    _refuses_with(lambda: tp._tree_manifest(tmp_path, workers=2), later)


def test_interrupt_in_walk_is_not_replaced_by_earlier_worker_failure(tmp_path, monkeypatch):
    (tmp_path / "a-large").write_bytes(b"a" * BIG)
    (tmp_path / "z-small").write_bytes(b"z")
    earlier, interrupt = PermissionError("a"), KeyboardInterrupt()
    _fail_opens(monkeypatch, {"a-large": earlier, "z-small": interrupt})
    _refuses_with(lambda: tp._tree_manifest(tmp_path, workers=2), interrupt)


def test_pooled_alias_owner_failure_releases_waiter_which_hashes_itself(tmp_path, monkeypatch):
    payload = b"p" * 300_000
    target = tmp_path / "a-large"
    target.write_bytes(payload)
    (tmp_path / "b-alias").symlink_to(target)
    monkeypatch.setattr(tp, "_read_only_mount", lambda value: True)
    real_hash = hashlib.sha256
    hashes, second_opened = [], threading.Event()
    real_open = Path.open

    def opening(path, *args, **kw):
        if path.name == "b-alias":
            second_opened.set()
        return real_open(path, *args, **kw)

    class Hash:
        def __init__(self):
            hashes.append(threading.current_thread().name)
            self.inner = real_hash()
            self.owner = len(hashes) == 1

        def update(self, chunk):
            if self.owner:
                assert second_opened.wait(5)
                raise OSError("owner read failed")
            self.inner.update(chunk)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(hashlib, "sha256", Hash)
    with pytest.raises(OSError, match="owner read failed"):
        tp._tree_manifest(tmp_path, workers=2, reuse_readonly_aliases=True)
    assert len(hashes) == 2  # the waiter was released and hashed the bytes itself
