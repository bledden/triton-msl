"""Full native reads may seed one read-only toolchain inventory, never an SDK."""

import hashlib
import mmap
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from triton_msl.backend import _native_dependencies as native, _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError
from test_native_dependency_contract import ARM64, X86_64, _command, _image


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "Toolchain"
    root.mkdir()
    (root / "tool").write_bytes(_image(file_type=2) + b"native payload")
    (root / "alias").symlink_to("tool")
    monkeypatch.setattr(tc, "_read_only_mount", lambda value: True)
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    return root


def collect(root, cache):
    return tc._toolchain_native_dependencies({"metal": root, "metallib": root}, {}, provider_digests=cache)


def inventory(root, cache, monkeypatch, *, reuse=True):
    reads = []
    real = hashlib.sha256

    def counted(data=b""):
        reads.append(True)
        return real(data)

    with monkeypatch.context() as patch:
        patch.setattr(tc.hashlib, "sha256", counted)
        rows = tc._tree_manifest(root, reuse_readonly_aliases=reuse, provider_digests=cache)
    return rows, len(reads)


@pytest.mark.parametrize("fat", [False, True])
def test_native_digest_seeds_inventory_without_erasing_aliases(bundle, monkeypatch, fat):
    if fat:
        arm, x86 = _image(file_type=2), _image(cpu=X86_64) + b"other-slice payload"
        payload = (
            struct.pack(">II", 0xCAFEBABE, 2)
            + struct.pack(">IIIII", ARM64, 0, 48, len(arm), 0)
            + struct.pack(">IIIII", X86_64, 0, 48 + len(arm), len(x86), 0)
            + arm
            + x86
        )
        (bundle / "tool").write_bytes(payload)
    cache = {}
    baseline_graph = collect(bundle, None)
    assert collect(bundle, cache) == baseline_graph
    assert len(cache) == 1
    before = dict(cache)
    baseline = tc._tree_manifest(bundle)
    rows, reads = inventory(bundle, cache, monkeypatch)
    assert rows == baseline
    assert [r[0] for r in rows] == ["alias", "tool"]
    assert reads == 0
    assert cache == before


@pytest.mark.parametrize("stage", ["producer", "consumer"])
@pytest.mark.parametrize("gate", ["root", "descriptor", "both"])
def test_both_ends_require_readonly_root_and_descriptor(bundle, monkeypatch, stage, gate):
    cache = {}

    def mount(value):
        return not (gate == "both" or (isinstance(value, int) == (gate == "descriptor")))

    if stage == "producer":
        monkeypatch.setattr(tc, "_read_only_mount", mount)
    collect(bundle, cache)
    if stage == "producer":
        assert cache == {}
        monkeypatch.setattr(tc, "_read_only_mount", lambda value: True)
    else:
        assert cache
        monkeypatch.setattr(tc, "_read_only_mount", mount)
    rows, reads = inventory(bundle, cache, monkeypatch)
    assert rows == tc._tree_manifest(bundle)
    assert reads == (1 if stage == "producer" else 2)


def test_external_readonly_provider_not_exported(bundle, tmp_path):
    external = tmp_path / "external.dylib"
    external.write_bytes(_image() + b"external")
    (bundle / "tool").write_bytes(_image(_command(0xC, str(external)), file_type=2))
    cache = {}
    graph = collect(bundle, cache)
    assert sum(len(rows) for rows in graph.values()) == 2
    assert set(cache) == {tc._file_state((bundle / "tool").stat())}


def test_writable_symlink_escape_is_reread_after_open_mapping_write(bundle, tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.write_bytes(_image(file_type=2) + b"old")
    (bundle / "escape").symlink_to(external)
    inode = external.stat().st_ino
    monkeypatch.setattr(
        tc, "_read_only_mount", lambda value: not isinstance(value, int) or os.fstat(value).st_ino != inode
    )
    cache = {}
    with external.open("r+b") as handle, mmap.mmap(handle.fileno(), 0) as mapping:
        collect(bundle, cache)
        assert tc._file_state(external.stat()) not in cache
        mapping[-3:] = b"new"
        rows, reads = inventory(bundle, cache, monkeypatch)
        assert reads == 1
        assert next(row[2] for row in rows if row[0] == "escape") == hashlib.sha256(mapping).hexdigest()


@pytest.mark.parametrize("change", ["bytes", "replace", "mtime"])
def test_changed_witness_between_stages_forces_read(bundle, monkeypatch, change):
    cache = {}
    collect(bundle, cache)
    path = bundle / "tool"
    if change == "bytes":
        path.write_bytes(path.read_bytes() + b"new")
    elif change == "replace":
        path.unlink()
        path.write_bytes(b"replacement")
    else:
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    rows, reads = inventory(bundle, cache, monkeypatch)
    assert reads == 1
    assert rows == tc._tree_manifest(bundle)


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"])
def test_producer_descriptor_disagreement_cannot_publish(bundle, monkeypatch, field):
    real = os.fstat

    def mismatch(fd):
        status = real(fd)
        values = {key: getattr(status, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
        values[field] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", mismatch)
    cache = {}
    collect(bundle, cache)
    assert cache == {}


@pytest.mark.parametrize("damage", ["parse", "mutation", "header"])
def test_failed_or_header_read_does_not_publish(bundle, monkeypatch, damage):
    path = bundle / "tool"
    seen = []
    if damage == "parse":
        path.write_bytes(b"not Mach-O")
    if damage == "mutation":
        real = native._parse

        def mutate(data, cpu):
            result = real(data, cpu)
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            return result

        monkeypatch.setattr(native, "_parse", mutate)
    if damage == "header":
        native._read_image(path, ARM64, digest=False, digest_observer=lambda *args: seen.append(args))
    else:
        with pytest.raises(ValueError):
            native._read_image(path, ARM64, digest=True, digest_observer=lambda *args: seen.append(args))
    assert seen == []


def test_unselected_payload_still_gets_full_inventory_read(bundle, monkeypatch):
    (bundle / "unused.dylib").write_bytes(_image() + b"unloaded")
    cache = {}
    collect(bundle, cache)
    assert len(cache) == 1
    rows, reads = inventory(bundle, cache, monkeypatch)
    assert reads == 1
    assert rows == tc._tree_manifest(bundle)


def test_no_opt_in_means_every_alias_is_read_even_with_seed(bundle, monkeypatch):
    cache = {}
    collect(bundle, cache)
    rows, reads = inventory(bundle, cache, monkeypatch, reuse=False)
    assert reads == 2
    assert rows == tc._tree_manifest(bundle)


@pytest.mark.parametrize("fail_first", [False, True])
def test_identity_cache_lifetime_and_sdk_role_even_at_same_root(bundle, monkeypatch, fail_first):
    for key in tc._EXTERNAL_SEARCH:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tc, "_snapshot", None)
    caches, calls = [], []
    roots = {"metal": bundle, "metallib": bundle, "sdk": bundle}

    def resolve(*, provider_digests):
        assert provider_digests == {}
        caches.append(provider_digests)
        graph = collect(bundle, provider_digests)
        if fail_first and len(caches) == 1:
            raise ValueError("late graph refusal")
        return roots, {"native_providers": graph}

    real = tc._tree_manifest

    def tree(root, **kw):
        calls.append(dict(kw))
        return real(root, **kw)

    monkeypatch.setattr(tc, "_resolve_toolchain", resolve)
    monkeypatch.setattr(tc, "_tree_manifest", tree)
    if fail_first:
        with pytest.raises(MetalNonRecoverableError, match="late graph refusal"):
            tc.toolchain_identity()
        assert tc._snapshot is None
    first = tc.toolchain_identity()
    assert tc.toolchain_identity() == first
    assert len(calls) == 2
    assert calls[0]["provider_digests"] is caches[-1]
    assert calls[1] == {}  # SDK must not inherit toolchain policy even at the same root.
    monkeypatch.setattr(tc, "_snapshot", None)
    assert tc.toolchain_identity() == first
    assert caches[-1] is not caches[-2]


def test_reuse_still_opens_every_alias(bundle, monkeypatch):
    cache = {}
    collect(bundle, cache)
    real = Path.open

    def deny(path, *args, **kwargs):
        if path.name == "tool":
            raise PermissionError("cannot open logical alias")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(PermissionError):
        tc._tree_manifest(bundle, reuse_readonly_aliases=True, provider_digests=cache)
