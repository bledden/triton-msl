"""CPU ownership and publication controls; all devices are recording stand-ins."""

import gc
import hashlib
import sys
import weakref
from types import SimpleNamespace

import pytest
import triton  # noqa: F401

from triton_msl.backend import driver, _pipeline_ownership as ownership
from triton_msl.backend._argument_abi import LayoutError


class Object:
    def maxTotalThreadsPerThreadgroup(self):
        return 1024

    def newFunctionWithName_(self, name):
        return name


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "_PIPELINE_BUCKETS", {})
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))


def bind(owner, primary, module, direct, binary=b"library"):
    return driver._bind_pipeline_binary(
        primary, "kernel", binary, owner=owner, module=module, direct=direct, loaded=True
    )


def valid(primary, binary=b"library", owner=None):
    driver._validate_packed_pipeline(primary, "kernel", hashlib.sha256(binary).hexdigest(), owner=owner)


@pytest.mark.parametrize(
    "order",
    [
        (b"library", "path", bytearray(b"library")),
        ("path", bytearray(b"library"), b"library"),
        (bytearray(b"library"), b"library", "path"),
    ],
)
def test_mixed_loads_count_once_and_never_grant_mutable_binary_proof(order):
    owner, module = driver.MetalUtils(), Object()
    primaries = [Object() for _ in order]
    directs = [Object() for _ in order]
    for binary, primary, direct in zip(order, primaries, directs):
        bind(owner, primary, module, direct, binary)
        assert driver._direct_pipeline(primary) is direct
        if type(binary) is bytes:
            valid(primary)
        else:
            with pytest.raises(LayoutError):
                valid(primary)
    assert owner._pipeline_ownership.modules[id(module)].count == 3
    for _ in range(2):
        owner.unload_module(module)
        assert all(driver._direct_pipeline(p) is d for p, d in zip(primaries, directs))
    owner.unload_module(module)
    assert not owner._pipeline_ownership.modules and not driver._PIPELINE_BUCKETS


@pytest.mark.parametrize("release", ["unload", "owner"])
@pytest.mark.parametrize("first", [0, 1])
def test_shared_primary_restores_latest_live_owner_and_binary_proof(release, first):
    owners = [driver.MetalUtils(), driver.MetalUtils()]
    modules = [Object(), Object()]
    primary, directs = Object(), [Object(), Object()]
    for i in range(2):
        bind(owners[i], primary, modules[i], directs[i], bytes([i]))
    assert driver._direct_pipeline(primary) is directs[1]
    valid(primary, b"\0", owner=owners[0])
    valid(primary, b"\1")
    with pytest.raises(LayoutError):
        valid(primary, b"\0")
    if release == "unload":
        owners[first].unload_module(modules[first])
    else:
        owners[first] = None
        gc.collect()
    assert driver._direct_pipeline(primary) is directs[1 - first]
    valid(primary, bytes([1 - first]))
    owners[1 - first].unload_module(modules[1 - first])
    assert driver._direct_pipeline(primary) is None
    with pytest.raises(LayoutError):
        valid(primary, bytes([1 - first]))


def test_same_owner_shared_primary_distinct_modules_restore_previous():
    owner, primary = driver.MetalUtils(), Object()
    modules, directs = [Object(), Object()], [Object(), Object()]
    for i in range(2):
        bind(owner, primary, modules[i], directs[i], bytes([i]))
    owner.unload_module(modules[1])
    assert driver._direct_pipeline(primary) is directs[0]
    valid(primary, b"\0")
    owner.unload_module(modules[0])
    assert not driver._PIPELINE_BUCKETS


def test_duplicate_module_retains_both_pairs_until_last_unload():
    owner, module = driver.MetalUtils(), Object()
    primaries, directs = [Object(), Object()], [Object(), Object()]
    for primary, direct in zip(primaries, directs):
        bind(owner, primary, module, direct)
    refs = [weakref.ref(p) for p in (*primaries, *directs)]
    del primary, direct, primaries, directs
    owner.unload_module(Object())
    owner.unload_module(module)
    assert all(r() is not None for r in refs)
    owner.unload_module(module)
    gc.collect()
    assert all(r() is None for r in refs)
    owner.unload_module(module)
    assert not driver._PIPELINE_BUCKETS


def test_long_lived_oldest_does_not_accumulate_dead_records():
    oldest, primary, module, direct = driver.MetalUtils(), Object(), Object(), Object()
    bind(oldest, primary, module, direct)
    for _ in range(200):
        later = driver.MetalUtils()
        bind(later, primary, Object(), Object())
        assert len(driver._PIPELINE_BUCKETS[id(primary)].records) == 2
        del later
        assert len(driver._PIPELINE_BUCKETS[id(primary)].records) == 1
        assert driver._direct_pipeline(primary) is direct
    oldest.unload_module(module)
    assert not driver._PIPELINE_BUCKETS


@pytest.mark.parametrize("boundary", ["records", "modules", "cache"])
@pytest.mark.parametrize("after", [False, True])
def test_failed_publication_restores_prior_acquisition(monkeypatch, boundary, after):
    owner, primary, module, direct = driver.MetalUtils(), Object(), Object(), Object()
    bind(owner, primary, module, direct)
    prior = owner._pipeline_ownership.modules[id(module)]
    failure = RuntimeError("publication boundary")

    class FailDict(dict):
        armed = True

        def __setitem__(self, key, value):
            if self.armed:
                self.armed = False
                if after:
                    super().__setitem__(key, value)
                raise failure
            super().__setitem__(key, value)

    if boundary == "records":
        bucket = driver._PIPELINE_BUCKETS[id(primary)]
        bucket.records = FailDict(bucket.records)
    elif boundary == "modules":
        owner._pipeline_ownership.modules = FailDict(owner._pipeline_ownership.modules)
    else:
        original = ownership.Bucket.cache

        def fail_cache(bucket, record):
            if after:
                original(bucket, record)
            raise failure

        monkeypatch.setattr(ownership.Bucket, "cache", fail_cache)
    newer = Object()
    newer_ref = weakref.ref(newer)
    try:
        bind(owner, primary, module, newer, b"new")
    except RuntimeError as observed:
        assert observed is failure
    else:
        pytest.fail("publication failure swallowed")
    failure.__traceback__ = None
    del newer
    gc.collect()
    assert newer_ref() is None
    current = owner._pipeline_ownership.modules[id(module)]
    assert current.count == 1 and current.records == prior.records
    assert driver._direct_pipeline(primary) is direct
    valid(primary)
    assert len(driver._PIPELINE_BUCKETS[id(primary)].records) == 1


@pytest.mark.parametrize("which", ["primary", "direct", "module"])
def test_finalizers_observe_detached_old_records_and_keep_replacements(which):
    owner = driver.MetalUtils()
    owner_ref = weakref.ref(owner)
    seen, keys, replacement = [], [], Object()

    class Finalizer(Object):
        def __del__(self):
            actual = owner_ref()
            seen.append(all(key not in driver._PIPELINE_BUCKETS for key in keys))
            bind(actual, replacement, replacement, replacement, b"replacement")

    primary = Finalizer() if which == "primary" else Object()
    direct = Finalizer() if which == "direct" else Object()
    module = Finalizer() if which == "module" else Object()
    keys.append(id(primary))
    module_key = id(module)
    bind(owner, primary, module, direct)
    del primary, direct, module
    # Resolve the module through its live owner, not a retained test handle.
    owner.unload_module(owner._pipeline_ownership.modules[module_key].module)
    gc.collect()
    assert seen == [True]
    valid(replacement, b"replacement")
    assert driver._direct_pipeline(replacement) is replacement


def test_nested_acquisition_during_group_preparation_is_not_overwritten(monkeypatch):
    owner, module, outer, nested = driver.MetalUtils(), Object(), Object(), Object()
    original = ownership.Group
    entered = False

    def reentrant(*args):
        nonlocal entered
        if not entered:
            entered = True
            bind(owner, nested, module, nested, b"nested")
        return original(*args)

    monkeypatch.setattr(ownership, "Group", reentrant)
    bind(owner, outer, module, outer, b"outer")
    assert owner._pipeline_ownership.modules[id(module)].count == 2
    valid(nested, b"nested")
    owner.unload_module(module)
    valid(outer, b"outer")
    valid(nested, b"nested")
    owner.unload_module(module)
    assert not driver._PIPELINE_BUCKETS


def test_lazy_registry_adopts_nested_successful_acquisition(monkeypatch):
    owner = driver.MetalUtils.__new__(driver.MetalUtils)
    module, outer, nested = Object(), Object(), Object()
    original = ownership.Registry
    entered = False

    def reentrant():
        nonlocal entered
        if not entered:
            entered = True
            bind(owner, nested, module, nested, b"nested")
        return original()

    monkeypatch.setattr(ownership, "Registry", reentrant)
    bind(owner, outer, module, outer, b"outer")
    assert owner._pipeline_ownership.modules[id(module)].count == 2
    valid(nested, b"nested")
    valid(outer, b"outer")
    owner.unload_module(module)
    valid(nested, b"nested")
    owner.unload_module(module)
    assert not driver._PIPELINE_BUCKETS
