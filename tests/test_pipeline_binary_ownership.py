"""Loader receipts must support non-weakrefable objects, without a Metal device."""

import hashlib
import gc
import weakref
from types import SimpleNamespace
import sys

import pytest
import triton  # noqa: F401 -- complete backend discovery before importing driver

from triton_msl.backend import driver
from triton_msl.backend._argument_abi import LayoutError


class NoWeakPipeline:
    __slots__ = ()


def test_nonweak_pipeline_has_a_binary_receipt():
    owner = driver.MetalUtils()
    pipeline = NoWeakPipeline()
    with pytest.raises(TypeError):
        weakref.ref(pipeline)
    driver._bind_pipeline_binary(pipeline, "packed", b"library", owner=owner)
    driver._validate_packed_pipeline(pipeline, "packed", hashlib.sha256(b"library").hexdigest(), owner=owner)
    owner.unload_module(pipeline)
    with pytest.raises(LayoutError):
        driver._validate_packed_pipeline(pipeline, "packed", hashlib.sha256(b"library").hexdigest(), owner=owner)


@pytest.mark.parametrize("kind", ["NSObject", "MetalOptions"])
def test_actual_pyobjc_receipt_without_device(kind):
    import Foundation
    import Metal

    obj = Foundation.NSObject.alloc().init() if kind == "NSObject" else Metal.MTLCompileOptions.alloc().init()
    with pytest.raises(TypeError):
        weakref.ref(obj)
    owner = driver.MetalUtils()
    driver._bind_pipeline_binary(obj, "packed", b"library", owner=owner)
    driver._validate_packed_pipeline(obj, "packed", hashlib.sha256(b"library").hexdigest(), owner=owner)
    owner.unload_module(obj)
    assert not owner._pipeline_ownership.modules


class Tracked:
    pass


def test_ownership_is_scoped_to_utils_and_teardown_releases_every_reference():
    first, second = driver.MetalUtils(), driver.MetalUtils()
    pipeline, library = Tracked(), Tracked()
    pipe_ref, lib_ref = weakref.ref(pipeline), weakref.ref(library)
    driver._bind_pipeline_binary(pipeline, "packed", b"library", owner=first, module=library)
    with pytest.raises(LayoutError):
        driver._validate_packed_pipeline(pipeline, "packed", hashlib.sha256(b"library").hexdigest(), owner=second)
    del pipeline, library
    gc.collect()
    assert pipe_ref() is not None and lib_ref() is not None
    del first
    gc.collect()
    assert pipe_ref() is None and lib_ref() is None


def test_duplicate_library_loads_retire_only_at_last_unload():
    owner = driver.MetalUtils()
    module = NoWeakPipeline()
    first, second = Tracked(), Tracked()
    keys = (id(first), id(second))
    refs = [weakref.ref(first), weakref.ref(second)]
    driver._bind_pipeline_binary(first, "one", b"one", owner=owner, module=module)
    driver._bind_pipeline_binary(second, "two", b"two", owner=owner, module=module)
    owner.unload_module(NoWeakPipeline())
    owner.unload_module(module)
    driver._validate_packed_pipeline(first, "one", hashlib.sha256(b"one").hexdigest(), owner=owner)
    driver._validate_packed_pipeline(second, "two", hashlib.sha256(b"two").hexdigest(), owner=owner)
    del first, second
    owner.unload_module(module)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not owner._pipeline_ownership.modules
    assert not any(key in driver._PIPELINE_BUCKETS for key in keys)
    owner.unload_module(module)


def test_finalizer_reentry_sees_complete_retirement_and_keeps_new_record():
    owner = driver.MetalUtils()
    module = NoWeakPipeline()
    seen = []
    replacement = NoWeakPipeline()

    class Reentrant:
        def __del__(self):
            seen.append(
                (
                    sum(len(g.records) for g in owner._pipeline_ownership.modules.values()),
                    len(owner._pipeline_ownership.modules),
                )
            )
            owner.unload_module(module)
            driver._bind_pipeline_binary(replacement, "new", b"new", owner=owner)

    first, second = Reentrant(), Tracked()
    driver._bind_pipeline_binary(first, "one", b"one", owner=owner, module=module)
    driver._bind_pipeline_binary(second, "two", b"two", owner=owner, module=module)
    del first, second
    owner.unload_module(module)
    assert seen == []
    owner.unload_module(module)
    gc.collect()
    assert seen == [(0, 0)]
    driver._validate_packed_pipeline(replacement, "new", hashlib.sha256(b"new").hexdigest(), owner=owner)


def test_loader_receipt_reaches_dispatch_with_different_utils(monkeypatch, tmp_path):
    class NonWeakMetalPipeline:
        __slots__ = ()

        def maxTotalThreadsPerThreadgroup(self):
            return 1024

    pipeline = NonWeakMetalPipeline()

    class Device:
        def newLibraryWithURL_error_(self, *args):
            return self, None

        def newFunctionWithName_(self, name):
            return None if name.endswith("__mmdirect") else name

        def newComputePipelineStateWithFunction_error_(self, *args):
            return pipeline, None

    loader, dispatcher = driver.MetalUtils(), driver.MetalUtils()
    loader._device = Device()
    monkeypatch.setattr(driver, "_get_utils", lambda: dispatcher)
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))
    library, loaded, *_ = loader.load_binary("packed", b"library", 0)
    assert loaded is pipeline
    with pytest.raises(TypeError):
        weakref.ref(loaded)
    assert not dispatcher._pipeline_ownership.modules
    driver._validate_packed_pipeline(loaded, "packed", hashlib.sha256(b"library").hexdigest())
    loader.unload_module(library)
    assert id(loaded) not in driver._PIPELINE_BUCKETS
    with pytest.raises(LayoutError):
        driver._validate_packed_pipeline(loaded, "packed", hashlib.sha256(b"library").hexdigest())


def test_repeated_load_unload_does_not_accumulate_receipts():
    owner = driver.MetalUtils()
    initial = set(driver._PIPELINE_BUCKETS.keys())
    for _ in range(100):
        pipeline, module = NoWeakPipeline(), NoWeakPipeline()
        driver._bind_pipeline_binary(pipeline, "packed", b"library", owner=owner, module=module)
        owner.unload_module(module)
        assert not owner._pipeline_ownership.modules
    assert set(driver._PIPELINE_BUCKETS.keys()) <= initial
