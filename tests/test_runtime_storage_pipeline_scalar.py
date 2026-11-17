"""CPU-only runtime regression witnesses with recording buffers and pipelines."""

import struct
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from triton_msl.backend import driver, _cache_contract, _launch_contract, _launch_signature
from triton_msl.errors import MetalNonRecoverableError


META = (4, 1, 0, 32, None, False, None, None, None, None, None, None)


class _Buffer:
    def __init__(self, size):
        self.data = bytearray(size)

    def contents(self):
        return self

    def as_buffer(self, size):
        assert size <= len(self.data)
        return memoryview(self.data)[:size]


class _Pool:
    def __init__(self):
        self.acquired = []

    def acquire(self, size):
        buffer = _Buffer(size)
        self.acquired.append(buffer)
        return buffer, None, size

    def release(self, *args):
        pass

    def acquire_scalar(self, size):
        return _Buffer(size)

    def release_scalar(self, *args):
        pass


class _Utils:
    def __init__(self):
        self.buffer_pool = _Pool()
        self.calls = []
        self.groups = []

    def launch(self, pipeline, grid, group, buffers):
        self.calls.append((pipeline, tuple(buffers)))
        self.groups.append(tuple(group))


def _launcher(names, signatures, metadata=META, msl=None):
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.arg_names = names
    launcher.signature = signatures
    launcher.constexpr_indices = set()
    launcher._binding_plan = _launch_signature.make_binding_plan(names, signatures)
    launcher._execution_contract = "recording-runtime-stamp"
    launcher._packed_contract = _launch_contract._canonical(metadata)
    launcher._msl, launcher._msl_block_size, launcher.kernel_name = msl, 32, "record_only"
    return launcher


@pytest.fixture
def recording_host(monkeypatch):
    utils = _Utils()
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: "recording-runtime-stamp")
    monkeypatch.setattr(driver, "_get_utils", lambda: utils)
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: SimpleNamespace(available=lambda: False))
    return utils


@pytest.mark.parametrize("other_dtype,other_sig", [(torch.float32, "*fp32"), (torch.int64, "*i64")])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("outputs", [None, [], [0], [1], [0, 1]])
def test_mixed_fp64_storage_refuses_before_allocation(recording_host, other_dtype, other_sig, reverse, outputs):
    base = torch.zeros(4, dtype=torch.float64)
    args, sigs = [base, base.view(other_dtype)], ["*fp64", other_sig]
    if reverse:
        args.reverse()
        sigs.reverse()
    metadata = (*META[:4], outputs, *META[5:])
    launcher = _launcher(["a", "b"], dict(zip(["a", "b"], sigs)), metadata)
    with pytest.raises(MetalNonRecoverableError, match="aliased float64"):
        launcher(1, 1, 1, None, None, metadata, None, None, None, *args)
    assert recording_host.buffer_pool.acquired == []
    assert recording_host.calls == []
    assert torch.equal(base, torch.zeros_like(base))


def test_independent_fp64_storages_keep_conversion_path(recording_host):
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
    b = a.clone()
    launcher = _launcher(["a", "b"], dict(a="*fp64", b="*fp64"))
    launcher(1, 1, 1, None, None, META, None, None, None, a, b)
    buffers = recording_host.calls[0][1]
    assert buffers[0][0] is not buffers[1][0]
    assert [len(buf.data) for buf, _ in buffers] == [16, 16]
    assert torch.equal(a, b) and a.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("kind", ["duplicate", "partial", "strided"])
def test_unrepresentable_fp64_storage_still_refuses(recording_host, kind):
    base = torch.zeros(8, dtype=torch.float64)
    args = [base, base] if kind == "duplicate" else [base[1:] if kind == "partial" else base[::2]]
    names = [str(i) for i in range(len(args))]
    launcher = _launcher(names, {n: "*fp64" for n in names})
    with pytest.raises(MetalNonRecoverableError, match="float64"):
        launcher(1, 1, 1, None, None, META, None, None, None, *args)
    assert recording_host.buffer_pool.acquired == [] and recording_host.calls == []


def test_ordinary_dtype_aliases_keep_one_mirror_and_byte_offsets(recording_host):
    base = torch.arange(8, dtype=torch.float32)
    launcher = _launcher(["a", "b"], dict(a="*fp32", b="*i32"))
    launcher(1, 1, 1, None, None, META, None, None, None, base[1:], base.view(torch.int32)[2:])
    first, second = recording_host.calls[0][1]
    assert first[0] is second[0] and (first[1], second[1]) == (4, 8)
    assert len(recording_host.buffer_pool.acquired) == 1
    assert torch.equal(base, torch.arange(8, dtype=torch.float32))


class _Pipeline:
    def __init__(self, name):
        self.name = name

    def maxTotalThreadsPerThreadgroup(self):
        return 1024


class _Device:
    def __init__(self, name, mode="success"):
        self.name, self.mode = name, mode

    def newLibraryWithURL_error_(self, url, error):
        return self, None

    def newFunctionWithName_(self, name):
        return None if name.endswith("__mmdirect") and self.mode == "absent" else name

    def newComputePipelineStateWithFunction_error_(self, function, error):
        if function.endswith("__mmdirect") and self.mode == "failed":
            return None, "recorded direct pipeline failure"
        return _Pipeline(self.name + (" direct" if function.endswith("__mmdirect") else " staged")), None


@pytest.fixture
def pipeline_loader(monkeypatch):
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    monkeypatch.setattr(driver, "_PIPELINE_BUCKETS", {})
    loader = driver.MetalUtils.__new__(driver.MetalUtils)

    def load(name, mode="success"):
        loader._device = _Device(name, mode)
        return loader.load_binary("kernel", "record-only.metallib", 0, 0)[1]

    load.unload = lambda: loader.unload_module(loader._device)
    return load


def _invoke_matrix(pipeline, dims=(32, 32, 8), split=True):
    descriptor = dict(
        direct_name="kernel__mmdirect", direct_block=64, block_m=32, block_n=32, m_idx=0, n_idx=1, k_idx=2
    )
    metadata = (*META[:6], descriptor if split else None, *META[7:])
    launcher = _launcher(["M", "N", "K"], dict(M="i32", N="i32", K="i32"), metadata)
    launcher(1, 1, 1, None, pipeline, metadata, None, None, None, *dims)


def test_loader_owns_primary_until_matching_unload(pipeline_loader):
    primary = pipeline_loader("A")
    key = id(primary)
    reference = weakref.ref(primary)
    del primary
    assert reference() is not None
    pipeline_loader.unload()
    assert reference() is None
    assert key not in driver._PIPELINE_BUCKETS


@pytest.mark.parametrize("mode", ["absent", "failed"])
def test_colliding_foreign_pipeline_keeps_staged_fallback(monkeypatch, recording_host, pipeline_loader, mode):
    # Deterministic cache-key collision; no claim of actual PyObjC address reuse.
    monkeypatch.setattr(driver, "id", lambda _: 741, raising=False)
    first = pipeline_loader("A")
    second = pipeline_loader("B", mode)
    _invoke_matrix(second)
    assert recording_host.calls[-1][0] is second
    _invoke_matrix(first)
    assert recording_host.calls[-1][0].name == "A direct"


@pytest.mark.parametrize(
    "dims,split,direct",
    [
        ((32, 32, 8), True, True),
        ((31, 32, 8), True, False),
        ((32, 31, 8), True, False),
        ((32, 32, 7), True, False),
        ((32, 32, 8), False, False),
    ],
)
def test_matching_pipeline_keeps_alignment_and_metadata_checks(recording_host, pipeline_loader, dims, split, direct):
    primary = pipeline_loader("A")
    _invoke_matrix(primary, dims, split)
    selected = recording_host.calls[-1][0]
    assert selected.name == ("A direct" if direct else "A staged")
    # The twin runs at the width its descriptor carries, the staged entry at its own.
    assert recording_host.groups[-1] == ((64, 1, 1) if direct else (META[3], 1, 1))
    if not direct:
        assert selected is primary


@pytest.mark.parametrize(
    "change",
    [
        dict(direct_block=0),
        dict(direct_block=48),
        dict(direct_block=256),
        dict(direct_block=True),
        dict(direct_name=""),
        dict(block_m=0),
        dict(m_idx=3),
        dict(extra=1),
        "missing-width",
    ],
)
def test_unrecognised_twin_descriptor_keeps_staged_entry(recording_host, pipeline_loader, monkeypatch, change):
    primary = pipeline_loader("A")
    descriptor = dict(
        direct_name="kernel__mmdirect", direct_block=64, block_m=32, block_n=32, m_idx=0, n_idx=1, k_idx=2
    )
    if change == "missing-width":
        del descriptor["direct_block"]
    else:
        descriptor.update(change)
    metadata = (*META[:6], descriptor, *META[7:])
    launcher = _launcher(["M", "N", "K"], dict(M="i32", N="i32", K="i32"), metadata)
    launcher(1, 1, 1, None, primary, metadata, None, None, None, 32, 32, 8)
    assert recording_host.calls[-1][0] is primary
    assert recording_host.groups[-1] == (META[3], 1, 1)


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize(
    "value,eligible",
    [(0, True), ((1 << 63) - 1, True), (1 << 63, False), ((1 << 63) + 3, False), ((1 << 64) - 1, False)],
)
def test_u64_bridge_range_precedes_cached_packing(checked, value, eligible):
    pointer = torch.empty(1, device="cpu")
    launcher = _launcher(["p", "v"], dict(p="*fp32", v="u64"))
    args = (pointer, value)
    bound = _launch_signature.bind_arguments(args, launcher.arg_names, launcher.signature)
    assert bound[3][1] == struct.pack("<Q", value)
    assert driver._compile_shader_scalars_ok(launcher, args, checked=bound if checked else None) is eligible


@pytest.mark.parametrize(
    "declared,value",
    [
        ("i64", -(1 << 63)),
        ("i64", (1 << 63) - 1),
        ("u32", (1 << 32) - 1),
        ("i8", -128),
        ("u8", 255),
        ("i1", True),
        ("i1", False),
    ],
)
def test_signed_bridge_and_small_scalar_boundaries_still_admit(declared, value):
    launcher = _launcher(["v"], dict(v=declared))
    assert driver._compile_shader_scalars_ok(launcher, [value])


def test_high_u64_reaches_host_with_full_unsigned_bytes(monkeypatch, recording_host):
    events = []
    runtime = SimpleNamespace(
        available=lambda: True,
        is_unsupported=lambda _: False,
        get_library=lambda _: pytest.fail("high u64 entered compile_shader"),
        dispatch=lambda *a, **k: pytest.fail("high u64 dispatched"),
        mark_unsupported=lambda _: events.append("blacklist"),
    )
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: runtime)
    # All-scalar launch satisfies all_mps; the scalar gate must select the host.
    value = (1 << 63) + 3
    launcher = _launcher(["v"], dict(v="u64"), msl="record-only")
    launcher(1, 1, 1, None, None, META, None, None, None, value)
    assert recording_host.calls[0][1][0][0].data == struct.pack("<Q", value)
    assert events == []
