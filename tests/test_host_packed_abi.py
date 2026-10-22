"""CPU recording controls for host Packed materialization, not producer admission."""

import gc
import hashlib
import struct
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
import triton.backends

from triton_msl.backend import _argument_abi, _cache_contract, _host_dispatch, _host_extent, _launch_contract, driver
from triton_msl.errors import MetalNonRecoverableError, PostSubmitError


class Buffer:
    def __init__(self, size, scalar=False, fail=None):
        self.data = bytearray(size)
        self.scalar = scalar
        self.fail = fail

    def contents(self):
        return self

    def as_buffer(self, size):
        if self.scalar and self.fail is not None:
            raise self.fail
        assert size <= len(self.data)
        return memoryview(self.data)[:size]


class Pipeline:
    pass


class RecordingUtils(SimpleNamespace):
    pass


class Pool:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.failure = RuntimeError("controlled Packed failure")
        self.mode = None

    def acquire(self, size):
        row = (Buffer(size), object(), size)
        self.acquired.append(row)
        return row

    def acquire_scalar(self, size):
        if self.mode == "acquire":
            raise self.failure
        buf = Buffer(size, scalar=True, fail=self.failure if self.mode == "fill" else None)
        self.acquired.append(("scalar", buf, size))
        return buf

    def release(self, *row):
        self.released.append(row)

    def release_scalar(self, buf, size):
        self.released.append(("scalar", buf, size))


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setattr(_host_dispatch, "_retained", {})
    # These are modeled marshalling controls. Primary separately tests real
    # construction/sealing; this does NOT grant a source certificate to Layout.
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda _: None)
    monkeypatch.setattr(_launch_contract, "validate_packed_launch", lambda metadata, _: metadata)
    monkeypatch.setattr(_host_extent, "select", lambda *a, **k: ("mirror", a[2], "controlled mirror"))
    pool = Pool()
    utils = RecordingUtils(buffer_pool=pool, calls=[], action=lambda _: None, bridge_calls=[])

    def launch(pipeline, grid, group, buffers):
        utils.calls.append((pipeline, grid, group, tuple(buffers)))
        utils.action(buffers)

    utils.launch = launch
    monkeypatch.setattr(driver, "_get_utils", lambda: utils)

    def bridge():
        utils.bridge_calls.append("bridge")
        raise AssertionError("host recording route must not enter compile_shader")

    monkeypatch.setattr(driver, "_get_compile_shader_runtime", bridge)
    return utils


def panel(count):
    declarations = ["i32", "*fp32", "fp32", "*fp32", "*fp32", "i1", "*fp32", "u32", "*fp32", "*fp32", "*fp32"]
    declarations.extend(["fp32"] + ["i32"] * (count - 12))
    values = []
    words = []
    for i, ty in enumerate(declarations):
        if ty.startswith("*"):
            values.append(torch.arange(4, dtype=torch.float32) + i)
        else:
            value = (
                -7
                if i == 0
                else 1
                if i == 2
                else True
                if ty == "i1"
                else 0xFFFFFFFF
                if ty == "u32"
                else -0.0
                if ty == "fp32"
                else i
            )
            values.append(value)
            words.append(
                struct.pack("<f", value)
                if ty == "fp32"
                else struct.pack("<I", int(value))
                if ty in ("u32", "i1")
                else struct.pack("<i", value)
            )
    return tuple(declarations), values, b"".join(words)


def launcher(declarations, *, layout=True):
    result = driver.MetalLauncher.__new__(driver.MetalLauncher)
    result.arg_names = [f"a{i}" for i in range(len(declarations))]
    result.signature = dict(zip(result.arg_names, declarations))
    result.constexpr_indices = set()
    result._msl = None
    result._host_packed_layout = _argument_abi.make_layout(declarations, kind="uint32-packed-v1") if layout else None
    result.kernel_name = "packed_test"
    result._host_packed_binary_sha = hashlib.sha256(b"fixture-binary").hexdigest()
    result._pipeline = Pipeline()
    driver._bind_pipeline_binary(result._pipeline, "packed_test", b"fixture-binary")
    return result


def call(launch, args, *, outputs=(1, 3), native_grid=True):
    metadata = (4, 1, 0, 32, list(outputs), native_grid, None, None, None, None, None, None)
    launch(2, 3, 4, None, launch._pipeline, metadata, None, None, None, *args)


@pytest.mark.parametrize("count", [32, 36])
def test_nonweak_pipeline_reaches_recording_host_dispatch(host, count):
    class NonWeakPipeline:
        __slots__ = ()

    declarations, args, expected = panel(count)
    target = launcher(declarations)
    driver.MetalUtils.unload_module(host, target._pipeline)
    target._pipeline = NonWeakPipeline()
    with pytest.raises(TypeError):
        weakref.ref(target._pipeline)
    driver._bind_pipeline_binary(target._pipeline, target.kernel_name, b"fixture-binary")
    call(target, args)
    assert len(host.calls) == 1 and host.calls[0][0] is target._pipeline
    bindings = host.calls[0][3]
    assert len(bindings) == 8 and bytes(bindings[-1][0].data) == expected


def write_float(binding, value):
    buffer, offset = binding
    buffer.data[offset : offset + 4] = struct.pack("<f", value)


@pytest.mark.parametrize("count", [32, 36])
def test_packed_interleaved_words_and_original_output_indices(host, count):
    declarations, args, expected = panel(count)
    before = {i: arg.clone() for i, arg in enumerate(args) if declarations[i].startswith("*")}

    def mutate(bindings):
        write_float(bindings[0], 111)
        write_float(bindings[1], 222)
        write_float(bindings[2], 999)  # Negative copyback-selection control.

    host.action = mutate
    call(launcher(declarations), args)
    _, grid, group, bindings = host.calls[0]
    assert grid == (2, 3, 4) and group == (32, 1, 1)
    assert len(bindings) == 8 and bindings[7][1] == 0
    assert bytes(bindings[7][0].data) == expected and len(expected) == (count - 7) * 4
    assert len(host.buffer_pool.acquired) == 8
    assert [row[2] for row in host.buffer_pool.acquired if row[0] == "scalar"] == [len(expected)]
    for physical, logical in enumerate((1, 3, 4, 6, 8, 9, 10)):
        assert bindings[physical][0] is host.buffer_pool.acquired[physical][0]
        wanted = before[logical]
        if logical in (1, 3):
            wanted[0] = 111 if logical == 1 else 222
        assert torch.equal(args[logical], wanted)
    assert host.buffer_pool.released == host.buffer_pool.acquired
    assert host.bridge_calls == [] and not _host_dispatch._retained


@pytest.mark.parametrize("same_offset", [False, True])
def test_packed_shared_storage_offsets_and_alias_output_order(host, same_offset):
    declarations, args, _ = panel(36)
    base = torch.arange(130, dtype=torch.float32)
    wanted = base.clone()
    args[1] = base[1:65]
    args[3] = base[1:65] if same_offset else base[65:129]

    def mutate(bindings):
        assert bindings[0][0] is bindings[1][0]
        assert [bindings[0][1], bindings[1][1]] == [4, 4 if same_offset else 260]
        write_float(bindings[1], 222)
        write_float(bindings[0], 111)

    host.action = mutate
    call(launcher(declarations), args)
    wanted[1] = 111
    if not same_offset:
        wanted[65] = 222
    assert torch.equal(base, wanted)
    assert len(host.buffer_pool.acquired) == 7  # Six storage groups plus Packed.
    assert host.buffer_pool.released == host.buffer_pool.acquired
    assert len({id(row[1] if row[0] == "scalar" else row[0]) for row in host.buffer_pool.released}) == 7


@pytest.mark.parametrize("count", [32, 36])
@pytest.mark.parametrize("mode", ["acquire", "fill"])
def test_packed_acquisition_and_fill_failures_release_safe_owners(host, count, mode):
    declarations, args, _ = panel(count)
    before = [value.clone() for value in args if torch.is_tensor(value)]
    host.buffer_pool.mode = mode
    with pytest.raises(RuntimeError) as raised:
        call(launcher(declarations), args)
    assert raised.value is host.buffer_pool.failure
    assert host.calls == [] and host.bridge_calls == [] and not _host_dispatch._retained
    assert len(host.buffer_pool.acquired) == (7 if mode == "acquire" else 8)
    assert host.buffer_pool.released == host.buffer_pool.acquired
    assert all(torch.equal(a, b) for a, b in zip((v for v in args if torch.is_tensor(v)), before))


@pytest.mark.parametrize("reason", ["missing", "null", "mismatch"])
def test_packed_refusal_precedes_pool_allocation(host, reason):
    declarations, args, _ = panel(32)
    target = launcher(declarations, layout=reason != "missing")
    if reason == "null":
        args[1] = None
    if reason == "mismatch":
        changed = list(declarations)
        changed[0] = "fp32"
        target._host_packed_layout = _argument_abi.make_layout(tuple(changed), kind="uint32-packed-v1")
    with pytest.raises(MetalNonRecoverableError):
        call(target, args)
    assert host.buffer_pool.acquired == [] and host.calls == [] and host.bridge_calls == []


@pytest.mark.parametrize("reason", ["missing", "foreign", "entry", "binary"])
def test_packed_pipeline_binding_refusal_precedes_pool_allocation(host, reason):
    declarations, args, _ = panel(36)
    target = launcher(declarations)
    if reason == "missing":
        target._pipeline = None
    elif reason == "foreign":
        target._pipeline = Pipeline()
    elif reason == "entry":
        target.kernel_name = "different_entry"
    else:
        target._host_packed_binary_sha = hashlib.sha256(b"different-binary").hexdigest()
    with pytest.raises(MetalNonRecoverableError):
        call(target, args)
    assert host.buffer_pool.acquired == [] and host.calls == [] and host.bridge_calls == []


@pytest.mark.parametrize("failure", ["commit", "wait", "status", "interrupt"])
def test_uncertain_real_runtime_retains_packed_and_blocks_replay(host, monkeypatch, failure):
    monkeypatch.setitem(
        sys.modules,
        "Metal",
        SimpleNamespace(MTLSizeMake=lambda *x: x, MTLCommandBufferStatusCompleted=4, MTLCommandBufferStatusError=5),
    )
    error = KeyboardInterrupt("controlled interrupt") if failure == "interrupt" else RuntimeError(failure)
    events, encoded = [], []

    class Encoder:
        def setComputePipelineState_(self, _):
            pass

        def setBuffer_offset_atIndex_(self, buf, offset, index):
            encoded.append((buf, offset, index))

        def dispatchThreadgroups_threadsPerThreadgroup_(self, grid, group):
            assert grid == (2, 3, 4) and group == (32, 1, 1)
            # Distinguish forbidden copyback from a vacuous unchanged mirror.
            write_float(encoded[0][:2], 111)
            write_float(encoded[1][:2], 222)

        def endEncoding(self):
            pass

    class Command:
        final = False
        readable = failure != "status"

        def computeCommandEncoder(self):
            return Encoder()

        def commit(self):
            events.append("commit")
            if failure == "commit":
                raise error

        def waitUntilCompleted(self):
            if failure in ("wait", "interrupt"):
                raise error
            self.final = True

        def status(self):
            if not self.readable:
                raise error
            return 4 if self.final else 3

    command = Command()
    host.command_queue = SimpleNamespace(commandBuffer=lambda: command)
    host.launch = lambda *a: driver.MetalUtils.launch(host, *a)
    declarations, args, _ = panel(36)
    target = launcher(declarations)
    before = [v.clone() for v in args if torch.is_tensor(v)]
    with pytest.raises(type(error)) as raised:
        call(target, args)
    assert raised.value is error
    error.__traceback__ = None
    del raised
    lease = next(iter(_host_dispatch._retained.values()))
    assert lease.stage == "submission_attempted" and lease.command is command
    assert [entry[2] for entry in encoded] == list(range(8))
    assert lease.buffers[7][0] is encoded[7][0] and lease.buffers[7][0].scalar
    assert lease.owners[3][-1] == ("scalar", lease.buffers[7][0], 116)
    packed = weakref.ref(lease.buffers[7][0])
    assert host.buffer_pool.released == [] and host.bridge_calls == []
    assert all(torch.equal(a, b) for a, b in zip((v for v in args if torch.is_tensor(v)), before))
    count = len(host.buffer_pool.acquired)
    with pytest.raises(PostSubmitError, match="unconfirmed completion"):
        call(target, args)
    assert len(host.buffer_pool.acquired) == count and events == ["commit"]
    assert _host_dispatch.reap_completed() == 1
    # Remove recording logs as alternative owners: only retained runtime state
    # should keep the Packed buffer alive, not fixture bookkeeping/tracebacks.
    host.buffer_pool.acquired.clear()
    encoded.clear()
    del lease
    error.__traceback__ = None
    gc.collect()
    assert packed() is not None
    command.final, command.readable = True, True
    assert _host_dispatch.reap_completed() == 0
    error.__traceback__ = None
    gc.collect()
    assert packed() is None and host.buffer_pool.released == []
    assert events == ["commit"] and host.bridge_calls == []


def test_plain_direct_host_route_keeps_original_bindings(host):
    target = launcher(("*fp32", "*fp32", "i32"), layout=False)
    x, y = torch.ones(4), torch.zeros(4)
    call(target, [x, y, 4], outputs=(1,), native_grid=False)
    _, grid, group, bindings = host.calls[0]
    assert grid == (24, 1, 1) and group == (32, 1, 1)
    assert len(bindings) == 3 and bytes(bindings[2][0].data) == struct.pack("<i", 4)
    assert host.buffer_pool.released == host.buffer_pool.acquired
    assert host.bridge_calls == [] and not _host_dispatch._retained


def test_pipeline_registry_owns_pipeline_until_matching_unload(host):
    pipeline = Pipeline()
    key, observed = id(pipeline), weakref.ref(pipeline)
    driver._bind_pipeline_binary(pipeline, "packed_test", b"fixture-binary")
    assert driver._PIPELINE_BUCKETS[key].latest("binary").primary is pipeline
    del pipeline
    gc.collect()
    assert observed() is not None
    driver.MetalUtils.unload_module(host, observed())
    gc.collect()
    assert observed() is None and key not in driver._PIPELINE_BUCKETS


def test_stale_pipeline_identity_refuses_even_with_matching_digest(host):
    declarations, args, _ = panel(32)
    target = launcher(declarations)
    key = id(target._pipeline)
    replacement = Pipeline()
    substituted = (replacement, target.kernel_name, target._host_packed_binary_sha, replacement)
    # Model a reused identity/address deterministically instead of waiting for
    # nondeterministic allocator id reuse. Referent identity must be checked.
    record = driver._PIPELINE_BUCKETS[key].latest("binary")
    record.proof = substituted
    with pytest.raises(MetalNonRecoverableError):
        call(target, args)
    assert host.buffer_pool.acquired == [] and host.calls == []
    assert record.proof is substituted


@pytest.mark.parametrize("count", [32, 36])
def test_packed_narrower_actual_storage_refuses_before_allocation(host, count):
    declarations, args, _ = panel(count)
    args[1] = torch.zeros(4, dtype=torch.float16)
    assert args[1].untyped_storage().nbytes() == 8  # Four source fp32 elements need16 bytes.
    with pytest.raises(MetalNonRecoverableError, match="backing width"):
        call(launcher(declarations), args)
    assert host.buffer_pool.acquired == [] and host.calls == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_packed_same_width_reinterpretation_keeps_storage_bytes(host, wrapped):
    import triton.language as tl
    from triton.runtime.jit import TensorWrapper

    declarations, args, _ = panel(32)
    backing = torch.tensor([0x3F800000, 0x40000000, 0x40400000, 0x40800000], dtype=torch.int32)
    args[1] = TensorWrapper(backing, tl.float32) if wrapped else backing
    wanted = bytes(backing.view(torch.uint8).tolist())
    call(launcher(declarations), args)
    bindings = host.calls[0][3]
    assert bytes(bindings[0][0].data) == wanted
    assert torch.equal(backing, torch.tensor([0x3F800000, 0x40000000, 0x40400000, 0x40800000], dtype=torch.int32))
    assert host.buffer_pool.released == host.buffer_pool.acquired


def test_packed_width_changing_wrapper_refuses_before_allocation(host):
    import triton.language as tl
    from triton.runtime.jit import TensorWrapper

    declarations, args, _ = panel(36)
    args[1] = TensorWrapper(torch.zeros(4, dtype=torch.float16), tl.float32)
    with pytest.raises(MetalNonRecoverableError, match="backing width"):
        call(launcher(declarations), args)
    assert host.buffer_pool.acquired == [] and host.calls == []


def test_real_scalar_pool_packed_capacity_and_size_keyed_reuse(monkeypatch):
    from triton_msl.buffer_pool import MetalBufferPool

    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLResourceStorageModeShared=0))
    allocations = []

    class Device:
        def newBufferWithLength_options_(self, size, options):
            assert options == 0
            allocations.append(size)
            return Buffer(size, scalar=True)

    pool = MetalBufferPool(Device())
    first = pool.acquire_scalar(100)
    first.contents().as_buffer(100)[:] = bytes(range(100))
    pool.release_scalar(first, 100)
    second = pool.acquire_scalar(116)
    assert second is not first
    second.contents().as_buffer(116)[:] = bytes(range(116))
    pool.release_scalar(second, 116)
    assert pool.acquire_scalar(100) is first and pool.acquire_scalar(116) is second
    assert bytes(first.data) == bytes(range(100)) and bytes(second.data) == bytes(range(116))
    assert allocations == [100, 116]
