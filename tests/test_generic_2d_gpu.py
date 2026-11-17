"""Source/oracle pins for generic 2D binding, aliases and assertion boundaries."""

import pytest
import torch
import triton
import triton.language as tl


def _save_evidence(label, arrays, dispatches, host, launchers):
    """Optional durable raw evidence for the coordinated validation runner."""
    import os

    destination = os.environ.get("TRITON_MSL_GPU_EVIDENCE")
    if not destination:
        return
    import hashlib, json
    from pathlib import Path
    import numpy as np

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / (label + ".npz"), **{name: value.detach().cpu().numpy().copy() for name, value in arrays.items()})
    launcher, args = launchers[-1]
    (out / (label + ".msl")).write_text(launcher._msl)
    (out / (label + ".json")).write_text(
        json.dumps(
            dict(
                dispatches=dispatches,
                host_dispatches=len(host),
                source_sha256=hashlib.sha256(launcher._msl.encode()).hexdigest(),
                signature=launcher.signature,
                grid=list(args[:3]),
                group=launcher._msl_block_size,
                metadata=args[5],
            ),
            indent=2,
        )
        + "\n"
    )


@triton.jit
def affine_2d(X, O, N, SCALE, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    offset = row * N + col
    value = tl.load(X + offset, col < N, other=0.0)
    tl.store(O + offset, value * SCALE + 1.0, col < N)


@triton.jit
def integer_2d(O, N, BIAS, UNSIGNED, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    value = BIAS.to(tl.int64) + UNSIGNED.to(tl.int64) + row.to(tl.int64) * N + col
    tl.store(O + row * N + col, value, col < N)


def integer_source():
    from triton.compiler import ASTSource

    return ASTSource(
        integer_2d, signature=dict(O="*i64", N="i32", BIAS="i64", UNSIGNED="u32"), constexprs=dict(BLOCK=32)
    )


@triton.jit
def checked_accumulate_2d(X, O, N, LIMIT, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    offset = row * N + col
    value = tl.load(X + offset, col < N, other=0)
    tl.device_assert((value < LIMIT) | (col >= N), "generic 2d input bound")
    tl.atomic_add(O + offset, value, col < N, sem="relaxed")


def _trace(monkeypatch):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend import driver

    dispatches = []
    host = []
    launchers = []
    launch = driver.MetalLauncher.__call__

    def observe_dispatch(instance, dispatch):
        def record(lib, name, args, **kw):
            dispatches.append(dict(name=name, **kw))
            return dispatch(lib, name, args, **kw)

        return record

    def observe_host(instance, host_launch):
        def record(*a, **kw):
            host.append(a)
            return host_launch(*a, **kw)

        return record

    def record_launcher(self, *a, **kw):
        launchers.append((self, a))
        return launch(self, *a, **kw)

    # Fixture undo can leave a bound method on these long-lived instances.
    # Observe the exact objects used by production, including that shadow.
    patch_live_singleton_method(monkeypatch, driver._get_compile_shader_runtime, "dispatch", observe_dispatch)
    patch_live_singleton_method(monkeypatch, driver._get_utils, "launch", observe_host)
    # Python resolves the special call method on the class.
    monkeypatch.setattr(driver.MetalLauncher, "__call__", record_launcher)
    return dispatches, host, launchers


def test_live_singleton_patch_leaves_no_instance_shadow_cpu(monkeypatch):
    """Undoing the helper must not pin a class-level replacement on the singleton."""
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend import driver

    utils = object.__new__(driver.MetalUtils)
    monkeypatch.setattr(driver, "_metal_utils", utils)
    with pytest.MonkeyPatch.context() as outer:
        outer.setattr(driver.MetalUtils, "launch", lambda self, *a, **kw: "refused")
        with pytest.MonkeyPatch.context() as inner:
            patch_live_singleton_method(
                inner, driver._get_utils, "launch", lambda owner, real: lambda *a, **kw: real(*a, **kw)
            )
            assert "launch" in vars(utils) and utils.launch() == "refused"
        assert "launch" not in vars(utils)
    assert utils.launch.__func__ is driver.MetalUtils.launch


@pytest.mark.parametrize("shadowed", [False, True])
def test_trace_records_live_routes_across_fixture_undo_cpu(monkeypatch, shadowed):
    from types import SimpleNamespace
    from triton_msl.backend import driver
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    runtime = CompileShaderRuntime()
    # No device construction: these stand-ins only exercise Python dispatch.
    utils = object.__new__(driver.MetalUtils)
    launcher = object.__new__(driver.MetalLauncher)
    actual_dispatches = []
    actual_host = []
    actual_launchers = []
    lib = SimpleNamespace(kernel=lambda *a, **kw: actual_dispatches.append((a, kw)))

    def host_call(self, *a, **kw):
        actual_host.append((a, kw))
        return "host result"

    def launcher_call(self, *a, **kw):
        actual_launchers.append((self, a, kw))
        return "launcher result"

    monkeypatch.setattr(driver.MetalUtils, "launch", host_call)
    monkeypatch.setattr(driver.MetalLauncher, "__call__", launcher_call)
    monkeypatch.setattr(driver, "_COMPILE_SHADER_RUNTIME", runtime)
    monkeypatch.setattr(driver, "_metal_utils", utils)
    if shadowed:
        with pytest.MonkeyPatch.context() as previous:
            # A plain instance patch is what leaves a shadow: its undo restores
            # the looked-up bound method as an instance attribute.
            for owner, name in ((runtime, "dispatch"), (utils, "launch")):
                previous.setattr(owner, name, getattr(owner, name))
        assert "dispatch" in vars(runtime) and "launch" in vars(utils)

    recordings = []
    for iteration in range(2):
        with pytest.MonkeyPatch.context() as observer:
            dispatches, host, launchers = _trace(observer)
            assert not dispatches and not host and not launchers
            runtime.dispatch(lib, "kernel", [iteration], threads=(96, 5, 1), group_size=(32, 1, 1))
            assert dispatches == [dict(name="kernel", threads=(96, 5, 1), group_size=(32, 1, 1))]
            assert actual_dispatches[-1] == ((iteration,), dict(threads=(96, 5, 1), group_size=(32, 1, 1)))
            assert utils.launch(iteration, tag="host") == "host result"
            assert host == [(iteration,)] and actual_host[-1] == ((iteration,), dict(tag="host"))
            assert launcher(iteration, tag="launcher") == "launcher result"
            assert launchers == [(launcher, (iteration,))]
            assert actual_launchers[-1] == (launcher, (iteration,), dict(tag="launcher"))
            recordings.append((dispatches, host, launchers))
    assert len(actual_dispatches) == len(actual_host) == len(actual_launchers) == 2
    # A completed fixture must not keep collecting the following test's calls.
    assert all(len(records) == 1 for observation in recordings for records in observation)


requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


@requires_gpu
@pytest.mark.parametrize("alias", ["separate", "shared_disjoint", "inplace"])
def test_affine_nonsquare_offsets_and_defined_aliases(monkeypatch, alias):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    dispatches, host, launchers = _trace(monkeypatch)
    count = 5 * 93
    sentinel = 98765.0
    owner = torch.full((count * 2 + 96,), sentinel, device="mps")
    source = torch.arange(count, dtype=torch.float32).reshape(5, 93) / 16
    x = owner[16 : 16 + count].view(5, 93)
    x.copy_(source.to("mps"))
    if alias == "inplace":
        out = x
    elif alias == "shared_disjoint":
        out = owner[count + 48 : 2 * count + 48].view(5, 93)
    else:
        out_owner = torch.full((count + 64,), sentinel, device="mps")
        out = out_owner[32 : 32 + count].view(5, 93)
    owners = {v.untyped_storage()._cdata: v for v in ([owner, out_owner] if alias == "separate" else [owner])}
    before = {key: value.cpu().clone() for key, value in owners.items()}
    affine_2d[(3, 5)](x, out, 93, 1.25, BLOCK=32)
    torch.mps.synchronize()
    expected_owners = {}
    for key, value in owners.items():
        expected = before[key].clone()
        if key == out.untyped_storage()._cdata:
            expected[out.storage_offset() : out.storage_offset() + count] = (source * 1.25 + 1).reshape(-1)
        expected_owners[key] = expected
    _save_evidence(
        "affine-" + alias,
        dict(
            actual=out,
            expected=source * 1.25 + 1,
            source=source,
            **{f"owner{i}": v for i, v in enumerate(owners.values())},
            **{f"expected_owner{i}": v for i, v in enumerate(expected_owners.values())},
        ),
        dispatches,
        host,
        launchers,
    )
    torch.testing.assert_close(out.cpu(), source * 1.25 + 1.0, rtol=0, atol=0)
    assert len(dispatches) == 1 and not host
    launcher, args = launchers[-1]
    group = launcher._msl_block_size
    assert dispatches[-1]["threads"] == (3 * group, 5, 1) and dispatches[-1]["group_size"] == (group, 1, 1)
    assert args[5][5] is True and 0 < group < 1024
    for key, value in owners.items():
        expected = before[key]
        if key == out.untyped_storage()._cdata:
            expected[out.storage_offset() : out.storage_offset() + count] = (source * 1.25 + 1).reshape(-1)
        assert torch.equal(value.cpu(), expected)


@requires_gpu
@pytest.mark.parametrize("bias", [1 << 40, -(1 << 40)])
def test_integer_scalar_signedness_and_high_bits(monkeypatch, bias):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    dispatches, host, launchers = _trace(monkeypatch)
    n = 75
    rows = 5
    owner = torch.full((rows * n + 64,), -7, dtype=torch.int64, device="mps")
    out = owner[16 : 16 + rows * n].view(rows, n)
    # The public explicit source signature is the ABI under test. A Python
    # integer's inferred JIT type is not evidence for a u32 declaration.
    kernel = triton.compile(integer_source(), options=dict(num_warps=4))
    kernel[(3, rows, 1)](out, n, bias, (1 << 32) - 1)
    torch.mps.synchronize()
    expected = torch.arange(rows * n, dtype=torch.int64).view(rows, n) + bias + (1 << 32) - 1
    _save_evidence(
        "integer-" + str(bias), dict(actual=out, expected=expected, owner=owner), dispatches, host, launchers
    )
    assert torch.equal(out.cpu(), expected) and torch.all(owner[:16] == -7) and torch.all(owner[16 + rows * n :] == -7)
    assert len(dispatches) == 1 and not host
    assert launchers[-1][0].signature["BIAS"] == "i64"
    assert launchers[-1][0].signature["UNSIGNED"] == "u32"


@requires_gpu
def test_assertion_valid_invalid_valid_exact_no_write(monkeypatch):
    from triton_msl.errors import MetalDeviceAssertionError

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    dispatches, host, launchers = _trace(monkeypatch)
    n = 93
    count = 5 * n
    x = torch.ones(count, device="mps")
    owner = torch.full((count + 64,), 54321.0, device="mps")
    out = owner[16 : 16 + count]
    out.fill_(3)
    checked_accumulate_2d[(3, 5)](x, out, n, 10, BLOCK=32, debug=True)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((count,), 4.0))
    before = owner.cpu().clone()
    x.fill_(11)
    with pytest.raises(MetalDeviceAssertionError):
        checked_accumulate_2d[(3, 5)](x, out, n, 10, BLOCK=32, debug=True)
    torch.mps.synchronize()
    _save_evidence("assert-invalid", dict(actual=owner, expected=before), dispatches, host, launchers)
    assert torch.equal(owner.cpu(), before)
    x.fill_(1)
    checked_accumulate_2d[(3, 5)](x, out, n, 10, BLOCK=32, debug=True)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((count,), 5.0))
    _save_evidence(
        "assert-recovery",
        dict(actual=out, expected=torch.full((count,), 5.0), owner=owner),
        dispatches,
        host,
        launchers,
    )
    assert len(dispatches) == 3 and not host
    assert torch.all(owner[:16] == 54321) and torch.all(owner[16 + count :] == 54321)


@requires_gpu
def test_attempted_atomic_is_not_replayed_after_exit_failure(monkeypatch):
    from triton.knobs import runtime
    from triton_msl.errors import PostSubmitError

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    dispatches, host, launchers = _trace(monkeypatch)
    n = 93
    count = 5 * n
    x = torch.ones(count, device="mps")
    out = torch.zeros(count, device="mps")
    checked_accumulate_2d[(3, 5)](x, out, n, 10, BLOCK=32, debug=True)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.ones(count))
    out.zero_()
    dispatches.clear()

    def fail(_):
        raise RuntimeError("after real invocation")

    monkeypatch.setattr(runtime, "launch_exit_hook", fail)
    with pytest.raises(PostSubmitError):
        checked_accumulate_2d[(3, 5)](x, out, n, 10, BLOCK=32, debug=True)
    torch.mps.synchronize()
    _save_evidence("exit-no-replay", dict(actual=out, expected=torch.ones(count)), dispatches, host, launchers)
    assert torch.equal(out.cpu(), torch.ones(count))
    assert len(dispatches) == 1 and not host
