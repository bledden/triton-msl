"""Grouped tutorial coordinates: prove the source mapping before replaying it.

Reconstructs the grouped spelling described in PR6, not a reporter-exact kernel.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _grouped(
    A,
    B,
    C,
    M,
    N,
    K,
    SAM,
    SAK,
    SBK,
    SBN,
    SCM,
    SCN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MUT: tl.constexpr,
):
    pid = tl.program_id(1 if MUT == 5 else 0)
    nm = tl.cdiv(M.to(tl.int16).to(tl.int32) if MUT == 7 else M, BM)
    nn = tl.cdiv(N, BN)
    group_id = pid // (GM * nn)
    first_m = group_id * (GM + 1 if MUT == 1 else GM)
    gm = tl.minimum((nn if MUT == 2 else nm) - first_m, GM)
    pm = first_m + (pid % (GM * nn)) % gm
    pn = (pid % (GM * nn)) // (GM if MUT == 3 else gm)
    am = (pm * BM + tl.arange(0, BM)) % M
    if MUT == 6:
        am = am.to(tl.int16).to(tl.int32)
    bn = (pn * BN + tl.arange(0, BN)) % N
    kk = tl.arange(0, BK)
    ap = A + am[:, None] * SAM + kk[None, :] * SAK
    bp = B + kk[:, None] * SBK + bn[None, :] * SBN
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, kk[None, :] < K - k * BK, other=0.0)
        b = tl.load(bp, kk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc=acc)
        ap += BK * SAK
        bp += BK * SBK
    cm = (pm + 1 if MUT == 4 else pm) * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(C + cm[:, None] * SCM + cn[None, :] * SCN, acc, (cm[:, None] < M) & (cn[None, :] < N))


def _lower(gm=8, mutation=0):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    signature = {
        n: "*fp16" if n in ("A", "B", "C") else "i32"
        for n in _grouped.arg_names
        if n not in ("BM", "BN", "BK", "GM", "MUT")
    }
    source = ASTSource(_grouped, signature=signature, constexprs=dict(BM=32, BN=32, BK=32, GM=gm, MUT=mutation))
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    lowerer = GenericLowerer(walk_ttgir(mod, options), options)
    return lowerer, str(mod)


@pytest.mark.parametrize("gm", [4, 8])
def test_grouped_coordinates_are_proved_and_replayed(gm):
    lowerer, ttgir = _lower(gm)
    assert "arith.minsi" in ttgir
    msl = lowerer.lower()
    assert "grouped source tile mapping" in msl
    assert f"* {gm}u" in msl
    assert "min(as_type<int>(as_type<uint>(_npm) - as_type<uint>(_first_m))" in msl
    assert "pid_m = as_type<uint>(_first_m) + as_type<uint>(_within % _group_m)" in msl
    assert "pid_n = as_type<uint>(_within / _group_m)" in msl


@pytest.mark.parametrize("mutation", [1, 2, 3, 4, 5, 6, 7])
def test_unreplayed_grouped_near_miss_refuses(mutation):
    lowerer, _ = _lower(mutation=mutation)
    with pytest.raises(MetalNonRecoverableError):
        lowerer.lower()


def _observe_grouped_routes(monkeypatch):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend import driver

    host = []
    direct = []
    launchers = []

    def observe_host(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kw):
            result = real(pipeline, grid, group, buffers, **kw)
            host.append((pipeline, grid, group))
            return result

        return wrapper

    def observe_direct(instance, real):
        def wrapper(lib, name, args, **kw):
            result = real(lib, name, args, **kw)
            direct.append((lib, name, args, kw))
            return result

        return wrapper

    launch = driver.MetalLauncher.__call__

    def observe_launcher(self, *args, **kw):
        launchers.append((self, args))
        return launch(self, *args, **kw)

    utils = patch_live_singleton_method(monkeypatch, driver._get_utils, "launch", observe_host)
    runtime = patch_live_singleton_method(monkeypatch, driver._get_compile_shader_runtime, "dispatch", observe_direct)
    monkeypatch.setattr(driver.MetalLauncher, "__call__", observe_launcher)
    return host, direct, launchers, runtime


@pytest.mark.parametrize("route", ["host", "direct"])
def test_grouped_route_observer_records_shadowed_singleton_cpu(monkeypatch, route):
    from types import SimpleNamespace
    from triton_msl.backend import driver
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    runtime = CompileShaderRuntime()
    utils = object.__new__(driver.MetalUtils)
    actual = []

    def host_call(self, *args, **kw):
        actual.append((args, kw))
        return "host returned"

    monkeypatch.setattr(driver.MetalUtils, "launch", host_call)
    monkeypatch.setattr(driver, "_metal_utils", utils)
    monkeypatch.setattr(driver, "_COMPILE_SHADER_RUNTIME", runtime)
    with pytest.MonkeyPatch.context() as previous:
        # A plain instance patch is what leaves a shadow: its undo restores
        # the looked-up bound method as an instance attribute.
        for owner, method in ((utils, "launch"), (runtime, "dispatch")):
            previous.setattr(owner, method, getattr(owner, method))
    assert "launch" in vars(utils) and "dispatch" in vars(runtime)
    host, direct, launchers, observed_runtime = _observe_grouped_routes(monkeypatch)
    assert observed_runtime is runtime and not host and not direct and not launchers
    if route == "host":
        pipeline = object()
        buffers = []
        assert utils.launch(pipeline, (9, 1, 1), (128, 1, 1), buffers, tag="test") == "host returned"
        assert host == [(pipeline, (9, 1, 1), (128, 1, 1))] and not direct
        assert actual == [((pipeline, (9, 1, 1), (128, 1, 1), buffers), dict(tag="test"))]
    else:
        lib = SimpleNamespace(kernel=lambda *a, **kw: actual.append((a, kw)))
        args = [42]
        geometry = dict(threads=(1152, 1, 1), group_size=(128, 1, 1))
        runtime.dispatch(lib, "kernel", args, **geometry)
        assert direct == [(lib, "kernel", args, geometry)] and not host
        assert actual == [((42,), geometry)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("compile_shader", ["0", "1"])
@pytest.mark.parametrize(
    "m,n,programs,overflow", [(256, 64, None, False), (288, 70, None, False), (288, 96, 25, False), (288, 64, 9, True)]
)
def test_grouped_full_tail_and_partial_launch_compute_source(monkeypatch, m, n, programs, overflow, compile_shader):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)

    gen = torch.Generator().manual_seed(459)
    a_cpu = torch.randint(-2, 3, (m, 32), generator=gen).to(torch.float16)
    b_cpu = torch.randint(-2, 3, (32, n), generator=gen).to(torch.float16)
    a, b = a_cpu.to("mps"), b_cpu.to("mps")
    out = torch.full((m, n), -8192.0, dtype=torch.float16, device="mps")
    cm, cn = triton.cdiv(m, 32), triton.cdiv(n, 32)
    count = cm * cn if programs is None else programs
    host, direct, launchers, runtime = _observe_grouped_routes(monkeypatch)
    logical_m = 2147483647 if overflow else m
    compiled = _grouped[(count,)](a, b, out, logical_m, n, 32, 32, 1, n, 1, n, 1, BM=32, BN=32, BK=32, GM=8, MUT=0)
    torch.mps.synchronize()
    assert len(launchers) == 1
    launcher, args = launchers[0]
    assert tuple(args[:3]) == (count, 1, 1) and args[4] is compiled.function
    assert launcher.kernel_name == compiled.metadata.name
    assert launcher._msl == compiled.asm["msl"]
    group = launcher._msl_block_size
    assert 0 < group <= 1024 and group == args[5][3]
    if compile_shader == "0":
        assert len(host) == 1 and not direct
        assert host[0][0] is compiled.function
        assert tuple(host[0][1]) == (count, 1, 1)
        assert tuple(host[0][2]) == (group, 1, 1)
    else:
        assert len(direct) == 1 and not host
        lib, name, bound, geometry = direct[0]
        assert runtime._lib_cache[launcher._msl] is lib
        assert name == launcher.kernel_name
        assert len(bound) >= 3 and bound[0] is a and bound[1] is b and bound[2] is out
        assert args[5][5] is True and all(v is None for v in args[5][6:11])
        assert geometry == dict(threads=(count * group, 1, 1), group_size=(group, 1, 1))
    assert any("grouped source tile mapping" in value for value in compiled.asm.values() if isinstance(value, str))
    # Enumerate rows in groups independently of the shader's quotient/remainder DAG.
    tile_order = [
        (row, col) for first in range(0, cm, 8) for col in range(cn) for row in range(first, min(first + 8, cm))
    ]
    ref = a_cpu.double() @ b_cpu.double()
    if overflow:
        # Source i32 cdiv numerator wraps negative; gm is negative. For these
        # nine programs remsi(0..8, gm) is 0..8 and divsi(0..8, gm) is zero.
        # Only allocated rows 0..287 / columns 0..31 are accessed; all in bounds.
        tile_order = [(row, 0) for row in range(count)]
    expected = torch.full_like(out.cpu(), -8192.0)
    for row, col in tile_order[:count]:
        section = (slice(row * 32, (row + 1) * 32), slice(col * 32, (col + 1) * 32))
        expected[section] = ref[section].half()
    assert torch.equal(out.cpu(), expected), "computed tiles and untouched sentinel tiles must both match"
