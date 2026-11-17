"""N-fastest flat traversal: source-coordinate recovery, not fast-path credit."""

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
def _flat(
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
    MODE: tl.constexpr,
):
    pid = tl.program_id(1 if MODE == 6 else 0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N.to(tl.int16).to(tl.int32) if MODE == 4 else N, BN)
    if MODE == 0:
        pm = pid % nm
        pn = pid // nm
    else:
        pm = pid // (nm if MODE == 2 else nn)
        pn = pid % nn
    am = (pm * BM + tl.arange(0, BM)) % M
    if MODE == 5:
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
    cm = (pm + 1 if MODE == 3 else pm) * BM + tl.arange(0, BM)
    cn = pn * (BN + 1 if MODE == 7 else BN) + tl.arange(0, BN)
    tl.store(C + cm[:, None] * SCM + cn[None, :] * SCN, acc, (cm[:, None] < M) & (cn[None, :] < N))


def _lower(mode=1):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    signature = {
        n: "*fp16" if n in ("A", "B", "C") else "i32" for n in _flat.arg_names if n not in ("BM", "BN", "BK", "MODE")
    }
    source = ASTSource(_flat, signature=signature, constexprs=dict(BM=32, BN=32, BK=32, MODE=mode))
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


def test_n_fastest_coordinates_preserve_signed_source_mapping():
    lowerer = _lower()
    msl = lowerer.lower()
    assert "N-fastest source tile mapping" in msl
    assert "int _npn = as_type<int>((uint)_mmarg4 + 31u) / 32;" in msl
    assert "uint pid_m = as_type<uint>(as_type<int>(pid3.x) / _npn);" in msl
    assert "uint pid_n = as_type<uint>(as_type<int>(pid3.x) % _npn);" in msl
    assert lowerer._fast_matmul is None, "partial source grids cannot become whole-output fast launches"


def test_existing_m_fastest_mapping_still_lowers():
    msl = _lower(0).lower()
    assert "uint pid_m = pid3.x % _npm;" in msl
    assert "uint pid_n = pid3.x / _npm;" in msl
    assert "N-fastest source tile mapping" not in msl


@pytest.mark.parametrize("mode", [2, 3, 4, 5, 6, 7])
def test_flat_near_misses_refuse(mode):
    with pytest.raises(MetalNonRecoverableError):
        _lower(mode).lower()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize(
    "m,n,programs,overflow", [(64, 96, None, False), (70, 97, None, False), (96, 128, 5, False), (32, 288, 9, True)]
)
def test_flat_full_tail_partial_and_overflow_compute_source(monkeypatch, m, n, programs, overflow):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    gen = torch.Generator().manual_seed(483)
    a_cpu = torch.randint(-2, 3, (m, 32), generator=gen).half()
    b_cpu = torch.randint(-2, 3, (32, n), generator=gen).half()
    a, b = a_cpu.to("mps"), b_cpu.to("mps")
    out = torch.full((m, n), -8192.0, dtype=torch.float16, device="mps")
    count = triton.cdiv(m, 32) * triton.cdiv(n, 32) if programs is None else programs
    launches = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            result = real(pipeline, grid, group, buffers, **kwargs)
            launches.append((pipeline, grid, group))
            return result

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    compiled = _flat[(count,)](
        a, b, out, m, 2147483647 if overflow else n, 32, 32, 1, n, 1, n, 1, BM=32, BN=32, BK=32, MODE=1
    )
    torch.mps.synchronize()
    assert len(launches) == 1 and launches[0][0] is compiled.function
    assert tuple(launches[0][1]) == (count, 1, 1)
    assert any("N-fastest source tile mapping" in s for s in compiled.asm.values() if isinstance(s, str))
    ref = a_cpu.double() @ b_cpu.double()
    expected = torch.full((m, n), -8192.0, dtype=torch.float16)
    order = [(r, c) for r in range(triton.cdiv(m, 32)) for c in range(triton.cdiv(n, 32))]
    if overflow:
        # Signed cdiv wraps negative. For pid0..8 quotient is0, remainder ispid.
        # All source addresses stay in physical row0..31 / columns0..287.
        order = [(0, c) for c in range(count)]
    for r, c in order[:count]:
        section = (slice(r * 32, (r + 1) * 32), slice(c * 32, (c + 1) * 32))
        expected[section] = ref[section].half()
    assert torch.equal(out.cpu(), expected), "computed tiles and untouched sentinel tiles must both match"
