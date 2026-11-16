"""K-loop source iteration extent is distinct from a proved zero-masked tail."""

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _body(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, mode: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap, bp = A + rm[:, None] * K + rk[None, :], B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, K, BK):
        valid = base + rk < K
        if mode == 0:
            av, bv = tl.load(ap), tl.load(bp)
        elif mode == 1:
            av = tl.load(ap, mask=valid[None, :], other=0.0)
            bv = tl.load(bp, mask=valid[:, None], other=0.0)
        elif mode == 2:
            av = tl.load(ap, mask=valid[None, :], other=0.0)
            bv = tl.load(bp)
        else:
            av = tl.load(ap, mask=valid[None, :], other=1.0)
            bv = tl.load(bp, mask=valid[:, None], other=1.0)
        acc += tl.dot(av, bv)
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _unmasked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _body(A, B, C, M, N, K, BM, BN, BK, 0)


@triton.jit
def _masked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _body(A, B, C, M, N, K, BM, BN, BK, 1)


@triton.jit
def _a_only(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _body(A, B, C, M, N, K, BM, BN, BK, 2)


@triton.jit
def _nonzero_other(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _body(A, B, C, M, N, K, BM, BN, BK, 3)


def _emit(fn):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    sig = {
        "A": "*fp32",
        "B": "*fp32",
        "C": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "BM": "constexpr",
        "BN": "constexpr",
        "BK": "constexpr",
    }
    module = ASTSource(fn, sig, {"BM": 32, "BN": 32, "BK": 32}).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    text = str(module)
    return text, backend.make_msl(module, {}, options)


def test_unmasked_source_replays_complete_final_block_and_bites_parent():
    ttgir, msl = _emit(_unmasked)
    assert "scf.for" in ttgir and "step %c32_i32" in ttgir
    assert "tt.load %ap" in ttgir and "tt.load %bp" in ttgir
    assert msl.count("long _K = (long)_mmarg5;") == 2
    assert msl.count("long _K_source = _K;") == 2
    assert msl.count("((_K_source + 31L) / 32L) * 32L") == 2
    a = [1.0] * 33 + [2.0] * 31
    b = [1.0] * 33 + [3.0] * 31
    assert sum(x * y for x, y in zip(a, b)) == 219.0


def test_both_zero_masked_operands_preserve_logical_k_tail():
    ttgir, msl = _emit(_masked)
    loads = [line for line in ttgir.splitlines() if "tt.load" in line]
    assert len(loads) == 2 and all(", %" in line for line in loads)
    assert "both operands prove a zero-masked K tail" in msl
    assert "_K_source" not in msl
    assert msl.count("both operands prove a zero-masked K tail") == 2


@pytest.mark.parametrize("fn", [_a_only, _nonzero_other])
def test_asymmetric_or_nonzero_tail_mask_refuses(fn):
    with pytest.raises(MetalNonRecoverableError, match="both zero-mask|other.*not a literal zero"):
        _emit(fn)
