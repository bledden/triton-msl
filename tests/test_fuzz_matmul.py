"""Systemic matmul/dot differential fuzzer (re-audit #14 -> systemic gate).

The matmul template family (simple-dot, K-loop, strided-pid, epilogue) has been the
dominant silent-wrong surface across re-audits #11-14 — point patches found edges one
per round and hit over-refusal walls. This is the matmul analog of
``tests/test_fuzz_combinations.py``: it sweeps the fast-path forms x shapes x dtypes x
{plain, const-bias, loaded-bias, masked/padded-K, batch-pid} and asserts the ONE
invariant for every cell:

    correct (vs torch, dtype-appropriate tol)  OR  loud MetalNonRecoverableError

never a silent-wrong (computed-but-wrong) and never a cryptic crash (any other
exception). A computed result is checked against the exact torch reference for that
cell; a refusal is always acceptable (a fast-path may decline an edge it can't do).

Run deep:  TRITON_MSL_FALLBACK=error python tests/test_fuzz_matmul.py 400
"""
import math
import os
import shutil
import sys

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

_CACHE = os.path.expanduser("~/.cache/triton_msl")
_TCACHE = os.path.expanduser("~/.triton/cache")


def _clear_cache():
    for d in (_CACHE, _TCACHE):
        shutil.rmtree(d, ignore_errors=True)


def _tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=2e-3, atol=2e-3)
    return dict(rtol=4e-2, atol=4e-2)   # fp16/bf16 compute-in-fp32, narrowed output


# --------------------------------------------------------------------------- kernels
@triton.jit
def _k_simple(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    av = tl.load(a + om[:, None] * K + ok[None, :])
    bv = tl.load(b + ok[:, None] * N + on[None, :])
    tl.store(c + om[:, None] * N + on[None, :], tl.dot(av, bv).to(c.dtype.element_ty))


@triton.jit
def _k_simple_bias(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    av = tl.load(a + om[:, None] * K + ok[None, :])
    bv = tl.load(b + ok[:, None] * N + on[None, :])
    acc = tl.dot(av, bv, tl.full((M, N), 3.0, tl.float32))
    tl.store(c + om[:, None] * N + on[None, :], acc.to(c.dtype.element_ty))


@triton.jit
def _k_kloop(a, b, c, sam, sak, sbk, sbn, scm, scn, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM); rn = tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty))


@triton.jit
def _k_kloop_bias(a, b, c, sam, sak, sbk, sbn, scm, scn, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM); rn = tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 3.0, tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty))


@triton.jit
def _k_strided_masked(a, b, c, sam, sak, sbk, sbn, scm, scn, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        am = (rm[:, None] < M) & (kk[None, :] < K)
        bm = (kk[:, None] < K) & (rn[None, :] < N)
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak, mask=am, other=0.0),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn, mask=bm, other=0.0))
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty), mask=cm)


# --- transposed / strided operands (the coverage hole that hid BLOCKER 1/2) ---
# A fully-stride-parameterized pid-tiled K-loop matmul; the runner passes the
# REAL .stride() of operands that may be transposed (inner dim non-contiguous),
# column-major, or otherwise strided. Row-major addressing would be silently
# wrong, so the backend must use the inferred strides (correct) or refuse loudly.
@triton.jit
def _k_kloop_strided(a, b, c, sam, sak, sbk, sbn, scm, scn, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty))


# Single-tile (no program_id) strided matmul with an optional fused epilogue.
@triton.jit
def _k_single_strided(a, b, c, sam, sak, sbk, sbn, scm, scn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK)
    av = tl.load(a + om[:, None] * sam + ok[None, :] * sak)
    bv = tl.load(b + ok[:, None] * sbk + on[None, :] * sbn)
    tl.store(c + om[:, None] * scm + on[None, :] * scn, tl.dot(av, bv).to(c.dtype.element_ty))


@triton.jit
def _k_single_strided_epi(a, b, c, sam, sak, sbk, sbn, scm, scn,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK)
    av = tl.load(a + om[:, None] * sam + ok[None, :] * sak)
    bv = tl.load(b + ok[:, None] * sbk + on[None, :] * sbn)
    acc = tl.dot(av, bv) * 2.0 + 1.0       # fused scale+bias epilogue
    tl.store(c + om[:, None] * scm + on[None, :] * scn, acc.to(c.dtype.element_ty))


# --- STANDARD stride_* arg-naming convention (Triton-tutorial names) ---
# BLOCKER 2 hid behind ``if has_strides: return None`` (any "stride"-substring arg
# name skipped the address-traced safety path). These kernels use the canonical
# ``stride_am, stride_ak, ...`` names so the fuzzer exercises the naming-independent
# routing (a transposed/strided operand must still be correct or refuse, NOT the old
# name-match row-major guess).
@triton.jit
def _k_kloop_stdname(a_ptr, b_ptr, c_ptr,
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :] * stride_ak),
                      tl.load(b_ptr + kk[:, None] * stride_bk + rn[None, :] * stride_bn))
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _k_single_stdname(a_ptr, b_ptr, c_ptr,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK)
    av = tl.load(a_ptr + om[:, None] * stride_am + ok[None, :] * stride_ak)
    bv = tl.load(b_ptr + ok[:, None] * stride_bk + on[None, :] * stride_bn)
    tl.store(c_ptr + om[:, None] * stride_cm + on[None, :] * stride_cn,
             tl.dot(av, bv).to(c_ptr.dtype.element_ty))


# --- BATCHED 3-D matmul (the docstring advertised a batch-pid case but defined none) ---
# Batch on program_id(0); each batch advances A/B/C base pointers. Batched MMA is not
# implemented -> the backend must REFUSE loudly (never compute batch 0 for every batch).
@triton.jit
def _k_batched(a, b, c, Bz, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0); pm = tl.program_id(1); pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = a + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
    bp = b + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK * sak; bp += BK * sbk
    tl.store(c + pb * scb + (rm[:, None] * scm + rn[None, :] * scn),
             acc.to(c.dtype.element_ty))


# --------------------------------------------------------------------------- runner
def _run_cell(form, M, N, K, dtype, seed):
    """Return ('correct'|'refused'|'wrong:<err>'|'crash:<type>', detail).

    Cache is NOT cleared per cell — Triton/triton-msl key by dtype + constexpr shape, so
    cells never collide, and clearing on-disk mid-session races the in-memory<->disk cache
    (spurious FileNotFoundError under heavy load). A FileNotFoundError is retried once.
    """
    torch.manual_seed(seed)
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=dtype)
    ref = (A.float() @ B.float())
    bias = 0.0
    try:
        if form == "simple":
            _k_simple[(1,)](A, B, C, M=M, N=N, K=K)
        elif form == "simple_bias":
            _k_simple_bias[(1,)](A, B, C, M=M, N=N, K=K); bias = 3.0
        elif form == "kloop":
            _k_kloop[(1, 1)](A, B, C, *A.stride(), *B.stride(), *C.stride(), K,
                             BM=M, BN=N, BK=min(16, K))
        elif form == "kloop_bias":
            _k_kloop_bias[(1, 1)](A, B, C, *A.stride(), *B.stride(), *C.stride(), K,
                                  BM=M, BN=N, BK=min(16, K)); bias = 3.0
        elif form == "strided_masked":
            BM, BN, BK = 32, 32, 16
            grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
            _k_strided_masked[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                    M, N, K, BM=BM, BN=BN, BK=BK)
        else:
            return ("crash:badform", form)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return ("refused", None)
    except FileNotFoundError:
        _clear_cache()                                     # transient cache race; retry
        try:
            return _run_cell(form, M, N, K, dtype, seed + 100000)
        except MetalNonRecoverableError:
            return ("refused", None)
        except Exception as e:                             # noqa: BLE001
            return (f"crash:{type(e).__name__}", str(e)[:80])
    except Exception as e:                                 # noqa: BLE001
        return (f"crash:{type(e).__name__}", str(e)[:80])
    got = C.float()
    exp = ref + bias
    err = (got - exp).abs().max().item()
    scale = max(exp.abs().max().item(), 1e-6)
    rel = err / scale
    t = _tol(dtype)
    if err <= t["atol"] + t["rtol"] * scale:
        return ("correct", rel)
    return (f"wrong:{rel:.2e}", rel)


def _make_operand(shape, layout, dtype):
    """Build an operand of logical ``shape`` in the requested memory ``layout``.

    - "rowmaj": dense row-major (the contiguous baseline).
    - "trans":  the transpose of a (cols, rows) tensor -> inner dim NON-contiguous
                (this is the x @ w.t() layout that was BLOCKER 1).
    - "sliced": a column slice ``randn(r, c+pad)[:, :c]`` -> inner CONTIGUOUS but the
                ROW stride (c+pad) != the matrix dim c (the audit's sliced-A/B case;
                the row stride is a runtime arg the simdgroup leading dim must honor).
    """
    r, c = shape
    if layout == "trans":
        return torch.randn(c, r, device="mps", dtype=dtype).t()   # logical (r,c), stride (1,c)
    if layout == "sliced":
        pad = 8
        return torch.randn(r, c + pad, device="mps", dtype=dtype)[:, :c]  # stride (c+pad, 1)
    return torch.randn(r, c, device="mps", dtype=dtype)


def _run_strided_cell(form, M, N, K, dtype, a_lay, b_lay, c_lay, seed):
    """Run a transposed/strided/column-major matmul cell.

    Invariant (same as the rest of this fuzzer): the result is either CORRECT vs
    the torch reference, or the backend REFUSED loudly — never silent-wrong, never
    a cryptic crash. The operand layouts (transposed inner dim, column-major C)
    are exactly the coverage the contiguous-row-major-only forms above missed.
    """
    torch.manual_seed(seed)
    A = _make_operand((M, K), a_lay, dtype)
    B = _make_operand((K, N), b_lay, dtype)
    if c_lay == "trans":            # column-major output: logical (M,N), stride (1,M)
        C = torch.empty(N, M, device="mps", dtype=dtype).t()
    elif c_lay == "sliced":         # row-sliced output: logical (M,N), stride (N+pad,1)
        C = torch.empty(M, N + 8, device="mps", dtype=dtype)[:, :N]
    else:
        C = torch.empty(M, N, device="mps", dtype=dtype)
    ref = (A.float() @ B.float())
    epi = form.endswith("_epi")
    if epi:
        ref = ref * 2.0 + 1.0
    try:
        if form.startswith("kloop") and "stdname" not in form:
            BM, BN, BK = M, N, min(16, K)
            _k_kloop_strided[(1, 1)](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                     M, N, K, BM=BM, BN=BN, BK=BK)
        elif form == "kloop_stdname":
            BM, BN, BK = M, N, min(16, K)
            _k_kloop_stdname[(1, 1)](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                     M, N, K, BM=BM, BN=BN, BK=BK)
        elif form == "single_stdname":
            _k_single_stdname[(1,)](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                    BM=M, BN=N, BK=K)
        elif form == "single":
            _k_single_strided[(1,)](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                    BM=M, BN=N, BK=K)
        elif form == "single_epi":
            _k_single_strided_epi[(1,)](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                                        BM=M, BN=N, BK=K)
        else:
            return ("crash:badform", form)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return ("refused", None)
    except FileNotFoundError:
        _clear_cache()
        try:
            return _run_strided_cell(form, M, N, K, dtype, a_lay, b_lay, c_lay, seed + 100000)
        except MetalNonRecoverableError:
            return ("refused", None)
        except Exception as e:                             # noqa: BLE001
            return (f"crash:{type(e).__name__}", str(e)[:80])
    except Exception as e:                                 # noqa: BLE001
        return (f"crash:{type(e).__name__}", str(e)[:80])
    got = C.float()
    err = (got - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-6)
    rel = err / scale
    t = _tol(dtype)
    if err <= t["atol"] + t["rtol"] * scale:
        return ("correct", rel)
    return (f"wrong:{rel:.2e}", rel)


# power-of-2 tiles (arange needs power of 2); strided form covers unaligned M/N/K.
_PO2 = [16, 32, 64]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_PO2_FORMS = ["simple", "simple_bias", "kloop", "kloop_bias"]


def _po2_cells():
    cells = []
    for form in _PO2_FORMS:
        for dt in _DTYPES:
            for M in _PO2:
                for K in _PO2:
                    cells.append((form, M, M, K, dt))   # N=M to bound the matrix
    return cells


def _strided_cells():
    # unaligned M/N/K through the strided+masked form
    cells = []
    for dt in _DTYPES:
        for (M, N, K) in [(48, 48, 48), (40, 24, 40), (33, 65, 17), (96, 32, 80), (17, 17, 33)]:
            cells.append(("strided_masked", M, N, K, dt))
    return cells


# transposed / column-major / strided-operand cells (cover the BLOCKER 1/2 hole).
# Each cell: (form, M, N, K, dtype, a_layout, b_layout, c_layout).
def _transposed_cells():
    cells = []
    shapes = [(32, 32, 32), (64, 32, 64), (32, 64, 32)]
    for dt in _DTYPES:
        for (M, N, K) in shapes:
            # transposed B (x @ w.t()) — the canonical case, pid-tiled + single-tile.
            cells.append(("kloop", M, N, K, dt, "rowmaj", "trans", "rowmaj"))
            cells.append(("single", M, N, K, dt, "rowmaj", "trans", "rowmaj"))
            cells.append(("single_epi", M, N, K, dt, "rowmaj", "trans", "rowmaj"))
            # transposed A.
            cells.append(("kloop", M, N, K, dt, "trans", "rowmaj", "rowmaj"))
            # column-major output C.
            cells.append(("kloop", M, N, K, dt, "rowmaj", "rowmaj", "trans"))
            # general: both inputs transposed + column-major C.
            cells.append(("kloop", M, N, K, dt, "trans", "trans", "trans"))
    return cells


# STANDARD stride_*-named cells (BLOCKER 2): the canonical Triton-tutorial naming
# must route through the address-traced path, NOT the old name-match row-major guess.
def _stdname_cells():
    cells = []
    shapes = [(32, 32, 32), (64, 32, 64), (32, 64, 32)]
    for dt in _DTYPES:
        for (M, N, K) in shapes:
            for bl in ("rowmaj", "trans"):     # contiguous + transposed (x @ w.t())
                cells.append(("kloop_stdname", M, N, K, dt, "rowmaj", bl, "rowmaj"))
            cells.append(("single_stdname", M, N, K, dt, "rowmaj", "trans", "rowmaj"))
            cells.append(("kloop_stdname", M, N, K, dt, "trans", "rowmaj", "rowmaj"))  # trans-A
            cells.append(("kloop_stdname", M, N, K, dt, "rowmaj", "rowmaj", "trans"))  # col-major C
    return cells


# SLICED-operand cells (BLOCKER 1, generalized): a column slice randn(r,c+pad)[:, :c]
# has a CONTIGUOUS inner dim but a ROW stride != the matrix dim — the simdgroup leading
# dim must use the inferred runtime row stride, or refuse. Single-tile AND K-loop, for
# A, B, and C, across dtypes.
def _sliced_cells():
    cells = []
    shapes = [(32, 32, 32), (64, 32, 64), (32, 64, 32)]
    for dt in _DTYPES:
        for (M, N, K) in shapes:
            cells.append(("single", M, N, K, dt, "sliced", "rowmaj", "rowmaj"))  # sliced A
            cells.append(("single", M, N, K, dt, "rowmaj", "sliced", "rowmaj"))  # sliced B
            cells.append(("single", M, N, K, dt, "rowmaj", "rowmaj", "sliced"))  # sliced C
            cells.append(("kloop", M, N, K, dt, "sliced", "rowmaj", "rowmaj"))
            cells.append(("kloop", M, N, K, dt, "rowmaj", "sliced", "rowmaj"))
            cells.append(("kloop_stdname", M, N, K, dt, "sliced", "sliced", "rowmaj"))
    return cells


def _batched_cells():
    cells = []
    for dt in _DTYPES:
        for (Bz, M, N, K) in [(4, 32, 32, 32), (3, 64, 32, 64), (2, 32, 64, 48)]:
            cells.append((Bz, M, N, K, dt))
    return cells


def _run_batched_cell(Bz, M, N, K, dtype, seed):
    """A real 3-D batched matmul. Batched MMA is not implemented, so the ONLY
    acceptable outcomes are a LOUD MetalNonRecoverableError (preferred) or a CORRECT
    result — never silent-wrong (computing batch 0's region for every batch)."""
    torch.manual_seed(seed)
    A = torch.randn(Bz, M, K, device="mps", dtype=dtype)
    B = torch.randn(Bz, K, N, device="mps", dtype=dtype)
    C = torch.empty(Bz, M, N, device="mps", dtype=dtype)
    ref = torch.bmm(A.float(), B.float())
    BM, BN, BK = M, N, min(16, K)
    try:
        _k_batched[(Bz, 1, 1)](A, B, C, Bz, M, N, K,
                               *A.stride(), *B.stride(), *C.stride(),
                               BM=BM, BN=BN, BK=BK)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return ("refused", None)
    except FileNotFoundError:
        _clear_cache()
        try:
            return _run_batched_cell(Bz, M, N, K, dtype, seed + 100000)
        except MetalNonRecoverableError:
            return ("refused", None)
        except Exception as e:                             # noqa: BLE001
            return (f"crash:{type(e).__name__}", str(e)[:80])
    except Exception as e:                                 # noqa: BLE001
        return (f"crash:{type(e).__name__}", str(e)[:80])
    err = (C.float() - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-6)
    rel = err / scale
    t = _tol(dtype)
    if err <= t["atol"] + t["rtol"] * scale:
        return ("correct", rel)
    return (f"wrong:{rel:.2e}", rel)


# --------------------------------------------------------------------------- pytest
@requires
@pytest.mark.parametrize("form,M,N,K,dtype", _po2_cells())
def test_matmul_po2_correct_or_refuse(form, M, N, K, dtype):
    status, detail = _run_cell(form, M, N, K, dtype, seed=0)
    assert not status.startswith("wrong"), \
        f"SILENT-WRONG {form} {M}x{N}x{K} {dtype}: rel_err {detail}"
    assert not status.startswith("crash"), \
        f"CRYPTIC-CRASH {form} {M}x{N}x{K} {dtype}: {status} {detail}"


@requires
@pytest.mark.parametrize("form,M,N,K,dtype", _strided_cells())
def test_matmul_strided_unaligned_correct_or_refuse(form, M, N, K, dtype):
    status, detail = _run_cell(form, M, N, K, dtype, seed=0)
    assert not status.startswith("wrong"), \
        f"SILENT-WRONG {form} {M}x{N}x{K} {dtype}: rel_err {detail}"
    assert not status.startswith("crash"), \
        f"CRYPTIC-CRASH {form} {M}x{N}x{K} {dtype}: {status} {detail}"


@requires
@pytest.mark.parametrize("form,M,N,K,dtype,al,bl,cl", _transposed_cells())
def test_matmul_transposed_strided_correct_or_refuse(form, M, N, K, dtype, al, bl, cl):
    status, detail = _run_strided_cell(form, M, N, K, dtype, al, bl, cl, seed=0)
    tag = f"{form} {M}x{N}x{K} {dtype} A={al} B={bl} C={cl}"
    assert not status.startswith("wrong"), f"SILENT-WRONG {tag}: rel_err {detail}"
    assert not status.startswith("crash"), f"CRYPTIC-CRASH {tag}: {status} {detail}"


@requires
@pytest.mark.parametrize("form,M,N,K,dtype,al,bl,cl", _stdname_cells())
def test_matmul_stdname_correct_or_refuse(form, M, N, K, dtype, al, bl, cl):
    """STANDARD ``stride_*`` arg-naming (BLOCKER 2) must be correct or refuse —
    the naming must NOT route a transposed operand to the row-major name-match guess."""
    status, detail = _run_strided_cell(form, M, N, K, dtype, al, bl, cl, seed=0)
    tag = f"{form} {M}x{N}x{K} {dtype} A={al} B={bl} C={cl}"
    assert not status.startswith("wrong"), f"SILENT-WRONG {tag}: rel_err {detail}"
    assert not status.startswith("crash"), f"CRYPTIC-CRASH {tag}: {status} {detail}"


@requires
@pytest.mark.parametrize("form,M,N,K,dtype,al,bl,cl", _sliced_cells())
def test_matmul_sliced_correct_or_refuse(form, M, N, K, dtype, al, bl, cl):
    """SLICED operands (row stride != matrix dim, contiguous inner; BLOCKER 1) must
    use the inferred row stride as the simdgroup leading dim — correct or refuse."""
    status, detail = _run_strided_cell(form, M, N, K, dtype, al, bl, cl, seed=0)
    tag = f"{form} {M}x{N}x{K} {dtype} A={al} B={bl} C={cl}"
    assert not status.startswith("wrong"), f"SILENT-WRONG {tag}: rel_err {detail}"
    assert not status.startswith("crash"), f"CRYPTIC-CRASH {tag}: {status} {detail}"


@requires
@pytest.mark.parametrize("Bz,M,N,K,dtype", _batched_cells())
def test_matmul_batched_correct_or_refuse(Bz, M, N, K, dtype):
    """A real 3-D BATCHED matmul (BLOCKER 4). Batched MMA is unimplemented, so this
    must REFUSE loudly (or be correct) — never compute batch 0's region for all."""
    status, detail = _run_batched_cell(Bz, M, N, K, dtype, seed=0)
    tag = f"batched Bz={Bz} {M}x{N}x{K} {dtype}"
    assert not status.startswith("wrong"), f"SILENT-WRONG {tag}: rel_err {detail}"
    assert not status.startswith("crash"), f"CRYPTIC-CRASH {tag}: {status} {detail}"


# --------------------------------------------------------------------------- deep run
def _deep(n_seeds):
    cells = _po2_cells() + _strided_cells()
    tcells = _transposed_cells() + _stdname_cells() + _sliced_cells()
    bcells = _batched_cells()
    tally = {"correct": 0, "refused": 0, "wrong": 0, "crash": 0}
    bad = []
    for seed in range(n_seeds):
        for (form, M, N, K, dt) in cells:
            status, detail = _run_cell(form, M, N, K, dt, seed=seed)
            kind = status.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind in ("wrong", "crash"):
                bad.append((seed, form, M, N, K, str(dt), status, detail))
        for (form, M, N, K, dt, al, bl, cl) in tcells:
            status, detail = _run_strided_cell(form, M, N, K, dt, al, bl, cl, seed=seed)
            kind = status.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind in ("wrong", "crash"):
                bad.append((seed, form, M, N, K, str(dt), f"{al}/{bl}/{cl}", status, detail))
        for (Bz, M, N, K, dt) in bcells:
            status, detail = _run_batched_cell(Bz, M, N, K, dt, seed=seed)
            kind = status.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind in ("wrong", "crash"):
                bad.append((seed, "batched", M, N, K, str(dt), f"Bz={Bz}", status, detail))
    ncells = len(cells) + len(tcells) + len(bcells)
    print(f"matmul fuzz: {tally} over {n_seeds} seed(s) x {ncells} cells")
    for b in bad[:40]:
        print("  BAD", b)
    return 1 if (tally["wrong"] or tally["crash"]) else 0


if __name__ == "__main__":
    sys.exit(_deep(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
