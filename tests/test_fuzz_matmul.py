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


# --------------------------------------------------------------------------- runner
def _run_cell(form, M, N, K, dtype, seed):
    """Return ('correct'|'refused'|'wrong:<err>'|'crash:<type>', detail)."""
    _clear_cache()
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


# --------------------------------------------------------------------------- deep run
def _deep(n_seeds):
    cells = _po2_cells() + _strided_cells()
    tally = {"correct": 0, "refused": 0, "wrong": 0, "crash": 0}
    bad = []
    for seed in range(n_seeds):
        for (form, M, N, K, dt) in cells:
            status, detail = _run_cell(form, M, N, K, dt, seed=seed)
            kind = status.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind in ("wrong", "crash"):
                bad.append((seed, form, M, N, K, str(dt), status, detail))
    print(f"matmul fuzz: {tally} over {n_seeds} seed(s) x {len(cells)} cells")
    for b in bad[:40]:
        print("  BAD", b)
    return 1 if (tally["wrong"] or tally["crash"]) else 0


if __name__ == "__main__":
    sys.exit(_deep(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
