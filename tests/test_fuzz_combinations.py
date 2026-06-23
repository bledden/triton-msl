"""Seeded property-based COMBINATION fuzzer — the systemic net for the bug class
that re-audits #5/#6 surfaced: op INTERACTIONS, not single ops.

Every recent silent-wrong came from PAIRING ops, where a hand-written fast-path
template silently dropped or mis-handled an op the generic path lowers correctly:
a looped matmul + epilogue (epilogue dropped), reduce + cooperative op (tail
under-computed), scan inside a loop (cross-iteration buffer race), reduce over a
splat (summed num_threads copies), matmul-epilogue NaN-drop, ...

This fuzzer exercises op COMBINATIONS over sizes spanning the dispatched-thread
(~num_warps*32) and 1024-thread boundaries, and asserts the one invariant:

    correct (matches torch) OR loud MetalNonRecoverableError — never silently
    wrong, never a cryptic crash, never an OOB write (canary-checked).

torch is the reference. Each runner returns (out, ref, canary_or_None, tol, desc)
or raises MetalNonRecoverableError (a clean refusal == pass).
"""
import os
import random
import shutil

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

_TOL = {torch.float32: 3e-3, torch.float16: 3e-2, torch.bfloat16: 6e-2}
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SENT = 2048.0   # exactly representable in fp32/fp16/bf16, far from any randn value


def _clear():
    # Clear ONLY the triton-msl codegen cache (force re-codegen). Never delete the
    # shared content-addressed ~/.triton/cache (it races sibling pipelines).
    shutil.rmtree(os.path.expanduser("~/.cache/triton_msl"), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Combination kernels (the fast-path interactions under test)
# --------------------------------------------------------------------------- #
@triton.jit
def _mm_epi(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, OP: tl.constexpr):
    # NON-looped single dot -> the fused-epilogue template (which applies the epilogue
    # correctly for any size). The LOOPED variant is pinned to refuse in
    # test_matmul_epilogue.py::test_matmul_kloop_epilogue_refuses_not_dropped.
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    acc = tl.dot(tl.load(a + om[:, None] * K + ok[None, :]),
                 tl.load(b + ok[:, None] * N + on[None, :]))
    if OP == 0:
        r = acc * 2.0 + 1.0
    elif OP == 1:
        r = tl.maximum(acc, 0.0)
    elif OP == 2:
        r = tl.math.fma(acc, 3.0, 0.5)
    else:
        r = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    tl.store(c + om[:, None] * N + on[None, :], r)


@triton.jit
def _scan_gather(a, idx, o, N: tl.constexpr):
    i = tl.arange(0, N)
    s = tl.cumsum(tl.load(a + i), 0)
    tl.store(o + i, tl.gather(s, tl.load(idx + i), axis=0))


@triton.jit
def _cumsum_loop(a, o, N: tl.constexpr, REP: tl.constexpr):
    i = tl.arange(0, N); x = tl.load(a + i)
    for _ in range(REP):
        x = tl.cumsum(x, 0)
    tl.store(o + i, x)


@triton.jit
def _splat_reduce(o, N: tl.constexpr, V: tl.constexpr):
    tl.store(o, tl.sum(tl.full((N,), V, tl.float32), 0))


@triton.jit
def _trans_ew(a, o, S: tl.constexpr):
    i = tl.arange(0, S); j = tl.arange(0, S)
    t = tl.trans(tl.load(a + i[:, None] * S + j[None, :]))
    tl.store(o + i[:, None] * S + j[None, :], t * 2.0 + 1.0)


@triton.jit
def _cat_ew(a, b, o, N: tl.constexpr):
    i = tl.arange(0, N)
    z = tl.cat(tl.load(a + i), tl.load(b + i), can_reorder=True)
    tl.store(o + tl.arange(0, 2 * N), z * 2.0)


@triton.jit
def _reduce_bcast(a, o, R, C, BC: tl.constexpr):
    r = tl.program_id(0); cc = tl.arange(0, BC)
    x = tl.load(a + r * C + cc, mask=cc < C, other=0.0)
    s = tl.sum(x, 0)
    tl.store(o + r * C + cc, (x - s), mask=cc < C)   # row-centering: reduce then broadcast-sub


# --------------------------------------------------------------------------- #
# Runners: (metal_out_float, torch_ref_float, canary_or_None, tol, desc) or raise
# --------------------------------------------------------------------------- #
def _run_mm_epi(rng):
    dt = rng.choice(_DTYPES); out = rng.choice(_DTYPES)
    M = rng.choice([16, 32, 64, 128]); N = rng.choice([16, 32, 64, 128]); K = rng.choice([16, 32, 64])
    op = rng.choice([0, 1, 2, 3])
    A = torch.randn(M, K, device="mps", dtype=dt) * 0.3
    B = torch.randn(K, N, device="mps", dtype=dt) * 0.3
    C = torch.empty(M, N, device="mps", dtype=out)
    _mm_epi[(1,)](A, B, C, M=M, N=N, K=K, OP=op)
    mm = A.float() @ B.float()
    ref = {0: mm * 2.0 + 1.0, 1: mm.clamp(min=0), 2: mm * 3.0 + 0.5,
           3: mm.clamp(-1.0, 1.0)}[op]
    return C.float(), ref, None, max(_TOL[dt], _TOL[out]), f"mm+epi op{op} {dt}->{out} M{M}N{N}K{K}"


def _run_scan_gather(rng):
    N = rng.choice([8, 32, 64, 128, 256, 512, 1024, 2048])
    torch.manual_seed(rng.randrange(1 << 30))
    a = torch.randn(N, device="mps")
    idx = torch.randint(0, N, (N,), device="mps", dtype=torch.int32)
    o = torch.empty(N, device="mps")
    _scan_gather[(1,)](a, idx, o, N=N)
    ref = torch.cumsum(a.cpu(), 0)[idx.cpu().long()].to("mps")
    return o.float(), ref, None, 1e-3, f"scan+gather N{N}"


def _run_cumsum_loop(rng):
    N = rng.choice([64, 128, 256, 512, 1024]); rep = rng.choice([1, 2, 3])
    a = torch.ones(N, device="mps")
    o = torch.empty(N, device="mps")
    _cumsum_loop[(1,)](a, o, N=N, REP=rep)
    ref = a.cpu()
    for _ in range(rep):
        ref = torch.cumsum(ref, 0)
    return o.float(), ref.to("mps"), None, 1e-2, f"cumsum-loop N{N} rep{rep}"


def _run_splat_reduce(rng):
    N = rng.choice([8, 16, 32, 64, 128, 256]); v = float(rng.choice([1.0, 2.0, 3.0]))
    o = torch.empty(1, device="mps")
    _splat_reduce[(1,)](o, N=N, V=v)
    return o.float(), torch.tensor([N * v], device="mps"), None, 1e-3, f"splat-reduce N{N} v{v}"


def _run_trans_ew(rng):
    S = rng.choice([8, 16, 32, 64])
    torch.manual_seed(rng.randrange(1 << 30))
    a = torch.randn(S, S, device="mps"); o = torch.empty(S, S, device="mps")
    _trans_ew[(1,)](a, o, S=S)
    return o.float(), a.t().contiguous() * 2.0 + 1.0, None, 1e-3, f"trans+ew S{S}"


def _run_cat_ew(rng):
    N = rng.choice([8, 32, 64, 128, 256, 512, 1024])
    torch.manual_seed(rng.randrange(1 << 30))
    a = torch.randn(N, device="mps"); b = torch.randn(N, device="mps")
    PAD = 64
    obuf = torch.full((2 * N + PAD,), _SENT, device="mps"); o = obuf[:2 * N]
    _cat_ew[(1,)](a, b, o, N=N)
    canary = int((obuf[2 * N:] != _SENT).sum().item())
    ref = torch.cat([a, b]) * 2.0
    return o.float(), ref, canary, 1e-3, f"cat+ew N{N}"


def _run_reduce_bcast(rng):
    R = rng.choice([1, 4, 8]); C = rng.choice([8, 16, 32, 64, 128])
    BC = 1 << (C - 1).bit_length()
    torch.manual_seed(rng.randrange(1 << 30))
    a = torch.randn(R, C, device="mps"); o = torch.empty(R, C, device="mps")
    _reduce_bcast[(R,)](a, o, R, C, BC=BC)
    ref = a - a.sum(dim=1, keepdim=True)
    return o.float(), ref, None, 3e-3, f"reduce+bcast R{R} C{C}"


_RUNNERS = [_run_mm_epi, _run_scan_gather, _run_cumsum_loop, _run_splat_reduce,
            _run_trans_ew, _run_cat_ew, _run_reduce_bcast]


@requires
@pytest.mark.parametrize("seed", range(120))
def test_fuzz_combinations(seed):
    rng = random.Random(seed)
    runner = rng.choice(_RUNNERS)
    _clear()
    try:
        out, ref, canary, tol, desc = runner(rng)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # loud refusal satisfies the never-silent-wrong contract
    except Exception as e:
        pytest.fail(f"[seed {seed}] cryptic crash (not a clean refusal): "
                    f"{type(e).__name__}: {str(e)[:200]}")
    if canary is not None and canary != 0:
        pytest.fail(f"[seed {seed}] OOB WRITE ({canary} bytes past buffer): {desc}")
    rel = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < tol, f"[seed {seed}] SILENT-WRONG: rel_err {rel:.3e} > {tol}: {desc}"
