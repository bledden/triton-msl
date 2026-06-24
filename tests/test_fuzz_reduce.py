"""Systemic reduce/scan differential fuzzer (re-audit -> systemic gate, reduce surface).

The reduce surface (1-D/2-D/3-D x sum/max/min/argmax x axis x dtype x no-loop/in-loop x
under/exact/over-fill of the threadgroup) has produced silent-wrongs across many
re-audits — under-fill, over-fill, i32 precision, i64 truncation, sub-warp argmax,
in-loop accumulation. This is the reduce analog of tests/test_fuzz_matmul.py: it sweeps
that cross-product and asserts the ONE invariant per cell:

    correct (vs torch, dtype-appropriate tol)  OR  loud MetalNonRecoverableError

never silent-wrong (computed-but-wrong) and never a cryptic crash (any other exception).

Run deep:  TRITON_MSL_FALLBACK=error python tests/test_fuzz_reduce.py 3
"""
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


def _clear_cache():
    for d in (os.path.expanduser("~/.cache/triton_msl"),
              os.path.expanduser("~/.triton/cache")):
        shutil.rmtree(d, ignore_errors=True)


def _is_float(dt):
    return dt in (torch.float32, torch.float16, torch.bfloat16)


def _tol(dt):
    return dict(rtol=3e-3, atol=3e-3) if dt == torch.float32 else dict(rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- kernels
@triton.jit
def _r1d_sum(a, o, N: tl.constexpr):
    tl.store(o, tl.sum(tl.load(a + tl.arange(0, N)), 0))


@triton.jit
def _r1d_max(a, o, N: tl.constexpr):
    tl.store(o, tl.max(tl.load(a + tl.arange(0, N)), 0))


@triton.jit
def _r1d_argmax(a, o, N: tl.constexpr):
    tl.store(o, tl.argmax(tl.load(a + tl.arange(0, N)), 0))


@triton.jit
def _r1d_inloop_sum(a, o, N: tl.constexpr, T: tl.constexpr):
    acc = tl.zeros((), tl.float32) if False else 0.0
    for t in range(0, T):
        acc += tl.sum(tl.load(a + t * N + tl.arange(0, N)), 0)
    tl.store(o, acc)


@triton.jit
def _r2d_sum_ax1(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    tl.store(o + i, tl.sum(tl.load(a + i[:, None] * N + j[None, :]), 1))


@triton.jit
def _r2d_max_ax1(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    tl.store(o + i, tl.max(tl.load(a + i[:, None] * N + j[None, :]), 1))


@triton.jit
def _r2d_argmax_ax1(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    _v, idx = tl.max(tl.load(a + i[:, None] * N + j[None, :]), 1, return_indices=True)
    tl.store(o + i, idx)


@triton.jit
def _r2d_sum_ax0(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    tl.store(o + j, tl.sum(tl.load(a + i[:, None] * N + j[None, :]), 0))


@triton.jit
def _r2d_inloop_sum_ax1(a, o, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M); acc = tl.zeros((M,), tl.float32)
    for t in range(0, T):
        j = tl.arange(0, N)
        acc += tl.sum(tl.load(a + t * M * N + i[:, None] * N + j[None, :]), 1)
    tl.store(o + i, acc)


@triton.jit
def _r3d_sum_ax2(a, o, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    b = tl.arange(0, B); r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(a + b[:, None, None] * R * C + r[None, :, None] * C + c[None, None, :])
    tl.store(o + b[:, None] * R + r[None, :], tl.sum(x, 2))


@triton.jit
def _r3d_sum_ax0(a, o, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    b = tl.arange(0, B); r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(a + b[:, None, None] * R * C + r[None, :, None] * C + c[None, None, :])
    tl.store(o + r[:, None] * C + c[None, :], tl.sum(x, 0))


# --------------------------------------------------------------------------- runner
def _ref(form, A):
    if form == "1d_sum":
        return A.float().sum()
    if form == "1d_max":
        return A.float().max()
    if form == "1d_argmax":
        return A.float().argmax()
    if form == "1d_inloop_sum":
        return A.float().sum()
    if form == "2d_sum_ax1":
        return A.float().sum(1)
    if form == "2d_max_ax1":
        return A.float().max(1).values
    if form == "2d_argmax_ax1":
        return A.float().argmax(1)
    if form == "2d_sum_ax0":
        return A.float().sum(0)
    if form == "2d_inloop_sum_ax1":
        return A.float().sum(dim=(0, 2))
    if form == "3d_sum_ax2":
        return A.float().sum(2)
    if form == "3d_sum_ax0":
        return A.float().sum(0)
    raise ValueError(form)


def _run_cell(form, dims, dtype, seed):
    """Return ('correct'|'refused'|'wrong:<rel>'|'crash:<type>', detail).

    Cache is NOT cleared per cell: Triton specializes (and keys the metallib) by dtype +
    constexpr shape, so distinct cells never collide, and clearing the on-disk cache
    mid-session races Triton's in-memory<->disk state (spurious FileNotFoundError).
    """
    torch.manual_seed(seed)
    is_arg = "argmax" in form
    odt = torch.int32 if is_arg else dtype
    try:
        if form in ("1d_sum", "1d_max", "1d_argmax"):
            (N,) = dims
            A = (torch.randn(N, device="mps") if _is_float(dtype)
                 else torch.randint(-50, 50, (N,), device="mps", dtype=dtype))
            o = torch.empty(1, device="mps", dtype=odt)
            {"1d_sum": _r1d_sum, "1d_max": _r1d_max, "1d_argmax": _r1d_argmax}[form][(1,)](A, o, N=N)
        elif form == "1d_inloop_sum":
            N, T = dims
            A = torch.randn(T, N, device="mps", dtype=dtype) if _is_float(dtype) else \
                torch.randint(-20, 20, (T, N), device="mps", dtype=dtype)
            o = torch.empty(1, device="mps", dtype=odt)
            _r1d_inloop_sum[(1,)](A, o, N=N, T=T)
        elif form in ("2d_sum_ax1", "2d_max_ax1", "2d_argmax_ax1", "2d_sum_ax0"):
            M, N = dims
            A = torch.randn(M, N, device="mps") if _is_float(dtype) else \
                torch.randint(-50, 50, (M, N), device="mps", dtype=dtype)
            osz = N if form == "2d_sum_ax0" else M
            o = torch.empty(osz, device="mps", dtype=odt)
            {"2d_sum_ax1": _r2d_sum_ax1, "2d_max_ax1": _r2d_max_ax1,
             "2d_argmax_ax1": _r2d_argmax_ax1, "2d_sum_ax0": _r2d_sum_ax0}[form][(1,)](A, o, M=M, N=N)
        elif form == "2d_inloop_sum_ax1":
            M, N, T = dims
            A = torch.randn(T, M, N, device="mps", dtype=dtype)
            o = torch.empty(M, device="mps", dtype=odt)
            _r2d_inloop_sum_ax1[(1,)](A, o, M=M, N=N, T=T)
        elif form in ("3d_sum_ax2", "3d_sum_ax0"):
            B, R, C = dims
            A = torch.randn(B, R, C, device="mps") if _is_float(dtype) else \
                torch.randint(-30, 30, (B, R, C), device="mps", dtype=dtype)
            osz = (R, C) if form == "3d_sum_ax0" else (B, R)
            o = torch.empty(osz, device="mps", dtype=odt)
            {"3d_sum_ax2": _r3d_sum_ax2, "3d_sum_ax0": _r3d_sum_ax0}[form][(1,)](A, o, B=B, R=R, C=C)
        else:
            return ("crash:badform", form)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return ("refused", None)
    except FileNotFoundError:
        # transient on-disk cache race (Triton in-memory<->disk); retry once cleanly
        _clear_cache()
        try:
            return _run_cell(form, dims, dtype, seed + 100000)
        except MetalNonRecoverableError:
            return ("refused", None)
        except Exception as e:                             # noqa: BLE001
            return (f"crash:{type(e).__name__}", str(e)[:80])
    except Exception as e:                                 # noqa: BLE001
        return (f"crash:{type(e).__name__}", str(e)[:80])
    ref = _ref(form, A)
    got = o.reshape(ref.shape) if hasattr(ref, "shape") and ref.dim() > 0 else o[0]
    if is_arg:
        ok = bool((got.long().cpu() == ref.long().cpu()).all())
        return ("correct" if ok else "wrong:idx", None)
    g = got.float().cpu(); e = ref.float().cpu()
    err = (g - e).abs().max().item()
    scale = max(e.abs().max().item(), 1e-6)
    t = _tol(dtype)
    return ("correct" if err <= t["atol"] + t["rtol"] * scale else f"wrong:{err/scale:.2e}", err / scale)


_FLOAT = [torch.float32, torch.float16, torch.bfloat16]
_INT = [torch.int32, torch.int64]


def _cells():
    cells = []
    for N in [16, 32, 64, 128, 256, 512, 1024]:
        for dt in _FLOAT + _INT:
            cells.append(("1d_sum", (N,), dt))
            cells.append(("1d_max", (N,), dt))
        for dt in _FLOAT:
            cells.append(("1d_argmax", (N,), dt))
    for dt in _FLOAT + _INT:
        cells.append(("1d_inloop_sum", (64, 4), dt))
    for (M, N) in [(8, 16), (8, 32), (16, 16), (16, 32), (32, 32), (8, 64), (16, 64)]:
        for dt in _FLOAT:
            for form in ("2d_sum_ax1", "2d_max_ax1", "2d_argmax_ax1", "2d_sum_ax0"):
                cells.append((form, (M, N), dt))
        cells.append(("2d_sum_ax1", (M, N), torch.int32))
    for (M, N) in [(8, 16), (8, 64), (16, 32)]:
        for dt in _FLOAT:
            cells.append(("2d_inloop_sum_ax1", (M, N, 3), dt))
    for (B, R, C) in [(2, 4, 4), (2, 8, 8), (4, 4, 8)]:
        for dt in _FLOAT + _INT:
            cells.append(("3d_sum_ax2", (B, R, C), dt))
            cells.append(("3d_sum_ax0", (B, R, C), dt))
    return cells


# --------------------------------------------------------------------------- pytest
@requires
@pytest.mark.parametrize("form,dims,dtype", _cells())
def test_reduce_correct_or_refuse(form, dims, dtype):
    status, detail = _run_cell(form, dims, dtype, seed=0)
    assert not status.startswith("wrong"), \
        f"SILENT-WRONG {form} {dims} {dtype}: {status} ({detail})"
    assert not status.startswith("crash"), \
        f"CRYPTIC-CRASH {form} {dims} {dtype}: {status} {detail}"


def _deep(n_seeds):
    cells = _cells()
    tally = {}
    bad = []
    for seed in range(n_seeds):
        for (form, dims, dt) in cells:
            status, detail = _run_cell(form, dims, dt, seed=seed)
            kind = status.split(":")[0]
            tally[kind] = tally.get(kind, 0) + 1
            if kind in ("wrong", "crash"):
                bad.append((seed, form, dims, str(dt), status, detail))
    print(f"reduce fuzz: {tally} over {n_seeds} seed(s) x {len(cells)} cells")
    for b in bad[:60]:
        print("  BAD", b)
    return 1 if (tally.get("wrong") or tally.get("crash")) else 0


if __name__ == "__main__":
    sys.exit(_deep(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
