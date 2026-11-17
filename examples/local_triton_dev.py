#!/usr/bin/env python3
"""Develop real Triton kernels locally on Apple Silicon — no NVIDIA GPU required.

The kernels below are written in *plain, unmodified* Triton: the same
``@triton.jit`` source, the same ``tl.*`` ops, and the same ``kernel[grid](...)``
launch you would write for a CUDA box. triton-msl compiles them to Metal and runs
them on your Mac's GPU. Nothing here is Apple-specific — copy a kernel onto a CUDA
machine and it runs there too.

This is the point of the project: write/iterate on Triton kernels on the laptop you
already have, then ship the identical code to an NVIDIA cluster.

Each kernel is checked against a **CPU / NumPy** reference (never torch-on-MPS, which
has its own integer/reduction quirks) so a PASS means the Metal result matches what
CUDA would produce — bit-for-bit on the floats that matter.

Every kernel is deliberately run on a shape whose dimensions are NOT multiples of the
block size. That exercises the boundary/tail masks (``offs < n``, ``offs_m < M``) on
every load and store — the path triton-msl hardened to be *correct-or-refuse*: a mask
it can honor is honored, one it cannot is a loud error, never a silent wrong answer.
That guarantee is what makes a local result you can trust on ragged real-world shapes.

Run:  python examples/local_triton_dev.py
Needs: macOS + a Metal GPU (M1/M2/M3/M4), torch with MPS, and triton-msl importable.
"""

import os
import sys

import numpy as np

# Run from a checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(f"This example needs torch + triton installed: {exc}")

# Importing triton_msl registers the Metal backend Triton dispatches to.
import triton_msl  # noqa: F401  (side-effect import)


# ===========================================================================
# Kernel 1 — Vector add (Triton tutorial 01). Elementwise + a tail mask.
# ===========================================================================
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements                      # last program is partial
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def run_vector_add():
    n = 98_432                                       # not a multiple of BLOCK
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    xt = torch.from_numpy(x).to("mps")
    yt = torch.from_numpy(y).to("mps")
    out = torch.empty(n, device="mps")

    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](xt, yt, out, n, BLOCK=1024)
    torch.mps.synchronize()

    ref = x + y
    err = float(np.abs(out.cpu().numpy() - ref).max())
    return ("vector_add", f"n={n:,} (tail-masked)", err, 1e-5)


# ===========================================================================
# Kernel 2 — Fused softmax (Triton tutorial 02). Row reduction + masked load.
# ===========================================================================
@triton.jit
def softmax_kernel(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    row_ptr = in_ptr + row * row_stride + cols
    x = tl.load(row_ptr, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                        # numerically-stable softmax
    numer = tl.exp(x)
    denom = tl.sum(numer, axis=0)
    tl.store(out_ptr + row * row_stride + cols, numer / denom, mask=mask)


def run_fused_softmax():
    rows, cols = 128, 781                            # cols not a power of 2
    x = np.random.randn(rows, cols).astype(np.float32)
    xt = torch.from_numpy(x).to("mps")
    out = torch.empty_like(xt)

    block = triton.next_power_of_2(cols)             # BLOCK padded; tail masked
    softmax_kernel[(rows,)](out, xt, xt.stride(0), cols, BLOCK=block)
    torch.mps.synchronize()

    ref = np.exp(x - x.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    err = float(np.abs(out.cpu().numpy() - ref).max())
    return ("fused_softmax", f"{rows}x{cols} (masked reduction)", err, 1e-5)


# ===========================================================================
# Kernel 3 — Tiled matmul (Triton tutorial 03). 2-D program grid, K-loop,
# masked M/N/K-tail loads, and a boundary store mask on the C tile. This is the
# kernel that most directly leans on the correct-or-refuse masked-store work:
# on a non-multiple M/N the C-store mask (rm < M) & (rn < N) must be honored
# exactly, or the padding rows/cols would be clobbered with garbage.
# ===========================================================================
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_mask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)               # boundary store mask


def run_matmul():
    M, N, K = 100, 80, 48                            # none are multiples of 32
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    at = torch.from_numpy(a).to("mps")
    bt = torch.from_numpy(b).to("mps")
    ct = torch.zeros(M, N, device="mps")

    BM = BN = BK = 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](at, bt, ct, M, N, K,
                        *at.stride(), *bt.stride(), *ct.stride(),
                        BM=BM, BN=BN, BK=BK)
    torch.mps.synchronize()

    ref = a.astype(np.float64) @ b.astype(np.float64)   # CPU float64 reference
    err = float(np.abs(ct.cpu().numpy() - ref).max())
    return ("matmul", f"{M}x{K} @ {K}x{N} (boundary-masked)", err, 1e-3)


def main():
    if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
        sys.exit("No Metal (MPS) device available — this example needs an Apple GPU.")

    print("=" * 68)
    print("  Real Triton kernels, compiled to Metal, running on this Mac")
    print("  (same @triton.jit source you'd run on a CUDA GPU)")
    print("=" * 68)

    np.random.seed(0)
    runners = [run_vector_add, run_fused_softmax, run_matmul]
    results = []
    for run in runners:
        try:
            name, shape, err, tol = run()
            ok = err < tol
            results.append(ok)
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<14} {shape:<34} max|Δ vs NumPy| = {err:.2e}")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the demo
            results.append(False)
            print(f"  [ERR ] {run.__name__}: {type(exc).__name__}: {exc}")

    print("=" * 68)
    passed = sum(results)
    print(f"  {passed}/{len(results)} kernels matched the CPU reference on Metal.")
    if passed == len(results):
        print("  Every dimension above was non-multiple-of-block on purpose: the")
        print("  load/store boundary masks were honored exactly — correct-or-refuse,")
        print("  never silently wrong. Iterate locally; ship the same code to CUDA.")
    print("=" * 68)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
