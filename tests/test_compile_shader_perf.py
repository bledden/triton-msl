"""Perf gate: compile_shader zero-copy fast-path must deliver ≥ 250 GB/s for
vector_add@16M on MPS.

Rationale: the host-round-trip path (TRITON_MSL_COMPILE_SHADER=0) achieves
~28 GB/s. The fast-path should reach ≥ 250 GB/s (~45% of 546 GB/s peak on
M4 Max). If this test fails, the fast-path is either not firing through the
real Triton→driver path or measurement includes unexpected overhead.

Run with:
    pytest tests/test_compile_shader_perf.py -v
"""

import os
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton.testing import do_bench

    _HAS_MPS = torch.backends.mps.is_available()
    _HAS_CS = hasattr(torch.mps, "compile_shader")
except Exception:
    _HAS_MPS = False
    _HAS_CS = False

_ELIGIBLE = _HAS_MPS and _HAS_CS and os.environ.get("TRITON_MSL_COMPILE_SHADER", "1") != "0"

requires_fast_path = pytest.mark.skipif(
    not _ELIGIBLE,
    reason="MPS + torch.mps.compile_shader needed and TRITON_MSL_COMPILE_SHADER must be '1' (default)",
)

# Peak M4 Max memory bandwidth (GB/s)
PEAK_BW = 546.0

# Gate: fast-path must reach at least this fraction of peak
MIN_GBPS = 250.0


# ---------------------------------------------------------------------------
# Kernel under test
# ---------------------------------------------------------------------------

@triton.jit
def _vadd_perf(A, B, OUT, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(OUT + o, tl.load(A + o, mask=m) + tl.load(B + o, mask=m), mask=m)


# ---------------------------------------------------------------------------
# Perf gate test
# ---------------------------------------------------------------------------

@requires_fast_path
def test_vector_add_fast_path_throughput():
    """vector_add@16M through compile_shader must achieve ≥ 250 GB/s."""
    N = 16 * 1024 * 1024
    BLOCK = 1024
    dtype = torch.float32

    A = torch.randn(N, device="mps", dtype=dtype)
    B = torch.randn(N, device="mps", dtype=dtype)
    OUT = torch.empty(N, device="mps", dtype=dtype)

    grid = (triton.cdiv(N, BLOCK),)

    def fn():
        _vadd_perf[grid](A, B, OUT, N, BLOCK=BLOCK)

    # Warmup: allow compilation + first few dispatches to settle.
    for _ in range(5):
        fn()
    torch.mps.synchronize()

    # Take min over 3 independent do_bench calls for stability.
    ms_runs = [do_bench(fn, warmup=25, rep=100, return_mode="min") for _ in range(3)]
    ms = min(ms_runs)

    # 3 tensors × N × sizeof(float32) bytes: read A, read B, write OUT
    bytes_moved = 3 * N * dtype.itemsize
    gbps = bytes_moved / ms * 1e-6  # bytes / ms → GB/s

    pct_peak = gbps / PEAK_BW * 100

    print(
        f"\n  vector_add@16M fast-path: {ms:.3f} ms → {gbps:.1f} GB/s "
        f"({pct_peak:.1f}% of {PEAK_BW} GB/s peak)"
    )

    assert gbps >= MIN_GBPS, (
        f"compile_shader fast-path throughput {gbps:.1f} GB/s < {MIN_GBPS} GB/s gate. "
        f"Fast-path may not be firing — check TRITON_MSL_COMPILE_SHADER=1 and "
        f"that torch.mps.compile_shader is available."
    )
