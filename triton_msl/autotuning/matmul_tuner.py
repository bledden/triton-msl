# triton_msl/autotuning/matmul_tuner.py
"""Deterministic, occupancy-gated selection of the fast-matmul register blocking
(rr, rc) per shape.

This is NOT a GPU-timed autotuner. An earlier version timed candidates and disk-
cached the winner; measurement through the production dispatch showed that tuning
is within noise for aligned shapes (the hand-written (4,4) fast path is already as
fast as any blocking) AND it introduced a silent-wrong (an N-contract miss). So the
tuning is removed. What DID measure as a real, reachable win is **fast-path
ENABLEMENT for unaligned M**: the fixed (4,4) needs M%32==0, so M%32!=0 matmuls
fall to the ~2.4 TFLOP/s generic path; a smaller tile (rr=2 -> tile_m=16, rr=1 ->
tile_m=8) keeps them on the ~8-11 TFLOP/s fast path (3-5x) -- when there is enough
parallelism. For small/low-occupancy shapes the fine tile LOSES to generic, so we
gate on occupancy and fall back to generic below it (never-regress).

All candidates compute a CORRECT matmul (the kernel is correct for any blocking
meeting its size contract), so selection is perf-only -- never correctness.
"""
import math
import os

# Ordered preference: (4,4) first (the default / aligned common case, unchanged),
# then rr=2 tiles (serve M%16), then rr=1 tiles (serve M%8). Within an rr, larger rc
# first (more register blocking). Register budget rr*rc <= 32.
# NOTE: (4,8) is effectively UNREACHABLE via best_rrrc — it needs N%256, which implies
# N%128, so (4,4) is always also valid and is returned first. It is kept only for
# valid_candidates completeness/symmetry; it is never *selected*.
CANDIDATES = [(4, 4), (4, 2), (4, 8),
              (2, 8), (2, 4), (2, 2),
              (1, 8), (1, 4), (1, 2),
              # rc=1 (per-simdgroup strip width 8) — serve N % 8 == 0 but N % 16 != 0,
              # the finest column tiling; tried last (smallest tile) so an N%16-aligned
              # shape still prefers a wider rc.
              (4, 1), (2, 1), (1, 1)]

_CORES = None


def _gpu_cores(default=40):
    global _CORES
    if _CORES is None:
        try:
            from triton_msl.backend.device_detect import get_device_info
            _CORES = get_device_info().gpu_core_count or default
        except Exception:
            _CORES = default
    return _CORES


def _enabled():
    # Kill-switch. Default ON: the selector is deterministic + never-regress
    # (occupancy-gated), so it is safe to leave on; =0 pins the fixed (4,4) path.
    return os.environ.get("TRITON_MSL_MATMUL_AUTOTUNE", "1") != "0"


def valid_candidates(M, N, K):
    """Candidates whose size contract this shape satisfies. The fast template needs
    M % (8*rr) == 0 (row tile), N % (8*rc) == 0 (the per-simdgroup STRIP width — each of
    the 4 simdgroups owns an 8*rc-wide strip and the kernel's ``if (col0 >= N) return``
    guard skips whole strips beyond N, so N need only align to the strip, NOT the full
    32*rc threadgroup tile), and K % 8 == 0 (8-deep K fragments).

    Relaxed from N % (32*rc) (the matmul N%32 perf cliff, 2026-06-26): the dispatch already
    gates on exactly N % sel_strip (= N % (8*rc)) and the fast template is verified byte-
    exact there, but the tuner's stricter 32*rc gate was returning None for any N%32!=0
    shape — forcing it to the ~1.2-1.8 TF generic path (BELOW the ~2.4 generic floor) even
    when N%8==0. With the strip-width gate + the rc=1 candidates, N%8==0 shapes (odd
    vocab/hidden dims) now reach the fast path. N%8!=0 still routes to generic (a partial
    strip would write past N)."""
    if K % 8 != 0:
        return []
    return [(rr, rc) for (rr, rc) in CANDIDATES
            if M % (8 * rr) == 0 and N % (8 * rc) == 0]


def _occupancy_ok(M, N, rr, rc):
    """Enough threadgroups to beat the generic fallback. The fast-vs-generic
    crossover for fine tiles is noisy/non-monotonic (measured: a shape at 4x cores
    can still lose), so the gate is CONSERVATIVE at 8x cores -- reliably >generic in
    measurement -- to guarantee no regression (low-occupancy shapes route to generic)."""
    n_groups = math.ceil(M / (8 * rr)) * math.ceil(N / (32 * rc))
    return n_groups >= 8 * _gpu_cores()


def best_rrrc(msl_dtype, msl_out, M, N, K, runtime=None, cache_dir=None):
    """Deterministic (rr,rc) for this shape, or None to use the generic path.

    - (4,4) when valid: the aligned common case, behavior unchanged.
    - else (M%32!=0 or N%128!=0): the largest valid tile (rr=2 before rr=1) that
      passes the occupancy gate -- the fast-path-enablement win. If nothing valid
      passes the gate, return None so the caller uses the generic path.
    No GPU timing; `runtime`/`cache_dir` are accepted for signature compatibility
    and ignored. The kill-switch pins (4,4)-or-None."""
    valid = valid_candidates(M, N, K)
    if not valid:
        return None
    if not _enabled():
        return (4, 4) if (4, 4) in valid else None
    if (4, 4) in valid:
        return (4, 4)
    for (rr, rc) in valid:            # CANDIDATES order: largest tile first
        if _occupancy_ok(M, N, rr, rc):
            return (rr, rc)
    return None                       # low occupancy -> generic (never-regress)
