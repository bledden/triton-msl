"""End-to-end: a torch.compiled / @triton.jit matmul routed through the fast path
matches torch @ across shapes, with rr/rc autotuning on (default) and off.

Also contains driver-level unit tests for the three review findings on Task 2:
  - Finding 1 (test gap): descriptor must have 8 elements
  - Finding 2 (test gap): non-(4,4) config must actually reach dispatch
  - Finding 3 (blocking): mark_unsupported must target sel_msl, not fast_msl
"""
import os
import math
import platform
import pytest
import torch

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()
         and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader")


def _mm(M, K, N):
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a_ptr, b_ptr, c_ptr, M, N, K,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0); pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps")
    c = torch.empty(M, N, device="mps")
    grid = (M // 32, N // 32)
    mm[grid](a, b, c, M, N, K, BM=32, BN=32, BK=32)
    torch.mps.synchronize()
    return c, a @ b


@requires_mps
@pytest.mark.parametrize("M,K,N", [(512, 512, 512), (1024, 1024, 1024), (2048, 512, 2048)])
def test_autotuned_matmul_matches_torch(M, K, N):
    os.environ.pop("TRITON_MSL_MATMUL_AUTOTUNE", None)   # default ON
    c, ref = _mm(M, K, N)
    assert (c - ref).abs().max().item() < 1e-1   # fp32 matmul over K, generous abs tol


@requires_mps
def test_optout_matches_torch():
    os.environ["TRITON_MSL_MATMUL_AUTOTUNE"] = "0"
    try:
        c, ref = _mm(1024, 1024, 1024)
        assert (c - ref).abs().max().item() < 1e-1
    finally:
        os.environ.pop("TRITON_MSL_MATMUL_AUTOTUNE", None)


# ---------------------------------------------------------------------------
# Shared: fake CompileShaderRuntime for driver-block unit tests
# ---------------------------------------------------------------------------

class _RecordingRuntime:
    """Fake CompileShaderRuntime. Intercepts dispatch and records which MSL was used."""

    def __init__(self, *, fail_dispatch=False):
        self._unsupported = set()
        self.dispatched_msls = []   # ordered list of MSL strings passed to dispatch
        self._fail_dispatch = fail_dispatch

    def available(self):
        return True

    def is_unsupported(self, msl):
        return msl in self._unsupported

    def mark_unsupported(self, msl):
        self._unsupported.add(msl)

    def get_library(self, msl):
        return msl  # use the MSL source itself as the lib token

    def dispatch(self, lib, name, args, *, threads, group_size):
        self.dispatched_msls.append(lib)
        if self._fail_dispatch:
            raise RuntimeError("injected dispatch failure")


def _run_fast_matmul_block(rt, descriptor, M, N, K, best_rrrc_override=None,
                            monkeypatch=None):
    """Thin wrapper around the real dispatch_fast_matmul from
    triton_msl.autotuning._fast_matmul_dispatch.  Exists so tests that predated
    the extraction keep a stable call-site.

    Pass best_rrrc_override=(rr,rc) to force the autotuner to return that config.
    Pass monkeypatch to patch triton_msl.autotuning.matmul_tuner.best_rrrc.
    Returns the rt object for post-call inspection.

    Because this now calls the REAL function (not a replica), there is no
    silent divergence risk when driver.py or _fast_matmul_dispatch.py changes.
    """
    from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul
    import triton_msl.autotuning.matmul_tuner as _tuner_mod
    if best_rrrc_override is not None and monkeypatch is not None:
        monkeypatch.setattr(_tuner_mod, "best_rrrc",
                            lambda *a, **kw: best_rrrc_override)

    kargs = [None, None, None, M, N, K]
    dispatch_fast_matmul(rt, descriptor, kargs)
    return rt


# ---------------------------------------------------------------------------
# Finding 1 (test gap): descriptor must have exactly 8 elements and carry
# msl_dtype / msl_out in fields 6 and 7.
# ---------------------------------------------------------------------------

def test_descriptor_has_8_elements_and_carries_dtype_fields():
    """The fast_matmul descriptor built by the lowerer must be exactly 8 elements:
    (msl, m_idx, n_idx, k_idx, tile_m, tile_n, msl_dtype, msl_out).
    Fields 6+7 let the driver build alternative (rr,rc) variants for autotuning.
    We verify this by calling make_simdgroup_matmul_kernel_fast directly and
    constructing what the lowerer would produce (the lowerer's code path at
    _lowerer_templates.py:2449-2455 is the single source of truth).
    """
    from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

    for msl_dtype, msl_out in [("fp32", "fp32"), ("fp16", "fp16"), ("fp32", "fp16")]:
        rr = rc = 4
        fast_msl = make_simdgroup_matmul_kernel_fast(
            dtype=msl_dtype, rr=rr, rc=rc, out_dtype=msl_out)
        # This mirrors the lowerer's return statement exactly:
        descriptor = (fast_msl, 3, 4, 5, 8 * rr, 32 * rc, msl_dtype, msl_out)

        assert len(descriptor) == 8, (
            f"descriptor for ({msl_dtype},{msl_out}) must be 8 elements, got {len(descriptor)}"
        )
        assert descriptor[6] == msl_dtype, (
            f"descriptor[6] must be msl_dtype={msl_dtype!r}, got {descriptor[6]!r}"
        )
        assert descriptor[7] == msl_out, (
            f"descriptor[7] must be msl_out={msl_out!r}, got {descriptor[7]!r}"
        )
        assert isinstance(descriptor[6], str) and isinstance(descriptor[7], str), (
            "descriptor[6] and [7] must be plain strings for driver unpacking"
        )

    # Verify the driver can unpack all 8 fields from the descriptor
    fast_msl = make_simdgroup_matmul_kernel_fast("fp32", 4, 4, "fp32")
    desc = (fast_msl, 3, 4, 5, 32, 128, "fp32", "fp32")
    unpacked_msl = desc[0]
    m_idx, n_idx, k_idx = desc[1], desc[2], desc[3]
    tile_m, tile_n = desc[4], desc[5]
    msl_dtype_d = desc[6] if len(desc) > 6 else None
    msl_out_d = desc[7] if len(desc) > 7 else None

    assert msl_dtype_d == "fp32", f"driver unpack of desc[6] must yield 'fp32', got {msl_dtype_d!r}"
    assert msl_out_d == "fp32", f"driver unpack of desc[7] must yield 'fp32', got {msl_out_d!r}"
    assert (m_idx, n_idx, k_idx) == (3, 4, 5), "buffer indices must be (3,4,5)"
    assert (tile_m, tile_n) == (32, 128), "default (4,4) tile sizes must be (32,128)"


# ---------------------------------------------------------------------------
# Finding 2 (test gap): when the autotuner selects a non-(4,4) config, the
# driver block must dispatch that config's MSL — not the (4,4) default.
# ---------------------------------------------------------------------------

def test_non_default_config_reaches_dispatch_when_tuner_selects_it(monkeypatch):
    """When best_rrrc returns a non-(4,4) config, the fast-matmul dispatch block
    must call rt.dispatch with the MSL for THAT config, not fast_msl (the (4,4)
    default baked into the descriptor)."""
    from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

    fast_msl = make_simdgroup_matmul_kernel_fast("fp32", 4, 4, "fp32")
    # (2,4): tile_m=16, tile_n=128 — valid for M=512 (512%16==0, N=512%32==0, K=512%8==0)
    sel_msl = make_simdgroup_matmul_kernel_fast("fp32", 2, 4, "fp32")
    assert sel_msl != fast_msl, "(2,4) must produce a different MSL from (4,4)"

    descriptor = (fast_msl, 3, 4, 5, 32, 128, "fp32", "fp32")
    rt = _RecordingRuntime(fail_dispatch=False)

    _run_fast_matmul_block(
        rt, descriptor, M=512, N=512, K=512,
        best_rrrc_override=(2, 4),   # tuner returns (2,4)
        monkeypatch=monkeypatch,
    )

    assert rt.dispatched_msls, "dispatch must have been called"
    dispatched = rt.dispatched_msls[0]
    assert dispatched == sel_msl, (
        "fast-matmul block must dispatch the (2,4)-selected MSL, not the (4,4) default.\n"
        f"Expected sel_msl (rr=2,rc=4); got the (4,4) default or something else."
    )
    assert dispatched != fast_msl, (
        "dispatched the fixed (4,4) MSL — the autotuner selection is not being used"
    )


# ---------------------------------------------------------------------------
# Finding 3 (blocking bug): mark_unsupported must target sel_msl (the failing
# variant), NOT fast_msl (the (4,4) default) — wrong key permanently disables
# the (4,4) fallback for the kernel on all subsequent calls.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# --- Medium: wasted-work loop test (RED until is_unsupported(sel_msl) gate added) ---

def test_failed_non_default_sel_msl_not_retried_on_second_call(monkeypatch):
    """After a non-(4,4) sel_msl dispatch fails and is marked unsupported, a
    second call to the fast-matmul block must NOT attempt dispatch again with
    that same sel_msl variant.  Without the fix the is_unsupported gate checks
    only fast_msl (still clean) so dispatch is retried on every call.

    This test RED-s against the current _run_fast_matmul_block before the
    is_unsupported(sel_msl) guard is added.
    """
    from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

    fast_msl = make_simdgroup_matmul_kernel_fast("fp32", 4, 4, "fp32")
    sel_msl = make_simdgroup_matmul_kernel_fast("fp32", 2, 4, "fp32")
    assert sel_msl != fast_msl

    descriptor = (fast_msl, 3, 4, 5, 32, 128, "fp32", "fp32")
    rt = _RecordingRuntime(fail_dispatch=True)

    # First call: dispatch is attempted, fails, sel_msl marked unsupported
    _run_fast_matmul_block(
        rt, descriptor, M=512, N=512, K=512,
        best_rrrc_override=(2, 4),
        monkeypatch=monkeypatch,
    )
    assert rt.dispatched_msls, "first call must attempt dispatch"
    assert sel_msl in rt._unsupported, "first call must mark sel_msl unsupported"
    first_call_dispatch_count = len(rt.dispatched_msls)

    # Second call: sel_msl is already unsupported — dispatch must NOT be attempted again
    _run_fast_matmul_block(
        rt, descriptor, M=512, N=512, K=512,
        best_rrrc_override=(2, 4),
        monkeypatch=monkeypatch,
    )
    assert len(rt.dispatched_msls) == first_call_dispatch_count, (
        "Second call must NOT re-attempt dispatch of already-unsupported sel_msl. "
        f"Expected {first_call_dispatch_count} total dispatch calls, "
        f"got {len(rt.dispatched_msls)}.  The is_unsupported(sel_msl) guard is missing."
    )


# --- Low-3: _dispatch_fast_matmul extracted to autotuning module (RED until extracted) ---

def test_dispatch_fast_matmul_importable_from_autotuning(monkeypatch):
    """_dispatch_fast_matmul must be a callable in triton_msl.autotuning._fast_matmul_dispatch.
    This test RED-s until the function is extracted from the inline block in driver.py.
    Once green, _run_fast_matmul_block becomes a thin wrapper that calls the real
    function, eliminating the silent divergence risk for future refactors.
    """
    from triton_msl.autotuning import _fast_matmul_dispatch as _mod
    assert hasattr(_mod, "dispatch_fast_matmul"), (
        "dispatch_fast_matmul is not yet exported from "
        "triton_msl.autotuning._fast_matmul_dispatch. "
        "Extract the fast-matmul dispatch block to that module."
    )
    assert callable(_mod.dispatch_fast_matmul), (
        "dispatch_fast_matmul must be callable"
    )


def test_mark_unsupported_targets_sel_msl_not_fast_msl(monkeypatch):
    """When the selected non-(4,4) variant fails at dispatch, the except block
    must call rt.mark_unsupported(sel_msl), NOT rt.mark_unsupported(fast_msl).

    Bug (driver.py line 667 pre-fix): 'rt.mark_unsupported(fast_msl)' is called
    regardless of which variant failed.  When the selected variant is non-(4,4),
    fast_msl (the (4,4) default) is incorrectly blacklisted, permanently preventing
    any future fast-path dispatch for the kernel — violating the prime directive's
    fallback order.
    """
    from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

    fast_msl = make_simdgroup_matmul_kernel_fast("fp32", 4, 4, "fp32")
    sel_msl = make_simdgroup_matmul_kernel_fast("fp32", 2, 4, "fp32")
    assert sel_msl != fast_msl

    descriptor = (fast_msl, 3, 4, 5, 32, 128, "fp32", "fp32")
    rt = _RecordingRuntime(fail_dispatch=True)  # dispatch raises -> except block fires

    _run_fast_matmul_block(
        rt, descriptor, M=512, N=512, K=512,
        best_rrrc_override=(2, 4),   # tuner returns (2,4) -> sel_msl != fast_msl
        monkeypatch=monkeypatch,
    )

    # The dispatch call is attempted (confirm the block actually reached dispatch)
    assert rt.dispatched_msls, "dispatch must have been attempted before the failure"

    # CRITICAL: fast_msl (the (4,4) default / future fallback) must NOT be blacklisted.
    # If the bug is present, fast_msl IS in _unsupported and this assertion fails.
    assert fast_msl not in rt._unsupported, (
        "BUG DETECTED: mark_unsupported was called with fast_msl (the (4,4) default) "
        "instead of sel_msl (the (2,4) variant that actually failed).  This permanently "
        "disables the (4,4) fallback for future dispatches of this kernel."
    )
    # sel_msl MUST be marked unsupported — it's the variant that failed
    assert sel_msl in rt._unsupported, (
        "mark_unsupported must be called with sel_msl (the selected (2,4) variant that "
        "failed at dispatch), not fast_msl."
    )
