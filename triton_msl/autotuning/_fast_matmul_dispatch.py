# triton_msl/autotuning/_fast_matmul_dispatch.py
"""Extracted fast-matmul dispatch logic.

This module exists so the dispatch block is testable without triggering the
full Triton backend discovery that importing driver.py causes.  The driver
imports and calls dispatch_fast_matmul; tests import it directly.

Signature:
    dispatch_fast_matmul(rt, descriptor, kargs, *, launch_exit_hook=None,
                         launch_metadata=None) -> bool

    Returns True if the fast path dispatched successfully, False otherwise.
    On any error the function returns False (no exception escapes); the caller
    falls through to the generic Metal/CPU path.
"""
import math as _math

# Variant MSL strings cached by (msl_dtype, msl_out, rr, rc) — avoids rebuilding
# the ~5KB kernel source on every dispatch (per-call overhead that swamped the GPU
# saving on small matmuls).
_VARIANT_MSL_CACHE = {}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dispatch_fast_matmul(rt, descriptor, kargs, *, launch_exit_hook=None,
                         launch_metadata=None):
    """Attempt to dispatch via the simdgroup fast-matmul template.

    Parameters
    ----------
    rt : CompileShaderRuntime-like
        Must expose: is_unsupported(msl), mark_unsupported(msl),
        get_library(msl), dispatch(lib, name, args, *, threads, group_size).
    descriptor : tuple
        8-element: (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n, msl_dtype, msl_out)
        OR 6-element legacy: (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n)
        Absent trailing fields -> fixed (4,4) path only (no tile selection).
    kargs : list
        Non-constexpr kernel args in buffer order (kargs[:6] = [A,B,C,M,N,K]).
    launch_exit_hook, launch_metadata : optional
        Forwarded to the exit hook after a successful dispatch (same as driver).

    Returns
    -------
    bool
        True  — fast path dispatched; caller should return immediately.
        False — fast path skipped or failed; caller falls through to generic path.
    """
    # Unpack descriptor defensively (6/8/9-element; older descriptors lack dtype
    # / stride-check fields).
    try:
        fast_msl = descriptor[0]
        m_idx, n_idx, k_idx = descriptor[1], descriptor[2], descriptor[3]
        tile_m, tile_n = descriptor[4], descriptor[5]
        msl_dtype = descriptor[6] if len(descriptor) > 6 else None
        msl_out   = descriptor[7] if len(descriptor) > 7 else None
        stride_checks = descriptor[8] if len(descriptor) > 8 else ()
    except (TypeError, ValueError, IndexError):
        return False  # malformed descriptor -> skip

    if fast_msl is None or rt.is_unsupported(fast_msl):
        return False

    # Initialise sel_msl BEFORE the try so the except block always has it in scope.
    sel_msl = fast_msl
    try:
        M = int(kargs[m_idx])
        N = int(kargs[n_idx])
        K = int(kargs[k_idx])

        # RUNTIME STRIDE CONTRACT: the fast template assumes a ROW-MAJOR layout
        # (leading dims = M/N/K, inner stride 1). When the kernel passes explicit
        # stride args, verify each runtime row stride equals the dim it must be —
        # otherwise a transposed/column-sliced operand would be SILENTLY WRONG.
        # Mismatch -> skip the fast path; the generic stride-aware kernel handles
        # it. (expected_idx == -1 means "must equal literal 1".)
        for arg_idx, expected_idx in stride_checks:
            try:
                actual = int(kargs[arg_idx])
                expected = 1 if expected_idx < 0 else int(kargs[expected_idx])
            except (TypeError, ValueError, IndexError):
                return False
            if actual != expected:
                return False

        # --- Per-shape deterministic tile selection (safe: every CANDIDATES config
        #     computes a correct matmul; selection only affects perf).  Any error in this
        #     inner block falls back to the fixed (4,4) tile dims and fast_msl. ---
        sel_tm, sel_tn = tile_m, tile_n
        if msl_dtype is not None and msl_out is not None:
            try:
                from triton_msl.autotuning.matmul_tuner import best_rrrc
                from triton_msl.codegen._msl_templates import (
                    make_simdgroup_matmul_kernel_fast)
                rrrc = best_rrrc(msl_dtype, msl_out, M, N, K, rt)
                if rrrc is not None and rrrc != (4, 4):
                    rr, rc = rrrc
                    # Cache the variant MSL by config: rebuilding the ~5KB string
                    # every dispatch added per-call Python overhead that swamped the
                    # GPU saving on sub-ms matmuls (net SLOWER at 512^3). Built once
                    # per (dtype,out,rr,rc), then a dict hit.
                    vkey = (msl_dtype, msl_out, rr, rc)
                    sel_msl = _VARIANT_MSL_CACHE.get(vkey)
                    if sel_msl is None:
                        sel_msl = make_simdgroup_matmul_kernel_fast(msl_dtype, rr, rc, msl_out)
                        _VARIANT_MSL_CACHE[vkey] = sel_msl
                    sel_tm, sel_tn = 8 * rr, 32 * rc
            except Exception:
                # Autotuning failed -> fall back to the baked (4,4) variant.
                sel_msl = fast_msl
                sel_tm, sel_tn = tile_m, tile_n

        # --- Guard: skip if sel_msl was already blacklisted (e.g. a previous
        #     dispatch of this very variant failed).  Without this check the
        #     block would re-attempt dispatch on every call after the first
        #     failure, wasting get_library + dispatch round-trips. ---
        if rt.is_unsupported(sel_msl):
            return False

        # --- Size-contract gate. The fast template guards the boundary COLUMN tile per
        #     simdgroup: each of the 4 simdgroups owns a strip of width 8*rc (= sel_tn/4)
        #     and `if (col0 >= N) return;` skips strips entirely beyond N. So the real
        #     requirement is N % (8*rc) == 0 (the per-simdgroup strip width) — NOT
        #     N % (32*rc) (the full threadgroup tile). The grid uses ceil(N/sel_tn) and the
        #     per-strip guard masks the overshoot, so any N that is a multiple of the strip
        #     width is correct. The old N%sel_tn gate was 4x too strict and dropped e.g.
        #     N=2080 (%32, not %128) to the ~3 TF generic path despite the fast template
        #     handling it (verified byte-exact vs torch). M still needs %sel_tm (no row
        #     guard). [The descriptor (_maybe_fast_matmul_descriptor) verifies the kernel's
        #     M/N/K binding via the K-loop bound, so a non-canonical signature can't reach
        #     here with mis-bound dims.]
        sel_strip = sel_tn // 4
        if not (M > 0 and N > 0 and K > 0
                and M % sel_tm == 0 and N % sel_strip == 0 and K % 8 == 0):
            return False

        n_groups = _math.ceil(M / sel_tm) * _math.ceil(N / sel_tn)
        lib = rt.get_library(sel_msl)
        # The fast template declares exactly 6 buffers (A,B,C,M,N,K = kargs[:6]);
        # pass only those so we don't rely on compile_shader silently ignoring
        # trailing stride args.
        rt.dispatch(lib, "simdgroup_matmul_fast", kargs[:6],
                    threads=n_groups * 128, group_size=128)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True

    except Exception:
        # Fast path failed -> mark the SELECTED variant (sel_msl) unsupported,
        # NOT fast_msl (the (4,4) default).  Marking fast_msl would permanently
        # disable the (4,4) fallback for any shape, not just the failing variant.
        try:
            rt.mark_unsupported(sel_msl)
        except Exception:
            pass
        return False
