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
    # Unpack descriptor defensively (6- or 8-element; older descriptors lack dtype fields)
    try:
        fast_msl = descriptor[0]
        m_idx, n_idx, k_idx = descriptor[1], descriptor[2], descriptor[3]
        tile_m, tile_n = descriptor[4], descriptor[5]
        msl_dtype = descriptor[6] if len(descriptor) > 6 else None
        msl_out   = descriptor[7] if len(descriptor) > 7 else None
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

        # --- Size-contract gate. valid_candidates guarantees M%sel_tm==0 and
        #     N%sel_tn==0 for the chosen config; we re-check here for defence-in-
        #     depth (e.g. a caller bypassing best_rrrc, or a future candidate-set
        #     change) so a config can never be dispatched for a shape it mis-fits.
        #     NOTE: N%32 alone is NOT sufficient — tile_n = 32*rc can be 64..256;
        #     an N that is a multiple of 32 but not of tile_n produces OOB writes.
        if not (M > 0 and N > 0 and K > 0
                and M % sel_tm == 0 and N % sel_tn == 0 and K % 8 == 0):
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
