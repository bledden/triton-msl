# triton_msl/autotuning/_quant_matmul_dispatch.py
"""Quantized-matmul dispatch (mirror of _fast_matmul_dispatch).

Routes a recognized weight-only int8 quantized matmul to the dedicated
``make_int8_matmul_fast(layout='kn')`` simdgroup kernel via compile_shader.
The descriptor is built at compile time by
``_maybe_quant_matmul_descriptor`` (correct-or-refuse; only the CANONICAL
shape produces a descriptor). Kept in its own module so the dispatch is unit-
testable without triggering Triton backend discovery.

Signature:
    dispatch_quant_matmul(rt, descriptor, kargs, *, launch_exit_hook=None,
                          launch_metadata=None) -> bool

Returns True if it dispatched (caller returns immediately), False otherwise
(caller decides the fallback — for quantized the driver REFUSES rather than
run the mismatched compiled kernel, so a False here is a hard error upstream).
On any internal error returns False (no exception escapes).
"""

import math as _math

# The kernel bk (K-tile) is fixed at 32 in the descriptor's make_int8_matmul_fast.
_BK = 32


from triton_msl.autotuning._operand_overlap import output_overlaps_operand
from triton_msl.autotuning._submission import SubmissionState


def _cdiv(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _launch_grid_ok(grid, expected):
    """Packet 105 B: every quantized template computes the FULL output from M/N/K; it may
    stand in for the compiled kernel only when the caller launched EXACTLY the program grid
    the kernel maps (``cdiv(M, BM) x cdiv(N, BN)`` on program_id(0)/(1), or ``cdiv(N, BN)``
    for a GEMV; unused axes 1). A partial launch used to run the whole output (GPU: N = 64,
    BN = 32, ``grid=(1,)`` filled outputs 32..63 the kernel never wrote). No grid or no
    contract in the descriptor -> not provably equivalent -> False (the driver refuses)."""
    if grid is None or expected is None:
        return False
    try:
        g = tuple(int(x) for x in grid)
        e = tuple(int(x) for x in expected)
    except (TypeError, ValueError):
        return False
    return len(g) == 3 and len(e) == 3 and g == e


def _gemm_grid_ok(grid, bmbn, M, N):
    """2-D quantized GEMM: the exact program grid ``(cdiv(M, BM), cdiv(N, BN), 1)`` the
    kernel maps. (Tail tiles of a non-aligned extent: the unmasked kernel's own semantics
    there is an out-of-bounds write — undefined — and the template's clip at M/N is the only
    defined refinement; the program COUNT is still required to be the kernel's.)"""
    try:
        bm, bn = int(bmbn[0]), int(bmbn[1])
    except (TypeError, ValueError, IndexError):
        return False
    if bm <= 0 or bn <= 0:
        return False
    return _launch_grid_ok(grid, (_cdiv(M, bm), _cdiv(N, bn), 1))


def _gemv_grid_ok(grid, bn, N):
    try:
        bn = int(bn)
    except (TypeError, ValueError):
        return False
    if bn <= 0:
        return False
    return _launch_grid_ok(grid, (_cdiv(N, bn), 1, 1))


def dispatch_quant_matmul(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """Attempt to dispatch the fast quantized-matmul kernel.

    Parameters
    ----------
    rt : CompileShaderRuntime-like
        Exposes is_unsupported / mark_unsupported / get_library / dispatch.
    descriptor : tuple
        (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n, stride_checks).
    kargs : list
        Non-constexpr kernel args in buffer order. kargs[:8] =
        [input, weight, output, scale, zero, M, N, K]; strides follow.

    Returns
    -------
    bool : True if dispatched; False on a pre-invocation miss (misaligned shape,
           runtime stride mismatch, or library failure). Attempted-invocation
           errors propagate; fast-to-scalar fallback cannot replay caller work.
    """
    _submission = submission_state if submission_state is not None else SubmissionState()
    # Tagged descriptors: "gemv_int4" (per-group int4 decode) / "gemv" (int8 decode).
    # The GEMM's index 0 is its MSL string (starts with "#include"), never these tags.
    if isinstance(descriptor, (tuple, list)) and len(descriptor):
        if descriptor[0] == "gemv_int4":
            return _dispatch_int4_gemv(
                rt,
                descriptor,
                kargs,
                grid=grid,
                launch_exit_hook=launch_exit_hook,
                launch_metadata=launch_metadata,
                submission_state=_submission,
            )
        if descriptor[0] == "gemv":
            return _dispatch_gemv(
                rt,
                descriptor,
                kargs,
                grid=grid,
                launch_exit_hook=launch_exit_hook,
                launch_metadata=launch_metadata,
                submission_state=_submission,
            )
        if descriptor[0] in ("pergroup_int8", "pergroup_int4"):
            return _dispatch_pergroup_int8(
                rt,
                descriptor,
                kargs,
                grid=grid,
                launch_exit_hook=launch_exit_hook,
                launch_metadata=launch_metadata,
                submission_state=_submission,
            )
        if descriptor[0] == "sym_int8":
            return _dispatch_sym_int8(
                rt,
                descriptor,
                kargs,
                grid=grid,
                launch_exit_hook=launch_exit_hook,
                launch_metadata=launch_metadata,
                submission_state=_submission,
            )

    try:
        fast_msl = descriptor[0]
        m_idx, n_idx, k_idx = descriptor[1], descriptor[2], descriptor[3]
        tile_m, tile_n = descriptor[4], descriptor[5]
        stride_checks = descriptor[6] if len(descriptor) > 6 else ()
    except (TypeError, ValueError, IndexError):
        return False

    if fast_msl is None or rt.is_unsupported(fast_msl):
        return False

    try:
        M = int(kargs[m_idx])
        N = int(kargs[n_idx])
        K = int(kargs[k_idx])

        # RUNTIME STRIDE CONTRACT: the fast kernel assumes row-major (A row == K,
        # B row == N, C row == N). A stride arg that differs at runtime (a
        # transposed / column-sliced operand) would be SILENTLY WRONG -> skip.
        for arg_idx, expected_idx in stride_checks:
            actual = int(kargs[arg_idx])
            expected = 1 if expected_idx < 0 else int(kargs[expected_idx])
            if actual != expected:
                return False

        # SIZE CONTRACT: the kernel has NO edge handling and floor-divides N by the
        # column tile, so all three dims must divide exactly. A non-conforming shape
        # is skipped (the driver then refuses -> never silent-wrong).
        if not (M > 0 and N > 0 and K > 0 and M % tile_m == 0 and N % tile_n == 0 and K % _BK == 0):
            return False
        # LAUNCH CONTRACT (packet 105 B): descriptor[7] = the source (BM, BN).
        if not _gemm_grid_ok(grid, descriptor[7] if len(descriptor) > 7 else None, M, N):
            return False
        # OUTPUT ALIAS: see _operand_overlap. The template stores while other tiles load.
        if output_overlaps_operand(kargs, 2, M * N, {0: M * K}):
            return False

        n_groups = _math.ceil(M / tile_m) * _math.ceil(N / tile_n)
        lib = rt.get_library(fast_msl)
        # The kernel declares exactly 8 buffers (input,weight,output,scale,zero,M,N,K);
        # pass only those (kargs[:8]) — trailing stride args have no buffer slot.
        _submission.begin()
        rt.dispatch(lib, "int8_matmul_fast", kargs[:8], threads=n_groups * 32, group_size=32)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True

    except Exception as _error:
        _submission.reraise_if_attempted(_error)
        try:
            rt.mark_unsupported(fast_msl)
        except Exception as _error:
            _submission.reraise_if_attempted(_error)
            pass
        return False


def _dispatch_int4_gemv(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """Dispatch make_int4_gemv (weight-only int4 per-group decode GEMV).

    descriptor = ("gemv_int4", int4_msl, in, w, out, scale, zero, n_idx, k_idx,
                  swn_idx, ssn_idx, ng_idx, group). Buffer order (input, weight,
                  output, scales, zeros, K, N); N simdgroups; K % 4 == 0, K % group == 0.
    """
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        int4_msl = descriptor[1]
        in_idx, w_idx, out_idx, scale_idx, zero_idx = descriptor[2:7]
        n_idx, k_idx = descriptor[7], descriptor[8]
        swn_idx, ssn_idx, group = descriptor[9], descriptor[10], descriptor[12]
    except (TypeError, ValueError, IndexError):
        return False
    if int4_msl is None or rt.is_unsupported(int4_msl):
        return False
    try:
        N = int(kargs[n_idx])
        K = int(kargs[k_idx])
        # RUNTIME STRIDE CONTRACT: weight row stride == K/2 (packed 2 nibbles/byte),
        # scale/zero row stride == K/group (per-group). A mismatch would read the wrong
        # bytes → skip (the driver then refuses).
        if swn_idx is not None and int(kargs[swn_idx]) * 2 != K:
            return False
        if ssn_idx is not None and int(kargs[ssn_idx]) * group != K:
            return False
        if not (N > 0 and K > 0 and K % 4 == 0 and K % group == 0):
            return False
        # Packet 105 A: the template dequantizes both nibbles of a byte with ONE group ->
        # the group must be byte-aligned (even). B: exact launch grid (descriptor[13] = BN).
        if int(group) % 2 != 0:
            return False
        if not _gemv_grid_ok(grid, descriptor[13] if len(descriptor) > 13 else None, N):
            return False
        # OUTPUT ALIAS: see _operand_overlap. The template stores while other tiles load.
        if output_overlaps_operand(kargs, out_idx, N, {in_idx: K}):
            return False
        gsz = 256
        threads = ((N * 32 + gsz - 1) // gsz) * gsz
        buffers = [kargs[in_idx], kargs[w_idx], kargs[out_idx], kargs[scale_idx], kargs[zero_idx], K, N]
        lib = rt.get_library(int4_msl)
        _submission.begin()
        rt.dispatch(lib, "int4_gemv", buffers, threads=threads, group_size=gsz)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception as _error:
        _submission.reraise_if_attempted(_error)
        try:
            rt.mark_unsupported(int4_msl)
        except Exception as _error:
            _submission.reraise_if_attempted(_error)
            pass
        return False


def _dispatch_sym_int8(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """Dispatch a SYMMETRIC int8 GEMM (no zero-point) via make_int8_matmul_pergroup.

    descriptor = ("sym_int8", pg_msl, m_idx, n_idx, k_idx,
                  (isr, isc, wsk, wsn, osr, osc, ssn)).
    kargs = [input, weight, output, scale, M, N, K, ...strides] (4 ptrs, NO zeros).
    Synthesize an all-zero zeros buffer and set ssg=0/zsg=0/zsn=0 so the per-group
    template reads scale[col*ssn] and subtracts 0 -> w = weight*scale (per-N symmetric).
    """
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        pg_msl = descriptor[1]
        m_idx, n_idx, k_idx = descriptor[2], descriptor[3], descriptor[4]
        isr, isc, wsk, wsn, osr, osc, ssn_idx = descriptor[5]
    except (TypeError, ValueError, IndexError):
        return False
    if pg_msl is None or rt.is_unsupported(pg_msl):
        return False
    try:
        import torch as _torch

        M, N, K = int(kargs[m_idx]), int(kargs[n_idx]), int(kargs[k_idx])
        if not (M > 0 and N > 0 and K > 0):
            return False
        if not _gemm_grid_ok(grid, descriptor[6] if len(descriptor) > 6 else None, M, N):  # packet 105 B
            return False
        _s = lambda i: (int(kargs[i]) if i >= 0 else 1)
        scale = kargs[3]
        strides = tuple(_s(i) for i in (isr, isc, wsk, wsn, osr, osc, ssn_idx))
        isr_v, isc_v, wsk_v, wsn_v, osr_v, osc_v, ssn_v = strides

        # M/N POSITIONAL-SANITY + memory-safety gate, matching the asymmetric
        # per-group path but omitting its real zeros tensor. M/N are dead in the
        # recognized Triton body, so prove the positional interpretation at runtime
        # against every buffer extent before launching the hard-coded template ABI.
        if all(s >= 0 for s in strides):
            bounds = (
                ((M - 1) * isr_v + (K - 1) * isc_v, kargs[0]),
                ((K - 1) * wsk_v + (N - 1) * wsn_v, kargs[1]),
                ((M - 1) * osr_v + (N - 1) * osc_v, kargs[2]),
                ((N - 1) * ssn_v, scale),
            )
            for max_index, tensor in bounds:
                if hasattr(tensor, "numel") and max_index >= tensor.numel():
                    return False
            # OUTPUT ALIAS: see _operand_overlap. The template stores while other tiles
            # load. The furthest cell each pointer is addressed at is known here.
            last_in, last_w, last_out, last_scale = (max_index for max_index, _ in bounds)
            if output_overlaps_operand(
                kargs[:4], 2, last_out + 1, {0: last_in + 1, 1: last_w + 1, 3: last_scale + 1}, exact=True
            ):
                return False
        elif output_overlaps_operand(kargs, 2, 0, {}):
            return False

        zeros = _torch.zeros(1, dtype=getattr(scale, "dtype", None), device=getattr(scale, "device", None))
        # input(0), weight(1), output(2), scale(3), zeros(synth,4), M,N,K, strides...
        buffers = list(kargs[0:4]) + [zeros, M, N, K, *strides[:6], 0, ssn_v, 0, 0]  # ssg=0, ssn, zsg=0, zsn=0
        lib = rt.get_library(pg_msl)
        _grp = 256
        threads = _math.ceil((M * N) / _grp) * _grp
        _submission.begin()
        rt.dispatch(lib, "int8_matmul_pergroup", buffers, threads=threads, group_size=_grp)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception as _error:  # noqa: BLE001
        _submission.reraise_if_attempted(_error)
        return False


def _dispatch_pergroup_int8(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """Dispatch the scalar per-group int8 GEMM (make_int8_matmul_pergroup).

    descriptor = ("pergroup_int8"|"pergroup_int4", pg_msl, m_idx, n_idx, k_idx,
                  (isr, isc, wsk/wbk, wsn, osr, osc, ssg, ssn, zsg, zsn)).
    Buffer order = (input, weight, output, scales, zeros, M, N, K, + the 10 strides);
    one thread per output element (gid = row*N + col). The int4 tag routes to
    make_int4_matmul_pergroup (uchar weight, byte//2 index) — same buffer layout, and
    buffer 10 is the packed weight's byte-row stride.
    """
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        pg_msl = descriptor[1]
        m_idx, n_idx, k_idx = descriptor[2], descriptor[3], descriptor[4]
        sidx = descriptor[5]
        _kname = "int4_matmul_pergroup" if descriptor[0] == "pergroup_int4" else "int8_matmul_pergroup"
    except (TypeError, ValueError, IndexError):
        return False
    if pg_msl is None or rt.is_unsupported(pg_msl):
        return False
    try:
        M, N, K = int(kargs[m_idx]), int(kargs[n_idx]), int(kargs[k_idx])
        if not (M > 0 and N > 0 and K > 0):
            return False
        # LAUNCH CONTRACT (packet 105 B) — checked BEFORE the fast sub-path selection below.
        if not _gemm_grid_ok(grid, descriptor[11] if len(descriptor) > 11 else None, M, N):
            return False
        # a stride index of -1 means the stride was folded to a compile-time 1.
        strides = [(int(kargs[i]) if i >= 0 else 1) for i in sidx]
        isr, isc, wsk, wsn, osr, osc, ssg, ssn = strides[0:8]
        zsg, zsn = strides[8], strides[9]

        # M/N POSITIONAL-SANITY + memory-safety gate (re-review 2026-08-25): the descriptor
        # hard-codes m_idx/n_idx/k_idx = 5/6/7, but in the canonical maskless 2-D-grid
        # kernel M and N are DEAD args — a kernel declaring them as (N, M, K) routed and the
        # template then wrote/read OUT OF BOUNDS (partial output + NaN from OOB scale
        # reads). There is no structural anchor for dead args, so anchor at RUNTIME: the
        # furthest element each tensor is indexed at must lie inside that tensor. This makes
        # the launch memory-safe unconditionally and refuses (fail-closed driver raise)
        # every non-square M/N swap under standard layouts. Skipped only for negative
        # strides (formula invalid; the scalar template doesn't support those anyway).
        _g_s = descriptor[10] if len(descriptor) > 10 else None
        if _g_s and all(_s >= 0 for _s in strides):
            _ng = (K + _g_s - 1) // _g_s
            _wrow = ((K - 1) // 2) * wsk if descriptor[0] == "pergroup_int4" else (K - 1) * wsk
            _bounds = (
                ((M - 1) * isr + (K - 1) * isc, kargs[0]),
                (_wrow + (N - 1) * wsn, kargs[1]),
                ((M - 1) * osr + (N - 1) * osc, kargs[2]),
                ((_ng - 1) * ssg + (N - 1) * ssn, kargs[3]),
                ((_ng - 1) * zsg + (N - 1) * zsn, kargs[4]),
            )
            for _mx, _t in _bounds:
                if hasattr(_t, "numel") and _mx >= _t.numel():
                    return False
            # OUTPUT ALIAS: see _operand_overlap. The template stores while other tiles
            # load. The furthest cell each pointer is addressed at is known here.
            _reads = {_i: _mx + 1 for _i, (_mx, _t) in enumerate(_bounds) if _i != 2}
            if output_overlaps_operand(kargs[:5], 2, _bounds[2][0] + 1, _reads, exact=True):
                return False
        elif output_overlaps_operand(kargs, 2, 0, {}):
            return False

        # FAST simdgroup-MMA per-group path (descriptor[6..10] = fast_msl, rr, rc, bk,
        # group_size), selected ONLY when the runtime shape meets its contract: contiguous
        # row-major input [M,K] / weight [K,N] kn (for int4, packed [K/2,N] with byte-row
        # stride wsk==N) / output [M,N], aligned M/N/K/group, and ZEROS laid out like the
        # SCALES (the fast MSL indexes zeros with ssg/ssn — re-review 2026-08-25: a zeros
        # tensor with its own layout, e.g. GPTQ qzeros transposed vs scales, was silently
        # mis-indexed, err ~3.6-6.5). Otherwise fall through to the stride-generic scalar
        # kernel below (correct for any layout — it consumes zsg/zsn separately).
        _fast = descriptor[6] if len(descriptor) > 6 else None
        if _fast is not None and descriptor[0] in ("pergroup_int8", "pergroup_int4") and not rt.is_unsupported(_fast):
            try:
                rr, rc, bk, g_s = descriptor[7], descriptor[8], descriptor[9], descriptor[10]
                tm, tn = 8 * rr, 8 * rc
                if (
                    M % tm == 0
                    and N % tn == 0
                    and K % bk == 0
                    and g_s % bk == 0
                    and isc == 1
                    and wsn == 1
                    and osc == 1
                    and isr == K
                    and wsk == N
                    and osr == N
                    and zsg == ssg
                    and zsn == ssn
                ):
                    _fkname = (
                        "int4_matmul_pergroup_fast" if descriptor[0] == "pergroup_int4" else "int8_matmul_pergroup_fast"
                    )
                    flib = rt.get_library(_fast)
                    fbuf = list(kargs[0:5]) + [M, N, K, ssg, ssn]
                    fthreads = (M // tm) * (N // tn) * 32
                    _submission.begin()
                    rt.dispatch(flib, _fkname, fbuf, threads=fthreads, group_size=32)
                    if launch_exit_hook:
                        launch_exit_hook(launch_metadata)
                    return True
            except Exception as _error:  # noqa: BLE001
                _submission.reraise_if_attempted(_error)
                # A fast-path failure (e.g. compile error) must not kill the launch: mark
                # the fast MSL unsupported (no per-launch retry) and fall through to the
                # scalar per-group kernel in the same descriptor (correct for any layout).
                try:
                    rt.mark_unsupported(_fast)
                except Exception as _error:  # noqa: BLE001
                    _submission.reraise_if_attempted(_error)
                    pass

        buffers = list(kargs[0:5]) + [M, N, K] + strides
        lib = rt.get_library(pg_msl)
        _grp = 256
        threads = _math.ceil((M * N) / _grp) * _grp
        _submission.begin()
        rt.dispatch(lib, _kname, buffers, threads=threads, group_size=_grp)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception as _error:  # noqa: BLE001
        _submission.reraise_if_attempted(_error)
        return False


def _dispatch_gemv(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """Dispatch the dedicated make_int8_gemv kernel (weight-only int8 decode GEMV).

    descriptor = ("gemv", gemv_msl, in_idx, w_idx, out_idx, scale_idx, zero_idx,
                  n_idx, k_idx, stride_checks).
    make_int8_gemv buffer order is (input, weight, output, scales, zeros, K, N) with
    one simdgroup per output column (N*32 threads); K % 4 == 0.
    """
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        gemv_msl = descriptor[1]
        in_idx, w_idx, out_idx, scale_idx, zero_idx = descriptor[2:7]
        n_idx, k_idx = descriptor[7], descriptor[8]
        stride_checks = descriptor[9] if len(descriptor) > 9 else ()
    except (TypeError, ValueError, IndexError):
        return False

    if gemv_msl is None or rt.is_unsupported(gemv_msl):
        return False

    try:
        N = int(kargs[n_idx])
        K = int(kargs[k_idx])

        # RUNTIME STRIDE CONTRACT: make_int8_gemv reads weight[n*K + k], so the weight
        # row stride must equal K. A different stride (transposed / sliced) -> skip.
        for arg_idx, expected_idx in stride_checks:
            actual = int(kargs[arg_idx])
            expected = 1 if expected_idx < 0 else int(kargs[expected_idx])
            if actual != expected:
                return False

        # SIZE CONTRACT: the kernel strides K in char4/float4, so K % 4 == 0.
        if not (N > 0 and K > 0 and K % 4 == 0):
            return False
        # LAUNCH CONTRACT (packet 105 B): descriptor[10] = the source BN.
        if not _gemv_grid_ok(grid, descriptor[10] if len(descriptor) > 10 else None, N):
            return False
        # OUTPUT ALIAS: see _operand_overlap. The template stores while other tiles load.
        if output_overlaps_operand(kargs, out_idx, N, {in_idx: K}):
            return False

        group = 256
        threads = ((N * 32 + group - 1) // group) * group  # pad up; kernel guards n >= N
        # Buffer order the kernel declares: (input, weight, output, scales, zeros, K, N).
        buffers = [
            kargs[in_idx],
            kargs[w_idx],
            kargs[out_idx],
            kargs[scale_idx],
            kargs[zero_idx],
            K,
            N,
        ]
        lib = rt.get_library(gemv_msl)
        _submission.begin()
        rt.dispatch(lib, "int8_gemv", buffers, threads=threads, group_size=group)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True

    except Exception as _error:
        _submission.reraise_if_attempted(_error)
        try:
            rt.mark_unsupported(gemv_msl)
        except Exception as _error:
            _submission.reraise_if_attempted(_error)
            pass
        return False
