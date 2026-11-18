"""Packet 103/104: the quantized templates re-emit a FIXED address contract per role; the
descriptors must prove the kernel's addresses are EXACTLY that contract — required terms
present with the right axis, coefficient and range bounds, the K-loop's bound and step, the
output store's address, mask and base — not merely that no foreign term appears.

On clean ``ba21e9d`` (and on the P0 tree ``e2330ec``) every mode below ran the specialized route
and stored the template's canonical result while the kernel's semantics differed (packet 103
§2 for the int8 GEMV; the per-group / symmetric GEMM and int4 GEMV siblings measured in 104):

  30 no program_id on the tile index     31 pid coefficient 2*BLOCK      32 input K index without the loop term
  33 weight K index without the loop term 34 output base +1              35 output index +1
  36 output store masked to half a tile   37 make_range(BN, 2BN)          38 make_range(BK, 2BK)
  39 loop step 2*BK                        40 loop bound K/2               41 duplicate pid term
  42 output pid axes swapped (GEMM)        43 scale index scaled by 2      44 duplicate induction-variable term (GEMV x)

The oracle for each mode is a torch SIMULATION of the kernel's own semantics (flat addressing
into the same contiguous slack buffers, per program and per loop iteration), so a witness
asserts "refuse, or match what the kernel means" — never the template's raw result. The
compile-time pins go through ``emit_msl`` with launch-specialized signatures.

2026-11-18 (reduce of a row reduction, the held-layout record): the int8 GEMV violations no
longer refuse at the lowering boundary. Their descriptor still declines every one of them (no
``quant_matmul``), and the generic path now lowers the kernel op by op: ``acc * scale`` is a
loop-carried row reduction times a load through the row index, and a load through an index is
now known to be held the way the index is, so the 1-D store is proved instead of refused. The
fourteen ``gemv-int8`` refusal pins became ``test_lowering_boundary_int8_gemv_declines`` plus
``test_int8_gemv_address_contract_computes_exactly``, which compares the device result with
the kernel's own semantics EXACTLY (integer data), at sizes that are and are not multiples
of the tile. The GEMM routes still refuse.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.msl_emitter import emit_msl
    from triton_msl.errors import MetalNonRecoverableError
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"
SENTINEL = 98765.0

if HAS:

    @triton.jit
    def _pg(
        a_ptr,
        w_ptr,
        c_ptr,
        scale_ptr,
        zero_ptr,
        M,
        N,
        K,
        sam,
        sak,
        swk,
        swn,
        ssg,
        ssn,
        zsg,
        zsn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        G: tl.constexpr,
        MODE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        if MODE == 30:
            om = rm
            on = rn
        elif MODE == 31:
            om = pid_m * (2 * BM) + rm
            on = pid_n * BN + rn
        elif MODE == 37:
            om = pid_m * BM + rm
            on = pid_n * BN + tl.arange(BN, 2 * BN)
        elif MODE == 41:
            om = pid_m * BM + pid_m * BM + rm
            on = pid_n * BN + rn
        else:
            om = pid_m * BM + rm
            on = pid_n * BN + rn
        if MODE == 38:
            ok = tl.arange(BK, 2 * BK)
        else:
            ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if MODE == 39:
            STEP: tl.constexpr = 2 * BK
        else:
            STEP: tl.constexpr = BK
        if MODE == 40:
            KB = K // 2
        else:
            KB = K
        for k in range(0, KB, STEP):
            g = k // G
            if MODE == 43:
                s = tl.load(scale_ptr + g * ssg + (on * 2) * ssn)
            else:
                s = tl.load(scale_ptr + g * ssg + on * ssn)
            z = tl.load(zero_ptr + g * zsg + on * zsn)
            if MODE == 33:
                w = (tl.load(w_ptr + ok[:, None] * swk + on[None, :] * swn).to(tl.float32) - z[None, :]) * s[None, :]
            else:
                w = (tl.load(wp).to(tl.float32) - z[None, :]) * s[None, :]
            if MODE == 32:
                a = tl.load(a_ptr + om[:, None] * sam + ok[None, :] * sak)
            else:
                a = tl.load(ap)
            acc += tl.dot(a, w)
            ap += BK * sak
            wp += BK * swk
        if MODE == 34:
            tl.store(c_ptr + 1 + om[:, None] * scm + on[None, :] * scn, acc)
        elif MODE == 35:
            tl.store(c_ptr + om[:, None] * scm + (on[None, :] + 1) * scn, acc)
        elif MODE == 36:
            tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc, mask=(rn < (BN // 2))[None, :])
        elif MODE == 42:
            tl.store(c_ptr + (pid_n * BM + rm)[:, None] * scm + (pid_m * BN + rn)[None, :] * scn, acc)
        else:
            tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _sym(
        a_ptr,
        w_ptr,
        c_ptr,
        s_ptr,
        M,
        N,
        K,
        sam,
        sak,
        swk,
        swn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        MODE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        if MODE == 30:
            om = rm
            on = rn
        elif MODE == 31:
            om = pid_m * (2 * BM) + rm
            on = pid_n * BN + rn
        elif MODE == 37:
            om = pid_m * BM + rm
            on = pid_n * BN + tl.arange(BN, 2 * BN)
        elif MODE == 41:
            om = pid_m * BM + pid_m * BM + rm
            on = pid_n * BN + rn
        else:
            om = pid_m * BM + rm
            on = pid_n * BN + rn
        if MODE == 38:
            ok = tl.arange(BK, 2 * BK)
        else:
            ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        if MODE == 43:
            scale = tl.load(s_ptr + on * 2)
        else:
            scale = tl.load(s_ptr + on)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if MODE == 39:
            STEP: tl.constexpr = 2 * BK
        else:
            STEP: tl.constexpr = BK
        if MODE == 40:
            KB = K // 2
        else:
            KB = K
        for _ in range(0, KB, STEP):
            if MODE == 33:
                w = tl.load(w_ptr + ok[:, None] * swk + on[None, :] * swn).to(tl.float32) * scale[None, :]
            else:
                w = tl.load(wp).to(tl.float32) * scale[None, :]
            if MODE == 32:
                a = tl.load(a_ptr + om[:, None] * sam + ok[None, :] * sak)
            else:
                a = tl.load(ap)
            acc += tl.dot(a, w)
            ap += BK * sak
            wp += BK * swk
        if MODE == 34:
            tl.store(c_ptr + 1 + om[:, None] * scm + on[None, :] * scn, acc)
        elif MODE == 35:
            tl.store(c_ptr + om[:, None] * scm + (on[None, :] + 1) * scn, acc)
        elif MODE == 36:
            tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc, mask=(rn < (BN // 2))[None, :])
        elif MODE == 42:
            tl.store(c_ptr + (pid_n * BM + rm)[:, None] * scm + (pid_m * BN + rn)[None, :] * scn, acc)
        else:
            tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _g8(
        x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr
    ):
        pid = tl.program_id(0)
        rn = tl.arange(0, BN)
        on_out = pid * BN + rn
        if MODE == 30:
            on = rn
        elif MODE == 31:
            on = pid * (2 * BN) + rn
        elif MODE == 37:
            on = pid * BN + tl.arange(BN, 2 * BN)
        elif MODE == 41:
            on = pid * BN + pid * BN + rn
        else:
            on = on_out
        if MODE == 43:
            scale = tl.load(scale_ptr + on * 2)
        else:
            scale = tl.load(scale_ptr + on)
        zero = tl.load(zero_ptr + on)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        if MODE == 39:
            STEP: tl.constexpr = 2 * BK
        else:
            STEP: tl.constexpr = BK
        if MODE == 40:
            KB = K // 2
        else:
            KB = K
        for k in range(0, KB, STEP):
            if MODE == 32:
                x = tl.load(x_ptr + rk)
            elif MODE == 38:
                x = tl.load(x_ptr + tl.arange(BK, 2 * BK) + k)
            elif MODE == 44:
                x = tl.load(x_ptr + rk + k + k)
            else:
                x = tl.load(x_ptr + rk + k)
            if MODE == 33:
                w = tl.load(w_ptr + on[:, None] * swn + rk[None, :] * swk).to(tl.float32)
            else:
                w = tl.load(w_ptr + on[:, None] * swn + (rk[None, :] + k) * swk).to(tl.float32)
            acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
        if MODE == 34:
            tl.store(o_ptr + 1 + on_out, acc * scale)
        elif MODE == 35:
            tl.store(o_ptr + (on_out + 1), acc * scale)
        elif MODE == 36:
            tl.store(o_ptr + on_out, acc * scale, mask=rn < (BN // 2))
        else:
            tl.store(o_ptr + on_out, acc * scale)

    @triton.jit
    def _g4(
        x_ptr,
        w_ptr,
        o_ptr,
        s_ptr,
        z_ptr,
        N,
        K,
        ng,
        swn,
        ssn,
        BN: tl.constexpr,
        BK: tl.constexpr,
        G: tl.constexpr,
        MODE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rn = tl.arange(0, BN)
        on_out = pid * BN + rn
        if MODE == 30:
            on = rn
        elif MODE == 31:
            on = pid * (2 * BN) + rn
        elif MODE == 37:
            on = pid * BN + tl.arange(BN, 2 * BN)
        elif MODE == 41:
            on = pid * BN + pid * BN + rn
        else:
            on = on_out
        if MODE == 38:
            ok = tl.arange(BK, 2 * BK)
        else:
            ok = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        if MODE == 39:
            STEP: tl.constexpr = 2 * BK
        else:
            STEP: tl.constexpr = BK
        if MODE == 40:
            KB = K // 2
        else:
            KB = K
        for k in range(0, KB, STEP):
            kk = k + ok
            if MODE == 33:
                kw = ok
            else:
                kw = kk
            packed = tl.load(w_ptr + on[:, None] * swn + (kw // 2)[None, :])
            w4 = (packed >> ((kw % 2) * 4)[None, :]) & 0xF
            g = kk // G
            if MODE == 43:
                s = tl.load(s_ptr + (on * 2)[:, None] * ssn + g[None, :])
            else:
                s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
            z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
            if MODE == 32:
                x = tl.load(x_ptr + ok)
            elif MODE == 44:
                x = tl.load(x_ptr + kk + k)
            else:
                x = tl.load(x_ptr + kk)
            acc += tl.sum(x[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
        if MODE == 34:
            tl.store(o_ptr + 1 + on_out, acc)
        elif MODE == 35:
            tl.store(o_ptr + (on_out + 1), acc)
        elif MODE == 36:
            tl.store(o_ptr + on_out, acc, mask=rn < (BN // 2))
        else:
            tl.store(o_ptr + on_out, acc)

    def _emit_for(fn, signature, constexprs):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({"num_warps": 4})
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        mod = src.make_ir(
            target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
        )
        metadata = {}
        mod = backend.make_ttir(mod, metadata, options)
        mod = backend.make_ttgir(mod, metadata, options)
        out = {}
        return emit_msl(mod, out, options), out


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (_pg, _sym, _g8, _g4):
        if hasattr(fn, "device_caches"):
            fn.device_caches.clear()


@pytest.fixture
def quant_dispatch_spy(monkeypatch):
    hits = []
    orig = quant_dispatch.dispatch_quant_matmul

    def spy(*a, **k):
        r = orig(*a, **k)
        hits.append(bool(r))
        return r

    monkeypatch.setattr(quant_dispatch, "dispatch_quant_matmul", spy)
    return hits


# ---------------------------------------------------------------------------- simulation
# Each simulator mirrors its kernel per program and per loop iteration with FLAT addressing
# into the contiguous buffers (a shifted address that spills into the next row reads and
# writes exactly what the kernel would), and stores into a NaN-initialised output.


def _cmp(a, b):
    return (torch.nan_to_num(a, nan=SENTINEL) - torch.nan_to_num(b, nan=SENTINEL)).abs().max().item()


def _sim_gemm(mode, a, w, c_shape, M, N, K, BM, BN, BK, G, s=None, z=None, sc=None):
    """Per-group (s, z) or symmetric (sc) GEMM kernel semantics for ``mode``; returns C."""
    dev = a.device
    af, wf = a.reshape(-1).float(), w.reshape(-1).float()
    sam, sak, swk, swn = a.stride(0), a.stride(1), w.stride(0), w.stride(1)
    c = torch.full(c_shape, float("nan"), device=dev)
    cf = c.reshape(-1)
    scm, scn = c.stride(0), c.stride(1)
    rm, rn = torch.arange(BM, device=dev), torch.arange(BN, device=dev)
    step = 2 * BK if mode == 39 else BK
    kb = K // 2 if mode == 40 else K
    ok = torch.arange(BK, 2 * BK, device=dev) if mode == 38 else torch.arange(BK, device=dev)
    for pm in range(M // BM):
        for pn in range(N // BN):
            if mode == 30:
                om, on = rm, rn
            elif mode == 31:
                om, on = pm * (2 * BM) + rm, pn * BN + rn
            elif mode == 37:
                om, on = pm * BM + rm, pn * BN + torch.arange(BN, 2 * BN, device=dev)
            elif mode == 41:
                om, on = pm * BM + pm * BM + rm, pn * BN + rn
            else:
                om, on = pm * BM + rm, pn * BN + rn
            acc = torch.zeros(BM, BN, device=dev)
            for i, k in enumerate(range(0, kb, step)):
                a_cols = ok if mode == 32 else ok + i * BK
                w_rows = ok if mode == 33 else ok + i * BK
                at = af[(om[:, None] * sam + a_cols[None, :] * sak)]
                wt = wf[(w_rows[:, None] * swk + on[None, :] * swn)]
                if sc is not None:
                    scale = sc[on * 2] if mode == 43 else sc[on]
                    wd = wt * scale[None, :]
                else:
                    g = k // G
                    ssg, ssn, zsg, zsn = s.stride(0), s.stride(1), z.stride(0), z.stride(1)
                    scale = s.reshape(-1)[g * ssg + (on * 2 if mode == 43 else on) * ssn]
                    zero = z.reshape(-1)[g * zsg + on * zsn]
                    wd = (wt - zero[None, :]) * scale[None, :]
                acc = acc + at @ wd
            if mode == 34:
                idx = 1 + om[:, None] * scm + on[None, :] * scn
            elif mode == 35:
                idx = om[:, None] * scm + (on[None, :] + 1) * scn
            elif mode == 42:
                idx = (pn * BM + rm)[:, None] * scm + (pm * BN + rn)[None, :] * scn
            else:
                idx = om[:, None] * scm + on[None, :] * scn
            vals = acc
            if mode == 36:
                keep = (rn < BN // 2)[None, :].expand(BM, BN)
                idx, vals = idx[keep], acc[keep]
            cf[idx.reshape(-1)] = vals.reshape(-1)
    return c


def _sim_gemv8(mode, x, w, s, z, N, K, BN, BK, progs=None):
    """``progs`` is the launched grid when N is not a multiple of BN (the kernel has no masks,
    so its last program stores a whole tile); the default is the N // BN of the witnesses."""
    dev = x.device
    wf = w.reshape(-1).float()
    swn, swk = w.stride(0), w.stride(1)
    progs = N // BN if progs is None else progs
    o = torch.full((max(N, progs * BN) + 8,), float("nan"), device=dev)
    rn, rk = torch.arange(BN, device=dev), torch.arange(BK, device=dev)
    step = 2 * BK if mode == 39 else BK
    kb = K // 2 if mode == 40 else K
    for pid in range(progs):
        on_out = pid * BN + rn
        if mode == 30:
            on = rn
        elif mode == 31:
            on = pid * (2 * BN) + rn
        elif mode == 37:
            on = pid * BN + torch.arange(BN, 2 * BN, device=dev)
        elif mode == 41:
            on = pid * BN + pid * BN + rn
        else:
            on = on_out
        scale = s[on * 2] if mode == 43 else s[on]
        zero = z[on]
        acc = torch.zeros(BN, device=dev)
        for k in range(0, kb, step):
            if mode == 32:
                xk = x[rk]
            elif mode == 38:
                xk = x[torch.arange(BK, 2 * BK, device=dev) + k]
            elif mode == 44:
                xk = x[rk + k + k]
            else:
                xk = x[rk + k]
            wcols = rk if mode == 33 else rk + k
            wt = wf[on[:, None] * swn + wcols[None, :] * swk]
            acc = acc + (xk[None, :] * (wt - zero[:, None])).sum(1)
        val = acc * scale
        if mode == 34:
            o[1 + on_out] = val
        elif mode == 35:
            o[on_out + 1] = val
        elif mode == 36:
            keep = rn < BN // 2
            o[on_out[keep]] = val[keep]
        else:
            o[on_out] = val
    return o


def _sim_gemv4(mode, x, packed, s, z, N, K, ng, BN, BK, G):
    dev = x.device
    pf = packed.reshape(-1).to(torch.int32)
    swn, ssn = packed.stride(0), s.stride(0)
    sf, zf = s.reshape(-1), z.reshape(-1)
    o = torch.full((N + 8,), float("nan"), device=dev)
    rn = torch.arange(BN, device=dev)
    ok = torch.arange(BK, 2 * BK, device=dev) if mode == 38 else torch.arange(BK, device=dev)
    step = 2 * BK if mode == 39 else BK
    kb = K // 2 if mode == 40 else K
    for pid in range(N // BN):
        on_out = pid * BN + rn
        if mode == 30:
            on = rn
        elif mode == 31:
            on = pid * (2 * BN) + rn
        elif mode == 37:
            on = pid * BN + torch.arange(BN, 2 * BN, device=dev)
        elif mode == 41:
            on = pid * BN + pid * BN + rn
        else:
            on = on_out
        acc = torch.zeros(BN, device=dev)
        for k in range(0, kb, step):
            kk = k + ok
            kw = ok if mode == 33 else kk
            byte = pf[on[:, None] * swn + (kw // 2)[None, :]]
            w4 = ((byte >> ((kw % 2) * 4)[None, :]) & 0xF).float()
            g = kk // G
            srow = on * 2 if mode == 43 else on
            sc = sf[srow[:, None] * ssn + g[None, :]]
            ze = zf[on[:, None] * ssn + g[None, :]]
            xk = x[ok] if mode == 32 else (x[kk + k] if mode == 44 else x[kk])
            acc = acc + (xk[None, :] * ((w4 - ze) * sc)).sum(1)
        if mode == 34:
            o[1 + on_out] = acc
        elif mode == 35:
            o[on_out + 1] = acc
        elif mode == 36:
            keep = rn < BN // 2
            o[on_out[keep]] = acc[keep]
        else:
            o[on_out] = acc
    return o


# ---------------------------------------------------------------------------- data + cases
_GEMM_MODES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
_SYM_MODES = list(_GEMM_MODES)
_GEMV_MODES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44]  # 44 duplicate induction-variable term on x


def _pg_case(mode):
    M, N, K, G, BM, BN, BK = 64, 32, 128, 32, 32, 32, 32
    torch.manual_seed(10301)
    a = torch.randn(4 * M, K, device=D)  # extra ROWS only: contiguous, canonical strides
    a0 = a[:M]
    w = torch.randint(-8, 8, (K + BK, N), device=D, dtype=torch.int8).contiguous()
    s = (torch.rand(K // G + 1, 2 * N, device=D) * 0.1 + 0.02).contiguous()
    z = torch.randint(-3, 4, (K // G + 1, 2 * N), device=D).float().contiguous()
    c = torch.full((4 * M, N), float("nan"), device=D)
    st = lambda t: t.stride()
    launch = lambda: _pg[(M // BM, N // BN)](
        a0,
        w,
        c,
        s,
        z,
        M,
        N,
        K,
        *st(a0),
        *st(w),
        s.stride(0),
        s.stride(1),
        z.stride(0),
        z.stride(1),
        *st(c),
        BM=BM,
        BN=BN,
        BK=BK,
        G=G,
        MODE=mode,
    )
    # simulate on the FULL backing buffer (same base, same strides): the shifted modes read
    # past the launched view into its slack rows exactly as the kernel does
    raw = _sim_gemm(0, a, w, c.shape, M, N, K, BM, BN, BK, G, s=s, z=z)
    intended = _sim_gemm(mode, a, w, c.shape, M, N, K, BM, BN, BK, G, s=s, z=z)
    return launch, c, raw, intended


def _sym_case(mode):
    M, N, K, BM, BN, BK = 64, 32, 128, 32, 32, 32
    torch.manual_seed(10301)
    a = torch.randn(4 * M, K, device=D)
    a0 = a[:M]
    w = torch.randint(-127, 127, (K + BK, N), device=D, dtype=torch.int8).contiguous()
    sc = torch.rand(4 * N, device=D) * 0.05 + 0.01
    c = torch.full((4 * M, N), float("nan"), device=D)
    launch = lambda: _sym[(M // BM, N // BN)](
        a0,
        w,
        c,
        sc,
        M,
        N,
        K,
        a0.stride(0),
        a0.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(0),
        c.stride(1),
        BM,
        BN,
        BK,
        MODE=mode,
    )
    raw = _sim_gemm(0, a, w, c.shape, M, N, K, BM, BN, BK, 1, sc=sc)  # full backing buffer, see _pg_case
    intended = _sim_gemm(mode, a, w, c.shape, M, N, K, BM, BN, BK, 1, sc=sc)
    return launch, c, raw, intended


def _g8_case(mode):
    N, K, BN, BK = 64, 128, 32, 32
    torch.manual_seed(10301)
    x = torch.randn(2 * K + BK, device=D) * 2  # slack for the shifted / doubled K indices
    w = torch.randint(-32, 32, (4 * N, K), device=D, dtype=torch.int8).contiguous()
    s = (torch.rand(4 * N, device=D) * 0.08 + 0.02).contiguous()
    z = torch.randint(-4, 5, (4 * N,), device=D).float().contiguous()
    o = torch.full((N + 8,), float("nan"), device=D)
    launch = lambda: _g8[(N // BN,)](x, w, o, s, z, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK, MODE=mode)
    return launch, o, _sim_gemv8(0, x, w, s, z, N, K, BN, BK), _sim_gemv8(mode, x, w, s, z, N, K, BN, BK)


def _g8_exact_case(mode, N, K, BN, BK):
    """Integer data for an EXACT comparison: x and zero are small integers, w is int8 and the
    scale a power of two, so every partial sum is an integer far below 2**24 and fp32 holds it
    exactly in any summation order. N and K need not be tile multiples: the kernel then reads
    and stores whole tiles inside the same contiguous buffers (flat addressing, as simulated).
    Built on the CPU; only the launch sees the device."""
    g = torch.Generator().manual_seed(10501)
    progs, iters = -(-N // BN), -(-K // BK)
    rows = 4 * progs * BN + 2 * BN + 8  # the doubled / shifted tile indices, and a ragged K spilling a row
    x = torch.randint(-4, 5, (2 * iters * BK + 3 * BK + 8,), generator=g).float()
    w = torch.randint(-32, 32, (rows, K), generator=g, dtype=torch.int8)
    s = (2.0 ** torch.randint(-1, 3, (2 * rows + 8,), generator=g).float()).contiguous()
    z = torch.randint(-4, 5, (rows + 8,), generator=g).float()
    raw = _sim_gemv8(0, x, w, s, z, N, K, BN, BK, progs=progs)
    intended = _sim_gemv8(mode, x, w, s, z, N, K, BN, BK, progs=progs)
    gx, gw, gs, gz = (t.to(D) for t in (x, w, s, z))
    o = torch.full((progs * BN + 8,), float("nan"), device=D)
    launch = lambda: _g8[(progs,)](gx, gw, o, gs, gz, N, K, gw.stride(0), gw.stride(1), BN=BN, BK=BK, MODE=mode)
    return launch, o, raw, intended


def _g4_case(mode):
    N, K, G, BN, BK = 64, 256, 128, 32, 32
    torch.manual_seed(10301)
    ng = K // G
    x = torch.randn(2 * K + BK, device=D) * 2
    w4 = torch.randint(0, 16, (4 * N, K), device=D, dtype=torch.int32)
    packed = (w4[:, 0::2] | (w4[:, 1::2] << 4)).to(torch.uint8).contiguous()
    s = (torch.rand(4 * N, ng, device=D) * 0.02 + 0.005).contiguous()
    z = torch.randint(0, 16, (4 * N, ng), device=D).float().contiguous()
    o = torch.full((N + 8,), float("nan"), device=D)
    launch = lambda: _g4[(N // BN,)](
        x, packed, o, s, z, N, K, ng, packed.stride(0), s.stride(0), BN=BN, BK=BK, G=G, MODE=mode
    )
    return (
        launch,
        o,
        _sim_gemv4(0, x, packed, s, z, N, K, ng, BN, BK, G),
        _sim_gemv4(mode, x, packed, s, z, N, K, ng, BN, BK, G),
    )


def _witness(launch, out, raw, intended, hits):
    """Refuse (touching nothing), or match the kernel's simulated semantics — never the
    template's canonical result; and the specialized quant dispatch must not be what produced
    a raw result."""
    sep = _cmp(raw, intended)
    assert sep > 0, "mode does not change the kernel's semantics on this data"
    try:
        launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(out.isnan().all()), "a refused kernel must touch nothing"
        return "refused"
    err = _cmp(out, intended)
    err_raw = _cmp(out, raw)
    assert err <= sep / 10, (
        f"stored the template's raw result / something else (err vs intended {err}, vs raw {err_raw}, sep {sep}; quant dispatch {hits})"
    )
    assert hits != [True] or err_raw > sep / 10, "the specialized quant dispatcher produced the raw result"
    return "computed"


@requires_gpu
@pytest.mark.parametrize("mode", _GEMM_MODES)
def test_pergroup_address_contract_never_raw(cold_gpu_caches, quant_dispatch_spy, mode):
    launch, c, raw, intended = _pg_case(mode)
    _witness(launch, c, raw, intended, quant_dispatch_spy)


@requires_gpu
@pytest.mark.parametrize("mode", _SYM_MODES)
def test_symmetric_address_contract_never_raw(cold_gpu_caches, quant_dispatch_spy, mode):
    launch, c, raw, intended = _sym_case(mode)
    _witness(launch, c, raw, intended, quant_dispatch_spy)


@requires_gpu
@pytest.mark.parametrize("mode", _GEMV_MODES)
def test_int8_gemv_address_contract_never_raw(cold_gpu_caches, quant_dispatch_spy, mode):
    launch, o, raw, intended = _g8_case(mode)
    _witness(launch, o, raw, intended, quant_dispatch_spy)


# (N, K, BN, BK): the witness launch; N and K off the tile; a wider K tile; a taller row tile.
_G8_EXACT_DIMS = [(64, 128, 32, 32), (40, 100, 32, 32), (24, 96, 16, 64), (70, 50, 64, 16)]


@requires_gpu
@pytest.mark.parametrize("dims", _G8_EXACT_DIMS, ids=lambda d: "N{}-K{}-BN{}-BK{}".format(*d))
@pytest.mark.parametrize("mode", _GEMV_MODES)
def test_int8_gemv_address_contract_computes_exactly(cold_gpu_caches, quant_dispatch_spy, mode, dims):
    """Every int8 GEMV contract violation now COMPUTES on the generic path, and what it stores
    is the kernel's own semantics exactly: every stored cell equal, every other cell untouched,
    and not through the specialized template. (Before the held-layout record these refused at
    the 1-D store of ``acc * scale``; the collapse to row 0 that refusal guarded against would
    show here as every row of a tile holding the first row's value.)"""
    launch, o, raw, intended = _g8_exact_case(mode, *dims)
    same = lambda a, b: torch.equal(torch.nan_to_num(a, nan=SENTINEL), torch.nan_to_num(b, nan=SENTINEL))
    assert not same(raw, intended), "mode does not change the kernel's semantics on this data"
    launch()
    torch.mps.synchronize()
    got = o.cpu()
    assert True not in quant_dispatch_spy, "a contract violation reached the specialized template"
    wrong = (torch.nan_to_num(got, nan=SENTINEL) != torch.nan_to_num(intended, nan=SENTINEL)).nonzero().flatten()
    assert same(got, intended), (
        f"{wrong.numel()} cells differ from the kernel's semantics, first {wrong[:4].tolist()}: "
        f"got {got[wrong[:4]].tolist()}, want {intended[wrong[:4]].tolist()}, template {raw[wrong[:4]].tolist()}"
    )


@requires_gpu
@pytest.mark.parametrize("mode", _GEMV_MODES)
def test_int4_gemv_address_contract_never_raw(cold_gpu_caches, quant_dispatch_spy, mode):
    launch, o, raw, intended = _g4_case(mode)
    _witness(launch, o, raw, intended, quant_dispatch_spy)


@requires_gpu
@pytest.mark.parametrize(
    "case", [_pg_case, _sym_case, _g8_case, _g4_case], ids=["pergroup", "symmetric", "gemv-int8", "gemv-int4"]
)
def test_native_forms_route_and_compute(cold_gpu_caches, quant_dispatch_spy, case):
    """Positives: mode 0 routes to the specialized template and matches its own simulation."""
    launch, out, raw, _ = case(0)
    launch()
    torch.mps.synchronize()
    assert quant_dispatch_spy == [True]
    assert _cmp(out, raw) < 3e-3 * max(1.0, torch.nan_to_num(raw, nan=0.0).abs().max().item())


# ---------------------------------------------------------------------------- lowering boundary
_SIG_PG = {
    "a_ptr": "*fp32",
    "w_ptr": "*i8",
    "c_ptr": "*fp32",
    "scale_ptr": "*fp32",
    "zero_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "sam": "i32",
    "sak": "constexpr",
    "swk": "i32",
    "swn": "constexpr",
    "ssg": "i32",
    "ssn": "constexpr",
    "zsg": "i32",
    "zsn": "constexpr",
    "scm": "i32",
    "scn": "constexpr",
}
_CEX_PG = {"BM": 32, "BN": 32, "BK": 32, "G": 32, "sak": 1, "swn": 1, "ssn": 1, "zsn": 1, "scn": 1}
_SIG_SYM = {
    "a_ptr": "*fp32",
    "w_ptr": "*i8",
    "c_ptr": "*fp32",
    "s_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "sam": "i32",
    "sak": "constexpr",
    "swk": "i32",
    "swn": "constexpr",
    "scm": "i32",
    "scn": "constexpr",
}
_CEX_SYM = {"BM": 32, "BN": 32, "BK": 32, "sak": 1, "swn": 1, "scn": 1}
_SIG_G8 = {
    "x_ptr": "*fp32",
    "w_ptr": "*i8",
    "o_ptr": "*fp32",
    "scale_ptr": "*fp32",
    "zero_ptr": "*fp32",
    "N": "i32",
    "K": "i32",
    "swn": "i32",
    "swk": "constexpr",
}
_CEX_G8 = {"BN": 32, "BK": 32, "swk": 1}
_SIG_G4 = {
    "x_ptr": "*fp32",
    "w_ptr": "*u8",
    "o_ptr": "*fp32",
    "s_ptr": "*fp32",
    "z_ptr": "*fp32",
    "N": "i32",
    "K": "i32",
    "ng": "i32",
    "swn": "i32",
    "ssn": "i32",
}
_CEX_G4 = {"BN": 32, "BK": 32, "G": 128}

_REFUSING = [pytest.param(_pg, _SIG_PG, _CEX_PG, m, id=f"pergroup-m{m}") for m in _GEMM_MODES] + [
    pytest.param(_sym, _SIG_SYM, _CEX_SYM, m, id=f"symmetric-m{m}") for m in _SYM_MODES
]


@requires
@pytest.mark.parametrize("fn,sig,cex,mode", _REFUSING)
def test_lowering_boundary_address_contract_refuses(fn, sig, cex, mode):
    """Rule 9: through emit_msl, every contract violation refuses (no descriptor, and the
    generic path refuses these forms) on the GEMM routes."""
    with pytest.raises(MetalNonRecoverableError):
        _emit_for(fn, sig, {**cex, "MODE": mode})


@requires
@pytest.mark.parametrize("mode", _GEMV_MODES)
def test_lowering_boundary_int8_gemv_declines(mode):
    """The int8 descriptor declines every contract violation (no quant descriptor is built);
    the generic path lowers the kernel op by op, with no stub
    (``test_int8_gemv_address_contract_computes_exactly`` proves on the GPU that it computes
    the kernel's semantics). These fourteen were refusal pins until the generic path could
    prove the layout of ``acc * scale``."""
    msl, md = _emit_for(_g8, _SIG_G8, {**_CEX_G8, "MODE": mode})
    assert "UNSUPPORTED" not in msl
    assert md.get("quant_matmul") is None


@requires
@pytest.mark.parametrize("mode", _GEMV_MODES)
def test_lowering_boundary_int4_gemv_declines(mode):
    """The int4 descriptor declines every contract violation (no quant descriptor is built);
    the generic path lowers the kernel op by op (GPU witnesses above prove it computes the
    kernel's semantics)."""
    msl, md = _emit_for(_g4, _SIG_G4, {**_CEX_G4, "MODE": mode})
    assert md.get("quant_matmul") is None


@requires
@pytest.mark.parametrize(
    "fn,sig,cex",
    [
        pytest.param(_pg, _SIG_PG, _CEX_PG, id="pergroup"),
        pytest.param(_sym, _SIG_SYM, _CEX_SYM, id="symmetric"),
        pytest.param(_g8, _SIG_G8, _CEX_G8, id="gemv-int8"),
        pytest.param(_g4, _SIG_G4, _CEX_G4, id="gemv-int4"),
    ],
)
def test_lowering_boundary_native_routes(fn, sig, cex):
    msl, md = _emit_for(fn, sig, {**cex, "MODE": 0})
    assert "UNSUPPORTED" not in msl
    assert md.get("quant_matmul") is not None
