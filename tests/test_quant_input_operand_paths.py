"""Round 2.1 F3 + packet 102: every value the quantized templates stage from a raw pointer
(the input A / x, the weight, the scale, the zero) must reach its use EXACTLY as the template
replays it — replay-based, default-deny.

On clean ``ba21e9d`` every quantized route — per-group int8 fast and scalar fallback,
symmetric int8, int8 decode GEMV, int4 decode GEMV — stored the RAW-POINTER result (err vs raw
0, err vs the IR-ordered oracle up to 372) for each of these transforms, without an exception:
a cast on the input (F3), a mask/other on the input load (101), a square in-tile transpose of
the input (101), a base or index offset on the input (102), a wrong-axis ``x[:, None]`` expand
at N == K (102), a mask on the weight / scale / zero loads (102), an index offset on the
symmetric scale (102). Now ``_quant_input_path_ok`` admits only the input grammar each role
replays (gemm: layout ops; gemv: ``x[None, :]``), every staged load must be PLAIN
(``_quant_plain_load``), the GEMV input and the symmetric scale addresses are proven exactly
(``_quant_1d_addr_ok``), and P0's strict stride tracer admits the in-loop ``(range + k)``
spelling only with the loop's induction variable. A failed proof returns no descriptor: the
GEMM chain refuses with its canonical-form message, the GEMV chain refuses on the generic path.

MODES (per route, where applicable): 0 native · 1 masked input · 2 square-tile transposed input
· 3 select on input · 4 arithmetic on input · 5 input base +1 · 6 in-loop A spelling (positive)
· 7 one-row broadcast input · 8 wrong-axis expand (GEMV, BN == BK) · 12 input index +1 ·
13 masked weight · 14 masked scale · 15 masked zero · 16 scale index +1 · 17 weight index +1 ·
20 input cast fp32→fp16→fp32 (F3).

Compile-time pins go through ``emit_msl`` with the LAUNCH-SPECIALIZED signature (unit strides
as ``constexpr`` 1 — the GEMV descriptors check that exact arg layout).

2026-11-18 (reduce of a row reduction, the held-layout record): the int8 wrong-axis expand
(mode 8 at BN == BK == 32) no longer refuses at the lowering boundary. Its descriptor still
declines it, and the generic path now proves the layout of ``acc * scale`` (a loop-carried row
reduction times a load through the row index) and lowers the kernel, as it already did for
the int4 route. The ``gemv-int8-m8`` refusal pin became
``test_lowering_boundary_int8_wrong_axis_declines_no_stub`` and the exact device tests
``test_int8_gemv_wrong_axis_expand_declines_and_computes`` / ``..._computes_exactly``.
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

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"

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
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        if MODE == 5:
            ap = a_ptr + 1 + om[:, None] * sam + ok[None, :] * sak
        elif MODE == 12:
            ap = a_ptr + om[:, None] * sam + (ok[None, :] + 1) * sak
        else:
            ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        if MODE == 17:
            wp = w_ptr + (ok[:, None] + 1) * swk + on[None, :] * swn
        else:
            wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            g = k // G
            if MODE == 14:
                s = tl.load(scale_ptr + g * ssg + on * ssn, mask=on < (BN // 2), other=0.0)
            elif MODE == 16:
                s = tl.load(scale_ptr + g * ssg + (on + 1) * ssn)
            else:
                s = tl.load(scale_ptr + g * ssg + on * ssn)
            if MODE == 15:
                z = tl.load(zero_ptr + g * zsg + on * zsn, mask=on < (BN // 2), other=0.0)
            else:
                z = tl.load(zero_ptr + g * zsg + on * zsn)
            if MODE == 13:
                w = (tl.load(wp, mask=ok[:, None] < (BK // 2), other=0).to(tl.float32) - z[None, :]) * s[None, :]
            else:
                w = (tl.load(wp).to(tl.float32) - z[None, :]) * s[None, :]
            if MODE == 1:
                a = tl.load(ap, mask=ok[None, :] < (BK // 2), other=0.0)
            elif MODE == 6:
                a = tl.load(a_ptr + om[:, None] * sam + (ok[None, :] + k) * sak)
            elif MODE == 7:
                arow = tl.load(a_ptr + k * sak + ok * sak)
                a = tl.broadcast_to(arow[None, :], (BM, BK))
            else:
                a = tl.load(ap)
            if MODE == 2:
                a = tl.trans(a)
            elif MODE == 3:
                a = tl.where(ok[None, :] < (BK // 2), a, 0.0)
            elif MODE == 4:
                a = a * 2.0
            elif MODE == 20:
                a = a.to(tl.float16).to(tl.float32)
            acc += tl.dot(a, w)
            ap += BK * sak
            wp += BK * swk
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
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        if MODE == 5:
            ap = a_ptr + 1 + om[:, None] * sam + ok[None, :] * sak
        else:
            ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        if MODE == 17:
            wp = w_ptr + (ok[:, None] + 1) * swk + on[None, :] * swn
        else:
            wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        if MODE == 14:
            scale = tl.load(s_ptr + on, mask=on < (BN // 2), other=0.0)
        elif MODE == 16:
            scale = tl.load(s_ptr + on + 1)
        else:
            scale = tl.load(s_ptr + on)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            if MODE == 13:
                w = tl.load(wp, mask=ok[:, None] < (BK // 2), other=0).to(tl.float32) * scale[None, :]
            else:
                w = tl.load(wp).to(tl.float32) * scale[None, :]
            if MODE == 1:
                a = tl.load(ap, mask=ok[None, :] < (BK // 2), other=0.0)
            else:
                a = tl.load(ap)
            if MODE == 2:
                a = tl.trans(a)
            elif MODE == 3:
                a = tl.where(ok[None, :] < (BK // 2), a, 0.0)
            elif MODE == 4:
                a = a * 2.0
            elif MODE == 20:
                a = a.to(tl.float16).to(tl.float32)
            acc += tl.dot(a, w)
            ap += BK * sak
            wp += BK * swk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _g8(
        x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        if MODE == 14:
            scale = tl.load(scale_ptr + offs_n, mask=(offs_n % BN) < (BN // 2), other=0.0)
        elif MODE == 16:
            scale = tl.load(scale_ptr + offs_n + 1)
        else:
            scale = tl.load(scale_ptr + offs_n)
        if MODE == 15:
            zero = tl.load(zero_ptr + offs_n, mask=(offs_n % BN) < (BN // 2), other=0.0)
        else:
            zero = tl.load(zero_ptr + offs_n)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            if MODE == 1:
                x = tl.load(x_ptr + offs_k + k, mask=offs_k < (BK // 2), other=0.0)
            elif MODE == 5:
                x = tl.load(x_ptr + 1 + offs_k + k)
            elif MODE == 12:
                x = tl.load(x_ptr + (offs_k + 1) + k)
            else:
                x = tl.load(x_ptr + offs_k + k)
            if MODE == 3:
                x = tl.where(offs_k < (BK // 2), x, 0.0)
            elif MODE == 4:
                x = x * 2.0
            elif MODE == 20:
                x = x.to(tl.float16).to(tl.float32)
            if MODE == 13:
                w = tl.load(
                    w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk,
                    mask=offs_k[None, :] < (BK // 2),
                    other=0,
                ).to(tl.float32)
            elif MODE == 17:
                w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k + 1) * swk).to(tl.float32)
            else:
                w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk).to(tl.float32)
            if MODE == 8:
                acc += tl.sum(x[:, None] * (w - zero[:, None]), axis=1)
            else:
                acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
        tl.store(o_ptr + offs_n, acc * scale)

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
        on = pid * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            kk = k + ok
            if MODE == 13:
                packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :], mask=ok[None, :] < (BK // 2), other=0)
            elif MODE == 17:
                packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2 + 1)[None, :])
            else:
                packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
            w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
            g = kk // G
            if MODE == 14:
                s = tl.load(s_ptr + on[:, None] * ssn + g[None, :], mask=(on % BN)[:, None] < (BN // 2), other=0.0)
            elif MODE == 16:
                s = tl.load(s_ptr + on[:, None] * ssn + g[None, :] + 1)
            else:
                s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
            if MODE == 15:
                z = tl.load(z_ptr + on[:, None] * ssn + g[None, :], mask=(on % BN)[:, None] < (BN // 2), other=0.0)
            else:
                z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
            if MODE == 1:
                x = tl.load(x_ptr + kk, mask=ok < (BK // 2), other=0.0)
            elif MODE == 5:
                x = tl.load(x_ptr + 1 + kk)
            elif MODE == 12:
                x = tl.load(x_ptr + (kk + 1))
            else:
                x = tl.load(x_ptr + kk)
            if MODE == 3:
                x = tl.where(ok < (BK // 2), x, 0.0)
            elif MODE == 4:
                x = x * 2.0
            elif MODE == 20:
                x = x.to(tl.float16).to(tl.float32)
            wd = (w4.to(tl.float32) - z) * s
            if MODE == 8:
                acc += tl.sum(x[:, None] * wd, axis=1)
            else:
                acc += tl.sum(x[None, :] * wd, axis=1)
        tl.store(o_ptr + on, acc)

    def _emit_for(fn, signature, constexprs):
        """The compiler's own pipeline through ``emit_msl`` (the lowering boundary)."""
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


def _witness(launch, c, raw, intended):
    """Refuse (touching nothing) or match the IR-ordered oracle; never the raw-pointer result."""
    sep = (raw - intended).abs().max().item()
    assert sep > 0
    try:
        launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(c.isnan().all()), "a refused kernel must touch nothing"
        return "refused"
    err = (c - intended).abs().max().item()
    assert err == err and err <= sep / 10, (
        f"computed the raw-pointer result, not the intended one (err {err}, separation {sep})"
    )
    return "computed"


def _tile_mask_zero(a, bk):
    a2 = a.clone().reshape(a.shape[0], -1, bk)
    a2[:, :, bk // 2 :] = 0
    return a2.reshape(a.shape)


def _tile_trans(a, bm, bk):
    t = torch.empty_like(a)
    for m0 in range(0, a.shape[0], bm):
        for k0 in range(0, a.shape[1], bk):
            t[m0 : m0 + bm, k0 : k0 + bk] = a[m0 : m0 + bm, k0 : k0 + bk].T
    return t


# ---------------------------------------------------------------- per-group int8 (fast + scalar)
_PG = dict(M=64, K=128, G=32, BM=32, BN=32, BK=32)


def _pg_case(N, mode):
    M, K, G, BM, BN, BK = (_PG[k] for k in ("M", "K", "G", "BM", "BN", "BK"))
    torch.manual_seed(10111)
    ng = K // G
    # Slack for the +1 modes lives INSIDE contiguous tensors (an extra row), so the
    # launched tensors stay contiguous with their canonical strides.
    a_buf = torch.randn(M * K + 8, device=D)
    a = a_buf[: M * K].view(M, K)
    w = torch.randint(-8, 8, (K + 1, N), device=D, dtype=torch.int8).contiguous()  # row K = slack
    s = (torch.rand(ng + 1, N, device=D) * 0.1 + 0.02).contiguous()  # row ng = slack
    z = torch.randint(-3, 4, (ng, N), device=D).float().contiguous()
    gi = torch.arange(K, device=D) // G
    w0, s0 = w[:K].float(), s[:ng]
    wd = (w0 - z[gi, :]) * s0[gi, :]
    raw = a @ wd
    half_n = (torch.arange(N, device=D) < BN // 2).float()
    ia, iwd = a, wd
    if mode in (1, 3):
        ia = _tile_mask_zero(a, BK)
    elif mode == 2:
        ia = _tile_trans(a, BM, BK)
    elif mode == 4:
        ia = 2 * a
    elif mode in (5, 12):
        ia = a_buf[1 : 1 + M * K].view(M, K)
    elif mode == 7:
        ia = a[0:1, :].expand(M, K)
    elif mode == 20:
        ia = a.half().float()
    elif mode == 13:
        wm = w0.clone().reshape(K // BK, BK, N)
        wm[:, BK // 2 :, :] = 0
        iwd = (wm.reshape(K, N) - z[gi, :]) * s0[gi, :]
    elif mode == 14:
        iwd = (w0 - z[gi, :]) * (s0 * half_n[None, :])[gi, :]
    elif mode == 15:
        iwd = (w0 - (z * half_n[None, :])[gi, :]) * s0[gi, :]
    elif mode == 16:
        s_shift = s.reshape(-1)[1 : 1 + ng * N].view(ng, N)  # s[g, n + 1] in flat order
        iwd = (w0 - z[gi, :]) * s_shift[gi, :]
    elif mode == 17:
        iwd = (w[1 : K + 1].float() - z[gi, :]) * s0[gi, :]
    c = torch.full((M, N), float("nan"), device=D)
    st = lambda t: t.stride()
    launch = lambda: _pg[(M // BM, triton.cdiv(N, BN))](
        a,
        w,
        c,
        s,
        z,
        M,
        N,
        K,
        *st(a),
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
    return launch, c, raw, ia @ iwd


_PG_BITES = [1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 20]


@requires_gpu
@pytest.mark.parametrize("mode", _PG_BITES)
@pytest.mark.parametrize("N", [32, 24], ids=["fast", "scalar-fallback"])
def test_pergroup_transform_never_raw(cold_gpu_caches, N, mode):
    """Bites on ba21e9d (raw result stored) for every mode except 7 (already refused)."""
    launch, c, raw, intended = _pg_case(N, mode)
    _witness(launch, c, raw, intended)


@requires_gpu
@pytest.mark.parametrize("mode", [0, 6], ids=["native", "in-loop-A-spelling"])
@pytest.mark.parametrize("N", [32, 24], ids=["fast", "scalar-fallback"])
def test_pergroup_native_forms_compute(cold_gpu_caches, N, mode):
    """Positives: the native form and the in-loop ``a_ptr + om*sam + (ok + k)*sak`` spelling
    (mainstream; computed on ba21e9d, refused by P0's strict tracer until packet 102)."""
    launch, c, raw, _ = _pg_case(N, mode)
    launch()
    torch.mps.synchronize()
    assert (c - raw).abs().max().item() < 3e-3


# ---------------------------------------------------------------- symmetric int8
def _sym_case(mode):
    M, N, K, BM, BN, BK = 64, 32, 128, 32, 32, 32
    torch.manual_seed(10111)
    a_buf = torch.randn(M * K + 8, device=D)
    a = a_buf[: M * K].view(M, K)
    w = torch.randint(-127, 127, (K + 1, N), device=D, dtype=torch.int8).contiguous()  # row K = slack
    sc = torch.rand(N + 8, device=D) * 0.05 + 0.01  # contiguous 1-D with slack
    w0, sc0 = w[:K].float(), sc[:N]
    wd = w0 * sc0
    raw = a @ wd
    half_n = (torch.arange(N, device=D) < BN // 2).float()
    ia, iwd = a, wd
    if mode in (1, 3):
        ia = _tile_mask_zero(a, BK)
    elif mode == 2:
        ia = _tile_trans(a, BM, BK)
    elif mode == 4:
        ia = 2 * a
    elif mode == 5:
        ia = a_buf[1 : 1 + M * K].view(M, K)
    elif mode == 20:
        ia = a.half().float()
    elif mode == 13:
        wm = w0.clone().reshape(K // BK, BK, N)
        wm[:, BK // 2 :, :] = 0
        iwd = wm.reshape(K, N) * sc0
    elif mode == 14:
        iwd = w0 * (sc0 * half_n)
    elif mode == 16:
        iwd = w0 * sc[1 : N + 1]
    elif mode == 17:
        iwd = w[1 : K + 1].float() * sc0
    c = torch.full((M, N), float("nan"), device=D)
    launch = lambda: _sym[(M // BM, N // BN)](
        a,
        w,
        c,
        sc,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(0),
        c.stride(1),
        BM,
        BN,
        BK,
        MODE=mode,
    )
    return launch, c, raw, ia @ iwd


_SYM_BITES = [1, 2, 3, 4, 5, 13, 14, 16, 17, 20]


@requires_gpu
@pytest.mark.parametrize("mode", _SYM_BITES)
def test_symmetric_transform_never_raw(cold_gpu_caches, mode):
    launch, c, raw, intended = _sym_case(mode)
    _witness(launch, c, raw, intended)


@requires_gpu
def test_symmetric_native_computes(cold_gpu_caches):
    launch, c, raw, _ = _sym_case(0)
    launch()
    torch.mps.synchronize()
    assert (c - raw).abs().max().item() < 3e-3


# ---------------------------------------------------------------- int8 decode GEMV
def _g8_case(mode, BK=64):
    N, K, BN = 256, 512, 32
    torch.manual_seed(10111)
    x_buf = torch.randn(K + 8, device=D)
    x = x_buf[:K]
    w = torch.randint(-128, 128, (N + 1, K), device=D, dtype=torch.int8).contiguous()  # row N = slack, swn == K
    s = torch.rand(N + 8, device=D) * 0.05 + 0.01  # contiguous 1-D with slack
    z = torch.randint(-4, 4, (N,), device=D).float()
    w0, s0 = w[:N].float(), s[:N]
    wz = w0 - z[:, None]
    wd = wz * s0[:, None]
    raw = wd @ x
    half = ((torch.arange(N, device=D) % BN) < BN // 2).float()
    if mode in (1, 3):
        xi = x.clone().reshape(-1, BK)
        xi[:, BK // 2 :] = 0
        intended = wd @ xi.reshape(-1)
    elif mode == 4:
        intended = wd @ (2 * x)
    elif mode in (5, 12):
        intended = wd @ x_buf[1 : K + 1]
    elif mode == 20:
        intended = wd @ x.half().float()
    elif mode == 8:
        T = K // BK
        tiles = wz.reshape(N, T, BK).sum(2)
        xn = x.reshape(T, BK).T[torch.arange(N, device=D) % BN]
        intended = s0 * (xn * tiles).sum(1)
    elif mode == 13:
        wm = wz.clone().reshape(N, K // BK, BK)
        wm[:, :, BK // 2 :] = 0
        intended = (wm.reshape(N, K) * s0[:, None]) @ x
    elif mode == 14:
        intended = (wz * (s0 * half)[:, None]) @ x
    elif mode == 15:
        intended = ((w0 - (z * half)[:, None]) * s0[:, None]) @ x
    elif mode == 16:
        intended = (wz * s[1 : N + 1][:, None]) @ x
    elif mode == 17:
        w_shift = w.reshape(-1)[1 : 1 + N * K].view(N, K).float()  # w[n, k + 1] in flat order
        intended = ((w_shift - z[:, None]) * s0[:, None]) @ x
    else:
        intended = raw
    o = torch.full((N,), float("nan"), device=D)
    launch = lambda: _g8[(N // BN,)](x, w, o, s, z, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK, MODE=mode)
    return launch, o, raw, intended


_G8_BITES = [1, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20]


@requires_gpu
@pytest.mark.parametrize("mode", _G8_BITES)
def test_int8_gemv_transform_never_raw(cold_gpu_caches, mode):
    launch, o, raw, intended = _g8_case(mode)
    _witness(launch, o, raw, intended)


@requires_gpu
def test_int8_gemv_wrong_axis_expand_never_raw(cold_gpu_caches):
    """``x[:, None] * (w - zero)`` at BN == BK is a different contraction; the template
    replayed ``x[None, :]`` (raw result, err 372 on ba21e9d)."""
    launch, o, raw, intended = _g8_case(8, BK=32)
    _witness(launch, o, raw, intended)


@requires_gpu
def test_int8_gemv_wrong_axis_expand_declines_and_computes(cold_gpu_caches):
    """The int8 descriptor declines ``x[:, None]``; the generic path now replays it (it
    refused until the layout of ``acc * scale`` could be proved)."""
    launch, o, raw, intended = _g8_case(8, BK=32)
    assert _witness(launch, o, raw, intended) == "computed"


def _g8_wrong_axis_exact_case(N, K, B):
    """Integer data for an EXACT comparison of the wrong-axis expand at BN == BK == B: x and
    zero are small integers, w is int8 and the scale a power of two, so every partial sum is an
    integer far below 2**24 and fp32 holds it exactly in any summation order. N and K need not
    be multiples of B: the kernel has no masks, so it reads and stores whole tiles inside the
    same contiguous buffers (flat addressing, mirrored here). Returns the launch, the output,
    the kernel's own semantics and the template's (``x[None, :]``) result."""
    g = torch.Generator().manual_seed(10502)
    progs, iters = -(-N // B), -(-K // B)
    rows = progs * B + 2  # a ragged K spills the last tile into the next row
    x = torch.randint(-4, 5, (iters * B + 8,), generator=g).float()
    w = torch.randint(-32, 32, (rows, K), generator=g, dtype=torch.int8)
    s = 2.0 ** torch.randint(-1, 3, (progs * B + 8,), generator=g).float()
    z = torch.randint(-4, 5, (progs * B + 8,), generator=g).float()
    wf, r = w.reshape(-1).float(), torch.arange(B)
    intended = torch.full((progs * B + 8,), float("nan"))
    raw = intended.clone()
    for pid in range(progs):
        rn = pid * B + r
        acc, acc_raw = torch.zeros(B), torch.zeros(B)
        for k in range(0, K, B):
            wz = wf[rn[:, None] * K + (r[None, :] + k)] - z[rn][:, None]
            acc += (x[r + k][:, None] * wz).sum(1)  # x[:, None]: row i scaled by x[k + i]
            acc_raw += (x[r + k][None, :] * wz).sum(1)
        intended[rn], raw[rn] = acc * s[rn], acc_raw * s[rn]
    gx, gw, gs, gz = (t.to(D) for t in (x, w, s, z))
    o = torch.full((progs * B + 8,), float("nan"), device=D)
    launch = lambda: _g8[(progs,)](gx, gw, o, gs, gz, N, K, gw.stride(0), gw.stride(1), BN=B, BK=B, MODE=8)
    return launch, o, intended, raw


@requires_gpu
@pytest.mark.parametrize(
    "N,K,B",
    [(256, 512, 32), (64, 128, 32), (40, 100, 32), (32, 32, 32), (64, 128, 16), (24, 40, 16)],
    ids=lambda v: str(v),
)
def test_int8_gemv_wrong_axis_expand_computes_exactly(cold_gpu_caches, N, K, B):
    """The wrong-axis expand on the generic path equals the kernel's own semantics EXACTLY,
    at sizes that are and are not multiples of the tile: every stored cell equal, every other
    cell untouched, and different from the template's contraction."""
    launch, o, intended, raw = _g8_wrong_axis_exact_case(N, K, B)
    same = lambda a, b: torch.equal(torch.nan_to_num(a, nan=98765.0), torch.nan_to_num(b, nan=98765.0))
    assert not same(raw, intended), "the expand axis does not change the result on this data"
    launch()
    torch.mps.synchronize()
    got = o.cpu()
    wrong = (torch.nan_to_num(got, nan=98765.0) != torch.nan_to_num(intended, nan=98765.0)).nonzero().flatten()
    assert same(got, intended), (
        f"{wrong.numel()} cells differ from the kernel's semantics, first {wrong[:4].tolist()}: "
        f"got {got[wrong[:4]].tolist()}, want {intended[wrong[:4]].tolist()}, template {raw[wrong[:4]].tolist()}"
    )


@requires_gpu
def test_int8_gemv_native_computes(cold_gpu_caches):
    launch, o, raw, _ = _g8_case(0)
    launch()
    torch.mps.synchronize()
    assert (o - raw).abs().max().item() < 3e-3


# ---------------------------------------------------------------- int4 decode GEMV
def _g4_case(mode, BK=64):
    N, K, G, BN = 256, 512, 128, 32
    torch.manual_seed(10111)
    ng = K // G
    x_buf = torch.randn(K + 8, device=D)
    x = x_buf[:K]
    w4_ext = torch.randint(0, 16, (N + 1, K), device=D, dtype=torch.int32)  # row N = slack, swn == K/2
    packed = (w4_ext[:, 0::2] | (w4_ext[:, 1::2] << 4)).to(torch.uint8).contiguous()
    w4 = w4_ext[:N]
    s = (torch.rand(N + 1, ng, device=D) * 0.02 + 0.005).contiguous()  # row N = slack, ssn == ng
    z = torch.randint(0, 16, (N, ng), device=D).float().contiguous()
    gi = torch.arange(K, device=D) // G
    s0 = s[:N]
    wd = (w4.float() - z[:, gi]) * s0[:, gi]
    raw = wd @ x
    half = ((torch.arange(N, device=D) % BN) < BN // 2).float()
    if mode in (1, 3):
        xi = x.clone().reshape(-1, BK)
        xi[:, BK // 2 :] = 0
        intended = wd @ xi.reshape(-1)
    elif mode == 4:
        intended = wd @ (2 * x)
    elif mode in (5, 12):
        intended = wd @ x_buf[1 : K + 1]
    elif mode == 20:
        intended = wd @ x.half().float()
    elif mode == 8:
        T = K // BK
        tiles = wd.reshape(N, T, BK).sum(2)
        xn = x.reshape(T, BK).T[torch.arange(N, device=D) % BN]
        intended = (xn * tiles).sum(1)
    elif mode == 13:
        keep = ((torch.arange(K, device=D) % BK) < BK // 2).float()
        intended = ((w4.float() * keep - z[:, gi]) * s0[:, gi]) @ x
    elif mode == 14:
        intended = ((w4.float() - z[:, gi]) * (s0 * half[:, None])[:, gi]) @ x
    elif mode == 15:
        intended = ((w4.float() - (z * half[:, None])[:, gi]) * s0[:, gi]) @ x
    elif mode == 16:
        s_shift = s.reshape(-1)[1 : 1 + N * ng].view(N, ng)  # s[n, g + 1] in flat order
        intended = ((w4.float() - z[:, gi]) * s_shift[:, gi]) @ x
    elif mode == 17:
        w_shift = w4_ext.reshape(-1)[2 : 2 + N * K].view(N, K).float()  # byte + 1 = nibble k + 2, flat order
        intended = ((w_shift - z[:, gi]) * s0[:, gi]) @ x
    else:
        intended = raw
    o = torch.full((N,), float("nan"), device=D)
    launch = lambda: _g4[(N // BN,)](
        x, packed, o, s, z, N, K, ng, packed.stride(0), s.stride(0), BN=BN, BK=BK, G=G, MODE=mode
    )
    return launch, o, raw, intended


_G4_BITES = [1, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20]


@requires_gpu
@pytest.mark.parametrize("mode", _G4_BITES)
def test_int4_gemv_transform_never_raw(cold_gpu_caches, mode):
    launch, o, raw, intended = _g4_case(mode)
    _witness(launch, o, raw, intended)


@requires_gpu
def test_int4_gemv_wrong_axis_expand_declines_and_computes(cold_gpu_caches):
    """The int4 descriptor declines ``x[:, None]``; the generic path replays it exactly."""
    launch, o, raw, intended = _g4_case(8, BK=32)
    assert _witness(launch, o, raw, intended) == "computed"


@requires_gpu
def test_int4_gemv_native_computes(cold_gpu_caches):
    launch, o, raw, _ = _g4_case(0)
    launch()
    torch.mps.synchronize()
    torch.testing.assert_close(o, raw, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------- lowering boundary (rule 9)
# Launch-specialized signatures: Triton passes unit strides as constexpr 1 (dropped from the
# runtime arg list); the descriptors see exactly this layout at launch.
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
_CEX_G8 = {"BN": 32, "BK": 64, "swk": 1}
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
_CEX_G4 = {"BN": 32, "BK": 64, "G": 128}

_BOUNDARY = (
    [pytest.param(_pg, _SIG_PG, _CEX_PG, m, id=f"pergroup-m{m}") for m in _PG_BITES]
    + [pytest.param(_sym, _SIG_SYM, _CEX_SYM, m, id=f"symmetric-m{m}") for m in _SYM_BITES]
    + [pytest.param(_g8, _SIG_G8, _CEX_G8, m, id=f"gemv-int8-m{m}") for m in _G8_BITES]
    + [pytest.param(_g4, _SIG_G4, _CEX_G4, m, id=f"gemv-int4-m{m}") for m in _G4_BITES]
)


@requires
@pytest.mark.parametrize("fn,sig,cex,mode", _BOUNDARY)
def test_lowering_boundary_quant_transform_refuses(fn, sig, cex, mode):
    """Rule 9: fresh TTGIR through emit_msl, every transform refuses on every route (the
    wrong-axis expand, mode 8, is the exception on both GEMV routes: declined, then lowered
    by the generic path — see the two ``..._wrong_axis_declines_no_stub`` tests below)."""
    with pytest.raises(MetalNonRecoverableError):
        _emit_for(fn, sig, {**cex, "MODE": mode})


@requires
@pytest.mark.parametrize(
    "fn,sig,cex,mode",
    [
        pytest.param(_pg, _SIG_PG, _CEX_PG, 0, id="pergroup-native"),
        pytest.param(_pg, _SIG_PG, _CEX_PG, 6, id="pergroup-in-loop-spelling"),
        pytest.param(_sym, _SIG_SYM, _CEX_SYM, 0, id="symmetric-native"),
        pytest.param(_g8, _SIG_G8, _CEX_G8, 0, id="gemv-int8-native"),
        pytest.param(_g4, _SIG_G4, _CEX_G4, 0, id="gemv-int4-native"),
    ],
)
def test_lowering_boundary_quant_native_routes(fn, sig, cex, mode):
    """Positive controls: the native forms (and the in-loop A spelling) still build their
    quantized dispatch descriptor — no capability lost to the stricter proofs."""
    msl, md = _emit_for(fn, sig, {**cex, "MODE": mode})
    assert "UNSUPPORTED" not in msl
    assert md.get("quant_matmul") is not None, "native quantized form no longer routes to its template"


@requires
def test_lowering_boundary_int8_wrong_axis_declines_no_stub():
    """The int8 descriptor declines ``x[:, None]`` and the generic path lowers it (no stub).
    This was a refusal pin until the generic path could prove the layout of ``acc * scale``."""
    msl, md = _emit_for(_g8, _SIG_G8, {**_CEX_G8, "BK": 32, "MODE": 8})
    assert "UNSUPPORTED" not in msl
    assert md.get("quant_matmul") is None


@requires
def test_lowering_boundary_int4_wrong_axis_declines_no_stub():
    """The int4 descriptor declines ``x[:, None]`` and the generic path lowers it (no stub)."""
    msl, md = _emit_for(_g4, _SIG_G4, {**_CEX_G4, "BK": 32, "MODE": 8})
    assert "UNSUPPORTED" not in msl
    assert md.get("quant_matmul") is None
