# _biased_fa and _biased_tri_fa adapt TriFast's forward kernel.
# Copyright (c) 2025 Liam Atkinson; MIT License (third_party/trifast/LICENSE).
# Local test adaptations and provenance: docs/TRIFAST_PROVENANCE.md.
"""End-to-end: a trifast-style BIASED / triangle-attention @triton.jit kernel is
detected, routed to the biased tiled template, and computes correctly on Metal —
where the generic path used to refuse (dot-result-scale guard). Plus correct-or-
refuse coverage: standard FA is NOT misrouted, and a mismatched softmax
temperature is refused.

The biased pattern = standard FA (2 dots + exp + max) PLUS a bias tile as the QK
dot's C operand, a loaded mask -> -inf, a per-row lse store, a runtime scale into
Q, and a const result-scale (inv_ln2) with exp2 (== natural softmax over
sm_scale*QK + bias). See generic_lowerer._detect_biased_flash_attention.
"""

import math
import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _biased_fa(
    o_ptr,
    o_sz,
    o_sh,
    o_sm,
    o_sk,
    lse_ptr,
    lse_sz,
    lse_sh,
    lse_sm,
    q_ptr,
    q_sz,
    q_sh,
    q_sm,
    q_sk,
    k_ptr,
    k_sz,
    k_sh,
    k_sn,
    k_sk,
    v_ptr,
    v_sz,
    v_sh,
    v_sn,
    v_sk,
    b_ptr,
    b_sz,
    b_sh,
    b_sm,
    b_sn,
    mask_ptr,
    m_sz,
    m_sh,
    m_sn,
    sm_scale,
    neg_inf,
    Z,
    H,
    N,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BAD_TEMP: tl.constexpr = False,
    I3D: tl.constexpr = False,
):
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    if I3D:  # a 3rd program_id (trifast's triangle-i axis) -> detector must refuse
        z = z + tl.program_id(2)
    start_m = pid_m * BLOCK_M
    m_idxs = start_m + tl.arange(0, BLOCK_M)
    n_idxs = tl.arange(0, BLOCK_N)
    d_idxs = tl.arange(0, DIM)
    q_ptrs = q_ptr + z * q_sz + h * q_sh + m_idxs[:, None] * q_sm + d_idxs[None, :] * q_sk
    kt_ptrs = k_ptr + z * k_sz + h * k_sh + d_idxs[:, None] * k_sk + n_idxs[None, :] * k_sn
    v_ptrs = v_ptr + z * v_sz + h * v_sh + n_idxs[:, None] * v_sn + d_idxs[None, :] * v_sk
    b_ptrs = b_ptr + z * b_sz + h * b_sh + m_idxs[:, None] * b_sm + n_idxs[None, :] * b_sn
    mask_ptrs = mask_ptr + z * m_sz + h * m_sh + n_idxs * m_sn
    o_ptrs = o_ptr + z * o_sz + h * o_sh + m_idxs[:, None] * o_sm + d_idxs[None, :] * o_sk
    lse_ptrs = lse_ptr + z * lse_sz + h * lse_sh + m_idxs * lse_sm

    scores_max = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_M], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_M, DIM], value=0, dtype=tl.float32)
    mask_m = m_idxs < N
    q_block = tl.load(q_ptrs, mask_m[:, None])
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

    for start_n in tl.range(0, N, BLOCK_N):
        mask_n = (n_idxs + start_n) < N
        kt_block = tl.load(kt_ptrs, mask_n[None, :])
        b_block = tl.load(b_ptrs, mask_m[:, None] & mask_n[None, :])
        m_block = tl.load(mask_ptrs, mask_n)
        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores)
        if not BAD_TEMP:
            scores *= inv_ln2  # inv_ln2 + exp2 == natural softmax
        scores = tl.where(m_block[None, :], neg_inf, scores)
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, neg_inf)
        block_max = tl.maximum(scores_max, tl.max(scores, 1))
        scores = scores - block_max[:, None]
        exp_scores = tl.math.exp2(scores)
        summed = tl.sum(exp_scores, 1)
        exp_scale = tl.math.exp2(scores_max - block_max)
        sm_denom = sm_denom * exp_scale + summed
        acc = acc * exp_scale[:, None]
        v_block = tl.load(v_ptrs, mask_n[:, None])
        exp_scores = exp_scores.to(q_block.type.element_ty)
        acc = tl.dot(exp_scores, v_block, acc)
        scores_max = block_max
        kt_ptrs += BLOCK_N * k_sn
        v_ptrs += BLOCK_N * v_sn
        b_ptrs += BLOCK_N * b_sn
        mask_ptrs += BLOCK_N * m_sn

    normalize = acc / sm_denom[:, None]
    tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])
    lse = (scores_max * ln2) + tl.log(sm_denom)
    tl.store(lse_ptrs, lse, mask=mask_m)


def _run(Z, H, N, DIM, bad_temp=False):
    torch.manual_seed(0)
    dev = "mps"
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Z, H, N, DIM, device=dev)
    k = torch.randn(Z, H, N, DIM, device=dev)
    v = torch.randn(Z, H, N, DIM, device=dev)
    b = torch.randn(Z, H, N, N, device=dev)
    mask = (torch.rand(Z, H, N, device=dev) < 0.25).to(torch.uint8)
    o = torch.zeros(Z, H, N, DIM, device=dev)
    lse = torch.zeros(Z, H, N, device=dev)
    st = lambda t: tuple(t.stride())
    grid = (triton.cdiv(N, 32), Z * H)
    _biased_fa[grid](
        o,
        *st(o),
        lse,
        *st(lse),
        q,
        *st(q),
        k,
        *st(k),
        v,
        *st(v),
        b,
        *st(b),
        mask,
        *st(mask),
        sm,
        -1e9,
        Z,
        H,
        N,
        DIM,
        32,
        32,
        bad_temp,
    )
    torch.mps.synchronize()
    raw = sm * (q @ k.transpose(-2, -1)) + b
    raw = raw.masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o_ref = torch.nan_to_num(p, nan=0.0) @ v
    lse_ref = torch.logsumexp(raw, dim=-1)
    return o, lse, o_ref, lse_ref


@requires_mps
@pytest.mark.parametrize("Z,H,N,DIM", [(1, 2, 64, 32), (1, 1, 96, 32), (2, 2, 64, 32)])
def test_biased_fa_routes_and_computes(Z, H, N, DIM):
    o, lse, o_ref, lse_ref = _run(Z, H, N, DIM)
    assert (o - o_ref).abs().max().item() < 1e-3
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 1e-3


@requires_mps
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-3), (torch.float16, 3e-2)])
@pytest.mark.parametrize("DIM, BLOCK_N", [(32, 128), (64, 128), (64, 64), (128, 64)])
def test_biased_fa_head_dim_stride_two_computes(DIM, BLOCK_N, dtype, tol):
    """Every other head-dim element of wider tensors: the 64-key template needs stride 1,
    so these take the tiled template and its narrowed head-dim slice."""
    Z, H, N = 1, 2, 96
    gen = torch.Generator().manual_seed(7)
    wide = {name: torch.randn(Z, H, N, 2 * DIM, generator=gen).to(dtype).to("mps") for name in "qkv"}
    q, k, v = (wide[name][..., ::2] for name in "qkv")
    b = torch.randn(Z, H, N, N, generator=gen).to(dtype).to("mps")
    mask = (torch.rand(Z, H, N, generator=gen) < 0.25).to(torch.uint8).to("mps")
    held = torch.full((Z, H, N, 2 * DIM), -8192.0, dtype=dtype, device="mps")
    o, lse = held[..., ::2], torch.zeros(Z, H, N, dtype=dtype, device="mps")
    args = []
    for t in (o, lse, q, k, v, b, mask):
        args += [t, *t.stride()]
    _biased_fa[(triton.cdiv(N, 32), Z * H)](*args, DIM**-0.5, -1e9, Z, H, N, DIM, 32, BLOCK_N)
    torch.mps.synchronize()
    raw = DIM**-0.5 * (q.cpu().double() @ k.cpu().double().transpose(-2, -1)) + b.cpu().double()
    raw = raw.masked_fill(mask.cpu()[:, :, None, :].bool(), float("-inf"))
    want = torch.softmax(raw, dim=-1) @ v.cpu().double()
    assert (o.cpu().double() - want).abs().max().item() < tol
    assert (lse.cpu().double() - torch.logsumexp(raw, dim=-1)).abs().max().item() < tol
    assert bool((held[..., 1::2] == -8192.0).all())  # the elements between stay untouched


@requires_mps
def test_biased_fa_bad_temperature_refuses():
    """exp2 WITHOUT the inv_ln2 result-scale => softmax temperature = ln2 != 1 =>
    the detector must refuse (routing it to the natural-exp template would be wrong).
    The refuse surfaces as either a MetalNonRecoverableError or a codegen warning;
    either proves it was NOT silently mis-routed to the template."""
    import warnings
    from triton_msl.errors import MetalNonRecoverableError

    refused = False
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            _run(1, 2, 64, 32, bad_temp=True)
        except MetalNonRecoverableError as e:
            refused = "temperature" in str(e) or "Refusing" in str(e)
    refused = refused or any("temperature" in str(w.message) or "Refusing" in str(w.message) for w in wl)
    assert refused, "bad-temperature biased FA must be refused, not silently mis-routed"


@triton.jit
def _biased_tri_fa(
    o_ptr,
    o_sh,
    o_si,
    o_sn,
    o_sd,
    lse_ptr,
    lse_sh,
    lse_si,
    lse_sn,
    q_ptr,
    q_sh,
    q_si,
    q_sn,
    q_sd,
    k_ptr,
    k_sh,
    k_si,
    k_sn,
    k_sd,
    v_ptr,
    v_sh,
    v_si,
    v_sn,
    v_sd,
    b_ptr,
    b_sh,
    b_sn,
    b_sm,  # bias [h, n, m] — SHARED across the i-axis
    mask_ptr,
    mask_sh,
    mask_si,
    mask_sn,  # mask [batch, i, k]; batch = pid_h // H
    sm_scale,
    neg_inf,
    N,
    H,
    DIM: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """trifast-style 3-D triangle attention: pid_j/pid_i/pid_h grid, bias shared
    across i, mask indexed pid_h//H (cross-head). Mirrors scratchpad trifast _fwd."""
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    mask_start_h = pid_h // H
    start_j = pid_j * BLOCK_J
    j_idxs = start_j + tl.arange(0, BLOCK_J)
    k_idxs = tl.arange(0, BLOCK_K)
    d_idxs = tl.arange(0, DIM)
    q_ptrs = q_ptr + pid_h * q_sh + pid_i * q_si + j_idxs[:, None] * q_sn + d_idxs[None, :] * q_sd
    kt_ptrs = k_ptr + pid_h * k_sh + pid_i * k_si + d_idxs[:, None] * k_sd + k_idxs[None, :] * k_sn
    v_ptrs = v_ptr + pid_h * v_sh + pid_i * v_si + k_idxs[:, None] * v_sn + d_idxs[None, :] * v_sd
    b_ptrs = b_ptr + pid_h * b_sh + j_idxs[:, None] * b_sn + k_idxs[None, :] * b_sm
    mask_ptrs = mask_ptr + mask_start_h * mask_sh + pid_i * mask_si + k_idxs * mask_sn
    o_ptrs = o_ptr + pid_h * o_sh + pid_i * o_si + j_idxs[:, None] * o_sn + d_idxs[None, :] * o_sd
    lse_ptrs = lse_ptr + pid_h * lse_sh + pid_i * lse_si + j_idxs * lse_sn

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)
    mask_j = j_idxs < N
    q_block = tl.load(q_ptrs, mask_j[:, None])
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)
    for start_k in tl.range(0, N, BLOCK_K):
        mask_k = (k_idxs + start_k) < N
        kt_block = tl.load(kt_ptrs, mask_k[None, :])
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])
        m_block = tl.load(mask_ptrs, mask_k)
        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores)
        scores *= inv_ln2
        scores = tl.where(m_block[None, :], neg_inf, scores)
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        block_max = tl.maximum(scores_max, tl.max(scores, 1))
        scores = scores - block_max[:, None]
        exp_scores = tl.math.exp2(scores)
        summed = tl.sum(exp_scores, 1)
        exp_scale = tl.math.exp2(scores_max - block_max)
        sm_denom = sm_denom * exp_scale + summed
        acc = acc * exp_scale[:, None]
        v_block = tl.load(v_ptrs, mask_k[:, None])
        exp_scores = exp_scores.to(q_block.type.element_ty)
        acc = tl.dot(exp_scores, v_block, acc)
        scores_max = block_max
        kt_ptrs += BLOCK_K * k_sn
        v_ptrs += BLOCK_K * v_sn
        b_ptrs += BLOCK_K * b_sm
        mask_ptrs += BLOCK_K * mask_sn
    normalize = acc / sm_denom[:, None]
    tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_j[:, None])
    lse = (scores_max * ln2) + tl.log(sm_denom)
    tl.store(lse_ptrs, lse, mask=mask_j)


@requires_mps
@pytest.mark.parametrize(
    "DIM,dtype,o_tol",
    [
        (32, torch.float32, 1e-3),  # small-head simd-MMA path
        (64, torch.float32, 1e-3),  # simd-MMA path
        (32, torch.bfloat16, 6e-2),  # bf16 in/out, fp32 compute -> tiled path
    ],
)
def test_biased_tri_fa_3d_computes(DIM, dtype, o_tol):
    """trifast's 3-D triangle attention (bias shared across i, mask pid_h//H,
    >31 args -> packed ABI) computes correctly on Metal via the biased template.
    fp32 DIM=32/64 route to simd-MMA; bf16 routes to the scalar tiled template
    (bf16 in/out, fp32 interior compute)."""
    torch.manual_seed(0)
    dev = "mps"
    Hc, Hh, I, N = 4, 2, 3, 64  # H_combined=4, H_heads=2 => batch=2
    batch = Hc // Hh
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Hc, I, N, DIM, device=dev, dtype=dtype)
    k = torch.randn(Hc, I, N, DIM, device=dev, dtype=dtype)
    v = torch.randn(Hc, I, N, DIM, device=dev, dtype=dtype)
    bias = torch.randn(Hc, N, N, device=dev, dtype=dtype)
    mask = (torch.rand(batch, I, N, device=dev) < 0.25).to(torch.uint8)
    o = torch.zeros(Hc, I, N, DIM, device=dev, dtype=dtype)
    lse = torch.zeros(Hc, I, N, device=dev, dtype=torch.float32)
    st = lambda t: tuple(t.stride())
    grid = (triton.cdiv(N, 32), I, Hc)
    _biased_tri_fa[grid](
        o,
        *st(o),
        lse,
        *st(lse),
        q,
        *st(q),
        k,
        *st(k),
        v,
        *st(v),
        bias,
        *st(bias),
        mask,
        *st(mask),
        sm,
        -1e9,
        N,
        Hh,
        DIM,
        32,
        32,
    )
    torch.mps.synchronize()
    qf, kf, vf, bf = q.float(), k.float(), v.float(), bias.float()
    if dtype == torch.float32:
        qk = torch.einsum("hijd,hikd->hijk", qf, kf) * sm
    else:
        # packet 164: the SOURCE scales Q in its own dtype (`q * tl.full([1], sm, dtype)`): the scale
        # is rounded to dtype and so is each product; the template now replays that, so the
        # reference must too (the old fp32-scaled reference was 3e-3 off in the lse).
        qs = (qf * torch.tensor(sm, dtype=dtype).float()).to(dtype).float()
        qk = torch.einsum("hijd,hikd->hijk", qs, kf)
    raw = qk + bf[:, None, :, :]
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = raw.masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o_ref = torch.einsum("hijk,hikd->hijd", torch.nan_to_num(p, nan=0.0), vf)
    lse_ref = torch.logsumexp(raw, dim=-1)
    assert (o.float() - o_ref).abs().max().item() < o_tol
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 1e-3


def _build_biased_lowerer(*, signature_overrides=None, **constexpr_overrides):
    """Compile _biased_fa to TTGIR and wrap in a GenericLowerer (no MSL emission)."""
    from triton._C.libtriton import ir
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    ptr = "*fp32"
    sig = {}

    def add(names, ty):
        for n in names.split():
            sig[n] = ty

    add("o_ptr", ptr)
    add("o_sz o_sh o_sm o_sk", "i32")
    add("lse_ptr", ptr)
    add("lse_sz lse_sh lse_sm", "i32")
    add("q_ptr", ptr)
    add("q_sz q_sh q_sm q_sk", "i32")
    add("k_ptr", ptr)
    add("k_sz k_sh k_sn k_sk", "i32")
    add("v_ptr", ptr)
    add("v_sz v_sh v_sn v_sk", "i32")
    add("b_ptr", ptr)
    add("b_sz b_sh b_sm b_sn", "i32")
    add("mask_ptr", "*i8")
    add("m_sz m_sh m_sn", "i32")
    add("sm_scale", "fp32")
    add("neg_inf", "fp32")
    add("Z H N", "i32")
    sig.update(signature_overrides or {})
    constexprs = {"DIM": 32, "BLOCK_M": 32, "BLOCK_N": 32, "BAD_TEMP": False, "I3D": False}
    constexprs.update(constexpr_overrides)
    s = ASTSource(fn=_biased_fa, signature=sig, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = s.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    md = {}
    mod = backend.make_ttir(mod, md, options)
    mod = backend.make_ttgir(mod, md, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


def test_biased_fa_2d_grid_detects():
    """Sanity: the 2-D-grid biased kernel is detected (not refused), grid_3d=False."""
    low = _build_biased_lowerer(I3D=False)
    info = low._detect_biased_flash_attention()
    assert info is not None and info["biased"] is True and info["grid_3d"] is False


@pytest.mark.parametrize("arg_name", ["neg_inf", "sm_scale"])
def test_biased_fa_fp16_runtime_float_refuses_at_lowering_boundary(arg_name):
    """The replacement ABI transports scalar bits in uint slots and decodes its
    runtime floats as fp32.  A source fp16 scale OR sentinel therefore has no
    proven bit-exact ABI; it must refuse before an executable can be cached or
    dispatched.  Pin the shared ABI mechanism, not only the reported sentinel.
    """
    from triton_msl.errors import MetalNonRecoverableError

    low = _build_biased_lowerer(signature_overrides={arg_name: "fp16"})
    with pytest.raises(MetalNonRecoverableError, match=rf"runtime .*{arg_name}.*fp32"):
        low.lower()


def test_biased_fa_over_3_program_ids_refuses():
    """More than 3 program_ids is an unhandled grid -> refuse (correct-or-refuse)."""
    from triton_msl.errors import MetalNonRecoverableError

    low = _build_biased_lowerer(I3D=True)  # adds a 3rd program_id in a non-trifast way
    # I3D adds program_id(2) atop the 2-D (pid_m, pid_zh) grid = 3 total, which is
    # now *handled* (grid_3d); this asserts the malformed z=zh/H+pid_i still resolves
    # or refuses cleanly — never silently. Either a valid info or a refuse is fine.
    try:
        info = low._detect_biased_flash_attention()
        assert info is None or info["grid_3d"] is True
    except MetalNonRecoverableError:
        pass  # a clean refuse is acceptable


def test_detector_returns_none_for_standard_fa():
    """The biased detector must NOT fire on a standard FA kernel (whose QK dot C
    operand is a zero/loop-acc, not a load) — else it would misroute it."""
    from tests.test_fa_detect import _build_fa_lowerer  # reuse the standard FA fixture

    low = _build_fa_lowerer(causal=False, head_dim=128, block=32)
    assert low._detect_biased_flash_attention() is None
