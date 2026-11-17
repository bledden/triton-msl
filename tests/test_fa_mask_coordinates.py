# _biased_mask adapts the TriFast-derived biased forward test kernel.
# Copyright (c) 2025 Liam Atkinson; MIT License (third_party/trifast/LICENSE).
# Other kernels in this file are separate tests; see docs/TRIFAST_PROVENANCE.md.
"""Packet 111/112: a replacement FA template's load/store masks must be EXACTLY the
coordinates the template clips with — role by role, numerically — on every route.

Packet 110's rule compared only the bound (N_CTX) and the program-id axis; it recorded the
range bounds, the program-id coefficient and the loop-IV participation but never compared
them, and it used one grammar for Q, K and V. GPT's 111 rows on the packet-110 candidate
(all fresh caches, one SIMD dispatch each): Q coefficient 2*BM err 0.94, K masked by the
QUERY coordinate err 0.73, V likewise err 1.25, Q range(BM, 2BM) err 0.91. The varlen route
had no rule at all: Q value mask err 0.69, K value mask err 1.68. The biased route (trifast)
reconstructs Q/K/V, Bias, Mask, Out and Lse and checked none of their load masks.

Now (``_fa_verify_value_paths``): BM / BN come from the VERIFIED dot operand shapes; a
row-coordinate mask (Q, Q-rope, Out, Lse) must be exactly ``program_id(0)*BM +
make_range(0,BM) < bound``; a column-coordinate mask (K, K-rope, V, Mask) exactly
``<K-loop iv> + make_range(0,BN) < bound``; every ``andi`` leaf independently; the bound is
the N_CTX argument (dense, biased) or the detector's verified ``seqlen_q`` / ``seqlen_k``
value (varlen); ``other`` absent (this compiler's generic path substitutes zero — pinned
below) or the literal zero; the varlen Out store must carry exactly the seqlen_q boundary.
"""

import math
import sys

import pytest

try:
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl

    import triton_msl.autotuning._fa_dispatch as fa_dispatch
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@pytest.fixture
def fa_spy(monkeypatch):
    """Spy on the replacement FA dispatch (dense simd, varlen and biased all route through it)."""
    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(*a, **k):
        r = real(*a, **k)
        hits.append(bool(r))
        return r

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


def _st(t):
    return [t.stride(i) for i in range(t.dim())]


if HAS:
    # ------------------------------------------------------------------ dense simd
    @triton.jit
    def _fa_mask_coordinate(
        Q,
        K,
        V,
        Out,
        qz,
        qh,
        qm,
        qk_,
        kz,
        kh,
        kn,
        kk,
        vz,
        vh,
        vn,
        vk,
        oz,
        oh,
        om,
        ok_,
        Z,
        H,
        N_CTX,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """GPT's 111 harness (modes 0-4) + two positives: 5 = boundary mask with NO ``other``,
        6 = K and V masked at their own column boundary."""
        pid = tl.program_id(0)
        zh = tl.program_id(1)
        z = zh // H
        h = zh % H
        rm = pid * BM + tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rd = tl.arange(0, D)
        qptr = Q + z * qz + h * qh + rm[:, None] * qm + rd[None, :] * qk_
        if MODE == 1:
            qmask = (pid * (2 * BM) + tl.arange(0, BM))[:, None] < N_CTX  # wrong coefficient
        elif MODE == 4:
            qmask = (pid * BM + tl.arange(BM, 2 * BM))[:, None] < N_CTX  # wrong range
        else:
            qmask = rm[:, None] < N_CTX
        if MODE == 5:
            q = tl.load(qptr, mask=qmask) * (1.0 / tl.sqrt(float(D)))  # no ``other``
        else:
            q = tl.load(qptr, mask=qmask, other=0.0) * (1.0 / tl.sqrt(float(D)))
        mi = tl.full([BM], float("-inf"), tl.float32)
        li = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        for start in range(0, N_CTX, BN):
            kptr = K + z * kz + h * kh + (start + rn)[:, None] * kn + rd[None, :] * kk
            vptr = V + z * vz + h * vh + (start + rn)[:, None] * vn + rd[None, :] * vk
            if MODE == 2:
                k = tl.load(kptr, mask=(pid * (2 * BM) + tl.arange(0, BN))[:, None] < N_CTX, other=0.0)
            elif MODE == 6:
                k = tl.load(kptr, mask=(start + rn)[:, None] < N_CTX, other=0.0)
            else:
                k = tl.load(kptr)
            if MODE == 3:
                v = tl.load(vptr, mask=(pid * (2 * BM) + tl.arange(0, BN))[:, None] < N_CTX, other=0.0)
            elif MODE == 6:
                v = tl.load(vptr, mask=(start + rn)[:, None] < N_CTX, other=0.0)
            else:
                v = tl.load(vptr)
            qk = tl.dot(q, tl.trans(k).to(q.dtype))
            mij = tl.max(qk, 1)
            mnew = tl.maximum(mi, mij)
            alpha = tl.exp(mi - mnew)
            p = tl.exp(qk - mnew[:, None])
            li = li * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            mi = mnew
        acc = acc / li[:, None]
        optr = Out + z * oz + h * oh + rm[:, None] * om + rd[None, :] * ok_
        tl.store(optr, acc.to(Out.dtype.element_ty))

    # ------------------------------------------------------------------ varlen
    @triton.jit
    def _varlen_mask(
        Q,
        K,
        V,
        Out,
        CUQ,
        CUK,
        sqt,
        sqh,
        sqd,
        skt,
        skh,
        skd,
        svt,
        svh,
        svd,
        sot,
        soh,
        sod,
        H,
        MAXS,
        SCALE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """GPT's 111 varlen harness (modes 0-2) + V value mask (3), Out store value mask (4),
        Out store UNMASKED (5), Q mask bound = seqlen_k (6)."""
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        qs = tl.load(CUQ + b)
        qlen = tl.load(CUQ + b + 1) - qs
        ks = tl.load(CUK + b)
        klen = tl.load(CUK + b + 1) - ks
        rm = pid * BM + tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rd = tl.arange(0, D)
        qptr = Q + (qs + rm)[:, None] * sqt + h * sqh + rd[None, :] * sqd
        if MODE == 1:
            qmask = rm[:, None] < 16
        elif MODE == 6:
            qmask = rm[:, None] < klen
        else:
            qmask = rm[:, None] < qlen
        q = tl.load(qptr, mask=qmask, other=0.0) * SCALE
        mi = tl.full([BM], float("-inf"), tl.float32)
        li = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        for start in range(0, MAXS, BN):
            kn = start + rn
            kptr = K + (ks + kn)[:, None] * skt + h * skh + rd[None, :] * skd
            kmask = (kn[:, None] < 16) if MODE == 2 else (kn[:, None] < klen)
            k = tl.load(kptr, mask=kmask, other=0.0)
            qk = tl.dot(q, tl.trans(k).to(q.dtype))
            qk = tl.where(kn[None, :] < klen, qk, float("-inf"))
            mij = tl.max(qk, 1)
            mnew = tl.maximum(mi, mij)
            alpha = tl.exp(mi - mnew)
            p = tl.exp(qk - mnew[:, None])
            li = li * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            vptr = V + (ks + kn)[:, None] * svt + h * svh + rd[None, :] * svd
            vmask = (kn[:, None] < 16) if MODE == 3 else (kn[:, None] < klen)
            v = tl.load(vptr, mask=vmask, other=0.0)
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
            mi = mnew
        acc = acc / li[:, None]
        optr = Out + (qs + rm)[:, None] * sot + h * soh + rd[None, :] * sod
        if MODE == 4:
            tl.store(optr, acc.to(Out.dtype.element_ty), mask=rm[:, None] < 16)
        elif MODE == 5:
            tl.store(optr, acc.to(Out.dtype.element_ty))
        else:
            tl.store(optr, acc.to(Out.dtype.element_ty), mask=rm[:, None] < qlen)

    # ------------------------------------------------------------------ biased (trifast)
    @triton.jit
    def _biased_mask(
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
        MODE: tl.constexpr,
    ):
        """tests/test_fa_biased_routing.py::_biased_fa (trifast's forward spelling) with ONE
        memory boundary varied per MODE: 1 Q value mask, 2 K value mask, 3 V value mask,
        4 Bias value mask (col < 16), 5 Bias row-only mask, 6 Mask-load value mask,
        7 Lse store value mask, 8 Out store value mask, 9 / 10 Out store clipped on the next
        row / twice the row."""
        inv_ln2: tl.constexpr = 1.4426950408889634
        ln2: tl.constexpr = 0.6931471824645996
        pid_m = tl.program_id(0)
        pid_zh = tl.program_id(1)
        z = pid_zh // H
        h = pid_zh % H
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
        if MODE == 1:
            q_block = tl.load(q_ptrs, (m_idxs < 16)[:, None])
        else:
            q_block = tl.load(q_ptrs, mask_m[:, None])
        q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

        for start_n in tl.range(0, N, BLOCK_N):
            mask_n = (n_idxs + start_n) < N
            val_n = (n_idxs + start_n) < 16
            if MODE == 2:
                kt_block = tl.load(kt_ptrs, val_n[None, :])
            else:
                kt_block = tl.load(kt_ptrs, mask_n[None, :])
            if MODE == 4:
                b_block = tl.load(b_ptrs, mask_m[:, None] & val_n[None, :])
            elif MODE == 5:
                b_block = tl.load(b_ptrs, mask_m[:, None])
            else:
                b_block = tl.load(b_ptrs, mask_m[:, None] & mask_n[None, :])
            if MODE == 6:
                m_block = tl.load(mask_ptrs, val_n)
            else:
                m_block = tl.load(mask_ptrs, mask_n)
            scores = b_block.to(tl.float32)
            scores = tl.dot(q_block, kt_block, scores)
            scores *= inv_ln2
            scores = tl.where(m_block[None, :], neg_inf, scores)
            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, neg_inf)
            block_max = tl.maximum(scores_max, tl.max(scores, 1))
            scores = scores - block_max[:, None]
            exp_scores = tl.math.exp2(scores)
            summed = tl.sum(exp_scores, 1)
            exp_scale = tl.math.exp2(scores_max - block_max)
            sm_denom = sm_denom * exp_scale + summed
            acc = acc * exp_scale[:, None]
            if MODE == 3:
                v_block = tl.load(v_ptrs, val_n[:, None])
            else:
                v_block = tl.load(v_ptrs, mask_n[:, None])
            exp_scores = exp_scores.to(q_block.type.element_ty)
            acc = tl.dot(exp_scores, v_block, acc)
            scores_max = block_max
            kt_ptrs += BLOCK_N * k_sn
            v_ptrs += BLOCK_N * v_sn
            b_ptrs += BLOCK_N * b_sn
            mask_ptrs += BLOCK_N * m_sn

        normalize = acc / sm_denom[:, None]
        if MODE == 8:
            tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=(m_idxs < 16)[:, None])
        elif MODE == 9:
            tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=(m_idxs + 1)[:, None] < N)
        elif MODE == 10:
            tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=(m_idxs * 2)[:, None] < N)
        else:
            tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])
        lse = (scores_max * ln2) + tl.log(sm_denom)
        if MODE == 7:
            tl.store(lse_ptrs, lse, mask=m_idxs < 16)
        else:
            tl.store(lse_ptrs, lse, mask=mask_m)

    # ------------------------------------------------------------------ generic default
    @triton.jit
    def _masked_load_no_other(X, Y, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs, mask=offs < 8)
        tl.store(Y + offs, x)


# =========================================================================== dense simd


def _dense_oracle(q, k, v, mode):
    """The SOURCE kernel's semantics: block 1 (rows 32-63) sees zeroed Q / K / V under the
    wrong-coordinate masks (GPT's 111 oracle)."""
    out = torch.empty_like(q, dtype=torch.float32)
    for block in range(2):
        qb = q[..., block * 32 : (block + 1) * 32, :].float().clone()
        kb, vb = k.float().clone(), v.float().clone()
        if mode in (1, 4) and block == 1:
            qb.zero_()
        if mode == 2 and block == 1:
            kb.zero_()
        if mode == 3 and block == 1:
            vb.zero_()
        out[..., block * 32 : (block + 1) * 32, :] = F.scaled_dot_product_attention(qb, kb, vb)
    return out


def _run_dense(mode):
    if hasattr(_fa_mask_coordinate, "device_caches"):
        _fa_mask_coordinate.device_caches.clear()
    z, h, n, d = 1, 2, 64, 128
    torch.manual_seed(1110 + mode)
    q, k, v = (torch.randn(z, h, n, d, device=D, dtype=torch.float16) for _ in range(3))
    out = torch.full_like(q, float("nan"))
    try:
        _fa_mask_coordinate[(2, z * h)](q, k, v, out, *_st(q), *_st(k), *_st(v), *_st(out), z, h, n, 32, 32, d, mode)
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    return state, q, k, v, out


@requires_gpu
@pytest.mark.parametrize(
    "mode,label", [(0, "canonical row boundary"), (5, "row boundary, no other"), (6, "K/V column boundary")]
)
def test_dense_exact_boundary_masks_route_and_match(cold_gpu_caches, fa_spy, mode, label):
    state, q, k, v, out = _run_dense(mode)
    assert state == "computed", label
    assert True in fa_spy, f"{label}: the simd replacement must be the path taken"
    assert (out.float() - _dense_oracle(q, k, v, 0)).abs().max().item() < 5e-2, label


@requires_gpu
@pytest.mark.parametrize(
    "mode,label",
    [
        (1, "Q mask program-id coefficient 2*BM"),
        (2, "K mask uses the QUERY coordinate"),
        (3, "V mask uses the QUERY coordinate"),
        (4, "Q mask make_range(BM, 2BM)"),
    ],
)
def test_dense_wrong_coordinate_masks_refuse_or_match(cold_gpu_caches, fa_spy, mode, label):
    """Bites on the packet-110 candidate and on e2330ec: one SIMD dispatch each, err 0.94 /
    0.73 / 1.25 / 0.91 against the source (111 §2). Now: refused before any dispatch."""
    state, q, k, v, out = _run_dense(mode)
    if state == "computed":
        assert (out.float() - _dense_oracle(q, k, v, mode)).abs().max().item() < 5e-2, label
    else:
        assert True not in fa_spy, f"{label}: refusal must happen before any dispatch"


# =========================================================================== varlen


def _varlen_oracle(q, k, v, cuq, cuk, mode, sentinel):
    """Per head: the SOURCE kernel's written rows and their values (tokens packed on dim 0)."""
    n, h, d = q.shape
    qlen, klen = int(cuq[1]), int(cuk[1])
    out = torch.full_like(q, sentinel)
    for hh in range(h):
        qb, kb, vb = q[:, hh].float().clone(), k[:, hh].float().clone(), v[:, hh].float().clone()
        q_valid = 16 if mode == 1 else (klen if mode == 6 else qlen)
        qb[q_valid:].zero_()
        kb[(16 if mode == 2 else klen) :].zero_()
        vb[(16 if mode == 3 else klen) :].zero_()
        s = (qb @ kb.transpose(0, 1)) / (d**0.5)
        s[:, klen:] = float("-inf")
        o = torch.softmax(s, dim=-1) @ vb
        rows = 16 if mode == 4 else (n if mode == 5 else qlen)
        out[:rows, hh] = o[:rows].to(out.dtype)
    return out


def _run_varlen(mode, qlen, klen):
    if hasattr(_varlen_mask, "device_caches"):
        _varlen_mask.device_caches.clear()
    n, h, d = 32, 2, 32
    torch.manual_seed(1120 + mode)
    q, k, v = (torch.randn(n, h, d, device=D, dtype=torch.float32) for _ in range(3))
    out = torch.full_like(q, float("nan"))
    cuq = torch.tensor([0, qlen], device=D, dtype=torch.int32)
    cuk = torch.tensor([0, klen], device=D, dtype=torch.int32)
    try:
        _varlen_mask[(1, h)](
            q, k, v, out, cuq, cuk, *_st(q), *_st(k), *_st(v), *_st(out), h, n, 1.0 / (d**0.5), 32, 32, d, mode
        )
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    return state, q, k, v, out, cuq, cuk


def _assert_varlen_matches(out, exp, label):
    fin = ~torch.isnan(exp)
    assert torch.equal(torch.isnan(out), ~fin), f"{label}: written-row set differs from the source"
    assert (out[fin] - exp[fin]).abs().max().item() < 1e-3, label


@requires_gpu
@pytest.mark.parametrize("qlen,klen", [(32, 32), (32, 20)])
def test_varlen_canonical_seqlen_masks_route_and_match(cold_gpu_caches, fa_spy, qlen, klen):
    """The canonical masks on the detector's seqlen_q / seqlen_k values route and match —
    including UNEQUAL lengths, which proves the bound is keyed per role."""
    state, q, k, v, out, cuq, cuk = _run_varlen(0, qlen, klen)
    assert state == "computed"
    assert True in fa_spy, "the varlen replacement must be the path taken"
    _assert_varlen_matches(out, _varlen_oracle(q, k, v, cuq, cuk, 0, float("nan")), "varlen canonical")


@requires_gpu
@pytest.mark.parametrize(
    "mode,qlen,klen,label",
    [
        (1, 32, 32, "Q load value mask rows < 16"),
        (2, 32, 32, "K load value mask cols < 16"),
        (3, 32, 32, "V load value mask cols < 16"),
        (4, 32, 32, "Out store value mask rows < 16"),
        (5, 20, 20, "Out store unmasked (seqlen_q < BM)"),
        (6, 32, 20, "Q load mask bound is seqlen_k, not seqlen_q"),
    ],
)
def test_varlen_unreplayed_masks_refuse_or_match(cold_gpu_caches, fa_spy, mode, qlen, klen, label):
    """Bites on the packet-110 candidate and e2330ec (111 §3: Q err 0.69, K err 1.68, one
    varlen dispatch each). Now: refused before any dispatch."""
    state, q, k, v, out, cuq, cuk = _run_varlen(mode, qlen, klen)
    if state == "computed":
        _assert_varlen_matches(out, _varlen_oracle(q, k, v, cuq, cuk, mode, float("nan")), label)
    else:
        assert True not in fa_spy, f"{label}: refusal must happen before any dispatch"


# =========================================================================== biased (trifast)


def _biased_oracle(q, k, v, b, mask, sm, mode, out_sentinel, lse_sentinel):
    """The SOURCE kernel's semantics with N = 64 (two full K tiles): every value mask zeroes
    the affected rows/cols of the loaded operand; a value-masked Mask load reads 0 (not
    masked) beyond col 16; value-masked stores leave the sentinel."""
    q, k, v, b = (t.float().clone() for t in (q, k, v, b))
    m = mask.bool().clone()
    if mode == 1:
        q[..., 16:, :] = 0
    if mode == 2:
        k[..., 16:, :] = 0
    if mode == 3:
        v[..., 16:, :] = 0
    if mode == 4:
        b[..., :, 16:] = 0
    if mode == 6:
        m[..., 16:] = False
    raw = sm * (q @ k.transpose(-2, -1)) + b
    raw = raw.masked_fill(m[:, :, None, :], float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.nan_to_num(p, nan=0.0) @ v
    lse = torch.logsumexp(raw, dim=-1)
    if mode in (8, 9, 10):
        o = o.clone()
        o[..., {8: 16, 9: 63, 10: 32}[mode] :, :] = out_sentinel
    if mode == 7:
        lse = lse.clone()
        lse[..., 16:] = lse_sentinel
    return o, lse


def _run_biased(mode):
    if hasattr(_biased_mask, "device_caches"):
        _biased_mask.device_caches.clear()
    Z, H, N, DIM = 1, 2, 64, 64
    torch.manual_seed(1130 + mode)
    sm = 1.0 / math.sqrt(DIM)
    q, k, v = (torch.randn(Z, H, N, DIM, device=D) for _ in range(3))
    b = torch.randn(Z, H, N, N, device=D)
    mask = (torch.rand(Z, H, N, device=D) < 0.25).to(torch.uint8)
    o = torch.full((Z, H, N, DIM), 77.0, device=D)
    lse = torch.full((Z, H, N), 77.0, device=D)
    st = lambda t: tuple(t.stride())
    try:
        _biased_mask[(N // 32, Z * H)](
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
            mode,
        )
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    return state, (q, k, v, b, mask, sm), o, lse


def _assert_biased_matches(o, lse, o_ref, lse_ref, label):
    assert (o - o_ref).abs().max().item() < 2e-3, label
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 2e-3, label
    if (~fin).any():
        assert (lse[~fin] < -1e30).all(), label


@requires_gpu
def test_biased_canonical_boundaries_route_and_match(cold_gpu_caches, fa_spy):
    """trifast's spelling: Q/K/V at their boundaries, Bias at row AND col, Mask at col, Out
    and Lse at row — all with NO ``other`` (accepted as the generic path's zero)."""
    state, (q, k, v, b, mask, sm), o, lse = _run_biased(0)
    assert state == "computed"
    assert True in fa_spy, "the biased replacement must be the path taken"
    o_ref, lse_ref = _biased_oracle(q, k, v, b, mask, sm, 0, 77.0, 77.0)
    _assert_biased_matches(o, lse, o_ref, lse_ref, "biased canonical")


@requires_gpu
@pytest.mark.parametrize(
    "mode,label",
    [
        (1, "Q load value mask rows < 16"),
        (2, "K load value mask cols < 16"),
        (3, "V load value mask cols < 16"),
        (4, "Bias load value mask cols < 16"),
        (5, "Bias load row-only mask (conservative refusal; equivalent at N = 64)"),
        (6, "Mask load value mask cols < 16"),
        (7, "Lse store value mask rows < 16"),
        (8, "Out store value mask rows < 16 (existing output-mask guard)"),
        (9, "Out store clipped on the next row: row 63 keeps the sentinel"),
        (10, "Out store clipped on twice the row: rows 32-63 keep the sentinel"),
    ],
)
def test_biased_unreplayed_boundaries_refuse_or_match(cold_gpu_caches, fa_spy, mode, label):
    """Bites on e2330ec for modes 1-4, 6, 7 (the biased route checked none of its load
    masks nor the Lse store mask) and on 8f2c58f for modes 9, 10 (the Out guard read any
    index derived from the row as the row). Now: refused before any dispatch."""
    state, (q, k, v, b, mask, sm), o, lse = _run_biased(mode)
    if state == "computed":
        o_ref, lse_ref = _biased_oracle(q, k, v, b, mask, sm, mode, 77.0, 77.0)
        _assert_biased_matches(o, lse, o_ref, lse_ref, label)
    else:
        assert True not in fa_spy, f"{label}: refusal must happen before any dispatch"


# =========================================================================== generic default


@requires_gpu
def test_generic_masked_load_without_other_reads_zero(cold_gpu_caches):
    """The invariant the mask proof relies on: this compiler's generic path substitutes ZERO
    for a masked-out lane when ``other`` is absent (``_lower_load``), which is exactly what
    the replacement templates supply beyond their boundaries."""
    if hasattr(_masked_load_no_other, "device_caches"):
        _masked_load_no_other.device_caches.clear()
    x = torch.randn(16, device=D) + 3.0  # nowhere near zero
    y = torch.full((16,), float("nan"), device=D)
    _masked_load_no_other[(1,)](x, y, 16)
    torch.mps.synchronize()
    assert torch.equal(y[:8], x[:8])
    assert (y[8:] == 0).all(), y[8:]
