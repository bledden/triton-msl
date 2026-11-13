# TriFast reporter kernels; upstream MIT license: third_party/trifast/LICENSE.
# Preserved candidate adaptation: derive contiguous strides; transpose loaded K.
import triton
import triton.language as tl
@triton.jit
def _fwd(
    o_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    mask_ptr,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    PADDED: tl.constexpr = False,
):
    # Strides are derived here rather than passed: Metal exposes only 31 buffer
    # slots and these are ~20 arguments. Every tensor is contiguous, so they follow
    # from N and DIM -- which also makes the compiled kernel independent of N.
    _s4h, _s4m, _s4n = N * N * DIM, N * DIM, DIM
    _s3h, _s3m = N * N, N
    stride_oh = _s4h
    stride_om = _s4m
    stride_on = _s4n
    stride_od = 1
    stride_lseh = _s3h
    stride_lsem = _s3m
    stride_lsen = 1
    stride_qh = _s4h
    stride_qm = _s4m
    stride_qn = _s4n
    stride_qd = 1
    stride_kh = _s4h
    stride_km = _s4m
    stride_kn = _s4n
    stride_kd = 1
    stride_vh = _s4h
    stride_vm = _s4m
    stride_vn = _s4n
    stride_vd = 1
    stride_bh = _s3h
    stride_bm = _s3m
    stride_bn = 1
    stride_maskh = _s3h
    stride_maskm = _s3m
    stride_maskn = 1
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_h = tl.program_id(2)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2)

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0 # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    # kt/b/v/mask pointers are built inside the k-loop (see the note there).
    base_kt_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    base_b_ptr = b_ptr + (start_h * stride_bh)
    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)

    base_lse_ptr = lse_ptr + (start_h * stride_lseh) + (start_i * stride_lsem)
    lse_ptrs = base_lse_ptr + (j_idxs * stride_lsen) # [j]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
    # `inv_ln2` is folded into q (and into b below) rather than applied to the dot RESULT.
    # Algebraically identical — inv_ln2 * (q @ kᵀ * sm_scale + b) == (q * sm_scale * inv_ln2) @ kᵀ
    # + b * inv_ln2 — but it keeps the scale off the accumulator, which the Metal backend
    # (triton-msl) refuses because its attention lowering silently drops a scale on a dot result.
    q_block = q_block * tl.full([1], value=sm_scale * inv_ln2, dtype=q_block.type.element_ty)

    for start_k in tl.range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        kk_idxs = k_idxs + start_k
        mask_k = kk_idxs < N

        # Pointers are rebuilt from their bases each iteration rather than advanced across the
        # loop. Identical arithmetic, but it keeps pointer *tensors* out of the loop-carried
        # state: the Metal backend emits a loop-carried pointer tensor as a per-thread scalar and
        # then subscripts it, which does not compile.
        kt_ptrs = base_kt_ptr + (d_idxs[:, None] * stride_kd) + (kk_idxs[None, :] * stride_kn)
        b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (kk_idxs[None, :] * stride_bn)
        mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (kk_idxs * stride_maskn)
        v_ptrs = base_v_ptr + (kk_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd)

        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
        b_block = tl.load(b_ptrs,  mask_j[:, None] & mask_k[None, :])  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") # [k]

        scores = b_block.to(tl.float32) * inv_ln2
        scores = tl.dot(q_block, kt_block, scores, input_precision="ieee")  # [j,k]

        # we want to make scores -inf at mask locations
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        scores = scores - block_max[:, None]  # [j,k]
        exp_scores = tl.math.exp2(scores)  # [j,k]

        summed_exp_scores = tl.sum(exp_scores, 1)  # [j]
        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + summed_exp_scores  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        scores_max = block_max


    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)

    lse = (scores_max * ln2) + tl.log(sm_denom)

    if PADDED:
        # Buffers are padded to whole tiles, so the boundary masks are unnecessary. The Metal
        # backend needs both of these: it refuses a matmul store carrying a tighter-than-tile
        # mask, and — worse — it MIS-COMPILES a 1-D store of a row-reduced value, writing row 0
        # of the tile to every row. A [j,1] 2-D store uses the correct row map at no extra cost.
        tl.store(o_ptrs, final_output)
        tl.store(lse_ptrs[:, None], lse[:, None])
    else:
        tl.store(o_ptrs, final_output, mask=mask_j[:, None])
        tl.store(lse_ptrs, lse, mask=mask_j)

@triton.jit
def _bwd_kv(
    d_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    l_ptr,
    m_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    PADDED: tl.constexpr = False,
):
    # Strides are derived here rather than passed: Metal exposes only 31 buffer
    # slots and these are ~20 arguments. Every tensor is contiguous, so they follow
    # from N and DIM -- which also makes the compiled kernel independent of N.
    _s4h, _s4m, _s4n = N * N * DIM, N * DIM, DIM
    _s3h, _s3m = N * N, N
    stride_dh = _s3h
    stride_dm = _s3m
    stride_dn = 1
    stride_qh = _s4h
    stride_qm = _s4m
    stride_qn = _s4n
    stride_qd = 1
    stride_kh = _s4h
    stride_km = _s4m
    stride_kn = _s4n
    stride_kd = 1
    stride_vh = _s4h
    stride_vm = _s4m
    stride_vn = _s4n
    stride_vd = 1
    stride_bh = _s3h
    stride_bm = _s3m
    stride_bn = 1
    stride_lh = _s3h
    stride_lm = _s3m
    stride_ln = 1
    stride_mh = _s3h
    stride_mm = _s3m
    stride_mn = 1
    stride_doh = _s4h
    stride_dom = _s4m
    stride_don = _s4n
    stride_dod = 1
    stride_dkh = _s4h
    stride_dkm = _s4m
    stride_dkn = _s4n
    stride_dkd = 1
    stride_dvh = _s4h
    stride_dvm = _s4m
    stride_dvn = _s4n
    stride_dvd = 1
    input_dtype = q_ptr.dtype.element_ty

    # program id
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = 0 # we iterate over j, so each pid starts at 0
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    # q/b/l/do/d pointers are built inside the j-loop (see the note there); the rest are loop
    # invariant.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    base_b_ptr = b_ptr + (start_h * stride_bh)
    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)   # [d,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None,:] * stride_vn)  # [d,k]

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs= base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn) # [k]

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd) # [k,d]

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd) # [k,d]

    mask_k = k_idxs < N

    # load k/v once per pid
    vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
    kt_block = tl.load(kt_ptrs, mask_k[None, :]) # [d,k]
    kt_block = kt_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [k,d]
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") # [k]

    # accumulate over j for dk/dv
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        jj_idxs = j_idxs + start_j
        mask_j = jj_idxs < N

        # Rebuilt from bases rather than advanced — see the note in _fwd.
        q_ptrs = base_q_ptr + (jj_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)
        b_ptrs = base_b_ptr + (jj_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)
        l_ptrs = base_l_ptr + (jj_idxs * stride_ln)
        do_ptrs = base_do_ptr + (jj_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)
        d_ptrs = base_d_ptr + (jj_idxs * stride_dn)

        q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32) # [j,k]

        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee") # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores= tl.where(m_block[None, :], neg_inf, scores)

        row_max = tl.load(l_ptrs, mask=mask_j) # [j]
        sm_value = tl.math.exp2((scores  - row_max[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
        dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee") # [k,d]

        delta = tl.load(d_ptrs, mask_j) # [j]

        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        dscores = dscores.to(input_dtype) # [j,k]

        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee") # [k,d]


    dk_block *= sm_scale
    if PADDED:
        tl.store(dk_ptrs, dk_block.to(input_dtype))
        tl.store(dv_ptrs, dv_block.to(input_dtype))
    else:
        tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None])
        tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])

@triton.jit
def _bwd_q(
    d_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    l_ptr,
    mask_ptr,
    o_ptr,
    do_ptr,
    dq_ptr,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,  BLOCK_K: tl.constexpr,
    PADDED: tl.constexpr = False,
):
    # Strides are derived here rather than passed: Metal exposes only 31 buffer
    # slots and these are ~20 arguments. Every tensor is contiguous, so they follow
    # from N and DIM -- which also makes the compiled kernel independent of N.
    _s4h, _s4m, _s4n = N * N * DIM, N * DIM, DIM
    _s3h, _s3m = N * N, N
    stride_dh = _s3h
    stride_dm = _s3m
    stride_dn = 1
    stride_qh = _s4h
    stride_qm = _s4m
    stride_qn = _s4n
    stride_qd = 1
    stride_kh = _s4h
    stride_km = _s4m
    stride_kn = _s4n
    stride_kd = 1
    stride_vh = _s4h
    stride_vm = _s4m
    stride_vn = _s4n
    stride_vd = 1
    stride_bh = _s3h
    stride_bm = _s3m
    stride_bn = 1
    stride_lh = _s3h
    stride_lm = _s3m
    stride_ln = 1
    stride_maskh = _s3h
    stride_maskm = _s3m
    stride_maskn = 1
    stride_oh = _s4h
    stride_om = _s4m
    stride_on = _s4n
    stride_od = 1
    stride_doh = _s4h
    stride_dom = _s4m
    stride_don = _s4n
    stride_dod = 1
    stride_dqh = _s4h
    stride_dqm = _s4m
    stride_dqn = _s4n
    stride_dqd = 1
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0 # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    # k/b/v/mask pointers are built inside the k-loop (see the note in _fwd).
    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    base_b_ptr = b_ptr + (start_h * stride_bh)
    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    base_dq_ptr = dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
    dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs[None, :] * stride_dqd) # [j,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
    sm_denom = tl.load(l_ptrs, mask_j) # [j]
    do_block = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
    o_block = tl.load(o_ptrs, mask_j[:, None]) # [j,d]

    delta = tl.sum(o_block * do_block, axis=1) # [j]

    if PADDED:
        # 2-D [j,1] store — see the note in _fwd; a 1-D store of this row-reduced value is
        # silently mis-compiled by the Metal backend.
        tl.store(d_ptrs[:, None], delta.to(input_dtype)[:, None])
    else:
        tl.store(d_ptrs, delta.to(input_dtype), mask=mask_j)

    dq_block = tl.zeros([BLOCK_J, DIM], dtype=tl.float32)

    # iterte over k for dq = \sum_{k} ds_{jk} k_{k}
    for start_k in range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        kk_idxs = k_idxs + start_k
        mask_k = kk_idxs < N

        # Rebuilt from bases rather than advanced — see the note in _fwd.
        k_ptrs = base_k_ptr + (kk_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd)
        b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (kk_idxs[None, :] * stride_bn)
        vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (kk_idxs[None, :] * stride_vn)
        mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (kk_idxs * stride_maskn)

        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") # [k]
        k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
        k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [j,d]

        # The proved backward template handles the shared K value in both
        # orientations; remove the historical generic-lowering workaround.
        kt_block = tl.trans(k_block)

        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee") # [j,k]
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        sm_value = tl.math.exp2((scores  - sm_denom[:, None]) * inv_ln2) # [j,k]

        vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
        dsm_value = tl.dot(do_block, vt_block, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        dscores = dscores.to(input_dtype) # [j,k]

        dq_block += tl.dot(dscores, k_block, input_precision="ieee")

    if PADDED:
        tl.store(dq_ptrs, dq_block.to(input_dtype))
    else:
        tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])

@triton.jit
def _bwd_b(
    d_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    l_ptr,
    m_ptr,
    do_ptr,
    db_ptr,
    sm_scale,
    neg_inf,
    H, N, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    PADDED: tl.constexpr = False,
):
    # Strides are derived here rather than passed: Metal exposes only 31 buffer
    # slots and these are ~20 arguments. Every tensor is contiguous, so they follow
    # from N and DIM -- which also makes the compiled kernel independent of N.
    _s4h, _s4m, _s4n = N * N * DIM, N * DIM, DIM
    _s3h, _s3m = N * N, N
    stride_dh = _s3h
    stride_dm = _s3m
    stride_dn = 1
    stride_qh = _s4h
    stride_qm = _s4m
    stride_qn = _s4n
    stride_qd = 1
    stride_kh = _s4h
    stride_km = _s4m
    stride_kn = _s4n
    stride_kd = 1
    stride_vh = _s4h
    stride_vm = _s4m
    stride_vn = _s4n
    stride_vd = 1
    stride_bh = _s3h
    stride_bm = _s3m
    stride_bn = 1
    stride_lh = _s3h
    stride_lm = _s3m
    stride_ln = 1
    stride_mh = _s3h
    stride_mm = _s3m
    stride_mn = 1
    stride_doh = _s4h
    stride_dom = _s4m
    stride_don = _s4n
    stride_dod = 1
    stride_dbh = _s3h
    stride_dbm = _s3m
    stride_dbn = 1
    input_dtype = q_ptr.dtype.element_ty
    BLOCK_I: tl.constexpr = 1

    # program id
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = 0 # we iterate over j, so each pid starts at 0
    start_j = pid_j * BLOCK_J
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    # Everything the i-loop touches is built inside it (see the note in _fwd); only the db output
    # pointer is loop invariant.
    base_q_ptr = q_ptr + (start_h * stride_qh)
    base_k_ptr = k_ptr + (start_h * stride_kh)
    base_b_ptr = b_ptr + (start_h * stride_bh)
    base_v_ptr = v_ptr + (start_h * stride_vh)
    base_l_ptr = l_ptr + (start_h * stride_lh)
    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    base_do_ptr = do_ptr + (start_h * stride_doh)
    base_d_ptr = d_ptr + (start_h * stride_dh)

    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_db_ptr = db_ptr + (pid_h * stride_dbh)
    db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)

    mask_k = k_idxs < N
    mask_j = j_idxs < N

    db_block = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)

    # loop over i
    for start_i in range(0, N, BLOCK_I):
        start_i = tl.multiple_of(start_i, BLOCK_I)

        # Rebuilt from bases rather than advanced — see the note in _fwd.
        q_ptrs = (base_q_ptr + (start_i * stride_qm)
                  + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd))
        k_ptrs = (base_k_ptr + (start_i * stride_km)
                  + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd))
        v_ptrs = (base_v_ptr + (start_i * stride_vm)
                  + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd))
        do_ptrs = (base_do_ptr + (start_i * stride_dom)
                   + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod))
        l_ptrs = base_l_ptr + (start_i * stride_lm) + (j_idxs * stride_ln)
        mask_ptrs = base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn)
        d_ptrs = base_d_ptr + (start_i * stride_dm) + (j_idxs * stride_dn)

        q_block = tl.load(q_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        q_block *= tl.full([1], value=sm_scale, dtype=input_dtype)
        k_block = tl.load(k_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]

        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :], cache_modifier=".cg").to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") # [k]

        scores = tl.dot(q_block, tl.trans(k_block), b_block, input_precision="ieee") # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores = tl.where(m_block[None, :], neg_inf, scores)

        sm_denom = tl.load(l_ptrs, mask=mask_j, cache_modifier=".cg") # [j]
        sm_score = tl.math.exp2((scores  - sm_denom[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        delta = tl.load(d_ptrs, mask_j, cache_modifier=".cg") # [j]

        v_block = tl.load(v_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]
        dsm_value = tl.dot(do, tl.trans(v_block), input_precision="ieee") # [j,k]

        dscores = sm_score * (dsm_value - delta[:, None]) # [j,k]

        db_block += dscores

    if PADDED:
        tl.store(db_ptrs, db_block.to(input_dtype))
    else:
        tl.store(db_ptrs, db_block.to(input_dtype), mask=mask_j[:, None] & mask_k[None, :])

