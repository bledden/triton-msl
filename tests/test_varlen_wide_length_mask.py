"""A packed-sequence attention kernel that widens its lengths compares them at 64 bits.

`tl.load(cu + b).to(tl.int64)` before the subtraction keeps a length of 2**31 or more
positive, where the int32 difference of tests/test_varlen_fa_routing._varlen_fwd wraps
negative and selects nothing. The widened spelling takes the generic lowering, whose
staged store and load masks were rebuilt with a 32-bit cast, `(int)(row) < (int)(length)`
against a `long` length: a length whose low half is negative stored nothing. Measured
before the fix on 34378c9, 49156a0 and 12382ca alike.

Rows of Q and Out use a zero row stride, so every address stays inside the tensors
whatever the cumulative start: each launched row reads Q[0] and stores the same cells.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

from tests.test_varlen_fa_routing import _ref_varlen, _run_varlen
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)
POISON = -16384.0


@triton.jit
def _varlen_fwd_widened(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ZEXT_ROWS: tl.constexpr = False,
):  # fmt: skip
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_start = tl.load(cu_q + off_b).to(tl.int64)
    seqlen_q = tl.load(cu_q + off_b + 1).to(tl.int64) - q_start
    k_start = tl.load(cu_k + off_b).to(tl.int64)
    seqlen_k = tl.load(cu_k + off_b + 1).to(tl.int64) - k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = q * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = tl.where(kn[None, :] < seqlen_k, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = V + (k_start + kn)[:, None] * stride_vt + off_h * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    if ZEXT_ROWS:
        rows = offs_m.to(tl.uint32).to(tl.int64)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=rows[:, None] < seqlen_q)
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def _launch(cu_q, dtype, heads=2, dim=64, **constants):
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(4, heads, dim, generator=gen).to(dtype).to("mps")
    k = torch.randn(64, heads, dim, generator=gen).to(dtype).to("mps")
    v = torch.randn(64, heads, dim, generator=gen).to(dtype).to("mps")
    out = torch.full((4, heads, dim), POISON, dtype=dtype, device="mps")
    starts = torch.tensor(cu_q, dtype=torch.int32, device="mps")
    keys = torch.tensor([0, 45], dtype=torch.int32, device="mps")
    handle = _varlen_fwd_widened[(2, heads)](
        q, k, v, out, starts, keys,
        0, q.stride(1), q.stride(2), *k.stride(), *v.stride(), 0, out.stride(1), out.stride(2),
        heads, 64, 1.0 / math.sqrt(dim), 32, 32, dim, **constants,
    )  # fmt: skip
    torch.mps.synchronize()
    scores = torch.einsum("hd,nhd->hn", q[0].cpu().double(), k[:45].cpu().double()) / math.sqrt(dim)
    row = torch.einsum("hn,nhd->hd", torch.softmax(scores, -1), v[:45].cpu().double())
    return out.cpu().double(), row, handle


@requires_mps
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.float16, 2e-2)])
@pytest.mark.parametrize(
    "cu_q",
    [[0, 40], [-(2**31) + 10, 2**31 - 10], [-(2**31) + 10, 30]],
    ids=["length-40", "length-2**32-20", "length-2**31+20"],
)
def test_a_widened_length_selects_its_rows(cu_q, dtype, tol):
    out, row, handle = _launch(cu_q, dtype)
    # Every launched row is below the length: row 0's cells hold the selected value and
    # the zero row stride leaves the other rows alone.
    assert torch.allclose(out[0], row, atol=tol, rtol=tol)
    assert bool((out[1:] == POISON).all())
    masks = [line for line in handle.asm["msl"].splitlines() if "_fill_row) <" in line]
    assert masks and all("(long)(" in line and "(int)(r_" not in line for line in masks)


@requires_mps
def test_a_widened_kernel_matches_the_reference_on_packed_sequences():
    q, k, v, out, cu_q, cu_k = _run_varlen(_varlen_fwd_widened, [40, 17], [45, 20], 2, 64, torch.float32, 0.125)
    want = _ref_varlen(q, k, v, cu_q, cu_k, 2, 64, 0.125)
    assert torch.allclose(out.cpu(), want.cpu(), atol=1e-4, rtol=1e-4)


@requires_mps
def test_a_zero_extended_row_index_at_64_bits_refuses():
    # The rebuilt mask renders the row sum at its own width and sign-extends it; a zero
    # extension between the row index and a 64-bit comparison is not that.
    with pytest.raises(MetalNonRecoverableError, match="masked cooperative store"):
        _launch([0, 40], torch.float32, ZEXT_ROWS=True)
