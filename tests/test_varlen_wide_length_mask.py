"""A packed-sequence attention kernel that widens its lengths compares them at 64 bits.

`tl.load(cu + b).to(tl.int64)` before the subtraction keeps a length of 2**31 or more
positive, where the int32 difference of tests/test_varlen_fa_routing._varlen_fwd wraps
negative and selects nothing. The widened spelling takes the generic lowering, whose
staged store and load masks were rebuilt with a 32-bit cast, `(int)(row) < (int)(length)`
against a `long` length: a length whose low half is negative stored nothing. Measured
before the fix on 34378c9, 49156a0 and 12382ca alike.

The rebuild flattens an index into one sum, which is exact while every term is 32 bits
wide. Beside a 64-bit value it is not. A 32-bit partial sum that holds the program id, a
`uint` in the shader, is zero-extended where the source sign-extends it: with the rows
spelled `arange - program_id`, row -1 of the second program was dropped from the mask by
5af3906's `(long)` cast, and on every earlier tree its address, a 32-bit row offset added
to a pointer that already carries a 64-bit offset, landed 2**32 elements away. A zero
extension in the source was sign-extended by the 32-bit cast and refused by 5af3906. The
masks and addresses of a 64-bit comparison or offset are now rendered operation by
operation at the source's widths. The rows expected below are those Triton's interpreter
stores.

Rows of Q use a zero row stride, so each launched row reads Q[0] and every stored row holds
the same values. Out has a zero row stride where a length of 2**31 or more would leave the
tensor, and real rows elsewhere, so the rows that pass the mask and where they land show.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

from tests.test_varlen_fa_routing import _ref_varlen, _run_varlen

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)
POISON = -16384.0
# Spellings of the query rows, their mask and their address.
SPELLINGS = (
    "canonical",  # program * block + arange
    "minus",  # arange - program
    "plus-minus-one",  # program + arange - 1
    "zext",  # arange - program, zero-extended for the mask
    "zext-canonical",
    "unsigned",  # arange - program, compared unsigned at 64 bits
    "wide-difference",  # the mask subtracts two zero-extended terms at 64 bits
    "split-before",  # the 32-bit row offset added after a 64-bit base offset
    "split-after",  # the 64-bit start offset added after the 32-bit row offset
)


@triton.jit
def _varlen_fwd_widened(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr = "canonical",
):  # fmt: skip
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_start = tl.load(cu_q + off_b).to(tl.int64)
    seqlen_q = tl.load(cu_q + off_b + 1).to(tl.int64) - q_start
    k_start = tl.load(cu_k + off_b).to(tl.int64)
    seqlen_k = tl.load(cu_k + off_b + 1).to(tl.int64) - k_start

    if ROWS == "canonical" or ROWS == "zext-canonical":
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    elif ROWS == "plus-minus-one":
        offs_m = start_m + tl.arange(0, BLOCK_M) - 1
    else:
        offs_m = tl.arange(0, BLOCK_M) - start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if ROWS == "unsigned":
        rmask = offs_m.to(tl.int64).to(tl.uint64)[:, None] < seqlen_q.to(tl.uint64)
    elif ROWS == "zext" or ROWS == "zext-canonical":
        rmask = offs_m.to(tl.uint32).to(tl.int64)[:, None] < seqlen_q
    elif ROWS == "wide-difference":
        rmask = (tl.arange(0, BLOCK_M).to(tl.uint32).to(tl.int64) - start_m.to(tl.uint32).to(tl.int64))[
            :, None
        ] < seqlen_q
    else:
        rmask = offs_m[:, None] < seqlen_q

    if ROWS == "split-before":
        q_ptrs = (
            Q + q_start * stride_qt + (offs_m[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd)
        )
    elif ROWS == "split-after":
        q_ptrs = (
            Q + (offs_m[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd) + q_start * stride_qt
        )
    else:
        q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=rmask, other=0.0)
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
    if ROWS == "split-before":
        o_ptrs = (
            Out + q_start * stride_ot + (offs_m[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od)
        )
    elif ROWS == "split-after":
        o_ptrs = (
            Out + (offs_m[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od) + q_start * stride_ot
        )
    else:
        o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=rmask)


def _launch(cu_q, dtype, heads=2, dim=64, out_rows=None, **constants):
    """Out with a zero row stride (4 rows), or ``out_rows`` real rows."""
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(4, heads, dim, generator=gen).to(dtype).to("mps")
    k = torch.randn(64, heads, dim, generator=gen).to(dtype).to("mps")
    v = torch.randn(64, heads, dim, generator=gen).to(dtype).to("mps")
    out = torch.full((out_rows or 4, heads, dim), POISON, dtype=dtype, device="mps")
    starts = torch.tensor(cu_q, dtype=torch.int32, device="mps")
    keys = torch.tensor([0, 45], dtype=torch.int32, device="mps")
    handle = _varlen_fwd_widened[(2, heads)](
        q, k, v, out, starts, keys,
        0, q.stride(1), q.stride(2), *k.stride(), *v.stride(),
        out.stride(0) if out_rows else 0, out.stride(1), out.stride(2),
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
    # Each rebuilt mask compares two 64-bit values.
    masks = [line for line in handle.asm["msl"].splitlines() if "_fill_row" in line and " < " in line]
    assert masks and all(") < (long)(" in line and ") < (int)(" not in line for line in masks)


@requires_mps
def test_a_widened_kernel_matches_the_reference_on_packed_sequences():
    q, k, v, out, cu_q, cu_k = _run_varlen(_varlen_fwd_widened, [40, 17], [45, 20], 2, 64, torch.float32, 0.125)
    want = _ref_varlen(q, k, v, cu_q, cu_k, 2, 64, 0.125)
    assert torch.allclose(out.cpu(), want.cpu(), atol=1e-4, rtol=1e-4)


def _rows_stored(spelling, start, length):
    """The Out rows the source stores for a two-program launch of 32-row blocks."""
    rows = set()
    for program in range(2):
        for lane in range(32):
            row = {"canonical": program * 32 + lane, "zext-canonical": program * 32 + lane}.get(
                spelling, lane - program
            )
            row = program + lane - 1 if spelling == "plus-minus-one" else row
            if spelling in ("zext", "zext-canonical"):
                passes = row % 2**32 < length
            elif spelling == "unsigned":
                passes = row % 2**64 < length % 2**64
            else:
                passes = row < length
            if passes:
                rows.add(start + row)
    return rows


@requires_mps
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.float16, 2e-2)])
@pytest.mark.parametrize("cu_q", [[5, 45], [5, 5], [8, 9]], ids=["length-40", "length-0", "length-1"])
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_row_index_beside_a_64_bit_length_selects_and_addresses_its_rows(spelling, cu_q, dtype, tol):
    out, row, _handle = _launch(cu_q, dtype, out_rows=80, ROWS=spelling)
    want = _rows_stored(spelling, cu_q[0], cu_q[1] - cu_q[0])
    stored = {index for index in range(80) if not bool((out[index] == POISON).all())}
    assert stored == want
    assert all(torch.allclose(out[index], row, atol=tol, rtol=tol) for index in want)
    if spelling not in ("canonical", "zext-canonical"):
        # Row -1 of the second program: below a signed length of zero, far above it
        # once zero-extended or compared unsigned.
        assert (cu_q[0] - 1 in want) == (spelling not in ("zext", "unsigned"))


@requires_mps
@pytest.mark.parametrize("spelling", ["zext", "zext-canonical", "unsigned"])
def test_a_zero_extended_row_index_is_compared_at_64_bits(spelling):
    # A length of 2**31 + 20: above every zero-extended 32-bit row, so each launched row
    # is stored, row -1 (2**32 - 1 once extended, 2**64 - 1 unsigned) excepted.
    out, row, handle = _launch([-(2**31) + 10, 30], torch.float32, ROWS=spelling)
    assert torch.allclose(out[0], row, atol=1e-4, rtol=1e-4) and bool((out[1:] == POISON).all())
    masks = [line for line in handle.asm["msl"].splitlines() if "_fill_row" in line and "Out[" in line]
    extension = " < (ulong)(" if spelling == "unsigned" else "(long)((uint)("
    assert masks and all(extension in line for line in masks)
