"""A replacement FlashAttention template stores exactly the rows the source's Out mask keeps.

The template clips the row it stores, ``program_id(0) * BM + arange(BM) < N_CTX``. Through
8f2c58f an Out mask on any index derived from that one, ``(offs_m + 1)[:, None] < N_CTX``,
``(offs_m - 16)``, ``(offs_m * 2)``, was read as that clip, and the head_dim 64 and 128
templates stored other rows than the source (GPU: 128 to 10240 cells per case below, values
correct, no error). head_dim 32 runs the source's own store and was never affected.
"""

import pytest
import torch
import triton  # noqa: F401

import triton_msl.autotuning._fa_dispatch as fa_dispatch
from tests.test_matmul_source_extents import _kernel
from triton_msl.errors import MetalNonRecoverableError

SOURCE = """
@triton.jit
def {{name}}(Q, K, V, Out, qz, qh, qm, qd, kz, kh, kn, kd, vz, vh, vn, vd, oz, oh, om, od, Z, H, N_CTX,
           BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    zh = tl.program_id(1)
    z = zh // H
    h = zh % H
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + z * qz + h * qh + offs_m[:, None] * qm + offs_d[None, :] * qd,
                mask=offs_m[:, None] < N_CTX, other=0.0) * (1.0 / tl.sqrt(float(D)))
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    hi = N_CTX
    if CAUSAL:
        hi = min((start_m + 1) * BM, N_CTX)
    for start_n in range(0, hi, BN):
        k = tl.load(K + z * kz + h * kh + (start_n + offs_n)[:, None] * kn + offs_d[None, :] * kd,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + z * vz + h * vh + (start_n + offs_n)[:, None] * vn + offs_d[None, :] * vd,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p, v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + z * oz + h * oh + offs_m[:, None] * om + offs_d[None, :] * od, acc, mask={smask})
"""
ROWS, N_CTX, BLOCK, HEADS = 96, 80, 32, 2
POISON = -16384.0
# tag: (Out mask, the rows of [0, ROWS) it keeps, whether the template may stand in)
MASKS = {
    "canonical": ("offs_m[:, None] < N_CTX", lambda r: r < N_CTX, True),
    "compared_before_lifting": ("(offs_m < N_CTX)[:, None]", lambda r: r < N_CTX, True),
    "with_whole_head_columns": ("(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D)", lambda r: r < N_CTX, True),
    "shifted_rows": ("(offs_m + 1)[:, None] < N_CTX", lambda r: r + 1 < N_CTX, False),
    "lowered_rows": ("(offs_m - 16)[:, None] < N_CTX", lambda r: r - 16 < N_CTX, False),
    "scaled_rows": ("(offs_m * 2)[:, None] < N_CTX", lambda r: r * 2 < N_CTX, False),
    "inclusive_bound": ("offs_m[:, None] <= N_CTX", lambda r: r <= N_CTX, False),
}


@pytest.fixture
def replaced(monkeypatch):
    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


def _oracle(q, k, v, head_dim, causal, keep):
    """The source: zero-masked loads past N_CTX, whole key blocks, then the Out mask's rows."""
    q, k, v = (t.double().clone() for t in (q, k, v))
    for t in (q, k, v):
        t[:, :, N_CTX:] = 0
    rows = torch.arange(ROWS)
    hi = torch.full((ROWS,), N_CTX)
    if causal:
        hi = torch.minimum((rows // BLOCK + 1) * BLOCK, hi)
    seen = rows[None, :] < (((hi + BLOCK - 1) // BLOCK) * BLOCK)[:, None]
    if causal:
        seen &= rows[:, None] >= rows[None, :]
    scores = ((q / head_dim**0.5) @ k.transpose(-2, -1)).masked_fill(~seen, float("-inf"))
    expected = torch.full(q.shape, POISON, dtype=torch.float64)
    kept = keep(rows)
    expected[:, :, kept] = (torch.softmax(scores, -1) @ v)[:, :, kept]
    return expected


@pytest.mark.parametrize("tag", sorted(MASKS))
@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_out_store_mask_keeps_the_source_rows(tmp_path, replaced, head_dim, causal, tag):
    smask, keep, template = MASKS[tag]
    fn = _kernel(tmp_path, f"fa_store_{tag}_{head_dim}_{causal}", SOURCE.format(smask=smask))
    generator = torch.Generator().manual_seed(head_dim + causal)
    q, k, v = (torch.randn(1, HEADS, ROWS, head_dim, generator=generator) for _ in range(3))
    out = torch.full((1, HEADS, ROWS, head_dim), POISON, device="mps")
    arrays = [t.to("mps") for t in (q, k, v)] + [out]
    strides = [s for t in arrays for s in t.stride()]
    try:
        fn[(ROWS // BLOCK, HEADS)](*arrays, *strides, 1, HEADS, N_CTX, BLOCK, BLOCK, head_dim, causal)
        torch.mps.synchronize()
    except MetalNonRecoverableError as error:
        # 34378c9 and 8f2c58f ran all four of these on the head_dim 64 and 128 templates.
        assert not template and head_dim != 32, error
        assert "non-tile-boundary output store mask" in str(error)
        assert True not in replaced
        return
    assert template or head_dim == 32, f"{tag}: ran on a template that clips offs_m < N_CTX"
    expected = _oracle(q, k, v, head_dim, causal, keep)
    actual = out.cpu().double()
    written = expected != POISON
    assert torch.equal(actual != POISON, written)
    torch.testing.assert_close(actual[written], expected[written], atol=2e-3, rtol=2e-3)
