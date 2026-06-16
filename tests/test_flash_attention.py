"""Flash Attention v2 on Metal GPU via @triton.jit.

Tests the standard Triton FlashAttention kernel (non-causal and causal)
compiled through the generic lowerer to MSL and executed on Apple GPU.

This exercises the full 2D stack: tiled matmul (tl.dot), online softmax
(tl.max + tl.exp + tl.sum), masking (tl.where), K-loop (scf.for with
matrix iter_args), and strided 2D loads/stores.
"""

import pytest
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")

try:
    from triton_metal.errors import MetalNonRecoverableError
except Exception:
    MetalNonRecoverableError = Exception


@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash Attention v2 forward kernel.

    Each program handles one BLOCK_M chunk of queries.
    Grid: (N_CTX // BLOCK_M, Z * H)
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Scale
    qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    q = q * qk_scale

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only attend to positions <= current
    hi = N_CTX
    if IS_CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, N_CTX)

    # K/V loop
    for start_n in range(0, hi, BLOCK_N):
        # Load K block [BLOCK_N, HEAD_DIM]
        k_ptrs = K + off_z * stride_kz + off_h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))

        # Causal mask
        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Load V block [BLOCK_N, HEAD_DIM]
        v_ptrs = V + off_z * stride_vz + off_h * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # Accumulate P @ V
        acc += tl.dot(p.to(tl.float32), v)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _ref_attention(q, k, v, causal=False):
    """Reference attention using PyTorch."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    if causal:
        N = attn.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=attn.device))
        attn = attn.masked_fill(mask[None, None] == 0, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    # Replace NaN from all-masked rows with 0
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


class TestFlashAttention:
    """Flash Attention tests via @triton.jit → Metal GPU."""

    @requires_triton
    @pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM", [
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 1, 64, 64),
        (1, 2, 64, 32),
        (2, 2, 64, 32),
        (1, 1, 128, 32),
    ])
    def test_non_causal(self, Z, H, N_CTX, HEAD_DIM):
        """Non-causal attention."""
        BLOCK_M = min(32, N_CTX)
        BLOCK_N = min(32, N_CTX)
        torch.manual_seed(42)
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        out = torch.empty_like(q)

        grid = (N_CTX // BLOCK_M, Z * H)
        _flash_attn_fwd[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=False,
        )
        ref = _ref_attention(q, k, v, causal=False)
        assert (out - ref).abs().max().item() < 0.01, \
            f"Non-causal attention max error: {(out - ref).abs().max().item()}"

    @requires_triton
    @pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM", [
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 1, 64, 64),
        (1, 2, 64, 32),
        (1, 1, 128, 32),
    ])
    def test_causal(self, Z, H, N_CTX, HEAD_DIM):
        """Causal (autoregressive) attention."""
        BLOCK_M = min(32, N_CTX)
        BLOCK_N = min(32, N_CTX)
        torch.manual_seed(42)
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        out = torch.empty_like(q)

        grid = (N_CTX // BLOCK_M, Z * H)
        _flash_attn_fwd[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=True,
        )
        ref = _ref_attention(q, k, v, causal=True)
        assert (out - ref).abs().max().item() < 0.01, \
            f"Causal attention max error: {(out - ref).abs().max().item()}"

    @requires_triton
    @pytest.mark.parametrize("BLOCK", [16, 8])
    def test_head_dim_over_64_refuses(self, BLOCK):
        """Integrity guard: FlashAttention at head_dim > 64 must REFUSE loudly
        (`MetalNonRecoverableError`), never silently mis-compute. head_dim=128 with
        small blocks previously produced garbage (max error ~1000) with no error
        raised; the prescan guard in `lower()` now refuses it. head_dim 32 and 64
        (tested above) stay supported. Large-head_dim FA is future work."""
        Z, H, N_CTX, HEAD_DIM = 1, 1, 128, 128
        torch.manual_seed(42)
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device='cpu', dtype=torch.float32)
        out = torch.empty_like(q)
        grid = (N_CTX // BLOCK, Z * H)
        with pytest.raises(MetalNonRecoverableError):
            _flash_attn_fwd[grid](
                q, k, v, out,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                Z, H, N_CTX,
                BLOCK_M=BLOCK, BLOCK_N=BLOCK, HEAD_DIM=HEAD_DIM,
                IS_CAUSAL=False,
            )
