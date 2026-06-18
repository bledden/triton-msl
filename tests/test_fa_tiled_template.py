"""Standalone parity + structure tests for the head-dim-tiled FA2 MSL template.

The template (`make_flash_attention_kernel_tiled`) returns an MSL kernel STRING;
these tests compile it with `torch.mps.compile_shader` and run it on the Metal
GPU, checking against a torch reference. fp32, non-causal, head_dim=128.

Buffer ABI (binding order), matched in both the MSL signature and the launch:
    Q, K, V, Out          (device pointers)
    Z, H, N_CTX           (uint)
    16 strides (uint)     Q.{z,h,m,k}, K.{z,h,n,k}, V.{z,h,n,k}, O.{z,h,m,k}
    scale                 (float)

Launch uses the REAL torch.mps.compile_shader binding API (see
triton_metal/backend/compile_shader_runtime.py):
    lib.<kernel_name>(*args, threads=TOTAL_THREADS, group_size=THREADS_PER_GROUP)
where `threads` is the TOTAL number of threads (n_groups * group_size), NOT the
number of threadgroups.
"""
import math
import pytest
import torch

from triton_metal.codegen._msl_templates import make_flash_attention_kernel_tiled

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _ref(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    a = (q * scale) @ k.transpose(-2, -1)
    if causal:
        n = a.shape[-1]
        a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    a = torch.softmax(a, dim=-1)
    return torch.nan_to_num(a, nan=0.0) @ v


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM", [(1, 1, 64, 128), (1, 2, 96, 128)])
def test_tiled_fa_fp32_noncausal(Z, H, N_CTX, HEAD_DIM):
    BLOCK_M = BLOCK_N = 32
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)

    src = make_flash_attention_kernel_tiled(
        HEAD_DIM, BLOCK_M, BLOCK_N, Dc=64, causal=False, out_dtype="fp32"
    )
    lib = torch.mps.compile_shader(src)

    n_q_blocks = N_CTX // BLOCK_M
    n_groups = n_q_blocks * (Z * H)
    group_size = BLOCK_M * BLOCK_N  # 1024
    threads = n_groups * group_size

    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    # ABI order: Q, K, V, Out, Z, H, N_CTX, <16 strides>, scale
    lib.flash_attention(
        q, k, v, out,
        Z, H, N_CTX,
        *s,
        1.0 / math.sqrt(HEAD_DIM),
        threads=threads, group_size=group_size,
    )
    torch.mps.synchronize()

    ref = _ref(q, k, v, causal=False)
    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-3, max_err


def test_tiled_fa_emits_chunk_loop():
    src = make_flash_attention_kernel_tiled(128, 32, 32, Dc=64, causal=False, out_dtype="fp32")
    assert "device const float* Q" in src and "device float* Out" in src
    assert "for (uint dc = 0" in src  # head-dim chunk loop present
    assert "threadgroup float acc" in src
