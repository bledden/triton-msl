# tests/test_fa_simdgroup_template.py
"""Standalone parity tests for the simdgroup-MMA FA template vs a torch reference."""
import math
import pytest
import torch

from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _ref(q, k, v, causal=False):
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(qf.shape[-1])
    a = (qf * scale) @ kf.transpose(-2, -1)
    if causal:
        n = a.shape[-1]
        a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    a = torch.softmax(a, dim=-1)
    return torch.nan_to_num(a, nan=0.0) @ vf


def _launch(lib, name, q, k, v, out):
    Z, H, N_CTX, _ = q.shape
    n_q_blocks = (N_CTX + 31) // 32          # ceil(N_CTX / BLOCK_M=32)
    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    getattr(lib, name)(q, k, v, out, *s, Z, H, N_CTX,
                       threads=(n_q_blocks * 256, Z * H), group_size=(256, 1))


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX", [(1, 1, 64), (1, 2, 128), (1, 8, 256)])
def test_simd_fa_fp32_noncausal(Z, H, N_CTX):
    HEAD_DIM = 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=False, out_dtype="fp32")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out - _ref(q, k, v)).abs().max().item() < 1e-3


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX", [(1, 2, 128), (1, 8, 256)])
def test_simd_fa_fp16_noncausal(Z, H, N_CTX):
    HEAD_DIM = 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    out = torch.empty(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=False, out_dtype="fp16")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out.float() - _ref(q, k, v)).abs().max().item() < 5e-2
