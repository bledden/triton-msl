# tests/test_fa_simdgroup_diff.py
"""Differential gate: the simd FA template must match the scalar template (the
validated oracle) AND the torch reference, across dtype x causal x alignment."""
import math
import platform
import pytest
import torch

from triton_msl.codegen._msl_templates import (
    make_flash_attention_kernel_simdgroup, make_flash_attention_kernel_tiled)

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()
         and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader")


def _ref(q, k, v, causal):
    qf, kf, vf = q.float(), k.float(), v.float()
    sc = 1.0 / math.sqrt(qf.shape[-1])
    a = (qf*sc) @ kf.transpose(-2, -1)
    if causal:
        n = a.shape[-1]; a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    return torch.nan_to_num(torch.softmax(a, -1), nan=0.0) @ vf


def _run(src, name, q, k, v, out, threads_pg):
    lib = torch.mps.compile_shader(src)
    Z, H, N, _ = q.shape
    nqb = (N + 31)//32
    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    getattr(lib, name)(q, k, v, out, *s, Z, H, N, threads=(nqb*threads_pg, Z*H), group_size=(threads_pg, 1))
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [128, 100, 192])
def test_simd_matches_scalar_and_torch(dt, causal, N):
    HD, Z, H = 128, 1, 2
    od = "f16" if dt == torch.float16 else "f32"
    tol = 5e-2 if dt == torch.float16 else 1e-3
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    k = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    v = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    ref = _ref(q, k, v, causal)
    o_sd = torch.empty_like(q); o_sc = torch.empty_like(q)
    _run(make_flash_attention_kernel_simdgroup(HD, 32, 64, causal=causal, out_dtype=od),
         "flash_attention", q, k, v, o_sd, 256)
    _run(make_flash_attention_kernel_tiled(HD, 32, 32, Dc=64, causal=causal, out_dtype=od),
         "flash_attention", q, k, v, o_sc, 1024)
    assert (o_sd.float() - ref).abs().max().item() < tol     # simd vs torch
    assert (o_sd.float() - o_sc.float()).abs().max().item() < tol  # simd vs scalar oracle
