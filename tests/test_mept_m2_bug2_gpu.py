"""MEPT M2 GPU correctness: the tridec Bug-2 reduction-in-loop kernel computes
correctly at BLOCK>=256 under flag-ON (previously refused with
MetalNonRecoverableError). Run with TRITON_METAL_MEPT=1. Serial only.
"""
import os

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")
requires_mept = pytest.mark.skipif(
    os.environ.get("TRITON_METAL_MEPT") != "1",
    reason="requires TRITON_METAL_MEPT=1 (M2 register-array form)")

if HAS:
    @triton.jit
    def _sum_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = 0.0
        for i in range(n_tiles):
            idx = i * BLOCK + offs
            v = tl.load(X + idx, mask=idx < N, other=0.0)
            total += tl.sum(v)
        tl.store(OUT, total)


@requires_metal
@requires_mept
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_sum_in_loop_computes_flag_on(BLOCK):
    N = 4096
    X = torch.randn(N)
    OUT = torch.zeros(1)
    n_tiles = (N + BLOCK - 1) // BLOCK
    _sum_in_loop[(1,)](X, OUT, N, n_tiles, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-1, (
        f"BLOCK={BLOCK}: got {float(OUT[0])}, want {X.sum().item()}")
