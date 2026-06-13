"""MEPT M3a GPU correctness: a multi-element value carried as an scf.for
iter-arg (per-element accumulator) computes the column-sum correctly under
flag-ON. Previously the array iter-arg was emitted as a scalar -> invalid
MSL / refusal. Run with TRITON_METAL_MEPT=1. Serial only.
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
    reason="requires TRITON_METAL_MEPT=1 (M3 register-array iter-arg)")

if HAS:
    @triton.jit
    def _vec_accumulate(X, OUT, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)   # per-element array iter-arg
        for i in range(n_tiles):
            acc = acc + tl.load(X + i * BLOCK + offs)
        tl.store(OUT + offs, acc)


@requires_metal
@requires_mept
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_vec_accumulate_column_sum(BLOCK):
    n_tiles = 8
    X = torch.randn(n_tiles * BLOCK)
    OUT = torch.zeros(BLOCK)
    _vec_accumulate[(1,)](X, OUT, n_tiles, BLOCK=BLOCK)
    want = X.view(n_tiles, BLOCK).sum(0)
    assert torch.allclose(OUT, want, atol=1e-2), (
        f"BLOCK={BLOCK}: max|diff|={float((OUT-want).abs().max()):.4g}")
