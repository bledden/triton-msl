"""MEPT M3c: the single-pass register-array form lifts the >1024 threadgroup
ceiling for 1-D kernels. Apple threadgroups cap at 1024 threads; the array
form keeps num_threads <= 128 and gives each thread block_size/num_threads
elements, so BLOCK 2048/4096 compute in one pass.

Two patterns, both validated above 1024:
  - loop-carried register-array iter-arg (column-sum), and
  - reduction-in-loop (tridec Bug-2 shape) — this one REFUSES with MEPT off
    (the UNKNOWN_ backstop, BLOCK > threadgroup), so it is a genuine flag-ON
    guard, not a path that would pass either way.

Run with TRITON_METAL_MEPT=1. Serial only.
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
    reason="requires TRITON_METAL_MEPT=1 (M3 register-array form)")

if HAS:
    @triton.jit
    def _vec_acc(X, OUT, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for i in range(n_tiles):
            acc = acc + tl.load(X + i * BLOCK + offs)
        tl.store(OUT + offs, acc)

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
@pytest.mark.parametrize("BLOCK", [2048, 4096])
def test_iter_carry_above_1024(BLOCK):
    n_tiles = 4
    X = torch.randn(n_tiles * BLOCK)
    OUT = torch.zeros(BLOCK)
    _vec_acc[(1,)](X, OUT, n_tiles, BLOCK=BLOCK)
    want = X.view(n_tiles, BLOCK).sum(0)
    assert torch.allclose(OUT, want, atol=1e-2), (
        f"iter-carry BLOCK={BLOCK}: max|diff|={float((OUT-want).abs().max()):.4g}")


@requires_metal
@requires_mept
@pytest.mark.parametrize("BLOCK", [2048, 4096])
def test_reduce_in_loop_above_1024(BLOCK):
    N = 4 * BLOCK
    X = torch.randn(N)
    OUT = torch.zeros(1)
    _sum_in_loop[(1,)](X, OUT, N, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 5e-1, (
        f"reduce-in-loop BLOCK={BLOCK}: got {float(OUT[0])} want {X.sum().item()}")
