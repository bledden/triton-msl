"""tridec Bug-2 remaining case: a masked cross-lane reduce (tl.sum over
tl.where) inside a runtime-bound loop must compute at BLOCK>=256, not refuse.
The trigger was arith.select not being MEPT-array-wired. Serial GPU."""
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

if HAS:
    @triton.jit
    def _sum_where_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = tl.zeros((), dtype=tl.float32)
        for i in range(n_tiles):
            idx = i * BLOCK + offs
            m = idx < N
            total += tl.sum(tl.where(m, tl.load(X + idx, mask=m, other=0.0), 0.0))
        tl.store(OUT, total)

    @triton.jit
    def _sum_where_nested(X, OUT, N, n_legs, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = tl.zeros((), dtype=tl.float32)
        for leg in range(n_legs):
            for i in range(n_tiles):
                idx = i * BLOCK + offs
                m = idx < N
                total += tl.sum(tl.where(m, tl.load(X + idx, mask=m, other=0.0), 0.0))
        tl.store(OUT, total)


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_sum_where_in_loop(BLOCK):
    N = 4096
    X = torch.randn(N); OUT = torch.zeros(1)
    _sum_where_in_loop[(1,)](X, OUT, N, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-1, (
        f"BLOCK={BLOCK}: got {float(OUT[0])} want {X.sum().item()}")


@requires_metal
@pytest.mark.parametrize("BLOCK,N", [(256, 3000), (512, 4100)])
def test_sum_where_partial_mask(BLOCK, N):
    # N not a multiple of BLOCK -> the last tile has m=False elements, so the
    # select's FALSE branch (the 0.0) is actually exercised (a branch-swap bug
    # would otherwise pass test_sum_where_in_loop, whose mask is all-True).
    X = torch.randn(N); OUT = torch.zeros(1)
    _sum_where_in_loop[(1,)](X, OUT, N, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X[:N].sum().item()) < 1e-1, (
        f"BLOCK={BLOCK} N={N}: got {float(OUT[0])} want {X[:N].sum().item()}")


@requires_metal
def test_sum_where_nested():
    # nested loops (tridec relay shape): total = n_legs * sum(X)
    N, BLOCK, n_legs = 2048, 256, 3
    X = torch.randn(N); OUT = torch.zeros(1)
    _sum_where_nested[(1,)](X, OUT, N, n_legs, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - n_legs * X.sum().item()) < 2e-1
