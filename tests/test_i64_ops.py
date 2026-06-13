"""int64/uint64 reduce / where / transpose. Values exceed 2^32 to prove no
truncation. Run with METAL_TEST_INT64=1 for the upstream corpus; the project
tests here exercise the paths directly. Serial GPU."""
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

if HAS:
    @triton.jit
    def _sum_i64(X, OUT, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(OUT, tl.sum(x))

    @triton.jit
    def _max_i64(X, OUT, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(OUT, tl.max(x))

    @triton.jit
    def _where_i64(C, A, B, OUT, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        c = tl.load(C + i) != 0
        tl.store(OUT + i, tl.where(c, tl.load(A + i), tl.load(B + i)))


@requires_metal
def test_i64_sum():
    BLOCK = 256
    X = torch.randint(2**40, 2**41, (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(1, dtype=torch.int64)
    _sum_i64[(1,)](X, OUT, BLOCK=BLOCK)
    assert int(OUT[0]) == int(X.sum()), f"got {int(OUT[0])} want {int(X.sum())}"


@requires_metal
def test_i64_max():
    BLOCK = 256
    X = torch.randint(-(2**41), 2**41, (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(1, dtype=torch.int64)
    _max_i64[(1,)](X, OUT, BLOCK=BLOCK)
    assert int(OUT[0]) == int(X.max()), f"got {int(OUT[0])} want {int(X.max())}"


@requires_metal
def test_i64_where():
    BLOCK = 256
    C = torch.randint(0, 2, (BLOCK,), dtype=torch.int64)
    A = torch.randint(2**40, 2**41, (BLOCK,), dtype=torch.int64)
    B = torch.randint(-(2**41), -(2**40), (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(BLOCK, dtype=torch.int64)
    _where_i64[(1,)](C, A, B, OUT, BLOCK=BLOCK)
    want = torch.where(C != 0, A, B)
    assert torch.equal(OUT, want)
