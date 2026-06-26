"""Regressions for the two BLOCKER silent-wrongs the 2026-06-25 Triton-lens audit found
on core language semantics the matmul/reduce campaign never touched:

  B1. tl.reduce with a CUSTOM associative combine was string-sniffed and silently
      mis-computed — a product (arith.mulf) combine returned the SUM; a max-by-magnitude
      (where(abs(a)>abs(b),a,b)) returned the plain MAX. Now refused loudly; plain
      sum/max/min still compute.
  B2. an scf.if yielding an INTEGER produced inside the branch (an inner scf.for
      accumulator) was declared `float` — silent precision loss for i32 > 2^24 and i64.
      Now derives the dtype from the IR result type + has a long/ulong branch.
"""
import pytest
import torch

try:
    import triton
    import triton.language as tl
    _HAS = True
except Exception:
    _HAS = False

from triton_msl.errors import MetalNonRecoverableError

requires = pytest.mark.skipif(not _HAS, reason="triton not available")

if _HAS:
    @triton.jit
    def _mul(a, b):
        return a * b

    @triton.jit
    def _k_prod(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _mul))

    @triton.jit
    def _maxmag(a, b):
        return tl.where(tl.abs(a) > tl.abs(b), a, b)

    @triton.jit
    def _k_maxmag(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _maxmag))

    @triton.jit
    def _k_sum(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_max(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.max(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_if_i32(COND, OUT):
        c = tl.load(COND)
        acc = 1
        if c > 0:
            for _ in range(10):
                acc += 2000003          # 1 + 10*2000003 = 20000031, odd, > 2^24
        tl.store(OUT, acc)

    @triton.jit
    def _k_if_i64(COND, INIT, OUT):
        c = tl.load(COND)
        acc = tl.load(INIT)             # i64 (forces the scf.if result dtype to i64)
        if c > 0:
            for _ in range(10):
                acc += 1600000003       # 16000000031, well beyond fp32 mantissa
        tl.store(OUT, acc)


def _clear():
    import os
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")


@requires
def test_product_reduce_refuses_not_sum():
    _clear()
    X = torch.tensor([1., 2., 3., 4., 1., 1., 1., 1.], device="mps")
    OUT = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_prod[(1,)](X, OUT, N=8); torch.mps.synchronize()


@requires
def test_max_by_magnitude_reduce_refuses_not_plain_max():
    _clear()
    X = torch.tensor([1., -5., 3., -2., 4., -9., 0.5, 2.], device="mps")
    OUT = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_maxmag[(1,)](X, OUT, N=8); torch.mps.synchronize()


@requires
def test_plain_sum_max_still_compute():
    _clear()
    X = torch.tensor([1., -5., 3., -2., 4., -9., 0.5, 2.], device="mps")
    for kern, ref in ((_k_sum, X.sum().item()), (_k_max, X.max().item())):
        OUT = torch.zeros(1, device="mps")
        kern[(1,)](X, OUT, N=8); torch.mps.synchronize()
        assert abs(OUT.item() - ref) < 1e-3, (OUT.item(), ref)


@requires
def test_scf_if_int_accumulator_not_float_rounded():
    _clear()
    COND = torch.ones(1, dtype=torch.int32, device="mps")
    OUT = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_if_i32[(1,)](COND, OUT); torch.mps.synchronize()
    assert OUT.item() == 1 + 10 * 2000003   # 20000031 exactly, no fp32 rounding


@requires
def test_scf_if_i64_accumulator_not_truncated():
    _clear()
    COND = torch.ones(1, dtype=torch.int32, device="mps")
    INIT = torch.ones(1, dtype=torch.int64, device="mps")
    OUT = torch.zeros(1, dtype=torch.int64, device="mps")
    try:
        _k_if_i64[(1,)](COND, INIT, OUT); torch.mps.synchronize()
    except MetalNonRecoverableError:
        return   # refusing is acceptable (correct-or-refuse); silently truncating is not
    assert OUT.item() == 1 + 10 * 1600000003   # 16000000031 exactly, no i64->i32/float loss
