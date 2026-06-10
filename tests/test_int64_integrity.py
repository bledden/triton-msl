"""int64 integrity (audit C1).

The Metal backend declares int64 support (load/add/store work), but
``arith.cmpi`` hardcoded a 32-bit ``(int)``/``(uint)`` cast on its operands,
truncating i64 values before comparing — a silent-wrong that was hidden because
the upstream int64 tests are blanket-skipped. These tests pin the correct
behavior so the skip can be narrowed honestly.
"""
import numpy as np
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
    def _i64_gt(X, O, thresh, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(X + i)
        tl.store(O + i, (x > thresh).to(tl.int32))

    @triton.jit
    def _i64_lt_unsigned(X, O, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(X + i)
        # values that differ only above bit 32 must compare correctly
        tl.store(O + i, (x < 4_000_000_000).to(tl.int32))


@requires_metal
def test_i64_signed_gt_no_truncation():
    # Values straddling 2^31 must compare correctly; a 32-bit truncation would
    # wrap 3e9/5e9 to negative and report them as NOT greater than 2e9.
    x = torch.tensor([1_000_000_000, 3_000_000_000, 2_000_000_001, 0, -5,
                      5_000_000_000, 2_000_000_000, 9], dtype=torch.int64)
    o = torch.zeros(8, dtype=torch.int32)
    _i64_gt[(1,)](x, o, 2_000_000_000, N=8)
    ref = (x.numpy() > 2_000_000_000).astype(np.int32)
    np.testing.assert_array_equal(o.numpy(), ref)


@requires_metal
def test_i64_compare_above_bit32():
    # 4e9 > 2^32? compare a mix; truncation to 32 bits loses the high word.
    x = torch.tensor([3_999_999_999, 4_000_000_001, 4_294_967_297, 0,
                      4_000_000_000, 8_000_000_000, 1, 2], dtype=torch.int64)
    o = torch.zeros(8, dtype=torch.int32)
    _i64_lt_unsigned[(1,)](x, o, N=8)
    ref = (x.numpy() < 4_000_000_000).astype(np.int32)
    np.testing.assert_array_equal(o.numpy(), ref)


@requires_metal
def test_i64_loop_bounds_refuse_not_hang():
    # i64 scf.for induction lowering does not terminate (test_for_iv hang).
    # Until implemented it must refuse loudly, not hang. Run in a subprocess
    # with a hard timeout so a regression cannot wedge the suite/GPU.
    import subprocess, sys, os
    code = (
        "import os; os.environ.setdefault('TRITON_DEFAULT_BACKEND','metal')\n"
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(O, lo, hi, N: tl.constexpr):\n"
        "    s = tl.zeros((N,), tl.int64)\n"
        "    for i in range(lo, hi):\n"
        "        s += i\n"
        "    tl.store(O + tl.arange(0, N), s)\n"
        "o = torch.zeros(8, dtype=torch.int64)\n"
        "try:\n"
        "    k[(1,)](o, 2**33, 2**33 + 4, N=8)\n"
        "    print('RAN')\n"
        "except Exception as e:\n"
        "    print('REFUSED' if 'NonRecoverable' in type(e).__name__ else 'OTHER:' + type(e).__name__)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60, env=env)
    assert "REFUSED" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr[-400:]!r}"
