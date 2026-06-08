"""Atomic RMW integrity (audit C2).

Metal has no 16-bit device atomic. The float-atomic CAS loop operates on 32-bit
words, so an fp16/bf16 atomic_add silently corrupted the 2-byte slot (reproduced:
scatter-add of eight 1.0s into two bins gave [0.0, 2.5] instead of [4, 4]). These
tests pin that such atomics now REFUSE loudly instead of returning wrong output,
and that the supported fp32 atomic still works.
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
    def _scatter_add(Idx, Val, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.atomic_add(Out + tl.load(Idx + i), tl.load(Val + i))


@requires_metal
def test_fp16_atomic_add_refuses_not_silentwrong():
    from triton_metal.errors import MetalNonRecoverableError
    idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    val = torch.ones(8, dtype=torch.float16)
    out = torch.zeros(2, dtype=torch.float16)
    with pytest.raises(MetalNonRecoverableError):
        _scatter_add[(1,)](idx, val, out, N=8)


@requires_metal
def test_fp32_atomic_add_still_correct():
    # The supported 32-bit float atomic must keep working.
    idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    val = torch.ones(8, dtype=torch.float32)
    out = torch.zeros(2, dtype=torch.float32)
    _scatter_add[(1,)](idx, val, out, N=8)
    np.testing.assert_allclose(out.numpy(), [4.0, 4.0], atol=1e-5)
