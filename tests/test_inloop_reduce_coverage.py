"""In-loop reduction coverage (spec 2026-06-13-inloop-reduce-coverage).

A tt.reduce inside a runtime loop must NEVER silently sum only the first
num_threads elements when block_size > num_threads. Under MEPT=0 (no register
arrays) such a reduce must refuse loudly (Stage B); under the default flag the
common where-on-reduce shape must compute correctly (Stage C). Serial GPU.
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    from triton_metal.errors import MetalNonRecoverableError
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _sum_carry_in_loop(X, OUT, C: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, C):
            v = tl.load(X + i * BLOCK + tl.arange(0, BLOCK))
            acc = acc + tl.sum(v)
        tl.store(OUT + tl.arange(0, 1), acc)


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512])
def test_inloop_reduce_mept0_refuses(BLOCK, monkeypatch):
    """MEPT=0: an in-loop reduce with block>num_threads is uncovered → refuse
    loudly (was silent-wrong before Stage B)."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    C = 4
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)


@requires_metal
def test_inloop_reduce_small_block_ok(monkeypatch):
    """block_size <= num_threads is fully covered (one elem/thread) → never
    refused, correct under both flags."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    BLOCK, C = 128, 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    torch.testing.assert_close(OUT[0], X.sum(), rtol=1e-3, atol=1e-3)
