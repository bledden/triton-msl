"""num_stages is a documented no-op on Metal (Triton-API compat).

CUDA uses num_stages for software-pipelining depth (multi-buffering staged
operands). That win doesn't transfer here: the fast paths stream operands
directly from device with register blocking / explicit prefetch, already
saturating load/compute overlap, and Apple has no cp.async. A num_stages=2
double-buffered matmul measured flat-to-slower (11.13 vs 11.2 TFLOP/s, 2048^3).

Contract: passing num_stages>1 must (a) NOT change the result (it's ignored) and
(b) not be SILENT — _warn_inert_num_stages notes it once at debug level >= 1.
"""
import platform

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()),
    reason="Metal backend requires macOS + MPS",
)


@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_num_stages_is_a_correct_noop(num_stages):
    """A kernel compiled with any num_stages produces the SAME correct result."""
    import triton
    import triton.language as tl

    @triton.jit
    def addk(a, b, o, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)

    n = 4096
    a = torch.randn(n, device="mps")
    b = torch.randn(n, device="mps")
    o = torch.empty(n, device="mps")
    addk[(n // 256,)](a, b, o, n, BLOCK=256, num_stages=num_stages)
    torch.mps.synchronize()
    assert torch.allclose(o, a + b, atol=1e-5)


def test_warn_inert_num_stages_is_one_shot():
    """The inert-num_stages note fires at most once (no log spam)."""
    import triton_msl.backend.compiler as C

    saved = C._NUM_STAGES_WARNED
    try:
        C._NUM_STAGES_WARNED = False
        C._warn_inert_num_stages(2)   # first call may emit (depending on debug level)
        assert C._NUM_STAGES_WARNED is True
        C._warn_inert_num_stages(4)   # second call: must be a no-op (already warned)
        assert C._NUM_STAGES_WARNED is True
    finally:
        C._NUM_STAGES_WARNED = saved
