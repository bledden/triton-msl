"""emit_msl refuses on unresolved UNKNOWN_<id> instead of emitting invalid MSL.

A value defined outside a runtime-bound loop and used inside it, when BLOCK
exceeds the threadgroup size (multi-element-per-thread regime), can't be
resolved yet (register-array spine = roadmap Phase 2). It previously emitted
UNKNOWN_<addr> -> cryptic xcrun compile error. Now it refuses loudly with an
actionable message. BLOCK <= 128 still runs correctly. (downstream tridec bug 2)
"""
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


@pytest.fixture(autouse=True)
def _force_mept_off(monkeypatch):
    # These tests assert the DEFAULT (MEPT-off) behavior. The lowerer reads
    # TRITON_METAL_MEPT per-compile, so pin it off here — otherwise a leaked
    # "1" from an earlier test file turns the BLOCK=256 refusal into a (valid)
    # M2 computation and this test fails on ordering. monkeypatch restores the
    # prior value on teardown, so this fixture doesn't itself pollute.
    monkeypatch.delenv("TRITON_METAL_MEPT", raising=False)


if HAS:
    @triton.jit
    def _sum_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)            # hoisted outside the runtime loop
        total = 0.0
        for i in range(n_tiles):
            idx = i * BLOCK + offs
            v = tl.load(X + idx, mask=idx < N, other=0.0)
            total += tl.sum(v)
        tl.store(OUT, total)


@requires_metal
def test_sum_in_loop_block128_runs():
    from triton_metal.errors import MetalNonRecoverableError
    N = 1024; X = torch.randn(N); OUT = torch.zeros(1)
    _sum_in_loop[(1,)](X, OUT, N, (N + 127) // 128, BLOCK=128)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-2


# NOTE: refuses only with MEPT OFF (default). Under TRITON_METAL_MEPT=1 this
# kernel computes correctly — see tests/test_mept_m2_bug2_gpu.py (M2).
@requires_metal
def test_sum_in_loop_block256_refuses_not_compile_error():
    from triton_metal.errors import MetalNonRecoverableError
    N = 1024; X = torch.randn(N); OUT = torch.zeros(1)
    with pytest.raises(MetalNonRecoverableError):
        _sum_in_loop[(1,)](X, OUT, N, (N + 255) // 256, BLOCK=256)
