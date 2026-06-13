"""emit_msl refuses on unresolved UNKNOWN_<id> instead of emitting invalid MSL.

This pins the TRITON_METAL_MEPT=0 ESCAPE-HATCH behavior. As of M5 the
register-array model is default-ON and computes this kernel at BLOCK>=256 (see
tests/test_mept_m5_default_gpu.py). With the escape hatch (MEPT=0, pinned by the
autouse fixture below), a value defined outside a runtime-bound loop and used
inside it at BLOCK>threadgroup-size can't be resolved on the scalar path -> it
refuses loudly (the UNKNOWN_ backstop) rather than emitting invalid MSL.
BLOCK<=128 still runs on the scalar path. (downstream tridec bug 2)
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
    # These tests assert the TRITON_METAL_MEPT=0 ESCAPE-HATCH behavior (the
    # legacy scalar/wrap-loop path). Pin the flag to "0" explicitly — NOT
    # delenv: as of M5 the default is ON, so removing the var would let the
    # kernel compute and break the refusal assertion. setenv auto-restores.
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")


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


# Escape-hatch behavior: with MEPT=0 (pinned by _force_mept_off) this kernel
# refuses at BLOCK=256. Default-ON it computes — see test_mept_m5_default_gpu.py.
@requires_metal
def test_sum_in_loop_block256_refuses_not_compile_error():
    from triton_metal.errors import MetalNonRecoverableError
    N = 1024; X = torch.randn(N); OUT = torch.zeros(1)
    with pytest.raises(MetalNonRecoverableError):
        _sum_in_loop[(1,)](X, OUT, N, (N + 255) // 256, BLOCK=256)
