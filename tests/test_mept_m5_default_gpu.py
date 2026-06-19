"""MEPT M5: with NO TRITON_MSL_MEPT env var (the default), the register-array
model is ON, so the tridec Bug-2 reduction-in-loop kernel computes at BLOCK>=256
instead of refusing. The inverse of test_unknown_value_backstop's escape-hatch
(MEPT=0) refusal. Serial GPU.
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
def _default_env(monkeypatch):
    # Exercise the DEFAULT: ensure no explicit flag is set, so mept_enabled
    # resolves to its built-in default (post-flip: ON).
    monkeypatch.delenv("TRITON_MSL_MEPT", raising=False)


if HAS:
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
@pytest.mark.parametrize("BLOCK", [256, 512])
def test_bug2_computes_by_default(BLOCK):
    N = 4096
    X = torch.randn(N)
    OUT = torch.zeros(1)
    _sum_in_loop[(1,)](X, OUT, N, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-1, (
        f"BLOCK={BLOCK}: got {float(OUT[0])}, want {X.sum().item()}")
