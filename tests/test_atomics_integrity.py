"""Atomic RMW integrity (audit C2).

Metal has no 16-bit device atomic. The float-atomic CAS loop operates on 32-bit
words, so an fp16/bf16 atomic_add silently corrupted the 2-byte slot (reproduced:
scatter-add of eight 1.0s into two bins gave [0.0, 2.5] instead of [4, 4]). These
tests pin that such atomics now REFUSE loudly instead of returning wrong output,
and that the supported fp32 atomic still works.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    from triton_metal.backend.compiler import MetalBackend
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
    HAS_CPP = MetalBackend._has_cpp_passes()
except Exception:
    HAS = False
    HAS_CPP = False

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


_FP16_ATOMIC_SCRIPT = '''import os
os.environ.setdefault("TRITON_DEFAULT_BACKEND", "metal")
import torch, triton, triton.language as tl
@triton.jit
def k(Idx, Val, Out, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_add(Out + tl.load(Idx + i), tl.load(Val + i))
idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
val = torch.ones(8, dtype=torch.float16)
out = torch.zeros(2, dtype=torch.float16)
k[(1,)](idx, val, out, N=8)
'''


@requires_metal
@pytest.mark.skipif(not HAS_CPP, reason="cpp not built")
def test_fp16_atomic_refusal_under_default_route():
    """Refusal parity on the DEFAULT route (Phase 1, T4).

    The in-process test above may inherit TRITON_METAL_FORCE_PYTHON from the
    surrounding session. This pins the refusal in a fresh subprocess with the
    C++ passes built and no FORCE_PYTHON: default routing must still raise
    MetalNonRecoverableError, never produce silently-corrupt output.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ,
               PYTHONPATH=root,
               TRITON_METAL_CACHE_DIR=tempfile.mkdtemp(),
               TRITON_CACHE_DIR=tempfile.mkdtemp())
    env.pop("TRITON_METAL_FORCE_PYTHON", None)
    r = subprocess.run([sys.executable, "-c", _FP16_ATOMIC_SCRIPT],
                       env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode != 0, (
        f"fp16 atomic_add must refuse under the default route; "
        f"exited 0\nstdout: {r.stdout}\nstderr: {r.stderr}")
    assert "MetalNonRecoverableError" in r.stderr, (
        f"expected MetalNonRecoverableError in stderr, got:\n{r.stderr}")


@requires_metal
def test_fp32_atomic_add_still_correct():
    # The supported 32-bit float atomic must keep working.
    idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    val = torch.ones(8, dtype=torch.float32)
    out = torch.zeros(2, dtype=torch.float32)
    _scatter_add[(1,)](idx, val, out, N=8)
    np.testing.assert_allclose(out.numpy(), [4.0, 4.0], atol=1e-5)
