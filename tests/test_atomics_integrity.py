"""Atomic RMW integrity (audit C2 → Phase 3 feature 1).

Metal has no 16-bit device atomic. A naive float-atomic CAS loop operates on
32-bit words, so an fp16/bf16 atomic_add would corrupt the 2-byte slot AND its
neighbor (reproduced: scatter-add of eight 1.0s into two bins gave [0.0, 2.5]
instead of [4, 4]). The backend originally REFUSED such atomics loudly.

As of 2026-06-13 the refusal is replaced by a neighbor-preserving 32-bit
word-CAS (_emit_atomic_rmw_16bit in _lowerer_control.py; see
docs/superpowers/specs/2026-06-13-fp16-bf16-atomics-design.md). These tests now
pin that fp16/bf16 atomic_add produces the CORRECT result (no corruption), and
that the supported fp32 atomic still works. Detailed correctness coverage
(accumulation, neighbor preservation, bf16) lives in tests/test_fp16_atomics.py.
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
    from triton_msl.backend.compiler import MetalBackend
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
def test_fp16_atomic_add_correct_not_silentwrong():
    # The exact reproduction that previously gave [0.0, 2.5] (corruption) and
    # then loudly refused. The word-CAS now makes it correct: eight 1.0s
    # scatter-added into two bins => [4, 4]. No neighbor corruption.
    idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    val = torch.ones(8, dtype=torch.float16)
    out = torch.zeros(2, dtype=torch.float16)
    _scatter_add[(1,)](idx, val, out, N=8)
    np.testing.assert_allclose(out.float().numpy(), [4.0, 4.0], atol=1e-3)


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
got = out.float().tolist()
assert abs(got[0] - 4.0) < 1e-3 and abs(got[1] - 4.0) < 1e-3, got
print("OK", got)
'''


@requires_metal
@pytest.mark.skipif(not HAS_CPP, reason="cpp not built")
def test_fp16_atomic_correct_under_default_route():
    """Correctness parity on the DEFAULT route (Phase 3 feature 1).

    The in-process test above may inherit TRITON_MSL_FORCE_PYTHON from the
    surrounding session. This pins the word-CAS correctness in a fresh
    subprocess with the C++ passes built and no FORCE_PYTHON: default routing
    must produce the right result ([4, 4]), never silently-corrupt output.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ,
               PYTHONPATH=root,
               TRITON_MSL_CACHE_DIR=tempfile.mkdtemp(),
               TRITON_CACHE_DIR=tempfile.mkdtemp())
    env.pop("TRITON_MSL_FORCE_PYTHON", None)
    r = subprocess.run([sys.executable, "-c", _FP16_ATOMIC_SCRIPT],
                       env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, (
        f"fp16 atomic_add must succeed on the default route; "
        f"exited {r.returncode}\nstdout: {r.stdout}\nstderr: {r.stderr}")
    assert "OK" in r.stdout, (
        f"expected correct [4, 4] result on default route, got:\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")


@requires_metal
def test_fp32_atomic_add_still_correct():
    # The supported 32-bit float atomic must keep working.
    idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    val = torch.ones(8, dtype=torch.float32)
    out = torch.zeros(2, dtype=torch.float32)
    _scatter_add[(1,)](idx, val, out, N=8)
    np.testing.assert_allclose(out.numpy(), [4.0, 4.0], atol=1e-5)
