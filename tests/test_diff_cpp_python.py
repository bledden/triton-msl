"""Differential harness: C++ vs Python lowering must produce identical bytes.

Each path runs in its own subprocess (fresh cache dir, fresh process state)
and saves the kernel output buffer; the parent byte-compares the two .npy
files. Skips entirely when the C++ pass library isn't built.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import triton  # noqa: F401  (backend discovery before triton_metal imports)
from triton_metal.backend.compiler import MetalBackend

pytestmark = pytest.mark.skipif(not MetalBackend._has_cpp_passes(), reason="cpp not built")

KERNEL = '''import os, sys
os.environ.setdefault("TRITON_DEFAULT_BACKEND", "metal")
import numpy as np, torch, triton, triton.language as tl
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N); x = tl.load(X + i)
    tl.store(O + i, (x * 2.0 + 1.0).to(tl.float16))
torch.manual_seed(0)
x = torch.randn(256); o = torch.zeros(256, dtype=torch.float16)
k[(1,)](x, o, N=256); np.save(sys.argv[1], o.numpy())'''

def _run(out, force_python):
    # Fresh TRITON_CACHE_DIR per run: TRITON_METAL_FORCE_PYTHON is not in
    # Triton's cache key, so a shared ~/.triton/cache would replay the first
    # run's binary in the second run and make the differential vacuous.
    env = dict(os.environ, PYTHONPATH=os.getcwd(),
               TRITON_METAL_CACHE_DIR=tempfile.mkdtemp(),
               TRITON_CACHE_DIR=tempfile.mkdtemp())
    env["TRITON_METAL_FORCE_PYTHON"] = "1" if force_python else "0"
    subprocess.run([sys.executable, "-c", KERNEL, out], check=True, env=env, timeout=180)

def test_elementwise_matches():
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.npy"), os.path.join(d, "b.npy")
        _run(a, True)
        _run(b, False)
        np.testing.assert_array_equal(np.load(a), np.load(b))
