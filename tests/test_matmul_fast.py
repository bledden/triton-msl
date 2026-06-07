"""Correctness for the fast direct-load + register-blocked matmul (WS1 C.2).

The fast kernel reaches MLX parity but uses direct simdgroup_load (no masking),
so correctness hinges on the dispatch grid + the per-simdgroup N-guard handling
partial column tiles. These tests exercise the boundary cases the design
flagged: the smallest 32x32 (where 3 of 4 simdgroups are guarded off), N a
multiple of 32 but NOT of 32*rc=128 (the col-guard), rectangular, fp16 + fp32.
"""
import subprocess

import numpy as np
import pytest

try:
    import Metal
    import Foundation
    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS_METAL = False

requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel_fast

_RR = _RC = 4  # the productionized config


def _np_dtype(dtype):
    return np.float16 if dtype in ("fp16", "f16") else np.float32


def _run_fast(dtype, M, N, K, seed=0):
    """Compile + dispatch the fast kernel with its documented grid; return
    (got, ref) as float32 MxN arrays."""
    import struct
    import tempfile
    import os
    src = make_simdgroup_matmul_kernel_fast(dtype=dtype, rr=_RR, rc=_RC)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "k.metal")
    open(p, "w").write(src)
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", p,
                           "-o", d + "/k.air", "-std=metal3.2", "-O2"],
                          stderr=subprocess.DEVNULL)
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", d + "/k.air",
                           "-o", d + "/k.metallib"], stderr=subprocess.DEVNULL)
    dev = Metal.MTLCreateSystemDefaultDevice()
    lib, _ = dev.newLibraryWithURL_error_(
        Foundation.NSURL.fileURLWithPath_(d + "/k.metallib"), None)
    pso, _ = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_("simdgroup_matmul_fast"), None)
    SH = Metal.MTLResourceStorageModeShared
    npt = _np_dtype(dtype)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((M, K)) * 0.1).astype(npt)
    b = (rng.standard_normal((K, N)) * 0.1).astype(npt)

    def buf(arr):
        f = np.ascontiguousarray(arr).flatten()
        bb = dev.newBufferWithLength_options_(f.nbytes, SH)
        bb.contents().as_buffer(f.nbytes)[:] = f.tobytes()
        return bb

    def ub(v):
        bb = dev.newBufferWithLength_options_(4, SH)
        bb.contents().as_buffer(4)[:] = struct.pack("I", v)
        return bb

    C = dev.newBufferWithLength_options_(M * N * 4, SH)
    ng = ((M + 8 * _RR - 1) // (8 * _RR)) * ((N + 32 * _RC - 1) // (32 * _RC))
    q = dev.newCommandQueue()
    cmd = q.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, bb in enumerate([buf(a), buf(b), C, ub(M), ub(N), ub(K)]):
        enc.setBuffer_offset_atIndex_(bb, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(ng, 1, 1), Metal.MTLSizeMake(128, 1, 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    got = np.frombuffer(C.contents().as_buffer(M * N * 4),
                        dtype=np.float32).reshape(M, N)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    return got, ref


@requires_metal
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("M,N,K", [
    (32, 32, 32),      # smallest: ntc=1, only sgitg=0 valid, 1/2/3 guarded off
    (64, 64, 64),
    (128, 128, 128),   # exactly one 32x128 tile
    (256, 256, 128),
    (64, 96, 128),     # N=96: %32 but NOT %128 -> col-guard (sg col0=96 OOB)
    (96, 160, 64),     # rectangular, M%32, N=160 (%32 not %128)
    (128, 256, 512),   # bigger rectangular, deeper K
    (32, 512, 256),    # wide
])
def test_fast_matmul_correct(dtype, M, N, K):
    got, ref = _run_fast(dtype, M, N, K)
    tol = 2e-2 if dtype == "fp16" else 1e-3
    np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)


@requires_metal
def test_fast_matmul_genuine_fp16():
    src = make_simdgroup_matmul_kernel_fast(dtype="fp16")
    assert "simdgroup_half8x8" in src
    import re
    assert not re.search(r"float\(\s*A\[", src)


def test_fast_matmul_rejects_bad_dtype():
    with pytest.raises(ValueError):
        make_simdgroup_matmul_kernel_fast(dtype="fp64")
