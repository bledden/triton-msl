"""De-risk for WS1 Phase C.1 — the HARD GATE.

The entire genuine-fp16 strategy rests on one unproven assumption: that Metal
supports `simdgroup_half8x8` INPUT fragments multiply-accumulated into a
`simdgroup_float8x8` ACCUMULATOR (half x half -> float). This is not in the
public MSL spec, so we prove it compiles AND is numerically correct here
before touching any emitter.

Two cases (the second added per the pre-execution audit):
  1. K=8, single MMA — does the mixed form compile + compute at all?
  2. K=256, K-loop accumulation through HALF threadgroup staging — does the
     real (many-staging-round-trips) path stay correct? A green K=8 with a
     silently-wrong K=256 is exactly the failure mode the audit flagged.

If the xcrun compile fails on simdgroup_half8x8 / the mixed accumulate, STOP:
pivot to the documented fallback (half x half -> half with periodic float
accumulation, or honest docs that fp16 runs fp32). Do not work around it.
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

_SHARED = None  # set lazily


def _shared():
    global _SHARED
    if _SHARED is None:
        _SHARED = Metal.MTLResourceStorageModeShared
    return _SHARED


def _compile(msl_src, name, tmp_path):
    metal_p = str(tmp_path / "k.metal")
    air = str(tmp_path / "k.air")
    lib_p = str(tmp_path / "k.metallib")
    with open(metal_p, "w") as f:
        f.write(msl_src)
    # This compile is the make-or-break for the genuine-fp16 path.
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", metal_p,
                           "-o", air, "-std=metal3.2", "-O2"])
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", air,
                           "-o", lib_p])
    dev = Metal.MTLCreateSystemDefaultDevice()
    lib, err = dev.newLibraryWithURL_error_(
        Foundation.NSURL.fileURLWithPath_(lib_p), None)
    assert err is None, f"load: {err}"
    fn = lib.newFunctionWithName_(name)
    pso, err = dev.newComputePipelineStateWithFunction_error_(fn, None)
    assert err is None, f"pipeline: {err}"
    return dev, pso


def _hbuf(dev, arr):
    flat = np.ascontiguousarray(arr, dtype=np.float16).flatten()
    buf = dev.newBufferWithLength_options_(flat.nbytes, _shared())
    buf.contents().as_buffer(flat.nbytes)[:] = flat.tobytes()
    return buf


def _fout(dev, n):
    return dev.newBufferWithLength_options_(n * 4, _shared())


def _dispatch(dev, pso, buffers, threads):
    q = dev.newCommandQueue()
    cmd = q.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(threads, 1, 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()


_K8_MSL = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void half_mma(device const half* A [[buffer(0)]],
                     device const half* B [[buffer(1)]],
                     device float* C [[buffer(2)]],
                     uint tiitg [[thread_index_in_threadgroup]]) {
    threadgroup half tgA[64], tgB[64];
    for (uint i = tiitg; i < 64u; i += 32u) { tgA[i] = A[i]; tgB[i] = B[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 a, b;            // GENUINE half input fragments
    simdgroup_float8x8 acc(0);          // float accumulator (precision)
    simdgroup_load(a, tgA, 8);
    simdgroup_load(b, tgB, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);   // half x half -> float
    simdgroup_store(acc, C, 8);
}
"""

# 8x8 output, K=256 — exercises K-loop accumulation through HALF staging.
_KLOOP_MSL = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void half_mma_kloop(device const half* A [[buffer(0)]],   // 8 x 256
                           device const half* B [[buffer(1)]],   // 256 x 8
                           device float* C [[buffer(2)]],         // 8 x 8
                           uint tiitg [[thread_index_in_threadgroup]]) {
    threadgroup half tgA[64], tgB[64];
    simdgroup_float8x8 acc(0);
    simdgroup_half8x8 a, b;
    for (uint k = 0u; k < 256u; k += 8u) {
        for (uint i = tiitg; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u;
            tgA[i] = A[r * 256u + (k + c)];   // A[r][k+c]
            tgB[i] = B[(k + r) * 8u + c];      // B[k+r][c]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a, tgA, 8);
        simdgroup_load(b, tgB, 8);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, C, 8);
}
"""


@requires_metal
def test_half_mma_compiles_and_is_correct(tmp_path):
    """K=8 single MMA: does half x half -> float compile and compute?"""
    dev, pso = _compile(_K8_MSL, "half_mma", tmp_path)
    a = (np.arange(64).reshape(8, 8) * 0.1).astype(np.float16)
    b = ((np.arange(64).reshape(8, 8) * 0.1)[::-1].copy()).astype(np.float16)
    A, B, C = _hbuf(dev, a), _hbuf(dev, b), _fout(dev, 64)
    _dispatch(dev, pso, [A, B, C], 32)
    got = np.frombuffer(C.contents().as_buffer(64 * 4),
                        dtype=np.float32).reshape(8, 8)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)


@requires_metal
def test_half_mma_kloop_256_precision(tmp_path):
    """K=256 K-loop through half staging: real-path precision, not just K=8."""
    dev, pso = _compile(_KLOOP_MSL, "half_mma_kloop", tmp_path)
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((8, 256)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((256, 8)) * 0.1).astype(np.float16)
    A, B, C = _hbuf(dev, a), _hbuf(dev, b), _fout(dev, 64)
    _dispatch(dev, pso, [A, B, C], 32)
    got = np.frombuffer(C.contents().as_buffer(64 * 4),
                        dtype=np.float32).reshape(8, 8)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    # fp16 inputs, float accumulation: expect close but not exact.
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


def _uint(dev, v):
    import struct
    b = dev.newBufferWithLength_options_(4, _shared())
    b.contents().as_buffer(4)[:] = struct.pack("I", v)
    return b


@requires_metal
def test_genuine_fp16_full_template_is_numerically_correct(tmp_path):
    """The REAL make_simdgroup_matmul_kernel(fp16) — full 32x32 tile + K-loop +
    boundary — must compute a correct matmul, not just the de-risk's 8x8. This
    is what proves the genuine-fp16 fix is correct end-to-end, not only that it
    compiles."""
    from triton_metal.codegen._msl_templates import make_simdgroup_matmul_kernel
    M = N = 64
    K = 128
    dev, pso = _compile(make_simdgroup_matmul_kernel(dtype="fp16"),
                        "simdgroup_matmul", tmp_path)
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    A, B = _hbuf(dev, a), _hbuf(dev, b)
    C = _fout(dev, M * N)
    Mb, Nb, Kb = _uint(dev, M), _uint(dev, N), _uint(dev, K)
    n_groups = ((M + 31) // 32) * ((N + 31) // 32)
    q = dev.newCommandQueue(); cmd = q.commandBuffer()
    enc = cmd.computeCommandEncoder(); enc.setComputePipelineState_(pso)
    for i, bf in enumerate([A, B, C, Mb, Nb, Kb]):
        enc.setBuffer_offset_atIndex_(bf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1), Metal.MTLSizeMake(128, 1, 1))
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    got = np.frombuffer(C.contents().as_buffer(M * N * 4),
                        dtype=np.float32).reshape(M, N)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)
