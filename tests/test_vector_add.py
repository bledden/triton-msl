"""End-to-end tests for Metal compute kernel pipeline.

Tests the full pipeline: MSL -> xcrun metal -> metallib -> Metal GPU -> results.
These tests work standalone (no triton dependency required).
"""

import hashlib
import os
import platform
import struct
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


def _has_metal():
    try:
        import Metal

        return Metal.MTLCreateSystemDefaultDevice() is not None
    except ImportError:
        return False


def _has_metal_compiler():
    try:
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _compile_msl(msl_src, kernel_name):
    """Compile MSL source to a metallib file and return its path."""
    cache_dir = os.path.join(tempfile.gettempdir(), "triton_msl_test_cache")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
    base = f"{kernel_name}_{src_hash}"
    metal_path = os.path.join(cache_dir, f"{base}.metal")
    air_path = os.path.join(cache_dir, f"{base}.air")
    metallib_path = os.path.join(cache_dir, f"{base}.metallib")

    if os.path.exists(metallib_path):
        return metallib_path

    with open(metal_path, "w") as f:
        f.write(msl_src)

    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path, "-o", air_path,
         "-std=metal3.2", "-O2"],
        stderr=subprocess.PIPE,
    )
    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", metallib_path],
        stderr=subprocess.PIPE,
    )
    return metallib_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_metal(), reason="No Metal GPU")
def test_metal_device():
    """Verify Metal device is available and reports expected properties."""
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None
    assert "apple" in device.name().lower()
    assert device.maxThreadgroupMemoryLength() >= 32768


@pytest.mark.skipif(not _has_metal_compiler(), reason="Metal compiler not installed")
def test_msl_compilation():
    """Compile MSL source -> metallib and verify the magic bytes."""
    msl = """\
#include <metal_stdlib>
using namespace metal;
kernel void noop(uint tid [[thread_position_in_grid]]) {}
"""
    path = _compile_msl(msl, "noop")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        magic = f.read(4)
    assert magic == b"MTLB", f"Expected MTLB magic, got {magic}"


@pytest.mark.skipif(not _has_metal(), reason="No Metal GPU")
@pytest.mark.skipif(not _has_metal_compiler(), reason="Metal compiler not installed")
def test_load_and_launch_noop():
    """Load a metallib, launch a noop kernel, verify it completes."""
    import Metal
    import Foundation

    msl = """\
#include <metal_stdlib>
using namespace metal;
kernel void noop(uint tid [[thread_position_in_grid]]) {}
"""
    metallib_path = _compile_msl(msl, "noop")

    device = Metal.MTLCreateSystemDefaultDevice()
    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    assert error is None, f"Load failed: {error}"

    function = library.newFunctionWithName_("noop")
    assert function is not None

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    assert error is None

    queue = device.newCommandQueue()
    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(1, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    # status 4 = MTLCommandBufferStatusCompleted
    assert cmd.status() == 4


@pytest.mark.skipif(not _has_metal(), reason="No Metal GPU")
@pytest.mark.skipif(not _has_metal_compiler(), reason="Metal compiler not installed")
def test_add_one_kernel():
    """Compile and run add_one kernel, verify output[i] = input[i] + 1."""
    import Metal
    import Foundation

    msl = """\
#include <metal_stdlib>
using namespace metal;

kernel void add_one(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        output[tid] = input[tid] + 1.0f;
    }
}
"""
    metallib_path = _compile_msl(msl, "add_one")

    device = Metal.MTLCreateSystemDefaultDevice()
    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    assert error is None

    function = library.newFunctionWithName_("add_one")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    assert error is None

    n = 1024
    input_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    output_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )

    # Fill input
    input_view = input_buf.contents().as_buffer(n * 4)
    for i in range(n):
        struct.pack_into("f", input_view, i * 4, float(i))

    n_view = n_buf.contents().as_buffer(4)
    struct.pack_into("I", n_view, 0, n)

    # Launch
    queue = device.newCommandQueue()
    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 2)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(4, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    # Verify
    output_view = output_buf.contents().as_buffer(n * 4)
    for i in range(n):
        result = struct.unpack_from("f", output_view, i * 4)[0]
        expected = float(i) + 1.0
        assert abs(result - expected) < 1e-6, (
            f"[{i}] got {result}, expected {expected}"
        )


@pytest.mark.skipif(not _has_metal(), reason="No Metal GPU")
@pytest.mark.skipif(not _has_metal_compiler(), reason="Metal compiler not installed")
def test_vector_add_kernel():
    """output = a + b, verified across 2048 elements."""
    import Metal
    import Foundation

    msl = """\
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        output[tid] = a[tid] + b[tid];
    }
}
"""
    metallib_path = _compile_msl(msl, "vector_add")

    device = Metal.MTLCreateSystemDefaultDevice()
    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    assert error is None

    function = library.newFunctionWithName_("vector_add")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    assert error is None

    n = 2048
    a_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    b_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    out_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )

    a_view = a_buf.contents().as_buffer(n * 4)
    b_view = b_buf.contents().as_buffer(n * 4)
    for i in range(n):
        struct.pack_into("f", a_view, i * 4, float(i))
        struct.pack_into("f", b_view, i * 4, float(i) * 2.0)

    n_view = n_buf.contents().as_buffer(4)
    struct.pack_into("I", n_view, 0, n)

    queue = device.newCommandQueue()
    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 3)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(8, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    out_view = out_buf.contents().as_buffer(n * 4)
    for i in range(n):
        result = struct.unpack_from("f", out_view, i * 4)[0]
        expected = float(i) + float(i) * 2.0
        assert abs(result - expected) < 1e-4, (
            f"[{i}] got {result}, expected {expected}"
        )
