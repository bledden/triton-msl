"""Shared test fixtures for Metal GPU tests."""

import hashlib
import os
import platform
import struct
import subprocess
import tempfile

import pytest


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


requires_metal = pytest.mark.skipif(
    not _has_metal() or not _has_metal_compiler(),
    reason="Requires Metal GPU and compiler",
)


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests on non-macOS platforms."""
    if platform.system() != "Darwin":
        skip = pytest.mark.skip(reason="Metal tests require macOS")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _fresh_inductor_cache():
    """Clear PyTorch Inductor's persistent kernel cache ONCE at session start.

    Inductor caches compiled kernels in /var/folders/.../torchinductor_* keyed by ITS own
    hash (Triton source + config) — NOT by triton-msl's CODEGEN_VERSION. So a triton-msl
    lowering change does NOT invalidate inductor's cache, and a torch.compile test can
    silently run kernels a PRIOR session compiled before the change. That masked a
    reduce-classifier regression (it refused inductor's NaN-propagating max -> broke
    softmax/training) behind a green suite. Clearing once per session makes the
    torch.compile tests exercise the CURRENT codegen rather than stale cached kernels.
    """
    try:
        import shutil
        import torch._inductor.runtime.cache_dir_utils as _cdu
        shutil.rmtree(_cdu.cache_dir(), ignore_errors=True)
    except Exception:
        pass
    yield


class MetalKernelRunner:
    """Compile MSL, load metallib, run kernel, read results."""

    def __init__(self):
        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self._cache_dir = os.path.join(
            tempfile.gettempdir(), "triton_msl_test_cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    def compile(self, msl_src, kernel_name):
        """Compile MSL to metallib path."""
        src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
        base = f"{kernel_name}_{src_hash}"
        metal_path = os.path.join(self._cache_dir, f"{base}.metal")
        air_path = os.path.join(self._cache_dir, f"{base}.air")
        metallib_path = os.path.join(self._cache_dir, f"{base}.metallib")

        if not os.path.exists(metallib_path):
            with open(metal_path, "w") as f:
                f.write(msl_src)
            subprocess.check_call(
                ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
                 "-o", air_path, "-std=metal3.2", "-O2"],
                stderr=subprocess.PIPE,
            )
            subprocess.check_call(
                ["xcrun", "-sdk", "macosx", "metallib", air_path,
                 "-o", metallib_path],
                stderr=subprocess.PIPE,
            )
        return metallib_path

    def load(self, metallib_path, kernel_name):
        """Load metallib and create pipeline state."""
        import Foundation

        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        library, error = self.device.newLibraryWithURL_error_(url, None)
        assert error is None, f"Load failed: {error}"

        function = library.newFunctionWithName_(kernel_name)
        assert function is not None, f"Kernel '{kernel_name}' not found"

        pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(
                function, None
            )
        )
        assert error is None, f"Pipeline failed: {error}"
        return pipeline

    def make_float_buffer(self, data):
        """Create a Metal buffer filled with float data."""
        import Metal

        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * 4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * 4)
        for i, val in enumerate(data):
            struct.pack_into("f", view, i * 4, float(val))
        return buf

    def make_empty_buffer(self, n):
        """Create an empty float buffer of n elements."""
        import Metal

        return self.device.newBufferWithLength_options_(
            n * 4, Metal.MTLResourceStorageModeShared
        )

    def make_uint_buffer(self, value):
        """Create a buffer with a single uint32."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(4)
        struct.pack_into("I", view, 0, value)
        return buf

    def make_int_buffer(self, value):
        """Create a buffer with a single int32."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(4)
        struct.pack_into("i", view, 0, value)
        return buf

    def make_float_scalar_buffer(self, value):
        """Create a buffer with a single float."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(4)
        struct.pack_into("f", view, 0, value)
        return buf

    def run(self, pipeline, buffers, n_elements, block_size=256):
        """Dispatch a compute kernel."""
        import Metal

        n_groups = (n_elements + block_size - 1) // block_size

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_groups, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    def read_float_buffer(self, buf, n):
        """Read n floats from a Metal buffer."""
        view = buf.contents().as_buffer(n * 4)
        return [struct.unpack_from("f", view, i * 4)[0] for i in range(n)]

    # -- Half-precision (FP16) helpers --

    def make_half_buffer(self, data):
        """Create a Metal buffer filled with half-precision (FP16) data."""
        import Metal

        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * 2)
        for i, val in enumerate(data):
            struct.pack_into("e", view, i * 2, float(val))
        return buf

    def make_empty_half_buffer(self, n):
        """Create an empty half-precision buffer of n elements."""
        import Metal

        return self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )

    def read_half_buffer(self, buf, n):
        """Read n half-precision values from a Metal buffer."""
        view = buf.contents().as_buffer(n * 2)
        return [struct.unpack_from("e", view, i * 2)[0] for i in range(n)]

    # -- BFloat16 helpers --

    @staticmethod
    def _float_to_bf16_bytes(f):
        """Convert float to bfloat16 bytes (truncate lower 16 bits of float32)."""
        b = struct.pack("f", f)
        return b[2:4]  # upper 2 bytes on little-endian

    @staticmethod
    def _bf16_bytes_to_float(b):
        """Convert bfloat16 bytes to float."""
        return struct.unpack("f", b"\x00\x00" + b)[0]

    def make_bf16_buffer(self, data):
        """Create a Metal buffer filled with bfloat16 data."""
        import Metal

        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * 2)
        for i, val in enumerate(data):
            bf16 = self._float_to_bf16_bytes(float(val))
            view[i * 2] = bf16[0]
            view[i * 2 + 1] = bf16[1]
        return buf

    def make_empty_bf16_buffer(self, n):
        """Create an empty bfloat16 buffer of n elements."""
        import Metal

        return self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )

    def read_bf16_buffer(self, buf, n):
        """Read n bfloat16 values from a Metal buffer."""
        view = buf.contents().as_buffer(n * 2)
        results = []
        for i in range(n):
            b = bytes([view[i * 2], view[i * 2 + 1]])
            results.append(self._bf16_bytes_to_float(b))
        return results


@pytest.fixture
def metal_device():
    """Provide a Metal device for tests that need one."""
    if not _has_metal():
        pytest.skip("No Metal GPU available")
    import Metal
    return Metal.MTLCreateSystemDefaultDevice()


@pytest.fixture
def runner():
    """Provide a MetalKernelRunner for GPU tests."""
    if not _has_metal() or not _has_metal_compiler():
        pytest.skip("Requires Metal GPU and compiler")
    return MetalKernelRunner()
