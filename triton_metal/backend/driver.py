import os
import platform
import struct
import subprocess
import tempfile

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def ty_to_cpp(ty):
    """Map Triton type strings to C++ type strings for Metal."""
    if ty[0] == "*":
        # Metal uses raw device pointers.
        return "uint64_t"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
    }[ty]


# Two-kernel-split matmul (#159): when a metallib carries both a staged matmul
# kernel and its pure-direct (no-threadgroup) variant, load_binary resolves both
# and records the direct pipeline here, keyed by id() of the primary (staged)
# pipeline. The launcher looks it up to dispatch the direct kernel for
# fully-aligned matmuls (max occupancy / MLX parity).
_MM_DIRECT_PIPELINES = {}


class MetalUtils:
    """Manages Metal device, command queue, and kernel dispatch.

    Supports batched dispatch: multiple kernel encodings share a single
    MTLCommandBuffer, committed once on flush(). This reduces per-kernel
    overhead from ~0.15ms to ~0.05ms for sequences of launches.
    """

    def __init__(self):
        self._device = None
        self._command_queue = None
        self._buffer_pool = None
        # Batched dispatch state
        self._batch_cb = None       # Active MTLCommandBuffer (None = no batch)
        self._batch_count = 0       # Dispatches encoded in current batch
        self._batch_max = 16        # Max dispatches per command buffer
        self._batch_mode = False    # True = defer commit until flush()
        # Deferred copy-backs waiting on current batch
        self._deferred_copies = []  # [(metal_buf, tensor, nbytes, cpu_tensor)]
        self._deferred_releases = []  # [(metal_buf, aligned_mem, size_class) or ("scalar", buf, sz)]

    @property
    def device(self):
        if self._device is None:
            import sys
            import Metal
            from triton_metal.backend.device_detect import get_device_info
            from triton_metal.debug import _debug_level

            self._device = Metal.MTLCreateSystemDefaultDevice()
            if self._device is None:
                raise RuntimeError("No Metal GPU device found")

            # Log device info at debug level 1+.
            if _debug_level() >= 1:
                info = get_device_info()
                print(
                    f"[triton-metal] device: {info.chip_family} {info.chip_variant} "
                    f"| GPU cores: {info.gpu_core_count} "
                    f"| Metal {info.metal_version} "
                    f"| max threads/tg: {info.max_threads_per_threadgroup} "
                    f"| bf16: {info.has_bfloat16} "
                    f"| Metal4: {info.supports_metal4} "
                    f"| neural accel: {info.has_neural_accelerator} "
                    f"| tensor ops: {info.supports_tensor_ops}",
                    file=sys.stderr,
                )
        return self._device

    @property
    def command_queue(self):
        if self._command_queue is None:
            self._command_queue = self.device.newCommandQueue()
        return self._command_queue

    @property
    def buffer_pool(self):
        if self._buffer_pool is None:
            from triton_metal.buffer_pool import MetalBufferPool
            self._buffer_pool = MetalBufferPool(self.device)
        return self._buffer_pool

    @property
    def batch_active(self):
        """True if batched dispatch mode is active."""
        return self._batch_mode

    def begin_batch(self):
        """Start batched dispatch mode.

        Subsequent launch() calls encode into a shared command buffer.
        Call flush() to commit and wait.
        """
        self._batch_mode = True

    def end_batch(self):
        """End batched dispatch mode and flush any pending work."""
        self.flush()
        self._batch_mode = False

    def load_binary(self, name, kernel, shared_mem, device=None):
        """Load a metallib and create a compute pipeline state.

        Uses newLibraryWithURL instead of newLibraryWithData to avoid a
        PyObjC segfault in NSData's interaction with Metal's internal
        SHA256 hashing.

        Args:
            name: kernel function name.
            kernel: metallib bytes (Triton framework) or file path str (legacy).
            shared_mem: bytes of shared memory needed.
            device: ignored (Metal has a single GPU).

        Returns 5-tuple: (library, pipeline_state, n_regs, n_spills, n_max_threads).
        """
        import Foundation

        if isinstance(kernel, (bytes, bytearray)):
            # Triton framework path: write bytes to temp file.
            with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
                f.write(kernel)
                tmp_path = f.name
            url = Foundation.NSURL.fileURLWithPath_(tmp_path)
        else:
            # Legacy path: kernel is a file path string.
            url = Foundation.NSURL.fileURLWithPath_(kernel)

        library, error = self.device.newLibraryWithURL_error_(url, None)
        if error is not None:
            raise RuntimeError(f"Failed to load metallib: {error}")

        function = library.newFunctionWithName_(name)
        if function is None:
            available = [
                library.functionNames().objectAtIndex_(i)
                for i in range(library.functionNames().count())
            ]
            raise RuntimeError(
                f"Kernel '{name}' not found in metallib. Available: {available}"
            )

        pipeline_state, error = (
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if error is not None:
            # Pipeline-state creation can fail for resource reasons
            # (threadgroup memory > 32 KB, register pressure, compiler
            # complexity limit) or genuinely malformed code. Triton\'s
            # autotuner catches ``OutOfResources`` to silently skip
            # configs that don\'t fit; without translating these errors
            # the autotuner aborts on the first oversized config
            # instead of moving on. Detect resource-style failures by
            # the error\'s description and raise ``OutOfResources`` so
            # the autotuner can continue.
            err_str = str(error).lower()
            resource_markers = (
                "internal error",          # AGXMetalG16X Code=3
                "threadgroup memory",
                "out of memory",
                "register",
                "too large",
                "exceeds",
            )
            if any(m in err_str for m in resource_markers):
                try:
                    from triton.runtime.errors import OutOfResources
                    raise OutOfResources(0, 0, f"pipeline state: {error}")
                except ImportError:
                    raise RuntimeError(f"Failed to create pipeline state: {error}")
            raise RuntimeError(f"Failed to create pipeline state: {error}")

        n_max_threads = pipeline_state.maxTotalThreadsPerThreadgroup()
        # Apple doesn\'t expose per-kernel register usage to client code,
        # so we report a conservative ``32`` (one full register file
        # slot per thread) instead of 0. Returning 0 here triggers
        # ``ZeroDivisionError`` in upstream tutorials that compute
        # ``NUM_REGS // (n_regs * ...)`` to estimate occupancy.
        n_regs = 32
        n_spills = 0

        # Two-kernel split (#159): if the metallib also defines a pure-direct
        # matmul variant, resolve its pipeline and stash it so the launcher can
        # dispatch it for fully-aligned matmuls. Cheap no-op for other kernels
        # (newFunctionWithName_ returns None when absent).
        try:
            direct_fn = library.newFunctionWithName_(name + "__mmdirect")
            if direct_fn is not None:
                direct_ps, derr = (
                    self.device.newComputePipelineStateWithFunction_error_(
                        direct_fn, None))
                if derr is None and direct_ps is not None:
                    _MM_DIRECT_PIPELINES[id(pipeline_state)] = direct_ps
        except Exception:
            pass

        return library, pipeline_state, n_regs, n_spills, n_max_threads

    def launch(
        self,
        pipeline_state,
        grid,
        threadgroup_size,
        buffers,
        sync=True,
    ):
        """Dispatch a compute kernel.

        In batch mode, encodes the dispatch into the current command buffer
        without committing. Call flush() to commit and wait.

        Args:
            pipeline_state: MTLComputePipelineState from load_binary.
            grid: (grid_x, grid_y, grid_z) threadgroup counts.
            threadgroup_size: (threads_x, threads_y, threads_z) per threadgroup.
            buffers: list of (MTLBuffer, offset) tuples bound to sequential indices.
            sync: if True and not in batch mode, wait for completion immediately.
        """
        import Metal

        # In batch mode, reuse the current command buffer
        if self._batch_mode:
            if self._batch_cb is None or self._batch_count >= self._batch_max:
                # Need a new command buffer (first dispatch or batch full)
                if self._batch_cb is not None:
                    self._flush_current_batch()
                self._batch_cb = self.command_queue.commandBuffer()
                self._batch_count = 0

            encoder = self._batch_cb.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline_state)
            for i, (buf, offset) in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, offset, i)
            grid_size = Metal.MTLSizeMake(*grid)
            tg_size = Metal.MTLSizeMake(*threadgroup_size)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
            encoder.endEncoding()
            self._batch_count += 1
            return

        # Non-batch mode: immediate dispatch
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        for i, (buf, offset) in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, offset, i)

        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup_size)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()

        if sync:
            command_buffer.waitUntilCompleted()
            status = command_buffer.status()
            if status == Metal.MTLCommandBufferStatusError:
                error = command_buffer.error()
                raise RuntimeError(f"Metal kernel execution failed: {error}")
        else:
            self._batch_cb = command_buffer  # Track for later flush

    def flush(self):
        """Commit and wait on any pending command buffer.

        Processes all deferred copy-backs and releases pool buffers.
        Safe to call when no batch is active (no-op).
        """
        if self._batch_cb is not None:
            self._flush_current_batch()
        self._process_deferred_copies()

    def _flush_current_batch(self):
        """Commit the current batch command buffer and wait."""
        import Metal

        cb = self._batch_cb
        self._batch_cb = None
        self._batch_count = 0

        cb.commit()
        cb.waitUntilCompleted()

        status = cb.status()
        if status == Metal.MTLCommandBufferStatusError:
            error = cb.error()
            raise RuntimeError(f"Metal kernel execution failed: {error}")

    def defer_copy_back(self, tensor_copies, pool_releases):
        """Register copy-back operations to execute on flush()."""
        self._deferred_copies.extend(tensor_copies)
        self._deferred_releases.extend(pool_releases)

    def _process_deferred_copies(self):
        """Execute all deferred copy-backs and release pool buffers."""
        import ctypes

        for entry in self._deferred_copies:
            metal_buf, tensor, nbytes = entry[0], entry[1], entry[2]
            cpu_tensor = entry[3] if len(entry) > 3 else None

            import torch as _torch
            is_f64_downcast = (
                cpu_tensor is not None
                and hasattr(tensor, "dtype")
                and tensor.dtype == _torch.float64
                and hasattr(cpu_tensor, "dtype")
                and cpu_tensor.dtype == _torch.float32
            )

            if is_f64_downcast:
                src_view = metal_buf.contents().as_buffer(nbytes)
                dst = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)
                tensor.copy_(cpu_tensor.double())
            elif cpu_tensor is not None:
                src_view = metal_buf.contents().as_buffer(nbytes)
                dst = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)
                tensor.copy_(cpu_tensor)
                _torch.mps.synchronize()
            else:
                src_view = metal_buf.contents().as_buffer(nbytes)
                dst = (ctypes.c_char * nbytes).from_address(tensor.data_ptr())
                ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)

        self._deferred_copies.clear()

        pool = self.buffer_pool
        for release_entry in self._deferred_releases:
            if release_entry[0] == "scalar":
                pool.release_scalar(release_entry[1], release_entry[2])
            else:
                pool.release(release_entry[0], release_entry[1], release_entry[2])
        self._deferred_releases.clear()

    def make_buffer_from_ptr(self, ptr, nbytes):
        """Create a Metal buffer wrapping an existing pointer (zero-copy UMA).

        Uses a ctypes array (not c_void_p) so PyObjC can validate the
        buffer size for newBufferWithBytesNoCopy.
        """
        import ctypes
        import Metal

        # Wrap pointer as a sized ctypes array so PyObjC accepts it.
        src = (ctypes.c_char * nbytes).from_address(ptr)
        buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            src,
            nbytes,
            Metal.MTLResourceStorageModeShared,
            None,
        )
        if buf is None:
            raise RuntimeError(
                f"Failed to create Metal buffer from pointer {ptr:#x} ({nbytes} bytes)"
            )
        return buf

    def make_buffer(self, nbytes):
        """Allocate a new Metal buffer."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError(f"Failed to allocate Metal buffer ({nbytes} bytes)")
        return buf

    def make_buffer_with_data(self, data, nbytes):
        """Create a Metal buffer by copying data (single-copy via Metal API)."""
        import Metal

        buf = self.device.newBufferWithBytes_length_options_(
            data, nbytes, Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError(
                f"Failed to create Metal buffer with data ({nbytes} bytes)"
            )
        return buf

    def get_device_properties(self, device=0):
        # Estimate GPU core count from device name
        name = self.device.name().lower() if self.device.name() else ""
        if "ultra" in name:
            mp_count = 80
        elif "max" in name:
            mp_count = 40
        elif "pro" in name:
            mp_count = 18
        else:
            mp_count = 10  # M-series base
        return {
            "max_shared_mem": 32768,  # 32 KB threadgroup memory
            # 32K 32-bit registers per SIMD-group on Apple M4 (and
            # similar on M1/M2/M3 — the exact value isn\'t queryable, so
            # we report Apple\'s documented per-SIMD-group register file
            # size). Tutorials use this to compute occupancy as
            # ``NUM_REGS // (n_regs * WARP_SIZE * num_warps)`` and need
            # a non-zero value to avoid ZeroDivisionError.
            "max_num_regs": 32 * 1024,
            "multiprocessor_count": mp_count,
            # Apple\'s per-SIMD-group max resident threads. Used in HIP
            # tutorial branches; harmless to include for Metal too.
            "max_threads_per_sm": 1024,
            # Both spellings — upstream tutorials use ``warpSize`` (camelCase,
            # matches CUDA\'s property name) while internal triton code uses
            # ``warp_size``. Tutorials currently fail otherwise.
            "warp_size": 32,
            "warpSize": 32,
        }

    def unload_module(self, module):
        pass  # Metal libraries are reference-counted by ObjC ARC


_metal_utils = None


def _get_utils():
    """Module-level MetalUtils singleton (Metal device is a system singleton)."""
    global _metal_utils
    if _metal_utils is None:
        _metal_utils = MetalUtils()
    return _metal_utils


_COMPILE_SHADER_RUNTIME = None


def _get_compile_shader_runtime():
    global _COMPILE_SHADER_RUNTIME
    if _COMPILE_SHADER_RUNTIME is None:
        from triton_metal.backend.compile_shader_runtime import CompileShaderRuntime
        _COMPILE_SHADER_RUNTIME = CompileShaderRuntime()
    return _COMPILE_SHADER_RUNTIME


# Scalar Triton signature types that torch.mps.compile_shader binds CORRECTLY
# when passed a raw Python int/float (what the fast-path does). Determined
# empirically (2026-06-14): for each type, a trivial kernel
# `kernel void k(device float* o, constant <T>& v, uint i){ o[i]=(float)v; }`
# was dispatched with a known Python scalar and the result checked.
#   SAFE  : i32, u32, i64, i8, i16, i1 (bool), fp32
#           - i64 binds CORRECTLY even for high-bit values (1<<40 verified);
#             PyTorch passes the full 64-bit value, no 32-bit truncation.
#   UNSAFE: fp16, bf16 — a raw Python float binds as 0.0 (PyTorch writes 4/8
#           fp32/fp64 bytes into a 2-byte half/bfloat slot -> garbage). These
#           are excluded so such kernels fall back to the existing path.
# Anything not in this set (or unresolvable) -> NOT ok -> fall back. Conservative.
_COMPILE_SHADER_SAFE_SCALAR_SIGS = frozenset({
    "i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "fp32",
})


def _compile_shader_scalars_ok(launcher, kargs) -> bool:
    """True iff every NON-tensor arg in ``kargs`` has a declared scalar
    signature type that compile_shader binds correctly (see the SAFE set).

    ``kargs`` is the ordered non-constexpr arg list (the same one the fast-path
    dispatches). The j-th non-constexpr arg maps back to original arg index
    ``orig_idx[j]``; its declared type is ``signature[arg_names[orig_idx[j]]]``.
    Any uncertainty (tuple arg, unresolvable type, unknown/unsafe scalar type)
    -> return False so the launch falls back to the existing path.
    """
    try:
        # Original arg indices of the non-constexpr args, in kargs order.
        # __call__ builds kargs as [a for i,a in enumerate(args)
        # if i not in constexpr_indices]; mirror that index mapping here.
        orig_idx = [i for i in range(len(launcher.arg_names))
                    if i not in launcher.constexpr_indices]
        if len(orig_idx) < len(kargs):
            return False  # can't resolve every karg's declared type -> fall back
        for j, a in enumerate(kargs):
            if hasattr(a, "data_ptr"):
                continue  # tensor (pointer arg) — handled by the MPS check
            # Non-tensor scalar: resolve its declared signature type.
            oi = orig_idx[j]
            name = launcher.arg_names[oi]
            sig = launcher.signature.get(name)
            if isinstance(sig, tuple):
                return False  # tuple/aggregate scalar — uncertain, fall back
            if sig not in _COMPILE_SHADER_SAFE_SCALAR_SIGS:
                return False  # unknown or unsafe scalar type (fp16/bf16/...) -> fall back
        return True
    except Exception:
        return False  # any resolution failure -> conservative fall back


class MetalLauncher:
    """Triton kernel launcher for Metal backend.

    Instantiated by the Triton framework as launcher_cls(src, metadata).
    Called as launcher(gridX, gridY, gridZ, stream, function, kernel_metadata,
                       launch_metadata, launch_enter_hook, launch_exit_hook, *args).
    """

    def __init__(self, src, metadata):
        self.constants = src.constants if hasattr(src, "constants") else {}
        self.arg_names = src.fn.arg_names if hasattr(src, "fn") else []
        self.signature = src.signature if hasattr(src, "signature") else {}
        # Identify constexpr arg indices — these are compiled into the kernel
        # and must NOT be packed as Metal buffers at launch time.
        self.constexpr_indices = set()
        for name, sig in self.signature.items():
            if sig == "constexpr" and name in self.arg_names:
                self.constexpr_indices.add(self.arg_names.index(name))

        # Kernel name for the compile_shader fast-path MSL lookup (Phase 4).
        self.kernel_name = getattr(metadata, "name", None)
        # Resolve the MSL source by the kernel's content hash (msl_hash), NOT by
        # name: inductor reuses kernel names across compiled graphs in one
        # process, so a name lookup can return a DIFFERENT kernel's MSL and the
        # fast-path would dispatch the wrong shader. msl_hash is content-unique.
        # Missing/None -> self._msl is None -> the (always-correct) host path.
        try:
            from triton_metal.backend.compiler import _MSL_BY_KEY
            msl_key = getattr(metadata, "msl_hash", None)
            self._msl = _MSL_BY_KEY.get(msl_key) if msl_key else None
        except Exception:
            self._msl = None

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,  # MTLComputePipelineState from load_binary
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        import ctypes

        if launch_enter_hook:
            # Upstream Triton\'s hook convention is a single
            # ``LaunchMetadata`` argument. Passing both kernel_metadata
            # and launch_metadata trips registered ``hook(launch_metadata)``
            # users with ``TypeError: hook() takes 1 positional argument
            # but 2 were given`` (test_launch::test_metadata).
            launch_enter_hook(launch_metadata)

        utils = _get_utils()

        # Unpack kernel metadata: (num_warps, num_ctas, shared, block_size, output_indices, needs_2d_grid)
        num_warps = kernel_metadata[0] if kernel_metadata else 4
        block_size = kernel_metadata[3] if kernel_metadata and len(kernel_metadata) > 3 else num_warps * 32
        needs_2d_grid = kernel_metadata[5] if kernel_metadata and len(kernel_metadata) > 5 else False

        # compile_shader zero-copy fast-path (Phase 4). Purely additive: any
        # failure or ineligibility falls through to the existing (correct)
        # host-round-trip driver path below. A wrong result here is the one
        # unacceptable outcome, so eligibility is CONSERVATIVE — fire only for
        # the well-understood common case (1-D grid, MPS tensors, safe scalar
        # types) and fall back on anything uncertain. NEVER silent-wrong.
        #
        # The entire attempt (runtime acquisition + checks + dispatch) is inside
        # one try/except: on ANY exception we fall through to the existing path,
        # marking the MSL unsupported only when it is known.
        import os as _os
        fast_matmul = (kernel_metadata[7]
                       if (kernel_metadata and len(kernel_metadata) > 7) else None)
        if ((self._msl is not None or fast_matmul is not None)
                and _os.environ.get("TRITON_METAL_COMPILE_SHADER", "1") != "0"):
            try:
                _rt = _get_compile_shader_runtime()
                if _rt.available():
                    # Ordered non-constexpr args (match [[buffer(i)]] order).
                    kargs = [a for i, a in enumerate(args) if i not in self.constexpr_indices]
                    tensors = [a for a in kargs if hasattr(a, "data_ptr")]
                    all_mps = bool(tensors) and all(
                        getattr(a, "device", None) is not None
                        and str(a.device).startswith("mps") for a in tensors)

                    # --- Fast-matmul runtime dispatch (Phase 4) ---
                    # Dispatch the proven simdgroup fast template ONLY for MPS
                    # tensors with aligned runtime dims. The compiled metallib is
                    # the generic (bounds-checked, row-major) kernel and is the
                    # fallback on ANY miss. M%32 is MANDATORY: otherwise the grid
                    # rounds M up and the no-edge-handling template writes past C
                    # (OOB). N%32 / K%8 are the col-strip / MMA-depth requirements.
                    if (fast_matmul is not None and all_mps
                            and _os.environ.get("TRITON_METAL_FAST_MATMUL", "1") != "0"):
                        try:
                            fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n = fast_matmul
                        except (TypeError, ValueError):
                            fast_msl = None
                        if fast_msl is not None and not _rt.is_unsupported(fast_msl):
                            try:
                                M = int(kargs[m_idx]); N = int(kargs[n_idx]); K = int(kargs[k_idx])
                                if (M > 0 and N > 0 and K > 0
                                        and M % tile_m == 0 and N % 32 == 0 and K % 8 == 0):
                                    import math as _math
                                    n_groups = _math.ceil(M / tile_m) * _math.ceil(N / tile_n)
                                    lib = _rt.get_library(fast_msl)
                                    # The fast template declares exactly 6 buffers
                                    # (A,B,C,M,N,K = kargs[:6]); pass only those so we
                                    # don't rely on compile_shader silently ignoring the
                                    # trailing stride args (which would self-disable the
                                    # fast path if that tolerance ever changed).
                                    _rt.dispatch(lib, "simdgroup_matmul_fast", kargs[:6],
                                                 threads=n_groups * 128, group_size=128)
                                    if launch_exit_hook:
                                        launch_exit_hook(launch_metadata)
                                    return
                            except Exception:
                                # Fast path failed -> mark its MSL unsupported and
                                # fall through to the generic metallib (correct).
                                try:
                                    _rt.mark_unsupported(fast_msl)
                                except Exception:
                                    pass

                    # --- Existing elementwise 1-D-grid fast path (needs self._msl) ---
                    if self._msl is not None and not _rt.is_unsupported(self._msl):
                        # Every NON-tensor scalar arg must have a compile_shader-safe
                        # declared type (fp16/bf16 scalars mis-bind to 0.0). (Fix 4.)
                        scalars_ok = _compile_shader_scalars_ok(self, kargs)
                        # 1-D-grid only: anything needing a 2-D grid (or gridY/gridZ > 1)
                        # falls back to the existing path (correct, just slower).
                        if (all_mps and scalars_ok and not needs_2d_grid
                                and gridY == 1 and gridZ == 1):
                            tg = min(block_size, 1024)
                            threads, group_size = gridX * tg, tg
                            lib = _rt.get_library(self._msl)
                            _rt.dispatch(lib, self.kernel_name, kargs,
                                         threads=threads, group_size=group_size)
                            if launch_exit_hook:
                                launch_exit_hook(launch_metadata)
                            return
            except Exception:
                # Any failure -> mark unsupported (when MSL is known) + fall
                # through to the existing driver path (correct, just slower).
                try:
                    if self._msl is not None:
                        _get_compile_shader_runtime().mark_unsupported(self._msl)
                except Exception:
                    pass

        # Pack arguments into Metal buffers.
        # Strategy:
        # 1. Page-aligned tensors → zero-copy via newBufferWithBytesNoCopy (UMA)
        # 2. Non-aligned tensors → buffer pool (pre-allocated page-aligned memory,
        #    memmove in → zero-copy Metal wrap → kernel → memmove back)
        # 3. Scalars → scalar buffer pool (reusable small buffers)
        PAGE_SIZE = 16384  # ARM64 page size
        pool = utils.buffer_pool
        buffers = []
        tensor_copies = []  # (metal_buf, tensor, nbytes, cpu_tensor, pool_info)
        pool_releases = []  # (metal_buf, aligned_mem, size_class) to release after dispatch

        # Output arg indices: only these need copy-back after dispatch.
        # If not provided (None), copy back all tensors (conservative).
        # output_arg_indices from the lowerer are relative to TTGIR args
        # (which exclude constexpr params). Remap to runtime arg indices.
        # Flatten tuple args recursively so nested tuples (including tuples
        # of tensors / pointers) are marshalled element-by-element. The
        # Triton frontend flattens tuple arguments at the TTGIR level using
        # dot-indexed names (e.g. `Ptrs.0`, `Ptrs.1`), so the launcher must
        # emit one Metal buffer per leaf element. Per-element signatures are
        # pulled from the top-level tuple signature (e.g. `('*fp32',)`).
        def _flatten_arg(arg, sig):
            """Yield (leaf_arg, leaf_sig) pairs in Triton flatten order.

            `sig` may be None (unknown), a scalar string (e.g. 'i32',
            '*fp32', 'constexpr'), or a tuple matching the structure of
            `arg`. Nested tuples on either side are flattened together.
            """
            if isinstance(arg, tuple):
                # Signature may be a tuple of per-element sigs; else None.
                sig_tuple = sig if isinstance(sig, tuple) else None
                for i, elem in enumerate(arg):
                    elem_sig = sig_tuple[i] if (
                        sig_tuple is not None and i < len(sig_tuple)
                    ) else None
                    yield from _flatten_arg(elem, elem_sig)
            else:
                yield arg, sig

        flat_args = []       # list of leaf args in flattened order
        flat_sigs = []       # parallel list of per-leaf signature strings
        flat_origin = []     # original top-level arg index for each leaf
        for orig_idx, arg in enumerate(args):
            sig_ty = None
            if orig_idx < len(self.arg_names):
                sig_ty = self.signature.get(self.arg_names[orig_idx])
            # Fully-constexpr tuple (e.g. `('constexpr',)`) is compiled
            # into the kernel — skip it entirely.
            if (
                isinstance(arg, tuple)
                and isinstance(sig_ty, tuple)
                and all(s == "constexpr" for s in sig_ty)
            ):
                continue
            if orig_idx in self.constexpr_indices:
                continue
            for leaf, leaf_sig in _flatten_arg(arg, sig_ty):
                flat_args.append(leaf)
                flat_sigs.append(leaf_sig)
                flat_origin.append(orig_idx)

        output_arg_indices = None
        if kernel_metadata and len(kernel_metadata) > 4 and kernel_metadata[4] is not None:
            ttgir_indices = kernel_metadata[4]
            # TTGIR position i corresponds directly to flat_args[i] since
            # both exclude constexpr args and both flatten tuple args.
            output_arg_indices = set()
            for ti in ttgir_indices:
                if ti < len(flat_args):
                    output_arg_indices.add(ti)

        for arg_idx, arg in enumerate(flat_args):
            # Per-leaf constexpr entries (e.g. constexpr element inside a
            # mixed tuple) are already compiled into the kernel — skip.
            if flat_sigs[arg_idx] == "constexpr":
                continue
            if hasattr(arg, "data_ptr"):
                import torch as _torch
                is_mps = hasattr(arg, "device") and str(arg.device).startswith("mps")
                # Metal has no float64 — downcast to float32 transparently.
                is_f64 = hasattr(arg, "dtype") and arg.dtype == _torch.float64
                if is_f64:
                    arg_f32 = arg.float()  # float64 → float32
                    nbytes = arg_f32.nelement() * arg_f32.element_size()
                    is_output = (output_arg_indices is None or arg_idx in output_arg_indices)
                    metal_buf, aligned_mem, size_class = pool.acquire(nbytes)
                    src = (ctypes.c_char * nbytes).from_address(arg_f32.data_ptr())
                    dst_view = metal_buf.contents().as_buffer(nbytes)
                    dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                    ctypes.memmove(dst, src, nbytes)
                    buffers.append((metal_buf, 0))
                    pool_releases.append((metal_buf, aligned_mem, size_class))
                    if is_output:
                        tensor_copies.append((metal_buf, arg, nbytes, arg_f32, None))
                    continue
                # TensorWrapper (unsigned int tensors) may lack nelement()
                if hasattr(arg, 'nelement'):
                    nbytes = arg.nelement() * arg.element_size()
                elif hasattr(arg, 'numel'):
                    nbytes = arg.numel() * arg.element_size()
                else:
                    import functools, operator
                    nbytes = functools.reduce(operator.mul, arg.shape, 1) * arg.element_size()
                is_output = (output_arg_indices is None or arg_idx in output_arg_indices)

                if is_mps:
                    # MPS tensors: copy via CPU intermediate to avoid
                    # ctypes.memmove corruption of MPS buffer tracking.
                    import torch
                    torch.mps.synchronize()
                    cpu_tensor = arg.cpu()
                    # Use buffer pool for page-aligned zero-copy
                    metal_buf, aligned_mem, size_class = pool.acquire(nbytes)
                    src = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                    dst_view = metal_buf.contents().as_buffer(nbytes)
                    dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                    ctypes.memmove(dst, src, nbytes)
                    buffers.append((metal_buf, 0))
                    pool_releases.append((metal_buf, aligned_mem, size_class))
                    if is_output:
                        tensor_copies.append((metal_buf, arg, nbytes, cpu_tensor, None))
                else:
                    ptr = arg.data_ptr()
                    page_aligned = (ptr % PAGE_SIZE == 0) and (nbytes % PAGE_SIZE == 0)

                    if page_aligned and nbytes >= PAGE_SIZE:
                        # Zero-copy: wrap existing memory as Metal buffer.
                        buf = utils.make_buffer_from_ptr(ptr, nbytes)
                        buffers.append((buf, 0))
                        # No copy-back needed — same physical memory (UMA).
                    else:
                        # Pool path: acquire page-aligned buffer, memmove in
                        metal_buf, aligned_mem, size_class = pool.acquire(nbytes)
                        src = (ctypes.c_char * nbytes).from_address(ptr)
                        dst_view = metal_buf.contents().as_buffer(nbytes)
                        dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                        ctypes.memmove(dst, src, nbytes)
                        buffers.append((metal_buf, 0))
                        pool_releases.append((metal_buf, aligned_mem, size_class))
                        if is_output:
                            tensor_copies.append((metal_buf, arg, nbytes, None, None))
            elif isinstance(arg, bool):
                buf = pool.acquire_scalar(4)
                view = buf.contents().as_buffer(4)
                struct.pack_into("i", view, 0, int(arg))
                buffers.append((buf, 0))
                pool_releases.append(("scalar", buf, 4))
            elif isinstance(arg, int):
                if arg > 0x7FFFFFFFFFFFFFFF:
                    buf = pool.acquire_scalar(8)
                    view = buf.contents().as_buffer(8)
                    struct.pack_into("Q", view, 0, arg)  # uint64_t
                elif arg < -(1 << 31) or arg > 0xFFFFFFFF:
                    buf = pool.acquire_scalar(8)
                    view = buf.contents().as_buffer(8)
                    struct.pack_into("q", view, 0, arg)  # int64_t
                elif arg < 0:
                    buf = pool.acquire_scalar(4)
                    view = buf.contents().as_buffer(4)
                    struct.pack_into("i", view, 0, arg)  # int32_t (signed)
                else:
                    buf = pool.acquire_scalar(4)
                    view = buf.contents().as_buffer(4)
                    struct.pack_into("I", view, 0, arg)  # uint32_t
                buffers.append((buf, 0))
                sz = 8 if (arg > 0x7FFFFFFFFFFFFFFF or arg < -(1 << 31) or arg > 0xFFFFFFFF) else 4
                pool_releases.append(("scalar", buf, sz))
            elif isinstance(arg, float):
                # Determine the declared scalar float width from the kernel
                # signature so that bf16/fp16 scalar args are marshalled as
                # 2 bytes (matching MSL `half&` / `bfloat&` parameters)
                # instead of being silently packed as 4-byte fp32.
                sig_ty = flat_sigs[arg_idx]
                if sig_ty == "fp16":
                    buf = pool.acquire_scalar(2)
                    view = buf.contents().as_buffer(2)
                    # struct 'e' = IEEE 754 binary16 (fp16)
                    struct.pack_into("e", view, 0, arg)
                    buffers.append((buf, 0))
                    pool_releases.append(("scalar", buf, 2))
                elif sig_ty == "bf16":
                    buf = pool.acquire_scalar(2)
                    view = buf.contents().as_buffer(2)
                    # bf16 = upper 16 bits of fp32 (with round-to-nearest-even).
                    # Using torch's conversion keeps nan/inf/denorm handling
                    # consistent with the rest of the backend.
                    import torch as _torch_bf16
                    bf16_bits = _torch_bf16.tensor(
                        [arg], dtype=_torch_bf16.float32
                    ).to(_torch_bf16.bfloat16).view(_torch_bf16.int16).item()
                    struct.pack_into("h", view, 0, bf16_bits)
                    buffers.append((buf, 0))
                    pool_releases.append(("scalar", buf, 2))
                else:
                    buf = pool.acquire_scalar(4)
                    view = buf.contents().as_buffer(4)
                    struct.pack_into("f", view, 0, arg)
                    buffers.append((buf, 0))
                    pool_releases.append(("scalar", buf, 4))
            elif arg is None:
                # Optional pointer argument (mask, other, scale) — pack as null.
                buf = pool.acquire_scalar(8)
                view = buf.contents().as_buffer(8)
                struct.pack_into("Q", view, 0, 0)
                buffers.append((buf, 0))
                pool_releases.append(("scalar", buf, 8))
            elif isinstance(arg, str):
                # Constexpr string argument — already compiled into kernel.
                continue
            elif hasattr(arg, '__module__') and 'triton' in str(type(arg)):
                # tl.dtype or similar constexpr type — skip.
                continue
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        threads_per_tg = min(block_size, 1024)  # Metal max threads_per_threadgroup
        if needs_2d_grid:
            # Kernel uses program_id(1) or program_id(2) — preserve grid dimensions
            grid = (gridX, gridY, gridZ)
        else:
            # Kernel uses only program_id(0) — flatten to 1D for scalar pid
            grid = (gridX * gridY * gridZ, 1, 1)
        threadgroup_size = (threads_per_tg, 1, 1)

        # Two-kernel split (#159): for a fully-aligned float matmul, dispatch the
        # pure-direct (no-threadgroup) kernel instead of the staged one — same
        # grid, max occupancy. Falls back to the staged kernel on any
        # uncertainty (different size, non-aligned, read error). Other kernels
        # have no descriptor and are unaffected.
        dispatch_fn = function
        mm_two = kernel_metadata[6] if (
            kernel_metadata and len(kernel_metadata) > 6) else None
        if mm_two is not None:
            direct_ps = _MM_DIRECT_PIPELINES.get(id(function))
            if direct_ps is not None:
                try:
                    _M = int(flat_args[mm_two["m_idx"]])
                    _N = int(flat_args[mm_two["n_idx"]])
                    _K = int(flat_args[mm_two["k_idx"]])
                    if (_M % mm_two["block_m"] == 0 and _N % mm_two["block_n"] == 0
                            and _K % 8 == 0):
                        dispatch_fn = direct_ps
                except Exception:
                    dispatch_fn = function

        if utils.batch_active:
            # Batched mode: encode dispatch, defer copy-back until flush()
            utils.launch(dispatch_fn, grid, threadgroup_size, buffers)
            if tensor_copies or pool_releases:
                utils.defer_copy_back(tensor_copies, pool_releases)
        else:
            # Immediate mode: dispatch, wait, copy-back
            utils.launch(dispatch_fn, grid, threadgroup_size, buffers)

            # Copy results back from Metal buffers to tensor memory.
            for entry in tensor_copies:
                metal_buf, tensor, nbytes = entry[0], entry[1], entry[2]
                cpu_tensor = entry[3] if len(entry) > 3 else None

                import torch as _torch
                is_f64_downcast = (
                    cpu_tensor is not None
                    and hasattr(tensor, "dtype")
                    and tensor.dtype == _torch.float64
                    and hasattr(cpu_tensor, "dtype")
                    and cpu_tensor.dtype == _torch.float32
                )

                if is_f64_downcast:
                    src_view = metal_buf.contents().as_buffer(nbytes)
                    dst = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                    ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)
                    tensor.copy_(cpu_tensor.double())
                elif cpu_tensor is not None:
                    src_view = metal_buf.contents().as_buffer(nbytes)
                    dst = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                    ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)
                    tensor.copy_(cpu_tensor)
                    _torch.mps.synchronize()
                else:
                    src_view = metal_buf.contents().as_buffer(nbytes)
                    dst = (ctypes.c_char * nbytes).from_address(tensor.data_ptr())
                    ctypes.memmove(dst, (ctypes.c_char * nbytes).from_buffer(src_view), nbytes)

            # Release pool buffers back to pool for reuse
            for release_entry in pool_releases:
                if release_entry[0] == "scalar":
                    pool.release_scalar(release_entry[1], release_entry[2])
                else:
                    pool.release(release_entry[0], release_entry[1], release_entry[2])

        if launch_exit_hook:
            launch_exit_hook(launch_metadata)


def _detect_metal_arch():
    """Detect the Apple GPU architecture from the Metal device name."""
    try:
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return "apple-unknown"
        name = device.name()
        # e.g. "Apple M4 Max" -> "apple-m4-max"
        return name.lower().replace(" ", "-")
    except (ImportError, Exception):
        return "apple-unknown"


class _MetalTimerEvent:
    """Timer-based event for Metal benchmarking.

    MPS Events have ordering constraints that conflict with
    triton.testing.do_bench's benchmark loop. This uses wall-clock
    timing with MPS synchronization instead.
    """

    def __init__(self, enable_timing=True):
        self._time = None

    def record(self, stream=None):
        import torch
        torch.mps.synchronize()
        import time
        self._time = time.perf_counter()

    def elapsed_time(self, end_event):
        # Return milliseconds
        return (end_event._time - self._time) * 1000.0


class _MetalDeviceInterface:
    """Device interface for Metal, used by triton.testing.do_bench."""

    Event = _MetalTimerEvent

    @staticmethod
    def synchronize(device=None):
        import torch
        torch.mps.synchronize()

    @staticmethod
    def current_device():
        return 0


class MetalDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @classmethod
    def is_active(cls):
        if platform.system() != "Darwin":
            return False
        # Check for Xcode Command Line Tools (xcrun is required for MSL compilation)
        try:
            subprocess.run(
                ["xcrun", "--find", "metal"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            import warnings
            warnings.warn(
                "triton-metal requires Xcode Command Line Tools for MSL compilation. "
                "Install with: xcode-select --install",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        try:
            import Metal

            device = Metal.MTLCreateSystemDefaultDevice()
            return device is not None
        except ImportError:
            return False

    def get_current_target(self):
        arch = _detect_metal_arch()
        return GPUTarget("metal", arch, 32)

    def get_current_device(self):
        return 0  # Metal has a single GPU

    def get_current_stream(self, device=0):
        return 0  # Metal has no CUDA-style streams

    def get_active_torch_device(self):
        import torch

        # Explicit device index so equality comparisons with tensor
        # devices succeed: ``torch.device(\"mps\") != torch.device(\"mps:0\")``
        # but ``torch.rand(..., device=torch.device(\"mps\")).device``
        # returns the ``mps:0`` form. Tutorials commonly assert
        # ``tensor.device == DEVICE`` and would otherwise fail.
        return torch.device("mps", 0)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_device_interface(self):
        return _MetalDeviceInterface

    def get_empty_cache_for_benchmark(self):
        import torch

        # Apple Silicon\'s UMA means CPU and GPU share memory; there\'s no
        # separate L2 cache to evict between benchmark iterations the way
        # discrete NVIDIA GPUs require. Allocate on CPU so we don\'t
        # contend with torch\'s MPS allocator (which can mis-report ``other
        # allocations`` after our zero-copy ``newBufferWithBytesNoCopy``
        # mappings churn and surface as
        # ``RuntimeError: MPS backend out of memory``).
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")

    def clear_cache(self, cache):
        cache.zero_()

    def get_benchmarker(self):
        from triton_metal.profiling.metal_bench import metal_do_bench

        return metal_do_bench
