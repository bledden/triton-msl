import os
import platform
import struct
import subprocess
import tempfile
from types import BuiltinFunctionType

# Only the ordinary builtin consumer may omit the per-launch generator.
_standard_all = (
    all if type(all) is BuiltinFunctionType and all.__module__ == "builtins" and all.__name__ == "all" else None
)

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _release_host_pool_buffers(pool, entries):
    """Release every registered lease without hiding an active launch exception."""
    import sys

    # sys.exc_info() is the Python 3.10 spelling of 3.11's sys.exception().
    active_error = sys.exc_info()[1]
    cleanup_error = None
    for entry in entries:
        try:
            if type(entry[0]) is str and entry[0] == "scalar":
                pool.release_scalar(entry[1], entry[2])
            else:
                pool.release(*entry)
        except Exception as error:
            if active_error is not None:
                # BaseException.add_note is 3.11+; on 3.10 the note is skipped
                # and the active exception still propagates unchanged.
                add_note = getattr(active_error, "add_note", None)
                if add_note is not None:
                    add_note(f"Host buffer cleanup also failed: {error}")
            elif cleanup_error is None:
                cleanup_error = error
    if cleanup_error is not None:
        raise cleanup_error


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


# Loaders own Python acquisition records. Global discovery owns no loader or
# Metal pipeline. The Python buckets/records, not PyObjC objects, support weakrefs.
_PIPELINE_BUCKETS = {}


# PyObjC pipelines do not support weak references. Receipts own their pipeline
# and library until the loader's matching unload_module calls (or utils teardown).
# They are scoped to MetalUtils, not a process-global pipeline side table.
def _bind_pipeline_binary(pipeline, name, binary, *, owner=None, module=None, direct=None, loaded=False):
    import hashlib
    from ._pipeline_ownership import Registry

    proven = type(binary) is bytes and type(name) is str
    if not proven and not loaded:
        return
    owner = _get_utils() if owner is None else owner
    module = pipeline if module is None else module
    proof = (pipeline, name, hashlib.sha256(binary).hexdigest(), module) if proven else None
    registry = getattr(owner, "_pipeline_ownership", None)
    if registry is None:
        prepared = Registry()
        registry = getattr(owner, "_pipeline_ownership", None)
        if registry is None:
            owner._pipeline_ownership = registry = prepared
    return registry.bind(_PIPELINE_BUCKETS, id(pipeline), pipeline, module, proof, direct)


def _direct_pipeline(pipeline):
    bucket = _PIPELINE_BUCKETS.get(id(pipeline))
    record = None if bucket is None else bucket.latest("direct")
    return record.direct if record is not None and record.primary is pipeline else None


def _validate_packed_pipeline(pipeline, name, digest, *, owner=None):
    from ._argument_abi import LayoutError

    bucket = _PIPELINE_BUCKETS.get(id(pipeline))
    registry = None if owner is None else getattr(owner, "_pipeline_ownership", None)
    record = None if bucket is None or (owner is not None and registry is None) else bucket.latest("binary", registry)
    entry = None if record is None else record.proof
    if (
        entry is None
        or record.primary is not pipeline
        or entry[0] is not pipeline
        or entry[1] != name
        or entry[2] != digest
    ):
        raise LayoutError("host packed argument ABI does not match the loaded pipeline; recompile")


def _strided_storage_reach(t):
    """Number of elements from a tensor's storage_offset to its last addressable
    element + 1 — i.e. how many elements a stride-faithful device buffer must hold.

    For a contiguous tensor this equals ``numel``. For a non-contiguous view (a
    column slice / transpose / dilation) the last element lives at
    ``sum((dim-1) * stride)``, which can EXCEED numel — a numel-sized buffer would
    truncate it. Used so the metallib copy path mirrors the full strided extent
    (the kernel addresses with the user's runtime strides). Returns ``numel`` on any
    error (the safe contiguous default).
    """
    try:
        shape = list(t.shape)
        strides = list(t.stride())
        if not shape:
            return 1
        reach = 1
        for dim, st in zip(shape, strides):
            if dim > 0:
                reach += (dim - 1) * abs(int(st))
        numel = 1
        for d in shape:
            numel *= d
        return max(reach, numel)
    except Exception:
        try:
            return t.numel()
        except Exception:
            return 0


def _batched_dot_host_bounds_reason(descriptor, kargs, grid):
    """Return why a batched-dot address escapes its backing storage, else ``None``.

    The compile-time descriptor supplies the exact affine A/B/C address
    contracts.  Evaluate their extrema with the runtime strides and each
    argument's storage offset before host dispatch.  The storage-faithful host
    marshaller can replay addresses outside the logical view, but never outside
    the allocation that it mirrors.
    """

    try:
        if not (
            isinstance(descriptor, (tuple, list))
            and len(descriptor) == 12
            and descriptor[0] == "batched_dot_host_bounds_v1"
        ):
            return "the batched-dot host-bounds descriptor is missing or malformed"
        batch, m, n, k = (int(value) for value in descriptor[1:5])
        pid_tiled = bool(descriptor[5])
        tensor_indices = tuple(int(value) for value in descriptor[6:9])
        stride_specs = descriptor[9:12]
        gx, gy, gz = (int(value) for value in grid)
        if min(batch, m, n, k, gx, gy, gz) <= 0:
            return "the batched-dot launch has a non-positive extent or grid dimension"
        row_extent = m * gx if pid_tiled else m
        col_extent = n * gy if pid_tiled else n

        def _runtime_stride(spec):
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise ValueError("malformed stride reference")
            if spec[0] == "arg":
                return int(kargs[int(spec[1])])
            if spec[0] == "literal":
                return int(spec[1])
            raise ValueError("unknown stride reference")

        roles = (
            ("A", tensor_indices[0], stride_specs[0], (batch, row_extent, k)),
            ("B", tensor_indices[1], stride_specs[1], (batch, k, col_extent)),
            ("C", tensor_indices[2], stride_specs[2], (batch, row_extent, col_extent)),
        )
        for role, tensor_index, specs, extents in roles:
            tensor = kargs[tensor_index]
            layout = tensor if hasattr(tensor, "stride") else getattr(tensor, "base", None)
            if (
                layout is None
                or not hasattr(layout, "shape")
                or not hasattr(layout, "element_size")
                or not hasattr(layout, "storage_offset")
                or not hasattr(layout, "untyped_storage")
            ):
                return f"batched-dot {role} tensor layout cannot be inspected safely"
            strides = tuple(_runtime_stride(spec) for spec in specs)
            if len(strides) != 3:
                return f"batched-dot {role} stride descriptor is malformed"
            lower = 0
            upper = 0
            for extent, stride in zip(extents, strides):
                delta = (extent - 1) * stride
                lower += min(0, delta)
                upper += max(0, delta)

            elem_size = int(layout.element_size())
            storage_nbytes = int(layout.untyped_storage().nbytes())
            base_byte = int(layout.storage_offset()) * elem_size
            lower_byte = base_byte + lower * elem_size
            upper_byte_exclusive = base_byte + (upper + 1) * elem_size
            if elem_size <= 0 or storage_nbytes <= 0:
                return f"batched-dot {role} backing storage is empty or malformed"
            if lower_byte < 0:
                return (
                    f"batched-dot {role} forms byte offset {lower_byte} before its backing storage; "
                    "the host-roundtrip path cannot preserve this runtime stride"
                )
            if upper_byte_exclusive > storage_nbytes:
                return (
                    f"batched-dot {role} reaches byte {upper_byte_exclusive - 1} past its "
                    f"{storage_nbytes}-byte backing storage; the host-roundtrip path cannot "
                    "preserve this runtime stride"
                )
        return None
    except Exception:
        return "the batched-dot runtime address bounds could not be proven"


def _matmul_tail_bounds_reason(descriptor, kargs, signatures=None, scalar_payloads=None, grid=None):
    """Return a refusal reason for an unmasked rounded-K access plan, else None."""
    try:
        grid_plan = (
            type(descriptor) in (tuple, list)
            and len(descriptor) == 7
            and descriptor[0] in ("matmul_grid_storage_v1", "matmul_grid_storage_v2")
        )
        if not (
            grid_plan
            or (
                type(descriptor) in (tuple, list)
                and len(descriptor) == 6
                and descriptor[0] == "matmul_full_k_storage_v1"
            )
        ):
            return "the K-tail allocation descriptor is missing or malformed; mask the K tail"
        block_k, tensor_indices, extent_refs, stride_refs, arithmetic = descriptor[1:6]
        if (
            type(block_k) is not int
            or block_k <= 0
            or type(tensor_indices) not in (tuple, list)
            or len(tensor_indices) != 3
            or type(extent_refs) not in (tuple, list)
            or len(extent_refs) != 3
            or type(stride_refs) not in (tuple, list)
            or len(stride_refs) != 6
            or type(arithmetic) not in (tuple, list)
            or len(arithmetic) != 3
        ):
            return "the K-tail allocation descriptor is malformed; mask the K tail"
        limits = {
            "i32": (-(1 << 31), (1 << 31) - 1),
            "u32": (0, (1 << 32) - 1),
            "i64": (-(1 << 63), (1 << 63) - 1),
            "u64": (0, (1 << 64) - 1),
        }

        def _value(ref):
            if type(ref) not in (tuple, list) or len(ref) != 3 or ref[2] not in limits:
                raise ValueError("malformed scalar reference")
            if ref[0] == "arg":
                index = int(ref[1])
                if signatures is not None and scalar_payloads is not None:
                    from triton_msl.backend._launch_signature import scalar_bytes

                    captured = scalar_payloads[index]
                    if (
                        signatures[index] != ref[2]
                        or type(captured) is not bytes
                        or scalar_bytes(kargs[index], signatures[index]) != captured
                    ):
                        raise ValueError("runtime scalar changed after its captured launch payload")
                    # __int__ on an int subclass can disagree with the bytes
                    # packed by the launch ABI. Prove the value the GPU reads.
                    value = int.from_bytes(captured, "little", signed=ref[2].startswith("i"))
                elif type(kargs[index]) in (int, bool):
                    value = kargs[index]
                else:
                    from triton_msl.backend._launch_signature import scalar_bytes

                    raw = scalar_bytes(kargs[index], ref[2])
                    value = int.from_bytes(raw, "little", signed=ref[2].startswith("i"))
            else:
                value = int(ref[1]) if ref[0] == "literal" else None
            if value is None or not limits[ref[2]][0] <= value <= limits[ref[2]][1]:
                raise ValueError("runtime scalar is outside its declared source width")
            return value

        # Legacy fully-masked plans need only their rounded-K obligation. A
        # grid plan proves the source's complete domain even for aligned K.
        k = _value(extent_refs[2])
        if not grid_plan and (k <= 0 or k % block_k == 0):
            return None
        rounded_k = ((max(k, 0) + block_k - 1) // block_k) * block_k
        if rounded_k > (1 << 63) - 1:
            return "rounded K exceeds the signed source address range; mask the K tail"
        strides = tuple(_value(ref) for ref in stride_refs)
        if grid_plan:
            from ._matmul_domain import matmul_grid_rectangles, validate_matmul_no_wrap_moduli

            source_plan = descriptor[6]
            identity_refs = None
            if descriptor[0] == "matmul_grid_storage_v2":
                if type(source_plan) not in (tuple, list) or len(source_plan) != 8:
                    return "the matmul no-wrap source grid descriptor is malformed"
                identity_refs = source_plan[7]
                source_plan = source_plan[:7]
            rectangles = matmul_grid_rectangles(source_plan, grid, _value)
            full_m, full_n = max(r[1] for r in rectangles), max(r[3] for r in rectangles)
            load_refs, full_k = source_plan[5:]

            def clipped(ref, full):
                return full if ref is None else min(full, max(0, _value(ref)))

            store_m, store_n = (clipped(ref, full) for ref, full in zip(extent_refs[:2], (full_m, full_n)))
            if descriptor[0] == "matmul_grid_storage_v2":
                stored_rectangles = tuple((m0, min(m1, store_m), n0, min(n1, store_n)) for m0, m1, n0, n1 in rectangles)
                try:
                    validate_matmul_no_wrap_moduli(identity_refs, stored_rectangles, _value)
                except ValueError as exc:
                    return str(exc)
            load_m, load_n = (clipped(ref, full) for ref, full in zip(load_refs, (full_m, full_n)))
            load_k = rounded_k if full_k else max(k, 0)
            role_ranges = [((0, load_m), (0, load_k)), ((0, load_k), (0, load_n))]
            role_ranges.extend(((m0, min(m1, store_m)), (n0, min(n1, store_n))) for m0, m1, n0, n1 in rectangles)
        else:
            m, n = (_value(ref) for ref in extent_refs[:2])
            role_ranges = [
                ((0, max(m, 0)), (0, rounded_k)),
                ((0, rounded_k), (0, max(n, 0))),
                ((0, max(m, 0)), (0, max(n, 0))),
            ]
        roles = tuple(
            (
                ("A", "B", "C")[min(i, 2)],
                tensor_indices[min(i, 2)],
                ranges,
                strides[2 * min(i, 2) : 2 * min(i, 2) + 2],
                arithmetic[min(i, 2)],
            )
            for i, ranges in enumerate(role_ranges)
        )
        signed_min, signed_max = -(1 << 63), (1 << 63) - 1
        for role, tensor_index, ranges, role_strides, arith in roles:
            if any(hi <= lo for lo, hi in ranges):
                continue
            if (
                type(arith) not in (tuple, list)
                or len(arith) != 3
                or arith[0] not in ("i32", "i64")
                or arith[1] not in ("i32", "i64")
                or type(arith[2]) is not bool
            ):
                return f"{role} source address widths are malformed; mask the K tail"
            term_ranges = []
            for (start, end), stride, width in zip(ranges, role_strides, arith[:2]):
                endpoints = (start * stride, (end - 1) * stride)
                width_min, width_max = limits[width]
                lo, hi = min(endpoints), max(endpoints)
                if lo < width_min or hi > width_max:
                    return f"{role} source {width} address term can wrap; mask the K tail"
                term_ranges.append((lo, hi))
            if arith[2]:
                fused_min = sum(item[0] for item in term_ranges)
                fused_max = sum(item[1] for item in term_ranges)
                width_min, width_max = limits[arith[0]]
                if fused_min < width_min or fused_max > width_max:
                    return f"{role} fused source address can wrap; mask the K tail"
            tensor = kargs[int(tensor_index)]
            layout = tensor if hasattr(tensor, "untyped_storage") else getattr(tensor, "base", None)
            if layout is None or any(
                not hasattr(layout, name) for name in ("element_size", "storage_offset", "untyped_storage", "data_ptr")
            ):
                return f"{role} backing storage cannot be inspected; mask the K tail"
            lower = upper = 0
            for lo, hi in term_ranges:
                if not signed_min <= lo <= hi <= signed_max:
                    return f"{role} source address arithmetic can overflow signed 64-bit; mask the K tail"
                lower += lo
                upper += hi
                if not signed_min <= lower <= signed_max or not signed_min <= upper <= signed_max:
                    return f"{role} source address sum can overflow signed 64-bit; mask the K tail"
            if not hasattr(tensor, "element_size"):
                return f"{role} dispatched pointer element width cannot be inspected; mask the K tail"
            elem_size = int(tensor.element_size())
            if elem_size != int(layout.element_size()):
                return f"{role} wrapper/base element widths disagree; mask the K tail"
            storage = layout.untyped_storage()
            storage_nbytes = int(storage.nbytes())
            base_byte = int(layout.storage_offset()) * elem_size
            if (
                elem_size <= 0
                or storage_nbytes <= 0
                or base_byte < 0
                or int(tensor.data_ptr()) != int(storage.data_ptr()) + base_byte
            ):
                return f"{role} tensor/storage identity cannot be proven; mask the K tail"
            lower_byte = base_byte + lower * elem_size
            upper_byte = base_byte + (upper + 1) * elem_size
            if lower_byte < 0 or upper_byte > storage_nbytes:
                return (
                    f"{role} full-block K tail reaches outside its {storage_nbytes}-byte backing allocation; "
                    "pad the backing allocation or mask the K tail"
                )
        return None
    except Exception:
        return "the K-tail runtime allocation bounds could not be proven; mask the K tail"


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
        from ._pipeline_ownership import Registry

        self._pipeline_ownership = Registry()

    @property
    def device(self):
        if self._device is None:
            import sys
            import Metal
            from triton_msl.backend.device_detect import get_device_info
            from triton_msl.debug import _debug_level

            self._device = Metal.MTLCreateSystemDefaultDevice()
            if self._device is None:
                raise RuntimeError("No Metal GPU device found")

            # Log device info at debug level 1+.
            if _debug_level() >= 1:
                info = get_device_info()
                print(
                    f"[triton-msl] device: {info.chip_family} {info.chip_variant} "
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
            from triton_msl.buffer_pool import MetalBufferPool

            self._buffer_pool = MetalBufferPool(self.device)
        return self._buffer_pool

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
            available = [library.functionNames().objectAtIndex_(i) for i in range(library.functionNames().count())]
            raise RuntimeError(f"Kernel '{name}' not found in metallib. Available: {available}")

        pipeline_state, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if error is not None:
            # Free-form diagnostics (including "internal error" or "exceeds")
            # do not prove resource exhaustion. Translating them would let the
            # autotuner hide compiler defects. Only typed, numeric capacity
            # checks upstream are prunable; unclassified pipeline errors stay loud.
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
        direct_pipeline = None
        try:
            direct_fn = library.newFunctionWithName_(name + "__mmdirect")
            if direct_fn is not None:
                direct_ps, derr = self.device.newComputePipelineStateWithFunction_error_(direct_fn, None)
                if derr is None and direct_ps is not None:
                    direct_pipeline = direct_ps
        except Exception:
            pass

        result = (library, pipeline_state, n_regs, n_spills, n_max_threads)
        _bind_pipeline_binary(
            pipeline_state, name, kernel, owner=self, module=library, direct=direct_pipeline, loaded=True
        )
        return result

    def launch(
        self,
        pipeline_state,
        grid,
        threadgroup_size,
        buffers,
        sync=True,
    ):
        """Dispatch a compute kernel and (by default) wait for completion.

        Args:
            pipeline_state: MTLComputePipelineState from load_binary.
            grid: (grid_x, grid_y, grid_z) threadgroup counts.
            threadgroup_size: (threads_x, threads_y, threads_z) per threadgroup.
            buffers: list of (MTLBuffer, offset) tuples bound to sequential indices.
            sync: if True, wait for completion immediately.
        """
        import Metal

        from ._host_dispatch import RetainedBuffers

        lease = buffers.lease if isinstance(buffers, RetainedBuffers) else None
        if lease is not None:
            lease.stage = "encoding"
        command_buffer = self.command_queue.commandBuffer()
        if lease is not None:
            lease.command = command_buffer
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        for i, (buf, offset) in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, offset, i)

        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup_size)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        if lease is not None:
            # commit may enqueue before the Python/ObjC call raises.
            lease.stage = "submission_attempted"
        command_buffer.commit()

        if sync:
            command_buffer.waitUntilCompleted()
            status = command_buffer.status()
            if status == Metal.MTLCommandBufferStatusError:
                error = command_buffer.error()
                raise RuntimeError(f"Metal kernel execution failed: {error}")

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
            raise RuntimeError(f"Failed to create Metal buffer from pointer {ptr:#x} ({nbytes} bytes)")
        return buf

    def make_buffer(self, nbytes):
        """Allocate a new Metal buffer."""
        import Metal

        buf = self.device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise RuntimeError(f"Failed to allocate Metal buffer ({nbytes} bytes)")
        return buf

    def make_buffer_with_data(self, data, nbytes):
        """Create a Metal buffer by copying data (single-copy via Metal API)."""
        import Metal

        buf = self.device.newBufferWithBytes_length_options_(data, nbytes, Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise RuntimeError(f"Failed to create Metal buffer with data ({nbytes} bytes)")
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
        registry = getattr(self, "_pipeline_ownership", None)
        if registry is not None:
            registry.unload(_PIPELINE_BUCKETS, module)


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
        from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

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
_COMPILE_SHADER_SAFE_SCALAR_SIGS = frozenset(
    {
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "fp32",
    }
)


def _compile_shader_scalars_ok(launcher, kargs, *, checked=None) -> bool:
    """True iff every NON-tensor arg in ``kargs`` has a declared scalar
    signature type that compile_shader binds correctly (see the SAFE set).

    ``kargs`` is the ordered non-constexpr arg list (the same one the fast-path
    dispatches). The j-th non-constexpr arg maps back to original arg index
    ``orig_idx[j]``; its declared type is ``signature[arg_names[orig_idx[j]]]``.
    Any uncertainty (tuple arg, unresolvable type, unknown/unsafe scalar type)
    -> return False so the launch falls back to the existing path.
    """
    global _SCALAR_GATE
    if _SCALAR_GATE is False:
        _SCALAR_GATE = _select_scalar_gate()
    try:
        if _SCALAR_GATE is not None:
            _fast_scalar = _SCALAR_GATE(launcher, kargs, checked)
            if _fast_scalar is not None:
                return _fast_scalar
        # Original arg indices of the non-constexpr args, in kargs order.
        # __call__ builds kargs as [a for i,a in enumerate(args)
        # if i not in constexpr_indices]; mirror that index mapping here.
        orig_idx = [i for i in range(len(launcher.arg_names)) if i not in launcher.constexpr_indices]
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
            # A declaration alone does not determine the raw Python argument's
            # representation in compile_shader. In particular, int/bool supplied
            # to an explicitly fp32 IRSource would bind integer bits, not 1.0f.
            if sig == "fp32" and type(a) is not float:
                return False
            # The raw bridge casts Python integers to signed int64, even for
            # a u64 declaration. Keep the full unsigned range on the host path.
            # This must precede reuse of the binder's declared-width proof.
            if sig != "fp32" and (type(a) not in (int, bool) or not -(1 << 63) <= a < (1 << 63)):
                return False
            # The binder already packed this invocation's live scalar. Reuse
            # that proof only for the same exact immutable value, declaration
            # and source position, as observed HERE after hooks/runtime calls.
            # Changed declarations, positions or custom values still repack.
            if (
                checked is not None
                and type(sig) is str
                and (type(a) is int or type(a) is bool or type(a) is float)
                and j < len(checked[0])
                and checked[0][j] is a
                and checked[1][j] is sig
                and checked[2][j] == oi
                and type(checked[3][j]) is bytes
            ):
                continue
            from triton_msl.backend._launch_signature import scalar_bytes

            scalar_bytes(a, sig)  # prove representability, never truncate by guess
        return True
    except Exception:
        return False  # any resolution failure -> conservative fall back


def _select_scalar_gate():
    from ._launch_signature import _binder_native

    if _binder_native is None:
        return None
    configure = getattr(_binder_native, "configure_scalar", None)
    if configure is None:
        return None
    gate = getattr(_binder_native, "scalar_gate", None)
    for entry, name in ((configure, "configure_scalar"), (gate, "scalar_gate")):
        if type(entry) is not BuiltinFunctionType or entry.__self__ is not _binder_native or entry.__name__ != name:
            return None
    return gate if configure(_compile_shader_scalars_ok) else None


_SCALAR_GATE = False


_MAX_METAL_BUFFERS = 31
_HOST_MIRROR_MAX_BYTES = 1 << 30


def _pack_overflow_scalars(kargs):
    """Argument-buffer packing (GitHub issue #4.7): a kernel with > 31 args has its runtime
    SCALARS bundled by the emitter into one ``constant uint*`` buffer after the pointers.
    Mirror that here for compile_shader dispatch: return ``[pointer tensors..., packed]``
    where ``packed`` is one int32 tensor holding each scalar's 32-bit representation, in the
    SAME (arg) order the emitter unpacks -- ints by value, floats by bit pattern (the kernel
    ``as_type<>``-casts back). Pointers keep their own buffers."""
    import struct

    import torch

    tensors = [a for a in kargs if hasattr(a, "data_ptr")]
    scalars = [a for a in kargs if not hasattr(a, "data_ptr")]
    bits = []
    for s in scalars:
        if isinstance(s, bool):
            bits.append(1 if s else 0)
        elif isinstance(s, float):
            bits.append(struct.unpack("<i", struct.pack("<f", s))[0])  # float bits -> int32
        else:
            bits.append(int(s) & 0xFFFFFFFF if int(s) < 0 else int(s))
    dev = tensors[0].device if tensors else "mps"
    packed = torch.tensor([_i32(b) for b in bits], dtype=torch.int32, device=dev)
    return tensors + [packed]


def _i32(v):
    """Wrap a Python int into signed 32-bit range so torch.int32 accepts the bit pattern."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


class MetalLauncher:
    """Triton kernel launcher for Metal backend.

    Instantiated by the Triton framework as launcher_cls(src, metadata).
    Called as launcher(gridX, gridY, gridZ, stream, function, kernel_metadata,
                       launch_metadata, launch_enter_hook, launch_exit_hook, *args).
    """

    def __init__(self, src, metadata):
        from triton_msl.backend._cache_contract import validate_execution_contract, install_jit_policy_guard

        self._execution_contract = validate_execution_contract(getattr(metadata, "execution_contract", None))
        from triton_msl.backend._launch_contract import validate_launch_products

        self._packed_contract, _argument_abi = validate_launch_products(metadata)
        from ._argument_abi import capture_host_layout

        self._host_packed_layout, self._host_packed_binary_sha = capture_host_layout(_argument_abi)
        install_jit_policy_guard(getattr(src, "fn", None), self._execution_contract)
        self.constants = src.constants if hasattr(src, "constants") else {}
        from triton_msl.backend._launch_signature import ordered_source_signature, make_binding_plan

        self.arg_names, self.signature = ordered_source_signature(src)
        self._binding_plan = make_binding_plan(self.arg_names, self.signature)
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
        # _MSL_BY_KEY stores (msl_src, msl_block_size): the MSL's OWN threadgroup
        # size, which compile_shader must use — the C++ LLVM path may have
        # clobbered metadata["block_size"] with a different (one-thread-per-element)
        # value meant for its host metallib launch, which would mis-launch the
        # stashed (MEPT) MSL here.
        self._msl_block_size = None
        try:
            # In-memory stash first, then the persistent disk stash — the latter
            # is what keeps the zero-copy fast-path alive when inductor restores a
            # compiled kernel from its own cache without re-running make_msl (which
            # is the only thing that fills the in-memory _MSL_BY_KEY). Keyed by the
            # content-unique cache_key, so a hit is the exact MSL for this kernel.
            from triton_msl.backend.compiler import _load_stashed_msl

            msl_key = getattr(metadata, "msl_hash", None)
            _stashed = _load_stashed_msl(msl_key)
            if _stashed is not None:
                self._msl, self._msl_block_size = _stashed
            else:
                self._msl = None
        except Exception:
            self._msl = None

        # Field 9 is the existing sealed attention-descriptor union. The
        # experimental sibling never replaces the scalar source in its stash.
        import json

        _attention = json.loads(self._packed_contract)[9]
        if type(_attention) is dict and _attention.get("kind") == "cooperative-row-v1":
            from ._attention_variant import validate_descriptor

            validate_descriptor(
                _attention,
                primary=self.kernel_name,
                scalar_source=self._msl,
                scalar_threads=self._msl_block_size if self._msl is not None else 32,
            )

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
        # This also runs for an already-resident JIT handle whose outer lookup
        # bypassed backend.hash(). It precedes launch hooks and runtime/pipeline
        # access. Packed-descriptor/argument ABI validation is a separate layer.
        from triton_msl.backend._cache_contract import validate_execution_contract

        from triton_msl.backend import _host_extent

        _host_extent.clear_launch()
        validate_execution_contract(getattr(self, "_execution_contract", None))
        from triton_msl.backend._launch_contract import validate_packed_launch

        kernel_metadata = validate_packed_launch(kernel_metadata, getattr(self, "_packed_contract", None))
        from triton_msl.backend._launch_signature import bind_arguments, bind_arguments_with_plan

        binding_plan = getattr(self, "_binding_plan", None)
        if binding_plan is None:
            flat_args, flat_sigs, flat_origin, scalar_payloads = bind_arguments(args, self.arg_names, self.signature)
        else:
            flat_args, flat_sigs, flat_origin, scalar_payloads = bind_arguments_with_plan(
                args, self.arg_names, self.signature, binding_plan
            )
        if kernel_metadata[4] is not None and any(i >= len(flat_args) for i in kernel_metadata[4]):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError("source signature: output index is outside the constexpr-free argument list")
        import ctypes

        if launch_enter_hook:
            # Upstream Triton\'s hook convention is a single
            # ``LaunchMetadata`` argument. Passing both kernel_metadata
            # and launch_metadata trips registered ``hook(launch_metadata)``
            # users with ``TypeError: hook() takes 1 positional argument
            # but 2 were given`` (test_launch::test_metadata).
            launch_enter_hook(launch_metadata)

        # Run after the enter hook and before runtime acquisition or either
        # dispatch route, using the immutable checked descriptor snapshot.
        address_bounds = kernel_metadata[10]
        tail_access_bounds = batched_dot_bounds = None
        validated_tail_bounds = None
        if address_bounds is not None:
            if not isinstance(address_bounds, (tuple, list)) or not address_bounds:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing launch: packed address-bounds contract is malformed",
                    op_name="tt.dot",
                )
            if address_bounds[0] == "source_program_pending_v1":
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing launch: source-program numerical, storage and dispatch admission is not qualified",
                    op_name="tt.dot",
                )
            if address_bounds[0] in ("matmul_full_k_storage_v1", "matmul_grid_storage_v1", "matmul_grid_storage_v2"):
                tail_access_bounds = address_bounds
            elif address_bounds[0] == "batched_dot_host_bounds_v1":
                batched_dot_bounds = address_bounds
            else:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing launch: packed address-bounds contract has an unknown discriminator",
                    op_name="tt.dot",
                )
        if tail_access_bounds is not None:
            _tail_reason = _matmul_tail_bounds_reason(
                tail_access_bounds, flat_args, flat_sigs, scalar_payloads, (gridX, gridY, gridZ)
            )
            if _tail_reason is not None:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing unmasked partial-K matmul launch: " + _tail_reason,
                    op_name="tt.dot",
                )
            # Evidence belongs only to this checked metadata snapshot and live
            # invocation. Never remember successful bounds on the launcher.
            validated_tail_bounds = tail_access_bounds
        if batched_dot_bounds is not None:
            _bounds_reason = _batched_dot_host_bounds_reason(batched_dot_bounds, flat_args, (gridX, gridY, gridZ))
            if _bounds_reason is not None:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing batched dot on the host-roundtrip launch path: " + _bounds_reason,
                    op_name="tt.dot",
                )

        utils = _get_utils()

        # Unpack kernel metadata: (num_warps, num_ctas, shared, block_size, output_indices, needs_2d_grid)
        num_warps = kernel_metadata[0] if kernel_metadata else 4
        block_size = kernel_metadata[3] if kernel_metadata and len(kernel_metadata) > 3 else num_warps * 32
        needs_2d_grid = kernel_metadata[5] if kernel_metadata and len(kernel_metadata) > 5 else False

        # compile_shader zero-copy fast-path (Phase 4). A PRE-invocation
        # failure or ineligibility falls through to the existing (correct)
        # host-round-trip driver path below. A wrong result here is the one
        # unacceptable outcome, so eligibility is CONSERVATIVE — fire only for
        # the well-understood common case (1-D grid, MPS tensors, safe scalar
        # types) and fall back on anything uncertain. NEVER silent-wrong.
        #
        # The entire attempt (runtime acquisition + checks + dispatch) is inside
        # one try/except, with monotonic state shared by every nested helper.
        # After attempted invocation, exceptions cannot trigger fallback/replay,
        # even when a return callback raises after the helper's own try completes.
        import os as _os
        import torch as _torch
        from triton_msl.autotuning._submission import SubmissionState

        _submission = SubmissionState()
        fast_matmul = kernel_metadata[7] if (kernel_metadata and len(kernel_metadata) > 7) else None
        quant_matmul = kernel_metadata[8] if (kernel_metadata and len(kernel_metadata) > 8) else None
        flash_attention = kernel_metadata[9] if (kernel_metadata and len(kernel_metadata) > 9) else None
        # FAIL-CLOSED for quantized: the compiled kernel IS the fast dequant kernel,
        # which the host-roundtrip path below cannot dispatch correctly. So a quantized
        # launch must be handled by dispatch_quant_matmul (compile_shader) or REFUSED —
        # start "unhandled" and clear it only on a successful dispatch (the `return`).
        _quant_unhandled = quant_matmul is not None
        # FAIL-CLOSED for MLA too: the 'mla' descriptor's kernel (concatenated qk=192)
        # has a DIFFERENT ABI than the @jit kernel, so the host path can't run it. A
        # symmetric ('flash_attention') descriptor is fail-OPEN (host path is correct).
        _mla_unhandled = (
            isinstance(flash_attention, (tuple, list)) and len(flash_attention) and flash_attention[0] == "mla"
        )
        if (
            self._msl is not None or fast_matmul is not None or quant_matmul is not None or flash_attention is not None
        ) and _os.environ.get("TRITON_MSL_COMPILE_SHADER", "1") != "0":
            try:
                _rt = _get_compile_shader_runtime()
                if _rt.available():
                    # Ordered non-constexpr args (match [[buffer(i)]] order).
                    kargs = [a for i, a in enumerate(args) if i not in self.constexpr_indices]
                    tensors = [a for a in kargs if hasattr(a, "data_ptr")]
                    all_mps = bool(tensors)
                    if all_mps:
                        if _standard_all is not None and all is _standard_all:
                            # Keep each live type/device read in its original order.
                            # TensorWrapper is still ineligible; no dispatch is retried.
                            try:
                                for a in tensors:
                                    if not (isinstance(a, _torch.Tensor) and str(a.device).startswith("mps")):
                                        all_mps = False
                                        break
                            except StopIteration as error:
                                # The original generator applies PEP 479 here.
                                raise RuntimeError("generator raised StopIteration") from error
                        else:
                            # A custom consumer retains its lazy iterator and result.
                            all_mps = all(
                                isinstance(a, _torch.Tensor) and str(a.device).startswith("mps") for a in tensors
                            )

                    # --- Quantized-matmul dispatch (compile_shader-only) ---
                    # Route a recognized weight-only int8 matmul to the fast dequant
                    # kernel. On success we return; otherwise _quant_unhandled stays
                    # True and the launch is refused after this block (never fall
                    # through to the host path, which would mis-run the fast kernel).
                    if quant_matmul is not None and all_mps and _os.environ.get("TRITON_MSL_QUANT_MATMUL", "1") != "0":
                        from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul

                        if dispatch_quant_matmul(
                            _rt,
                            quant_matmul,
                            kargs,
                            grid=(gridX, gridY, gridZ),
                            launch_exit_hook=launch_exit_hook,
                            launch_metadata=launch_metadata,
                            submission_state=_submission,
                        ):
                            return

                    # --- Fast-matmul runtime dispatch (Phase 4) ---
                    # Dispatch the proven simdgroup fast template ONLY for MPS
                    # tensors with aligned runtime dims.  Logic lives in
                    # triton_msl.autotuning._fast_matmul_dispatch so it can be
                    # unit-tested without triggering Triton backend discovery.
                    if fast_matmul is not None and all_mps and _os.environ.get("TRITON_MSL_FAST_MATMUL", "1") != "0":
                        from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul

                        if dispatch_fast_matmul(
                            _rt,
                            fast_matmul,
                            kargs,
                            grid=(gridX, gridY, gridZ),
                            launch_exit_hook=launch_exit_hook,
                            launch_metadata=launch_metadata,
                            submission_state=_submission,
                        ):
                            return

                    # --- FlashAttention dispatch (compile_shader zero-copy, 2-D grid) ---
                    # The simdgroup FA kernel's 2-D grid disqualifies it from the 1-D
                    # fast path below, so it would otherwise fall to the host-roundtrip
                    # path (~2.5-4.2x slower; the whole gap is dispatch overhead, not the
                    # kernel). Route it via compile_shader with its native grid. FAIL-OPEN:
                    # unlike quantized, the host path is equally correct, so a miss (False)
                    # simply falls through -- never wrong, never refused.
                    if flash_attention is not None and all_mps and _os.environ.get("TRITON_MSL_FA_FAST", "1") != "0":
                        if (
                            self._msl is not None
                            and type(flash_attention) is dict
                            and flash_attention.get("kind") == "cooperative-row-v1"
                        ):
                            from ._cooperative_dispatch import dispatch_cooperative_attention

                            if dispatch_cooperative_attention(
                                _rt,
                                flash_attention,
                                kargs,
                                (gridX, gridY, gridZ),
                                device_limits=(
                                    utils.device.maxThreadsPerThreadgroup().width,
                                    utils.device.maxThreadgroupMemoryLength(),
                                ),
                                submission_state=_submission,
                                launch_exit_hook=launch_exit_hook,
                                launch_metadata=launch_metadata,
                            ):
                                return
                        from triton_msl.autotuning._fa_dispatch import dispatch_flash_attention

                        if dispatch_flash_attention(
                            _rt,
                            flash_attention,
                            self.kernel_name,
                            kargs,
                            gridX,
                            gridY,
                            gridZ,
                            launch_exit_hook=launch_exit_hook,
                            launch_metadata=launch_metadata,
                            submission_state=_submission,
                        ):
                            return

                    # --- Existing elementwise 1-D-grid fast path (needs self._msl) ---
                    # FAIL-CLOSED kernels must NOT take this path: a quantized (or MLA)
                    # kernel's self._msl has a template ABI the positional 1-thread-per-
                    # element dispatch mis-binds (re-review 2026-08-25: a quant launch the
                    # dequant dispatch declined — e.g. the M/N-swap bounds gate — fell
                    # through HERE when its grid was 1-D and ran the template mis-bound:
                    # garbage output instead of the refusal below).
                    if (
                        self._msl is not None
                        and not _rt.is_unsupported(self._msl)
                        and not _quant_unhandled
                        and not _mla_unhandled
                    ):
                        # Every NON-tensor scalar arg must have a compile_shader-safe
                        # declared type (fp16/bf16 scalars mis-bind to 0.0). (Fix 4.)
                        scalars_ok = _compile_shader_scalars_ok(
                            self, kargs, checked=(flat_args, flat_sigs, flat_origin, scalar_payloads)
                        )
                        _generic_geometry = None
                        if all_mps and scalars_ok and needs_2d_grid:
                            from ._generic_launch import generic_2d_geometry

                            # The direct positional list must be identical to
                            # the binder's constexpr-free leaves. Aggregates
                            # and argument-buffer packing keep the host route.
                            if len(kargs) == len(flat_args) and all(a is b for a, b in zip(kargs, flat_args)):
                                _generic_geometry = generic_2d_geometry(
                                    self,
                                    kargs,
                                    flat_sigs,
                                    kernel_metadata,
                                    (gridX, gridY, gridZ),
                                    validated_tail_bounds=validated_tail_bounds,
                                )
                        if (
                            all_mps
                            and scalars_ok
                            and ((not needs_2d_grid and gridY == 1 and gridZ == 1) or _generic_geometry is not None)
                        ):
                            # The stashed MSL's OWN threadgroup size — NOT the
                            # metadata block_size, which the C++ LLVM path may have
                            # clobbered with a value meant for its host metallib
                            # (mis-launches MEPT MSL kernels here -> wrong results).
                            tg = min(self._msl_block_size or block_size, 1024)
                            threads, group_size = gridX * tg, tg
                            if _generic_geometry is not None:
                                threads, group_size = _generic_geometry
                            lib = _rt.get_library(self._msl)
                            # >31 args -> the MSL packs scalars into one buffer; pack the
                            # dispatch args to match (issue #4.7).
                            _dk = _pack_overflow_scalars(kargs) if len(kargs) > _MAX_METAL_BUFFERS else kargs
                            _assert_desc = kernel_metadata[11]
                            if _assert_desc is not None:
                                from ._device_assert import check_binding, check_failure
                                import torch as _assert_torch

                                check_binding(_dk, _assert_desc)
                                # Fresh per invocation; neither a failed launch nor
                                # another concurrent launch can poison this flag.
                                _assert_flag = _assert_torch.zeros(1, dtype=_assert_torch.int32, device="mps")
                                _dk = list(_dk) + [_assert_flag]
                            # A coordinate-preserving matmul twin shares this entry's grid
                            # and buffers; aligned extents run it at its own group width in
                            # place of the staged entry, as the host route below chooses it.
                            _entry = self.kernel_name
                            if _generic_geometry is not None and kernel_metadata[6] is not None:
                                from ._generic_launch import direct_twin

                                _twin = direct_twin(kernel_metadata[6], kargs, gridX * gridY * gridZ)
                                if _twin:
                                    _entry, _twin_width = _twin
                                    threads = (threads[0] // group_size[0] * _twin_width, *threads[1:])
                                    group_size = (_twin_width, *group_size[1:])
                            _submission.begin()
                            _rt.dispatch(lib, _entry, _dk, threads=threads, group_size=group_size)
                            if _assert_desc is not None:
                                check_failure(int(_assert_flag.cpu().item()), _assert_desc)
                            if launch_exit_hook:
                                launch_exit_hook(launch_metadata)
                            return
            except Exception as _error:
                _submission.reraise_if_attempted(_error)
                # Only a pre-invocation failure can mark unsupported and fall
                # through to the existing driver path (correct, just slower).
                try:
                    if self._msl is not None:
                        _get_compile_shader_runtime().mark_unsupported(self._msl)
                except Exception:
                    pass

        # A batched-dot descriptor that did not return through zero-copy
        # compile_shader is about to use the host-roundtrip mirror. Prove that
        # every affine A/B/C offset formed by the runtime grid and strides lies
        # inside the complete backing allocation the storage-faithful marshaller
        # will mirror below.  Offsets outside the logical view are valid; offsets
        # outside the allocation still refuse before any unsafe dispatch.
        # Quantized matmul is compile_shader-only: if it wasn't dispatched above
        # (non-MPS, compile_shader unavailable, opt-out, or a shape the edge-free
        # fast kernel can't tile), REFUSE. The compiled self._msl is the fast dequant
        # kernel; running it via the host-roundtrip path below would mis-dispatch it.
        if _quant_unhandled:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "quantized weight-only matmul runs only on MPS tensors via "
                "compile_shader, with dims M % 32 == 0, N % 16 == 0, K % 32 == 0 (the "
                "fast dequant kernel has no edge handling). This launch is non-MPS, has "
                "compile_shader disabled/unavailable, has non-conforming dims, or its "
                "launch grid differs from the kernel's program mapping (the quantized "
                "template computes the FULL output, so it stands in only for a launch of "
                "exactly cdiv(M, BM) x cdiv(N, BN) programs; a partial or padded grid is "
                "refused). Pad the dims / launch the full grid, or dequantize the weight "
                "on the host and pass a float/half weight to a normal matmul."
            )

        # MLA (nope/rope) attention is compile_shader-only: the 'mla' descriptor's
        # concatenated qk=192 kernel has a different ABI than this @jit kernel, so the
        # host path can't run it. If it wasn't dispatched above (non-MPS, compile_shader
        # unavailable/opt-out), REFUSE rather than mis-run.
        if _mla_unhandled:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "MLA (nope/rope) attention runs only on MPS tensors via compile_shader "
                "(the concatenated qk=head_dim kernel). This launch is non-MPS or has "
                "compile_shader disabled/unavailable."
            )

        # Overflow is admitted only by the fresh producer's explicit MSL ABI,
        # not by argument count or attention-family name. Geometry stays native.
        _n_nonconstexpr = len(flat_args)
        _packed_layout = None
        _packed_payload = None
        if _n_nonconstexpr > _MAX_METAL_BUFFERS:
            from ._argument_abi import prepare_host_payload

            _packed_layout = getattr(self, "_host_packed_layout", None)
            _packed_payload = prepare_host_payload(_packed_layout, flat_sigs, scalar_payloads, flat_args)
            _validate_packed_pipeline(function, self.kernel_name, self._host_packed_binary_sha)

        from . import _host_dispatch

        _host_dispatch.check_pending()

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
        pool_releases = []  # Register each successful acquisition before using its bytes.

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
        # Source-ordered leaves were bound and validated before launch hooks.
        # Nested constexpr leaves are absent, exactly like TTGIR arguments.

        output_arg_indices = None
        if kernel_metadata and len(kernel_metadata) > 4 and kernel_metadata[4] is not None:
            ttgir_indices = kernel_metadata[4]
            # TTGIR position i corresponds directly to flat_args[i] since
            # both exclude constexpr args and both flatten tuple args.
            output_arg_indices = set()
            for ti in ttgir_indices:
                if ti < len(flat_args):
                    output_arg_indices.add(ti)

        # Preserve STORAGE identity on the host-roundtrip path.  A tensor view's
        # data_ptr starts at its storage_offset, but a Triton kernel may legally
        # use a runtime stride to reach other elements of the SAME allocation.
        # Mirroring only [view_base, view_base + reach(view)) therefore drops
        # before-base / past-view bytes.  Packing each argument independently
        # also destroys aliases, so store/load ordering and multiple-output
        # semantics can change even when every address lies inside a view.
        #
        # Group ordinary tensors and TensorWrapper.base objects by their actual
        # backing storage.  Each group gets ONE full-storage Metal mirror, and
        # each argument binds that buffer at its own byte storage_offset.  One
        # group-level copy-back then preserves aliases and kernel store order.
        # Float64 remains on its established conversion path below: Metal has no
        # double, so its device buffer has a different element width and cannot
        # share this raw-byte mirror.
        storage_groups = {}
        storage_bindings = {}
        float64_storage_owners = set()
        from triton_msl.errors import MetalNonRecoverableError

        def _storage_refusal(reason):
            raise MetalNonRecoverableError(
                "Refusing host-roundtrip tensor marshalling: " + reason,
                op_name="kernel",
            )

        for arg_idx, arg in enumerate(flat_args):
            if _packed_layout is not None and arg_idx not in _packed_layout.pointer_indices:
                continue
            if flat_sigs[arg_idx] == "constexpr" or not hasattr(arg, "data_ptr"):
                continue
            import torch as _torch

            if _packed_layout is not None:
                from ._argument_abi import pointee_storage_width

                # A low-level caller can supply a tensor whose storage width
                # disagrees with its specialization. Prove width on the actual
                # backing owner, including reinterpret wrappers, before copying.
                _decl = _packed_layout.declarations[arg_idx]
                _owner = arg if hasattr(arg, "untyped_storage") else getattr(arg, "base", None)
                try:
                    _actual_width = int(_owner.element_size())
                except Exception:
                    _storage_refusal(f"packed pointer {arg_idx} has no proved backing element width")
                if _actual_width != pointee_storage_width(_decl):
                    _storage_refusal(f"packed pointer {arg_idx} backing width differs from its source declaration")
                _converts_fp64 = hasattr(arg, "dtype") and arg.dtype == _torch.float64
                if (_decl == "*fp64") != _converts_fp64:
                    _storage_refusal(f"packed pointer {arg_idx} has no matching fp64 conversion contract")

            if hasattr(arg, "dtype") and arg.dtype == _torch.float64:
                # The established fp64 compatibility path converts a dense
                # argument to a separate fp32 buffer.  It cannot preserve an
                # offset/non-contiguous backing span or aliases because element
                # widths differ.  Keep the supported full-contiguous case, but
                # fail closed on shapes that would otherwise silently lose
                # before-base bytes or storage identity.
                layout = arg if hasattr(arg, "untyped_storage") else getattr(arg, "base", None)
                try:
                    storage = layout.untyped_storage()
                    storage_nbytes = int(storage.nbytes())
                    storage_ptr = int(storage.data_ptr())
                    logical_nbytes = int(arg.nelement()) * int(arg.element_size())
                    full_contiguous = (
                        bool(arg.is_contiguous())
                        and int(arg.storage_offset()) == 0
                        and logical_nbytes == storage_nbytes
                    )
                except Exception:
                    full_contiguous = False
                    storage_ptr = 0
                    storage_nbytes = 0
                if not full_contiguous:
                    _storage_refusal(
                        f"float64 tensor argument {arg_idx} must cover one complete contiguous storage; "
                        "Metal's fp32 conversion path cannot preserve a partial or strided fp64 backing span"
                    )
                f64_key = (
                    "mps" if hasattr(layout, "device") and str(layout.device).startswith("mps") else "host",
                    storage_ptr,
                    storage_nbytes,
                )
                if f64_key in float64_storage_owners or f64_key in storage_groups:
                    _storage_refusal(
                        "aliased float64 arguments cannot preserve storage identity through separate fp32 conversions"
                    )
                float64_storage_owners.add(f64_key)
                continue
            layout = arg if hasattr(arg, "untyped_storage") else getattr(arg, "base", None)
            if layout is None or not hasattr(layout, "untyped_storage") or not hasattr(layout, "storage_offset"):
                _storage_refusal(
                    f"tensor argument {arg_idx} exposes no inspectable backing storage; "
                    "the host path cannot preserve view bounds or aliases"
                )
            try:
                storage = layout.untyped_storage()
                storage_nbytes = int(storage.nbytes())
                storage_ptr = int(storage.data_ptr())
                elem_size = int(layout.element_size())
                byte_offset = int(layout.storage_offset()) * elem_size
                is_mps = hasattr(layout, "device") and str(layout.device).startswith("mps")
                full_storage_view = (
                    byte_offset == 0
                    and hasattr(layout, "is_contiguous")
                    and bool(layout.is_contiguous())
                    and hasattr(layout, "numel")
                    and int(layout.numel()) * elem_size == storage_nbytes
                )
                if storage_nbytes <= 0 or elem_size <= 0:
                    continue
                if byte_offset < 0 or byte_offset >= storage_nbytes:
                    _storage_refusal(
                        f"tensor argument {arg_idx} has byte offset {byte_offset} outside "
                        f"its {storage_nbytes}-byte backing storage"
                    )
            except MetalNonRecoverableError:
                raise
            except Exception as error:
                _storage_refusal(f"tensor argument {arg_idx}'s backing storage cannot be inspected ({error})")
            key = ("mps" if is_mps else "host", storage_ptr, storage_nbytes)
            if key in float64_storage_owners:
                _storage_refusal(
                    "aliased float64 arguments cannot preserve storage identity through separate fp32 conversions"
                )
            group = storage_groups.setdefault(
                key,
                {
                    "layout": layout,
                    "storage": storage,  # retain the live owner through synchronous completion
                    "storage_nbytes": storage_nbytes,
                    "storage_ptr": storage_ptr,
                    "elem_size": elem_size,
                    "is_mps": is_mps,
                    "members": [],
                    "is_output": False,
                },
            )
            # One backing storage can be observed through dtype reinterpretation.
            # The full mirror is raw bytes, so different element sizes are fine;
            # each binding offset is calculated in that view's own element units.
            group["members"].append((arg_idx, byte_offset, full_storage_view))
            if output_arg_indices is None or arg_idx in output_arg_indices:
                group["is_output"] = True

        from triton_msl.backend import _cache_contract as _extent_contract

        extent_state = _extent_contract.effective_policy()["HOST_EXTENT"]
        extent_query = getattr(_extent_contract._validation_native, "host_extent", None)
        pool_reusable = True
        try:
            for group in storage_groups.values():
                layout = group["layout"]
                storage_nbytes = group["storage_nbytes"]
                storage_ptr = group["storage_ptr"]
                elem_size = group["elem_size"]
                single_complete_view = len(group["members"]) == 1 and group["members"][0][2]
                if storage_nbytes > _HOST_MIRROR_MAX_BYTES and not single_complete_view:
                    _storage_refusal(
                        f"a tensor alias group needs its full {storage_nbytes}-byte backing storage, "
                        f"exceeding the {_HOST_MIRROR_MAX_BYTES}-byte safety limit"
                    )
                if storage_nbytes % elem_size != 0:
                    _storage_refusal(
                        f"a {storage_nbytes}-byte backing storage is not divisible by its "
                        f"{elem_size}-byte representative element size"
                    )

                if group["is_mps"]:
                    # Copy the COMPLETE MPS allocation to a CPU byte-faithful mirror.
                    # as_strided(..., storage_offset=0) deliberately ignores the
                    # representative view's base and exposes the whole typed storage.
                    try:
                        import torch as _torch

                        _torch.mps.synchronize()
                        storage_elems = storage_nbytes // elem_size
                        flat_storage = layout.as_strided((storage_elems,), (1,), 0)
                        source = flat_storage.cpu().contiguous()
                        if source.numel() * source.element_size() != storage_nbytes:
                            _storage_refusal("the complete MPS backing storage could not be mirrored byte-for-byte")
                    except MetalNonRecoverableError:
                        raise
                    except Exception as error:
                        _storage_refusal(f"the complete MPS backing storage could not be mirrored ({error})")
                    metal_buf, aligned_mem, size_class = pool.acquire(storage_nbytes)
                    pool_releases.append((metal_buf, aligned_mem, size_class))
                    src = (ctypes.c_char * storage_nbytes).from_address(source.data_ptr())
                    dst_view = metal_buf.contents().as_buffer(storage_nbytes)
                    dst = (ctypes.c_char * storage_nbytes).from_buffer(dst_view)
                    ctypes.memmove(dst, src, storage_nbytes)
                    if group["is_output"]:
                        tensor_copies.append(
                            (
                                metal_buf,
                                layout,
                                storage_nbytes,
                                None,
                                ("whole_storage_mps", elem_size),
                            )
                        )
                else:
                    # CPU tensors can expose their complete storage directly when it
                    # satisfies Metal's page-wrapping contract; otherwise use one
                    # pooled raw-byte mirror for the whole alias group.
                    route, borrow_nbytes, reason = _host_extent.select(
                        group["storage"],
                        storage_ptr,
                        storage_nbytes,
                        extent_state,
                        extent_query,
                        PAGE_SIZE,
                        owner=layout,
                    )
                    if extent_state is not False:
                        _host_extent.record(
                            {
                                "route": route,
                                "reason": reason,
                                "storage_nbytes": storage_nbytes,
                                "borrow_nbytes": borrow_nbytes if route in ("borrow", "existing") else 0,
                                "members": tuple(member[0] for member in group["members"]),
                            },
                            launch_metadata,
                        )
                    if route == "refuse":
                        _storage_refusal(reason)
                    if route in ("borrow", "existing"):
                        metal_buf = utils.make_buffer_from_ptr(storage_ptr, borrow_nbytes)
                    else:
                        metal_buf, aligned_mem, size_class = pool.acquire(storage_nbytes)
                        pool_releases.append((metal_buf, aligned_mem, size_class))
                        src = (ctypes.c_char * storage_nbytes).from_address(storage_ptr)
                        dst_view = metal_buf.contents().as_buffer(storage_nbytes)
                        dst = (ctypes.c_char * storage_nbytes).from_buffer(dst_view)
                        ctypes.memmove(dst, src, storage_nbytes)
                        if group["is_output"]:
                            tensor_copies.append(
                                (
                                    metal_buf,
                                    layout,
                                    storage_nbytes,
                                    None,
                                    ("whole_storage_host", storage_ptr),
                                )
                            )
                for arg_idx, byte_offset, _full_storage_view in group["members"]:
                    storage_bindings[arg_idx] = (metal_buf, byte_offset)

            for arg_idx, arg in enumerate(flat_args):
                if _packed_layout is not None and arg_idx not in _packed_layout.pointer_indices:
                    continue
                # Per-leaf constexpr entries (e.g. constexpr element inside a
                # mixed tuple) are already compiled into the kernel — skip.
                if flat_sigs[arg_idx] == "constexpr":
                    continue
                if hasattr(arg, "data_ptr"):
                    import torch as _torch

                    is_mps = hasattr(arg, "device") and str(arg.device).startswith("mps")
                    # Metal has no float64 — downcast to float32 transparently.
                    is_f64 = hasattr(arg, "dtype") and arg.dtype == _torch.float64
                    if not is_f64 and arg_idx in storage_bindings:
                        buffers.append(storage_bindings[arg_idx])
                        continue
                    if is_f64:
                        arg_f32 = arg.float()  # float64 → float32
                        nbytes = arg_f32.nelement() * arg_f32.element_size()
                        is_output = output_arg_indices is None or arg_idx in output_arg_indices
                        metal_buf, aligned_mem, size_class = pool.acquire(nbytes)
                        pool_releases.append((metal_buf, aligned_mem, size_class))
                        src = (ctypes.c_char * nbytes).from_address(arg_f32.data_ptr())
                        dst_view = metal_buf.contents().as_buffer(nbytes)
                        dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                        ctypes.memmove(dst, src, nbytes)
                        buffers.append((metal_buf, 0))
                        if is_output:
                            tensor_copies.append((metal_buf, arg, nbytes, arg_f32, None))
                        continue
                    # TensorWrapper (unsigned int tensors) may lack nelement()
                    if hasattr(arg, "nelement"):
                        nbytes = arg.nelement() * arg.element_size()
                    elif hasattr(arg, "numel"):
                        nbytes = arg.numel() * arg.element_size()
                    else:
                        import functools, operator

                        nbytes = functools.reduce(operator.mul, arg.shape, 1) * arg.element_size()
                    is_output = output_arg_indices is None or arg_idx in output_arg_indices

                    if is_mps:
                        # MPS tensors: copy via CPU intermediate to avoid
                        # ctypes.memmove corruption of MPS buffer tracking.
                        import torch

                        torch.mps.synchronize()
                        # A kernel addresses each operand with the user-passed RUNTIME
                        # stride args, which describe the tensor's actual (possibly
                        # NON-contiguous) storage layout. ``arg.cpu()`` may DENSIFY a
                        # non-contiguous view (e.g. a column slice randn(K,N+pad)[:, :N]
                        # comes back contiguous with stride N, not N+pad), so a buffer of
                        # just ``numel`` dense bytes no longer matches the kernel's
                        # stride-N+pad addressing -> silently wrong / OOB (audit sliced-B).
                        # Faithfully mirror the storage instead: copy the FULL strided
                        # extent (storage_offset .. storage_offset + reach) verbatim, so
                        # ``buf[i*row_stride + j*col_stride]`` lands on the same element
                        # the user's strides select. Contiguous tensors keep the fast
                        # numel-sized copy (reach == numel).
                        cpu_tensor = arg.cpu()
                        reach_elems = _strided_storage_reach(arg)
                        copy_bytes = max(nbytes, reach_elems * arg.element_size())
                        # Try to build the FULL-strided faithful source buffer. ``arg.cpu()``
                        # densifies a non-contiguous view (losing the original row stride),
                        # so re-read the underlying storage via an as_strided flat view over
                        # [storage_offset, +reach), which keeps ``buf[i*row_stride +
                        # j*col_stride]`` pointing at the right element. Done on the MPS
                        # tensor (then .cpu()) so the strided storage is mirrored, not the
                        # densified copy. Only take the faithful path if it actually
                        # produced a reach-sized contiguous source — otherwise fall through
                        # to the dense numel copy (never memmove copy_bytes from a smaller
                        # source, which would over-read the heap).
                        # TensorWrapper (triton.reinterpret, e.g. uint8 over int8) lacks
                        # is_contiguous/as_strided/storage_offset but shares its base
                        # tensor's storage BYTE-FOR-BYTE -- only the dtype label differs.
                        # The base therefore answers every layout question and stands in
                        # for the faithful mirror AND the copy-back tensor (the copy-back
                        # builds a scratch via _torch.empty(dtype=tensor.dtype), which
                        # crashes on a wrapper's triton dtype). For a plain tensor,
                        # _layout_t IS arg, so this is a no-op. A wrapper-like object with
                        # neither method nor base falls back to the dense copy, matching
                        # _strided_storage_reach's safe reach==numel default.
                        _layout_t = arg if hasattr(arg, "is_contiguous") else getattr(arg, "base", None)
                        _faithful_src = None
                        if _layout_t is not None and (not _layout_t.is_contiguous()) and copy_bytes > nbytes:
                            try:
                                flat = _layout_t.as_strided((reach_elems,), (1,), _layout_t.storage_offset())
                                _cand = flat.cpu().contiguous()
                                if _cand.numel() * _cand.element_size() >= copy_bytes:
                                    _faithful_src = _cand
                            except Exception:
                                _faithful_src = None
                        if _faithful_src is not None:
                            src_t = _faithful_src
                            metal_buf, aligned_mem, size_class = pool.acquire(copy_bytes)
                            pool_releases.append((metal_buf, aligned_mem, size_class))
                            src = (ctypes.c_char * copy_bytes).from_address(src_t.data_ptr())
                            dst_view = metal_buf.contents().as_buffer(copy_bytes)
                            dst = (ctypes.c_char * copy_bytes).from_buffer(dst_view)
                            ctypes.memmove(dst, src, copy_bytes)
                            buffers.append((metal_buf, 0))
                            if is_output:
                                # The copy-back must mirror the FULL strided extent back
                                # through the tensor's own layout — NOT memmove the dense
                                # numel-prefix over the storage, which would densify and
                                # CORRUPT a strided operand (e.g. overwrite a column-sliced
                                # input's rows). The 6th tuple element flags a faithful
                                # strided round-trip (carrying reach + the device-elem
                                # count); the copy-back below scatters via as_strided.
                                # (Inputs are restored byte-identical; a strided output
                                # is written correctly. Most callers pass output indices
                                # so inputs never reach here at all.)
                                # _layout_t, not arg: the copy-back scatters through
                                # tensor.shape/stride and allocates scratch by
                                # tensor.dtype -- for a TensorWrapper that must be the
                                # base (same storage bytes); for a plain tensor it IS arg.
                                tensor_copies.append(
                                    (
                                        metal_buf,
                                        _layout_t,
                                        nbytes,
                                        cpu_tensor,
                                        ("faithful_strided", reach_elems, arg.element_size()),
                                    )
                                )
                        else:
                            # Use buffer pool for page-aligned zero-copy
                            metal_buf, aligned_mem, size_class = pool.acquire(nbytes)
                            pool_releases.append((metal_buf, aligned_mem, size_class))
                            src = (ctypes.c_char * nbytes).from_address(cpu_tensor.data_ptr())
                            dst_view = metal_buf.contents().as_buffer(nbytes)
                            dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                            ctypes.memmove(dst, src, nbytes)
                            buffers.append((metal_buf, 0))
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
                            pool_releases.append((metal_buf, aligned_mem, size_class))
                            src = (ctypes.c_char * nbytes).from_address(ptr)
                            dst_view = metal_buf.contents().as_buffer(nbytes)
                            dst = (ctypes.c_char * nbytes).from_buffer(dst_view)
                            ctypes.memmove(dst, src, nbytes)
                            buffers.append((metal_buf, 0))
                            if is_output:
                                tensor_copies.append((metal_buf, arg, nbytes, None, None))
                elif isinstance(arg, (bool, int, float)):
                    payload = scalar_payloads[arg_idx]
                    size = len(payload)
                    buf = pool.acquire_scalar(size)
                    pool_releases.append(("scalar", buf, size))
                    view = buf.contents().as_buffer(size)
                    view[:size] = payload
                    buffers.append((buf, 0))
                elif arg is None:
                    # Optional pointer argument (mask, other, scale) — pack as null.
                    buf = pool.acquire_scalar(8)
                    pool_releases.append(("scalar", buf, 8))
                    view = buf.contents().as_buffer(8)
                    struct.pack_into("Q", view, 0, 0)
                    buffers.append((buf, 0))
                elif isinstance(arg, str):
                    # Constexpr string argument — already compiled into kernel.
                    continue
                elif hasattr(arg, "__module__") and "triton" in str(type(arg)):
                    # tl.dtype or similar constexpr type — skip.
                    continue
                else:
                    raise TypeError(f"Unsupported argument type: {type(arg)}")

            if _packed_layout is not None:
                if len(buffers) != _packed_layout.packed_slot:
                    raise MetalNonRecoverableError("host packed pointer bindings disagree with producer ABI")
                _packed_size = len(_packed_payload)
                _packed_buffer = pool.acquire_scalar(_packed_size)
                # Register before property/fill calls that may raise.
                pool_releases.append(("scalar", _packed_buffer, _packed_size))
                _packed_view = _packed_buffer.contents().as_buffer(_packed_size)
                _packed_view[:_packed_size] = _packed_payload
                buffers.append((_packed_buffer, 0))

            from ._generic_launch import host_launch_geometry

            grid, threadgroup_size = host_launch_geometry((gridX, gridY, gridZ), block_size, needs_2d_grid)

            # Two-kernel split (#159): for a fully-aligned float matmul, dispatch the
            # pure-direct (no-threadgroup) kernel instead of the staged one — same
            # grid, max occupancy. Falls back to the staged kernel on any
            # uncertainty (different size, non-aligned, read error). Other kernels
            # have no descriptor and are unaffected.
            dispatch_fn = function
            mm_two = kernel_metadata[6] if (kernel_metadata and len(kernel_metadata) > 6) else None
            if mm_two is not None:
                direct_ps = _direct_pipeline(function)
                if direct_ps is not None:
                    from ._generic_launch import direct_twin

                    _twin = direct_twin(mm_two, flat_args, gridX * gridY * gridZ)
                    if _twin:
                        # The twin runs the same grid at its own threadgroup width.
                        dispatch_fn = direct_ps
                        threadgroup_size = (_twin[1], *threadgroup_size[1:])

            # Immediate mode: dispatch, wait, copy-back
            _assert_desc = kernel_metadata[11]
            if _assert_desc is not None:
                from ._device_assert import check_binding, check_failure

                check_binding(buffers, _assert_desc)
                _assert_buffer = utils.make_buffer_with_data(bytes(4), 4)
                buffers.append((_assert_buffer, 0))
            # Only recycle after a normal synchronous return. A raised dispatch may
            # have an uncertain submission/completion state; preserve non-reuse there.
            with _host_dispatch.retain(
                buffers,
                owners=(flat_args, storage_groups, tensor_copies, pool_releases, pool),
            ) as retained_buffers:
                pool_reusable = False
                utils.launch(dispatch_fn, grid, threadgroup_size, retained_buffers)
                pool_reusable = True

            if _assert_desc is not None:
                check_failure(int.from_bytes(_assert_buffer.contents().as_buffer(4), "little"), _assert_desc)

            # Copy results back from Metal buffers to tensor memory.
            for entry in tensor_copies:
                metal_buf, tensor, nbytes = entry[0], entry[1], entry[2]
                cpu_tensor = entry[3] if len(entry) > 3 else None
                _marker = entry[4] if len(entry) > 4 else None

                import torch as _torch

                if isinstance(_marker, tuple) and _marker[0] == "whole_storage_mps":
                    _esz = _marker[1]
                    if nbytes % _esz != 0:
                        raise RuntimeError("host-roundtrip whole-storage copy-back has a partial element")
                    _count = nbytes // _esz
                    src_view = metal_buf.contents().as_buffer(nbytes)
                    flat_cpu = _torch.empty(_count, dtype=tensor.dtype)
                    ctypes.memmove(
                        (ctypes.c_char * nbytes).from_address(flat_cpu.data_ptr()),
                        (ctypes.c_char * nbytes).from_buffer(src_view),
                        nbytes,
                    )
                    flat_device = tensor.as_strided((_count,), (1,), 0)
                    flat_device.copy_(flat_cpu.to(tensor.device))
                    _torch.mps.synchronize()
                    continue

                if isinstance(_marker, tuple) and _marker[0] == "whole_storage_host":
                    _storage_ptr = _marker[1]
                    src_view = metal_buf.contents().as_buffer(nbytes)
                    ctypes.memmove(
                        (ctypes.c_char * nbytes).from_address(_storage_ptr),
                        (ctypes.c_char * nbytes).from_buffer(src_view),
                        nbytes,
                    )
                    continue

                # Faithful strided round-trip: the device buffer mirrors the full
                # strided storage (reach >= numel). Read it ALL back into a flat CPU
                # buffer, then write the values into the tensor through an as_strided
                # view that maps each logical element to its strided storage slot —
                # so a strided operand is restored/written WITHOUT densifying (the
                # numel-prefix memmove below would corrupt a column-sliced layout).
                if isinstance(_marker, tuple) and _marker[0] == "faithful_strided":
                    _reach, _esz = _marker[1], _marker[2]
                    _rbytes = _reach * _esz
                    src_view = metal_buf.contents().as_buffer(_rbytes)
                    flat_cpu = _torch.empty(_reach, dtype=tensor.dtype)
                    ctypes.memmove(
                        (ctypes.c_char * _rbytes).from_address(flat_cpu.data_ptr()),
                        (ctypes.c_char * _rbytes).from_buffer(src_view),
                        _rbytes,
                    )
                    # Index the flat buffer by the tensor's own strides so each logical
                    # [i,j,...] reads flat[i*s0 + j*s1 + ...]. ``flat_cpu`` was built from
                    # ``arg.as_strided((reach,),(1,),storage_offset)`` (dispatch), so it is
                    # ALREADY re-based to the storage offset — index it from 0, NOT from
                    # tensor.storage_offset() again (double-applying the offset would push
                    # the view out of bounds for any non-zero-offset output and silently
                    # drop the write). Failure here is a silent-wrong (output not written),
                    # so do NOT swallow it — raise loudly.
                    strided_logical = flat_cpu.as_strided(tuple(tensor.shape), tuple(tensor.stride()), 0)
                    tensor.copy_(strided_logical.to(tensor.device))
                    _torch.mps.synchronize()
                    continue

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

        finally:
            if pool_reusable:
                _release_host_pool_buffers(pool, pool_releases)

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
        # Use the same SDK selection and pinned resolver as compiler/probe calls.
        # Selection is sufficient here; full content identity belongs to compilation.
        try:
            from ._toolchain_contract import compiler_process_context

            compiler_process_context()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            import warnings

            warnings.warn(
                f"triton-msl cannot select a usable Metal compiler: {error}. "
                "Install/select Xcode and its Metal Toolchain component; "
                "restart after changing the toolchain selection.",
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
        from triton_msl.profiling.metal_bench import metal_do_bench

        return metal_do_bench
