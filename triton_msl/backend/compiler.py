import functools
import hashlib
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field, MISSING
from typing import Dict

from types import ModuleType

# Import BaseBackend from triton.backends.compiler.
# This can trigger a circular import during Triton's backend discovery:
#   triton.backends.__init__ → _discover_backends() → import this module
#   → import triton.backends.compiler → import triton.backends.__init__ (cycle)
#
# Fix: if triton.backends is currently being loaded (partially initialized),
# import BaseBackend directly from the compiler submodule without going
# through triton.backends.__init__.
from triton.backends.compiler import BaseBackend, GPUTarget


# Content-key -> MSL source, populated when MSL is emitted so the driver's
# launcher can retrieve the source for the torch.mps.compile_shader fast-path
# (the launcher only receives the compiled metallib + metadata, not the source).
#
# Keyed by the per-kernel content hash (the same ``cache_key`` used for the
# on-disk metallib), NOT the kernel name. Inductor reuses names like
# ``triton_poi_fused_0`` across DIFFERENT compiled graphs in one process, so a
# name key lets a later compile's MSL clobber an earlier kernel's entry and the
# fast-path would JIT + dispatch the WRONG shader (silent-wrong, observed as a
# roving cold-cache failure across torch.compile models). The content hash is
# unique per distinct kernel — identical kernels share, distinct kernels never
# collide — and the launcher resolves it via ``metadata.msl_hash``.
_MSL_BY_KEY: dict = {}


def _stash_msl(msl_src, key, block_size=None):
    """Register ``msl_src`` (and its threadgroup size) under the content-unique
    ``key`` (the kernel's ``cache_key``) for the launcher's compile_shader
    fast-path.

    The launcher resolves this via ``metadata.msl_hash`` (set to the same
    ``cache_key`` alongside every stash). Keying by content rather than by the
    parsed entry-point name prevents cross-graph name collisions from serving
    one kernel's launcher the wrong shader source.

    ``block_size`` is the MSL kernel's OWN threadgroup size, captured BEFORE the
    C++ LLVM path may clobber ``metadata["block_size"]`` with its different
    ("one thread per element") value. compile_shader launches the stashed MSL,
    so it must use this size — using the clobbered metadata size silently
    mis-launches MEPT kernels (MSL sizePerThread>1 uses fewer threads).
    """
    if key:
        _MSL_BY_KEY[key] = (msl_src, block_size)


def _get_cache_dir():
    """Return the persistent cache directory for compiled kernels.

    The directory is created if it does not exist.  Users can override the
    location by setting the ``TRITON_MSL_CACHE_DIR`` environment variable.
    """
    cache_dir = os.environ.get("TRITON_MSL_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    # Default: ~/.cache/triton_msl/
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "triton_msl")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _msl_cache_key(mod_text, options_hash):
    """Persistent-cache key for emitted MSL.

    Includes CODEGEN_VERSION and the MEPT flag alongside TTGIR text + options:
    without them, emitter/lowerer changes (or toggling TRITON_MSL_MEPT)
    silently replay stale compiled kernels (Phase 0, audit debt #1/#2).
    """
    from triton_msl import CODEGEN_VERSION
    # Effective MEPT flag (default ON as of M5): no-env and "1" share a key;
    # "0" (escape hatch) is distinct. Must match generic_lowerer's default.
    mept = "0" if os.environ.get("TRITON_MSL_MEPT") == "0" else "1"
    return hashlib.sha256(
        (mod_text + options_hash + CODEGEN_VERSION + mept).encode("utf-8")
    ).hexdigest()[:16]


# Number of attempts for the metal→air→metallib pipeline.
# Transient toolchain flakes (e.g. xcrun metal -c exits 0 but the .air
# is not yet durable) are retried up to this many times.  Genuine MSL/IR
# compile errors (CalledProcessError from the metal -c step) are raised
# immediately on the first attempt without retrying.
_METALLIB_COMPILE_ATTEMPTS = 3

_NUM_STAGES_WARNED = False


def _warn_inert_num_stages(n):
    """Note ONCE (debug level >= 1) that num_stages > 1 is a no-op on Metal.

    Keeps the contract honest: a user who sets num_stages expecting software
    pipelining should know it has no effect here (correct result, just not
    pipelined) rather than be silently ignored. One-shot to avoid log spam.
    """
    global _NUM_STAGES_WARNED
    if _NUM_STAGES_WARNED:
        return
    _NUM_STAGES_WARNED = True
    try:
        from triton_msl.debug import _debug_level
        if _debug_level() < 1:
            return
        import sys
        print(f"[triton-msl] num_stages={n} requested but is a no-op on Metal: the "
              f"backend's fast paths (direct-load + register-blocked matmul, "
              f"prefetch+MLP FlashAttention) already saturate load/compute overlap; "
              f"CUDA-style multi-stage pipelining measured no benefit. Result is "
              f"correct, just not pipelined.", file=sys.stderr)
    except Exception:
        pass


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    # num_stages is accepted for Triton-API compatibility but is INTENTIONALLY a
    # no-op on Metal (default 1). On CUDA it sets the software-pipelining depth —
    # multi-buffering threadgroup-staged operands to overlap global loads with MMA.
    # That win does not transfer here: our fast paths stream operands DIRECTLY from
    # device with register blocking (matmul) / explicit prefetch + MLP (FA), which
    # already saturates load/compute overlap, and Apple GPUs have no cp.async to
    # multi-stage. A num_stages=2 double-buffered matmul was measured flat-to-slower
    # (11.13 vs 11.2 TFLOP/s at 2048^3 fp32 — extra live fragments just add register
    # pressure), matching the FA K-prefetch (flat) and block-pipelining (wash)
    # spikes. We do NOT silently imply pipelining: see _warn_inert_num_stages.
    num_stages: int = 1
    num_ctas: int = 1
    # Apple GPU SIMD-groups are always 32-wide.
    warp_size: int = 32
    # Metal threadgroup memory is capped at 32 KB.
    max_threadgroup_memory: int = 32768
    enable_fp_fusion: bool = True
    # FP8 via software emulation — store as uchar, convert to/from float.
    supported_fp8_dtypes: tuple = ("fp8e4nv", "fp8e5", "fp8e4b15", "fp8e4b8", "fp8e5b16")
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = field(default_factory=dict)
    debug: bool = False
    backend_name: str = "metal"
    arch: str = "apple-m4"
    sanitize_overflow: bool = True
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    instrumentation_mode: str = ""
    # Metal Shading Language version for xcrun compilation.
    # "auto" (default) detects from the current device and SDK.
    target_metal_version: str = "auto"
    # HIP-specific knobs that upstream tutorials pass for any non-CUDA
    # backend. Apple GPUs ignore them, but they need to be accepted as
    # kwargs so ``triton.runtime.jit`` doesn\'t raise
    # ``KeyError: Keyword argument <name> was specified but
    # unrecognised`` when running e.g. tutorial 03 (matmul) with the
    # HIP autotune config as fallback.
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    waves_per_eu: int = 0

    def __post_init__(self):
        # Match NVIDIA/AMD: ``extern_libs`` is exposed downstream as a
        # tuple of (name, path) pairs so ``options.__dict__`` produces
        # a hashable, deterministic value. Tests query
        # ``compile_info[\"extern_libs\"]`` and compare against
        # ``tuple(option_val.items())``; storing a dict here breaks
        # that contract (test_launch_with_options[options2]).
        if isinstance(self.extern_libs, dict):
            object.__setattr__(
                self, "extern_libs", tuple(sorted(self.extern_libs.items()))
            )

    @staticmethod
    def _make_hashable(value):
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join(
            f"{name}-{self._make_hashable(val)}"
            for name, val in sorted(hash_dict.items())
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ("metal", "mps")

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    def parse_options(self, opts: dict) -> MetalOptions:
        result = {}
        for k, f in MetalOptions.__dataclass_fields__.items():
            if k in opts:
                result[k] = opts[k]
            elif f.default is not MISSING:
                result[k] = f.default
            elif f.default_factory is not MISSING:
                result[k] = f.default_factory()

        # Validate num_warps is a power of 2
        num_warps = result.get("num_warps", 4)
        assert num_warps > 0 and (num_warps & (num_warps - 1)) == 0, \
            f"num_warps ({num_warps}) must be a power of 2"

        # Validate: no FP64 on Metal.
        arch = result.get("arch", "apple-m4")
        result["arch"] = arch

        # num_stages is a no-op on Metal (see MetalOptions.num_stages). If the user
        # explicitly asks for pipelining (>1), say so ONCE at debug level rather than
        # silently ignoring it — the result is correct either way, just not pipelined.
        if opts.get("num_stages", 1) and opts.get("num_stages", 1) > 1:
            _warn_inert_num_stages(opts["num_stages"])

        return MetalOptions(**result)

    def pack_metadata(self, metadata):
        block_size = getattr(metadata, "block_size", None) or metadata.num_warps * 32
        output_arg_indices = getattr(metadata, "output_arg_indices", None)
        needs_2d_grid = getattr(metadata, "needs_2d_grid", False)
        # IRSource-loaded kernels skip the codegen stage that sets
        # ``shared``; fall back to 0 instead of raising. The Metal
        # driver doesn\'t use this value (threadgroup allocs are
        # baked into the MSL), but unpacking still expects a tuple
        # of fixed shape (test_irsource::test_mlir_attribute_parsing).
        shared = getattr(metadata, "shared", 0)
        # Two-kernel-split matmul descriptor (#159); None for other kernels.
        mm_two_kernel = getattr(metadata, "mm_two_kernel", None)
        # Fast-matmul runtime-dispatch descriptor (Phase 4); None for other kernels.
        fast_matmul = getattr(metadata, "fast_matmul", None)
        return (
            metadata.num_warps,
            metadata.num_ctas,
            shared,
            block_size,
            output_arg_indices,
            needs_2d_grid,
            mm_two_kernel,
            fast_matmul,
        )

    def get_codegen_implementation(self, options):
        return {
            "min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1),
        }

    def get_module_map(self) -> Dict[str, ModuleType]:
        # Remap triton.language.extra.libdevice (stubs that return None) to
        # our Metal-compatible implementation. Mirrors NVIDIA's pattern of
        # pointing at triton.language.extra.cuda.libdevice.
        from triton_msl.inductor.metal_libdevice import metal_libdevice
        return {"triton.language.extra.libdevice": metal_libdevice}

    def load_dialects(self, ctx):
        # No custom MLIR dialects for now.
        pass

    def add_stages(self, stages, options, language=None):
        from triton.compiler.compiler import Language

        if language == Language.GLUON:
            # Gluon: skip TTIR, use gluon-specific passes to reach TTGIR.
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        else:
            # Triton (default): TTIR → TTGIR
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)

        # Always generate MSL (needed by MLX dispatch path).
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)

        # Optional C++ LLVM IR lowering: when enabled, the metallib stage
        # uses LLVM IR → xcrun metal instead of MSL → xcrun metal.
        # MSL is STILL generated above for MLX compatibility.
        # Phase 1: C++ is OPT-IN (TRITON_MSL_USE_CPP=1), default Python.
        # The default-on flip was REVERTED: the full corpus gate showed the C++
        # elementwise path fails beyond f16 (fp32/i32 bin_ops) and compiles far
        # slower; one differential kernel (T2) was insufficient to flip a family.
        # The diff harness + family table + dtype gate stay as the foundation for
        # a corpus-validated re-flip. FORCE_PYTHON kept as an explicit override.
        use_cpp = (os.environ.get("TRITON_MSL_FORCE_PYTHON") != "1"
                   and os.environ.get("TRITON_MSL_USE_CPP", "") == "1")
        if use_cpp and self._has_cpp_passes():
            def _metallib_via_cpp(src, metadata):
                """Compile metallib from C++ LLVM IR when possible, MSL otherwise.

                For simple kernels (no complex ops), uses the C++ path:
                TTGIR → C++ MLIR passes → LLVM IR → xcrun metal → metallib.
                For complex kernels, uses the MSL path as usual.
                The MSL is always available (from the msl stage) for MLX.
                """
                ttgir_text = metadata.pop("cpp_ttgir", None)
                if ttgir_text and not MetalBackend._has_complex_ops(ttgir_text):
                    # C++ per-thread model: one thread per element.
                    # For kernels with >1024 elements, make_llir injects a
                    # wrapping loop so 1024 threads cover all elements.
                    try:
                        # Don't inherit MSL's block_size (which reflects
                        # wrapping loops / sizePerThread). Let make_llir
                        # compute the correct value from make_range end.
                        cpp_meta = dict(metadata)
                        cpp_meta.pop("block_size", None)
                        llir = MetalBackend.make_llir(ttgir_text, cpp_meta, options)
                        cpp_meta["name"] = metadata["name"]
                        # Compile metallib FIRST — only then commit the
                        # C++ path's block_size to shared metadata. This
                        # keeps MSL fallback's metadata pristine if
                        # compilation fails.
                        result = MetalBackend.make_metallib_from_llir(
                            llir, cpp_meta, options)
                        metadata["block_size"] = cpp_meta["block_size"]
                        return result
                    except Exception as _e:
                        # Debug: expose the silently-swallowed error when
                        # TRITON_MSL_CPP_TRACE=1 is set.
                        if os.environ.get("TRITON_MSL_CPP_TRACE"):
                            import traceback
                            print(f"[triton-msl] C++ path failed for "
                                  f"{metadata.get('name', 'kernel')}: {_e}",
                                  file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                # Complex kernels or C++ failure: use MSL metallib
                return MetalBackend.make_metallib(src, metadata, options)

            # Override the msl stage to ALSO generate LLVM IR
            orig_make_msl = stages["msl"]
            def _msl_with_cpp(src, metadata):
                """Generate MSL and save TTGIR for C++ metallib compilation."""
                # Save TTGIR text before MSL generation mutates the module
                metadata["cpp_ttgir"] = str(src)
                # Generate MSL (primary path)
                return MetalBackend.make_msl(src, metadata, options)

            stages["msl"] = _msl_with_cpp
            stages["metallib"] = _metallib_via_cpp
        else:
            stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @staticmethod
    def _has_complex_ops(ttgir_text):
        """Check if TTGIR has ops that the C++ path doesn't handle correctly.

        Returns True for kernels with reductions, dot, trans, or other ops
        that require Python codegen for correctness. Simple elementwise
        kernels return False and can use the C++ LLVM IR path.
        """
        # ALLOWLIST: ops the C++ metallib path handles correctly.
        # Only use C++ metallib for kernels using ONLY these ops.
        # The per-family op table lives in cpp_families.py (Phase 1 spec).
        import re
        from triton_msl.backend.cpp_families import enabled_ops
        allowed_ops = enabled_ops()
        from triton_msl.backend.cpp_families import cpp_safe_text
        if not cpp_safe_text(ttgir_text):
            return True  # unsafe dtype for C++ AIR path -> Python route
        # The C++ run_to_llvm pass aborts (assertion) on a SCALAR (0-D) load —
        # a `tt.load %p : !tt.ptr<T>` whose result is a scalar, not a tensor of
        # pointers `tensor<Nx!tt.ptr<T>>` (e.g. batchnorm's i64
        # num_batches_tracked increment: `tt.load %in : !tt.ptr<i64>`). The MSL
        # path handles these; route them there rather than crash the process.
        if re.search(r'tt\.load\b[^\n]*:\s*!tt\.ptr<', ttgir_text):
            return True
        # Extract actual MLIR operations from the TTGIR text.
        # Operations appear as either:
        #   %result = tt.load %ptr   (result-producing op)
        #   tt.store %ptr, %val      (side-effecting op at line start)
        #   tt.func public @name     (function definition at line start)
        #   tt.return                (terminator at line start)
        # We must NOT match attribute/type references like:
        #   !tt.ptr<f32>             (type annotation, preceded by !)
        #   #ttg.blocked<{...}>      (encoding attribute, preceded by #)
        #   "ttg.num-warps"          (module attribute key, inside quotes)
        #   "ttg.target"             (module attribute key, inside quotes)
        ops_in_kernel = set()
        for line in ttgir_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                continue
            # Match "= <op>" pattern (result-producing ops)
            for m in re.finditer(r'=\s+((?:tt|arith|math|scf|ttg)\.\w+)', stripped):
                ops_in_kernel.add(m.group(1))
            # Match ops at line start (side-effecting ops like tt.store, tt.return, tt.func)
            m = re.match(r'((?:tt|arith|math|scf|ttg)\.\w+)', stripped)
            if m:
                ops_in_kernel.add(m.group(1))
        # If any op is NOT in the allowlist, use MSL
        for op in ops_in_kernel:
            if op not in allowed_ops:
                return True

        # History (2026-04-16):
        #   Bug A: ReduceOpConversion ignored op.getAxis() and collapsed
        #          ALL threadgroup elements to one scalar — broken for any
        #          2D tensor reduction.  FIXED: ReduceOpConversion now
        #          dispatches to a shape-aware axis-scoped per-thread
        #          threadgroup-memory scan for 2D inputs (axis=0 and
        #          axis=1).
        #   Bug B: SharedMemoryAliasingPass merged pre-loop `q` with
        #          in-loop `p`, clobbering `q` on iteration 2+. FIXED
        #          by extending live ranges across loop back-edges.
        #   Bug C: DotOpConversion treated B operand as row-major
        #          (K_dim x N_dim) regardless of whether the source
        #          memdesc had a `ttg.memdesc_trans` in its chain. For
        #          FA's qk = q @ trans(k) the B buffer actually holds
        #          K un-transposed, so the dot was computing q @ k
        #          instead of q @ k^T.  FIXED: DotOpConversion now walks
        #          through memdesc_trans, computes swapped-index tile
        #          offsets, and passes `transpose=true` to the simdgroup
        #          load intrinsic.
        #   Bug D: The >1024-element wrapping-loop injection in
        #          _inject_wrapping_loop (compiler.py) moved the original
        #          entry compute lines into the new `_wl_header` block
        #          without rewriting phi-node predecessor labels, so any
        #          scf.for loop-header phi that referenced the original
        #          entry block still named the wrong predecessor. FIXED
        #          by rewriting phi incoming labels from %<entry> to
        #          %_wl_header after the entry split.
        #
        # tt.dot + wrap-loop fundamental incompatibility guard.
        #
        # When the kernel tile size (block_size) exceeds Metal's 1024
        # thread cap, make_llir injects a wrapping loop in
        # _inject_wrapping_loop: `for (_wlid = lid; _wlid < total;
        # _wlid += 1024)`. Each thread iterates, processing multiple
        # elements. This works for elementwise / reduce kernels where
        # each thread's computation is per-element.
        #
        # For tt.dot, it breaks in multiple ways at 2048-element tiles:
        #
        # 1) The populate phase (stores scalar per-thread to TG[lid])
        #    only covers positions 0..1023. The simdgroup_matrix_8x8_load
        #    that follows reads a full M*N tile (e.g. 2048 floats) from TG
        #    and gets garbage at positions 1024..2047. Could be fixed with
        #    a populate-phase-only wrap (each thread does 2 stores), but:
        #
        # 2) FA's scf.for has per-thread scalar loop-carried state
        #    (acc, m_i, l_i) as phi nodes. For a 2048-element acc tensor
        #    with only 1024 threads, each thread would need to carry 2
        #    scalars across the loop, but the phi is a single scalar.
        #
        # 3) The alpha = exp(m_prev - m_new) rescale of acc happens as
        #    `acc_scalar * alpha_scalar` per thread, writing back to
        #    TG[lid] (positions 0..1023). Positions 1024..2047 of the acc
        #    TG buffer are never rescaled, so the next iter's dot_cin
        #    reads stale data from half the acc buffer.
        #
        # 4) The final output store uses the per-thread scalar acc/l_i,
        #    so only rows 0..15 (of 32) get written to global memory.
        #
        # Issues (2), (3), (4) all stem from the per-thread scalar model
        # in DotOpConversion. Fixing requires keeping acc/m_i/l_i purely
        # in TG memory and using stride loops for all per-element ops.
        # That is a substantial refactor of the C++ conversion.
        #
        # Route to MSL, which uses a different code model: acc/m_i/l_i
        # live in threadgroup memory throughout, with stride loops
        # (`for _sb = lid; _sb < 2048; _sb += 1024`) for populate, apply
        # alpha, and write-out. No per-thread scalar roundtrip.
        #
        # Investigated 2026-04-21: attempted populate-only phase split and
        # `__tg_dot_cin_` aliasing; aliasing broke FA HEAD_DIM=32 (live
        # range of dot_cin is not captured correctly by the range-based
        # pass for this buffer's bracketed store-barrier-simdload pattern).
        # Phase split would fix (1) but not (2)/(3)/(4).
        if 'tt.dot' in ops_in_kernel:
            # Compute block_size the same way make_llir does: product of
            # max `end` per slice dim for 2D kernels, else max end.
            dim_to_end = {}
            max_end_any = 0
            for mr in re.finditer(
                r'tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)[^}]*\}\s*:\s*'
                r'tensor<\d+xi32(?:,\s*#ttg\.slice<\{dim\s*=\s*(\d+))?',
                ttgir_text,
            ):
                end = int(mr.group(1))
                dim = int(mr.group(2)) if mr.group(2) is not None else None
                if end > max_end_any:
                    max_end_any = end
                if dim is not None:
                    dim_to_end[dim] = max(dim_to_end.get(dim, 0), end)
            if len(dim_to_end) > 1:
                block_size = 1
                for e in dim_to_end.values():
                    block_size *= e
            else:
                block_size = max_end_any
            if block_size > 1024:
                return True

        # Shared memory budget check: detect kernels whose effective
        # threadgroup memory demand exceeds Metal's 32KB cap. This is a
        # pre-lowering safety net; the C++ path also enforces this inside
        # LocalAllocOpConversion for `ttg.local_alloc`. For tt.reduce, the
        # C++ per-thread model can only correctly handle tensors whose size
        # fits within the thread cap (1024). Larger reductions must go via
        # MSL, which has its own shared-memory aliasing pass.
        # Element byte sizes keyed by MLIR type suffix.
        elem_bytes = {
            'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
            'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8,
        }
        # Find any tt.reduce and check the input tensor type. The signature
        # line follows the op body on a later line, so scan multi-line.
        for m in re.finditer(
            r'tt\.reduce["\w]*\b[^}]*?\}\)\s*:\s*\(tensor<([\dx]+)x(\w+)',
            ttgir_text,
            re.DOTALL,
        ):
            dims_str, elem_ty = m.group(1), m.group(2)
            try:
                dims = [int(d) for d in dims_str.split('x') if d]
            except ValueError:
                continue
            n = 1
            for d in dims:
                n *= d
            # If the reduction tensor has more elements than the thread cap,
            # our per-thread SIMD-then-crossSIMD lowering is incorrect.
            if n > 1024:
                return True
            # Also guard against exceeding the 32KB shared-memory budget.
            eb = elem_bytes.get(elem_ty, 4)
            if n * eb > 32 * 1024:
                return True
        return False

    @staticmethod
    def _has_cpp_passes():
        """Check if the C++ MLIR pass library is available."""
        try:
            import triton_msl._triton_msl_cpp
            return True
        except ImportError:
            return False

    @staticmethod
    def _strip_ttg_annotations(ttgir_text):
        """Strip TritonGPU encoding attributes, loc annotations, and TTG ops.

        Our C++ pass doesn't need the encoding annotations (it converts
        tensor types to scalars regardless), and parsing them would require
        the TritonGPU dialect which pulls in NVIDIA-specific dependencies.

        When `tt.dot` is NOT present (common case: elementwise/reduce
        kernels), this method also replaces TritonGPU shared memory ops
        with their operand passthroughs at the text level:
        - ttg.local_alloc %x -> deleted (local_load will use %x directly)
        - ttg.local_load %x  -> replaced with the original alloc operand
        - ttg.convert_layout %x -> replaced with %x (passthrough)
        - ttg.memdesc_trans %x -> replaced with %x (passthrough)
        - tt.trans %x -> replaced with %x (passthrough in per-thread model)

        When `tt.dot` IS present, these ops are preserved so the C++
        patterns (LocalAllocOpConversion, LocalLoadOpConversion,
        ConvertLayoutOpConversion, etc.) can lower them with proper
        memdesc semantics. DotOpConversion needs to see the real
        ttg.local_load defining ops to locate the memdesc pointers.

        tt.dot is always preserved.
        """
        import re

        # Detect whether tt.dot is present. When it is, the matmul chain
        # (local_alloc -> local_load -> tt.dot) must survive this pass so
        # the C++ DotOpConversion pattern can locate memdesc operands.
        # Use a line-aware check to avoid matching attribute references.
        has_dot = False
        for _line in ttgir_text.splitlines():
            if re.search(r'(?:^|=\s|\s)tt\.dot\b', _line):
                has_dot = True
                break

        # Phase 0.5: Detect 2D blocks and annotate make_range with dimension info.
        #
        # In 2D kernels (matmul), each make_range maps to a different
        # dimension of the output block. We detect slice<{dim = N}> on
        # make_range ops and add a metal.dim attribute so the C++ backend
        # can decompose the linear thread ID:
        #   dim=1 → row index:  lid / BLOCK_COL
        #   dim=0 → col index:  lid % BLOCK_COL
        lines = ttgir_text.split('\n')
        make_range_dims = {}  # %name -> (dim, block_size)
        for line in lines:
            mr_m = re.search(
                r'(%\S+)\s*=\s*tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)[^}]*\}\s*:\s*tensor<\d+xi32,\s*#ttg\.slice<\{dim\s*=\s*(\d+)',
                line
            )
            if mr_m:
                name = mr_m.group(1)
                block_sz = int(mr_m.group(2))
                dim = int(mr_m.group(3))
                make_range_dims[name] = (dim, block_sz)

        # Determine if this is a 2D kernel
        has_2d_block = len(set(d for d, _ in make_range_dims.values())) > 1

        # For 2D blocks, find the column block size (dim=0)
        col_block_size = 32  # default
        if has_2d_block:
            for dim, bsz in make_range_dims.values():
                if dim == 0:
                    col_block_size = bsz
                    break

        # Annotate make_range ops with metal.dim and metal.col_block_size.
        # This is done before stripping so the dim info is preserved.
        annotated_lines = []
        for line in lines:
            if has_2d_block and 'tt.make_range' in line:
                # Find the make_range attribute dict and slice dim
                mr_m = re.search(
                    r'(tt\.make_range\s*\{)([^}]*)(}\s*:)',
                    line
                )
                slice_m = re.search(
                    r'#ttg\.slice<\{dim\s*=\s*(\d+)',
                    line
                )
                if mr_m and slice_m:
                    dim_val = int(slice_m.group(1))
                    # Inject metal.dim and metal.col_block_size attributes
                    new_attrs = f'{mr_m.group(2)}, metal.dim = {dim_val} : i32, metal.col_block_size = {col_block_size} : i32'
                    line = line[:mr_m.start(1)] + mr_m.group(1) + new_attrs + mr_m.group(3) + line[mr_m.end():]
            annotated_lines.append(line)

        # Phase 1: Strip encoding attributes and loc annotations.
        #
        # When tt.dot is present, preserve encoding attribute aliases
        # (e.g. `#shared = #ttg.swizzled_shared<...>`) and their
        # references. The TTG MemDescType requires a shared encoding,
        # and MMA lowering may read dot_op / blocked layouts. The C++
        # pipeline can parse these now that the TTG dialect is linked.
        phase1_lines = []
        for line in annotated_lines:
            stripped = line.rstrip()

            # Handle #name = #ttg.X<...> alias definitions
            if re.match(r'^#\w+ = #ttg\.', stripped):
                if has_dot:
                    phase1_lines.append(stripped)
                continue

            # Skip #locN = loc(...) alias definitions
            if re.match(r'^#loc\d*\s*=\s*loc\(', stripped):
                continue

            if not has_dot:
                # Remove trailing encoding alias references from tensor/memdesc
                # types. Loop to handle multiple (e.g. `<32x32xf16, #shared,
                # #smem>`) since re.sub only replaces non-overlapping matches
                # in one pass.
                while True:
                    new_stripped = re.sub(r',\s*#\w+>', '>', stripped)
                    if new_stripped == stripped:
                        break
                    stripped = new_stripped

                # Remove inline TTG type annotations:
                #   , #ttg.slice<{dim = 1, parent = #blocked1}>
                #   , #ttg.dot_op<{opIdx = 0, parent = #blocked}>
                #   , #ttg.swizzled_shared<{...}>
                # These appear inside tensor<...> types. Match , #ttg.X<{...}>
                stripped = re.sub(
                    r',\s*#ttg\.\w+<\{[^}]*\}>', '', stripped
                )

            # Remove ttg.* module attributes — but preserve ttg.num-warps,
            # ttg.num-ctas, ttg.threads-per-warp, ttg.target when tt.dot is
            # present. TTG layout verifiers require these to compute warp
            # layout for MMA operands.
            if not has_dot:
                stripped = re.sub(r'"ttg\.[^"]*"\s*=\s*[^,}]+[,]?\s*', '', stripped)
                stripped = re.sub(r'ttg\.\w+\s*=\s*"[^"]*"[,]?\s*', '', stripped)

            # Remove loc(...) annotations — handle nested parens.
            while 'loc(' in stripped:
                prev = stripped
                stripped = re.sub(
                    r'\s*loc\((?:[^()]*|\([^()]*\))*\)', '', stripped
                )
                if stripped == prev:
                    break

            # Clean up empty module attributes
            stripped = re.sub(r'module attributes \{\s*\}', 'module', stripped)

            # Skip now-empty lines
            if stripped.strip():
                phase1_lines.append(stripped)

        # Phase 2: Strip TTG ops — SKIPPED when tt.dot is present.
        #
        # When tt.dot is in the kernel, we preserve ttg.local_alloc/load/
        # convert_layout/memdesc_trans so the C++ DotOpConversion +
        # SharedMemoryOpToLLVM patterns can lower them with proper memdesc
        # semantics. tt.trans inside a matmul chain stays too.
        #
        # For non-matmul kernels, we continue the text-level passthrough
        # rewrites (this keeps the existing elementwise/reduce path
        # intact).
        if has_dot:
            return '\n'.join(phase1_lines) + '\n'

        alloc_map = {}    # ttg.local_alloc result -> input operand
        replace_map = {}  # value to replace -> replacement value

        # First pass: scan for TTG op mappings
        for line in phase1_lines:
            # ttg.local_alloc %x : (...) -> !ttg.memdesc<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.local_alloc\s+(%\S+)', line)
            if m:
                alloc_map[m.group(1)] = m.group(2)
                continue

            # ttg.local_load %x : !ttg.memdesc<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.local_load\s+(%\S+)', line)
            if m:
                alloc_result = m.group(2)
                original = alloc_map.get(alloc_result, alloc_result)
                replace_map[m.group(1)] = original
                continue

            # ttg.convert_layout %x : tensor<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.convert_layout\s+(%\S+)', line)
            if m:
                replace_map[m.group(1)] = m.group(2)
                continue

            # ttg.memdesc_trans %x {order = ...} : ...
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.memdesc_trans\s+(%\S+)', line)
            if m:
                alloc_map[m.group(1)] = alloc_map.get(m.group(2), m.group(2))
                continue

            # tt.trans %x : tensor<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*tt\.trans\s+(%\S+)', line)
            if m:
                replace_map[m.group(1)] = m.group(2)
                continue

        # Second pass: emit lines with TTG ops removed and references replaced.
        # tt.dot is preserved — handled by C++ DotOpToLLVM pattern or MSL
        # _detect_simple_dot path.
        out_lines = []
        # Sort replacements longest-first to avoid partial matches
        sorted_replacements = sorted(replace_map.items(),
                                     key=lambda x: len(x[0]), reverse=True)

        for line in phase1_lines:
            # Skip ttg.* op lines entirely
            if re.match(r'\s*%\S+\s*=\s*ttg\.\w+', line):
                continue

            # Skip tt.trans lines (already mapped as passthrough)
            if re.match(r'\s*%\S+\s*=\s*tt\.trans\s+', line):
                continue

            # Apply replacements
            for old, new in sorted_replacements:
                # Use word-boundary-aware replacement to avoid partial matches
                line = re.sub(re.escape(old) + r'(?=[\s,):}\]]|$)', new, line)

            # tt.dot is preserved — handled by C++ DotOpToLLVM pattern or MSL
            # _detect_simple_dot path.

            # Remove residual TTG type annotations that slipped through
            # e.g. !ttg.memdesc<...> in type positions
            line = re.sub(r'!ttg\.memdesc<[^>]*>', 'tensor<1xf32>', line)

            if line.strip():
                out_lines.append(line)

        return '\n'.join(out_lines) + '\n'

    # Map LLVM IR element types to their byte size and AIR metadata name.
    _ELEM_TYPE_INFO = {
        'half':  (2, 2, 'half'),
        'float': (4, 4, 'float'),
        'i8':    (1, 1, 'char'),
        'i16':   (2, 2, 'short'),
        'i32':   (4, 4, 'int'),
        'i64':   (8, 8, 'long'),
        'bfloat':(2, 2, 'bfloat'),
    }

    @staticmethod
    def _opaque_to_typed_ptrs(llir_text):
        """Convert LLVM IR with opaque pointers to typed pointers for Metal AIR.

        Metal's GPU JIT compiler requires typed pointers and does not support
        generic address space stores. This conversion:
        1. Eliminates addrspacecasts (inlines source pointers)
        2. Infers element types from GEP instructions for each device buffer
        3. Converts all pointer types to typed equivalents
        4. Fixes metadata to use typed function pointer references and correct
           type names/sizes for non-float buffers
        """
        import re

        lines = llir_text.split('\n')

        # Pass -1: Collect addrspace(3) global declarations so we can rewrite
        # opaque-pointer references to them (including in !air.threadgroup_buffers
        # metadata) into typed-pointer form.
        # E.g. @__reduce_shared_0 = internal addrspace(3) global [32 x float] undef
        tg_global_types = {}  # "@__reduce_shared_0" -> "[32 x float]"
        for line in lines:
            m = re.match(
                r'\s*(@[\w.]+)\s*=\s*[^=]*addrspace\(3\)\s+global\s+(\[[^\]]+\]|\S+)',
                line
            )
            if m:
                tg_global_types[m.group(1)] = m.group(2)

        # Pass 0: Collect addrspacecast mappings and their source address spaces.
        # E.g. %.generic = addrspacecast ptr addrspace(1) %0 to ptr
        #   -> cast_map["%.generic"] = ("%0", "1")
        cast_map = {}
        for line in lines:
            m = re.match(
                r'\s*(%\S+)\s*=\s*addrspacecast\s+ptr\s+addrspace\((\d+)\)\s+(%\S+)\s+to\s+ptr',
                line
            )
            if m:
                cast_map[m.group(1)] = (m.group(3), m.group(2))

        # Pass 0.5: Infer element types from GEP and load instructions.
        # GEP pattern: getelementptr <elem_type>, ptr <ptr_name>, ...
        # Load pattern: load <elem_type>, ptr addrspace(N) <ptr_name>, ...
        # Follow addrspacecast chains to resolve back to the original param.
        param_types = {}  # param_name -> element_type (e.g. "%0" -> "half")
        for line in lines:
            # GEP: getelementptr <type>, ptr %name
            m = re.match(
                r'\s*%\S+\s*=\s*getelementptr\s+(\w+),\s*ptr\s+(%([\w.]+))',
                line
            )
            if m:
                elem_type = m.group(1)
                ptr_name = m.group(2)
                actual_ptr = cast_map.get(ptr_name, (ptr_name,))[0] if ptr_name in cast_map else ptr_name
                if actual_ptr not in param_types:
                    param_types[actual_ptr] = elem_type
            # Load from constant buffer: load <type>, ptr addrspace(2) %name
            m = re.match(
                r'\s*(%\S+)\s*=\s*load\s+(\w+),\s*ptr\s+addrspace\(2\)\s+(%([\w.]+))',
                line
            )
            if m:
                elem_type = m.group(2)
                ptr_name = m.group(3)  # e.g. "%4"
                if ptr_name not in param_types:
                    param_types[ptr_name] = elem_type

        # Parse function signature to get ordered parameter names.
        # Pattern: define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ...)
        sig_param_names = []  # ordered list of param names from define line
        sig_param_addrspaces = {}  # param_name -> addrspace string or None
        for line in lines:
            m = re.match(
                r'\s*define\s+void\s+@\w+\((.*)\)',
                line
            )
            if m:
                for param in m.group(1).split(','):
                    param = param.strip()
                    # Extract param name (last %word in the param)
                    name_m = re.search(r'(%\S+)\s*$', param)
                    if name_m:
                        pname = name_m.group(1)
                        sig_param_names.append(pname)
                        # Check if it has an addrspace
                        as_m = re.search(r'addrspace\((\d+)\)', param)
                        if as_m:
                            sig_param_addrspaces[pname] = as_m.group(1)
                break

        # For device buffer params (addrspace 1), determine the typed pointer.
        # If we found a GEP that uses the param, use that type; else default to float.
        def _get_device_ptr_type(param_name):
            """Return the element type for a device buffer parameter."""
            return param_types.get(param_name, 'float')

        # Build maps: param_index -> element_type for device and constant buffers
        param_elem_types = {}  # index -> elem_type string (device buffers)
        const_elem_types = {}  # index -> elem_type string (constant buffers)
        for i, pname in enumerate(sig_param_names):
            if sig_param_addrspaces.get(pname) == '1':
                param_elem_types[i] = _get_device_ptr_type(pname)
            elif sig_param_addrspaces.get(pname) == '2':
                const_elem_types[i] = param_types.get(pname, 'i32')

        # Build per-param constant buffer type lookup by name (for non-sig lines)
        const_param_type_by_name = {}
        for i, pname in enumerate(sig_param_names):
            if sig_param_addrspaces.get(pname) == '2':
                const_param_type_by_name[pname] = const_elem_types.get(i, 'i32')

        # Pass 0.75: Rewrite AIR intrinsic declarations (and their call-site
        # pointer-operand types) from opaque `ptr addrspace(N)` to typed
        # pointers. The element type is encoded in the intrinsic's `.pNT`
        # suffix (e.g. `p3f32` -> `float addrspace(3)*`). Metal's compiler
        # rejects opaque pointers in `declare` lines with:
        #   "ptr type is only supported in -opaque-pointers mode".
        _PTR_SUFFIX_MAP = {
            'p0f32': ('float*', '0', 'float'),
            'p1f32': ('float addrspace(1)*', '1', 'float'),
            'p2f32': ('float addrspace(2)*', '2', 'float'),
            'p3f32': ('float addrspace(3)*', '3', 'float'),
            'p0f16': ('half*', '0', 'half'),
            'p1f16': ('half addrspace(1)*', '1', 'half'),
            'p2f16': ('half addrspace(2)*', '2', 'half'),
            'p3f16': ('half addrspace(3)*', '3', 'half'),
            'p0i32': ('i32*', '0', 'i32'),
            'p1i32': ('i32 addrspace(1)*', '1', 'i32'),
            'p2i32': ('i32 addrspace(2)*', '2', 'i32'),
            'p3i32': ('i32 addrspace(3)*', '3', 'i32'),
        }
        # Map intrinsic-name -> typed-pointer string (for the pointer operand).
        air_intrinsic_ptr_types = {}
        for line in lines:
            m = re.match(r'\s*declare\s+.*@(air\.[\w.]+)\(', line)
            if not m:
                continue
            iname = m.group(1)
            # Find the FIRST pN<type> suffix; AIR names may stack suffixes
            # like `.v64f32.p3f32` so we want the *pointer* one.
            psuf = re.search(r'\.(p[0-3][a-z0-9]+)', iname)
            if not psuf:
                continue
            entry = _PTR_SUFFIX_MAP.get(psuf.group(1))
            if not entry:
                continue
            air_intrinsic_ptr_types[iname] = entry
        # Rewrite `ptr addrspace(N)` occurrences on the declare/call lines
        # of each AIR intrinsic to the typed form.
        if air_intrinsic_ptr_types:
            new_lines = []
            for line in lines:
                for iname, (typed_ptr, addr, _elem) in (
                        air_intrinsic_ptr_types.items()):
                    if f'@{iname}' not in line:
                        continue
                    line = re.sub(
                        rf'ptr\s+addrspace\({addr}\)',
                        typed_ptr,
                        line,
                    )
                new_lines.append(line)
            lines = new_lines

        # Pass 1: Process each line
        out_lines = []
        fn_name = None
        fn_param_types = []

        for line in lines:
            # Skip addrspacecast lines
            if re.match(r'\s*%\S+\s*=\s*addrspacecast\s+.*to\s+ptr', line):
                continue

            # Strip `nuw` / `nsw` flags from getelementptr (both instruction
            # and constant-expression forms). Metal's older LLVM parser
            # rejects these flags with "expected '(' in constantexpr".
            line = re.sub(
                r'getelementptr\s+((?:inbounds\s+)?)(?:nuw|nsw)\s+',
                r'getelementptr \1',
                line,
            )

            # Normalize byte-offset GEPs back to element-typed GEPs.
            # LLVM InstCombine rewrites `GEP(half*, i64 8)` into
            # `GEP(i8, half addrspace(3)* p, i64 16)` — in typed-pointer
            # mode this is rejected ("i8 vs half"). Convert back to
            # `GEP(half, half addrspace(3)* p, i64 8)`.
            _ELEM_BYTES = {'half': 2, 'float': 4, 'bfloat': 2,
                           'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8}
            def _fix_byte_gep(m):
                prefix = m.group(1)        # `inbounds ` or ''
                elem_ptr_ty = m.group(2)   # `half` or `float` etc.
                addr_space = m.group(3)    # '1', '2', '3'
                ptr_expr = m.group(4)      # `@name` or `%name` or nested GEP
                byte_off = int(m.group(5))
                size = _ELEM_BYTES.get(elem_ptr_ty)
                if size is None or byte_off % size != 0:
                    return m.group(0)
                elem_off = byte_off // size
                return (f'getelementptr {prefix}({elem_ptr_ty}, '
                        f'{elem_ptr_ty} addrspace({addr_space})* {ptr_expr}'
                        f', i64 {elem_off})')
            # This handles `getelementptr [inbounds] (i8, <T> addrspace(N)* <P>,
            # i64 <N>)` where <P> may itself be a constant expression without
            # parens in it. We match non-greedily up to the matching `)` by
            # disallowing any other comma after the second arg.
            line = re.sub(
                r'getelementptr\s+((?:inbounds\s+)?)'
                r'\(i8,\s*(\w+)\s+addrspace\((\d+)\)\*\s+'
                r'(@[\w.]+|getelementptr\([^)]*\))\s*,\s*i64\s+(\d+)\)',
                _fix_byte_gep,
                line,
            )

            # Replace cast names with source names (longest first for safety)
            for name in sorted(cast_map.keys(), key=len, reverse=True):
                src, addrspace = cast_map[name]
                # Word-boundary replacement
                line = re.sub(re.escape(name) + r'(?=[\s,)\]]|$)', src, line)

            # Convert function signature with per-param types
            def_m = re.match(
                r'(\s*define\s+void\s+@\w+\()(.*?)(\)\s*\{.*)$',
                line
            )
            if def_m:
                prefix, params_str, suffix = def_m.group(1), def_m.group(2), def_m.group(3)
                new_params = []
                for i, param in enumerate(params_str.split(',')):
                    param = param.strip()
                    if 'ptr addrspace(1)' in param:
                        ety = param_elem_types.get(i, 'float')
                        param = param.replace('ptr addrspace(1)', f'{ety} addrspace(1)*')
                    elif 'ptr addrspace(2)' in param:
                        ety = const_elem_types.get(i, 'i32')
                        param = param.replace('ptr addrspace(2)', f'{ety} addrspace(2)*')
                    new_params.append(param)
                line = prefix + ', '.join(new_params) + suffix
            else:
                # Non-signature lines: convert ptr addrspace(2) with correct type.
                # Match "ptr addrspace(2) %name" and use the inferred type.
                def _replace_const_ptr(m):
                    pname = m.group(1)
                    ety = const_param_type_by_name.get(pname, 'i32')
                    return f'{ety} addrspace(2)* {pname}'
                line = re.sub(
                    r'ptr\s+addrspace\(2\)\s+(%([\w.]+))',
                    _replace_const_ptr,
                    line
                )
                # Do NOT blindly replace ptr addrspace(1) here; GEP/load/store
                # handlers below will use the correct element type.

            # GEP: getelementptr <type>, ptr %X → getelementptr <type>, <type> addrspace(1)* %X
            line = re.sub(
                r'getelementptr\s+(\w+),\s*ptr\s+(%\S+)',
                r'getelementptr \1, \1 addrspace(1)* \2',
                line
            )

            # Load from GEP result (device buffer, addrspace 1):
            # load <type>, ptr %X -> load <type>, <type> addrspace(1)* %X
            # By this point, constant buffer loads (addrspace 2) already have
            # their ptr replaced with i32 addrspace(2)*, so any remaining
            # "load <type>, ptr %X" refers to a device buffer GEP result.
            line = re.sub(
                r'load\s+(\w+),\s*ptr\s+(%\S+)',
                r'load \1, \1 addrspace(1)* \2',
                line
            )

            # Store: store <type> %v, ptr %X → store <type> %v, <type> addrspace(1)* %X
            # Handle any scalar type. The stored value may be an SSA register
            # (%v) OR a literal constant (e.g. `store float 0.0, ptr %6` from a
            # zero-init); match both, otherwise the pointer is left opaque and
            # Metal rejects the IR ("ptr type is only supported in
            # -opaque-pointers mode") — which then AGX-rejects the metallib.
            line = re.sub(
                r'store\s+(\w+)\s+(\S+),\s*ptr\s+(%\S+)',
                r'store \1 \2, \1 addrspace(1)* \3',
                line
            )

            # PHI nodes with ptr type: phi ptr [ %a, %bb1 ], [ %b, %bb2 ]
            # These come from scf.for loop iter_args that carry pointers.
            # Convert to typed pointer: phi float addrspace(1)* [ ... ]
            # The element type is inferred from uses (GEP/load/store).
            # For simplicity, default to float addrspace(1)* since that's
            # the most common case for device buffer pointers in loops.
            line = re.sub(
                r'phi\s+ptr\s+\[',
                r'phi float addrspace(1)* [',
                line
            )

            # ---- Threadgroup shared memory (addrspace 3) patterns ----
            # These are generated by the reduce op lowering.

            # GEP with array type base: getelementptr [N x <elem>], ptr addrspace(3) @name, ...
            # -> getelementptr [N x <elem>], [N x <elem>] addrspace(3)* @name, ...
            # Generalized over element types (float, half, i32, ...).
            line = re.sub(
                r'getelementptr\s+(\[\d+ x \w+\]),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                r'getelementptr \1, \1 addrspace(3)* \2',
                line
            )

            # When an AIR intrinsic (or other typed-pointer call) receives a
            # threadgroup *global* directly — e.g. `half addrspace(3)* @__tg_shared_0`
            # — the global's actual type is `[N x half]`, so Metal's typed-
            # pointer mode rejects the implicit decay. Wrap those references
            # with a GEP to the first element:
            #   half addrspace(3)* @g   -->
            #   half addrspace(3)* getelementptr([N x half], [N x half]
            #                      addrspace(3)* @g, i32 0, i32 0)
            def _wrap_tg_global(m):
                elem = m.group(1)
                gname = m.group(2)
                gty = tg_global_types.get(gname)
                if gty is None:
                    return m.group(0)
                # Only wrap when the stored global is an array of the same
                # element type (i.e. `[N x <elem>]`), otherwise leave alone.
                arr_m = re.match(r'\[(\d+) x (\w+)\]', gty)
                if not arr_m or arr_m.group(2) != elem:
                    return m.group(0)
                return (f'{elem} addrspace(3)* getelementptr({gty}, {gty}'
                        f' addrspace(3)* {gname}, i32 0, i32 0)')
            line = re.sub(
                r'(\w+)\s+addrspace\(3\)\*\s+(@[\w.]+)',
                _wrap_tg_global,
                line,
            )

            # Store to threadgroup: store <elem> %v, ptr addrspace(3) %slot or @global
            # For direct global refs to reduce/merged pools (declared as
            # `[N x float]`), insert an extra GEP to get the scalar pointer.
            # `__tg_merged_*` globals are emitted by SharedMemoryAliasingPass
            # when it coalesces reduce globals with non-overlapping live
            # ranges; they keep the `[32 x float]` type.
            m_tg_store = re.match(
                r'(\s*store\s+float\s+\S+),\s*ptr addrspace\(3\)\s+'
                r'(@(?:__reduce_shared_|__tg_merged_)\d+)(.*)',
                line
            )
            if m_tg_store:
                line = (f'{m_tg_store.group(1)}, float addrspace(3)* '
                        f'getelementptr([32 x float], [32 x float] addrspace(3)* '
                        f'{m_tg_store.group(2)}, i32 0, i32 0){m_tg_store.group(3)}')
            else:
                # Generalized scalar store: any scalar element type.
                line = re.sub(
                    r'store\s+(\w+)\s+(%\S+),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                    r'store \1 \2, \1 addrspace(3)* \3',
                    line
                )

            # Load from threadgroup: load <elem>, ptr addrspace(3) %slot or @global
            m_tg_load = re.match(
                r'(\s*%\S+\s*=\s*load\s+float),\s*ptr addrspace\(3\)\s+'
                r'(@(?:__reduce_shared_|__tg_merged_)\d+)(.*)',
                line
            )
            if m_tg_load:
                line = (f'{m_tg_load.group(1)}, float addrspace(3)* '
                        f'getelementptr([32 x float], [32 x float] addrspace(3)* '
                        f'{m_tg_load.group(2)}, i32 0, i32 0){m_tg_load.group(3)}')
            else:
                line = re.sub(
                    r'load\s+(\w+),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                    r'load \1, \1 addrspace(3)* \2',
                    line
                )

            # Capture function name and param types for metadata
            m = re.match(
                r'\s*define\s+void\s+@(\w+)\(((?:[^()]*|\([^()]*\))*)\)',
                line
            )
            if m:
                fn_name = m.group(1)
                for param in m.group(2).split(','):
                    param = param.strip()
                    idx = param.rfind('%')
                    if idx > 0:
                        fn_param_types.append(param[:idx].rstrip())
                    else:
                        fn_param_types.append(param)

            # Metadata: ptr @fn -> typed function pointer
            if fn_name and line.strip().startswith('!') and f'ptr @{fn_name}' in line:
                typed_sig = ', '.join(fn_param_types)
                fn_ptr_type = f'void ({typed_sig})*'
                line = line.replace(f'ptr @{fn_name}', f'{fn_ptr_type} @{fn_name}')

            # Metadata: ptr addrspace(3) @global -> typed pointer to the global.
            # Emitted by the C++ bridge for !air.threadgroup_buffers entries.
            # Metal rejects opaque `ptr` in metadata operands.
            if line.strip().startswith('!') and 'ptr addrspace(3)' in line:
                def _rewrite_tg_global(m):
                    gname = m.group(1)
                    gty = tg_global_types.get(gname)
                    if gty is None:
                        return m.group(0)
                    return f'{gty} addrspace(3)* {gname}'
                line = re.sub(
                    r'ptr\s+addrspace\(3\)\s+(@[\w.]+)',
                    _rewrite_tg_global,
                    line
                )

            # Fix metadata arg_type_name and arg_type_size for non-float device buffers.
            # The C++ pass hardcodes "float" / size 4 for all device buffers.
            # Replace with the correct type based on GEP-inferred element types.
            if line.strip().startswith('!') and '!"air.buffer"' in line and '!"air.address_space", i32 1' in line:
                # This is a device buffer metadata entry. Extract the arg index.
                arg_idx_m = re.match(r'(\s*!\d+\s*=\s*!\{i32\s+)(\d+)', line)
                if arg_idx_m:
                    arg_idx = int(arg_idx_m.group(2))
                    if arg_idx in param_elem_types:
                        ety = param_elem_types[arg_idx]
                        type_info = MetalBackend._ELEM_TYPE_INFO.get(ety)
                        if type_info:
                            byte_size, align_size, air_name = type_info
                            # Replace arg_type_size
                            line = re.sub(
                                r'(!"air\.arg_type_size",\s*i32\s+)\d+',
                                rf'\g<1>{byte_size}',
                                line
                            )
                            # Replace arg_type_align_size
                            line = re.sub(
                                r'(!"air\.arg_type_align_size",\s*i32\s+)\d+',
                                rf'\g<1>{align_size}',
                                line
                            )
                            # Replace arg_type_name
                            line = re.sub(
                                r'(!"air\.arg_type_name",\s*!")(\w+)(")',
                                rf'\g<1>{air_name}\3',
                                line
                            )

            out_lines.append(line)

        return '\n'.join(out_lines)

    @staticmethod
    def _strip_unsupported_llvm_attrs(llir_text):
        """Strip LLVM function attributes that Metal's compiler doesn't support.

        Metal's compiler is based on an older LLVM and doesn't understand
        newer attributes like nocreateundeforpoison, memory(none), etc.
        We remove attribute group definitions and their references from
        function declarations.
        """
        import re

        lines = llir_text.split('\n')
        out_lines = []
        for line in lines:
            # Remove "attributes #N = { ... }" lines entirely
            if re.match(r'\s*attributes\s+#\d+\s*=\s*\{', line):
                continue
            # Remove #N references from declare/define lines
            line = re.sub(r'\s+#\d+\b', '', line)
            out_lines.append(line)

        return '\n'.join(out_lines)

    @staticmethod
    def _rename_llvm_kernel(llir_text, new_name):
        """Rename the LLVM kernel function from @kernel to @<new_name>.

        Triton's TTGIR uses a generic symbol @kernel; Metal's runtime
        launcher resolves kernels by their user-specified name. We rewrite
        the LLVM IR to use the correct symbol (including references in
        AIR metadata nodes like !air.kernel).
        """
        import re
        if not new_name or new_name == "kernel":
            return llir_text
        # Rewrite define line and all @kernel references (as a whole word)
        # so substrings like @kernel_fn aren't accidentally matched.
        pattern = re.compile(r'@kernel\b')
        return pattern.sub(f'@{new_name}', llir_text)

    # Mapping from LLVM intrinsics to AIR intrinsics for math functions
    # that Metal's runtime doesn't resolve as standard LLVM intrinsics.
    # Most llvm.* intrinsics work (exp, sin, cos, ...) but this provides
    # a safety net for any that don't.
    _LLVM_TO_AIR_INTRINSICS = {
        # These are only needed if Metal's runtime fails to resolve them.
        # Currently llvm.exp.f32, llvm.sin.f32, etc. all work.
        # Add entries here if specific intrinsics cause "Undefined symbols" errors.
    }

    # All Triton/TTG ops handled by the C++ path are listed in _has_complex_ops.
    # - tt.trans is a passthrough in the per-thread model
    # - ttg.local_alloc/local_load/convert_layout/memdesc_trans are stripped
    # - scf.for/if/yield are lowered by the scf-to-cf pass
    # tt.dot is NOT in the allowlist — it cleanly falls back to the MSL path
    # (_detect_simple_dot + template matmul), which is what FlashAttention uses.
    _CPP_UNSUPPORTED_OPS = set()  # empty — no fallback needed

    @staticmethod
    def _has_unsupported_ops(ttgir_text):
        """Check if TTGIR contains ops the C++ path cannot handle.

        Currently returns False for all inputs — all ops are handled.
        """
        return False

    @staticmethod
    def make_llir(mod, metadata, options):
        """Lower TTGIR to AIR-compatible LLVM IR using C++ MLIR passes.

        Pipeline: TTGIR text → strip TritonGPU annotations → C++ pass
        pipeline (Triton ops → LLVM dialect → LLVM IR) → AIR LLVM IR
        with Metal kernel metadata.

        The output is LLVM IR text that can be fed directly to Metal's
        compiler (xcrun metal -Xclang -opaque-pointers -c -x ir).

        If the TTGIR contains ops the C++ path cannot handle (tt.reduce,
        tt.dot, tt.trans), raises RuntimeError to trigger fallback to the
        Python/MSL path.
        """
        import triton_msl._triton_msl_cpp as cpp
        from triton_msl.debug import _debug_level, _dump_dir

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        # DEBUG: force named kernels off the C++ path (raise -> MSL fallback) to
        # bisect which C++-path kernel produces a wrong result. Comma-separated
        # substrings in TRITON_MSL_CPP_SKIP.
        _skip = os.environ.get("TRITON_MSL_CPP_SKIP", "")
        if _skip and any(s and s in kernel_name for s in _skip.split(",")):
            raise RuntimeError(f"C++ path skipped for {kernel_name} (TRITON_MSL_CPP_SKIP)")

        # Get TTGIR text (accept either MLIR module or pre-saved text)
        ttgir_text = mod if isinstance(mod, str) else str(mod)

        # Strip TritonGPU annotations; tt.dot is preserved and either
        # handled by the C++ DotOpToLLVM pattern or triggers fallback to
        # the MSL path (via _has_complex_ops).
        stripped = MetalBackend._strip_ttg_annotations(ttgir_text)

        if level >= 1:
            debug_dir = _dump_dir()
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{kernel_name}.ttgir"), "w") as f:
                f.write(ttgir_text)
            with open(os.path.join(debug_dir, f"{kernel_name}.stripped.mlir"), "w") as f:
                f.write(stripped)

        # -- Compute block_size early (needed for wrapping loop decision) ----
        import re
        block_size = options.num_warps * 32  # default
        # Collect (end, slice_dim) pairs, one per tt.make_range occurrence,
        # so we can compute block_size = prod(end for each distinct slice dim)
        # for 2D kernels (e.g. 32x32 matmul = 1024 threads), not prod(set()).
        mr_entries = []  # list of (end, slice_dim or None)
        for mr_match in re.finditer(
            r'tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)[^}]*\}\s*:\s*tensor<'
            r'\d+xi32(?:,\s*#ttg\.slice<\{dim\s*=\s*(\d+))?',
            ttgir_text,
        ):
            end = int(mr_match.group(1))
            dim = int(mr_match.group(2)) if mr_match.group(2) is not None else None
            mr_entries.append((end, dim))
        if mr_entries:
            dim_to_end = {}
            for end, dim in mr_entries:
                if dim is None:
                    # No slice -> 1D kernel; fall back to max end.
                    continue
                # If multiple make_range ops map to the same dim, take the
                # largest end (they represent the same axis of the tile).
                dim_to_end[dim] = max(dim_to_end.get(dim, 0), end)
            if len(dim_to_end) > 1:
                product = 1
                for e in dim_to_end.values():
                    product *= e
                block_size = product
            else:
                block_size = max(end for end, _ in mr_entries)
        # Run C++ pass pipeline: TTGIR → LLVM IR with AIR metadata
        air_llvm_ir_opaque = cpp.run_to_llvm(stripped)

        if level >= 2:
            debug_dir = _dump_dir()
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{kernel_name}.opaque.ll"), "w") as f:
                f.write(air_llvm_ir_opaque)

        # If total elements > 1024, inject a wrapping loop so that 1024
        # threads can cover all elements via a stride loop.
        if block_size > 1024:
            cap = 1024
            if level >= 1:
                debug_dir = _dump_dir()
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"{kernel_name}.pre_wrap.ll"), "w") as f:
                    f.write(air_llvm_ir_opaque)
            air_llvm_ir_opaque = MetalBackend._inject_wrapping_loop(
                air_llvm_ir_opaque, block_size, cap,
            )
            if level >= 1:
                with open(os.path.join(debug_dir, f"{kernel_name}.wrapped.ll"), "w") as f:
                    f.write(air_llvm_ir_opaque)

        # Metal's GPU JIT compiler requires typed pointers (old LLVM IR format).
        # Convert opaque pointers to typed pointers.
        air_llvm_ir = MetalBackend._opaque_to_typed_ptrs(air_llvm_ir_opaque)

        # Strip LLVM function attributes that Metal's compiler doesn't
        # understand (nocreateundeforpoison, memory(none), etc.).
        air_llvm_ir = MetalBackend._strip_unsupported_llvm_attrs(air_llvm_ir)

        # Rename the LLVM function from the generic @kernel to the user's
        # kernel name. The Metal launcher resolves kernels by metadata["name"]
        # which was set earlier in the pipeline (make_ttir/make_ttgir).
        # Without this, the metallib exports @kernel but the launcher asks
        # for @<user_fn_name>, producing "Kernel ... not found" errors.
        user_name = metadata.get("name")
        if user_name and user_name != "kernel":
            air_llvm_ir = MetalBackend._rename_llvm_kernel(air_llvm_ir, user_name)

        if level >= 1:
            with open(os.path.join(debug_dir, f"{kernel_name}.ll"), "w") as f:
                f.write(air_llvm_ir)

        # Extract kernel name from the LLVM IR (sanity check — should match
        # the name we just renamed to above).
        m = re.search(r'define\s+void\s+@(\w+)\s*\(', air_llvm_ir)
        if m:
            metadata["name"] = m.group(1)

        metadata.setdefault("block_size", min(block_size, 1024))

        # Detect 2D grid usage from program_id axes in the TTGIR.
        needs_2d = bool(re.search(r'tt\.get_program_id\s+y\b', ttgir_text))
        metadata.setdefault("needs_2d_grid", needs_2d)

        return air_llvm_ir

    @staticmethod
    def _inject_wrapping_loop(ir_text, total_elems, cap):
        """Inject a stride loop into LLVM IR so *cap* threads cover *total_elems*.

        Operates on opaque-pointer LLVM IR (before typed-pointer conversion).
        Each thread iterates: for (_wlid = lid; _wlid < total_elems; _wlid += cap).

        The generated IR replaces the single-pass per-thread model with a loop
        where each of the *cap* threads processes multiple elements.
        """
        import re

        # ---- locate function body ------------------------------------------
        # The argument list contains nested parens (e.g. addrspace(1)),
        # so we match the define line ending with ') {' then the body.
        fn_match = re.search(
            r'(define void @\w+\(.*?\) \{)\n(.*?)\n(\})',
            ir_text, re.DOTALL,
        )
        if not fn_match:
            raise RuntimeError("_inject_wrapping_loop: cannot find function body")

        fn_header = fn_match.group(1)   # define void @name(...) {
        body = fn_match.group(2)
        fn_close = fn_match.group(3)    # }

        # ---- determine entry block implicit label ---------------------------
        # The entry block label = next unused SSA number after unnamed args.
        # Named args (%pid, %lid) don't consume SSA numbers.
        # Extract args from fn_header by stripping 'define void @name(' ... ') {'
        args_match = re.search(r'@\w+\((.*)\)\s*\{', fn_header, re.DOTALL)
        unnamed_count = 0
        if args_match:
            for arg in args_match.group(1).split(','):
                arg = arg.strip()
                if arg and re.search(r'%\d+\s*$', arg):
                    unnamed_count += 1
        entry_label = str(unnamed_count)

        # ---- split body into lines and identify blocks ----------------------
        lines = body.split('\n')

        # Separate setup lines (addrspacecasts + addrspace(2) loads) from
        # compute lines in the entry block.
        setup_lines = []
        compute_lines = []
        past_setup = False
        in_entry = True
        other_blocks = []

        for line in lines:
            stripped = line.strip()
            # Detect start of a new basic block (numeric or named label)
            if re.match(r'^\d+:', stripped) or re.match(r'^[a-zA-Z_]\w*:', stripped):
                in_entry = False

            if in_entry:
                if not past_setup and (
                    'addrspacecast' in stripped
                    or ('load' in stripped and 'addrspace(2)' in stripped)
                    or stripped == ''
                ):
                    setup_lines.append(line)
                else:
                    past_setup = True
                    compute_lines.append(line)
            else:
                other_blocks.append(line)

        # ---- replace %lid with %_wlid in compute + other blocks ------------
        def replace_lid(text):
            # Replace %lid as a whole word (not inside other identifiers)
            return re.sub(r'%lid\b', '%_wlid', text)

        compute_text = replace_lid('\n'.join(compute_lines))
        other_text = replace_lid('\n'.join(other_blocks))

        # ---- rewrite phi incoming labels from %<entry> to %_wl_header -------
        # After wrapping, the entry block ends with `br label %_wl_header`
        # (not fall-through into the original compute block), so any phi in
        # `other_text` that named `%<entry_label>` as its incoming predecessor
        # must now name `%_wl_header`. Without this the LLVM verifier errors
        # out with "PHI node entries do not match predecessors!" and the
        # metallib compile aborts (seen on FA HEAD_DIM=64 where the scf.for
        # header phi referenced the entry block directly).
        def rewrite_phi_preds(text):
            # Match phi lines: `%foo = phi T [ val, %N ], [ val, %M ], ...`
            # Inside each bracketed pair, replace `%<entry>` with
            # `%_wl_header`. We scan line-by-line to avoid touching lines
            # like `br label %N` where %N is a branch target, not a phi pred.
            out_lines = []
            for ln in text.split('\n'):
                if ' = phi ' in ln:
                    ln = re.sub(
                        r'(\[\s*[^,\]]+,\s*)%' + re.escape(entry_label) + r'(\s*\])',
                        r'\g<1>%_wl_header\g<2>', ln,
                    )
                out_lines.append(ln)
            return '\n'.join(out_lines)

        compute_text = rewrite_phi_preds(compute_text)
        other_text = rewrite_phi_preds(other_text)

        # ---- find the merge block (the one with 'ret void') ----------------
        # In the other_blocks text, replace 'ret void' with a branch to the
        # loop latch. Also need to redirect any 'br label %<merge>' in the
        # conditional store block to branch to latch instead.

        # Find the merge block label (block containing ret void)
        merge_label = None
        for m in re.finditer(r'^(\d+):\s*;.*$', other_text, re.MULTILINE):
            # Check if this block contains ret void
            block_start = m.end()
            next_block = re.search(r'^\d+:', other_text[block_start:], re.MULTILINE)
            block_end = block_start + next_block.start() if next_block else len(other_text)
            block_body = other_text[block_start:block_end]
            if 'ret void' in block_body:
                merge_label = m.group(1)
                break

        # Also check if ret void is directly in compute_text (no conditional store)
        ret_in_compute = 'ret void' in compute_text and merge_label is None

        if ret_in_compute:
            # Simple case: no conditional branch, ret void directly in compute.
            # The compute block has a direct store + ret void.
            # Replace ret void with branch to latch.
            compute_text = compute_text.replace('ret void', 'br label %_wl_latch')

            new_body_parts = [
                '\n'.join(setup_lines),
                f'  br label %_wl_header',
                '',
                f'_wl_header:',
                f'  %_wlid = phi i32 [ %lid, %{entry_label} ], [ %_wlid_next, %_wl_latch ]',
                compute_text,
                other_text,
                '',
                f'_wl_latch:',
                f'  %_wlid_next = add i32 %_wlid, {cap}',
                f'  %_wl_cmp = icmp slt i32 %_wlid_next, {total_elems}',
                f'  br i1 %_wl_cmp, label %_wl_header, label %_wl_exit',
                '',
                f'_wl_exit:',
                f'  ret void',
            ]
        else:
            # Common case: conditional store with merge block.
            # The merge block has ret void. Replace it with branch to latch.
            # Also redirect branches to the merge block from the entry compute.
            if merge_label is None:
                # ret void might be at the end of other_text without a labeled block
                # (shouldn't happen with our IR, but handle gracefully)
                raise RuntimeError(
                    "_inject_wrapping_loop: cannot find merge block with ret void"
                )

            # Replace 'ret void' in the merge block with branch to latch
            other_text = other_text.replace('ret void', 'br label %_wl_latch')

            # In compute_text, redirect branches to merge label to go to latch.
            # e.g., "br i1 %6, label %11, label %12" where %12 is the merge.
            # The false branch (mask=false, skip store) should go to latch.
            compute_text = re.sub(
                r'br label %' + merge_label + r'\b',
                'br label %_wl_latch',
                compute_text,
            )
            compute_text = re.sub(
                r'(br i1 [^,]+, label %\d+), label %' + merge_label + r'\b',
                r'\1, label %_wl_latch',
                compute_text,
            )

            # In other_text, redirect branches to merge to go to latch
            other_text = re.sub(
                r'br label %' + merge_label + r'\b',
                'br label %_wl_latch',
                other_text,
            )

            # Update the preds comment in the merge block to reference latch
            # (cosmetic, not functionally required)

            new_body_parts = [
                '\n'.join(setup_lines),
                f'  br label %_wl_header',
                '',
                f'_wl_header:',
                f'  %_wlid = phi i32 [ %lid, %{entry_label} ], [ %_wlid_next, %_wl_latch ]',
                compute_text,
                other_text,
                '',
                f'_wl_latch:',
                f'  %_wlid_next = add i32 %_wlid, {cap}',
                f'  %_wl_cmp = icmp slt i32 %_wlid_next, {total_elems}',
                f'  br i1 %_wl_cmp, label %_wl_header, label %_wl_exit',
                '',
                f'_wl_exit:',
                f'  ret void',
            ]

        new_body = '\n'.join(new_body_parts)
        new_fn = f'{fn_header}\n{new_body}\n{fn_close}'

        # Replace the old function in the full IR text
        return ir_text[:fn_match.start()] + new_fn + ir_text[fn_match.end():]

    @staticmethod
    def make_metallib_from_llir(src, metadata, options):
        """Compile AIR LLVM IR to metallib using Metal's compiler.

        Pipeline: LLVM IR text → metal -c -x ir → .air → metallib
        This bypasses MSL entirely.
        """
        import shutil
        import time
        import warnings
        from triton_msl.debug import _debug_level, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        if level >= 2:
            t0 = time.perf_counter()

        try:
            cache_dir = _get_cache_dir()
            src_hash = _msl_cache_key(src, "")  # versioned (Phase 0)
            base = f"{kernel_name}_{src_hash}"

            metallib_path = os.path.join(cache_dir, f"{base}.metallib")

            # Skip compilation if cached metallib exists.  The read is wrapped:
            # a concurrent cache clear (or external rm -rf) can delete the file
            # between os.path.exists and open() (TOCTOU).  Treat a vanished file
            # as a cache miss and fall through to (re)compile rather than raising.
            if os.path.exists(metallib_path):
                if level >= 2:
                    print(
                        f"[triton-msl] make_metallib_from_llir({kernel_name}): cache hit",
                        file=sys.stderr,
                    )
                try:
                    with open(metallib_path, "rb") as f:
                        return f.read()
                except FileNotFoundError:
                    pass  # cached metallib vanished mid-read → recompile below

            # Bounded retry loop: each attempt gets a fresh private work dir so
            # intermediates from failed attempts never interfere with the next.
            # A CalledProcessError from the IR→AIR step is a REAL deterministic
            # compile error → raised immediately (no retry).  All other failures
            # (missing .air after exit-0, metallib link error, os.replace
            # FileNotFoundError) are treated as transient toolchain flakes and
            # retried; the last attempt re-raises as MetalCompilationError.
            _last_transient_exc: Exception | None = None
            for _attempt in range(_METALLIB_COMPILE_ATTEMPTS):
                # Per-call private work directory: each concurrent invocation of the
                # same kernel gets its own unique dir, so intermediates never collide.
                # All paths are on the same filesystem as metallib_path, so the final
                # os.replace() stays atomic.  The work dir is removed in finally even
                # if compilation fails.
                work = tempfile.mkdtemp(dir=cache_dir)
                try:
                    ll_path = os.path.join(work, f"{base}.ll")
                    air_path = os.path.join(work, f"{base}.air")
                    tmp_metallib_path = os.path.join(work, "out.metallib")

                    with open(ll_path, "w") as f:
                        f.write(src)

                    # Compile LLVM IR → AIR using Metal's compiler
                    # Our IR uses typed pointers (Metal's GPU JIT requires them).
                    # CalledProcessError here = real deterministic IR error → raise
                    # immediately, no retry (retrying a syntax error wastes time).
                    try:
                        subprocess.run(
                            [
                                "xcrun", "-sdk", "macosx", "metal",
                                "-c", "-x", "ir",
                                ll_path,
                                "-o", air_path,
                            ],
                            capture_output=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        from triton_msl.errors import MetalCompilationError
                        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                        raise MetalCompilationError(
                            f"Metal IR compilation failed (exit {e.returncode})",
                            msl_source=ll_path,
                            stderr=stderr,
                        ) from None

                    # Explicit transient guard: metal -c returned exit 0 but the
                    # .air was not durably written.  Treat as transient (retry).
                    if not os.path.exists(air_path):
                        _last_transient_exc = FileNotFoundError(
                            f"metal -c exited 0 but {air_path!r} is missing (attempt {_attempt + 1})"
                        )
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal IR compilation produced no .air (transient; all {_METALLIB_COMPILE_ATTEMPTS} attempts failed)",
                            msl_source=ll_path,
                            stderr=str(_last_transient_exc),
                        ) from _last_transient_exc

                    # Link AIR → metallib (atomic rename onto content-addressed final path).
                    # Concurrent callers each rename their OWN unique tmp onto the same
                    # final path — last-writer-wins with identical content; all safe.
                    try:
                        subprocess.run(
                            [
                                "xcrun", "-sdk", "macosx", "metallib",
                                air_path,
                                "-o", tmp_metallib_path,
                            ],
                            capture_output=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                        _last_transient_exc = subprocess.CalledProcessError(
                            e.returncode, e.cmd, e.output, e.stderr
                        )
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library linking failed (exit {e.returncode})",
                            msl_source=air_path,
                            stderr=stderr,
                        ) from None

                    try:
                        os.replace(tmp_metallib_path, metallib_path)
                    except (FileNotFoundError, OSError) as e:
                        _last_transient_exc = e
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library linking failed (os.replace error: {e})",
                            msl_source=air_path,
                            stderr=str(e),
                        ) from e

                    # Read the freshly-placed metallib INSIDE the retry loop, so a
                    # concurrent cache clear that deletes it between os.replace and
                    # this read is handled like the other transients (retry; raise
                    # MetalCompilationError on the last attempt) rather than escaping
                    # as a bare FileNotFoundError.
                    try:
                        with open(metallib_path, "rb") as f:
                            data = f.read()
                    except FileNotFoundError as e:
                        _last_transient_exc = e
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library read failed (metallib vanished after replace: {e})",
                            msl_source=air_path,
                            stderr=str(e),
                        ) from e

                    # Success — break out of the retry loop.
                    break

                finally:
                    shutil.rmtree(work, ignore_errors=True)

            if level >= 2:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[triton-msl] make_metallib_from_llir({kernel_name}): {elapsed_ms:.1f}ms",
                    file=sys.stderr,
                )

            return data

        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-msl: Metal IR compilation failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                raise
            raise

    @staticmethod
    def gluon_to_ttgir(mod, metadata, options):
        """Convert Gluon IR to TTGIR for Metal.

        Gluon is Triton's higher-level language. The conversion applies
        Gluon-specific passes (inliner, encoding resolution) then standard
        TTGIR conversion passes. Metal-specific: no TMA, no NVIDIA passes.
        """
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        pm.run(mod, "gluon_to_ttgir")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_ttir(mod, metadata, options):
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        # Block pointers are now lowered in the Python frontend (upstream
        # b939621a0 "Rewrite block pointer to be python-only"); the C++
        # add_rewrite_tensor_pointer pass was removed in that change.
        # Metal has no TMA — always rewrite tensor descriptors to pointers.
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        import sys
        import time
        from triton._C.libtriton import ir, passes
        from triton_msl.debug import _debug_level

        level = _debug_level()
        if level >= 2:
            t0 = time.perf_counter()

        pm = ir.pass_manager(mod.context)
        target_str = f"metal:{options.arch}"
        passes.ttir.add_convert_to_ttgpuir(
            pm, target_str, options.num_warps, 32, options.num_ctas
        )

        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, False)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttgir")
        metadata["tensordesc_meta"] = None
        # Extract shared memory requirement from MLIR module if available.
        try:
            shared = mod.get_int_attr("ttg.shared")
        except Exception:
            shared = None
        # ``shared`` reports threadgroup-memory usage in bytes. Upstream
        # tutorials compute occupancy as ``SIZE_SMEM // metadata.shared``
        # and crash on 0 (ZeroDivisionError) — report at least 1 byte so
        # the division evaluates to ``SIZE_SMEM`` (i.e., no constraint),
        # which is the correct semantic when the kernel uses no TG memory.
        metadata["shared"] = max(1, shared if shared is not None else 0)

        if level >= 2:
            kernel_name = metadata.get("name", "kernel")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[triton-msl] make_ttgir({kernel_name}): {elapsed_ms:.1f}ms",
                file=sys.stderr,
            )

        return mod

    @staticmethod
    def make_msl(mod, metadata, options):
        import sys
        import time
        import warnings
        from triton_msl.codegen.msl_emitter import emit_msl
        from triton_msl.debug import _debug_level, _dump_dir, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        # Level 1+: dump raw TTGIR before lowering
        if level >= 1:
            debug_dir = _dump_dir()
            os.makedirs(debug_dir, exist_ok=True)
            ttgir_path = os.path.join(debug_dir, f"{kernel_name}.ttgir")
            with open(ttgir_path, "w") as f:
                f.write(str(mod))

        # Check persistent MSL cache (TTGIR text + options + codegen version
        # + MEPT flag → MSL string).
        mod_text = str(mod)
        cache_key = _msl_cache_key(mod_text, options.hash())
        cache_dir = _get_cache_dir()
        msl_cache_path = os.path.join(cache_dir, f"{kernel_name}_{cache_key}.msl")

        if os.path.exists(msl_cache_path):
            with open(msl_cache_path, "r") as f:
                msl_src = f.read()

            # Populate metadata that emit_msl would normally set.
            # Extract kernel name from MSL: "kernel void NAME("
            import re as _re
            m = _re.search(r'kernel\s+void\s+(\w+)\s*\(', msl_src)
            if m:
                metadata["name"] = m.group(1)
            else:
                metadata["name"] = kernel_name
            # block_size and output_arg_indices default if not cached
            metadata.setdefault("block_size", options.num_warps * 32)
            metadata.setdefault("needs_2d_grid", False)

            # Try to load cached metadata alongside the MSL
            meta_cache_path = msl_cache_path.replace(".msl", ".meta.json")
            if os.path.exists(meta_cache_path):
                import json
                with open(meta_cache_path, "r") as f:
                    cached_meta = json.load(f)
                metadata.update(cached_meta)

            if level >= 2:
                print(
                    f"[triton-msl] make_msl({kernel_name}): cache hit",
                    file=sys.stderr,
                )

            # Stash MSL keyed on the kernel's content hash (cache_key) and
            # expose that key to the launcher via metadata for the launcher's
            # compile_shader fast-path (Phase 4). Content-keyed so a later
            # compile in the same process can't clobber this kernel's MSL.
            metadata["msl_hash"] = cache_key
            _stash_msl(msl_src, cache_key, metadata.get("block_size"))
            return msl_src

        # Level 2: time the MSL emission
        if level >= 2:
            t0 = time.perf_counter()

        try:
            msl_src = emit_msl(mod, metadata, options)
        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-msl: MSL codegen failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                # Re-raise without fallback hint — user wants hard errors.
                raise
            # "silent" and "warn" both re-raise so Triton/torch.compile
            # can route to CPU fallback.
            raise

        if level >= 2:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[triton-msl] make_msl({kernel_name}): {elapsed_ms:.1f}ms",
                file=sys.stderr,
            )

        # Cache the generated MSL and metadata atomically.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=cache_dir, suffix=".msl.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(msl_src)
            os.replace(tmp_path, msl_cache_path)
            # Cache metadata (name, block_size, etc.) alongside the MSL.
            import json
            meta_cache_path = msl_cache_path.replace(".msl", ".meta.json")
            cacheable = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, type(None), list, tuple))
            }
            tmp_meta_fd, tmp_meta_path = tempfile.mkstemp(
                dir=cache_dir, suffix=".meta.tmp"
            )
            with os.fdopen(tmp_meta_fd, "w") as f:
                json.dump(cacheable, f)
            os.replace(tmp_meta_path, meta_cache_path)
        except Exception:
            # Best-effort cleanup on failure; compilation still succeeds.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Level 1+: dump generated MSL
        if level >= 1:
            msl_path = os.path.join(debug_dir, f"{kernel_name}.msl")
            with open(msl_path, "w") as f:
                f.write(msl_src)

        # Stash MSL keyed on the kernel's content hash (cache_key) and expose
        # that key to the launcher via metadata for the launcher's
        # compile_shader fast-path (Phase 4). Content-keyed so a later compile
        # in the same process can't clobber this kernel's MSL.
        metadata["msl_hash"] = cache_key
        _stash_msl(msl_src, cache_key, metadata.get("block_size"))
        return msl_src

    @staticmethod
    def make_metallib(src, metadata, options):
        import shutil
        import sys
        import time
        import warnings
        from triton_msl.debug import _debug_level, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        if level >= 2:
            t0 = time.perf_counter()

        try:
            # Persistent cache directory (survives reboots).
            cache_dir = _get_cache_dir()

            # Use content hash for deterministic naming.
            src_hash = _msl_cache_key(src, "")  # versioned (Phase 0)
            base = f"{kernel_name}_{src_hash}"

            metallib_path = os.path.join(cache_dir, f"{base}.metallib")

            # Skip compilation if cached metallib exists.  The read is wrapped:
            # a concurrent cache clear (or external rm -rf) can delete the file
            # between os.path.exists and open() (TOCTOU).  Treat a vanished file
            # as a cache miss and fall through to (re)compile rather than raising.
            if os.path.exists(metallib_path):
                if level >= 2:
                    print(
                        f"[triton-msl] make_metallib({kernel_name}): cache hit",
                        file=sys.stderr,
                    )
                try:
                    with open(metallib_path, "rb") as f:
                        return f.read()
                except FileNotFoundError:
                    pass  # cached metallib vanished mid-read → recompile below

            # Resolve Metal standard version for compilation (done once, outside
            # the retry loop — it's deterministic and has no side-effects).
            if options.target_metal_version == "auto":
                from triton_msl.backend.device_detect import get_device_info
                metal_std_flag = get_device_info().metal_std_flag
            else:
                metal_std_flag = f"-std=metal{options.target_metal_version}"

            # Bounded retry loop: each attempt gets a fresh private work dir so
            # intermediates from failed attempts never interfere with the next.
            # A CalledProcessError from the MSL→AIR step is a REAL deterministic
            # compile error → raised immediately (no retry).  All other failures
            # (missing .air after exit-0, metallib link error, os.replace
            # FileNotFoundError) are treated as transient toolchain flakes and
            # retried; the last attempt re-raises as MetalCompilationError.
            _last_transient_exc: Exception | None = None
            for _attempt in range(_METALLIB_COMPILE_ATTEMPTS):
                # Per-call private work directory: each concurrent invocation of the
                # same kernel gets its own unique dir, so intermediates never collide.
                # All paths are on the same filesystem as metallib_path, so the final
                # os.replace() stays atomic.  The work dir is removed in finally even
                # if compilation fails.
                work = tempfile.mkdtemp(dir=cache_dir)
                try:
                    metal_path = os.path.join(work, f"{base}.metal")
                    air_path = os.path.join(work, f"{base}.air")
                    tmp_metallib_path = os.path.join(work, "out.metallib")

                    with open(metal_path, "w") as f:
                        f.write(src)

                    # Compile MSL -> AIR
                    # -fno-fast-math: disable algebraic re-association so IEEE-754
                    # rounding tricks like (x + 2^23) - 2^23 are preserved verbatim.
                    # The Triton test suite (test_conversions.py) relies on this
                    # idiom for round-to-nearest-even emulation; with fast-math the
                    # add/sub gets folded away, breaking rounding correctness.
                    # CalledProcessError here = real deterministic MSL error → raise
                    # immediately, no retry (retrying a syntax error wastes time).
                    try:
                        subprocess.run(
                            [
                                "xcrun", "-sdk", "macosx", "metal",
                                "-c", metal_path,
                                "-o", air_path,
                                metal_std_flag,
                                "-mmacosx-version-min=15.0",
                                "-O2",
                                "-fno-fast-math",
                            ],
                            capture_output=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        import re
                        from triton_msl.errors import MetalCompilationError
                        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                        # Distinguish a REAL deterministic MSL error from a
                        # TRANSIENT toolchain flake. A genuine compile error
                        # carries a source-location diagnostic
                        # (`file:line:col: error:`) — raise it immediately
                        # (retrying a syntax error wastes time). A nonzero exit
                        # WITHOUT such a diagnostic (empty stderr, a spawn/signal
                        # failure, or an SDK/temp race under heavy parallel load)
                        # is transient: retry like the other metallib steps so a
                        # flaky `metal -c` doesn't surface as a failed (NaN)
                        # kernel in the torch.compile / training path.
                        if re.search(r":\d+:\d+:\s+(error|fatal error):", stderr):
                            raise MetalCompilationError(
                                f"Metal shader compilation failed (exit {e.returncode})",
                                msl_source=metal_path,
                                stderr=stderr,
                            ) from None
                        _last_transient_exc = e
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        raise MetalCompilationError(
                            f"Metal shader compilation failed (exit {e.returncode}; "
                            f"transient, all {_METALLIB_COMPILE_ATTEMPTS} attempts failed)",
                            msl_source=metal_path,
                            stderr=stderr,
                        ) from None

                    # Explicit transient guard: metal -c returned exit 0 but the
                    # .air was not durably written.  Treat as transient (retry).
                    if not os.path.exists(air_path):
                        _last_transient_exc = FileNotFoundError(
                            f"metal -c exited 0 but {air_path!r} is missing (attempt {_attempt + 1})"
                        )
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal shader compilation produced no .air (transient; all {_METALLIB_COMPILE_ATTEMPTS} attempts failed)",
                            msl_source=metal_path,
                            stderr=str(_last_transient_exc),
                        ) from _last_transient_exc

                    # Link AIR -> metallib (atomic rename onto content-addressed final path).
                    # Concurrent callers each rename their OWN unique tmp onto the same
                    # final path — last-writer-wins with identical content; all safe.
                    try:
                        subprocess.run(
                            [
                                "xcrun", "-sdk", "macosx", "metallib",
                                air_path,
                                "-o", tmp_metallib_path,
                            ],
                            capture_output=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                        _last_transient_exc = subprocess.CalledProcessError(
                            e.returncode, e.cmd, e.output, e.stderr
                        )
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library linking failed (exit {e.returncode})",
                            msl_source=air_path,
                            stderr=stderr,
                        ) from None

                    try:
                        os.replace(tmp_metallib_path, metallib_path)
                    except (FileNotFoundError, OSError) as e:
                        _last_transient_exc = e
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library linking failed (os.replace error: {e})",
                            msl_source=air_path,
                            stderr=str(e),
                        ) from e

                    # Read the freshly-placed metallib INSIDE the retry loop, so a
                    # concurrent cache clear that deletes it between os.replace and
                    # this read is handled like the other transients (retry; raise
                    # MetalCompilationError on the last attempt) rather than escaping
                    # as a bare FileNotFoundError.
                    try:
                        with open(metallib_path, "rb") as f:
                            data = f.read()
                    except FileNotFoundError as e:
                        _last_transient_exc = e
                        if _attempt < _METALLIB_COMPILE_ATTEMPTS - 1:
                            time.sleep(0.05 * (_attempt + 1))
                            continue
                        from triton_msl.errors import MetalCompilationError
                        raise MetalCompilationError(
                            f"Metal library read failed (metallib vanished after replace: {e})",
                            msl_source=air_path,
                            stderr=str(e),
                        ) from e

                    # Success — break out of the retry loop.
                    break

                finally:
                    shutil.rmtree(work, ignore_errors=True)

            if level >= 2:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[triton-msl] make_metallib({kernel_name}): {elapsed_ms:.1f}ms",
                    file=sys.stderr,
                )

            return data

        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-msl: Metal compilation failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                # Re-raise without fallback hint -- user wants hard errors.
                raise
            # "silent" and "warn" both re-raise so Triton/torch.compile
            # can route to CPU fallback.
            raise

    @functools.lru_cache()
    def hash(self):
        try:
            sdk_version = subprocess.check_output(
                ["xcrun", "--show-sdk-version"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sdk_version = "unknown"
        return f"metal-{sdk_version}-{self.target.arch}"
