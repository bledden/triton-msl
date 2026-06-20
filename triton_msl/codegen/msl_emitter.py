"""Emit Metal Shading Language (MSL) source from kernel descriptions.

Two-layer architecture:
1. KernelBuilder — captures kernel semantics (args, block size, ops)
2. MSLCodeGen — emits valid MSL compute kernel source from a KernelBuilder

The KernelBuilder can be driven by:
- Direct Python API (standalone, no triton required)
- TTGIR MLIR walking (when triton is available)

Supports:
- Elementwise ops: vector add, scalar mul, activation functions (silu, gelu)
- Reductions: sum, max, min via SIMD-group intrinsics + threadgroup shared memory
- Softmax: fused row-wise max → subtract → exp → sum → divide
- Matmul: tiled matrix multiplication with threadgroup shared memory
- Layer norm: mean → variance → normalize with gamma/beta
- Cross-entropy: fused log-softmax + target selection loss
- Flash Attention: online softmax with tiled Q@K^T and P@V accumulation
"""

import os

from triton_msl.codegen.msl_types import triton_type_to_msl


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _msl_compute_type(dtype):
    """Get the MSL compute type for a Triton dtype.

    For fp16, computations are done in float and cast back to half on store.
    This matches Triton's behavior and avoids precision issues.
    FP8 types always compute in float — stored as uchar, converted on load/store.
    """
    if dtype in ("fp16", "bf16"):
        return "float"
    from triton_msl.codegen.msl_builtins import is_fp8_type
    if is_fp8_type(dtype):
        return "float"
    return triton_type_to_msl(dtype)


def _msl_zero(dtype):
    """Get the zero literal for a MSL type."""
    if dtype in ("fp16", "bf16", "fp32", "f32"):
        return "0.0f"
    from triton_msl.codegen.msl_builtins import is_fp8_type
    if is_fp8_type(dtype):
        return "0.0f"  # FP8 computes in float
    return "0"


# ---------------------------------------------------------------------------
# Kernel description API
# ---------------------------------------------------------------------------

class Arg:
    """A kernel argument (buffer pointer or scalar)."""

    def __init__(self, name, dtype, is_ptr=False, const=False):
        self.name = name
        self.dtype = dtype  # Triton type string: "fp32", "i32", etc.
        self.is_ptr = is_ptr
        self.const = const  # Read-only buffer

    def msl_param(self, index):
        """Emit MSL kernel parameter declaration.

        Non-const pointers use 'volatile device' to prevent the Metal shader
        compiler from hoisting loads out of while/for loops, which breaks
        read-modify-write patterns on the same memory location.
        """
        if self.is_ptr:
            inner = triton_type_to_msl(self.dtype)
            if self.const:
                return f"device const {inner}* {self.name} [[buffer({index})]]"
            else:
                return f"volatile device {inner}* {self.name} [[buffer({index})]]"
        else:
            msl_ty = triton_type_to_msl(self.dtype)
            return f"constant {msl_ty}& {self.name} [[buffer({index})]]"


def _sanitize_msl_name(name: str) -> str:
    """Ensure a kernel name doesn't clash with MSL reserved words."""
    if name in _MSL_RESERVED:
        return f"{name}_fn"
    return name


_MSL_RESERVED = frozenset({
    "kernel", "vertex", "fragment", "device", "constant", "threadgroup",
    "thread", "texture", "sampler", "float", "half", "int", "uint",
    "bool", "char", "short", "void", "using", "namespace", "metal",
    "return", "if", "else", "for", "while", "do", "switch", "case",
    "break", "continue", "struct", "class", "enum", "true", "false",
})


class KernelBuilder:
    """Describes a compute kernel's structure for MSL emission."""

    def __init__(self, name, block_size=256):
        self.name = _sanitize_msl_name(name)
        self.block_size = block_size
        self.args = []
        self._body_lines = []
        self._locals = {}
        self._indent = 1
        self._needs_simd_qualifiers = False
        self._needs_num_programs = False  # Whether kernel uses tt.get_num_programs
        self._threadgroup_arrays = []  # (name, dtype, size) for static tg memory
        self._prebuilt_msl = None  # Raw MSL string when using pre-made kernels
        self._device_functions = []  # List of MSL device function source strings

    def set_prebuilt_msl(self, msl_source):
        """Set a pre-generated MSL string, bypassing the builder's code gen."""
        self._prebuilt_msl = msl_source

    # -- Argument registration --

    def add_ptr_arg(self, name, dtype="fp32", const=False):
        """Add a device buffer pointer argument."""
        self.args.append(Arg(name, dtype, is_ptr=True, const=const))
        return name

    def add_scalar_arg(self, name, dtype="i32"):
        """Add a scalar argument (passed as constant buffer)."""
        self.args.append(Arg(name, dtype, is_ptr=False))
        return name

    # -- Code generation helpers --

    def _emit(self, line):
        prefix = "    " * self._indent
        self._body_lines.append(f"{prefix}{line}")

    def _var(self, name, expr, ty="auto"):
        """Declare and assign a local variable."""
        self._emit(f"{ty} {name} = {expr};")
        self._locals[name] = ty
        return name

    # -- Triton-like operations --

    def get_program_id(self, var_name="pid"):
        """tt.get_program_id(0) -> threadgroup_position_in_grid."""
        return var_name  # injected as kernel parameter

    def make_block_offsets(self, pid_var="pid", out_var="offsets"):
        """Compute per-thread offsets within a 1D block.

        offsets = pid * BLOCK_SIZE + thread_position_in_threadgroup
        """
        self._var(out_var, f"{pid_var} * {self.block_size} + lid")
        return out_var

    def make_mask(self, offsets_var, n_var, out_var="mask"):
        """Generate a bounds mask: offsets < n_elements."""
        self._var(out_var, f"{offsets_var} < {n_var}", ty="bool")
        return out_var

    def load(self, ptr_var, offsets_var, mask_var=None, out_var=None, dtype="fp32"):
        """Masked load from a buffer pointer + offset.

        For FP16/BF16 buffers, the value is promoted to float for computation.
        """
        if out_var is None:
            out_var = f"{ptr_var}_val"
        compute_ty = _msl_compute_type(dtype)
        zero = _msl_zero(dtype)
        if mask_var:
            self._emit(f"{compute_ty} {out_var} = {mask_var} ? "
                       f"static_cast<{compute_ty}>({ptr_var}[{offsets_var}]) : {zero};")
        else:
            self._emit(f"{compute_ty} {out_var} = "
                       f"static_cast<{compute_ty}>({ptr_var}[{offsets_var}]);")
        return out_var

    def store(self, ptr_var, offsets_var, val_var, mask_var=None, dtype="fp32"):
        """Masked store to a buffer pointer + offset.

        For FP16/BF16 buffers, casts from float compute type back to storage type.
        """
        store_ty = triton_type_to_msl(dtype)
        compute_ty = _msl_compute_type(dtype)
        needs_cast = (store_ty != compute_ty)
        cast_val = f"static_cast<{store_ty}>({val_var})" if needs_cast else val_var
        if mask_var:
            self._emit(f"if ({mask_var}) {{ {ptr_var}[{offsets_var}] = {cast_val}; }}")
        else:
            self._emit(f"{ptr_var}[{offsets_var}] = {cast_val};")

    def binary_op(self, op, a_var, b_var, out_var):
        """Emit a binary operation: out = a op b."""
        op_map = {
            "add": "+", "sub": "-", "mul": "*", "div": "/",
            "mod": "%", "and": "&", "or": "|", "xor": "^",
        }
        if op in op_map:
            self._var(out_var, f"{a_var} {op_map[op]} {b_var}", ty="float")
        else:
            raise ValueError(f"Unknown binary op: {op}")
        return out_var

    def unary_op(self, op, x_var, out_var):
        """Emit a unary operation."""
        op_map = {
            "neg": f"-{x_var}",
            "exp": f"exp({x_var})",
            "log": f"log({x_var})",
            "sqrt": f"sqrt({x_var})",
            "rsqrt": f"rsqrt({x_var})",
            "abs": f"abs({x_var})",
            "sigmoid": f"(1.0f / (1.0f + exp(-{x_var})))",
            "tanh": f"tanh({x_var})",
            "sin": f"sin({x_var})",
            "cos": f"cos({x_var})",
        }
        if op in op_map:
            self._var(out_var, op_map[op], ty="float")
        else:
            raise ValueError(f"Unknown unary op: {op}")
        return out_var

    def fused_op(self, op_name, args_vars, out_var):
        """Emit a fused multi-input operation."""
        fused_map = {
            "fma": lambda a: f"fma({a[0]}, {a[1]}, {a[2]})",
            "silu": lambda a: f"({a[0]} / (1.0f + exp(-{a[0]})))",
            # MSL has no erf(). Both gelu variants use the tanh approximation
            # which is standard in ML frameworks.
            "gelu": lambda a: (
                f"({a[0]} * 0.5f * (1.0f + tanh(0.7978845608028654f * "
                f"({a[0]} + 0.044715f * {a[0]} * {a[0]} * {a[0]}))))"
            ),
            "gelu_tanh": lambda a: (
                f"({a[0]} * 0.5f * (1.0f + tanh(0.7978845608028654f * "
                f"({a[0]} + 0.044715f * {a[0]} * {a[0]} * {a[0]}))))"
            ),
        }
        if op_name in fused_map:
            self._var(out_var, fused_map[op_name](args_vars), ty="float")
        else:
            raise ValueError(f"Unknown fused op: {op_name}")
        return out_var

    def raw_line(self, line):
        """Emit a raw MSL line."""
        self._emit(line)

    def comment(self, text):
        """Emit a comment."""
        self._emit(f"// {text}")

    # -- Indentation control --

    def indent(self):
        """Increase indentation level."""
        self._indent += 1

    def dedent(self):
        """Decrease indentation level."""
        self._indent = max(1, self._indent - 1)

    def begin_if(self, condition):
        """Emit an if statement and increase indent."""
        self._emit(f"if ({condition}) {{")
        self._indent += 1

    def end_block(self):
        """Close a block and decrease indent."""
        self._indent -= 1
        self._emit("}")

    # -- Shared memory and barriers --

    def declare_threadgroup_array(self, name, dtype="fp32", size=None):
        """Declare a static threadgroup memory array."""
        if size is None:
            size = (self.block_size + 31) // 32  # one slot per SIMD group
        self._threadgroup_arrays.append((name, dtype, size))
        return name

    def barrier(self, kind="threadgroup"):
        """Emit a memory barrier."""
        from triton_msl.codegen.msl_builtins import BARRIERS
        self._emit(f"{BARRIERS[kind]};")

    # -- Reduction operations --

    def simd_reduce(self, op, val_var, out_var):
        """Emit a SIMD-group reduction: out = simd_op(val).

        Uses hardware SIMD intrinsics (32-wide on Apple Silicon).
        """
        self._needs_simd_qualifiers = True
        from triton_msl.codegen.msl_builtins import SIMD_REDUCTIONS
        intrinsic = SIMD_REDUCTIONS[op]
        self._var(out_var, f"{intrinsic}({val_var})", ty="float")
        return out_var

    def threadgroup_reduce(self, op, val_var, shared_var, out_var):
        """Emit a full threadgroup reduction: SIMD reduce → shared mem → final SIMD reduce.

        Standard two-level pattern:
        1. simd_op within each SIMD group
        2. Lane 0 writes to shared memory
        3. Barrier
        4. SIMD group 0 reads shared and does final reduction

        Variable names are suffixed with out_var to avoid collisions when
        called multiple times in the same kernel.
        """
        # Defense-in-depth: the step-1 `simd_op(val)` below reduces over the full
        # 32-wide SIMD group. When block_size > 32 and is NOT a multiple of 32, the
        # trailing SIMD group has inactive lanes (e.g. block_size=48 -> lanes 48-63
        # of group 1 never execute), and Apple leaves simd_* over inactive lanes
        # UNDEFINED — so the first-level reduction would fold in garbage, silently.
        # Triton's tl.arange is power-of-2, so block_size is normally pow2 (mult of
        # 32) and templates always pad to a multiple of 32; this guard exists for an
        # out-of-contract block_size (e.g. an inductor-fused odd tile) reaching here.
        # Refuse loudly rather than emit a silently-wrong reduction.
        if self.block_size > 32 and self.block_size % 32 != 0:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"threadgroup reduction over a {self.block_size}-element tile spans "
                f"a partial trailing SIMD group (block_size is not a multiple of 32). "
                f"Apple does not define simd-group reductions over inactive lanes, so "
                f"the result would be silently wrong; refusing. Pad the reduction tile "
                f"to a multiple of 32 (the matmul/softmax templates already do this).")
        self._needs_simd_qualifiers = True
        from triton_msl.codegen.msl_builtins import SIMD_REDUCTIONS

        intrinsic = SIMD_REDUCTIONS[op]
        identity = {
            "sum": "0.0f",
            "max": "-INFINITY",
            "min": "INFINITY",
        }[op]

        # Unique intermediate variable names
        simd_var = f"simd_{out_var}"
        read_var = f"shared_{out_var}"

        # Step 1: SIMD-level reduction
        self._var(simd_var, f"{intrinsic}({val_var})", ty="float")

        n_simd_groups = (self.block_size + 31) // 32

        # Step 2: Initialize shared memory (bounds-guarded). A leading
        # barrier here is required because ``shared_var`` may be reused
        # by an earlier reduction in the same kernel (e.g. softmax\'s
        # max then sum, or any loop body re-entering the reduction).
        # Without it, SG 0\'s init writes can race with another SG\'s
        # final read in step 4 — manifests as 350/1823 rows mismatching
        # with the persistent-softmax tutorial pattern.
        self.barrier("threadgroup")
        self.begin_if(f"sgitg == 0 && tiisg < {n_simd_groups}u")
        self._emit(f"{shared_var}[tiisg] = {identity};")
        self.end_block()
        self.barrier("threadgroup")

        # Step 3: Lane 0 of each SIMD group writes to shared
        self.begin_if("tiisg == 0")
        self._emit(f"{shared_var}[sgitg] = {simd_var};")
        self.end_block()
        self.barrier("threadgroup")

        # Step 4: SIMD group 0 reads back and does final reduction (bounds-guarded)
        self._var(read_var, f"(tiisg < {n_simd_groups}u) ? {shared_var}[tiisg] : {identity}", ty="float")
        self._var(out_var, f"{intrinsic}({read_var})", ty="float")
        return out_var

    # -- Build the MSL source --

    def build(self):
        """Generate the complete MSL kernel source."""
        gen = MSLCodeGen(self)
        return gen.emit()


class MSLCodeGen:
    """Generates MSL source from a KernelBuilder."""

    def __init__(self, builder):
        self.builder = builder

    def emit(self):
        # Return prebuilt MSL if available (e.g., matmul kernels).
        # Substitute the kernel function name to match the TTGIR function name.
        if self.builder._prebuilt_msl is not None:
            msl = self.builder._prebuilt_msl
            import re as _re
            msl = _re.sub(
                r'kernel\s+void\s+\w+\s*\(',
                f'kernel void {self.builder.name}(',
                msl,
                count=1,
            )
            return msl

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")

        # Device functions (noinline callees) — must appear before the kernel
        for dev_fn in self.builder._device_functions:
            lines.append(dev_fn)
            lines.append("")

        # Kernel signature
        params = []
        for i, arg in enumerate(self.builder.args):
            params.append(f"    {arg.msl_param(i)}")

        # Thread position qualifiers — Metal requires all position attrs same type
        used_axes = getattr(self.builder, '_used_pid_axes', {0})
        if used_axes and max(used_axes) > 0:
            params.append("    uint3 pid3 [[threadgroup_position_in_grid]]")
            params.append("    uint3 lid3 [[thread_position_in_threadgroup]]")
            params.append("    uint3 tid3 [[thread_position_in_grid]]")
        else:
            params.append("    uint pid [[threadgroup_position_in_grid]]")
            params.append("    uint lid [[thread_position_in_threadgroup]]")
            params.append("    uint tid [[thread_position_in_grid]]")

        # SIMD qualifiers (only when reductions are used)
        if self.builder._needs_simd_qualifiers:
            params.append("    uint sgitg [[simdgroup_index_in_threadgroup]]")
            params.append("    uint tiisg [[thread_index_in_simdgroup]]")

        # Grid size (when kernel uses tt.get_num_programs)
        if self.builder._needs_num_programs:
            if used_axes and max(used_axes) > 0:
                params.append("    uint3 tpg3 [[threadgroups_per_grid]]")
            else:
                params.append("    uint tpg [[threadgroups_per_grid]]")

        lines.append(f"kernel void {self.builder.name}(")
        lines.append(",\n".join(params))
        lines.append(") {")

        # Decompose uint3 position attrs into scalar values
        if used_axes and max(used_axes) > 0:
            lines.append("    uint pid = pid3.x;")
            lines.append("    uint lid = lid3.x;")
            lines.append("    uint tid = tid3.x;")
            if 1 in used_axes:
                lines.append("    uint pid_y = pid3.y;")
            if 2 in used_axes:
                lines.append("    uint pid_z = pid3.z;")
            if self.builder._needs_num_programs:
                lines.append("    uint tpg = tpg3.x;")
                if 1 in used_axes:
                    lines.append("    uint tpg_y = tpg3.y;")
                if 2 in used_axes:
                    lines.append("    uint tpg_z = tpg3.z;")

        # Static threadgroup memory declarations
        # Use compute type (float) for fp16/bf16 to avoid precision loss in reductions
        for tg_name, tg_dtype, tg_size in self.builder._threadgroup_arrays:
            msl_ty = _msl_compute_type(tg_dtype)
            lines.append(f"    threadgroup {msl_ty} {tg_name}[{tg_size}];")

        # Body
        for line in self.builder._body_lines:
            lines.append(line)

        lines.append("}")
        lines.append("")

        return "\n".join(lines)



# ---------------------------------------------------------------------------
# TTGIR integration (requires triton)
# ---------------------------------------------------------------------------

def _mept_path_log(tag, detail):
    """Diagnostic: record which lowering path produced a kernel.

    No-op unless ``TRITON_MSL_PATH_LOG`` names a file. Used by the
    fallback-integrity audit to measure how load-bearing the (silent-
    capable) legacy fallback is across the test suite.
    """
    import os
    path = os.environ.get("TRITON_MSL_PATH_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(f"{tag}\t{detail}\n")
    except Exception:
        pass


def emit_msl(mod, metadata, options):
    """Convert a TritonGPU IR module to MSL source code.

    This is the entry point called by MetalBackend.make_msl().

    Uses the MLIR walker + generic op-by-op lowerer. Falls back to
    the legacy text-based parser only if the new pipeline fails.

    Args:
        mod: The MLIR module after TTGIR passes.
        metadata: Compilation metadata dict.
        options: MetalOptions instance.

    Returns:
        MSL source code as a string.
    """
    from triton_msl.errors import MetalNonRecoverableError
    # Primary path: new walker + generic lowerer
    try:
        from triton_msl.codegen.mlir_walker import walk_ttgir
        from triton_msl.codegen.generic_lowerer import lower_ir_graph

        graph = walk_ttgir(mod, options)
        metadata["name"] = _sanitize_msl_name(graph.func_name)

        from triton_msl.codegen.generic_lowerer import GenericLowerer
        lowerer = GenericLowerer(graph, options)
        msl_src = lowerer.lower()

        # Use lowerer's effective block_size (may differ from graph for matmul templates)
        metadata["block_size"] = lowerer.effective_block_size

        # Integrity backstop: an UNKNOWN_<id> in the emitted source is an
        # UNRESOLVED SSA reference (e.g. _lookup of a value not in env). It is
        # never valid MSL — it would fail xcrun with a cryptic compile error.
        # Refuse loudly with an actionable message instead. The common cause is
        # a value defined OUTSIDE a runtime-bound loop referenced INSIDE it in
        # the multi-element-per-thread regime (BLOCK > threadgroup size, e.g.
        # the tl.arange / other= constant in a tl.sum-in-loop at BLOCK>=256) —
        # the register-array spine (roadmap Phase 2). (downstream tridec bug 2)
        if "UNKNOWN_" in msl_src and "UNSUPPORTED" not in msl_src:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"codegen left an unresolved value (UNKNOWN_<id>) in kernel "
                f"'{metadata.get('name', '?')}'. This usually means a value defined "
                f"outside a runtime-bound loop is used inside it when BLOCK "
                f"exceeds the threadgroup size (multi-element-per-thread). "
                f"Use BLOCK <= 128, or restructure the loop, until the "
                f"register-array spine lands.")

        # Verify no UNSUPPORTED markers in output
        if "UNSUPPORTED" not in msl_src:
            metadata["output_arg_indices"] = lowerer.get_output_arg_indices()
            # Flag whether the kernel uses multi-axis program_id (needs 2D/3D grid)
            used_axes = getattr(lowerer, '_used_pid_axes', {0})
            metadata["needs_2d_grid"] = max(used_axes) > 0 if used_axes else False
            # Two-kernel-split matmul descriptor (#159); None for other kernels.
            metadata["mm_two_kernel"] = getattr(lowerer, "_mm_two_kernel", None)
            # Fast-matmul runtime-dispatch descriptor (Phase 4); None for other kernels.
            metadata["fast_matmul"] = getattr(lowerer, "_fast_matmul", None)
            _mept_path_log("primary", metadata.get("name", "?"))
            return msl_src

        # Fall through to legacy parser if unsupported ops remain
        _mept_path_log("fallback-unsupported", metadata.get("name", "?"))
    except MetalNonRecoverableError:
        # Deliberate refusal: the lowerer recognized a kernel it cannot lower
        # correctly AND knows the legacy parser can't either. Re-raise instead
        # of falling back — returning silently-wrong numbers is worse than a
        # clear error. This is the integrity backstop (PR1).
        _mept_path_log("refused-nonrecoverable", metadata.get("name", "?"))
        raise
    except Exception as e:
        import warnings
        warnings.warn(
            f"emit_msl: generic lowerer failed: {e}. "
            "Falling back to legacy text-based parser.",
            stacklevel=2,
        )
        _mept_path_log("fallback-exception", str(e)[:80])

    return _legacy_fallback(str(mod), metadata, options,
                            "generic lowerer could not lower this kernel")


def _legacy_fallback(ir_text, metadata, options, reason):
    """Legacy text-based parser — OPT-IN only (Phase 0 T4).

    The heuristic parser can emit plausible-but-wrong kernels (it has produced
    verified silent-wrongs). By default an unlowerable kernel REFUSES; set
    TRITON_MSL_LEGACY=1 to accept the risk for debugging.
    """
    import os
    if os.environ.get("TRITON_MSL_LEGACY") != "1":
        from triton_msl.errors import MetalNonRecoverableError
        raise MetalNonRecoverableError(
            f"Refusing to emit possibly-wrong output: {reason}, and the legacy "
            "text parser is heuristic (has produced silent-wrongs). Set "
            "TRITON_MSL_LEGACY=1 to opt in for debugging.")
    from triton_msl.codegen.ttgir_parser import parse_ttgir
    kernel_name = _extract_kernel_name(ir_text)
    metadata["name"] = _sanitize_msl_name(kernel_name)
    kb = parse_ttgir(ir_text, options)
    metadata["block_size"] = kb.block_size
    return kb.build()


def _extract_kernel_name(ir_text):
    """Extract the kernel function name from MLIR text."""
    import re

    match = re.search(r"tt\.func\s+public\s+@(\w+)\s*\(", ir_text)
    if match:
        return match.group(1)
    match = re.search(r"func\.func\s+@(\w+)\s*\(", ir_text)
    if match:
        return match.group(1)
    return "triton_kernel"

# ---------------------------------------------------------------------------
# Re-export pre-baked kernel templates from _msl_templates so callers can
# continue to do `from triton_msl.codegen.msl_emitter import make_X_kernel`.
# The split keeps msl_emitter.py focused on the KernelBuilder API + emit_msl;
# the 65 make_* templates live in _msl_templates.py.
# ---------------------------------------------------------------------------

from triton_msl.codegen._msl_templates import *  # noqa: E402, F401, F403
