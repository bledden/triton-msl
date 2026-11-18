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


_MSL_RESERVED = frozenset(
    {
        "kernel",
        "vertex",
        "fragment",
        "device",
        "constant",
        "threadgroup",
        "thread",
        "texture",
        "sampler",
        "float",
        "half",
        "int",
        "uint",
        "bool",
        "char",
        "short",
        "void",
        "using",
        "namespace",
        "metal",
        "return",
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "break",
        "continue",
        "struct",
        "class",
        "enum",
        "true",
        "false",
    }
)


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
        self.unsupported_reasons = []  # Explicit lowering outcomes, never inferred from MSL text.

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
            self._emit(
                f"{compute_ty} {out_var} = {mask_var} ? static_cast<{compute_ty}>({ptr_var}[{offsets_var}]) : {zero};"
            )
        else:
            self._emit(f"{compute_ty} {out_var} = static_cast<{compute_ty}>({ptr_var}[{offsets_var}]);")
        return out_var

    def store(self, ptr_var, offsets_var, val_var, mask_var=None, dtype="fp32"):
        """Masked store to a buffer pointer + offset.

        For FP16/BF16 buffers, casts from float compute type back to storage type.
        """
        store_ty = triton_type_to_msl(dtype)
        compute_ty = _msl_compute_type(dtype)
        needs_cast = store_ty != compute_ty
        cast_val = f"static_cast<{store_ty}>({val_var})" if needs_cast else val_var
        if mask_var:
            self._emit(f"if ({mask_var}) {{ {ptr_var}[{offsets_var}] = {cast_val}; }}")
        else:
            self._emit(f"{ptr_var}[{offsets_var}] = {cast_val};")

    def binary_op(self, op, a_var, b_var, out_var):
        """Emit a binary operation: out = a op b."""
        op_map = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "mod": "%",
            "and": "&",
            "or": "|",
            "xor": "^",
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

    def unsupported(self, reason):
        """Record an unlowered operation while retaining its diagnostic comment."""
        self.unsupported_reasons.append(reason)
        self.comment(reason)

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

    @staticmethod
    def ordered_combine_expr(op, left, right):
        """Render one exact ordered cmp/select combine."""
        try:
            _, pred, true_side, nan_side = op.split("_", 3)
            cmp = {"ogt": ">", "oge": ">=", "olt": "<", "ole": "<="}[pred]
        except (ValueError, KeyError) as exc:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(f"unknown ordered reduction combine '{op}'") from exc
        if true_side not in ("a", "b"):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(f"unknown ordered reduction true side '{true_side}'")
        cond = f"({left} {cmp} {right})"
        if nan_side == "a":
            cond = f"({cond} || ({left} != {left}))"
        elif nan_side == "b":
            cond = f"({cond} || ({right} != {right}))"
        elif nan_side != "none":
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(f"unknown ordered reduction NaN side '{nan_side}'")
        tval, fval = (left, right) if true_side == "a" else (right, left)
        return f"({cond} ? {tval} : {fval})"

    def threadgroup_reduce(self, op, val_var, shared_var, out_var, reduce_ty="float"):
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
                f"to a multiple of 32 (the matmul/softmax templates already do this)."
            )
        self._needs_simd_qualifiers = True
        from triton_msl.codegen.msl_builtins import SIMD_REDUCTIONS

        _ordered = op.startswith("ordered_")

        if _ordered:
            if reduce_ty not in ("float", "half", "bfloat") or (
                self.block_size != 16 and (self.block_size < 32 or self.block_size % 32)
            ):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError("ordered cmp/select reduction requires a complete float SIMD-group tile")
            # All lanes execute the same five shuffle calls. Build adjacent
            # contiguous ranges ([0,1], [2,3], then [0,3], ...) rather than the
            # ordinary shuffle-down interleaving (0,16,8,24,...). The operator
            # is associative but can be non-commutative: reassociation is legal,
            # changing leaf order is not.
            lane_val = f"(({reduce_ty}){val_var})"
            self._var(f"ordered_{out_var}", lane_val, ty=reduce_ty)
            _last_active_lane = min(self.block_size, 32) - 1
            for _offset in (1, 2, 4, 8, 16):
                if _offset >= min(self.block_size, 32):
                    continue
                _right = f"ordered_right_{out_var}_{_offset}"
                # Clamp indices unused by this round so every lane executes a
                # defined, convergent shuffle before the leader-only update.
                self._var(
                    _right,
                    f"simd_shuffle(ordered_{out_var}, min((uint)tiisg + {_offset}u, {_last_active_lane}u))",
                    ty=reduce_ty,
                )
                self.begin_if(f"(((uint)tiisg & {(2 * _offset) - 1}u) == 0u)")
                self._emit(f"ordered_{out_var} = " + self.ordered_combine_expr(op, f"ordered_{out_var}", _right) + ";")
                self.end_block()
            n_simd_groups = (self.block_size + 31) // 32
            self.barrier("threadgroup")
            self.begin_if("tiisg == 0")
            self._emit(f"{shared_var}[sgitg] = ordered_{out_var};")
            self.end_block()
            self.barrier("threadgroup")
            # Do not let non-leader threads read slot zero while lid 0 may be
            # writing the final group fold.  Their temporary value is dead;
            # all threads consume the published slot only after the barrier.
            self._var(out_var, f"({reduce_ty})0", ty=reduce_ty)
            self.begin_if("lid == 0")
            self._emit(f"{out_var} = {shared_var}[0];")
            self._emit(f"for (uint _ord_group = 1u; _ord_group < {n_simd_groups}u; ++_ord_group) {{")
            self._indent += 1
            self._emit(f"{out_var} = " + self.ordered_combine_expr(op, out_var, f"{shared_var}[_ord_group]") + ";")
            self._indent -= 1
            self._emit("}")
            self._emit(f"{shared_var}[0] = {out_var};")
            self.end_block()
            self.barrier("threadgroup")
            self._emit(f"{out_var} = {shared_var}[0];")
            return out_var

        intrinsic = SIMD_REDUCTIONS[op]
        # Reduce IN the element type. Integer reductions were emitted in float
        # (ty="float"), silently losing precision above 2^24 (re-audit #9: i32
        # tl.sum/max/min). simd_sum/max/min are defined for integer types, so reduce
        # natively. float-family (fp16/bf16/fp32) stays "float" — that's deliberately
        # MORE precise than reducing in half/bfloat.
        _is_float = reduce_ty in ("float", "half", "bfloat")
        # NaN-PROPAGATING max/min (inductor maximum/minimum): the simd intrinsic is
        # NaN-quiet, so carry an any-NaN flag through both reduction levels below.
        _nan_prop = op in ("nanmax", "nanmin")
        if _is_float:
            identity = {
                # Compiler-added padding must be neutral for both zero signs.
                # +0 changes an all-negative-zero sum to +0; -0 preserves -0
                # and still yields +0 when any source term is positive zero.
                "sum": "-0.0f",
                "prod": "1.0f",
                "max": "-INFINITY",
                "min": "INFINITY",
                "nanmax": "-INFINITY",
                "nanmin": "INFINITY",
            }[op]
        else:
            identity = {
                "sum": f"({reduce_ty})0",
                "prod": f"({reduce_ty})1",
                "max": f"metal::numeric_limits<{reduce_ty}>::lowest()",
                "min": f"metal::numeric_limits<{reduce_ty}>::max()",
                # bitwise reductions (all()/any()/xor) — identities so a 1-D and/or/xor
                # reduce computes instead of hitting a KeyError -> generic-fallback refusal.
                "and": f"(~({reduce_ty})0)",
                "or": f"({reduce_ty})0",
                "xor": f"({reduce_ty})0",
            }[op]

        # Unique intermediate variable names
        simd_var = f"simd_{out_var}"
        read_var = f"shared_{out_var}"

        # Step 1: SIMD-level reduction. Cast val_var to reduce_ty BEFORE the simd op so the
        # comparison runs in the reduce type — for an UNSIGNED max/min (reduce_ty='uint',
        # val_var typed 'int' because Triton types uint32 as i32), simd_max(int) would compare
        # SIGNED and return the wrong element; (uint)val_var reinterprets to compare unsigned.
        # A no-op when val_var already matches reduce_ty (the normal int/float/long cases).
        # (Triton-lens re-audit 2026-06-25.)
        if _nan_prop:
            # simd_max/min is NaN-quiet (returns the non-NaN). For NaN-PROPAGATING max/min,
            # force this group's partial result to NaN if ANY lane is NaN (`v != v`).
            _v = f"(({reduce_ty}){val_var})"
            self._var(
                simd_var,
                f"(simd_max(({_v} != {_v}) ? 1.0f : 0.0f) > 0.0f) ? ({reduce_ty})NAN : {intrinsic}({_v})",
                ty=reduce_ty,
            )
        else:
            self._var(simd_var, f"{intrinsic}(({reduce_ty}){val_var})", ty=reduce_ty)

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
        self._var(read_var, f"(tiisg < {n_simd_groups}u) ? {shared_var}[tiisg] : {identity}", ty=reduce_ty)
        if _nan_prop:
            # Final cross-group level: same NaN-quiet caveat — a per-group partial that is
            # NaN (set above) must make the whole-tile result NaN.
            self._var(
                out_var,
                f"(simd_max(({read_var} != {read_var}) ? 1.0f : 0.0f) > 0.0f) "
                f"? ({reduce_ty})NAN : {intrinsic}({read_var})",
                ty=reduce_ty,
            )
        else:
            self._var(out_var, f"{intrinsic}({read_var})", ty=reduce_ty)
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
        self.builder._argument_abi = None
        # Return prebuilt MSL if available (e.g., matmul kernels).
        # Substitute the kernel function name to match the TTGIR function name.
        if self.builder._prebuilt_msl is not None:
            msl = self.builder._prebuilt_msl
            import re as _re

            msl = _re.sub(
                r"kernel\s+void\s+\w+\s*\(",
                f"kernel void {self.builder.name}(",
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

        # Metal binds kernel arguments to buffer slots 0..30 (31 total). Every pointer
        # and every runtime scalar takes one, so a kernel passing ~20 strides overflows
        # well before it looks large; the Metal frontend then emits a wall of cryptic
        # "'buffer' attribute parameter is out of bounds" (one line per overflowing arg)
        # naming nothing. Refuse with a clear, actionable message instead of miscompiling
        # into that wall (GitHub issue #4.7).
        # Metal binds kernel arguments to buffer slots 0..30 (31 total). When a kernel
        # exceeds that, pack every runtime SCALAR into a single ``uint`` argument buffer
        # (each pointer still needs its own slot) and unpack them as locals at the body top
        # (GitHub issue #4.7). If even the pointers don't fit, or a scalar isn't 32-bit,
        # that's a true hardware wall -> refuse.
        _MAX_METAL_BUFFERS = 31
        _args = self.builder.args
        _pack_scalars = None  # list[Arg] packed into one buffer, or None
        if len(_args) > _MAX_METAL_BUFFERS:
            from triton_msl.errors import MetalNonRecoverableError

            _ptr_args = [a for a in _args if getattr(a, "is_ptr", False)]
            _scalar_args = [a for a in _args if not getattr(a, "is_ptr", False)]
            _kname = getattr(self.builder, "name", "kernel")
            if len(_ptr_args) + 1 > _MAX_METAL_BUFFERS:
                raise MetalNonRecoverableError(
                    f"kernel '{_kname}' needs {len(_ptr_args)} pointer buffers, but Metal "
                    f"binds at most {_MAX_METAL_BUFFERS} (slots 0..{_MAX_METAL_BUFFERS - 1}) "
                    "and pointers cannot be packed into an argument buffer. Reduce the "
                    "pointer count.",
                    op_name="kernel",
                )
            for _a in _scalar_args:
                if triton_type_to_msl(_a.dtype) not in ("int", "uint", "float", "bool"):
                    raise MetalNonRecoverableError(
                        f"kernel '{_kname}' exceeds Metal's {_MAX_METAL_BUFFERS} buffer slots "
                        f"and scalar '{_a.name}' is not 32-bit ({triton_type_to_msl(_a.dtype)}); "
                        "argument-buffer packing supports int/uint/float/bool scalars only. "
                        "Make it `tl.constexpr` or reduce the argument count.",
                        op_name="kernel",
                    )
            _pack_scalars = _scalar_args

        # Kernel signature
        params = []
        if _pack_scalars is not None:
            _pi = 0
            for arg in _args:
                if getattr(arg, "is_ptr", False):
                    params.append(f"    {arg.msl_param(_pi)}")
                    _pi += 1
            params.append(f"    constant uint* _packed_scalars [[buffer({_pi})]]")
        else:
            for i, arg in enumerate(self.builder.args):
                params.append(f"    {arg.msl_param(i)}")

        # Thread position qualifiers — Metal requires all position attrs same type
        used_axes = getattr(self.builder, "_used_pid_axes", {0})
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

        # Unpack packed scalars into locals so the body references them by name unchanged
        # (issue #4.7). Each 32-bit slot is bitcast to the scalar's type; the launcher packs
        # the values into ``_packed_scalars`` in this same (scalar-arg) order.
        if _pack_scalars is not None:
            for _j, _sa in enumerate(_pack_scalars):
                _mt = triton_type_to_msl(_sa.dtype)
                if _mt == "uint":
                    lines.append(f"    uint {_sa.name} = _packed_scalars[{_j}];")
                elif _mt == "bool":
                    lines.append(f"    bool {_sa.name} = _packed_scalars[{_j}] != 0u;")
                else:
                    lines.append(f"    {_mt} {_sa.name} = as_type<{_mt}>(_packed_scalars[{_j}]);")

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

        source = "\n".join(lines)
        if _pack_scalars is not None:
            from triton_msl.backend._argument_abi import producer_record

            self.builder._argument_abi = producer_record(_args, "generic-msl-v1", source)
        return source


# ---------------------------------------------------------------------------
# TTGIR integration (requires triton)
# ---------------------------------------------------------------------------


def _mept_path_log(tag, detail):
    """Diagnostic: record which lowering path produced a kernel.

    No-op unless ``TRITON_MSL_PATH_LOG`` names a file. Used by the
    fallback-integrity audit to measure how load-bearing the (silent-
    capable) legacy fallback is across the test suite.
    """
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
    from triton_msl.errors import MetalNonRecoverableError, MetalResourceError

    # Native operation identity, including callees: no legacy fallback is allowed
    # to erase an assertion if walking, planning, or generic emission fails.
    retained_asserts = []
    mod.walk(lambda op: retained_asserts.append(op) if op.get_name() == "tt.assert" else None)

    # Primary path: new walker + generic lowerer
    try:
        from triton_msl.codegen.mlir_walker import walk_ttgir
        from triton_msl.codegen.generic_lowerer import lower_ir_graph

        graph = walk_ttgir(mod, options)
        if retained_asserts:
            from triton_msl.codegen._constant_assertions import discharge_scalar_true

            graph = discharge_scalar_true(graph, mod)
        metadata["name"] = _sanitize_msl_name(graph.func_name)

        from triton_msl.codegen.generic_lowerer import GenericLowerer

        from triton_msl.codegen._source_program.entry import emit as emit_source_program

        fragment_request = getattr(options, "experimental_fa_fragment_lowering", "off")
        if fragment_request != "off":
            metadata["fa_fragment_lowering"] = {
                "requested": fragment_request,
                "selected": "off",
                "reason": "not-dense-simd",
            }
        source_program = emit_source_program(mod, graph, options, metadata["name"], GenericLowerer)
        if source_program is not None:
            msl_src, source_metadata = source_program
            metadata.update(source_metadata)
            _mept_path_log("source-program-pending", metadata["name"])
            return msl_src

        lowerer = GenericLowerer(graph, options)
        msl_src = lowerer.lower()
        if fragment_request != "off" and hasattr(lowerer, "_fa_fragment_lowering"):
            metadata["fa_fragment_lowering"] = dict(lowerer._fa_fragment_lowering)

        # Use lowerer's effective block_size (may differ from graph for matmul templates)
        metadata["block_size"] = lowerer.effective_block_size
        if type(lowerer._flash_attention) is dict and lowerer._flash_attention.get("kind") == "cooperative-row-v1":
            # Preserve the public occupancy denominator convention. Actual
            # per-entry static bytes (scalar0/sibling16384) live in the descriptor.
            metadata["shared"] = max(1, lowerer._replay_shared_bytes)

        # Unresolved SSA references raise at lookup. Unsupported operations are
        # recorded by their lowering sites, including noinline callees. Searching
        # the emitted text confuses legal kernel names/comments with diagnostics.
        # Template paths may return without constructing a KernelBuilder.
        if lowerer.kb is None or not lowerer.kb.unsupported_reasons:
            metadata["output_arg_indices"] = lowerer.get_output_arg_indices()
            # Flag whether the kernel uses multi-axis program_id (needs 2D/3D grid)
            used_axes = getattr(lowerer, "_used_pid_axes", {0})
            metadata["needs_2d_grid"] = max(used_axes) > 0 if used_axes else False
            # Two-kernel-split matmul descriptor (#159); None for other kernels.
            metadata["mm_two_kernel"] = getattr(lowerer, "_mm_two_kernel", None)
            # Fast-matmul runtime-dispatch descriptor (Phase 4); None for other kernels.
            metadata["fast_matmul"] = getattr(lowerer, "_fast_matmul", None)
            # Quantized-matmul runtime-dispatch descriptor; None for other kernels.
            metadata["quant_matmul"] = getattr(lowerer, "_quant_matmul", None)
            # FlashAttention zero-copy-dispatch descriptor; None for other kernels.
            metadata["flash_attention"] = getattr(lowerer, "_flash_attention", None)
            # Outputs of an attention-shaped kernel, for the launcher's overlap refusal.
            # Outside the packed descriptor: it grants no route, it only refuses.
            metadata["attention_outputs"] = getattr(lowerer, "_attention_outputs", None)
            # Where the kernel loads and stores, for the launcher's lane-order refusal.
            # It grants no route either. A plan that cannot be built is recorded as one
            # that proves nothing, never left out.
            from triton_msl.codegen._access_plan import build_access_plan

            try:
                _plan = build_access_plan(
                    lowerer.graph,
                    msl_src,
                    lowerer._memory_emitted,
                    None if lowerer.kb is None else lowerer.kb._body_lines,
                    single=lowerer._single_lane_stores,
                    rows=lowerer._row_lane_stores,
                    threads=(
                        None
                        if lowerer.kb is None
                        else [int(lowerer.kb.block_size), bool(getattr(lowerer, "_needs_wrapping", False))]
                    ),
                )
            except Exception as _error:
                _plan = {"version": 0, "why": "not built: " + type(_error).__name__}
            # A replacement shader is written from a recognised whole kernel, not operation
            # by operation, and its own launch checks decide about shared memory.
            _plan["replaced"] = lowerer.kb is None and [
                name
                for name in ("mm_two_kernel", "fast_matmul", "quant_matmul", "flash_attention", "attention_outputs")
                if metadata[name]
            ]
            metadata["access_plan"] = _plan
            # Batched-dot host-roundtrip address-bounds descriptor; None for
            # other kernels. Runtime strides must stay inside each view mirror.
            # Mutually exclusive address-plan union, sealed by the packed
            # launch contract. Rank-3 batched dot and the K-loop templates do
            # not share a lowering route.
            _batched_bounds = getattr(lowerer, "_batched_dot_bounds", None)
            _tail_bounds = getattr(lowerer, "_tail_access_bounds", None)
            if _batched_bounds is not None and _tail_bounds is not None:
                raise MetalNonRecoverableError(
                    "lowering produced conflicting launch address contracts", op_name="tt.dot"
                )
            metadata["batched_dot_bounds"] = _batched_bounds or _tail_bounds
            plan = lowerer._assert_plan
            metadata["device_assert"] = (
                {"schema": 1, "messages": plan["messages"], "buffer_index": len(graph.args)}
                if plan is not None
                else None
            )
            _mept_path_log("primary", metadata.get("name", "?"))
            metadata["argument_abi"] = lowerer._argument_abi
            return msl_src

        if retained_asserts:
            raise MetalNonRecoverableError(
                "retained assertion kernel has unsupported generic operations", op_name="tt.assert"
            )
        # Fall through to legacy parser if unsupported ops remain
        _mept_path_log("fallback-unsupported", metadata.get("name", "?"))
    except MetalResourceError:
        # A concrete capacity failure is terminal for this configuration, even
        # with legacy opt-in. Keep its type for the compiler/autotuner boundary.
        _mept_path_log("refused-resource", metadata.get("name", "?"))
        raise
    except MetalNonRecoverableError:
        # Deliberate refusal: the lowerer recognized a kernel it cannot lower
        # correctly AND knows the legacy parser can't either. Re-raise instead
        # of falling back — returning silently-wrong numbers is worse than a
        # clear error. This is the integrity backstop (PR1).
        _mept_path_log("refused-nonrecoverable", metadata.get("name", "?"))
        raise
    except Exception as e:
        if retained_asserts:
            raise MetalNonRecoverableError(
                "retained assertion lowering failed; no legacy fallback is safe: " + str(e), op_name="tt.assert"
            ) from e
        import warnings

        warnings.warn(
            f"emit_msl: generic lowerer failed: {e}. Falling back to legacy text-based parser.",
            stacklevel=2,
        )
        _mept_path_log("fallback-exception", str(e)[:80])

    return _legacy_fallback(str(mod), metadata, options, "generic lowerer could not lower this kernel")


def _legacy_fallback(ir_text, metadata, options, reason):
    """Legacy text-based parser — OPT-IN only (Phase 0 T4).

    The heuristic parser can emit plausible-but-wrong kernels (it has produced
    verified silent-wrongs). By default an unlowerable kernel REFUSES; set
    TRITON_MSL_LEGACY=1 to accept the risk for debugging.
    """
    if os.environ.get("TRITON_MSL_LEGACY") != "1":
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(
            f"Refusing to emit possibly-wrong output: {reason}, and the legacy "
            "text parser is heuristic (has produced silent-wrongs). Set "
            "TRITON_MSL_LEGACY=1 to opt in for debugging."
        )
    from triton_msl.codegen.ttgir_parser import parse_ttgir

    kernel_name = _extract_kernel_name(ir_text)
    metadata["name"] = _sanitize_msl_name(kernel_name)
    kb = parse_ttgir(ir_text, options)
    metadata["block_size"] = kb.block_size
    # The opt-in heuristic parser grants no new host ABI certificate.
    metadata["argument_abi"] = None
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
