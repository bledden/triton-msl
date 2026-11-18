"""Generic op-by-op lowering from IRGraph to MSL via KernelBuilder.

Processes each TTGIR operation independently, mapping it to MSL code.
This replaces the 30+ pattern matchers in ttgir_parser.py with a single
pass that lowers each op based on its type.

Metal-specific considerations:
- No tensor abstraction: each thread processes one element
- tt.splat is a no-op (scalar→per-thread is free in SIMT)
- tt.make_range → thread_position_in_threadgroup (lid)
- tt.reduce → SIMD intrinsics + threadgroup shared memory
- All FP16/BF16 computation done in float, cast at load/store
"""

import math
import os
import re
from typing import Any, Dict, List, Optional

from triton_msl.codegen.mlir_walker import IRGraph, SSAValue, FuncArg, CalledFunc, _extract_shape
from triton_msl.codegen.msl_emitter import KernelBuilder, _msl_compute_type, _sanitize_msl_name
from triton_msl.codegen.msl_types import triton_type_to_msl
from triton_msl.errors import (
    MetalCodegenError,
    MetalNotImplementedError,
    MetalNonRecoverableError,
)


# Free helpers extracted to keep this file navigable; see
# _lowerer_helpers.py and _device_func_lowerer.py for the split-out code.
from triton_msl.codegen._lowerer_helpers import (
    CMPI_PREDICATES,
    CMPF_PREDICATES,
    CMPI_NAMED,
    CMPF_NAMED,
    _UINT_TYPE_MAP,
    _mlir_to_triton_dtype,
    _msl_int_type,
    _shape_numel,
    _alias_shared_memory,
)
from triton_msl.codegen._device_func_lowerer import _DeviceFuncLowerer
from triton_msl.codegen._lowerer_templates import _TemplateMixin


def _op_is_exp(op_name: str) -> bool:
    """True iff ``op_name`` is the exponential op (``math.exp`` / ``math.exp2``
    / ``tt.exp``), used by the FlashAttention softmax gate.

    Matches the op suffix precisely rather than a substring: a bare
    ``"exp" in op`` also matches ``tt.expand_dims``, which would wrongly pull a
    non-FA kernel (>=2 dots + max + expand_dims) into FlashAttention handling.
    """
    if not op_name:
        return False
    return op_name in ("math.exp", "math.exp2", "tt.exp")


def _v_head_dim_of(info):
    """A missing optional value width denotes symmetric attention (PR #7)."""
    value = info.get("v_head_dim")
    return info["head_dim"] if value is None else value


def _simd_fa_eligible(info):
    """True if the detected FA can use the simdgroup template: head_dim in {64, 128},
    block 32x32, fp32/fp16, AND contiguous innermost (head-dim) stride for all of
    Q/K/V/Out. info["strides"][role] is [z,h,m,k]; index 3 (k) == "c1" means the
    innermost stride folded to 1 (contiguous). Non-contiguous -> scalar template
    (handles general strides) -> never silent-wrong.

    Registers scale with the OUTPUT width (v_head_dim, via TPG = v_head_dim//64), NOT the
    QK contraction width. So v_head_dim must be a supported multiple of 64 ({64, 128}), but
    head_dim (the QK contraction) only adds loop iterations and CAN exceed 128 -- the
    ASYMMETRIC MLA / DeepSeek case (qk=192, v=128). Symmetric attention has
    v_head_dim == head_dim so hd64/hd128 behave exactly as before. The Q staging is
    BM*head_dim*elem bytes of threadgroup memory, so fp32 head_dim>128 overflows the 32KB
    budget -> head_dim>128 is fp16/bf16-only (192 validated, cap 256)."""
    qk = info.get("head_dim")
    if qk is None:
        return False
    vd = _v_head_dim_of(info)
    out_dtype = info.get("out_dtype")
    if vd not in (64, 128):
        return False
    if out_dtype not in ("f32", "f16"):
        return False
    # QK contraction width: the Q staging is BM*qk*elem bytes of threadgroup memory. The
    # 32KB budget caps qk at 192 for fp16 (qk=256 overflows by 384B) and 128 for fp32
    # (2x the bytes). MLA's qk=192 fits fp16; symmetric hd128 fits both.
    qk_cap = 192 if out_dtype == "f16" else 128
    if not (isinstance(qk, int) and qk % 8 == 0 and 64 <= qk <= qk_cap):
        return False
    if not (info.get("block_m") == 32 and info.get("block_n") == 32):
        return False
    st = info.get("strides", {})
    if not all(st.get(r, [None] * 4)[3] == "c1" for r in ("q", "k", "v", "o")):
        return False
    # MLA: the rope Q/K sources must also be contiguous-innermost (the cat dispatch
    # concatenates them; requiring c1 keeps the concat + kernel a clean fast path).
    if info.get("is_mla"):
        mst = info.get("mla_strides") or {}
        if not all((mst.get(r) or [None] * 4)[3] == "c1" for r in ("q_rope", "k_rope")):
            return False
    return True


def _fa_half_accumulate(out_dtype) -> bool:
    """OPT-IN (default OFF): TRITON_MSL_FA_HALF_ACCUM=1 makes the simd FA use fp16
    (half8x8) MMA accumulators instead of fp32. ~4% faster at ~1% max-abs error — a
    latency/accuracy trade for fp16 inference, mirroring the int8/int4 opt-ins. Applies
    ONLY to fp16 output (there is no half input to accumulate for fp32), so it is a
    no-op — the shipped float-accumulate kernel — for fp32 regardless of the env var.
    Must match _msl_cache_key's TRITON_MSL_FA_HALF_ACCUM handling (both key the flag).
    """
    if os.environ.get("TRITON_MSL_FA_HALF_ACCUM", "0") not in ("1", "true", "True"):
        return False
    return out_dtype in ("f16", "fp16")


from triton_msl.codegen._lowerer_detection import _DetectionMixin
from triton_msl.codegen._lowerer_batched_dot import _BatchedDotMixin
from triton_msl.codegen._lowerer_dot_kchunk import _DotKChunkMixin
from triton_msl.codegen._lowerer_emission import _EmissionMixin
from triton_msl.codegen._lowerer_reduce import _ReduceScanMixin
from triton_msl.codegen._lowerer_control import _ControlFlowMixin


# ---------------------------------------------------------------------------
# Multi-element-per-thread (MEPT) op whitelist
# ---------------------------------------------------------------------------
#
# MEPT single-pass activation is *default-deny*: it fires only for kernels
# composed entirely of ops whose array-form lowering is verified wired (see
# Phase 4b/4c). Any op outside this set forces the scalar wrap-loop path,
# which is always correct. A too-narrow set merely forgoes the MEPT speedup
# (still correct); a too-wide set would risk an op consuming a register
# array it can't handle (wrong results). Err narrow.
#
# Deliberately EXCLUDED (not array-wired):
# arith.bitcast, math.erf/log1p/expm1, every shape op (tt.trans/join/cat/
# split/gather/broadcast/expand_dims/reshape), atomics, reduce/scan/
# histogram, tt.dot, control flow (scf.*), ttg.*. FP8 kernels are excluded
# separately (fp8 truncf emits a scalar conversion chain).
_MEPT_SAFE_OPS = frozenset(
    {
        # producers / memory (wired array path)
        "tt.make_range",
        "tt.addptr",
        "tt.load",
        "tt.store",
        # scalar / broadcast operands (never produce a register array)
        "tt.get_program_id",
        "tt.get_num_programs",
        "arith.constant",
        "tt.splat",
        "tt.return",
        # elementwise binary -> _emit_binary / _emit_builtin_binary /
        # _emit_nan_propagating_minmax (all route through _mept_binary_dispatch)
        "arith.addf",
        "arith.addi",
        "arith.subf",
        "arith.subi",
        "arith.mulf",
        "arith.muli",
        "arith.divf",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.remf",
        "arith.maxf",
        "arith.maxsi",
        "arith.maxui",
        "arith.minf",
        "arith.minsi",
        "arith.minui",
        "arith.maxnumf",
        "arith.minnumf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrsi",
        "arith.shrui",
        # comparison (cmpi and cmpf are array-wired)
        "arith.cmpi",
        "arith.cmpf",
        # ternary select (tl.where) -> _lower_select / _mept_select_dispatch
        "arith.select",
        # unary
        "arith.negf",
        # casts (passthrough or _emit_cast / _emit_int_cast / _emit_uitofp)
        "arith.extf",
        "arith.truncf",
        "arith.sitofp",
        "arith.uitofp",
        "arith.fptosi",
        "arith.fptoui",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.index_cast",
        "arith.index_castui",
        # math (array-wired in _lower_math)
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.sqrt",
        "math.rsqrt",
        "math.abs",
        "math.absf",
        "math.absi",
        "math.sin",
        "math.cos",
        "math.tanh",
        "math.floor",
        "math.ceil",
        "math.round",
        "math.fma",
        "math.powf",
        "math.copysign",
        "math.atan2",
        "math.roundeven",
        "math.trunc",
    }
)


# Compute ops that, when present at TOP LEVEL (not inside the K-loop scf.for
# region, where `acc += dot` lives), indicate a trailing matmul epilogue
# (bias / activation / fma) on the matmul RESULT. The inline-dot and prebuilt
# matmul templates store the raw accumulator and would SILENTLY DROP these, so a
# matmul kernel containing one is routed to the generic op-by-op lowerer instead
# (re-audit #6). Pure casts/broadcasts are intentionally EXCLUDED (the templates
# cast on store), as are comparisons/selects (matmuls don't carry those).
_MATMUL_EPILOGUE_COMPUTE_OPS = frozenset(
    {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "math.fma",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.sqrt",
        "math.rsqrt",
        "math.sin",
        "math.cos",
        "math.erf",
        "math.tanh",
        "math.floor",
        "math.ceil",
        "math.absf",
        "math.powf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.maxnumf",
        "arith.minnumf",
        "tt.clampf",
        # tl.where(cond, acc, x) — e.g. relu-via-where — is a TOP-LEVEL arith.select on
        # the matmul result. (A masked LOAD's select lives inside the scf.for region, not
        # at top level, so this does NOT over-refuse bounds-masked matmuls.) re-audit #8:
        # a K-loop matmul + relu-via-where silently dropped the mask, storing the raw dot.
        "arith.select",
    }
)

_MATMUL_EPILOGUE_REFUSE_MSG = (
    "matmul with a trailing compute epilogue (bias/activation/fma/clamp) that the "
    "fused-epilogue template can't emit (e.g. the dot is inside an scf.for K-loop): the "
    "inline/prebuilt matmul templates store the raw accumulator and would SILENTLY DROP "
    "the epilogue, and the generic fallback mis-tiles it. Refusing rather than "
    "mis-compute. Split the epilogue into a separate kernel after the matmul."
)


# ---------------------------------------------------------------------------
# Generic Lowerer
# ---------------------------------------------------------------------------


class GenericLowerer(
    _ControlFlowMixin,
    _ReduceScanMixin,
    _EmissionMixin,
    _DetectionMixin,
    _BatchedDotMixin,
    _DotKChunkMixin,
    _TemplateMixin,
):
    """Lower an IRGraph to MSL source code via KernelBuilder."""

    def __init__(self, graph: IRGraph, options=None):
        self.graph = graph
        # Fast-matmul runtime-dispatch descriptor (Phase 4). Set by
        # _lower_dot_simple_template when an eligible matmul is detected;
        # read by emit_msl into metadata. None for every other kernel.
        self._fast_matmul = None
        # FlashAttention zero-copy-dispatch descriptor. Set in the simdgroup-FA
        # branch of _lower_flash_attention_template; read by emit_msl into
        # metadata. Routes the 2-D-grid FA kernel through compile_shader (its
        # 2-D grid blocks the generic 1-D fast-path). None for every other kernel.
        self._flash_attention = None
        # Argument positions an attention-shaped kernel stores through. The launcher
        # refuses such a launch when one of them overlaps another tensor argument.
        self._attention_outputs = None
        self._argument_abi = None
        self._tail_access_bounds = None
        self.options = options
        self.env = {}  # ssa_id -> MSL variable name
        self.env_types = {}  # ssa_id -> triton dtype string
        self.env_is_mask = {}  # ssa_id -> True if this is a bool mask
        self.env_is_ptr = {}  # ssa_id -> (base_ptr_name, offsets_var)
        self.env_shapes = {}  # ssa_id -> shape tuple, e.g., (32, 64)
        # Phase 4 foundation: track how many tensor elements per thread each
        # SSA value carries. The current lowerer emits 1 scalar per thread for
        # every tensor, so the default is 1; ``_track_n_elems`` reads the
        # TTGIR layout attached to each op\'s ``type_str`` and stores the
        # computed value here. Op handlers can opt into the multi-element
        # path later; today this is purely diagnostic and informs the
        # ``_lower_convert_layout`` guard about when a real shuffle is
        # required. See ``docs/superpowers/plans/2026-05-21-multi-element-
        # per-thread.md`` for the staged refactor plan.
        self.env_n_elems = {}  # ssa_id -> int (elements per thread, ≥ 1)
        # Phase 4c: alongside env_n_elems, store the resolved
        # ``LinearLayout`` when one is available. Consulted by MEPT
        # producers (e.g. ``_lower_make_range``) to emit the correct
        # per-register position math via ``msl_position_expr``. When
        # absent (the synthetic / contiguous case), producers fall back
        # to the simple ``lid*N + i`` formula.
        self.env_layout = {}  # ssa_id -> LinearLayout
        # Some per-thread scalar values come from a broadcast-redundant layout:
        # thread `lid` does not hold the element at flat index `lid`, but at a
        # different index (e.g., after a 3D reduce that broadcasts the reduced
        # axis's K copies). `self._bcast_layout[ssa_id]` records the MSL
        # expression that, given `lid`, yields the flat index into the logical
        # result tensor that thread `lid` actually holds. Consumers (e.g., a
        # subsequent 2D reduce or a tt.store) use this to re-stage correctly.
        self._bcast_layout = {}
        # Track ssa_ids whose value is the SAME on every thread (splat-like).
        # Includes: arith.constant tensors, tt.splat, and elementwise ops whose
        # operands are all splat.  Used to decide whether combining with a
        # bcast-laid-out value preserves the layout (splat operand doesn't
        # absorb the broadcast-redundancy) or collapses it to canonical (a
        # per-thread distinct operand does absorb).
        self._is_splat = set()
        self.kb = None
        self._var_counter = 0

        # Track stores for output detection
        self._store_ptr_ids = set()

        # Shared memory counter for reductions
        self._shared_counter = 0

        # Whether kernel uses tt.get_num_programs (needs grid size parameter)
        self._needs_num_programs = False

        # 2D kernel info (populated by _prescan_2d_info)
        self._is_2d = False
        self._effective_2d_shape = None  # e.g., (32, 64)
        self._make_range_dim = {}  # ssa_id -> dimension index (0=row, 1=col)

        # Track which program_id axes are used (for kernel signature)
        self._used_pid_axes = set()  # {0, 1, 2}

        # SSA ids to skip (handled as part of a fused pattern)
        self._skip_ids = set()

        # ── multi-element-per-thread (MEPT) register-array model ───────────
        # Charter (read before changing this): MEPT lets a thread hold N tensor
        # elements as a register array instead of one scalar. As of M5
        # (2026-06-13) it is the DEFAULT codegen path; ``TRITON_MSL_MEPT=0``
        # is an escape hatch to the legacy scalar/wrap-loop path.
        # Why default-on is CORRECTNESS, not perf (it benchmarks perf-neutral):
        #   - It is the general register-array path that carries per-element
        #     state across data-dependent control flow. Patterns that REFUSE on
        #     the scalar path compute correctly here: reduction-in-loop at
        #     BLOCK>=256 (tridec Bug 2), loop-carried register arrays, and the
        #     >1024 threadgroup ceiling for 1-D kernels.
        #   - Zero regression: the full upstream test_core suite is 5335/0 with
        #     MEPT on (default) as well as off (MEPT=0); the project suite is
        #     622/0 (default) / 620/0 (MEPT=0). Verified 2026-06-13.
        # Every consumer is gated on ``mept_enabled``; the producer (make_range)
        # is the single activation root. ``TRITON_MSL_MEPT=0`` restores the
        # exact pre-flip scalar route (the parity gate proves byte-equivalence
        # on scalar kernels). Specs: docs/superpowers/specs/2026-06-11-mept-
        # register-array-spine-design.md, 2026-06-13-mept-m5-flip-design.md.
        self.mept_enabled = os.environ.get("TRITON_MSL_MEPT", "1") != "0"
        # Depth of nested control-flow bodies (scf.for/if/while) currently
        # being lowered. An in-loop reduce (depth > 0) that is not register-
        # array-covered cannot use the top-level multipass wrap and would
        # silently under-cover block_size > num_threads — see _lower_reduce.
        self._control_flow_depth = 0
        self._current_loop_body_ops = None
        self.env_array = {}  # ssa_id -> (var_name: str, n_elems: int, ty: str)
        # Phase 4c: parallel to env_is_ptr but for the case where the
        # tt.addptr offset is an env_array. Maps ssa_id -> (base_ptr,
        # offset_array_var, n) so tt.load / tt.store can emit per-position
        # memory accesses.
        self.env_ptr_array = {}

    def _next_var(self, prefix="r") -> str:
        name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        return name

    def _pointer_offset_expr(self, value_id: int) -> str:
        """Preserve signed integer offsets at the MSL pointer boundary.

        Index arithmetic may contain ``uint lid`` even when the MLIR result is
        signless i32. C++ usual conversions otherwise turn ``start + lid``
        into uint, so start=-2 indexes 2**32-2. Cast the completed expression
        back to its received signed width before pointer addition/indexing.
        Explicit unsigned values keep their unsigned contract.
        """
        expr = self._lookup(value_id)
        dtype = self.env_types.get(value_id)
        if dtype in ("i1", "i8", "i16", "i32"):
            return f"int({expr})"
        if dtype == "i64":
            return f"long({expr})"
        return expr

    def _var_array(self, prefix: str, exprs, ty: str) -> str:
        """Emit ``ty name[N];`` followed by ``name[i] = exprs[i];``.

        Phase 4b primitive for multi-element-per-thread emission. Returns
        the variable name; callers record the entry in ``env_array`` so
        downstream ops know to read array elements with ``name[i]`` rather
        than the plain scalar form. ``len(exprs)`` must be ≥ 1; a length
        of 1 is allowed and equivalent to a plain ``_var`` decl, but the
        env entry signals to consumers that array-loop emission is in use.

        This helper assumes the caller has already verified that the MEPT
        feature flag is on and that the target tensor genuinely carries
        multiple elements per thread (consult ``env_n_elems`` for the
        source SSA value before deciding to emit an array).
        """
        if not exprs:
            raise ValueError("_var_array requires at least one expression")
        n = len(exprs)
        name = self._next_var(prefix)
        # Single-line declaration + per-element initializer keeps the
        # emitted MSL compact and lets the Metal compiler unroll trivially
        # at small N. For larger N, switch to a loop later (Phase 4b
        # follow-up; profile first).
        self.kb.raw_line(f"    {ty} {name}[{n}];")
        for i, expr in enumerate(exprs):
            self.kb.raw_line(f"    {name}[{i}] = {expr};")
        return name

    def _materialize(self, regval, body, base="t"):
        """Emit the cheapest correct form for a value and return its RegVal.

        ``body(e)`` -> MSL expression string for element index e. MEPT spine
        (milestone 1). scalar (n_elems==1): ``ty name = body(0);`` (identical
        to a plain _var) -- the scalar-collapse that keeps the common path
        byte-identical to today. array: ``ty name[n]; name[e] = body(e);`` via
        _var_array. wraploop: a single expression emitted inside the caller's
        _loop_e loop (scalar-shaped, indexed by _loop_e in body(0))."""
        from triton_msl.codegen.regval import RegVal

        if regval.form == "array" and regval.n_elems > 1:
            arr = self._var_array(base, [body(e) for e in range(regval.n_elems)], regval.ty)
            return RegVal(name=arr, n_elems=regval.n_elems, ty=regval.ty, form="array")
        name = self._next_var(base)
        self.kb.raw_line(f"    {regval.ty} {name} = {body(0)};")
        return RegVal(name=name, n_elems=1, ty=regval.ty, form=regval.form)

    # -- Shape tracking helpers --------------------------------------------------

    def _native_result_ids_for_op(self, ssa: SSAValue) -> tuple[int, ...]:
        """Return only native result ids owned by ``ssa``.

        Void operations have neither ``result_ids`` nor a ResultMeta entry and
        therefore contribute no value. Single-result operations from walkers
        that omit the convenience list are keyed by ``ssa.id``.
        """
        result_ids = tuple(ssa.result_ids or ())
        if result_ids:
            return result_ids
        # A non-empty printed result type is used only as an arity witness so
        # a missing metadata record cannot make a real value disappear from
        # the prescan. Its shape/type contents are never consumed here.
        return (ssa.id,) if ssa.id in self.graph.result_meta or ssa.type_str else ()

    def _native_shape(self, ssa_id: int, *, op_name: str) -> tuple:
        """Return an ownership-checked, fully static native shape.

        Kernel extent and coverage decisions cannot use mutable replay caches
        or printed type summaries: either native metadata proves the logical
        shape or lowering refuses before choosing a dispatch width.
        """
        from triton_msl.errors import MetalNonRecoverableError

        facts = self._native_value_facts(ssa_id, op_name=op_name)
        shape = facts.shape
        if facts.raw.strip().startswith("tensor<") != facts.is_tensor:
            raise MetalNonRecoverableError(
                f"{op_name} native metadata has contradictory tensor-kind facts for SSA value {ssa_id}",
                op_name=op_name,
            )
        if facts.is_tensor:
            if shape is None or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise MetalNonRecoverableError(
                    f"{op_name} native metadata does not prove a positive static tensor shape for SSA value {ssa_id}",
                    op_name=op_name,
                )
            return tuple(shape)
        if shape != ():
            raise MetalNonRecoverableError(
                f"{op_name} native scalar metadata carries a contradictory shape for SSA value {ssa_id}",
                op_name=op_name,
            )
        return ()

    def _native_shapes_for_op(self, ssa: SSAValue) -> tuple[tuple, ...]:
        shapes = []
        for result_id in self._native_result_ids_for_op(ssa):
            meta = self.graph.result_meta.get(result_id)
            if meta is None:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"{ssa.op} native per-result metadata is missing for SSA value {result_id}",
                    op_name=ssa.op,
                )
            # Shared-memory descriptors and other non-register native values do
            # not define the logical thread-tile extent. A malformed or unknown
            # *tensor* still goes through the strict parser/ownership check.
            if not meta.type.is_tensor and not meta.type.raw.strip().startswith("tensor<"):
                continue
            shapes.append(self._native_shape(result_id, op_name=ssa.op))
        return tuple(shapes)

    def _get_shape(self, ssa_id: int) -> tuple:
        """Return the tracked shape for an SSA value.

        Returns the shape tuple from env_shapes if tracked, otherwise
        attempts to infer from the op's type_str via _extract_shape.
        Falls back to () (scalar) if no shape information is available.
        """
        if ssa_id in self.env_shapes:
            return self.env_shapes[ssa_id]
        # Try to infer from the op's type_str
        for op in self.graph.ops:
            if op.id == ssa_id and op.type_str:
                shape = _extract_shape(op.type_str)
                if shape:
                    self.env_shapes[ssa_id] = shape
                    return shape
        return ()

    def _propagate_shape_from_type(self, ssa: SSAValue):
        """Set env_shapes[ssa.id] from the op's result type_str.

        Used as a common shape-propagation step after lowering an op.
        If the type_str contains a tensor shape, record it; otherwise
        the value is implicitly scalar (shape = ()).
        """
        if ssa.type_str:
            shape = _extract_shape(ssa.type_str)
            if shape:
                self.env_shapes[ssa.id] = shape
                # Also track elements-per-thread for the Phase 4 refactor.
                self._track_n_elems(ssa.id, ssa.type_str, shape)
                return
        # No tensor type → scalar
        self.env_shapes[ssa.id] = ()
        self.env_n_elems[ssa.id] = 1

    def _track_n_elems(self, ssa_id: int, type_str: str, shape: tuple):
        """Compute elements-per-thread for ``ssa_id`` from its TTGIR layout.

        For ``tensor<NxT, #alias>`` we look up ``#alias = #ttg.blocked<...>``
        or ``#ttg.linear<...>`` in ``self.graph.mod_text`` and use the
        ``LinearLayout`` machinery to count the number of register basis
        vectors (each doubles the elements-per-thread).

        When the layout can\'t be resolved (no alias, or inline layout),
        defaults to ``max(1, numel // num_threads)`` — the value the
        per-thread scalar model implicitly assumes today.

        Storing this lets the convert_layout guard distinguish
        \"genuinely needs MEPT shuffle\" from \"1-elem-per-thread is fine.\"
        See ``docs/superpowers/plans/2026-05-21-multi-element-per-thread.md``.
        """
        if not shape:
            self.env_n_elems[ssa_id] = 1
            return
        numel = 1
        for d in shape:
            numel *= d
        num_threads = (self.options.num_warps if self.options else 4) * 32
        default = max(1, numel // num_threads)

        mod_text = getattr(self.graph, "mod_text", "") or ""
        if not mod_text:
            self.env_n_elems[ssa_id] = default
            return

        alias_match = re.search(r",\s*#(\w+)\s*>\s*$", type_str)
        if not alias_match:
            self.env_n_elems[ssa_id] = default
            return
        alias = alias_match.group(1)

        # #ttg.linear → use parser directly
        if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.linear<", mod_text):
            try:
                from triton_msl.codegen._linear_layout import parse_linear_layout

                ll = parse_linear_layout(mod_text, alias)
                if ll:
                    self.env_n_elems[ssa_id] = ll.num_registers_per_thread
                    self.env_layout[ssa_id] = ll
                    return
            except Exception:
                pass

        # #ttg.blocked → convert to linear
        if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.blocked<", mod_text):
            try:
                from triton_msl.codegen._linear_layout import blocked_to_linear

                spt = self._parse_blocked_field(mod_text, alias, "sizePerThread")
                tpw = self._parse_blocked_field(mod_text, alias, "threadsPerWarp")
                wpc = self._parse_blocked_field(mod_text, alias, "warpsPerCTA")
                order = self._parse_blocked_field(mod_text, alias, "order")
                if spt and tpw and wpc and order:
                    ll = blocked_to_linear(spt, tpw, wpc, order, tuple(shape))
                    if ll:
                        self.env_n_elems[ssa_id] = ll.num_registers_per_thread
                        self.env_layout[ssa_id] = ll
                        return
                # Fall back: product of sizePerThread is the local-tile size
                # per thread; the rest is replication.
                if spt:
                    n_per_thread = 1
                    for s in spt:
                        n_per_thread *= s
                    self.env_n_elems[ssa_id] = max(1, n_per_thread)
                    return
            except Exception:
                pass

        self.env_n_elems[ssa_id] = default

    def _resolve_linear_layout(self, type_str: str, shape: tuple):
        """Resolve a ``tensor<NxT, #alias>`` type to a ``LinearLayout``.

        Returns the ``LinearLayout`` for the layout aliased in ``type_str``
        (``#ttg.linear`` directly, ``#ttg.blocked`` via ``blocked_to_linear``)
        or ``None`` if it can't be resolved (no alias, inline layout, parse
        failure). Used by the Phase 4d ``convert_layout`` shuffle to obtain
        both the source and destination layouts. Mirrors the resolution in
        ``_track_n_elems`` but returns the layout instead of just the count.
        """
        mod_text = getattr(self.graph, "mod_text", "") or ""
        if not mod_text or not type_str:
            return None
        alias_match = re.search(r",\s*#(\w+)\s*>\s*$", type_str)
        if not alias_match:
            return None
        alias = alias_match.group(1)
        try:
            if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.linear<", mod_text):
                from triton_msl.codegen._linear_layout import (
                    parse_linear_layout,
                )

                return parse_linear_layout(mod_text, alias)
            if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.blocked<", mod_text):
                from triton_msl.codegen._linear_layout import (
                    blocked_to_linear,
                )

                spt = self._parse_blocked_field(mod_text, alias, "sizePerThread")
                tpw = self._parse_blocked_field(mod_text, alias, "threadsPerWarp")
                wpc = self._parse_blocked_field(mod_text, alias, "warpsPerCTA")
                order = self._parse_blocked_field(mod_text, alias, "order")
                if spt and tpw and wpc and order:
                    return blocked_to_linear(spt, tpw, wpc, order, tuple(shape))
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_blocked_field(mod_text: str, alias: str, field: str):
        """Pull an ``int`` list field out of ``#alias = #ttg.blocked<{...}>``.

        ``field`` is one of ``sizePerThread``, ``threadsPerWarp``,
        ``warpsPerCTA``, ``order``. Returns the list or ``None`` if the
        layout/field can\'t be parsed.
        """
        m = re.search(
            rf"#{re.escape(alias)}\s*=\s*#ttg\.blocked<\{{(.+?)\}}>",
            mod_text,
            re.DOTALL,
        )
        if not m:
            return None
        body = m.group(1)
        fm = re.search(rf"{field}\s*=\s*\[([^\]]*)\]", body)
        if not fm:
            return None
        try:
            return [int(p.strip()) for p in fm.group(1).split(",") if p.strip()]
        except ValueError:
            return None

    def _propagate_shape_elementwise(self, ssa: SSAValue):
        """Propagate shape for element-wise ops (arith, math, select, etc.).

        Element-wise ops inherit the shape of their operands.  When operands
        have different shapes (e.g., scalar + vector due to implicit broadcast),
        we take the "largest" shape — the one with the most elements.

        Falls back to _propagate_shape_from_type if no operand shapes are
        available.
        """
        best_shape = ()
        for op_id in ssa.operand_ids:
            s = self._get_shape(op_id)
            if len(s) > len(best_shape):
                best_shape = s
            elif len(s) == len(best_shape):
                # Same rank — pick the one with more total elements
                if _shape_numel(s) > _shape_numel(best_shape):
                    best_shape = s
        if best_shape != ():
            self.env_shapes[ssa.id] = best_shape
        else:
            self._propagate_shape_from_type(ssa)

    @property
    def _lid_expr(self):
        """Return the per-element index expression.

        When total elements > 1024 and a wrapping loop is active,
        returns '_loop_e' (the loop variable). Otherwise returns 'lid'.
        """
        return "_loop_e" if getattr(self, "_needs_wrapping", False) else "lid"

    # -- Multi-pass reduction helpers ------------------------------------------

    def _split_ops_by_reductions(self):
        """Split ops into phases separated by tt.reduce ops.

        Returns a list of (ops_list, is_reduce) tuples. Reduce ops are
        isolated in their own single-element phases so they can be emitted
        between per-element loops.
        """
        phases = []
        current_phase = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.reduce":
                if current_phase:
                    phases.append((current_phase, False))
                phases.append(([ssa], True))
                current_phase = []
            else:
                current_phase.append(ssa)
        if current_phase:
            phases.append((current_phase, False))
        return phases

    def _collect_tensor_deps(self, target_ops, all_preceding_ops, reduce_result_ids):
        """Find all ops from earlier phases needed to compute target_ops.

        Walks backward through operand_ids from target_ops, collecting any
        ops from all_preceding_ops whose results are tensor-shaped (per-element)
        and therefore must be re-computed inside the current loop.

        Args:
            target_ops: ops in the current phase that need per-element inputs
            all_preceding_ops: all ops from earlier phases (ordered)
            reduce_result_ids: set of SSA IDs that are reduce results (scalars,
                available outside loops)

        Returns:
            List of ops (in original order) that must be re-emitted in the loop.
        """
        # Build lookup: SSA ID → op
        op_by_id = {}
        for ssa in all_preceding_ops:
            op_by_id[ssa.id] = ssa

        # Collect IDs we need by walking dependencies backward
        needed_ids = set()
        worklist = []
        for ssa in target_ops:
            for dep_id in ssa.operand_ids:
                if dep_id in op_by_id and dep_id not in reduce_result_ids:
                    worklist.append(dep_id)

        while worklist:
            dep_id = worklist.pop()
            if dep_id in needed_ids:
                continue
            if dep_id in reduce_result_ids:
                continue
            if dep_id not in op_by_id:
                continue
            needed_ids.add(dep_id)
            dep_op = op_by_id[dep_id]
            for sub_id in dep_op.operand_ids:
                if sub_id not in needed_ids and sub_id in op_by_id:
                    worklist.append(sub_id)

        # Return in original order
        return [op for op in all_preceding_ops if op.id in needed_ids]

    @staticmethod
    def _is_scalar_op(ssa):
        """Return True if an op produces a scalar value (not per-element).

        Scalar ops can be emitted outside loops because they don't depend
        on the per-element index. This includes program_id, scalar
        constants, scalar arithmetic, and passthrough ops like splat on
        a scalar. Tensor ops (loads, stores, tensor arithmetic) must go
        inside per-element loops.
        """
        # Tensor-flagged ops are per-element
        if ssa.is_tensor:
            return False
        # Loads and stores are always per-element
        if ssa.op in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"):
            return False
        # tt.reduce is handled separately
        if ssa.op == "tt.reduce":
            return False
        return True

    # Ops/contexts with no correct lowering AND no viable fallback: their
    # presence means the kernel can only produce silently-wrong output, so the
    # lowerer refuses instead. The catalog (the unsafe-op set and every case
    # predicate) lives in ``triton_msl.codegen.refusal_catalog`` — the single
    # source of truth shared with the C++ path and the generated docs. Keep it
    # tight: only ops that are both unsupported and *unsafe to approximate*.

    def _refuse_unsafe_unsupported_ops(self):
        """Raise MetalNonRecoverableError for op/context combinations that can
        only be lowered to silently-wrong output.

        The cases are defined once in
        ``triton_msl.codegen.refusal_catalog`` (the single source of truth
        shared with the C++ path and the generated docs); this method just
        builds the :class:`RefusalContext` and walks the catalog. Each case
        was a confirmed silent-wrong producer found by classifying skip-listed
        tests; refusing keeps the integrity guarantee (never return numbers we
        can't vouch for). Guards are scoped so the supported cases (test_cat
        1-D, test_join, test_noinline non-shared, top-level tt.dot,
        value-returning early returns) are untouched.
        """
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen import refusal_catalog as _rc

        def _walk(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _walk(s.region_ops)
                if s.else_ops:
                    yield from _walk(s.else_ops)

        ctx = _rc.RefusalContext(
            all_ops=list(_walk(self.graph.ops)),
            top_level_ops=self.graph.ops,
            called_funcs=self.graph.called_funcs or [],
            find_op_type_str=self._find_op_type_str,
            extract_shape=_extract_shape,
            supported_assert_ids=frozenset(self._assert_plan["ids"])
            if getattr(self, "_assert_plan", None)
            else frozenset(),
        )
        violation = _rc.check_all(ctx)
        if violation is not None:
            raise MetalNonRecoverableError(violation.message, op_name=violation.op_name)

    def lower(self) -> str:
        """Lower the IRGraph to MSL source code."""
        msl = self._lower_routes()
        if getattr(self, "_store_mask_replay_only", False) and not getattr(self, "_source_epilogue_replay", False):
            from triton_msl.errors import MetalNonRecoverableError

            # Fail closed: this store mask was admitted for the mask-executing route only.
            raise MetalNonRecoverableError(
                "Refusing to emit silently-wrong output: the output store mask is executed only by the "
                "source-replayed K-loop epilogue route, and another route claimed this kernel.",
                op_name="tt.dot",
            )
        return msl

    def _lower_routes(self) -> str:
        from .assertions import prepare

        self._assert_plan = prepare(self)
        # Integrity prescan (PR1): some ops have no correct lowering on this
        # backend AND the legacy fallback can't help — emitting anything for
        # them yields silently-wrong output (a kernel that runs and returns
        # wrong numbers). Refuse rather than approximate. ``tt.dot_scaled``
        # is microscaling (mxfp) matmul: Apple GPUs have no microscaling
        # units and our codegen has no handler, so the result tensor is
        # never computed -> garbage (test_scaled_dot, ~all configs mismatch).
        self._refuse_unsafe_unsupported_ops()

        if self._assert_plan is not None:
            # Templates do not replay assertions. The proved generic path below
            # owns both the rendezvous and the sealed host-result check.
            self._attention_outputs = self._attention_store_arguments(forward_only=True)
            return self._lower_generic()

        # Integrity prescan (never silent-wrong): FlashAttention-style kernels
        # (>= 2 tt.dot with a softmax/exp between them) are validated only for
        # head_dim <= 64. Above that the attention lowering SILENTLY mis-computes
        # (head_dim=128 produced garbage — max error ~1000 — with NO error raised
        # at BLOCK_M=16/8; only the large-block case happened to fail loudly with
        # OutOfResources). Refuse loudly instead. head_dim 32 and 64 are validated
        # + supported; large-head_dim FA (tiled threadgroup memory) is future work.
        def _fa_walk(ops):
            for _s in ops:
                yield _s
                if getattr(_s, "region_ops", None):
                    yield from _fa_walk(_s.region_ops)
                if getattr(_s, "else_ops", None):
                    yield from _fa_walk(_s.else_ops)

        _fa_ops = list(_fa_walk(self.graph.ops))
        _fa_dots = [s for s in _fa_ops if s.op == "tt.dot"]
        # FA-specific gate: >=2 dots with BOTH exp and a max (softmax with the
        # numerical-stability max between the dots). Requiring the max excludes
        # non-FA >=2-dot kernels that merely use exp (e.g. a fused MLP with a
        # gelu/silu activation), so this guard can't over-refuse them.
        # Match the actual exp op (math.exp / math.exp2 / tt.exp) precisely — a
        # bare ``"exp" in op`` also matches ``tt.expand_dims``, which would wrongly
        # pull a non-FA >=2-dot + max + expand_dims kernel into FA handling.
        _fa_has_exp = any(_op_is_exp(s.op) for s in _fa_ops)
        _fa_has_max = any("max" in (s.op or "") for s in _fa_ops)
        # Biased-FA BACKWARD (trifast _bwd_kv/_bwd_q/_bwd_b): FA-shaped (>=2 dots +
        # exp2) but NO stability-max (P is recomputed from the saved lse), so it does
        # NOT enter the forward FA block below. Route it to the backward templates
        # here. Returns None for non-backward kernels (falls through unchanged);
        # raises on ambiguity (never silently-wrong gradients).
        if len(_fa_dots) >= 2 and _fa_has_exp and not _fa_has_max:
            _bwd_info = self._detect_biased_fa_backward()
            if _bwd_info is not None:
                self._attention_outputs = self._attention_store_arguments()
                return self._lower_biased_fa_backward(_bwd_info)

        # SYSTEMIC anti-silent-wrong gate for the remaining matmul/forward-FA
        # template store sites. Run it after backward detection because that
        # detector independently proves every backward output store mask and
        # Triton's N_CTX==1 specialization replaces the boundary arg with a
        # constant. Only a fully proved backward route bypasses this gate;
        # an unrecognized or ambiguous graph still reaches the fail-closed
        # matmul classification below.
        self._refuse_nontrivial_template_output_mask()

        if len(_fa_dots) >= 2 and _fa_has_exp and _fa_has_max:
            self._attention_outputs = self._attention_store_arguments()
            # VARLEN (packed cu_seqlens) FA-2: the dominant real-world FA shape (flash-attn /
            # vLLM), which the dense [Z,H,N,D] templates can't express and the generic path
            # runs ~0.04 TF. Route the tiled varlen template FIRST — its 4-float + 2-int-ptr +
            # 2-seqlen-subi signature cannot collide with dense/biased/MLA FA (checked before
            # the dense detectors below). Returns None for non-varlen (falls through); routes
            # both non-causal AND standard within-seq causal (a non-standard mask refuses).
            _varlen_info = self._detect_varlen_flash_attention()
            if _varlen_info is not None:
                return self._lower_varlen_flash_attention(_varlen_info)
            _fa_maxdim = 0
            _fa_mindim = 1 << 30
            for _d in _fa_dots:
                for _oid in _d.operand_ids or []:
                    _t = self._find_op_type_str(_oid)
                    _sh = _extract_shape(_t) if _t else None
                    if _sh:
                        _dims = [d for d in _sh if d > 1]  # ignore broadcast/size-1
                        if _dims:
                            _fa_maxdim = max([_fa_maxdim] + _dims)
                            _fa_mindim = min([_fa_mindim] + _dims)
            if _fa_mindim in (8, 16) and _fa_maxdim == 64:
                from ._decode_template import detect as _detect_small_query

                _decode_info = _detect_small_query(self)
                if _decode_info is not None:
                    return self._lower_flash_attention_template(_decode_info)
            # BIASED / triangle attention (trifast): a bias tile as the QK dot's C
            # operand + a loaded mask -> -inf + a per-row lse + a const result-scale
            # (inv_ln2) with exp2. Route to the tiled/simd template WITH bias/mask/lse
            # fused into the online-softmax loop, BEFORE the dot-result-scale guard
            # below (which would otherwise refuse the inv_ln2 scale) AND before the
            # bf16 gate — the biased path SUPPORTS bf16 via the scalar tiled template
            # (loads promote bfloat->fp32; simd stays fp16/fp32). Returns None for a
            # non-biased FA (the QK dot's C is a zero/loop-acc, not a load), so
            # standard FA flows through to the gates below unchanged.
            from ._biased_replay import lower_wide_rows as _lower_wide_biased_rows
            from ._biased_replay import wide_row_plan as _wide_biased_row_plan

            _wide_row_plan = _wide_biased_row_plan(self)
            if _wide_row_plan is not None:
                self._source_biased_replay = True
                scalar = _lower_wide_biased_rows(self, _wide_row_plan)
                # One threadgroup per program with a barrier between its last load and
                # its stores, and the cooperative candidate checks overlap itself: an
                # output over an input keeps the source's result (pinned in
                # tests/test_attention_capability.py), so there is nothing to refuse.
                self._attention_outputs = None
                if getattr(self.options, "experimental_cooperative_attention", False) is True:
                    from ._cooperative_attention import build_candidate, make_descriptor

                    candidate = build_candidate(self, _wide_row_plan)
                    if candidate is not None:
                        self._flash_attention = make_descriptor(self, _wide_row_plan, scalar, candidate)
                return scalar

            from ._biased_replay import eligible as _biased_replay_eligible

            if _biased_replay_eligible(self):
                self._source_biased_replay = True
                replayed = "// Source-replayed biased attention\n" + self._lower_generic()
                from ._biased_replay import store_follows_every_load as _store_follows_every_load

                if _store_follows_every_load(self, replayed):
                    # One threadgroup per program and a barrier between its last load
                    # and its one store: an output over an input keeps the source's
                    # result, so there is nothing to refuse.
                    self._attention_outputs = None
                return replayed
            _biased_info = self._detect_biased_flash_attention()
            if _biased_info is not None:
                return self._lower_biased_flash_attention_template(_biased_info)
            # DTYPE GATE (2026-06-21 audit): the attention lowering — generic,
            # C++, and the tiled/simdgroup templates — is validated only for
            # fp32/fp16 dot operands. bf16 dot operands SILENTLY mis-compute
            # (head_dim 32/64 dispatched wrong numbers, max err 0.4..218; the
            # head_dim>64 guard never fired, no dtype guard existed). Refuse them
            # LOUDLY here, before any head_dim routing, rather than emit garbage.
            # Covers bf16 introduced via explicit .to(tl.bfloat16) on Q/K/V or the
            # scores (any dot operand). NON-biased FA stays fp16/fp32 only.
            _fa_has_bf16 = any(
                (self._find_op_type_str(_oid) or "")
                and ("bf16" in (self._find_op_type_str(_oid) or "") or "bfloat" in (self._find_op_type_str(_oid) or ""))
                for _d in _fa_dots
                for _oid in (_d.operand_ids or [])
            )
            if _fa_has_bf16:
                raise MetalNonRecoverableError(
                    "FlashAttention with bf16 dot operands is not supported: the "
                    "attention lowering (fp32/fp16 only) silently mis-computes for "
                    "bf16 at any head_dim. Refusing to emit silently-wrong output. "
                    "Use fp16 or fp32 for Q/K/V in attention."
                )
            # FUSED-SCALE-ON-DOT-RESULT GATE (naming-independence follow-up): the generic
            # attention lowering silently mis-computes when a tt.dot RESULT feeds an
            # elementwise scale/bias before the softmax — e.g. the scores scaled INSIDE
            # the loop ``qk = tl.dot(q, kᵀ); qk = qk * (1/√d)`` (rel_err ~1.1: the scale
            # is dropped/mis-applied). The validated form scales Q BEFORE the dot
            # (``q = q * scale; qk = tl.dot(q, kᵀ)``) — no compute op on the dot result —
            # so this gate doesn't touch it. (A general dot-result compute epilogue in a
            # K-loop is a known unhandled case for single-dot matmuls too; FA is the
            # multi-dot instance.) This was previously masked by the name-match
            # ``has_strides`` gate in _detect_simple_dot; guard it HERE for any naming.
            _fa_by_id = {s.id: s for s in _fa_ops}

            def _is_scalar_splat(_id):
                # True if SSA _id is a scalar broadcast into the tile — a splat of a
                # constant/scalar (the scale/bias factor), NOT a per-element tensor
                # (like the loop-carried accumulator). Walks layout-only wrappers.
                _o = _fa_by_id.get(_id)
                _seen = set()
                while _o is not None and _o.id not in _seen:
                    _seen.add(_o.id)
                    if _o.op in ("tt.splat",):
                        return True
                    if _o.op in (
                        "tt.broadcast",
                        "ttg.convert_layout",
                        "tt.reshape",
                        "tt.expand_dims",
                        "arith.constant",
                    ):
                        if _o.op == "arith.constant":
                            return True
                        _o = _fa_by_id.get(_o.operand_ids[0]) if _o.operand_ids else None
                        continue
                    break
                return False

            def _dot_result_scaled(_dot):
                # True only when the dot's result is multiplied/divided (always a scale),
                # or added/subtracted with a SCALAR-SPLAT other operand (a fused bias).
                # The standard FA accumulation ``acc += tt.dot(p, v)`` adds the dot to a
                # loop-carried TENSOR accumulator (not a scalar splat) -> NOT flagged, so
                # the validated FA is unaffected; only a true scale/bias on the scores is.
                _dot_ids = {_dot.id} | set(getattr(_dot, "result_ids", None) or [])
                for _s in _fa_ops:
                    if not _dot_ids.intersection(_s.operand_ids or []):
                        continue
                    # Triton's N_CTX==1 specialization erases the one-iteration K
                    # loop.  The canonical P@V accumulator is then the second dot
                    # itself and the source's final ``acc / l`` becomes a direct
                    # tensor division of that dot.  It is not an unreplayed scalar
                    # epilogue.  Admit only the exact terminal shape here; routing
                    # below is mandatory and the dense value-path verifier still
                    # proves the denominator recurrence and stored result.
                    if _is_terminal_pv_normalization(_dot, _s):
                        continue
                    if _s.op in ("arith.mulf", "arith.divf", "arith.muli"):
                        return True
                    if _s.op in ("arith.addf", "arith.subf", "arith.addi"):
                        # the operand that is NOT the dot must be a scalar splat (bias);
                        # a tensor accumulator (acc += dot) is the valid FA form.
                        _others = [o for o in (_s.operand_ids or []) if o != _dot.id]
                        if any(_is_scalar_splat(o) for o in _others):
                            return True
                return False

            def _is_terminal_pv_normalization(_dot, _consumer):
                if _dot is not _fa_dots[-1] or _consumer.op != "arith.divf":
                    return False
                _dot_ids = {_dot.id} | set(getattr(_dot, "result_ids", None) or [])
                _operands = list(_consumer.operand_ids or [])
                if len(_operands) != 2 or _operands[0] not in _dot_ids:
                    return False
                # A scalar divisor is the P-side scale the guard exists to catch.
                if _is_scalar_splat(_operands[1]):
                    return False
                _direct = [_op for _op in _fa_ops if _dot_ids.intersection(_op.operand_ids or [])]
                if _direct != [_consumer]:
                    return False
                _stores = [_op for _op in _fa_ops if _op.op == "tt.store"]
                if len(_stores) != 1 or len(_stores[0].operand_ids or []) < 2:
                    return False
                _cur = _stores[0].operand_ids[1]
                _wrappers = {
                    "ttg.convert_layout",
                    "arith.extf",
                    "arith.truncf",
                    "tt.fp_to_fp",
                }
                _seen = set()
                while _cur not in _seen:
                    if _cur == _consumer.id or _cur in set(getattr(_consumer, "result_ids", None) or []):
                        return True
                    _seen.add(_cur)
                    _op = _fa_by_id.get(_cur)
                    if _op is None or _op.op not in _wrappers or len(_op.operand_ids or []) != 1:
                        return False
                    _cur = _op.operand_ids[0]
                return False

            def _refuse_dot_result_epilogue():
                raise MetalNonRecoverableError(
                    "FlashAttention that applies an elementwise scale/bias to a tt.dot "
                    "RESULT is supported only when the complete QK-result-to-softmax "
                    "path is a proven positive finite CONSTANT multiply/divide chain "
                    "and a specialized dense/varlen template can replay it. Runtime "
                    "scales, bias/addition, P-side scaling, and unrecognized chains "
                    "would be silently dropped or mis-applied by the generic lowering; "
                    "refusing to emit silently-wrong output.",
                    op_name="tt.dot",
                )

            _terminal_pv_normalization = any(_is_terminal_pv_normalization(_fa_dots[-1], _s) for _s in _fa_ops)
            _scaled_dots = [_d for _d in _fa_dots if _dot_result_scaled(_d)]
            _constant_result_scale = None
            if _scaled_dots:
                # #6a is specifically the first (QK) dot's score scale.  Keep every
                # P@V/result epilogue and multi-dot/MLA spelling under the old refusal.
                # The varlen route already ran above; dense routing below must consume
                # this proof, otherwise falling into generic lowering is still unsafe.
                if len(_fa_dots) == 2 and _scaled_dots == [_fa_dots[0]]:
                    _constant_result_scale = self._fa_analyze_constant_score_scale(_fa_dots[0], _fa_dots[1])
                if _constant_result_scale is None or not _constant_result_scale["op_ids"]:
                    _refuse_dot_result_epilogue()
            # Empirically mapped failure boundary (2026-06-17): the attention
            # lowering is validated only at BLOCK_M = BLOCK_N = 32, head_dim in
            # {32, 64}. Two distinct out-of-range failure modes — only ONE was
            # being guarded before, leaving a silent-wrong hole:
            #   (a) max dot-tile dim > 64 (head_dim > 64): at the real BLOCK=32 this
            #       fails LOUDLY with OutOfResources, but refuse here for a clear
            #       message rather than a cryptic pipeline-state error.
            #   (b) min dot-tile dim < 32 (BLOCK_M/BLOCK_N below 32): SILENTLY
            #       mis-computes — rows past the first become garbage (max err
            #       28..1e4) — for ANY head_dim, INCLUDING the otherwise-supported
            #       32 and 64. The old head_dim>64 guard missed this entirely
            #       (head_dim=64, BLOCK=16 returned silent garbage). (BLOCK > 32 is
            #       already refused by other guards before reaching here.)
            #
            # ROUTE the supported large-head_dim configs to the head-dim-tiled
            # FA2 template (Task 1) BEFORE the head_dim>64 refusal: head_dim=128,
            # BLOCK_M=BLOCK_N=32, fp32 (Task 3) OR fp16 (Task 4 — fp16 in /
            # fp32 accumulate / fp16 out via the cast epilogue), causal OR
            # non-causal (Task 5 — causal mask in the online-softmax loop).
            # _detect_flash_attention raises on ANY ambiguity (a mis-read stride
            # would silently mis-compute), so reaching the template means every
            # param resolved unambiguously.
            #
            # Only consult the detector when a dot tile reaches 128 — i.e. the
            # head_dim the tiled template targets. head_dim 32/64 lower through
            # the normal generic/C++ path (validated, supported) and must NOT be
            # routed here; invoking the refuse-on-ambiguity detector on them
            # would wrongly block a previously-working kernel whose (different)
            # addressing the FA stride-walker doesn't model.
            if _fa_maxdim in (128, 192, 256):
                # Symmetric hd128 (existing) AND asymmetric MLA (qk up to 256, v in
                # {64,128}): _fa_maxdim is the LARGER qk dim, so MLA (qk=192) lands here
                # too. maxdim>128 was previously an unconditional refuse, so detection
                # here can only ADD capability or refuse identically -- no regression.
                # This prescan is intentionally broader than proven attention
                # (e.g. a ReLU MLP followed by softmax also has two dots, exp,
                # and max). Keep the established role-specific diagnostic when
                # the conventional FA stride ABI is present; otherwise use a
                # neutral subject. This changes wording only, never admission.
                _fa_arg_names = {str(a.name).lower() for a in self.graph.args}
                _named_fa_abi = {
                    "stride_qm",
                    "stride_qk",
                    "stride_kn",
                    "stride_kk",
                }.issubset(_fa_arg_names)
                info = self._detect_flash_attention(
                    unresolved_subject=None if _named_fa_abi else "multi-dot softmax-shaped kernel",
                )
                if (
                    info is not None
                    and info["block_m"] == 32
                    and info["block_n"] == 32
                    and info["out_dtype"] in ("f32", "f16")
                ):
                    # Symmetric hd128 -> template (picks simd if contiguous, else tiled).
                    if not info.get("is_mla") and info["head_dim"] == 128 and info.get("v_head_dim", 128) == 128:
                        return self._lower_flash_attention_template(info)
                    # MLA (nope/rope 3-dot): the two QK tensors are separate; route to the
                    # cat-dispatch (concat q_nope|q_rope, k_nope|k_rope -> the validated
                    # asymmetric qk=head_dim/v=v_head_dim kernel). fail-CLOSED (the qk=192
                    # kernel has a different ABI than the @jit kernel, so the host path
                    # can't run it) -> refuse below on ineligibility.
                    if info.get("is_mla") and _simd_fa_eligible(info):
                        return self._lower_mla_attention_template(info)
                    # Asymmetric single-tensor (qk != v): simd-only. Route iff simd-eligible
                    # (contiguous, v in {64,128}, fp16 for qk>128); else fall through -> refuse.
                    if not info.get("is_mla") and _v_head_dim_of(info) != info["head_dim"] and _simd_fa_eligible(info):
                        return self._lower_flash_attention_template(info)

            elif _fa_maxdim == 64:
                # hd64: route fp16/fp32 to the simd MMA template when eligible. A native,
                # uniform-bf16 32x32 source instead uses the scalar/tiled template, whose
                # bfloat ABI promotes loads and computes in fp32. Other hd64 graphs retain
                # their generic path/refusal. We must NEVER regress that path, so detection
                # ambiguity or ineligibility falls through unchanged. Unlike the hd128 branch
                # (where a detect refusal is intentional -> the head_dim>64 refusal below), a
                # refusal here is NOT a hard error: hd64 has a correct generic fallback.
                if _constant_result_scale is not None or _terminal_pv_normalization:
                    # Once the generic-result guard is relaxed, detection/routing is
                    # mandatory: generic lowering is the GPU-confirmed silent-wrong.
                    _info64 = self._detect_flash_attention()
                else:
                    try:
                        _info64 = self._detect_flash_attention()
                    except MetalNonRecoverableError:
                        _info64 = None
                if (
                    _info64 is not None
                    and _constant_result_scale is None
                    and not _terminal_pv_normalization
                    and self._fa_single_query_generic_eligible(_info64)
                ):
                    # A folded query length does not make this a single-length
                    # source. Keep its actual masks in generic lowering; never
                    # weaken the template's own output-mask safety check.
                    _info64 = None
                _bf16_tiled64 = (
                    _info64 is not None
                    and _info64["head_dim"] == 64
                    and _info64.get("v_head_dim", 64) == 64
                    and _info64["block_m"] == 32
                    and _info64["block_n"] == 32
                    and _info64["out_dtype"] == "bf16"
                    # This is the native-bf16 buffer ABI. Explicit bf16 dot
                    # operands remain behind the integrity refusal above: the
                    # source here promotes its loads before both dots, while
                    # the tiled maker promotes the same bfloat loads to fp32.
                    and not _fa_has_bf16
                    and not _info64.get("is_mla")
                    and all(str(self.graph.args[_info64[role]].elem_type) == "bf16" for role in ("q", "k", "v", "out"))
                )
                if (
                    _info64 is not None
                    and _info64["head_dim"] == 64
                    and (
                        _simd_fa_eligible(_info64)
                        or (
                            _info64["out_dtype"] in ("f32", "f16")
                            and _info64["block_m"] in (32, 64)
                            and _info64["block_n"] in (32, 64)
                            and (_info64["block_m"], _info64["block_n"]) != (32, 32)
                            and _info64.get("v_head_dim", 64) == 64
                            and not _info64.get("is_mla")
                        )
                        or _bf16_tiled64
                    )
                ):
                    return self._lower_flash_attention_template(_info64)

            if _constant_result_scale is not None or _terminal_pv_normalization:
                # Exact constant algebra alone is insufficient when the template's
                # supported shape/ABI gates did not claim the kernel.
                _refuse_dot_result_epilogue()

            if _fa_maxdim > 64:
                raise MetalNonRecoverableError(
                    f"FlashAttention head_dim > 64 (a dot tile dimension is "
                    f"{_fa_maxdim}) is not supported: the attention lowering is "
                    f"validated only for head_dim <= 64. Refusing to emit "
                    f"silently-wrong output. Use head_dim <= 64."
                )
            if _fa_mindim < 32:
                raise MetalNonRecoverableError(
                    f"FlashAttention with a block tile dimension < 32 (smallest "
                    f"dot tile dimension is {_fa_mindim}) is not supported: the "
                    f"attention lowering silently mis-computes for BLOCK_M/BLOCK_N "
                    f"below 32 (rows past the first become garbage), for any "
                    f"head_dim. Refusing to emit silently-wrong output. Use "
                    f"BLOCK_M = BLOCK_N = 32."
                )

        # Check for the fused matmul + row-softmax pattern FIRST — before the
        # simple/K-loop dot detectors. _detect_simple_dot matches any
        # load→dot→store and emits a BARE matmul, silently dropping a softmax
        # (or any) epilogue (a simple matmul→softmax kernel returned A@B with
        # no softmax — row sums != 1). matmul_softmax only matches the full
        # dot→max→sub→exp→sum→div pattern, so a pure matmul still falls through
        # to simple_dot below. (Also must precede _requires_matmul_template,
        # whose legacy fallback drops the softmax too.)
        matmul_softmax_info = self._detect_matmul_softmax()
        if matmul_softmax_info:
            msl = self._lower_matmul_softmax_template(matmul_softmax_info)
            if msl is not None:
                self._prescan_stores()
                return msl

        # Fused matmul + pointwise/broadcast epilogue (bias, activation, scale,
        # chains). Same staged vehicle as softmax; the epilogue op chain is
        # lowered per-element on the staged tg_C. Checked after softmax (which
        # claims the reduce-bearing case) and before simple_dot (whose #157
        # epilogue-refusal is the catch-all for anything not matched here). #158.
        matmul_epilogue_info = self._detect_matmul_epilogue()
        if matmul_epilogue_info:
            msl = self._lower_matmul_softmax_template(matmul_epilogue_info)
            if msl is not None:
                self._prescan_stores()
                return msl

        # QUANTIZED weight-only matmul: B is an integer weight dequantized to float
        # before the dot ((w_i8.to(float) - zero) * scale, out = A @ B). The simdgroup
        # templates load B DIRECTLY from its pointer as float/half — a char* (int8)
        # weight either fails to compile (`simdgroup_load(float, char*)`) or, if types
        # matched, would read the weight RAW (no dequant) = silently wrong. Route the
        # CANONICAL case to the dedicated fast dequant kernel (make_int8_matmul_fast,
        # layout kn) via the quant_matmul dispatch descriptor; refuse anything else.
        if self._detect_quantized_dot():
            _qdesc = self._maybe_quant_matmul_descriptor() or self._maybe_quant_matmul_symmetric_descriptor()
            if _qdesc is None:
                raise MetalNonRecoverableError(
                    "quantized matmul (a tt.dot whose B operand is an integer weight "
                    "dequantized to float, e.g. (w_i8.to(float) - zero) * scale) is "
                    "supported only in the CANONICAL weight-only form: the standard "
                    "(input, weight, output, scale, zero, M, N, K, ...strides) signature "
                    "with weight stored [K,N] contiguous, per-N float scale/zero, and "
                    "fp32 input/output. This kernel deviates from that shape. Dequantize "
                    "on the host and pass a float/half weight, or restructure to the "
                    "canonical quantized-matmul signature."
                )
            # The compiled kernel IS the fast dequant kernel; dispatch_quant_matmul runs
            # it via compile_shader (MPS) with the simdgroup grid. Non-MPS refuses in the
            # driver (this path is compile_shader-only). effective_block_size is a
            # placeholder — the runtime dispatch computes the real grid.
            #
            # The compiled metallib is looked up by the kernel's func name, so rename the
            # fast kernel's function to match. dispatch_quant_matmul compiles the
            # descriptor's own copy (named int8_matmul_fast) separately via compile_shader,
            # so this rename is local to self._msl (which is never itself dispatched — the
            # driver either routes to compile_shader or refuses).
            self._quant_matmul = _qdesc
            self.effective_block_size = 32
            import re as _re

            # Untagged fast per-N descriptor: [0] IS the MSL (starts with #include). A
            # TAGGED descriptor (e.g. "pergroup_int8") carries its MSL at [1].
            _qmsl = _qdesc[1] if (isinstance(_qdesc[0], str) and not _qdesc[0].startswith("#include")) else _qdesc[0]
            _kname = _sanitize_msl_name(self.graph.func_name)
            return _re.sub(r"kernel\s+void\s+\w+\s*\(", f"kernel void {_kname}(", _qmsl, count=1)

        # QUANTIZED INT4 GEMV (decode): weight-only int4 (GPTQ/AWQ per-group) decode GEMV
        # with a packed-nibble unpack + per-group scale/zero. Checked BEFORE int8 (it is
        # the more specific pattern — the nibble unpack distinguishes it). Routes to
        # make_int4_gemv; correct-or-refuse (every packing constant verified).
        _gemv4 = self._maybe_quant_gemv_int4_descriptor()
        if _gemv4 is not None:
            self._quant_matmul = _gemv4
            self.effective_block_size = 32
            import re as _re

            _n4 = _sanitize_msl_name(self.graph.func_name)
            return _re.sub(r"kernel\s+void\s+\w+\s*\(", f"kernel void {_n4}(", _gemv4[1], count=1)

        # QUANTIZED GEMV (decode / M=1): a weight-only int8 GEMV written as an in-loop
        # row reduce of a dequantized [N,K] weight (`acc += tl.sum(x[None,:]*(w-zero),
        # axis=1)`, then `out = acc*scale`). This is the loop-carried 2-D-reduce pattern
        # the reduce guard refuses; route the CANONICAL shape to the dedicated
        # make_int8_gemv kernel (memory roofline) instead. Same dispatch descriptor +
        # compile_shader path as the quantized matmul, tagged "gemv".
        _gemv = self._maybe_quant_gemv_descriptor()
        if _gemv is not None:
            self._quant_matmul = _gemv
            self.effective_block_size = 32
            import re as _re

            _gname = _sanitize_msl_name(self.graph.func_name)
            return _re.sub(r"kernel\s+void\s+\w+\s*\(", f"kernel void {_gname}(", _gemv[1], count=1)

        # Round 3: a rank-3 dot is admitted only when the dedicated detector
        # proves the complete load/address/dot/store graph.  Every near miss
        # continues to the established rank>=3 refusal below.
        batched_dot = self._detect_batched_dot()
        if batched_dot:
            return self._lower_batched_dot_template(batched_dot)

        # Check for simple dot (no stride args, no scf.for) — use inline
        # scalar matmul that loads from global into shared memory, then
        # does per-thread dot product.
        from ._loop_epilogue import eligible as _loop_epilogue_eligible

        if _loop_epilogue_eligible(self):
            from ._loop_epilogue import deferred_wide_ops

            self._source_epilogue_replay = True
            self._skip_ids.update(deferred_wide_ops(self))
            return "// Source-replayed K-loop epilogue\n" + self._lower_generic()
        simple_dot = self._detect_simple_dot()
        if simple_dot:
            return self._lower_simple_dot_inline(simple_dot)

        # Check for tt.dot — switch to prebuilt matmul template
        if self._requires_matmul_template():
            msl = self._lower_dot_via_prebuilt_template()
            # Matmul template needs block_m * block_n threads (typically 1024)
            self.effective_block_size = self._matmul_block_size
            return msl

        # Check for tl.flip's reshape+xor-reduce pattern — emit direct flip.
        # Must run before _detect_3d_reduce, since the single-step flip case
        # (size-2 flip dim) looks like a 3D xor-reduce to that detector.
        flip_info = self._detect_flip()
        if flip_info:
            msl = self._lower_flip_template(flip_info)
            self.effective_block_size = flip_info["block_size"]
            # Record output arg indices for driver copy-back
            self._prescan_stores()
            return msl

        # Check for row-wise softmax (max + exp + sum + div). The generic
        # phase lowerer reads x_ptr 3 times and computes exp() twice; the
        # template caches the row in TG memory once and computes exp() once.
        # ~2x faster on M4 Max for n=1024.
        softmax_info = self._detect_softmax()
        if softmax_info:
            msl = self._lower_softmax_template(softmax_info)
            self.effective_block_size = (self.options.num_warps if self.options else 4) * 32
            self._prescan_stores()
            return msl

        # Check for row-wise layer norm (sum + sum_sq + sub + rsqrt). Same
        # TG-cache + float4 vectorization shape as softmax. Must be tried
        # AFTER softmax because the patterns share the "two reduces + sub"
        # signature (softmax: max+sum+exp; layer_norm: sum+sum+rsqrt).
        layer_norm_info = self._detect_layer_norm()
        if layer_norm_info:
            msl = self._lower_layer_norm_template(layer_norm_info)
            self.effective_block_size = (self.options.num_warps if self.options else 4) * 32
            self._prescan_stores()
            return msl

        # Check for the test_trans_reshape pattern: load 2-D, reshape to 4-D,
        # permute (1,2,3,0), reshape to 1-D, store. The generic phase lowerer
        # routes this through ttg.convert_layout from a multi-element-per-thread
        # #linear layout into #blocked, which the per-thread scalar model can\'t
        # honor. The template emits a closed-form transpose lookup directly.
        trans_reshape_info = self._detect_transpose_via_reshape()
        if trans_reshape_info:
            msl = self._lower_transpose_via_reshape_template(trans_reshape_info)
            self.effective_block_size = (self.options.num_warps if self.options else 4) * 32
            self._prescan_stores()
            return msl

        # Check for the generic N-D transpose pattern: load(rank>=3)->trans->
        # [reshape]->store with no reduce/control-flow (test_trans_4d). After
        # the more-specific detector above (which returns None here), emit a
        # closed-form direct-copy kernel (out[k] = in[src_flat(k)]) in a
        # strided loop that handles the >1024-element shapes the shared-memory
        # _lower_tt_trans path can't. The generic _lower_tt_trans rank>=3
        # refusal stays as the backstop for un-matched cases.
        nd_trans_info = self._detect_nd_trans()
        if nd_trans_info:
            msl = self._lower_nd_trans_template(nd_trans_info)
            self.effective_block_size = (self.options.num_warps if self.options else 4) * 32
            self._prescan_stores()
            return msl

        # Check for tl.sort / tl.topk applied to each row of a 2D tensor.
        # When total > 1024 threads are needed, the generic reduce path can't
        # run (threadgroup cap), but each row can be sorted independently in
        # a single thread with an in-register bitonic sort.
        sort_info = self._detect_row_wise_sort()
        if sort_info:
            msl = self._lower_row_wise_sort_template(sort_info)
            self.effective_block_size = sort_info["block_size"]
            # _prescan_stores already ran inside _detect_row_wise_sort
            return msl

        # Check for permute + chained sum-reduce (test_chained_reductions):
        # a large N-D tensor permuted then reduced over several axes into a
        # tiny output. Fuse the permute into the reduction index math and
        # cooperatively scatter-add — materializing the permute would exceed
        # threadgroup memory. Must run before _detect_3d_reduce (multi-reduce).
        pcr_info = self._detect_permute_chained_reduce()
        if pcr_info:
            msl = self._lower_permute_chained_reduce_template(pcr_info)
            self.effective_block_size = (self.options.num_warps if self.options else 4) * 32
            self._prescan_stores()
            return msl

        # Check for 3D reduce — switch to prebuilt template
        reduce_3d_info = self._detect_3d_reduce()
        if reduce_3d_info:
            if reduce_3d_info["combine_op"] in ("argmin", "argmax"):
                msl = self._lower_3d_argminmax_template(reduce_3d_info)
            else:
                msl = self._lower_3d_reduce_template(reduce_3d_info)
            self.effective_block_size = reduce_3d_info["block_size"]
            return msl

        return self._lower_generic()

    def _fa_single_query_generic_eligible(self, info):
        """Prove the narrow Q=1/K-runtime source which the dense detector conflates.

        This selects operation-by-operation lowering, not a mask-dropping template.
        The caller separately excludes both protected dot-result epilogue classes.
        No name-based length guess or exception-driven fallback is used here.
        """
        if (
            info.get("head_dim") != 64
            or info.get("v_head_dim", 64) != 64
            or info.get("block_m") != 32
            or info.get("block_n") != 32
            or info.get("out_dtype") not in ("f16", "f32")
            or info.get("is_mla")
            or type(info.get("N_CTX")) is not int
        ):
            return False

        def walk(ops):
            for op in ops:
                yield op
                yield from walk(op.region_ops or [])
                yield from walk(op.else_ops or [])

        ops = list(walk(self.graph.ops))
        by_id = {op.id: op for op in ops}
        args = {arg.id: arg for arg in self.graph.args}
        if not 0 <= info["N_CTX"] < len(self.graph.args):
            return False
        length = self.graph.args[info["N_CTX"]]
        if length.is_ptr:
            return False
        loops = [op for op in ops if op.op == "scf.for"]
        if len(loops) != 1:
            return False
        loop = loops[0]
        if (
            len(loop.operand_ids or []) < 3
            or loop.operand_ids[1] != length.id
            or self._const_int(loop.operand_ids[0], by_id) != 0
            or self._const_int(loop.operand_ids[2], by_id) != 32
        ):
            return False
        iv = (loop.attrs.get("block_arg_ids") or [None])[0]
        if iv is None:
            return False

        wrappers = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"}

        def core(vid):
            seen = set()
            while vid not in seen:
                seen.add(vid)
                op = by_id.get(vid)
                if op is None or op.op not in wrappers:
                    return vid
                if len(op.operand_ids or []) != 1:
                    return None
                vid = op.operand_ids[0]
            return None

        def base(vid):
            seen = set()
            while vid not in seen:
                seen.add(vid)
                if vid in args:
                    return args[vid].index if args[vid].is_ptr else None
                op = by_id.get(vid)
                if (
                    op is None
                    or op.op not in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout")
                    or not op.operand_ids
                ):
                    return None
                vid = op.operand_ids[0]
            return None

        def boundary(mask, row):
            cmp = by_id.get(core(mask))
            if (
                cmp is None
                or cmp.op != "arith.cmpi"
                or len(cmp.operand_ids or []) != 2
                or cmp.attrs.get("predicate_name") != "slt"
            ):
                return False
            lhs, rhs = cmp.operand_ids
            rhs = core(rhs)
            if row:
                if self._const_int(rhs, by_id) != 1:
                    return False
            elif rhs != length.id:
                return False
            sh = self._index_shape(lhs, by_id, args, iv, 1)
            if sh["bad"] or sh["mod"] is not None or sh["index_casts"] or sh["range"] != 1 or sh["bounds"] != (0, 32):
                return False
            return (
                (sh["iv"] == 0 and sh["pid"] == 0 and sh["pid_div"] is None and sh["coef"] == 32)
                if row
                else (sh["iv"] == 1 and sh["pid"] is None)
            )

        memory = [op for op in ops if op.op in ("tt.load", "tt.store")]
        if len(memory) != 4:
            return False
        expected = {
            info["q"]: ("tt.load", True),
            info["k"]: ("tt.load", False),
            info["v"]: ("tt.load", False),
            info["out"]: ("tt.store", True),
        }
        if len(expected) != 4:
            return False
        seen = set()
        for op in memory:
            role = base(op.operand_ids[0]) if op.operand_ids else None
            if role not in expected or role in seen or op.op != expected[role][0]:
                return False
            seen.add(role)
            mask_slot = 1 if op.op == "tt.load" else 2
            if len(op.operand_ids) <= mask_slot or not boundary(op.operand_ids[mask_slot], expected[role][1]):
                return False
            if op.op == "tt.load":
                if len(op.operand_ids) != 3:
                    return False
                other = by_id.get(core(op.operand_ids[2]))
                if other is None or other.op != "arith.constant" or other.attrs.get("value") != 0:
                    return False
        return True

    def _lower_generic(self) -> str:
        # Detect 2D kernel patterns (expand_dims + broadcast)
        self._prescan_2d_info()

        # Use BLOCK_SIZE from the kernel (graph.block_size), not num_warps * 32.
        # For 2D kernels, block_size = product of all dims.
        # For scalar-only kernels (no tt.make_range), use 1 thread.
        if self._is_2d and self._effective_2d_shape:
            block_size = 1
            for d in self._effective_2d_shape:
                block_size *= d
        else:
            block_size = self.graph.block_size
        if not self._has_tensor_ops():
            block_size = 1

        # For kernels with constant tensors (e.g. tl.full) but no make_range,
        # graph.block_size may be too small (defaults to num_warps*32).
        # Scan immutable native result shapes to find the actual max tensor size.
        # Walk nested regions too: a tensor living only inside an scf.for / scf.if body
        # is just as real, and missing it leaves the kernel sized for the narrower
        # top-level tiles -- the same silent truncation this scan exists to prevent.
        def _scan_nested(ops):
            for _o in ops:
                yield _o
                if getattr(_o, "region_ops", None):
                    yield from _scan_nested(_o.region_ops)
                if getattr(_o, "else_ops", None):
                    yield from _scan_nested(_o.else_ops)

        max_tensor_size = block_size
        for ssa in _scan_nested(self.graph.ops):
            for shape in self._native_shapes_for_op(ssa):
                if not shape:
                    continue
                total = 1
                for d in shape:
                    total *= d
                if total > max_tensor_size:
                    max_tensor_size = total
        if max_tensor_size > block_size:
            # No 1024 cap here. Capping meant that when the WIDEST tensor exceeded 1024
            # the widening was skipped entirely and the kernel kept the SMALLER tile's
            # block size -- silently truncating every wider tensor (a tl.sum over a
            # 2048-element load returned 16). Sizes above 1024 are already handled
            # below: `block_size > 1024` takes the wrapping / multipass path, which
            # dispatches 1024 threads and strides over _total_elements. Widening here is
            # what lets that path see the real extent.
            block_size = max_tensor_size

        # If total elements exceed the thread count, use a wrapping loop so
        # each thread processes multiple elements.
        self._needs_wrapping = False
        self._total_elements = block_size
        # Phase 4c: set True only when the prescan proves this kernel is
        # safe for single-pass MEPT (exact tile cover, no barriers, every
        # op in _MEPT_SAFE_OPS, no fp8). The MEPT producers (make_range)
        # gate on this flag, so it controls the whole MEPT chain.
        self._mept_single_pass = False

        # Determine optimal thread count from TTGIR layout.
        # When sizePerThread > 1, Triton expects fewer threads each handling
        # multiple elements. Use num_warps * warp_size as the thread count
        # and emit a per-thread loop for the extra elements.
        size_per_thread = 1
        if self.graph.size_per_thread:
            for s in self.graph.size_per_thread:
                size_per_thread *= s

        # Scan for reduces/barriers recursively (ops may be inside scf.for body)
        def _scan_all_ops(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _scan_all_ops(s.region_ops)
                if s.else_ops:
                    yield from _scan_all_ops(s.else_ops)

        all_ops_iter = list(_scan_all_ops(self.graph.ops))
        # Identify cooperative staging once, before choosing the physical dispatch.
        # ``_staging_op`` is retained for the fail-closed error's op_name; the narrow
        # dot predicate below decides whether every >1024 value path is actually wired
        # for multi-element-per-thread loops.
        _staging_op = next(
            (
                ssa.op
                for ssa in all_ops_iter
                if ssa.op
                in (
                    "tt.dot",
                    "ttg.local_alloc",
                    "tt.trans",
                    "tt.gather",
                    "tt.cat",
                    "tt.join",
                    "tt.split",
                    "tt.scan",
                    "tt.histogram",
                )
            ),
            None,
        )
        _staging_ops = _staging_op is not None
        _dot_cooperative_mept = self._dot_cooperative_mept_eligible() or getattr(self, "_source_epilogue_replay", False)
        has_reduce_ops = any(ssa.op == "tt.reduce" for ssa in all_ops_iter)
        has_barrier_ops = any(
            ssa.op
            in (
                "tt.reduce",
                "tt.scan",
                "tt.debug_barrier",
                "ttg.barrier",
                "tt.trans",
                "tt.dot",
                "ttg.local_alloc",
                "tt.gather",
                "tt.cat",
                "tt.join",
                "tt.split",
            )
            for ssa in all_ops_iter
        )
        # Cooperative shared-memory-staging ops (one element per thread). When a
        # kernel ALSO contains a tt.reduce that forces the multipass path (which
        # dispatches only num_threads / 1024 threads), the staging tail is
        # under-computed — a reduce+scan re-accumulated the partial, histogram
        # re-counted each bin num_warps times (re-audit #5 silent-wrong). These two
        # dispatch strategies are mutually exclusive, so the combination refuses.
        _coop_staging_ops = any(
            ssa.op in ("tt.scan", "tt.trans", "tt.gather", "tt.cat", "tt.join", "tt.split", "tt.histogram")
            for ssa in all_ops_iter
        )
        # Multi-value reduces (argmin/argmax) need per-element indices which
        # are incompatible with the multi-pass accumulation loop (the loop
        # variable goes out of scope before the reduce handler runs).
        has_multivalue_reduce = any(
            ssa.op == "tt.reduce" and ssa.result_ids and len(ssa.result_ids) >= 2 for ssa in all_ops_iter
        )
        num_threads = self.graph.num_warps * 32

        if self._assert_plan is not None and block_size <= 1024:
            # A rendezvous is not movable across multipass reduction phases.
            # Use the existing one-logical-element-per-Metal-thread form when
            # the complete tile fits, rather than splitting source assertions
            # between passes or dropping them from a template.
            num_threads = block_size

        # Phase 4c: is this kernel safe for single-pass MEPT? Default-deny —
        # every op must be array-wired and no fp8 dtype may appear (fp8
        # truncf emits a scalar conversion chain). Computed here; combined
        # with the exact-cover + no-barrier checks at the wrapping decision.
        from triton_msl.codegen.msl_builtins import is_fp8_type as _is_fp8

        def _op_is_fp8(s):
            if not s.elem_type:
                return False
            return _is_fp8(_mlir_to_triton_dtype(s.elem_type))

        # Phase 4d: ttg.convert_layout is MEPT-safe when both its source and
        # destination layouts resolve to LinearLayouts of the same total size
        # — the shuffle (_lower_convert_layout_mept_shuffle) can then
        # redistribute the register array. Multi-element converts get the
        # shuffle; scalar (1-elem) converts stay a passthrough (no env_array).
        # Gating eligibility on resolution guarantees the shuffle never sees
        # an array it can't place.
        def _convert_resolves(s):
            if not s.operand_ids:
                return False
            src_t = self._find_op_type_str(s.operand_ids[0]) or ""
            src_sh = self._native_shape(s.operand_ids[0], op_name=s.op)
            dst_sh = self._native_shape(s.id, op_name=s.op)
            if not src_sh or not dst_sh:
                return False
            sll = self._resolve_linear_layout(src_t, src_sh)
            dll = self._resolve_linear_layout(s.type_str or "", dst_sh)
            return sll is not None and dll is not None and sll.total_elements == dll.total_elements

        def _op_mept_ok(s):
            if s.op in _MEPT_SAFE_OPS:
                return True
            if s.op == "ttg.convert_layout":
                return _convert_resolves(s)
            return False

        mept_kernel_safe = (
            self.mept_enabled
            and all(_op_mept_ok(s) for s in all_ops_iter)
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )

        # Phase 4e: is this kernel eligible for single-pass MEPT *with* a
        # 1-D full reduce? The reduce operand (a per-thread register array)
        # is folded to a scalar partial, then the existing cross-thread
        # SIMD/threadgroup reduce finishes. Only safe when:
        #   - every TOP-LEVEL op is array-wired or tt.reduce (reduce *body*
        #     ops are the combiner, parsed separately, not lowered as array
        #     ops — so they are not checked here),
        #   - every reduce is a single-result 1-D full reduce (multi-dim /
        #     axis / argmin-max reduces need the unimplemented multi-dim
        #     fold and stay on the multipass path),
        #   - no fp8.
        _top_ops = self.graph.ops

        def _reduce_is_1d_full(r):
            if r.result_ids and len(r.result_ids) >= 2:
                return False  # multi-value (argmin/argmax/Welford)
            if not r.operand_ids:
                return False
            return len(self._native_shape(r.operand_ids[0], op_name=r.op)) == 1

        mept_reduce_eligible = (
            self.mept_enabled
            and any(s.op == "tt.reduce" for s in _top_ops)
            and all(_op_mept_ok(s) or s.op == "tt.reduce" for s in _top_ops)
            and all(_reduce_is_1d_full(s) for s in _top_ops if s.op == "tt.reduce")
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )

        # Packet 325: a wide scan uses one exact flattened ownership map for
        # its complete producer/consumer cone.  This is intentionally a
        # separate, default-deny proof rather than adding shape operations to
        # the general MEPT allowlist: expand/broadcast are only array-safe here
        # because every make_range is projected from the same flat element
        # index (row=e//N, col=e%N) by _lower_make_range.
        _scan_ops = [s for s in _top_ops if s.op == "tt.scan"]
        _scan_shape = None
        self._scan_native_contracts = {}
        self._scan_native_errors = {}
        _scan_top_by_result = {}
        for _s in _top_ops:
            _scan_top_by_result[_s.id] = _s
            for _rid in _s.result_ids or []:
                _scan_top_by_result[_rid] = _s

        def _scan_region_is_exact(sc):
            nonlocal _scan_shape
            try:
                contract = self._prove_scan_native_contract(sc)
            except MetalNonRecoverableError as exc:
                reason = str(exc).splitlines()[0]
                self._scan_native_errors[sc.id] = reason.removeprefix("Refusing to emit silently-wrong output: ")
                return False
            self._scan_native_contracts[sc.id] = contract
            _shape = contract["shape"]
            if _scan_shape is None:
                _scan_shape = tuple(_shape)
            elif tuple(_shape) != _scan_shape:
                return False
            _n = len(sc.operand_ids)
            _args = list((sc.attrs or {}).get("block_arg_ids") or [])
            _returns = [b for b in (sc.region_ops or []) if b.op == "tt.scan.return"]
            if len(_args) != 2 * _n or len(_returns) != 1:
                return False
            if len(_returns[0].operand_ids or []) != _n:
                return False
            return all(b.op == "tt.scan.return" or b.op in _MEPT_SAFE_OPS for b in (sc.region_ops or []))

        _scan_arg_ids = {a.id for a in self.graph.args}

        def _scan_value_proof(vid, seen=None):
            """Return ``coordinate`` / ``uniform`` or None for a whole value DAG.

            It is not enough for *one* operand path to reach a proved range: an
            elementwise value may combine that path with a second tensor whose
            ownership is unknown.  Every sibling must therefore prove either the
            same flattened coordinate ownership or true scalar uniformity.
            """
            if seen is None:
                seen = set()
            if vid in seen:
                return None
            seen.add(vid)
            try:
                _facts = self._scan_native_value_facts(vid)
            except MetalNonRecoverableError:
                return None
            if _facts.elem and _is_fp8(_mlir_to_triton_dtype(_facts.elem)):
                return None
            _op = _scan_top_by_result.get(vid)
            if _op is None:
                # Kernel arguments are scalar values or scalar pointer bases.
                # Tensor uses of them must still pass through a proved splat,
                # addptr or load before becoming a scan operand.
                return "uniform" if vid in _scan_arg_ids and not _facts.is_tensor and _facts.shape == () else None
            if _op.op == "tt.make_range":
                return "coordinate"
            if _op.op == "tt.scan":
                return "coordinate"
            if _op.op == "arith.constant":
                # MLIR's ``dense<scalar>`` is a splat, but ``dense<[...]>`` is
                # not.  The walker leaves a non-splat payload as a string; do
                # not silently replay it as one scalar in every array slot.
                return "uniform" if isinstance((_op.attrs or {}).get("value"), (bool, int, float)) else None
            if _op.op in ("tt.get_program_id", "tt.get_num_programs"):
                return "uniform"
            if _op.op == "tt.splat":
                if len(_op.operand_ids or []) != 1:
                    return None
                return "uniform" if _scan_value_proof(_op.operand_ids[0], set(seen)) == "uniform" else None
            if _op.op == "ttg.convert_layout":
                if len(_op.operand_ids or []) != 1:
                    return None
                if not (_convert_resolves(_op) or _scan_convert_is_flat_noop(_op)):
                    return None
                return _scan_value_proof(_op.operand_ids[0], set(seen))
            if _op.op in _MEPT_SAFE_OPS or _op.op in ("tt.expand_dims", "tt.broadcast"):
                if not _op.operand_ids:
                    return None
                proofs = [_scan_value_proof(o, set(seen)) for o in _op.operand_ids]
                if any(p is None for p in proofs):
                    return None
                return "coordinate" if "coordinate" in proofs else "uniform"
            return None

        def _scan_convert_is_flat_noop(op):
            if len(op.operand_ids or []) != 1:
                return False
            if _scan_shape is None:
                return False
            try:
                source_shape = self._scan_native_value_facts(op.operand_ids[0]).shape
                dest_shape = self._scan_native_value_facts(op.id).shape
            except MetalNonRecoverableError:
                return False
            # Packet 325 replaces every layout *inside the proved scan
            # value cone* with one canonical flat ownership projection.
            # A full-tile layout conversion is consequently a value
            # identity whether it appears before or after the scan.  Both
            # ends must name the exact proved tile: admitting a partial or
            # reshaped conversion would silently invent element ownership.
            return bool(
                source_shape
                and dest_shape
                and tuple(source_shape) == tuple(_scan_shape)
                and tuple(dest_shape) == tuple(_scan_shape)
            )

        _scan_allowed_top = _MEPT_SAFE_OPS | {"tt.expand_dims", "tt.broadcast", "tt.scan", "ttg.convert_layout"}
        # Establish the common scan shape before asking whether layout
        # conversions are identities with respect to that shape.  Doing this
        # inside the later boolean expression made the answer depend on
        # evaluation order: a conversion was queried while ``_scan_shape``
        # was still None, over-refusing the four-warp bf16 layouts while the
        # independently-resolvable sixteen-warp spellings happened to pass.
        _scan_regions_exact = bool(_scan_ops) and all(_scan_region_is_exact(s) for s in _scan_ops)
        mept_scan_eligible = (
            self.mept_enabled
            and bool(_scan_ops)
            and not any(s.op == "tt.reduce" for s in all_ops_iter)
            and all(s.op in _scan_allowed_top for s in _top_ops)
            and all(
                s.op != "ttg.convert_layout" or _convert_resolves(s) or _scan_convert_is_flat_noop(s) for s in _top_ops
            )
            and _scan_regions_exact
            and all(_scan_value_proof(o) in ("coordinate", "uniform") for s in _scan_ops for o in (s.operand_ids or []))
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )

        # Phase 4f (MEPT M2): control-flow kernels that carry a multi-element
        # value across a data-dependent scf.for/while/if. The re-execution
        # wrap-loop cannot carry per-element state across the control-flow
        # boundary, so values hoisted before the loop (a tl.arange register
        # array, a masked-load `other=` constant) used to be unresolved inside
        # it (tridec Bug 2, BLOCK>=256).
        # Register arrays declared once before the loop persist into the body
        # naturally (env_array is instance state), so the existing array-wired
        # body handlers resolve them. Eligible iff: MEPT on; a control-flow op
        # is present; the region references/carries a multi-element value;
        # every op is array-wired (or control-flow / reduce / yield /
        # condition); every reduce is a 1-D full reduce; the tile cover is
        # exact; no fp8. See
        # docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md
        from triton_msl.codegen.regval import (
            region_needs_arrays as _region_needs_arrays,
            tensor_value_ids as _tensor_value_ids,
            _CONTROL_OPS as _CF_OPS,
        )

        def _reshape_preserves_elems(s):
            # A rank- and element-count-preserving tt.reshape is a layout-only
            # change (e.g. the sizePerThread=[2]→[1] reshape Triton inserts
            # before a tt.reduce of a tl.sum, which is rank-1→rank-1). It lowers
            # via _emit_passthrough, which propagates env_array unchanged, so the
            # per-thread register array survives into the reduce. We require both
            # the same total element count AND the same number of dimensions:
            # a (N,)→(a,b) reshape keeps the element count but changes the
            # rank/layout, which is a real reinterpretation, so it is rejected.
            # This keeps the helper self-contained — it does not lean on a
            # downstream gate (_reduce_is_1d_full) to reject rank-changing
            # reshapes. Stays conservative-False on any unresolvable shape.
            if not s.operand_ids:
                return False
            in_sh = self._native_shape(s.operand_ids[0], op_name=s.op)
            out_sh = self._native_shape(s.id, op_name=s.op)
            if not in_sh or not out_sh:
                return False
            tin = 1
            for d in in_sh:
                tin *= d
            tout = 1
            for d in out_sh:
                tout *= d
            return tin == tout and len(in_sh) == len(out_sh)

        def _arrayform_op_ok(s):
            if s.op in _CF_OPS:
                body = list(s.region_ops or []) + list(getattr(s, "else_ops", None) or [])
                return all(_arrayform_op_ok(b) for b in body)
            if s.op in ("scf.yield", "scf.condition", "tt.reduce"):
                return True
            if s.op == "tt.reshape":
                return _reshape_preserves_elems(s)
            return _op_mept_ok(s)

        def _all_reduces(op_list):
            for s in op_list:
                if s.op == "tt.reduce":
                    yield s
                if s.region_ops:
                    yield from _all_reduces(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _all_reduces(s.else_ops)

        def _value_is_multi(s):
            for shp in self._native_shapes_for_op(s):
                if not shp:
                    continue
                tot = 1
                for d in shp:
                    tot *= d
                if tot > num_threads:
                    return True
            return False

        mept_arrayform_eligible = (
            self.mept_enabled
            and any(s.op in _CF_OPS for s in _top_ops)
            and _region_needs_arrays(_top_ops, _tensor_value_ids(_top_ops, _value_is_multi))
            and all(_arrayform_op_ok(s) for s in _top_ops)
            and all(_reduce_is_1d_full(r) for r in _all_reduces(_top_ops))
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )

        # Detect if this 2D kernel has axis-specific reductions that produce
        # per-row/per-column results (not full-array reductions).
        # Multipass is incompatible with these because the per-thread
        # accumulator mixes values from different rows/columns.
        # Also covers N-D axis reductions (e.g. tl.sort reshapes to (2,)*n and
        # reduces along a specific axis per compare-and-swap step).
        has_2d_axis_reduce = False
        if self._is_2d and self._effective_2d_shape and has_reduce_ops:
            for ssa in all_ops_iter:
                if ssa.op == "tt.reduce" and ssa.operand_ids:
                    reduce_axis = ssa.attrs.get("axis", 0)
                    inp_shape = self._native_shape(ssa.operand_ids[0], op_name=ssa.op)
                    # A true axis-specific reduce: multi-dim input where more
                    # than one non-reduced axis has size > 1. For N-D, check
                    # whether the non-reduced axes together have > 1 element.
                    if inp_shape and len(inp_shape) >= 2:
                        other_size = 1
                        for i, s in enumerate(inp_shape):
                            if i != reduce_axis:
                                other_size *= s
                        if other_size > 1:
                            has_2d_axis_reduce = True

        # Decide wrapping strategy:
        # 1. sizePerThread > 1 with reductions → multi-pass reduction
        # 2. sizePerThread > 1 without barriers → simple wrapping loop
        # 3. block_size > 1024 with reductions → multi-pass reduction
        # 4. block_size > 1024 without reductions → simple wrapping loop (capped at 1024)
        #
        # EXCEPTION: 2D kernels with axis-specific reductions (dim_0 > 1)
        # cannot use multipass because the per-thread accumulator mixes
        # values across rows. For these, keep block_size = total (up to 1024)
        # so each thread handles exactly one element and _lower_reduce_2d
        # can correctly collect all values in shared memory.
        use_multipass = False
        # MEPT arrayform exact-cover count. The raw ``size_per_thread`` is the
        # product of the layout's first-level sizePerThread tile, which
        # UNDERCOUNTS when the blocked tile is smaller than the tensor and the
        # layout replicates (e.g. BLOCK=1024 has sizePerThread=[4],
        # threadsPerWarp=[32], warpsPerCTA=[4] → a 512-wide tile covering a
        # 1024 tensor, so each thread really holds 8 registers, not 4). The
        # MEPT array producer (_track_n_elems / num_registers_per_thread) uses
        # the true per-thread count, which for a contiguous 1-D cover equals
        # ``block_size // num_threads``. The arrayform exact-cover gate must use
        # the SAME count or it disagrees with the codegen (BLOCK=1024 was
        # rejected with size_per_thread=4 even though the array path emits 8).
        mept_elems_per_thread = block_size // num_threads if num_threads and block_size % num_threads == 0 else 0
        if (
            mept_scan_eligible
            and block_size > 1024
            and mept_elems_per_thread > 1
            and num_threads * mept_elems_per_thread == block_size
            and _scan_shape is not None
        ):
            # The array is the multiplicity; no wrapping loop.  Every array
            # slot owns flat element e=lid*width+slot and every 2-D coordinate
            # derives from that same e.  _lower_scan validates staging budget,
            # operand widths, slot types, block arguments and return arity.
            self._total_elements = block_size
            self._scan_mept_shape = _scan_shape
            self._scan_mept_width = mept_elems_per_thread
            block_size = num_threads
            self._mept_single_pass = True
        elif has_2d_axis_reduce and block_size <= 1024:
            # Skip multipass; use full block_size with one element per thread.
            # _lower_reduce_2d handles the sequential reduction internally.
            pass
        elif has_2d_axis_reduce and block_size > 1024:
            # Total 2D elements exceed 1024. Cap block_size to 1024.
            # The dot path uses strided loops for outputs > block_size.
            # The reduce path stores to shared[lid] which needs lid < total,
            # so check that reduce inputs fit within 1024.
            max_reduce_size = 0

            def _scan_reduce_sizes(ops):
                nonlocal max_reduce_size
                for op in ops:
                    if op.op == "tt.reduce" and op.operand_ids:
                        inp_shape = self._native_shape(op.operand_ids[0], op_name=op.op)
                        if inp_shape:
                            rs = 1
                            for d in inp_shape:
                                rs *= d
                            max_reduce_size = max(max_reduce_size, rs)
                    if op.region_ops:
                        _scan_reduce_sizes(op.region_ops)

            _scan_reduce_sizes(self.graph.ops)
            if max_reduce_size <= 1024:
                block_size = max(max_reduce_size, 1024)
                block_size = min(block_size, 1024)
            else:
                self._flash_too_large = True
        elif has_multivalue_reduce and block_size <= 1024:
            # Multi-value reduces (argmin/argmax) need per-element indices that
            # go out of scope in the multi-pass accumulation loop. Use full
            # block_size with one element per thread.
            pass
        elif size_per_thread > 1 and block_size > num_threads:
            if (
                mept_arrayform_eligible
                and mept_elems_per_thread > 1
                and num_threads * mept_elems_per_thread == block_size
            ):
                # MEPT M2 single-pass register-array form for control-flow
                # kernels. Each thread owns mept_elems_per_thread contiguous
                # elements as a register array (idx[i] = lid*N + i). The array
                # IS the per-thread multiplicity, so there is NO wrap-loop
                # (_needs_wrapping stays False). Arrays declared before a
                # data-dependent scf.for persist into its body, so hoisted
                # values (arange, masked-load `other`) resolve inside the loop.
                # Uses mept_elems_per_thread (= block_size // num_threads), the
                # replication-aware count that matches the array producer; the
                # raw size_per_thread undercounts on replicated layouts.
                self._total_elements = block_size
                block_size = num_threads
                self._mept_single_pass = True
            elif mept_reduce_eligible and num_threads * size_per_thread == block_size:
                # Phase 4e MEPT-reduce single-pass: each thread loads its
                # ``size_per_thread`` elements as a register array, folds
                # them to a scalar partial (_mept_reduce_fold), then the
                # existing cross-thread reduce finishes. Replaces the
                # multipass wrap-loop for eligible 1-D full reduces. The
                # default flag-off path still uses multipass below.
                self._total_elements = block_size
                block_size = num_threads
                self._mept_single_pass = True
            elif has_reduce_ops:
                if _coop_staging_ops:
                    raise MetalNonRecoverableError(
                        "Kernel combines a reduce with a cooperative shared-memory "
                        "op (scan/transpose/gather/cat/join/split/histogram) over a "
                        "tile wider than the threadgroup: the multipass reduce "
                        "dispatches only num_threads threads while the cooperative op "
                        "stages one element per thread, so the staging tail is "
                        "under-computed (silent-wrong). Split into separate kernels "
                        "or use BLOCK <= num_threads.",
                        op_name="tt.reduce",
                    )
                use_multipass = True
                self._total_elements = block_size
                block_size = num_threads
            elif _dot_cooperative_mept and block_size > 1024:
                # Stage-1B generic dot: the logical output tile is wider than Metal's
                # physical threadgroup, but every load/local_alloc/dot/store leg above
                # has a cooperative strided loop.  Set the PHYSICAL dispatch here; the
                # logical extent stays in _total_elements and is never used as a loop
                # stride. Value-changing epilogues do not satisfy the predicate and
                # continue to hit the fail-closed backstop below.
                self._total_elements = block_size
                block_size = self._cooperative_dispatch_threads(block_size)
            elif mept_kernel_safe and not has_barrier_ops and num_threads * size_per_thread == block_size:
                # Phase 4c MEPT single-pass: each of ``num_threads`` threads
                # owns ``size_per_thread`` contiguous elements via a register
                # array (idx[i] = lid*N + i). One pass covers the whole tile,
                # so NO wrap-loop — the array IS the per-thread multiplicity.
                # Enabling _needs_wrapping here would double-count: the
                # wrap-loop strides _loop_e over threads AND the array
                # multiplies by N, producing N x out-of-bounds indices.
                # ``mept_kernel_safe`` guarantees every op in the kernel is
                # array-wired, so the whole MEPT chain is correct; anything
                # else falls through to the scalar wrapping path below.
                self._total_elements = block_size
                block_size = num_threads
                self._mept_single_pass = True
                # _needs_wrapping intentionally stays False.
            elif not has_barrier_ops:
                self._needs_wrapping = True
                self._total_elements = block_size
                block_size = num_threads
        elif block_size > 1024:
            if _dot_cooperative_mept:
                self._total_elements = block_size
                block_size = self._cooperative_dispatch_threads(block_size)
            elif has_reduce_ops:
                if _coop_staging_ops:
                    raise MetalNonRecoverableError(
                        "Kernel combines a reduce with a cooperative shared-memory "
                        "op (scan/transpose/gather/cat/join/split/histogram) over a "
                        ">1024 tile: the multipass reduce and one-element-per-thread "
                        "staging are mutually exclusive, so the staging tail is "
                        "under-computed (silent-wrong). Split into separate kernels.",
                        op_name="tt.reduce",
                    )
                use_multipass = True
                self._total_elements = block_size
                block_size = 1024
            else:
                self._needs_wrapping = True
                self._total_elements = block_size
                block_size = 1024  # Cap dispatch to Metal max

        # From this point onward ``actual_dispatch_threads`` is the physical Metal
        # threadgroup width.  Logical coverage remains in ``_total_elements``.  Generic
        # cooperative emitters must use the former as their loop stride.
        actual_dispatch_threads = block_size
        self._actual_dispatch_threads = actual_dispatch_threads
        self.effective_block_size = actual_dispatch_threads

        if (
            self._assert_plan is not None
            and not self._is_2d
            and any(
                op.op == "tt.make_range" and op.attrs.get("end", 0) - op.attrs.get("start", 0) != self._total_elements
                for op in all_ops_iter
            )
        ):
            from .assertions import refuse

            refuse("mixed 1-D tile widths lack a predicate ownership proof")

        if self._assert_plan is not None and (
            use_multipass
            or self._mept_single_pass
            or (self._needs_wrapping and self._total_elements % actual_dispatch_threads != 0)
        ):
            from .assertions import refuse

            refuse("execution multiplicity does not prove uniform assertion rendezvous")

        # COVERAGE BACKSTOP -- should be unreachable, and is kept because the thing it
        # guards already regressed once.
        #
        # The real fix for mixed tile widths is the max-tensor widening above: block_size
        # is set to the WIDEST tensor, so nothing is left uncovered. That widening used
        # to be capped at 1024, and above the cap it was skipped entirely -- the kernel
        # then kept the narrower tile's size and silently truncated every wider tensor
        # (`tl.sum` over a 2048-element load returned 16). This check makes a
        # reintroduction of that class loud instead of silent: if any tensor is wider
        # than the threadgroup with no mechanism covering it, refuse.
        #
        # Only the plain scalar path is checked. The wrapping, MEPT and multipass paths
        # each carry their own multiplicity. Cooperative staging (tt.dot operands,
        # scan/trans/gather/...) covers wider tensors through strided loops only while
        # its emitted stride equals the actual dispatch; the explicit >1024 check below
        # enforces that precondition. A bare tt.reduce does NOT stage, so a reduce wider
        # than the threadgroup with no wrap loop is exactly the truncation this catches.
        # Cooperative staging covers wide tiles only when its emitted loop stride
        # equals the number of threads Metal actually dispatches.  Above 1024 the
        # driver clamps the dispatch, but this unwrapped path still emits
        # ``block_size`` as the stride.  GPU bite at 2048 elements: indices
        # 1024..2047 remained NaN (exactly half the output), with no exception.
        if _staging_ops and actual_dispatch_threads > 1024:
            raise MetalNonRecoverableError(
                f"kernel requires cooperative shared-memory staging for {actual_dispatch_threads} "
                "elements, exceeding Metal's 1024-thread threadgroup cap: the "
                f"emitted loops would stride by {actual_dispatch_threads} while only 1024 threads "
                "run, leaving the tail unwritten. Refusing (correct-or-refuse).",
                op_name=_staging_op,
            )
        if (
            not self._needs_wrapping
            and not getattr(self, "_mept_single_pass", False)
            and not use_multipass
            and not _staging_ops
        ):
            _widest, _widest_op = 0, None
            for _ssa in all_ops_iter:
                for _sh in self._native_shapes_for_op(_ssa):
                    if not _sh:
                        continue
                    _t = 1
                    for _d in _sh:
                        _t *= _d
                    if _t > _widest:
                        _widest, _widest_op = _t, _ssa.op
            if _widest > actual_dispatch_threads:
                raise MetalNonRecoverableError(
                    f"kernel mixes tile widths: a {_widest}-element tensor (from "
                    f"{_widest_op}) would be processed by only {actual_dispatch_threads} threads with "
                    "no wrapping loop, silently truncating it to the first "
                    f"{actual_dispatch_threads} elements. Refusing (correct-or-refuse). Use a single "
                    "tile width, or split into separate kernels.",
                    op_name=_widest_op or "tt.load",
                )

        # If the kernel is too large for the generic lowerer (cooperative ops
        # with > 1024 total elements), emit a minimal kernel with UNSUPPORTED
        # so the legacy parser can handle it via prebuilt templates.
        if getattr(self, "_flash_too_large", False):
            self.kb = KernelBuilder(self.graph.func_name, block_size=actual_dispatch_threads)
            self._register_args()
            self.kb.unsupported("UNSUPPORTED: 2D kernel with cooperative ops exceeds 1024 elements")
            return self.kb.build()

        self.kb = KernelBuilder(self.graph.func_name, block_size=actual_dispatch_threads)

        # Generate device functions for noinline callees (must appear before kernel)
        if self.graph.called_funcs:
            self._lower_called_funcs()

        # Pre-scan stores to identify output pointers
        self._prescan_stores()

        # Register function arguments
        self._register_args()

        if self._assert_plan is not None:
            self.kb.add_ptr_arg("_assert_status", dtype="u32")
            self.kb.raw_line("    threadgroup atomic_uint _assert_group_failed;")
            self.kb.raw_line(
                "    if (lid == 0u) atomic_store_explicit(&_assert_group_failed, 0u, memory_order_relaxed);"
            )
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        if use_multipass:
            # Multi-pass reduction: split kernel into phases separated by
            # reductions, wrap each phase in a per-element loop, emit
            # reductions between loops operating on thread-local accumulators.
            self._lower_multipass_reduction(actual_dispatch_threads)
        else:
            # Standard path: single wrapping loop or no loop
            if self._needs_wrapping:
                self.kb.raw_line(
                    f"    for (uint _loop_e = lid; _loop_e < {self._total_elements}u; "
                    f"_loop_e += {actual_dispatch_threads}u) {{"
                )

            # Lower each op
            for ssa in self.graph.ops:
                self._lower_op(ssa)

            # Close wrapping loop
            if self._needs_wrapping:
                self.kb.raw_line(f"    }}")

        # Propagate flags to KernelBuilder for MSL emission
        if self._needs_num_programs:
            self.kb._needs_num_programs = True
        if self._used_pid_axes:
            self.kb._used_pid_axes = self._used_pid_axes

        msl = self.kb.build()
        allocation_aliases = {}
        msl = _alias_shared_memory(msl, allocation_aliases=allocation_aliases)

        # Safety net: the generic op-by-op path must never silently emit a
        # kernel with an *empty body* when the graph clearly has a
        # tt.store/tt.atomic_* — that compiles but produces zeros (exactly
        # how the unhandled N-D permute+reduce pattern manifested before its
        # template existed: ``kernel void k(...) { }``). Marking it
        # UNSUPPORTED makes ``emit_msl`` fall back to the legacy parser
        # instead of returning silently-wrong output. Templates return
        # earlier, so this only guards the generic path.
        #
        # The check is intentionally narrow — *no emitted statements at all*
        # — rather than "no recognized write": atomic kernels write via
        # ``atomic_compare_exchange_weak_explicit`` / ``atomic_load_explicit``
        # CAS loops (no ``[idx] =`` and no ``atomic_store``/``_fetch``), and
        # an over-broad heuristic wrongly forced them to the legacy parser
        # (regressed 106 atomic tests). A truly empty body has zero
        # statement lines; any real kernel has many.
        graph_has_store = any(op.op in ("tt.store", "tt.atomic_rmw", "tt.atomic_cas") for op in self.graph.ops)
        body_lines = getattr(self.kb, "_body_lines", [])
        body_has_stmt = any(";" in ln for ln in body_lines)
        if graph_has_store and not body_has_stmt:
            self.kb.unsupported(
                "UNSUPPORTED: generic lowering produced an empty body for a "
                "kernel that stores (e.g. an unhandled N-D permute/reduce) — "
                "falling back to the legacy parser."
            )
            msl = self.kb.build()
            msl = _alias_shared_memory(msl, allocation_aliases=allocation_aliases)
        if getattr(self, "_source_epilogue_replay", False) or getattr(self, "_source_biased_replay", False):
            from ._lowerer_helpers import _check_replay_shared_memory_budget

            # Every source op/proof has already lowered. Capacity must not hide
            # an integrity failure. The bounded replay families use builder
            # arrays; the assertion channel, if present, adds one atomic uint.
            self._replay_shared_bytes = _check_replay_shared_memory_budget(
                self.kb._threadgroup_arrays,
                allocation_aliases,
                extra_bytes=4 if self._assert_plan is not None else 0,
            )
        if self.kb._argument_abi is not None and self._assert_plan is None:
            from triton_msl.backend._argument_abi import producer_record

            # Generic emission's shared-memory alias pass can rewrite the text.
            # Certify the final source, not the builder's pre-alias intermediate.
            self._argument_abi = producer_record(self.kb.args, "generic-msl-v1", msl)
        return msl

    def get_output_arg_indices(self):
        """Return list of arg positions that are output (stored-to) pointers.

        Must be called after lower(). Returns None if _prescan_stores()
        was not called (e.g., matmul template path), which means the
        driver should conservatively copy back all tensors.
        """
        if not hasattr(self, "_output_arg_ids") or not self._output_arg_ids:
            return None
        indices = []
        for i, arg in enumerate(self.graph.args):
            if arg.id in self._output_arg_ids:
                indices.append(i)
        return indices

    @staticmethod
    def _cooperative_dispatch_threads(logical_elements: int) -> int:
        """Physical Metal threads for a cooperatively strided logical tile.

        ``logical_elements`` is the number of tensor elements the emitted loops must
        cover.  The return value is the ACTUAL threadgroup width and therefore the only
        legal stride for those loops.  Keeping this conversion named and centralized
        prevents the pre-1B defect where both quantities were called ``block_size`` and
        a 2048/4096-element loop stepped by its logical size even though Metal launched
        only 1024 threads.
        """
        return min(max(1, int(logical_elements)), 1024)

    def _dot_cooperative_mept_eligible(self) -> bool:
        """Whether the >1024 generic-dot graph is fully wired for strided staging.

        Stage 1B deliberately admits only ONE bare 2-D dot whose two operands come from
        load→local_alloc→local_load and whose result reaches exactly one store through
        representation-only casts/layout conversions.  Value-changing epilogues,
        non-zero per-element accumulator inits, chain dots, reductions, control flow,
        and other cooperative operations remain behind the 1024-thread refusal until
        their own per-element semantics are wired.  This makes the dispatch repair a
        narrow capability proof rather than a wholesale guard deletion.
        """

        # Stage 1C's whole-graph proof owns its accumulator/chain value paths
        # and K-sliced storage.  Consult it before the deliberately narrower
        # stage-1B bare-dot proof below.
        if self._dot_kchunk_plan() is not None:
            return True

        def _all_ops(ops):
            for op in ops:
                yield op
                if op.region_ops:
                    yield from _all_ops(op.region_ops)
                if op.else_ops:
                    yield from _all_ops(op.else_ops)

        ops = list(_all_ops(self.graph.ops))
        dots = [op for op in ops if op.op == "tt.dot"]
        if len(dots) != 1:
            return False
        dot = dots[0]
        if len(dot.operand_ids or []) < 3:
            return False

        dot_shape = _extract_shape(dot.type_str or self._find_op_type_str(dot.id) or "")
        if not dot_shape or len(dot_shape) != 2:
            return False
        dot_total = dot_shape[0] * dot_shape[1]
        if dot_total <= 1024:
            return False

        forbidden = {
            "scf.for",
            "scf.while",
            "scf.if",
            "tt.reduce",
            "tt.scan",
            "tt.trans",
            "tt.gather",
            "tt.cat",
            "tt.join",
            "tt.split",
            "tt.histogram",
        }
        if any(op.op in forbidden for op in ops):
            return False

        op_by_id = {op.id: op for op in ops}

        # This path reconstructs the full result tile in shared memory.  Accept only a
        # literal zero accumulator, which is exactly the bare-matmul contract.  The
        # broader detector helper intentionally treats UNKNOWN as "not a known bias";
        # using that negative result here would be an unsafe proof and could admit an
        # unrecognized per-element initializer that the cooperative loop cannot replay.
        acc_init = op_by_id.get(dot.operand_ids[2])
        if acc_init is None or acc_init.op != "arith.constant":
            return False
        try:
            if float(acc_init.attrs.get("value")) != 0.0:
                return False
        except (TypeError, ValueError):
            return False

        def _is_staged_load(value_id):
            local_load = op_by_id.get(value_id)
            if local_load is None or local_load.op != "ttg.local_load" or not local_load.operand_ids:
                return False
            source = op_by_id.get(local_load.operand_ids[0])
            if source is not None and source.op == "ttg.memdesc_trans" and source.operand_ids:
                source = op_by_id.get(source.operand_ids[0])
            if source is None or source.op != "ttg.local_alloc" or not source.operand_ids:
                return False
            load = op_by_id.get(source.operand_ids[0])
            return load is not None and load.op == "tt.load"

        if not _is_staged_load(dot.operand_ids[0]) or not _is_staged_load(dot.operand_ids[1]):
            return False

        if sum(1 for op in ops if op.op == "ttg.local_alloc") != 2:
            return False
        if sum(1 for op in ops if op.op == "tt.store") != 1:
            return False

        output_passthrough = {
            "ttg.convert_layout",
            "arith.truncf",
            "arith.extf",
            "tt.fp_to_fp",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "arith.trunci",
            "arith.extsi",
            "arith.extui",
        }
        seen = set()
        frontier = [dot.id]
        terminal_stores = set()
        while frontier:
            value_id = frontier.pop()
            if value_id in seen:
                continue
            seen.add(value_id)
            consumers = [op for op in ops if value_id in (op.operand_ids or [])]
            if not consumers:
                return False
            for consumer in consumers:
                if consumer.op == "tt.store":
                    if len(consumer.operand_ids) < 2 or consumer.operand_ids[1] != value_id:
                        return False
                    terminal_stores.add(consumer.id)
                    continue
                if (
                    consumer.op not in output_passthrough
                    or not consumer.operand_ids
                    or consumer.operand_ids[0] != value_id
                ):
                    return False
                frontier.append(consumer.id)

        return len(terminal_stores) == 1

    def _dot_has_reshape_provenance(self) -> bool:
        """True iff any tt.dot operand is staged from a ``tt.reshape`` (the
        ``load(1-D) -> reshape -> local_alloc -> [memdesc_trans] -> local_load``
        chain upstream test_dot_multidim emits). Such operands carry no 2-D
        addptr arithmetic, so the template stride tracer can never resolve them;
        detectors consult this together with ``_dot_generic_eligible`` to decline
        to the generic path instead of refusing (dot-recovery 2B, 2026-09-01)."""
        by_id = {}

        def _index(ops):
            for _o in ops:
                by_id[_o.id] = _o
                if _o.region_ops:
                    _index(_o.region_ops)

        _index(self.graph.ops)

        def _reshape_fed(vid):
            cur = vid
            for _ in range(6):
                _o = by_id.get(cur)
                if _o is None or not _o.operand_ids:
                    return False
                if _o.op == "tt.reshape":
                    return True
                if _o.op in ("ttg.local_alloc", "ttg.local_load", "ttg.memdesc_trans", "ttg.convert_layout"):
                    cur = _o.operand_ids[0]
                    continue
                return False
            return False

        return any(_reshape_fed(opnd) for d in by_id.values() if d.op == "tt.dot" for opnd in (d.operand_ids or [])[:2])

    def _dot_generic_eligible(self, *, allow_flat: bool = False) -> bool:
        """True iff every tt.dot sits inside the PROVEN envelope of the generic
        per-thread dot path (dot-recovery stage 1a, 2026-08-30).

        The matmul refusal sites consult this to FALL THROUGH to the generic
        lowerer instead of refusing — recovering fused-accumulator adds
        (add-matrix/rows/cols), compute epilogues (scale/relu/softmax
        spellings the fused templates don't claim), and chain-dot. Envelope
        (every bound below is probe-verified; outside it the sites keep
        refusing):

        - NO structured control flow (single-tile, non-looped kernels only —
          the historical fuzzer runs that found generic matmul+epilogue
          "grossly wrong" used K-loops);
        - the stage-1A scalar path requires every dot to be 2-D with
          M == N == K in {16, 32}: the 2026-08-30
          shapes probe confirmed the generic mapping silently mis-tiles
          NON-UNIFORM shapes (M32xN64 err 0.2..4.0);
        - an oversized dot is eligible only when ``_dot_kchunk_plan`` proves
          the COMPLETE accumulator/chain graph and computes a KC=16 footprint
          at least 4 KiB below the 32 KiB threadgroup limit;
        - on the stage-1A scalar path, every tensor in the kernel is (S,S),
          (S,1), (1,S), (S,) or scalar for that single S — its generic
          per-thread mapping assumes one uniform 2-D tile shape per kernel.
        """
        # The K-chunk plan is already a stronger whole-graph proof than the
        # scalar shape checks below (including op/value-path and budget gates).
        if self._dot_kchunk_plan() is not None:
            return True

        S = None
        dots = []
        for o in self.graph.ops:
            if o.op in ("scf.for", "scf.while", "scf.if"):
                return False
            if o.op == "tt.dot":
                dots.append(o)
        if not dots:
            return False
        for d in dots:
            if len(d.operand_ids or []) < 2:
                return False
            a_t = self._find_op_type_str(d.operand_ids[0])
            b_t = self._find_op_type_str(d.operand_ids[1])
            a_s = _extract_shape(a_t) if a_t else None
            b_s = _extract_shape(b_t) if b_t else None
            if not a_s or not b_s or len(a_s) != 2 or len(b_s) != 2:
                return False
            _m, _k = a_s
            _k2, _n = b_s
            if not (_m == _k == _k2 == _n) or _m not in (16, 32):
                return False
            if S is None:
                S = _m
            elif S != _m:
                return False
        for o in self.graph.ops:
            if not o.type_str:
                continue
            sh = _extract_shape(o.type_str)
            if not sh:
                continue
            if len(sh) == 1:
                # ``allow_flat`` admits the FLAT form of the same tile: a contiguous
                # 1-D load reshaped to (S, S) for the dot, and the dot result
                # reshaped back to 1-D for the store (upstream test_dot_multidim
                # rank-2, dot-recovery 2B). Element count equals the tile's, so the
                # per-thread mapping the envelope guarantees is unchanged; at S=32
                # the flat tensor is exactly the 1024-thread cap the tile occupies.
                # OPT-IN ONLY (protocol rule 8, packet 063): the two reshape-
                # provenance routing sites pass True; every other caller of this
                # shared predicate keeps the default and is byte-identical to the
                # pre-2B envelope — a flat S*S tensor must not silently widen the
                # accumulator/epilogue/detector decisions that also consult it.
                _flat_ok = (S, 1, S * S) if allow_flat else (S, 1)
                if sh[0] not in _flat_ok:
                    return False
            elif len(sh) == 2:
                if tuple(sh) not in ((S, S), (S, 1), (1, S)):
                    return False
            else:
                return False
        return True

    def _masked_dot_generic_eligible(self) -> bool:
        """Whether a masked, single-tile dot may fall through to generic lowering.

        The inline and prebuilt matmul templates reconstruct raw pointer loads and do
        not replay ``tt.load`` masks.  The generic local-allocation path *does* rebuild
        each staged address, comparison, and scalar ``other`` structurally.  Route to
        it only for this complete, budgeted graph; mask details that cannot be rebuilt
        still refuse inside ``_rebuild_staged_fill_mask`` rather than being guessed.

        This is intentionally separate from ``_dot_generic_eligible``: the latter
        proves fused accumulator/epilogue and K-chunk forms, while this predicate proves
        a bare masked dot whose output and both input tiles each fit one threadgroup.
        """
        if self.graph is None:
            return False

        def _all_ops(ops):
            for op in ops:
                yield op
                if op.region_ops:
                    yield from _all_ops(op.region_ops)
                if op.else_ops:
                    yield from _all_ops(op.else_ops)

        ops = list(_all_ops(self.graph.ops))
        if any(op.region_ops or op.else_ops for op in ops):
            return False
        if any(op.op in ("scf.for", "scf.while", "scf.if", "tt.reduce", "tt.scan") for op in ops):
            return False

        dots = [op for op in ops if op.op == "tt.dot"]
        if len(dots) != 1:
            return False
        dot = dots[0]
        if len(dot.operand_ids or []) < 3:
            return False

        op_by_id = {op.id: op for op in ops}
        a_shape = _extract_shape(self._find_op_type_str(dot.operand_ids[0]) or "")
        b_shape = _extract_shape(self._find_op_type_str(dot.operand_ids[1]) or "")
        out_shape = _extract_shape(dot.type_str or self._find_op_type_str(dot.id) or "")
        if not a_shape or not b_shape or not out_shape:
            return False
        if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
            return False
        M, K = a_shape
        K2, N = b_shape
        if K != K2 or tuple(out_shape) != (M, N):
            return False
        if min(M, N, K) <= 0 or max(M, N, K) > 64:
            return False
        tile_counts = (M * K, K * N, M * N)
        if any(count > 1024 for count in tile_counts):
            return False
        # Generic local_alloc/result storage is fp32.  Keep 4 KiB headroom under
        # the project's 32 KiB threadgroup policy rather than admitting at the cap.
        if 4 * sum(tile_counts) > 28 * 1024:
            return False

        # The bare-dot path initializes once from a literal zero.  UNKNOWN is not
        # evidence of zero and must not enlarge the route.
        acc_init = op_by_id.get(dot.operand_ids[2])
        if acc_init is None or acc_init.op != "arith.constant":
            return False
        try:
            if float(acc_init.attrs.get("value")) != 0.0:
                return False
        except (TypeError, ValueError):
            return False

        def _masked_staged_load(value_id):
            local_load = op_by_id.get(value_id)
            if local_load is None or local_load.op != "ttg.local_load" or not local_load.operand_ids:
                return None
            source = op_by_id.get(local_load.operand_ids[0])
            if source is not None and source.op == "ttg.memdesc_trans" and source.operand_ids:
                source = op_by_id.get(source.operand_ids[0])
            if source is None or source.op != "ttg.local_alloc" or len(source.operand_ids or []) != 1:
                return None
            load = op_by_id.get(source.operand_ids[0])
            if load is None or load.op != "tt.load" or len(load.operand_ids or []) < 2:
                return None
            return load

        a_load = _masked_staged_load(dot.operand_ids[0])
        b_load = _masked_staged_load(dot.operand_ids[1])
        if a_load is None or b_load is None:
            return False
        if sum(1 for op in ops if op.op == "tt.load") != 2:
            return False
        if sum(1 for op in ops if op.op == "ttg.local_alloc") != 2:
            return False
        if sum(1 for op in ops if op.op == "tt.store") != 1:
            return False

        # The scalar generic emitter is validated for floating dot operands and an
        # fp32 dot result.  Storage may be narrowed by the terminal store.
        for value_id in dot.operand_ids[:2]:
            type_str = self._find_op_type_str(value_id) or ""
            if not re.search(r"x(?:bf16|f16|f32)(?:,|>)", type_str):
                return False
        if not re.search(r"xf32(?:,|>)", dot.type_str or ""):
            return False

        passthrough = {
            "ttg.convert_layout",
            "arith.truncf",
            "arith.extf",
            "tt.fp_to_fp",
        }
        seen = set()
        frontier = [dot.id]
        terminal_stores = set()
        while frontier:
            value_id = frontier.pop()
            if value_id in seen:
                continue
            seen.add(value_id)
            consumers = [op for op in ops if value_id in (op.operand_ids or [])]
            if not consumers:
                return False
            for consumer in consumers:
                if consumer.op == "tt.store":
                    if len(consumer.operand_ids or []) < 2 or consumer.operand_ids[1] != value_id:
                        return False
                    terminal_stores.add(consumer.id)
                    continue
                if consumer.op not in passthrough or not consumer.operand_ids or consumer.operand_ids[0] != value_id:
                    return False
                frontier.append(consumer.id)
        return len(terminal_stores) == 1

    def _requires_matmul_template(self) -> bool:
        """Check if the kernel is a pure matmul that needs the prebuilt template.

        Returns True only for PURE matmul kernels (dot + loads + stores).
        Returns False for complex kernels that have dot mixed with reductions,
        masking, or other ops (like flash attention) — these go through the
        generic op-by-op lowerer which handles tt.dot via _lower_dot.

        Note: _detect_simple_dot() is checked before this in lower() and
        handles simple dot patterns with an inline simdgroup MMA template.
        """
        has_dot = False
        has_reduce = False
        has_where = False

        def _scan_ops(ops):
            nonlocal has_dot, has_reduce, has_where
            for ssa in ops:
                if ssa.op == "tt.dot":
                    has_dot = True
                elif ssa.op == "tt.reduce":
                    has_reduce = True
                elif ssa.op == "arith.select":
                    has_where = True
                if ssa.region_ops:
                    _scan_ops(ssa.region_ops)

        _scan_ops(self.graph.ops)

        if not has_dot:
            return False

        # The matmul templates discard input-load masks.  A complete, budgeted
        # masked-dot graph instead falls through to the generic staged-fill path,
        # which structurally replays address + mask + scalar other and refuses any
        # mask it cannot reconstruct.
        if self._masked_dot_generic_eligible():
            return False

        # Pure matmul: dot without reductions or conditional masking.
        # Complex kernels (flash attention, fused matmul+softmax) have
        # reductions/masking alongside dot and must go through generic lowerer.
        if has_reduce or has_where:
            return False

        # RESHAPE-provenance operands (dot-recovery 2B, 2026-09-01): a dot operand
        # staged from `tt.load(1-D) -> tt.reshape -> ttg.local_alloc` carries no 2-D
        # addptr arithmetic, so the template's infer_dot_strides CANNOT resolve it and
        # refuses ("operand stride could not be inferred"). The generic path stages
        # exactly this chain correctly (probe-verified all four upstream
        # test_dot_multidim rank-2 variants at err 0.0, bf16 + memdesc_trans included) —
        # so inside the proven envelope, decline the template and fall through instead
        # of refusing. Outside the envelope the template's loud refusal remains the
        # authority (a reshape it cannot address must never be guessed at).
        if self._dot_has_reshape_provenance() and self._dot_generic_eligible(allow_flat=True):
            return False

        # A single-dot matmul with a trailing compute epilogue the fused template
        # didn't claim (looped dot-in-K-loop, etc.) must REFUSE — the prebuilt template
        # would silently DROP it and routing to generic mis-tiles (see the helper) —
        # UNLESS the kernel is inside the generic dot path's proven envelope
        # (dot-recovery stage 1a): then fall through to the generic lowerer, which
        # computes the epilogue correctly there.
        if self._has_unhandled_matmul_compute_epilogue():
            if self._dot_generic_eligible():
                return False
            raise MetalNonRecoverableError(_MATMUL_EPILOGUE_REFUSE_MSG)

        # Shapes the prebuilt template mis-models (it emits ONE bare 2-D matmul):
        # a multi-dot kernel (chain-dot) or a fused non-zero accumulator init.
        # Inside the proven envelope these fall through to the generic lowerer
        # (which lowers each dot and the accumulator correctly); outside it the
        # template's own refusals fire as before (dot-recovery stage 1a).
        if self._dot_generic_eligible():
            _dots = []

            def _collect(ops):
                for _o in ops:
                    if _o.op == "tt.dot":
                        _dots.append(_o)
                    if _o.region_ops:
                        _collect(_o.region_ops)

            _collect(self.graph.ops)
            if len(_dots) > 1:
                return False
            _by = {o.id: o for o in self.graph.ops}
            if (
                _dots
                and len(_dots[0].operand_ids or []) >= 3
                and not self._acc_init_is_literal_zero(_dots[0].operand_ids[2], _by)
            ):
                return False

        return True

    def _has_unhandled_matmul_compute_epilogue(self):
        """True if this is a SINGLE-dot matmul carrying a trailing TOP-LEVEL compute
        epilogue (bias/activation/fma/clamp) that the fused-epilogue template
        (_detect_matmul_epilogue, checked FIRST in lower()) did NOT claim. What remains
        is the inline-dot / prebuilt matmul template, which stores the RAW accumulator
        and would SILENTLY DROP the epilogue (re-audit #6: looped matmul + fma/scale/relu
        returned the bare dot). Such kernels must REFUSE: routing them to the generic
        op-by-op lowerer is NOT reliable — the combination fuzzer showed generic matmul +
        epilogue is correct at some shapes (M32xN32) yet grossly WRONG at others
        (M16xN16, M32xN64), mis-tiling the dot in a size-dependent way. Restricted to
        EXACTLY ONE dot so multi-dot kernels (FlashAttention = 2 dots + softmax) are
        never touched (the K-loop accumulation `acc += dot` is in the scf.for region, so
        a TOP-LEVEL compute op is the epilogue). Proper follow-up: teach the fused
        template the looped case so large fused matmuls stay fast + correct.
        """
        dots = []

        def _find(ops):
            for o in ops:
                if o.op == "tt.dot":
                    dots.append(o)
                if o.region_ops:
                    _find(o.region_ops)

        _find(self.graph.ops)
        if len(dots) != 1:
            return False
        if any(ssa.op in _MATMUL_EPILOGUE_COMPUTE_OPS for ssa in self.graph.ops):
            return True

        # The denylist above is FLOAT-only (addresses use integer arith, so a
        # top-level float-arith op reliably indicates an epilogue). It therefore
        # MISSES an INTEGER output epilogue — a K-loop int8 matmul + int32 bias on
        # the dot RESULT silently dropped the bias (re-audit 2026-06-28). Catch it
        # ADDITIVELY (the float case already returned True above) with a forward-walk
        # from the matmul OUTPUT: if the dot sits in a top-level region op (the
        # scf.for K-loop) the loop result carries the accumulator, else the dot
        # result. A consumer that is neither a value-preserving passthrough nor the
        # terminal tt.store is an epilogue -> refuse. Crucially this follows ONLY
        # consumers of the matmul output, never the integer ADDRESS arithmetic that
        # independently feeds the store pointer, so a plain / cast / reshape matmul
        # is NOT over-refused.
        dot = dots[0]
        _passthrough = {
            "ttg.convert_layout",
            "tt.reshape",
            "tt.trans",
            "arith.truncf",
            "arith.extf",
            "arith.bitcast",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "tt.fp_to_fp",
        }

        def _region_has(ops, target):
            for o in ops:
                if o.id == target:
                    return True
                if o.region_ops and _region_has(o.region_ops, target):
                    return True
                if o.else_ops and _region_has(o.else_ops, target):
                    return True
            return False

        output_id = dot.id
        for top in self.graph.ops:
            if top.region_ops and _region_has(top.region_ops, dot.id):
                output_id = top.id  # the scf.for loop result carries the accumulator
                break

        _seen = set()
        _frontier = [output_id]
        while _frontier:
            _vid = _frontier.pop()
            if _vid in _seen:
                continue
            _seen.add(_vid)
            for _op in self.graph.ops:
                if _vid not in (_op.operand_ids or []):
                    continue
                if _op.op == "tt.store":
                    continue  # terminal — fine
                if _op.op in _passthrough:
                    _frontier.append(_op.id)  # follow a representation change
                else:
                    return True  # integer (or any) output epilogue
        return False

    def _has_tensor_ops(self) -> bool:
        """Check if the kernel has any tensor-producing ops (tt.make_range, etc.).

        Scalar-only kernels (no tensor operations) should use block_size=1
        to avoid multiple threads racing on the same scalar memory locations.
        """

        def _check_ops(ops):
            for ssa in ops:
                if any(self._native_shapes_for_op(ssa)):
                    return True
                if ssa.region_ops and _check_ops(ssa.region_ops):
                    return True
                if ssa.else_ops and _check_ops(ssa.else_ops):
                    return True
            return False

        return _check_ops(self.graph.ops)

    def _attention_store_arguments(self, forward_only=False):
        """Argument positions an attention kernel stores through, for the launch refusal.

        OUTPUT ALIAS: a source program loads its rows, keys and values before it stores,
        so one whose output shares memory with an input is defined. The attention
        lowerings do not keep that order (a 64-row program runs as two 32-row passes;
        the dQ backward template stores delta before it loads lse) and only the two
        biased source replays show it: the wide-row replay by construction, the narrow
        one by _biased_replay.store_follows_every_load on its emitted text. Every other
        forward attention kernel (two or more dots, an exponential and a max) and every
        backward template route publishes these positions, and the launcher refuses one
        that overlaps another tensor argument. A two-dot kernel that is neither stays on
        the generic lowering, which stages every dot operand behind a barrier, and
        publishes nothing.

        ``forward_only`` is for the caller that has not tested the shape. None when a
        store's pointer does not trace to an argument: with no output to compare, the
        refusal would have to cover inputs that share memory, and attention over one
        tensor as query, key and value is ordinary.
        """

        def walk(ops):
            for ssa in ops:
                yield ssa
                yield from walk(ssa.region_ops or ())
                yield from walk(ssa.else_ops or ())

        ops = list(walk(self.graph.ops))
        if forward_only and (
            sum(ssa.op == "tt.dot" for ssa in ops) < 2
            or not any(_op_is_exp(ssa.op) for ssa in ops)
            or not any("max" in (ssa.op or "") for ssa in ops)
        ):
            return None
        stored = set()
        for ssa in ops:
            if ssa.op in ("tt.store", "tt.atomic_rmw") and ssa.operand_ids:
                arg = self._trace_ptr_source(ssa.operand_ids[0])
                if arg is None:
                    return None
                stored.add(arg.id)
        return [index for index, arg in enumerate(self.graph.args) if arg.id in stored] or None

    def _prescan_stores(self):
        """Scan ops to find which pointer args are stored to (outputs).

        Recursively scans nested regions (scf.for, scf.while, scf.if bodies)
        to find stores inside loops and conditionals.
        """
        self._prescan_stores_recursive(self.graph.ops)

        # Resolve each store's pointer back to its func arg with the SAME cycle-safe
        # tracer the FA role verification uses (_trace_ptr_source, depth 32, follows
        # every operand-0 pass-through) — NOT a private addptr/splat-only walk.
        #
        # The old hand-rolled walk linked only tt.addptr/tt.splat and gave up after 5
        # hops, so a 2-D output address (which crosses tt.broadcast and runs deeper)
        # never resolved. In a MULTI-output kernel that produced a PARTIAL
        # output_arg_indices, and the host dispatch path faithfully obeyed it: the 2-D
        # output was computed correctly on the GPU and then simply never copied back —
        # the caller saw stale memory (GPU-measured: a plain 2-output kernel returned
        # its 1-D output at 1e-6 and its 2-D output UNTOUCHED). Single-output kernels
        # were safe only by accident: total failure yields an empty set, which
        # get_output_arg_indices maps to None = copy-everything.
        #
        # Two safety properties, both load-bearing:
        #   * PARTIAL knowledge is treated as NO knowledge. If any store pointer fails
        #     to trace to a pointer arg, fall back to copy-everything rather than emit
        #     a list that omits a real output.
        #   * The tracer is MORE permissive than the old walk, so a mis-trace could in
        #     principle mark an INPUT as an output; that only widens copy-back to a
        #     buffer the kernel didn't write, which the driver copies back unchanged.
        #     The pin guarding this (input contents byte-identical after a host-path
        #     launch) lives in tests/test_multi_output_copyback.py.
        self._output_arg_ids = set()
        _unresolved = False
        for store_ptr_id in self._store_ptr_ids:
            _arg = self._trace_ptr_source(store_ptr_id)
            if _arg is not None:
                self._output_arg_ids.add(_arg.id)
            else:
                _unresolved = True
        if _unresolved:
            # Copy-everything fallback: correctness over the copy-back optimization.
            self._output_arg_ids = set()

    def _prescan_stores_recursive(self, ops):
        """Recursively find all tt.store and tt.atomic_rmw ops including in nested regions."""
        for ssa in ops:
            if ssa.op == "tt.store":
                if ssa.operand_ids:
                    self._store_ptr_ids.add(ssa.operand_ids[0])
            # tt.atomic_rmw modifies memory in-place — treat target as output
            if ssa.op == "tt.atomic_rmw":
                if ssa.operand_ids:
                    self._store_ptr_ids.add(ssa.operand_ids[0])
            # Recurse into nested regions
            if ssa.region_ops:
                self._prescan_stores_recursive(ssa.region_ops)
            if ssa.else_ops:
                self._prescan_stores_recursive(ssa.else_ops)

    def _prescan_2d_info(self):
        """Detect 2D kernel patterns and compute make_range → dimension mappings.

        Scans the op graph for expand_dims + broadcast chains to determine:
        1. Whether this is a 2D kernel
        2. The effective 2D shape (M, N) from broadcast target types
        3. Which make_range ops correspond to which dimensions

        The pattern is:
            make_range(0, M) → expand_dims(axis=1) → broadcast → tensor<MxNx...>
            make_range(0, N) → expand_dims(axis=0) → broadcast → tensor<MxNx...>

        For a 2D kernel with shape (M, N), thread lid maps to:
            dim 0 (row): lid / N
            dim 1 (col): lid % N
        """
        self._prescan_2d_info_recursive(self.graph.ops)

    def _prescan_2d_info_recursive(self, ops, parent_op_by_id=None):
        """Recursively scan ops for 2D patterns."""
        # Build lookup tables for ops in this scope, including parent scope
        # so tracing can cross scope boundaries (e.g. expand_dims in scf.for
        # body can trace back to make_range in parent scope)
        op_by_id = dict(parent_op_by_id) if parent_op_by_id else {}
        for ssa in ops:
            op_by_id[ssa.id] = ssa

        # Find the max 2D shape from immutable native results in this scope.
        max_2d_shape = None
        for ssa in ops:
            for shape in self._native_shapes_for_op(ssa):
                if len(shape) < 2:
                    continue
                total = 1
                for d in shape:
                    total *= d
                if max_2d_shape is None:
                    max_2d_shape = shape
                else:
                    cur_total = 1
                    for d in max_2d_shape:
                        cur_total *= d
                    if total > cur_total:
                        max_2d_shape = shape
            # Recurse into nested regions
            if ssa.region_ops:
                self._prescan_2d_info_recursive(ssa.region_ops, op_by_id)
            if ssa.else_ops:
                self._prescan_2d_info_recursive(ssa.else_ops, op_by_id)

        if max_2d_shape is None or len(max_2d_shape) < 2:
            return

        self._is_2d = True
        if self._effective_2d_shape is None:
            self._effective_2d_shape = max_2d_shape

        # Build a users map (value id -> list of ops that use it as operand)
        # so we can walk expand_dims chains forward to find the final N-D shape.
        # Limited to ops in this scope; for chains crossing scopes we fall back
        # to the axis-based heuristic.
        users_map = {}
        for ssa in ops:
            for oid in ssa.operand_ids:
                users_map.setdefault(oid, []).append(ssa)

        def _final_expand_shape(first_ed_ssa):
            """Walk forward through consecutive expand_dims ops and return
            the shape of the outermost expand_dims (the one whose result
            is consumed by broadcast/addi/load/etc, not by another expand_dims).
            """
            cur = first_ed_ssa
            while True:
                next_ed = None
                for user in users_map.get(cur.id, []):
                    if user.op == "tt.expand_dims":
                        next_ed = user
                        break
                if next_ed is None:
                    break
                cur = next_ed
            return self._native_shape(cur.id, op_name=cur.op)

        # Find expand_dims ops and trace back to make_range.
        # Also record expand_dims by parent layout to pair dim=0/dim=1 siblings
        # so each make_range can know its tile's inner dimension.
        # Use an instance-level dict to accumulate across recursive prescan calls.
        if not hasattr(self, "_expand_by_parent"):
            self._expand_by_parent = {}
        if not hasattr(self, "_make_range_stride_below"):
            self._make_range_stride_below = {}
        if not hasattr(self, "_make_range_full_shape"):
            self._make_range_full_shape = {}
        expand_by_parent = self._expand_by_parent

        # Also detect tt.reshape of a make_range (or a chain ending in one) to
        # a shape where exactly one axis has non-1 size — semantically
        # equivalent to an expand_dims chain for our lowering purposes. This
        # is how tl.sort's _indicator emits its per-axis ranges.
        # The stride_below is computed against the eventual broadcast target
        # (the largest same-rank tensor in the kernel) so it reflects the
        # enclosing tensor's actual strides.
        def _max_shape_with_rank(rank):
            """Find the largest tensor (by element count) with the given rank."""
            best = None
            best_numel = 0
            for s_ssa in ops:
                for s_shape in self._native_shapes_for_op(s_ssa):
                    if not s_shape or len(s_shape) != rank:
                        continue
                    numel = 1
                    for d in s_shape:
                        numel *= d
                    if numel > best_numel:
                        best = s_shape
                        best_numel = numel
            return best

        # Per-make_range broadcast target: each make_range's stride_below must come
        # from the broadcast IT actually feeds, not the largest same-rank tensor in the
        # kernel. Two make_ranges of the same rank broadcast to different shapes (e.g.
        # 1x2x1 -> 1x2x4 vs 1x2x1 -> 4x2x1) have different strides; the global-max
        # heuristic silently gave both the same one (issue #9). Map each make_range to
        # the distinct broadcast output shape(s) it reaches. A range feeding MULTIPLE
        # distinct broadcast shapes (tl.sort indicator, etc.) records no per-range target
        # and keeps the existing heuristic + reshape rewrite below; there is no
        # conflict-refusal here.
        if not hasattr(self, "_make_range_bcast_target"):
            self._make_range_bcast_target = {}

        # Elementwise/value ops that PRESERVE a make_range's per-axis coordinate: the
        # result at axis-k position i is a function of range[i] alone, so the range still
        # indexes the same axis in any broadcast of the result. Shape ops are consumed by
        # the stage-1 walk below; these are the value-combining ops that can sit between a
        # range and its broadcast (casts, comparisons, select, arithmetic, gather/load).
        _VALUE_PRESERVING = (
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "arith.truncf",
            "arith.index_cast",
            "arith.index_castui",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "arith.bitcast",
            "tt.bitcast",
            "tt.fp_to_fp",
            "ttg.convert_layout",
            "arith.cmpi",
            "arith.cmpf",
            "arith.select",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.minsi",
            "arith.maxsi",
            "arith.minui",
            "arith.maxui",
            "arith.shli",
            "arith.shrsi",
            "arith.shrui",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.minnumf",
            "arith.maxnumf",
            "arith.minimumf",
            "arith.maximumf",
            "tt.load",
            "tt.addptr",
        )

        def _is_coord_invariant(_vid):
            # An operand is coordinate-invariant when it contributes the SAME value to
            # every coordinate, so it cannot change which axis a sibling make_range
            # indexes. A tt.splat always broadcasts one scalar. An arith.constant is
            # invariant ONLY when proven uniform: a scalar/splat value parses to a Python
            # number (int/float/bool), whereas a NONUNIFORM dense tensor constant
            # (dense<[...]>) parses to a non-number and varies per coordinate -- op name
            # alone does not prove uniformity. An all-ones/scalar shape is a single
            # broadcast value regardless.
            _o = op_by_id.get(_vid)
            if _o is not None:
                if _o.op == "tt.splat":
                    return True
                if _o.op == "arith.constant":
                    _v = (_o.attrs or {}).get("value")
                    if isinstance(_v, (int, float)):  # bool is an int subclass
                        return True
                    # else fall through to the shape check below
            if _o is None:
                return False
            try:
                _sh = self._native_shape(_vid, op_name=_o.op)
            except Exception:
                return False
            return _sh is not None and (len(_sh) == 0 or all(_d == 1 for _d in _sh))

        def _ranges_of(_vid, _seen):
            # Frozenset of make_range ids this value is a PURE per-axis function of,
            # following only coordinate-bearing operands; None when a coordinate-bearing
            # operand's correspondence cannot be proven (unknown op, block arg, or a
            # non-range coordinate-bearing leaf). An empty set = coordinate-invariant.
            if _vid in _seen:
                return frozenset()
            _seen = _seen | {_vid}
            if _is_coord_invariant(_vid):
                return frozenset()
            _o = op_by_id.get(_vid)
            if _o is None:
                return None
            if _o.op == "tt.make_range":
                return frozenset((_vid,))
            if _o.op in _VALUE_PRESERVING:
                _acc = frozenset()
                for _oid in _o.operand_ids or ():
                    if _is_coord_invariant(_oid):
                        continue
                    _sub = _ranges_of(_oid, _seen)
                    if _sub is None:
                        return None
                    _acc = _acc | _sub
                return _acc
            return None

        _bt = {}
        for _b in ops:
            if _b.op != "tt.broadcast" or not _b.operand_ids:
                continue
            # Stage 1: consume the leading shape-op chain down to the first value op.
            _cur = _b.operand_ids[0]
            _seen_shape = set()
            while _cur in op_by_id and _cur not in _seen_shape:
                _seen_shape.add(_cur)
                _o = op_by_id[_cur]
                if _o.op in ("tt.expand_dims", "tt.broadcast", "tt.reshape", "ttg.convert_layout") and _o.operand_ids:
                    _cur = _o.operand_ids[0]
                    continue
                break
            # Stage 2: prove the value under the broadcast is a pure per-axis function of
            # exactly ONE make_range. Comparison-, select-, arithmetic- and gather-derived
            # broadcasts (issue #9 residuals) are coordinate-preserving and resolve here;
            # outer products (two distinct ranges) and unprovable inputs record nothing
            # and keep the old heuristic. _trace_to_make_range is NOT used (it is shared
            # with pointer/load consumers and does not traverse cmpi/select).
            _rs = _ranges_of(_cur, set())
            if not _rs or len(_rs) != 1:
                continue
            _mr = next(iter(_rs))
            _bshape = self._native_shape(_b.id, op_name=_b.op)
            if _bshape:
                _bt.setdefault(_mr, set()).add(tuple(_bshape))
        for _mr, _shapes in _bt.items():
            if len(_shapes) == 1:
                # Only when a make_range feeds a SINGLE broadcast is its target
                # unambiguous. Multiple broadcasts (tl.sort indicator, etc.) keep the
                # existing heuristic + slice-layout reshape rewrite, which handle them.
                self._make_range_bcast_target[_mr] = next(iter(_shapes))

        for ssa in ops:
            if ssa.op != "tt.reshape" or not ssa.operand_ids:
                continue
            src_id = ssa.operand_ids[0]
            mr_id = self._trace_to_make_range(src_id, ops, op_by_id)
            if mr_id is None:
                _rs = _ranges_of(src_id, set())
                mr_id = next(iter(_rs)) if (_rs and len(_rs) == 1) else None
            if mr_id is None:
                continue
            if mr_id in self._make_range_dim:
                # Already assigned by an expand_dims chain; do not override.
                continue
            out_shape = self._native_shape(ssa.id, op_name=ssa.op)
            if not out_shape or len(out_shape) < 2:
                continue
            non_one = [i for i, s in enumerate(out_shape) if s != 1]
            if len(non_one) != 1:
                continue
            dim = non_one[0]
            self._make_range_dim[mr_id] = dim
            # Broadcast target: largest same-rank tensor in the kernel. For
            # tl.sort, the reshape output is e.g. (1,1,1,2) and the broadcast
            # target is (2,2,2,2). Using the actual broadcast shape ensures
            # stride_below reflects the enclosing tensor's strides.
            own = self._make_range_bcast_target.get(mr_id)
            if own is not None and len(own) == len(out_shape):
                broadcast_shape = own
            else:
                same_rank = _max_shape_with_rank(len(out_shape))
                broadcast_shape = tuple(same_rank) if same_rank else tuple(out_shape)
            self._make_range_full_shape[mr_id] = broadcast_shape
            stride_below = 1
            for s in broadcast_shape[dim + 1 :]:
                stride_below *= s
            self._make_range_stride_below[mr_id] = stride_below

        for ssa in ops:
            if ssa.op == "tt.expand_dims" and ssa.operand_ids:
                axis = ssa.attrs.get("axis", 0)
                src_id = ssa.operand_ids[0]
                # Only start a chain-walk at the INNERMOST expand_dims: the one
                # whose source is NOT itself an expand_dims. This avoids
                # assigning dims multiple times per make_range.
                src_op = op_by_id.get(src_id)
                if src_op is not None and src_op.op == "tt.expand_dims":
                    continue
                # Trace back through passthroughs to find the make_range
                mr_id = self._trace_to_make_range(src_id, ops, op_by_id)
                if mr_id is None:
                    _rs = _ranges_of(src_id, set())
                    mr_id = next(iter(_rs)) if (_rs and len(_rs) == 1) else None
                if mr_id is not None:
                    # Walk forward through the expand_dims chain to find the
                    # final N-D shape (all dims are 1 except the make_range's
                    # position). The position of the non-1 axis gives the
                    # true dim in the broadcast target.
                    final_shape = _final_expand_shape(ssa)
                    dim = None
                    if final_shape and len(final_shape) >= 2:
                        non_one = [i for i, s in enumerate(final_shape) if s != 1]
                        if len(non_one) == 1:
                            dim = non_one[0]
                        elif len(non_one) == 0:
                            # Range of size 1 (degenerate). Fall back to axis heuristic.
                            dim = 0 if axis == 1 else (len(final_shape) - 1)
                    if dim is None:
                        # Fallback to the original 2D axis-based heuristic
                        dim = 0 if axis == 1 else (len(max_2d_shape) - 1)
                        final_shape = max_2d_shape
                    self._make_range_dim[mr_id] = dim
                    # Use the (broadcast) max_2d_shape for stride_below, since
                    # broadcast expands size-1 dims to the enclosing tensor's
                    # sizes. The position `dim` is the make_range's axis in
                    # that shape; stride_below = product of broadcast dims
                    # after `dim`.
                    own = self._make_range_bcast_target.get(mr_id)
                    if own is not None and final_shape and len(own) == len(final_shape):
                        broadcast_shape = own
                    else:
                        broadcast_shape = (
                            max_2d_shape if (max_2d_shape and len(max_2d_shape) == len(final_shape)) else final_shape
                        )
                    self._make_range_full_shape[mr_id] = broadcast_shape
                    stride_below = 1
                    for s in broadcast_shape[dim + 1 :]:
                        stride_below *= s
                    self._make_range_stride_below[mr_id] = stride_below

                    # Extract the parent layout from immutable native metadata
                    # e.g., "tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>"
                    # to pair with siblings from the same tile.
                    layout_value_id = src_id if src_id in self.graph.result_meta else mr_id
                    src_layout = self._native_value_facts(layout_value_id, op_name="2-D shape prescan").layout or ""
                    # Extract the parent layout identifier. The type string
                    # may use aliases (#blocked, #blocked1) or inline defs
                    # (#ttg.blocked<{...}>).  Use a nested-brace-aware match.
                    parent_key = "default"
                    pidx = src_layout.find("parent")
                    if pidx >= 0:
                        # Find the `=` and the start of the layout spec
                        eq_idx = src_layout.find("=", pidx)
                        if eq_idx >= 0:
                            rest = src_layout[eq_idx + 1 :].strip()
                            # Capture everything up to matching `>` or `}`
                            depth = 0
                            end_idx = 0
                            for ci, ch in enumerate(rest):
                                if ch in ("<", "{", "["):
                                    depth += 1
                                elif ch in (">", "}", "]"):
                                    if depth == 0:
                                        end_idx = ci
                                        break
                                    depth -= 1
                                    if depth == 0:
                                        end_idx = ci + 1
                                        break
                            parent_key = rest[:end_idx].strip() if end_idx > 0 else rest[:40]
                    mr_op = op_by_id.get(mr_id)
                    range_size = 0
                    if mr_op:
                        range_size = mr_op.attrs.get("end", 0) - mr_op.attrs.get("start", 0)
                    expand_by_parent.setdefault(parent_key, []).append((dim, mr_id, range_size))

        # For each parent layout, pair dim=0 and dim=1 make_ranges to
        # determine the tile inner dim for dim=0 (row) make_ranges.
        if not hasattr(self, "_make_range_inner_N"):
            self._make_range_inner_N = {}
        for parent_key, entries in expand_by_parent.items():
            dim0_entries = [(mr_id, rs) for d, mr_id, rs in entries if d == 0]
            dim1_entries = [(mr_id, rs) for d, mr_id, rs in entries if d != 0]
            # The dim=1 range_size IS the inner dim N for all dim=0 siblings
            if dim1_entries:
                inner_N = max(rs for _, rs in dim1_entries)
                for mr_id, _ in dim0_entries:
                    self._make_range_inner_N[mr_id] = inner_N

        # Use analysis: figure out which make_ranges flow ONLY to tt.store
        # (not to tt.load). For such store-only make_ranges, when the tile
        # (range × inner_N) is smaller than the kernel's total element count,
        # the default lid/inner_N row expression aliases multiple rows onto
        # the same M coord. Mark them for inner_N scaling at lowering time.
        #
        # This happens in tl.topk M>1: the Z-offset make_range has parent
        # layout for the (M, k) output tile, which is smaller than the block
        # that processes the (M, N) input.
        if not hasattr(self, "_make_range_store_only"):
            self._make_range_store_only = set()

        # Build a use map: SSA id -> set of op IDs that use it
        use_of = {}
        all_oplist = []

        def _coll(ops):
            for o in ops:
                all_oplist.append(o)
                if o.region_ops:
                    _coll(o.region_ops)
                if o.else_ops:
                    _coll(o.else_ops)

        _coll(ops)
        for o in all_oplist:
            if o.operand_ids:
                for oid in o.operand_ids:
                    use_of.setdefault(oid, set()).add(o.id)

        # For each make_range, do a transitive use walk; check if it reaches
        # any tt.load (or tt.gather etc.) as well as any tt.store.
        op_by_id2 = {o.id: o for o in all_oplist}

        def _transitive_uses(start_id):
            seen = {start_id}
            stack = [start_id]
            reaches_load = False
            reaches_store = False
            while stack:
                cur = stack.pop()
                for u in use_of.get(cur, ()):
                    if u in seen:
                        continue
                    seen.add(u)
                    u_op = op_by_id2.get(u)
                    if u_op is None:
                        continue
                    # A make_range used as the PTR operand of a store would
                    # make that store a store-target user. But here we care
                    # about offset_operand: make_range flows through expand,
                    # muli, broadcast, addi into addptr, then into store.
                    if u_op.op == "tt.load":
                        reaches_load = True
                    elif u_op.op in ("tt.store", "tt.atomic_rmw"):
                        reaches_store = True
                    stack.append(u)
            return reaches_load, reaches_store

        for parent_key, entries in expand_by_parent.items():
            for _d, mr_id, _rs in entries:
                rl, rs = _transitive_uses(mr_id)
                if rs and not rl:
                    self._make_range_store_only.add(mr_id)

    def _trace_to_make_range(self, ssa_id, ops, op_by_id):
        """Trace an SSA ID back through passthrough ops to find a make_range.

        Follows: passthroughs (extsi, convert_layout, etc.), tt.load (through
        the pointer operand), tt.addptr (through the offset operand), and
        arithmetic ops (muli, addi — tries both operands).
        This allows tracing from expand_dims through load→addptr→make_range
        chains, which is needed when a 1D load result gets expand_dims'd to 2D.
        """
        visited = set()
        current = ssa_id
        while current not in visited:
            visited.add(current)
            if current in op_by_id:
                op = op_by_id[current]
                if op.op == "tt.make_range":
                    return current
                # Follow through passthroughs (first operand)
                if (
                    op.op
                    in (
                        "arith.extsi",
                        "arith.extui",
                        "arith.trunci",
                        "arith.index_cast",
                        "arith.index_castui",
                        "arith.sitofp",
                        "arith.uitofp",
                        "ttg.convert_layout",
                        "tt.load",
                    )
                    and op.operand_ids
                ):
                    current = op.operand_ids[0]
                    continue
                # tt.addptr: follow the offset (second operand) to reach make_range
                if op.op == "tt.addptr" and len(op.operand_ids) >= 2:
                    current = op.operand_ids[1]
                    continue
                # Arithmetic ops (muli, addi): the make_range could be either
                # operand (e.g. arange*SIZE or SIZE*arange). Try both.
                if op.op in ("arith.muli", "arith.addi") and op.operand_ids:
                    for oid in op.operand_ids:
                        result = self._trace_to_make_range(oid, ops, op_by_id)
                        if result is not None:
                            return result
                    break
            break
        return None

    def _register_args(self):
        """Register function arguments with KernelBuilder."""
        for arg in self.graph.args:
            triton_dtype = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                # Never use const for generic-lowered kernels — prescan can miss
                # stores through block args, reductions, and complex chains.
                self.kb.add_ptr_arg(arg.name, dtype=triton_dtype, const=False)
            else:
                self.kb.add_scalar_arg(arg.name, dtype=triton_dtype)
            self.env[arg.id] = arg.name
            self.env_types[arg.id] = triton_dtype
            # Shape: function arguments are always scalar (pointers are
            # base addresses, scalars are single values).  tt.splat lifts
            # them to tensor shapes downstream.
            self.env_shapes[arg.id] = ()

    def _lookup(self, ssa_id: int) -> str:
        """Look up MSL variable name for an SSA value."""
        if ssa_id in self.env:
            return self.env[ssa_id]
        raise MetalNonRecoverableError(f"codegen encountered unresolved SSA value {ssa_id}")

    def _lookup_array(self, ssa_id: int):
        """Return ``(name, n_elems, ty)`` for an SSA value's MEPT storage.

        Phase 4b helper. If ``ssa_id`` has an entry in ``env_array``,
        returns it directly. Otherwise returns ``(scalar_name, 1, ty)``
        so call sites can iterate ``range(n_elems)`` uniformly without
        branching on storage form. ``ty`` defaults to the empty string
        when the type isn't tracked — callers that need the type for an
        array declaration should resolve it separately.

        Always check ``self.mept_enabled`` before deciding to emit array
        code; the scalar form remains correct when the flag is off.
        """
        if ssa_id in self.env_array:
            return self.env_array[ssa_id]
        return (self._lookup(ssa_id), 1, "")

    def _lookup_regval(self, ssa_id):
        """Unified RegVal view over env / env_array. Does not change emission;
        callers migrate to this incrementally (MEPT spine, milestone 1)."""
        from triton_msl.codegen.regval import RegVal

        if ssa_id in getattr(self, "env_array", {}):
            name, n, ty = self.env_array[ssa_id]
            return RegVal(name=name, n_elems=n, ty=ty, form="array")
        name = self._lookup(ssa_id)
        n = self.env_n_elems.get(ssa_id, 1)
        ty = self.env_types.get(ssa_id, "")
        return RegVal(name=name, n_elems=n, ty=ty, form="scalar")

    def _lower_op(self, ssa: SSAValue):
        """Lower a single SSA operation to MSL."""
        # Skip ops that were handled as part of a fused pattern
        if ssa.id in self._skip_ids:
            return

        try:
            self._lower_op_dispatch(ssa)
        except (MetalCodegenError, MetalNotImplementedError):
            raise  # Already has context
        except Exception as e:
            raise MetalCodegenError(
                f"Failed to lower operation: {e}",
                op_name=ssa.op,
                ssa_id=ssa.id,
                type_str=ssa.type_str,
            ) from e

    def _lower_op_dispatch(self, ssa: SSAValue):
        """Dispatch a single SSA operation to its lowering handler."""
        op = ssa.op
        ids = ssa.operand_ids

        # Dispatch by op name
        if op == "tt.get_program_id":
            self._lower_get_program_id(ssa)
        elif op == "tt.get_num_programs":
            self._lower_get_num_programs(ssa)
        elif op == "tt.make_range":
            self._lower_make_range(ssa)
        elif op == "tt.splat":
            self._lower_splat(ssa)
        elif op == "tt.expand_dims":
            self._lower_expand_dims(ssa)
        elif op == "tt.broadcast":
            self._lower_broadcast(ssa)
        elif op == "tt.addptr":
            self._lower_addptr(ssa)
        elif op == "tt.load":
            self._lower_load(ssa)
        elif op == "tt.store":
            self._lower_store(ssa)
        elif op == "tt.reduce":
            self._lower_reduce(ssa)
        elif op == "tt.scan":
            self._lower_scan(ssa)
        elif op == "tt.clampf":
            self._lower_clampf(ssa)
        elif op == "tt.dot":
            self._lower_dot(ssa)
        elif op == "arith.constant":
            self._lower_constant(ssa)
        elif op.startswith("arith."):
            self._lower_arith(ssa)
        elif op.startswith("math."):
            self._lower_math(ssa)
        elif op == "scf.for":
            self._control_flow_depth += 1
            try:
                self._lower_scf_for(ssa)
            finally:
                self._control_flow_depth -= 1
        elif op == "scf.if":
            self._control_flow_depth += 1
            try:
                self._lower_scf_if(ssa)
            finally:
                self._control_flow_depth -= 1
        elif op == "scf.while":
            self._control_flow_depth += 1
            try:
                self._lower_scf_while(ssa)
            finally:
                self._control_flow_depth -= 1
        elif op in ("scf.yield", "scf.condition"):
            pass  # Handled by parent op
        elif op == "tt.call":
            self._lower_call(ssa)
        elif op == "tt.return":
            pass  # Kernel return — nothing to emit
        elif op.startswith("ttg."):
            self._lower_ttg(ssa)
        elif op == "tt.reshape":
            self._lower_reshape(ssa)
        elif op == "tt.trans":
            self._lower_tt_trans(ssa)
        elif op == "tt.join":
            self._lower_tt_join(ssa)
        elif op == "tt.cat":
            self._lower_tt_cat(ssa)
        elif op == "tt.split":
            self._lower_tt_split(ssa)
        elif op == "tt.histogram":
            self._lower_tt_histogram(ssa)
        elif op == "tt.gather":
            self._lower_tt_gather(ssa)
        elif op == "tt.unsplat":
            # tt.unsplat: extract scalar from 1-element tensor (inverse of splat)
            # In per-thread model, this is a passthrough.
            self._emit_passthrough(ssa)
        elif op == "tt.map_elementwise":
            self._lower_map_elementwise(ssa)
        elif op == "tt.atomic_rmw":
            self._lower_atomic_rmw(ssa)
        elif op == "tt.atomic_cas":
            self._lower_atomic_cas(ssa)
        elif op == "tt.debug_barrier":
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_device);")
        elif op == "tt.mulhiui":
            self._lower_mulhiui(ssa)
        elif op == "tt.bitcast":
            # tt.bitcast can be:
            # 1. Pointer bitcast (e.g. !tt.ptr<i1> -> !tt.ptr<i8>)
            # 2. Value bitcast (e.g. f32 -> i32 for float atomic max)
            # For case 2, we need as_type<T>() in MSL.
            self._lower_tt_bitcast(ssa)
        elif op == "tt.precise_sqrt":
            self._lower_precise_math(ssa, "sqrt")
        elif op == "tt.precise_divf":
            self._lower_precise_math(ssa, "divf")
        elif op == "tt.extern_elementwise":
            self._lower_extern_elementwise(ssa)
        elif op == "tt.fp_to_fp":
            self._lower_fp_to_fp(ssa)
        elif op == "tt.assert":
            from .assertions import emit

            emit(self, ssa)
        elif op in ("tt.print", "tt.device_print"):
            # Device-side print: Apple GPUs have no device printf channel. This
            # is an output-NEUTRAL diagnostic side effect, so dropping it is
            # correct (does not affect the kernel's result buffers) — unlike a
            # store/atomic, which the no-result refusal below catches. Documented
            # unsupported (test_print is skip-listed). (audit #165)
            pass
        else:
            # Unknown op. A NO-RESULT op (negative synthetic id, see mlir_walker
            # SSAValue construction) is a side-effect / terminator the lowerer
            # doesn't model — store/atomic/scatter/cf variants — and dropping it
            # silently loses its effect (audit #165). Refuse loudly. A
            # result-producing unknown op records an unsupported outcome even
            # when dead. A consumer of its undefined result refuses at lookup;
            # otherwise emit_msl handles the explicit unsupported outcome.
            if ssa.id is not None and ssa.id < 0:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"Refusing to emit silently-wrong output: unsupported "
                    f"side-effecting op '{op}' (no result) has no Metal lowering "
                    f"and would be silently dropped, losing its effect. Add a "
                    f"handler or an explicit refusal for it."
                )
            self.kb.unsupported(f"UNSUPPORTED: {op}")

    # -- Program ID and indexing --

    def _lower_get_program_id(self, ssa: SSAValue):
        """tt.get_program_id → pid / pid_y / pid_z.

        Axis 0 (x) → pid, axis 1 (y) → pid_y, axis 2 (z) → pid_z.
        Tracks which axes are used so the kernel signature includes them.
        """
        axis = ssa.attrs.get("axis", 0)
        self._used_pid_axes.add(axis)
        if axis == 0:
            self.env[ssa.id] = "pid"
        elif axis == 1:
            self.env[ssa.id] = "pid_y"
        else:
            self.env[ssa.id] = "pid_z"
        self.env_types[ssa.id] = "i32"
        # Shape: program_id is always scalar
        self.env_shapes[ssa.id] = ()

    def _lower_get_num_programs(self, ssa: SSAValue):
        """tt.get_num_programs → grid dimension (threadgroups_per_grid).

        Uses Metal's [[threadgroups_per_grid]] kernel parameter.
        Since other grid attributes (pid, lid, tid) use scalar uint,
        threadgroups_per_grid must also be scalar uint (Metal requires
        all position attributes to have matching dimensionality).
        For axis 0 this is just 'tpg'. Multi-axis dispatch would need
        uint3 for all grid attributes.
        """
        axis = ssa.attrs.get("axis", 0)
        if axis == 0:
            self.env[ssa.id] = "tpg"
        elif axis == 1:
            self.env[ssa.id] = "tpg_y"
        else:
            self.env[ssa.id] = "tpg_z"
        self.env_types[ssa.id] = "i32"
        # Track which axes need num_programs
        self._used_pid_axes.add(axis)
        # Flag that we need the threadgroups_per_grid parameter
        self._needs_num_programs = True
        # Shape: num_programs is always scalar
        self.env_shapes[ssa.id] = ()

    def _lower_make_range(self, ssa: SSAValue):
        """tt.make_range → lid or 2D index expression.

        In Metal SIMT, tt.make_range {start=0, end=BLOCK_SIZE}
        produces per-thread indices [0, 1, ..., BLOCK_SIZE-1].

        For 1D kernels: maps directly to lid.
        For 2D kernels: maps to lid/N (row dim) or lid%N (col dim),
        determined by the expand_dims + broadcast pre-pass analysis.
        """
        start = ssa.attrs.get("start", 0)
        end = ssa.attrs.get("end", self.graph.block_size)

        # Check if this make_range is part of a 2D/N-D pattern
        if self._is_2d and ssa.id in self._make_range_dim:
            dim = self._make_range_dim[ssa.id]
            range_size = end - start
            lid = self._lid_expr
            scan_shape = getattr(self, "_scan_mept_shape", None)
            scan_width = getattr(self, "_scan_mept_width", 0)
            if getattr(self, "_mept_single_pass", False) and scan_shape is not None:
                if len(scan_shape) != 2 or dim not in (0, 1) or range_size != scan_shape[dim] or scan_width <= 1:
                    raise MetalNonRecoverableError(
                        "tt.scan: a 2-D make_range is outside the proved flat "
                        "row/column projection cone; refusing rather than inventing "
                        "per-slot coordinates.",
                        op_name="tt.make_range",
                    )
                inner = scan_shape[1]
                exprs = []
                for i in range(scan_width):
                    flat = f"({lid} * {scan_width}u + {i}u)"
                    coord = f"({flat} / {inner}u)" if dim == 0 else f"({flat} % {inner}u)"
                    exprs.append(f"({coord} + {start}u)" if start else coord)
                var_name = self._var_array("idx", exprs, "uint")
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, scan_width, "uint")
                self.env_types[ssa.id] = "i32"
                self.env_shapes[ssa.id] = (range_size,)
                return

            var_name = self._next_var("idx")
            # Use _total_elements (not capped effective_block_size) for
            # correct index decomposition when wrapping loop is active.
            total = getattr(self, "_total_elements", self.effective_block_size)

            # If the prescan computed a per-range stride_below (product of dims
            # after `dim` in the make_range's final N-D shape), use the general
            # decomposition: (lid / stride_below) % range_size. This handles
            # both 2D and 3D+ patterns correctly:
            #   2D (M, N), dim 0 (row):   stride_below = N, expr = (lid / N) % M
            #   2D (M, N), dim 1 (col):   stride_below = 1, expr = lid % N
            #   3D (M, N, K), dim 0:      stride_below = N*K
            #   3D (M, N, K), dim 1:      stride_below = K
            #   3D (M, N, K), dim 2:      stride_below = 1
            stride_below_map = getattr(self, "_make_range_stride_below", {})
            full_shape = getattr(self, "_make_range_full_shape", {}).get(ssa.id)
            if ssa.id in stride_below_map and full_shape and len(full_shape) >= 3:
                stride_below = stride_below_map[ssa.id]
                if stride_below == 1:
                    expr = f"({lid} % {range_size}u)"
                else:
                    expr = f"(({lid} / {stride_below}u) % {range_size}u)"
            else:
                # 2D path: preserved for backward compatibility and the
                # dim-0 inner_N pairing (transpose tiles, FlashAttention, etc).
                # Compute the inner dimension N from range_size and total.
                # For row (dim 0): range covers rows, N = total / range = inner dim
                # For col (dim 1): range IS the inner dim, N = range_size
                if dim == 0:
                    inner_N_map = getattr(self, "_make_range_inner_N", {})
                    if ssa.id in inner_N_map:
                        N = inner_N_map[ssa.id]
                    else:
                        N = total // range_size if range_size > 0 else 1
                    # Store-only make_ranges whose tile is smaller than the
                    # kernel's element count: scale inner_N so lid/N still
                    # yields the correct M-index (see tl.topk M>1 case).
                    store_only = getattr(self, "_make_range_store_only", set())
                    if ssa.id in store_only and range_size > 0 and total > range_size * N:
                        N = total // range_size
                    expr = f"{lid} / {N}u"
                else:
                    expr = f"{lid} % {range_size}u"

            if start != 0:
                expr = f"({expr} + {start}u)"

            self.kb.raw_line(f"    uint {var_name} = {expr};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = (range_size,)
            return

        # 1D make_range in a 2D kernel: this range is used for a 1D operation
        # (like a load) that later gets expand_dims'd to 2D. The range values
        # need to cycle within [start, end) for each thread, so use modular
        # indexing: lid % range_size. This gives each column/row of threads
        # a valid index within the original 1D array.
        range_size = end - start
        total = getattr(self, "_total_elements", self.effective_block_size)
        if self._is_2d and range_size < total:
            lid = self._lid_expr
            scan_shape = getattr(self, "_scan_mept_shape", None)
            scan_width = getattr(self, "_scan_mept_width", 0)
            if getattr(self, "_mept_single_pass", False) and scan_shape is not None:
                # Packet 325 option A: all 2-D scan indices are projections
                # of the same flattened ownership coordinate.  Do not use
                # the blocked layout's sizePerThread here (it is [1,1] for
                # the upstream 2-D scans and therefore undercounts the tile).
                # The prescan must have identified this make_range's axis and
                # its extent must match that axis exactly.
                dim = self._make_range_dim.get(ssa.id)
                if len(scan_shape) != 2 or dim not in (0, 1) or range_size != scan_shape[dim] or scan_width <= 1:
                    raise MetalNonRecoverableError(
                        "tt.scan: a 2-D make_range is outside the proved flat "
                        "row/column projection cone; refusing rather than inventing "
                        "per-slot coordinates.",
                        op_name="tt.make_range",
                    )
                inner = scan_shape[1]
                exprs = []
                for i in range(scan_width):
                    flat = f"({lid} * {scan_width}u + {i}u)"
                    coord = f"({flat} / {inner}u)" if dim == 0 else f"({flat} % {inner}u)"
                    exprs.append(f"({coord} + {start}u)" if start else coord)
                var_name = self._var_array("idx", exprs, "uint")
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, scan_width, "uint")
                self.env_types[ssa.id] = "i32"
                self.env_shapes[ssa.id] = (range_size,)
                return
            var_name = self._next_var("idx")
            if start != 0:
                expr = f"({lid} % {range_size}u + {start}u)"
            else:
                expr = f"{lid} % {range_size}u"
            self.kb.raw_line(f"    uint {var_name} = {expr};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = (range_size,)
            return

        # Pure 1D case (original behavior)
        lid = self._lid_expr
        # Phase 4c: MEPT producer. When the result tensor carries >1
        # elements per thread, emit an array form. If a resolved
        # ``LinearLayout`` is available, use its register-basis
        # position math (correct for any 1D layout, including
        # interleaved warps and non-default order). Otherwise default
        # to the contiguous formula ``idx[i] = start + lid*N + i``
        # (valid for synthetic / default blocked layouts).
        # ``_lower_make_range`` writes env_shapes directly (not via
        # ``_propagate_shape_from_type``), so ensure env_n_elems is
        # populated from the result's TTGIR layout here.
        if ssa.id not in self.env_n_elems and ssa.type_str:
            self._track_n_elems(ssa.id, ssa.type_str, (end - start,))
        n_per_thread = getattr(self, "_scan_mept_width", 0) or self.env_n_elems.get(ssa.id, 1)
        # The MEPT array path activates ONLY when the prescan proved this
        # kernel is single-pass MEPT-safe (exact tile cover, no barriers,
        # every op array-wired, no fp8). That flag is the sole gate for
        # the whole MEPT chain: if make_range stays scalar, no env_array
        # propagates and every downstream consumer stays scalar too.
        # This is what keeps MEPT from firing inside reduction/multipass
        # kernels (where _needs_wrapping is False but the array form is
        # unsupported) or kernels containing unwired ops (select, atomics,
        # shape ops, ...).
        if n_per_thread > 1 and getattr(self, "_mept_single_pass", False):
            ll = self.env_layout.get(ssa.id)
            if getattr(self, "_scan_mept_shape", None) is not None:
                # Scan option A promises contiguous flattened ownership, not
                # the source layout's register basis.
                exprs = [
                    f"{start}u + {lid} * {n_per_thread}u + {i}u" if start != 0 else f"{lid} * {n_per_thread}u + {i}u"
                    for i in range(n_per_thread)
                ]
            elif ll is not None and ll.num_registers_per_thread == n_per_thread:
                # Compute lane / warp from lid:
                #   lane = lid & 31u
                #   warp = lid >> 5u
                # The XOR-basis position(register=i, lane, warp) is the
                # global element index this thread holds at register i.
                exprs = []
                for i in range(n_per_thread):
                    pos_expr = ll.msl_position_expr(
                        reg_var=f"{i}u",
                        lane_var=f"({lid} & 31u)",
                        warp_var=f"({lid} >> 5u)",
                    )
                    if start != 0:
                        exprs.append(f"(uint)({start} + ({pos_expr}))")
                    else:
                        exprs.append(f"(uint)({pos_expr})")
            else:
                exprs = [
                    f"{start}u + {lid} * {n_per_thread}u + {i}u" if start != 0 else f"{lid} * {n_per_thread}u + {i}u"
                    for i in range(n_per_thread)
                ]
            var_name = self._var_array("idx", exprs, "uint")
            self.env[ssa.id] = var_name
            self.env_array[ssa.id] = (var_name, n_per_thread, "uint")
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = (end - start,)
            return
        if start != 0:
            var_name = self._next_var("range")
            self.kb.raw_line(f"    uint {var_name} = {lid} + {start}u;")
            self.env[ssa.id] = var_name
        else:
            self.env[ssa.id] = lid
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = (end - start,)

    def _lower_splat(self, ssa: SSAValue):
        """tt.splat → pass through (broadcast is free in SIMT).

        In Triton IR, tt.splat broadcasts a scalar to a tensor.
        In Metal per-thread execution, every thread already has the scalar,
        so this is a no-op.

        Special case: splatting a pointer arg without addptr (e.g. for
        scalar loads like tl.load(X)) registers the result with offset "0"
        so that tt.load reads from index 0 for all threads.

        Shape tracking: the source is scalar (), the result gets the tensor
        shape from type_str (e.g. tensor<256xf32> → (256,)).  This records
        that the value has been "splatted" but each thread still holds a
        single scalar copy.
        """
        # Splat produces a value that's the same on every thread.
        self._is_splat.add(ssa.id)
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            elif "!tt.ptr" in ssa.type_str:
                # Splatting a raw pointer arg (no addptr) — all threads
                # point to the same address, so offset is 0
                self.env_is_ptr[ssa.id] = (self._lookup(src_id), "0")
        # Track splat output shape from result type
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape
        else:
            # Scalar splat (no tensor wrapper) — record as scalar
            self.env_shapes[ssa.id] = ()

    def _lower_expand_dims(self, ssa: SSAValue):
        """tt.expand_dims → passthrough with shape tracking.

        In the 2D model, expand_dims inserts a size-1 dimension.
        The per-thread value doesn't change (index remapping was done
        at make_range level by the 2D pre-pass), so this is a passthrough.

        Shape tracking: records the new shape with the inserted dimension.
        For example, tensor<64xi32> with axis=1 → tensor<64x1xi32>,
        giving shape (64, 1).

        Why this is a passthrough (not a "TODO" to fix later):
        -- 1D-per-thread model owns one scalar per lid. expand_dims doesn\'t
        change the per-thread value, only the type-level shape annotation.
        Whether this scalar represents a row index or column index of the
        eventual 2D tensor is decided by the make_range pre-pass
        (``_prescan_2d_info``), which inspects the make_range/expand_dims
        chain in the IR and assigns each thread the correct (row, col)
        decomposition. Doing it via the expand_dims axis attribute would
        be more "principled" but produces identical MSL — the prescan
        already gives us the same information. Tests covering 2D shapes
        (test_index1d, test_broadcast, test_reshape, etc.) pass through
        this passthrough.
        """
        self._emit_passthrough(ssa)
        # Track shape from the result type (overrides passthrough shape)
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape

    def _lower_broadcast(self, ssa: SSAValue):
        """tt.broadcast → passthrough with shape tracking.

        In the 2D model, broadcasting is handled implicitly:
        - make_range already computes the correct 2D index (lid/N or lid%N)
        - Intermediate values (loads, arithmetic) propagate correctly
        - The broadcast just changes the "shape" annotation

        This works because each thread's value is already the correct
        broadcast result based on the 2D index computed at make_range time.

        Shape tracking: records the broadcast target shape.  For example,
        tensor<64x1xi32> broadcast to tensor<64x128xi32> gives shape
        (64, 128).  The source shape (64, 1) → target shape (64, 128)
        tells us dimension 1 was broadcast.

        Why broadcast is a passthrough (not a "TODO" to fix later):
        -- the 1D-per-thread model makes broadcast implicit. When (M, 1) is
        broadcast to (M, N), each thread already computes its own value
        independently — the make_range pre-pass assigned each thread a
        (row, col) tuple, and expand_dims placed the value on the row axis
        only. Reading it from any column produces the same value because
        the make_range gave the value `lid / N` (the row index), which
        doesn\'t depend on the column. So all threads in the same row get
        the same value automatically. The shape annotation here exists
        only so downstream ops (like addptr emitting row*stride + col)
        can compose correctly with the broadcast dim.
        """
        if getattr(self, "_source_biased_replay", False):
            from ._biased_replay import emit_row_broadcast

            if emit_row_broadcast(self, ssa):
                return
        self._emit_passthrough(ssa)
        # Track shape from the result type (overrides passthrough shape)
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape

    def _lower_reshape(self, ssa: SSAValue):
        """tt.reshape → passthrough with optional bcast-layout rewrite.

        Most reshapes are passthrough (same per-thread value, just a type
        change). But there's a critical case for tl.sort's bitonic topk:

        When a make_range is reshaped into a slice layout (e.g.
        ``#ttg.slice<{dim = 5, parent = #blocked}>``), the reshape operates
        in the post-reduce broadcast-layout regime.  In that regime, thread
        ``lid`` no longer canonically holds element ``tensor[lid]``; it
        holds ``tensor[bcast_layout(lid)]``.  The original make_range
        variable (computed at the top of the kernel with ``(lid/stride) %
        range``) is wrong in this context.

        The fix: when the reshape output is in a slice layout AND there is
        an active bcast_layout from a preceding reduce, emit a fresh
        variable computing ``(bcast_layout / stride_below) % range_size``.
        The fresh variable shadows the original make_range value for this
        reshape SSA (and thus for all its downstream consumers).

        Otherwise (no slice layout, or source isn't a make_range), fall
        back to the passthrough behavior.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            if ssa.type_str:
                out_shape = _extract_shape(ssa.type_str)
                if out_shape:
                    self.env_shapes[ssa.id] = out_shape
            return

        # Check whether this reshape needs a bcast-layout rewrite.
        did_rewrite = self._maybe_rewrite_make_range_reshape(ssa)
        if not did_rewrite:
            self._emit_passthrough(ssa)

        # Propagate output shape from type_str for downstream reduce detection
        if ssa.type_str:
            out_shape = _extract_shape(ssa.type_str)
            if out_shape:
                self.env_shapes[ssa.id] = out_shape

    def _maybe_rewrite_make_range_reshape(self, ssa: SSAValue) -> bool:
        """Emit a bcast-layout-aware expression for a reshape-of-make_range.

        Returns True if the reshape was rewritten, False otherwise (caller
        should fall back to passthrough).
        """
        if not ssa.operand_ids:
            return False

        native = self._native_value_facts(ssa.id, op_name=ssa.op)

        # Only rewrite when the native reshape output is in a slice layout.
        # Printed aliases and mutable summaries are not evidence that this
        # result participates in the reduce's broadcast projection.
        if native.layout is None or not native.layout.startswith("#ttg.slice<"):
            return False

        # The source must trace to a make_range, and the output shape must
        # have exactly one non-1 axis (so we can interpret it as a per-axis
        # range indicator).
        out_shape = self._native_shape(ssa.id, op_name=ssa.op)
        if not out_shape or len(out_shape) < 2:
            return False
        non_one = [i for i, s in enumerate(out_shape) if s != 1]
        if len(non_one) != 1:
            return False
        dim = non_one[0]
        range_size = out_shape[dim]

        # Trace the source through passthroughs to find the make_range.
        ops_list = list(self.graph.ops)
        op_by_id = {o.id: o for o in ops_list}
        mr_id = self._trace_to_make_range(ssa.operand_ids[0], ops_list, op_by_id)
        if mr_id is None:
            return False

        # Match by exact native layout.  The reshape's output
        # layout (e.g. ``#ttg.slice<{dim = 5, parent = #blocked}>``) should
        # exactly match a reduce output in the same bitonic stage. Shape
        # compatibility alone cannot identify that stage: chained sort
        # reductions can have compatible shapes but different projections.
        layouts_by_layout = getattr(self, "_bcast_layouts_by_layout", {})
        registered = layouts_by_layout.get(native.layout)
        if registered is None:
            return False
        bcast_shape, consumer_layout = registered

        stride_below = 1
        for s in bcast_shape[dim + 1 :]:
            stride_below *= s

        var_name = self._next_var("idx")
        if stride_below == 1:
            expr = f"(({consumer_layout}) % {range_size}u)"
        else:
            expr = f"((({consumer_layout}) / {stride_below}u) % {range_size}u)"
        self.kb.raw_line(f"    uint {var_name} = {expr};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = tuple(out_shape)
        return True

    def _lower_addptr(self, ssa: SSAValue):
        """tt.addptr → pointer + offset indexing.

        tt.addptr(%ptr_tensor, %offset_tensor) computes element addresses.
        In MSL, this becomes array indexing: ptr[offset].
        We track the (base_ptr, offset) pair for use in load/store.
        Chained addptrs accumulate offsets: addptr(addptr(p, a), b) → p[a + b].
        """
        if len(ssa.operand_ids) >= 2:
            ptr_id = ssa.operand_ids[0]
            offset_id = ssa.operand_ids[1]
            offset_var = self._pointer_offset_expr(offset_id)

            # Phase 4c: MEPT array-of-offsets. If the offset operand is
            # an env_array, record env_ptr_array so tt.load / tt.store
            # can emit per-position accesses. The base pointer is always
            # scalar (one buffer); only the offset varies per array slot.
            if self.mept_enabled and offset_id in self.env_array:
                offset_arr, n, _ = self.env_array[offset_id]
                # Chained addptr with array offset: combine with any
                # existing scalar parent offset.
                parent_ptr_info = self.env_is_ptr.get(ptr_id)
                parent_ptr_array = self.env_ptr_array.get(ptr_id)
                if parent_ptr_array:
                    base_ptr, parent_arr, parent_n = parent_ptr_array
                    if parent_n == n:
                        # Both array: produce a combined-offset array.
                        # Pointer differences are signed.  ``make_range`` is
                        # represented as uint, but either side of a chained
                        # addptr may contribute a negative runtime offset.
                        # Widen before the addition so C++ usual conversions
                        # cannot turn that offset into a huge unsigned index.
                        combined_exprs = [f"long({parent_arr}[{i}]) + long({offset_arr}[{i}])" for i in range(n)]
                        combined_name = self._var_array("off", combined_exprs, "long")
                        self.env_ptr_array[ssa.id] = (base_ptr, combined_name, n)
                        # env[ssa.id] is mostly informational here;
                        # tt.load reads env_ptr_array directly.
                        self.env[ssa.id] = f"{base_ptr}[{combined_name}[0]]"
                        self._propagate_shape_elementwise(ssa)
                        return
                if parent_ptr_info:
                    base_ptr, existing_offset = parent_ptr_info
                    combined_exprs = [f"long({existing_offset}) + long({offset_arr}[{i}])" for i in range(n)]
                    combined_name = self._var_array("off", combined_exprs, "long")
                    self.env_ptr_array[ssa.id] = (base_ptr, combined_name, n)
                    self.env[ssa.id] = f"{base_ptr}[{combined_name}[0]]"
                    self._propagate_shape_elementwise(ssa)
                    return
                # No parent — base pointer is the operand directly.
                ptr_var = self._lookup(ptr_id)
                self.env_ptr_array[ssa.id] = (ptr_var, offset_arr, n)
                self.env[ssa.id] = f"{ptr_var}[{offset_arr}[0]]"
                self._propagate_shape_elementwise(ssa)
                return

            # Chained addptr: array-offset parent advanced by a SCALAR. Folding a
            # scalar into an array-of-offsets base (e.g. `x_ptr + offs*K + k`
            # inside a loop) must add the scalar into each array slot. Without
            # this, the scalar-offset fall-through below emits
            # `base[arr[0]][scalar]` — a double subscript that fails to compile.
            # MEPT-only: the non-MEPT path chains through env_is_ptr (below) and
            # flattens to a single offset correctly.
            if self.mept_enabled:
                parent_ptr_array = self.env_ptr_array.get(ptr_id)
                if parent_ptr_array:
                    base_ptr, parent_arr, parent_n = parent_ptr_array
                    combined_exprs = [f"long({parent_arr}[{i}]) + long({offset_var})" for i in range(parent_n)]
                    combined_name = self._var_array("off", combined_exprs, "long")
                    self.env_ptr_array[ssa.id] = (base_ptr, combined_name, parent_n)
                    self.env[ssa.id] = f"{base_ptr}[{combined_name}[0]]"
                    self._propagate_shape_elementwise(ssa)
                    return

            # Check if this is a chained addptr (ptr_id is itself an addptr result)
            parent_ptr_info = self.env_is_ptr.get(ptr_id)
            if parent_ptr_info:
                base_ptr, existing_offset = parent_ptr_info
                combined = f"({existing_offset} + {offset_var})"
                self.env_is_ptr[ssa.id] = (base_ptr, combined)
                self.env[ssa.id] = f"{base_ptr}[{combined}]"
            else:
                ptr_var = self._lookup(ptr_id)
                self.env_is_ptr[ssa.id] = (ptr_var, offset_var)
                self.env[ssa.id] = f"{ptr_var}[{offset_var}]"
            # Shape: addptr inherits shape from its operands (typically the
            # offset tensor dictates the shape, or the pointer tensor from
            # splat). For 2D addptr the offset arithmetic
            # (row * stride + col) is already baked into the offset operand
            # before it reaches us, so we just propagate the elementwise
            # shape — the per-thread `offset_var` already encodes the
            # threads\'s (row, col) memory address.
            self._propagate_shape_elementwise(ssa)

    # -- Load and Store --

    def _lower_load(self, ssa: SSAValue):
        """tt.load → masked buffer read with optional 'other' default value."""
        if not ssa.operand_ids:
            return

        ptr_id = ssa.operand_ids[0]

        # Phase 4c: MEPT array load. When the pointer was assembled by
        # tt.addptr with an array offset, emit per-position reads into
        # a result array. Honors mask + "other" (array or splat). FP8
        # loads emit two arrays (raw uchar + converted float) so the
        # uchar→float conversion stays per-position.
        ptr_arr_info = self.env_ptr_array.get(ptr_id)
        if self.mept_enabled and ptr_arr_info is not None:
            dtype = _mlir_to_triton_dtype(ssa.elem_type)
            from triton_msl.codegen.msl_builtins import (
                is_fp8_type,
                fp8_to_float_func,
            )

            base_ptr, off_arr, n = ptr_arr_info
            is_fp8 = is_fp8_type(dtype)
            if is_fp8:
                self._inject_fp8_device_functions(dtype)
                compute_type = "float"
                zero = "0.0f"
            else:
                compute_type = _msl_compute_type(dtype)
                zero = "0.0f" if dtype in ("fp32", "fp16", "bf16") else "0"
            # Find mask + other operands the same way the scalar
            # path does, but also accept array-form values.
            mask_var = None
            mask_is_array = False
            other_val = zero
            other_val_is_array = False
            for op_id in ssa.operand_ids[1:]:
                if op_id in self.env_is_mask or self._is_mask(op_id):
                    if op_id in self.env_array:
                        mask_var, _mn, _ = self.env_array[op_id]
                        mask_is_array = True
                    else:
                        mask_var = self._lookup(op_id)
                elif mask_var is not None:
                    if op_id in self.env_array:
                        other_val = self.env_array[op_id][0]
                        other_val_is_array = True
                    else:
                        other_val = self._lookup(op_id)
            if is_fp8:
                # Step 1: gather raw uchar values (mask-aware).
                raw_exprs = []
                for i in range(n):
                    pos = f"{base_ptr}[{off_arr}[{i}]]"
                    if mask_var is None:
                        raw_exprs.append(pos)
                    else:
                        mask_expr = f"{mask_var}[{i}]" if mask_is_array else mask_var
                        raw_exprs.append(f"{mask_expr} ? {pos} : uchar(0)")
                raw_name = self._var_array("raw", raw_exprs, "uchar")
                # Step 2: convert uchar → float per position. When
                # masked, fall back to ``other``; when unmasked,
                # convert all positions unconditionally.
                to_float = fp8_to_float_func(dtype)
                val_exprs = []
                for i in range(n):
                    convert = f"{to_float}({raw_name}[{i}])"
                    if mask_var is None:
                        val_exprs.append(convert)
                    else:
                        mask_expr = f"{mask_var}[{i}]" if mask_is_array else mask_var
                        other_expr = (
                            f"static_cast<float>({other_val}[{i}])"
                            if other_val_is_array
                            else f"static_cast<float>({other_val})"
                        )
                        val_exprs.append(f"{mask_expr} ? {convert} : {other_expr}")
                var_name = self._var_array("val", val_exprs, "float")
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, n, "float")
                self.env_types[ssa.id] = dtype
                self._propagate_shape_from_type(ssa)
                return
            exprs = []
            for i in range(n):
                pos = f"{base_ptr}[{off_arr}[{i}]]"
                loaded = f"static_cast<{compute_type}>({pos})"
                if mask_var is None:
                    exprs.append(loaded)
                    continue
                mask_expr = f"{mask_var}[{i}]" if mask_is_array else mask_var
                other_expr = (
                    f"static_cast<{compute_type}>({other_val}[{i}])"
                    if other_val_is_array
                    else f"static_cast<{compute_type}>({other_val})"
                )
                exprs.append(f"{mask_expr} ? {loaded} : {other_expr}")
            var_name = self._var_array("val", exprs, compute_type)
            self.env[ssa.id] = var_name
            self.env_array[ssa.id] = (var_name, n, compute_type)
            self.env_types[ssa.id] = dtype
            self._propagate_shape_from_type(ssa)
            return

        ptr_info = self.env_is_ptr.get(ptr_id)

        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            # Direct pointer (no addptr)
            base_ptr = self._lookup(ptr_id)
            # Scalar load (non-tensor result) → always load from index 0
            # Tensor load without addptr → use lid as offset
            offsets = "0" if not ssa.is_tensor else self._lid_expr

        # Determine dtype from pointer type
        dtype = _mlir_to_triton_dtype(ssa.elem_type)
        compute_type = _msl_compute_type(dtype)
        zero = "0.0f" if dtype in ("fp32", "fp16", "bf16") else "0"

        # Check if this is an FP8 load — needs software conversion from uchar
        from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_to_float_func

        fp8_load = is_fp8_type(dtype)
        if fp8_load:
            zero = "0.0f"  # FP8 computes in float
            self._inject_fp8_device_functions(dtype)

        # Parse operands: tt.load(ptr, mask?, other?)
        # Operands after the pointer: mask (i1 tensor), then other (default value)
        mask_var = None
        other_val = zero

        remaining_ids = ssa.operand_ids[1:]
        for op_id in remaining_ids:
            if op_id in self.env_is_mask or self._is_mask(op_id):
                mask_var = self._lookup(op_id)
            elif mask_var is not None:
                # After mask comes the 'other' value
                other_val = self._lookup(op_id)

        var_name = self._next_var("val")

        # Source replay can use the widest operand's threadgroup.
        # Lanes beyond this load's own flat tensor extent have NO source load.
        # They must rendezvous at barriers, but must not read an adjacent tile.
        if getattr(self, "_source_epilogue_replay", False) or getattr(self, "_source_biased_replay", False):
            load_shape = self._native_shape(ssa.id, op_name="tt.load")
            if len(load_shape) == 2 and load_shape[0] * load_shape[1] < self.effective_block_size:
                extent = f"({self._lid_expr} < {load_shape[0] * load_shape[1]}u)"
                mask_var = f"({extent} && ({mask_var}))" if mask_var else extent

        if fp8_load:
            # FP8: load as uchar, then convert to float
            to_float = fp8_to_float_func(dtype)
            raw_var = self._next_var("raw")
            if mask_var:
                self.kb.raw_line(f"    uchar {raw_var} = {mask_var} ? {base_ptr}[{offsets}] : uchar(0);")
                self.kb.raw_line(
                    f"    float {var_name} = {mask_var} ? {to_float}({raw_var}) : static_cast<float>({other_val});"
                )
            else:
                self.kb.raw_line(f"    uchar {raw_var} = {base_ptr}[{offsets}];")
                self.kb.raw_line(f"    float {var_name} = {to_float}({raw_var});")
        elif mask_var:
            self.kb.raw_line(
                f"    {compute_type} {var_name} = {mask_var} ? "
                f"static_cast<{compute_type}>({base_ptr}[{offsets}]) : "
                f"static_cast<{compute_type}>({other_val});"
            )
        else:
            self.kb.raw_line(f"    {compute_type} {var_name} = static_cast<{compute_type}>({base_ptr}[{offsets}]);")

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: load inherits shape from pointer operand. For 2D shapes
        # the (row, col) decomposition (row = lid / N, col = lid % N) is
        # already done by the make_range pre-pass when computing the
        # offsets that feed addptr — by the time we get here, `offsets`
        # is already the correct linearized memory index for this thread,
        # so we just emit base[offsets].
        ptr_shape = self.env_shapes.get(ptr_id)
        if ptr_shape:
            self.env_shapes[ssa.id] = ptr_shape
        else:
            self._propagate_shape_from_type(ssa)

    def _lower_store(self, ssa: SSAValue):
        """tt.store -> masked buffer write.

        When the value to store is backed by a shared-memory array (total >
        block_size), emits a cooperative strided store loop that reads from
        shared memory and writes to global memory with reconstructed 2D
        addressing.
        """
        if len(ssa.operand_ids) < 2:
            return

        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]

        # Prove the native store relation before selecting *any* emission
        # path, including the MEPT early return below.  Replay-derived array
        # lengths may corroborate this relation, but cannot establish it.
        native_ptr_shape = self._native_shape(ptr_id, op_name="tt.store")
        native_val_shape = self._native_shape(val_id, op_name="tt.store")
        if (not native_ptr_shape and native_val_shape) or (
            native_ptr_shape and native_val_shape and native_ptr_shape != native_val_shape
        ):
            raise MetalNonRecoverableError(
                f"tt.store native pointer/value shapes disagree ({native_ptr_shape} vs {native_val_shape})",
                op_name="tt.store",
            )
        native_mask_id = ssa.operand_ids[2] if len(ssa.operand_ids) >= 3 else None
        if native_mask_id is not None:
            mask_facts = self._native_value_facts(native_mask_id, op_name="tt.store")
            native_mask_shape = self._native_shape(native_mask_id, op_name="tt.store")
            if mask_facts.kind != "integer" or mask_facts.elem != "i1" or mask_facts.width != 1:
                raise MetalNonRecoverableError(
                    "tt.store native mask is not an i1 value",
                    op_name="tt.store",
                )
            if (not native_ptr_shape and native_mask_shape) or (
                native_ptr_shape and native_mask_shape and native_ptr_shape != native_mask_shape
            ):
                raise MetalNonRecoverableError(
                    f"tt.store native pointer/mask shapes disagree ({native_ptr_shape} vs {native_mask_shape})",
                    op_name="tt.store",
                )

        if getattr(self, "_source_epilogue_replay", False):
            from ._loop_epilogue import emit_store

            if emit_store(self, ssa):
                return

        if getattr(self, "_source_biased_replay", False):
            from ._biased_replay import emit_small_store

            if emit_small_store(self, ssa):
                return

        # Phase 4c: MEPT array scatter. When the pointer was assembled
        # via a tt.addptr with an array offset, emit per-position writes for
        # all n MEPT elements. The value is either an env_array of matching
        # length (write val[i]) OR a splat-like scalar/constant (broadcast the
        # same value to every position). The splat case matters because a
        # constant-0 store (e.g. embedding_dense_backward's grad zero-init)
        # registers as a SCALAR in self.env, not env_array — without handling
        # it here the store falls through to the scalar path, which can't see
        # the MEPT pointer array (it lives in env_ptr_array, not env_is_ptr)
        # and emits a malformed `ptr[off][lid]` double-subscript that fails to
        # compile. Only broadcast values KNOWN to be splat (uniform across
        # positions) — never guess an untracked array (that would be silent-wrong).
        ptr_arr_info = self.env_ptr_array.get(ptr_id)
        val_arr_info = self.env_array.get(val_id)
        val_is_splat = val_id in self._is_splat
        if (
            self.mept_enabled
            and ptr_arr_info is not None
            and (
                (val_arr_info is not None and ptr_arr_info[2] == val_arr_info[1])
                or (val_arr_info is None and val_is_splat)
            )
        ):
            base_ptr, off_arr, n = ptr_arr_info
            if val_arr_info is not None:
                _val_arr = val_arr_info[0]

                def _mept_val_at(i, _a=_val_arr):
                    return f"{_a}[{i}]"
            else:
                # Splat/constant: same scalar written to every MEPT position.
                _splat_val = self._lookup(val_id)

                def _mept_val_at(i, _v=_splat_val):
                    return _v

            # Optional mask is the third operand if present.
            mask_var = None
            mask_is_array = False
            if native_mask_id is not None:
                if native_mask_id in self.env_array:
                    mask_var, mask_n, _ = self.env_array[native_mask_id]
                    if mask_n != n:
                        raise MetalNonRecoverableError(
                            "tt.store MEPT mask register-array width does not match the pointer/value width",
                            op_name="tt.store",
                        )
                    mask_is_array = True
                else:
                    # Legal Triton IR requires a tensor mask for this tensor
                    # pointer. A scalar emission value can represent that
                    # tensor only when its producer proved a uniform splat;
                    # otherwise one lane's predicate would be broadcast over
                    # every register-array position.
                    if native_mask_shape and native_mask_id not in self._is_splat:
                        raise MetalNonRecoverableError(
                            "tt.store MEPT tensor mask has no per-position array and is not a proved uniform splat",
                            op_name="tt.store",
                        )
                    mask_var = self._lookup(native_mask_id)
            # Cast each value to the buffer element type, exactly as the
            # scalar store does. Values are computed in `float`; narrowing
            # to bf16 (and fp8) needs an explicit cast — MSL won't
            # implicitly convert float -> bfloat (it does for half, which
            # is why only bf16 surfaced this). The MEPT pointer lives in
            # env_ptr_array (not env_is_ptr), so _trace_ptr_dtype can't
            # see it — derive the dtype from the base buffer arg directly.
            store_dtype = "fp32"
            for arg in self.graph.args:
                if arg.name == base_ptr and arg.is_ptr:
                    store_dtype = _mlir_to_triton_dtype(arg.elem_type)
                    break
            for i in range(n):
                cast_val = self._fp8_cast_val(_mept_val_at(i), store_dtype)
                write = f"{base_ptr}[{off_arr}[{i}]] = {cast_val};"
                if mask_var is None:
                    self.kb.raw_line(f"    {write}")
                else:
                    cond = f"{mask_var}[{i}]" if mask_is_array else mask_var
                    self.kb.raw_line(f"    if ({cond}) {{ {write} }}")
            return

        # Every remaining store decision that depends on logical extent uses
        # immutable native pointer/value facts. ``env_shapes`` is a replay
        # cache populated while emitting earlier ops; it is not an ownership
        # or representation proof and must not select guards or refusals.
        # Check if the value to store is smem-backed with total > block_size
        smem_descs = getattr(self, "_shared_mem_descs", {})
        val_smem = smem_descs.get(val_id)
        if val_smem is None:
            # The store value may be a dtype CAST / layout convert of a smem-backed
            # dot result (e.g. for an fp16/bf16 output: store(convert_layout(
            # truncf(dot)))). Those ops' results are not themselves smem-backed, so
            # without following them the cooperative strided-store loop below is
            # skipped and the store collapses to a flat per-thread write -- silently
            # dropping every element past block_size (only block_size of a
            # total>block_size tile get written; the rest stay at their init value).
            # Walk the passthrough (convert_layout) + dtype-cast chain to a
            # smem-backed source for the LOOP decision; the loop re-casts smem[_st]
            # to the output pointer dtype, so the narrowing is preserved. Do NOT
            # rebind val_id (the total<=block_size flat path needs the cast value).
            # Re-audit 2026-06-27.
            _FOLLOW = {
                "ttg.convert_layout",
                "arith.truncf",
                "arith.extf",
                "tt.fp_to_fp",
                "arith.sitofp",
                "arith.uitofp",
                "arith.fptosi",
                "arith.fptoui",
                "arith.trunci",
                "arith.extsi",
                "arith.extui",
            }

            def _flat_ops(ops):
                for o in ops:
                    yield o
                    if o.region_ops:
                        yield from _flat_ops(o.region_ops)
                    if o.else_ops:
                        yield from _flat_ops(o.else_ops)

            _by_id = {o.id: o for o in _flat_ops(self.graph.ops)}
            _cur = val_id
            for _ in range(8):
                _o = _by_id.get(_cur)
                if _o is None or _o.op not in _FOLLOW or not _o.operand_ids:
                    break
                _cur = _o.operand_ids[0]
                val_smem = smem_descs.get(_cur)
                if val_smem is not None:
                    break
        dispatch_threads = getattr(self, "_actual_dispatch_threads", self.effective_block_size)
        if val_smem:
            val_shape = val_smem[1]
            val_total = 1
            for d in val_shape:
                val_total *= d
            if val_total > dispatch_threads and len(val_shape) >= 2:
                smem_name = val_smem[0]
                M, N = val_shape[0], val_shape[1]

                # Get mask if provided
                mask_id = native_mask_id

                # Get the pointer info
                ptr_info = self.env_is_ptr.get(ptr_id)
                if ptr_info:
                    base_ptr, offset_expr = ptr_info

                    # Rebuild the output address per (_fill_row, _fill_col)
                    # STRUCTURALLY via the single-source per-load helper -- the
                    # same one ttg.local_alloc staging uses. This replaces the
                    # old global-_make_range_dim + string-`.replace` heuristic
                    # (the buggy "third twin"): the helper walks THIS store's
                    # addptr -> (broadcast) -> expand_dims -> make_range chain
                    # and refuses (correct-or-refuse) rather than guess a
                    # possibly-transposed address.
                    def _all_ops(ops):
                        for o in ops:
                            yield o
                            if o.region_ops:
                                yield from _all_ops(o.region_ops)
                            if o.else_ops:
                                yield from _all_ops(o.else_ops)

                    all_ops = list(_all_ops(self.graph.ops))
                    op_by_id = {o.id: o for o in all_ops}

                    new_offset = self._rebuild_staged_fill_offset(ptr_id, op_by_id, base_ptr, M, N)

                    # Masked over-threadgroup store: reconstruct the per-element
                    # mask STRUCTURALLY via the same term-walk (row/col bounds
                    # comparison) so a boundary store stays correct, instead of
                    # the OLD code which silently DROPPED the mask (a latent OOB
                    # write for non-block-multiple bounds) or hardcoded
                    # `(_fill_row + r_8) < N_CTX`. Unresolvable masks refuse.
                    mask_expr = None
                    if mask_id is not None:
                        mask_expr = self._rebuild_staged_fill_mask(mask_id, op_by_id, M, N)

                    self.kb.raw_line(f"    for (uint _st = lid; _st < {val_total}u; _st += {dispatch_threads}u) {{")
                    self.kb.raw_line(f"        uint _fill_row = _st / {N}u;")
                    self.kb.raw_line(f"        uint _fill_col = _st % {N}u;")

                    store_val = f"{smem_name}[_st]"
                    store_dtype = self._trace_ptr_dtype(ptr_id)
                    store_type = triton_type_to_msl(store_dtype)
                    compute_type = _msl_compute_type(store_dtype)
                    if store_type != compute_type:
                        store_val = f"static_cast<{store_type}>({store_val})"

                    if mask_expr is not None:
                        self.kb.raw_line(f"        if ({mask_expr}) {base_ptr}[{new_offset}] = {store_val};")
                    else:
                        self.kb.raw_line(f"        {base_ptr}[{new_offset}] = {store_val};")
                    self.kb.raw_line(f"    }}")
                    return

        # Detect reduce keep_dims pattern: store to (M, 1) or (1, N) shaped pointer
        # where the value comes from a reduce. The generic 2D index decomposition
        # is broken for this case, so we use guarded lid-based indexing instead.
        # Skip when dim_0 == 1 (triton_per_* pattern): the ptr already has
        # the row offset from addptr, and using base_ptr[idx] would lose it.
        ptr_shape = native_ptr_shape
        val_shape = native_val_shape
        if (
            self._is_2d
            and ptr_shape
            and len(ptr_shape) == 2
            and (ptr_shape[0] == 1 or ptr_shape[1] == 1)
            and ptr_shape[0] != ptr_shape[1]
            and ptr_shape[0] != 1
        ):
            result_size = max(ptr_shape)
            ptr_info = self.env_is_ptr.get(ptr_id)
            if ptr_info:
                base_ptr, offsets = ptr_info
            else:
                base_ptr = self._lookup(ptr_id)
                offsets = self._lid_expr
            val_var = self._lookup(val_id)
            store_dtype = self._trace_ptr_dtype(ptr_id)
            cast_val = self._fp8_cast_val(val_var, store_dtype)
            idx = self._lid_expr
            # In 2D kernels, lid maps to row index via lid/N where N is the
            # inner dimension.  The guard must cover all threads whose
            # lid/N < result_size, i.e. lid < result_size * N.  Using just
            # result_size (the number of rows) cuts off the upper threads
            # and leaves half the rows unwritten when N > 1.
            guard_size = result_size
            if self._effective_2d_shape and len(self._effective_2d_shape) == 2:
                inner_N = self._effective_2d_shape[1]
                guard_size = result_size * inner_N
            self.kb.raw_line(f"    if ({idx} < {guard_size}u) {base_ptr}[{offsets}] = {cast_val};")
            return

        # Refuse a BLOCK-wide tensor store the base path can only emit as one
        # element per thread (PTR[k + lid] = val). When the tile exceeds the
        # threadgroup (block_size > num_threads, i.e. n>1 elements per thread),
        # that silently covers only `lid` of each tile-stride and DROPS the
        # rest — a silent-wrong. The correct n>1 store paths are the MEPT
        # register-array scatter (_mept_single_pass), the smem-backed path
        # above, and _loop_e-wrapped stores (_needs_wrapping); none apply here.
        # Refuse loudly: the user must launch with num_warps = BLOCK/32 so
        # num_threads == BLOCK (n=1), or reduce BLOCK.
        _val_shape = native_val_shape
        _num_threads = self.kb.block_size
        if (
            not self._mept_single_pass
            and not self._needs_wrapping
            and _val_shape is not None
            and len(_val_shape) >= 1
            and _val_shape[0] > _num_threads
        ):
            raise MetalNonRecoverableError(
                f"Refusing a {_val_shape[0]}-element tensor store with only "
                f"{_num_threads} threads: the base path stores one element per "
                f"thread, so a tile wider than the threadgroup would silently "
                f"drop the rest. Launch with num_warps = BLOCK/32 (so "
                f"num_threads == BLOCK), or reduce BLOCK."
            )

        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            # Scalar pointer (direct arg, not tensor of pointers) → offset 0
            # Tensor pointer → use lid as offset
            offsets = "0" if self._is_scalar_ptr(ptr_id) else self._lid_expr

        val_var = self._lookup(val_id)

        # Determine storage type
        # Trace back to the function arg to find the pointer dtype
        store_dtype = self._trace_ptr_dtype(ptr_id)
        cast_val = self._fp8_cast_val(val_var, store_dtype)

        # Get mask if provided
        mask_var = None
        if native_mask_id is not None:
            mask_var = self._lookup(native_mask_id)

        # In 2D kernels, 1D store tensors must be guarded to prevent
        # duplicate writes from extra threads (e.g. after 2D→1D reduce).
        store_1d_guard = None
        # Check if the kernel has any ttg.convert_layout that did a real
        # shared memory redistribution. If so, all 1D stores in this kernel
        # should use simple lid < N guards because the convert_layout
        # changed the thread-to-element mapping to simple (thread i = element i).
        val_converted = hasattr(self, "_converted_layout_ids") and bool(getattr(self, "_converted_layout_ids", set()))
        if self._is_2d and not self._is_scalar_ptr(ptr_id):
            store_shape = native_ptr_shape
            if store_shape and len(store_shape) == 1 and store_shape[0] < self.effective_block_size:
                store_1d_guard = store_shape[0]

        # The wrapping regime emits ONE loop over the kernel's dominant tile, so every
        # op inside it runs that many times. A store whose own tile is SMALLER (a
        # histogram's bins, a small output beside a large input) then runs past the end
        # of its buffer and corrupts whatever follows -- silently, since the store's own
        # elements are still written correctly. Guard by the store's own extent.
        store_wrap_guard = None
        # Only meaningful when the store index is driven by the lid / wrap variable, so
        # that bounding that variable actually bounds the write.
        _idx_var = self._lid_expr
        _idx_driven = _idx_var == str(offsets) or _idx_var in re.findall(r"[A-Za-z_]\w*", str(offsets))
        if _idx_driven and not self._is_scalar_ptr(ptr_id):
            wrap_shape = native_ptr_shape
            if wrap_shape:
                wrap_total = 1
                for _d in wrap_shape:
                    wrap_total *= _d
                # How many distinct index values actually execute this store: the wrap
                # extent when wrapping, otherwise the dispatched threadgroup.
                span = (
                    getattr(self, "_total_elements", 0)
                    if getattr(self, "_needs_wrapping", False)
                    else self.effective_block_size
                )
                if 0 < wrap_total < span:
                    store_wrap_guard = wrap_total

        def remap_blocked_address(inner):
            # The result is broadcast at row=lid/inner, whereas the pointer
            # and mask may have been computed from a separate direct 1-D range.
            # Replay THEIR SSA cones at that row, preserving every scalar base,
            # stride, integer conversion and predicate. Reusing the old offset
            # keeps the wrong coordinate; replacing it drops the pointer DAG.
            import copy

            replay = copy.copy(self)
            # Emission shares the builder, but store-local names/layout facts
            # must not replace values used by later source operations.
            for key, value in vars(self).items():
                if isinstance(value, (dict, set)):
                    setattr(replay, key, value.copy())
            by_id = self._layout_op_index()
            done = set()

            def visit(vid):
                if vid in done:
                    return
                done.add(vid)
                op = by_id.get(vid)
                shape = self._native_shape(vid, op_name="tt.store address")
                if not shape:
                    return  # Scalar args and program/loop offsets.
                if op is not None and op.op == "arith.constant":
                    # Only a parsed scalar/splat is coordinate-invariant. A
                    # dense tensor mask cannot be treated as coordinate-
                    # invariant merely because its operation is a constant.
                    if isinstance(op.attrs.get("value"), (bool, int, float)):
                        return
                    raise MetalNonRecoverableError(
                        "blocked 1-D store address/mask contains a nonuniform or unparsed tensor constant",
                        op_name="tt.store",
                    )
                if op is not None and op.op == "tt.make_range":
                    start = int(op.attrs.get("start", 0))
                    row = f"({self._lid_expr} / {inner}u)"
                    replay.env[vid] = f"({row} + {start}u)" if start else row
                    return
                wrappers = ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout")
                if op is not None and (op.op in wrappers or op.op == "tt.addptr" or op.op.startswith("arith.")):
                    for operand in op.operand_ids:
                        visit(operand)
                    if op.op in ("ttg.convert_layout", "tt.broadcast", "tt.expand_dims"):
                        replay._emit_passthrough(op)
                    else:
                        replay._lower_op_dispatch(op)
                    return
                # An already blocked/uniform value has the requested row.
                # Do not replay loads, reductions or effects at a different time.
                if self._value_1d_layout_of(vid) in ("blocked", "any"):
                    return
                raise MetalNonRecoverableError(
                    "blocked 1-D store address/mask depends on a tensor whose "
                    "row coordinate cannot be reconstructed without replaying "
                    "a load or effect",
                    op_name="tt.store",
                )

            try:
                visit(ptr_id)
                if native_mask_id is not None:
                    visit(native_mask_id)
                ptr = replay.env_is_ptr.get(ptr_id)
                if ptr is None:
                    raise MetalNonRecoverableError("blocked 1-D store lacks its traced pointer", op_name="tt.store")
                mask = replay._lookup(native_mask_id) if native_mask_id is not None else None
                return ptr[0], ptr[1], mask
            finally:
                self._var_counter = replay._var_counter

        clauses = []
        if store_1d_guard is not None:
            lid = self._lid_expr
            # Did the value come from a 2-D reduce whose broadcast layout we recorded?
            # That is authoritative; the size-matching below cannot tell an axis=0 result
            # from an axis=1 one on a square tile.
            _rec_layout = self._value_1d_layout_of(val_id)
            if self._traces_to_converted_layout(val_id):
                # THIS value was redistributed: thread i has element i. Simple guard.
                clauses.append(f"{lid} < {store_1d_guard}u")
            elif _rec_layout in ("direct", "any"):
                # "any" = uniform across elements (constant/splat-derived): every
                # thread holds the same value, so the direct store is correct.
                # Thread i holds element i (axis=0 reduce broadcast as shared[lid % N],
                # or a tt.split pair). Store from the first `size` threads, index by lid.
                clauses.append(f"{lid} < {store_1d_guard}u")
            elif _rec_layout == "blocked":
                # Thread i holds element i / N. One thread per row block.
                shape = self._effective_2d_shape
                if shape and len(shape) >= 2 and shape[1] > 0:
                    N = shape[1]
                    base_ptr, offsets, mask_var = remap_blocked_address(N)
                    clauses.append(f"{lid} % {N}u == 0u && {lid} / {N}u < {store_1d_guard}u")
                else:
                    clauses.append(f"{lid} < {store_1d_guard}u")
            elif val_converted:
                # Kernel-wide convert_layout flag: blunter than the per-value check
                # above, and reached only when no producer declared. Reported so its
                # remaining reach can be measured and shrunk.
                self._report_undeclared_layout("kernel-wide-convert-flag", store_1d_guard)
                clauses.append(f"{lid} < {store_1d_guard}u")
            else:
                # UNDECLARED: no producer told us this value's layout, so fall back to
                # inferring it from the result's SIZE. That inference is what produced
                # three separate silent-wrongs (reduce-probe #5/#6, re-audit #8), so it
                # is a last resort, not a design.
                self._report_undeclared_layout("size-inference", store_1d_guard)
                shape = self._effective_2d_shape
                if shape and len(shape) >= 2 and store_1d_guard == shape[0] and shape[1] > 0:
                    N = shape[1]
                    base_ptr, offsets, mask_var = remap_blocked_address(N)
                    clauses.append(f"{lid} % {N}u == 0u && {lid} / {N}u < {store_1d_guard}u")
                else:
                    clauses.append(f"{lid} < {store_1d_guard}u")
        if store_wrap_guard is not None:
            clauses.append(f"{self._lid_expr} < {store_wrap_guard}u")
        if mask_var:
            clauses.append(str(mask_var))

        if clauses:
            self.kb.raw_line(f"    if ({' && '.join(clauses)}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
        else:
            self.kb.raw_line(f"    {base_ptr}[{offsets}] = {cast_val};")

    def _fp8_cast_val(self, val_var: str, store_dtype: str) -> str:
        """Return the MSL expression to convert a float value to FP8 uchar.

        If store_dtype is not FP8, returns a regular static_cast or passthrough.
        """
        from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_from_float_func

        if is_fp8_type(store_dtype):
            self._inject_fp8_device_functions(store_dtype)
            return f"{fp8_from_float_func(store_dtype)}({val_var})"
        store_type = triton_type_to_msl(store_dtype)
        compute_type = _msl_compute_type(store_dtype)
        if store_type != compute_type:
            return f"static_cast<{store_type}>({val_var})"
        return val_var

    def _inject_fp8_device_functions(self, dtype: str):
        """Inject FP8 conversion device functions into the kernel builder."""
        from triton_msl.codegen.msl_builtins import fp8_device_functions

        if not hasattr(self, "_fp8_injected"):
            self._fp8_injected = set()
        if dtype not in self._fp8_injected:
            self._fp8_injected.add(dtype)
            for fn_src in fp8_device_functions(dtype):
                self.kb._device_functions.append(fn_src)

    def _trace_ptr_dtype(self, ptr_id: int) -> str:
        """Trace a pointer SSA value back to its function arg dtype."""
        # ptr_id might be an addptr result
        info = self.env_is_ptr.get(ptr_id)
        if info:
            base_name = info[0]
            # Find the function arg with this name
            for arg in self.graph.args:
                if arg.name == base_name and arg.is_ptr:
                    return _mlir_to_triton_dtype(arg.elem_type)

        # Direct lookup from env
        if ptr_id in self.env_types:
            return self.env_types[ptr_id]

        return "fp32"

    def _is_mask(self, ssa_id: int) -> bool:
        """Check if an SSA value is a boolean mask."""
        # Entry arguments have no defining operation. Searching only graph.ops
        # silently discarded a runtime scalar i1 load mask (False read memory
        # instead of returning `other`). Bind their role to native owned facts,
        # not the argument name, replay cache, or a printed signature.
        if any(arg.id == ssa_id for arg in self.graph.args):
            facts = self._native_value_facts(ssa_id, op_name="mask")
            return (
                not facts.is_tensor
                and facts.shape == ()
                and facts.kind == "integer"
                and facts.elem == "i1"
                and facts.width == 1
            )
        if ssa_id in self.env_is_mask:
            return True

        # Check type: i1 or tensor<...xi1> (exact match, not substring).
        # Walk nested regions too — ops inside scf.for / scf.if bodies live
        # in region_ops, not at the top level, so a top-level-only scan
        # misses masks defined inside a K-loop.
        def _find(ops):
            for ssa in ops:
                if ssa.id == ssa_id:
                    return ssa
                if ssa.region_ops:
                    hit = _find(ssa.region_ops)
                    if hit is not None:
                        return hit
                if ssa.else_ops:
                    hit = _find(ssa.else_ops)
                    if hit is not None:
                        return hit
            return None

        ssa = _find(self.graph.ops)
        if ssa is not None:
            return ssa.elem_type == "i1" or ssa.op in ("arith.cmpi", "arith.cmpf")
        return False

    def _is_scalar_ptr(self, ssa_id: int) -> bool:
        """Check if an SSA value is a scalar pointer (not a tensor of pointers).

        A scalar pointer like !tt.ptr<i32> should be indexed at [0],
        while a tensor of pointers like tensor<256x!tt.ptr<i32>> uses [lid].
        """
        # Check function args first
        for arg in self.graph.args:
            if arg.id == ssa_id:
                return arg.is_ptr and "tensor<" not in arg.type_str
        # Check ops
        for ssa in self.graph.ops:
            if ssa.id == ssa_id:
                return "!tt.ptr" in ssa.type_str and "tensor<" not in ssa.type_str
        return False

    # -- Constants --

    def _lower_constant(self, ssa: SSAValue):
        """arith.constant → literal value.

        Handles int, float, bool, and hex-encoded IEEE 754 bit patterns
        (MLIR uses hex integers for special floats like inf/nan).
        """
        import math
        import struct as _struct

        value = ssa.attrs.get("value")
        # Only a decoded scalar/splat literal is uniform. Non-splat DenseAttr
        # text is not a scalar truth value (e.g. [true,false] is nonempty but
        # must not become an all-true store mask). Missing values aren't zero.
        if isinstance(value, str) and ssa.elem_type == "i1" and value in ("true", "false", "-1", "0", "1"):
            value = value not in ("false", "0")
        if not isinstance(value, (bool, int, float)):
            raise MetalNonRecoverableError(
                "nonuniform or unparsed tensor constant (or unknown scalar literal) "
                "cannot be lowered as a uniform scalar; a decoded numeric scalar/splat is required",
                op_name="arith.constant",
            )
        self._is_splat.add(ssa.id)
        var_name = self._next_var("c")

        # Check if this is a hex integer that should be interpreted as float
        is_float_type = ssa.elem_type in ("f32", "f16", "bf16", "f64")
        if isinstance(value, int) and is_float_type:
            # Hex-encoded IEEE 754 bit pattern — width depends on elem_type.
            # MLIR encodes special floats (inf/nan) as hex integers of the
            # corresponding float type's width, so we must unpack using the
            # matching width (not always f32) to correctly recover NaN/Inf.
            try:
                if ssa.elem_type == "f64":
                    float_val = _struct.unpack("<d", _struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))[0]
                elif ssa.elem_type == "f16":
                    float_val = _struct.unpack("<e", _struct.pack("<H", value & 0xFFFF))[0]
                elif ssa.elem_type == "bf16":
                    # bfloat16: upper 16 bits of an f32 bit pattern
                    float_val = _struct.unpack("<f", _struct.pack("<I", (value & 0xFFFF) << 16))[0]
                else:  # f32
                    float_val = _struct.unpack("<f", _struct.pack("<I", value & 0xFFFFFFFF))[0]
            except _struct.error:
                float_val = 0.0

            if math.isinf(float_val):
                msl_val = "INFINITY" if float_val > 0 else "(-INFINITY)"
            elif math.isnan(float_val):
                msl_val = "NAN"
            else:
                msl_val = f"{float_val}f"

            if ssa.is_tensor:
                self.env[ssa.id] = msl_val
                self.env_types[ssa.id] = "fp32"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    float {var_name} = {msl_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
            self.env_shapes[ssa.id] = ()
            return

        # Determine type and format
        if isinstance(value, bool) or (isinstance(value, str) and value in ("true", "false")) or ssa.elem_type == "i1":
            # i1 must render as 0/1 (or true/false). An i1 all-ones constant arrives as
            # the integer -1 (or "-1"); the int path below would emit it verbatim, and
            # `mask ^ -1` (bool NOT via arith.xori) is then ALWAYS truthy — re-audit #10:
            # fp32 atomic_min/max ran BOTH the signed and unsigned branches. Map any i1
            # value to its truthiness so the bool render (1/0) is correct.
            if isinstance(value, bool):
                bool_val = "true" if value else "false"
            elif isinstance(value, str) and value in ("true", "false"):
                bool_val = value
            else:
                try:
                    bool_val = "true" if int(str(value).strip()) != 0 else "false"
                except (TypeError, ValueError):
                    bool_val = "true" if value else "false"
            if ssa.is_tensor:
                # Tensor bool: store as int (1/0) for SIMD reduction compatibility
                int_val = "1" if bool_val == "true" else "0"
                self.env[ssa.id] = int_val
                self.env_types[ssa.id] = "i1"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    bool {var_name} = {bool_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i1"
        elif isinstance(value, int):
            is_i64 = ssa.elem_type == "i64" or abs(value) > 0x7FFFFFFF
            int_type = "long" if is_i64 else "int"
            int_dtype = "i64" if is_i64 else "i32"
            if ssa.is_tensor:
                self.env[ssa.id] = str(value)
                self.env_types[ssa.id] = int_dtype
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    {int_type} {var_name} = {value};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = int_dtype
        elif isinstance(value, float):
            if math.isinf(value):
                msl_val = "INFINITY" if value > 0 else "(-INFINITY)"
            elif math.isnan(value):
                msl_val = "NAN"
            else:
                msl_val = f"{value}f"
            if ssa.is_tensor:
                self.env[ssa.id] = msl_val
                self.env_types[ssa.id] = "fp32"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    float {var_name} = {msl_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        else:
            self.env[ssa.id] = str(value)
            self.env_types[ssa.id] = "i32"
        # Shape: constants are scalar unless they have a tensor type
        self._propagate_shape_from_type(ssa)

    # -- Arithmetic ops --

    def _lower_arith(self, ssa: SSAValue):
        """Lower arith.* operations."""
        op = ssa.op
        ids = ssa.operand_ids

        if op in ("arith.addf", "arith.addi"):
            self._emit_binary(ssa, "+")
        elif op in ("arith.subf", "arith.subi"):
            self._emit_binary(ssa, "-")
        elif op in ("arith.mulf", "arith.muli"):
            self._emit_binary(ssa, "*")
        elif op == "arith.divf":
            self._emit_binary(ssa, "/")
        elif op == "arith.divsi":
            self._emit_binary(ssa, "/")
        elif op == "arith.divui":
            self._emit_binary(ssa, "/", force_unsigned=True)
        elif op == "arith.remsi":
            self._emit_binary(ssa, "%")
        elif op == "arith.remui":
            self._emit_binary(ssa, "%", force_unsigned=True)
        elif op == "arith.remf":
            self._emit_builtin_binary(ssa, "fmod")
        elif op == "arith.negf":
            self._emit_unary(ssa, "-")
        elif op == "arith.maxsi":
            self._emit_builtin_binary(ssa, "max")
        elif op == "arith.maxui":
            self._emit_builtin_binary(ssa, "max", force_unsigned=True)
        elif op == "arith.minsi":
            self._emit_builtin_binary(ssa, "min")
        elif op == "arith.minui":
            self._emit_builtin_binary(ssa, "min", force_unsigned=True)
        # NaN-quiet min/max (IEEE 754 minNum/maxNum): return the non-NaN operand.
        # The deprecated arith.maxf/minf (never emitted by Triton 3.7) are the
        # NaN-quiet variant (llvm.maxnum lineage) too, so they map here — kept OFF
        # the integer max()/min() path, which would mis-handle float NaN (re-audit #4).
        elif op in ("arith.maxnumf", "arith.maxf"):
            self._emit_builtin_binary(ssa, "fmax")
        elif op in ("arith.minnumf", "arith.minf"):
            self._emit_builtin_binary(ssa, "fmin")
        # NaN-propagating min/max: if either operand is NaN, result is NaN
        elif op == "arith.maximumf":
            self._emit_nan_propagating_minmax(ssa, "fmax")
        elif op == "arith.minimumf":
            self._emit_nan_propagating_minmax(ssa, "fmin")
        elif op == "arith.cmpi":
            self._lower_cmpi(ssa)
        elif op == "arith.cmpf":
            self._lower_cmpf(ssa)
        elif op == "arith.select":
            self._lower_select(ssa)
        elif op == "arith.extf":
            self._lower_extf(ssa)
        elif op == "arith.truncf":
            self._lower_truncf(ssa)
        elif op == "arith.sitofp":
            self._emit_cast(ssa, "float")
            self.env_types[ssa.id] = "fp32"
        elif op == "arith.uitofp":
            self._emit_uitofp(ssa)
            self.env_types[ssa.id] = "fp32"
        elif op in ("arith.fptosi",):
            msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
            self._emit_cast(ssa, msl_ty, dtype=dtype)
        elif op == "arith.fptoui":
            msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
            self._emit_cast(ssa, msl_ty, dtype=dtype)
        elif op == "arith.extsi":
            self._emit_int_cast(ssa, unsigned=False)
        elif op == "arith.extui":
            self._emit_int_cast(ssa, unsigned=True)
        elif op in ("arith.trunci",):
            self._emit_int_cast(ssa, unsigned=False)
        elif op in ("arith.index_cast", "arith.index_castui"):
            self._emit_cast(ssa, "int")
            self.env_types[ssa.id] = "i32"
        elif op == "arith.bitcast":
            self._lower_arith_bitcast(ssa)
        elif op == "arith.andi":
            self._emit_binary(ssa, "&")
        elif op == "arith.ori":
            self._emit_binary(ssa, "|")
        elif op == "arith.xori":
            self._emit_binary(ssa, "^")
        elif op == "arith.shli":
            self._emit_binary(ssa, "<<")
        elif op == "arith.shrsi":
            self._emit_binary(ssa, ">>")
        elif op == "arith.shrui":
            self._emit_binary(ssa, ">>", force_unsigned=True)
        else:
            self.kb.unsupported(f"UNSUPPORTED arith: {op}")

    def _propagate_bcast_layout_binary(self, ssa: SSAValue) -> None:
        """Propagate `_bcast_layout` from operands to the result of an
        elementwise binary op.

        Rules:
          1. Both operands have the same layout → keep it.
          2. Both different layouts → pick the one with the larger shape
             (the broader frame); this happens during chained N-D reduces
             in tl.sort where 8D ⊃ 7D layouts xor within a stage.
          3. One operand has a layout and the other does not → propagate
             (matching the original simple behavior that many existing
             patterns depend on, including scf.for loop-carried accums
             with constant init).
        """
        if len(ssa.operand_ids) < 2:
            return
        a_id = ssa.operand_ids[0]
        b_id = ssa.operand_ids[1]
        a_lay = self._bcast_layout.get(a_id)
        b_lay = self._bcast_layout.get(b_id)
        if a_lay is None and b_lay is None:
            return
        if a_lay is not None and b_lay is not None:
            if a_lay == b_lay:
                self._bcast_layout[ssa.id] = a_lay
                # The result inherits splat-ness if both operands are splats.
                if a_id in self._is_splat and b_id in self._is_splat:
                    self._is_splat.add(ssa.id)
                return
            # Different layouts — pick broader (more elements).
            shapes_by_expr = getattr(self, "_bcast_shapes_by_expr", {})
            a_shapes = shapes_by_expr.get(a_lay, set())
            b_shapes = shapes_by_expr.get(b_lay, set())
            a_shape = max(a_shapes, key=_shape_numel) if a_shapes else None
            b_shape = max(b_shapes, key=_shape_numel) if b_shapes else None
            if a_shape is not None and b_shape is not None:
                self._bcast_layout[ssa.id] = a_lay if _shape_numel(a_shape) >= _shape_numel(b_shape) else b_lay
            else:
                self._bcast_layout[ssa.id] = a_lay
            return
        # Exactly one operand has layout.  Check whether the unlaid operand
        # is "splat-like" (same value on every thread: constant, tt.splat,
        # or transitively derived from them).  Splat operand preserves
        # broadcast redundancy; canonical per-thread operand collapses it.
        other_id = b_id if a_lay is not None else a_id
        lay = a_lay if a_lay is not None else b_lay
        if other_id in self._is_splat:
            self._bcast_layout[ssa.id] = lay
            return
        other_shape = self._native_shape(other_id, op_name=ssa.op)
        if other_shape == ():
            # Scalar — preserve layout.
            self._bcast_layout[ssa.id] = lay
            return
        # A full per-thread-distinct canonical tensor absorbs the layout
        # back to canonical. Drop.
        # (This matches tl.sort's `val_load ^ reduce_result` pattern.)
        return

    def _is_float_op(self, ssa: SSAValue) -> bool:
        """Check if an SSA op produces a float result."""
        # Check element type first (most reliable)
        if ssa.elem_type in ("f32", "f16", "bf16", "f64"):
            return True
        # FP8 MLIR element types (f8E4M3FN, f8E5M2, etc.) also produce floats
        if ssa.elem_type and ssa.elem_type.startswith("f8"):
            return True
        # Check op suffix
        if ssa.op.endswith("f") or ssa.op.endswith("fp"):
            return True
        # Check operand types
        from triton_msl.codegen.msl_builtins import is_fp8_type

        for op_id in ssa.operand_ids[:2]:
            dtype = self.env_types.get(op_id)
            if dtype and (dtype.startswith("fp") or dtype in ("bf16",) or is_fp8_type(dtype)):
                return True
        return False

    def _lower_clampf(self, ssa: SSAValue):
        """tt.clampf → clamp(x, min, max) with optional NaN propagation.

        propagateNan = "none": fmin(fmax(x, min), max)  (NaN-quiet)
        propagateNan = "all":  NaN if x is NaN, else fmin(fmax(x, min), max)
        """
        if len(ssa.operand_ids) < 3:
            return
        x = self._lookup(ssa.operand_ids[0])
        lo = self._lookup(ssa.operand_ids[1])
        hi = self._lookup(ssa.operand_ids[2])
        var_name = self._next_var("r")
        propagate = ssa.attrs.get("propagateNan", "none")
        if propagate == "all":
            # NaN-propagating: if x is NaN, result is NaN
            self.kb.raw_line(f"    float {var_name} = isnan({x}) ? NAN : fmin(fmax({x}, {lo}), {hi});")
        else:
            # NaN-quiet: standard clamp
            self.kb.raw_line(f"    float {var_name} = fmin(fmax({x}, {lo}), {hi});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: clamp is element-wise
        self._propagate_shape_elementwise(ssa)

    def _lower_tt_bitcast(self, ssa: SSAValue):
        """tt.bitcast → reinterpret bits or change pointer element type.

        Handles both pointer bitcasts (passthrough) and value bitcasts
        (float <-> int requiring MSL as_type<>()).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]

        # Pointer bitcast — preserve ptr tracking
        if src_id in self.env_is_ptr:
            self._emit_passthrough(ssa)
            self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Update type to destination
            self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type) if ssa.elem_type else "i32"
            return

        # Check if pointer type in type_str (ptr-to-ptr bitcast)
        if "!tt.ptr" in ssa.type_str:
            self._emit_passthrough(ssa)
            return

        # Value bitcast — delegate to arith.bitcast handler
        self._lower_arith_bitcast(ssa)

    def _lower_arith_bitcast(self, ssa: SSAValue):
        """arith.bitcast → reinterpret bits without changing value.

        When source and destination types differ (float <-> int),
        emit as_type<T>() in MSL. MSL's as_type<>() requires matching
        bit widths, so we pick the MSL destination type to match the
        source value's width. When source and dest are the same category
        (e.g., ptr bitcast), pass through.
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)
        src_dtype = self.env_types.get(src_id, "fp32")
        dst_elem = ssa.elem_type or "f32"

        src_is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        dst_is_float = dst_elem in ("f32", "f16", "bf16", "f64")
        dst_is_int = dst_elem.startswith("i")

        # Mapping from Triton float dtype -> MSL type name
        _FP_DTYPE_TO_MSL = {
            "fp16": "half",
            "fp32": "float",
            "bf16": "bfloat",
            "fp64": "double",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        _FP_ELEM_TO_MSL = {"f16": "half", "bf16": "bfloat", "f32": "float", "f64": "double"}
        _FP_ELEM_TO_DTYPE = {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}

        # Width mapping for MSL types
        _FP_WIDTH = {"half": 16, "bfloat": 16, "float": 32, "double": 64}
        _INT_WIDTH_FROM_DTYPE = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64}
        _INT_MSL_FROM_WIDTH = {8: ("char", "i8"), 16: ("short", "i16"), 32: ("int", "i32"), 64: ("long", "i64")}

        if src_is_float and dst_is_int:
            # float -> int bitcast. MSL as_type requires matching widths, so
            # pick the int MSL type to match source float width; then cast
            # to the requested destination int width if different.
            src_msl_fp = _FP_DTYPE_TO_MSL.get(src_dtype, "float")
            src_width = _FP_WIDTH.get(src_msl_fp, 32)
            matched_msl_int, matched_dtype = _INT_MSL_FROM_WIDTH.get(src_width, ("int", "i32"))
            # Narrow-type floats (f16/bf16) are promoted to float at load; we
            # must narrow back to the matching MSL fp type before the bitcast
            # so that as_type<short>() sees a 16-bit operand.
            src_op = src_var
            if src_width != 32:
                narrow_name = self._next_var("bc")
                self.kb.raw_line(f"    {src_msl_fp} {narrow_name} = static_cast<{src_msl_fp}>({src_var});")
                src_op = narrow_name
            var_name = self._next_var("bc")
            self.kb.raw_line(f"    {matched_msl_int} {var_name} = as_type<{matched_msl_int}>({src_op});")
            # If destination int type has a different width, narrow/widen
            dst_int_width = _INT_WIDTH_FROM_DTYPE.get(dst_elem, src_width)
            if dst_int_width != src_width:
                dst_msl_int, dst_int_dtype = _INT_MSL_FROM_WIDTH.get(dst_int_width, ("int", "i32"))
                cast_name = self._next_var("bc")
                self.kb.raw_line(f"    {dst_msl_int} {cast_name} = static_cast<{dst_msl_int}>({var_name});")
                self.env[ssa.id] = cast_name
                self.env_types[ssa.id] = dst_int_dtype
            else:
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = matched_dtype
            self._propagate_shape_elementwise(ssa)
            # Bitcast preserves per-thread element identity: propagate bcast
            # layout so chained reduces in tl.sort can recognise post-reduce
            # slice-layout operands.
            if src_id in self._bcast_layout:
                self._bcast_layout[ssa.id] = self._bcast_layout[src_id]
        elif not src_is_float and dst_is_float:
            # int -> float bitcast. MSL as_type requires matching widths.
            # When the destination is bf16, bitcast directly to bfloat — using
            # half as an intermediate would later get value-converted to bfloat
            # at the store boundary, corrupting the bit pattern. Otherwise pick
            # the width-matching float type and cast if widths differ.
            src_int_width = _INT_WIDTH_FROM_DTYPE.get(src_dtype, 32)
            # Find matching float type by width
            _WIDTH_TO_FP = {16: ("half", "fp16"), 32: ("float", "fp32"), 64: ("double", "fp64")}
            dst_msl_fp = _FP_ELEM_TO_MSL.get(dst_elem, "float")
            dst_width = _FP_WIDTH.get(dst_msl_fp, 32)
            dst_dtype_name = _FP_ELEM_TO_DTYPE.get(dst_elem, "fp32")
            if dst_width == src_int_width:
                # Bitcast directly to the requested destination float type so
                # the bit pattern is preserved (especially important for bf16
                # vs half which share width but differ in encoding).
                var_name = self._next_var("bc")
                self.kb.raw_line(f"    {dst_msl_fp} {var_name} = as_type<{dst_msl_fp}>({src_var});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = dst_dtype_name
            else:
                # Widths disagree — bitcast to width-matching float then value
                # convert to the destination type.
                matched_fp, matched_fp_dtype = _WIDTH_TO_FP.get(src_int_width, ("float", "fp32"))
                var_name = self._next_var("bc")
                self.kb.raw_line(f"    {matched_fp} {var_name} = as_type<{matched_fp}>({src_var});")
                cast_name = self._next_var("bc")
                self.kb.raw_line(f"    {dst_msl_fp} {cast_name} = static_cast<{dst_msl_fp}>({var_name});")
                self.env[ssa.id] = cast_name
                self.env_types[ssa.id] = dst_dtype_name
            self._propagate_shape_elementwise(ssa)
            if src_id in self._bcast_layout:
                self._bcast_layout[ssa.id] = self._bcast_layout[src_id]
        else:
            # Same category — passthrough (shape propagation handled inside)
            self._emit_passthrough(ssa)
            # Update type to reflect the destination type
            self.env_types[ssa.id] = _mlir_to_triton_dtype(dst_elem)

    def _lower_mulhiui(self, ssa: SSAValue):
        """tt.mulhiui → upper 32 bits of unsigned 32x32→64 multiply."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        self.kb.raw_line(f"    uint {var_name} = mulhi(as_type<uint>({a}), as_type<uint>({b}));")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "i32"
        # Shape: element-wise binary
        self._propagate_shape_elementwise(ssa)

    def _lower_fp_to_fp(self, ssa: SSAValue):
        """tt.fp_to_fp — Triton's FP8 cast operation.

        FP8 → wider (extf direction): load already converted to float, passthrough.
        wider → FP8 (truncf direction): convert float to FP8 encoding.
        Non-FP8 narrowing with rounding=rtz: pre-mask mantissa bits before
        the standard cast so the (default RNE) hardware cast produces the
        truncated value.
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)
        src_dtype = self.env_types.get(src_id, "fp32")
        dst_elem = ssa.elem_type or "f32"
        dst_dtype = _mlir_to_triton_dtype(dst_elem)
        rounding = ssa.attrs.get("rounding") if ssa.attrs else None

        from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_to_float_func, fp8_from_float_func

        if is_fp8_type(src_dtype) and not is_fp8_type(dst_dtype):
            # FP8 → wider float: already in float from load, passthrough
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = dst_dtype
        elif not is_fp8_type(src_dtype) and is_fp8_type(dst_dtype):
            # wider float → FP8: round-trip through encode/decode to emulate truncation
            self._inject_fp8_device_functions(dst_dtype)
            var_name = self._next_var("fp8")
            from_func = fp8_from_float_func(dst_dtype)
            to_func = fp8_to_float_func(dst_dtype)
            # Convert to fp8 encoding and back to float (the actual fp8 byte
            # is materialized at the store boundary when writing to uchar buffer)
            self.kb.raw_line(f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dst_dtype
        elif is_fp8_type(src_dtype) and is_fp8_type(dst_dtype):
            # FP8 → FP8: re-encode through float
            self._inject_fp8_device_functions(src_dtype)
            self._inject_fp8_device_functions(dst_dtype)
            var_name = self._next_var("fp8")
            from_func = fp8_from_float_func(dst_dtype)
            to_func = fp8_to_float_func(dst_dtype)
            self.kb.raw_line(f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dst_dtype
        elif rounding == "rtz" and src_dtype == "fp32" and dst_dtype in ("fp16", "bf16"):
            # f32 → narrow float with round-toward-zero. MSL's static_cast uses
            # round-to-nearest-even; pre-mask bits the destination cannot
            # represent so the cast becomes exact (equivalent to truncation
            # toward zero) for the bulk of values. For values that fall in the
            # destination's subnormal range, fall back to: cast back, compare,
            # nudge by 1 ULP toward zero if we rounded away. fp16 keeps 10
            # mantissa bits (clear bottom 13); bf16 keeps 7 (clear bottom 16).
            mask = 0xFFFFE000 if dst_dtype == "fp16" else 0xFFFF0000
            dst_msl = "half" if dst_dtype == "fp16" else "bfloat"
            bits_var = self._next_var("rtzb")
            masked_var = self._next_var("rtzm")
            f32_var = self._next_var("rtzf")
            cand_var = self._next_var("rtzc")
            back_var = self._next_var("rtzbk")
            adj_var = self._next_var("rtzab")
            adj_bits = self._next_var("rtzai")
            out_var = self._next_var("rtz")
            self.kb.raw_line(f"    int {bits_var} = as_type<int>(static_cast<float>({src_var}));")
            self.kb.raw_line(f"    int {masked_var} = {bits_var} & static_cast<int>({mask:#x}u);")
            self.kb.raw_line(f"    float {f32_var} = as_type<float>({masked_var});")
            self.kb.raw_line(f"    {dst_msl} {cand_var} = static_cast<{dst_msl}>({f32_var});")
            # Round-trip back to f32 to detect cases where the cast still
            # rounded away from zero (i.e., the destination subnormal range
            # where the mask trick is insufficient).
            self.kb.raw_line(f"    float {back_var} = static_cast<float>({cand_var});")
            # If |back| > |orig| and orig is finite, nudge toward zero by 1 ULP.
            self.kb.raw_line(f"    bool {adj_var} = isfinite({f32_var}) && (fabs({back_var}) > fabs({f32_var}));")
            # Compute one-ULP-nudge toward zero. IEEE 754 bit patterns grow
            # in magnitude with the float's magnitude on each side of zero;
            # in two's-complement representation of the underlying short,
            # decrementing reduces magnitude regardless of sign (e.g. f16
            # 0xC000 = -2.0 → 0xBFFF = -1.999, magnitude shrinks toward 0).
            self.kb.raw_line(f"    short {adj_bits} = as_type<short>({cand_var});")
            self.kb.raw_line(f"    {adj_bits} = {adj_var} ? (short)({adj_bits} - 1) : {adj_bits};")
            self.kb.raw_line(f"    {dst_msl} {out_var} = as_type<{dst_msl}>({adj_bits});")
            self.env[ssa.id] = out_var
            self.env_types[ssa.id] = dst_dtype
        else:
            # non-FP8 → non-FP8: passthrough (regular float precision change)
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = dst_dtype
        self._propagate_shape_elementwise(ssa)

    def _lower_extf(self, ssa: SSAValue):
        """arith.extf — extend float precision.

        FP8 → FP32/FP16: call software conversion function.
        FP16 → FP32: passthrough (compute already in float).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_dtype = self.env_types.get(src_id, "fp32")
        from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_to_float_func

        if is_fp8_type(src_dtype):
            # FP8 → float: the load already converted to float, so this is a passthrough.
            # However, if the source is a raw uchar (from bitcast or constant), convert.
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = "fp32"
        else:
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = "fp32"
        self._propagate_shape_elementwise(ssa)

    def _lower_truncf(self, ssa: SSAValue):
        """arith.truncf — truncate float precision.

        FP32 → FP8: software conversion (round-trip, keep float register).
        FP32 → FP16/BF16: round-trip through the narrow type to QUANTIZE the value,
            then keep a float register for downstream arithmetic. (A plain passthrough
            here was a silent-wrong: an explicit ``.to(tl.float16)`` mid-computation
            never lost precision — 2026-06-22 re-audit: 2049.0 stayed 2049.0 instead
            of quantizing to 2048.0.)
        """
        if not ssa.operand_ids:
            return
        dst_elem = ssa.elem_type or "f16"
        dst_dtype = _mlir_to_triton_dtype(dst_elem)
        from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_from_float_func, fp8_to_float_func

        if is_fp8_type(dst_dtype):
            # FP32 → FP8: emit conversion call
            src_var = self._lookup(ssa.operand_ids[0])
            self._inject_fp8_device_functions(dst_dtype)
            var_name = self._next_var("fp8")
            from_func = fp8_from_float_func(dst_dtype)
            to_func = fp8_to_float_func(dst_dtype)
            # Convert to fp8 and immediately back to float for further computation
            # The actual fp8 encoding is stored at the store boundary
            self.kb.raw_line(f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dst_dtype
        else:
            narrow = "half" if dst_dtype in ("fp16", "f16") else "bfloat" if dst_dtype in ("bf16",) else None
            # Quantize when the source is a SCALAR float (the static_cast round-trip
            # is valid MSL). That covers BOTH the pure one-element-per-thread kernel
            # AND the wrap-loop body (where the value is scalar-shaped per iteration,
            # indexed by _loop_e). It is NOT valid only for the MEPT single-pass
            # array form (env value is a `float*` register array) — excluded by
            # _mept_single_pass / env_n_elems>1 (that array case is a documented
            # per-element follow-up). The earlier `not _needs_wrapping` over-excluded
            # the wrap-loop and left `.to(fp16)` un-quantized for large N (re-audit #2).
            _scalar_mode = (
                not getattr(self, "_mept_single_pass", False) and self.env_n_elems.get(ssa.operand_ids[0], 1) == 1
            )
            # A SHARED-MEMORY-backed source (an oversized tile / loop-carried
            # accumulator) resolves through ``_lookup`` to the bare threadgroup ARRAY
            # NAME, so the scalar round-trip below emitted
            # ``static_cast<half>(smem_iter_N)`` — casting a ``threadgroup float*`` to a
            # scalar, which Metal rejects (invalid MSL; loud, never silently wrong).
            # Quantize cooperatively in place instead, mirroring what the binary
            # elementwise path already does for smem operands (_lowerer_emission), and
            # re-register the result as smem-backed so the store keeps its cooperative
            # path. Storage stays fp32 (a narrow->wide round-trip) exactly like the
            # scalar branch, so the VALUE quantizes while downstream arithmetic still
            # reads floats. Reached by an fp16 varlen-shaped FA kernel that falls to the
            # generic path (2026-08-26); the loop degenerates safely for any tile size.
            _smem_src = getattr(self, "_shared_mem_descs", {}).get(ssa.operand_ids[0])
            if narrow is not None and _smem_src is not None:
                # This path updates the source array in place to avoid allocating a
                # second full-tile buffer (the generic varlen kernel already sits at
                # Metal's threadgroup-memory limit). That is only SSA-correct when the
                # cast is the source value's sole consumer. If another op also reads
                # the fp32 source, in-place quantization would silently change that
                # consumer to fp16/bf16 values. Refuse that uncommon aliasing shape
                # until the lowerer can allocate/lifetime-reuse distinct storage.
                def _all_ops(ops):
                    for _op in ops:
                        yield _op
                        if getattr(_op, "region_ops", None):
                            yield from _all_ops(_op.region_ops)
                        if getattr(_op, "else_ops", None):
                            yield from _all_ops(_op.else_ops)

                _src_id = ssa.operand_ids[0]
                _other_consumers = [
                    _op for _op in _all_ops(self.graph.ops) if _op is not ssa and _src_id in (_op.operand_ids or [])
                ]
                if _other_consumers:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "fp16/bf16 conversion of a shared-memory-backed value with "
                        "another consumer cannot quantize the source array in place "
                        "without violating SSA semantics. Refusing rather than "
                        "silently rounding the original value; materialize the cast "
                        "in a separate kernel or remove the additional consumer.",
                        op_name="arith.truncf",
                    )
                _smem_name, _smem_shape, _ = _smem_src
                _smem_total = 1
                for _d in _smem_shape:
                    _smem_total *= _d
                _bs = self.effective_block_size
                self.kb.raw_line(f"    for (uint _tq = lid; _tq < {_smem_total}u; _tq += {_bs}u) {{")
                self.kb.raw_line(
                    f"        {_smem_name}[_tq] = static_cast<float>(static_cast<{narrow}>({_smem_name}[_tq]));"
                )
                self.kb.raw_line("    }")
                self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Expose a per-thread SCALAR as this value's SSA name — NOT the bare
                # array name: flat consumers (a total<=block_size store, local_alloc
                # staging) interpolate the env value into element expressions, and the
                # array name there is invalid MSL (`static_cast<half>(smem_N)` /
                # `smem_0[_sa] = smem_N` — dot-recovery recon 2026-08-30). Cooperative
                # consumers keep using the smem desc registered below, exactly as they
                # do for the dot result itself.
                _tv = self._next_var("truncf")
                self.kb.raw_line(f"    float {_tv} = (lid < {_smem_total}u) ? {_smem_name}[lid] : 0.0f;")
                self.env[ssa.id] = _tv
                self._shared_mem_descs[ssa.id] = (_smem_name, _smem_shape, "fp32")
            elif narrow is not None and _scalar_mode:
                # SCALAR fp32->fp16/bf16: round-trip narrow->wide so the VALUE
                # actually quantizes (e.g. 2049.0 -> 2048.0) while the register
                # stays float for downstream arithmetic — mirrors the FP8 branch.
                # (The old _emit_passthrough aliased the fp32 source unchanged, so
                # an explicit `.to(tl.float16)` mid-computation did not lose
                # precision — 2026-06-22 re-audit silent-wrong.)
                src_var = self._lookup(ssa.operand_ids[0])
                var_name = self._next_var("truncf")
                self.kb.raw_line(f"    float {var_name} = static_cast<float>(static_cast<{narrow}>({src_var}));")
                self.env[ssa.id] = var_name
            elif narrow is not None and ssa.operand_ids[0] in self.env_array:
                # MEPT single-pass ARRAY form: the source is a per-thread register
                # array (float src[n]); quantize EACH element (round-trip narrow->wide)
                # so a mid-computation `.to(fp16)` that then feeds arithmetic loses
                # precision, exactly as in the scalar path. (Was a passthrough that
                # dropped quantization for large array kernels — re-audit #3 silent-wrong.)
                _src_name, _n, _ = self.env_array[ssa.operand_ids[0]]
                _exprs = [f"static_cast<float>(static_cast<{narrow}>({_src_name}[{_i}]))" for _i in range(_n)]
                _arr = self._var_array("truncf", _exprs, "float")
                self.env[ssa.id] = _arr
                self.env_array[ssa.id] = (_arr, _n, "float")
                self.env_n_elems[ssa.id] = _n
            else:
                # fp64->fp32 or unknown: passthrough (the store boundary narrows on
                # write, so a cast that is only stored is still correct).
                self._emit_passthrough(ssa)
            self.env_types[ssa.id] = _mlir_to_triton_dtype(dst_elem) if dst_elem else "fp16"
        self._propagate_shape_elementwise(ssa)

    def _lower_cmpi(self, ssa: SSAValue):
        """arith.cmpi → comparison with unsigned cast when needed.

        Uses pred_name (from MLIR text, reliable) over pred_int (enum may
        differ between MLIR versions).
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        # Get predicate — prefer pred_name (text-parsed, authoritative)
        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        if pred_name and pred_name in CMPI_NAMED:
            op_str = CMPI_NAMED[pred_name]
        elif pred_int is not None and pred_int in CMPI_PREDICATES:
            op_str = CMPI_PREDICATES[pred_int]
        else:
            op_str = "<"

        # Unsigned predicates need (uint) cast for correct semantics.
        # Signed predicates need (int) cast to prevent C++ implicit
        # unsigned promotion when comparing int vs uint (e.g. int32 vs
        # zero-extended uint8 → uint32).
        is_unsigned = pred_name in ("ult", "ule", "ugt", "uge") if pred_name else False
        is_signed = pred_name in ("slt", "sle", "sgt", "sge") if pred_name else False

        # The normalizing cast must match the operand WIDTH, not assume 32 bits:
        # i64 operands are declared ``long`` in MSL, so a hardcoded ``(int)``
        # truncates them to 32 bits before the compare — a silent-wrong on any
        # value outside int32 range (audit C1). Pick the 64-bit cast when either
        # operand is 64-bit; ``eq``/``ne`` (no cast) were already width-safe.
        def _is64(oid):
            return self.env_types.get(oid) in ("i64", "u64", "ui64")

        wide64 = _is64(ssa.operand_ids[0]) or _is64(ssa.operand_ids[1])
        s_cast = "(long)" if wide64 else "(int)"
        u_cast = "(ulong)" if wide64 else "(uint)"

        # Phase 4b/c: MEPT array path. Wraps the same cast convention
        # (uint / int / none) per array position. Result is bool[N];
        # downstream consumers (tt.load mask, tt.store mask, arith.andi
        # on i1) read it as an env_array.
        if is_unsigned:

            def _make_expr(av, bv, _op=op_str):
                return f"{u_cast}{av} {_op} {u_cast}{bv}"
        elif is_signed:

            def _make_expr(av, bv, _op=op_str):
                return f"{s_cast}{av} {_op} {s_cast}{bv}"
        else:

            def _make_expr(av, bv, _op=op_str):
                return f"{av} {_op} {bv}"

        if self._mept_binary_dispatch(ssa, ssa.operand_ids[0], ssa.operand_ids[1], a, b, _make_expr, "bool", "i1"):
            # _mept_binary_dispatch set env / env_array; mark as mask.
            self.env_is_mask[ssa.id] = True
            return

        var_name = self._next_var("mask")
        if is_unsigned:
            self.kb.raw_line(f"    bool {var_name} = {u_cast}{a} {op_str} {u_cast}{b};")
        elif is_signed:
            self.kb.raw_line(f"    bool {var_name} = {s_cast}{a} {op_str} {s_cast}{b};")
        else:
            self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        # Shape: comparison inherits shape from operands
        self._propagate_shape_elementwise(ssa)
        # Propagate bcast layout across the comparison — the result has the
        # same (lid → flat-index) mapping as its operands.
        self._propagate_bcast_layout_binary(ssa)

    def _lower_cmpf(self, ssa: SSAValue):
        """arith.cmpf → float comparison with NaN-aware unordered predicates.

        pred_name (from MLIR text parsing) is the primary predicate source.
        pred_int is used as fallback only. Array path (MEPT) mirrors
        _lower_cmpi: a register-array operand yields a bool[N] per-position
        mask. The scalar fallback emits exactly the original text so MEPT=0
        codegen is byte-identical.
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        # Per-element boolean expression for operands (av, bv). Returns the
        # SAME unparenthesised text the scalar path historically emitted, so
        # `bool m = {_expr(a, b)};` is byte-identical to the old code.
        def _expr(av, bv):
            if pred_name == "false":
                return "false"
            if pred_name == "true":
                return "true"
            if pred_name == "uno":
                return f"isnan({av}) || isnan({bv})"
            if pred_name == "ord":
                return f"!isnan({av}) && !isnan({bv})"
            if pred_name == "une":
                return f"{av} != {bv}"
            if pred_name and pred_name in CMPF_NAMED:
                op_str = CMPF_NAMED[pred_name]
                if pred_name.startswith("u"):
                    return f"isnan({av}) || isnan({bv}) || ({av} {op_str} {bv})"
                return f"{av} {op_str} {bv}"
            if pred_int is not None and pred_int in CMPF_PREDICATES:
                return f"{av} {CMPF_PREDICATES[pred_int]} {bv}"
            return f"{av} < {bv}"

        # MEPT array path: if either operand is a register array, emit bool[N].
        if self._mept_binary_dispatch(ssa, ssa.operand_ids[0], ssa.operand_ids[1], a, b, _expr, "bool", "i1"):
            self.env_is_mask[ssa.id] = True
            return

        # Scalar fallback (byte-identical to the original emission).
        var_name = self._next_var("mask")
        self.kb.raw_line(f"    bool {var_name} = {_expr(a, b)};")
        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        self._propagate_shape_elementwise(ssa)
        self._propagate_bcast_layout_binary(ssa)

    def _lower_select(self, ssa: SSAValue):
        """arith.select → ternary operator with inferred type.

        Preserves the logical float dtype (fp16/bf16/fp32/fp64) from the IR
        when available so that subsequent arith.bitcast uses the correct width.
        Narrow floats (fp16/bf16) still compute in ``float`` (matching the rest
        of the codegen), but their logical dtype is tracked in env_types so
        that a later bitcast to i16/i32 narrows the operand first.
        """
        if len(ssa.operand_ids) < 3:
            return
        if "!tt.ptr<" in (ssa.type_str or ""):
            from triton_msl.errors import MetalNonRecoverableError

            if ssa.operand_ids[0] in self.env_array or any(oid in self.env_ptr_array for oid in ssa.operand_ids[1:]):
                raise MetalNonRecoverableError(
                    "pointer-valued arith.select has a per-thread register-array operand; "
                    "address-array selection is not implemented and numeric fallback is unsafe",
                    op_name="arith.select",
                )
            true_ptr = self._loop_pointer_parts(ssa.operand_ids[1])
            false_ptr = self._loop_pointer_parts(ssa.operand_ids[2])
            meta = self.graph.result_meta.get(ssa.id)
            if (
                true_ptr is None
                or false_ptr is None
                or meta is None
                or meta.schema_version != 1
                or meta.value_id != ssa.id
                or meta.kind != "result"
                or meta.producer_id != ssa.id
                or meta.result_index != 0
                or meta.type.kind != "pointer"
                or meta.type.address_space != 1
                or meta.type.pointee is None
                or meta.type.pointee.unknown_reason
                or meta.type.pointee.elem is None
            ):
                raise MetalNonRecoverableError(
                    "pointer-valued arith.select lacks two proved addresses or native result metadata",
                    op_name="arith.select",
                )
            cond = self._lookup(ssa.operand_ids[0])

            def _address(info):
                base, offset = info
                return base if not offset or offset == "0" else f"({base} + {offset})"

            dtype = _mlir_to_triton_dtype(meta.type.pointee.elem)
            var_name = self._next_var("ptr")
            self.kb.raw_line(
                f"    volatile device {triton_type_to_msl(dtype)}* {var_name} = "
                f"{cond} ? {_address(true_ptr)} : {_address(false_ptr)};"
            )
            self.env[ssa.id] = var_name
            self.env_is_ptr[ssa.id] = (var_name, "0")
            self.env_types[ssa.id] = dtype
            self._propagate_shape_elementwise(ssa)
            return
        cond = self._lookup(ssa.operand_ids[0])
        true_val = self._lookup(ssa.operand_ids[1])
        false_val = self._lookup(ssa.operand_ids[2])
        var_name = self._next_var("r")

        # Prefer ssa.elem_type (from the IR's result type) over operand tracking.
        # This preserves fp16/bf16 across select even though operands may have
        # been widened to float earlier.
        ir_dtype = _mlir_to_triton_dtype(ssa.elem_type) if ssa.elem_type else None
        true_dtype = self.env_types.get(ssa.operand_ids[1], "fp32")

        if ir_dtype and (ir_dtype.startswith("fp") or ir_dtype.startswith("bf")):
            # Metal has no fp64 on Apple Silicon; fp64 downcasts to float32 to
            # match the rest of the codegen (see msl_emitter.triton_type_to_msl).
            # Narrow float (fp16/bf16) also computes in float.
            ty = "float"
            dtype = "fp32" if ir_dtype == "fp64" else ir_dtype
        elif ir_dtype in ("i64", "u64", "ui64"):
            ty = "ulong" if ir_dtype != "i64" else "long"
            dtype = "u64" if ir_dtype != "i64" else "i64"
        elif ir_dtype and ir_dtype.startswith("i"):
            ty = "int"
            dtype = ir_dtype
        elif ir_dtype and ir_dtype.startswith("u"):
            ty = "uint"
            dtype = ir_dtype
        elif true_dtype in ("i64", "u64", "ui64"):
            ty = "ulong" if true_dtype != "i64" else "long"
            dtype = "u64" if true_dtype != "i64" else "i64"
        elif true_dtype.startswith("fp") or true_dtype.startswith("bf"):
            ty = "float"
            dtype = true_dtype if true_dtype in ("fp16", "bf16", "fp32", "fp64") else "fp32"
        elif true_dtype.startswith("u"):
            ty = "uint"
            dtype = "u32"
        else:
            ty = "int"
            dtype = "i32"

        # MEPT array path: if any operand is a register array, emit a
        # per-element select and return. Scalar/flag-off select falls through
        # to the byte-identical scalar emission below.
        if self._mept_select_dispatch(
            ssa, ssa.operand_ids[0], ssa.operand_ids[1], ssa.operand_ids[2], cond, true_val, false_val, ty, dtype
        ):
            return

        self.kb.raw_line(f"    {ty} {var_name} = {cond} ? {true_val} : {false_val};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: select inherits shape from operands (cond, true, false)
        self._propagate_shape_elementwise(ssa)
        # Propagate bcast layout across the ternary select — if any operand
        # has a tracked layout, the result carries it.
        for oid in ssa.operand_ids:
            if oid in self._bcast_layout:
                self._bcast_layout[ssa.id] = self._bcast_layout[oid]
                break

    # -- Math ops --

    def _lower_math(self, ssa: SSAValue):
        """Lower math.* operations to MSL intrinsics."""
        op = ssa.op
        if not ssa.operand_ids:
            return

        # Map math ops to MSL functions
        unary_map = {
            "math.exp": "exp",
            "math.exp2": "exp2",
            "math.log": "log",
            "math.log2": "log2",
            "math.sqrt": "sqrt",
            "math.rsqrt": "rsqrt",
            "math.abs": "abs",
            "math.absf": "abs",
            "math.sin": "sin",
            "math.cos": "cos",
            "math.tanh": "tanh",
            "math.floor": "floor",
            "math.ceil": "ceil",
            "math.round": "round",
        }

        if op in unary_map:
            src_id = ssa.operand_ids[0]
            fn = unary_map[op]
            # Phase 4b: MEPT array path.
            if self.mept_enabled and src_id in self.env_array:
                src_name, n, _ = self.env_array[src_id]
                exprs = [f"{fn}({src_name}[{i}])" for i in range(n)]
                var_name = self._var_array("r", exprs, "float")
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, n, "float")
                self.env_types[ssa.id] = "fp32"
                return
            a = self._lookup(src_id)
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = {fn}({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.absi":
            # Integer absolute value
            src_id = ssa.operand_ids[0]
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
            if self.mept_enabled and src_id in self.env_array:
                src_name, n, _ = self.env_array[src_id]
                exprs = [f"abs({src_name}[{i}])" for i in range(n)]
                var_name = self._var_array("r", exprs, ty)
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, n, ty)
                self.env_types[ssa.id] = dtype
                return
            a = self._lookup(src_id)
            var_name = self._next_var("r")
            self.kb.raw_line(f"    {ty} {var_name} = abs({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dtype

        elif op == "math.fma":
            if len(ssa.operand_ids) >= 3:
                a_id, b_id, c_id = ssa.operand_ids[:3]
                # Phase 4b: ternary MEPT — require all-or-nothing array
                # form. If any operand is an array, all three must be
                # arrays of the same length (or scalar to broadcast).
                if self.mept_enabled and (a_id in self.env_array or b_id in self.env_array or c_id in self.env_array):
                    a = self._lookup(a_id)
                    b = self._lookup(b_id)
                    c = self._lookup(c_id)
                    arrs = [self.env_array.get(x) for x in (a_id, b_id, c_id)]
                    ns = [arr[1] for arr in arrs if arr is not None]
                    n = ns[0] if ns else 1
                    if all((arr is None or arr[1] == n) for arr in arrs):

                        def _read(i, sid, scalar):
                            arr = self.env_array.get(sid)
                            return f"{arr[0]}[{i}]" if arr else scalar

                        exprs = [
                            f"fma({_read(i, a_id, a)}, {_read(i, b_id, b)}, {_read(i, c_id, c)})" for i in range(n)
                        ]
                        var_name = self._var_array("r", exprs, "float")
                        self.env[ssa.id] = var_name
                        self.env_array[ssa.id] = (var_name, n, "float")
                        self.env_types[ssa.id] = "fp32"
                        return
                a = self._lookup(a_id)
                b = self._lookup(b_id)
                c = self._lookup(c_id)
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = fma({a}, {b}, {c});")
                self.env[ssa.id] = var_name

        elif op == "math.erf":
            # MSL has no erf() — Abramowitz & Stegun approximation (max error ~1.5e-7)
            a = self._lookup(ssa.operand_ids[0])
            abs_var = self._next_var("erf_abs")
            t_var = self._next_var("erf_t")
            y_var = self._next_var("erf_y")
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {abs_var} = abs({a});")
            self.kb.raw_line(f"    float {t_var} = 1.0f / (1.0f + 0.3275911f * {abs_var});")
            self.kb.raw_line(
                f"    float {y_var} = 1.0f - (((((1.061405429f * {t_var} "
                f"- 1.453152027f) * {t_var}) + 1.421413741f) * {t_var} "
                f"- 0.284496736f) * {t_var} + 0.254829592f) * {t_var} "
                f"* exp(-{abs_var} * {abs_var});"
            )
            self.kb.raw_line(f"    float {var_name} = copysign({y_var}, {a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op in ("math.powf", "math.copysign", "math.atan2"):
            if len(ssa.operand_ids) >= 2:
                bin_fn = {"math.powf": "pow", "math.copysign": "copysign", "math.atan2": "atan2"}[op]
                a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
                a = self._lookup(a_id)
                b = self._lookup(b_id)

                # Phase 4b: MEPT array path via shared dispatcher.
                def _make_expr(av, bv, _fn=bin_fn):
                    return f"{_fn}({av}, {bv})"

                if self._mept_binary_dispatch(ssa, a_id, b_id, a, b, _make_expr, "float", "fp32"):
                    return
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = {bin_fn}({a}, {b});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = "fp32"

        elif op in ("math.roundeven", "math.trunc"):
            # Both are simple unary float ops with a different MSL name.
            un_fn = {"math.roundeven": "rint", "math.trunc": "trunc"}[op]
            src_id = ssa.operand_ids[0]
            if self.mept_enabled and src_id in self.env_array:
                src_name, n, _ = self.env_array[src_id]
                exprs = [f"{un_fn}({src_name}[{i}])" for i in range(n)]
                var_name = self._var_array("r", exprs, "float")
                self.env[ssa.id] = var_name
                self.env_array[ssa.id] = (var_name, n, "float")
                self.env_types[ssa.id] = "fp32"
                return
            a = self._lookup(src_id)
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = {un_fn}({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.log1p":
            # log1p(x) = log(1 + x), more numerically stable near zero
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = log(1.0f + {a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.expm1":
            # expm1(x) = exp(x) - 1, more numerically stable near zero
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = (exp({a}) - 1.0f);")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        else:
            self.kb.unsupported(f"UNSUPPORTED math: {op}")
            return
        # Shape: all math ops are element-wise — inherit from operands
        self._propagate_shape_elementwise(ssa)

    def _lower_precise_math(self, ssa: SSAValue, kind: str):
        """Lower tt.precise_sqrt / tt.precise_divf to MSL."""
        if kind == "sqrt":
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = precise::sqrt({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        elif kind == "divf":
            a = self._lookup(ssa.operand_ids[0])
            b = self._lookup(ssa.operand_ids[1])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = precise::divide({a}, {b});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        # Shape: precise math is element-wise
        self._propagate_shape_elementwise(ssa)

    # -- Extern elementwise --

    def _lower_extern_elementwise(self, ssa: SSAValue):
        """tt.extern_elementwise → direct MSL function call.

        Handles the common case where the extern function maps to a Metal
        standard library function (e.g., sin, cos, exp, etc.).

        The symbol name is extracted from the op's attributes. The TTGIR text
        typically contains: tt.extern_elementwise ... {symbol = "func_name", ...}
        The walker stores raw attributes, so we check for 'symbol', 'libname',
        and 'pure' attributes.
        """
        # Extract function name from attributes
        func_name = ssa.attrs.get("symbol", "")
        if not func_name:
            func_name = ssa.attrs.get("libname", "")
        if not func_name:
            # Fallback: try to extract from the raw_line or op string
            self.kb.unsupported("UNSUPPORTED: tt.extern_elementwise (no symbol)")
            return

        # Sanitize function name for MSL (strip leading underscores from __nv_* etc.)
        # Common pattern: __nv_sinf → sin, __nv_expf → exp
        safe_name = func_name

        # Explicit CUDA→MSL renames for functions whose MSL name doesn't match
        # the "drop __nv_ prefix and trailing f" rule. MSL's isfinite/isinf/isnan
        # take a single fp arg and return bool. Precedence over the prefix strip.
        _NV_TO_MSL = {
            "__nv_finitef": "isfinite",
            "__nv_isfinited": "isfinite",
            "__nv_isinff": "isinf",
            "__nv_isinfd": "isinf",
            "__nv_isnanf": "isnan",
            "__nv_isnand": "isnan",
            "__nv_signbitf": "signbit",
            "__nv_signbitd": "signbit",
        }
        if safe_name in _NV_TO_MSL:
            safe_name = _NV_TO_MSL[safe_name]
        elif safe_name.startswith("__nv_"):
            # CUDA libdevice function — strip prefix and trailing 'f' if present
            stripped = safe_name[5:]  # remove "__nv_"
            if stripped.endswith("f") and len(stripped) > 1:
                stripped = stripped[:-1]
            safe_name = stripped

        # Build argument list
        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        # Determine result type
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        if triton_dtype.startswith("fp") or triton_dtype.startswith("bf"):
            msl_ty = "float"
        elif triton_dtype.startswith("u"):
            msl_ty = "uint"
        elif triton_dtype == "i64":
            msl_ty = "long"
        else:
            msl_ty = triton_type_to_msl(triton_dtype)

        var_name = self._next_var("r")
        self.kb.raw_line(f"    {msl_ty} {var_name} = {safe_name}({args_str});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = triton_dtype

    # -- Transpose --

    def _lower_tt_trans(self, ssa: SSAValue):
        """tt.trans → shared memory transpose.

        In TTGIR, tt.trans {order = array<i32: 1, 0>} swaps dimensions.
        For a 2D tensor of shape (M, N), the transpose requires each
        thread to exchange data with another thread:
        - Thread at (row, col) in source needs value from (col, row) in source
        - Source lid → (lid/N, lid%N), transposed → (lid%N, lid/N)
        - Source lid for transposed value: (lid%N)*N + (lid/N) → wrong!
        - Source lid for transposed value: (lid%M)*M + (lid/M) if target is (N,M)

        Uses threadgroup shared memory for the data exchange.
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)

        # Get source and destination shapes
        src_shape = _extract_shape(
            # Find source op's type_str
            self._find_op_type_str(src_id)
        )
        dst_shape = _extract_shape(ssa.type_str)

        # Integrity guard (PR1): the generic transpose only implements the
        # 2-D shared-memory exchange below. A rank>=3 transpose falls through
        # to a passthrough that silently DROPS the permutation — correct only
        # when the permute is the identity, wrong for any real reordering
        # (test_trans_4d: identity perms pass, the rest return the input
        # unpermuted). Templates that do handle high-rank transposes
        # (_detect_transpose_via_reshape, _detect_permute_chained_reduce)
        # return before the generic path, so they never reach here. When we
        # can confirm a non-identity rank>=3 permute, refuse rather than emit
        # wrong output. (If the order can't be parsed we keep the old
        # passthrough — no regression, just no new guarantee.)
        if len(src_shape) >= 3:
            order = self._parse_trans_order(ssa, len(src_shape))
            if order is not None and order != list(range(len(src_shape))):
                raise MetalNonRecoverableError(
                    f"rank-{len(src_shape)} tt.trans with a non-identity "
                    f"permutation {order} is not supported by the generic "
                    "lowerer (only 2-D transpose is). Refusing rather than "
                    "dropping the permutation and returning wrong output "
                    "(e.g. test_trans_4d).",
                    op_name="tt.trans",
                )

        if len(src_shape) < 2 or not self._is_2d:
            # 1D or unknown — passthrough
            self._emit_passthrough(ssa)
            return

        M, N = src_shape[0], src_shape[1]
        total = M * N

        # The shared-memory exchange below maps EXACTLY ONE element per thread
        # (shared[lid] = src; then read shared[transposed(lid)]). Apple caps a
        # threadgroup at 1024 threads, so for total > 1024 the threads cover only
        # the first 1024 elements and the rest are left UNTRANSPOSED — a silent
        # garbage result (2026-06-22 fuzz: a 64x64 = 4096-element tt.trans returned
        # err 3.8). The exchange is not multi-element-per-thread-coverable (a fixed
        # lid per element), so REFUSE rather than mis-compute. (2-D transposes with
        # M*N <= 1024, and the nd/reshape transpose templates that return before
        # this generic path, are unaffected.)
        _TG_MAX = 1024
        if total > _TG_MAX:
            from ._biased_replay import emit_loaded_transpose

            if getattr(self, "_source_biased_replay", False) and emit_loaded_transpose(self, ssa):
                return
            raise MetalNonRecoverableError(
                f"2-D tt.trans of a {M}x{N} ({total}-element) block exceeds the "
                f"{_TG_MAX}-thread threadgroup: the shared-memory transpose maps one "
                f"element per thread, so elements past {_TG_MAX} would be left "
                f"untransposed (silent-wrong). Use a tile with M*N <= {_TG_MAX}.",
                op_name="tt.trans",
            )

        # Determine types
        input_dtype = self.env_types.get(src_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        if input_dtype in ("i64", "u64", "ui64"):
            msl_type = "ulong" if input_dtype != "i64" else "long"
            shared_dtype = "u64" if input_dtype != "i64" else "i64"
        elif is_float:
            msl_type, shared_dtype = "float", "fp32"
        else:
            msl_type, shared_dtype = "int", "i32"

        # Allocate shared memory for transpose
        shared_name = f"trans_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        result_var = self._next_var("trans")

        # Write to shared in row-major order
        self.kb.raw_line(f"    {shared_name}[lid] = {src_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read from transposed position
        # Source: (row, col) = (lid/N, lid%N) → linear lid
        # Transposed: want value from (col, row) = (lid%N, lid/N)
        # → linear index in source: (lid%N)*N + (lid/N)... wait, that's wrong.
        # Source stored as row-major (M rows, N cols): index = row*N + col
        # We want element at transposed position: (col, row) in source
        # = source[col * N + row]... no, source is M×N row-major.
        # Source element (r, c) is at index r*N + c.
        # After transpose, output position (i, j) = source (j, i).
        # Output is N×M. Thread lid in output maps to (lid/M, lid%M).
        # We want source value at (lid%M, lid/M) = source[(lid%M)*N + (lid/M)].
        self.kb.raw_line(f"    {msl_type} {result_var} = {shared_name}[(lid % {M}u) * {N}u + (lid / {M}u)];")
        # Scratch may be reused by a subsequent staging operation. All lanes
        # must finish the transposed read before any lane writes that storage.
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = input_dtype
        if dst_shape:
            self.env_shapes[ssa.id] = dst_shape

    def _fa_analyze_constant_score_scale(self, dot_qk, dot_pv, certified_selects=None):
        """Return the exact constant scale between QK dot and softmax, or ``None``.

        A specialized FA template reloads raw operands and rebuilds the score path, so
        a multiply/divide after ``tt.dot`` is safe only when the complete path from the
        probability exponential back to that dot is structural and constant-valued.
        This helper is shared by detection and final value-path verification so the
        routing guard cannot prove a weaker spelling than the verifier replays.

        The result is ``{"factor": float, "op_ids": tuple}``.  ``factor == 1`` and an
        empty tuple mean that no post-dot scale exists.  Runtime factors, addition,
        asymmetric selects, or any other unreplayed operation return ``None``.
        """
        import math as _math

        def _walk(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _walk(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _walk(s.else_ops)

        ops = list(_walk(self.graph.ops))
        by_id = {}
        for op in ops:
            by_id[op.id] = op
            for result_id in getattr(op, "result_ids", None) or []:
                by_id[result_id] = op

        dot_ids = {dot_qk.id} | set(getattr(dot_qk, "result_ids", None) or [])

        def _depends_on(start_id, targets, limit=256):
            seen, stack = set(), [start_id]
            while stack and len(seen) < limit:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if cur in targets:
                    return True
                op = by_id.get(cur)
                if op is not None:
                    stack.extend(op.operand_ids or [])
            return False

        def _depends_on_dot(start_id, limit=256):
            return _depends_on(start_id, dot_ids, limit)

        scalar_wrappers = {
            "tt.splat",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
            "ttg.convert_layout",
            "arith.extf",
            "arith.truncf",
            "arith.fpext",
            "arith.fptrunc",
        }
        value_wrappers = {
            "ttg.local_load",
            "ttg.local_alloc",
            "ttg.convert_layout",
            "arith.extf",
            "arith.truncf",
            "tt.fp_to_fp",
        }

        def _constant_float(start_id):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                op = by_id.get(cur)
                if op is None:
                    return None
                if op.op == "arith.constant":
                    value = (op.attrs or {}).get("value")
                    return float(value) if isinstance(value, (int, float)) else None
                if op.op not in scalar_wrappers or len(op.operand_ids or []) != 1:
                    return None
                cur = op.operand_ids[0]
            return None

        def _peel(start_id, allowed, limit=64):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < limit:
                seen.add(cur)
                op = by_id.get(cur)
                if op is None or op.op not in allowed:
                    return cur, op
                if len(op.operand_ids or []) != 1:
                    return cur, None
                cur = op.operand_ids[0]
            return cur, None

        # Find the probability exp through the representation-only P@V operand.
        cur = dot_pv.operand_ids[0]
        seen = set()
        p_exp = None
        while cur not in seen and len(seen) < 64:
            seen.add(cur)
            op = by_id.get(cur)
            if op is None:
                return None
            if _op_is_exp(op.op):
                p_exp = op
                break
            if op.op not in value_wrappers or len(op.operand_ids or []) != 1:
                return None
            cur = op.operand_ids[0]
        if p_exp is None or len(p_exp.operand_ids or []) != 1:
            return None

        # The exp-base conversion may be written as exp2((score-max)*log2e).
        # It belongs to the probability transform, not to the post-dot score scale.
        cur, core = _peel(
            p_exp.operand_ids[0],
            {"ttg.convert_layout", "arith.extf", "arith.truncf"},
        )
        if core is not None and core.op == "arith.mulf" and len(core.operand_ids or []) == 2:
            constants = [oid for oid in core.operand_ids if _constant_float(oid) is not None]
            if len(constants) != 1:
                return None
            cur = next(oid for oid in core.operand_ids if oid != constants[0])
            cur, core = _peel(
                cur,
                {"ttg.convert_layout", "arith.extf", "arith.truncf"},
            )
        if core is None or core.op != "arith.subf" or len(core.operand_ids or []) != 2:
            return None

        score_value_id = core.operand_ids[0]

        # The online-softmax row maximum must consume the SAME score value as the
        # probability subtraction. Scaling only after tl.max is not equivalent: it
        # changes both the recurrence and normalization even though the P leg still
        # looks like a constant chain from the dot.
        score_core_id, _ = _peel(score_value_id, value_wrappers)
        max_reductions = []
        for candidate in ops:
            if candidate.op != "tt.reduce" or not candidate.operand_ids:
                continue
            if not any("max" in (nested.op or "") for nested in (candidate.region_ops or [])):
                continue
            input_core_id, _ = _peel(candidate.operand_ids[0], value_wrappers)
            if input_core_id == score_core_id:
                max_reductions.append(candidate)
        max_result_ids = {
            result_id
            for reduction in max_reductions
            for result_id in ([reduction.id] + list(getattr(reduction, "result_ids", None) or []))
        }
        if not max_result_ids or not _depends_on(core.operand_ids[1], max_result_ids):
            return None

        # Walk the value-minus-max score operand back to the QK dot.  Masks are
        # admitted only as transparent value selectors here; their exact predicates
        # are independently proven by the dense/varlen detectors before this result
        # is used for routing.
        cur = score_value_id
        factor = 1.0
        scale_ops = []
        select_ids = []
        seen = set()
        while cur not in seen and len(seen) < 128:
            if cur in dot_ids:
                if not (_math.isfinite(factor) and factor > 0.0):
                    return None
                return {"factor": factor, "op_ids": tuple(scale_ops), "select_ids": tuple(select_ids)}
            seen.add(cur)
            op = by_id.get(cur)
            if op is None:
                return None
            if op.op in value_wrappers and len(op.operand_ids or []) == 1:
                cur = op.operand_ids[0]
                continue
            if op.op == "arith.select" and len(op.operand_ids or []) == 3:
                # Packet 114: on a route that supplies its proof object, ONLY the selects that
                # proof certified (predicate, polarity, sentinel) are transparent here.
                if certified_selects is not None and op.id not in certified_selects:
                    return None
                branches = [oid for oid in op.operand_ids[1:] if _depends_on_dot(oid)]
                if len(branches) != 1:
                    return None
                select_ids.append(op.id)
                cur = branches[0]
                continue
            if op.op == "arith.mulf" and len(op.operand_ids or []) == 2:
                data = [oid for oid in op.operand_ids if _depends_on_dot(oid)]
                if len(data) != 1:
                    return None
                other = next(oid for oid in op.operand_ids if oid != data[0])
                value = _constant_float(other)
                if value is None:
                    return None
                factor *= value
                scale_ops.append(op.id)
                cur = data[0]
                continue
            if op.op == "arith.divf" and len(op.operand_ids or []) == 2:
                numerator, denominator = op.operand_ids
                if not _depends_on_dot(numerator) or _depends_on_dot(denominator):
                    return None
                value = _constant_float(denominator)
                if value is None or value == 0.0:
                    return None
                factor /= value
                scale_ops.append(op.id)
                cur = numerator
                continue
            return None
        return None

    def _detect_varlen_flash_attention(self):
        """Recognize a VARLEN (packed cu_seqlens) FlashAttention-2 forward kernel and
        resolve its ABI for the tiled varlen template. Returns None (correct-or-refuse)
        for anything that is not an UNAMBIGUOUS non-causal varlen FA — the caller then
        falls through to the dense-FA path (which loudly refuses varlen) so a
        mis-recognition can never silently mis-compute.

        The varlen signature (flash-attn / vLLM ABI) is highly distinctive and CANNOT
        collide with the dense/biased/MLA FA kernels:
          * exactly 4 float pointer args (Q,K,V,Out) + exactly 2 int pointer args
            (cu_seqlens_q, cu_seqlens_k) — dense FA has 0 int ptrs, biased/MLA have
            >4 float ptrs;
          * exactly 2 dots + a math.exp (FA2 online softmax);
          * exactly 2 ``arith.subi``, each = (load(cu[b+1]) - load(cu[b])) over the SAME
            int pointer, across 2 DISTINCT cu pointers (the per-batch seqlens);
          * heads count H = the divisor of a divsi/remsi(program_id, H).
        Both NON-CAUSAL and standard within-sequence lower-triangular CAUSAL route: causal
        is recognized PRECISELY (a row-vs-col ``>=`` where the row reaches program_id(0) and
        the col does not) and any other row-vs-col compare (strict ``>``, reversed, absolute,
        windowed) refuses -> the generic path, never a wrong mask.
        """

        def _flat(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _flat(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _flat(s.else_ops)

        def _is_int(et):
            return (et or "").lstrip("s").lstrip("u").startswith("i")

        def _is_float(et):
            return (et or "").startswith("f") or (et or "").startswith("bf")  # f16/f32 + bf16

        ops = list(_flat(self.graph.ops))
        obid = {o.id: o for o in ops}
        args = self.graph.args
        argid = {getattr(a, "id", None): a for a in args}
        ptrs = [a for a in args if a.is_ptr]
        int_ptrs = [a for a in ptrs if _is_int(getattr(a, "elem_type", None))]
        flt_ptrs = [a for a in ptrs if _is_float(getattr(a, "elem_type", None))]
        if len(flt_ptrs) != 4 or len(int_ptrs) != 2:
            return None
        dots = [o for o in ops if o.op == "tt.dot"]
        if len(dots) != 2:
            return None
        if not any(_op_is_exp(o.op) for o in ops):
            return None

        def _trace_to_arg(oid, d=0):
            if d > 40:
                return None
            if oid in argid:
                return argid[oid]
            o = obid.get(oid)
            if o is None or not o.operand_ids:
                return None
            return _trace_to_arg(o.operand_ids[0], d + 1)

        def _load_ptr_arg(load):
            return _trace_to_arg(load.operand_ids[0]) if load.operand_ids else None

        # exactly 2 subi, each subtracting two i32 loads of the SAME int ptr; 2 distinct cus
        subis = [o for o in ops if o.op == "arith.subi" and len(o.operand_ids or []) == 2]
        seqlen_subis = []
        cu_used = set()
        for s in subis:
            lo0, lo1 = obid.get(s.operand_ids[0]), obid.get(s.operand_ids[1])
            if lo0 is None or lo1 is None or lo0.op != "tt.load" or lo1.op != "tt.load":
                continue
            p0, p1 = _load_ptr_arg(lo0), _load_ptr_arg(lo1)
            if p0 is None or p0 is not p1 or not _is_int(getattr(p0, "elem_type", None)):
                continue
            seqlen_subis.append((s, p0))
            cu_used.add(p0.index)
        if len(seqlen_subis) != 2 or len(cu_used) != 2:
            return None

        # H = divsi/remsi(program_id, H_arg) divisor
        H_arg = None
        for o in ops:
            if o.op in ("arith.divsi", "arith.remsi", "arith.divui", "arith.remui") and o.operand_ids:
                if any(
                    obid.get(x) is not None and obid.get(x).op in ("tt.get_program_id", "tt.program_id")
                    for x in o.operand_ids
                ):
                    for x in o.operand_ids:
                        a = argid.get(x)
                        if a is not None and not a.is_ptr:
                            H_arg = a
            if H_arg is not None:
                break
        if H_arg is None:
            return None

        def _find_load(oid, seen=None, d=0):
            if seen is None:
                seen = set()
            if oid in seen or d > 40:
                return None
            seen.add(oid)
            o = obid.get(oid)
            if o is None:
                return None
            if o.op == "tt.load":
                return o
            for x in o.operand_ids or []:
                r = _find_load(x, seen, d + 1)
                if r is not None:
                    return r
            return None

        d0, d1 = dots
        q_load = _find_load(d0.operand_ids[0])
        k_load = _find_load(d0.operand_ids[1])
        v_load = _find_load(d1.operand_ids[1]) or _find_load(d1.operand_ids[0])
        stores = [o for o in ops if o.op == "tt.store"]
        if not stores or q_load is None or k_load is None or v_load is None:
            return None
        q_arg = _load_ptr_arg(q_load)
        k_arg = _load_ptr_arg(k_load)
        v_arg = _load_ptr_arg(v_load)
        o_arg = _trace_to_arg(stores[0].operand_ids[0])
        if any(r is None for r in (q_arg, k_arg, v_arg, o_arg)):
            return None
        if not (_is_float(q_arg.elem_type) and _is_float(o_arg.elem_type)):
            return None

        # tile shapes from the load tensor types: Q=[BM,D], K/V=[BN,D]. The template
        # HARD-CODES BM==BN==32 and loops a single head_dim D over Q/K/V, so:
        #  * a kernel with BLOCK_M != 32 launches cdiv(seqlen, BLOCK_M) blocks but the
        #    template advances 32 rows/block -> rows past 32*nblocks are never written
        #    (silent-wrong); likewise BLOCK_N != 32 mis-stages the K/V tile;
        #  * a different Q vs K/V head_dim (asymmetric) mis-sizes the K/V loop.
        # Require BM==BN==32 and a single shared head_dim; otherwise refuse (-> generic).
        import re

        def _shape(o):
            mm = re.search(r"tensor<(\d+)x(\d+)x", getattr(o, "type_str", "") or "")
            return (int(mm.group(1)), int(mm.group(2))) if mm else None

        qsh, ksh, vsh = _shape(q_load), _shape(k_load), _shape(v_load)
        if qsh is None or ksh is None or vsh is None:
            return None
        head_dim = qsh[1]
        if qsh[0] != 32 or ksh[0] != 32 or vsh[0] != 32:
            return None  # template is fixed BLOCK_M==BLOCK_N==32
        if ksh[1] != head_dim or vsh[1] != head_dim:
            return None  # Q/K/V must share head_dim (symmetric); asymmetric -> generic
        _od = {"f32": "f32", "f16": "f16", "bf16": "bf16"}.get(o_arg.elem_type)
        if _od is None:
            return None  # varlen template is fp32/fp16/bf16 only

        def _cone_has(oid, pred, seen=None, d=0):
            if seen is None:
                seen = set()
            if oid in seen or d > 48:
                return False
            seen.add(oid)
            o = obid.get(oid)
            if o is None:
                return False
            if pred(o):
                return True
            return any(_cone_has(x, pred, seen, d + 1) for x in (o.operand_ids or []))

        _is_mr = lambda o: o.op == "tt.make_range"
        _is_cu_load = lambda o: (
            o.op == "tt.load" and (lambda p: p is not None and _is_int(getattr(p, "elem_type", None)))(_load_ptr_arg(o))
        )
        _is_offh = lambda o: o.op in ("arith.remsi", "arith.remui", "arith.divsi", "arith.divui")

        def _muli_terms(addr_oid, seen=None, acc=None):
            if seen is None:
                seen, acc = set(), []
            if addr_oid in seen:
                return acc
            seen.add(addr_oid)
            o = obid.get(addr_oid)
            if o is None:
                return acc
            if o.op == "arith.muli":
                acc.append(o)
            for x in o.operand_ids or []:
                _muli_terms(x, seen, acc)
            return acc

        C1 = "c1"

        def _resolve_strides(load):
            # (token_idx, head_idx, dim_entry): token/head = int arg indices (required);
            # dim = C1 when folded (equal_to_1) else the stride arg index. Classify each
            # address muli by its index: cu-load -> token, remsi/divsi(pid) -> head,
            # make_range -> dim. Order matters (token's cu-load address also carries an
            # off_z divsi, so cu-load is tested first).
            st = sh = None
            sk = C1
            for mul in _muli_terms(load.operand_ids[0]):
                if len(mul.operand_ids or []) != 2:
                    continue
                a0, a1 = mul.operand_ids
                sa0, sa1 = _trace_to_arg(a0), _trace_to_arg(a1)
                if sa0 is not None and not sa0.is_ptr and (sa1 is None or sa1.is_ptr):
                    stride_arg, idx = sa0, a1
                elif sa1 is not None and not sa1.is_ptr and (sa0 is None or sa0.is_ptr):
                    stride_arg, idx = sa1, a0
                else:
                    continue
                if _cone_has(idx, _is_cu_load):
                    st = stride_arg.index
                elif _cone_has(idx, _is_offh):
                    sh = stride_arg.index
                elif _cone_has(idx, _is_mr):
                    sk = stride_arg.index
            return st, sh, sk

        def _cu_in_cone(load):
            found = set()

            def walk(oid, seen=None, d=0):
                if seen is None:
                    seen = set()
                if oid in seen or d > 48:
                    return
                seen.add(oid)
                o = obid.get(oid)
                if o is None:
                    return
                if o.op == "tt.load":
                    p = _load_ptr_arg(o)
                    if p is not None and _is_int(getattr(p, "elem_type", None)):
                        found.add(p.index)
                for x in o.operand_ids or []:
                    walk(x, seen, d + 1)

            walk(load.operand_ids[0])
            return found

        q = _resolve_strides(q_load)
        k = _resolve_strides(k_load)
        v = _resolve_strides(v_load)
        o = _resolve_strides(stores[0])
        # token + head strides MUST resolve for all four (a folded/unresolved token or
        # head offset would silently address the wrong row/head).
        if any(t[0] is None or t[1] is None for t in (q, k, v, o)):
            return None

        # GQA/MQA GUARD (correct-or-refuse): the template indexes K/V with the SAME head
        # index as Q (h = bh % nheads). Grouped-query attention indexes K/V by a DIFFERENT
        # head (off_h_kv = off_h // group, fewer kv heads), which the template would silently
        # mis-address. Require Q, K, V to share the SAME head-index SSA (peeled through
        # layout ops); a K/V head that goes through an extra divsi (GQA) or a different
        # remsi (MQA) differs -> refuse (return None -> generic stays correct).
        def _head_index(load):
            for mul in _muli_terms(load.operand_ids[0]):
                if len(mul.operand_ids or []) != 2:
                    continue
                a0, a1 = mul.operand_ids
                sa0, sa1 = _trace_to_arg(a0), _trace_to_arg(a1)
                if sa0 is not None and not sa0.is_ptr and (sa1 is None or sa1.is_ptr):
                    idx = a1
                elif sa1 is not None and not sa1.is_ptr and (sa0 is None or sa0.is_ptr):
                    idx = a0
                else:
                    continue
                if not _cone_has(idx, _is_cu_load) and _cone_has(idx, _is_offh):
                    return idx
            return None

        def _peel_core(oid):
            op = obid.get(oid)
            seen = set()
            while (
                op is not None
                and op.id not in seen
                and op.operand_ids
                and op.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape")
            ):
                seen.add(op.id)
                op = obid.get(op.operand_ids[0])
            return op.id if op is not None else None

        # Re-review 2026-08-25: the OUTPUT store's head index must ALSO match (a kernel
        # writing O at off_h+1 otherwise routed and the template wrote the WRONG head).
        qh_i, kh_i, vh_i, oh_i = (_head_index(q_load), _head_index(k_load), _head_index(v_load), _head_index(stores[0]))
        if qh_i is None or kh_i is None or vh_i is None or oh_i is None:
            return None
        qh_c, kh_c, vh_c, oh_c = (_peel_core(qh_i), _peel_core(kh_i), _peel_core(vh_i), _peel_core(oh_i))
        if not (qh_c is not None and qh_c == kh_c == vh_c == oh_c):
            return None

        cuq = _cu_in_cone(q_load)
        cuk = _cu_in_cone(k_load)
        if len(cuq) != 1 or len(cuk) != 1:
            return None
        cuq_idx, cuk_idx = next(iter(cuq)), next(iter(cuk))
        if cuq_idx == cuk_idx or {cuq_idx, cuk_idx} != cu_used:
            return None

        # CU-LOAD PIN (re-review 2026-08-25): the token offset in each address must use the
        # SEQUENCE-START cu load — the SUBTRAHEND of that pointer's seqlen subi
        # (q_start = load(cu[b]); seqlen = load(cu[b+1]) - q_start). A kernel addressing Q
        # at load(cu[b+1]) (or any other cu read) otherwise routed while the template
        # hard-codes cu[b]. Pin Q + the output store to cu_q's subtrahend, K/V to cu_k's.
        _start_load_of = {}

        def _int_constant(oid):
            op = obid.get(oid)
            if op is None or op.op != "arith.constant":
                return None
            try:
                return int(str((op.attrs or {}).get("value", "")).split(":")[0].strip())
            except (TypeError, ValueError):
                return None

        def _canonical_cu_pair(seqlen_subi, cu_arg):
            """Classify an exact adjacent CU pair by its batch-index convention."""
            end_load = obid.get(seqlen_subi.operand_ids[0])
            start_load = obid.get(seqlen_subi.operand_ids[1])
            if end_load is None or start_load is None:
                return "unrecognized"
            end_addr = obid.get(end_load.operand_ids[0]) if end_load.operand_ids else None
            start_addr = obid.get(start_load.operand_ids[0]) if start_load.operand_ids else None
            if (
                end_addr is None
                or start_addr is None
                or end_addr.op != "tt.addptr"
                or start_addr.op != "tt.addptr"
                or len(end_addr.operand_ids or []) != 2
                or len(start_addr.operand_ids or []) != 2
                or start_addr.operand_ids[0] != getattr(cu_arg, "id", None)
            ):
                return "unrecognized"

            def _base_plus_constant(oid):
                """Return an exact ``(base SSA, constant)`` for base or base+constant."""
                core_id = _peel_core(oid)
                core = obid.get(core_id)
                seen = set()
                while (
                    core is not None
                    and core.id not in seen
                    and core.op in ("arith.extui", "arith.extsi", "arith.index_cast")
                    and len(core.operand_ids or []) == 1
                ):
                    seen.add(core.id)
                    core_id = _peel_core(core.operand_ids[0])
                    core = obid.get(core_id)
                if core is None or core.op != "arith.addi" or len(core.operand_ids or []) != 2:
                    return core_id, 0
                lhs, rhs = core.operand_ids
                lhs_c, rhs_c = _int_constant(lhs), _int_constant(rhs)
                if lhs_c is not None and rhs_c is None:
                    return rhs, lhs_c
                if rhs_c is not None and lhs_c is None:
                    return lhs, rhs_c
                return None

            batch_off = start_addr.operand_ids[1]
            start_form = _base_plus_constant(batch_off)
            if start_form is None:
                return "unrecognized"
            batch_base, start_delta = start_form

            # Triton emits canonical ``cu + b + 1`` as addptr(addptr(cu, b), 1).
            if end_addr.operand_ids[0] == start_addr.id:
                adjacent = _int_constant(end_addr.operand_ids[1]) == 1
            else:
                # Also accept the equivalent fused form addptr(cu, addi(b, 1)). Keep the
                # relationship SSA-exact: both indices must reduce to the same base with
                # constants differing by exactly one.
                adjacent = False
                if end_addr.operand_ids[0] == getattr(cu_arg, "id", None):
                    end_form = _base_plus_constant(end_addr.operand_ids[1])
                    if end_form is not None:
                        end_base, end_delta = end_form
                        adjacent = end_base == batch_base and end_delta == start_delta + 1
            if not adjacent:
                return "unrecognized"

            core = obid.get(_peel_core(batch_base))
            if core is None or len(core.operand_ids or []) != 2:
                return "unrecognized"
            divisor = _trace_to_arg(core.operand_ids[1])
            pid_h = (
                _cone_has(
                    core.operand_ids[0],
                    lambda o: o.op in ("tt.get_program_id", "tt.program_id"),
                )
                and divisor is not None
                and divisor.index == H_arg.index
            )
            if core.op in ("arith.divsi", "arith.divui") and pid_h:
                return "canonical" if start_delta == 0 else "shifted"
            if core.op in ("arith.remsi", "arith.remui") and pid_h and start_delta == 0:
                return "batch_inner"
            return "unrecognized"

        for _s, _p in seqlen_subis:
            _sub_op = obid.get(_s.operand_ids[1])
            _pair_kind = _canonical_cu_pair(_s, _p)
            if _sub_op is None or _sub_op.op != "tt.load" or _pair_kind == "unrecognized":
                return None
            if _pair_kind == "batch_inner":
                return None
            if _pair_kind == "shifted":
                raise MetalNonRecoverableError(
                    "varlen FlashAttention cannot route a proven shifted cu_seqlens "
                    "convention; the generic fallback is not yet safe for this dot layout."
                )
            _start_load_of[_p.index] = _sub_op.id

        def _cu_load_ids(load):
            found = set()

            def walk(oid, seen=None, d=0):
                if seen is None:
                    seen = set()
                if oid in seen or d > 48:
                    return
                seen.add(oid)
                o = obid.get(oid)
                if o is None:
                    return
                if o.op == "tt.load":
                    p = _load_ptr_arg(o)
                    if p is not None and _is_int(getattr(p, "elem_type", None)):
                        found.add(o.id)
                for x in o.operand_ids or []:
                    walk(x, seen, d + 1)

            walk(load.operand_ids[0])
            return found

        if _cu_load_ids(q_load) != {_start_load_of[cuq_idx]}:
            return None
        if _cu_load_ids(stores[0]) != {_start_load_of[cuq_idx]}:
            return None
        for _ld in (k_load, v_load):
            if _cu_load_ids(_ld) != {_start_load_of[cuk_idx]}:
                return None

        # MAX_SEQLEN BOUND (re-review 2026-08-25 V1): the kernel's kv loop runs to its own
        # ``max_seqlen`` ARG (masking kn < seqlen_k), i.e. it attends the keys below seqlen_k
        # in its range(0, max_seqlen, 32) tiles — but the templates looped to seqlen_k
        # unconditionally, silently attending MORE keys whenever max_seqlen < seqlen_k (err
        # ~0.6-0.8). Resolve the scf.for upper bound to its arg so the templates can clamp
        # (to max_seqlen rounded up to a whole tile: _varlen_kmax_expr); unresolvable -> refuse.
        _ms_arg = None
        for _sf in ops:
            if _sf.op != "scf.for" or len(_sf.operand_ids or []) < 2:
                continue
            _b = _sf.operand_ids[1]
            for _ in range(8):
                if _b in argid:
                    break
                _bo = obid.get(_b)
                if _bo is None or not _bo.operand_ids:
                    break
                _b = _bo.operand_ids[0]
            _a = argid.get(_b)
            if _a is None or _a.is_ptr:
                return None  # kv-loop bound isn't a plain scalar arg -> refuse
            if _ms_arg is not None and _ms_arg != _a.index:
                return None  # multiple loops with different bounds -> refuse
            _ms_arg = _a.index
        if _ms_arg is None:
            return None

        # CONVENTION GUARD (correct-or-refuse): the template assumes HEAD-INNER batch/head
        # packing -- h = bh % nheads, batch = bh // nheads. Verify off_h = remsi(pid, H) and
        # the cu_seqlens batch offset = divsi(pid, H), same H. A batch-inner kernel
        # (off_h = pid // NB) otherwise routes with NB mistaken for H, swapping batch<->head
        # (silent-wrong) AND reading cu_seqlens out of bounds (garbage seqlen -> runaway loop).
        _pid = lambda o: o.op in ("tt.get_program_id", "tt.program_id")

        def _div_by_H(op, kinds):
            if op is None or op.op not in kinds or len(op.operand_ids or []) != 2:
                return False
            num, den = op.operand_ids
            da = _trace_to_arg(den)
            return da is not None and da.index == H_arg.index and _cone_has(num, _pid)

        if not _div_by_H(obid.get(_peel_core(qh_i)), ("arith.remsi", "arith.remui")):
            return None

        def _cu_batch_divH(cu_idx):
            for ld in ops:
                if ld.op != "tt.load" or not ld.operand_ids:
                    continue
                p = _load_ptr_arg(ld)
                if p is None or p.index != cu_idx:
                    continue
                addr = obid.get(ld.operand_ids[0])
                if addr is None or addr.op != "tt.addptr" or len(addr.operand_ids or []) < 2:
                    continue
                if _cone_has(addr.operand_ids[1], lambda o: _div_by_H(o, ("arith.divsi", "arith.divui"))):
                    return True
            return False

        if not (_cu_batch_divH(cuq_idx) and _cu_batch_divH(cuk_idx)):
            return None

        # --- causal: PRECISE correct-or-refuse. The template's causal_guard is the standard
        # within-sequence lower-triangular ``qrow >= krow``. Recognize EXACTLY that and route
        # causal=True; refuse (return None -> generic stays correct) on any other row-vs-col
        # compare (strict >, reversed direction, absolute q_start/k_start positions, windowed).
        # A "row-vs-col compare" = a cmpi whose BOTH operand cones reach a make_range (the
        # bounds masks compare a make_range to a scalar seqlen, so only ONE side is a range).
        # Direction is read from program-vs-loop dependence, NOT make_range identity: with
        # BLOCK_M==BLOCK_N the offs_m/offs_n aranges are CSE'd to ONE op, so identity can't
        # tell row from col. The query ROW (offs_m = pid(0)*BM + arange) reaches
        # program_id(0); the kv COL (kn = loop_start + arange) reaches NO program_id. Require
        # LHS=row (pid0, within-seq), RHS=col (a make_range, no program_id, within-seq).
        _pid0 = lambda o: o.op == "tt.get_program_id" and int(o.attrs.get("axis", 0)) == 0
        _anypid = lambda o: o.op == "tt.get_program_id"
        both_mr = [
            c
            for c in ops
            if c.op == "arith.cmpi"
            and len(c.operand_ids or []) == 2
            and _cone_has(c.operand_ids[0], _is_mr)
            and _cone_has(c.operand_ids[1], _is_mr)
        ]
        causal = False
        if both_mr:

            def _std_causal(c):
                if c.attrs.get("predicate_name") not in ("sge", "uge"):
                    return False  # only >= (lower-tri incl. diagonal); strict/reversed -> refuse
                lhs, rhs = c.operand_ids
                return (
                    _cone_has(lhs, _pid0)
                    and not _cone_has(lhs, _is_cu_load)
                    and _cone_has(rhs, _is_mr)
                    and not _cone_has(rhs, _anypid)
                    and not _cone_has(rhs, _is_cu_load)
                )

            if all(_std_causal(c) for c in both_mr):
                causal = True
            else:
                return None

        # Softmax scale: accept either the traditional single constant multiplied into
        # Q before dot0, or a completely proven constant multiply/divide chain on the
        # dot RESULT before softmax (#6a).  The template bakes their PRODUCT.  Runtime
        # factors, score bias, P-side scaling, and any unrecognized value operation stay
        # outside the detector and are refused by the common FA guard.
        _LAYOUT = (
            "ttg.local_load",
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "tt.trans",
            "tt.reshape",
            "ttg.convert_layout",
            "tt.broadcast",
            "tt.expand_dims",
            "arith.truncf",
            "arith.extf",
            "tt.fp_to_fp",
        )

        def _const_factors(oid):
            """All arith.constant factors multiplied into this dot-operand cone (walked
            through layout/cast ops and CHAINED mulfs, stopping at the load). Re-review
            2026-08-25: the old walk baked only the FIRST mulf's constant, silently
            DROPPING any other factor — a split scale (q*=a; k*=b, or q*a*b) mis-scaled."""
            out = []
            stack, seen3 = [oid], set()
            while stack:
                cur = stack.pop()
                if cur in seen3:
                    continue
                seen3.add(cur)
                op = obid.get(cur)
                if op is None:
                    continue
                if op.op == "arith.mulf":
                    for x in op.operand_ids or []:
                        sub = obid.get(x)
                        if sub is not None and sub.op == "arith.constant":
                            val = sub.attrs.get("value")
                            if isinstance(val, (int, float)):
                                out.append(float(val))
                                continue
                        stack.append(x)
                elif op.op in _LAYOUT and op.operand_ids:
                    stack.append(op.operand_ids[0])
                # anything else (tt.load, splat of a runtime scalar, ...) terminates
            return out

        _qf = _const_factors(d0.operand_ids[0])
        _kf = _const_factors(d0.operand_ids[1])
        _vf = _const_factors(d1.operand_ids[0]) + _const_factors(d1.operand_ids[1])
        _score_scale = self._fa_analyze_constant_score_scale(d0, d1)
        # At most one Q-side constant; no K-, V-, or P-side factor.  A post-dot
        # constant chain can supply all or part of the effective scale, but an entirely
        # unscaled kernel remains outside the specialized template's contract.
        if len(_qf) > 1 or _kf or _vf or _score_scale is None:
            return None
        if not _qf and not _score_scale["op_ids"]:
            return None
        scale = (_qf[0] if _qf else 1.0) * _score_scale["factor"]

        # the verified per-batch seqlen values (the cu_seqlens subi results) — the ONLY
        # bounds the varlen template clips with (packet 112 load/store mask proof)
        _seqlen_id_of = {p.index: s.id for s, p in seqlen_subis}
        if cuq_idx not in _seqlen_id_of or cuk_idx not in _seqlen_id_of:
            return None
        return {
            "roles": {
                "q": q_arg.index,
                "k": k_arg.index,
                "v": v_arg.index,
                "o": o_arg.index,
                "cuq": cuq_idx,
                "cuk": cuk_idx,
            },
            "H": H_arg.index,
            "head_dim": head_dim,
            "out_dtype": _od,
            "scale": scale,
            "q_scale": bool(_qf),
            "score_scale": _score_scale,
            "max_seqlen": _ms_arg,
            "causal": causal,
            "q": q,
            "k": k,
            "v": v,
            "o": o,
            "seqlen_q_id": _seqlen_id_of[cuq_idx],
            "seqlen_k_id": _seqlen_id_of[cuk_idx],
        }

    def _witness_varlen_flash_attention(self, info: dict) -> dict:
        """Prove every source address of a varlen kernel is the template's, and bind the strides.

        Template coordinates (``make_varlen_flash_attention`` and its MMA form), with
        ``b = program id 1 / H`` and ``h = program id 1 % H``: Q and Out are addressed at
        ``(cu_q[b] + program id 0 * 32 + row) * st + h * sh + d * sk``; K and V at
        ``(cu_k[b] + loop start + row) * st + h * sh + d * sk`` over ``range(0, max_seqlen, 32)``;
        the tables are read at ``b`` and ``b + 1`` only. The detector's stride lists are
        positional ([token, head, depth]); see ``_address_witness``.
        """
        from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

        roles, heads, bound_arg = info["roles"], info["H"], info["max_seqlen"]
        head_dim = int(info["head_dim"])
        if type(heads) is not int or type(bound_arg) is not int:
            raise MetalNonRecoverableError(
                "varlen FlashAttention: the head count or the key-loop bound is not an argument; refusing."
            )
        batch, head = ("div", 1, ("arg", heads)), ("rem", 1, ("arg", heads))

        def _start(table):
            return lambda env: ("load", table, env.atom(batch))

        def _entries(env):
            return (env.atom(batch), env.atom(batch) + 1)

        def _rows(atom):
            return lambda env: [env.atom(atom) * 32 + i for i in range(32)]

        def _role(key, name, access, table, rows):
            return {
                "key": key,
                "name": name,
                "arg": roles[key],
                "kind": access,
                "strides": list(info[key]),
                "shape": (32, head_dim),
                "scalars": [(0, _start(table)), (1, head)],
                "axes": [(0, 0, rows), (2, 1, lambda env: list(range(head_dim)))],
            }

        spec = [
            _role("q", "Q", "load", roles["cuq"], _rows(("pid", 0))),
            _role("k", "K", "load", roles["cuk"], _rows(("t",))),
            _role("v", "V", "load", roles["cuk"], _rows(("t",))),
            _role("o", "Out", "store", roles["cuq"], _rows(("pid", 0))),
            {"name": "cu_seqlens_q", "arg": roles["cuq"], "table": _entries},
            {"name": "cu_seqlens_k", "arg": roles["cuk"], "table": _entries},
        ]
        try:
            bound = prove_template_addresses(self.graph, spec, (bound_arg, 32))
        except WitnessRefusal as exc:
            raise MetalNonRecoverableError(
                f"varlen FlashAttention: {exc}. The template replays its own coordinates and cannot "
                "prove they are the source's; refusing."
            ) from None
        proved = dict(info)
        for role in spec[:4]:
            proved[role["key"]] = bound[role["name"]]
        return proved

    def _lower_varlen_flash_attention(self, info: dict) -> str:
        """Emit the tiled varlen FA-2 template (make_varlen_flash_attention) for a detected
        packed-cu_seqlens kernel + set the native-2-D-grid ('flash_attention', msl, BM=32)
        dispatch descriptor. info from _detect_varlen_flash_attention (which refuses on any
        ambiguity, so every arg/stride here resolved unambiguously)."""
        from triton_msl.codegen._msl_templates import make_varlen_flash_attention

        C1 = "c1"
        MAX_BUFFERS = 31  # mirrors driver._MAX_METAL_BUFFERS
        args = self.graph.args
        if len(args) > MAX_BUFFERS:
            raise MetalNonRecoverableError(
                "varlen FlashAttention with >31 args (packed-scalar ABI) is not supported; "
                "refusing rather than risk a mis-bound buffer."
            )
        for i, a in enumerate(args):
            if i != a.index:
                raise MetalNonRecoverableError("varlen FlashAttention arg list is not densely indexed; refusing.")
        roles = info["roles"]
        # Packet 112: varlen loads are plain or exactly ``pid*32 + range(0,32) < seqlen_q``
        # (Q) / ``iv + range(0,32) < seqlen_k`` (K, V) on the detector's verified seqlen
        # values; the Out store MUST carry exactly the seqlen_q boundary (the template stores
        # only inside the sequence; an unmasked or tighter source store is not equivalent).
        info["scale"] *= self._fa_verify_value_paths(
            detected_q_index=roles["q"],
            detected_k_index=roles["k"],
            detected_v_index=roles["v"],
            detected_out_index=roles["o"],
            q_scale_required=info.get("q_scale", True),
            k_transposes=1,
            verify_score_path=True,
            expected_score_scale=info.get("score_scale"),
            load_mask_policy="varlen",
            varlen_bounds=(info["seqlen_q_id"], info["seqlen_k_id"]),
            varlen_causal=bool(info.get("causal")),
            q_rows=32,
            kv_cols=32,
            extra_stores=[(roles["o"], "Out", ("Q",), True)],
        )
        info = self._witness_varlen_flash_attention(info)
        # ``index < length`` with a length that came out negative: no index when the source
        # compares signed, every launched index when it compares unsigned (uint32 or uint64
        # cumulative lengths). The template computes each length the way its masks read it.
        _signs = [self._fa_boundary_signs.get(r, set()) for r in ("Q", "K")]
        if any(s not in ({"s"}, {"u"}) for s in _signs):
            raise MetalNonRecoverableError(
                "varlen FlashAttention compares one sequence length as signed in one mask and "
                "unsigned in another; the template holds one length per sequence. Refusing."
            )
        _unsigned_q, _unsigned_k = (s == {"u"} for s in _signs)
        role_of_idx = {
            roles["q"]: "Q",
            roles["k"]: "K",
            roles["v"]: "V",
            roles["o"]: "Out",
            roles["cuq"]: "CUQ",
            roles["cuk"]: "CUK",
        }
        arg_decls = []
        for a in args:
            if a.is_ptr:
                role = role_of_idx.get(a.index)
                if role is None:
                    raise MetalNonRecoverableError(
                        "varlen FlashAttention has a pointer arg with no resolved role; refusing."
                    )
                if role in ("CUQ", "CUK"):
                    # The template indexes the cumulative lengths as stored: an int64 tensor
                    # read through ``int*`` gives every second entry a zero high word. The mask
                    # proof refuses a narrower length before this point; the check stays so the
                    # template never declares a width it does not index.
                    if a.elem_type not in ("i32", "i64"):
                        raise MetalNonRecoverableError(
                            "varlen FlashAttention cumulative sequence lengths are neither int32 nor int64; refusing."
                        )
                    arg_decls.append(
                        f"    device const {triton_type_to_msl(a.elem_type)}* {role} [[buffer({a.index})]]"
                    )
                else:
                    msl_ty = triton_type_to_msl(a.elem_type)
                    qual = "device" if role == "Out" else "device const"
                    arg_decls.append(f"    {qual} {msl_ty}* {role} [[buffer({a.index})]]")
            else:
                arg_decls.append(f"    constant uint& bsc_{a.index} [[buffer({a.index})]]")

        unsigned_args, signed_args = set(), set()

        def _uint_expr(entry, signed=False):
            if entry == C1:
                return "1u"
            if isinstance(entry, int):
                (signed_args if signed else unsigned_args).add(entry)
                return f"bsc_{entry}"
            raise MetalNonRecoverableError(
                "varlen FlashAttention stride/dim could not be mapped to a kernel arg; refusing."
            )

        q, k, v, o = info["q"], info["k"], info["v"], info["o"]
        bindings = {
            "Q": "Q",
            "K": "K",
            "V": "V",
            "O": "Out",
            "CUQ": "CUQ",
            "CUK": "CUK",
            "H": _uint_expr(info["H"]),
            # The key-extent expression reads this one as ``int`` (_varlen_kmax_expr).
            "MAXS": _uint_expr(info["max_seqlen"], signed=True),
            "q_st": _uint_expr(q[0]),
            "q_sh": _uint_expr(q[1]),
            "q_sk": _uint_expr(q[2]),
            "k_st": _uint_expr(k[0]),
            "k_sh": _uint_expr(k[1]),
            "k_sk": _uint_expr(k[2]),
            "v_st": _uint_expr(v[0]),
            "v_sh": _uint_expr(v[1]),
            "v_sk": _uint_expr(v[2]),
            "o_st": _uint_expr(o[0]),
            "o_sh": _uint_expr(o[1]),
            "o_sk": _uint_expr(o[2]),
            "unsigned_q": _unsigned_q,
            "unsigned_k": _unsigned_k,
            "wide_q": any(a.index == roles["cuq"] and a.elem_type == "i64" for a in args),
            "wide_k": any(a.index == roles["cuk"] and a.elem_type == "i64" for a in args),
        }
        # FAST PATH: the register-O simdgroup-MMA varlen kernel (~7.6x the scalar one-thread-
        # per-row template at D=64, ~4x pad+SDPA; ~1.4x pad+SDPA at D=128) when eligible —
        # fp16/bf16 (2-byte tiles, BN=32, hd<=128) or fp32 (TRUE float tiles, BN=16,
        # hd<=~112), head_dim %% 8 == 0, tiles within the 32KB tg budget. Q/K/V stage into
        # threadgroup memory before the MMA so ANY global strides work; O stays register-
        # resident (diag-MMA rescale). Ineligible dims fall to the scalar template.
        D = info["head_dim"]
        # register-O MMA tg budget, matching make_varlen_flash_attention_mma: fp16/bf16 use
        # 2-byte tiles at BN=32; fp32 uses true float tiles at BN=16 (so the doubled bytes
        # still fit 32KB). tg = BM*D*TB + BN*D*TB*2 + BM*BN*TB + BM*BN*4 + BM*4*2 + 4*64*4.
        _tb, _bn = (4, 16) if info["out_dtype"] == "f32" else (2, 32)
        _mma_tg_bytes = 32 * D * _tb + _bn * D * _tb * 2 + 32 * _bn * _tb + 32 * _bn * 4 + 32 * 4 * 2 + 4 * 64 * 4
        mma_eligible = info["out_dtype"] in ("f16", "bf16", "f32") and D % 8 == 0 and _mma_tg_bytes <= 32768
        if mma_eligible:
            from triton_msl.codegen._msl_templates import make_varlen_flash_attention_mma

            msl = make_varlen_flash_attention_mma(
                head_dim=D,
                causal=bool(info.get("causal")),
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                scale=info["scale"],
            )
            self.effective_block_size = 128
            self._flash_attention = ("flash_attention", msl, 128)
        else:
            if 2 * 32 * D * 4 > 32768:  # scalar template stages float tg_K/tg_V tiles
                raise MetalNonRecoverableError(
                    f"varlen FlashAttention head_dim={D} exceeds the scalar template's "
                    "32KB threadgroup budget (float K/V tiles); refusing at lowering "
                    "rather than failing at Metal pipeline creation. Use head_dim <= 128."
                )
            msl = make_varlen_flash_attention(
                head_dim=D,
                causal=bool(info.get("causal")),
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                scale=info["scale"],
            )
            self.effective_block_size = 32
            self._flash_attention = ("flash_attention", msl, 32)
        self._attention_unsigned_scalars(unsigned_args, signed_args)
        self._used_pid_axes = {0, 1}
        self._prescan_stores()
        return msl

    def _detect_flash_attention(self, *, sequence_index=None, unresolved_subject=None):
        """Recognize a FlashAttention forward kernel and extract its params.

        Returns ``None`` when the kernel is NOT an FA pattern (fewer than two
        ``tt.dot`` ops, or no exp + max — i.e. no softmax-with-stability
        between the dots). Returns a fully-resolved param dict when EVERY field
        is recovered unambiguously::

            {
              "q", "k", "v", "out": kernel-arg indices of the four pointers,
              "strides": {"q":[z,h,m,k], "k":[...], "v":[...], "o":[...]},
                          (16 stride kernel-arg indices, 4 per pointer),
              "Z", "H", "N_CTX": kernel-arg indices of the batch/head/ctx
                          scalars,
              "block_m", "block_n", "head_dim": tile dims from the dot shapes,
              "causal": bool, "scale": float, "out_dtype": str,
            }

        Raises ``MetalNonRecoverableError`` when the kernel IS an FA pattern
        (>= 2 dots + exp + max) but ANY field — a pointer role, any of the 16
        strides, a constexpr, or the dtype — cannot be resolved unambiguously.
        This refuse-on-ambiguity is the project's prime directive: a single
        mis-read stride would silently mis-compute, so we never return a
        partially-guessed dict. (Refusing a *non-canonical* FA kernel is the
        correct, safe outcome — Task 3's tiled MSL template only handles the
        canonical addressing.)

        This is pure IR analysis (no GPU). It is NOT yet wired into
        ``lower()`` — Task 3 does the routing.
        """
        from triton_msl.errors import MetalNonRecoverableError

        # --- FA gate: identical to the prescan in lower() ---------------------
        # >= 2 tt.dot with BOTH exp and a max between them (softmax with the
        # numerical-stability max). Requiring the max excludes non-FA >=2-dot
        # kernels that merely use exp (e.g. a fused MLP with a gelu/silu).
        def _fa_walk(ops):
            for _s in ops:
                yield _s
                if getattr(_s, "region_ops", None):
                    yield from _fa_walk(_s.region_ops)
                if getattr(_s, "else_ops", None):
                    yield from _fa_walk(_s.else_ops)

        all_ops = list(_fa_walk(self.graph.ops))
        dots = [s for s in all_ops if s.op == "tt.dot"]
        # Match the exp op (math.exp / math.exp2 / tt.exp) precisely — ``"exp" in
        # op`` would also match ``tt.expand_dims`` and mis-classify a non-FA
        # >=2-dot + max + expand_dims kernel as FA.
        has_exp = any(_op_is_exp(s.op) for s in all_ops)
        has_max = any("max" in (s.op or "") for s in all_ops)
        if not (len(dots) >= 2 and has_exp and has_max):
            return None

        # Past this point the kernel IS an FA pattern: any unresolved field is
        # a hard refusal, never a guess.
        # Before the pointer roles resolve, a kernel that is not attention at all can reach a
        # refusal: its message must not name an attention role. After, the field is a proof
        # about a kernel with attention's roles and is worth naming.
        roles_resolved = False

        def _refuse(field):
            if unresolved_subject is not None:
                raise MetalNonRecoverableError(
                    f"{unresolved_subject}: required pointer/stride roles could not be "
                    "resolved; refusing rather than guess" + (f" (unproved: {field})" if roles_resolved else "")
                )
            raise MetalNonRecoverableError(
                f"FlashAttention recognized but {field} could not be resolved; refusing rather than guess"
            )

        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)

        _collect(self.graph.ops)

        arg_by_id = {a.id: a for a in self.graph.args}

        # The two dots, in IR order: dot 0 = QK^T (operands Q, K), dot 1 = P@V
        # (operand 1 = V; operand 0 = the softmax probs P, which has NO load).
        # Order by position in the flat op list so dot 0 is the QK matmul.
        order = {s.id: i for i, s in enumerate(all_ops)}
        dot_qk, dot_pv = sorted(dots, key=lambda d: order[d.id])[:2]

        # --- helpers: trace an addptr chain back to (base_ptr_arg, 4 strides) -
        def _skip_layout(sid):
            seen = set()
            while sid in op_by_id and sid not in seen:
                seen.add(sid)
                op = op_by_id[sid]
                if op.op == "ttg.convert_layout" and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return sid

        # A stride entry is EITHER a runtime kernel-arg index (int) OR the
        # marker ``"c1"`` (the stride was specialized to the constant 1 and
        # folded out of the arg list). Triton's ``equal_to_1`` specialization
        # elides any arg whose value is 1 (e.g. a contiguous tensor's innermost
        # stride, or Z/H == 1), so the real compiled FA kernel has FEWER than 16
        # stride args. The detector must report which strides are present-as-args
        # vs folded so the template binds only the runtime args and bakes the
        # constants; refusing here would wrongly reject the canonical kernel.
        C1 = "c1"

        def _stride_from_index_term(term_id):
            """An index term contributing ``index * STRIDE`` to an address.
            Returns the stride kernel-arg index (int) for
            ``arith.muli(index, tt.splat(STRIDE_ARG))`` (through an optional
            ``tt.broadcast``); returns ``C1`` when the term is a bare index
            (``broadcast(expand_dims(make_range))`` etc.) — i.e. the stride was
            folded to 1; returns ``None`` only when the term is an ``arith.muli``
            whose multiplier is NOT a kernel-arg splat (genuinely unresolvable)."""
            tid = _skip_layout(term_id)
            op = op_by_id.get(tid)
            if op is None:
                return None
            if op.op == "tt.broadcast" and op.operand_ids:
                return _stride_from_index_term(op.operand_ids[0])
            if op.op != "arith.muli":
                # No explicit multiply: the index is added directly => stride 1
                # (folded). The op is the index expression itself (make_range /
                # expand_dims / splat-of-pid / addi), not an unresolved value.
                return C1
            for oid in op.operand_ids:
                sub = op_by_id.get(oid)
                if sub is not None and sub.op == "tt.splat" and sub.operand_ids:
                    a = arg_by_id.get(sub.operand_ids[0])
                    if a is not None and not a.is_ptr:
                        return a.index
            return None

        def _scalar_stride_from_muli(muli_id):
            """``arith.muli(program_id_or_derived, STRIDE_ARG)`` scalar ->
            stride kernel-arg index, or None."""
            op = op_by_id.get(muli_id)
            if op is None or op.op != "arith.muli":
                return None
            for oid in op.operand_ids:
                a = arg_by_id.get(oid)
                if a is not None and not a.is_ptr:
                    return a.index
            return None

        def _extract_ptr_strides(addr_id):
            """Walk the ``[BLOCK, HEAD_DIM]`` tensor-of-pointers addptr chain
            built by the standard 4-D addressing pattern::

                ptr + off_z*sz + off_h*sh
                    + row_idx[:,None]*srow + col_idx[None,:]*scol

            Returns ``(base_ptr_arg_index, [sz, sh, srow, scol])`` where each
            stride is a runtime kernel-arg index (int) or the marker ``C1``
            (folded to 1). The scalar batch/head offset chain may be 2-level
            (``addptr(addptr(PTR, muli(off_z,SZ)), muli(off_h,SH))``) or
            collapsed to 1-level (when ``off_h*SH`` folded because H==1 ->
            off_h==0); a missing leg is reported as ``C1``. Returns ``None`` only
            if a genuinely-required leg (the terminal addptr / row leg / base
            pointer) is structurally absent or unresolvable.
            """
            if addr_id is None:
                return None
            addr_id = _skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            # terminal: addptr(broadcast(row_ptr_2d), broadcast(col_term))
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            col_stride = _stride_from_index_term(op.operand_ids[1])
            # row side: broadcast -> addptr(splat(scalar_base), row_term)
            row_side = _skip_layout(op.operand_ids[0])
            rop = op_by_id.get(row_side)
            if rop is not None and rop.op == "tt.broadcast" and rop.operand_ids:
                row_side = _skip_layout(rop.operand_ids[0])
                rop = op_by_id.get(row_side)
            if rop is None or rop.op != "tt.addptr" or len(rop.operand_ids) < 2:
                return None
            row_stride = _stride_from_index_term(rop.operand_ids[1])
            splat_base = op_by_id.get(rop.operand_ids[0])
            if splat_base is None or splat_base.op != "tt.splat" or not splat_base.operand_ids:
                return None
            # scalar chain: addptr([addptr(PTR, muli(off_z, SZ))], muli(off_h, SH))
            # (2-level), or addptr(PTR, muli(off_hz, SZ)) (1-level, when off_h*SH
            # folded because H==1 -> off_h==0). Discriminate on whether the outer
            # addptr's pointer operand is the base-ptr ARG (1-level) or another
            # addptr (2-level).
            scal = op_by_id.get(splat_base.operand_ids[0])
            if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                return None
            outer_stride = _scalar_stride_from_muli(scal.operand_ids[1])
            inner_arg = arg_by_id.get(scal.operand_ids[0])
            inner = op_by_id.get(scal.operand_ids[0])
            if inner_arg is not None and inner_arg.is_ptr:
                # 1-level: addptr(PTR, muli(off_hz, SZ)); h-stride folded (H==1).
                z_stride = outer_stride
                h_stride = C1
                base = inner_arg
            elif inner is not None and inner.op == "tt.addptr" and len(inner.operand_ids) >= 2:
                # 2-level: addptr(addptr(PTR, muli(off_z,SZ)), muli(off_h,SH)).
                z_stride = _scalar_stride_from_muli(inner.operand_ids[1])
                h_stride = outer_stride
                base = arg_by_id.get(inner.operand_ids[0])
            else:
                return None
            if base is None or not base.is_ptr:
                return None
            strides = [z_stride, h_stride, row_stride, col_stride]
            # row stride (the BLOCK dimension) must be a real runtime arg or a
            # resolved fold — a None here means the row leg didn't parse.
            if any(s is None for s in strides):
                return None
            # Defensive: the ROW stride (strides[2]) is the BLOCK_M/BLOCK_N-row
            # stride, equal to head_dim for a [Z,H,N_CTX,head_dim] tensor. It is
            # NEVER legitimately 1/folded (head_dim is always >1 for FA). If it
            # resolved to C1, the addptr chain was mis-parsed and we'd silently
            # compute wrong addresses — refuse rather than risk it.  The COLUMN
            # stride (strides[3]) IS legitimately C1 (contiguous innermost stride).
            if strides[2] == C1:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "FlashAttention row stride resolved to 1; expected head_dim "
                    "— refusing rather than risk wrong addressing"
                )
            return base.index, strides

        def _load_addr_for_dot_operand(operand_id, _depth=0):
            """From a dot operand SSA id, walk back through the
            local_load -> (memdesc_trans?) -> local_alloc -> (mulf scale?) ->
            (extf/truncf dtype-cast?) -> tt.load chain to the load's ADDRESS
            operand id, or None when the operand has no backing load (e.g. the
            softmax probs P).

            ``arith.extf``/``arith.truncf`` are walked through because a fp16 FA
            kernel casts a dot operand to a common dtype before the dot
            (``tl.dot(q, tl.trans(k).to(q.dtype))`` — the QK operands must share
            a dtype; for fp16 inputs ``q`` is already fp32 after ``q*scale``, so
            ``k`` widens via ``arith.extf``). These conversions do not change the
            ADDRESS the load reads from, so it is safe to see through them to the
            backing load (and refusing-on-ambiguity still holds — the address
            chain itself is unchanged)."""
            if _depth > 32:
                return None
            sid = operand_id
            seen = set()
            while sid in op_by_id and sid not in seen:
                seen.add(sid)
                op = op_by_id[sid]
                if (
                    op.op
                    in (
                        "ttg.local_load",
                        "ttg.local_alloc",
                        "ttg.memdesc_trans",
                        "tt.trans",
                        "tt.reshape",
                        "ttg.convert_layout",
                        "arith.extf",
                        "arith.truncf",
                    )
                    and op.operand_ids
                ):
                    sid = op.operand_ids[0]
                    continue
                if op.op == "tt.load" and op.operand_ids:
                    return op.operand_ids[0]
                if op.op == "arith.mulf":
                    # Q is scaled (q * qk_scale) before the local_alloc; follow
                    # whichever operand traces to a load.
                    for oid in op.operand_ids:
                        r = _load_addr_for_dot_operand(oid, _depth + 1)
                        if r is not None:
                            return r
                    return None
                break
            return None

        # --- MLA (nope/rope) 3-dot case: re-identify roles + the rope QK dot ----
        # Triton fuses `dot_nope + dot_rope` into a dot CHAIN (rope's accumulator
        # operand == nope's result), so a real MLA attention has 3 dots: two chained
        # QK dots + one PV dot. The PV dot's A operand (softmax probs P) has NO backing
        # load. Symmetric FA is 2 dots and skips this entirely (dot_qk/dot_pv keep the
        # values identified above). DeepSeek/Kimi MLA: qk = qk_nope(128) + qk_rope(64).
        is_mla = False
        dot_rope = None
        if len(dots) == 3:
            _qk3 = [d for d in dots if _load_addr_for_dot_operand(d.operand_ids[0]) is not None]
            _pv3 = [d for d in dots if _load_addr_for_dot_operand(d.operand_ids[0]) is None]
            if len(_qk3) != 2 or len(_pv3) != 1:
                _refuse("the MLA dot roles (need exactly 2 QK-with-loads + 1 PV)")
            dot_pv = _pv3[0]
            _a, _b = _qk3
            if len(_b.operand_ids) > 2 and _b.operand_ids[2] == _a.id:
                dot_qk, dot_rope = _a, _b  # nope=_a; rope=_b accumulates onto nope
            elif len(_a.operand_ids) > 2 and _a.operand_ids[2] == _b.id:
                dot_qk, dot_rope = _b, _a
            else:
                _refuse("the MLA QK dot chain (rope must accumulate onto nope's result)")
            is_mla = True

        # --- pointer roles + strides -----------------------------------------
        q_addr = _load_addr_for_dot_operand(dot_qk.operand_ids[0])
        k_addr = _load_addr_for_dot_operand(dot_qk.operand_ids[1])
        v_addr = _load_addr_for_dot_operand(dot_pv.operand_ids[1])
        q_res = _extract_ptr_strides(q_addr)
        k_res = _extract_ptr_strides(k_addr)
        v_res = _extract_ptr_strides(v_addr)
        if q_res is None:
            _refuse("the Q pointer/stride chain")
        if k_res is None:
            _refuse("the K pointer/stride chain")
        if v_res is None:
            _refuse("the V pointer/stride chain")

        # Store target = Out. There must be exactly one tt.store.
        stores = [s for s in all_ops if s.op == "tt.store"]
        if len(stores) != 1 or not stores[0].operand_ids:
            _refuse("the output store")
        o_res = _extract_ptr_strides(stores[0].operand_ids[0])
        if o_res is None:
            _refuse("the Out pointer/stride chain")

        q_idx, q_strides = q_res
        k_idx, k_strides = k_res
        v_idx, v_strides = v_res
        o_idx, o_strides = o_res

        # MLA: the rope QK dot adds two more pointer roles (q_rope, k_rope). The cat
        # dispatch requires all sources contiguous, so their strides are captured for
        # the eligibility check too.
        qr_idx = kr_idx = None
        qr_strides = kr_strides = None
        if is_mla:
            qr_res = _extract_ptr_strides(_load_addr_for_dot_operand(dot_rope.operand_ids[0]))
            kr_res = _extract_ptr_strides(_load_addr_for_dot_operand(dot_rope.operand_ids[1]))
            if qr_res is None or kr_res is None:
                _refuse("the MLA rope Q/K pointer/stride chains")
            qr_idx, qr_strides = qr_res
            kr_idx, kr_strides = kr_res

        # DISTINCT pointer roles (4 for symmetric FA, 6 for MLA nope/rope).
        if is_mla:
            if len({q_idx, qr_idx, k_idx, kr_idx, v_idx, o_idx}) != 6:
                _refuse("six distinct MLA pointer roles (q_nope,q_rope,k_nope,k_rope,v,out)")
        elif len({q_idx, k_idx, v_idx, o_idx}) != 4:
            _refuse("four distinct pointer roles")
        roles_resolved = True

        # GQA/MQA GUARD (correct-or-refuse): the FA templates index K/V with the SAME head
        # offset as Q (h = zh % H applied to all three). Grouped-/multi-query attention
        # (Llama/Mistral/Qwen) gives K/V FEWER heads, indexed by a DIFFERENT head (off_h //
        # group, or a distinct remsi), which the template would silently mis-address
        # (measured max err ~1.1 at head_dim 128). Require Q, K, V to share the SAME head-
        # offset SSA (peeled through layout ops). off_h lives in the 2-level scalar batch/
        # head addptr chain as muli(off_h, SH); the 1-level chain (H==1, single head) has no
        # head index and can't be GQA. return None -> the caller falls through to the
        # head_dim>64 refusal / generic path (both correct), NOT a routed wrong result.
        # BATCH/HEAD CONVENTION GUARD (correct-or-refuse): the templates compute
        # z = zh / H, h = zh % H (head-inner packing). A batch-inner kernel (h = zh // Z,
        # z = zh % Z) otherwise routes with z/h swapped -> silent-wrong (measured err ~0.97
        # at head_dim 128). Verify off_h = remsi(pid, H) and off_z = divsi(pid, H) (same H).
        if not is_mla:

            def _bh_offsets_dense(addr_id):
                """(off_z, off_h) from the 2-level scalar batch/head addptr chain, or
                (None, None) for the 1-level (H==1, single-head) chain / an unresolved walk."""
                if addr_id is None:
                    return None, None
                a = _skip_layout(addr_id)
                op = op_by_id.get(a)
                if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                    return None, None
                row_side = _skip_layout(op.operand_ids[0])
                rop = op_by_id.get(row_side)
                if rop is not None and rop.op == "tt.broadcast" and rop.operand_ids:
                    row_side = _skip_layout(rop.operand_ids[0])
                    rop = op_by_id.get(row_side)
                if rop is None or rop.op != "tt.addptr" or len(rop.operand_ids) < 2:
                    return None, None
                splat_base = op_by_id.get(rop.operand_ids[0])
                if splat_base is None or splat_base.op != "tt.splat" or not splat_base.operand_ids:
                    return None, None
                scal = op_by_id.get(splat_base.operand_ids[0])
                if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                    return None, None
                inner_arg = arg_by_id.get(scal.operand_ids[0])
                inner = op_by_id.get(scal.operand_ids[0])
                if inner_arg is not None and inner_arg.is_ptr:
                    return None, None  # 1-level (H==1): single head
                if inner is None or inner.op != "tt.addptr" or len(inner.operand_ids) < 2:
                    return None, None

                def _off_of_muli(muli_id):
                    mop = op_by_id.get(muli_id)
                    if mop is None or mop.op != "arith.muli":
                        return None
                    off = None
                    for oid in mop.operand_ids:
                        aarg = arg_by_id.get(oid)
                        if aarg is not None and not aarg.is_ptr:
                            continue  # the stride arg
                        off = oid
                    return off

                return _off_of_muli(inner.operand_ids[1]), _off_of_muli(scal.operand_ids[1])

            def _peel_core_dense(oid):
                seen = set()
                cur = op_by_id.get(oid)
                while (
                    cur is not None
                    and cur.id not in seen
                    and cur.operand_ids
                    and cur.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape")
                ):
                    seen.add(cur.id)
                    cur = op_by_id.get(cur.operand_ids[0])
                return cur.id if cur is not None else None

            _is_pid_d = lambda o: o.op in ("tt.get_program_id", "tt.program_id")

            def _cone_pid_d(oid, seen=None, dep=0):
                if seen is None:
                    seen = set()
                if oid in seen or dep > 48:
                    return False
                seen.add(oid)
                o = op_by_id.get(oid)
                if o is None:
                    return False
                if _is_pid_d(o):
                    return True
                return any(_cone_pid_d(x, seen, dep + 1) for x in (o.operand_ids or []))

            def _divmod_arg(oid, kinds):
                # peeled core must be a kinds-op dividing a program_id by a non-ptr arg;
                # returns that divisor arg index (the head count) or None.
                c = op_by_id.get(_peel_core_dense(oid)) if oid is not None else None
                if c is None or c.op not in kinds or len(c.operand_ids) != 2:
                    return None
                num, den = c.operand_ids
                da = arg_by_id.get(den)
                if da is not None and not da.is_ptr and _cone_pid_d(num):
                    return da.index
                return None

            _qz, _qh = _bh_offsets_dense(q_addr)
            _kz, _kh = _bh_offsets_dense(k_addr)
            _vz, _vh = _bh_offsets_dense(v_addr)
            # Re-review 2026-08-25: the OUTPUT store's batch/head offsets must ALSO match —
            # a kernel writing O at off_h+1 otherwise routed and the template wrote the
            # WRONG head (the guard only compared the input loads).
            _oz, _ohh = _bh_offsets_dense(stores[0].operand_ids[0])
            if not (_qh is None and _kh is None and _vh is None and _ohh is None):
                # multi-head: GQA (Q/K/V + store must share off_h AND off_z) + convention
                # (off_h=remsi(pid,H), off_z=divsi(pid,H), same H).
                if _qh is None or _kh is None or _vh is None or _ohh is None:
                    return None  # mixed multi/single head (e.g. MQA H_kv==1) -> refuse
                _qc, _kc, _vc, _oc = (
                    _peel_core_dense(_qh),
                    _peel_core_dense(_kh),
                    _peel_core_dense(_vh),
                    _peel_core_dense(_ohh),
                )
                if not (_qc is not None and _qc == _kc == _vc == _oc):
                    return None  # GQA/MQA or store-head mismatch -> refuse
                _zq, _zk, _zv, _zo = (
                    _peel_core_dense(_off) if _off is not None else None for _off in (_qz, _kz, _vz, _oz)
                )
                if _zq is None or not (_zq == _zk == _zv == _zo):
                    return None  # K/V/store batch offset differs from Q's -> refuse
                _hH = _divmod_arg(_qh, ("arith.remsi", "arith.remui"))
                _zH = _divmod_arg(_qz, ("arith.divsi", "arith.divui"))
                if _hH is None or _zH is None or _hH != _zH:
                    return None  # not head-inner z=zh//H, h=zh%H (e.g. batch-inner) -> refuse

        # --- tile dims from the dot shapes -----------------------------------
        # dot 0 (QK^T): A = [block_m, head_dim], result = [block_m, block_n].
        a_shape = _extract_shape(self._find_op_type_str(dot_qk.operand_ids[0]))
        qk_out_shape = _extract_shape(dot_qk.type_str or "")
        # dot 1 (P@V): result = [block_m, head_dim].
        pv_out_shape = _extract_shape(dot_pv.type_str or "")
        if (
            not a_shape
            or len(a_shape) != 2
            or not qk_out_shape
            or len(qk_out_shape) != 2
            or not pv_out_shape
            or len(pv_out_shape) != 2
        ):
            _refuse("the dot tile shapes")
        block_m, head_dim = a_shape  # head_dim here == qk_head_dim (QK contraction)
        nope_dim = head_dim  # symmetric: the whole QK dim
        rope_dim = 0
        if is_mla:
            # The QK contraction spans BOTH dots: qk_head_dim = nope_dim + rope_dim.
            rope_a = _extract_shape(self._find_op_type_str(dot_rope.operand_ids[0]))
            if not rope_a or len(rope_a) != 2 or rope_a[0] != block_m:
                _refuse("the MLA rope dot A shape")
            rope_dim = rope_a[1]
            head_dim = nope_dim + rope_dim  # e.g. 128 (nope) + 64 (rope) = 192
        if qk_out_shape[0] != block_m:
            _refuse("a consistent block_m across the two dots")
        block_n = qk_out_shape[1]
        # P@V output is [block_m, v_head_dim]. v_head_dim MAY differ from the QK
        # head_dim (asymmetric MLA / DeepSeek attention: qk=192, v=128). Only block_m
        # must match across the dots; v_head_dim is read from the P@V output width.
        if pv_out_shape[0] != block_m:
            _refuse("a consistent block_m in the P@V output")
        v_head_dim = pv_out_shape[1]

        # --- Z / H / N_CTX scalar arg indices --------------------------------
        # Z and H are ``equal_to_1``-specialized away when they equal 1 (the
        # batch=1 / heads=1 case), so they may be ABSENT from the arg list; a
        # folded scalar is the constant 1 (reported as ``"c1"``). N_CTX drives
        # the bounds masks and is never specialized to 1, so it must be present.
        scalar_by_name = {a.name: a for a in self.graph.args if not a.is_ptr}
        z_arg = scalar_by_name.get("Z")
        h_arg = scalar_by_name.get("H")
        n_ctx_arg = scalar_by_name.get("N_CTX")
        n_ctx_index = (
            sequence_index if sequence_index is not None else (n_ctx_arg.index if n_ctx_arg is not None else None)
        )
        if n_ctx_index is None:
            # STRUCTURAL fallback (independent of the arg NAME, the trifast-#1
            # name-heuristic class): the bounds masks compare a make_range-derived
            # tile index to the sequence length (offs < N_CTX) in BOTH causal and
            # non-causal kernels — unlike the KV-loop upper bound, which is
            # (start_m+1)*BLOCK_M under causal and would resolve the WRONG arg. So
            # a kernel that names its seq-len anything but "N_CTX" (seqlen, L, ...)
            # still routes. Collect the distinct non-ptr scalar args that appear as
            # the (splat) operand of a less-than cmpi whose OTHER operand is a tile
            # index; require EXACTLY one -> that is N_CTX (2+ distinct -> ambiguous
            # -> refuse, never guess).
            _passthru = (
                "tt.splat",
                "tt.broadcast",
                "arith.extsi",
                "arith.extui",
                "arith.index_cast",
                "arith.index_castui",
                "ttg.convert_layout",
            )

            def _is_lt(_s):
                _pn = _s.attrs.get("predicate_name")
                if _pn is not None:
                    return _pn in ("slt", "sle", "ult", "ule")
                return _s.attrs.get("predicate") in (2, 3, 6, 7)

            def _cone_has_range(_start):
                # True if the SSA cone reaches a tt.make_range, unwrapping the
                # shape ops (_trace_to_make_range stops at expand_dims/broadcast,
                # but the mask operand is offs[:, None] = expand_dims(range)). The
                # splat-of-scalar-arg side bottoms out at a block arg -> False.
                _seen, _stack = set(), [_start]
                while _stack:
                    _c = _stack.pop()
                    if _c in _seen:
                        continue
                    _seen.add(_c)
                    _o = op_by_id.get(_c)
                    if _o is None:
                        continue
                    if _o.op == "tt.make_range":
                        return True
                    if (
                        _o.op
                        in (
                            "tt.expand_dims",
                            "tt.broadcast",
                            "tt.reshape",
                            "tt.splat",
                            "ttg.convert_layout",
                            "arith.extsi",
                            "arith.extui",
                            "arith.trunci",
                            "arith.index_cast",
                            "arith.index_castui",
                            "arith.muli",
                            "arith.addi",
                        )
                        and _o.operand_ids
                    ):
                        _stack.extend(_o.operand_ids)
                return False

            _nctx_cands = set()
            for _s in all_ops:
                if _s.op != "arith.cmpi" or len(_s.operand_ids or []) != 2 or not _is_lt(_s):
                    continue
                _x, _y = _s.operand_ids
                _x_mr = _cone_has_range(_x)
                _y_mr = _cone_has_range(_y)
                if _x_mr == _y_mr:
                    continue  # need exactly one tile-index side (excludes causal range-vs-range)
                _arg_side = _y if _x_mr else _x
                _root = _arg_side
                while True:
                    _o = op_by_id.get(_root)
                    if _o is not None and _o.op in _passthru and _o.operand_ids:
                        _root = _o.operand_ids[0]
                        continue
                    break
                _a = arg_by_id.get(_root)
                if _a is not None and not _a.is_ptr:
                    _nctx_cands.add(_a.index)
            if len(_nctx_cands) == 1:
                _cand = next(iter(_nctx_cands))
                # CROSS-CHECK (positively verify the pick is a LENGTH, not merely a unique
                # range-compared scalar): the seq-len drives the KV-loop extent, so the pick
                # must be the scf.for upper bound — directly (non-causal) or an operand of the
                # causal min((start_m+1)*BM, N_CTX). Trace every scf.for bound through
                # casts + min/select to its arg(s); require the pick among them. This rejects
                # a kernel that assumes divisibility (no seq-len mask) yet uniquely compares a
                # DIFFERENT scalar (e.g. valid_len) to a tile index. If no scf.for bound
                # resolves to any arg (e.g. a constexpr trip count), trust the unique pick.
                _bound_args = set()
                for _s in all_ops:
                    if _s.op != "scf.for" or len(_s.operand_ids or []) < 2:
                        continue
                    _stack, _seen2 = [_s.operand_ids[1]], set()
                    while _stack:
                        _bid = _stack.pop()
                        if _bid in _seen2:
                            continue
                        _seen2.add(_bid)
                        _ba = arg_by_id.get(_bid)
                        if _ba is not None and not _ba.is_ptr:
                            _bound_args.add(_ba.index)
                            continue
                        _bo = op_by_id.get(_bid)
                        if (
                            _bo is not None
                            and _bo.op
                            in (
                                "arith.minsi",
                                "arith.minui",
                                "arith.maxsi",
                                "arith.maxui",
                                "arith.select",
                                "arith.index_cast",
                                "arith.index_castui",
                                "arith.extsi",
                                "arith.extui",
                                "arith.trunci",
                            )
                            and _bo.operand_ids
                        ):
                            _stack.extend(_bo.operand_ids)
                if (not _bound_args) or (_cand in _bound_args):
                    n_ctx_index = _cand
                # else: the unique range-compared scalar is not the loop bound -> refuse
            if n_ctx_index is None and not any(_s.op == "scf.for" for _s in all_ops):
                # Equal-to-one specialization removes N_CTX and the one-trip KV
                # loop.  Propose C1 only when a real tile coordinate is still
                # bounded by the literal one.  The load/store/value verifier later
                # requires this exact bound on every Q/K/V/output role; this is not
                # a blanket "loop-free means singleton" inference.
                def _is_const_one(_start):
                    _cur, _seen = _start, set()
                    while _cur not in _seen:
                        _seen.add(_cur)
                        _op = op_by_id.get(_cur)
                        if _op is None:
                            return False
                        if _op.op == "arith.constant":
                            try:
                                return int((_op.attrs or {}).get("value")) == 1
                            except (TypeError, ValueError):
                                return False
                        if _op.op not in _passthru or len(_op.operand_ids or []) != 1:
                            return False
                        _cur = _op.operand_ids[0]
                    return False

                _c1_bounds = 0
                for _s in all_ops:
                    if _s.op != "arith.cmpi" or len(_s.operand_ids or []) != 2 or not _is_lt(_s):
                        continue
                    _x, _y = _s.operand_ids
                    if _cone_has_range(_x) and _is_const_one(_y):
                        _c1_bounds += 1
                    elif _cone_has_range(_y) and _is_const_one(_x):
                        _c1_bounds += 1
                if _c1_bounds:
                    n_ctx_index = C1
        if n_ctx_index is None:
            _refuse("the N_CTX scalar arg")
        z_val = z_arg.index if z_arg is not None else C1
        # H (heads) is baked into the template as the head divisor (z = zh / H,
        # h = zh % H), so a WRONG H silently collapses every head to head 0. Resolve
        # it STRUCTURALLY from the batch/head decomposition (off_hz // H, a divui/remui
        # on a program_id) so it is independent of the arg's NAME — a kernel that names
        # its heads scalar anything but "H" (num_heads, nheads, ...) must not silently
        # mis-compute (the trifast-#1 name-heuristic class). The name lookup is only a
        # fallback; a genuine single-head kernel has H==1 specialized away (no divui) -> C1.
        h_val = None
        for _s in all_ops:
            if _s.op in ("arith.divui", "arith.remui", "arith.divsi", "arith.remsi") and _s.operand_ids:
                if any(
                    op_by_id.get(o) is not None and op_by_id.get(o).op == "tt.get_program_id" for o in _s.operand_ids
                ):
                    for o in _s.operand_ids:
                        _a = arg_by_id.get(o)
                        if _a is not None and not _a.is_ptr:
                            h_val = _a.index
                            break
            if h_val is not None:
                break
        if h_val is None:
            h_val = h_arg.index if h_arg is not None else C1

        # --- score masks: classify the exact semantics, never "any select = causal".
        # The old shortcut sent an ordinary ``where(k < N_CTX, score, -inf)`` through
        # the triangular template and returned a GPU-confirmed wrong result (err 3.7).
        # Dense templates replay two masks: the N_CTX tile boundary and the canonical
        # within-sequence lower triangle.  Admit only those proofs; any other select on
        # the score path refuses rather than being dropped or mislabeled.
        _qk_ids = {dot_qk.id} | ({dot_rope.id} if is_mla and dot_rope is not None else set())
        causal = False

        def _mask_cone_has(start, predicate, limit=128):
            seen2, stack2 = set(), [start]
            while stack2 and len(seen2) < limit:
                cur2 = stack2.pop()
                if cur2 in seen2:
                    continue
                seen2.add(cur2)
                op2 = op_by_id.get(cur2)
                if op2 is not None:
                    if predicate(op2):
                        return True
                    stack2.extend(op2.operand_ids or [])
            return False

        def _mask_depends_on_score(start):
            return _mask_cone_has(start, lambda op: op.id in _qk_ids)

        def _mask_is_exact_arg(start, arg_index):
            if arg_index == C1:
                cur2, seen2 = start, set()
                wrappers = {
                    "tt.splat",
                    "tt.broadcast",
                    "tt.expand_dims",
                    "tt.reshape",
                    "ttg.convert_layout",
                    "arith.extsi",
                    "arith.extui",
                    "arith.index_cast",
                    "arith.index_castui",
                }
                while cur2 not in seen2:
                    seen2.add(cur2)
                    op2 = op_by_id.get(cur2)
                    if op2 is None:
                        return False
                    if op2.op == "arith.constant":
                        try:
                            return int((op2.attrs or {}).get("value")) == 1
                        except (TypeError, ValueError):
                            return False
                    if op2.op not in wrappers or len(op2.operand_ids or []) != 1:
                        return False
                    cur2 = op2.operand_ids[0]
                return False
            target = next((a.id for a in self.graph.args if a.index == arg_index), None)
            if target is None:
                return False
            cur2, seen2 = start, set()
            wrappers = {
                "tt.splat",
                "tt.broadcast",
                "tt.expand_dims",
                "tt.reshape",
                "ttg.convert_layout",
                "arith.extsi",
                "arith.extui",
                "arith.index_cast",
                "arith.index_castui",
            }
            while cur2 not in seen2:
                if cur2 == target:
                    return True
                seen2.add(cur2)
                op2 = op_by_id.get(cur2)
                if op2 is None or op2.op not in wrappers or len(op2.operand_ids or []) != 1:
                    return False
                cur2 = op2.operand_ids[0]
            return False

        def _mask_core_id(start):
            cur2, seen2 = start, set()
            wrappers = {
                "tt.splat",
                "tt.broadcast",
                "tt.expand_dims",
                "tt.reshape",
                "ttg.convert_layout",
            }
            while cur2 not in seen2:
                seen2.add(cur2)
                op2 = op_by_id.get(cur2)
                if op2 is None or op2.op not in wrappers or len(op2.operand_ids or []) != 1:
                    return cur2
                cur2 = op2.operand_ids[0]
            return None

        def _mask_core(start):
            return op_by_id.get(_mask_core_id(start))

        def _mask_is_neg_inf(start):
            op2 = _mask_core(start)
            if op2 is None or op2.op != "arith.constant":
                return False
            value = (op2.attrs or {}).get("value")
            if isinstance(value, float):
                return value == float("-inf")
            # TTGIR walkers expose floating -inf either as a float or its raw bit word.
            return value in (0xFC00, 0xFF80, 0xFF800000, 0xFFF0000000000000)

        _loop_iv_ids = {
            (s.attrs or {}).get("block_arg_ids", [None])[0]
            for s in all_ops
            if s.op == "scf.for" and (s.attrs or {}).get("block_arg_ids")
        }

        def _mask_constant(start):
            op2 = _mask_core(start)
            if op2 is None or op2.op != "arith.constant":
                return None
            value = (op2.attrs or {}).get("value")
            return int(value) if isinstance(value, (int, float)) else None

        def _mask_index_kind(start):
            """Exact dense-template row/column index, or None.

            Query rows are ``pid0*BLOCK_M + arange(0, BLOCK_M)``. Key columns are
            ``loop_iv + arange(0, BLOCK_N)``.  Admitting a mere cone dependency would
            also accept offsets such as ``arange+1`` that the template does not replay.
            """
            root = _mask_core(start)
            if root is None or root.op != "arith.addi" or len(root.operand_ids or []) != 2:
                return None
            range_id = base_id = None
            for oid in root.operand_ids:
                atom = _mask_core(oid)
                if atom is not None and atom.op == "tt.make_range":
                    range_id = oid
                else:
                    base_id = oid
            if range_id is None or base_id is None:
                return None
            range_op = _mask_core(range_id)
            start_v = int((range_op.attrs or {}).get("start", -1))
            end_v = int((range_op.attrs or {}).get("end", -1))
            base = _mask_core(base_id)
            if _mask_core_id(base_id) in _loop_iv_ids:
                return "key" if start_v == 0 and end_v == block_n else None
            if base is None or base.op != "arith.muli" or len(base.operand_ids or []) != 2:
                return None
            pid0 = any(
                (atom := _mask_core(oid)) is not None
                and atom.op == "tt.get_program_id"
                and int((atom.attrs or {}).get("axis", 0)) == 0
                for oid in base.operand_ids
            )
            block_factor = any(_mask_constant(oid) == block_m for oid in base.operand_ids)
            if pid0 and block_factor and start_v == 0 and end_v == block_m:
                return "query"
            return None

        _axis_wrappers = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"}

        def _mask_expands(start, stop_ops):
            """``(op reached, tt.expand_dims passed)`` peeling broadcasts from ``start``."""
            cur2, seen2, expands = start, set(), []
            while cur2 not in seen2 and len(seen2) < 64:
                seen2.add(cur2)
                op2 = op_by_id.get(cur2)
                if op2 is None:
                    return None, expands
                if op2.op in stop_ops:
                    return op2, expands
                if op2.op not in _axis_wrappers or len(op2.operand_ids or []) != 1:
                    return None, expands
                if op2.op == "tt.expand_dims":
                    expands.append(op2)
                cur2 = op2.operand_ids[0]
            return None, expands

        def _mask_index_axis(cond_id, side):
            """The score axis an index of the mask comparison varies along, or None.

            ``_mask_index_kind`` proves WHICH index a side is; on a square tile a query
            index broadcast along the columns has the same shape and masks other cells.
            """
            cmp2, above = _mask_expands(cond_id, {"arith.cmpi"})
            if cmp2 is None:
                return None
            add2, mid = _mask_expands(cmp2.operand_ids[side], {"arith.addi"})
            if add2 is None:
                return None
            below = None
            for oid in add2.operand_ids or []:
                rng, got = _mask_expands(oid, {"tt.make_range"})
                if rng is not None:
                    if below is not None:
                        return None
                    below = got
            if below is None:
                return None
            expands = above + mid + below
            if len(expands) != 1:
                return None
            src = op_by_id.get(expands[0].operand_ids[0])
            axis = (expands[0].attrs or {}).get("axis")
            if src is None or len(_extract_shape(src.type_str or "")) != 1 or axis not in (0, 1):
                return None
            return 1 - int(axis)

        for s in all_ops:
            if s.op != "arith.select" or len(s.operand_ids or []) != 3:
                continue
            if tuple(_extract_shape(s.type_str or "")) != (block_m, block_n):
                continue
            cond_id, true_id, false_id = s.operand_ids
            true_score = _mask_depends_on_score(true_id)
            false_score = _mask_depends_on_score(false_id)
            if not (true_score or false_score):
                continue
            if not (true_score and not false_score and _mask_is_neg_inf(false_id)):
                _refuse("a score mask of where(proven_mask, score, -inf)")

            cmp_op = _mask_core(cond_id)
            if cmp_op is None or cmp_op.op != "arith.cmpi" or len(cmp_op.operand_ids or []) != 2:
                _refuse("a directly provable comparison for the score mask")
            lhs, rhs = cmp_op.operand_ids
            pred = (cmp_op.attrs or {}).get("predicate_name")
            lhs_kind = _mask_index_kind(lhs)
            rhs_kind = _mask_index_kind(rhs)

            is_bounds = pred in ("slt", "ult") and lhs_kind in ("query", "key") and _mask_is_exact_arg(rhs, n_ctx_index)
            is_causal = pred in ("sge", "uge") and lhs_kind == "query" and rhs_kind == "key"
            # Queries are the score's rows and keys its columns. The dense templates prove
            # this in the value-path verifier; the MLA route has no later axis proof.
            _want = {"query": 0, "key": 1}
            if is_mla and (
                (is_bounds and _mask_index_axis(cond_id, 0) != _want[lhs_kind])
                or (is_causal and (_mask_index_axis(cond_id, 0) != 0 or _mask_index_axis(cond_id, 1) != 1))
            ):
                _refuse("a score mask whose query index runs along the rows and key index along the columns")
            if is_bounds:
                continue
            if is_causal:
                causal = True
                continue
            _refuse(
                "a score mask equivalent to either the N_CTX boundary or the canonical within-sequence lower triangle"
            )

        # SPLIT-SCALE GUARD (re-review 2026-08-25): scan every dot-operand cone
        # (through layout/cast ops and chained mulfs, stopping at loads). The template
        # may bake one Q-side factor plus a separately proven post-dot chain, but a K-,
        # V-, P-side, or second Q factor would still be silently dropped.
        _SG_LAYOUT = (
            "ttg.local_load",
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "tt.trans",
            "tt.reshape",
            "ttg.convert_layout",
            "tt.broadcast",
            "tt.expand_dims",
            "arith.truncf",
            "arith.extf",
            "tt.fp_to_fp",
        )

        def _sg_const_factors(oid):
            out = []
            stack, seen4 = [oid], set()
            while stack:
                cur = stack.pop()
                if cur in seen4:
                    continue
                seen4.add(cur)
                op4 = op_by_id.get(cur)
                if op4 is None:
                    continue
                if op4.op == "arith.mulf":
                    for x in op4.operand_ids or []:
                        sub4 = op_by_id.get(x)
                        if sub4 is not None and sub4.op == "arith.constant":
                            v4 = sub4.attrs.get("value")
                            if isinstance(v4, (int, float)):
                                out.append(float(v4))
                                continue
                        stack.append(x)
                elif op4.op in _SG_LAYOUT and op4.operand_ids:
                    stack.append(op4.operand_ids[0])
            return out

        _sg_q = _sg_const_factors(dot_qk.operand_ids[0])
        _sg_k = _sg_const_factors(dot_qk.operand_ids[1])
        _sg_pv = _sg_const_factors(dot_pv.operand_ids[0]) + _sg_const_factors(dot_pv.operand_ids[1])
        # MLA has two chained QK dots and is outside #6a's two-dot score-chain
        # proof. Preserve its already-verified no-result-scale contract unchanged.
        _score_scale = (
            {"factor": 1.0, "op_ids": ()} if is_mla else self._fa_analyze_constant_score_scale(dot_qk, dot_pv)
        )
        if len(_sg_q) > 1 or _sg_k or _sg_pv or _score_scale is None:
            _refuse(
                "at most one Q-side scale plus an exact constant score-result scale "
                "(a runtime, K/V/P-side, bias, or second Q factor would be silently "
                "dropped by the template)"
            )
        if not _sg_q and not _score_scale["op_ids"]:
            _refuse("the softmax scale constant")
        scale = (_sg_q[0] if _sg_q else 1.0) * _score_scale["factor"]

        # --- out_dtype: the output pointer's element type --------------------
        out_arg = arg_by_id.get(self.graph.args[o_idx].id)
        out_dtype = out_arg.elem_type if out_arg is not None else None
        if not out_dtype:
            _refuse("the output dtype")

        return {
            "q": q_idx,
            "k": k_idx,
            "v": v_idx,
            "out": o_idx,
            "strides": {
                "q": q_strides,
                "k": k_strides,
                "v": v_strides,
                "o": o_strides,
            },
            "Z": z_val,
            "H": h_val,
            "N_CTX": n_ctx_index,
            "block_m": block_m,
            "block_n": block_n,
            "head_dim": head_dim,
            "v_head_dim": v_head_dim,
            "causal": causal,
            "scale": scale,
            "q_scale": bool(_sg_q),
            "score_scale": _score_scale,
            "verify_score_path": not is_mla,
            "out_dtype": out_dtype,
            # MLA (nope/rope): the QK score is TWO chained dots over separate tensors.
            # is_mla=True carries the rope Q/K pointer roles + their strides so the
            # dispatch can concat [q_nope|q_rope] / [k_nope|k_rope] into contiguous
            # [.,head_dim] and run the (validated) asymmetric qk=head_dim / v=v_head_dim
            # kernel. False/absent for symmetric FA (all fields below are ignored).
            "is_mla": is_mla,
            "q_rope": qr_idx,
            "k_rope": kr_idx,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
            "mla_strides": {"q_rope": qr_strides, "k_rope": kr_strides} if is_mla else None,
        }

    def _detect_biased_flash_attention(self):
        """Recognize trifast-style BIASED / triangle FlashAttention and resolve it.

        On top of standard FA (>=2 dots + exp + max) this pattern adds:
          (a) an additive BIAS tile as the QK dot's C/accumulator operand
              (``scores = tl.dot(q, kᵀ, bias)``) — the trigger; a standard FA's C
              is a zero/loop-carried acc (no backing load), so a non-biased kernel
              returns ``None`` here and flows to the normal path.
          (b) a const result-scale ``c`` with exp base ``b`` (``scores*=c; exp_b``)
              where ``c*ln(b) == 1`` — i.e. numerically identical to a natural-exp
              softmax over ``(sm_scale*QK + bias)`` (trifast: c=1/ln2, exp2). A
              mismatch (different temperature) is REFUSED.
          (c) a loaded MASK tile applied ``-> -inf``.
          (d) a per-query LSE store (a 1-D store distinct from the Out store).

        Returns a fully-resolved dict (mapped to the tiled template's ABI
        convention), ``None`` when the kernel is not this pattern, or raises
        ``MetalNonRecoverableError`` on ANY ambiguity once the pattern is
        confirmed (correct-or-refuse — a mis-read stride would silently mis-
        compute). Handles loop-carried pointers (K/V/bias loaded inside the
        K-loop address via scf.for block-args) and the direct-Kᵀ load
        orientation (row=head_dim, col=kv) that trifast uses.
        """
        from triton_msl.errors import MetalNonRecoverableError
        import math as _math

        def _walk(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _walk(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _walk(s.else_ops)

        allops = list(_walk(self.graph.ops))
        dots = [s for s in allops if s.op == "tt.dot"]
        has_exp = any(_op_is_exp(s.op) for s in allops)
        has_max = any("max" in (s.op or "") for s in allops)
        if not (len(dots) >= 2 and has_exp and has_max):
            return None

        op_by_id = {s.id: s for s in allops}
        arg_by_id = {a.id: a for a in self.graph.args}
        C1 = "c1"

        # Loop-carried pointer: a load inside scf.for addresses via a block-arg;
        # map each block-arg id -> its init value (the addptr chain before the loop).
        blockarg_init = {}
        for s in allops:
            if s.op == "scf.for":
                bids = (s.attrs or {}).get("block_arg_ids", [])
                inits = list(s.operand_ids or [])[3:]
                for i, init_id in enumerate(inits):
                    if i + 1 < len(bids):
                        blockarg_init[bids[i + 1]] = init_id

        def skip_layout(sid):
            seen = set()
            while sid not in seen:
                seen.add(sid)
                if sid in blockarg_init:
                    sid = blockarg_init[sid]
                    continue
                op = op_by_id.get(sid)
                if op is not None and op.op == "ttg.convert_layout" and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return sid

        def stride_from_term(tid):
            tid = skip_layout(tid)
            op = op_by_id.get(tid)
            if op is None:
                return None
            if op.op == "tt.broadcast" and op.operand_ids:
                return stride_from_term(op.operand_ids[0])
            if op.op != "arith.muli":
                return C1
            for oid in op.operand_ids:
                sub = op_by_id.get(oid)
                if sub is not None and sub.op == "tt.splat" and sub.operand_ids:
                    a = arg_by_id.get(sub.operand_ids[0])
                    if a is not None and not a.is_ptr:
                        return a.index
            return None

        def scalar_stride_from_muli(mid):
            op = op_by_id.get(mid)
            if op is None or op.op != "arith.muli":
                return None
            for oid in op.operand_ids:
                a = arg_by_id.get(oid)
                if a is not None and not a.is_ptr:
                    return a.index
            return None

        def _base_and_scalar(scal_id):
            """scalar batch chain: addptr(addptr(PTR, muli(off_z,SZ)), muli(off_h,SH))
            (2-level) OR addptr(PTR, muli(off_hz,SZ)) (1-level, h folded). Returns
            (base_arg, z_stride, h_stride) or None."""
            scal = op_by_id.get(scal_id)
            if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                return None
            outer = scalar_stride_from_muli(scal.operand_ids[1])
            inner_arg = arg_by_id.get(scal.operand_ids[0])
            inner = op_by_id.get(scal.operand_ids[0])
            if inner_arg is not None and inner_arg.is_ptr:
                return inner_arg, outer, C1
            if inner is not None and inner.op == "tt.addptr" and len(inner.operand_ids) >= 2:
                z = scalar_stride_from_muli(inner.operand_ids[1])
                base = arg_by_id.get(inner.operand_ids[0])
                if base is not None and base.is_ptr:
                    return base, z, outer
            return None

        def resolve_2d(addr_id):
            """addptr(broadcast(addptr(splat(base), row_term)), col_term) ->
            (base_idx, [z, h, row_stride, col_stride]). Orientation-agnostic
            (row_stride may be 1 for a direct-Kᵀ load)."""
            if addr_id is None:
                return None
            addr_id = skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            col = stride_from_term(op.operand_ids[1])
            row_side = skip_layout(op.operand_ids[0])
            rop = op_by_id.get(row_side)
            if rop is not None and rop.op == "tt.broadcast" and rop.operand_ids:
                row_side = skip_layout(rop.operand_ids[0])
                rop = op_by_id.get(row_side)
            if rop is None or rop.op != "tt.addptr" or len(rop.operand_ids) < 2:
                return None
            row = stride_from_term(rop.operand_ids[1])
            sb = op_by_id.get(rop.operand_ids[0])
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            bs = _base_and_scalar(sb.operand_ids[0])
            if bs is None:
                return None
            base, z, h = bs
            s = [z, h, row, col]
            return None if any(x is None for x in s) else (base.index, s)

        def resolve_1d(addr_id):
            """addptr(splat(base), idx_term) -> (base_idx, [z, h, col_stride])."""
            if addr_id is None:
                return None
            addr_id = skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            col = stride_from_term(op.operand_ids[1])
            sb = op_by_id.get(skip_layout(op.operand_ids[0]))
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            bs = _base_and_scalar(sb.operand_ids[0])
            if bs is None:
                return None
            base, z, h = bs
            s = [z, h, col]
            return None if any(x is None for x in s) else (base.index, s)

        def load_addr(oid, depth=0):
            if depth > 32:
                return None
            sid = oid
            seen = set()
            while sid not in seen:
                seen.add(sid)
                if sid in blockarg_init:
                    sid = blockarg_init[sid]
                    continue
                op = op_by_id.get(sid)
                if op is None:
                    break
                if (
                    op.op
                    in (
                        "ttg.local_load",
                        "ttg.local_alloc",
                        "ttg.memdesc_trans",
                        "tt.trans",
                        "tt.reshape",
                        "ttg.convert_layout",
                        "arith.extf",
                        "arith.truncf",
                    )
                    and op.operand_ids
                ):
                    sid = op.operand_ids[0]
                    continue
                if op.op == "tt.load" and op.operand_ids:
                    return op.operand_ids[0]
                if op.op == "arith.mulf":
                    for o in op.operand_ids:
                        r = load_addr(o, depth + 1)
                        if r is not None:
                            return r
                    return None
                break
            return None

        order = {s.id: i for i, s in enumerate(allops)}
        dot_qk, dot_pv = sorted(dots, key=lambda d: order[d.id])[:2]

        # --- BIAS trigger: the QK dot's C operand must trace to a load. --------
        if len(dot_qk.operand_ids) < 3:
            return None
        bias_addr = load_addr(dot_qk.operand_ids[2])
        if bias_addr is None:
            return None  # C is a zero/loop-acc (standard FA) -> not the biased pattern

        # Past here the kernel IS biased-FA: any unresolved field is a hard refuse.
        def _refuse(field):
            raise MetalNonRecoverableError(
                f"Biased/triangle FlashAttention recognized but {field} could not be "
                "resolved unambiguously; refusing rather than emit silently-wrong output."
            )

        # Grid dimensionality. 2 program_ids = the 2-D (q-block, z*h) grid. 3 =
        # trifast's triangle attention (pid_j q-block, pid_i i-axis, pid_h
        # batch*heads): the two batch axes are SEPARATE grid dims, bias is shared
        # across the i-axis (its i-stride is 0), and the mask is shared across heads
        # (indexed pid_h//H_heads). The template's grid_3d + mask_batch_div model
        # this. More than 3 program_ids is unhandled -> refuse.
        n_pid = len({s.id for s in allops if s.op == "tt.get_program_id"})
        if n_pid > 3 or n_pid < 2:
            _refuse(f"a 2-D or 3-D program grid (got {n_pid} program_ids)")
        grid_3d = n_pid == 3

        q_res = resolve_2d(load_addr(dot_qk.operand_ids[0]))
        k_res = resolve_2d(load_addr(dot_qk.operand_ids[1]))
        v_res = resolve_2d(load_addr(dot_pv.operand_ids[1]))
        b_res = resolve_2d(bias_addr)
        if q_res is None:
            _refuse("the Q pointer/strides")
        if k_res is None:
            _refuse("the K pointer/strides")
        if v_res is None:
            _refuse("the V pointer/strides")
        if b_res is None:
            _refuse("the bias pointer/strides")

        # Out + lse: exactly two tt.store — a 2-D Out store and a 1-D lse store.
        stores = [s for s in allops if s.op == "tt.store"]
        o_res = lse_res = None
        for st_ in stores:
            if not st_.operand_ids:
                continue
            r2 = resolve_2d(st_.operand_ids[0])
            r1 = resolve_1d(st_.operand_ids[0])
            if r2 is not None and o_res is None:
                o_res = r2
            elif r1 is not None and lse_res is None:
                lse_res = r1
        if o_res is None:
            _refuse("the Out store pointer/strides")
        if lse_res is None:
            _refuse("the lse store pointer/strides")

        # Mask: OPTIONAL — a boolean tt.load of an i1/i8 tensor. Zero mask loads =>
        # ALiBi/T5-style additive bias with NO masking (the template's mask is a no-op).
        # Exactly one => resolve it (trifast). More than one => REFUSE, never return None
        # (a bias-in-dot kernel routed to standard FA would silently DROP the bias).
        mask_loads = [
            s for s in allops if s.op == "tt.load" and ("i1" in (s.type_str or "") or "i8" in (s.type_str or ""))
        ]
        if len(mask_loads) > 1:
            _refuse("at most one boolean mask load (ambiguous biased attention)")
        has_mask = len(mask_loads) == 1
        m_res = None
        if has_mask:
            m_res = resolve_1d(mask_loads[0].operand_ids[0])
            if m_res is None:
                _refuse("the mask pointer/strides")

        def _mask_batch_divisor(mask_addr):
            """The mask's batch (z) offset. Returns the divisor arg index iff it is
            ``divui(program_id, arg)`` (trifast's cross-head mask_start_h = pid_h //
            H_heads) so the 3-D template can index the shared mask by (z / div);
            returns None when the offset is the batch id directly (a per-head mask, no
            divisor). This VERIFIES the divisor feeds the mask rather than reusing the
            first pid-division found — a 3-D kernel with an unrelated pid-division and
            a per-head mask must NOT get a spurious divisor."""
            a = skip_layout(mask_addr)
            op = op_by_id.get(a)
            if op is None or op.op != "tt.addptr":
                return None
            sb = op_by_id.get(skip_layout(op.operand_ids[0]))
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            scal = op_by_id.get(sb.operand_ids[0])
            if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                return None
            inner = op_by_id.get(scal.operand_ids[0])
            z_muli_id = (
                inner.operand_ids[1]
                if (inner is not None and inner.op == "tt.addptr" and len(inner.operand_ids) >= 2)
                else scal.operand_ids[1]
            )
            z_muli = op_by_id.get(z_muli_id)
            if z_muli is None or z_muli.op != "arith.muli":
                return None
            # SCALAR muli(offset, STRIDE_ARG): the stride is a direct kernel arg; the
            # offset is the other (computed) operand (e.g. mask_start_h = pid_h // H).
            off_id = None
            for oid in z_muli.operand_ids:
                if arg_by_id.get(oid) is not None:
                    continue  # the stride arg
                off_id = oid
            if off_id is None:
                return None
            off = op_by_id.get(skip_layout(off_id))
            if off is None or off.op not in ("arith.divui", "arith.divsi"):
                return None
            if not any(
                op_by_id.get(o) is not None and op_by_id.get(o).op == "tt.get_program_id" for o in off.operand_ids
            ):
                return None
            for o in off.operand_ids:
                ar = arg_by_id.get(o)
                if ar is not None and not ar.is_ptr:
                    return ar.index
            return None

        mask_div = _mask_batch_divisor(mask_loads[0].operand_ids[0]) if has_mask else None

        # Runtime Q-scale: mulf(load(Q), splat(scale_arg)). For fp16/bf16 inputs the
        # scale is cast to the input dtype first (tl.full([1], sm_scale, dtype=q.dtype)
        # -> truncf(splat(sm_scale))), so trace THROUGH dtype casts to the splat.
        def _splat_scale_arg(oid):
            # Walk through splat + dtype casts (in ANY order: splat(truncf(arg)) for
            # bf16/fp16, truncf(splat(arg)), or splat(arg) for fp32) to the scalar arg.
            sid = skip_layout(oid)
            seen = set()
            while sid not in seen:
                seen.add(sid)
                a = arg_by_id.get(sid)
                if a is not None:  # reached a kernel arg
                    return a.index if (not a.is_ptr and not str(a.elem_type).startswith("i")) else None
                op = op_by_id.get(sid)
                if op is None:
                    return None
                if (
                    op.op
                    in (
                        "tt.splat",
                        "arith.truncf",
                        "arith.extf",
                        "arith.fptrunc",
                        "arith.fpext",
                        "arith.sitofp",
                        "arith.uitofp",
                        "tt.bitcast",
                        "ttg.convert_layout",
                    )
                    and op.operand_ids
                ):
                    sid = skip_layout(op.operand_ids[0])
                    continue
                return None
            return None

        scale_arg = None
        for s in allops:
            if s.op == "arith.mulf" and s.operand_ids:
                if not any(load_addr(o) is not None for o in s.operand_ids):
                    continue
                for o in s.operand_ids:
                    idx = _splat_scale_arg(o)
                    if idx is not None:
                        scale_arg = idx
                        break
            if scale_arg is not None:
                break
        if scale_arg is None:
            _refuse("the runtime softmax scale (q * scale)")

        # Result-scale const + exp base: verify c*ln(base) == 1 (== natural softmax
        # over (sm_scale*QK + bias)). No result-scale + natural exp is also valid.
        result_const = None
        for s in allops:
            if s.op in ("arith.mulf",) and dot_qk.id in (s.operand_ids or []):
                for o in s.operand_ids:
                    if o == dot_qk.id:
                        continue
                    cop = op_by_id.get(skip_layout(o))
                    if cop is not None and cop.op == "tt.splat" and cop.operand_ids:
                        cc = op_by_id.get(cop.operand_ids[0])
                        if cc is not None and cc.op == "arith.constant":
                            result_const = cc.attrs.get("value")
                    elif cop is not None and cop.op == "arith.constant":
                        result_const = cop.attrs.get("value")
        if any(s.op == "math.exp2" for s in allops):
            exp_base = 2.0
        elif any(s.op in ("math.exp", "tt.exp") for s in allops):
            exp_base = _math.e
        else:
            _refuse("the softmax exp op")
        c_eff = float(result_const) if result_const is not None else 1.0
        if abs(c_eff * _math.log(exp_base) - 1.0) > 1e-4:
            _refuse(
                f"a softmax temperature of 1 (result-scale {c_eff} with exp base "
                f"{exp_base:.4f} gives c*ln(base)={c_eff * _math.log(exp_base):.5f}, "
                "not 1 — the biased template computes natural softmax over sm_scale*QK+bias)"
            )

        # --- tile dims from the dot shapes -----------------------------------
        a_shape = _extract_shape(self._find_op_type_str(dot_qk.operand_ids[0]))
        qk_out = _extract_shape(dot_qk.type_str or "")
        pv_out = _extract_shape(dot_pv.type_str or "")
        k_shape = _extract_shape(self._find_op_type_str(dot_qk.operand_ids[1]))
        if not (
            a_shape
            and len(a_shape) == 2
            and qk_out
            and len(qk_out) == 2
            and pv_out
            and len(pv_out) == 2
            and k_shape
            and len(k_shape) == 2
        ):
            _refuse("the dot tile shapes")
        block_m, head_dim = a_shape
        block_n = qk_out[1]
        v_head_dim = pv_out[1]
        if qk_out[0] != block_m or pv_out[0] != block_m:
            _refuse("a consistent block_m across the two dots")
        if v_head_dim != head_dim:
            _refuse("a symmetric head_dim (v_head_dim must equal qk head_dim)")
        # Direct-Kᵀ orientation check: the K operand is [head_dim, block_n].
        if k_shape[0] != head_dim or k_shape[1] != block_n:
            _refuse(
                f"the direct-Kᵀ orientation (K operand shape {k_shape} != "
                f"[head_dim={head_dim}, block_n={block_n}] — a transposed/canonical K "
                "load is not handled by the biased path)"
            )

        # --- N_CTX (the K-loop upper bound) + H (zh // H decomposition) -------
        def _arg_through_casts(sid):
            seen = set()
            while sid not in seen:
                seen.add(sid)
                a = arg_by_id.get(sid)
                if a is not None:
                    return a.index if not a.is_ptr else None
                op = op_by_id.get(sid)
                if op is not None and op.op in ("arith.extsi", "arith.trunci", "arith.index_cast") and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return None

        n_ctx_arg = None
        for s in allops:
            if s.op == "scf.for" and len(s.operand_ids or []) >= 2:
                n_ctx_arg = _arg_through_casts(s.operand_ids[1])
                if n_ctx_arg is not None:
                    break
        if n_ctx_arg is None:
            _refuse("the N_CTX loop bound")

        h_arg = None
        h_div_exists = False
        for s in allops:
            if s.op in ("arith.divui", "arith.remui", "arith.divsi", "arith.remsi") and s.operand_ids:
                pid_side = any(
                    (op_by_id.get(o) is not None and op_by_id.get(o).op == "tt.get_program_id") for o in s.operand_ids
                )
                if not pid_side:
                    continue
                h_div_exists = True
                for o in s.operand_ids:
                    a = arg_by_id.get(o)
                    if a is not None and not a.is_ptr:
                        h_arg = a.index
                        break
            if h_arg is not None:
                break
        if h_arg is None:
            if h_div_exists:
                # A pid decomposition exists but its divisor isn't a resolvable arg.
                _refuse("the H (heads) decomposition arg (zh // H)")
            # No pid division at all -> H was equal_to_1-specialized (single head):
            # z = pid_zh, h = 0. Bind H = 1u (the template computes h = zh % 1 = 0).
            h_arg = C1

        # DISTINCT pointer roles: Q, K, V, Out, bias, lse (+ mask when present).
        roles = [q_res[0], k_res[0], v_res[0], o_res[0], b_res[0], lse_res[0]]
        if has_mask:
            roles.append(m_res[0])
        if len(set(roles)) != len(roles):
            _refuse("distinct pointer roles (q,k,v,out,bias,lse[,mask])")

        _o_elem = str(self.graph.args[o_res[0]].elem_type)
        if _o_elem in ("fp16", "f16"):
            out_dtype = "f16"
        elif _o_elem in ("bf16", "bfloat16"):
            out_dtype = "bf16"
        else:
            out_dtype = "f32"

        # 3-D grid (trifast): a tensor whose batch chain is 1-level has NO
        # dependence on the second batch axis (the triangle-i axis, h = pid3.y).
        # Its resolver h-slot is C1 (a missing leg), but the CORRECT stride is 0
        # (SHARED across i) — e.g. trifast's bias [h_combined, n, n] is shared over
        # i. In 2-D grid this can't happen (a 1-level chain there means H=1 -> h=0,
        # so the stride is irrelevant). Convert C1 h-slots to 0 in 3-D mode.
        if grid_3d:
            for _res in (q_res, k_res, v_res, o_res, b_res):
                if _res[1][1] == C1:
                    _res[1][1] = "c0"  # shared across the i-axis -> stride 0 (not 1)

        return {
            "biased": True,
            "grid_3d": grid_3d,
            "q": q_res[0],
            "k": k_res[0],
            "v": v_res[0],
            "out": o_res[0],
            "bias": b_res[0],
            "mask": (m_res[0] if has_mask else None),
            "lse": lse_res[0],
            "has_mask": has_mask,
            # strides in resolver order; the template mapping handles the Kᵀ swap.
            "q_strides": q_res[1],
            "k_strides": k_res[1],
            "v_strides": v_res[1],
            "o_strides": o_res[1],
            "b_strides": b_res[1],
            "m_strides": (m_res[1] if has_mask else None),
            "lse_strides": lse_res[1],
            "scale_arg": scale_arg,
            # In 2-D, H drives z=zh/H. In 3-D, the same resolved divui(pid_h, H) IS
            # the mask cross-head divisor (mask_start_h = pid_h // H_heads).
            "N_CTX": n_ctx_arg,
            "H": h_arg,
            "mask_div": mask_div,
            "block_m": block_m,
            "block_n": block_n,
            "head_dim": head_dim,
            "out_dtype": out_dtype,
            "causal": False,
            "c_eff": c_eff,  # the constant score scale before the exp (packet 114: sentinel units)
        }

    def _detect_biased_fa_backward(self):
        """Recognize trifast's biased-FA BACKWARD kernels and resolve them
        exact-or-refuse. Distinct from the forward: NO ``max`` op (P is recomputed
        from the saved lse). Classifies by outputs: a delta-reduce + 2 stores =>
        _bwd_q (dQ+delta); 2 stores, no reduce => _bwd_kv (dK/dV); 1 store => _bwd_b
        (dbias). Returns a dict with ``bwd_kind`` + resolved pointers/strides, ``None``
        when not a backward pattern, or raises on ambiguity. Only ``kv`` is fully
        resolved so far; ``q``/``b`` return None (fall through -> fail-closed).
        """
        from triton_msl.errors import MetalNonRecoverableError

        def _walk(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _walk(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _walk(s.else_ops)

        allops = list(_walk(self.graph.ops))
        dots = [s for s in allops if s.op == "tt.dot"]
        has_exp2 = any(s.op == "math.exp2" or _op_is_exp(s.op) for s in allops)
        has_max = any("max" in (s.op or "") for s in allops)
        # Backward = FA-shaped (>=2 dots + exp) but NO stability-max (lse is loaded).
        if not (len(dots) >= 2 and has_exp2) or has_max:
            return None

        op_by_id = {s.id: s for s in allops}
        arg_by_id = {a.id: a for a in self.graph.args}
        C1 = "c1"
        blockarg_init = {}
        for s in allops:
            if s.op == "scf.for":
                bids = (s.attrs or {}).get("block_arg_ids", [])
                inits = list(s.operand_ids or [])[3:]
                for i, init_id in enumerate(inits):
                    if i + 1 < len(bids):
                        blockarg_init[bids[i + 1]] = init_id

        def skip_layout(sid):
            seen = set()
            while sid not in seen:
                seen.add(sid)
                if sid in blockarg_init:
                    sid = blockarg_init[sid]
                    continue
                op = op_by_id.get(sid)
                if op is not None and op.op == "ttg.convert_layout" and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return sid

        def uniform_stride(sid, seen=frozenset()):
            # Exact i32 products, retaining source wraparound. Never derive a
            # stride from a program id, range, load, induction variable or float.
            if sid in seen or len(seen) >= 32:
                return None
            seen = seen | {sid}
            arg = arg_by_id.get(sid)
            if arg is not None:
                return arg.index if not arg.is_ptr and str(arg.type_str) == "i32" else None
            op = op_by_id.get(sid)
            if op is None:
                return None
            if (
                op.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout")
                and len(op.operand_ids or []) == 1
            ):
                return uniform_stride(op.operand_ids[0], seen)
            if op.op == "arith.constant" and "i32" in (op.type_str or ""):
                value = (op.attrs or {}).get("value")
                if type(value) is int and 0 <= value <= 0xFFFFFFFF:
                    return C1 if value == 1 else ("literal", value)
            if op.op == "arith.muli" and len(op.operand_ids or []) == 2:
                left, right = [uniform_stride(x, seen) for x in op.operand_ids]
                if left is not None and right is not None:
                    return ("mul", left, right)
            return None

        def split_stride(op):
            if op is None or op.op != "arith.muli" or len(op.operand_ids or []) != 2:
                return None
            sides = [uniform_stride(x) for x in op.operand_ids]
            # Exactly one coordinate-dependent side and one uniform stride.
            if (sides[0] is None) == (sides[1] is None):
                return None
            i = 1 if sides[0] is None else 0
            return op.operand_ids[1 - i], sides[i]

        def stride_from_term(tid):
            tid = skip_layout(tid)
            op = op_by_id.get(tid)
            if op is None:
                return None
            if op.op == "tt.broadcast" and op.operand_ids:
                return stride_from_term(op.operand_ids[0])
            if op.op != "arith.muli":
                return C1
            pair = split_stride(op)
            return pair[1] if pair is not None else None

        def scalar_stride_from_muli(mid):
            op = op_by_id.get(mid)
            if op is not None and op.op in ("tt.get_program_id", "tt.program_id"):
                return C1
            pair = split_stride(op)
            return pair[1] if pair is not None else None

        def _base_and_scalar(scal_id):
            scal = op_by_id.get(scal_id)
            if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                return None
            outer = scalar_stride_from_muli(scal.operand_ids[1])
            inner_arg = arg_by_id.get(scal.operand_ids[0])
            inner = op_by_id.get(scal.operand_ids[0])
            if inner_arg is not None and inner_arg.is_ptr:
                return inner_arg, outer, C1
            if inner is not None and inner.op == "tt.addptr" and len(inner.operand_ids) >= 2:
                z = scalar_stride_from_muli(inner.operand_ids[1])
                base = arg_by_id.get(inner.operand_ids[0])
                if base is not None and base.is_ptr:
                    return base, z, outer
            return None

        def resolve_2d(addr_id):
            if addr_id is None:
                return None
            addr_id = skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            col = stride_from_term(op.operand_ids[1])
            row_side = skip_layout(op.operand_ids[0])
            rop = op_by_id.get(row_side)
            if rop is not None and rop.op == "tt.broadcast" and rop.operand_ids:
                row_side = skip_layout(rop.operand_ids[0])
                rop = op_by_id.get(row_side)
            if rop is None or rop.op != "tt.addptr" or len(rop.operand_ids) < 2:
                return None
            row = stride_from_term(rop.operand_ids[1])
            sb = op_by_id.get(rop.operand_ids[0])
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            bs = _base_and_scalar(sb.operand_ids[0])
            if bs is None:
                return None
            base, z, h = bs
            s = [z, h, row, col]
            return None if any(x is None for x in s) else (base.index, s)

        def resolve_1d(addr_id):
            if addr_id is None:
                return None
            addr_id = skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            # Triton loads bool storage through an i8 pointer. Both occupy one
            # byte in the Metal ABI; no data cast or other pointer cast is erased.
            if (
                op is not None
                and op.op == "tt.bitcast"
                and len(op.operand_ids or []) == 1
                and "!tt.ptr<i8>" in (op.type_str or "")
            ):
                src = op_by_id.get(op.operand_ids[0])
                if src is not None and "!tt.ptr<i1>" in (src.type_str or ""):
                    op = src
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            col = stride_from_term(op.operand_ids[1])
            sb = op_by_id.get(skip_layout(op.operand_ids[0]))
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            bs = _base_and_scalar(sb.operand_ids[0])
            if bs is None:
                return None
            base, z, h = bs
            s = [z, h, col]
            return None if any(x is None for x in s) else (base.index, s)

        def load_addr(oid, depth=0):
            if depth > 32:
                return None
            sid = oid
            seen = set()
            while sid not in seen:
                seen.add(sid)
                if sid in blockarg_init:
                    sid = blockarg_init[sid]
                    continue
                op = op_by_id.get(sid)
                if op is None:
                    break
                if (
                    op.op
                    in (
                        "ttg.local_load",
                        "ttg.local_alloc",
                        "ttg.memdesc_trans",
                        "tt.trans",
                        "tt.reshape",
                        "ttg.convert_layout",
                        "arith.extf",
                        "arith.truncf",
                        # backward also loads lse/delta as [j] then broadcasts to [j,k]:
                        "tt.broadcast",
                        "tt.expand_dims",
                    )
                    and op.operand_ids
                ):
                    sid = op.operand_ids[0]
                    continue
                if op.op == "tt.load" and op.operand_ids:
                    return op.operand_ids[0]
                if op.op == "arith.mulf":
                    for o in op.operand_ids:
                        r = load_addr(o, depth + 1)
                        if r is not None:
                            return r
                    return None
                break
            return None

        # --- the QK "scores" dot: C operand traces to a LOAD (the bias). Other dots'
        # C is a loop-acc/zeros. Absent -> not a biased backward.
        scores_dot = None
        for d in dots:
            if len(d.operand_ids) >= 3 and load_addr(d.operand_ids[2]) is not None:
                scores_dot = d
                break
        if scores_dot is None:
            return None

        stores = [s for s in allops if s.op == "tt.store"]
        has_reduce = any(s.op == "tt.reduce" for s in allops)
        if has_reduce:
            kind = "q"
        elif len(stores) == 2:
            kind = "kv"
        elif len(stores) == 1:
            kind = "b"
        else:
            return None

        def _refuse(field):
            raise MetalNonRecoverableError(
                f"Biased-FA backward ({kind}) recognized but {field} could not be resolved; "
                "refusing rather than emit silently-wrong gradients."
            )

        # A 2-D-grid backward (fewer than 3 program ids) makes the q/kv templates use their
        # z = zh // H; h = zh % H head/batch decode -- a head-inner convention no kernel
        # exercises (trifast uses a 3-D grid). A batch-inner 2-D kernel would silently swap
        # batch<->head there; rather than route through the unvalidated 2-D branch, refuse
        # it (correct-or-refuse -> generic/CPU). Only the 3-D grid path is validated.
        _BWD_2D_MSG = (
            "Biased-FA backward on a 2-D grid (fewer than 3 program ids) is not supported: "
            "the q/kv head/batch decomposition (z = zh // H; h = zh % H) is unvalidated and "
            "could silently swap batch<->head. Refusing rather than emit silently-wrong "
            "gradients. Use a 3-D grid (separate program ids for k-block, i, and head)."
        )

        # DEFINITIVE backward signature, checked BEFORE any _refuse: P is recomputed as
        # exp2((scores - lse) * inv_ln2), i.e. an exp2 of (or scaling) a SUBTRACTION.
        # Absent (e.g. a fused GEMM+bias+exp2, or a raw-exp2 attention) => NOT a softmax-
        # recompute backward => return None (fall through to the generic path). Only once
        # this shape is confirmed do the resolvers below refuse-on-failure.
        _has_sub_exp2 = False
        for _e in allops:
            if _e.op != "math.exp2" or not _e.operand_ids:
                continue
            _inp = op_by_id.get(skip_layout(_e.operand_ids[0]))
            _subs = (
                list(_inp.operand_ids or [])
                if (_inp is not None and _inp.op == "arith.mulf")
                else ([_inp.id] if (_inp is not None and _inp.op == "arith.subf") else [])
            )
            for _c in _subs:
                _cop = op_by_id.get(skip_layout(_c))
                if _cop is not None and _cop.op == "arith.subf":
                    _has_sub_exp2 = True
        if not _has_sub_exp2:
            return None

        # 3-D grid + mask divisor + block dims (shared with the forward semantics).
        # A biased-FA backward has a 2-D or 3-D grid; any other grid shape means this
        # is NOT our pattern -> fall through (return None), do NOT refuse.
        n_pid = len({s.id for s in allops if s.op == "tt.get_program_id"})
        if n_pid > 3 or n_pid < 2:
            return None
        grid_3d = n_pid == 3

        # Q, K, bias from the scores dot.
        _q_addr0 = load_addr(scores_dot.operand_ids[0])
        _k_addr0 = load_addr(scores_dot.operand_ids[1])
        q_res = resolve_2d(_q_addr0)
        k_res = resolve_2d(_k_addr0)
        b_res = resolve_2d(load_addr(scores_dot.operand_ids[2]))
        if not (q_res and k_res and b_res):
            _refuse("Q/K/bias")

        # GQA/MQA GUARD (correct-or-refuse; parity with the forward paths): the backward
        # templates apply the SAME batch/head offset (z,h) to Q, K, V, dK, dV. Grouped-/
        # multi-query attention gives K/V a DIFFERENT head (off_h // group), which the
        # template would silently mis-address (wrong dK/dV). Q and K must share their scalar
        # batch/head OFFSET SSAs; a K head that goes through an extra divsi/distinct remsi
        # differs -> return None (fall through to the generic path). Verified: a biased-GQA
        # _bwd_kv otherwise routes (kind=kv) and mis-computes dK/dV.
        def _scal_of(addr_id):
            if addr_id is None:
                return None
            addr_id = skip_layout(addr_id)
            op = op_by_id.get(addr_id)
            if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                return None
            row_side = skip_layout(op.operand_ids[0])
            rop = op_by_id.get(row_side)
            if rop is not None and rop.op == "tt.broadcast" and rop.operand_ids:
                row_side = skip_layout(rop.operand_ids[0])
                rop = op_by_id.get(row_side)
            # SINGLE-addptr fused-offset form (the db store): the splat of the scalar base
            # sits directly under the terminal addptr — return its scalar chain.
            if rop is not None and rop.op == "tt.splat" and rop.operand_ids:
                return rop.operand_ids[0]
            if rop is None or rop.op != "tt.addptr" or len(rop.operand_ids) < 2:
                return None
            sb = op_by_id.get(rop.operand_ids[0])
            if sb is None or sb.op != "tt.splat" or not sb.operand_ids:
                return None
            return sb.operand_ids[0]

        def _scalar_offsets(scal_id):
            scal = op_by_id.get(scal_id)
            if scal is None or scal.op != "tt.addptr" or len(scal.operand_ids) < 2:
                return None

            def _muli_off(muli_id):
                m = op_by_id.get(muli_id)
                if m is None or m.op != "arith.muli":
                    return None
                pair = split_stride(m)
                return pair[0] if pair is not None else None

            # Preserve the address-chain order: for the canonical
            # ptr + h*stride_h + z*stride_z form this is (z_off, h_off).
            offs = []
            o1 = _muli_off(scal.operand_ids[1])
            if o1 is not None:
                offs.append(o1)
            inner = op_by_id.get(scal.operand_ids[0])
            if inner is not None and inner.op == "tt.addptr" and len(inner.operand_ids) >= 2:
                o2 = _muli_off(inner.operand_ids[1])
                if o2 is not None:
                    offs.append(o2)
            return tuple(offs)

        _q_offs = _scalar_offsets(_scal_of(_q_addr0))
        _k_offs = _scalar_offsets(_scal_of(_k_addr0))
        if _q_offs is None or _k_offs is None or _q_offs != _k_offs:
            return None  # GQA/MQA or Q/K batch-head offset mismatch -> generic path

        def _offs_match_q(addr_id, subset=False):
            """Re-review 2026-08-25: the backward templates apply Q's scalar
            (start_h, start_i) offsets to V, O/dO and EVERY gradient store — a V read or a
            dK/dV/dQ store at a different head (e.g. pid_h+1) routed and was silently
            mis-addressed. True iff this address's scalar offsets equal Q's (subset=True
            for the head-only bias-gradient chain: its offsets must be a subset of Q's,
            i.e. share the same head SSA)."""
            if addr_id is None:
                return False
            _o = _scalar_offsets(_scal_of(addr_id))
            if _o is None:
                return False
            if subset:
                # Bias-gradient addresses carry only the head leg. In a full
                # canonical (z, h) pair it is the inner/second entry; _bwd_b's
                # Q itself is head-only, so its singleton pair must match exactly.
                if len(_q_offs) == 1:
                    return _o == _q_offs
                return len(_o) == 1 and len(_q_offs) == 2 and _o[0] == _q_offs[1]
            return _o == _q_offs

        # lse: the exp2 input is mulf(subf(scores, splat(lse)), inv_ln2). Find the
        # subf feeding an exp2 whose non-dot operand traces to a load.
        def _find_sub_operands(target_op):
            for s in allops:
                if s.op in ("arith.subf",):
                    yield s

        # The DEFINITIVE backward signature is P = exp2((scores - lse) * inv_ln2): an
        # exp2 whose input is (or scales) a subtraction. If NO exp2 is of a subtraction
        # (e.g. a fused GEMM+bias+exp2, or a raw-exp2 attention), this is NOT a softmax-
        # recompute backward -> return None (fall through to the generic path), do NOT
        # refuse. Only once the (scores - lse) shape is confirmed do we resolve exact-or-
        # refuse: a pattern present but a pointer unresolvable IS a broken backward.
        lse_res = lse_addr = None
        lse_sub_op = None  # the (scores - lse) subf (packet 134 sentinel proof)
        lse_exp_op = lse_exp_inp = None  # the exp2 and its input (packet 136 constant proof)
        lse_pattern_found = False
        exp2_ops = [s for s in allops if s.op == "math.exp2"]
        for e in exp2_ops:
            inp = op_by_id.get(skip_layout(e.operand_ids[0])) if e.operand_ids else None
            # inp = mulf(subf(...), inv_ln2)  OR  subf(...) directly
            cands = []
            if inp is not None and inp.op == "arith.mulf":
                cands = list(inp.operand_ids or [])
            elif inp is not None and inp.op == "arith.subf":
                cands = [inp.id]
            for c in cands:
                cop = op_by_id.get(skip_layout(c))
                if cop is not None and cop.op == "arith.subf":
                    lse_pattern_found = True
                    for o in cop.operand_ids:
                        la = load_addr(o)
                        r = resolve_1d(la) if la is not None else None
                        if r is not None:
                            lse_res = r
                            lse_addr = la
                            lse_sub_op = cop
                            lse_exp_op, lse_exp_inp = e, inp
        if not lse_pattern_found:
            return None  # not a (scores - lse) softmax recompute -> not a biased-FA backward
        if lse_res is None:
            _refuse("the lse (row_max) pointer")

        # mask: exactly one i1/i8 load. A biased-FA backward loads exactly one boolean
        # mask; 0 or 2+ means this is NOT our pattern -> fall through (return None).
        # Present-but-unresolvable IS a broken backward -> refuse.
        mask_loads = [
            s for s in allops if s.op == "tt.load" and ("i1" in (s.type_str or "") or "i8" in (s.type_str or ""))
        ]
        if len(mask_loads) != 1:
            return None
        mask_addr = mask_loads[0].operand_ids[0]
        m_res = resolve_1d(mask_addr)
        if m_res is None:
            _refuse("the mask pointer")

        # ---- packet 134: the score sentinel the P recompute applies (both selects, one value) ----
        # Source: scores -> where(bounds, scores, SNT) / where(Mask != 0, SNT, scores) in either
        # order -> subf(., lse) -> mulf(., inv_ln2) -> exp2. The template replays SNT on
        # in-bounds masked cells; it must be ONE SSA scalar (a float arg or a literal).
        _bcast = ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout")

        def _peel_ids(vid, allowed):
            seen = set()
            while vid not in seen:
                seen.add(vid)
                o = op_by_id.get(vid)
                if o is None or o.op not in allowed or len(o.operand_ids or []) != 1:
                    return vid
                vid = o.operand_ids[0]
            return vid

        def _const_val(o):
            v = (o.attrs or {}).get("value")
            if isinstance(v, bool) or v is None:
                return None
            if isinstance(v, int):
                if v in (0xFC00, 0xFF80, 0xFF800000, 0xFFF0000000000000):
                    return float("-inf")
                if v in (0x7C00, 0x7F80, 0x7F800000, 0x7FF0000000000000):
                    return float("inf")
                return float(v) if v == 0 else None
            return float(v)

        sentinel_ref = None
        # packet 139/140: the N_CTX operand of a boundary compare may peel ONLY value-preserving
        # wrappers — broadcast-family ops and sign-extension (both compare operands are then
        # sign-extended, which preserves slt AND ult). A narrowing (`trunci`), zero-extension
        # or index cast changes the compared value (N=160 -> N.to(int8).to(int32) == -96).
        _IDXW = _bcast + ("arith.extsi",)
        _VALW = (
            "arith.extf",
            "arith.truncf",
            "ttg.convert_layout",
            "tt.reshape",
            "ttg.local_load",
            "ttg.local_alloc",
            "tt.fp_to_fp",
        )
        # packet 141/144: VALUE identity (not provenance) admits only wrappers that cannot change
        # a single element: layout changes and the shared-memory round trip. A float conversion
        # (`truncf`/`extf`/`fp_to_fp`) is a different value — `scores.to(f16).to(f32)` rounds, and
        # the template never replays that rounding — so it must refuse before any maker.
        _VALX = ("ttg.convert_layout", "ttg.local_load", "ttg.local_alloc")

        def _elem_is_f32(o):
            t = getattr(o, "type_str", None) or ""
            # "tensor<32x32xf32, #mma>" / "f32"; NOT "bf16" (and no wider/narrower float)
            return re.search(r"(?<!b)f32(?![0-9])", t) is not None

        _ivs = [((s_.attrs or {}).get("block_arg_ids") or [None])[0] for s_ in allops if s_.op == "scf.for"]
        _LOG2E = 1.4426950408889634

        def _arg_index(vid):
            core = _peel_ids(vid, _IDXW)
            a_ = arg_by_id.get(core)
            if a_ is not None:
                return None if a_.is_ptr else a_.index
            # Triton removes constexpr-equivalent runtime arguments. Preserve
            # the same closed proof for the only singleton extent this route
            # supports instead of treating every literal as an N_CTX bound.
            c_ = op_by_id.get(core)
            if c_ is not None and c_.op == "arith.constant":
                try:
                    return C1 if int(c_.attrs.get("value")) == 1 else None
                except (TypeError, ValueError):
                    return None
            return None

        def _leaf_extent(cid, n_ctx_arg, extents):
            """One boundary leaf: cmpi slt|ult(<tile index>, N_CTX). Returns the tile extent."""
            c_ = op_by_id.get(_peel_ids(cid, _bcast))
            if (
                c_ is None
                or c_.op != "arith.cmpi"
                or (c_.attrs or {}).get("predicate_name") not in ("slt", "ult")
                or len(c_.operand_ids or []) != 2
            ):
                _refuse("a strict less-than N_CTX boundary leaf in the bounds select")
            lhs, rhs = c_.operand_ids
            if n_ctx_arg is None or _arg_index(rhs) != n_ctx_arg:
                _refuse("the bounds select comparing the tile index with the kernel's N_CTX argument")
            for iv_ in [None] + [v for v in _ivs if v is not None]:
                sh = self._index_shape(lhs, op_by_id, arg_by_id, iv_, 1)
                if (
                    sh["bad"] is not None
                    or sh["mod"] is not None
                    or sh["pid_div"] is not None
                    or sh["range"] != 1
                    or not sh["bounds"]
                ):
                    continue
                ext = sh["bounds"][1]
                if sh["bounds"][0] != 0 or ext not in extents:
                    continue
                singleton_col = n_ctx_arg == C1 and not _ivs and sh["pid"] is None and sh["iv"] == 0
                if (
                    (sh["pid"] is not None and sh["iv"] == 0 and sh["coef"] == ext)
                    or (sh["pid"] is None and sh["iv"] == 1)
                    or singleton_col
                ):
                    return ext
            _refuse("a bounds-select index that is exactly one tile of the source's row/col coordinate")

        def _leaf_sig(cid, n_ctx_arg, extents, mirrored_ok=False):
            """The full coordinate signature of one boundary leaf (structural, remat-proof).

            ``mirrored_ok`` also reads ``N_CTX > index``, the same boundary written the
            other way round."""
            c_ = op_by_id.get(_peel_ids(cid, _bcast))
            pred_ = (c_.attrs or {}).get("predicate_name") if c_ is not None else None
            if (
                c_ is None
                or c_.op != "arith.cmpi"
                or pred_ not in (("slt", "ult", "sgt", "ugt") if mirrored_ok else ("slt", "ult"))
                or len(c_.operand_ids or []) != 2
            ):
                _refuse("a strict less-than N_CTX boundary leaf")
            lhs, rhs = c_.operand_ids
            if pred_ in ("sgt", "ugt"):
                lhs, rhs = rhs, lhs
            if n_ctx_arg is None or _arg_index(rhs) != n_ctx_arg:
                _refuse("a boundary leaf comparing the tile index with the kernel's N_CTX argument")
            for iv_ in [None] + [v for v in _ivs if v is not None]:
                sh = self._index_shape(lhs, op_by_id, arg_by_id, iv_, 1)
                if (
                    sh["bad"] is not None
                    or sh["mod"] is not None
                    or sh["range"] != 1
                    or not sh["bounds"]
                    or sh["bounds"][0] != 0
                    or sh["bounds"][1] not in extents
                ):
                    continue
                singleton_col = n_ctx_arg == C1 and not _ivs and sh["pid"] is None and sh["iv"] == 0
                if (
                    (sh["pid"] is not None and sh["iv"] == 0 and sh["coef"] == sh["bounds"][1])
                    or (sh["pid"] is None and sh["iv"] == 1)
                    or singleton_col
                ):
                    return (
                        n_ctx_arg,
                        sh["pid"],
                        sh["pid_div"],
                        sh["coef"],
                        sh["bounds"],
                        sh["iv"],
                        iv_ if sh["iv"] else None,
                    )
            _refuse("a boundary leaf index that is exactly one tile of a row/col coordinate")

        _OPW = _VALW + ("tt.trans", "ttg.memdesc_trans")

        def _load_op_of(vid):
            """The tt.load a dot operand comes from, through casts / layout / transpose and
            the scale multiply (the constant side is skipped); None if anything else appears."""
            cur, seen = vid, set()
            while cur not in seen and len(seen) < 64:
                seen.add(cur)
                o_ = op_by_id.get(cur)
                if o_ is None:
                    return None
                if o_.op == "tt.load":
                    return o_
                if o_.op in _OPW and len(o_.operand_ids or []) == 1:
                    cur = o_.operand_ids[0]
                    continue
                if o_.op == "arith.mulf" and len(o_.operand_ids or []) == 2:

                    def _is_scalar(x):
                        pid_ = _peel_ids(x, _bcast)
                        a_ = arg_by_id.get(pid_)
                        c_ = op_by_id.get(pid_)
                        if c_ is not None and c_.op == "arith.truncf" and len(c_.operand_ids or []) == 1:
                            # packet 174: `tl.full([1], sm_scale, dtype)` = splat(truncf(scalar arg))
                            a2_ = arg_by_id.get(_peel_ids(c_.operand_ids[0], _bcast))
                            return a2_ is not None and not a2_.is_ptr
                        return (a_ is not None and not a_.is_ptr) or (c_ is not None and c_.op == "arith.constant")

                    data = [x for x in o_.operand_ids if not _is_scalar(x)]
                    if len(data) != 1:
                        return None
                    cur = data[0]
                    continue
                return None
            return None

        def _single_mask_leaf(load, role):
            if load is None or len(load.operand_ids or []) < 2:
                _refuse(f"a masked {role} load (the bounds select must reuse its mask)")
            leaf = _peel_ids(load.operand_ids[1], _bcast)
            lo = op_by_id.get(leaf)
            if lo is None or lo.op != "arith.cmpi":
                _refuse(f"the {role} load mask being one boundary comparison")
            return leaf

        def _q_mask_leaf():
            return _single_mask_leaf(_load_op_of(scores_dot.operand_ids[0]), "Q")

        def _k_mask_leaf():
            return _single_mask_leaf(_load_op_of(scores_dot.operand_ids[1]), "K")

        def _axis_of(vid, limit=24):
            """Packet 145/146: the score axis a broadcast operand VARIES along when consumed at
            [j, k] — 'row' (expand_dims axis=1: one value per query row j), 'col' (axis=0: one
            value per key k), 'uniform' (a splat), or None (no / unknown projection: a transpose,
            a reshape, a double expansion). Walks broadcast / convert_layout / expand_dims and,
            through a compare against a uniform operand, its varying operand. The element proofs
            (which load, which index) say WHAT vector arrives; this says along WHICH axis."""
            cur, axis, seen = vid, None, set()
            while cur not in seen and len(seen) < limit:
                seen.add(cur)
                o = op_by_id.get(cur)
                if o is None:
                    # a kernel argument (scalar) is uniform; anything else unresolved is unknown
                    return (axis if axis is not None else "uniform") if cur in arg_by_id else axis
                if o.op == "arith.constant":
                    return axis if axis is not None else "uniform"  # splat or dense constant
                if not (o.operand_ids or []):
                    break
                if o.op in ("tt.broadcast", "ttg.convert_layout", "arith.extui", "arith.extsi", "arith.extf"):
                    cur = o.operand_ids[0]  # layout / widening: the projection is unchanged
                    continue
                if o.op == "tt.splat":
                    return axis if axis is not None else "uniform"
                if o.op == "tt.expand_dims":
                    try:
                        a = int((o.attrs or {}).get("axis"))
                    except (TypeError, ValueError):
                        return None
                    if axis is not None or a not in (0, 1):
                        return None
                    axis = "row" if a == 1 else "col"
                    cur = o.operand_ids[0]
                    continue
                if o.op == "arith.cmpi" and len(o.operand_ids) == 2:
                    if axis is not None:
                        return axis
                    _var = [x for x in o.operand_ids if _axis_of(x, limit) != "uniform"]
                    if len(_var) != 1:
                        return None
                    cur = _var[0]
                    continue
                break
            return axis

        def _prove(n_ctx_arg, block_j, block_k):
            """Packet 136: the WHOLE score path the backward templates replay —
            exp2((S2 - lse) * log2(e)), S2/S1 = the two selects (either order) on ONE
            sentinel with role-exact predicates, and the value under them = the recognised
            scores dot (Q@Kᵀ with the Bias load as accumulator). Anything else refuses."""
            if lse_sub_op is None or len(lse_sub_op.operand_ids or []) != 2:
                _refuse("the (scores - lse) subtraction")
            # packet 141/144: the subtrahend must BE the lse load's value through layout-only
            # wrappers — `load_addr` is provenance (it peels float casts and walks through
            # multiplies) and admitted `tl.load(l_ptrs).to(f16).to(f32)`; the template subtracts
            # the raw fp32 load, so anything between the load and the subtraction refuses.
            _lo = op_by_id.get(_peel_ids(lse_sub_op.operand_ids[1], _bcast))
            if _lo is None or _lo.op != "tt.load" or not _lo.operand_ids or _lo.operand_ids[0] != lse_addr:
                _refuse(
                    "the lse on the right-hand side of the (scores - lse) subtraction being the loaded lse VALUE itself (no conversion or arithmetic between the load and the subtraction)"
                )
            if not _elem_is_f32(_lo):
                _refuse("an fp32 lse load (the template subtracts the raw fp32 value)")
            if _axis_of(lse_sub_op.operand_ids[1]) != "row":
                _refuse("the lse broadcast along the score ROWS (lse[:, None]: one value per query row j)")
            if lse_exp_op is None or lse_exp_op.op != "math.exp2":
                _refuse("an exp2 softmax recompute")
            if lse_exp_inp is None or lse_exp_inp.op != "arith.mulf" or len(lse_exp_inp.operand_ids or []) != 2:
                _refuse("the log2(e) multiply between (scores - lse) and exp2")
            _sides = [_peel_ids(x, _bcast) for x in lse_exp_inp.operand_ids]
            if lse_sub_op.id not in _sides:
                _refuse("the exp2 input being (scores - lse) times a constant")
            _c = op_by_id.get(next(x for x in _sides if x != lse_sub_op.id))
            _cv = _const_val(_c) if (_c is not None and _c.op == "arith.constant") else None
            if _cv is None or abs(_cv - _LOG2E) > 1e-6 * _LOG2E:
                _refuse("the exp2 coefficient being exactly log2(e) (the template's inv_ln2)")
            _cur = _peel_ids(lse_sub_op.operand_ids[0], _bcast)
            _kinds, _snts = [], []
            for _ in range(2):
                _sel = op_by_id.get(_cur)
                if _sel is None or _sel.op != "arith.select" or len(_sel.operand_ids or []) != 3:
                    _refuse("the two score sentinel selects (where(bounds, s, SNT) and where(Mask != 0, SNT, s))")
                _pred = op_by_id.get(_peel_ids(_sel.operand_ids[0], _bcast))
                if _pred is not None and _pred.op == "arith.cmpi" and (_pred.attrs or {}).get("predicate_name") == "ne":
                    # packet 148 (147 §5): the explicit `m != 0` spelling widens the uint8 mask to
                    # i32 (`extui`) before the compare; zero- or sign-extension preserves the
                    # "is nonzero" test exactly, so it is peeled here. A truncation is NOT (it
                    # keeps the low bit: 2 -> 0) and stays refused.
                    _src = [_peel_ids(x, _bcast + ("arith.extui", "arith.extsi")) for x in _pred.operand_ids]
                    _zero = [op_by_id.get(x) for x in _src if x != mask_loads[0].id]
                    if (
                        mask_loads[0].id not in _src
                        or len(_zero) != 1
                        or _zero[0] is None
                        or _zero[0].op != "arith.constant"
                        or _const_val(_zero[0]) != 0.0
                    ):
                        _refuse("the loaded-mask select predicate (must be Mask != literal 0)")
                    if _axis_of(_sel.operand_ids[0]) != "col":
                        _refuse(
                            "the loaded Mask broadcast along the score COLUMNS (Mask[None, :]: one value per key k)"
                        )
                    _kinds.append("mask")
                    _snts.append(_peel_ids(_sel.operand_ids[1], _bcast))
                    _cur = _peel_ids(_sel.operand_ids[2], _bcast)
                elif _pred is not None and _pred.op == "arith.andi" and len(_pred.operand_ids or []) == 2:
                    _exts = sorted(_leaf_extent(x, n_ctx_arg, {block_j, block_k}) for x in _pred.operand_ids)
                    if _exts != sorted([block_j, block_k]):
                        _refuse("the bounds select covering BOTH the row and the column N_CTX boundary")
                    # packet 138 (137 P0): a leaf that merely LOOKS like a tile index (any program
                    # id times the tile) is not the coordinate; the leaves must BE the mask values
                    # the recognised Q load (row) and K load (column) are guarded with.
                    # The compiler rematerialises comparisons per layout, so SSA identity is not
                    # the test; the leaf must have the SAME coordinate signature (N_CTX arg, pid
                    # axis + divisor + coefficient, range bounds, loop-IV identity) as the mask on
                    # the recognised Q load (row) / K load (column).
                    # packet 146 (145 HOLD): the SAME leaf laid along the wrong axis is a different
                    # bound, so each leaf is a (signature, axis) pair: the Q-load mask must vary
                    # along the rows and the K-load mask along the columns. Commuting the AND
                    # is harmless and stays admitted.
                    _qs, _ks = (
                        _leaf_sig(_q_mask_leaf(), n_ctx_arg, {block_j, block_k}),
                        _leaf_sig(_k_mask_leaf(), n_ctx_arg, {block_j, block_k}),
                    )
                    _pairs = {(_leaf_sig(x, n_ctx_arg, {block_j, block_k}), _axis_of(x)) for x in _pred.operand_ids}
                    if _qs == _ks or len(_pred.operand_ids) != 2 or _pairs != {(_qs, "row"), (_ks, "col")}:
                        _refuse(
                            "the bounds select's leaves being exactly the Q-load mask along the score ROWS and the K-load mask along the COLUMNS — a lookalike coordinate or a swapped broadcast axis is not the source's bound"
                        )
                    _kinds.append("bounds")
                    _snts.append(_peel_ids(_sel.operand_ids[2], _bcast))
                    _cur = _peel_ids(_sel.operand_ids[1], _bcast)
                else:
                    _refuse("a recognizable predicate on a score sentinel select")
            if sorted(_kinds) != ["bounds", "mask"] or len(set(_snts)) != 1:
                _refuse("exactly one loaded-mask select and one bounds select sharing ONE sentinel value")
            _under = _peel_ids(_cur, _VALX)
            if _under != scores_dot.id and _under not in (getattr(scores_dot, "result_ids", None) or []):
                _refuse(
                    "the value under the two selects being exactly the recognised Q@Kᵀ + Bias dot value (a conversion, third select or epilogue is not replayed)"
                )
            if not _elem_is_f32(scores_dot):
                _refuse("an fp32 score dot (the template accumulates Q@Kᵀ + Bias in fp32)")
            _snt = _snts[0]
            _snt_op = op_by_id.get(_snt)
            if _snt_op is None and _snt in arg_by_id:
                _a = arg_by_id[_snt]
                if _a.is_ptr or str(_a.elem_type) not in ("f32", "fp32", "float"):
                    _refuse("an fp32 runtime sentinel argument (the packed scalar ABI reinterprets 32-bit words)")
                return ("arg", _a.index)
            if _snt_op is not None and _snt_op.op == "arith.constant":
                _v = _const_val(_snt_op)
                if _v is None or _v != _v or _v == float("inf"):
                    _refuse("a decodable finite-or-minus-infinity literal sentinel")
                return ("const", _v)
            _refuse("a sentinel that is a kernel argument or a literal")

        def _prove_downstream(
            kind,
            n_ctx_arg,
            block_j,
            block_k,
            head_dim,
            do_addr,
            v_addr,
            dlt_addr,
            out_stores,
            reduce_op=None,
            scale_arg=None,
            o_addr=None,
        ):
            """Packet 148 (147 HOLD): the WHOLE downstream graph each backward template re-emits,
            proved by ORIENTATION and VALUE, not by pointer role. Every 2-D value gets a pair of
            logical coordinates (row, col) in {j = query row, k = key, d = head dim}; 1-D values
            get (coord,); a broadcast 1-D operand carries ('row'|'col', coord); splats and
            constants are 'uniform'; a zero-initialised loop accumulator is the wildcard 'acc'.
            Rules: layout ops and the shared-memory round trip are transparent; a transpose
            swaps; a load is oriented by its MASK (leaf signature -> j / k via the Q / K load
            masks, expand axis -> which side; the unmasked side is d iff its extent is the head
            dim); the recognised P is (j, k); a dot contracts A.col == B.row and its C is the bias
            (scores), a zero constant, or a zero-initialised accumulator; elementwise ops require
            equal full orientations and projections that match the axis they lie along; a reduce
            over addf keeps the surviving axis; ANY dtype conversion refuses (fp32 interior).
            Then per kind: dP = dO(j,d) @ V(d,k); dS = P * (dP - delta[j]); kv: dV = Pᵀ(k,j) @ dO,
            dK = dSᵀ(k,j) @ Q, stored (k,d) under the K mask, dK scaled; q: dQ = dS @ K(k,d)
            stored (j,d) under the Q mask, delta = rowsum(O*dO) over d stored under the Q mask;
            b: dbias += dS stored (j,k) under both masks. Returns the raw K / V load
            orientations so the lowering derives its stride swaps from the proof."""
            _qs = _leaf_sig(_q_mask_leaf(), n_ctx_arg, {block_j, block_k})
            _ks = _leaf_sig(_k_mask_leaf(), n_ctx_arg, {block_j, block_k})
            if _qs == _ks:
                _refuse("distinct row (Q) and column (K) coordinates")
            P_id = lse_exp_op.id
            _TRANS = ("tt.trans", "ttg.memdesc_trans")
            _LAY = ("ttg.convert_layout", "ttg.local_load", "ttg.local_alloc")
            # packet 174 (158 row 6): the source's rounding points for narrow dtypes, recorded
            # here and replayed by the makers (`round_k` / `round_q` at the scale product,
            # `round_p` before the dV dot, `round_ds` before the dK / dQ dots). Output casts
            # are peeled at the stores (the makers' store_cast replays them). Reset per proof.
            _rounding = {}
            _rounded_truncf = {}
            _dS_holder = [None]
            _ELEM = {"f16": "half", "bf16": "bfloat"}

            def _narrow_delta_layout_is_exact(reduce, product, elem):
                """Prove the one narrow reduction layout the q maker can replay.

                In the runtime D=32 specialization, each lane owns eight
                contiguous D values, four adjacent lanes cooperate on one row,
                and each of four warps owns eight rows.  The maker replays the
                standard binary in-thread tree and then the lane XOR 2,1 tree,
                rounding through the source combiner after every edge.  Any
                shape/layout/axis variation stays refused: size alone is not
                evidence of the reduction association.
                """
                source_meta = self.graph.result_meta.get(product.id)
                result_meta = self.graph.result_meta.get(reduce.id)
                if source_meta is None or result_meta is None:
                    return False, "missing per-result metadata"
                source_type = source_meta.type
                result_type = result_meta.type
                compact = lambda value: re.sub(r"\s+", "", value or "")
                blocked = compact(
                    "#ttg.blocked<{sizePerThread = [1, 8], "
                    "threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], "
                    "order = [1, 0]}>"
                )
                sliced = compact(f"#ttg.slice<{{dim = 1, parent = {blocked}}}>")
                try:
                    axis = int((reduce.attrs or {})["axis"])
                except (KeyError, TypeError, ValueError):
                    return False, "missing/invalid axis"
                checks = {
                    "axis": axis == 1,
                    "metadata-kind": source_meta.kind == result_meta.kind == "result",
                    "producer-identity": (
                        source_meta.producer_id == product.id and result_meta.producer_id == reduce.id
                    ),
                    "known-types": (source_type.unknown_reason is None and result_type.unknown_reason is None),
                    "element": (
                        source_type.kind == result_type.kind == "float" and source_type.elem == result_type.elem == elem
                    ),
                    "tensor-shapes": (
                        source_type.is_tensor
                        and result_type.is_tensor
                        and source_type.shape == (32, 32)
                        and result_type.shape == (32,)
                    ),
                    "source-layout": compact(source_type.layout) == blocked,
                    "result-layout": compact(result_type.layout) == sliced,
                }
                failed = [name for name, passed in checks.items() if not passed]
                detail = ",".join(failed)
                if "source-layout" in failed or "result-layout" in failed:
                    detail += f"; source={compact(source_type.layout)!r}; result={compact(result_type.layout)!r}"
                return not failed, detail

            def _elem_of(type_str):
                t = type_str or ""
                el = t.rsplit("x", 1)[-1].split(",")[0].split(">")[0].strip() if "x" in t else t.strip()
                return el

            def _classify_truncf(o):
                """A `truncf` on the backward value path is admitted ONLY as one of the source's
                own rounding points: P (the exp) before the dV dot (kv), or dS before the dK / dQ
                dots (kv, q). Its target type is recorded once; anything else refuses."""
                if o.id in _rounded_truncf:
                    return _rounded_truncf[o.id]
                el = _ELEM.get(_elem_of(o.type_str))
                src = _root(o.operand_ids[0]) if o.operand_ids else None
                if el is None:
                    _refuse(
                        f"no conversion to {_elem_of(o.type_str) or '?'} on the backward value path (the templates replay fp16 / bf16 rounding only)"
                    )
                if src == P_id and kind == "kv":
                    key = "round_p"
                elif _dS_holder[0] is not None and src == _dS_holder[0].id and kind in ("kv", "q"):
                    key = "round_ds"
                else:
                    _refuse(
                        "no dtype conversion on the backward value path except the source's own rounding of P before the dV dot (kv) and of dS before the dK / dQ dots (kv, q): the template replays exactly those and computes everything else in fp32"
                    )
                if _rounding.get(key, el) != el:
                    _refuse(f"one target type per rounding point ({key})")
                _rounding[key] = el
                _rounded_truncf[o.id] = key
                return key

            # ---- loop plumbing: block arg / result / yield positions ----
            ba_pos, res_pos, yields, inits_of, bids_of = {}, {}, {}, {}, {}
            for L in allops:
                if L.op != "scf.for":
                    continue
                bids = (L.attrs or {}).get("block_arg_ids") or []
                bids_of[L.id] = bids
                inits = list(L.operand_ids or [])[3:]
                results = list(L.result_ids or [L.id])
                y = next((o for o in reversed(L.region_ops or []) if o.op == "scf.yield"), None)
                yields[L.id] = y
                for i in range(len(inits)):
                    inits_of[(L.id, i)] = inits[i]
                    if i + 1 < len(bids):
                        ba_pos[bids[i + 1]] = (L.id, i)
                    if i < len(results):
                        res_pos[results[i]] = (L.id, i)

            def _yielded(lid, i):
                y = yields.get(lid)
                return y.operand_ids[i] if (y is not None and i < len(y.operand_ids or [])) else None

            def _is_zero_const(vid):
                o = op_by_id.get(_peel_ids(vid, _bcast))
                if o is None or o.op != "arith.constant":
                    return False
                v = _const_val(o)
                if v is not None:
                    return v == 0.0
                sv = str((o.attrs or {}).get("value", ""))
                return ("dense<0" in sv) or sv.strip() in ("0", "0.0", "0.000000e+00")

            def _leaf_coord(cid, mirrored_ok=False):
                try:
                    sig = _leaf_sig(cid, n_ctx_arg, {block_j, block_k}, mirrored_ok)
                except MetalNonRecoverableError:
                    return None
                return "j" if sig == _qs else "k" if sig == _ks else None

            def _mask_orient(mask_vid, shape):
                m = op_by_id.get(_peel_ids(mask_vid, _bcast))
                if m is not None and m.op == "arith.andi" and len(m.operand_ids or []) == 2:
                    pairs = {}
                    for x in m.operand_ids:
                        ax, co = _axis_of(x), _leaf_coord(x)
                        if ax not in ("row", "col") or co is None or ax in pairs:
                            return None
                        pairs[ax] = co
                    return (pairs.get("row"), pairs.get("col"))
                ax, co = _axis_of(mask_vid), _leaf_coord(mask_vid)
                if co is None or not shape or len(shape) != 2:
                    return None
                if ax == "row":
                    return (co, "d") if shape[1] == head_dim else None
                if ax == "col":
                    return ("d", co) if shape[0] == head_dim else None
                return None

            def _plain_sum_reduce(o):
                """Packet 152 (151 F2): the reduction region is EXACTLY one `addf` of its two block
                arguments — the template's sum. "Contains an addf" admitted `a + b + 1.0`."""
                bids = (o.attrs or {}).get("block_arg_ids") or []
                body = [b for b in (o.region_ops or [])]
                if len(bids) != 2 or len(body) != 1:
                    return False
                b = body[0]
                if not (b.op == "arith.addf" and sorted(b.operand_ids or []) == sorted(bids)):
                    return False
                # packet 154 (153): the region must RETURN that addition — a dead add with a
                # projection returned (`return %a`) is legal TTGIR and was indistinguishable;
                # missing return information must not imply a sum.
                rets = (o.attrs or {}).get("return_ids")
                return rets == [b.id]

            def _bcast_source(vid):
                cur, seen = vid, set()
                while cur not in seen:
                    seen.add(cur)
                    o = op_by_id.get(cur)
                    if o is None or not (o.operand_ids or []):
                        return None
                    if o.op in ("tt.broadcast", "ttg.convert_layout", "arith.extui", "arith.extsi", "arith.extf"):
                        cur = o.operand_ids[0]
                        continue
                    if o.op == "tt.expand_dims":
                        return o.operand_ids[0]
                    return None
                return None

            memo, stack = {}, set()

            def _orient(vid):
                if vid in memo:
                    return memo[vid]
                if vid in stack:
                    return None
                stack.add(vid)
                try:
                    r = _orient_(vid)
                finally:
                    stack.discard(vid)
                memo[vid] = r
                return r

            def _full(x):
                return isinstance(x, tuple) and len(x) == 2 and x[0] not in ("row", "col")

            def _orient_(vid):
                if vid in ba_pos:
                    lid, i = ba_pos[vid]
                    if _is_zero_const(inits_of[(lid, i)]):
                        return "acc"
                    return None
                if vid in res_pos:
                    lid, i = res_pos[vid]
                    y = _yielded(lid, i)
                    return _orient(y) if y is not None else None
                o = op_by_id.get(vid)
                if o is None:
                    a = arg_by_id.get(vid)
                    return "uniform" if (a is not None and not a.is_ptr) else None
                if o.op == "tt.fp_to_fp":
                    _refuse("no fp8 conversion on the backward value path (P, dP, dS, dK/dV/dQ/dbias)")
                if o.op == "arith.extf" and o.operand_ids:
                    return _orient(o.operand_ids[0])  # exact widening (packet 174)
                if o.op == "arith.truncf" and o.operand_ids:
                    _classify_truncf(o)  # a recorded rounding point or a refusal (packet 174)
                    return _orient(o.operand_ids[0])
                if o.op in _LAY and o.operand_ids:
                    return _orient(o.operand_ids[0])
                if o.op in _TRANS and o.operand_ids:
                    inner = _orient(o.operand_ids[0])
                    return (inner[1], inner[0]) if _full(inner) else None
                if o.op in ("tt.splat", "arith.constant"):
                    return "uniform"
                if o.op in ("tt.broadcast", "tt.expand_dims"):
                    ax = _axis_of(vid)
                    if ax == "uniform":
                        return "uniform"
                    if ax not in ("row", "col"):
                        return None
                    src = _bcast_source(vid)
                    c1 = _orient(src) if src is not None else None
                    if not (isinstance(c1, tuple) and len(c1) == 1):
                        return None
                    return (ax, c1[0])
                if o.op == "math.exp2":
                    return ("j", "k") if o.id == P_id else None
                if o.op == "tt.load":
                    shape = _extract_shape(o.type_str or "")
                    if not shape or len(o.operand_ids or []) < 2:
                        return None  # an unmasked tile load is not oriented
                    if len(shape) == 2:
                        return _mask_orient(o.operand_ids[1], shape)
                    if len(shape) == 1:
                        co = _leaf_coord(o.operand_ids[1])
                        return (co,) if co else None
                    return None
                if o.op == "tt.reduce" and o.operand_ids:
                    inner = _orient(o.operand_ids[0])
                    ax = (o.attrs or {}).get("axis")
                    if not _full(inner) or not _plain_sum_reduce(o) or ax not in (0, 1):
                        return None
                    return (inner[0],) if ax == 1 else (inner[1],)
                if o.op in ("arith.mulf", "arith.subf", "arith.addf") and len(o.operand_ids or []) == 2:
                    parts = [_orient(x) for x in o.operand_ids]
                    if any(p is None for p in parts):
                        return None
                    fulls = [p for p in parts if _full(p)]
                    if not fulls or any(f != fulls[0] for f in fulls):
                        return None
                    r, c = fulls[0]
                    for pp in parts:
                        if isinstance(pp, tuple) and len(pp) == 2 and pp[0] in ("row", "col"):
                            if (pp[0] == "row" and pp[1] != r) or (pp[0] == "col" and pp[1] != c):
                                return None
                        elif isinstance(pp, tuple) and len(pp) == 1:
                            return None  # an un-projected 1-D operand in a tile op
                    return (r, c)
                if o.op == "tt.dot" and len(o.operand_ids or []) >= 2:
                    a, b = _orient(o.operand_ids[0]), _orient(o.operand_ids[1])
                    if not (_full(a) and _full(b)) or a[1] is None or a[1] != b[0]:
                        return None
                    if len(o.operand_ids) > 2 and o.id != scores_dot.id:
                        cid = o.operand_ids[2]
                        if cid in ba_pos:
                            lid, i = ba_pos[cid]
                            if not _is_zero_const(inits_of[(lid, i)]):
                                return None
                        elif not _is_zero_const(cid):
                            return None
                    return (a[0], b[1])
                return None

            def _root(vid):
                return _peel_ids(vid, _LAY + _TRANS)

            def _root_rd(vid):
                """`_root` that also peels a `truncf` IF it is one of the source's recorded rounding
                points (classified on the way). Used only where the template replays that rounding:
                the A operand of the dV / dK / dQ accumulation dots. Everywhere else the strict
                root applies, so a rounded P feeding dS, or a rounded dS feeding dbias, still refuses."""
                cur, seen = vid, set()
                while cur not in seen:
                    seen.add(cur)
                    cur = _peel_ids(cur, _LAY + _TRANS)
                    o = op_by_id.get(cur)
                    if o is not None and o.op == "arith.truncf" and o.operand_ids:
                        _classify_truncf(o)
                        cur = o.operand_ids[0]
                        continue
                    return cur
                return cur

            def _need(vid, want, what):
                got = _orient(vid)
                if got != want:
                    _refuse(f"{what} oriented as {want} (proved {got})")

            def _load_of(vid):
                o = _load_op_of(vid)
                if o is None or o.op != "tt.load":
                    _refuse("a load under the dot operand")
                return o

            # ---- packet 150 (149 F1): VALUE identity per role. `load_addr` walks through a
            # multiply and is provenance only; an operand IS a load only if it peels through
            # layout / shared-memory / transpose ops to that load. The one arithmetic the
            # templates replay is the scale: `x * splat(scale_arg)` at the K operand (kv, q) or
            # the Q operand (b) of the score dot, and the same scaled K value into dQ (q).
            def _raw_load(vid, addr, what):
                o = op_by_id.get(_root(vid))
                if o is None or o.op != "tt.load" or load_addr(o.id) != addr:
                    _refuse(
                        f"{what} being the recognised load VALUE itself (no arithmetic between the load and the dot)"
                    )
                return o

            def _scaled_load(vid, addr, what):
                m = op_by_id.get(_root(vid))
                if m is None or m.op != "arith.mulf" or len(m.operand_ids or []) != 2:
                    _refuse(f"{what} being exactly load * splat(sm_scale)")
                sides = [(_root(x), _peel_ids(x, _bcast)) for x in m.operand_ids]
                ld = [op_by_id.get(r) for r, _ in sides]
                li = next(
                    (i for i, o in enumerate(ld) if o is not None and o.op == "tt.load" and load_addr(o.id) == addr),
                    None,
                )
                if li is None:
                    _refuse(f"{what} being exactly the recognised load * splat(sm_scale)")
                _mel = _elem_of(m.type_str)
                _sid = sides[1 - li][1]
                _so = op_by_id.get(_sid)
                if _so is not None and _so.op == "arith.truncf" and len(_so.operand_ids or []) == 1:
                    # packet 174: `tl.full([1], sm_scale, dtype)` rounds the fp32 scale argument to
                    # the input dtype before the splat — admitted only to the product's own dtype.
                    if _elem_of(_so.type_str) != _mel:
                        _refuse(f"{what} scaled by sm_scale rounded to the product's own dtype")
                    _sid = _peel_ids(_so.operand_ids[0], _bcast)
                sc = arg_by_id.get(_sid)
                if sc is None or sc.is_ptr or scale_arg is None or sc.index != scale_arg:
                    _refuse(f"{what} scaled by exactly the sm_scale argument the template binds")
                # packet 174: the product's element type IS a rounding point when narrower than
                # fp32 (the source computes x * full(scale, dtype) in that dtype: the scale rounded,
                # each product rounded). Recorded for the makers; any other type refuses.
                if _mel in _ELEM:
                    _rounding["round_q" if kind == "b" else "round_k"] = _ELEM[_mel]
                elif _mel not in ("f32", ""):
                    _refuse(f"{what} computed in {_mel}, which the templates cannot replay")
                return ld[li], m

            # The templates supply ZERO for every element a load's mask drops, and skip the
            # rows past N_CTX. A source ``other`` is what the source computes with there
            # (NaN or -inf reach the sums through 0 * x and exp), and an unmasked lse load
            # reads the tensor there. The Mask load is the exception: a key its key-boundary
            # mask drops gets the sentinel from the bounds select whatever the Mask says there
            # (both selects supply the one sentinel), so any constant, such as other=1, is the
            # same source.
            for _ld in allops:
                if _ld.op != "tt.load" or len(_ld.operand_ids or []) < 3 or _is_zero_const(_ld.operand_ids[2]):
                    continue
                _oc = op_by_id.get(_peel_ids(_ld.operand_ids[2], _bcast))
                if _ld.id != mask_loads[0].id or _oc is None or _oc.op != "arith.constant":
                    _refuse("every masked load's `other` being absent or zero (the templates supply zero)")
            _lse_ld = op_by_id.get(_peel_ids(lse_sub_op.operand_ids[1], _bcast))
            if (
                _lse_ld is None
                or _lse_ld.op != "tt.load"
                or len(_lse_ld.operand_ids or []) < 2
                or _leaf_coord(_lse_ld.operand_ids[1], mirrored_ok=True) != "j"
            ):
                _refuse("the lse load masked by exactly the query-row boundary")

            # The Bias and Mask loads are operands of no dot, so no contraction orients them:
            # their masks are compared with the coordinates their ADDRESSES run along (Bias
            # rows are query rows and its columns keys; Mask holds one value per key). A
            # boundary on the other axis zeroes cells inside the tile the templates read.
            def _mask_along(mask_vid, want):
                m = op_by_id.get(_peel_ids(mask_vid, _bcast))
                both = m is not None and m.op == "arith.andi" and len(m.operand_ids or []) == 2
                seen = set()
                for x in m.operand_ids if both else (mask_vid,):
                    ax, co = _axis_of(x), _leaf_coord(x, mirrored_ok=True)
                    if ax in seen or want.get(ax) is None or co != want[ax]:
                        return False
                    seen.add(ax)
                return True

            _mask_ld = mask_loads[0]
            if len(_mask_ld.operand_ids or []) >= 2 and _leaf_coord(_mask_ld.operand_ids[1], mirrored_ok=True) != "k":
                _refuse("the Mask load masked by exactly the key boundary (one Mask value per key)")

            # ---- shared: scores, dP, dS ----
            _need(scores_dot.id, ("j", "k"), "the score dot Q(j,d) @ K(d,k)")
            q_addr = load_addr(scores_dot.operand_ids[0])
            k_addr = load_addr(scores_dot.operand_ids[1])
            if kind == "b":
                _scaled_load(scores_dot.operand_ids[0], q_addr, "the score dot's Q operand")
                k_load = _raw_load(scores_dot.operand_ids[1], k_addr, "the score dot's K operand")
                k_val_root = None
            else:
                _raw_load(scores_dot.operand_ids[0], q_addr, "the score dot's Q operand")
                k_load, _km = _scaled_load(scores_dot.operand_ids[1], k_addr, "the score dot's K operand")
                k_val_root = _km.id
            if len(scores_dot.operand_ids or []) > 2:
                bo = op_by_id.get(_peel_ids(scores_dot.operand_ids[2], _LAY + ("arith.extf",)))
                if bo is None or bo.op != "tt.load" or load_addr(bo.id) != load_addr(scores_dot.operand_ids[2]):
                    _refuse(
                        "the score dot's accumulator being the Bias load VALUE itself (a widening is replayed; arithmetic is not)"
                    )
                if len(bo.operand_ids or []) >= 2 and not _mask_along(bo.operand_ids[1], {"row": "j", "col": "k"}):
                    _refuse(
                        "the Bias load masked by the query-row boundary along its rows and the key boundary along its columns"
                    )
            k_orient = _orient(k_load.id)
            dp = next(
                (
                    d
                    for d in dots
                    if d.id != scores_dot.id
                    and len(d.operand_ids or []) >= 2
                    and load_addr(d.operand_ids[0]) == do_addr
                    and load_addr(d.operand_ids[1]) == v_addr
                ),
                None,
            )
            if dp is None:
                _refuse("the dP dot dO @ Vᵀ on the recognised dO and V loads")
            _need(dp.id, ("j", "k"), "the dP dot dO(j,d) @ V(d,k)")
            _raw_load(dp.operand_ids[0], do_addr, "the dP dot's dO operand")
            v_load = _raw_load(dp.operand_ids[1], v_addr, "the dP dot's V operand")
            if len(dp.operand_ids or []) < 3 or not _is_zero_const(dp.operand_ids[2]):
                _refuse("the dP dot accumulating from a literal zero (not a loop-carried value)")
            v_orient = _orient(v_load.id)
            if not (_full(k_orient) and _full(v_orient)):
                _refuse("oriented K and V loads")
            dS = None
            for s_ in allops:
                if s_.op != "arith.mulf" or len(s_.operand_ids or []) != 2:
                    continue
                roots = [_root(x) for x in s_.operand_ids]
                if P_id not in roots:
                    continue
                other = op_by_id.get(next(r for r in roots if r != P_id)) if len(set(roots)) == 2 else None
                if (
                    other is not None
                    and other.op == "arith.subf"
                    and len(other.operand_ids or []) == 2
                    and _root(other.operand_ids[0]) == dp.id
                ):
                    dS = s_
                    break
            if dS is None:
                _refuse("dS = P * (dP - delta)")
            _dS_holder[0] = dS
            _need(dS.id, ("j", "k"), "dS = P(j,k) * (dP(j,k) - delta[j])")
            sub = op_by_id.get(next(r for r in [_root(x) for x in dS.operand_ids] if r != P_id))
            if kind in ("kv", "b"):
                dl = op_by_id.get(_bcast_source(sub.operand_ids[1]) or -1)
                if dl is None or dl.op != "tt.load" or load_addr(dl.id) != dlt_addr:
                    _refuse("delta broadcast from the recognised delta load")
            else:
                if reduce_op is None or _bcast_source(sub.operand_ids[1]) != reduce_op.id:
                    _refuse("delta broadcast from the rowsum(O * dO) reduce")
                # packet 152 (151 F1/F2): the reduce INPUT must be exactly O-load * dO-load (either
                # order, through value-preserving wrappers only) and the combiner exactly a + b.
                rin = op_by_id.get(_peel_ids(reduce_op.operand_ids[0], _LAY + ("arith.extf",)))
                if rin is None or rin.op != "arith.mulf" or len(rin.operand_ids or []) != 2:
                    _refuse("the delta reduce input being exactly the product O * dO")
                _pel = _elem_of(rin.type_str)
                if _pel in _ELEM:
                    _rounding["round_dprod"] = _ELEM[_pel]  # packet 174: each O*dO product rounded, summed in fp32
                elif _pel not in ("f32", ""):
                    _refuse(f"the delta product computed in {_pel}, which the template cannot replay")
                _roots = [op_by_id.get(_root(x)) for x in rin.operand_ids]
                _addrs = sorted(load_addr(r.id) if (r is not None and r.op == "tt.load") else -1 for r in _roots)
                if o_addr is None or _addrs != sorted([o_addr, do_addr]):
                    _refuse(
                        "the delta reduce input being exactly the recognised O load VALUE times the recognised dO load VALUE (no other arithmetic)"
                    )
                if not _plain_sum_reduce(reduce_op):
                    _refuse("the delta reduction combiner being exactly a + b (the template sums)")
                _rel = _elem_of(reduce_op.type_str)
                if _rel in _ELEM:
                    _layout_ok, _layout_failure = _narrow_delta_layout_is_exact(reduce_op, rin, _rel)
                    if not _layout_ok:
                        _refuse(
                            "the narrow delta reduction having the exact D=32 "
                            "eight-values-per-lane blocked layout and axis-1 "
                            "binary/XOR tree "
                            f"(failed: {_layout_failure})"
                        )
                    _rounding["round_delta"] = _ELEM[_rel]
                _need(reduce_op.id, ("j",), "delta = rowsum over d of O(j,d) * dO(j,d)")

            _output_casts = {}
            _store_roles = {st.id: role for role, st in out_stores.items()}

            def _out_cast(st):
                """Prove and record the conversion of THIS output role, including no conversion.

                A sibling's dtype says nothing about this store. The dot accumulators are
                fp32: either store them to fp32, or replay their one narrowing to this
                pointer's own element type. The one proved narrow-delta reduction is
                already narrow at its result; its direct narrow store still requires the
                maker's float carrier to narrow back to that same pointer element type.
                The surrounding proof checks the underlying value in every case.
                """
                role = _store_roles[st.id]
                v = st.operand_ids[1]
                o = op_by_id.get(_peel_ids(v, _LAY))
                _po = op_by_id.get(st.operand_ids[0])
                _pm = re.search(r"!tt\.ptr<([A-Za-z0-9_]+)", (_po.type_str or "") if _po is not None else "")
                if o is None or o.op != "arith.truncf" or not o.operand_ids:
                    source_el = _ELEM.get(_elem_of(o.type_str)) if o is not None else None
                    pointer_el = _ELEM.get(_pm.group(1)) if _pm else None
                    if (
                        role == "delta"
                        and _rounding.get("round_delta") is not None
                        and source_el == pointer_el == _rounding["round_delta"]
                    ):
                        _output_casts[role] = source_el
                        return v
                    if _pm is None or _pm.group(1) != "f32":
                        _refuse(f"{role}: an unconverted fp32 accumulator stored to its own fp32 pointer")
                    _output_casts[role] = None
                    return v
                el = _ELEM.get(_elem_of(o.type_str))
                pel = _ELEM.get(_pm.group(1)) if _pm else None
                if el is None or pel != el:
                    _refuse(
                        "an output cast to exactly the stored pointer's element type (fp16 / bf16), which the template replays"
                    )
                _output_casts[role] = el
                _rounded_truncf[o.id] = "out"
                return o.operand_ids[0]

            def _store_check(st, want, what, mask_want=None):
                val = _out_cast(st)
                _need(val, want, what)
                # the stored tile's shape: the first typed op on the value's layout chain (the
                # store itself has no result type; a loop result id is not an op)
                shape, cur, seen = None, st.operand_ids[1], set()  # the typed chain (incl. an output cast)
                while cur not in seen and shape is None:
                    seen.add(cur)
                    o_ = op_by_id.get(cur)
                    if o_ is None:
                        break
                    shape = _extract_shape(o_.type_str or "") or None
                    cur = o_.operand_ids[0] if (o_.op in _LAY and o_.operand_ids) else cur
                if len(st.operand_ids or []) < 3:
                    _refuse(f"{what} stored under a mask")
                mo = _mask_orient(st.operand_ids[2], shape) if len(want) == 2 else ((_leaf_coord(st.operand_ids[2]),))
                if mo != (mask_want or want):
                    _refuse(f"{what} stored under the {mask_want or want} mask (proved {mo})")

            def _acc_of(st, scaled=False):
                """Packet 150 (149 F2): the exact recurrence behind a store — (loop, position) from
                the store's loop RESULT, that position's zero INIT, its BLOCK ARG and its YIELD.
                Returns (yielded op, block arg id). The accumulating op must consume THAT block arg."""
                v = _peel_ids(_out_cast(st), _LAY)
                o = op_by_id.get(v)
                if scaled:
                    if o is None or o.op != "arith.mulf" or len(o.operand_ids or []) != 2:
                        _refuse("the scaled output store being mulf(loop result, splat(sm_scale))")
                    cand = [
                        (_peel_ids(x, _LAY), _peel_ids(y_, _bcast)) for x, y_ in (o.operand_ids, o.operand_ids[::-1])
                    ]
                    pick = next(((r, sc) for r, sc in cand if r in res_pos), None)
                    if pick is None:
                        _refuse("the scaled output store multiplying the loop result")
                    v, sc_id = pick
                    sc = arg_by_id.get(sc_id)
                    if sc is None or sc.is_ptr or scale_arg is None or sc.index != scale_arg:
                        _refuse("the output scaled by exactly the sm_scale argument the template binds")
                elif o is not None and o.op == "arith.mulf":
                    _refuse("an unscaled output store (this output is not scaled by the template)")
                if v not in res_pos:
                    if kind == "q" and n_ctx_arg == C1:
                        direct = op_by_id.get(v)
                        if direct is not None and direct.op == "tt.dot":
                            return direct, None
                    _refuse("an output accumulated by the j/k loop")
                lid, i = res_pos[v]
                if not _is_zero_const(inits_of[(lid, i)]):
                    _refuse("the output accumulator initialised to literal zero")
                bids = bids_of.get(lid) or []
                if i + 1 >= len(bids):
                    _refuse("a resolvable accumulator block argument")
                y = op_by_id.get(_yielded(lid, i))
                return y, bids[i + 1]

            def _acc_dot(st, a_root, b_addr, what, scaled=False, b_root=None):
                y, ba = _acc_of(st, scaled)
                if y is None or y.op != "tt.dot" or len(y.operand_ids or []) < 3:
                    _refuse(f"{what} being a dot accumulated into the loop carry")
                if _root_rd(y.operand_ids[0]) != a_root:
                    _refuse(
                        f"{what}: the first operand being exactly the proved value (or the source's recorded rounding of it)"
                    )
                if b_root is not None:
                    if _root(y.operand_ids[1]) != b_root:
                        _refuse(f"{what}: the second operand being exactly the scaled K value used by the score dot")
                else:
                    _raw_load(y.operand_ids[1], b_addr, f"{what}: the second operand")
                if ba is None and not _is_zero_const(y.operand_ids[2]):
                    _refuse(f"{what} accumulating from literal zero in the proved singleton specialization")
                if ba is not None and y.operand_ids[2] != ba:
                    _refuse(
                        f"{what} accumulating into ITS OWN loop carry (C is the block argument of the stored result's position)"
                    )

            if kind == "kv":
                dk_st, dv_st = out_stores["dk"], out_stores["dv"]
                _store_check(dv_st, ("k", "d"), "dV = Pᵀ(k,j) @ dO(j,d)")
                _store_check(dk_st, ("k", "d"), "dK = dSᵀ(k,j) @ Q(j,d), scaled")
                _acc_dot(dv_st, P_id, do_addr, "the dV accumulation dot(Pᵀ, dO)")
                _acc_dot(dk_st, dS.id, q_addr, "the dK accumulation dot(dSᵀ, Q)", scaled=True)
            elif kind == "q":
                dq_st, dlt_st = out_stores["dq"], out_stores["delta"]
                _store_check(dq_st, ("j", "d"), "dQ = dS(j,k) @ K(k,d)")
                _store_check(dlt_st, ("j",), "delta(j)")
                _acc_dot(dq_st, dS.id, k_addr, "the dQ accumulation dot(dS, K·scale)", b_root=k_val_root)
                if _peel_ids(_out_cast(dlt_st), _LAY) != reduce_op.id:
                    _refuse("the delta store value being the rowsum reduce itself")
            else:
                db_st = out_stores["db"]
                _store_check(db_st, ("j", "k"), "dbias = Σ_i dS(j,k)")
                y, ba = _acc_of(db_st)
                if y is None or y.op != "arith.addf" or len(y.operand_ids or []) != 2:
                    _refuse("the dbias accumulation being acc + dS")
                ops_ = [_peel_ids(x, _LAY) for x in y.operand_ids]
                if sorted(ops_, key=lambda x: x == dS.id) != [ba, dS.id]:
                    _refuse("the dbias accumulation being exactly ITS OWN loop carry plus dS")
            if set(_output_casts) != set(out_stores):
                _refuse("a proved conversion for every output store role")
            return {
                "k_orient": k_orient,
                "v_orient": v_orient,
                "rounding": {**_rounding, "output_casts": _output_casts},
            }

        # ----- N_CTX (loop bound) + H (mask cross-head divisor): shared by q/kv -----
        def _arg_through_casts(sid):
            seen = set()
            while sid not in seen:
                seen.add(sid)
                a = arg_by_id.get(sid)
                if a is not None:
                    return a.index if not a.is_ptr else None
                op = op_by_id.get(sid)
                if op is not None and op.op == "arith.constant":
                    try:
                        return C1 if int(op.attrs.get("value")) == 1 else None
                    except (TypeError, ValueError):
                        return None
                if op is not None and op.op in ("arith.extsi", "arith.trunci", "arith.index_cast") and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return None

        def _resolve_n_ctx():
            saw_loop = False
            for s in allops:
                if s.op == "scf.for" and len(s.operand_ids or []) >= 2:
                    saw_loop = True
                    r = _arg_through_casts(s.operand_ids[1])
                    if r is not None:
                        return r
            # N_CTX == 1 is constexpr-specialised and the one-iteration K loop
            # disappears. The boundary/value proofs below still have to prove
            # the exact constant-one Q/K coordinates before this can route.
            return None if saw_loop else C1

        def _resolve_h_div():
            h = C1
            for s in allops:
                if (
                    s.op in ("arith.divui", "arith.divsi")
                    and s.operand_ids
                    and any(
                        op_by_id.get(o) is not None and op_by_id.get(o).op == "tt.get_program_id" for o in s.operand_ids
                    )
                ):
                    for o in s.operand_ids:
                        a = arg_by_id.get(o)
                        if a is not None and not a.is_ptr:
                            h = a.index
                            break
            return h

        # ====================== _bwd_q (dQ + delta) ==========================
        if kind == "q":
            # delta = rowsum(O * dO): the reduce input is mulf(O_load, dO_load).
            reduce_op = next((s for s in allops if s.op == "tt.reduce"), None)
            mul = op_by_id.get(skip_layout(reduce_op.operand_ids[0])) if (reduce_op and reduce_op.operand_ids) else None
            if mul is not None and mul.op == "arith.extf" and mul.operand_ids:
                # packet 174: `tl.sum((o * do).to(tl.float32), 1)` — an exact widening of the
                # narrow product; the proof records the product's rounding (round_dprod).
                mul = op_by_id.get(skip_layout(mul.operand_ids[0]))
            if mul is None or mul.op != "arith.mulf" or len(mul.operand_ids or []) < 2:
                _refuse("the delta reduce (O * dO)")
            od_cands = []
            for o in mul.operand_ids:
                la = load_addr(o)
                r = resolve_2d(la) if la is not None else None
                if r is not None:
                    od_cands.append((la, r))
            if len(od_cands) != 2:
                _refuse("the O and dO loads")

            # dP dot = dot(dO, Vᵀ): the dot (not scores) whose op0 load is one of
            # {O,dO} -> that's dO; op1 -> V. (tl.dot without an explicit acc still
            # emits a 3-operand tt.dot with a zero C, so match on op0/op1, not count.)
            do_addr = v_res = _v_addr_q = None
            for d in dots:
                if d.id == scores_dot.id or len(d.operand_ids or []) < 2:
                    continue
                a0 = load_addr(d.operand_ids[0])
                a1 = load_addr(d.operand_ids[1])
                if a0 is not None and a1 is not None and a0 in [c[0] for c in od_cands]:
                    do_addr = a0
                    v_res = resolve_2d(a1)
                    _v_addr_q = a1
                    break
            if do_addr is None or v_res is None:
                _refuse("the dO / V pointers (dP = dO @ Vᵀ)")
            _o_addr_q = next(la for la, r in od_cands if la != do_addr)
            if not (_offs_match_q(_v_addr_q) and _offs_match_q(do_addr) and _offs_match_q(_o_addr_q)):
                return None  # V/dO/O batch-head offsets differ from Q's -> generic path
            do_res = next(r for la, r in od_cands if la == do_addr)
            o_res = next(r for la, r in od_cands if la != do_addr)

            # delta OUTPUT store (value traces to the reduce) + dQ store (the other).
            def _traces_to_reduce(vid):
                sid = skip_layout(vid)
                seen = set()
                while sid not in seen:
                    seen.add(sid)
                    op = op_by_id.get(sid)
                    if op is None:
                        return False
                    if op.op == "tt.reduce":
                        return True
                    if (
                        op.op
                        in (
                            "arith.truncf",
                            "arith.extf",
                            "ttg.convert_layout",
                            "tt.reshape",
                            "ttg.local_load",
                            "ttg.local_alloc",
                        )
                        and op.operand_ids
                    ):
                        sid = op.operand_ids[0]
                        continue
                    return False
                return False

            dlt_res = dq_res = _dq_addr = None
            _dlt_store = _dq_store = None
            for st_ in stores:
                if len(st_.operand_ids) > 1 and _traces_to_reduce(st_.operand_ids[1]):
                    dlt_res = resolve_1d(st_.operand_ids[0])
                    _dlt_store = st_
                else:
                    dq_res = resolve_2d(st_.operand_ids[0])
                    _dq_addr = st_.operand_ids[0]
                    _dq_store = st_
            if dlt_res is None:
                _refuse("the delta output store")
            if dq_res is None:
                _refuse("the dQ output store")
            if not _offs_match_q(_dq_addr):
                return None  # dQ store batch-head offsets differ from Q's -> generic path

            # scale: sm_scale is folded into K (mulf with a splat of a float arg).
            scale_arg = None
            for s in allops:
                if s.op != "arith.mulf":
                    continue
                for o in s.operand_ids:
                    node = op_by_id.get(skip_layout(o))
                    if node is not None and node.op == "tt.splat" and node.operand_ids:
                        inner = node.operand_ids[0]
                        w = op_by_id.get(inner)
                        while w is not None and w.op in ("arith.truncf", "arith.extf") and w.operand_ids:
                            inner = w.operand_ids[0]
                            w = op_by_id.get(inner)
                        a = arg_by_id.get(inner)
                        if a is not None and not a.is_ptr and not str(a.elem_type).startswith("i"):
                            scale_arg = a.index
                if scale_arg is not None:
                    break
            if scale_arg is None:
                _refuse("the sm_scale argument")

            n_ctx_arg = _resolve_n_ctx()
            if n_ctx_arg is None:
                _refuse("the N_CTX loop bound")
            h_arg = _resolve_h_div()

            # shared-across-i pointers (bias) -> h-slot 0 on the 3-D grid.
            if grid_3d:
                for _res in (q_res, k_res, v_res, b_res, do_res, o_res, dq_res):
                    if _res[1][1] == C1:
                        _res[1][1] = "c0"

            a_shape = _extract_shape(self._find_op_type_str(scores_dot.operand_ids[0]))
            if not a_shape or len(a_shape) != 2:
                _refuse("the QK dot A shape")
            block_j, head_dim = a_shape
            qk_out = _extract_shape(scores_dot.type_str or "")
            block_k = qk_out[1] if (qk_out and len(qk_out) == 2) else block_j
            out_dtype = (
                "f16"
                if str(self.graph.args[dq_res[0]].elem_type) in ("fp16", "f16")
                else ("bf16" if str(self.graph.args[dq_res[0]].elem_type) in ("bf16", "bfloat16") else "f32")
            )

            sentinel_ref = _prove(n_ctx_arg, block_j, block_k)  # packet 136 whole-path proof
            _ds = _prove_downstream(
                "q",
                n_ctx_arg,
                block_j,
                block_k,
                head_dim,
                do_addr,
                _v_addr_q,
                None,
                {"dq": _dq_store, "delta": _dlt_store},
                reduce_op=reduce_op,
                scale_arg=scale_arg,
                o_addr=_o_addr_q,
            )
            roles = [
                q_res[0],
                k_res[0],
                v_res[0],
                b_res[0],
                m_res[0],
                lse_res[0],
                dlt_res[0],
                do_res[0],
                o_res[0],
                dq_res[0],
            ]
            if len(set(roles)) != 10:
                _refuse("ten distinct pointer roles")

            if not grid_3d:
                raise MetalNonRecoverableError(_BWD_2D_MSG)
            return {
                "bwd_kind": "q",
                "grid_3d": grid_3d,
                "q": q_res,
                "k": k_res,
                "v": v_res,
                "bias": b_res,
                "mask": m_res,
                "lse": lse_res,
                "delta": dlt_res,
                "do": do_res,
                "o": o_res,
                "dq": dq_res,
                "sentinel": sentinel_ref,
                "scale_arg": scale_arg,
                "N_CTX": n_ctx_arg,
                "H": h_arg,
                "block_j": block_j,
                "block_k": block_k,
                "head_dim": head_dim,
                "out_dtype": out_dtype,
                "k_orient": _ds["k_orient"],
                "v_orient": _ds["v_orient"],
                "rounding": _ds.get("rounding") or {},  # packet 174
            }

        # ======================= _bwd_b (dbias) ==============================
        # Here the triangle-i axis is a LOOP (grid = (pid_j, pid_k, pid_h)); Q/K/V/dO/
        # lse/delta/mask advance by their i-stride each iteration -> that stride lives
        # in the loop-body addptr, not the initial pointer. bias/db do not depend on i.
        if kind == "b":
            # per-iteration advance stride of a loop-carried pointer (its i-stride).
            def _off_arg(oid):
                direct = uniform_stride(oid)
                if direct is not None:
                    return direct
                seen = set()
                while oid not in seen:
                    seen.add(oid)
                    oid2 = skip_layout(oid)
                    op = op_by_id.get(oid2)
                    if op is not None and op.op in ("tt.splat", "tt.broadcast") and op.operand_ids:
                        oid = op.operand_ids[0]
                        continue
                    if op is not None and op.op == "arith.muli":
                        for o in op.operand_ids:
                            sub = op_by_id.get(skip_layout(o))
                            if sub is not None and sub.op in ("tt.splat", "tt.broadcast") and sub.operand_ids:
                                a = arg_by_id.get(sub.operand_ids[0])
                                if a is not None and not a.is_ptr:
                                    return a.index
                            a = arg_by_id.get(skip_layout(o))
                            if a is not None and not a.is_ptr:
                                return a.index
                        return None
                    a = arg_by_id.get(oid2)
                    return a.index if (a is not None and not a.is_ptr) else None
                return None

            def _advance_stride(ba_id):
                for s in allops:
                    if s.op == "tt.addptr" and len(s.operand_ids or []) >= 2 and s.operand_ids[0] == ba_id:
                        r = _off_arg(s.operand_ids[1])
                        if r is not None:
                            return r
                return None

            # Q/K/bias from the scores dot; capture raw (block-arg) addrs for advances.
            q_addr = load_addr(scores_dot.operand_ids[0])
            k_addr = load_addr(scores_dot.operand_ids[1])
            q_res = resolve_2d(q_addr)
            k_res = resolve_2d(k_addr)
            b_res = resolve_2d(load_addr(scores_dot.operand_ids[2]))
            if not (q_res and k_res and b_res):
                _refuse("Q/K/bias")

            # dP dot = the one non-scores dot: dot(dO, Vᵀ) -> op0=dO, op1=V.
            dp_dot = next((d for d in dots if d.id != scores_dot.id), None)
            if dp_dot is None or len(dp_dot.operand_ids or []) < 2:
                _refuse("the dP dot (dO @ Vᵀ)")
            do_addr = load_addr(dp_dot.operand_ids[0])
            v_addr = load_addr(dp_dot.operand_ids[1])
            do_res = resolve_2d(do_addr)
            v_res = resolve_2d(v_addr)
            if do_res is None or v_res is None:
                _refuse("the dO / V pointers (dP = dO @ Vᵀ)")
            if not (_offs_match_q(v_addr) and _offs_match_q(do_addr)):
                return None  # V/dO batch-head offsets differ from Q's -> generic path

            # delta LOADED: subf(dP_result, delta_load) -> the load.
            dlt_res = dlt_addr = None
            for s in allops:
                if s.op == "arith.subf" and s.operand_ids:
                    for o in s.operand_ids:
                        la = load_addr(o)
                        r = resolve_1d(la) if la is not None else None
                        if r is not None and r[0] not in (lse_res[0], m_res[0]):
                            others = [x for x in s.operand_ids if x != o]
                            if others and load_addr(others[0]) is None:
                                dlt_res, dlt_addr = r, la
            if dlt_res is None:
                _refuse("the delta pointer (dP - delta)")

            # db store: addptr(splat(base), addi(j*sm_row, k*sn_col)) -- a SINGLE addptr
            # with a FUSED offset (unlike q/k/v's chained addptrs), so resolve_2d's
            # chained form doesn't match. Resolve the fused offset directly, tagging
            # each addi term as row (broadcast input [BJ,1]) or col ([1,BK]).
            def _range_axis(tid):
                sid = skip_layout(tid)
                seen = set()
                while sid not in seen:
                    seen.add(sid)
                    op = op_by_id.get(sid)
                    if op is None:
                        return None
                    if op.op == "tt.broadcast" and op.operand_ids:
                        insh = _extract_shape(self._find_op_type_str(op.operand_ids[0]))
                        if insh and len(insh) == 2:
                            if insh[1] == 1:
                                return 0
                            if insh[0] == 1:
                                return 1
                        return None
                    nxt = None
                    for o in op.operand_ids or []:
                        sub = op_by_id.get(skip_layout(o))
                        if sub is not None and sub.op != "tt.splat":
                            nxt = o
                            break
                    if nxt is None:
                        return None
                    sid = skip_layout(nxt)
                return None

            def _resolve_db(addr_id):
                aid = skip_layout(addr_id)
                op = op_by_id.get(aid)
                if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
                    return None
                sp = op_by_id.get(skip_layout(op.operand_ids[0]))
                if sp is None or sp.op != "tt.splat" or not sp.operand_ids:
                    return None
                bs = _base_and_scalar(sp.operand_ids[0])
                if bs is None:
                    return None
                base, z, h = bs
                off = op_by_id.get(skip_layout(op.operand_ids[1]))
                if off is None or off.op != "arith.addi" or len(off.operand_ids) < 2:
                    return None
                row = col = None
                for term in off.operand_ids:
                    ax = _range_axis(term)
                    if ax == 0:
                        row = stride_from_term(term)
                    elif ax == 1:
                        col = stride_from_term(term)
                s = [z, h, row, col]
                return None if any(x is None for x in s) else (base.index, s)

            if len(stores) != 1:
                _refuse("a single dbias store")
            if not _offs_match_q(stores[0].operand_ids[0], subset=True):
                return None  # db store head offset differs from Q's -> generic path
            db_res = _resolve_db(stores[0].operand_ids[0])
            if db_res is None:
                _refuse("the dbias store pointer")

            # scale: sm_scale folded into Q (mulf with a splat of a float arg).
            scale_arg = None
            for s in allops:
                if s.op != "arith.mulf":
                    continue
                for o in s.operand_ids:
                    node = op_by_id.get(skip_layout(o))
                    if node is not None and node.op == "tt.splat" and node.operand_ids:
                        inner = node.operand_ids[0]
                        w = op_by_id.get(inner)
                        while w is not None and w.op in ("arith.truncf", "arith.extf") and w.operand_ids:
                            inner = w.operand_ids[0]
                            w = op_by_id.get(inner)
                        a = arg_by_id.get(inner)
                        if a is not None and not a.is_ptr and not str(a.elem_type).startswith("i"):
                            scale_arg = a.index
                if scale_arg is not None:
                    break
            if scale_arg is None:
                _refuse("the sm_scale argument")

            # i-advance strides -> baked into stride-slot [1] (the "i" slot) of each
            # loop-advanced pointer (bias/db keep C1; they don't depend on i).
            for _res, _addr in (
                (q_res, q_addr),
                (k_res, k_addr),
                (v_res, v_addr),
                (do_res, do_addr),
                (lse_res, lse_addr),
                (m_res, mask_addr),
                (dlt_res, dlt_addr),
            ):
                adv = _advance_stride(_addr)
                if adv is None:
                    # The reporter rebuilds each pointer from its base + i*stride,
                    # rather than carrying an advanced pointer around the loop.
                    # Use the already resolved i coefficient only for a single
                    # exact range(0, N, 1) loop and its own bare induction value.
                    loops = [x for x in allops if x.op == "scf.for"]
                    addr = _addr
                    cast = op_by_id.get(addr)
                    if cast is not None and cast.op == "tt.bitcast" and "!tt.ptr<i8>" in (cast.type_str or ""):
                        src = op_by_id.get(cast.operand_ids[0])
                        if src is not None and "!tt.ptr<i1>" in (src.type_str or ""):
                            addr = src.id
                    offsets = _scalar_offsets(_scal_of(addr))
                    if len(loops) == 1:
                        loop = loops[0]
                        bids = (loop.attrs or {}).get("block_arg_ids", [])
                        lo = op_by_id.get(loop.operand_ids[0])
                        step = op_by_id.get(loop.operand_ids[2])
                        if (
                            bids
                            and offsets
                            and len(offsets) == 2
                            and offsets[0] == bids[0]
                            and lo is not None
                            and lo.op == "arith.constant"
                            and lo.attrs.get("value") == 0
                            and step is not None
                            and step.op == "arith.constant"
                            and step.attrs.get("value") == 1
                        ):
                            adv = _res[1][1]
                    if adv is None:
                        _refuse("an i-loop advance stride")
                _res[1][1] = adv

            n_ctx_arg = _resolve_n_ctx()
            if n_ctx_arg is None:
                _refuse("the N_CTX loop bound")
            h_arg = _resolve_h_div()

            a_shape = _extract_shape(self._find_op_type_str(scores_dot.operand_ids[0]))
            if not a_shape or len(a_shape) != 2:
                _refuse("the QK dot A shape")
            block_j, head_dim = a_shape
            qk_out = _extract_shape(scores_dot.type_str or "")
            block_k = qk_out[1] if (qk_out and len(qk_out) == 2) else block_j
            out_dtype = (
                "f16"
                if str(self.graph.args[db_res[0]].elem_type) in ("fp16", "f16")
                else ("bf16" if str(self.graph.args[db_res[0]].elem_type) in ("bf16", "bfloat16") else "f32")
            )

            sentinel_ref = _prove(n_ctx_arg, block_j, block_k)  # packet 136 whole-path proof
            _ds = _prove_downstream(
                "b",
                n_ctx_arg,
                block_j,
                block_k,
                head_dim,
                do_addr,
                v_addr,
                dlt_addr,
                {"db": stores[0]},
                scale_arg=scale_arg,
            )
            roles = [q_res[0], k_res[0], v_res[0], b_res[0], m_res[0], lse_res[0], dlt_res[0], do_res[0], db_res[0]]
            if len(set(roles)) != 9:
                _refuse("nine distinct pointer roles")

            return {
                "bwd_kind": "b",
                "grid_3d": True,
                "q": q_res,
                "k": k_res,
                "v": v_res,
                "bias": b_res,
                "mask": m_res,
                "lse": lse_res,
                "delta": dlt_res,
                "do": do_res,
                "db": db_res,
                "sentinel": sentinel_ref,
                "scale_arg": scale_arg,
                "N_CTX": n_ctx_arg,
                "H": h_arg,
                "block_j": block_j,
                "block_k": block_k,
                "head_dim": head_dim,
                "out_dtype": out_dtype,
                "k_orient": _ds["k_orient"],
                "v_orient": _ds["v_orient"],
                "rounding": _ds.get("rounding") or {},  # packet 174
            }

        # ========================= _bwd_kv (dK/dV) ===========================
        def _traces_to_exp2(oid):
            sid = skip_layout(oid)
            seen = set()
            while sid not in seen:
                seen.add(sid)
                op = op_by_id.get(sid)
                if op is None:
                    return False
                if op.op == "math.exp2" or _op_is_exp(op.op):
                    return True
                if (
                    op.op
                    in (
                        "ttg.local_load",
                        "ttg.local_alloc",
                        "ttg.memdesc_trans",
                        "tt.trans",
                        "tt.reshape",
                        "ttg.convert_layout",
                        "arith.extf",
                        "arith.truncf",
                    )
                    and op.operand_ids
                ):
                    sid = op.operand_ids[0]
                    continue
                return False
            return False

        # dV dot = dot(Pᵀ, dO): its FIRST operand traces to the exp2 (P). dO = 2nd.
        do_addr = None
        for d in dots:
            if len(d.operand_ids or []) >= 2 and _traces_to_exp2(d.operand_ids[0]):
                do_addr = load_addr(d.operand_ids[1])
                break
        do_res = resolve_2d(do_addr) if do_addr is not None else None
        if do_res is None:
            _refuse("the dO pointer (dV = Pᵀ @ dO)")

        # dP dot = dot(dO, Vᵀ): its FIRST operand's load == dO. V = 2nd operand's load.
        v_res = _v_addr_kv = None
        for d in dots:
            if len(d.operand_ids or []) >= 2 and load_addr(d.operand_ids[0]) == do_addr:
                _va = load_addr(d.operand_ids[1])
                cand = resolve_2d(_va)
                if cand is not None and cand[0] not in (q_res[0], k_res[0], b_res[0], do_res[0]):
                    v_res = cand
                    _v_addr_kv = _va
                    break
        if v_res is None:
            _refuse("the V pointer (dP = dO @ Vᵀ)")
        if not (_offs_match_q(_v_addr_kv) and _offs_match_q(do_addr)):
            return None  # V/dO batch-head offsets differ from Q's -> generic path

        # delta: dscores = P * (dP - delta). Find subf(dot_result, load) -> the load.
        dlt_res = _dlt_addr_kv = None
        for s in allops:
            if s.op == "arith.subf" and s.operand_ids:
                for o in s.operand_ids:
                    la = load_addr(o)
                    r = resolve_1d(la) if la is not None else None
                    if r is not None and r[0] not in (lse_res[0], m_res[0]):
                        # the OTHER operand must be a dot result (dP), not a load
                        others = [x for x in s.operand_ids if x != o]
                        if others and load_addr(others[0]) is None:
                            dlt_res, _dlt_addr_kv = r, la
        if dlt_res is None:
            _refuse("the delta pointer (dP - delta)")

        # dK / dV stores: dK's value traces through mulf(_, splat(sm_scale)); dV doesn't.
        def _traces_scale_mul(vid):
            sid = skip_layout(vid)
            seen = set()
            while sid not in seen:
                seen.add(sid)
                op = op_by_id.get(sid)
                if op is None:
                    return None
                if op.op in ("arith.truncf", "arith.extf", "ttg.convert_layout") and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                if op.op == "arith.mulf":
                    for o in op.operand_ids:
                        sp = op_by_id.get(skip_layout(o))
                        if sp is not None and sp.op == "tt.splat" and sp.operand_ids:
                            a = arg_by_id.get(sp.operand_ids[0])
                            if a is not None and not a.is_ptr and not str(a.elem_type).startswith("i"):
                                return a.index
                return None
            return None

        dk_res = dv_res = scale_from_dk = None
        _dk_store = _dv_store = None
        for st_ in stores:
            addr_r = resolve_2d(st_.operand_ids[0])
            if addr_r is None:
                continue
            if not _offs_match_q(st_.operand_ids[0]):
                return None  # dK/dV store batch-head offsets differ from Q's -> generic path
            sc = _traces_scale_mul(st_.operand_ids[1]) if len(st_.operand_ids) > 1 else None
            if sc is not None and dk_res is None:
                dk_res, scale_from_dk, _dk_store = addr_r, sc, st_
            else:
                dv_res, _dv_store = addr_r, st_
        if dk_res is None or dv_res is None or scale_from_dk is None:
            _refuse("the dK (scaled) / dV store pointers")

        # N_CTX = the j-loop bound; H = the pid divisor (mask cross-head).
        def _arg_through_casts(sid):
            seen = set()
            while sid not in seen:
                seen.add(sid)
                a = arg_by_id.get(sid)
                if a is not None:
                    return a.index if not a.is_ptr else None
                op = op_by_id.get(sid)
                if op is not None and op.op in ("arith.extsi", "arith.trunci", "arith.index_cast") and op.operand_ids:
                    sid = op.operand_ids[0]
                    continue
                break
            return None

        n_ctx_arg = None
        for s in allops:
            if s.op == "scf.for" and len(s.operand_ids or []) >= 2:
                n_ctx_arg = _arg_through_casts(s.operand_ids[1])
                if n_ctx_arg is not None:
                    break
        if n_ctx_arg is None:
            _refuse("the N_CTX loop bound")

        h_arg = C1
        for s in allops:
            if (
                s.op in ("arith.divui", "arith.divsi")
                and s.operand_ids
                and any(
                    op_by_id.get(o) is not None and op_by_id.get(o).op == "tt.get_program_id" for o in s.operand_ids
                )
            ):
                for o in s.operand_ids:
                    a = arg_by_id.get(o)
                    if a is not None and not a.is_ptr:
                        h_arg = a.index
                        break

        # shared-across-i bias/pointers: 3-D 1-level chain -> h-slot 0.
        if grid_3d:
            for _res in (q_res, k_res, v_res, b_res, do_res, dk_res, dv_res):
                if _res[1][1] == C1:
                    _res[1][1] = "c0"

        a_shape = _extract_shape(self._find_op_type_str(scores_dot.operand_ids[0]))
        if not a_shape or len(a_shape) != 2:
            _refuse("the QK dot A shape")
        block_j, head_dim = a_shape
        qk_out = _extract_shape(scores_dot.type_str or "")
        block_k = qk_out[1] if (qk_out and len(qk_out) == 2) else block_j
        out_dtype = (
            "f16"
            if str(self.graph.args[dk_res[0]].elem_type) in ("fp16", "f16")
            else ("bf16" if str(self.graph.args[dk_res[0]].elem_type) in ("bf16", "bfloat16") else "f32")
        )

        sentinel_ref = _prove(n_ctx_arg, block_j, block_k)  # packet 136 whole-path proof
        _ds = _prove_downstream(
            "kv",
            n_ctx_arg,
            block_j,
            block_k,
            head_dim,
            do_addr,
            _v_addr_kv,
            _dlt_addr_kv,
            {"dk": _dk_store, "dv": _dv_store},
            scale_arg=scale_from_dk,
        )
        roles = [
            q_res[0],
            k_res[0],
            v_res[0],
            b_res[0],
            m_res[0],
            lse_res[0],
            dlt_res[0],
            do_res[0],
            dk_res[0],
            dv_res[0],
        ]
        if len(set(roles)) != 10:
            _refuse("ten distinct pointer roles")

        if not grid_3d:
            raise MetalNonRecoverableError(_BWD_2D_MSG)
        return {
            "bwd_kind": "kv",
            "grid_3d": grid_3d,
            "q": q_res,
            "k": k_res,
            "v": v_res,
            "bias": b_res,
            "mask": m_res,
            "lse": lse_res,
            "delta": dlt_res,
            "do": do_res,
            "dk": dk_res,
            "dv": dv_res,
            "sentinel": sentinel_ref,
            "scale_arg": scale_from_dk,
            "N_CTX": n_ctx_arg,
            "H": h_arg,
            "block_j": block_j,
            "block_k": block_k,
            "head_dim": head_dim,
            "out_dtype": out_dtype,
            "k_orient": _ds["k_orient"],
            "v_orient": _ds["v_orient"],
            "rounding": _ds.get("rounding") or {},  # packet 174
        }

    def _witness_biased_fa_backward(self, info: dict, kind: str) -> dict:
        """Prove every source address of a backward kernel is the template's, and bind the strides.

        The detector reads each stride by its position in the address chain. That fixes neither
        the index a stride multiplies nor the program id a batch stride follows nor the loop
        advance, and the template replays its own coordinates regardless. ``_address_witness``
        evaluates the source addresses exactly and compares them with the template's formula
        below; the returned stride lists are bound by value, so a respelled address is bound to
        the strides it uses.

        Template coordinates (``_msl_templates.make_flash_attention_bwd_*``): slots are the
        stride lists' positions. kv / q: [0] follows program id 2 (the mask: program id 2 over
        H), [1] program id 1, then the tile axes; the grid tile is program id 0 and the other
        axis is the loop. b: [0] follows program id 2, [1] the i loop, then j and k from program
        ids 0 and 1.
        """
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

        C1 = "c1"
        block_j, block_k, head_dim = int(info["block_j"]), int(info["block_k"]), int(info["head_dim"])
        n_ctx, h_div = info["N_CTX"], info["H"]
        if n_ctx != C1 and type(n_ctx) is not int:
            raise MetalNonRecoverableError("biased-FA backward: the sequence length is not an argument; refusing.")
        if h_div != C1 and type(h_div) is not int:
            raise MetalNonRecoverableError("biased-FA backward: the mask batch divisor is not an argument; refusing.")
        iteration, looped = ("t",), n_ctx != C1

        def _tile(atom, block):
            return lambda env: [env.atom(atom) * block + i for i in range(block)]

        if kind == "kv":
            coords = {"k": (_tile(("pid", 0), block_k), block_k), "j": (_tile(iteration, block_j), block_j)}
            loop_step = block_j
        elif kind == "q":
            coords = {"j": (_tile(("pid", 0), block_j), block_j), "k": (_tile(iteration, block_k), block_k)}
            loop_step = block_k
        else:
            coords = {"j": (_tile(("pid", 0), block_j), block_j), "k": (_tile(("pid", 1), block_k), block_k)}
            loop_step = 1
        coords["d"] = (lambda env: list(range(head_dim)), head_dim)
        mask_batch = ("pid", 2) if h_div == C1 else ("div", 2, ("arg", h_div))
        if kind == "b":
            inner = [(1, iteration)] if looped else []
            batch, mask_scalars, plane = [(0, ("pid", 2))] + inner, [(0, mask_batch)] + inner, [(0, ("pid", 2))]
        else:
            batch = plane = [(0, ("pid", 2)), (1, ("pid", 1))]
            mask_scalars = [(0, mask_batch), (1, ("pid", 1))]

        def _role(key, name, access, scalars, layout):
            if any(axis not in coords for axis in layout):
                raise MetalNonRecoverableError(f"biased-FA backward: unproved {name} orientation; refusing.")
            return {
                "key": key,
                "name": name,
                "arg": info[key][0],
                "kind": access,
                "strides": list(info[key][1]),
                "shape": tuple(coords[axis][1] for axis in layout),
                "scalars": scalars,
                "axes": [(2 + i, i, coords[axis][0]) for i, axis in enumerate(layout)],
            }

        roles = [
            _role("q", "Q", "load", batch, ("j", "d")),
            _role("k", "K", "load", batch, tuple(info.get("k_orient") or ())),
            _role("v", "V", "load", batch, tuple(info.get("v_orient") or ())),
            _role("do", "dO", "load", batch, ("j", "d")),
            _role("bias", "Bias", "load", plane, ("j", "k")),
            _role("mask", "Mask", "load", mask_scalars, ("k",)),
            _role("lse", "Lse", "load", batch, ("j",)),
        ]
        if kind == "kv":
            roles += [
                _role("delta", "Delta", "load", batch, ("j",)),
                _role("dk", "DK", "store", batch, ("k", "d")),
                _role("dv", "DV", "store", batch, ("k", "d")),
            ]
        elif kind == "q":
            roles += [
                _role("o", "O", "load", batch, ("j", "d")),
                _role("delta", "Delta", "store", batch, ("j",)),
                _role("dq", "DQ", "store", batch, ("j", "d")),
            ]
        else:
            roles += [
                _role("delta", "Delta", "load", batch, ("j",)),
                _role("db", "DB", "store", plane, ("j", "k")),
            ]
        try:
            bound = prove_template_addresses(self.graph, roles, (n_ctx, loop_step) if looped else None)
        except WitnessRefusal as exc:
            raise MetalNonRecoverableError(
                f"biased-FA backward: {exc}. The template replays its own coordinates and cannot "
                "prove they are the source's; refusing."
            ) from None
        proved = dict(info)
        for role in roles:
            proved[role["key"]] = (info[role["key"]][0], bound[role["name"]])
        return proved

    def _lower_biased_fa_backward(self, info: dict) -> str:
        """Emit the biased-FA backward MSL for a detected backward kernel + set the
        compile_shader dispatch descriptor. Mirrors the forward biased lowering's
        #7-packed ABI (pointers individual, scalars in one packed buffer when >31
        args). Routes bwd_kind 'kv' (dK/dV) and 'q' (dQ + delta).

        Orientation notes (resolver strides are [z, h, row, col] from the ADDRESS):
          - kv: K and V are loaded TRANSPOSED [d,k] -> template k_sn/v_sn (kv) come
            from the resolver COL and k_sk/v_sk (head_dim) from the resolver ROW.
          - q:  K is loaded UNtransposed [k,d] -> k_sn/k_sk map DIRECT (row/col);
            V is still transposed [d,k] -> v_sn/v_sk SWAP. Delta here is an OUTPUT
            (rowsum(O*dO)); sm_scale is folded into K, not a final store-scale.
          - b:  triangle-i is a LOOP (grid = j,k,h); K and V both UNtransposed ->
            direct. Each stride list's slot [1] is the i-loop ADVANCE stride (baked
            by the detector), which the template consumes as q_si/k_si/v_si/....
        """
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen.msl_types import triton_type_to_msl
        from triton_msl.codegen._msl_templates import (
            make_flash_attention_bwd_kv_kernel,
            make_flash_attention_bwd_q_kernel,
            make_flash_attention_bwd_b_kernel,
            make_flash_attention_bwd_kv_kernel_simd,
            make_flash_attention_bwd_q_kernel_simd,
        )

        kind = info.get("bwd_kind")
        if kind not in ("kv", "q", "b"):
            raise MetalNonRecoverableError("only biased-FA backward dK/dV, dQ, dbias are routed so far")
        output_roles = {"kv": ("dk", "dv"), "q": ("dq", "delta"), "b": ("db",)}[kind]
        output_casts = (info.get("rounding") or {}).get("output_casts")
        if not isinstance(output_casts, dict) or set(output_casts) != set(output_roles):
            raise MetalNonRecoverableError("biased-FA backward requires a proved cast for every output role")
        storage_cast = {"f32": None, "fp32": None, "f16": "half", "fp16": "half", "bf16": "bfloat"}
        for role in output_roles:
            dtype = str(self.graph.args[info[role][0]].elem_type)
            if dtype not in storage_cast or output_casts[role] != storage_cast[dtype]:
                raise MetalNonRecoverableError(f"biased-FA backward {role}: cast does not match its own output pointer")
        MAX_BUFFERS = 31
        C1 = "c1"
        args = self.graph.args
        for i, a in enumerate(args):
            if i != a.index:
                raise MetalNonRecoverableError("biased-FA backward arg list not densely indexed; refusing.")
        info = self._witness_biased_fa_backward(info, kind)

        if kind == "kv":
            role_name = {
                info["dk"][0]: "DK",
                info["dv"][0]: "DV",
                info["q"][0]: "Q",
                info["k"][0]: "K",
                info["v"][0]: "V",
                info["bias"][0]: "Bias",
                info["mask"][0]: "Mask",
                info["lse"][0]: "Lse",
                info["delta"][0]: "Delta",
                info["do"][0]: "dO",
            }
            write_roles = ("DK", "DV")
        elif kind == "q":
            role_name = {
                info["dq"][0]: "DQ",
                info["delta"][0]: "Delta",
                info["q"][0]: "Q",
                info["k"][0]: "K",
                info["v"][0]: "V",
                info["bias"][0]: "Bias",
                info["mask"][0]: "Mask",
                info["lse"][0]: "Lse",
                info["o"][0]: "O",
                info["do"][0]: "dO",
            }
            write_roles = ("DQ", "Delta")
        else:  # b
            role_name = {
                info["db"][0]: "DB",
                info["q"][0]: "Q",
                info["k"][0]: "K",
                info["v"][0]: "V",
                info["bias"][0]: "Bias",
                info["mask"][0]: "Mask",
                info["lse"][0]: "Lse",
                info["delta"][0]: "Delta",
                info["do"][0]: "dO",
            }
            write_roles = ("DB",)
        n_args = len(args)
        n_ptr = sum(1 for a in args if a.is_ptr)
        packed = n_args > MAX_BUFFERS
        arg_decls = []
        scalar_pos = {}

        def _ptr_decl(a, buf):
            role = role_name[a.index]
            if role == "Mask":
                return f"    device const uchar* Mask [[buffer({buf})]]"
            m = triton_type_to_msl(a.elem_type)
            qual = "device" if role in write_roles else "device const"
            return f"    {qual} {m}* {role} [[buffer({buf})]]"

        if packed:
            ptr_ord = j = 0
            for a in args:
                if a.is_ptr:
                    arg_decls.append(_ptr_decl(a, ptr_ord))
                    ptr_ord += 1
                else:
                    scalar_pos[a.index] = j
                    j += 1
            arg_decls.append(f"    constant uint* _bpk [[buffer({n_ptr})]]")
        else:
            for a in args:
                if a.is_ptr:
                    arg_decls.append(_ptr_decl(a, a.index))
                else:
                    arg_decls.append(f"    constant uint& bsc_{a.index} [[buffer({a.index})]]")
                    scalar_pos[a.index] = a.index

        unsigned_args = set()

        def _u(entry):
            if entry == C1:
                return "1u"
            if entry == "c0":
                return "0u"
            if isinstance(entry, int) and entry in scalar_pos:
                unsigned_args.add(entry)
                return f"_bpk[{scalar_pos[entry]}]" if packed else f"bsc_{entry}"
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and entry[0] == "literal"
                and type(entry[1]) is int
                and 0 <= entry[1] <= 0xFFFFFFFF
            ):
                return f"{entry[1]}u"
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "mul":
                return f"({_u(entry[1])} * {_u(entry[2])})"
            raise MetalNonRecoverableError("biased-FA backward stride unmapped; refusing.")

        def _scale(entry):
            if isinstance(entry, int) and entry in scalar_pos:
                return f"as_type<float>(_bpk[{scalar_pos[entry]}])" if packed else f"as_type<float>(bsc_{entry})"
            raise MetalNonRecoverableError("biased-FA backward scale unmapped; refusing.")

        qs, ks, vs = info["q"][1], info["k"][1], info["v"][1]
        os_, bs, ms, ls = info["do"][1], info["bias"][1], info["mask"][1], info["lse"][1]

        # packet 148: the template addresses K / V as [k (sn), d (sk)]; the source may hold
        # either orientation, so the resolver's [row, col] strides map by the PROVED load
        # orientation rather than by kind: (d, k) -> sn = col, sk = row; (k, d) -> direct.
        def _sn_sk(res, orient, what):
            if orient == ("d", "k"):
                return res[1][3], res[1][2]
            if orient == ("k", "d"):
                return res[1][2], res[1][3]
            raise MetalNonRecoverableError(f"biased-FA backward: unproved {what} orientation {orient!r}; refusing.")

        _k_sn, _k_sk = _sn_sk(info["k"], info.get("k_orient"), "K")
        _v_sn, _v_sk = _sn_sk(info["v"], info.get("v_orient"), "V")
        ds = info["delta"][1]
        grid_3d = bool(info.get("grid_3d"))
        mask_batch_div = _u(info["H"]) if (grid_3d and info["H"] != C1) else None
        head_dim, block_j, block_k = info["head_dim"], info["block_j"], info["block_k"]
        common = {
            "q_sz": _u(qs[0]),
            "q_sh": _u(qs[1]),
            "q_sm": _u(qs[2]),
            "q_sk": _u(qs[3]),
            "v_sz": _u(vs[0]),
            "v_sh": _u(vs[1]),
            "v_sn": _u(_v_sn),
            "v_sk": _u(_v_sk),  # from the proven V orientation
            "b_sz": _u(bs[0]),
            "b_sh": _u(bs[1]),
            "b_sm": _u(bs[2]),
            "b_sn": _u(bs[3]),
            "mask_sz": _u(ms[0]),
            "mask_sh": _u(ms[1]),
            "mask_sn": _u(ms[2]),
            "lse_sz": _u(ls[0]),
            "lse_sh": _u(ls[1]),
            "lse_sm": _u(ls[2]),
            "dlt_sz": _u(ds[0]),
            "dlt_sh": _u(ds[1]),
            "dlt_sm": _u(ds[2]),
            "do_sz": _u(os_[0]),
            "do_sh": _u(os_[1]),
            "do_sm": _u(os_[2]),
            "do_sk": _u(os_[3]),
            "H": _u(info["H"]),
            "N_CTX": _u(info["N_CTX"]),
            "scale": _scale(info["scale_arg"]),
        }
        # packet 134: forward the proven score sentinel (natural units in the backward: the
        # source applies it BEFORE the inv_ln2 multiply) — runtime arg or baked literal.
        _snt = info.get("sentinel")
        if not _snt:
            raise MetalNonRecoverableError("biased-FA backward: the score sentinel was not proven; refusing.")
        if _snt[0] == "arg":
            common["neg_inf"] = _scale(_snt[1])
        else:
            common["neg_inf"] = "(-INFINITY)" if _snt[1] == float("-inf") else f"({float(_snt[1])!r}f)"

        # simdgroup-MMA fast path (dK/dV, dQ): ~2-2.5x the scalar template at the tile
        # sizes it supports. dbias stays scalar (memory-bound i-loop; MMA lost).
        simd_ok = (
            head_dim % 8 == 0
            and block_j % 8 == 0
            and block_k % 8 == 0
            and head_dim <= 32
            and block_j <= 32
            and block_k <= 32
        )
        if kind == "kv":
            dks, dvs = info["dk"][1], info["dv"][1]
            bindings = dict(common)
            bindings.update(
                {
                    "k_sz": _u(ks[0]),
                    "k_sh": _u(ks[1]),
                    "k_sn": _u(_k_sn),
                    "k_sk": _u(_k_sk),  # from the proven K orientation
                    "dk_sz": _u(dks[0]),
                    "dk_sh": _u(dks[1]),
                    "dk_sn": _u(dks[2]),
                    "dk_sk": _u(dks[3]),
                    "dv_sz": _u(dvs[0]),
                    "dv_sh": _u(dvs[1]),
                    "dv_sn": _u(dvs[2]),
                    "dv_sk": _u(dvs[3]),
                }
            )
            # Error-type parity with the q/b branches: refuse LOUDLY (MetalNonRecoverableError
            # -> CPU fallback) if the scalar path can't tile this head_dim within 32 KB
            # (e.g. head_dim=256), instead of letting the maker raise a bare ValueError.
            if not simd_ok:
                from triton_msl.codegen._msl_templates import _bwd_kv_subtile_size

                if _bwd_kv_subtile_size(head_dim, block_j, block_k) is None:
                    raise MetalNonRecoverableError(
                        f"Refusing to emit silently-wrong output: backward dK/dV head_dim={head_dim} "
                        f"(BLOCK_J={block_j}, BLOCK_K={block_k}) cannot be k-subtiled within Metal's "
                        f"1024-thread / 32 KB threadgroup limits."
                    )
            kv_maker = make_flash_attention_bwd_kv_kernel_simd if simd_ok else make_flash_attention_bwd_kv_kernel
            msl = kv_maker(
                head_dim,
                block_j,
                block_k,
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                grid_3d=grid_3d,
                mask_batch_div=mask_batch_div,
                runtime_neg_inf=True,
                **(info.get("rounding") or {}),
            )
            # scalar path k-subtiles so a large head_dim fits: TPG = KS*head_dim (KS==block_k
            # for head_dim<=32 -> unchanged block_k*head_dim). Must match the template's KS.
            if simd_ok:
                self.effective_block_size = 128
            else:
                from triton_msl.codegen._msl_templates import _bwd_kv_subtile_size

                _ks = _bwd_kv_subtile_size(head_dim, block_j, block_k)
                self.effective_block_size = (_ks * head_dim) if _ks is not None else block_k * head_dim
        elif kind == "q":
            oo, dqs = info["o"][1], info["dq"][1]
            bindings = dict(common)
            bindings.update(
                {
                    "k_sz": _u(ks[0]),
                    "k_sh": _u(ks[1]),
                    "k_sn": _u(_k_sn),
                    "k_sk": _u(_k_sk),  # from the proven K orientation
                    "o_sz": _u(oo[0]),
                    "o_sh": _u(oo[1]),
                    "o_sm": _u(oo[2]),
                    "o_sk": _u(oo[3]),
                    "dq_sz": _u(dqs[0]),
                    "dq_sh": _u(dqs[1]),
                    "dq_sm": _u(dqs[2]),
                    "dq_sk": _u(dqs[3]),
                }
            )
            _q_rounding = info.get("rounding") or {}
            if _q_rounding.get("round_delta") is not None and not simd_ok:
                raise MetalNonRecoverableError(
                    "Refusing biased-FA backward narrow delta reduction outside "
                    "the proved D=32 simdgroup layout; the scalar template's "
                    "sequential sum has a different association."
                )
            q_maker = make_flash_attention_bwd_q_kernel_simd if simd_ok else make_flash_attention_bwd_q_kernel
            # scalar path j-subtiles so a large head_dim fits: TPG = JS*head_dim (JS==block_j
            # for head_dim<=32 -> unchanged). The J-independent K/V staging (2*BK*D) can make
            # dQ unfittable at head_dim>=128 (JS is None) -> refuse LOUDLY (dK/dV still lower).
            if simd_ok:
                self.effective_block_size = 128
            else:
                from triton_msl.codegen._msl_templates import _bwd_q_subtile_size

                _js = _bwd_q_subtile_size(head_dim, block_j, block_k)
                if _js is None:
                    raise MetalNonRecoverableError(
                        f"Refusing to emit silently-wrong output: backward dQ head_dim={head_dim} "
                        f"(BLOCK_J={block_j}, BLOCK_K={block_k}) — the J-independent K/V threadgroup "
                        f"staging (2*BLOCK_K*head_dim) exceeds 32 KB at any j-subtile, so dQ has no "
                        f"correct lowering here (dK/dV still lower via the K-subtiled bwd_kv)."
                    )
                self.effective_block_size = _js * head_dim
            msl = q_maker(
                head_dim,
                block_j,
                block_k,
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                grid_3d=grid_3d,
                mask_batch_div=mask_batch_div,
                runtime_neg_inf=True,
                **(info.get("rounding") or {}),
            )
        else:  # b — triangle-i is a loop; slot [1] of each stride list is the i-stride.
            dbs = info["db"][1]
            bindings = {
                "q_sh": _u(qs[0]),
                "q_si": _u(qs[1]),
                "q_sm": _u(qs[2]),
                "q_sk": _u(qs[3]),
                "k_sh": _u(ks[0]),
                "k_si": _u(ks[1]),
                "k_sn": _u(_k_sn),
                "k_sk": _u(_k_sk),  # from the proven orientations
                "v_sh": _u(vs[0]),
                "v_si": _u(vs[1]),
                "v_sn": _u(_v_sn),
                "v_sk": _u(_v_sk),
                "b_sh": _u(bs[0]),
                "b_sm": _u(bs[2]),
                "b_sn": _u(bs[3]),
                "mask_sz": _u(ms[0]),
                "mask_si": _u(ms[1]),
                "mask_sn": _u(ms[2]),
                "lse_sh": _u(ls[0]),
                "lse_si": _u(ls[1]),
                "lse_sm": _u(ls[2]),
                "dlt_sh": _u(ds[0]),
                "dlt_si": _u(ds[1]),
                "dlt_sm": _u(ds[2]),
                "do_sh": _u(os_[0]),
                "do_si": _u(os_[1]),
                "do_sm": _u(os_[2]),
                "do_sk": _u(os_[3]),
                "db_sh": _u(dbs[0]),
                "db_sm": _u(dbs[2]),
                "db_sn": _u(dbs[3]),
                "H": _u(info["H"]),
                "N_CTX": _u(info["N_CTX"]),
                "scale": _scale(info["scale_arg"]),
                "neg_inf": common["neg_inf"],  # packet 134: the proven score sentinel
            }
            # dbias de-stages Q for a large head_dim; if even that overflows 32 KB
            # (head_dim>=128 at BLOCK_K=32) it has no correct lowering here -> refuse LOUDLY.
            from triton_msl.codegen._msl_templates import _bwd_b_config

            _bfits = _bwd_b_config(head_dim, block_j, block_k)[1]
            if not _bfits:
                raise MetalNonRecoverableError(
                    f"Refusing to emit silently-wrong output: backward dbias head_dim={head_dim} "
                    f"(BLOCK_J={block_j}, BLOCK_K={block_k}) — the J-independent K/V threadgroup "
                    f"staging exceeds 32 KB even with Q de-staged, so dbias has no correct lowering."
                )
            msl = make_flash_attention_bwd_b_kernel(
                head_dim,
                block_j,
                block_k,
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                mask_batch_div=mask_batch_div,
                runtime_neg_inf=True,
                **(info.get("rounding") or {}),
            )
            self.effective_block_size = block_j * block_k
        self._flash_attention = ("flash_attention", msl, self.effective_block_size)
        from triton_msl.backend._argument_abi import producer_record

        self._argument_abi = producer_record(args, "biased-backward-v1", msl)
        self._attention_unsigned_scalars(unsigned_args)
        self._used_pid_axes = {0, 1, 2} if grid_3d else {0, 1}
        self._prescan_stores()
        return msl

    def _attention_unsigned_scalars(self, unsigned, signed=()) -> None:
        """Carry the arguments an attention template reads as 32-bit values to the launcher.

        Every attention template binds its strides, head count and lengths as ``uint`` and
        adds them in 32-bit arithmetic. The source reads the same arguments as signed
        integers of their declared width, so a negative stride (a pointer at the end of a
        view), a negative length or a 64-bit value gives the template another cell or
        another loop count. ``signed`` names the arguments a template reinterprets as
        ``int``: any 32-bit value is the source's. The launcher refuses the other values
        (``attention_unsigned_scalars_v1``); a route's own launch contract rides inside.
        """
        unsigned = {int(i) for i in unsigned}
        signed = {int(i) for i in signed} - unsigned
        self._tail_access_bounds = (
            "attention_unsigned_scalars_v1",
            tuple(sorted(unsigned)),
            tuple(sorted(signed)),
            self._tail_access_bounds,
        )

    def _fa_unmasked_store_rows(self, row_bound, block_m) -> None:
        """Carry an unmasked FlashAttention store's row domain to the launcher.

        Every FlashAttention template stores only the rows below the sequence length. A
        source store masked by that row boundary is exactly that set. One WITHOUT the row
        clip (no mask, or a mask that is trivial on the columns alone such as ``offs_d <
        HEAD_DIM``) stores every row of every launched tile, which is the same set only when
        no launched tile crosses the length, and that is known only from the runtime length
        and grid. The launcher refuses the launch that would lose rows
        (``fa_whole_row_tiles_v1``).
        """
        from triton_msl.errors import MetalNonRecoverableError

        def _walk(ops):
            for op in ops:
                yield op
                yield from _walk(op.region_ops or [])
                yield from _walk(op.else_ops or [])

        clipped = set()
        self._template_output_mask_nontrivial(is_fa=True, fa_ctx_index=row_bound, row_clipped=clipped)
        if all(op.id in clipped for op in _walk(self.graph.ops) if op.op == "tt.store"):
            return
        if isinstance(row_bound, bool) or not isinstance(row_bound, int):
            raise MetalNonRecoverableError(
                "FlashAttention with an unmasked store and no runtime sequence-length argument: the "
                "template stores only the rows below the sequence length and cannot prove the launch "
                "stores whole tiles. Mask the store with the row boundary. Refusing.",
                op_name="tt.store",
            )
        self._tail_access_bounds = ("fa_whole_row_tiles_v1", int(row_bound), int(block_m))

    def _fa_unmasked_value_keys(self, key_bound, block_n) -> None:
        """Carry an unmasked FlashAttention V load's key domain to the launcher.

        The biased template supplies zero for the V rows past the sequence length. A source
        that loads V without a mask reads what the tensor holds there, and those rows carry
        weight whenever the sentinel leaves them any (a finite sentinel on a fully masked
        row, a sentinel that is not very negative) or are not finite. The two agree only
        when the last key tile holds no row past the length, which is known only from the
        runtime length. The launcher refuses the other launches (``fa_whole_key_tiles_v1``);
        an unmasked store's row contract rides inside.
        """
        from triton_msl.errors import MetalNonRecoverableError

        if isinstance(key_bound, bool) or not isinstance(key_bound, int):
            # A specialized length is refused earlier, at the loop bound; this guards the
            # launch contract below, which needs the argument's index.
            raise MetalNonRecoverableError(
                "FlashAttention with an unmasked V load and no runtime sequence-length argument: the "
                "template supplies zero past the sequence length and cannot prove the launch attends "
                "whole key tiles. Mask the V load with the key boundary. Refusing.",
                op_name="tt.load",
            )
        self._tail_access_bounds = (
            "fa_whole_key_tiles_v1",
            int(key_bound),
            int(block_n),
            self._tail_access_bounds,
        )

    def _prove_biased_fa_addresses(self, info: dict) -> dict:
        """Tie every address the biased/triangle template reconstructs to the source.

        The detector only EXTRACTS strides, by their position in the address
        expression. It never looks at the index a stride multiplies, the batch
        coordinate, or what a loop-carried pointer advances by, so before this proof
        ``(m_idxs + 1)[:, None] * q_sm``, ``(z + 1) * q_sz``, ``kt_ptrs += 2 * BLOCK_N *
        k_sn`` and the reordered but equivalent ``d_idxs[None, :] * q_sk + m_idxs[:,
        None] * q_sm`` all ran the template: the first three as another program, the
        last with its strides bound to the wrong axes.

        Each access must be exactly ``base + batch terms + one term per tile axis``:

        * rows   ``program_id(0) * BLOCK_M + arange(0, BLOCK_M)``
        * keys   ``start_n + arange(0, BLOCK_N)``, or ``arange(0, BLOCK_N)`` on a pointer
          the loop carries and advances by exactly ``BLOCK_N * <that term's stride>``
        * depth  ``arange(0, HEAD_DIM)``
        * batch  the template's own decode of the grid (``program_id(1) // H`` and
          ``% H``; ``program_id(2)`` and ``program_id(1)`` on a 3-D grid)

        each multiplied by a scalar argument or by nothing. The strides returned are
        read from the proved terms, in the detector's slot order, so a respelled
        address binds the stride of the axis it varies along and not the stride of
        the term the source happened to add first.
        """
        from triton_msl.errors import MetalNonRecoverableError

        C0, C1 = "c0", "c1"
        casts = self._IDX_CASTS
        layout = {"tt.broadcast", "ttg.convert_layout"}
        grid_3d = bool(info.get("grid_3d"))
        block_m, block_n, head_dim = int(info["block_m"]), int(info["block_n"]), int(info["head_dim"])
        has_mask = bool(info.get("has_mask", info.get("mask") is not None))

        def _refuse(role, why):
            raise MetalNonRecoverableError(
                f"Biased/triangle FlashAttention {role} address is not exactly the template's ({why}). "
                "The template reconstructs every address from the tile coordinates, so a source "
                "address it cannot reproduce would be replaced by a different one. Refusing.",
                op_name="tt.addptr",
            )

        def _walk(ops):
            for o in ops:
                yield o
                if getattr(o, "region_ops", None):
                    yield from _walk(o.region_ops)
                if getattr(o, "else_ops", None):
                    yield from _walk(o.else_ops)

        ops = list(_walk(self.graph.ops))
        by_id = {}
        for o in ops:
            by_id[o.id] = o
            for rid in getattr(o, "result_ids", None) or []:
                by_id[rid] = o
        arg_by_id = {a.id: a for a in self.graph.args}

        def _scalar_arg(vid):
            a = arg_by_id.get(self._strip_wrappers(vid, by_id, casts))
            return a.index if a is not None and not a.is_ptr else None

        def _pid_axis(vid):
            o = by_id.get(self._strip_wrappers(vid, by_id, casts))
            if o is None or o.op not in self._PID_OPS:
                return None
            try:
                return int(str(o.attrs.get("axis", o.attrs.get("dim", 0))))
            except (TypeError, ValueError):
                return None

        # ---- the one KV loop: range(0, N_CTX, BLOCK_N), and what each carried pointer advances by
        loops = [o for o in ops if o.op == "scf.for"]
        if len(loops) != 1:
            _refuse("K/V", f"expected the one key/value loop, found {len(loops)}")
        loop = loops[0]
        bids = list((loop.attrs or {}).get("block_arg_ids", []))
        lo_hi_step = list(loop.operand_ids or [])[:3]
        if len(lo_hi_step) != 3 or not bids:
            _refuse("K/V", "the key/value loop bounds could not be read")
        iv = bids[0]
        if self._const_int(self._strip_wrappers(lo_hi_step[0], by_id, casts), by_id) != 0:
            _refuse("K/V", "the key/value loop does not start at key 0")
        if self._const_int(self._strip_wrappers(lo_hi_step[2], by_id, casts), by_id) != block_n:
            _refuse("K/V", f"the key/value loop does not step by BLOCK_N={block_n}")
        if _scalar_arg(lo_hi_step[1]) != info.get("N_CTX"):
            _refuse("K/V", "the key/value loop does not end at the sequence-length argument")
        yields = [o for o in (loop.region_ops or []) if o.op == "scf.yield"]
        if len(yields) != 1:
            _refuse("K/V", "the key/value loop does not have exactly one yield")
        carried = {}
        for i, init in enumerate(list(loop.operand_ids or [])[3:]):
            if i + 1 < len(bids) and i < len(yields[0].operand_ids or []):
                carried[bids[i + 1]] = (init, yields[0].operand_ids[i])

        def _increment(role, bid):
            """The stride a carried pointer advances by BLOCK_N of each iteration."""
            nxt = by_id.get(carried[bid][1])
            if nxt is None or nxt.op != "tt.addptr" or len(nxt.operand_ids or []) != 2 or nxt.operand_ids[0] != bid:
                _refuse(role, "the loop-carried pointer is not advanced by one addition per iteration")
            inc = self._strip_wrappers(nxt.operand_ids[1], by_id, layout | {"tt.splat"} | casts)
            if self._const_int(inc, by_id) == block_n:
                return C1
            mul = by_id.get(inc)
            if mul is not None and mul.op in ("arith.muli", "arith.mul") and len(mul.operand_ids or []) == 2:
                for x, y in (mul.operand_ids, mul.operand_ids[::-1]):
                    stride = _scalar_arg(y)
                    if stride is not None and self._const_int(self._strip_wrappers(x, by_id, casts), by_id) == block_n:
                        return stride
            _refuse(role, f"the loop-carried pointer does not advance by BLOCK_N={block_n} keys")

        def _address(role, access, ptr_index):
            """(offsets, carried block arg or None) of an access rooted at ``ptr_index``."""
            offsets, bid, cur = [], None, access.operand_ids[0]
            for _ in range(64):
                arg = arg_by_id.get(cur)
                if arg is not None:
                    if not arg.is_ptr or arg.index != ptr_index:
                        _refuse(role, f"the address is rooted at argument '{arg.name}'")
                    return offsets, bid
                if cur in carried:
                    if bid is not None:
                        _refuse(role, "the address crosses more than one loop-carried pointer")
                    bid, cur = cur, carried[cur][0]
                    continue
                o = by_id.get(cur)
                if o is not None and o.op in (layout | {"tt.splat"}) and len(o.operand_ids or []) == 1:
                    cur = o.operand_ids[0]
                elif o is not None and o.op == "tt.addptr" and len(o.operand_ids or []) == 2:
                    offsets.append(o.operand_ids[1])
                    cur = o.operand_ids[0]
                else:
                    _refuse(role, f"'{o.op if o is not None else 'an unresolved value'}' in the pointer chain")
            _refuse(role, "the pointer chain exceeds the proved depth")

        def _has_lift(vid, depth=0):
            o = by_id.get(vid)
            if o is None or depth > 24:
                return o is not None
            return o.op == "tt.expand_dims" or any(_has_lift(x, depth + 1) for x in (o.operand_ids or []))

        def _tile_term(role, vid):
            """A tensor offset -> (matrix axis it varies along or None, stride, index facts)."""
            axis, stride, cur = None, None, vid
            for _ in range(16):
                cur = self._strip_wrappers(cur, by_id, layout | casts)
                o = by_id.get(cur)
                if o is None:
                    _refuse(role, "an unresolved tile offset")
                if o.op == "tt.expand_dims" and len(o.operand_ids or []) == 1:
                    if axis is not None:
                        _refuse(role, "a tile index is lifted more than once")
                    try:
                        # x[:, None] is expand_dims axis 1: the vector then varies along axis 0
                        axis = 1 - int(str(o.attrs.get("axis")))
                    except (TypeError, ValueError):
                        _refuse(role, "a tile index lift has no readable axis")
                    if axis not in (0, 1):
                        _refuse(role, "a tile index is lifted outside the 2-D tile")
                    cur = o.operand_ids[0]
                    continue
                if o.op in ("arith.muli", "arith.mul") and len(o.operand_ids or []) == 2 and stride is None:
                    found = None
                    for x, y in (o.operand_ids, o.operand_ids[::-1]):
                        sp = by_id.get(self._strip_wrappers(y, by_id, layout | casts))
                        if sp is not None and sp.op == "tt.splat" and len(sp.operand_ids or []) == 1:
                            arg = _scalar_arg(sp.operand_ids[0])
                            if arg is not None:
                                found = (arg, x)
                                break
                    if found is not None:
                        stride, cur = found
                        continue
                break
            if _has_lift(cur):
                _refuse(role, "a tile index is lifted more than once")
            sh = self._index_shape(cur, by_id, arg_by_id, iv, 1)
            if sh["bad"] is not None or sh["mod"] is not None or sh["range"] != 1 or sh["bounds"] is None:
                _refuse(role, f"a tile offset is not a plain tile index ({sh['bad'] or 'no single arange'})")
            return axis, (C1 if stride is None else stride), sh

        def _is_row(sh):
            return (
                sh["bounds"] == (0, block_m)
                and sh["iv"] == 0
                and sh["pid"] == 0
                and sh["pid_div"] is None
                and sh["coef"] == block_m
            )

        def _is_key(sh, carried_ptr):
            return sh["bounds"] == (0, block_n) and sh["pid"] is None and sh["iv"] == (0 if carried_ptr else 1)

        def _is_depth(sh):
            return sh["bounds"] == (0, head_dim) and sh["pid"] is None and sh["iv"] == 0

        h_arg = info.get("H")
        proved_mask_div = None

        def _batch_coordinate(role, vid):
            """A scalar batch coordinate -> its template slot (0 = z, 1 = h), else None."""
            nonlocal proved_mask_div
            axis = _pid_axis(vid)
            if grid_3d:
                # template: h = program_id(1), z = program_id(2)
                if axis == 2:
                    return 0
                if axis == 1:
                    return 1
            elif h_arg == C1:
                # template: z = program_id(1) / 1, h = 0
                if axis == 1:
                    return 0
            o = by_id.get(self._strip_wrappers(vid, by_id, casts))
            if o is None or len(o.operand_ids or []) != 2:
                return None
            if not grid_3d and h_arg != C1 and o.op == "arith.subi" and _pid_axis(o.operand_ids[0]) == 1:
                # h spelled as the remainder's definition: program_id(1) - (program_id(1) // H) * H
                mul = by_id.get(self._strip_wrappers(o.operand_ids[1], by_id, casts))
                if mul is not None and mul.op in ("arith.muli", "arith.mul") and len(mul.operand_ids or []) == 2:
                    for x, y in (mul.operand_ids, mul.operand_ids[::-1]):
                        div = by_id.get(self._strip_wrappers(x, by_id, casts))
                        if (
                            _scalar_arg(y) == h_arg
                            and div is not None
                            and div.op in ("arith.divsi", "arith.divui")
                            and len(div.operand_ids or []) == 2
                            and _pid_axis(div.operand_ids[0]) == 1
                            and _scalar_arg(div.operand_ids[1]) == h_arg
                        ):
                            return 1
                return None
            divisor = _scalar_arg(o.operand_ids[1])
            if divisor is None:
                return None
            if grid_3d:
                # only the shared mask is indexed by (z / divisor)
                if role == "Mask" and o.op in ("arith.divsi", "arith.divui") and _pid_axis(o.operand_ids[0]) == 2:
                    proved_mask_div = divisor
                    return 0
                return None
            if h_arg == C1 or divisor != h_arg or _pid_axis(o.operand_ids[0]) != 1:
                return None
            if o.op in ("arith.divsi", "arith.divui"):
                return 0
            if o.op in ("arith.remsi", "arith.remui"):
                return 1
            return None

        def _batch_term(role, vid):
            vid = self._strip_wrappers(vid, by_id, casts)
            if vid in arg_by_id:
                _refuse(role, f"a runtime offset '{arg_by_id[vid].name}'")
            o = by_id.get(vid)
            if o is not None and o.op in ("arith.muli", "arith.mul") and len(o.operand_ids or []) == 2:
                for x, y in (o.operand_ids, o.operand_ids[::-1]):
                    stride = _scalar_arg(y)
                    slot = _batch_coordinate(role, x) if stride is not None else None
                    if slot is not None:
                        return slot, stride
            slot = _batch_coordinate(role, vid)
            if slot is None:
                _refuse(role, "a batch offset is not the template's batch or head coordinate times a stride")
            return slot, C1

        # role -> (pointer arg, {matrix axis or None: index kind}, loop-carried allowed)
        layouts = {
            "Q": (info["q"], {0: "row", 1: "depth"}, False),
            "K": (info["k"], {0: "depth", 1: "key"}, True),
            "V": (info["v"], {0: "key", 1: "depth"}, True),
            "Out": (info["out"], {0: "row", 1: "depth"}, False),
            "Bias": (info["bias"], {0: "row", 1: "key"}, True),
            "Lse": (info["lse"], {None: "row"}, False),
        }
        if has_mask:
            layouts["Mask"] = (info["mask"], {None: "key"}, True)
        stores = {"Out", "Lse"}
        # 2-D with H specialized to one: h is identically 0, so the detector's 1 is kept
        absent = C1 if (not grid_3d and h_arg == C1) else C0

        def _rooted_at(access, limit=64):
            cur = access.operand_ids[0]
            for _ in range(limit):
                if cur in arg_by_id:
                    return arg_by_id[cur].index
                if cur in carried:
                    cur = carried[cur][0]
                    continue
                o = by_id.get(cur)
                if o is None or not o.operand_ids or o.op not in (layout | {"tt.splat", "tt.addptr"}):
                    return None
                cur = o.operand_ids[0]
            return None

        proved = {}
        for role, (ptr_index, axes, may_carry) in layouts.items():
            want = "tt.store" if role in stores else "tt.load"
            found = [o for o in ops if o.op == want and o.operand_ids and _rooted_at(o) == ptr_index]
            if len(found) != 1:
                _refuse(role, f"expected exactly one {want.split('.')[-1]}, found {len(found)}")
            offsets, bid = _address(role, found[0], ptr_index)
            if bid is not None and not may_carry:
                _refuse(role, "the pointer is carried through the key/value loop")
            batch, tile = {}, {}
            for off in offsets:
                o = by_id.get(off)
                if o is not None and o.is_tensor:
                    axis, stride, sh = _tile_term(role, off)
                    kind = axes.get(axis) if axis in axes else None
                    ok = (
                        (kind == "row" and _is_row(sh))
                        or (kind == "key" and _is_key(sh, bid is not None))
                        or (kind == "depth" and _is_depth(sh))
                    )
                    if not ok or axis in tile:
                        _refuse(role, "a tile offset is not this role's row, key or head-dim index on its own axis")
                    tile[axis] = stride
                else:
                    slot, stride = _batch_term(role, off)
                    if slot in batch:
                        _refuse(role, "a batch coordinate is added more than once")
                    batch[slot] = stride
            if set(tile) != set(axes):
                _refuse(role, "the address does not index every axis of the tile exactly once")
            if bid is not None:
                key_axis = next(a for a, kind in axes.items() if kind == "key")
                if _increment(role, bid) != tile[key_axis]:
                    _refuse(role, "the loop-carried pointer does not advance by BLOCK_N of its own key stride")
            proved[role] = [batch.get(0, C0), batch.get(1, absent)] + [tile[a] for a in (0, 1, None) if a in tile]

        out = {
            "q_strides": proved["Q"],
            "k_strides": proved["K"],
            "v_strides": proved["V"],
            "o_strides": proved["Out"],
            "b_strides": proved["Bias"],
            "lse_strides": proved["Lse"],
        }
        if has_mask:
            out["m_strides"] = proved["Mask"]
            if grid_3d:
                out["mask_div"] = proved_mask_div
        return out

    def _lower_biased_flash_attention_template(self, info: dict) -> str:
        """Emit the biased/triangle-attention tiled FA MSL for a detected biased-FA
        kernel (``info`` from ``_detect_biased_flash_attention``) + set the
        compile_shader dispatch descriptor.

        ABI: pointers (Q,K,V,Out,Bias,Mask,Lse) keep individual buffers in their
        kernel-arg order; when the kernel has > 31 runtime args, ALL scalars pack
        into ONE ``constant uint*`` buffer AFTER the pointers, matching
        ``driver._pack_overflow_scalars`` (scalars in arg order; ints by value,
        the float scale by bit pattern -> ``as_type<float>``). The Kᵀ stride swap
        (template k_sk=head-dim row, k_sn=kv col) is applied from the detector's
        resolver-order strides.
        """
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen.msl_types import triton_type_to_msl
        from triton_msl.codegen._msl_templates import (
            make_flash_attention_kernel_tiled,
            flash_attention_tiled_bytes,
            flash_attention_tiled_threads,
        )

        MAX_BUFFERS = 31  # mirrors driver._MAX_METAL_BUFFERS
        C1 = "c1"
        _LSE_ELEMS = {"fp32": "float", "f32": "float", "fp16": "half", "f16": "half", "bf16": "bfloat"}
        args = self.graph.args
        for i, a in enumerate(args):
            if i != a.index:
                raise MetalNonRecoverableError(
                    "biased FlashAttention arg list is not densely indexed; refusing "
                    "rather than risk a mis-bound buffer."
                )

        head_dim = info["head_dim"]
        block_m, block_n = info["block_m"], info["block_n"]

        # Anti-silent-wrong (parity with the standard hd128 path): the tiled biased
        # template computes the FULL output tile and gates writes only on the N_CTX
        # boundary, so a tighter WITHIN-tile output store mask would be SILENTLY DROPPED
        # (clobbering the masked-off rows). Refuse it here using the SAME shared triviality
        # check the standard path uses; fa_ctx_index = the resolved N_CTX arg so a
        # renamed-seqlen tile boundary (om < N / om < seqlen) is still recognized as
        # trivial and passes through, while a tighter/value mask refuses.
        if self._template_output_mask_nontrivial(is_fa=True, fa_ctx_index=info.get("N_CTX")):
            raise MetalNonRecoverableError(
                "Biased/triangle FlashAttention with a non-tile-boundary output store mask "
                "is not supported: the tiled biased template computes the FULL output tile "
                "and gates writes only on the N_CTX boundary, silently DROPPING any tighter "
                "store mask (e.g. tl.store(o, acc, mask=om < BOUND) with BOUND < N_CTX, or a "
                "value mask). Refusing to emit silently-wrong output. Use a full-tile store "
                "(the om < N_CTX tile boundary is honored automatically) or apply the "
                "partial-output mask in a separate elementwise kernel."
            )

        has_mask = bool(info.get("has_mask", info.get("mask") is not None))
        # Packet 112: every memory boundary the biased template reconstructs — Q/K/V loads,
        # the Bias load (row AND col boundary), the Mask load (col boundary), the Lse store
        # (row boundary) — must be plain or exactly the template's boundary; the Out store
        # is guarded above (_template_output_mask_nontrivial).
        self._fa_verify_value_paths(
            detected_q_index=info["q"],
            detected_k_index=info["k"],
            detected_v_index=info["v"],
            detected_out_index=info["out"],
            q_scale_arg=info["scale_arg"],
            scalar_rounding_replay=True,
            k_transposes=0,
            detected_n_ctx_index=info.get("N_CTX"),
            load_mask_policy="boundary",
            q_rows=block_m,
            kv_cols=block_n,
            extra_loads=[(info["bias"], "Bias", ("Q", "K"))] + ([(info["mask"], "Mask", ("K",))] if has_mask else []),
            # A masked store must carry exactly the row boundary. An unmasked one also
            # writes the rows past N_CTX, which the template does not: the launcher is
            # handed that condition below (_fa_unmasked_store_rows).
            extra_stores=[(info["out"], "Out", ("Q",), False), (info["lse"], "Lse", ("Q",), False)],
            # Packet 114: the VALUES — Bias as the dot accumulator, Mask as the select
            # predicate, both selects with ONE sentinel (reported in _bout), the Lse value.
            biased={
                "bias": info["bias"],
                "mask": (info["mask"] if has_mask else None),
                "lse": info["lse"],
                "c_eff": info.get("c_eff", 1.0),
            },
            biased_out=(_bout := {}),
        )
        # Every stride below is the one the PROVED address multiplies on that axis, not
        # the detector's positional guess.
        info = {**info, **self._prove_biased_fa_addresses(info)}
        self._fa_unmasked_store_rows(info.get("N_CTX"), block_m)
        if _bout.get("v_unmasked"):
            self._fa_unmasked_value_keys(info.get("N_CTX"), block_n)
        role_name = {
            info["q"]: "Q",
            info["k"]: "K",
            info["v"]: "V",
            info["out"]: "Out",
            info["bias"]: "Bias",
            info["lse"]: "Lse",
        }
        if has_mask:
            role_name[info["mask"]] = "Mask"

        # Partition args into pointers + scalars, matching the launcher's binding.
        #  - <= 31 args: NO packing; the launcher binds kargs[i] -> buffer(i)
        #    POSITIONALLY, so every arg (pointer or scalar) declares buffer(a.index).
        #  - > 31 args: driver._pack_overflow_scalars returns [pointers..., packed],
        #    so pointers take buffers 0..n_ptr-1 in arg order, the packed scalar
        #    buffer is n_ptr, and packed[j] is the j-th scalar in arg order.
        n_args = len(args)
        n_ptr = sum(1 for a in args if a.is_ptr)
        packed = n_args > MAX_BUFFERS
        arg_decls = []
        scalar_pos = {}  # scalar arg index -> j (packed slot); identity-ish when unpacked

        def _ptr_decl(a, buf):
            role = role_name.get(a.index)
            if role is None:
                raise MetalNonRecoverableError(
                    "biased FlashAttention has a pointer arg with no resolved role; refusing."
                )
            if role == "Mask":
                return f"    device const uchar* Mask [[buffer({buf})]]"
            if role == "Lse":
                # The store rounds to the tensor's own element type: a half tensor written
                # through ``float*`` took four bytes for every two it holds.
                if a.elem_type not in _LSE_ELEMS:
                    raise MetalNonRecoverableError(
                        "biased FlashAttention log-sum-exp tensor is not fp32, fp16 or bf16; refusing."
                    )
                return f"    device {_LSE_ELEMS[a.elem_type]}* Lse [[buffer({buf})]]"
            m = triton_type_to_msl(a.elem_type)
            qual = "device" if role == "Out" else "device const"
            return f"    {qual} {m}* {role} [[buffer({buf})]]"

        if packed:
            ptr_ord = 0
            j = 0
            for a in args:
                if a.is_ptr:
                    arg_decls.append(_ptr_decl(a, ptr_ord))
                    ptr_ord += 1
                else:
                    scalar_pos[a.index] = j
                    j += 1
            arg_decls.append(f"    constant uint* _bpk [[buffer({n_ptr})]]")
        else:
            for a in args:
                if a.is_ptr:
                    arg_decls.append(_ptr_decl(a, a.index))
                else:
                    arg_decls.append(f"    constant uint& bsc_{a.index} [[buffer({a.index})]]")
                    scalar_pos[a.index] = a.index

        unsigned_args = set()

        def _uint_expr(entry):
            """A stride/dim entry -> MSL uint expression (packed slot / individual
            buffer / baked 1u / baked 0u for a shared-across-i axis)."""
            if entry == C1:
                return "1u"
            if entry == "c0":
                return "0u"
            if isinstance(entry, int) and entry in scalar_pos:
                unsigned_args.add(entry)
                return f"_bpk[{scalar_pos[entry]}]" if packed else f"bsc_{entry}"
            raise MetalNonRecoverableError(
                "biased FlashAttention stride/dim could not be mapped to a kernel arg; refusing."
            )

        def _scale_expr(entry, role):
            """A runtime float -> reinterpret its packed/individual uint slot.

            The replacement ABI stores one uint-sized slot per scalar and this
            template decodes the slot with ``as_type<float>``.  Only a source
            fp32 value has that bit-exact representation.  In particular, a
            source fp16 scale/sentinel cannot be widened *after* its 16 bits
            have already been packed into the slot.
            """
            if isinstance(entry, int) and entry in scalar_pos:
                source_type = args[entry].elem_type
                if source_type not in ("f32", "fp32", "float"):
                    raise MetalNonRecoverableError(
                        f"biased FlashAttention runtime {role} arg {args[entry].name} "
                        f"must be fp32 for the replacement scalar ABI (got {source_type}); "
                        "refusing rather than reinterpret non-fp32 bits as float."
                    )
                return f"as_type<float>(_bpk[{scalar_pos[entry]}])" if packed else f"as_type<float>(bsc_{entry})"
            raise MetalNonRecoverableError(f"biased FlashAttention runtime {role} arg unmapped; refusing.")

        qs, ks, vs = info["q_strides"], info["k_strides"], info["v_strides"]
        os_, bs, ls = info["o_strides"], info["b_strides"], info["lse_strides"]
        # mask strides only exist when a mask is present; 0u placeholders otherwise (the
        # template's mask code is gated off, so they are never read).
        ms = info["m_strides"] if has_mask else None
        _msk = (lambda i: _uint_expr(ms[i])) if has_mask else (lambda i: "0u")
        bindings = {
            "q_sz": _uint_expr(qs[0]),
            "q_sh": _uint_expr(qs[1]),
            "q_sm": _uint_expr(qs[2]),
            "q_sk": _uint_expr(qs[3]),
            # Kᵀ: resolver row (ks[2]) = head-dim stride -> k_sk; col (ks[3]) = kv stride -> k_sn.
            "k_sz": _uint_expr(ks[0]),
            "k_sh": _uint_expr(ks[1]),
            "k_sn": _uint_expr(ks[3]),
            "k_sk": _uint_expr(ks[2]),
            "v_sz": _uint_expr(vs[0]),
            "v_sh": _uint_expr(vs[1]),
            "v_sn": _uint_expr(vs[2]),
            "v_sk": _uint_expr(vs[3]),
            "o_sz": _uint_expr(os_[0]),
            "o_sh": _uint_expr(os_[1]),
            "o_sm": _uint_expr(os_[2]),
            "o_sk": _uint_expr(os_[3]),
            "b_sz": _uint_expr(bs[0]),
            "b_sh": _uint_expr(bs[1]),
            "b_sm": _uint_expr(bs[2]),
            "b_sn": _uint_expr(bs[3]),
            "mask_sz": _msk(0),
            "mask_sh": _msk(1),
            "mask_sn": _msk(2),
            "lse_sz": _uint_expr(ls[0]),
            "lse_sh": _uint_expr(ls[1]),
            "lse_sm": _uint_expr(ls[2]),
            "Z": "1u",  # unused in the tiled template body (z = zh / H computed locally)
            "H": _uint_expr(info["H"]),
            "N_CTX": _uint_expr(info["N_CTX"]),
            "scale": _scale_expr(info["scale_arg"], "scale"),
        }
        # Packet 114: the source's sentinel, forwarded (runtime float arg, reinterpreted
        # from the packed scalar bits) or baked (finite literal); the template converts it
        # from the source's exp domain (after the c_eff multiply) to natural units.
        if "sentinel_arg" in _bout:
            bindings["neg_inf"] = _scale_expr(_bout["sentinel_arg"], "sentinel")
        elif "sentinel_const" in _bout:
            _sc = float(_bout["sentinel_const"])
            bindings["neg_inf"] = "(-INFINITY)" if _sc == float("-inf") else f"({_sc!r}f)"
        else:
            raise MetalNonRecoverableError("biased FlashAttention: the score sentinel was not proven; refusing.")
        _ni_unit = 1.0 / float(info.get("c_eff", 1.0))

        # 3-D grid (trifast): h = pid3.y (triangle-i), z = pid3.z (batch*heads). When
        # the mask's batch offset is divui(pid_h, H_heads) (cross-head-shared mask,
        # mask_start_h = pid_h // H_heads) the detector reports that divisor as
        # info["mask_div"]; the template then indexes the mask by (z / div). None
        # (per-head mask) -> no divisor. VERIFIED to feed the mask (not a stray divui).
        grid_3d = bool(info.get("grid_3d"))
        mask_batch_div = None
        if grid_3d and info.get("mask_div") is not None:
            mask_batch_div = _uint_expr(info["mask_div"])

        # FAST PATH: route to the simdgroup-MMA template (the shipped fast FA kernel,
        # ~10x the scalar tiled template) when eligible — head_dim %8==0, block_m==32,
        # head_dim<=192 (tg budget), and the CONTIGUOUS innermost strides the simd
        # transpose-loads require (head-dim stride == 1). Otherwise the scalar tiled
        # template handles arbitrary strides (correctness-first). Correct-or-refuse:
        # only take the fast path when provably contiguous; else the tiled fallback.
        qs2, ks2, vs2, os2 = info["q_strides"], info["k_strides"], info["v_strides"], info["o_strides"]
        # packet 164 (item F-a): the rounding points the value-path verifier recorded for THIS
        # kernel (bf16 / fp16 sources round q*scale and P; fp32 sources record nothing).
        _rq = getattr(self, "_fa_rounding", {}).get("round_q")
        _rp = getattr(self, "_fa_rounding", {}).get("round_p")
        _scale_chain = getattr(self, "_fa_rounding", {}).get("scale_chain", ())
        simd_eligible = (
            head_dim % 8 == 0
            # Valid v_head_dim tilings: <=64 (guarded surplus groups) OR the exact
            # multiples 128/192. 72..120 (non-64-multiple >64) mis-tile -> tiled path.
            and (8 <= head_dim <= 64 or head_dim in (128, 192))
            and info["out_dtype"] != "bf16"  # simd MMA fragments are fp16/fp32; bf16 -> tiled
            and block_m == 32
            and qs2[3] == C1  # Q head-dim (col) contiguous
            and ks2[2] == C1  # K head-dim (Kᵀ row) contiguous
            and vs2[3] == C1  # V head-dim (col) contiguous
            and os2[3] == C1  # Out head-dim (col) contiguous
            # packet 164: a source that rounds P to its dtype does so RELATIVE TO ITS OWN kv-block
            # running max; rounding then rescaling differs from rescaling then rounding, so the
            # replay must walk the source's block width. The simd maker is fixed at 64-wide
            # blocks: exact only when the source's kv block is 64. Otherwise the tiled maker
            # (which takes block_n from the source) replays it exactly — slower, correct.
            and (not _rp or block_n == 64)
            # A 64-wide tile lines up with the source's iterations when the source's key block
            # divides it or is a whole number of tiles; the template then tells the padding
            # keys the source visits from the ones it never reaches.
            and (64 % block_n == 0 or block_n % 64 == 0)
        )
        if simd_eligible:
            from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup

            msl = make_flash_attention_kernel_simdgroup(
                head_dim,
                32,
                64,
                causal=info["causal"],
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                bias=True,
                mask=has_mask,
                lse=True,
                lse_elem=_LSE_ELEMS[args[info["lse"]].elem_type],
                runtime_scale=True,
                grid_3d=grid_3d,
                mask_batch_div=mask_batch_div,
                runtime_neg_inf=True,
                neg_inf_unit=_ni_unit,
                round_q=_rq,
                round_p=_rp,
                scale_chain=_scale_chain,
                sentinel_kv_block=block_n,
            )
            self.effective_block_size = 256
        else:
            Dc = head_dim if head_dim <= 64 else 64
            # The tiled working set must fit Metal's 32 KB of threadgroup memory. Scores and
            # the accumulator are fixed by the source; only the staged head-dim slice narrows.
            while flash_attention_tiled_bytes(head_dim, block_m, block_n, Dc) > 32768 and Dc % 16 == 0:
                Dc //= 2
            if flash_attention_tiled_bytes(head_dim, block_m, block_n, Dc) > 32768:
                raise MetalNonRecoverableError(
                    f"biased FlashAttention: head_dim {head_dim} with a {block_m} x {block_n} block needs "
                    f"at least {flash_attention_tiled_bytes(head_dim, block_m, block_n, Dc)} bytes of threadgroup "
                    "memory on the tiled template, above Metal's 32768; refusing."
                )
            # Only the source-proved triangle family from the paired K-layout
            # screen opts in. Other tiled users (including dense attention) keep
            # the original layout. This changes no source admission or launch ABI.
            _transpose_k = (
                getattr(self.options, "experimental_tiled_k_layout", False) is True
                and (head_dim, block_m, block_n, Dc) == (64, 32, 32, 64)
                and info["out_dtype"] in ("fp16", "f16")
                and all(
                    args[info[role]].is_ptr and args[info[role]].elem_type in ("fp16", "f16")
                    for role in ("q", "k", "v", "out", "bias")
                )
                and _rq == "half"
                and _rp == "half"
                and not info["causal"]
                and grid_3d
                and has_mask
                and mask_batch_div is not None
            )
            msl = make_flash_attention_kernel_tiled(
                head_dim,
                block_m,
                block_n,
                Dc=Dc,
                causal=info["causal"],
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                bias=True,
                mask=has_mask,
                lse=True,
                lse_elem=_LSE_ELEMS[args[info["lse"]].elem_type],
                runtime_scale=True,
                grid_3d=grid_3d,
                mask_batch_div=mask_batch_div,
                runtime_neg_inf=True,
                neg_inf_unit=_ni_unit,
                round_q=_rq,
                round_p=_rp,
                scale_chain=_scale_chain,
                transpose_k=_transpose_k,
            )
            self.effective_block_size = flash_attention_tiled_threads(block_m, block_n)
        self._flash_attention = ("flash_attention", msl, self.effective_block_size)
        from triton_msl.backend._argument_abi import producer_record

        self._argument_abi = producer_record(args, "biased-forward-v1", msl)
        self._attention_unsigned_scalars(unsigned_args)
        self._used_pid_axes = {0, 1, 2} if grid_3d else {0, 1}
        self._prescan_stores()
        return msl

    def _witness_mla_attention(self, info: dict) -> dict:
        """Prove every source address of an MLA kernel is the dispatcher's view, and bind the strides.

        The dispatcher (``_dispatch_mla``) rebuilds each role as a ``[Z, H, N_CTX, width]``
        view from the role's four strides, with ``z = program id 1 / H`` and
        ``h = program id 1 % H``: the query roles and Out are addressed at
        ``z * sz + h * sh + (program id 0 * 32 + row) * sm + d * sk``, the key roles and V at
        ``z * sz + h * sh + (loop start + row) * sn + d * sk`` over ``range(0, N_CTX, BN)``.
        The detector's stride lists are positional; see ``_address_witness``.
        """
        from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

        heads, n_ctx, block_n = info["H"], info.get("N_CTX"), info.get("block_n")
        if type(n_ctx) is not int or type(block_n) is not int or (heads != "c1" and type(heads) is not int):
            raise MetalNonRecoverableError(
                "MLA attention: the head count, the sequence length or the key tile is not an "
                "argument the dispatcher can resolve; refusing.",
                op_name="tt.dot",
            )
        if heads == "c1":
            scalars = [(0, ("pid", 1))]
        else:
            scalars = [(0, ("div", 1, ("arg", heads))), (1, ("rem", 1, ("arg", heads)))]

        def _rows(atom, block):
            return lambda env: [env.atom(atom) * block + i for i in range(block)]

        def _role(group, key, name, arg, access, rows, block, width):
            return {
                "group": group,
                "key": key,
                "name": name,
                "arg": arg,
                "kind": access,
                "strides": list(info[group][key]),
                "shape": (block, int(width)),
                "scalars": scalars,
                "axes": [(2, 0, rows), (3, 1, lambda env, width=int(width): list(range(width)))],
            }

        nope, rope, value = info["nope_dim"], info["rope_dim"], info["v_head_dim"]
        query, keys = _rows(("pid", 0), 32), _rows(("t",), block_n)
        spec = [
            _role("strides", "q", "Q-nope", info["q"], "load", query, 32, nope),
            _role("mla_strides", "q_rope", "Q-rope", info["q_rope"], "load", query, 32, rope),
            _role("strides", "k", "K-nope", info["k"], "load", keys, block_n, nope),
            _role("mla_strides", "k_rope", "K-rope", info["k_rope"], "load", keys, block_n, rope),
            _role("strides", "v", "V", info["v"], "load", keys, block_n, value),
            _role("strides", "o", "Out", info["out"], "store", query, 32, value),
        ]
        try:
            bound = prove_template_addresses(self.graph, spec, (n_ctx, block_n))
        except WitnessRefusal as exc:
            raise MetalNonRecoverableError(
                f"MLA attention: {exc}. The dispatcher rebuilds each role from its strides and "
                "cannot prove that view is the source's; refusing.",
                op_name="tt.dot",
            ) from None
        proved = dict(info)
        proved["strides"], proved["mla_strides"] = dict(info["strides"]), dict(info["mla_strides"])
        for role in spec:
            proved[role["group"]][role["key"]] = bound[role["name"]]
        return proved

    def _lower_mla_attention_template(self, info: dict) -> str:
        """Emit the qk=head_dim / v=v_head_dim simd FA kernel for a detected MLA
        (3-dot nope/rope) attention + set the fail-CLOSED 'mla' dispatch descriptor.

        MLA's QK score is two chained dots over SEPARATE q_nope/q_rope and k_nope/k_rope
        tensors, so it can't be a single-dot @jit kernel. The kernel is generated with the
        DEFAULT (Q,K,V,Out) ABI; the dispatch ('mla' tag in _fa_dispatch) concatenates
        [q_nope|q_rope] and [k_nope|k_rope] into contiguous [.,head_dim] and runs the
        validated asymmetric kernel. FAIL-CLOSED: the qk=192 kernel's ABI differs from the
        @jit kernel's, so the host-roundtrip path can't run it -> the driver refuses on a
        miss (non-MPS / compile_shader unavailable) rather than mis-dispatch.
        """
        from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup

        name = _sanitize_msl_name(self.graph.func_name)
        # Packet 107 §4: the emitted MLA template is FIXED at BLOCK_M = 32; the replacement
        # launch (cdiv(N_CTX, 32) x Z*H) is the source kernel's program grid only when the
        # source q-block is 32 too. Bound at the producer AND re-checked by the dispatcher.
        if info.get("block_m") != 32:
            raise MetalNonRecoverableError(
                f"MLA attention: the source kernel tiles {info.get('block_m')} query rows per program but the "
                "emitted template is fixed at 32; refusing rather than launch a different program grid.",
                op_name="tt.dot",
            )
        # Packet 109 A: the emitted template declares ONE pointer type for Q, K, V and Out
        # (from out_dtype). Every source role must have exactly that element type; a
        # bf16 role behind an fp16 template (or any mixed ABI) is silently wrong.
        _norm = {
            "f16": "f16",
            "fp16": "f16",
            "half": "f16",
            "f32": "f32",
            "fp32": "f32",
            "float": "f32",
            "bf16": "bf16",
            "bfloat16": "bf16",
        }
        _role_types = {
            _norm.get(str(self.graph.args[i].elem_type), str(self.graph.args[i].elem_type))
            for i in (info["q"], info["q_rope"], info["k"], info["k_rope"], info["v"], info["out"])
        }
        if len(_role_types) != 1 or next(iter(_role_types)) not in ("f16", "f32"):
            raise MetalNonRecoverableError(
                f"MLA attention: the source roles have element types {sorted(_role_types)} but the emitted "
                "template declares one pointer type for Q/K/V/Out (f16 or f32); refusing a mixed or "
                "unsupported ABI rather than reinterpret buffers.",
                op_name="tt.dot",
            )
        # Packet 109 B: the WHOLE value path the template replaces — both QK dots (nope and
        # rope, same single scale, replayed transposes/casts only), plain or boundary-masked
        # loads, the exact nope + rope score into the softmax, the exact P and P@V path, the
        # canonical normalized Out — verified like the symmetric routes.
        info["scale"] *= self._fa_verify_value_paths(
            detected_q_index=info["q"],
            detected_k_index=info["k"],
            detected_v_index=info["v"],
            detected_out_index=info["out"],
            q_scale_required=info.get("q_scale", True),
            k_transposes=1,
            verify_score_path=True,
            expected_score_scale=info.get("score_scale"),
            mla_rope=(info["q_rope"], info["k_rope"]),
            detected_n_ctx_index=info.get("N_CTX"),
            load_mask_policy="boundary",
            q_rows=info.get("block_m"),
            kv_cols=info.get("block_n"),
        )
        # The template computes the FULL output tile and gates writes only at N_CTX: a
        # tighter store mask would be silently dropped (packet 109 mode 6).
        if self._template_output_mask_nontrivial(is_fa=True, fa_ctx_index=info.get("N_CTX")):
            raise MetalNonRecoverableError(
                "MLA attention with a non-tile-boundary output store mask is not supported: the template "
                "computes the FULL output tile and gates writes only on the N_CTX boundary, silently "
                "DROPPING any tighter store mask. Refusing to emit silently-wrong output.",
                op_name="tt.store",
            )
        # Every stride the dispatcher receives is the one the PROVED address multiplies on
        # that axis, not the detector's positional guess.
        info = self._witness_mla_attention(info)
        # The template attends the keys below N_CTX and stores the rows below it. The source
        # attends whole key tiles (a plain load reads storage past N_CTX, a masked one scores
        # those keys zero) and an unmasked store writes whole row tiles. The two agree only
        # when N_CTX is a whole number of both tiles, known only at launch.
        self._tail_access_bounds = ("mla_whole_tiles_v1", info["N_CTX"], math.lcm(32, info["block_n"]))
        # DEFAULT ABI (Q,K,V,Out + strides) — the dispatch supplies the concatenated
        # contiguous Q/K, so no @jit arg_decls/bindings. scale = 1/sqrt(qk_head_dim) as
        # detected (the user's per-part pre-scale is bypassed; the kernel scales Q).
        msl = make_flash_attention_kernel_simdgroup(
            info["head_dim"],
            32,
            64,
            causal=info["causal"],
            out_dtype=info["out_dtype"],
            kernel_name=name,
            scale=info["scale"],
            v_head_dim=info["v_head_dim"],
            half_accumulate=_fa_half_accumulate(info["out_dtype"]),
        )
        self.effective_block_size = 256
        self._used_pid_axes = {0, 1}
        # ('mla', msl, kernel_name, tg_size, q_nope, q_rope, k_nope, k_rope, v, out, Z, H, N)
        self._flash_attention = (
            "mla",
            msl,
            name,
            256,
            info["q"],
            info["q_rope"],
            info["k"],
            info["k_rope"],
            info["v"],
            info["out"],
            info["Z"],
            info["H"],
            info["N_CTX"],
            info.get("block_m"),  # [13] source q-block: the dispatcher requires the launch grid
            #                        (cdiv(N_CTX, block_m), Z*H, 1) exactly (packet 105 B)
            # Packet 107: the SOURCE ABI the dispatcher must reproduce — the kernel's
            # logical widths, its explicit stride args per role (arg index or the folded
            # "c1"), and each role's element type. The dispatcher builds exact logical
            # views from these; it never reads a host tensor's shape or .stride().
            (int(info["nope_dim"]), int(info["rope_dim"]), int(info["v_head_dim"])),  # [14]
            {  # [15]
                "q": tuple(info["strides"]["q"]),
                "q_rope": tuple(info["mla_strides"]["q_rope"]),
                "k": tuple(info["strides"]["k"]),
                "k_rope": tuple(info["mla_strides"]["k_rope"]),
                "v": tuple(info["strides"]["v"]),
                "out": tuple(info["strides"]["o"]),
            },
            tuple(  # [16] element types of (q_nope, q_rope, k_nope, k_rope, v, out)
                self.graph.args[i].elem_type
                for i in (info["q"], info["q_rope"], info["k"], info["k_rope"], info["v"], info["out"])
            ),
        )
        # The dispatcher reads these as host integers and rebuilds each role's view.
        self._attention_unsigned_scalars(
            entry
            for group in (*self._flash_attention[15].values(), (info["Z"], info["H"], info["N_CTX"]))
            for entry in group
            if isinstance(entry, int) and not isinstance(entry, bool)
        )
        self._prescan_stores()
        return msl

    def _fa_verify_value_paths(
        self,
        *,
        detected_q_index,
        detected_k_index,
        detected_v_index,
        detected_out_index,
        q_scale_arg=None,
        q_scale_required=True,
        scalar_rounding_replay=False,
        k_transposes=1,
        verify_score_path=False,
        expected_score_scale=None,
        mla_rope=None,
        detected_n_ctx_index=None,
        detected_query_index=None,
        load_mask_policy=None,
        q_rows=None,
        kv_cols=None,
        varlen_bounds=None,
        varlen_causal=False,
        extra_loads=(),
        extra_stores=(),
        biased=None,
        biased_out=None,
        dense_tail_out=None,
    ):
        """Verify every value boundary a forward-FA template replaces.

        Packet 114 (113 HOLD): ``biased={"bias": i, "mask": j|None, "lse": k, "c_eff": c}``
        adds the biased-route WHOLE-VALUE proof — the QK dot's accumulator IS the unique
        Bias load (casts/layout only); the score chain is exactly dot -> [mulf c] ->
        [select(Mask-load != 0, sentinel, score)] -> select(row&col boundary, score,
        sentinel) -> softmax, both selects sharing ONE sentinel SSA value (a float kernel
        arg, or a finite literal) which is reported in ``biased_out`` for the template;
        only those selects are transparent to the score-scale walker; the sole Lse store's
        value is exactly ``max * (1/c) + log(denominator)`` on the verified loop results.

        Packet 112 (111 HOLD): the mask proof is ROLE-EXACT and numeric. BM / BN are read
        from the VERIFIED dot operand shapes (Q = [BM, D], V = [BN, D]); a caller may pass
        ``q_rows`` / ``kv_cols`` and they must agree. A row-coordinate mask (Q, Q-rope, Out,
        Lse) must be exactly ``program_id(0) * BM + make_range(0, BM)``; a column-coordinate
        mask (K, K-rope, V, Mask) exactly ``<K-loop iv> + make_range(0, BN)``; every ``andi``
        leaf independently. ``load_mask_policy="varlen"`` keys the bound to the detector's
        ``seqlen_q`` / ``seqlen_k`` SSA values (``varlen_bounds``) instead of N_CTX.
        ``extra_loads`` = ``(arg_index, role, leaf_roles)`` for template-reconstructed loads
        beyond Q/K/V (Bias, Mask): plain, or masked with exactly those leaf roles;
        ``extra_stores`` = ``(arg_index, role, leaf_roles, mask_required)`` (Out, Lse).
        A masked load's ``other`` must be absent (this compiler's generic path substitutes
        zero — ``_lower_load``) or the literal zero.

        Packet 109/110 extensions:

        * ``mla_rope=(q_rope_index, k_rope_index)``: the MLA chain — the FIRST dot is the
          nope QK dot (its accumulator a literal zero), exactly one further dot accumulates
          onto it (the rope QK dot); its A/B operands are verified like Q/K with the SAME
          single scale constant as Q-nope, and the score path (softmax input, constant
          score-scale chain, alpha) is verified from the ROPE dot's result — the score the
          template replays is nope + rope with nothing in between.
        * ``load_mask_policy="boundary"``: every Q/K/V load the template re-reads must be
          plain, or carry exactly the sequence-boundary mask the template applies itself
          (``<tile row | tile col index> < N_CTX`` as ``slt``/``ult``, possibly two such
          compares joined by ``andi``, ``other`` absent or literal zero). A tighter or
          value mask, a different bound or a nonzero ``other`` refuses (GPU: ``mask=m < 16,
          other=0`` on Q stored the template's full result). ``None`` keeps the route's
          previous behaviour (routes whose templates replay their own masks — varlen,
          biased — are not covered by this rule; see packet 110 §7).

        The FA templates do not lower the accepted TTGIR operation by operation. They
        reload raw Q/K/V and re-emit canonical attention plus a canonical normalized
        Out store. Pointer/stride detection alone is therefore insufficient: a tracer
        can correctly find Q through ``transpose(Q)``, for example, while the template
        silently discards that transpose.

        This verifier is deliberately allowlist-based.  It requires the two dot
        operands and the output store to have exactly the transformations the selected
        template replays:

        * Q: the detector-proven zero or one scale multiply, no transpose, then a
          direct Q load;
        * K: the route-specific canonical transpose count, then a direct K load;
        * P: matching alpha/probability exponent chains whose combined base conversion
          can be folded into the template's natural-exp score scale, plus (for the
          dense/varlen routes) the exact detector-proven constant QK-result scale path;
        * V: representation/widening wrappers around a direct V load;
        * Out: representation/output-cast wrappers around the canonical division.

        Anything else refuses.  The return value is the positive factor by which the
        detected Q scale must be multiplied when the template substitutes natural exp
        (one for ``exp`` and ``exp2(x*log2e)``, ln(2) for bare ``exp2``).  In particular
        this replaces the old V-only verifier's fail-open returns: once a specialized
        FA route has been selected, an unprovable value path is not safe to template.
        """
        import math as _math

        from triton_msl.errors import MetalNonRecoverableError

        C1 = "c1"

        def _walk(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _walk(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _walk(s.else_ops)

        _all = list(_walk(self.graph.ops))
        # A correct value-path proof does not account for an extra observable
        # operation. Templates must never discard an atomic, call, synchronization
        # or unknown op merely because it does not feed the main output. This is
        # a closed vocabulary, not a denylist of the two atomics that exposed the
        # hole. Membership is necessary, NOT semantic-equivalence proof: the
        # role/value/address and loop checks below still apply independently.
        _effect_vocabulary = {
            "arith.constant",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.minsi",
            "arith.minui",
            "arith.maxsi",
            "arith.maxui",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrsi",
            "arith.shrui",
            "arith.cmpi",
            "arith.cmpf",
            "arith.select",
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "arith.index_cast",
            "arith.index_castui",
            "arith.bitcast",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "math.exp",
            "math.exp2",
            "math.log",
            "math.log2",
            "math.sqrt",
            "math.rsqrt",
            "math.absf",
            "math.fma",
            "scf.for",
            "scf.yield",
            "tt.addptr",
            "tt.broadcast",
            "tt.dot",
            "tt.expand_dims",
            "tt.exp",
            "tt.fp_to_fp",
            "tt.get_program_id",
            "tt.get_num_programs",
            "tt.load",
            "tt.make_range",
            "tt.reduce",
            "tt.reduce.return",
            "tt.reshape",
            "tt.return",
            "tt.splat",
            "tt.store",
            "tt.trans",
            "ttg.convert_layout",
            "ttg.local_alloc",
            "ttg.local_load",
            "ttg.memdesc_trans",
        }
        _foreign_effects = sorted({s.op for s in _all if s.op not in _effect_vocabulary})
        if _foreign_effects:
            raise MetalNonRecoverableError(
                f"FlashAttention effect proof cannot replay {_foreign_effects}; "
                "the template must not discard unproved operations or side effects.",
                op_name=_foreign_effects[0],
            )
        _dots = [s for s in _all if s.op == "tt.dot"]
        if len(_dots) < 2:
            raise MetalNonRecoverableError(
                "FlashAttention value path has fewer than two dots; refusing an unverifiable specialized route.",
                op_name="tt.dot",
            )
        _order = {s.id: i for i, s in enumerate(_all)}
        _dot_qk = min(_dots, key=lambda d: _order[d.id])
        _dot0_ids = {_dot_qk.id} | set(getattr(_dot_qk, "result_ids", None) or [])
        _by_id = {}
        for _s in _all:
            _by_id[_s.id] = _s
            for _rid in getattr(_s, "result_ids", None) or []:
                _by_id[_rid] = _s
        _arg_by_id = {a.id: a for a in self.graph.args}
        _loop_sources = {}
        for _s in _all:
            if _s.op != "scf.for":
                continue
            _bids = (_s.attrs or {}).get("block_arg_ids", [])
            _inits = list(_s.operand_ids or [])[3:]
            _yields = [op for op in (_s.region_ops or []) if op.op == "scf.yield"]
            for _i, _init in enumerate(_inits):
                if _i + 1 < len(_bids):
                    _loop_sources[_bids[_i + 1]] = [_init] + [
                        y.operand_ids[_i] for y in _yields if _i < len(y.operand_ids or [])
                    ]

        def _depends_on(start_id, targets, limit=256):
            seen, stack = set(), [start_id]
            while stack and len(seen) < limit:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if cur in targets:
                    return True
                _o = _by_id.get(cur)
                if _o is not None:
                    stack.extend(_o.operand_ids or [])
                    if _o.op == "scf.for":
                        stack.extend(
                            _oid
                            for _nested in (_o.region_ops or [])
                            if _nested.op == "scf.yield"
                            for _oid in (_nested.operand_ids or [])
                        )
            return False

        def _depends_on_dot0(start_id, limit=256):
            return _depends_on(start_id, _dot0_ids, limit)

        _dot_pv = next(
            (
                d
                for d in sorted(_dots, key=lambda d: _order[d.id])
                if d.id not in _dot0_ids and len(d.operand_ids) >= 2 and _depends_on_dot0(d.operand_ids[0])
            ),
            None,
        )
        if _dot_pv is None:
            raise MetalNonRecoverableError(
                "FlashAttention value path has no structurally verified P@V dot; "
                "refusing rather than template an ambiguous dot graph.",
                op_name="tt.dot",
            )

        # ---- MLA chain (packet 110): nope dot (acc = 0) -> rope dot accumulating onto it ----
        _dot_rope = None
        _score_dot = _dot_qk
        if mla_rope is not None:
            _ropes = [
                d
                for d in _dots
                if d.id not in _dot0_ids
                and d is not _dot_pv
                and len(d.operand_ids or []) > 2
                and _by_id.get(d.operand_ids[2]) is _dot_qk
            ]
            if len(_ropes) != 1 or len(_dots) != 3:
                raise MetalNonRecoverableError(
                    "MLA attention value path: expected exactly one rope QK dot accumulating onto the "
                    f"nope QK dot's result (found {len(_ropes)} of {len(_dots)} dots). Refusing.",
                    op_name="tt.dot",
                )
            _dot_rope = _ropes[0]
            if len(_dot_qk.operand_ids or []) < 3 or not self._acc_init_is_literal_zero(_dot_qk.operand_ids[2], _by_id):
                raise MetalNonRecoverableError(
                    "MLA attention value path: the nope QK dot's accumulator is not a literal zero "
                    "(a fused value would be silently dropped by the template). Refusing.",
                    op_name="tt.dot",
                )
            _score_dot = _dot_rope
            _dot0_ids = _dot0_ids | {_dot_rope.id} | set(getattr(_dot_rope, "result_ids", None) or [])

        _representation = {
            "ttg.local_load",
            "ttg.local_alloc",
            "ttg.convert_layout",
            "arith.extf",
        }
        _transpose = {"tt.trans", "ttg.memdesc_trans"}
        # packet 164 (item F-a): the source's rounding points found on the value paths — the
        # caller passes them to the makers (`round_q`, `round_p`) so the emitted kernel rounds
        # exactly where the source does. Reset per verification.
        _rounding = {}
        self._fa_rounding = _rounding
        _scalar_wrappers = {
            "tt.splat",
            "tt.broadcast",
            "tt.expand_dims",
            "ttg.convert_layout",
            "arith.extf",
            "arith.truncf",
            "arith.fpext",
            "arith.fptrunc",
            "arith.sitofp",
            "arith.uitofp",
        }

        def _refuse(role, detail):
            legacy = ""
            if role == "V":
                legacy = (
                    "V pointer role disagrees; "
                    if "load resolves to arg" in detail
                    else "P@V operand is not a direct V load; "
                )
            raise MetalNonRecoverableError(
                f"FlashAttention {legacy}{role} value path is not exactly reproduced by the "
                f"selected template ({detail}). Refusing rather than silently drop "
                "the transformation.",
                op_name="tt.dot",
            )

        def _cone_reaches_load(start_id, limit=128):
            seen, stack = set(), [start_id]
            while stack and len(seen) < limit:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    continue
                if op.op == "tt.load":
                    return True
                stack.extend(op.operand_ids or [])
            return False

        def _pointer_base_indices(start_id, limit=256):
            """Pointer bases of an address, crossing loop-carried init and yield.

            Biased FA advances K/V tensor pointers as ``scf.for`` iter args, so their
            load addresses begin at block arguments rather than at an ordinary SSA op.
            Address *offsets* can themselves depend on metadata pointers (varlen CUQ/CUK),
            so inspect the base leg of addptr/splat/layout operations rather than every
            operand in the cone.  For a loop carry, union the initializer and every
            yield source: a role swap after an iteration therefore produces two bases
            and refuses instead of being hidden by the canonical initializer.
            """
            found = set()
            seen = set()

            def walk(cur):
                if cur in seen or len(seen) >= limit:
                    return
                seen.add(cur)
                arg = _arg_by_id.get(cur)
                if arg is not None:
                    if arg.is_ptr:
                        found.add(arg.index)
                    return
                if cur in _loop_sources:
                    for source in _loop_sources[cur]:
                        walk(source)
                    return
                op = _by_id.get(cur)
                if op is None or not op.operand_ids:
                    return
                if op.op == "arith.select" and len(op.operand_ids) >= 3:
                    walk(op.operand_ids[1])
                    walk(op.operand_ids[2])
                    return
                # Pointer-producing address/layout ops preserve their base in operand 0.
                walk(op.operand_ids[0])

            walk(start_id)
            return found

        def _constant_cone(start_id):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    return False
                if op.op == "arith.constant":
                    return True
                if op.op not in _scalar_wrappers or len(op.operand_ids or []) != 1:
                    return False
                cur = op.operand_ids[0]
            return False

        def _scalar_arg_index(start_id):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                arg = _arg_by_id.get(cur)
                if arg is not None:
                    return arg.index if not arg.is_ptr else None
                op = _by_id.get(cur)
                if op is not None and op.op == "arith.constant":
                    try:
                        return C1 if int((op.attrs or {}).get("value")) == 1 else None
                    except (TypeError, ValueError):
                        return None
                if op is None or op.op not in _scalar_wrappers or len(op.operand_ids or []) != 1:
                    return None
                cur = op.operand_ids[0]
            return None

        def _scale_matches(start_id):
            if q_scale_arg is None:
                return _constant_cone(start_id)
            return _scalar_arg_index(start_id) == q_scale_arg

        def _scale_conversion_chain(start_id, role):
            """Separate scalar VALUE proof; never reuse the shape/address transparent peel.

            Only fp32 scalars, representation changes, exact narrow-to-fp32 widenings,
            and the explicitly replayed fp32-to-half/bfloat roundings are admitted.
            Record execution order, not the reverse order of this backward walk.
            """

            def element(oid):
                arg = _arg_by_id.get(oid)
                if arg is not None:
                    return str(arg.elem_type)
                op = _by_id.get(oid)
                text = (op.type_str or "") if op is not None else ""
                match = re.search(r"(?:^|x)(f32|f16|bf16)(?=[,>]|$)", text)
                return match.group(1) if match else None

            cur, seen, reversed_chain = start_id, set(), []
            layout = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"}
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                arg = _arg_by_id.get(cur)
                op = _by_id.get(cur)
                if arg is not None:
                    if arg.is_ptr or element(cur) not in ("f32", "fp32"):
                        _refuse(
                            role, f"runtime scalar {arg.name} must be fp32 at the root of the scalar conversion chain"
                        )
                    break
                if op is not None and op.op == "arith.constant":
                    if element(cur) not in ("f32", "f16", "bf16"):
                        _refuse(role, "the scalar conversion chain has an unsupported constant type")
                    break
                if op is None or len(op.operand_ids or []) != 1:
                    _refuse(role, "an unproved scalar conversion chain")
                src = op.operand_ids[0]
                if op.op in layout:
                    if element(cur) != element(src):
                        _refuse(role, "a scalar representation operation changed element type")
                elif op.op == "arith.truncf" and element(src) == "f32" and element(cur) in ("f16", "bf16"):
                    reversed_chain.append({"f16": "half", "bf16": "bfloat"}[element(cur)])
                elif op.op == "arith.extf" and element(src) in ("f16", "bf16") and element(cur) == "f32":
                    pass
                else:
                    _refuse(role, f"unreplayed scalar conversion {op.op}")
                cur = src
            else:
                _refuse(role, "a cyclic or overlong scalar conversion chain")
            chain = tuple(reversed(reversed_chain))
            if chain not in ((), ("half",), ("bfloat",), ("half", "bfloat")):
                _refuse(role, f"unproved scalar conversion chain {chain}")
            if chain and not scalar_rounding_replay:
                _refuse(role, "this route has no scalar conversion chain replay")
            return chain

        def _peel_to_core(start_id, allowed, limit=64):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < limit:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    return None
                if op.op not in allowed:
                    return op
                if len(op.operand_ids or []) != 1:
                    return None
                cur = op.operand_ids[0]
            return None

        def _is_neg_inf_constant(op):
            """A floating -inf literal: a Python float, or the raw bit word of ITS element type.

            The walker exposes an infinity as its bit word. Python numeric equality makes the
            finite float 64512.0 equal the f16 word 0xFC00, so a float is compared as a float and
            an integer word only against the pattern of the constant's own type.
            """
            if op is None or op.op != "arith.constant":
                return False
            value = (op.attrs or {}).get("value")
            if isinstance(value, bool):
                return False
            if isinstance(value, float):
                return value == float("-inf")
            if isinstance(value, int):
                words = {"f16": 0xFC00, "bf16": 0xFF80, "f32": 0xFF800000, "f64": 0xFFF0000000000000}
                return value == words.get(op.elem_type)
            return False

        # ---- load masks (packet 110): plain, or exactly the template's sequence-boundary mask ----
        _mask_wrappers = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"}
        _fors = [s for s in _all if s.op == "scf.for"]
        _kv_iv = ((_fors[0].attrs or {}).get("block_arg_ids") or [None])[0] if len(_fors) == 1 else None

        def _const_value(start_id):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    return None
                if op.op == "arith.constant":
                    value = (op.attrs or {}).get("value")
                    return float(value) if isinstance(value, (int, float)) else None
                if op.op not in _scalar_wrappers or len(op.operand_ids or []) != 1:
                    return None
                cur = op.operand_ids[0]
            return None

        # BM / BN from the VERIFIED dot operand shapes — never from whatever the mask says.
        def _shape_of(oid):
            o = _by_id.get(oid)
            t = getattr(o, "type_str", None) if o is not None else None
            return _extract_shape(t) if t else ()

        _q_shape, _v_shape = _shape_of(_dot_qk.operand_ids[0]), _shape_of(_dot_pv.operand_ids[1])
        _BM = int(_q_shape[0]) if len(_q_shape) == 2 else None
        _BN = int(_v_shape[0]) if len(_v_shape) == 2 else None
        if load_mask_policy is not None:
            if _BM is None or _BN is None:
                _refuse("Q/K/V", "the tile shapes could not be read from the verified dot operands")
            if q_rows is not None and int(q_rows) != _BM:
                _refuse("Q", f"the verified Q tile has {_BM} rows but the route expects {q_rows}")
            if kv_cols is not None and int(kv_cols) != _BN:
                _refuse("K/V", f"the verified K/V tile has {_BN} rows but the route expects {kv_cols}")
            if load_mask_policy == "varlen" and (varlen_bounds is None or len(varlen_bounds) != 2):
                _refuse("Q/K/V", "the varlen route did not supply its seqlen_q / seqlen_k values")
        _ROW_ROLES = {"Q", "Q-rope", "Out", "Lse"}  # program_id(0)*BM + range(0,BM)
        _COL_ROLES = {"K", "K-rope", "V", "Mask"}  # iv + range(0,BN)

        def _bound_ok(rhs, role):
            if load_mask_policy == "boundary":
                if detected_query_index is not None and role in _ROW_ROLES:
                    return (
                        _const_value(rhs) == 1
                        if detected_query_index == "c1"
                        else _scalar_arg_index(rhs) == detected_query_index
                    )
                return detected_n_ctx_index is not None and _scalar_arg_index(rhs) == detected_n_ctx_index
            if load_mask_policy == "varlen":
                core = _peel_to_core(rhs, _mask_wrappers)
                want = varlen_bounds[0] if role in _ROW_ROLES else varlen_bounds[1]
                return core is not None and core.id == want
            return False

        def _coord_ok(lhs, role):
            sh = self._index_shape(lhs, _by_id, _arg_by_id, _kv_iv, 1)
            if sh["bad"] is not None or sh["mod"] is not None or sh["range"] != 1 or sh["bounds"] is None:
                return False
            if role in _ROW_ROLES:
                return (
                    sh["bounds"] == (0, _BM)
                    and sh["iv"] == 0
                    and sh["pid"] == 0
                    and sh["pid_div"] is None
                    and sh["coef"] == _BM
                )
            if detected_n_ctx_index == C1 and _kv_iv is None:
                # The one-trip KV loop folds to a direct range(0, BN).  It is
                # still the exact column coordinate because the proved bound is
                # literal one and there is no surviving loop IV or program axis.
                return sh["bounds"] == (0, _BN) and sh["iv"] == 0 and sh["pid"] is None and sh["pid_div"] is None
            return sh["bounds"] == (0, _BN) and sh["iv"] == 1 and sh["pid"] is None and _kv_iv is not None

        def _leaf_compare(cid, role, negated=False):
            """The comparison a mask leaf is, as ``(coordinate id, bound id, expands above it)``.

            ``negated`` reads the complement instead: ``index >= bound`` is true exactly
            where the boundary ``index < bound`` is false.
            """
            cmp_op, above = _peel_expands(cid)
            if cmp_op is None or cmp_op.op != "arith.cmpi" or len(cmp_op.operand_ids or []) != 2:
                _refuse(role, "the mask is not a sequence-boundary comparison")
            pred = (cmp_op.attrs or {}).get("predicate_name")
            lhs, rhs = cmp_op.operand_ids
            same, mirrored = (("sge", "uge"), ("sle", "ule")) if negated else (("slt", "ult"), ("sgt", "ugt"))
            if pred in mirrored:
                # ``N > index`` is the same boundary written from the other side.
                lhs, rhs = rhs, lhs
            elif pred not in same:
                _refuse(role, "the mask is not a strict less-than boundary")
            return lhs, rhs, above

        def _peel_expands(start_id):
            """Peel the mask wrappers; returns ``(core op, the tt.expand_dims ops passed)``."""
            cur, seen, expands = start_id, set(), []
            while cur not in seen and len(seen) < 64:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    return None, expands
                if op.op not in _mask_wrappers:
                    return op, expands
                if len(op.operand_ids or []) != 1:
                    return None, expands
                if op.op == "tt.expand_dims":
                    expands.append(op)
                cur = op.operand_ids[0]
            return None, expands

        def _range_expands(vid, depth=0):
            """The tt.expand_dims ops between a coordinate and its one make_range, or None."""
            op = _by_id.get(vid)
            if op is None or depth > 32:
                return None
            if op.op == "tt.make_range":
                return []
            for x in op.operand_ids or []:
                below = _range_expands(x, depth + 1)
                if below is not None:
                    return below + ([op] if op.op == "tt.expand_dims" else [])
            return None

        def _expand_axis(expands):
            """Where a 1-D coordinate varies in the tensor it was broadcast into.

            ``-1`` while it is still a vector, ``0`` or ``1`` for the matrix axis its single
            ``tt.expand_dims`` leaves it on, None when that cannot be read. The coordinate
            proof says WHAT a mask leaf compares; a square tile gives a transposed broadcast
            the same coordinate and the same shape, so WHERE it varies is read from the
            expand itself.
            """
            if expands is None:
                return None
            if not expands:
                return -1
            if len(expands) != 1:
                return None
            axis = (expands[0].attrs or {}).get("axis")
            if len(_shape_of(expands[0].operand_ids[0])) != 1 or isinstance(axis, bool) or axis not in (0, 1):
                return None
            return 1 - axis

        # A boundary against a scalar argument is the same signed or unsigned (the launch
        # refuses a negative one). A varlen length is read from a tensor on the GPU, so the
        # varlen lowering needs the signedness each length is compared in.
        _boundary_signs = self._fa_boundary_signs = {}

        def _boundary_leaf(cid, role, allowed_roles, axes=None, above=(), negated=False):
            """One comparison leaf; returns the leaf role it is EXACTLY the boundary of.

            ``axes`` maps a leaf role to the axis its coordinate must vary along in the
            tensor the mask applies to (``-1`` for a vector); ``above`` are the expands
            between that tensor and this leaf.
            """
            lhs, rhs, own = _leaf_compare(cid, role, negated)
            for r in allowed_roles:
                if _bound_ok(rhs, r) and _coord_ok(lhs, r):
                    _pred = (_peel_expands(cid)[0].attrs or {}).get("predicate_name") or ""
                    _boundary_signs.setdefault("Q" if r in _ROW_ROLES else "K", set()).add(_pred[:1])
                    if axes is not None:
                        below = _range_expands(lhs)
                        got = _expand_axis(None if below is None else list(above) + own + below)
                        if got != axes[r]:
                            _refuse(
                                role,
                                f"the {r} boundary does not vary along the template's axis for this role "
                                f"(expected axis {axes[r]}, found {got}; a transposed broadcast masks other cells)",
                            )
                    return r
            _refuse(
                role,
                "the mask is not exactly the template's boundary for this role "
                f"(expected {'program_id(0)*BM + range(0,BM)' if role in _ROW_ROLES else 'iv + range(0,BN)'} "
                f"< {'seqlen' if load_mask_policy == 'varlen' else 'N_CTX'}, BM={_BM}, BN={_BN})",
            )

        def _mask_leaf_roles(mid, role, allowed_roles, axes, negated=False):
            core, above = _peel_expands(mid)
            if not negated and core is not None and core.op == "arith.andi" and len(core.operand_ids or []) == 2:
                return [_boundary_leaf(x, role, allowed_roles, axes, above) for x in core.operand_ids]
            return [_boundary_leaf(mid, role, allowed_roles, axes, negated=negated)]

        def _mask_axes(tensor_id, matrix_axes, role):
            """Leaf role -> required axis for a mask on ``tensor_id`` (a vector masks along -1)."""
            rank = len(_shape_of(tensor_id))
            if rank == 1:
                return dict.fromkeys(matrix_axes, -1)
            if rank != 2:
                _refuse(role, f"a masked tensor of rank {rank} is not a tile the template replays")
            return matrix_axes

        # Rows of the score are queries and its columns are keys. The dot's left operand
        # carries its coordinate on axis 0 and K, the right operand, on axis 1; V, the
        # right operand of P@V, carries the key on axis 0. Each transpose swaps the axes.
        _LOAD_AXIS = {"Q": 0, "Q-rope": 0, "K": 1, "K-rope": 1, "V": 0}

        def _check_load_mask(load, role, allowed_roles=None, exact_leaves=None, *, axes):
            if load_mask_policy is None or len(load.operand_ids or []) < 2:
                return
            leaves = _mask_leaf_roles(load.operand_ids[1], role, allowed_roles or (role,), axes)
            if exact_leaves is not None and sorted(leaves) != sorted(exact_leaves):
                _refuse(role, f"the mask leaves {leaves} are not exactly the template's {list(exact_leaves)} boundary")
            if len(load.operand_ids) >= 3 and _const_value(load.operand_ids[2]) != 0.0:
                _refuse(role, "the load's 'other' value is not zero (the template supplies zero beyond the boundary)")

        def _check_store_mask(store, role, leaf_roles, mask_required):
            if load_mask_policy is None:
                return
            if len(store.operand_ids or []) < 3:
                if mask_required:
                    _refuse(role, "the store is unmasked but the template stores only inside the sequence boundary")
                return
            # Out is [rows, head_dim] and Lse is [rows]: a row boundary varies along axis 0.
            axes = _mask_axes(store.operand_ids[1], {"Q": 0, "K": 1}, "Out")
            leaves = _mask_leaf_roles(store.operand_ids[2], role, tuple(leaf_roles), axes)
            if sorted(leaves) != sorted(leaf_roles):
                _refuse(
                    role, f"the store mask leaves {leaves} are not exactly the template's {list(leaf_roles)} boundary"
                )

        _verified_loads = {}

        def _verify_load_path(start_id, role, expected_index, *, scale=False, transposes=0):
            """Returns the scale constant applied on this path (None when arg-based or absent)."""
            cur, seen = start_id, set()
            n_scale = 0
            n_transpose = 0
            scale_value = None
            while cur not in seen and len(seen) < 64:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    _refuse(role, "the path terminates at an unknown SSA value")
                if op.op == "tt.load":
                    if n_scale != int(scale):
                        _refuse(role, f"expected {int(scale)} scale multiply, found {n_scale}")
                    if n_transpose != transposes:
                        _refuse(role, f"expected {transposes} transpose, found {n_transpose}")
                    ptrs = _pointer_base_indices(op.operand_ids[0]) if op.operand_ids else set()
                    if len(ptrs) != 1:
                        _refuse(role, f"the backing load address contains pointer args {sorted(ptrs)}")
                    ptr_index = next(iter(ptrs))
                    if ptr_index != expected_index:
                        _refuse(
                            role,
                            f"the load resolves to arg {ptr_index}, not detector arg {expected_index}",
                        )
                    _check_load_mask(
                        op, role, axes=_mask_axes(op.id, {role: _LOAD_AXIS[role] ^ (n_transpose & 1)}, role)
                    )
                    _verified_loads[role] = op
                    return scale_value
                if op.op in _representation and len(op.operand_ids or []) == 1:
                    cur = op.operand_ids[0]
                    continue
                if (
                    detected_query_index is not None
                    and role == "Q"
                    and op.op == "arith.truncf"
                    and len(op.operand_ids or []) == 1
                    and "round_q" not in _rounding
                ):
                    dst = self._native_value_facts(op.id, op_name=op.op)
                    src = self._native_value_facts(op.operand_ids[0], op_name=op.op)
                    producer = _by_id.get(op.operand_ids[0])
                    if (
                        dst.elem != "f16"
                        or src.elem != "f32"
                        or dst.shape != src.shape
                        or producer is None
                        or producer.op != "arith.mulf"
                    ):
                        _refuse(role, "decode Q narrowing is not the fp32 scale product rounded to fp16")
                    _rounding["round_q"] = "half"
                    cur = op.operand_ids[0]
                    continue
                if op.op in _transpose and len(op.operand_ids or []) == 1:
                    n_transpose += 1
                    cur = op.operand_ids[0]
                    continue
                if scale and op.op == "arith.mulf" and len(op.operand_ids or []) == 2:
                    if n_scale:
                        _refuse(role, "more than one scale multiply appears before the dot")
                    loaded = [oid for oid in op.operand_ids if _cone_reaches_load(oid)]
                    if len(loaded) != 1:
                        _refuse(role, "the scale multiply does not have one load-derived operand")
                    other = next(oid for oid in op.operand_ids if oid != loaded[0])
                    if not _scale_matches(other):
                        _refuse(role, "the multiply's scale source disagrees with detection")
                    _scale_chain = _scale_conversion_chain(other, role)
                    if _scale_chain:
                        _rounding["scale_chain"] = _scale_chain
                    scale_value = _const_value(other)
                    # packet 164 (item F-a): the multiply's element type IS a rounding point when it
                    # is narrower than fp32 — the source computes q * full(scale, dtype) in that
                    # dtype (scale rounded, each product rounded). Record it for the makers.
                    _mt = op.type_str or ""
                    _el = _mt.rsplit("x", 1)[-1].split(",")[0].split(">")[0].strip() if "x" in _mt else _mt.strip()
                    _round_q = {"f16": "half", "bf16": "bfloat"}.get(_el)
                    if _round_q is not None:
                        _rounding["round_q"] = _round_q
                    elif _el not in ("f32", ""):
                        _refuse(role, f"a scale multiply in {_el} the templates cannot replay")
                    n_scale += 1
                    cur = loaded[0]
                    continue
                _refuse(role, f"unsupported operation {op.op} appears before the dot")
            _refuse(role, "the path is cyclic or exceeds the verification bound")

        _q_scale_value = _verify_load_path(
            _dot_qk.operand_ids[0],
            "Q",
            detected_q_index,
            scale=q_scale_required,
            transposes=0,
        )
        _verify_load_path(_dot_qk.operand_ids[1], "K", detected_k_index, transposes=k_transposes)
        _verify_load_path(_dot_pv.operand_ids[1], "V", detected_v_index, transposes=0)
        # A V load without a mask reads the rows past the boundary; the varlen and biased
        # templates supply zero there. Q and K past the boundary reach only scores the
        # proved selects replace and rows the store drops, so their masks are optional.
        _v_unmasked = len(_verified_loads["V"].operand_ids or []) < 2
        if _v_unmasked and load_mask_policy == "varlen":
            _refuse(
                "V",
                "the V load is unmasked: the source reads the rows past each sequence's last key, "
                "where the template supplies zero",
            )
        if biased_out is not None:
            biased_out["v_unmasked"] = _v_unmasked
        if _dot_rope is not None:
            _qr_scale_value = _verify_load_path(
                _dot_rope.operand_ids[0], "Q-rope", mla_rope[0], scale=q_scale_required, transposes=0
            )
            _verify_load_path(_dot_rope.operand_ids[1], "K-rope", mla_rope[1], transposes=k_transposes)
            if q_scale_required and (
                _q_scale_value is None
                or _qr_scale_value is None
                or abs(_q_scale_value - _qr_scale_value) > 1e-7 * max(1.0, abs(_q_scale_value))
            ):
                _refuse(
                    "Q-rope",
                    "the rope scale is not the same single constant as the nope scale (the template bakes ONE Q scale)",
                )

        # P and alpha are one score->probability function.  The template always emits
        # natural exp, but equivalent source spellings may use exp2 with log2e either
        # after ``score-max`` or folded into Q's scale.  Parse BOTH exponent chains,
        # require their natural-exponent multipliers to agree, and return that multiplier
        # so dense/varlen callers normalize the template's baked score scale.  Arithmetic
        # after the accepted P exp remains forbidden.
        _p_passthrough = _representation | {"arith.truncf", "tt.fp_to_fp"}
        _cur = _dot_pv.operand_ids[0]
        _seen = set()
        _p_exp = None
        while _cur not in _seen and len(_seen) < 64:
            _seen.add(_cur)
            _op = _by_id.get(_cur)
            if _op is None:
                _refuse("P", "the path terminates at an unknown SSA value")
            if _op_is_exp(_op.op):
                _p_exp = _op
                break
            if _op.op not in _p_passthrough or len(_op.operand_ids or []) != 1:
                _refuse("P", f"unsupported operation {_op.op} appears after softmax")
            if _op.op in ("arith.truncf", "tt.fp_to_fp"):
                # packet 164 (item F-a): P rounded to the input dtype before P @ V is a rounding
                # point the makers replay; record its target type (one narrowing at most).
                _pt = _op.type_str or ""
                _pel = _pt.rsplit("x", 1)[-1].split(",")[0].split(">")[0].strip() if "x" in _pt else _pt.strip()
                _round_p = {"f16": "half", "bf16": "bfloat"}.get(_pel)
                if _round_p is None or "round_p" in _rounding:
                    _refuse("P", f"a P conversion to {_pel or '?'} the templates cannot replay")
                _rounding["round_p"] = _round_p
            _cur = _op.operand_ids[0]
        else:
            _refuse("P", "the path is cyclic or exceeds the verification bound")

        _exp_input_passthrough = {"ttg.convert_layout", "arith.extf", "arith.truncf"}

        def _constant_float(start_id):
            cur, seen = start_id, set()
            while cur not in seen and len(seen) < 32:
                seen.add(cur)
                op = _by_id.get(cur)
                if op is None:
                    return None
                if op.op == "arith.constant":
                    value = (op.attrs or {}).get("value")
                    return float(value) if isinstance(value, (int, float)) else None
                if op.op not in _scalar_wrappers or len(op.operand_ids or []) != 1:
                    return None
                cur = op.operand_ids[0]
            return None

        def _parse_exp(exp_op, role):
            if exp_op is None or len(exp_op.operand_ids or []) != 1:
                _refuse(role, "the softmax exponential does not have one input")
            core = _peel_to_core(exp_op.operand_ids[0], _exp_input_passthrough)
            factor = 1.0
            if core is not None and core.op == "arith.mulf":
                if len(core.operand_ids or []) != 2:
                    _refuse(role, "the exponential multiplier is not binary")
                constants = [(oid, _constant_float(oid)) for oid in core.operand_ids]
                constants = [(oid, value) for oid, value in constants if value is not None]
                if len(constants) != 1:
                    _refuse(
                        role,
                        "the exponential input does not have exactly one constant factor",
                    )
                const_id, factor = constants[0]
                data_id = next(oid for oid in core.operand_ids if oid != const_id)
                core = _peel_to_core(data_id, _exp_input_passthrough)
            if not (_math.isfinite(factor) and factor > 0.0):
                _refuse(role, f"the exponential factor {factor!r} is not positive finite")
            if core is None or core.op != "arith.subf" or len(core.operand_ids or []) != 2:
                _refuse(role, "the accepted exp is not the canonical value-minus-max form")
            if exp_op.op == "math.exp2":
                base_log = _math.log(2.0)
            elif exp_op.op in ("math.exp", "tt.exp"):
                base_log = 1.0
            else:
                _refuse(role, f"unsupported exponential operation {exp_op.op}")
            return core, factor * base_log

        _p_core, _p_multiplier = _parse_exp(_p_exp, "P")
        if not _depends_on_dot0(_p_core.operand_ids[0]):
            _refuse("P", "the probability score does not depend on the QK dot")
        _p_shape = _extract_shape(_p_core.type_str or "")
        if len(_p_shape) != 2:
            _refuse("P", f"the probability core has non-matrix shape {_p_shape}")

        # ---- template-reconstructed loads / stores beyond Q/K/V (packet 112) ----
        if load_mask_policy is not None:
            for _idx, _role, _leaf_roles in extra_loads:
                _lds = [
                    o
                    for o in _all
                    if o.op == "tt.load" and o.operand_ids and _pointer_base_indices(o.operand_ids[0]) == {_idx}
                ]
                if not _lds:
                    _refuse(_role, f"no load of arg {_idx} was found to verify")
                for _ld in _lds:
                    # Bias is [rows, keys]; Mask is a vector of keys.
                    _check_load_mask(
                        _ld,
                        _role,
                        tuple(_leaf_roles),
                        exact_leaves=tuple(_leaf_roles),
                        axes=_mask_axes(_ld.id, {"Q": 0, "K": 1}, _role),
                    )
            for _idx, _role, _leaf_roles, _req in extra_stores:
                _sts = [
                    o
                    for o in _all
                    if o.op == "tt.store" and o.operand_ids and _pointer_base_indices(o.operand_ids[0]) == {_idx}
                ]
                if len(_sts) != 1:
                    _refuse(_role, f"expected exactly one store of arg {_idx}, found {len(_sts)}")
                _check_store_mask(_sts[0], _role, tuple(_leaf_roles), _req)

        # ---- biased route (packet 114): Bias value, Mask value, both score selects, sentinel ----
        _certified_selects = None
        _biased_S2_id = None
        _b_wr = {
            "arith.extf",
            "arith.truncf",
            "ttg.convert_layout",
            "tt.reshape",
            "ttg.local_load",
            "ttg.local_alloc",
            "tt.fp_to_fp",
        }
        _b_bcast = _b_wr | {"tt.splat", "tt.broadcast", "tt.expand_dims"}

        def _peel_id(start, allowed):
            cur, seen = start, set()
            while cur not in seen and len(seen) < 64:
                seen.add(cur)
                o = _by_id.get(cur)
                if o is None or o.op not in allowed or len(o.operand_ids or []) != 1:
                    return cur
                cur = o.operand_ids[0]
            return cur

        def _expands_to(start, stop, allowed):
            """The tt.expand_dims ops from ``start`` down to ``stop`` through ``allowed``, or None."""
            cur, seen, expands = start, set(), []
            while cur not in seen and len(seen) < 64:
                if cur == stop:
                    return expands
                seen.add(cur)
                o = _by_id.get(cur)
                if o is None or o.op not in allowed or o.op == "tt.reshape" or len(o.operand_ids or []) != 1:
                    return None
                if o.op == "tt.expand_dims":
                    expands.append(o)
                cur = o.operand_ids[0]
            return None

        if biased is not None:
            _dot_ids_all = {_dot_qk.id} | set(getattr(_dot_qk, "result_ids", None) or [])

            def _unique_load_of(idx, role):
                lds = [
                    o
                    for o in _all
                    if o.op == "tt.load" and o.operand_ids and _pointer_base_indices(o.operand_ids[0]) == {idx}
                ]
                if len(lds) != 1:
                    _refuse(role, f"expected exactly one load of arg {idx}, found {len(lds)}")
                return lds[0]

            # (1) Bias value: the accumulator operand IS the Bias load (no arithmetic, no select)
            _bias_ld = _unique_load_of(biased["bias"], "Bias")
            if len(_dot_qk.operand_ids or []) < 3 or _peel_id(_dot_qk.operand_ids[2], _b_wr) != _bias_ld.id:
                _refuse(
                    "Bias",
                    "the QK dot's accumulator is not the Bias load itself (a transform of the bias is not replayed)",
                )

            # (2)+(3) the score chain, walked from the softmax input back to the dot
            _S2 = _by_id.get(_peel_id(_p_core.operand_ids[0], _b_wr))
            if _S2 is None or _S2.op != "arith.select" or len(_S2.operand_ids or []) != 3:
                _refuse("Score", "the softmax input is not the tile-boundary select the template replays")
            _leaves = _mask_leaf_roles(
                _S2.operand_ids[0], "Score", ("Q", "K"), _mask_axes(_S2.id, {"Q": 0, "K": 1}, "Score")
            )
            if sorted(_leaves) != ["K", "Q"]:
                _refuse(
                    "Score",
                    f"the boundary select's predicate leaves {_leaves} are not exactly the row & col N_CTX boundary",
                )
            _sentinel_id = _peel_id(_S2.operand_ids[2], _b_bcast)
            _chain = _peel_id(_S2.operand_ids[1], _b_wr)
            _S1 = None
            if biased.get("mask") is not None:
                _mask_ld = _unique_load_of(biased["mask"], "Mask")
                _S1 = _by_id.get(_chain)
                if _S1 is None or _S1.op != "arith.select" or len(_S1.operand_ids or []) != 3:
                    _refuse(
                        "Mask",
                        "the loaded-mask select is not where the template replays it (between the scale and the boundary select)",
                    )
                _cmp = _by_id.get(_peel_id(_S1.operand_ids[0], _b_bcast))
                if (
                    _cmp is None
                    or _cmp.op != "arith.cmpi"
                    or (_cmp.attrs or {}).get("predicate_name") != "ne"
                    or len(_cmp.operand_ids or []) != 2
                ):
                    _refuse("Mask", "the loaded-mask select's predicate is not `Mask != 0`")
                _cmp_src = [_peel_id(x, _b_bcast) for x in _cmp.operand_ids]
                if not (
                    (_cmp_src[0] == _mask_ld.id and _const_value(_cmp.operand_ids[1]) == 0.0)
                    or (_cmp_src[1] == _mask_ld.id and _const_value(_cmp.operand_ids[0]) == 0.0)
                ):
                    _refuse("Mask", "the loaded-mask select's predicate is not the Mask load compared with zero")
                # The template masks a KEY: the loaded vector must run along the score's
                # columns. ``m_block[:, None]`` has the same shape on a square tile and
                # masks whole query rows instead.
                _m_side = _cmp.operand_ids[0] if _cmp_src[0] == _mask_ld.id else _cmp.operand_ids[1]
                _m_path = _expands_to(_S1.operand_ids[0], _cmp.id, _b_bcast)
                _m_load = _expands_to(_m_side, _mask_ld.id, _b_bcast)
                if (
                    len(_shape_of(_S1.id)) != 2
                    or _m_path is None
                    or _m_load is None
                    or _expand_axis(_m_path + _m_load) != 1
                ):
                    _refuse(
                        "Mask",
                        "the loaded mask is not broadcast along the score's key axis (the template masks keys)",
                    )
                if _peel_id(_S1.operand_ids[1], _b_bcast) != _sentinel_id:
                    _refuse(
                        "Mask",
                        "the loaded-mask select's masked branch is not the same sentinel value as the boundary select's",
                    )
                _chain = _peel_id(_S1.operand_ids[2], _b_wr)
            # the constant score scale (c_eff) sits between the dot and the selects
            _c_eff = float(biased.get("c_eff", 1.0))
            _M = _by_id.get(_chain)
            if abs(_c_eff - 1.0) > 1e-9:
                if _M is None or _M.op != "arith.mulf" or len(_M.operand_ids or []) != 2:
                    _refuse(
                        "Score",
                        "the score scale multiply is not between the dot and the selects (the sentinel units would differ)",
                    )
                _cs = [(_const_value(x), x) for x in _M.operand_ids]
                _consts = [(v, x) for v, x in _cs if v is not None]
                if len(_consts) != 1 or abs(_consts[0][0] - _c_eff) > 1e-6 * abs(_c_eff):
                    _refuse("Score", "the score scale multiply's constant disagrees with the detector's proof")
                _data = next(x for v, x in _cs if v is None)
                if _peel_id(_data, _b_wr) not in _dot_ids_all:
                    _refuse("Score", "the score scale multiply does not consume the QK dot directly")
                _biased_expected = {"factor": _consts[0][0], "op_ids": (_M.id,)}
            else:
                if _chain not in _dot_ids_all:
                    _refuse("Score", "the selects do not consume the QK dot directly")
                _biased_expected = {"factor": 1.0, "op_ids": ()}
            # (4) the sentinel: one SSA scalar — an ABI-proven fp32 kernel arg
            # (forwarded) or a literal (baked).
            _snt_op = _by_id.get(_sentinel_id)
            if _snt_op is None and _sentinel_id in _arg_by_id:
                _a = _arg_by_id[_sentinel_id]
                if _a.is_ptr or str(_a.elem_type) not in ("f32", "fp32", "float"):
                    _refuse(
                        "Score",
                        f"runtime sentinel arg {_a.name} must be fp32 for the replacement scalar ABI",
                    )
                _biased_snt = {"sentinel_arg": _a.index}
            elif _snt_op is not None and _snt_op.op == "arith.constant":
                # TTGIR walkers expose a floating literal as a float, or as its RAW BIT WORD
                # (±inf / NaN): decode exactly; any other integer word is ambiguous -> refuse.
                _raw = (_snt_op.attrs or {}).get("value")
                if isinstance(_raw, bool) or _raw is None:
                    _refuse("Score", "the literal sentinel has no value")
                if isinstance(_raw, int):
                    if _raw in (0xFC00, 0xFF80, 0xFF800000, 0xFFF0000000000000):
                        _v = float("-inf")
                    elif _raw in (0x7C00, 0x7F80, 0x7F800000, 0x7FF0000000000000):
                        _v = float("inf")
                    elif _raw == 0:
                        _v = 0.0
                    else:
                        _refuse("Score", f"the literal sentinel's raw word {_raw:#x} is not a decodable float")
                else:
                    _v = float(_raw)
                if _v != _v or _v == float("inf"):
                    _refuse("Score", "a NaN / +inf literal sentinel is not replayed")
                _biased_snt = {"sentinel_const": _v}
            else:
                _refuse("Score", "the sentinel is neither a kernel argument nor a literal constant")
            _certified_selects = {_S2.id} | ({_S1.id} if _S1 is not None else set())
            _biased_S2_id = _S2.id
            expected_score_scale = _biased_expected
            verify_score_path = True
            if biased_out is not None:
                biased_out.update(_biased_snt)
                biased_out["certified_selects"] = tuple(sorted(_certified_selects))

        _varlen_select = None
        if load_mask_policy == "varlen":
            # The template gives every key past the sequence's last a score of negative
            # infinity. The source does the same only through selects on the score that keep
            # it and supply negative infinity: the key boundary exactly once (alone, or with
            # the row boundary, whose extra rows are never stored, and the causal triangle
            # when the template applies one), and-ed in one select or split over a chain of
            # them. A masked K load alone leaves those scores at zero.
            _sels = [s for s in _all if s.op == "arith.select"]
            if not 1 <= len(_sels) <= 3 or any(len(s.operand_ids or []) != 3 for s in _sels):
                _refuse("Score", f"expected the key-boundary select on the score, found {len(_sels)} selects")
            _leaves = []

            def _score_leaf(cid, above, depth, _axes):
                core, own = _peel_expands(cid)
                if core is not None and core.op == "arith.andi" and len(core.operand_ids or []) == 2 and depth < 3:
                    for x in core.operand_ids:
                        _score_leaf(x, above + own, depth + 1, _axes)
                    return
                if (
                    core is not None
                    and core.op == "arith.cmpi"
                    and len(core.operand_ids or []) == 2
                    and (core.attrs or {}).get("predicate_name") in ("sge", "uge")
                ):
                    # The template's causal guard: query row >= key column, within the sequence.
                    _row, _col = core.operand_ids
                    _row_x, _col_x = _range_expands(_row), _range_expands(_col)
                    if not (
                        _coord_ok(_row, "Q")
                        and _coord_ok(_col, "K")
                        and _row_x is not None
                        and _col_x is not None
                        and _expand_axis(above + own + _row_x) == 0
                        and _expand_axis(above + own + _col_x) == 1
                    ):
                        _refuse("Score", "a row-versus-key comparison is not the template's lower triangle")
                    _leaves.append("causal")
                    return
                _leaves.append(_boundary_leaf(cid, "Score", ("Q", "K"), _axes, above))

            for _sel in _sels:
                _cond, _yes, _no = _sel.operand_ids
                _cond_core = _peel_to_core(_cond, _mask_wrappers)
                # ``index >= bound`` supplying -inf is the boundary select written the other
                # way round; ``row >= key`` (both sides coordinates) is the causal triangle.
                _inverted = (
                    _cond_core is not None
                    and _cond_core.op == "arith.cmpi"
                    and (_cond_core.attrs or {}).get("predicate_name") in ("sge", "uge", "sle", "ule")
                    and not all(_range_expands(x) is not None for x in _cond_core.operand_ids or [])
                )
                _kept, _fill = (_no, _yes) if _inverted else (_yes, _no)
                _axes = _mask_axes(_sel.id, {"Q": 0, "K": 1}, "Score")
                if _inverted:
                    _leaves += _mask_leaf_roles(_cond, "Score", ("Q", "K"), _axes, negated=True)
                else:
                    _score_leaf(_cond, [], 0, _axes)
                _fill_core = _peel_to_core(_fill, _mask_wrappers)
                if not _depends_on_dot0(_kept) or _depends_on_dot0(_fill) or not _is_neg_inf_constant(_fill_core):
                    _refuse("Score", "the score select does not keep the score inside the key boundary and supply -inf")
            if (
                _leaves.count("K") != 1
                or _leaves.count("Q") > 1
                or _leaves.count("causal") != (1 if varlen_causal else 0)
                or len(_leaves) != _leaves.count("K") + _leaves.count("Q") + _leaves.count("causal")
            ):
                _refuse(
                    "Score",
                    f"the score select's leaves {_leaves} are not the key boundary"
                    + (" and the lower triangle" if varlen_causal else ""),
                )
            _varlen_select = _sels[0]
            _certified_selects = {s.id for s in _sels}

        if verify_score_path:
            _score_scale = self._fa_analyze_constant_score_scale(
                _score_dot, _dot_pv, certified_selects=_certified_selects
            )  # MLA: from the rope dot
            if _score_scale is None:
                _refuse(
                    "P",
                    "the QK-result-to-softmax path is not an exact constant scale chain",
                )
            if _varlen_select is not None and sorted(_score_scale.get("select_ids") or ()) != sorted(
                _certified_selects
            ):
                _refuse("Score", "a sequence-boundary select is not on the path from the QK dot to the softmax")
            if expected_score_scale is None:
                _refuse("P", "the detector supplied no score-scale proof")
            _factor_tol = 1e-7 * max(
                1.0,
                abs(_score_scale["factor"]),
                abs(expected_score_scale["factor"]),
            )
            if (
                _score_scale["op_ids"] != expected_score_scale["op_ids"]
                or abs(_score_scale["factor"] - expected_score_scale["factor"]) > _factor_tol
            ):
                _refuse(
                    "P",
                    "the verified score-scale chain disagrees with detection",
                )

        # Any second P-side exp (the post-softmax-exp bite) appears here in addition to
        # the one alpha exp and therefore refuses.  Conversely, requiring exactly one
        # alpha prevents a template from silently inventing or discarding a recurrence.
        _alpha_exps = [op for op in _all if _op_is_exp(op.op) and op is not _p_exp]
        if len(_alpha_exps) != 1:
            _refuse("P", f"expected one online-softmax alpha exp, found {len(_alpha_exps)}")
        _alpha_core, _alpha_multiplier = _parse_exp(_alpha_exps[0], "P")
        if not _depends_on_dot0(_alpha_core.operand_ids[1]):
            _refuse("P", "the alpha max does not depend on the QK dot")
        _max_passthrough = _representation | {"tt.broadcast", "tt.expand_dims"}
        _p_max = _peel_to_core(_p_core.operand_ids[1], _max_passthrough)
        _alpha_max = _peel_to_core(_alpha_core.operand_ids[1], _max_passthrough)
        if _p_max is None or _alpha_max is not _p_max:
            _refuse(
                "P",
                "the probability and alpha recurrences do not use the same score max",
            )
        _alpha_shape = _extract_shape(_alpha_core.type_str or "")
        if len(_alpha_shape) != 1:
            _refuse("P", f"the alpha core has non-vector shape {_alpha_shape}")
        _mult_tol = 1e-6 * max(1.0, abs(_p_multiplier), abs(_alpha_multiplier))
        if abs(_p_multiplier - _alpha_multiplier) > _mult_tol:
            _refuse(
                "P",
                "the probability and alpha exponent chains use different effective "
                f"bases ({_p_multiplier:.9g} vs {_alpha_multiplier:.9g})",
            )

        # Biased FA also stores LSE, so select Out by its independently traced pointer.
        _out_stores = []
        for _store in (s for s in _all if s.op == "tt.store"):
            if len(_store.operand_ids or []) < 2:
                continue
            _ptrs = _pointer_base_indices(_store.operand_ids[0])
            if _ptrs == {detected_out_index}:
                _out_stores.append(_store)
        if len(_out_stores) != 1:
            _refuse("Out", f"expected one store to detector arg {detected_out_index}, found {len(_out_stores)}")
        _allowed_store_args = {detected_out_index} | {entry[0] for entry in extra_stores}
        if biased is not None:
            _allowed_store_args.add(biased["lse"])
        for _store in (s for s in _all if s.op == "tt.store"):
            if not _store.operand_ids or not any(
                _pointer_base_indices(_store.operand_ids[0]) == {index} for index in _allowed_store_args
            ):
                _refuse("effect", "a store targets memory outside the proved template outputs")

        _out_passthrough = _representation | {"arith.truncf", "tt.fp_to_fp"}
        _out_loop, _out_den_id, _out_num_id = None, None, None
        _cur = _out_stores[0].operand_ids[1]
        _seen = set()
        while _cur not in _seen and len(_seen) < 64:
            _seen.add(_cur)
            _op = _by_id.get(_cur)
            if _op is None:
                _refuse("Out", "the path terminates at an unknown SSA value")
            if _op.op == "arith.divf":
                if len(_op.operand_ids or []) != 2:
                    _refuse("Out", "the normalization division does not have two inputs")
                _num = _peel_to_core(_op.operand_ids[0], _representation)
                _den = _peel_to_core(
                    _op.operand_ids[1],
                    _representation | {"tt.broadcast", "tt.expand_dims"},
                )
                _dot_pv_ids = {_dot_pv.id} | set(getattr(_dot_pv, "result_ids", None) or [])
                _singleton_direct = (
                    detected_n_ctx_index == C1
                    and not _fors
                    and _num is not None
                    and _num.id in _dot_pv_ids
                    and _den is not None
                    and _den.op == "arith.addf"
                )
                if not _singleton_direct and (_num is None or _num.op != "scf.for" or _den is not _num):
                    _refuse(
                        "Out",
                        "the division is not the canonical accumulator/denominator pair from one attention loop",
                    )
                if not _singleton_direct and not _depends_on(_num.id, _dot_pv_ids):
                    _refuse("Out", "the attention loop does not yield the verified P@V dot")
                _out_loop = None if _singleton_direct else _num
                _out_num_id = _peel_id(_op.operand_ids[0], _representation)
                _out_den_id = _peel_id(_op.operand_ids[1], _b_bcast)
                break
            if _op.op not in _out_passthrough or len(_op.operand_ids or []) != 1:
                _refuse("Out", f"unsupported operation {_op.op} appears after normalization")
            _cur = _op.operand_ids[0]
        else:
            _refuse("Out", "the path is cyclic or exceeds the verification bound")

        if dense_tail_out is not None:
            # Load padding is NOT a score mask: zero K contributes score zero,
            # and hence exp(0), through the source's last BN-wide iteration.
            # Prove the LIVE score path and the very loop that yields Out. A
            # dead select, query-only bound, or unrelated loop proves nothing.
            if load_mask_policy != "boundary" or biased is not None or mla_rope is not None:
                _refuse("Tail", "dense tail proof requested for a different route")
            _singleton_tail = detected_n_ctx_index == C1 and not _fors and _out_loop is None
            if not _singleton_tail:
                if len(_fors) != 1 or _out_loop is not _fors[0] or _kv_iv is None:
                    _refuse("Tail", "expected one identified KV loop yielding Out")
                _body_ids = {op.id for op in (_out_loop.region_ops or [])}
                if not {_dot_qk.id, _dot_pv.id} <= _body_ids:
                    _refuse("Tail", "both dots must execute directly in the KV loop")

            def _int_literal(oid, value):
                op = _by_id.get(oid)
                return (
                    op is not None
                    and op.op == "arith.constant"
                    and (op.type_str or "") in ("i32", "index")
                    and (op.attrs or {}).get("value") == value
                )

            def _length(oid):
                arg = _arg_by_id.get(oid)
                return arg is not None and not arg.is_ptr and arg.index == detected_n_ctx_index

            def _query_stop(oid):
                # Exactly (pid0 + 1) * BM, not a cone containing pid0/BM.
                op = _by_id.get(oid)
                if op is None or op.op != "arith.muli" or len(op.operand_ids or []) != 2:
                    return False
                for factor, value in (op.operand_ids, op.operand_ids[::-1]):
                    add = _by_id.get(value)
                    if not _int_literal(factor, _BM) or add is None or add.op != "arith.addi":
                        continue
                    if len(add.operand_ids or []) != 2:
                        continue
                    for one, pid in (add.operand_ids, add.operand_ids[::-1]):
                        p = _by_id.get(pid)
                        if (
                            _int_literal(one, 1)
                            and p is not None
                            and p.op == "tt.get_program_id"
                            and (p.attrs or {}).get("axis") == 0
                        ):
                            return True
                return False

            _causal_stop = False
            if not _singleton_tail:
                _lb, _ub, _step = _out_loop.operand_ids[:3]
                if not _int_literal(_lb, 0) or not _int_literal(_step, _BN):
                    _refuse("Tail", "KV loop must start at zero and step by the source BLOCK_N")
                _upper = _by_id.get(_ub)
                _causal_stop = (
                    _upper is not None
                    and _upper.op in ("arith.minsi", "arith.minui")
                    and len(_upper.operand_ids or []) == 2
                    and any(_length(a) and _query_stop(b) for a, b in (_upper.operand_ids, _upper.operand_ids[::-1]))
                )
                if not (_length(_ub) or _causal_stop):
                    _refuse("Tail", "KV loop upper bound is not N_CTX or min(query tile end, N_CTX)")

            def _mask_axis(oid):
                # The integer coordinate proves WHAT varies; this proves WHERE
                # it varies in the consuming matrix. Square scores do not expose
                # a swapped broadcast through a shape check alone.
                seen = set()
                while oid not in seen and len(seen) < 64:
                    seen.add(oid)
                    op = _by_id.get(oid)
                    if op is None or not op.operand_ids:
                        return None
                    if op.op in ("tt.broadcast", "ttg.convert_layout") and len(op.operand_ids) == 1:
                        oid = op.operand_ids[0]
                    elif op.op == "tt.expand_dims" and len(op.operand_ids) == 1:
                        axis = (op.attrs or {}).get("axis")
                        if len(_shape_of(op.operand_ids[0])) == 1 and axis in (0, 1):
                            return 1 - axis
                        return None
                    elif op.op == "arith.cmpi" and len(op.operand_ids) == 2:
                        oid = op.operand_ids[0]
                    elif op.op in ("arith.addi", "arith.muli") and len(op.operand_ids) == 2:

                        def uniform(x):
                            c = _peel_to_core(x, {"tt.broadcast", "ttg.convert_layout"})
                            return c is not None and (
                                c.op == "tt.splat"
                                or (c.op == "arith.constant" and isinstance((c.attrs or {}).get("value"), (int, float)))
                            )

                        varying = [x for x in op.operand_ids if not uniform(x)]
                        if len(varying) != 1:
                            return None
                        oid = varying[0]
                    else:
                        return None
                return None

            _tail_key_mask, _tail_causal = False, False
            _cur, _seen = _p_core.operand_ids[0], set()
            while _cur not in _dot0_ids:
                if _cur in _seen or len(_seen) >= 128:
                    _refuse("Tail", "cyclic or overlong live score path")
                _seen.add(_cur)
                _op = _by_id.get(_cur)
                if _op is None:
                    _refuse("Tail", "unknown live score value")
                if _op.op in _representation and len(_op.operand_ids or []) == 1:
                    _cur = _op.operand_ids[0]
                elif _op.op in ("arith.mulf", "arith.divf") and _op.id in _score_scale["op_ids"]:
                    _cur = next(x for x in _op.operand_ids if _depends_on_dot0(x))
                elif _op.op == "arith.select" and len(_op.operand_ids or []) == 3:
                    cond, yes, no = _op.operand_ids
                    _sentinel = _peel_to_core(no, _mask_wrappers)
                    if not _depends_on_dot0(yes) or _depends_on_dot0(no) or not _is_neg_inf_constant(_sentinel):
                        _refuse("Tail", "live score select must keep the score or supply negative infinity")
                    cmp = _peel_to_core(cond, _mask_wrappers)
                    if cmp is None or cmp.op != "arith.cmpi" or len(cmp.operand_ids or []) != 2:
                        _refuse("Tail", "unproved live score predicate")
                    a, b = cmp.operand_ids
                    pred = (cmp.attrs or {}).get("predicate_name")
                    if pred in ("slt", "ult"):
                        role = _boundary_leaf(cond, "Score", ("Q", "K"))
                        if _mask_axis(cond) != (0 if role == "Q" else 1):
                            _refuse("Tail", "score boundary is broadcast along the wrong matrix axis")
                        _tail_key_mask |= role == "K"
                    elif pred in ("sge", "uge") and _coord_ok(a, "Q") and _coord_ok(b, "K"):
                        if _mask_axis(a) != 0 or _mask_axis(b) != 1:
                            _refuse("Tail", "causal boundary has unproved query/key broadcast axes")
                        _tail_causal = True
                    else:
                        _refuse("Tail", "unreplayed live score mask")
                    _cur = yes
                else:
                    _refuse("Tail", f"unproved live score operation {_op.op}")
            if _tail_causal != dense_tail_out["causal"] or (_causal_stop and not _tail_causal):
                _refuse("Tail", "detected causal policy or loop stop disagrees with the live score path")
            _padding_contributes = not (_tail_key_mask or _tail_causal)
            _plain = []
            for role in ("K", "V"):
                load = _verified_loads[role]
                if len(load.operand_ids or []) == 1:
                    # The source reads the padded tile itself. Never invent zero
                    # fill for this role, even if its sibling has a masked load.
                    _plain.append(role)
                else:
                    _check_load_mask(load, role, exact_leaves=(role,), axes=_mask_axes(load.id, {role: 0}, role))
                    if _mask_axis(load.operand_ids[1]) != 0:
                        _refuse("Tail", f"{role} boundary is not broadcast along its load rows")

            # The same loop IV must drive the ADDRESS, not just its load mask.
            # Detection extracts stride slots but historically did not prove the
            # index multiplying each stride (e.g. K[(iv+range+32)*stride]).
            # Account for every addptr term, including scalar batch/head offsets.
            def _pid1(oid):
                op = _by_id.get(_peel_id(oid, _mask_wrappers))
                return op is not None and op.op == "tt.get_program_id" and (op.attrs or {}).get("axis") == 1

            def _batch_index(oid, slot):
                oid = _peel_id(oid, _mask_wrappers)
                h = dense_tail_out["H"]
                if h == "c1":
                    return _pid1(oid) if slot == 0 else _int_literal(oid, 0)
                op = _by_id.get(oid)
                allowed = ("arith.divsi", "arith.divui") if slot == 0 else ("arith.remsi", "arith.remui")
                return (
                    op is not None
                    and op.op in allowed
                    and len(op.operand_ids or []) == 2
                    and _pid1(op.operand_ids[0])
                    and (_arg_by_id.get(op.operand_ids[1]) is not None and _arg_by_id[op.operand_ids[1]].index == h)
                )

            for role, load in list(_verified_loads.items()) + [("Out", _out_stores[0])]:
                if role not in ("Q", "K", "V", "Out"):
                    _refuse("Tail", "unexpected dense pointer role")
                strides = dense_tail_out["strides"]["o" if role == "Out" else role.lower()]
                ptr = load.operand_ids[0]
                offsets, seen = [], set()
                while ptr not in _arg_by_id:
                    if ptr in seen or len(seen) >= 64:
                        _refuse("Tail", f"{role} has a cyclic address")
                    seen.add(ptr)
                    op = _by_id.get(ptr)
                    if op is None:
                        _refuse("Tail", f"{role} address has an unknown root")
                    if op.op in _mask_wrappers and len(op.operand_ids or []) == 1:
                        ptr = op.operand_ids[0]
                    elif op.op == "tt.addptr" and len(op.operand_ids or []) == 2:
                        offsets.append(op.operand_ids[1])
                        ptr = op.operand_ids[0]
                    else:
                        _refuse("Tail", f"{role} address contains unreplayed {op.op}")
                expected = {
                    "Q": detected_q_index,
                    "K": detected_k_index,
                    "V": detected_v_index,
                    "Out": detected_out_index,
                }[role]
                if _arg_by_id[ptr].index != expected:
                    _refuse("Tail", f"{role} address root disagrees with detection")
                shape = _shape_of(load.operand_ids[0])
                if len(shape) != 2:
                    _refuse("Tail", f"{role} address is not a matrix")
                found = set()
                for offset in offsets:
                    core = _by_id.get(_peel_id(offset, {"tt.broadcast", "ttg.convert_layout"}))
                    matches = []
                    for slot, stride in enumerate(strides):
                        indices = []
                        if stride == "c1":
                            indices.append(offset)
                        elif core is not None and core.op == "arith.muli" and len(core.operand_ids or []) == 2:
                            for a, b in (core.operand_ids, core.operand_ids[::-1]):
                                arg = _arg_by_id.get(_peel_id(a, _mask_wrappers))
                                if arg is not None and not arg.is_ptr and arg.index == stride:
                                    indices.append(b)
                        for index in indices:
                            if slot < 2:
                                ok = not _shape_of(index) and _batch_index(index, slot)
                            elif slot == 2:
                                ok = _coord_ok(index, role) and _mask_axis(index) == 0
                            else:
                                sh = self._index_shape(index, _by_id, _arg_by_id, _kv_iv, 1)
                                ok = (
                                    sh["bad"] is None
                                    and sh["mod"] is None
                                    and sh["range"] == 1
                                    and sh["bounds"] == (0, shape[1])
                                    and sh["pid"] is None
                                    and sh["iv"] == 0
                                    and _mask_axis(index) == 1
                                )
                            if ok:
                                matches.append(slot)
                    if len(matches) != 1 or matches[0] in found:
                        _refuse(
                            "Tail", f"{role} address term is not an exact, unique source coordinate times its stride"
                        )
                    found.add(matches[0])
                required = {0, 2, 3} if dense_tail_out["H"] == "c1" else {0, 1, 2, 3}
                if found != required:
                    _refuse("Tail", f"{role} address does not cover exactly the replayed batch/head/row/column terms")
            dense_tail_out.update(
                source_kv_block=_BN,
                mask_tail_scores=not _padding_contributes,
                unmasked_kv=tuple(_plain),
                source_causal_stop=_causal_stop,
            )

        # Prove the actual returned recurrence on EVERY forward caller. Belonging
        # to one loop does not prove which result is the numerator/denominator,
        # nor that its updates are the source operations the maker replays.
        # In particular, alpha*2 or max(P) used to emit unchanged canonical MSL.
        _rr_layout = {"ttg.convert_layout", "ttg.local_load", "ttg.local_alloc"}

        def _rr_id(sid):
            return _peel_id(sid, _rr_layout)

        def _rr_f32(op, shape):
            return (
                op is not None
                and _extract_shape(op.type_str or "") == tuple(shape)
                and (op.type_str or "").split(",", 1)[0].rstrip(">").endswith("xf32")
            )

        def _rr_binary(sid, kinds, shape):
            op = _by_id.get(_rr_id(sid))
            return op if (_rr_f32(op, shape) and op.op in kinds and len(op.operand_ids or []) == 2) else None

        def _rr_row(sid, target, cols):
            # A row's alpha/max must project across columns, never across rows.
            op = _by_id.get(_rr_id(sid))
            if op is not None and op.op == "tt.broadcast" and _rr_f32(op, (_BM, cols)):
                op = _by_id.get(_rr_id(op.operand_ids[0])) if len(op.operand_ids or []) == 1 else None
            return (
                op is not None
                and op.op == "tt.expand_dims"
                and _rr_f32(op, (_BM, 1))
                and (op.attrs or {}).get("axis") == 1
                and len(op.operand_ids or []) == 1
                and _rr_id(op.operand_ids[0]) == target
            )

        def _rr_reduce(sid, value, kinds):
            red = _by_id.get(_rr_id(sid))
            if (
                red is None
                or red.op != "tt.reduce"
                or not _rr_f32(red, (_BM,))
                or (red.attrs or {}).get("axis") != 1
                or len(red.operand_ids or []) != 1
                or _rr_id(red.operand_ids[0]) != value
            ):
                return False
            bids = list((red.attrs or {}).get("block_arg_ids") or [])
            returned = list((red.attrs or {}).get("return_ids") or [])
            body = {o.id: o for o in (red.region_ops or [])}
            combiner = body.get(returned[0]) if len(returned) == 1 else None
            return (
                len(bids) == 2
                and len(set(bids)) == 2
                and combiner is not None
                and combiner.op in kinds
                and combiner.type_str == "f32"
                and len(combiner.operand_ids or []) == 2
                and set(combiner.operand_ids) == set(bids)
            )

        def _rr_constant(sid, value, shape):
            op = _by_id.get(_rr_id(sid))
            if op is None or op.op != "arith.constant" or not _rr_f32(op, shape):
                return False
            actual = (op.attrs or {}).get("value")
            if value == -float("inf") and actual == 0xFF800000:
                actual = -float("inf")
            return actual == value

        _singleton_recurrence = (
            detected_n_ctx_index == C1
            and not _fors
            and dense_tail_out is not None
            and biased is None
            and mla_rope is None
        )
        if _singleton_recurrence:
            # Triton fully unrolls the one-trip KV loop at N_CTX==1.  Prove the
            # same recurrence directly rather than assuming that disappearance
            # of scf.for makes any two-dot graph canonical.
            if _out_num_id != _dot_pv.id:
                _refuse("recurrence", "singleton numerator is not the verified P@V dot result")
            _rr_max = _rr_binary(_p_max.id, {"arith.maxnumf", "arith.maximumf"}, (_BM,))
            _rr_max_ok = False
            if _rr_max is not None:
                for initial, reduced in (
                    _rr_max.operand_ids,
                    _rr_max.operand_ids[::-1],
                ):
                    if _rr_constant(initial, -float("inf"), (_BM,)) and _rr_reduce(
                        reduced,
                        _rr_id(_p_core.operand_ids[0]),
                        {"arith.maxnumf", "arith.maximumf"},
                    ):
                        _rr_max_ok = True
            if (
                not _rr_max_ok
                or _rr_id(_alpha_core.operand_ids[1]) != _p_max.id
                or not _rr_constant(_alpha_core.operand_ids[0], -float("inf"), (_BM,))
                or not _rr_row(_p_core.operand_ids[1], _p_max.id, _BN)
            ):
                _refuse("recurrence", "singleton maximum/alpha is not the exact unrolled first iteration")

            _rr_den = _rr_binary(_out_den_id, {"arith.addf"}, (_BM,))
            _rr_den_ok = False
            if _rr_den is not None:
                for scaled_initial, reduced in (
                    _rr_den.operand_ids,
                    _rr_den.operand_ids[::-1],
                ):
                    mul = _rr_binary(scaled_initial, {"arith.mulf"}, (_BM,))
                    if mul is None:
                        continue
                    alpha_and_zero = any(
                        _rr_id(alpha) == _alpha_exps[0].id and _rr_constant(zero, 0.0, (_BM,))
                        for alpha, zero in (mul.operand_ids, mul.operand_ids[::-1])
                    )
                    if alpha_and_zero and _rr_reduce(reduced, _p_exp.id, {"arith.addf"}):
                        _rr_den_ok = True
            if not _rr_den_ok:
                _refuse("recurrence", "singleton denominator is not 0 * alpha + returned sum(P)")

            if len(_dot_pv.operand_ids or []) != 3:
                _refuse("recurrence", "singleton P@V dot has no proved accumulator")
            acc = _peel_to_core(
                _dot_pv.operand_ids[2],
                _rr_layout | {"tt.broadcast", "tt.expand_dims"},
            )
            if (
                acc is None
                or acc.op != "arith.mulf"
                or len(acc.operand_ids or []) != 2
                or not any(
                    _rr_row(alpha, _alpha_exps[0].id, 1) and _rr_constant(zero, 0.0, (_BM, 1))
                    for alpha, zero in (acc.operand_ids, acc.operand_ids[::-1])
                )
            ):
                _refuse("recurrence", "singleton P@V accumulator is not 0 * row-broadcast(alpha)")
            dense_tail_out["initial_values"] = (0.0, 0.0)
            return _p_multiplier

        _rr_rids = list(getattr(_out_loop, "result_ids", None) or [])
        _rr_bids = list((_out_loop.attrs or {}).get("block_arg_ids") or [])
        _rr_inits = list(_out_loop.operand_ids or [])[3:]
        _rr_yields = [o for o in (_out_loop.region_ops or []) if o.op == "scf.yield"]
        if (
            len(_rr_yields) != 1
            or len(_rr_bids) != len(_rr_rids) + 1
            or len(_rr_inits) != len(_rr_rids)
            or len(_rr_yields[0].operand_ids or []) != len(_rr_rids)
            or _out_num_id not in _rr_rids
            or _out_den_id not in _rr_rids
            or _out_num_id == _out_den_id
        ):
            _refuse("recurrence", "Out does not select two distinct, completely recorded loop results")
        _rr_yids = [_rr_id(x) for x in _rr_yields[0].operand_ids]
        _rr_a, _rr_l = _rr_rids.index(_out_num_id), _rr_rids.index(_out_den_id)
        _rr_m_candidates = [i for i, x in enumerate(_rr_yids) if x == _p_max.id]
        if len(_rr_m_candidates) != 1 or _rr_m_candidates[0] in (_rr_a, _rr_l):
            _refuse("recurrence", "the loop does not yield the exact online maximum in its own slot")
        _rr_m = _rr_m_candidates[0]
        _rr_marg, _rr_larg, _rr_aarg = (_rr_bids[i + 1] for i in (_rr_m, _rr_l, _rr_a))
        _rr_initial_values = {}
        for index, value in ((_rr_m, -float("inf")), (_rr_l, 0.0), (_rr_a, 0.0)):
            initial = _by_id.get(_rr_id(_rr_inits[index]))
            initial_value = (initial.attrs or {}).get("value") if initial is not None else None
            # MLIR's dense hexadecimal f32 -inf is retained as its bit pattern
            # by the walker; do not decode an arbitrary integer as a float.
            if value == -float("inf") and initial_value == 0xFF800000:
                initial_value = -float("inf")
            replay_finite = (
                index != _rr_m
                and dense_tail_out is not None
                and (type(initial_value) is float or initial_value == 0)
                and _math.isfinite(initial_value)
            )
            if (
                initial is None
                or initial.op != "arith.constant"
                or initial.elem_type != "f32"
                or (initial_value != value and not replay_finite)
            ):
                _refuse("recurrence", "the max/denominator/numerator initial values are not the replayed -inf/0/0")
            _rr_initial_values[index] = float(initial_value)
        if dense_tail_out is not None:
            # Replay finite initializers explicitly. Do not over-refuse a
            # numerically correct init-one spelling, or assume alpha erases it
            # without proving every empty/nonfinite-loop case.
            dense_tail_out["initial_values"] = (_rr_initial_values[_rr_a], _rr_initial_values[_rr_l])

        _rr_max = _rr_binary(_p_max.id, {"arith.maxnumf", "arith.maximumf"}, (_BM,))
        if (
            _rr_max is None
            or _rr_id(_alpha_core.operand_ids[0]) != _rr_marg
            or not _rr_row(_p_core.operand_ids[1], _p_max.id, _BN)
        ):
            _refuse("recurrence", "the maximum/alpha pair or its row projection is not the carried source value")
        if not any(
            _rr_id(a) == _rr_marg and _rr_reduce(b, _rr_id(_p_core.operand_ids[0]), {"arith.maxnumf", "arith.maximumf"})
            for a, b in (_rr_max.operand_ids, _rr_max.operand_ids[::-1])
        ):
            _refuse("recurrence", "the online maximum is not max(carried max, returned row-max(score))")
        _rr_den = _rr_binary(_rr_yids[_rr_l], {"arith.addf"}, (_BM,))
        _rr_den_ok = False
        if _rr_den is not None:
            for a, b in (_rr_den.operand_ids, _rr_den.operand_ids[::-1]):
                mul = _rr_binary(a, {"arith.mulf"}, (_BM,))
                if (
                    mul is not None
                    and {_rr_id(x) for x in mul.operand_ids} == {_rr_larg, _alpha_exps[0].id}
                    and _rr_reduce(b, _p_exp.id, {"arith.addf"})
                ):
                    _rr_den_ok = True
        if not _rr_den_ok:
            _refuse("recurrence", "the denominator is not carried denominator * alpha + returned sum(P)")
        if _rr_yids[_rr_a] != _dot_pv.id or len(_dot_pv.operand_ids or []) != 3:
            _refuse("recurrence", "the numerator is not the verified P@V dot result")
        _rr_pv_shape = _extract_shape(_dot_pv.type_str or "")
        if len(_rr_pv_shape) != 2 or _rr_pv_shape[0] != _BM:
            _refuse("recurrence", "the numerator has an unproved output shape")
        _rr_acc = _rr_binary(_dot_pv.operand_ids[2], {"arith.mulf"}, _rr_pv_shape)
        if _rr_acc is None or not any(
            _rr_id(a) == _rr_aarg and _rr_row(b, _alpha_exps[0].id, _rr_pv_shape[1])
            for a, b in (_rr_acc.operand_ids, _rr_acc.operand_ids[::-1])
        ):
            _refuse("recurrence", "the P@V accumulator is not carried numerator * row-broadcast(alpha)")

        # ---- biased route (packet 114): the max reduce reads the certified select; the Lse value ----
        if biased is not None:
            # the row max the softmax subtracts is the max over the SAME boundary-select result
            _reds = [_by_id.get(_peel_id(x, _b_bcast)) for x in (_p_max.operand_ids or [])]
            _reds = [r for r in _reds if r is not None and r.op == "tt.reduce"]
            if len(_reds) != 1 or _peel_id(_reds[0].operand_ids[0], _b_wr) != _biased_S2_id:
                _refuse("Score", "the online-softmax row max is not reduced from the certified boundary select")
            if _out_loop is None or _out_den_id is None:
                _refuse("Lse", "the attention loop was not identified by the Out trace")
            _lse_st = [
                s
                for s in _all
                if s.op == "tt.store" and s.operand_ids and _pointer_base_indices(s.operand_ids[0]) == {biased["lse"]}
            ]
            if len(_lse_st) != 1:
                _refuse("Lse", f"expected exactly one store of arg {biased['lse']}, found {len(_lse_st)}")
            _lv = _by_id.get(_peel_id(_lse_st[0].operand_ids[1], _b_wr))
            if _lv is None or _lv.op != "arith.addf" or len(_lv.operand_ids or []) != 2:
                _refuse(
                    "Lse", "the stored value is not `max * ln(base) + log(denominator)` (an epilogue is not replayed)"
                )
            _parts = [(x, _by_id.get(_peel_id(x, _b_wr))) for x in _lv.operand_ids]
            _logs = [(x, o) for x, o in _parts if o is not None and o.op == "math.log"]
            if len(_logs) != 1 or len(_logs[0][1].operand_ids or []) != 1:
                _refuse("Lse", "the stored value does not add exactly one natural log")
            _R_l = _peel_id(_logs[0][1].operand_ids[0], _b_wr)
            _other = next(x for x, o in _parts if (x, o) != _logs[0])
            _c_eff = float(biased.get("c_eff", 1.0))
            if abs(_c_eff - 1.0) > 1e-9:
                _mo = _by_id.get(_peel_id(_other, _b_wr))
                if _mo is None or _mo.op != "arith.mulf" or len(_mo.operand_ids or []) != 2:
                    _refuse("Lse", "the max term is not converted by a constant multiply")
                _cs = [(_const_value(x), x) for x in _mo.operand_ids]
                _consts = [(v, x) for v, x in _cs if v is not None]
                _want = 1.0 / _c_eff
                if len(_consts) != 1 or abs(_consts[0][0] - _want) > 1e-6 * abs(_want):
                    _refuse("Lse", "the max conversion constant is not 1/c (the base conversion the template applies)")
                _R_m = _peel_id(next(x for v, x in _cs if v is None), _b_wr)
            else:
                _R_m = _peel_id(_other, _b_wr)
            _rids = list(getattr(_out_loop, "result_ids", None) or [])
            if _R_l not in _rids or _R_m not in _rids or _R_l == _R_m:
                _refuse("Lse", "its operands are not two distinct results of the verified attention loop")
            if _R_l != _out_den_id:
                _refuse("Lse", "the log operand is not the Out normalization denominator")
            _i_m, _i_l = _rids.index(_R_m), _rids.index(_R_l)
            _ys = [y for y in (_out_loop.region_ops or []) if y.op == "scf.yield"]
            if len(_ys) != 1 or max(_i_m, _i_l) >= len(_ys[0].operand_ids or []):
                _refuse("Lse", "the attention loop does not have one yield covering its results")
            if _peel_id(_ys[0].operand_ids[_i_m], _b_wr) != _p_max.id:
                _refuse("Lse", "the max result is not the verified online-softmax maximum")
            _bids = list((_out_loop.attrs or {}).get("block_arg_ids") or [])
            _barg_l = _bids[_i_l + 1] if _i_l + 1 < len(_bids) else None
            _yl = _by_id.get(_peel_id(_ys[0].operand_ids[_i_l], _b_wr))
            if _yl is None or _yl.op != "arith.addf" or len(_yl.operand_ids or []) != 2:
                _refuse("Lse", "the denominator recurrence is not `l * alpha + sum(P)`")
            _yparts = [_by_id.get(_peel_id(x, _b_wr)) for x in _yl.operand_ids]
            _mul = [o for o in _yparts if o is not None and o.op == "arith.mulf"]
            _red = [o for o in _yparts if o is not None and o.op == "tt.reduce"]
            if len(_mul) != 1 or len(_red) != 1:
                _refuse("Lse", "the denominator recurrence is not `l * alpha + sum(P)`")
            _alpha_ids = {_alpha_exps[0].id} | set(getattr(_alpha_exps[0], "result_ids", None) or [])
            _mul_src = {_peel_id(x, _b_bcast) for x in _mul[0].operand_ids}
            if _barg_l is None or _barg_l not in _mul_src or not (_mul_src & _alpha_ids) or len(_mul_src) != 2:
                _refuse("Lse", "the denominator rescale is not the carried denominator times the verified alpha")
            _p_ids = {_p_exp.id} | set(getattr(_p_exp, "result_ids", None) or [])
            _red_is_sum = any(o.op == "arith.addf" for o in (_red[0].region_ops or []))
            if not _red_is_sum or _peel_id(_red[0].operand_ids[0], _b_wr) not in _p_ids:
                _refuse("Lse", "the denominator sum is not the sum of the verified probabilities")

        return _p_multiplier

    def _lower_flash_attention_template(self, info: dict) -> str:
        """Emit the head-dim-tiled FA2 MSL (Task 1) for a detected FA kernel.

        Called from ``lower()`` only for validated configs: the head_dim=128
        fp32/fp16 routes and the uniform-bf16 head_dim=64, 32x32 tiled route.
        ``info`` comes from
        ``_detect_flash_attention`` (which refuses on any ambiguity).

        ABI reconciliation (the crux). The launcher binds the flat non-constexpr
        arg list positionally: ``kargs[N] -> buffer(N)``. So the emitted MSL's
        ``[[buffer(N)]]`` declarations are built DIRECTLY from ``self.graph.args``
        (the ground truth for binding order), one buffer per present arg, using
        each arg's own name. Triton's ``equal_to_1`` specialization elides any
        runtime arg whose value is 1 (a contiguous tensor's innermost stride;
        Z/H when batch/heads == 1), so the real kernel has FEWER than the
        idealized 23 args — and a varying count depending on the data. The
        detector (``info``) reports, per logical stride/dim, either the runtime
        arg INDEX or the marker ``"c1"`` (folded to 1); we map those to MSL
        expressions: the arg's name (a buffer) or the literal ``1u``. The real
        kernel computes ``qk_scale = 1/sqrt(HEAD_DIM)`` internally — there is NO
        scale arg — so the template BAKES the scale as a constant.
        """
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen.msl_types import triton_type_to_msl
        from triton_msl.codegen._msl_templates import (
            make_flash_attention_kernel_tiled,
            make_flash_attention_kernel_simdgroup,
            flash_attention_tiled_threads,
        )

        head_dim = info["head_dim"]
        v_head_dim = _v_head_dim_of(info)  # != head_dim for asymmetric MLA
        block_m = info["block_m"]
        block_n = info["block_n"]
        C1 = "c1"

        # COORDINATED REFUSE (the chokepoint's FA exemption lands here). The dispatch
        # gate ``_refuse_nontrivial_template_output_mask`` deliberately lets FA through —
        # head_dim<=64 routes to the mask-HONORING scalar/generic ``_lower_store``. But
        # BOTH head_dim=128 FA templates emitted below (simdgroup AND tiled) take NO mask
        # operand: they compute the FULL output tile, gate writes only on the N_CTX
        # boundary, and would SILENTLY DROP a tighter user output ``tt.store`` mask
        # (rm < BOUND with BOUND < N_CTX, or a value mask), clobbering the masked-off
        # rows. So the mask-dropping templates refuse such a mask THEMSELVES here, using
        # the SAME shared triviality check as the chokepoint (so the two cannot diverge).
        # A trivially-true tile-boundary mask (om < N_CTX) / the no-mask case pass through.
        if self._template_output_mask_nontrivial(is_fa=True, fa_ctx_index=info.get("query_length", info.get("N_CTX"))):
            raise MetalNonRecoverableError(
                f"FlashAttention (head_dim={head_dim}) with a non-tile-boundary output store "
                "mask is not supported: the simdgroup / tiled FA templates compute the "
                "FULL output tile and gate writes only on the N_CTX boundary, silently "
                "DROPPING any tighter store mask (e.g. tl.store(o, acc, mask=om < BOUND) "
                "with BOUND < N_CTX, or a value mask). Refusing to emit silently-wrong "
                "output. Use a full-tile store (the tile-boundary mask om < N_CTX is "
                "honored automatically by the template boundary), or apply the partial-"
                "output mask in a separate elementwise kernel.",
                op_name="tt.dot",
            )

        # The four pointer roles must be the first four args in canonical order
        # (Q,K,V,Out = 0,1,2,3) — the template hard-codes Q/K/V/Out at buffers
        # 0..3. The detector already required four DISTINCT roles; enforce the
        # ORDER the template assumes (a permuted order would bind wrongly).
        if not (info["q"] == 0 and info["k"] == 1 and info["v"] == 2 and info["out"] == 3):
            raise MetalNonRecoverableError(
                "FlashAttention recognized but Q/K/V/Out are not the first four "
                f"kernel args in order (got q={info['q']},k={info['k']},"
                f"v={info['v']},out={info['out']}). Refusing to bind buffers in "
                "the wrong order rather than risk silently-wrong output."
            )

        # Defense-in-depth (carry item 2): cross-check Q/K/Out roles against the
        # generic dot-pointer resolver. dot 0 is QK^T (operands Q, K) and the
        # single store targets Out, so _resolve_dot_ptr_roles(dot0, ...) returns
        # [Q, K, Out, <extras incl. V>]. V is cross-checked separately below
        # through dot 1 — see the V verification block.
        def _fa_walk(ops):
            for s in ops:
                yield s
                if getattr(s, "region_ops", None):
                    yield from _fa_walk(s.region_ops)
                if getattr(s, "else_ops", None):
                    yield from _fa_walk(s.else_ops)

        _all = list(_fa_walk(self.graph.ops))
        _dots = [s for s in _all if s.op == "tt.dot"]
        _order = {s.id: i for i, s in enumerate(_all)}
        _dot_qk = sorted(_dots, key=lambda d: _order[d.id])[0]
        _ptr_args = [a for a in self.graph.args if a.is_ptr]
        roles = self._resolve_dot_ptr_roles(_dot_qk, _ptr_args)
        if roles is not None and len(roles) >= 3:
            r_idx = {"q": roles[0].index, "k": roles[1].index, "out": roles[2].index}
            if any(r_idx[k] != info[k] for k in ("q", "k", "out")):
                raise MetalNonRecoverableError(
                    "FlashAttention Q/K/Out pointer roles disagree between the FA "
                    "detector and the generic dot-pointer resolver "
                    f"(detector q={info['q']},k={info['k']},out={info['out']}; "
                    f"resolver q={r_idx['q']},k={r_idx['k']},out={r_idx['out']}). "
                    "Refusing rather than risk a mis-bound buffer."
                )

        _dense_tail = {"causal": info["causal"], "H": info["H"], "strides": info["strides"]}
        info["scale"] *= self._fa_verify_value_paths(
            detected_q_index=info["q"],
            detected_k_index=info["k"],
            detected_v_index=info["v"],
            detected_out_index=info["out"],
            q_scale_required=info.get("q_scale", True),
            k_transposes=1,
            verify_score_path=info.get("verify_score_path", True),
            expected_score_scale=info.get("score_scale"),
            detected_n_ctx_index=info.get("N_CTX"),
            detected_query_index=info.get("query_length"),
            load_mask_policy="boundary",  # packet 110: the simd template gates at N_CTX itself
            q_rows=info["block_m"],
            kv_cols=info["block_n"],
            dense_tail_out=_dense_tail,
        )

        self._fa_unmasked_store_rows(info.get("query_length", info.get("N_CTX")), block_m)

        # Build the [[buffer(N)]] declarations from the ACTUAL arg list so the
        # binding order matches the launcher exactly: pointers as device buffers,
        # scalars as constant uint& (every present stride/Z/H/N_CTX is i32).
        # Pointers KEEP the names Q/K/V/Out the template body uses (buffers
        # 0..3). Scalar arg NAMES are positional (``fa_arg{i}``) and deliberately
        # NOT the triton names — those (``N_CTX``, ``Z``, ``H`` ...) would collide
        # with the template's logical-alias locals. Binding is by buffer INDEX,
        # so the scalar name is free. A index->msl-name map resolves logical dims.
        ptr_names = {info["q"]: "Q", info["k"]: "K", info["v"]: "V", info["out"]: "Out"}
        args = self.graph.args
        arg_decls = []
        name_by_index = {}
        for i, a in enumerate(args):
            if i != a.index:
                # Arg list must be densely 0-indexed by position for the
                # positional buffer binding to be valid.
                raise MetalNonRecoverableError(
                    "FlashAttention arg list is not densely indexed; refusing rather than risk a mis-bound buffer."
                )
            if a.is_ptr:
                m = triton_type_to_msl(a.elem_type)
                qual = "device const" if i != info["out"] else "device"
                arg_decls.append(f"    {qual} {m}* {ptr_names[i]} [[buffer({i})]]")
                name_by_index[i] = ptr_names[i]
            else:
                buf_name = f"fa_arg{i}"
                arg_decls.append(f"    constant uint& {buf_name} [[buffer({i})]]")
                name_by_index[i] = buf_name

        unsigned_args = set()

        def _expr(entry):
            """Map a detector stride/dim entry (arg index or 'c1') to an MSL
            expression: the buffer arg's name, or the literal 1u when folded."""
            if entry == C1:
                return "1u"
            if isinstance(entry, int) and entry in name_by_index:
                if not args[entry].is_ptr:
                    unsigned_args.add(entry)
                return name_by_index[entry]
            # Unresolvable — must not silently bake a wrong value.
            raise MetalNonRecoverableError(
                "FlashAttention stride/dim could not be mapped to a kernel arg; "
                "refusing rather than risk silently-wrong output."
            )

        qs, ks, vs, os_ = (info["strides"]["q"], info["strides"]["k"], info["strides"]["v"], info["strides"]["o"])
        bindings = {
            "q_sz": _expr(qs[0]),
            "q_sh": _expr(qs[1]),
            "q_sm": _expr(qs[2]),
            "q_sk": _expr(qs[3]),
            "k_sz": _expr(ks[0]),
            "k_sh": _expr(ks[1]),
            "k_sn": _expr(ks[2]),
            "k_sk": _expr(ks[3]),
            "v_sz": _expr(vs[0]),
            "v_sh": _expr(vs[1]),
            "v_sn": _expr(vs[2]),
            "v_sk": _expr(vs[3]),
            "o_sz": _expr(os_[0]),
            "o_sh": _expr(os_[1]),
            "o_sm": _expr(os_[2]),
            "o_sk": _expr(os_[3]),
            "Z": _expr(info["Z"]),
            "H": _expr(info["H"]),
            "N_CTX": _expr(info["N_CTX"]),
        }

        # Emit under the kernel's real (sanitized) name so it matches
        # metadata["name"] the driver loads from the metallib (emit_msl sets
        # name = _sanitize_msl_name(graph.func_name) before lower()).
        # out_dtype comes from the detector as an MLIR spelling. fp16 emits half
        # Q/K/V/Out; the narrow native-bf16 route emits bfloat Q/K/V/Out. The
        # arg_decls above use each argument's proved real element type, and both
        # tiled paths promote on load, compute in fp32 and cast on store.
        if _simd_fa_eligible(info) and "query_length" not in info:
            from triton_msl.codegen._fa_fragments import scope_reason, validate_mode

            # The simdgroup template stages Q through a scalar cast but loads K and V and
            # stores Out as matrices of ONE element type (out_dtype). A K or V of another
            # float type, or a bfloat Q beside half tiles, has no matching matrix load:
            # name the refusal here instead of surfacing the Metal compiler's error.
            _spell = {"fp16": "f16", "half": "f16", "fp32": "f32", "float": "f32", "bfloat16": "bf16"}
            _role_types = {
                r: _spell.get(str(args[info[r]].elem_type), str(args[info[r]].elem_type))
                for r in ("q", "k", "v", "out")
            }
            if len({_role_types[r] for r in ("k", "v", "out")}) != 1 or (
                _role_types["q"] == "bf16" and _role_types["out"] == "f16"
            ):
                raise MetalNonRecoverableError(
                    "FlashAttention: the source roles have element types "
                    f"Q={_role_types['q']}, K={_role_types['k']}, V={_role_types['v']}, "
                    f"Out={_role_types['out']} but the simdgroup template loads K and V and stores "
                    "Out as one matrix element type; refusing a mixed ABI rather than emit a "
                    "shader that does not compile.",
                    op_name="tt.dot",
                )
            requested = validate_mode(getattr(self.options, "experimental_fa_fragment_lowering", "off"))
            half_accumulate = _fa_half_accumulate(info["out_dtype"])
            rounding = getattr(self, "_fa_rounding", {})
            reason = scope_reason(
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                out_dtype=info["out_dtype"],
                causal=info["causal"],
                half_accumulate=half_accumulate,
                scale=info["scale"],
                round_q=rounding.get("round_q"),
                round_p=rounding.get("round_p"),
                scale_chain=rounding.get("scale_chain", ()),
                source_kv_block=_dense_tail["source_kv_block"],
                mask_tail_scores=_dense_tail["mask_tail_scores"],
                unmasked_kv=_dense_tail["unmasked_kv"],
                source_causal_stop=_dense_tail["source_causal_stop"],
                initial_values=_dense_tail["initial_values"],
            )
            if info.get("is_mla"):
                reason = "source-family"
            if not all(args[info[r]].elem_type in ("f16", "fp16") for r in ("q", "k", "v", "out")):
                reason = "role-types"
            selected = requested if reason is None else "off"
            self._fa_fragment_lowering = {
                "requested": requested,
                "selected": selected,
                "reason": "disabled" if requested == "off" else (reason or "selected"),
            }
            msl = make_flash_attention_kernel_simdgroup(
                head_dim,
                32,
                64,
                causal=info["causal"],
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                scale=info["scale"],
                v_head_dim=v_head_dim,
                half_accumulate=half_accumulate,
                fragment_lowering=selected,
                source_kv_block=_dense_tail["source_kv_block"],
                mask_tail_scores=_dense_tail["mask_tail_scores"],
                unmasked_kv=_dense_tail["unmasked_kv"],
                source_causal_stop=_dense_tail["source_causal_stop"],
                initial_values=_dense_tail["initial_values"],
            )
            # 256 threads/threadgroup (8 SIMD groups x 32 threads/group).
            self.effective_block_size = 256
            # Route this kernel through the zero-copy compile_shader path with its
            # NATIVE 2-D grid (dispatch bug fix: the 2-D grid otherwise falls to the
            # ~3x-slower host-roundtrip path). Fail-OPEN in the driver -- the host
            # path is equally correct, so a miss just falls through. Carry the MSL +
            # threadgroup size so the dispatch needs nothing from the stash.
            self._flash_attention = ("flash_attention", msl, 256)
        else:
            # The tiled/scalar template is SYMMETRIC-only. Asymmetric MLA (qk != v)
            # only reaches this method when simd-eligible (see the FA router), so it
            # always takes the simd branch above; refuse defensively rather than let
            # the symmetric tiled template silently mis-compute an asymmetric shape.
            if v_head_dim != head_dim:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "asymmetric-head_dim (MLA) FlashAttention requires the simdgroup "
                    "template (contiguous, fp16/bf16); the scalar/tiled fallback is "
                    "symmetric-only. Refusing rather than mis-compute."
                )
            msl = make_flash_attention_kernel_tiled(
                head_dim,
                block_m,
                block_n,
                # Source BN is unchanged; only resident query storage and D
                # staging shrink. The maker checks the exact five-array budget.
                # 32x64/Dc32 = 24832 bytes; 32x32/Dc64 = 20736 bytes.
                Dc=32 if head_dim == 64 and block_n == 64 else 64,
                causal=info["causal"],
                out_dtype=info["out_dtype"],
                arg_decls=arg_decls,
                bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name),
                scale=info["scale"],
                source_kv_block=_dense_tail["source_kv_block"],
                mask_tail_scores=_dense_tail["mask_tail_scores"],
                unmasked_kv=_dense_tail["unmasked_kv"],
                source_causal_stop=_dense_tail["source_causal_stop"],
                query_tile=32 if head_dim == 64 and block_m >= 32 and (block_m, block_n) != (32, 32) else None,
                initial_values=_dense_tail["initial_values"],
                query_length=_expr(info["query_length"]) if "query_length" in info else None,
                round_q=getattr(self, "_fa_rounding", {}).get("round_q") if "query_length" in info else None,
                round_p=getattr(self, "_fa_rounding", {}).get("round_p") if "query_length" in info else None,
            )
            # Match every cooperative loop's physical stride, even for >1024 scores.
            self.effective_block_size = flash_attention_tiled_threads(block_m, block_n)
        self._attention_unsigned_scalars(unsigned_args)
        # The kernel uses program_id(0) (q-block) AND program_id(1) (z*h); force
        # the 2-D threadgroup grid (n_q_blocks, Z*H, 1) so the template's pid3.x /
        # pid3.y resolve correctly (emit_msl derives needs_2d_grid from this).
        self._used_pid_axes = {0, 1}
        # Mark the Out pointer as the (only) output for precise copy-back.
        self._prescan_stores()
        return msl

    def _find_op_type_str(self, ssa_id: int) -> str:
        """Find the type_str for an SSA value by searching ops, recursing
        into nested scf regions (region_ops = body/then, else_ops = while
        body/else) at any depth. Returns "" if not found."""

        def _search(ops):
            for ssa in ops:
                if ssa.id == ssa_id:
                    return ssa.type_str or ""
                body = list(ssa.region_ops or []) + list(ssa.else_ops or [])
                if body:
                    r = _search(body)
                    if r is not None:
                        return r
            return None

        r = _search(self.graph.ops)
        if r is not None:
            return r
        # Check args
        for arg in self.graph.args:
            if arg.id == ssa_id:
                return arg.type_str
        return ""

    # -- Concatenation --

    def _lower_tt_join(self, ssa: SSAValue):
        """tt.join → fused cat (join + trans + reshape = concatenation).

        In Triton, tl.cat(a, b, can_reorder=False) compiles to:
          tt.join(a, b) → tensor<Nx2>
          tt.trans → tensor<2xN>
          tt.reshape → tensor<2*N>

        The result is concatenation: [a[0]..a[N-1], b[0]..b[N-1]].

        Since the kernel runs in 2D mode (due to intermediate 2D shapes),
        make_range(0, N) maps to lid % N, so all 2*N threads have valid
        loaded values. The fused cat is simply:
          result = (lid < N) ? a_val : b_val
        """
        if len(ssa.operand_ids) < 2:
            self.kb.unsupported("UNSUPPORTED: tt.join with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N from operand shape
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

        # Threadgroup-cap guard. The output is 2*N. With 2*N > 1024 only 1024 threads
        # dispatch (hard cap), so the `(lid < N) ? a : b` select is always-true and the
        # upper 1024 outputs are never written (re-audit #2 silent-wrong). tt.join is
        # in has_barrier_ops so 2*N <= 1024 dispatches one element per thread; refuse above.
        if 2 * N > 1024:
            raise MetalNonRecoverableError(
                f"tt.join / tl.cat producing {2 * N} elements exceeds the 1024-thread "
                f"threadgroup: only 1024 threads dispatch, so the upper half is never "
                f"written. Refusing rather than mis-compute.",
                op_name="tt.join",
            )

        # Detect the join → trans → reshape pattern
        trans_ssa = None
        reshape_ssa = None
        for op in self.graph.ops:
            if op.op == "tt.trans" and ssa.id in op.operand_ids:
                trans_ssa = op
                break
        if trans_ssa:
            for op in self.graph.ops:
                if op.op == "tt.reshape" and trans_ssa.id in op.operand_ids:
                    reshape_ssa = op
                    break

        # Determine type
        input_dtype = self.env_types.get(a_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        if input_dtype == "fp16":
            msl_type = "half"
        elif input_dtype == "bf16":
            msl_type = "bfloat"
        elif input_dtype in ("i64", "u64", "ui64"):
            # 64-bit ints must keep full width — int/i32 silently drops the high word
            # (Triton-lens re-audit 2026-06-25, the join/cat dtype-default sibling).
            msl_type = "long" if input_dtype == "i64" else "ulong"

        result_var = self._next_var("cat")

        if trans_ssa and reshape_ssa:
            # Fused cat: one element per thread (tt.join in has_barrier_ops), 2*N<=1024
            # guaranteed by the guard above. Use _lid_expr (== lid here) for the select.
            self.kb.raw_line(f"    {msl_type} {result_var} = ({self._lid_expr} < {N}u) ? {a_var} : {b_var};")
            # Register result for all intermediate SSA ids
            self.env[ssa.id] = result_var
            self.env_types[ssa.id] = input_dtype
            self.env[trans_ssa.id] = result_var
            self.env_types[trans_ssa.id] = input_dtype
            self.env[reshape_ssa.id] = result_var
            self.env_types[reshape_ssa.id] = input_dtype
            # Skip the trans and reshape ops
            self._skip_ids.add(trans_ssa.id)
            self._skip_ids.add(reshape_ssa.id)
        else:
            # Standalone join (no trans+reshape) — use shared memory
            shared_a = f"join_shared_a_{self._shared_counter}"
            shared_b = f"join_shared_b_{self._shared_counter}"
            self._shared_counter += 1
            shared_dtype = (
                "i64"
                if input_dtype == "i64"
                else "u64"
                if input_dtype in ("u64", "ui64")
                else "fp32"
                if is_float
                else "i32"
            )
            self.kb.declare_threadgroup_array(shared_a, dtype=shared_dtype, size=N)
            self.kb.declare_threadgroup_array(shared_b, dtype=shared_dtype, size=N)

            # Stage input values to shared memory — use lid (not loop var)
            # since we need all N values staged before any interleaving.
            # Each thread loads one element (lid < N).
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {shared_a}[lid] = {a_var};")
            self.kb.raw_line(f"        {shared_b}[lid] = {b_var};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # Interleave: join[i, 0] = a[i], join[i, 1] = b[i]
            # Linear (row-major): join[e] = a[e/2] if e%2==0, b[e/2] if e%2==1
            # Use _lid_expr so the loop variable covers all 2*N output elements
            elem = self._lid_expr
            self.kb.raw_line(
                f"    {msl_type} {result_var} = ({elem} % 2u == 0u) ? "
                f"{shared_a}[{elem} / 2u] : {shared_b}[{elem} / 2u];"
            )
            self.env[ssa.id] = result_var
            self.env_types[ssa.id] = input_dtype
            dst_shape = _extract_shape(ssa.type_str)
            if dst_shape:
                self.env_shapes[ssa.id] = dst_shape

    def _lower_tt_cat(self, ssa: SSAValue):
        """tt.cat → concatenation using shared memory.

        In Triton, tl.cat(a, b, can_reorder=True) may compile directly to
        tt.cat(a, b) → tensor<2*N>. The kernel is 1D with block_size=2*N.

        Since make_range(0, N) maps to lid (1D mode), only threads 0..N-1
        have valid loaded values. Use shared memory to stage and redistribute.
        """
        if len(ssa.operand_ids) < 2:
            self.kb.unsupported("UNSUPPORTED: tt.cat with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

        # Threadgroup-cap guard. The output is 2*N, staged one element per thread
        # (`if (lid < N) shared[lid] = ...`). With tt.cat in has_barrier_ops the
        # kernel dispatches 2*N threads (one element per thread, no wrap loop), but
        # Apple caps a threadgroup at 1024, so 2*N > 1024 can't be covered -> the
        # high half is left unwritten / OOB (re-audit #2 silent-wrong). Refuse.
        if 2 * N > 1024:
            raise MetalNonRecoverableError(
                f"tt.cat producing {2 * N} elements exceeds the 1024-thread "
                f"threadgroup: the one-element-per-thread shared-memory concat can't "
                f"cover it (high half unwritten). Refusing rather than mis-compute.",
                op_name="tt.cat",
            )

        # Determine type
        input_dtype = self.env_types.get(a_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"
        if input_dtype == "fp16":
            msl_type = "half"
            shared_dtype = "fp16"
        elif input_dtype == "bf16":
            msl_type = "bfloat"
            shared_dtype = "bf16"
        elif input_dtype in ("i64", "u64", "ui64"):
            # 64-bit ints must keep full width — int/i32 silently drops the high word
            # (Triton-lens re-audit 2026-06-25, the join/cat dtype-default sibling).
            msl_type = "long" if input_dtype == "i64" else "ulong"
            shared_dtype = "i64" if input_dtype == "i64" else "u64"

        # Allocate shared memory for both halves
        shared_a = f"cat_shared_a_{self._shared_counter}"
        shared_b = f"cat_shared_b_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_a, dtype=shared_dtype, size=N)
        self.kb.declare_threadgroup_array(shared_b, dtype=shared_dtype, size=N)

        result_var = self._next_var("cat")

        # Stage: only threads 0..N-1 have valid loaded values
        self.kb.raw_line(f"    if (lid < {N}u) {{")
        self.kb.raw_line(f"        {shared_a}[lid] = {a_var};")
        self.kb.raw_line(f"        {shared_b}[lid] = {b_var};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read: cat[lid] = a[lid] for lid < N, b[lid-N] for lid >= N
        self.kb.raw_line(f"    {msl_type} {result_var} = (lid < {N}u) ? {shared_a}[lid] : {shared_b}[lid - {N}u];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = input_dtype
        dst_shape = _extract_shape(ssa.type_str)
        if dst_shape:
            self.env_shapes[ssa.id] = dst_shape

    def _lower_tt_split(self, ssa: SSAValue):
        """tt.split → de-interleave using shared memory.

        Takes tensor<Nx2> and produces two tensor<N> results.
        split[i, 0] = z1[i], split[i, 1] = z2[i].

        In per-thread model with 2*N threads, each thread has one value
        from the flat input. Stage to shared memory, then read even/odd.
        """
        if not ssa.operand_ids:
            self.kb.unsupported("UNSUPPORTED: tt.split with no operands")
            return
        if not ssa.result_ids or len(ssa.result_ids) < 2:
            self.kb.unsupported("UNSUPPORTED: tt.split with < 2 results")
            return

        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)

        # Get input shape (N, 2)
        src_shape = _extract_shape(self._find_op_type_str(src_id))
        if src_shape and len(src_shape) >= 2:
            N = src_shape[0]
        else:
            N = self.effective_block_size // 2

        total = N * 2

        # (re-audit #8, FIXED 2026-08-26) This used to refuse unconditionally: the
        # de-interleave gives thread lid the pair at row lid, which is RIGHT, but the
        # store kept a join-era blocked index (`lid % 2 == 0 ? out[lid / 2]`), so
        # output[k] received source element 2k — every other element, at every size.
        #
        # The store was applying the blocked rule because it inferred the layout from the
        # result's SIZE, and a split of an (N,2) input yields size N, which equals the
        # input's first dim. Same root cause as the argmin/argmax index refusals. The
        # split now DECLARES its layout (registered on the results below) and the store
        # honours it, so this computes.
        #
        # The >1024 staging cap is a separate, still-real limit and is enforced below.
        if total > self.effective_block_size and not self._needs_wrapping:
            raise MetalNonRecoverableError(
                f"tt.split of a {total}-element input with only "
                f"{self.effective_block_size} threads and no wrapping loop: the staging "
                f"writes one element per thread, so the tail would be left uninitialized "
                f"and the de-interleave silently wrong. Refusing (correct-or-refuse).",
                op_name="tt.split",
            )

        # Determine types
        input_dtype = self.env_types.get(src_id, "i32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"
        if input_dtype == "fp16":
            msl_type = "half"
            shared_dtype = "fp16"

        # Allocate shared memory
        shared_name = f"split_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage all values to shared memory.
        # Must use a separate write loop + barrier BEFORE reading, because
        # all 2*N values need to be staged before any de-interleaving.
        elem = self._lid_expr  # _loop_e in wrapping mode
        bs = self.effective_block_size
        if self._needs_wrapping or total > bs:
            # Close the current wrapping loop temporarily
            # Actually — we can't easily close/reopen the wrapping loop.
            # Instead, stage using lid with a stride loop OUTSIDE the
            # wrapping context. Since we're inside the wrapping loop,
            # use the current element index for write, but ensure ALL
            # elements are written before reading by using the wrapping
            # loop's coverage guarantee: each element is visited exactly once.
            # The barrier syncs after each batch of writes.
            # Issue: reads in the same iteration can access unwritten slots.
            # Fix: break into two separate barriers — first write all, then read.
            self.kb.raw_line(f"    if ({elem} < {total}u) {shared_name}[{elem}] = {src_var};")
            # Need all elements written — close loop, barrier, reopen
            self.kb.raw_line(f"    }}")  # close wrapping loop
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # Reopen for reads
            self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {bs}u) {{")
        else:
            self.kb.raw_line(f"    {shared_name}[{elem}] = {src_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read de-interleaved: z1 = even elements, z2 = odd elements
        z1_var = self._next_var("split")
        z2_var = self._next_var("split")
        self.kb.raw_line(f"    {msl_type} {z1_var} = {shared_name}[({elem} % {N}u) * 2u];")
        self.kb.raw_line(f"    {msl_type} {z2_var} = {shared_name}[({elem} % {N}u) * 2u + 1u];")

        # Register both results
        rid1, rid2 = ssa.result_ids[0], ssa.result_ids[1]
        self.env[rid1] = z1_var
        self.env_types[rid1] = input_dtype
        self.env[rid2] = z2_var
        self.env_types[rid2] = input_dtype
        out_shape = (N,)
        self.env_shapes[rid1] = out_shape
        self.env_shapes[rid2] = out_shape
        # Thread lid holds the pair at row lid, so both results are in DIRECT layout.
        # Declaring it stops the store from re-deriving a (wrong) blocked layout from
        # the result's size — the re-audit #8 mis-compute.
        self._register_1d_layout(rid1, "direct")
        self._register_1d_layout(rid2, "direct")

    def _lower_tt_histogram(self, ssa: SSAValue):
        """tt.histogram → threadgroup atomic histogram.

        Input: tensor<Mxi32> of values in [0, N).
        Output: tensor<Nxi32> of bin counts.

        Uses threadgroup atomic_int array for thread-safe counting.
        """
        if not ssa.operand_ids:
            self.kb.unsupported("UNSUPPORTED: tt.histogram with no operands")
            return

        input_var = self._lookup(ssa.operand_ids[0])

        # Get N (number of bins) from output type
        out_shape = _extract_shape(ssa.type_str)
        if out_shape:
            N = out_shape[0]
        else:
            N = self.effective_block_size

        # Get M (input size) from input type
        in_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        M = in_shape[0] if in_shape else self.effective_block_size

        # Allocate threadgroup atomic histogram
        hist_name = f"hist_{self._shared_counter}"
        self._shared_counter += 1

        # Histogram requires all M elements to be processed before reading.
        # If inside a wrapping loop, close it, do histogram standalone, reopen.
        bs = self.effective_block_size
        in_loop = self._needs_wrapping
        if in_loop:
            self.kb.raw_line(f"    }}")  # close wrapping loop

        # Declare as atomic int array
        self.kb.raw_line(f"    threadgroup atomic_int {hist_name}[{N}];")

        # Initialize bins to 0
        self.kb.raw_line(f"    if (lid < {N}u) atomic_store_explicit(&{hist_name}[lid], 0, memory_order_relaxed);")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Each thread increments bins — use stride loop to cover all M elements.
        # Trace back to find the source pointer for re-loading values.
        src_ptr_name = None
        src_load_op = None
        trace_id = ssa.operand_ids[0]
        for op in self.graph.ops:
            if op.id == trace_id and op.op == "tt.load" and op.operand_ids:
                # Found the load — trace its ptr to the function arg
                src_load_op = op
                ptr_id = op.operand_ids[0]
                ptr_info = self.env_is_ptr.get(ptr_id)
                if ptr_info:
                    src_ptr_name = ptr_info[0]
                break

        # How the tile's elements reach the bins.
        #
        # Re-reading the input flatly as ``src[_h]`` discards everything the source
        # ``tl.load`` expressed -- its address arithmetic, its mask and its ``other=``.
        # Measured consequences: ``tl.load(a + OFF + i)`` binned ``a[0:N]`` instead of
        # the offset window, and a masked load both read past the end of the tensor and
        # counted the masked-out lanes into the bins. All silent.
        #
        # The per-thread loaded value already has all three applied correctly, so prefer
        # it. Re-reading is only necessary when one thread does not hold exactly one
        # element -- the wrapping regime -- and there the flat address is only valid if
        # the load really was ``base + arange(0, M)``; anything else refuses.
        from triton_msl.errors import MetalNonRecoverableError

        _val_id = ssa.operand_ids[0]
        _arr = getattr(self, "env_array", {}).get(_val_id)

        # tt.histogram carries its OWN mask operand, separate from anything the source
        # load had: `tl.histogram(x, N, mask)` bins only the lanes the mask selects.
        # Upstream test_histogram_mask does exactly this over an UNMASKED load, so the
        # two masks are independent and BOTH must be honoured.
        _hist_mask_id = ssa.operand_ids[1] if len(ssa.operand_ids) >= 2 else None
        _hist_mask_arr = getattr(self, "env_array", {}).get(_hist_mask_id) if _hist_mask_id is not None else None

        def _flat_arange_load() -> bool:
            """True when the load's address is exactly ``base + arange(0, M)``, so a
            flat ``src[_h]`` re-read reproduces it."""
            if src_load_op is None or not src_load_op.operand_ids:
                return False
            by_id = {}

            def _walk(ops):
                for _o in ops:
                    by_id[_o.id] = _o
                    if getattr(_o, "region_ops", None):
                        _walk(_o.region_ops)
                    if getattr(_o, "else_ops", None):
                        _walk(_o.else_ops)

            _walk(self.graph.ops)
            cur = src_load_op.operand_ids[0]
            for _ in range(16):
                op = by_id.get(cur)
                if op is None:
                    return False
                if op.op == "tt.addptr" and len(op.operand_ids) >= 2:
                    # The pointer half must be the bare arg; the offset half a make_range.
                    # Merely appearing in env_is_ptr is insufficient: chained addptr
                    # results live there too. Their tuple carries a non-zero accumulated
                    # offset, which a flat base[_h] re-read would silently discard.
                    _base_info = self.env_is_ptr.get(op.operand_ids[0])
                    if _base_info is None or _base_info[1] not in (0, "0", ""):
                        return False
                    cur = op.operand_ids[1]
                    continue
                if op.op in ("tt.broadcast", "tt.splat", "tt.expand_dims", "arith.extsi", "arith.extui"):
                    if not op.operand_ids:
                        return False
                    cur = op.operand_ids[0]
                    continue
                if op.op == "tt.make_range":
                    start = int(op.attrs.get("start", 0) or 0)
                    end = int(op.attrs.get("end", 0) or 0)
                    return start == 0 and end == M
                return False
            return False

        if _arr is not None:
            # MEPT register-array form: thread lid owns elements [lid*n, lid*n + n).
            # The array elements already carry mask/other, so bin them directly.
            _arr_name, _arr_n = _arr[0], _arr[1]
            if _hist_mask_arr is not None:
                _hm = f" && {_hist_mask_arr[0]}[_ai]"
            elif _hist_mask_id is not None:
                _hm = f" && {self._lookup(_hist_mask_id)}"
            else:
                _hm = ""
            self.kb.raw_line(f"    for (uint _ai = 0; _ai < {_arr_n}u; ++_ai) {{")
            self.kb.raw_line(f"        uint _h = lid * {_arr_n}u + _ai;")
            self.kb.raw_line(
                f"        if (_h < {M}u{_hm}) atomic_fetch_add_explicit("
                f"&{hist_name}[(uint){_arr_name}[_ai]], 1, memory_order_relaxed);"
            )
            self.kb.raw_line(f"    }}")
        elif not in_loop:
            # One element per thread: the loaded value IS this thread's element, and the
            # histogram's own mask variable is still in scope here.
            _hm = f" && {self._lookup(_hist_mask_id)}" if _hist_mask_id is not None else ""
            self.kb.raw_line(
                f"    if (lid < {M}u{_hm}) atomic_fetch_add_explicit("
                f"&{hist_name}[(uint){input_var}], 1, memory_order_relaxed);"
            )
        elif src_ptr_name and _flat_arange_load():
            mask_extra = ""
            other_expr = None
            if len(src_load_op.operand_ids) >= 2:
                _mask_id = None
                for _op_id in src_load_op.operand_ids[1:]:
                    if _mask_id is None and (_op_id in self.env_is_mask or self._is_mask(_op_id)):
                        _mask_id = _op_id
                    elif _mask_id is not None:
                        if _op_id in getattr(self, "env_array", {}):
                            raise MetalNonRecoverableError(
                                "tt.histogram over a load whose 'other' is a per-element "
                                "array: the staged re-read cannot re-address it. "
                                "Refusing (correct-or-refuse).",
                                op_name="tt.histogram",
                            )
                        other_expr = self._lookup(_op_id)
                        break
                if _mask_id is not None:
                    recomputed = self._synthesize_mask_for_index(_mask_id, "_h")
                    if recomputed is None:
                        raise MetalNonRecoverableError(
                            "masked tt.histogram over a tile larger than the threadgroup: "
                            "the load's mask cannot be re-synthesized against the "
                            "histogram index, so it could only be dropped -- which would "
                            "read past the input and count masked-out elements into the "
                            "bins. Refusing (correct-or-refuse).",
                            op_name="tt.histogram",
                        )
                    mask_extra = recomputed
            # The histogram's OWN mask must be re-synthesized against _h too: its
            # variable was computed in the enclosing wrapping loop and is out of scope.
            _hist_mask_expr = ""
            if _hist_mask_id is not None:
                _hm_syn = self._synthesize_mask_for_index(_hist_mask_id, "_h")
                if _hm_syn is None:
                    raise MetalNonRecoverableError(
                        "tt.histogram with a mask over a tile wider than the "
                        "threadgroup: the histogram's own mask cannot be re-synthesized "
                        "against the accumulation index, so it could only be dropped -- "
                        "which would bin the lanes it excludes. "
                        "Refusing (correct-or-refuse).",
                        op_name="tt.histogram",
                    )
                _hist_mask_expr = f"({_hm_syn})"
            self.kb.raw_line(f"    for (uint _h = lid; _h < {M}u; _h += {bs}u) {{")
            if mask_extra:
                # Masked-out lanes contribute the load's `other` fill, exactly as the
                # tile would hold it -- and are never dereferenced.
                self.kb.raw_line(
                    f"        int _hval = ({mask_extra}) ? static_cast<int>({src_ptr_name}[_h])"
                    f" : static_cast<int>({other_expr if other_expr is not None else '0'});"
                )
            else:
                self.kb.raw_line(f"        int _hval = static_cast<int>({src_ptr_name}[_h]);")
            _add = f"atomic_fetch_add_explicit(&{hist_name}[(uint)_hval], 1, memory_order_relaxed);"
            self.kb.raw_line(f"        {f'if ({_hist_mask_expr}) ' if _hist_mask_expr else ''}{_add}")
            self.kb.raw_line(f"    }}")
        else:
            raise MetalNonRecoverableError(
                "tt.histogram over a tile wider than the threadgroup whose source load "
                "is not a plain `base + arange(0, N)`: the histogram would have to "
                "re-read the input flatly, which drops the load's address arithmetic "
                "(and any mask/other) and bins the wrong elements. "
                "Refusing (correct-or-refuse).",
                op_name="tt.histogram",
            )
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read result. When N <= block_size each thread holds a unique bin
        # (bin[lid]). When N > block_size the result is consumed inside the
        # reopened wrapping loop, so we must load bin[_loop_e] per iteration
        # — otherwise every iteration writes the same bin[lid] and downstream
        # stores end up with wrong values at the outer indices.
        result_var = self._next_var("hist")
        if in_loop and N > bs:
            # Reopen wrapping loop and load hist_0[_loop_e] per iteration.
            total = getattr(self, "_total_elements", self.effective_block_size)
            self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {bs}u) {{")
            self.kb.raw_line(
                f"    int {result_var} = (_loop_e < {N}u) ? atomic_load_explicit(&{hist_name}[_loop_e], memory_order_relaxed) : 0;"
            )
        else:
            self.kb.raw_line(
                f"    int {result_var} = (lid < {N}u) ? atomic_load_explicit(&{hist_name}[lid], memory_order_relaxed) : 0;"
            )
            if in_loop:
                # Reopen wrapping loop for remaining ops (stores)
                total = getattr(self, "_total_elements", self.effective_block_size)
                self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {bs}u) {{")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = (N,)
        # Bins are read at hist[lid] (or hist[_loop_e]), so thread i holds bin i:
        # DIRECT layout. Declaring it means the store no longer depends on the
        # kernel-wide convert_layout flag happening to be set.
        self._register_1d_layout(ssa.id, "direct")
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, "direct")

    def _synthesize_mask_for_index(self, mask_id: int, index_var: str, _depth: int = 0):
        """Recompute a boolean mask expression using `index_var` as the loop index.

        Used by tt.histogram, which closes the enclosing wrapping loop and opens
        its own accumulation loop with a fresh index variable. Any mask computed
        inside the original loop references `_loop_e`, which is out of scope by
        then, so we trace the mask's producers and rebuild the expression with
        `index_var` substituted for make_range-derived values.

        Returns an MSL expression string (parenthesized) on success, or None if
        the mask is too complex to synthesize.

        None means REFUSE, not "drop the mask". The old contract dropped it and relied
        on the tile bounds check ``_h < M`` to keep reads safe -- but ``M`` bounds the
        TILE, not the tensor, so dropping it read past the input and counted masked-out
        elements into the bins (2026-08-26).
        """
        if _depth > 8:
            return None

        op_by_id = {op.id: op for op in self.graph.ops}
        op = op_by_id.get(mask_id)
        if op is None:
            # Not produced by an op: a kernel ARGUMENT (block argument), e.g. the bound
            # in `mask=i < n_elements`. Those are function parameters -- in scope
            # everywhere, including the histogram's own loop, and loop-invariant -- so
            # the name resolves correctly here. Without this the whole mask failed to
            # synthesize and the caller had to refuse a perfectly ordinary masked load.
            # Array- and shared-memory-backed values are NOT loop-invariant scalars.
            if (
                mask_id in self.env
                and mask_id not in getattr(self, "env_array", {})
                and mask_id not in getattr(self, "_shared_mem_descs", {})
                and isinstance(self.env.get(mask_id), str)
            ):
                return f"({self.env[mask_id]})"
            return None

        name = op.op

        # make_range(start, end) — per-thread index; substitute the loop index
        if name == "tt.make_range":
            start = op.attrs.get("start", 0)
            if start:
                return f"((int){index_var} + {int(start)})"
            return f"(int){index_var}"

        # splat/broadcast/convert_layout — pass through
        if name in ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.expand_dims", "tt.unsplat"):
            if not op.operand_ids:
                return None
            return self._synthesize_mask_for_index(op.operand_ids[0], index_var, _depth + 1)

        # arith.constant — emit the scalar value
        if name == "arith.constant":
            value = op.attrs.get("value")
            if value is None:
                return None
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, int):
                return f"({int(value)})"
            if isinstance(value, float):
                return f"({float(value)}f)"
            return None

        # arith.cmpi — binary comparison
        if name == "arith.cmpi":
            if len(op.operand_ids) < 2:
                return None
            a = self._synthesize_mask_for_index(op.operand_ids[0], index_var, _depth + 1)
            b = self._synthesize_mask_for_index(op.operand_ids[1], index_var, _depth + 1)
            if a is None or b is None:
                return None
            pred_name = op.attrs.get("predicate_name")
            pred_int = op.attrs.get("predicate")
            if pred_name and pred_name in CMPI_NAMED:
                op_str = CMPI_NAMED[pred_name]
            elif pred_int is not None and pred_int in CMPI_PREDICATES:
                op_str = CMPI_PREDICATES[pred_int]
            else:
                return None
            is_unsigned = pred_name in ("ult", "ule", "ugt", "uge") if pred_name else False

            # Width-correct cast (see _lower_cmpi, audit C1): a 64-bit bound
            # (e.g. a large tensor dim compared against an index) must not be
            # truncated to 32 bits. Falls back to 32-bit when the operand type
            # is unknown (index-derived operands, always small).
            def _is64m(oid):
                return self.env_types.get(oid) in ("i64", "u64", "ui64")

            wide64 = _is64m(op.operand_ids[0]) or _is64m(op.operand_ids[1])
            if is_unsigned:
                cast = "(ulong)" if wide64 else "(uint)"
            else:
                cast = "(long)" if wide64 else "(int)"
            return f"({cast}{a} {op_str} {cast}{b})"

        # arith.andi / arith.ori / arith.xori on i1 — logical combinators
        if name in ("arith.andi", "arith.ori", "arith.xori"):
            if len(op.operand_ids) < 2:
                return None
            a = self._synthesize_mask_for_index(op.operand_ids[0], index_var, _depth + 1)
            b = self._synthesize_mask_for_index(op.operand_ids[1], index_var, _depth + 1)
            if a is None or b is None:
                return None
            logical = {"arith.andi": "&&", "arith.ori": "||", "arith.xori": "!="}[name]
            return f"({a} {logical} {b})"

        # Simple integer arithmetic on the index — supports offset patterns like
        # `make_range + scalar` used as a mask input.
        if name in ("arith.addi", "arith.subi", "arith.muli"):
            if len(op.operand_ids) < 2:
                return None
            a = self._synthesize_mask_for_index(op.operand_ids[0], index_var, _depth + 1)
            b = self._synthesize_mask_for_index(op.operand_ids[1], index_var, _depth + 1)
            if a is None or b is None:
                return None
            op_str = {"arith.addi": "+", "arith.subi": "-", "arith.muli": "*"}[name]
            return f"({a} {op_str} {b})"

        return None

    @staticmethod
    def _gather_stage_types(src_dtype):
        """(msl_type, shared_dtype) for staging a value of ``src_dtype`` into the gather
        threadgroup array. SINGLE source of truth for both the 1-D and 2-D gather paths.
        64-bit ints MUST stage as long/ulong (shared_dtype i64/u64) — a 32-bit staging
        array silently truncates the high word (Triton-lens re-audit 2026-06-25: a gather
        of int64 source value 3_000_000_000 read back -1_294_967_296). Mirrors the
        i64/u64 -> long/ulong branch the select/scan siblings already carry.
        """
        if src_dtype.startswith("fp") or src_dtype.startswith("bf"):
            return "float", "fp32"
        if src_dtype in ("i64", "u64", "ui64"):
            return ("long", "i64") if src_dtype == "i64" else ("ulong", "u64")
        return "int", "i32"

    def _lower_tt_gather(self, ssa: SSAValue):
        """tt.gather → shared memory indexed lookup.

        For 1D: src (tensor<Sxf32>), indices (tensor<Ixi32>) → result (tensor<Ixf32>).
        Stage src to shared memory, each thread reads shared[indices[lid]].
        """
        if len(ssa.operand_ids) < 2:
            self.kb.unsupported("UNSUPPORTED: tt.gather with < 2 operands")
            return

        src_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Get source size from type
        src_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        idx_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[1]))

        # 2D gather has per-axis semantics (out[i,j] = src[idx[i,j], j] for
        # axis=0, src[i, idx[i,j]] for axis=1) that the 1D flat-staging path
        # below does NOT implement. Dispatch to the dedicated 2D handler, which
        # implements the supported shapes and refuses the rest loudly (never
        # silently wrong). See docs/superpowers/plans/2026-06-18-2d-gather-coverage.md.
        def _effective_rank(shape):
            return sum(1 for d in (shape or []) if d != 1)

        if _effective_rank(src_shape) > 1 or _effective_rank(idx_shape) > 1:
            self._lower_tt_gather_2d(ssa, src_var, idx_var, src_shape, idx_shape)
            return

        S = src_shape[0] if src_shape else self.effective_block_size

        # Threadgroup-cap guard (mirrors the 2-D gather guard, generic_lowerer ~5643).
        # The source is staged ONE element per thread (`if (lid < S) shared[lid] = src`)
        # and each thread gathers one result. Apple caps a threadgroup at 1024 threads,
        # so for src size S > 1024 the shared slots 1024..S-1 are NEVER written -> a
        # gather of any index >= 1024 reads UNINITIALIZED memory (silent-wrong;
        # 2026-06-22 re-audit: S=2048 gave out[1536]=0 vs 1537.5). An index tensor
        # larger than 1024 likewise under-computes the output. The 2-D sibling already
        # refuses this; the 1-D path was missing the guard. Refuse rather than mis-gather.
        _idx_total = 1
        for _d in idx_shape or [1]:
            _idx_total *= _d
        if S > 1024 or _idx_total > 1024:
            raise MetalNonRecoverableError(
                f"1-D tt.gather with source size {S} / index size {_idx_total} exceeds "
                f"the 1024-thread threadgroup: the source is staged one element per "
                f"thread, so slots/indices past 1024 read uninitialized memory "
                f"(silent-wrong). Refusing; strided multi-pass staging is future work.",
                op_name="tt.gather",
            )

        # Determine types
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        msl_type, shared_dtype = self._gather_stage_types(src_dtype)

        # Allocate shared memory for source
        shared_name = f"gather_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=S)

        # Stage source to shared (only threads with valid src index)
        self.kb.raw_line(f"    if (lid < {S}u) {shared_name}[lid] = {src_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Each thread gathers: result = shared[idx]
        result_var = self._next_var("gathered")
        self.kb.raw_line(f"    {msl_type} {result_var} = {shared_name}[(uint){idx_var}];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = src_dtype
        out_shape = _extract_shape(ssa.type_str)
        if out_shape:
            self.env_shapes[ssa.id] = tuple(out_shape)

    def _lower_tt_gather_2d(self, ssa: SSAValue, src_var, idx_var, src_shape, idx_shape):
        """2D tt.gather via full-tile shared-memory staging.

            axis=0:  out[i, j] = src[index[i, j], j]   ->  shared[idx*N + col]
            axis=1:  out[i, j] = src[i, index[i, j]]   ->  shared[row*N + idx]

        Each thread owns one output element at (row = lid / N, col = lid % N)
        and its own per-thread ``src``/``idx`` value. The gather needs an
        arbitrary OTHER element of the source row/column, so the whole source
        tile is staged into threadgroup memory first, then indexed.

        Supported (else refuse loudly -- never silently wrong):
          - src and index share the same 2D shape (so a thread's flat ``lid``
            equals its row*N+col in BOTH, making the staging
            ``shared[lid] = src`` exact). A ragged gather axis is refused.
          - one element per thread: the 2D kernel launches M*N threads, so the
            tile must fit Metal's 1024-thread threadgroup (and 32 KB shared).
          - scalar (non-register-array) operands.
        """
        from triton_msl.errors import MetalNonRecoverableError

        axis = ssa.attrs.get("axis", 0)

        if not src_shape or not idx_shape or len(src_shape) != 2 or len(idx_shape) != 2:
            raise MetalNonRecoverableError(
                f"2D tt.gather: expected 2D src and index (got src "
                f"{tuple(src_shape or ())}, index {tuple(idx_shape or ())})."
            )
        if axis not in (0, 1):
            raise MetalNonRecoverableError(f"2D tt.gather: unsupported axis {axis} (only 0 and 1).")
        # Register-array (multi-element-per-thread) operands aren't a single
        # per-thread scalar, so the flat ``shared[lid] = src`` staging would
        # mis-map. Refuse rather than guess.
        if ssa.operand_ids[0] in self.env_array or ssa.operand_ids[1] in self.env_array:
            raise MetalNonRecoverableError(
                "2D tt.gather: register-array (multi-element-per-thread) operands "
                "are not yet supported; refusing rather than mis-staging."
            )

        # Shape compatibility per axis. The output/index drives the thread grid
        # (block_size == idx rows*cols); the SOURCE tile is staged into shared.
        #   axis=0 (gather rows): out[i,j]=src[idx[i,j],j]. src and index must
        #     share the COLUMN count N; the row counts may differ (ragged) — the
        #     index picks any source row. Read shared[idx*N + col].
        #   axis=1 (gather cols): out[i,j]=src[i,idx[i,j]]. Supported only for the
        #     same-shape case (a ragged source column count remaps the per-thread
        #     src load in a way this flat staging doesn't track) — refuse ragged.
        N = src_shape[1]
        if axis == 0:
            if src_shape[1] != idx_shape[1]:
                raise MetalNonRecoverableError(
                    f"2D tt.gather axis=0 requires matching column counts "
                    f"(src {tuple(src_shape)} vs index {tuple(idx_shape)}); refusing."
                )
        else:  # axis == 1
            if tuple(src_shape) != tuple(idx_shape):
                raise MetalNonRecoverableError(
                    f"2D tt.gather axis=1 (ragged) is not yet lowered "
                    f"(src {tuple(src_shape)} vs index {tuple(idx_shape)}); refusing."
                )

        src_total = src_shape[0] * src_shape[1]
        idx_total = idx_shape[0] * idx_shape[1]
        # One thread per element: the index tile drives the 2D thread grid, which
        # must fit Metal's 1024-thread threadgroup. The source is staged one
        # element per thread too, so it must also fit AND not exceed the thread
        # count (else some source rows go un-staged). Larger tiles need a strided
        # multi-pass staging (future work) -- refuse rather than mis-stage.
        if idx_total > 1024 or src_total > 1024:
            raise MetalNonRecoverableError(
                f"2D tt.gather: tile too large (src {src_total}, index {idx_total} "
                "elems) for one-element-per-thread staging in a 1024-thread "
                "threadgroup. Refusing; strided multi-pass staging is future work."
            )
        if src_total > idx_total:
            raise MetalNonRecoverableError(
                f"2D tt.gather: source ({src_total} elems) larger than the index/"
                f"thread grid ({idx_total}); cannot stage every source element. Refusing."
            )

        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        msl_type, shared_dtype = self._gather_stage_types(src_dtype)
        zero = "0.0f" if is_float else "0"

        lid = self._lid_expr
        shared_name = f"gather2d_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=src_total)

        # Stage the full source tile: thread lid holds src flat[lid]
        # (row*N + col == lid for a row-major src_rows*N tile), for lid < src_total.
        self.kb.raw_line(f"    if ({lid} < {src_total}u) {shared_name}[{lid}] = {src_var};")
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        if axis == 0:
            # out[i,j] = src[idx[i,j], j]: row picked by idx, col = lid % N.
            gathered = f"(uint){idx_var} * {N}u + ({lid} % {N}u)"
        else:
            # out[i,j] = src[i, idx[i,j]]: row = lid / N, col picked by idx.
            gathered = f"({lid} / {N}u) * {N}u + (uint){idx_var}"

        # Guard the read by the OUTPUT thread grid (lid < idx_total); out-of-range
        # threads hold a garbage idx and the store masks them out, but avoid an
        # out-of-bounds shared read here. For in-range threads the computed index
        # is in [0, src_total) by construction (idx in [0, src_rows), col in [0,N)).
        result_var = self._next_var("gathered2d")
        self.kb.raw_line(f"    {msl_type} {result_var} = ({lid} < {idx_total}u) ? {shared_name}[{gathered}] : {zero};")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = src_dtype
        out_shape = _extract_shape(ssa.type_str)
        if out_shape:
            self.env_shapes[ssa.id] = tuple(out_shape)

    # -- Map elementwise --

    def _lower_map_elementwise(self, ssa: SSAValue):
        """tt.map_elementwise → apply body per element.

        The body region contains basic blocks with cf.cond_br (conditional
        branches) forming a decision tree. We convert this to nested
        ternary expressions in MSL.

        TTGIR pattern:
            %z = "tt.map_elementwise"(%x, %y) <{pack = 1}> ({
            ^bb0(%a: i32, %b: i32):
                %cmp = arith.cmpi slt, %a, %b : i32
                cf.cond_br %cmp, ^bb2(%c-1), ^bb1
            ^bb1:
                %cmp2 = arith.cmpi eq, %a, %b : i32
                cf.cond_br %cmp2, ^bb2(%c0), ^bb2(%c1)
            ^bb2(%result: i32):
                tt.map_elementwise.return %result : i32
            })

        MSL output:
            int v42 = (v10 < v11) ? -1 : ((v10 == v11) ? 0 : 1);
        """
        if not ssa.region_ops:
            self._emit_passthrough(ssa)
            return

        # Get input operands
        input_vars = [self.env.get(oid, f"v{oid}") for oid in ssa.operand_ids]
        msl_type = triton_type_to_msl(ssa.elem_type) if ssa.elem_type else "int"

        # Parse the body to extract the decision tree from the raw TTGIR text.
        # The region_ops contain all ops across basic blocks, but cf.cond_br
        # targets (^bb labels) are only in the raw text. We need to reconstruct
        # the control flow from the ops' _block_id attributes and cf.cond_br args.
        #
        # Strategy: Process ops by basic block. Each cf.cond_br creates a branch.
        # We build a nested ternary expression by resolving the branch targets.

        # Group ops by basic block
        blocks = {}  # block_id → list of ops
        block_order = []
        for op in ssa.region_ops:
            bid = op.attrs.get("_block_id", 0)
            if bid not in blocks:
                blocks[bid] = []
                block_order.append(bid)
            blocks[bid].append(op)

        # Find block args (bb entry parameters)
        block_args = {}  # block_id → list of arg SSA ids
        for op in ssa.region_ops:
            if op.op in ("tt.map_elementwise.return", "tt.reduce.return"):
                continue

        # Simple case: no cf.cond_br/cf.br (direct computation)
        has_cond_br = any(op.op in ("cf.cond_br", "cf.br") for op in ssa.region_ops)

        if not has_cond_br:
            # Direct lowering: process body ops with input bindings
            bb_args = ssa.attrs.get("block_arg_ids", [])
            for i, arg_id in enumerate(bb_args):
                if i < len(input_vars):
                    self.env[arg_id] = input_vars[i]
                    self.env_types[arg_id] = ssa.elem_type or "i32"

            for op in ssa.region_ops:
                if op.op == "tt.map_elementwise.return":
                    if op.operand_ids:
                        result_var = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}")
                        var_name = f"v{ssa.id}"
                        self.kb.raw_line(f"    {msl_type} {var_name} = {result_var};")
                        self.env[ssa.id] = var_name
                        self.env_types[ssa.id] = ssa.elem_type or "i32"
                else:
                    self._lower_op_dispatch(op)
            return

        # Complex case: cf.cond_br creates a decision tree
        # Parse from raw TTGIR text to resolve branch targets and block args
        self._lower_map_elementwise_cond_br(ssa, input_vars, msl_type)

    def _lower_map_elementwise_cond_br(self, ssa, input_vars, msl_type):
        """Lower map_elementwise with cf.cond_br decision tree.

        Reconstructs the basic block graph from region_ops and converts
        cf.cond_br branches to nested ternary/if-else expressions.

        cf.cond_br operand_ids are: [condition, true_args..., false_args...].
        The split between true/false args comes from n_true_operands/n_false_operands
        attrs (parsed from TTGIR text by the walker).
        """
        # Group ops by block
        blocks = {}
        block_order = []
        for op in ssa.region_ops:
            bid = op.attrs.get("_block_id", 0)
            if bid not in blocks:
                blocks[bid] = []
                block_order.append(bid)
            blocks[bid].append(op)

        # Bind entry block args to input vars
        bb_arg_ids = ssa.attrs.get("block_arg_ids", [])
        for i, arg_id in enumerate(bb_arg_ids):
            if i < len(input_vars):
                self.env[arg_id] = input_vars[i]
                self.env_types[arg_id] = ssa.elem_type or "i32"

        # Declare result variable
        var_name = f"v{ssa.id}"
        self.kb.raw_line(f"    {msl_type} {var_name};")

        # Process the decision tree using structured if/else
        self._emit_cond_br_block(blocks, block_order, 0, var_name, msl_type)

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = ssa.elem_type or "i32"

    # -- Prefix scan (tt.scan) --

    def _staged_fill_terms(self, nid, coeff, axis_dim, op_by_id, depth=0):
        """Decompose one (sub-)offset/index value into a flat list of additive
        contributions for the ``(_fill_row = _sa/N, _fill_col = _sa%N)``
        staged-fill decomposition, distributing any enclosing stride ``coeff``
        over sums.  Each contribution is either:

          ("range", dim, coeff_str, extent, start)  -- a make_range index, or
          ("const", expr_str)                        -- a block-constant.

        Returns ``None`` if any part cannot be reduced (callers refuse).
        ``coeff`` is the accumulated outer stride; ``axis_dim`` is the row/col
        position fixed by an enclosing ``tt.expand_dims`` (axis=1 -> dim 0 =
        row; axis=0 -> dim 1 = col).

        Shared by ``_rebuild_staged_fill_offset`` (staged dot-operand and store
        addresses) and ``_rebuild_staged_fill_mask`` (the cooperative store
        guard) so the structural row/col resolution is single-source -- no twin
        copy that a fix could land in only one of.
        """
        _PASS = (
            "arith.index_cast",
            "arith.index_castui",
            "arith.extsi",
            "arith.extui",
        )

        # A truncation is not an affine row/column term. Dropping it here
        # recreates the template's wrong address in the generic staging path.
        # Keep it unproved until this representation can replay its wrap point.

        def _const_str(tid):
            """MSL string for a compile-time / block-constant coefficient
            operand (dense constant or splat of a scalar arg / induction
            var), else None.  An index expression (idx_*/r_* derived) is NOT
            a constant coefficient."""
            op = op_by_id.get(tid)
            ev = self.env.get(tid)
            if op is None:
                if isinstance(ev, str) and ev and not ev.startswith("idx_") and not ev.startswith("r_"):
                    return ev
                return None
            if op.op in ("arith.constant", "tt.splat"):
                return ev if (isinstance(ev, str) and ev) else None
            return None

        def _mul_coeff(coeff, c):
            return c if coeff is None else f"({coeff} * {c})"

        def _neg_coeff(coeff):
            return "-1" if coeff is None else f"(-({coeff}))"

        def _apply_coeff(expr, coeff):
            return expr if coeff is None else f"({expr} * {coeff})"

        if depth > 64:
            return None
        op = op_by_id.get(nid)
        if op is None:
            ev = self.env.get(nid)
            if isinstance(ev, str) and ev:
                return [("const", _apply_coeff(ev, coeff))]
            return None
        name = op.op
        if name == "tt.make_range":
            if axis_dim is None:
                return None
            start = int(op.attrs.get("start", 0))
            end = int(op.attrs.get("end", 0))
            return [("range", axis_dim, coeff, end - start, start)]
        if name == "tt.expand_dims":
            # 2-D staged tile: a single expand_dims places the make_range
            # at the non-inserted axis. axis=1 -> dim 0 (row);
            # axis=0 -> dim 1 (col).
            axis = int(op.attrs.get("axis", 0))
            d = 1 - axis
            if d not in (0, 1):
                return None
            if axis_dim is not None and axis_dim != d:
                return None
            if not op.operand_ids:
                return None
            return self._staged_fill_terms(op.operand_ids[0], coeff, d, op_by_id, depth + 1)
        if name == "tt.broadcast" or name in _PASS:
            if not op.operand_ids:
                return None
            return self._staged_fill_terms(op.operand_ids[0], coeff, axis_dim, op_by_id, depth + 1)
        if name in ("tt.splat", "arith.constant"):
            ev = self.env.get(nid)
            if isinstance(ev, str) and ev:
                return [("const", _apply_coeff(ev, coeff))]
            return None
        if name in ("arith.addi", "arith.subi"):
            if len(op.operand_ids) < 2:
                return None
            left = self._staged_fill_terms(op.operand_ids[0], coeff, axis_dim, op_by_id, depth + 1)
            if left is None:
                return None
            rc = coeff if name == "arith.addi" else _neg_coeff(coeff)
            right = self._staged_fill_terms(op.operand_ids[1], rc, axis_dim, op_by_id, depth + 1)
            if right is None:
                return None
            return left + right
        if name == "arith.muli":
            if len(op.operand_ids) < 2:
                return None
            a, b = op.operand_ids[0], op.operand_ids[1]
            ca, cb = _const_str(a), _const_str(b)
            if cb is not None and ca is None:
                return self._staged_fill_terms(a, _mul_coeff(coeff, cb), axis_dim, op_by_id, depth + 1)
            if ca is not None and cb is None:
                return self._staged_fill_terms(b, _mul_coeff(coeff, ca), axis_dim, op_by_id, depth + 1)
            return None
        # Any other op breaks structural resolution.
        return None

    _INDEX_BITS = {"i32": 32, "u32": 32, "i64": 64, "u64": 64, "ui64": 64}

    def _staged_fill_typed(self, nid, axis_dim, op_by_id, M, N, depth=0):
        """Render one index value at ``(_fill_row, _fill_col)`` with every operation at
        the width the source gives it: ``(expr, bits)``, ``expr`` an ``int`` (32) or a
        ``long`` (64), or None when a part is not covered.

        ``_staged_fill_terms`` flattens a value into one sum. While every term is 32
        bits wide that is exact, because the sum is taken modulo 2**32 whatever C type
        it gets. Beside a 64-bit term it is not: a 32-bit partial sum that holds ``pid``
        (a ``uint``) is zero-extended where the source sign-extends it, and a zero
        extension in the source is dropped. Callers use this rendering whenever a 64-bit
        value takes part. Sums, differences and products are computed unsigned and cast
        back, so they wrap where the source's do.
        """
        if depth > 64:
            return None
        op = op_by_id.get(nid)
        bits = self._INDEX_BITS.get(self.env_types.get(nid) if op is None else op.elem_type)
        if bits is None:
            return None
        signed, unsigned = ("int", "uint") if bits == 32 else ("long", "ulong")
        if op is None or op.op in ("tt.splat", "arith.constant"):
            # A lowered value is recovered by a cast to its declared type, as at its
            # other uses; the declared type has to be the source's width for that.
            ev = self.env.get(nid)
            if not (isinstance(ev, str) and ev) or self._INDEX_BITS.get(self.env_types.get(nid)) != bits:
                return None
            return f"({signed})({ev})", bits
        if op.op == "tt.make_range":
            start, end = int(op.attrs.get("start", 0)), int(op.attrs.get("end", 0))
            if axis_dim is None or bits != 32 or end - start != (M if axis_dim == 0 else N):
                return None
            idx = "(int)_fill_row" if axis_dim == 0 else "(int)_fill_col"
            return (idx if start == 0 else f"({start} + {idx})"), 32
        if not op.operand_ids:
            return None
        if op.op == "tt.expand_dims":
            # axis=1 puts the range at dim 0 (row); axis=0 at dim 1 (column).
            d = 1 - int(op.attrs.get("axis", 0))
            if d not in (0, 1) or (axis_dim is not None and axis_dim != d):
                return None
            return self._staged_fill_typed(op.operand_ids[0], d, op_by_id, M, N, depth + 1)
        if op.op == "tt.broadcast":
            return self._staged_fill_typed(op.operand_ids[0], axis_dim, op_by_id, M, N, depth + 1)
        if op.op in ("arith.extsi", "arith.extui"):
            inner = self._staged_fill_typed(op.operand_ids[0], axis_dim, op_by_id, M, N, depth + 1)
            if inner is None or inner[1] != 32 or bits != 64:
                return None
            return (f"(long)({inner[0]})" if op.op == "arith.extsi" else f"(long)((uint)({inner[0]}))"), 64
        symbol = {"arith.addi": "+", "arith.subi": "-", "arith.muli": "*"}.get(op.op)
        if symbol is None or len(op.operand_ids) < 2:
            return None
        sides = [self._staged_fill_typed(oid, axis_dim, op_by_id, M, N, depth + 1) for oid in op.operand_ids[:2]]
        if any(side is None or side[1] != bits for side in sides):
            return None
        return f"({signed})(({unsigned})({sides[0][0]}) {symbol} ({unsigned})({sides[1][0]}))", bits

    # Producers that pass a value through unchanged as far as the thread-to-element
    # mapping is concerned (dtype/shape wrappers, layout conversions).
    _LAYOUT_PASSTHROUGH_OPS = (
        "ttg.convert_layout",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.sitofp",
        "arith.uitofp",
        "arith.fptosi",
        "arith.fptoui",
        "arith.bitcast",
        "arith.extf",
        "arith.truncf",
    )

    def _layout_op_index(self):
        """id -> op across nested regions (built per call; graphs here are small)."""
        by_id = {}

        def _walk(ops):
            for _o in ops:
                by_id[_o.id] = _o
                if getattr(_o, "region_ops", None):
                    _walk(_o.region_ops)
                if getattr(_o, "else_ops", None):
                    _walk(_o.else_ops)

        _walk(self.graph.ops)
        return by_id

    def _traces_to_converted_layout(self, val_id, depth=8):
        """True when THIS value went through a ttg.convert_layout redistribution.

        The store used to test ``bool(self._converted_layout_ids)`` -- a KERNEL-WIDE
        flag. When a 2-D argmin/argmax had its value converted but its index not, that
        flag put BOTH stores on the simple layout, so the index (still broadcast as
        ``shared[lid / N]``) stored row 0's index for every row (reduce-probe #5).
        """
        conv = getattr(self, "_converted_layout_ids", None)
        if not conv:
            return False
        by_id = self._layout_op_index()
        cur = val_id
        for _ in range(depth):
            if cur in conv:
                return True
            op = by_id.get(cur)
            if op is None or not op.operand_ids:
                return False
            if op.op in self._LAYOUT_PASSTHROUGH_OPS:
                cur = op.operand_ids[0]
                continue
            return False
        return False

    def _report_undeclared_layout(self, how, size):
        """A 1-D store whose thread-to-element layout no producer declared: REFUSE.

        Such a store can only be emitted by GUESSING the layout, and that guess is
        exactly what produced three silent-wrongs -- axis=0 argmin/argmax on a square
        tile (reduce-probe #6), axis=1 with both results consumed (#5), and tt.split
        (re-audit #8). In each case the emitted code looked reasonable and returned the
        wrong elements.

        Refusing is now the DEFAULT (flipped 2026-08-27). Every producer this path could
        reach across the full upstream conformance suite (9,342 tests) and the project
        suite now calls ``_register_1d_layout`` -- verified by running both suites with
        this refusal live: 198 upstream failures with 0 undeclared-layout errors, so
        this path is UNREACHABLE on every kernel either suite exercises. Measuring
        lesson, kept because it bit: an earlier flip attempt counted fallbacks via a
        WARN print, which pytest CAPTURES for passing tests (the count silently read as
        zero), and regressed 51 upstream ``test_reduce`` cases. Count refusals with an
        exception (cannot be captured away), never with prints under pytest.

        ``TRITON_MSL_INFER_LAYOUT=1`` opts back into the historical size/kernel-wide-flag
        inference for a kernel this project's test surface does not cover, in the same
        spirit as ``TRITON_MSL_LEGACY=1`` for the legacy parser -- it can be silently
        wrong, and exists as a way forward while the producer is taught to declare.
        """
        from triton_msl.errors import MetalNonRecoverableError

        msg = (
            f"1-D store layout was not declared by its producer, so it could only be "
            f"resolved by {how} -- a guess that has silently stored the wrong elements "
            f"before (kernel={self.graph.func_name}, size={size}, "
            f"2d_shape={self._effective_2d_shape}). Refusing (correct-or-refuse). The "
            f"producer should call _register_1d_layout; set TRITON_MSL_INFER_LAYOUT=1 to "
            f"fall back to the historical inference (may be silently wrong)."
        )
        if os.environ.get("TRITON_MSL_INFER_LAYOUT") == "1":
            if os.environ.get("TRITON_MSL_WARN_UNDECLARED_LAYOUT") == "1":
                import sys as _sys

                print(f"[triton-msl] {msg}", file=_sys.stderr)
            return
        raise MetalNonRecoverableError(msg, op_name="tt.store")

    def _register_1d_layout(self, value_id, layout):
        """Record the thread-to-element mapping a 1-D value is produced in.

        ``"direct"``  -- thread i holds element i (store: ``if (lid < size) out[lid]``)
        ``"blocked"`` -- thread i holds element i / N (store: one thread per row block)

        Producers must declare this because the STORE cannot infer it from the value's
        SIZE. A 2-D axis=0 reduce yields size N and an axis=1 reduce size M, so on a
        square tile they are indistinguishable; ``tt.split`` of an (N,2) input yields
        size N, which equals the input's first dim. In every one of those cases the
        size heuristic picked the blocked rule and silently stored the wrong elements
        (reduce-probe #5/#6, re-audit #8).
        """
        if not hasattr(self, "_value_1d_layout"):
            self._value_1d_layout = {}
        self._value_1d_layout[value_id] = layout

    # Elementwise ops that PRESERVE per-element layout: the result's element i is a
    # function of each operand's element i, so if every tensor operand shares a layout,
    # the result has that layout too.
    _LAYOUT_ELEMENTWISE_OPS = (
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.maxnumf",
        "arith.minnumf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.sqrt",
        "math.absf",
        "arith.negf",
        "arith.select",
    )

    def _value_1d_layout_of(self, val_id, depth=24):
        """The layout of a stored 1-D value: "direct", "blocked", "any", or None.

        "any" means the value is UNIFORM across elements (constants, splats of a
        scalar) — every layout reads it identically, so it combines with anything and
        a consumer may treat it as direct. None means unresolvable: the caller keeps
        its previous behaviour rather than guessing.

        Resolution follows, recursively:
          * the producer registry (_register_1d_layout);
          * single-operand pass-through wrappers (_LAYOUT_PASSTHROUGH_OPS);
          * elementwise ops (_LAYOUT_ELEMENTWISE_OPS): combine over the operands —
            "any" yields to anything, agreement passes through, conflict -> None;
          * constants and splats -> "any";
          * ``scf.for`` results and block args: a carried value's layout is
            combine(init[i], yield[i]) — the loop-carried lse case (`li = li*a +
            tl.sum(p,1)`; store of `mi + log(li)` after the loop) resolves to
            "blocked" because the in-loop reduce registered blocked and the
            `tl.full`/`tl.zeros` inits are "any". Cycles (mi = f(mi_arg, ...))
            resolve optimistically as "any" ON THE BACK-EDGE only, which is sound
            because init and yield are ALWAYS combined: a carry that rotates two
            different layouts conflicts at the combine and yields None.
        """
        rec = getattr(self, "_value_1d_layout", None)
        if not rec:
            return None

        by_id = self._layout_op_index()

        # scf.for block-arg id -> (for_op, iter index); result id -> (for_op, index).
        ba_of = {}
        res_of = {}
        for _o in by_id.values():
            if _o.op != "scf.for":
                continue
            _bas = (_o.attrs.get("block_arg_ids", []) if _o.attrs else [])[1:]  # skip IV
            for _i, _ba in enumerate(_bas):
                ba_of[_ba] = (_o, _i)
            # A SINGLE-result scf.for has result_ids == None and its result id IS the
            # op's own id (mlir_walker keeps result_ids only for multi-result ops).
            # Missing this is the historical single-iterarg trap: the carry edge never
            # existed, the in-loop reduce was invisible, and a blocked value staged
            # modular — the row-collapse silent-wrong, reintroduced.
            _rids = _o.result_ids or ([_o.id] if len((_o.operand_ids or [])) > 3 else [])
            for _i, _rid in enumerate(_rids):
                res_of[_rid] = (_o, _i)

        def _yield_of(for_op):
            for _b in for_op.region_ops or []:
                if _b.op == "scf.yield":
                    return _b
            return None

        def _combine(a, b):
            if a is None or b is None:
                return None
            if a == "any":
                return b
            if b == "any":
                return a
            return a if a == b else None

        def _resolve(cur, visiting, d):
            if d <= 0:
                return None
            if cur in visiting:
                return "any"  # optimistic back-edge; init⊓yield combine keeps it sound
            if cur in rec:
                return rec[cur]
            # Carried values: combine init with yield, positionally.
            hit = res_of.get(cur) or ba_of.get(cur)
            if hit is not None:
                for_op, i = hit
                inits = (for_op.operand_ids or [])[3:]
                y = _yield_of(for_op)
                if y is None or i >= len(y.operand_ids or []) or i >= len(inits):
                    return None
                visiting = visiting | {cur}
                return _combine(
                    _resolve(inits[i], visiting, d - 1),
                    _resolve(y.operand_ids[i], visiting, d - 1),
                )
            op = by_id.get(cur)
            if op is None:
                return None
            if op.op == "arith.constant":
                return "any"
            if op.op == "tt.splat":
                return "any"  # a splat of a scalar is uniform across elements
            if not op.operand_ids:
                return None
            if op.op in self._LAYOUT_PASSTHROUGH_OPS:
                return _resolve(op.operand_ids[0], visiting, d - 1)
            if op.op in self._LAYOUT_ELEMENTWISE_OPS:
                out = "any"
                # arith.select's condition (operand 0) selects per element; it shares
                # the layout requirement like any other operand, so combine all.
                for _oid in op.operand_ids:
                    out = _combine(out, _resolve(_oid, visiting, d - 1))
                    if out is None:
                        return None
                return out
            return None

        return _resolve(val_id, frozenset(), depth)

    def _rebuild_staged_fill_mask(self, mask_id, op_by_id, M, N, kind="store"):
        """Structurally rebuild a cooperative over-threadgroup store's or load's
        per-element mask as a boolean MSL expression in terms of
        ``(_fill_row, _fill_col)``, or raise ``MetalNonRecoverableError`` if it
        cannot be soundly reconstructed (correct-or-refuse).

        Real kernels guard such a store with a row/col bounds check, e.g.
        ``offs_m[:, None] < N_CTX``.  We resolve the comparison's two operands
        with the SAME structural term-walk used for the staged offset
        (``_staged_fill_terms``: make_range -> _fill_row/_fill_col, block
        constants pass through), then render the comparison.  Anything that is
        not a single ``arith.cmpi`` over structurally-resolvable operands
        refuses rather than guess -- the old global-make_range string heuristic
        either dropped the mask entirely (a latent OOB silent-wrong for
        non-block-multiple bounds) or hardcoded ``(_fill_row + r_8) < N_CTX``.
        """
        from triton_msl.errors import MetalNonRecoverableError

        def _refuse():
            raise MetalNonRecoverableError(
                f"masked cooperative {kind} of a tile larger than the "
                "threadgroup: the per-element mask is not a single "
                "structurally-resolvable row/col bounds comparison and cannot "
                "be safely reconstructed. Refusing (correct-or-refuse).",
                op_name=f"tt.{kind}",
            )

        def _render(terms, cast):
            """Render a _staged_fill_terms list to a scalar MSL expr (each
            comparison operand wrapped in the predicate's sign-cast, matching
            _lower_cmpi), or None if a make_range extent is wrong."""
            if terms is None:
                return None
            out = []
            for r in terms:
                if r[0] == "range":
                    _, dim, coeff, extent, start = r
                    want = M if dim == 0 else N
                    if extent != want:
                        return None
                    idx = "(int)_fill_row" if dim == 0 else "(int)_fill_col"
                    if start != 0:
                        idx = f"({start} + {idx})"
                    out.append(idx if coeff is None else f"({idx} * {coeff})")
                else:
                    out.append(r[1])
            expr = " + ".join(out) if out else "0"
            return f"{cast}({expr})" if cast else expr

        # Unwrap pass-through wrappers (broadcast/splat/extension) to the cmpi.
        cur = mask_id
        for _ in range(16):
            op = op_by_id.get(cur)
            if op is None:
                _refuse()
            if op.op == "arith.cmpi":
                if len(op.operand_ids) < 2:
                    _refuse()
                # Predicate: prefer the text-parsed name (authoritative), else
                # the int enum -- exactly as _lower_cmpi resolves it.
                pred_name = op.attrs.get("predicate_name")
                pred_int = op.attrs.get("predicate")
                if pred_name and pred_name in CMPI_NAMED:
                    sym = CMPI_NAMED[pred_name]
                elif pred_int is not None and pred_int in CMPI_PREDICATES:
                    sym = CMPI_PREDICATES[pred_int]
                else:
                    _refuse()
                # Sign-cast each operand consistently with _lower_cmpi so the
                # comparison can't be flipped by C++ unsigned promotion.
                is_unsigned = pred_name in ("ult", "ule", "ugt", "uge") if pred_name else pred_int in (6, 7, 8, 9)
                is_signed = pred_name in ("slt", "sle", "sgt", "sge") if pred_name else pred_int in (2, 3, 4, 5)
                cast = "(uint)" if is_unsigned else "(int)" if is_signed else ""
                lhs = _render(self._staged_fill_terms(op.operand_ids[0], None, None, op_by_id), cast)
                rhs = _render(self._staged_fill_terms(op.operand_ids[1], None, None, op_by_id), cast)
                if lhs is None or rhs is None:
                    _refuse()
                if any(self.env_types.get(oid) in ("i64", "u64", "ui64") for oid in op.operand_ids[:2]):
                    # A 64-bit comparison: the flattened sums above are exact at 32
                    # bits only (a 64-bit bound cast to int compares its low half, and
                    # a 32-bit partial sum is extended by its C type, not the
                    # source's). Render each side at the source's widths.
                    sides = [self._staged_fill_typed(oid, None, op_by_id, M, N) for oid in op.operand_ids[:2]]
                    if any(side is None or side[1] != 64 for side in sides):
                        _refuse()
                    lhs, rhs = (f"(ulong)({side[0]})" if is_unsigned else side[0] for side in sides)
                return f"({lhs} {sym} {rhs})"
            if op.op in ("tt.broadcast", "tt.splat", "arith.extsi", "arith.extui", "arith.trunci") and op.operand_ids:
                cur = op.operand_ids[0]
                continue
            _refuse()
        _refuse()

    def _rebuild_staged_fill_offset(self, addptr_id, op_by_id, base_ptr, M, N, axis_dim_hint=None):
        """Per-load structural rebuild of a shared-memory-staged tt.dot
        operand's global address as a function of (_fill_row, _fill_col).

        ``ttg.local_alloc`` stages a 2-D tile (M x N) into threadgroup memory
        for a downstream ``tt.dot``.  The cooperative fill loop walks element
        ``_sa`` over ``[0, M*N)`` with ``_fill_row = _sa / N`` and
        ``_fill_col = _sa % N``.  To stage each element from the correct global
        address we rebuild the load's offset in terms of those.

        We do this STRUCTURALLY and PER-LOAD by walking THIS load's
        ``addptr -> (broadcast) -> expand_dims -> make_range`` chain.  Each
        additive offset term reduces to either a 2-D index (row = dim 0 /
        col = dim 1, decided by the ``tt.expand_dims`` ``axis`` attribute:
        axis=1 => make_range sits at dim 0 = row; axis=0 => dim 1 = col) with
        an optional stride coefficient, or a block-constant scalar.  BOTH the
        bare and the strided term are rebuilt symmetrically -- we never assume
        "row == strided term, col == bare index".  That heuristic silently
        mis-stages a pre-transposed K operand
        (``K + offs_d[:,None] + offs_n[None,:]*HD``) whose bare stride-1 term
        is the row and whose strided term is the column.

        ``axis_dim_hint`` is reserved for a proven 1-D broadcast source: the
        consumer has already established whether its bare make_range is the row
        (0) or column (1) coordinate.  Ordinary 2-D staged operands leave it
        ``None`` and must carry their own expand_dims proof.

        Correct-or-refuse: if any term of a staged dot operand cannot be
        resolved to (dim, stride) | constant, raise ``MetalNonRecoverableError``
        rather than emit a guessed -- possibly transposed -- staging.
        """
        # The per-term structural decomposition lives in the shared
        # ``_staged_fill_terms`` method so the staged-offset rebuild here and
        # the staged-store-mask rebuild (``_rebuild_staged_fill_mask``) resolve
        # row/col indices identically (single-source, no twin to drift).

        # Walk the addptr chain, collecting every additive offset term plus any
        # block-constant offset folded onto the base pointer (a scalar
        # program-id / batch-stride addptr applied before the splat).
        parts = []  # row/col index contributions
        const_terms = []  # block-constant contributions (same for all _sa)
        offsets = []  # each tensor addptr's offset operand
        base_offset = None  # the scalar offset folded onto the splatted base pointer
        wide = ("i64", "u64", "ui64")

        def _refuse():
            raise MetalNonRecoverableError(
                "triton-msl: cannot structurally resolve an offset term of a "
                "shared-memory-staged tt.dot operand to a (row/col, stride) "
                "index or a block-constant. Refusing to emit a "
                "possibly-transposed staging rather than risk a silent "
                "miscompute (correct-or-refuse)."
            )

        def _add(sub):
            if sub is None:
                _refuse()
            for r in sub:
                if r[0] == "range":
                    _, dim, coeff, extent, start = r
                    want = M if dim == 0 else N
                    if extent != want:
                        raise MetalNonRecoverableError(
                            "triton-msl: shared-memory-staged tt.dot operand "
                            f"has a make_range extent ({extent}) that does not "
                            f"match the staged tile dimension ({want}); the "
                            "(_sa/N, _sa%N) fill decomposition would mis-stage "
                            "it. Refusing (correct-or-refuse)."
                        )
                    idx = "(int)_fill_row" if dim == 0 else "(int)_fill_col"
                    if start != 0:
                        idx = f"({start} + {idx})"
                    parts.append(idx if coeff is None else f"{idx} * {coeff}")
                else:
                    const_terms.append(r[1])

        cur = addptr_id
        reached_base = False
        for _ in range(64):
            op = op_by_id.get(cur)
            if op is None:
                _refuse()
            if op.op == "tt.addptr":
                if len(op.operand_ids) < 2:
                    _refuse()
                _add(self._staged_fill_terms(op.operand_ids[1], None, axis_dim_hint, op_by_id))
                offsets.append(op.operand_ids[1])
                cur = op.operand_ids[0]
                continue
            if op.op == "tt.broadcast" and op.operand_ids:
                cur = op.operand_ids[0]
                continue
            if op.op == "tt.splat":
                # Reached the splatted base pointer. If that base pointer
                # carries a scalar (block-constant, program-id/batch-stride)
                # addptr offset, fold it in as a constant term.
                if not op.operand_ids:
                    _refuse()
                # Raw pointer arguments are not themselves entries in
                # ``env_is_ptr``; the splat result is.  A scalar addptr below
                # the splat is propagated there as well, so this is the
                # complete base-address fact to validate and fold.
                sp = self.env_is_ptr.get(op.id)
                if sp is None or sp[0] != base_ptr:
                    _refuse()
                if isinstance(sp[1], str) and sp[1] not in ("", "0"):
                    const_terms.append(sp[1])
                    base_offset = sp[1]
                reached_base = True
                break
            # A value such as a pointer-valued select can have a valid
            # per-thread address in ``env_is_ptr`` while still lacking a
            # proof that reconstructs that address for every cooperative
            # fill coordinate.  Stopping here and returning offset zero used
            # to smear ptr[0] across the staged tile (GPU-confirmed silent
            # wrong).  Only the complete addptr/broadcast/splat proof above
            # may authorize reconstruction.
            _refuse()

        if not reached_base:
            _refuse()

        def _base_is_wide(pointer):
            """Does a scalar addptr below the splat add a 64-bit offset?"""
            for _ in range(64):
                node = op_by_id.get(pointer)
                if node is None or node.op != "tt.addptr" or len(node.operand_ids) < 2:
                    return False
                if self.env_types.get(node.operand_ids[1]) in wide:
                    return True
                pointer = node.operand_ids[0]
            return False

        if any(self.env_types.get(oid) in wide for oid in offsets) or _base_is_wide(op.operand_ids[0]):
            # The flattened sum is exact while every term is 32 bits wide. With a
            # 64-bit term in it, a 32-bit partial sum is extended by its C type: one
            # holding ``pid`` (a ``uint``) and a negative term lands 2**32 elements
            # away. Render each addptr's offset at the source's widths and add them as
            # the pointer does, sign-extended.
            typed = [self._staged_fill_typed(oid, axis_dim_hint, op_by_id, M, N) for oid in offsets]
            if any(term is None for term in typed):
                _refuse()
            terms = [f"(long)({term[0]})" for term in reversed(typed)]
            return " + ".join(terms + ([base_offset] if base_offset else []))

        all_parts = parts + const_terms
        return " + ".join(all_parts) if all_parts else "0"

    def _rebuild_staged_fill_load(self, ptr_id, op_by_id, M, N, axis_dim_hint=None, depth=4):
        """Rebuild one cooperative fill's source load at ``(_fill_row, _fill_col)``.

        Ordinary pointer tensors reduce to ``base[proved_offset]``.  A pointer-valued
        select is also exact when its condition is kernel-uniform: rebuild BOTH arm
        addresses independently and select their loaded values.  The per-thread pointer
        variable emitted by ``_lower_select`` cannot be reused here because it contains
        only the current lane's offset; doing so used ``ptr[0]`` for every additional
        cooperative slot and silently smeared one element across the staged tile.
        """
        from triton_msl.errors import MetalNonRecoverableError

        def _refuse(reason):
            raise MetalNonRecoverableError(
                "triton-msl: cannot reconstruct a shared-memory-staged tt.dot "
                f"load for every cooperative fill coordinate ({reason}). Refusing "
                "rather than change pointer-selection semantics.",
                op_name="tt.load",
            )

        if depth <= 0:
            _refuse("pointer expression nesting exceeds the proved limit")
        op = op_by_id.get(ptr_id)
        if op is not None and op.op == "arith.select":
            if len(op.operand_ids or []) != 3:
                _refuse("pointer select does not have exactly three operands")
            cond_id, true_id, false_id = op.operand_ids
            if cond_id in self.env_array:
                _refuse("pointer select condition is a per-element register array")
            cond_shape = self.env_shapes.get(cond_id)
            if cond_shape != () and cond_id not in self._is_splat:
                _refuse("pointer select condition is not proved kernel-uniform")
            true_expr = self._rebuild_staged_fill_load(true_id, op_by_id, M, N, axis_dim_hint, depth - 1)
            false_expr = self._rebuild_staged_fill_load(false_id, op_by_id, M, N, axis_dim_hint, depth - 1)
            return f"({self._lookup(cond_id)} ? ({true_expr}) : ({false_expr}))"

        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info is None:
            _refuse("pointer provenance is absent")
        base_ptr = ptr_info[0]
        offset = self._rebuild_staged_fill_offset(ptr_id, op_by_id, base_ptr, M, N, axis_dim_hint)
        return f"{base_ptr}[{offset}]"

    def _lower_local_alloc(self, ssa: SSAValue):
        """ttg.local_alloc -> write tensor to threadgroup shared memory.

        Cooperatively fills shared memory using a strided loop so all
        threads contribute, then barriers. When inside a wrapping loop,
        closes/reopens it so the fill is standalone.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            return

        # The proven oversized-dot plan declares K-slice buffers instead of
        # full input tiles (and fuses a chain intermediate rather than
        # materializing it).
        if self._lower_dot_kchunk_alloc(ssa):
            return

        src_var = self._lookup(ssa.operand_ids[0])
        # ttg.local_alloc output is !ttg.memdesc<MxNxT, ...> — _extract_shape
        # only handles tensor<...>, so try the input operand type first.
        shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        if not shape or len(shape) < 2:
            # Try parsing memdesc directly: !ttg.memdesc<32x32xf32, ...>
            import re

            m = re.search(r"memdesc<((?:\d+x)+)", ssa.type_str or "")
            if m:
                dims_str = m.group(1).rstrip("x")
                shape = tuple(int(d) for d in dims_str.split("x") if d)
        if not shape or len(shape) < 2:
            self._emit_passthrough(ssa)
            return

        M, N = shape[0], shape[1]
        total = M * N

        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        shared_dtype = "fp32" if is_float else "i32"

        shared_name = f"smem_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Close wrapping loop if active — the fill must be standalone.  Cooperative
        # loops stride by the PHYSICAL Metal dispatch, never by their logical tile.
        dispatch_threads = getattr(self, "_actual_dispatch_threads", self.effective_block_size)
        in_loop = self._needs_wrapping
        if in_loop:
            self.kb.raw_line(f"    }}")  # close wrapping loop

        # Trace back to find the source pointer for re-loading values
        # into shared memory directly from global memory.
        src_ptr_name = None
        trace_id = ssa.operand_ids[0]
        for op in self.graph.ops:
            if op.id == trace_id:
                if op.op == "tt.load" and op.operand_ids:
                    # The load's pointer operand
                    ptr_id = op.operand_ids[0]
                    ptr_info = self.env_is_ptr.get(ptr_id)
                    if ptr_info:
                        src_ptr_name = ptr_info[0]
                break

        # Cooperative strided fill: each thread writes elements stride-apart.
        # For 2D tiles where total > block_size, each thread must handle
        # multiple elements. The per-thread src_var only holds ONE value
        # (for lid), so we re-load from global memory directly into shared
        # memory with corrected row/col addressing for each _sa position.
        #
        # Strategy: trace back to the tt.load that produced the staged tile,
        # collect any post-load transform (e.g. * scale), then rebuild the
        # load's GLOBAL address as a function of (_fill_row, _fill_col) by
        # walking THIS load's addptr -> (broadcast) -> expand_dims ->
        # make_range chain structurally (see _rebuild_staged_fill_offset).
        # The structural, per-load analysis is correct even when a make_range
        # plays opposite (row/col) roles in two different loads -- e.g. a
        # pre-transposed K operand -- which the old global-_make_range_dim
        # string heuristic silently mis-staged.
        #
        #   for (_sa = lid; _sa < total; _sa += dispatch_threads) {
        #       row = _sa / N; col = _sa % N;
        #       shared[_sa] = ptr[<rebuilt offset(row, col)>];
        #   }

        # Build a flat op list / lookup that crosses scf.for + if regions so
        # the source trace can span scopes.
        def _all_ops(ops):
            for o in ops:
                yield o
                if o.region_ops:
                    yield from _all_ops(o.region_ops)
                if o.else_ops:
                    yield from _all_ops(o.else_ops)

        all_ops = list(_all_ops(self.graph.ops))
        op_by_id = {o.id: o for o in all_ops}

        # Trace the local_alloc's source back to its tt.load, recording the
        # load's pointer/addptr id and any post-load transforms applied
        # between the load and the local_alloc (e.g. arith.mulf by a scale).
        load_ptr_info = None
        load_addptr_id = None
        load_op = None
        post_load_ops = []  # (op_type, extra_operand_var) chain, in load order
        cur_id = ssa.operand_ids[0]
        for _depth in range(10):
            op = op_by_id.get(cur_id)
            if op is None:
                break
            if op.op == "tt.load" and op.operand_ids:
                load_addptr_id = op.operand_ids[0]
                load_ptr_info = self.env_is_ptr.get(load_addptr_id)
                load_op = op
                break
            if op.operand_ids:
                if len(op.operand_ids) >= 2:
                    post_load_ops.insert(0, (op.op, self._lookup(op.operand_ids[1])))
                cur_id = op.operand_ids[0]
            else:
                break

        def _apply_post_load(expr):
            for op_type, other_var in post_load_ops:
                if "mul" in op_type:
                    expr = f"({expr} * {other_var})"
                elif "sub" in op_type:
                    expr = f"({expr} - {other_var})"
                elif "add" in op_type:
                    expr = f"({expr} + {other_var})"
            return expr

        def _staged_load_guard():
            """Resolve the staged load's MASK (and 'other') per (_fill_row, _fill_col).

            The staged fill re-reads the tile straight from global memory, so it must
            honour the mask the Triton ``tl.load`` carried. Dropping it reads OUT OF
            BOUNDS whenever the tile overhangs the tensor -- e.g. a varlen FA kernel
            whose last K/V block starts past the final sequence: the masked lanes then
            multiply a correct ``p == 0`` by garbage, and ``0 * NaN == NaN`` poisons the
            accumulator (a NaN silent-wrong, not a loud failure). The scalar path
            already honours it; this is the same guard, resolved with the same
            structural machinery the cooperative STORE path uses, refusing rather than
            guessing when the mask is not structurally resolvable.

            Returns (mask_expr | None, other_expr).
            """
            if load_op is None or len(load_op.operand_ids) < 2:
                return None, "0.0f"
            mask_id = None
            other_expr = "0.0f"
            for _op_id in load_op.operand_ids[1:]:
                if mask_id is None and (_op_id in self.env_is_mask or self._is_mask(_op_id)):
                    mask_id = _op_id
                elif mask_id is not None:
                    # 'other' fill value: only a uniform scalar is sound here, since
                    # the staged element is addressed by (_fill_row, _fill_col) rather
                    # than by lid. A per-element 'other' array refuses below.
                    if _op_id in self.env_array or _op_id in getattr(self, "_shared_mem_descs", {}):
                        from triton_msl.errors import MetalNonRecoverableError

                        raise MetalNonRecoverableError(
                            "masked cooperative load of a tile larger than the "
                            "threadgroup: the load's 'other' operand is a per-element "
                            "array, which cannot be re-addressed for the staged fill. "
                            "Refusing (correct-or-refuse).",
                            op_name="tt.load",
                        )
                    other_expr = self._lookup(_op_id)
                    break
            if mask_id is None:
                return None, other_expr
            # Raises MetalNonRecoverableError when not structurally resolvable.
            return self._rebuild_staged_fill_mask(mask_id, op_by_id, M, N, kind="load"), other_expr

        def _emit_staged_fill(raw_expr):
            """Emit the guarded per-element staged assignment.

            The mask wraps the RAW load and any post-load transform is applied to the
            selected value -- i.e. ``(mask ? load : other) * scale``, matching Triton's
            order -- so a non-zero ``other`` is not silently scaled.
            """
            mask_expr, other_expr = _staged_load_guard()
            if mask_expr is None:
                self.kb.raw_line(f"        {shared_name}[_sa] = {_apply_post_load(raw_expr)};")
            else:
                sel = f"({mask_expr} ? ({raw_expr}) : ({other_expr}))"
                self.kb.raw_line(f"        {shared_name}[_sa] = {_apply_post_load(sel)};")

        # SMEM-BACKED SOURCE (a dot result or an smem-quantized cast, possibly through
        # a ttg.convert_layout): stage per element from the SOURCE ARRAY. The per-thread
        # env value holds only element lid, so the value fallbacks below smear it across
        # strided slots when total > block_size (silent-wrong) — and before 2026-08-30
        # they interpolated the raw array name (invalid MSL, chain-dot fp16 recon).
        _src_desc = getattr(self, "_shared_mem_descs", {}).get(ssa.operand_ids[0])
        if _src_desc is None:
            _cur_src = ssa.operand_ids[0]
            for _ in range(6):
                _o = op_by_id.get(_cur_src)
                if _o is None or _o.op != "ttg.convert_layout" or not _o.operand_ids:
                    break
                _cur_src = _o.operand_ids[0]
                _src_desc = getattr(self, "_shared_mem_descs", {}).get(_cur_src)
                if _src_desc is not None:
                    break
        if _src_desc is not None and (load_ptr_info is None or getattr(self, "_source_biased_replay", False)):
            _src_smem, _src_shape, _ = _src_desc
            _src_total = 1
            for _d in _src_shape:
                _src_total *= _d
            if _src_total != total:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"ttg.local_alloc of a shared-memory-backed value whose element "
                    f"count ({_src_total}) differs from the alloc shape ({M}x{N}): the "
                    f"element-wise copy has no proven index mapping. Refusing rather "
                    f"than mis-stage.",
                    op_name="ttg.local_alloc",
                )
            self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {dispatch_threads}u) {{")
            self.kb.raw_line(f"        {shared_name}[_sa] = {_src_smem}[_sa];")
            self.kb.raw_line(f"    }}")
        elif total > dispatch_threads and self._is_2d:
            if load_ptr_info is not None:
                # Per-load structural rebuild of each staged element's global
                # address (correct-or-refuse for transposed/strided/selected
                # operands).  This returns a complete load expression because
                # a selected pointer may have a different base in each arm.
                load_expr = self._rebuild_staged_fill_load(load_addptr_id, op_by_id, M, N)
                self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {dispatch_threads}u) {{")
                self.kb.raw_line(f"        uint _fill_row = _sa / {N}u;")
                self.kb.raw_line(f"        uint _fill_col = _sa % {N}u;")
                _emit_staged_fill(load_expr)
                self.kb.raw_line(f"    }}")
            else:
                # No load pointer and not smem-backed: the per-thread scalar holds ONE
                # element, and total > block_size here, so the strided copy would smear
                # each thread's value across its slots — silent-wrong (dot-recovery
                # hardening 2026-08-30; was the documented-but-unenforced assumption).
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"ttg.local_alloc staging of a computed (non-load, non-smem-backed) "
                    f"{M}x{N} value with {total} elements > {dispatch_threads} threads: each thread "
                    f"holds one element, so the cooperative fill cannot reconstruct the "
                    f"full tile. Refusing rather than smear per-thread values.",
                    op_name="ttg.local_alloc",
                )
        elif src_ptr_name:
            if self._is_2d and load_ptr_info is not None:
                # 2-D tile staged for a tt.dot but small enough that
                # total <= block_size. The flat base_ptr[_sa] copy assumes a
                # contiguous row-major source; rebuild structurally so a
                # transposed/strided source stages correctly (or refuses).
                load_expr = self._rebuild_staged_fill_load(load_addptr_id, op_by_id, M, N)
                self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {dispatch_threads}u) {{")
                self.kb.raw_line(f"        uint _fill_row = _sa / {N}u;")
                self.kb.raw_line(f"        uint _fill_col = _sa % {N}u;")
                _emit_staged_fill(load_expr)
                self.kb.raw_line(f"    }}")
            else:
                # Flat contiguous copy: the element address is _sa itself, so there is
                # no (_fill_row, _fill_col) to resolve a mask against. A masked load
                # here would read past the tensor exactly like the rebuilt paths did,
                # so refuse rather than drop the guard.
                if load_op is not None and len(load_op.operand_ids) >= 2:
                    from triton_msl.errors import MetalNonRecoverableError

                    for _op_id in load_op.operand_ids[1:]:
                        if _op_id in self.env_is_mask or self._is_mask(_op_id):
                            raise MetalNonRecoverableError(
                                "masked cooperative load staged as a flat contiguous "
                                "copy: the per-element mask cannot be reconstructed for "
                                "this fill, and dropping it would read out of bounds. "
                                "Refusing (correct-or-refuse).",
                                op_name="tt.load",
                            )
                self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {dispatch_threads}u) {{")
                self.kb.raw_line(f"        {shared_name}[_sa] = {src_ptr_name}[_sa];")
                self.kb.raw_line(f"    }}")
        else:
            # Fallback: use the per-thread value. Correct ONLY when total <= block_size
            # (each thread writes exactly its own element); a larger tile would smear
            # each thread's single value across its strided slots — enforce the
            # documented assumption instead of assuming it (dot-recovery 2026-08-30).
            if total > dispatch_threads:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"ttg.local_alloc staging of a per-thread value with {total} "
                    f"elements > {dispatch_threads} threads and no reconstructible source (no load "
                    f"pointer, not shared-memory-backed). Refusing rather than smear "
                    f"per-thread values across the tile.",
                    op_name="ttg.local_alloc",
                )
            self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {dispatch_threads}u) {{")
            self.kb.raw_line(f"        {shared_name}[_sa] = {src_var};")
            self.kb.raw_line(f"    }}")

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reopen wrapping loop if it was active
        if in_loop:
            total_elems = getattr(self, "_total_elements", self.effective_block_size)
            self.kb.raw_line(
                f"    for (uint _loop_e = lid; _loop_e < {total_elems}u; _loop_e += {dispatch_threads}u) {{"
            )

        # Store shared array name for local_load to reference
        self.env[ssa.id] = shared_name
        self.env_types[ssa.id] = src_dtype
        self.env_shapes[ssa.id] = shape
        # Mark as shared memory descriptor
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (shared_name, shape, shared_dtype)
        if not hasattr(self, "_shared_mem_orientations"):
            self._shared_mem_orientations = {}
        self._shared_mem_orientations[ssa.id] = False
        # Also mark the source operand as having its data in shared memory.
        # This allows downstream reduces on the source to skip redundant copies.
        if ssa.operand_ids:
            self._shared_mem_descs[ssa.operand_ids[0]] = (shared_name, shape, shared_dtype)

    def _lower_local_load(self, ssa: SSAValue):
        """ttg.local_load -> marker for shared memory access.

        Data is already in shared memory from local_alloc. This just
        propagates the shared memory descriptor so tt.dot can find it.
        No actual code is emitted — the dot reads directly from shared.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            return

        src_id = ssa.operand_ids[0]
        shared_info = getattr(self, "_shared_mem_descs", {}).get(src_id)
        if not shared_info:
            # No shared memory descriptor — passthrough
            self._emit_passthrough(ssa)
            return

        shared_name, shape, shared_dtype = shared_info

        # Propagate shared memory descriptor so tt.dot can find the array
        self.env[ssa.id] = shared_name
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = shape
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = shared_info
        if not hasattr(self, "_shared_mem_orientations"):
            self._shared_mem_orientations = {}
        orientations = self._shared_mem_orientations
        if src_id not in orientations:
            # A local_alloc creates the root descriptor view. Its physical fill
            # and logical shape have the same orientation; only a proved
            # memdesc_trans may toggle that fact. The K-chunk allocator owns its
            # descriptor through a specialized lowering path, so it can reach
            # here without the ordinary local_alloc helper having installed the
            # otherwise-identical identity fact.
            source = None
            pending = list(self.graph.ops)
            visited = set()
            while pending:
                candidate = pending.pop()
                if candidate.id in visited:
                    continue
                visited.add(candidate.id)
                if candidate.id == src_id:
                    source = candidate
                    break
                pending.extend(candidate.region_ops or [])
                pending.extend(candidate.else_ops or [])
            if source is not None and source.op == "ttg.local_alloc":
                orientations[src_id] = False
            else:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "ttg.local_load has shared descriptor data without proved orientation",
                    op_name="ttg.local_load",
                )
        orientations[ssa.id] = orientations[src_id]

    def _lower_memdesc_trans(self, ssa: SSAValue):
        """ttg.memdesc_trans -> transpose shared memory descriptor.

        This op changes the logical access order of a shared memory array
        without moving data. We propagate the shared memory descriptor and
        mark it as transposed so that tt.dot uses the correct indexing:
        B_trans[k, col] accesses physical B[col * K + k] instead of B[k * N + col].
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            return

        src_id = ssa.operand_ids[0]
        order = ssa.attrs.get("order")
        shape = self.env_shapes.get(src_id)
        if shape is None or not isinstance(order, (list, tuple)) or sorted(order) != list(range(len(shape))):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "ttg.memdesc_trans requires a proven operation-owned permutation; refusing to guess",
                op_name="ttg.memdesc_trans",
            )
        orientations = getattr(self, "_shared_mem_orientations", {})
        if src_id not in orientations:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "ttg.memdesc_trans source has no proved descriptor orientation",
                op_name="ttg.memdesc_trans",
            )

        if list(order) == list(range(len(shape))):
            # An identity descriptor view preserves every alias, shape and
            # transposed-state fact of its source exactly.
            self._emit_passthrough(ssa)
            if hasattr(self, "_shared_mem_descs") and src_id in self._shared_mem_descs:
                self._shared_mem_descs[ssa.id] = self._shared_mem_descs[src_id]
            orientations[ssa.id] = orientations[src_id]
            return

        inner_swap = list(range(len(shape)))
        if len(shape) < 2:
            inner_swap = []
        else:
            inner_swap[-2:] = [len(shape) - 1, len(shape) - 2]
        if list(order) != inner_swap:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "ttg.memdesc_trans supports only identity or the proven inner-dimension swap; "
                f"permutation {list(order)!r} is not reproduced",
                op_name="ttg.memdesc_trans",
            )

        # Propagate env, env_types, env_shapes from source
        self._emit_passthrough(ssa)

        # Propagate shared memory descriptor with transposed flag
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        shared_info = self._shared_mem_descs.get(src_id)
        if shared_info:
            shared_name, shape, shared_dtype = shared_info
            trans_shape = tuple(shape[i] for i in order)
            self._shared_mem_descs[ssa.id] = (shared_name, trans_shape, shared_dtype)
            self.env_shapes[ssa.id] = trans_shape
            orientations[ssa.id] = not orientations[src_id]

    # -- Matrix multiply (tt.dot) --

    def _lower_dot(self, ssa: SSAValue):
        """tt.dot -> generic scalar matmul using shared memory operands.

        For simple dot kernels (no stride args), each thread computes one
        element of C: C[row, col] = sum(A[row, k] * B[k, col]).
        This is a naive per-thread scalar loop — simdgroup MMA will be
        added in a follow-up for hardware-accelerated matmul.

        When inside a wrapping loop, closes/reopens it so the dot
        computation is standalone with its own strided loop.

        For strided kernels, this should not be reached — they go through
        _lower_dot_via_prebuilt_template() instead.
        """
        if len(ssa.operand_ids) < 3:
            self.kb.unsupported("UNSUPPORTED: tt.dot with < 3 operands")
            return
        if self._lower_dot_kchunk(ssa):
            return

        a_id = ssa.operand_ids[0]
        b_id = ssa.operand_ids[1]
        acc_var = self._lookup(ssa.operand_ids[2])

        # Get A shape (M, K) from type string of operand 0
        a_type = self._find_op_type_str(a_id)
        a_shape = _extract_shape(a_type) if a_type else None
        # Get B shape (K, N)
        b_type = self._find_op_type_str(b_id)
        b_shape = _extract_shape(b_type) if b_type else None

        if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
            # Returning the accumulator here drops A@B silently (audit #165 —
            # same class as the #157 matmul-epilogue refusals). Refuse loudly.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing to emit silently-wrong output: tt.dot operand shapes "
                f"({a_shape} x {b_shape}) are not both 2-D; the generic dot path "
                f"cannot lower this and would silently return the accumulator, "
                f"dropping the matmul. File an issue or restructure the dot."
            )

        M, K = a_shape[0], a_shape[1]
        K2, N = b_shape[0], b_shape[1]

        # PRIMITIVE-LEVEL large-tile guard (independent of the surrounding idiom).
        # The validated dot paths — matmul templates, the head_dim<=128 FA2 template,
        # _detect_simple_dot — all `return` early in lower(); a dot reaching this
        # generic per-thread fallback escaped every one of them. That fallback is
        # only correct for small tiles and silently mis-computes larger ones (the
        # per-thread mapping is wrong at 16/32/64+). The FlashAttention head_dim>64
        # refusal lives inside a max-substring heuristic, so a max-less head_dim-128
        # attention variant slips past it and lands here — refuse on the ACTUAL
        # unsupported primitive (the oversized dot tile), not the idiom, so no shape
        # of attention/fused-matmul can reach the wrong per-thread path silently.
        _dot_max = max(M, K, N)
        if _dot_max > 64:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing a tt.dot with tile dim {_dot_max} (> 64) on the generic "
                f"per-thread path: this fallback is validated only for tile dims <= 64 "
                f"and silently mis-computes larger matmuls. Large matmuls go through "
                f"the validated simdgroup templates and FlashAttention (head_dim <= "
                f"128) through the FA2 template; a dot reaching here escaped them "
                f"(e.g. a max-less head_dim-128 attention). Refusing rather than emit "
                f"silently-wrong output."
            )

        # Get shared memory names for A and B
        # Trace through local_load to find the shared memory arrays
        a_shared = getattr(self, "_shared_mem_descs", {}).get(a_id)
        b_shared = getattr(self, "_shared_mem_descs", {}).get(b_id)

        if not a_shared:
            for op in self.graph.ops:
                if op.id == a_id and op.op == "ttg.local_load" and op.operand_ids:
                    a_shared = getattr(self, "_shared_mem_descs", {}).get(op.operand_ids[0])
                    break
        if not b_shared:
            for op in self.graph.ops:
                if op.id == b_id and op.op == "ttg.local_load" and op.operand_ids:
                    b_shared = getattr(self, "_shared_mem_descs", {}).get(op.operand_ids[0])
                    break

        if not a_shared or not b_shared:
            # No shared-memory operands -> the generic simdgroup dot path can't
            # lower this. Returning the accumulator drops A@B silently; refuse
            # loudly instead (audit #165).
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "Refusing to emit silently-wrong output: tt.dot operands are not "
                "in threadgroup (shared) memory (no ttg.local_load source); the "
                "generic dot path would silently return the accumulator, dropping "
                "the matmul. File an issue."
            )

        a_smem, _, _ = a_shared
        b_smem, _, _ = b_shared

        # Check if operands were transposed via memdesc_trans.
        # Transposed arrays need swapped indexing: B_trans[k, col] reads
        # physical B_orig[col, k] = smem[col * K_orig + k].
        orientations = getattr(self, "_shared_mem_orientations", {})

        def descriptor_orientation(value_id):
            if value_id in orientations:
                return orientations[value_id]
            op = next((candidate for candidate in self.graph.ops if candidate.id == value_id), None)
            if op is not None and op.op == "ttg.local_load" and op.operand_ids:
                source_id = op.operand_ids[0]
                if source_id in orientations:
                    return orientations[source_id]
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "generic dot shared operand has no proved descriptor orientation",
                op_name="tt.dot",
            )

        a_trans = descriptor_orientation(a_id)
        b_trans = descriptor_orientation(b_id)

        # For transposed operands, the physical storage dimensions differ from
        # the logical ones. K is the inner/contraction dimension.
        # A normal: smem[row * K + k], A transposed: smem[k * M + row]
        # B normal: smem[k * N + col], B transposed: smem[col * K + k]
        if a_trans:
            a_index = f"{a_smem}[_dk * {M}u + _dot_row]"
        else:
            a_index = f"{a_smem}[_dot_row * {K}u + _dk]"
        if b_trans:
            b_index = f"{b_smem}[_dot_col * {K}u + _dk]"
        else:
            b_index = f"{b_smem}[_dk * {N}u + _dot_col]"

        total = M * N
        dispatch_threads = getattr(self, "_actual_dispatch_threads", self.effective_block_size)

        # Close wrapping loop if active -- dot needs standalone computation
        in_loop = self._needs_wrapping
        if in_loop:
            self.kb.raw_line(f"    }}")  # close wrapping loop

        # Check if the accumulator is a shared-memory-backed oversized array
        # (e.g. the 32x64 accumulator in flash attention with HEAD_DIM=64).
        # If so, read the init from shared memory per-element and write the
        # result back to the SAME shared array (no new allocation needed).
        acc_smem = getattr(self, "_shared_mem_descs", {}).get(ssa.operand_ids[2])
        acc_is_smem = False
        if acc_smem:
            acc_shape = acc_smem[1]
            acc_total = 1
            for d in acc_shape:
                acc_total *= d
            if acc_total > dispatch_threads:
                acc_is_smem = True

        if acc_is_smem:
            # Use the existing smem array for both init and result
            result_smem = acc_smem[0]
        else:
            # Declare a threadgroup result array to store per-thread dot results
            result_smem = f"smem_dot_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(result_smem, dtype="fp32", size=total)

        # Each thread computes one or more elements of C via strided loop
        self.kb.raw_line(f"    for (uint _de = lid; _de < {total}u; _de += {dispatch_threads}u) {{")
        self.kb.raw_line(f"        uint _dot_row = _de / {N}u;")
        self.kb.raw_line(f"        uint _dot_col = _de % {N}u;")
        if acc_is_smem:
            # Read accumulator init from shared memory per-element
            self.kb.raw_line(f"        float _dot_sum = {result_smem}[_de];")
        else:
            self.kb.raw_line(f"        float _dot_sum = {acc_var};")
        self.kb.raw_line(f"        for (uint _dk = 0; _dk < {K}u; _dk++) {{")
        self.kb.raw_line(f"            _dot_sum += {a_index} * {b_index};")
        self.kb.raw_line(f"        }}")
        self.kb.raw_line(f"        {result_smem}[_de] = _dot_sum;")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reopen wrapping loop if it was active
        if in_loop:
            total_elems = getattr(self, "_total_elements", self.effective_block_size)
            self.kb.raw_line(
                f"    for (uint _loop_e = lid; _loop_e < {total_elems}u; _loop_e += {dispatch_threads}u) {{"
            )

        # The result for each thread's element comes from the shared result array
        result_var = self._next_var("dot")
        elem = self._lid_expr
        self.kb.raw_line(f"    float {result_var} = ({elem} < {total}u) ? {result_smem}[{elem}] : 0.0f;")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "fp32"
        self.env_shapes[ssa.id] = (M, N)
        # Track that this variable's data is already in shared memory at lid.
        # Downstream reduce can skip the copy and read from result_smem directly.
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (result_smem, (M, N), "fp32")

    # -- SCF (structured control flow) --

    # -- Atomic ops --

    def _lower_called_funcs(self):
        """Lower all callee (noinline) functions to MSL device functions.

        Each callee function becomes an MSL device function placed before
        the kernel in the output. The lowering reuses the same op-by-op
        approach as the kernel but with a fresh environment.

        Functions are emitted in reverse order so that callees appear
        before their callers (dependency order).
        """
        # Reverse so leaf functions come first
        for cfunc in reversed(self.graph.called_funcs):
            msl = self._lower_one_called_func(cfunc)
            self.kb._device_functions.append(msl)

    def _lower_one_called_func(self, cfunc: CalledFunc) -> str:
        """Lower a single CalledFunc to an MSL device function string.

        Creates a temporary lowering context (env, env_types, etc.) to
        avoid polluting the main kernel's namespace.
        """
        safe_name = self._sanitize_func_name(cfunc.name)

        # Determine return type
        if len(cfunc.return_types) == 0:
            ret_type = "void"
        elif len(cfunc.return_types) == 1:
            ret_type = triton_type_to_msl(_mlir_to_triton_dtype(cfunc.return_types[0]))
        else:
            # Multi-value return: use a struct
            ret_type = f"_ret_{safe_name}"

        # Build parameter list
        # Pointer params use 'volatile device' to match kernel buffer qualifiers,
        # which prevent the Metal shader compiler from hoisting loads.
        params = []
        for arg in cfunc.args:
            triton_dtype = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                inner = triton_type_to_msl(triton_dtype)
                params.append(f"volatile device {inner}* {arg.name}")
            else:
                msl_ty = triton_type_to_msl(triton_dtype)
                params.append(f"{msl_ty} {arg.name}")

        params_str = ", ".join(params)

        # Generate the body using a sub-lowerer
        sub = _DeviceFuncLowerer(cfunc, self.options)
        body_lines = sub.lower_body()
        self.kb.unsupported_reasons.extend(sub.unsupported_reasons)

        # Assemble the function
        lines = []

        # Multi-return struct definition
        if len(cfunc.return_types) > 1:
            struct_name = ret_type
            lines.append(f"struct {struct_name} {{")
            for i, rt in enumerate(cfunc.return_types):
                msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(rt))
                lines.append(f"    {msl_ty} v{i};")
            lines.append("};")
            lines.append("")

        lines.append(f"{ret_type} {safe_name}({params_str}) {{")
        for line in body_lines:
            lines.append(line)
        lines.append("}")

        return "\n".join(lines)

    def _lower_call(self, ssa: SSAValue):
        """Lower tt.call to an MSL function call.

        Handles:
        - Void calls (no return value)
        - Single return value
        - Multiple return values (via struct)
        """
        callee = ssa.attrs.get("callee", "unknown_fn")
        safe_callee = self._sanitize_func_name(callee)
        # A tt.call pointer operand is an address, not a load. `_lookup` alone
        # loses addptr provenance and produced `callee(A[offset])` where MSL
        # requires `callee(A + offset)`.
        args = []
        for oid in ssa.operand_ids:
            pointer = self.env_is_ptr.get(oid)
            if pointer is not None:
                base, offset = pointer
                args.append(base if not offset else f"({base} + {offset})")
            elif self._is_scalar_ptr(oid):
                args.append(self._lookup(oid))
            else:
                args.append(self._lookup(oid))
        args_str = ", ".join(args)

        # Find the callee function definition to determine return types
        return_types = []
        if self.graph.called_funcs:
            for cfunc in self.graph.called_funcs:
                if cfunc.name == callee:
                    return_types = cfunc.return_types
                    break

        if not return_types:
            # Void call
            self.kb.raw_line(f"    {safe_callee}({args_str});")
        elif len(return_types) == 1:
            # Single return value
            msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(return_types[0]))
            var = self._next_var("r")
            self.kb.raw_line(f"    {msl_ty} {var} = {safe_callee}({args_str});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = _mlir_to_triton_dtype(return_types[0])
        else:
            # Multiple return values — call returns a struct
            ret_struct = f"_ret_{safe_callee}"
            var = self._next_var("rv")
            self.kb.raw_line(f"    {ret_struct} {var} = {safe_callee}({args_str});")

            # Map each result ID to its struct field
            if ssa.result_ids:
                for i, rid in enumerate(ssa.result_ids):
                    field_var = f"{var}.v{i}"
                    self.env[rid] = field_var
                    if i < len(return_types):
                        self.env_types[rid] = _mlir_to_triton_dtype(return_types[i])
            else:
                # Single result ID (shouldn't happen for multi-return, but be safe)
                self.env[ssa.id] = f"{var}.v0"
                self.env_types[ssa.id] = _mlir_to_triton_dtype(return_types[0])

    # -- TTG ops (TritonGPU dialect) --

    def _lower_ttg(self, ssa: SSAValue):
        """Lower ttg.* ops (TritonGPU dialect).

        Most ttg ops are layout annotations or shared memory management.
        convert_layout requires shared memory redistribution when the
        source and destination layouts map elements to different threads.
        """
        op = ssa.op
        if op == "ttg.convert_layout":
            self._lower_convert_layout(ssa)
        elif op == "ttg.local_alloc":
            self._lower_local_alloc(ssa)
        elif op == "ttg.local_load":
            self._lower_local_load(ssa)
        elif op == "ttg.local_store":
            # Store to shared memory — passthrough
            self._emit_passthrough(ssa)
        elif op == "ttg.memdesc_trans":
            self._lower_memdesc_trans(ssa)
        elif op == "ttg.barrier":
            # Triton's renamed cross-threadgroup barrier (was tt.debug_barrier).
            # Without this it fell to the passthrough below and was SILENTLY
            # dropped -> racy multi-SIMD-group kernels (downstream tridec bug 1).
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);")
        elif ssa.id is not None and ssa.id < 0:
            # Unknown NO-RESULT ttg op = a dropped side effect / synchronization
            # (async copy, barrier variant) the passthrough would lose silently.
            # Refuse loudly, consistent with the #165 generic default-deny.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing to emit silently-wrong output: unsupported "
                f"side-effecting op '{op}' (no result) would be dropped, losing "
                f"its effect. Add a handler or explicit refusal."
            )
        else:
            # Other (value-producing) ttg ops: passthrough. An undefined result
            # refuses at lookup if consumed.
            self._emit_passthrough(ssa)

    def _lower_convert_layout_mept_shuffle(self, ssa, src_type, dest_type, src_shape, N) -> bool:
        """Phase 4d: redistribute a per-thread register array via threadgroup
        memory according to source/destination LinearLayouts.

        Returns True if the shuffle was emitted, False if a layout couldn't be
        resolved (caller falls back). The element identity is preserved: each
        thread writes its source-register elements to the shared buffer at
        ``src_layout.position(reg, lane, warp)``, a barrier orders the
        exchange, then each thread reads its destination-register elements
        from ``dst_layout.position(reg, lane, warp)``. ``msl_position_expr``
        emits the XOR-basis index for each register.
        """
        src_id = ssa.operand_ids[0]
        src_arr, n_src, arr_ty = self.env_array[src_id]
        src_ll = self.env_layout.get(src_id)
        if src_ll is None:
            src_ll = self._resolve_linear_layout(src_type, src_shape)
        dst_ll = self._resolve_linear_layout(dest_type, src_shape)
        if src_ll is None or dst_ll is None:
            return False
        # Sanity: the source array length must match the source layout's
        # register count, and the layouts must describe the same tensor.
        if src_ll.num_registers_per_thread != n_src or src_ll.total_elements != dst_ll.total_elements:
            return False
        n_dst = dst_ll.num_registers_per_thread

        # Shared buffer holds the whole tile in compute form (float for fp/bf, int for
        # integer). 64-bit ints must stage at full width — the register array (arr_ty) is
        # already long/ulong, so an i32 buffer would truncate src_arr[i] on the store
        # (Triton-lens re-audit 2026-06-25: latent twin of the convert_layout i64 defect).
        src_dtype = self.env_types.get(src_id, "fp32")
        is_int = not (src_dtype.startswith("fp") or src_dtype.startswith("bf"))
        if not is_int:
            shared_dtype = "fp32"
        elif src_dtype in ("i64", "u64", "ui64"):
            shared_dtype = "i64" if src_dtype == "i64" else "u64"
        else:
            shared_dtype = "i32"
        shared_name = f"shuf_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=N)

        lane = "(lid & 31u)"
        warp = "(lid >> 5u)"
        # Barrier before writing in case the buffer aliases an earlier shuffle.
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        for i in range(n_src):
            pos = src_ll.msl_position_expr(f"{i}u", lane, warp)
            self.kb.raw_line(f"    {shared_name}[{pos}] = {src_arr}[{i}];")
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        read_exprs = [f"{shared_name}[{dst_ll.msl_position_expr(f'{j}u', lane, warp)}]" for j in range(n_dst)]
        out_var = self._var_array("shuf", read_exprs, arr_ty)
        self.env[ssa.id] = out_var
        self.env_array[ssa.id] = (out_var, n_dst, arr_ty)
        self.env_types[ssa.id] = src_dtype
        self._propagate_shape_from_type(ssa)
        return True

    def _lower_convert_layout(self, ssa: SSAValue):
        """ttg.convert_layout → shared memory redistribution.

        When the source layout's thread-to-element mapping differs from the
        destination layout, elements must be redistributed via shared memory:
          1. Each thread writes its value to shared[source_index]
          2. threadgroup_barrier
          3. Each thread reads shared[dest_index]

        For 1D tensors in our model: after a 2D reduce, the broadcast puts
        results on threads using lid/N or lid%M, but the destination layout
        expects a simple lid-based mapping. The shared memory shuffle fixes
        this mismatch.

        For cases where both layouts use the same mapping (e.g. same blocked
        layout), this is a passthrough.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            return

        src_var = self._lookup(ssa.operand_ids[0])
        src_shape = _extract_shape(ssa.type_str)
        if not src_shape:
            # Can't determine shape — passthrough
            self._emit_passthrough(ssa)
            return

        N = 1
        for d in src_shape:
            N *= d

        # Only do shared memory redistribution when the source layout is
        # a #ttg.slice (from a reduce). For #blocked → #blocked conversions,
        # our 1D model doesn't distinguish between blocked layouts, so
        # passthrough is correct.
        src_type = ""
        for op in self.graph.ops:
            if op.id == ssa.operand_ids[0] and op.type_str:
                src_type = op.type_str
                break
        if not src_type:
            src_type = self._find_op_type_str(ssa.operand_ids[0]) or ""

        # Only redistribute when converting FROM a slice layout TO a
        # non-slice layout (e.g., #ttg.slice → #blocked2). This indicates
        # the reduce result needs remapping to a new thread assignment.
        # When both source and dest are slice layouts, it's a layout
        # variant change that doesn't affect thread mapping in our model.
        dest_type = ssa.type_str or ""
        needs_redistribute = "ttg.slice" in src_type and "ttg.slice" not in dest_type

        # Packet 325: wide scans deliberately replace every participating
        # blocked layout with one canonical contiguous flat ownership map.
        # A full-tile convert inside that proved cone is therefore a logical
        # type annotation, not a physical redistribution: pre-scan values,
        # scan outputs, and independently rebuilt pointers all use
        # e=lid*width+slot.
        # Admit only an exact full-shape/width match; anything else continues
        # through the ordinary layout resolver (and fails closed if unresolved).
        if (
            getattr(self, "_scan_mept_shape", None) is not None
            and ssa.operand_ids[0] in self.env_array
            and tuple(src_shape) == tuple(self._scan_mept_shape)
            and self.env_array[ssa.operand_ids[0]][1] == self._scan_mept_width
        ):
            self._emit_passthrough(ssa)
            self._propagate_shape_from_type(ssa)
            return

        # Phase 4d: MEPT array shuffle. When the source value is a per-thread
        # register array and both layouts resolve to LinearLayouts, redistribute
        # the elements through threadgroup memory using the XOR-basis position
        # math. Each thread writes its source-register elements at their
        # src-layout positions, barriers, then reads its dest-register elements
        # from their dst-layout positions. This is the general layout-change
        # primitive for MEPT (and the eventual replacement for the targeted
        # transpose/interleave pattern detectors — Phase 4g).
        if getattr(self, "mept_enabled", False) and ssa.operand_ids[0] in self.env_array:
            shuffled = self._lower_convert_layout_mept_shuffle(ssa, src_type, dest_type, src_shape, N)
            if shuffled:
                return
            # If the shuffle couldn't resolve a layout, fall through. The
            # prescan eligibility check (_mept_convert_safe) guarantees this
            # path only carries an env_array when both layouts resolve, so a
            # fall-through here means a non-MEPT array (shouldn't happen) — the
            # passthrough below preserves the old behavior.

        # Multi-element-per-thread #linear source layouts are handled today
        # only by pattern detectors (e.g. ``_detect_transpose_via_reshape``)
        # that bypass this path entirely. Reaching here with an unhandled
        # ``#linear`` source means the kernel will produce wrong values.
        # ``#ttg.blocked`` sources still flow through the legacy text-based
        # fallback in ``emit_msl`` and produce correct output for current
        # kernels, so we only block ``#linear`` here. The architectural fix
        # for both layouts is tracked as the multi-element-per-thread
        # refactor (Phase 4 — see docs/superpowers/plans/
        # 2026-05-21-multi-element-per-thread.md).
        mod_text = getattr(self.graph, "mod_text", "") or ""
        alias_match = re.search(r",\s*#(\w+)\s*>\s*$", src_type)
        if alias_match and mod_text:
            from triton_msl.codegen._linear_layout import parse_linear_layout

            alias = alias_match.group(1)
            if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.linear<", mod_text):
                ll = parse_linear_layout(mod_text, alias)
                if ll and ll.num_registers_per_thread > 1:
                    # Corroborate against env_n_elems tracking (Phase 4a
                    # infrastructure). When they disagree, prefer the
                    # parser's reading — env_n_elems is best-effort.
                    raise MetalNotImplementedError(
                        "ttg.convert_layout from a multi-element-per-thread "
                        "#linear source layout is not yet implemented. The "
                        "kernel hit the generic convert_layout path with "
                        f"{ll.num_registers_per_thread} elements per thread; "
                        "add a pattern detector (see "
                        "_detect_transpose_via_reshape) or implement the "
                        "general per-register shuffle (Phase 4)."
                    )

        if not needs_redistribute or not self._is_2d or N <= 1:
            self._emit_passthrough(ssa)
            return

        # Determine source element type. 64-bit ints must re-stage at full width — an i32
        # buffer silently truncates the high word (Triton-lens re-audit 2026-06-25: a 2-D
        # int64 reduce accumulates correctly in long but convert_layout re-staged the
        # result through i32 and truncated it).
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (src_dtype.startswith("fp") or src_dtype.startswith("bf"))
        if not is_int:
            msl_type, shared_dtype = "float", "fp32"
        elif src_dtype in ("i64", "u64", "ui64"):
            msl_type = "long" if src_dtype == "i64" else "ulong"
            shared_dtype = "i64" if src_dtype == "i64" else "u64"
        else:
            msl_type, shared_dtype = "int", "i32"

        # Allocate shared memory for the redistribution
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=N)

        # Determine the source write index.
        #
        # After a 2-D reduce with broadcast, the thread-to-element mapping uses
        # lid / N_reduced (axis=1, blocked rows) or lid % N (axis=0), where N_reduced is
        # the inner dim of the REDUCE INPUT (not the global 2-D shape). Two mechanisms
        # decide, and both must agree with the emitted index:
        #
        #   * the declared-layout resolver (_value_1d_layout_of), which since 2026-08-29
        #     crosses scf.for carries and elementwise ops — the loop-carried lse value
        #     (`mi + log(li)` after the K-loop) resolves to "blocked" through the carry;
        #   * a carry-aware reduce search below, which supplies the DIVISOR (the layout
        #     registry records the kind, not the reduce-input inner dim).
        #
        # The old operand-0-only trace could not cross an scf.for (it walked into the
        # loop bound), so a carried blocked value fell into the "standard modular"
        # branch: threads 0..N-1 wrote shared[lid] — but in blocked layout those threads
        # ALL hold row 0, so every output row silently became row 0 (trifast #6b's lse
        # collapse, GPU-measured err 0.73). Unresolvable-but-reduce-derived now REFUSES
        # instead of guessing.
        src_id = ssa.operand_ids[0]
        _layout = self._value_1d_layout_of(src_id)

        # Carry-aware BFS over (operands + scf.for carry edges) for the producing
        # 2-D reduce(s). Consistency required: conflicting (axis, inner-dim) pairs
        # mean the value mixes reduces of different layouts -> unprovable.
        by_id = self._layout_op_index()
        _ba_edges = {}
        for _o in by_id.values():
            if _o.op != "scf.for":
                continue
            _bas = (_o.attrs.get("block_arg_ids", []) if _o.attrs else [])[1:]
            _inits = (_o.operand_ids or [])[3:]
            _yield = next((b for b in _o.region_ops or [] if b.op == "scf.yield"), None)
            for _i, _ba in enumerate(_bas):
                _dsts = []
                if _i < len(_inits):
                    _dsts.append(_inits[_i])
                if _yield is not None and _i < len(_yield.operand_ids or []):
                    _dsts.append(_yield.operand_ids[_i])
                _ba_edges[_ba] = _dsts
            # Single-result scf.for: result_ids is None and the result id IS _o.id
            # (see the resolver's res_of for the same trap).
            _rids = _o.result_ids or ([_o.id] if len(_inits) > 0 else [])
            for _i, _rid in enumerate(_rids):
                if _yield is not None and _i < len(_yield.operand_ids or []):
                    _ba_edges[_rid] = [_yield.operand_ids[_i]]

        _found = set()  # {(axis, inner_dim)}
        _seen = set()
        _queue = [src_id]
        while _queue and len(_seen) < 256:
            _cur = _queue.pop()
            if _cur in _seen:
                continue
            _seen.add(_cur)
            if _cur in _ba_edges:
                _queue.extend(_ba_edges[_cur])
                continue
            _op = by_id.get(_cur)
            if _op is None:
                continue
            if _op.op == "tt.reduce" and _op.operand_ids:
                _ish = self.env_shapes.get(_op.operand_ids[0])
                if not _ish:
                    _it = self._find_op_type_str(_op.operand_ids[0])
                    _ish = _extract_shape(_it) if _it else None
                if _ish and len(_ish) >= 2:
                    _ax = _op.attrs.get("axis", 0)
                    _found.add((_ax, _ish[1] if _ax == 1 else _ish[0]))
                continue  # do not walk past the reduce
            if _op.op in ("arith.constant", "tt.splat", "tt.make_range"):
                continue
            _queue.extend(_op.operand_ids or [])

        reduce_axis, N_reduced = next(iter(_found)) if len(_found) == 1 else (None, None)

        def _stage_modular():
            self.kb.raw_line(f"    if (lid < {N}u)")
            self.kb.raw_line(f"        {shared_name}[lid % {N}u] = {src_var};")

        def _stage_blocked(div):
            # thread lid holds row lid/div: one writer per row.
            self.kb.raw_line(f"    if (lid % {div}u == 0u && lid / {div}u < {N}u)")
            self.kb.raw_line(f"        {shared_name}[lid / {div}u] = {src_var};")

        def _refuse(why):
            raise MetalNonRecoverableError(
                f"ttg.convert_layout cannot prove its source's thread-to-element "
                f"mapping ({why}); staging it with a guessed mapping is the "
                f"row-collapse silent-wrong (trifast #6b). Refusing "
                f"(correct-or-refuse).",
                op_name="ttg.convert_layout",
            )

        if _layout in ("direct", "any"):
            # Resolver is decisive: thread i holds element i (or the value is
            # uniform). Modular staging is exact regardless of provenance.
            _stage_modular()
        elif _layout == "blocked":
            # Resolver proves blocked rows; the reduce search must supply ONE
            # consistent axis=1 divisor or there is no valid mapping.
            if reduce_axis == 1 and N_reduced and N_reduced > 1:
                _stage_blocked(N_reduced)
            else:
                _refuse(f"declared blocked, but reduce provenance is {sorted(_found)}")
        elif _found:
            # The provenance search can identify reduce DIVISORS, but it cannot prove
            # that every other elementwise operand shares that reduce's layout.  In
            # particular, DIRECT tt.split + BLOCKED axis=1 reduce has one perfectly
            # consistent reduce provenance while the resolver correctly returns None
            # for the mixed mapping.  Letting provenance override that conflict stages
            # the whole expression as BLOCKED and silently uses split row 0 for every
            # output row.  A declared resolver result is therefore mandatory whenever
            # reduce provenance exists; unresolved/conflicting means refuse.
            _refuse(f"source layout is unresolved despite reduce provenance {sorted(_found)}")
        else:
            # No reduce anywhere in the provenance and nothing declared: the
            # historical modular mapping for genuinely simple slice conversions.
            _stage_modular()

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read back in destination layout: thread i gets element i
        result_var = self._next_var("cvt")
        self.kb.raw_line(f"    {msl_type} {result_var} = (lid < {N}u) ? {shared_name}[lid] : ({msl_type})0;")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = src_dtype
        if src_shape:
            self.env_shapes[ssa.id] = src_shape
        # Mark this value as having been through convert_layout — the
        # thread-to-element mapping is now simple (thread i = element i).
        # This prevents the store from using 2D-aware guards.
        if not hasattr(self, "_converted_layout_ids"):
            self._converted_layout_ids = set()
        self._converted_layout_ids.add(ssa.id)
        # Read back as shared[lid], i.e. thread i holds element i -- the DIRECT layout.
        # Declared here too so the store can consult one registry instead of also
        # consulting the convert_layout side-channel.
        self._register_1d_layout(ssa.id, "direct")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower_ir_graph(graph: IRGraph, options=None) -> str:
    """Lower an IRGraph to MSL source code.

    Args:
        graph: The IRGraph from mlir_walker.walk_ttgir().
        options: MetalOptions instance.

    Returns:
        MSL source code string.
    """
    lowerer = GenericLowerer(graph, options)
    return lowerer.lower()
