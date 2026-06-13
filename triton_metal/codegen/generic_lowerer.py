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

import os
import re
from typing import Any, Dict, List, Optional

from triton_metal.codegen.mlir_walker import IRGraph, SSAValue, FuncArg, CalledFunc, _extract_shape
from triton_metal.codegen.msl_emitter import KernelBuilder, _msl_compute_type, _sanitize_msl_name
from triton_metal.codegen.msl_types import triton_type_to_msl
from triton_metal.errors import (
    MetalCodegenError, MetalNotImplementedError, MetalNonRecoverableError,
)


# Free helpers extracted to keep this file navigable; see
# _lowerer_helpers.py and _device_func_lowerer.py for the split-out code.
from triton_metal.codegen._lowerer_helpers import (
    CMPI_PREDICATES,
    CMPF_PREDICATES,
    CMPI_NAMED,
    CMPF_NAMED,
    _UINT_TYPE_MAP,
    _mlir_to_triton_dtype,
    _msl_int_type,
    _shape_numel,
    _extract_layout_signature,
    _alias_shared_memory,
)
from triton_metal.codegen._device_func_lowerer import _DeviceFuncLowerer
from triton_metal.codegen._lowerer_templates import _TemplateMixin
from triton_metal.codegen._lowerer_detection import _DetectionMixin
from triton_metal.codegen._lowerer_emission import _EmissionMixin
from triton_metal.codegen._lowerer_reduce import _ReduceScanMixin
from triton_metal.codegen._lowerer_control import _ControlFlowMixin


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
# Deliberately EXCLUDED (not array-wired): arith.cmpf, arith.select,
# arith.bitcast, math.erf/log1p/expm1, every shape op (tt.trans/join/cat/
# split/gather/broadcast/expand_dims/reshape), atomics, reduce/scan/
# histogram, tt.dot, control flow (scf.*), ttg.*. FP8 kernels are excluded
# separately (fp8 truncf emits a scalar conversion chain).
_MEPT_SAFE_OPS = frozenset({
    # producers / memory (wired array path)
    "tt.make_range", "tt.addptr", "tt.load", "tt.store",
    # scalar / broadcast operands (never produce a register array)
    "tt.get_program_id", "tt.get_num_programs", "arith.constant",
    "tt.splat", "tt.return",
    # elementwise binary -> _emit_binary / _emit_builtin_binary /
    # _emit_nan_propagating_minmax (all route through _mept_binary_dispatch)
    "arith.addf", "arith.addi", "arith.subf", "arith.subi",
    "arith.mulf", "arith.muli", "arith.divf", "arith.divsi", "arith.divui",
    "arith.remsi", "arith.remui", "arith.remf",
    "arith.maxf", "arith.maxsi", "arith.maxui",
    "arith.minf", "arith.minsi", "arith.minui",
    "arith.maxnumf", "arith.minnumf", "arith.maximumf", "arith.minimumf",
    "arith.andi", "arith.ori", "arith.xori",
    "arith.shli", "arith.shrsi", "arith.shrui",
    # comparison (only cmpi is array-wired; cmpf is NOT)
    "arith.cmpi",
    # unary
    "arith.negf",
    # casts (passthrough or _emit_cast / _emit_int_cast / _emit_uitofp)
    "arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
    "arith.fptosi", "arith.fptoui", "arith.extsi", "arith.extui",
    "arith.trunci", "arith.index_cast", "arith.index_castui",
    # math (array-wired in _lower_math)
    "math.exp", "math.exp2", "math.log", "math.log2", "math.sqrt",
    "math.rsqrt", "math.abs", "math.absf", "math.absi", "math.sin",
    "math.cos", "math.tanh", "math.floor", "math.ceil", "math.round",
    "math.fma", "math.powf", "math.copysign", "math.atan2",
    "math.roundeven", "math.trunc",
})


# ---------------------------------------------------------------------------
# Generic Lowerer
# ---------------------------------------------------------------------------

class GenericLowerer(_ControlFlowMixin, _ReduceScanMixin, _EmissionMixin, _DetectionMixin, _TemplateMixin):
    """Lower an IRGraph to MSL source code via KernelBuilder."""

    def __init__(self, graph: IRGraph, options=None):
        self.graph = graph
        self.options = options
        self.env = {}           # ssa_id -> MSL variable name
        self.env_types = {}     # ssa_id -> triton dtype string
        self.env_is_mask = {}   # ssa_id -> True if this is a bool mask
        self.env_is_ptr = {}    # ssa_id -> (base_ptr_name, offsets_var)
        self.env_shapes = {}    # ssa_id -> shape tuple, e.g., (32, 64)
        # Phase 4 foundation: track how many tensor elements per thread each
        # SSA value carries. The current lowerer emits 1 scalar per thread for
        # every tensor, so the default is 1; ``_track_n_elems`` reads the
        # TTGIR layout attached to each op\'s ``type_str`` and stores the
        # computed value here. Op handlers can opt into the multi-element
        # path later; today this is purely diagnostic and informs the
        # ``_lower_convert_layout`` guard about when a real shuffle is
        # required. See ``docs/superpowers/plans/2026-05-21-multi-element-
        # per-thread.md`` for the staged refactor plan.
        self.env_n_elems = {}   # ssa_id -> int (elements per thread, ≥ 1)
        # Phase 4c: alongside env_n_elems, store the resolved
        # ``LinearLayout`` when one is available. Consulted by MEPT
        # producers (e.g. ``_lower_make_range``) to emit the correct
        # per-register position math via ``msl_position_expr``. When
        # absent (the synthetic / contiguous case), producers fall back
        # to the simple ``lid*N + i`` formula.
        self.env_layout = {}    # ssa_id -> LinearLayout
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

        # ── EXPERIMENTAL: multi-element-per-thread (MEPT) ──────────────────
        # Charter (read before changing this): MEPT is an OPT-IN
        # (TRITON_METAL_MEPT=1), OFF-BY-DEFAULT experimental code path that
        # lets a thread hold N tensor elements as a register array instead of
        # one scalar. It is the prototype of the register-array programming
        # model that is the long-term path to retiring the pattern detectors
        # (see docs/ARCHITECTURE.md). It is NOT a performance feature:
        # benchmarked perf-neutral on elementwise/reduce kernels (deltas
        # within launch-overhead noise). It is kept because:
        #   1. it is correct — the full upstream test_core suite passes with
        #      the flag ON as well as OFF (4327/0 both ways), and
        #   2. it is the foundation for the generic convert_layout / dot
        #      lowering that would subsume the matmul + transpose detectors.
        # Default behavior (flag off) is byte-identical to not having MEPT:
        # every consumer is gated on ``mept_enabled`` and the producer
        # (make_range) is the single activation root. Do not enable by
        # default until it shows a measured win on a real workload.
        # Plan: docs/superpowers/plans/2026-05-21-multi-element-per-thread.md
        self.mept_enabled = os.environ.get("TRITON_METAL_MEPT", "0") == "1"
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
        from triton_metal.codegen.regval import RegVal
        if regval.form == "array" and regval.n_elems > 1:
            arr = self._var_array(
                base, [body(e) for e in range(regval.n_elems)], regval.ty)
            return RegVal(name=arr, n_elems=regval.n_elems, ty=regval.ty,
                          form="array")
        name = self._next_var(base)
        self.kb.raw_line(f"    {regval.ty} {name} = {body(0)};")
        return RegVal(name=name, n_elems=1, ty=regval.ty, form=regval.form)

    # -- Shape tracking helpers --------------------------------------------------

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

    def _is_scalar(self, ssa_id: int) -> bool:
        """Check if an SSA value has scalar shape (no dimensions).

        A value is scalar if its shape is () — i.e., it has no tensor
        dimensions.  Scalars don't need per-thread indexing; they are
        the same value on every thread.
        """
        return self._get_shape(ssa_id) == ()

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
                from triton_metal.codegen._linear_layout import parse_linear_layout
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
                from triton_metal.codegen._linear_layout import blocked_to_linear
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
                from triton_metal.codegen._linear_layout import (
                    parse_linear_layout,
                )
                return parse_linear_layout(mod_text, alias)
            if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.blocked<",
                         mod_text):
                from triton_metal.codegen._linear_layout import (
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
            mod_text, re.DOTALL,
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
    # predicate) lives in ``triton_metal.codegen.refusal_catalog`` — the single
    # source of truth shared with the C++ path and the generated docs. Keep it
    # tight: only ops that are both unsupported and *unsafe to approximate*.

    def _refuse_unsafe_unsupported_ops(self):
        """Raise MetalNonRecoverableError for op/context combinations that can
        only be lowered to silently-wrong output.

        The cases are defined once in
        ``triton_metal.codegen.refusal_catalog`` (the single source of truth
        shared with the C++ path and the generated docs); this method just
        builds the :class:`RefusalContext` and walks the catalog. Each case
        was a confirmed silent-wrong producer found by classifying skip-listed
        tests; refusing keeps the integrity guarantee (never return numbers we
        can't vouch for). Guards are scoped so the supported cases (test_cat
        1-D, test_join, test_noinline non-shared, top-level tt.dot,
        value-returning early returns) are untouched.
        """
        from triton_metal.errors import MetalNonRecoverableError
        from triton_metal.codegen import refusal_catalog as _rc

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
        )
        violation = _rc.check_all(ctx)
        if violation is not None:
            raise MetalNonRecoverableError(
                violation.message, op_name=violation.op_name)

    def lower(self) -> str:
        """Lower the IRGraph to MSL source code."""
        # Integrity prescan (PR1): some ops have no correct lowering on this
        # backend AND the legacy fallback can't help — emitting anything for
        # them yields silently-wrong output (a kernel that runs and returns
        # wrong numbers). Refuse rather than approximate. ``tt.dot_scaled``
        # is microscaling (mxfp) matmul: Apple GPUs have no microscaling
        # units and our codegen has no handler, so the result tensor is
        # never computed -> garbage (test_scaled_dot, ~all configs mismatch).
        self._refuse_unsafe_unsupported_ops()

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

        # Check for simple dot (no stride args, no scf.for) — use inline
        # scalar matmul that loads from global into shared memory, then
        # does per-thread dot product.
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
            self.effective_block_size = (
                (self.options.num_warps if self.options else 4) * 32)
            self._prescan_stores()
            return msl

        # Check for row-wise layer norm (sum + sum_sq + sub + rsqrt). Same
        # TG-cache + float4 vectorization shape as softmax. Must be tried
        # AFTER softmax because the patterns share the "two reduces + sub"
        # signature (softmax: max+sum+exp; layer_norm: sum+sum+rsqrt).
        layer_norm_info = self._detect_layer_norm()
        if layer_norm_info:
            msl = self._lower_layer_norm_template(layer_norm_info)
            self.effective_block_size = (
                (self.options.num_warps if self.options else 4) * 32)
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
            self.effective_block_size = (
                (self.options.num_warps if self.options else 4) * 32)
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
            self.effective_block_size = (
                (self.options.num_warps if self.options else 4) * 32)
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
        # Scan tensor type_strs to find the actual max tensor size.
        max_tensor_size = block_size
        for ssa in self.graph.ops:
            shape = _extract_shape(ssa.type_str)
            if shape:
                total = 1
                for d in shape:
                    total *= d
                if total > max_tensor_size:
                    max_tensor_size = total
        if max_tensor_size > block_size and max_tensor_size <= 1024:
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
        has_reduce_ops = any(
            ssa.op == "tt.reduce" for ssa in all_ops_iter
        )
        has_barrier_ops = any(
            ssa.op in ("tt.reduce", "tt.scan", "tt.debug_barrier", "ttg.barrier",
                       "tt.trans", "tt.dot", "ttg.local_alloc")
            for ssa in all_ops_iter
        )
        # Multi-value reduces (argmin/argmax) need per-element indices which
        # are incompatible with the multi-pass accumulation loop (the loop
        # variable goes out of scope before the reduce handler runs).
        has_multivalue_reduce = any(
            ssa.op == "tt.reduce" and ssa.result_ids and len(ssa.result_ids) >= 2
            for ssa in all_ops_iter
        )
        num_threads = self.graph.num_warps * 32

        # Phase 4c: is this kernel safe for single-pass MEPT? Default-deny —
        # every op must be array-wired and no fp8 dtype may appear (fp8
        # truncf emits a scalar conversion chain). Computed here; combined
        # with the exact-cover + no-barrier checks at the wrapping decision.
        from triton_metal.codegen.msl_builtins import is_fp8_type as _is_fp8

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
            src_sh = _extract_shape(src_t)
            dst_sh = _extract_shape(s.type_str or "")
            if not src_sh or not dst_sh:
                return False
            sll = self._resolve_linear_layout(src_t, src_sh)
            dll = self._resolve_linear_layout(s.type_str or "", dst_sh)
            return (sll is not None and dll is not None
                    and sll.total_elements == dll.total_elements)

        def _op_mept_ok(s):
            if s.op in _MEPT_SAFE_OPS:
                return True
            if s.op == "ttg.convert_layout":
                return _convert_resolves(s)
            return False

        mept_kernel_safe = self.mept_enabled and all(
            _op_mept_ok(s) for s in all_ops_iter
        ) and not any(_op_is_fp8(s) for s in all_ops_iter)

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
            t = self._find_op_type_str(r.operand_ids[0])
            shp = _extract_shape(t) if t else None
            return shp is not None and len(shp) == 1

        mept_reduce_eligible = (
            self.mept_enabled
            and any(s.op == "tt.reduce" for s in _top_ops)
            and all(_op_mept_ok(s) or s.op == "tt.reduce"
                    for s in _top_ops)
            and all(_reduce_is_1d_full(s) for s in _top_ops
                    if s.op == "tt.reduce")
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )

        # Phase 4f (MEPT M2): control-flow kernels that carry a multi-element
        # value across a data-dependent scf.for/while/if. The re-execution
        # wrap-loop cannot carry per-element state across the control-flow
        # boundary, so values hoisted before the loop (a tl.arange register
        # array, a masked-load `other=` constant) fall back to UNKNOWN_ inside
        # it and the integrity backstop refuses (tridec Bug 2, BLOCK>=256).
        # Register arrays declared once before the loop persist into the body
        # naturally (env_array is instance state), so the existing array-wired
        # body handlers resolve them. Eligible iff: MEPT on; a control-flow op
        # is present; the region references/carries a multi-element value;
        # every op is array-wired (or control-flow / reduce / yield /
        # condition); every reduce is a 1-D full reduce; the tile cover is
        # exact; no fp8. See
        # docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md
        from triton_metal.codegen.regval import (
            region_needs_arrays as _region_needs_arrays,
            tensor_value_ids as _tensor_value_ids,
            _CONTROL_OPS as _CF_OPS,
        )

        def _arrayform_op_ok(s):
            if s.op in _CF_OPS:
                body = list(s.region_ops or []) + list(getattr(s, "else_ops", None) or [])
                return all(_arrayform_op_ok(b) for b in body)
            if s.op in ("scf.yield", "scf.condition", "tt.reduce"):
                return True
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
            shp = _extract_shape(getattr(s, "type_str", "") or "")
            if not shp:
                return False
            tot = 1
            for d in shp:
                tot *= d
            return tot > num_threads

        mept_arrayform_eligible = (
            self.mept_enabled
            and any(s.op in _CF_OPS for s in _top_ops)
            and _region_needs_arrays(
                _top_ops, _tensor_value_ids(_top_ops, _value_is_multi))
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
                    # Extract shape from the reduce input's type_str in the IR
                    inp_type = self._find_op_type_str(ssa.operand_ids[0])
                    inp_shape = _extract_shape(inp_type) if inp_type else None
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
        if has_2d_axis_reduce and block_size <= 1024:
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
                        inp_type = self._find_op_type_str(op.operand_ids[0])
                        inp_shape = _extract_shape(inp_type) if inp_type else None
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
            if (mept_arrayform_eligible
                    and num_threads * size_per_thread == block_size):
                # MEPT M2 single-pass register-array form for control-flow
                # kernels. Each thread owns size_per_thread contiguous
                # elements as a register array (idx[i] = lid*N + i). The array
                # IS the per-thread multiplicity, so there is NO wrap-loop
                # (_needs_wrapping stays False). Arrays declared before a
                # data-dependent scf.for persist into its body, so hoisted
                # values (arange, masked-load `other`) resolve inside the loop.
                self._total_elements = block_size
                block_size = num_threads
                self._mept_single_pass = True
            elif (mept_reduce_eligible
                    and num_threads * size_per_thread == block_size):
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
                use_multipass = True
                self._total_elements = block_size
                block_size = num_threads
            elif (mept_kernel_safe and not has_barrier_ops
                  and num_threads * size_per_thread == block_size):
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
            if has_reduce_ops:
                use_multipass = True
                self._total_elements = block_size
                block_size = 1024
            else:
                self._needs_wrapping = True
                self._total_elements = block_size
                block_size = 1024  # Cap dispatch to Metal max

        self.effective_block_size = block_size

        # If the kernel is too large for the generic lowerer (cooperative ops
        # with > 1024 total elements), emit a minimal kernel with UNSUPPORTED
        # so the legacy parser can handle it via prebuilt templates.
        if getattr(self, '_flash_too_large', False):
            self.kb = KernelBuilder(self.graph.func_name, block_size=block_size)
            self._register_args()
            self.kb.comment("UNSUPPORTED: 2D kernel with cooperative ops exceeds 1024 elements")
            return self.kb.build()

        self.kb = KernelBuilder(self.graph.func_name, block_size=block_size)

        # Generate device functions for noinline callees (must appear before kernel)
        if self.graph.called_funcs:
            self._lower_called_funcs()

        # Pre-scan stores to identify output pointers
        self._prescan_stores()

        # Register function arguments
        self._register_args()

        if use_multipass:
            # Multi-pass reduction: split kernel into phases separated by
            # reductions, wrap each phase in a per-element loop, emit
            # reductions between loops operating on thread-local accumulators.
            self._lower_multipass_reduction(block_size)
        else:
            # Standard path: single wrapping loop or no loop
            if self._needs_wrapping:
                self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {self._total_elements}u; _loop_e += {block_size}u) {{")

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
        msl = _alias_shared_memory(msl)

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
        graph_has_store = any(
            op.op in ("tt.store", "tt.atomic_rmw", "tt.atomic_cas")
            for op in self.graph.ops
        )
        body_lines = getattr(self.kb, "_body_lines", [])
        body_has_stmt = any(";" in ln for ln in body_lines)
        if graph_has_store and not body_has_stmt:
            self.kb.comment(
                "UNSUPPORTED: generic lowering produced an empty body for a "
                "kernel that stores (e.g. an unhandled N-D permute/reduce) — "
                "falling back to the legacy parser.")
            msl = self.kb.build()
            msl = _alias_shared_memory(msl)
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

        # Pure matmul: dot without reductions or conditional masking.
        # Complex kernels (flash attention, fused matmul+softmax) have
        # reductions/masking alongside dot and must go through generic lowerer.
        if has_reduce or has_where:
            return False

        return True

    def _resolve_constant_int(self, ssa_id):
        """Resolve an SSA ID to its integer constant value, or None."""
        for ssa in self.graph.ops:
            if ssa.id == ssa_id and ssa.op == "arith.constant":
                val = ssa.attrs.get("value")
                if isinstance(val, int):
                    return val
        return None

    def _trace_dot_accumulator(self, acc_id) -> str:
        """Trace the 3rd operand of tt.dot to determine accumulator source.

        Returns: 'zero' (default), 'add-matrix', 'add-rows', 'add-cols'
        """
        # Build a quick lookup
        op_map = {ssa.id: ssa for ssa in self.graph.ops}

        # Follow the chain: convert_layout → load, or convert_layout → broadcast → load
        visited = set()
        current = acc_id
        has_broadcast = False
        expand_dims_shape = None  # Track expand_dims output to distinguish rows vs cols

        while current in op_map and current not in visited:
            visited.add(current)
            op = op_map[current]

            if op.op == "tt.load":
                # Found a load — it's an add epilogue
                if has_broadcast:
                    # Use expand_dims shape to distinguish rows vs cols:
                    # (M, 1) = add-rows ([:, None]), (1, N) = add-cols ([None, :])
                    if expand_dims_shape and len(expand_dims_shape) == 2:
                        if expand_dims_shape[0] == 1:
                            return "add-cols"
                        elif expand_dims_shape[1] == 1:
                            return "add-rows"
                    # Fallback: use load shape vs dot shape
                    load_shape = _extract_shape(op.type_str)
                    dot_shape = _extract_shape(
                        op_map[next(i for i in op_map
                                    if op_map[i].op == "tt.dot")].type_str
                    ) if any(op_map[i].op == "tt.dot" for i in op_map) else []
                    if load_shape and dot_shape and len(dot_shape) >= 2:
                        M_dim, N_dim = dot_shape[0], dot_shape[1]
                        load_size = load_shape[0] if len(load_shape) == 1 else max(load_shape)
                        if M_dim != N_dim:
                            if load_size == M_dim:
                                return "add-rows"
                            elif load_size == N_dim:
                                return "add-cols"
                    return "add-rows"  # default broadcast
                return "add-matrix"

            if op.op == "arith.constant":
                return "zero"

            # Follow through passthrough ops
            if op.op in ("ttg.convert_layout", "tt.broadcast",
                         "tt.expand_dims", "tt.splat",
                         "arith.extf", "arith.truncf",
                         "arith.sitofp", "arith.uitofp"):
                if op.op == "tt.broadcast":
                    has_broadcast = True
                if op.op == "tt.expand_dims":
                    has_broadcast = True
                    expand_dims_shape = _extract_shape(op.type_str)
                if op.operand_ids:
                    current = op.operand_ids[0]
                    continue

            # Unknown op — assume it's derived from a computation (zero)
            break

        return "zero"

    def _has_tensor_ops(self) -> bool:
        """Check if the kernel has any tensor-producing ops (tt.make_range, etc.).

        Scalar-only kernels (no tensor operations) should use block_size=1
        to avoid multiple threads racing on the same scalar memory locations.
        """
        def _check_ops(ops):
            for ssa in ops:
                if ssa.op in ("tt.make_range", "tt.splat", "tt.broadcast"):
                    return True
                if ssa.is_tensor:
                    return True
                if ssa.region_ops and _check_ops(ssa.region_ops):
                    return True
                if ssa.else_ops and _check_ops(ssa.else_ops):
                    return True
            return False
        return _check_ops(self.graph.ops)

    def _prescan_stores(self):
        """Scan ops to find which pointer args are stored to (outputs).

        Recursively scans nested regions (scf.for, scf.while, scf.if bodies)
        to find stores inside loops and conditionals.
        """
        self._prescan_stores_recursive(self.graph.ops)

        # Trace through tt.addptr → tt.splat → func_arg (or direct arg)
        # to identify which func_arg pointers are outputs
        self._output_arg_ids = set()
        arg_ids = {a.id for a in self.graph.args if a.is_ptr}

        # Build lookup: ssa_id -> first operand id (for addptr/splat chains)
        first_operand = {}
        self._build_first_operand_map(self.graph.ops, first_operand)

        for store_ptr_id in self._store_ptr_ids:
            # Walk the chain: store_ptr → addptr → splat → arg (or shorter)
            current = store_ptr_id
            for _ in range(5):  # Max chain depth
                if current in arg_ids:
                    self._output_arg_ids.add(current)
                    break
                next_id = first_operand.get(current)
                if next_id is None:
                    break
                current = next_id

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

    def _build_first_operand_map(self, ops, first_operand):
        """Recursively build first-operand lookup for addptr/splat chains."""
        for ssa in ops:
            if ssa.op in ("tt.addptr", "tt.splat") and ssa.operand_ids:
                first_operand[ssa.id] = ssa.operand_ids[0]
            if ssa.region_ops:
                self._build_first_operand_map(ssa.region_ops, first_operand)
            if ssa.else_ops:
                self._build_first_operand_map(ssa.else_ops, first_operand)

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

        # Find the max 2D shape from any tensor type in the kernel
        max_2d_shape = None
        for ssa in ops:
            shape = _extract_shape(ssa.type_str)
            if len(shape) >= 2:
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
            return _extract_shape(cur.type_str)

        # Find expand_dims ops and trace back to make_range.
        # Also record expand_dims by parent layout to pair dim=0/dim=1 siblings
        # so each make_range can know its tile's inner dimension.
        # Use an instance-level dict to accumulate across recursive prescan calls.
        if not hasattr(self, '_expand_by_parent'):
            self._expand_by_parent = {}
        if not hasattr(self, '_make_range_stride_below'):
            self._make_range_stride_below = {}
        if not hasattr(self, '_make_range_full_shape'):
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
                s_shape = _extract_shape(s_ssa.type_str)
                if s_shape and len(s_shape) == rank:
                    numel = 1
                    for d in s_shape:
                        numel *= d
                    if numel > best_numel:
                        best = s_shape
                        best_numel = numel
            return best

        for ssa in ops:
            if ssa.op != "tt.reshape" or not ssa.operand_ids:
                continue
            src_id = ssa.operand_ids[0]
            mr_id = self._trace_to_make_range(src_id, ops, op_by_id)
            if mr_id is None:
                continue
            if mr_id in self._make_range_dim:
                # Already assigned by an expand_dims chain; do not override.
                continue
            out_shape = _extract_shape(ssa.type_str)
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
            same_rank = _max_shape_with_rank(len(out_shape))
            broadcast_shape = tuple(same_rank) if same_rank else tuple(out_shape)
            self._make_range_full_shape[mr_id] = broadcast_shape
            stride_below = 1
            for s in broadcast_shape[dim + 1:]:
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
                    broadcast_shape = max_2d_shape if (max_2d_shape and
                        len(max_2d_shape) == len(final_shape)) else final_shape
                    self._make_range_full_shape[mr_id] = broadcast_shape
                    stride_below = 1
                    for s in broadcast_shape[dim + 1:]:
                        stride_below *= s
                    self._make_range_stride_below[mr_id] = stride_below

                    # Extract the parent layout from the expand_dims source type
                    # e.g., "tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>"
                    # to pair with siblings from the same tile.
                    src_type = ""
                    if src_id in op_by_id and op_by_id[src_id].type_str:
                        src_type = op_by_id[src_id].type_str
                    elif mr_id in op_by_id and op_by_id[mr_id].type_str:
                        src_type = op_by_id[mr_id].type_str
                    import re as _re
                    # Extract the parent layout identifier. The type string
                    # may use aliases (#blocked, #blocked1) or inline defs
                    # (#ttg.blocked<{...}>).  Use a nested-brace-aware match.
                    parent_key = "default"
                    pidx = src_type.find("parent")
                    if pidx >= 0:
                        # Find the `=` and the start of the layout spec
                        eq_idx = src_type.find("=", pidx)
                        if eq_idx >= 0:
                            rest = src_type[eq_idx + 1:].strip()
                            # Capture everything up to matching `>` or `}`
                            depth = 0
                            end_idx = 0
                            for ci, ch in enumerate(rest):
                                if ch in ('<', '{', '['):
                                    depth += 1
                                elif ch in ('>', '}', ']'):
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
                    expand_by_parent.setdefault(parent_key, []).append(
                        (dim, mr_id, range_size))

        # For each parent layout, pair dim=0 and dim=1 make_ranges to
        # determine the tile inner dim for dim=0 (row) make_ranges.
        if not hasattr(self, '_make_range_inner_N'):
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
        if not hasattr(self, '_make_range_store_only'):
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
                if op.op in ("arith.extsi", "arith.extui", "arith.trunci",
                              "arith.index_cast", "arith.index_castui",
                              "arith.sitofp", "arith.uitofp",
                              "ttg.convert_layout",
                              "tt.load") and op.operand_ids:
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
        return f"UNKNOWN_{ssa_id}"

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
        from triton_metal.codegen.regval import RegVal
        if ssa_id in getattr(self, "env_array", {}):
            name, n, ty = self.env_array[ssa_id]
            return RegVal(name=name, n_elems=n, ty=ty, form="array")
        name = self.env.get(ssa_id, "UNKNOWN_%s" % ssa_id)
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
            self._lower_scf_for(ssa)
        elif op == "scf.if":
            self._lower_scf_if(ssa)
        elif op == "scf.while":
            self._lower_scf_while(ssa)
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
            pass  # Runtime bounds check — skip in MSL
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
            # result-producing unknown op leaves its result undefined; any
            # consumer reads UNKNOWN_<id> and fails loud at MSL compile (or it's
            # dead -> harmless), so a comment is safe and tolerates dead-code ops.
            if ssa.id is not None and ssa.id < 0:
                from triton_metal.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    f"Refusing to emit silently-wrong output: unsupported "
                    f"side-effecting op '{op}' (no result) has no Metal lowering "
                    f"and would be silently dropped, losing its effect. Add a "
                    f"handler or an explicit refusal for it.")
            self.kb.comment(f"UNSUPPORTED: {op}")

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
            var_name = self._next_var("idx")
            lid = self._lid_expr
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
            stride_below_map = getattr(self, '_make_range_stride_below', {})
            full_shape = getattr(self, '_make_range_full_shape', {}).get(ssa.id)
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
                    inner_N_map = getattr(self, '_make_range_inner_N', {})
                    if ssa.id in inner_N_map:
                        N = inner_N_map[ssa.id]
                    else:
                        N = total // range_size if range_size > 0 else 1
                    # Store-only make_ranges whose tile is smaller than the
                    # kernel's element count: scale inner_N so lid/N still
                    # yields the correct M-index (see tl.topk M>1 case).
                    store_only = getattr(self, '_make_range_store_only', set())
                    if (ssa.id in store_only and range_size > 0
                            and total > range_size * N):
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
        n_per_thread = self.env_n_elems.get(ssa.id, 1)
        # The MEPT array path activates ONLY when the prescan proved this
        # kernel is single-pass MEPT-safe (exact tile cover, no barriers,
        # every op array-wired, no fp8). That flag is the sole gate for
        # the whole MEPT chain: if make_range stays scalar, no env_array
        # propagates and every downstream consumer stays scalar too.
        # This is what keeps MEPT from firing inside reduction/multipass
        # kernels (where _needs_wrapping is False but the array form is
        # unsupported) or kernels containing unwired ops (select, atomics,
        # shape ops, ...).
        if (n_per_thread > 1
                and getattr(self, "_mept_single_pass", False)):
            ll = self.env_layout.get(ssa.id)
            if ll is not None and ll.num_registers_per_thread == n_per_thread:
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
                    f"{start}u + {lid} * {n_per_thread}u + {i}u" if start != 0
                    else f"{lid} * {n_per_thread}u + {i}u"
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

        # Only rewrite when the reshape output is in a slice layout.  Other
        # reshapes (e.g. 1D → 9D parent) are canonical.
        if not ssa.type_str or "#ttg.slice<" not in ssa.type_str:
            return False

        # The source must trace to a make_range, and the output shape must
        # have exactly one non-1 axis (so we can interpret it as a per-axis
        # range indicator).
        out_shape = _extract_shape(ssa.type_str)
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
        mr_id = self._trace_to_make_range(ssa.operand_ids[0], ops_list,
                                          op_by_id)
        if mr_id is None:
            return False

        # Primary lookup: match by layout signature.  The reshape's output
        # layout (e.g. ``#ttg.slice<{dim = 5, parent = #blocked}>``) should
        # exactly match a reduce output in the same bitonic stage.
        consumer_layout = None
        bcast_shape = None
        layouts_by_layout = getattr(self, "_bcast_layouts_by_layout", {})
        sig = _extract_layout_signature(ssa.type_str)
        if sig is not None and sig in layouts_by_layout:
            bcast_shape, consumer_layout = layouts_by_layout[sig]

        # Fallback: match by shape compatibility.
        if consumer_layout is None:
            layouts_by_shape = getattr(self, "_bcast_layouts_by_shape", {})
            for shape, layout in layouts_by_shape.items():
                if len(shape) != len(out_shape):
                    continue
                compatible = True
                for i in range(len(out_shape)):
                    if out_shape[i] != 1 and out_shape[i] != shape[i]:
                        compatible = False
                        break
                if compatible:
                    if bcast_shape is None or _shape_numel(shape) > _shape_numel(bcast_shape):
                        bcast_shape = shape
                        consumer_layout = layout
        if consumer_layout is None:
            return False

        stride_below = 1
        for s in bcast_shape[dim + 1:]:
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
            offset_var = self._lookup(offset_id)

            # Phase 4c: MEPT array-of-offsets. If the offset operand is
            # an env_array, record env_ptr_array so tt.load / tt.store
            # can emit per-position accesses. The base pointer is always
            # scalar (one buffer); only the offset varies per array slot.
            if (self.mept_enabled
                    and offset_id in self.env_array):
                offset_arr, n, _ = self.env_array[offset_id]
                # Chained addptr with array offset: combine with any
                # existing scalar parent offset.
                parent_ptr_info = self.env_is_ptr.get(ptr_id)
                parent_ptr_array = self.env_ptr_array.get(ptr_id)
                if parent_ptr_array:
                    base_ptr, parent_arr, parent_n = parent_ptr_array
                    if parent_n == n:
                        # Both array: produce a combined-offset array.
                        combined_exprs = [
                            f"{parent_arr}[{i}] + {offset_arr}[{i}]"
                            for i in range(n)
                        ]
                        combined_name = self._var_array(
                            "off", combined_exprs, "uint")
                        self.env_ptr_array[ssa.id] = (
                            base_ptr, combined_name, n)
                        # env[ssa.id] is mostly informational here;
                        # tt.load reads env_ptr_array directly.
                        self.env[ssa.id] = (
                            f"{base_ptr}[{combined_name}[0]]")
                        self._propagate_shape_elementwise(ssa)
                        return
                if parent_ptr_info:
                    base_ptr, existing_offset = parent_ptr_info
                    combined_exprs = [
                        f"{existing_offset} + {offset_arr}[{i}]"
                        for i in range(n)
                    ]
                    combined_name = self._var_array(
                        "off", combined_exprs, "uint")
                    self.env_ptr_array[ssa.id] = (base_ptr, combined_name, n)
                    self.env[ssa.id] = (
                        f"{base_ptr}[{combined_name}[0]]")
                    self._propagate_shape_elementwise(ssa)
                    return
                # No parent — base pointer is the operand directly.
                ptr_var = self._lookup(ptr_id)
                self.env_ptr_array[ssa.id] = (ptr_var, offset_arr, n)
                self.env[ssa.id] = f"{ptr_var}[{offset_arr}[0]]"
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
            from triton_metal.codegen.msl_builtins import (
                is_fp8_type, fp8_to_float_func,
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
                        mask_expr = (f"{mask_var}[{i}]" if mask_is_array
                                     else mask_var)
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
                        mask_expr = (f"{mask_var}[{i}]" if mask_is_array
                                     else mask_var)
                        other_expr = (
                            f"static_cast<float>({other_val}[{i}])"
                            if other_val_is_array
                            else f"static_cast<float>({other_val})"
                        )
                        val_exprs.append(
                            f"{mask_expr} ? {convert} : {other_expr}")
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
                mask_expr = (f"{mask_var}[{i}]" if mask_is_array
                             else mask_var)
                other_expr = (
                    f"static_cast<{compute_type}>({other_val}[{i}])"
                    if other_val_is_array
                    else f"static_cast<{compute_type}>({other_val})"
                )
                exprs.append(
                    f"{mask_expr} ? {loaded} : {other_expr}")
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
        from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_to_float_func
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

        if fp8_load:
            # FP8: load as uchar, then convert to float
            to_float = fp8_to_float_func(dtype)
            raw_var = self._next_var("raw")
            if mask_var:
                self.kb.raw_line(
                    f"    uchar {raw_var} = {mask_var} ? "
                    f"{base_ptr}[{offsets}] : uchar(0);"
                )
                self.kb.raw_line(
                    f"    float {var_name} = {mask_var} ? "
                    f"{to_float}({raw_var}) : "
                    f"static_cast<float>({other_val});"
                )
            else:
                self.kb.raw_line(
                    f"    uchar {raw_var} = {base_ptr}[{offsets}];"
                )
                self.kb.raw_line(
                    f"    float {var_name} = {to_float}({raw_var});"
                )
        elif mask_var:
            self.kb.raw_line(
                f"    {compute_type} {var_name} = {mask_var} ? "
                f"static_cast<{compute_type}>({base_ptr}[{offsets}]) : "
                f"static_cast<{compute_type}>({other_val});"
            )
        else:
            self.kb.raw_line(
                f"    {compute_type} {var_name} = "
                f"static_cast<{compute_type}>({base_ptr}[{offsets}]);"
            )

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

        # Phase 4c: MEPT array scatter. When the pointer was assembled
        # via a tt.addptr with an array offset AND the value is an
        # env_array of matching length, emit per-position writes.
        # Honors an optional mask operand (array or scalar form).
        ptr_arr_info = self.env_ptr_array.get(ptr_id)
        val_arr_info = self.env_array.get(val_id)
        if (self.mept_enabled and ptr_arr_info is not None
                and val_arr_info is not None
                and ptr_arr_info[2] == val_arr_info[1]):
            base_ptr, off_arr, n = ptr_arr_info
            val_arr, _vn, _vty = val_arr_info
            # Optional mask is the third operand if present.
            mask_var = None
            mask_is_array = False
            if len(ssa.operand_ids) >= 3:
                mid = ssa.operand_ids[2]
                if mid in self.env_is_mask or self._is_mask(mid):
                    if mid in self.env_array:
                        mask_var, _mn, _ = self.env_array[mid]
                        mask_is_array = True
                    else:
                        mask_var = self._lookup(mid)
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
                cast_val = self._fp8_cast_val(f"{val_arr}[{i}]", store_dtype)
                write = f"{base_ptr}[{off_arr}[{i}]] = {cast_val};"
                if mask_var is None:
                    self.kb.raw_line(f"    {write}")
                else:
                    cond = (f"{mask_var}[{i}]" if mask_is_array
                            else mask_var)
                    self.kb.raw_line(f"    if ({cond}) {{ {write} }}")
            return

        # Check if the value to store is smem-backed with total > block_size
        smem_descs = getattr(self, '_shared_mem_descs', {})
        val_smem = smem_descs.get(val_id)
        bs = self.effective_block_size
        if val_smem:
            val_shape = val_smem[1]
            val_total = 1
            for d in val_shape:
                val_total *= d
            if val_total > bs and len(val_shape) >= 2:
                smem_name = val_smem[0]
                M, N = val_shape[0], val_shape[1]

                # Get mask if provided
                mask_id = None
                if len(ssa.operand_ids) >= 3:
                    mid = ssa.operand_ids[2]
                    if mid in self.env_is_mask or self._is_mask(mid):
                        mask_id = mid

                # Get the pointer info
                ptr_info = self.env_is_ptr.get(ptr_id)
                if ptr_info:
                    base_ptr, offset_expr = ptr_info

                    # Rebuild the offset expression with _fill_row / _fill_col
                    # substitution (same approach as _lower_local_alloc).
                    def _all_ops(ops):
                        for o in ops:
                            yield o
                            if o.region_ops:
                                yield from _all_ops(o.region_ops)
                            if o.else_ops:
                                yield from _all_ops(o.else_ops)
                    all_ops = list(_all_ops(self.graph.ops))

                    row_var = None
                    col_var = None
                    for mr_id, dim in self._make_range_dim.items():
                        v = self.env.get(mr_id, "")
                        if not isinstance(v, str) or not v.startswith("idx_"):
                            continue
                        if v in offset_expr:
                            if dim == 1 and col_var is None:
                                col_var = v
                            elif dim == 0 and row_var is None:
                                row_var = v
                            continue
                        # Transitive dependency check
                        if dim == 0 and row_var is None:
                            dep_names = {v}
                            changed = True
                            while changed:
                                changed = False
                                for dop in all_ops:
                                    dv = self.env.get(dop.id, "")
                                    if not isinstance(dv, str) or dv in dep_names:
                                        continue
                                    if not dop.operand_ids:
                                        continue
                                    if any(self.env.get(oid, "") in dep_names
                                           for oid in dop.operand_ids):
                                        dep_names.add(dv)
                                        changed = True
                            if any(dn in offset_expr for dn in dep_names):
                                row_var = v

                    self.kb.raw_line(
                        f"    for (uint _st = lid; _st < {val_total}u; "
                        f"_st += {bs}u) {{")
                    self.kb.raw_line(
                        f"        uint _fill_row = _st / {N}u;")
                    self.kb.raw_line(
                        f"        uint _fill_col = _st % {N}u;")

                    new_offset = offset_expr
                    emitted = set()
                    if col_var:
                        new_offset = new_offset.replace(col_var, "_fill_col")
                    if row_var:
                        # Build the set of variables in offset_expr that need
                        # substitution (directly or via dependencies).
                        needed_in_offset = set()
                        for op in all_ops:
                            v = self.env.get(op.id, "")
                            if isinstance(v, str) and v in offset_expr:
                                needed_in_offset.add(v)

                        for op in all_ops:
                            v = self.env.get(op.id, "")
                            if not isinstance(v, str) or not v.startswith("r_") or v in emitted:
                                continue
                            if not op.operand_ids:
                                continue
                            uses_row = any(self.env.get(oid, "") == row_var
                                           for oid in op.operand_ids)
                            if not uses_row:
                                continue
                            # Only emit if this var or a downstream var appears in offset
                            if v not in offset_expr:
                                # Check if any 2nd-level dep uses this var and appears in offset
                                has_downstream = False
                                for op2 in all_ops:
                                    v2 = self.env.get(op2.id, "")
                                    if isinstance(v2, str) and v2 in offset_expr and op2.operand_ids:
                                        if any(self.env.get(oid, "") == v for oid in op2.operand_ids):
                                            has_downstream = True
                                            break
                                if not has_downstream:
                                    continue
                            emitted.add(v)
                            a_ = self.env.get(op.operand_ids[0], "?")
                            b_ = self.env.get(op.operand_ids[1], "?") if len(op.operand_ids) > 1 else "0"
                            a_sub = "(int)_fill_row" if a_ == row_var else a_
                            b_sub = "(int)_fill_row" if b_ == row_var else b_
                            op_sym = " + " if "add" in (op.op or "") else " * " if "mul" in (op.op or "") else " + "
                            self.kb.raw_line(
                                f"        int _fill_{v} = {a_sub}{op_sym}{b_sub};")
                            new_offset = new_offset.replace(v, f"_fill_{v}")
                            # 2nd-level deps
                            for op2 in all_ops:
                                v2 = self.env.get(op2.id, "")
                                if not isinstance(v2, str) or not v2.startswith("r_") or v2 in emitted:
                                    continue
                                if not op2.operand_ids:
                                    continue
                                if not any(self.env.get(oid, "") == v
                                           for oid in op2.operand_ids):
                                    continue
                                if v2 not in offset_expr:
                                    continue
                                emitted.add(v2)
                                a2 = self.env.get(op2.operand_ids[0], "?")
                                b2 = self.env.get(op2.operand_ids[1], "?") if len(op2.operand_ids) > 1 else "0"
                                a2_sub = f"_fill_{v}" if a2 == v else a2
                                b2_sub = f"_fill_{v}" if b2 == v else b2
                                op2_sym = " + " if "add" in (op2.op or "") else " * " if "mul" in (op2.op or "") else " + "
                                self.kb.raw_line(
                                    f"        int _fill_{v2} = {a2_sub}{op2_sym}{b2_sub};")
                                new_offset = new_offset.replace(v2, f"_fill_{v2}")
                        new_offset = new_offset.replace(row_var, "(int)_fill_row")

                    # Mask: reconstruct per-element mask
                    mask_expr = None
                    if mask_id is not None:
                        # The mask is typically row < N_CTX. Rebuild with _fill_row.
                        mask_str = self._lookup(mask_id)
                        # Check if the mask depends on the row variable
                        if row_var and any(v in mask_str for v in emitted):
                            # Complex mask — use a simple bounds check
                            mask_expr = f"((int)_fill_row + r_8) < (int)N_CTX"
                        elif mask_str.startswith("mask_") or mask_str.startswith("("):
                            # Rebuild mask with _fill_row
                            # Simple approach: row-based mask
                            mask_expr = None  # Will use the existing mask pattern
                        else:
                            mask_expr = mask_str

                    store_val = f"{smem_name}[_st]"
                    store_dtype = self._trace_ptr_dtype(ptr_id)
                    store_type = triton_type_to_msl(store_dtype)
                    compute_type = _msl_compute_type(store_dtype)
                    if store_type != compute_type:
                        store_val = f"static_cast<{store_type}>({store_val})"

                    if mask_expr:
                        self.kb.raw_line(
                            f"        if ({mask_expr}) "
                            f"{base_ptr}[{new_offset}] = {store_val};")
                    else:
                        self.kb.raw_line(
                            f"        {base_ptr}[{new_offset}] = {store_val};")
                    self.kb.raw_line(f"    }}")
                    return

        # Detect reduce keep_dims pattern: store to (M, 1) or (1, N) shaped pointer
        # where the value comes from a reduce. The generic 2D index decomposition
        # is broken for this case, so we use guarded lid-based indexing instead.
        # Skip when dim_0 == 1 (triton_per_* pattern): the ptr already has
        # the row offset from addptr, and using base_ptr[idx] would lose it.
        ptr_shape = self.env_shapes.get(ptr_id)
        val_shape = self.env_shapes.get(val_id)
        if (self._is_2d and ptr_shape and len(ptr_shape) == 2
                and (ptr_shape[0] == 1 or ptr_shape[1] == 1)
                and ptr_shape[0] != ptr_shape[1]
                and ptr_shape[0] != 1):
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
            self.kb.raw_line(
                f"    if ({idx} < {guard_size}u) {base_ptr}[{offsets}] = {cast_val};")
            return

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
        if len(ssa.operand_ids) >= 3:
            mask_id = ssa.operand_ids[2]
            if mask_id in self.env_is_mask or self._is_mask(mask_id):
                mask_var = self._lookup(mask_id)

        # In 2D kernels, 1D store tensors must be guarded to prevent
        # duplicate writes from extra threads (e.g. after 2D→1D reduce).
        store_1d_guard = None
        # Check if the kernel has any ttg.convert_layout that did a real
        # shared memory redistribution. If so, all 1D stores in this kernel
        # should use simple lid < N guards because the convert_layout
        # changed the thread-to-element mapping to simple (thread i = element i).
        val_converted = hasattr(self, '_converted_layout_ids') and bool(getattr(self, '_converted_layout_ids', set()))
        if self._is_2d and not self._is_scalar_ptr(ptr_id):
            store_shape = self.env_shapes.get(ptr_id)
            if not store_shape:
                for op in self.graph.ops:
                    if op.id == ptr_id and op.type_str:
                        store_shape = _extract_shape(op.type_str)
                        break
            if store_shape and len(store_shape) == 1 and store_shape[0] < self.effective_block_size:
                store_1d_guard = store_shape[0]

        if store_1d_guard is not None:
            lid = self._lid_expr
            if val_converted:
                # After convert_layout, thread i has element i. Simple guard.
                guard = f"{lid} < {store_1d_guard}u"
            else:
                # After a 2D reduce (axis=1), the result is per-row and the
                # broadcast uses lid / N (blocked). Fix: use lid / N as the
                # store index and select one thread per row block.
                shape = self._effective_2d_shape
                if (shape and len(shape) >= 2 and store_1d_guard == shape[0]
                        and shape[1] > 0):
                    N = shape[1]
                    offsets = f"({lid} / {N}u)"
                    guard = f"{lid} % {N}u == 0u && {lid} / {N}u < {store_1d_guard}u"
                else:
                    guard = f"{lid} < {store_1d_guard}u"
            if mask_var:
                self.kb.raw_line(f"    if ({guard} && {mask_var}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
            else:
                self.kb.raw_line(f"    if ({guard}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
        elif mask_var:
            self.kb.raw_line(f"    if ({mask_var}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
        else:
            self.kb.raw_line(f"    {base_ptr}[{offsets}] = {cast_val};")

    def _fp8_cast_val(self, val_var: str, store_dtype: str) -> str:
        """Return the MSL expression to convert a float value to FP8 uchar.

        If store_dtype is not FP8, returns a regular static_cast or passthrough.
        """
        from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_from_float_func
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
        from triton_metal.codegen.msl_builtins import fp8_device_functions
        if not hasattr(self, '_fp8_injected'):
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

        # Constants are splat-like: every thread holds the same value.
        self._is_splat.add(ssa.id)

        value = ssa.attrs.get("value")
        var_name = self._next_var("c")

        if value is None:
            # Unknown constant — use 0
            self.env[ssa.id] = "0"
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = ()
            return

        # Check if this is a hex integer that should be interpreted as float
        is_float_type = ssa.elem_type in ("f32", "f16", "bf16", "f64")
        if isinstance(value, int) and is_float_type:
            # Hex-encoded IEEE 754 bit pattern — width depends on elem_type.
            # MLIR encodes special floats (inf/nan) as hex integers of the
            # corresponding float type's width, so we must unpack using the
            # matching width (not always f32) to correctly recover NaN/Inf.
            try:
                if ssa.elem_type == "f64":
                    float_val = _struct.unpack(
                        '<d', _struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF)
                    )[0]
                elif ssa.elem_type == "f16":
                    float_val = _struct.unpack(
                        '<e', _struct.pack('<H', value & 0xFFFF)
                    )[0]
                elif ssa.elem_type == "bf16":
                    # bfloat16: upper 16 bits of an f32 bit pattern
                    float_val = _struct.unpack(
                        '<f', _struct.pack('<I', (value & 0xFFFF) << 16)
                    )[0]
                else:  # f32
                    float_val = _struct.unpack(
                        '<f', _struct.pack('<I', value & 0xFFFFFFFF)
                    )[0]
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
        if isinstance(value, bool) or (isinstance(value, str) and value in ("true", "false")):
            bool_val = value if isinstance(value, str) else ("true" if value else "false")
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
        elif op in ("arith.maxf", "arith.maxsi"):
            self._emit_builtin_binary(ssa, "max")
        elif op == "arith.maxui":
            self._emit_builtin_binary(ssa, "max", force_unsigned=True)
        elif op in ("arith.minf", "arith.minsi"):
            self._emit_builtin_binary(ssa, "min")
        elif op == "arith.minui":
            self._emit_builtin_binary(ssa, "min", force_unsigned=True)
        # NaN-quiet min/max (IEEE 754 minNum/maxNum): return non-NaN operand
        elif op == "arith.maxnumf":
            self._emit_builtin_binary(ssa, "fmax")
        elif op == "arith.minnumf":
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
            self.kb.comment(f"UNSUPPORTED arith: {op}")

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
            layouts_by_shape = getattr(self, "_bcast_layouts_by_shape", {})
            a_shape = None
            b_shape = None
            for shape, lay in layouts_by_shape.items():
                if lay == a_lay and (a_shape is None or _shape_numel(shape) > _shape_numel(a_shape)):
                    a_shape = shape
                if lay == b_lay and (b_shape is None or _shape_numel(shape) > _shape_numel(b_shape)):
                    b_shape = shape
            if a_shape is not None and b_shape is not None:
                self._bcast_layout[ssa.id] = (
                    a_lay if _shape_numel(a_shape) >= _shape_numel(b_shape) else b_lay
                )
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
        other_shape = self.env_shapes.get(other_id)
        if other_shape is None or other_shape == ():
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
        from triton_metal.codegen.msl_builtins import is_fp8_type
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
            self.kb.raw_line(
                f"    float {var_name} = isnan({x}) "
                f"? NAN : fmin(fmax({x}, {lo}), {hi});"
            )
        else:
            # NaN-quiet: standard clamp
            self.kb.raw_line(
                f"    float {var_name} = fmin(fmax({x}, {lo}), {hi});"
            )
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
            "fp16": "half", "fp32": "float", "bf16": "bfloat", "fp64": "double",
            "f16": "half", "f32": "float", "f64": "double",
        }
        _FP_ELEM_TO_MSL = {"f16": "half", "bf16": "bfloat", "f32": "float", "f64": "double"}
        _FP_ELEM_TO_DTYPE = {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}

        # Width mapping for MSL types
        _FP_WIDTH = {"half": 16, "bfloat": 16, "float": 32, "double": 64}
        _INT_WIDTH_FROM_DTYPE = {"i8": 8, "i16": 16, "i32": 32, "i64": 64,
                                 "u8": 8, "u16": 16, "u32": 32, "u64": 64}
        _INT_MSL_FROM_WIDTH = {8: ("char", "i8"), 16: ("short", "i16"),
                               32: ("int", "i32"), 64: ("long", "i64")}

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
            _WIDTH_TO_FP = {16: ("half", "fp16"), 32: ("float", "fp32"),
                            64: ("double", "fp64")}
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

        from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_to_float_func, fp8_from_float_func

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
            self.kb.raw_line(
                f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));"
            )
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dst_dtype
        elif is_fp8_type(src_dtype) and is_fp8_type(dst_dtype):
            # FP8 → FP8: re-encode through float
            self._inject_fp8_device_functions(src_dtype)
            self._inject_fp8_device_functions(dst_dtype)
            var_name = self._next_var("fp8")
            from_func = fp8_from_float_func(dst_dtype)
            to_func = fp8_to_float_func(dst_dtype)
            self.kb.raw_line(
                f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));"
            )
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
            self.kb.raw_line(
                f"    bool {adj_var} = isfinite({f32_var}) && (fabs({back_var}) > fabs({f32_var}));"
            )
            # Compute one-ULP-nudge toward zero. IEEE 754 bit patterns grow
            # in magnitude with the float's magnitude on each side of zero;
            # in two's-complement representation of the underlying short,
            # decrementing reduces magnitude regardless of sign (e.g. f16
            # 0xC000 = -2.0 → 0xBFFF = -1.999, magnitude shrinks toward 0).
            self.kb.raw_line(
                f"    short {adj_bits} = as_type<short>({cand_var});"
            )
            self.kb.raw_line(
                f"    {adj_bits} = {adj_var} ? (short)({adj_bits} - 1) : {adj_bits};"
            )
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
        from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_to_float_func
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

        FP32 → FP8: call software conversion function.
        FP32 → FP16: passthrough (cast happens at store).
        """
        if not ssa.operand_ids:
            return
        dst_elem = ssa.elem_type or "f16"
        dst_dtype = _mlir_to_triton_dtype(dst_elem)
        from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_from_float_func, fp8_to_float_func
        if is_fp8_type(dst_dtype):
            # FP32 → FP8: emit conversion call
            src_var = self._lookup(ssa.operand_ids[0])
            self._inject_fp8_device_functions(dst_dtype)
            var_name = self._next_var("fp8")
            from_func = fp8_from_float_func(dst_dtype)
            to_func = fp8_to_float_func(dst_dtype)
            # Convert to fp8 and immediately back to float for further computation
            # The actual fp8 encoding is stored at the store boundary
            self.kb.raw_line(
                f"    float {var_name} = {to_func}({from_func}(static_cast<float>({src_var})));"
            )
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dst_dtype
        else:
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
        if self._mept_binary_dispatch(
                ssa, ssa.operand_ids[0], ssa.operand_ids[1], a, b,
                _make_expr, "bool", "i1"):
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
        pred_int is used as fallback only — its enum values can differ between
        MLIR/Triton versions, so we don't hardcode a mapping.
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        var_name = self._next_var("mask")

        # Use pred_name as primary source. Fall back to pred_int for op_str only.
        if pred_name == "false":
            self.kb.raw_line(f"    bool {var_name} = false;")
        elif pred_name == "true":
            self.kb.raw_line(f"    bool {var_name} = true;")
        elif pred_name == "uno":
            self.kb.raw_line(f"    bool {var_name} = isnan({a}) || isnan({b});")
        elif pred_name == "ord":
            self.kb.raw_line(f"    bool {var_name} = !isnan({a}) && !isnan({b});")
        elif pred_name == "une":
            # MSL != matches IEEE 754 une semantics (NaN != x is true)
            self.kb.raw_line(f"    bool {var_name} = {a} != {b};")
        elif pred_name and pred_name in CMPF_NAMED:
            op_str = CMPF_NAMED[pred_name]
            if pred_name.startswith("u"):
                self.kb.raw_line(
                    f"    bool {var_name} = isnan({a}) || isnan({b}) || ({a} {op_str} {b});"
                )
            else:
                self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        elif pred_int is not None and pred_int in CMPF_PREDICATES:
            # Fallback to pred_int when pred_name unavailable
            op_str = CMPF_PREDICATES[pred_int]
            self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        else:
            self.kb.raw_line(f"    bool {var_name} = {a} < {b};")

        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        # Shape: comparison inherits shape from operands
        self._propagate_shape_elementwise(ssa)
        # Propagate bcast layout across the comparison — the result has the
        # same (lid → flat-index) mapping as its operands.
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
        elif ir_dtype and ir_dtype.startswith("i"):
            ty = "int"
            dtype = ir_dtype
        elif ir_dtype and ir_dtype.startswith("u"):
            ty = "uint"
            dtype = ir_dtype
        elif true_dtype.startswith("fp") or true_dtype.startswith("bf"):
            ty = "float"
            dtype = true_dtype if true_dtype in ("fp16", "bf16", "fp32", "fp64") else "fp32"
        elif true_dtype.startswith("u"):
            ty = "uint"
            dtype = "u32"
        else:
            ty = "int"
            dtype = "i32"

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
                if self.mept_enabled and (
                    a_id in self.env_array
                    or b_id in self.env_array
                    or c_id in self.env_array
                ):
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
                            f"fma({_read(i, a_id, a)}, "
                            f"{_read(i, b_id, b)}, "
                            f"{_read(i, c_id, c)})"
                            for i in range(n)
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
                bin_fn = {"math.powf": "pow",
                          "math.copysign": "copysign",
                          "math.atan2": "atan2"}[op]
                a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
                a = self._lookup(a_id)
                b = self._lookup(b_id)
                # Phase 4b: MEPT array path via shared dispatcher.
                def _make_expr(av, bv, _fn=bin_fn):
                    return f"{_fn}({av}, {bv})"
                if self._mept_binary_dispatch(
                        ssa, a_id, b_id, a, b, _make_expr,
                        "float", "fp32"):
                    return
                var_name = self._next_var("r")
                self.kb.raw_line(
                    f"    float {var_name} = {bin_fn}({a}, {b});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = "fp32"

        elif op in ("math.roundeven", "math.trunc"):
            # Both are simple unary float ops with a different MSL name.
            un_fn = {"math.roundeven": "rint",
                     "math.trunc": "trunc"}[op]
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
            self.kb.comment(f"UNSUPPORTED math: {op}")
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
            self.kb.comment(f"UNSUPPORTED: tt.extern_elementwise (no symbol)")
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
                    "(e.g. test_trans_4d).", op_name="tt.trans")

        if len(src_shape) < 2 or not self._is_2d:
            # 1D or unknown — passthrough
            self._emit_passthrough(ssa)
            return

        M, N = src_shape[0], src_shape[1]
        total = M * N

        # Determine types
        input_dtype = self.env_types.get(src_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"

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
        self.kb.raw_line(
            f"    {msl_type} {result_var} = {shared_name}["
            f"(lid % {M}u) * {N}u + (lid / {M}u)];"
        )

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = input_dtype
        if dst_shape:
            self.env_shapes[ssa.id] = dst_shape

    def _find_op_type_str(self, ssa_id: int) -> str:
        """Find the type_str for an SSA value by searching ops."""
        for ssa in self.graph.ops:
            if ssa.id == ssa_id:
                return ssa.type_str
            if ssa.region_ops:
                for inner in ssa.region_ops:
                    if inner.id == ssa_id:
                        return inner.type_str
            if ssa.else_ops:
                for inner in ssa.else_ops:
                    if inner.id == ssa_id:
                        return inner.type_str
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
            self.kb.comment("UNSUPPORTED: tt.join with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N from operand shape
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

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

        result_var = self._next_var("cat")

        if trans_ssa and reshape_ssa:
            # Fused cat: in 2D mode, all threads have valid values via wrapping
            self.kb.raw_line(
                f"    {msl_type} {result_var} = (lid < {N}u) ? {a_var} : {b_var};"
            )
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
            shared_dtype = "fp32" if is_float else "i32"
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
            self.kb.comment("UNSUPPORTED: tt.cat with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

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
        self.kb.raw_line(
            f"    {msl_type} {result_var} = (lid < {N}u) ? "
            f"{shared_a}[lid] : {shared_b}[lid - {N}u];"
        )

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
            self.kb.comment("UNSUPPORTED: tt.split with no operands")
            return
        if not ssa.result_ids or len(ssa.result_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.split with < 2 results")
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
            self.kb.raw_line(
                f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {bs}u) {{"
            )
        else:
            self.kb.raw_line(f"    {shared_name}[{elem}] = {src_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read de-interleaved: z1 = even elements, z2 = odd elements
        z1_var = self._next_var("split")
        z2_var = self._next_var("split")
        self.kb.raw_line(
            f"    {msl_type} {z1_var} = {shared_name}[({elem} % {N}u) * 2u];"
        )
        self.kb.raw_line(
            f"    {msl_type} {z2_var} = {shared_name}[({elem} % {N}u) * 2u + 1u];"
        )

        # Register both results
        rid1, rid2 = ssa.result_ids[0], ssa.result_ids[1]
        self.env[rid1] = z1_var
        self.env_types[rid1] = input_dtype
        self.env[rid2] = z2_var
        self.env_types[rid2] = input_dtype
        out_shape = (N,)
        self.env_shapes[rid1] = out_shape
        self.env_shapes[rid2] = out_shape

    def _lower_tt_histogram(self, ssa: SSAValue):
        """tt.histogram → threadgroup atomic histogram.

        Input: tensor<Mxi32> of values in [0, N).
        Output: tensor<Nxi32> of bin counts.

        Uses threadgroup atomic_int array for thread-safe counting.
        """
        if not ssa.operand_ids:
            self.kb.comment("UNSUPPORTED: tt.histogram with no operands")
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
        trace_id = ssa.operand_ids[0]
        for op in self.graph.ops:
            if op.id == trace_id and op.op == "tt.load" and op.operand_ids:
                # Found the load — trace its ptr to the function arg
                ptr_id = op.operand_ids[0]
                ptr_info = self.env_is_ptr.get(ptr_id)
                if ptr_info:
                    src_ptr_name = ptr_info[0]
                break

        # Mask handling: the mask was computed inside the (now-closed) wrapping
        # loop, so its variable is out of scope in the histogram loop. Recompute
        # it symbolically using `_h` as the loop index. Fall back to the original
        # variable reference when we aren't inside a wrapping loop (still in scope).
        mask_extra = ""
        if len(ssa.operand_ids) >= 2:
            mask_operand_id = ssa.operand_ids[1]
            if in_loop and src_ptr_name:
                recomputed = self._synthesize_mask_for_index(mask_operand_id, "_h")
                if recomputed is not None:
                    mask_extra = f" && ({recomputed})"
                # If we couldn't recompute, omit the mask rather than reference
                # an out-of-scope variable. The bounds check `_h < M` still keeps
                # us from reading past the input.
            else:
                mask_var = self._lookup(mask_operand_id)
                mask_extra = f" && {mask_var}"

        if src_ptr_name:
            self.kb.raw_line(f"    for (uint _h = lid; _h < {M}u; _h += {bs}u) {{")
            self.kb.raw_line(f"        int _hval = static_cast<int>({src_ptr_name}[_h]);")
            self.kb.raw_line(f"        if (_h < {M}u{mask_extra}) atomic_fetch_add_explicit(&{hist_name}[(uint)_hval], 1, memory_order_relaxed);")
            self.kb.raw_line(f"    }}")
        else:
            # Fallback: use the loaded input_var (only works when not wrapping)
            self.kb.raw_line(f"    if (lid < {M}u{mask_extra}) atomic_fetch_add_explicit(&{hist_name}[(uint){input_var}], 1, memory_order_relaxed);")
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
            self.kb.raw_line(f"    int {result_var} = (_loop_e < {N}u) ? atomic_load_explicit(&{hist_name}[_loop_e], memory_order_relaxed) : 0;")
        else:
            self.kb.raw_line(f"    int {result_var} = (lid < {N}u) ? atomic_load_explicit(&{hist_name}[lid], memory_order_relaxed) : 0;")
            if in_loop:
                # Reopen wrapping loop for remaining ops (stores)
                total = getattr(self, "_total_elements", self.effective_block_size)
                self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {bs}u) {{")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = (N,)

    def _synthesize_mask_for_index(self, mask_id: int, index_var: str, _depth: int = 0):
        """Recompute a boolean mask expression using `index_var` as the loop index.

        Used by tt.histogram, which closes the enclosing wrapping loop and opens
        its own accumulation loop with a fresh index variable. Any mask computed
        inside the original loop references `_loop_e`, which is out of scope by
        then, so we trace the mask's producers and rebuild the expression with
        `index_var` substituted for make_range-derived values.

        Returns an MSL expression string (parenthesized) on success, or None if
        the mask is too complex to synthesize (caller falls back to dropping the
        mask, relying on the bounds check to keep reads safe).
        """
        if _depth > 8:
            return None

        op_by_id = {op.id: op for op in self.graph.ops}
        op = op_by_id.get(mask_id)
        if op is None:
            return None

        name = op.op

        # make_range(start, end) — per-thread index; substitute the loop index
        if name == "tt.make_range":
            start = op.attrs.get("start", 0)
            if start:
                return f"((int){index_var} + {int(start)})"
            return f"(int){index_var}"

        # splat/broadcast/convert_layout — pass through
        if name in ("tt.splat", "tt.broadcast", "ttg.convert_layout",
                    "tt.expand_dims", "tt.unsplat"):
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

    def _lower_tt_gather(self, ssa: SSAValue):
        """tt.gather → shared memory indexed lookup.

        For 1D: src (tensor<Sxf32>), indices (tensor<Ixi32>) → result (tensor<Ixf32>).
        Stage src to shared memory, each thread reads shared[indices[lid]].
        """
        if len(ssa.operand_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.gather with < 2 operands")
            return

        src_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Get source size from type
        src_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        S = src_shape[0] if src_shape else self.effective_block_size

        # Determine types
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"

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

    def _register_bcast_layout_by_type(self, type_str: str, shape: tuple,
                                       layout_expr: str) -> None:
        """Register a reduce output's bcast_layout, keyed both by shape and
        by the layout signature extracted from its type_str.

        This enables downstream reshape-of-make_range rewrites to find the
        layout matching the reshape's slice layout (rather than just any
        same-rank reduce, which may differ during chained compare-and-swaps
        within a single bitonic stage).
        """
        if not hasattr(self, "_bcast_layouts_by_shape"):
            self._bcast_layouts_by_shape = {}
        self._bcast_layouts_by_shape[shape] = layout_expr
        if not hasattr(self, "_bcast_layouts_by_layout"):
            self._bcast_layouts_by_layout = {}
        sig = _extract_layout_signature(type_str)
        if sig is not None:
            self._bcast_layouts_by_layout[sig] = (shape, layout_expr)

    # -- Prefix scan (tt.scan) --

    def _lower_local_alloc(self, ssa: SSAValue):
        """ttg.local_alloc -> write tensor to threadgroup shared memory.

        Cooperatively fills shared memory using a strided loop so all
        threads contribute, then barriers. When inside a wrapping loop,
        closes/reopens it so the fill is standalone.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
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

        # Close wrapping loop if active — the fill must be standalone
        bs = self.effective_block_size
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
        # Strategy: trace back to the tt.load's base pointer and strides,
        # then generate a simple cooperative loop:
        #   for (_sa = lid; _sa < total; _sa += bs) {
        #       row = _sa / N; col = _sa % N;
        #       shared[_sa] = ptr[row * stride_row + col * stride_col];
        #   }
        if total > bs and self._is_2d:
            # Trace the source chain to find the original tt.load and
            # its pointer info. Also collect any transformations applied
            # between the load and local_alloc (e.g., multiply by scale).
            load_ptr_info = None
            load_mask_op_id = None
            post_load_ops = []  # (op_type, extra_operand_var) chain

            # Build flat list of all ops including those inside scf.for bodies
            def _all_ops(ops):
                for o in ops:
                    yield o
                    if o.region_ops:
                        yield from _all_ops(o.region_ops)
                    if o.else_ops:
                        yield from _all_ops(o.else_ops)
            all_ops = list(_all_ops(self.graph.ops))

            cur_id = ssa.operand_ids[0]
            for _depth in range(10):
                for op in all_ops:
                    if op.id == cur_id:
                        if op.op == "tt.load" and op.operand_ids:
                            ptr_id = op.operand_ids[0]
                            load_ptr_info = self.env_is_ptr.get(ptr_id)
                            # Find mask
                            for oid in op.operand_ids[1:]:
                                if oid in self.env_is_mask or self._is_mask(oid):
                                    load_mask_op_id = oid
                        elif op.operand_ids:
                            # Record transformation op (e.g., arith.mulf)
                            if len(op.operand_ids) >= 2:
                                other_id = op.operand_ids[1]
                                other_var = self._lookup(other_id)
                                post_load_ops.insert(0, (op.op, other_var))
                            cur_id = op.operand_ids[0]
                        break
                if load_ptr_info is not None:
                    break

            if load_ptr_info:
                base_ptr, offset_expr = load_ptr_info
                # Find which idx variables are used in THIS offset expression.
                # Different loads may use different idx vars for the same dim.
                row_var = None
                col_var = None
                for mr_id, dim in self._make_range_dim.items():
                    v = self.env.get(mr_id, "")
                    if not isinstance(v, str) or not v.startswith("idx_"):
                        continue
                    # Direct: idx appears in offset expression
                    if v in offset_expr:
                        if dim == 1 and col_var is None:
                            col_var = v
                        elif dim == 0 and row_var is None:
                            row_var = v
                        continue
                    # Indirect: a TRANSITIVELY dependent variable appears
                    # in offset.  Build the set of all variable names that
                    # depend (directly or indirectly) on this make_range idx
                    # and check if any of them appear in offset_expr.
                    if dim == 0 and row_var is None:
                        dep_names = {v}
                        changed = True
                        while changed:
                            changed = False
                            for dop in all_ops:
                                dv = self.env.get(dop.id, "")
                                if not isinstance(dv, str) or dv in dep_names:
                                    continue
                                if not dop.operand_ids:
                                    continue
                                if any(self.env.get(oid, "") in dep_names
                                       for oid in dop.operand_ids):
                                    dep_names.add(dv)
                                    changed = True
                        if any(dn in offset_expr for dn in dep_names):
                            row_var = v

                self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {bs}u) {{")
                self.kb.raw_line(f"        uint _fill_row = _sa / {N}u;")
                self.kb.raw_line(f"        uint _fill_col = _sa % {N}u;")

                # Rebuild the offset expression by substituting row/col vars.
                new_offset = offset_expr
                emitted = set()
                if col_var:
                    new_offset = new_offset.replace(col_var, "_fill_col")
                if row_var:
                    # Find dependent variables and rebuild them
                    for op in all_ops:
                        v = self.env.get(op.id, "")
                        if not isinstance(v, str) or not v.startswith("r_") or v in emitted:
                            continue
                        if not op.operand_ids:
                            continue
                        uses_row = any(self.env.get(oid, "") == row_var for oid in op.operand_ids)
                        if not uses_row:
                            continue
                        emitted.add(v)
                        a = self.env.get(op.operand_ids[0], "?") if len(op.operand_ids) > 0 else "?"
                        b = self.env.get(op.operand_ids[1], "?") if len(op.operand_ids) > 1 else "0"
                        a_sub = "(int)_fill_row" if a == row_var else a
                        b_sub = "(int)_fill_row" if b == row_var else b
                        op_sym = " + " if "add" in (op.op or "") else " * " if "mul" in (op.op or "") else " + "
                        self.kb.raw_line(f"        int _fill_{v} = {a_sub}{op_sym}{b_sub};")
                        new_offset = new_offset.replace(v, f"_fill_{v}")
                        # 2nd-level deps
                        for op2 in all_ops:
                            v2 = self.env.get(op2.id, "")
                            if not isinstance(v2, str) or not v2.startswith("r_") or v2 in emitted:
                                continue
                            if not op2.operand_ids:
                                continue
                            if not any(self.env.get(oid, "") == v for oid in op2.operand_ids):
                                continue
                            emitted.add(v2)
                            a2 = self.env.get(op2.operand_ids[0], "?")
                            b2 = self.env.get(op2.operand_ids[1], "?") if len(op2.operand_ids) > 1 else "0"
                            a2_sub = f"_fill_{v}" if a2 == v else a2
                            b2_sub = f"_fill_{v}" if b2 == v else b2
                            op2_sym = " + " if "add" in (op2.op or "") else " * " if "mul" in (op2.op or "") else " + "
                            self.kb.raw_line(f"        int _fill_{v2} = {a2_sub}{op2_sym}{b2_sub};")
                            new_offset = new_offset.replace(v2, f"_fill_{v2}")
                    new_offset = new_offset.replace(row_var, "(int)_fill_row")

                # Load value from global memory
                val_expr = f"{base_ptr}[{new_offset}]"
                # Apply post-load transformations (e.g., * scale)
                for op_type, other_var in post_load_ops:
                    if "mul" in op_type:
                        val_expr = f"({val_expr} * {other_var})"
                    elif "add" in op_type:
                        val_expr = f"({val_expr} + {other_var})"
                self.kb.raw_line(f"        {shared_name}[_sa] = {val_expr};")
                self.kb.raw_line(f"    }}")
            else:
                # Couldn't find load pointer — fall back to per-thread value
                self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {bs}u) {{")
                self.kb.raw_line(f"        {shared_name}[_sa] = {src_var};")
                self.kb.raw_line(f"    }}")
        elif src_ptr_name:
            self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {bs}u) {{")
            self.kb.raw_line(f"        {shared_name}[_sa] = {src_ptr_name}[_sa];")
            self.kb.raw_line(f"    }}")
        else:
            # Fallback: use the value from the wrapping loop (only correct
            # when total == block_size or for a single element).
            self.kb.raw_line(f"    for (uint _sa = lid; _sa < {total}u; _sa += {bs}u) {{")
            self.kb.raw_line(f"        {shared_name}[_sa] = {src_var};")
            self.kb.raw_line(f"    }}")

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reopen wrapping loop if it was active
        if in_loop:
            total_elems = getattr(self, "_total_elements", self.effective_block_size)
            self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total_elems}u; _loop_e += {bs}u) {{")

        # Store shared array name for local_load to reference
        self.env[ssa.id] = shared_name
        self.env_types[ssa.id] = src_dtype
        self.env_shapes[ssa.id] = shape
        # Mark as shared memory descriptor
        if not hasattr(self, '_shared_mem_descs'):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (shared_name, shape, shared_dtype)
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
        shared_info = getattr(self, '_shared_mem_descs', {}).get(src_id)
        if not shared_info:
            # No shared memory descriptor — passthrough
            self._emit_passthrough(ssa)
            return

        shared_name, shape, shared_dtype = shared_info

        # Propagate shared memory descriptor so tt.dot can find the array
        self.env[ssa.id] = shared_name
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = shape
        if not hasattr(self, '_shared_mem_descs'):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = shared_info

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

        # Propagate env, env_types, env_shapes from source
        self._emit_passthrough(ssa)

        # Propagate shared memory descriptor with transposed flag
        if not hasattr(self, '_shared_mem_descs'):
            self._shared_mem_descs = {}
        shared_info = self._shared_mem_descs.get(src_id)
        if shared_info:
            shared_name, shape, shared_dtype = shared_info
            # Swap dimensions to reflect transposed access
            if len(shape) >= 2:
                trans_shape = (shape[1], shape[0]) + shape[2:]
            else:
                trans_shape = shape
            self._shared_mem_descs[ssa.id] = (shared_name, trans_shape, shared_dtype)
            self.env_shapes[ssa.id] = trans_shape
            # Mark this shared array as transposed for dot indexing
            if not hasattr(self, '_shared_mem_transposed'):
                self._shared_mem_transposed = set()
            self._shared_mem_transposed.add(shared_name)

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
            self.kb.comment("UNSUPPORTED: tt.dot with < 3 operands")
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
            from triton_metal.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Refusing to emit silently-wrong output: tt.dot operand shapes "
                f"({a_shape} x {b_shape}) are not both 2-D; the generic dot path "
                f"cannot lower this and would silently return the accumulator, "
                f"dropping the matmul. File an issue or restructure the dot.")

        M, K = a_shape[0], a_shape[1]
        K2, N = b_shape[0], b_shape[1]

        # Get shared memory names for A and B
        # Trace through local_load to find the shared memory arrays
        a_shared = getattr(self, '_shared_mem_descs', {}).get(a_id)
        b_shared = getattr(self, '_shared_mem_descs', {}).get(b_id)

        if not a_shared:
            for op in self.graph.ops:
                if op.id == a_id and op.op == "ttg.local_load" and op.operand_ids:
                    a_shared = getattr(self, '_shared_mem_descs', {}).get(op.operand_ids[0])
                    break
        if not b_shared:
            for op in self.graph.ops:
                if op.id == b_id and op.op == "ttg.local_load" and op.operand_ids:
                    b_shared = getattr(self, '_shared_mem_descs', {}).get(op.operand_ids[0])
                    break

        if not a_shared or not b_shared:
            # No shared-memory operands -> the generic simdgroup dot path can't
            # lower this. Returning the accumulator drops A@B silently; refuse
            # loudly instead (audit #165).
            from triton_metal.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "Refusing to emit silently-wrong output: tt.dot operands are not "
                "in threadgroup (shared) memory (no ttg.local_load source); the "
                "generic dot path would silently return the accumulator, dropping "
                "the matmul. File an issue.")

        a_smem, _, _ = a_shared
        b_smem, _, _ = b_shared

        # Check if operands were transposed via memdesc_trans.
        # Transposed arrays need swapped indexing: B_trans[k, col] reads
        # physical B_orig[col, k] = smem[col * K_orig + k].
        transposed = getattr(self, '_shared_mem_transposed', set())
        a_trans = a_smem in transposed
        b_trans = b_smem in transposed

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
        bs = self.effective_block_size

        # Close wrapping loop if active -- dot needs standalone computation
        in_loop = self._needs_wrapping
        if in_loop:
            self.kb.raw_line(f"    }}")  # close wrapping loop

        # Check if the accumulator is a shared-memory-backed oversized array
        # (e.g. the 32x64 accumulator in flash attention with HEAD_DIM=64).
        # If so, read the init from shared memory per-element and write the
        # result back to the SAME shared array (no new allocation needed).
        acc_smem = getattr(self, '_shared_mem_descs', {}).get(ssa.operand_ids[2])
        acc_is_smem = False
        if acc_smem:
            acc_shape = acc_smem[1]
            acc_total = 1
            for d in acc_shape:
                acc_total *= d
            if acc_total > bs:
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
        self.kb.raw_line(f"    for (uint _de = lid; _de < {total}u; _de += {bs}u) {{")
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
            self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total_elems}u; _loop_e += {bs}u) {{")

        # The result for each thread's element comes from the shared result array
        result_var = self._next_var("dot")
        elem = self._lid_expr
        self.kb.raw_line(f"    float {result_var} = ({elem} < {total}u) ? {result_smem}[{elem}] : 0.0f;")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "fp32"
        self.env_shapes[ssa.id] = (M, N)
        # Track that this variable's data is already in shared memory at lid.
        # Downstream reduce can skip the copy and read from result_smem directly.
        if not hasattr(self, '_shared_mem_descs'):
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
            ret_type = triton_type_to_msl(
                _mlir_to_triton_dtype(cfunc.return_types[0])
            )
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
        args = [self._lookup(oid) for oid in ssa.operand_ids]
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
            from triton_metal.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Refusing to emit silently-wrong output: unsupported "
                f"side-effecting op '{op}' (no result) would be dropped, losing "
                f"its effect. Add a handler or explicit refusal.")
        else:
            # Other (value-producing) ttg ops: passthrough. An undefined result
            # surfaces as UNKNOWN_<id> -> loud MSL compile error if consumed.
            self._emit_passthrough(ssa)

    def _lower_convert_layout_mept_shuffle(self, ssa, src_type, dest_type,
                                           src_shape, N) -> bool:
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
        if (src_ll.num_registers_per_thread != n_src
                or src_ll.total_elements != dst_ll.total_elements):
            return False
        n_dst = dst_ll.num_registers_per_thread

        # Shared buffer holds the whole tile in compute form (float for
        # fp/bf, int for integer), matching the existing convert_layout path.
        src_dtype = self.env_types.get(src_id, "fp32")
        is_int = not (src_dtype.startswith("fp") or src_dtype.startswith("bf"))
        shared_dtype = "i32" if is_int else "fp32"
        shared_name = f"shuf_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype,
                                          size=N)

        lane = "(lid & 31u)"
        warp = "(lid >> 5u)"
        # Barrier before writing in case the buffer aliases an earlier shuffle.
        self.kb.raw_line(
            "    threadgroup_barrier(mem_flags::mem_threadgroup);")
        for i in range(n_src):
            pos = src_ll.msl_position_expr(f"{i}u", lane, warp)
            self.kb.raw_line(f"    {shared_name}[{pos}] = {src_arr}[{i}];")
        self.kb.raw_line(
            "    threadgroup_barrier(mem_flags::mem_threadgroup);")
        read_exprs = [
            f"{shared_name}[{dst_ll.msl_position_expr(f'{j}u', lane, warp)}]"
            for j in range(n_dst)
        ]
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
        needs_redistribute = ("ttg.slice" in src_type
                              and "ttg.slice" not in dest_type)

        # Phase 4d: MEPT array shuffle. When the source value is a per-thread
        # register array and both layouts resolve to LinearLayouts, redistribute
        # the elements through threadgroup memory using the XOR-basis position
        # math. Each thread writes its source-register elements at their
        # src-layout positions, barriers, then reads its dest-register elements
        # from their dst-layout positions. This is the general layout-change
        # primitive for MEPT (and the eventual replacement for the targeted
        # transpose/interleave pattern detectors — Phase 4g).
        if (getattr(self, "mept_enabled", False)
                and ssa.operand_ids[0] in self.env_array):
            shuffled = self._lower_convert_layout_mept_shuffle(
                ssa, src_type, dest_type, src_shape, N)
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
            from triton_metal.codegen._linear_layout import parse_linear_layout
            alias = alias_match.group(1)
            if re.search(rf"#{re.escape(alias)}\s*=\s*#ttg\.linear<",
                         mod_text):
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
                        "general per-register shuffle (Phase 4).")

        if not needs_redistribute or not self._is_2d or N <= 1:
            self._emit_passthrough(ssa)
            return

        # Determine source element type
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (src_dtype.startswith("fp") or src_dtype.startswith("bf"))
        msl_type = "int" if is_int else "float"
        shared_dtype = "i32" if is_int else "fp32"

        # Allocate shared memory for the redistribution
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=N)

        # Determine the source write index.
        # After a 2D reduce with broadcast, the thread-to-element mapping
        # uses lid / N_reduced (blocked row indexing), where N_reduced is
        # the inner dim of the reduce input (NOT the global 2D shape).
        # Trace back to find the reduce that produced this value and get
        # its input inner dim.
        N_reduced = None
        reduce_axis = None
        src_id = ssa.operand_ids[0]
        # Trace through passthroughs (addf, etc.) to find the reduce
        visited = set()
        trace_id = src_id
        while trace_id not in visited:
            visited.add(trace_id)
            for op in self.graph.ops:
                if op.id == trace_id:
                    if op.op == "tt.reduce":
                        if op.operand_ids:
                            inp_shape = self.env_shapes.get(op.operand_ids[0])
                            if not inp_shape:
                                inp_type = self._find_op_type_str(op.operand_ids[0])
                                inp_shape = _extract_shape(inp_type) if inp_type else None
                            if inp_shape and len(inp_shape) >= 2:
                                reduce_axis = op.attrs.get("axis", 0)
                                if reduce_axis == 1:
                                    N_reduced = inp_shape[1]
                                elif reduce_axis == 0:
                                    N_reduced = inp_shape[0]
                        break
                    elif op.operand_ids:
                        trace_id = op.operand_ids[0]
                    break

        if N_reduced and N_reduced > 1:
            if reduce_axis == 1:
                # axis=1: broadcast used lid / N_reduced (blocked row)
                src_idx = f"lid / {N_reduced}u"
                self.kb.raw_line(f"    if (lid % {N_reduced}u == 0u && lid / {N_reduced}u < {N}u)")
                self.kb.raw_line(f"        {shared_name}[{src_idx}] = {src_var};")
            else:
                # axis=0: broadcast used lid % N (modular column)
                src_idx = f"lid % {N}u"
                self.kb.raw_line(f"    if (lid < {N}u)")
                self.kb.raw_line(f"        {shared_name}[{src_idx}] = {src_var};")
        else:
            # Standard modular mapping or no reduce found
            self.kb.raw_line(f"    if (lid < {N}u)")
            self.kb.raw_line(f"        {shared_name}[lid % {N}u] = {src_var};")

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
        if not hasattr(self, '_converted_layout_ids'):
            self._converted_layout_ids = set()
        self._converted_layout_ids.add(ssa.id)


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
