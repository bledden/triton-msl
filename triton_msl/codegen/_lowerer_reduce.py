"""Reduce / scan op lowering for ``GenericLowerer``.

Triton\'s ``tt.reduce`` and ``tt.scan`` carry a combiner body region that
defines the reduction operation (sum / max / min / argmax / Welford
moments / xor-flip / etc.). The lowering here:

  - inspects the body to identify the combine op (``_get_reduce_combine_info``)
  - dispatches to the right shape-specific path (1-D / 2-D / 3-D / N-D)
  - handles multi-value reduces (Welford, argmin/argmax tuple results)
  - integrates with the wrap-loop machinery for >1024-element tiles
    (``_lower_multipass_reduction``)

Mixed into ``GenericLowerer`` because every method reads instance state
(``self.kb``, ``self.env``, ``self.env_types``, ``self._next_var``, etc.)
and inserts MSL into ``self.kb``\'s body.
"""

import os
import re

from triton_msl.codegen.mlir_walker import SSAValue, _extract_shape
from triton_msl.codegen.msl_emitter import _msl_compute_type
from triton_msl.codegen.msl_types import triton_type_to_msl

from triton_msl.codegen._lowerer_helpers import _mlir_to_triton_dtype


class _ReduceScanMixin:
    """``tt.reduce`` and ``tt.scan`` lowering for ``GenericLowerer``."""

    def _native_value_facts(self, value_id, *, op_name):
        """Return ownership-checked native facts for one SSA value.

        ResultMeta is keyed by the native value id, but a key match alone is
        not an ownership proof: a damaged record could claim another producer,
        result position, region, or function. Build the expected ownership
        graph from explicit result/block-argument lists and bind each
        structural block to the native ``owner_id`` shared by its values.
        Consumers get no legacy ``type_str`` fallback from this helper.
        """
        from triton_msl.errors import MetalNonRecoverableError

        if not hasattr(self, "_native_value_owners"):
            owners = {}
            for index, arg in enumerate(self.graph.args):
                owners[arg.id] = ("entry_arg", None, index, ("entry",))

            def record(ops, block_token):
                for op in ops:
                    result_ids = list(op.result_ids or [])
                    if not result_ids and op.id in self.graph.result_meta:
                        result_ids = [op.id]
                    for index, result_id in enumerate(result_ids):
                        owners[result_id] = ("result", op.id, index, block_token)

                    region_args = list((op.attrs or {}).get("block_arg_ids") or [])
                    region_token = ("region", op.id)
                    for index, arg_id in enumerate(region_args):
                        owners[arg_id] = ("block_arg", None, index, region_token)
                    if op.region_ops:
                        record(op.region_ops, region_token)

                    # scf.if/while else regions have a distinct native block.
                    else_token = ("else", op.id)
                    if op.else_ops:
                        record(op.else_ops, else_token)

            record(self.graph.ops, ("entry",))
            self._native_value_owners = owners
            self._native_owner_ids = {}

        expected = self._native_value_owners.get(value_id)
        meta = self.graph.result_meta.get(value_id)
        if expected is None or meta is None:
            raise MetalNonRecoverableError(
                f"native per-result metadata is missing for SSA value {value_id}",
                op_name=op_name,
            )
        kind, producer_id, result_index, block_token = expected
        valid = (
            meta.schema_version == 1
            and meta.value_id == value_id
            and meta.kind == kind
            and meta.producer_id == producer_id
            and meta.result_index == result_index
            and meta.owner_id is not None
            and meta.function_name == self.graph.func_name
            and meta.type.unknown_reason is None
        )
        known_owner = self._native_owner_ids.get(block_token)
        if known_owner is None:
            self._native_owner_ids[block_token] = meta.owner_id
        elif known_owner != meta.owner_id:
            valid = False
        if not valid:
            raise MetalNonRecoverableError(
                f"native per-result metadata ownership is contradictory for SSA value {value_id}",
                op_name=op_name,
            )
        return meta.type

    def _scan_native_value_facts(self, value_id):
        return self._native_value_facts(value_id, op_name="tt.scan")

    @staticmethod
    def _native_element_signature(facts):
        return (facts.kind, facts.elem, facts.width, facts.signed)

    def _prove_scan_native_contract(self, ssa):
        """Prove the typed SSA boundary consumed by a scan lowering.

        The proof is deliberately per slot. A multi-result scan may combine
        unrelated widths (for example fp32 values and i64 indices), so neither
        result zero nor the mutable ``env_types`` map is an authority for its
        siblings. Region arguments and returned values must be scalar
        instances of the corresponding tensor slot.
        """
        from triton_msl.errors import MetalNonRecoverableError

        operand_ids = list(ssa.operand_ids or [])
        result_ids = list(ssa.result_ids or [ssa.id])
        block_arg_ids = list((ssa.attrs or {}).get("block_arg_ids") or [])
        returns = [op for op in (ssa.region_ops or []) if op.op == "tt.scan.return"]
        n_values = len(operand_ids)
        if (
            n_values == 0
            or len(result_ids) != n_values
            or len(block_arg_ids) != 2 * n_values
            or len(returns) != 1
            or len(returns[0].operand_ids or []) != n_values
        ):
            raise MetalNonRecoverableError(
                "tt.scan native contract requires exact operand, result, block-argument, and return arity",
                op_name="tt.scan",
            )

        operand_types = [self._scan_native_value_facts(value_id) for value_id in operand_ids]
        shape = operand_types[0].shape
        if (
            not operand_types[0].is_tensor
            or shape is None
            or len(shape) not in (1, 2)
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise MetalNonRecoverableError(
                "tt.scan native metadata does not prove a positive rank-1/rank-2 tensor shape",
                op_name="tt.scan",
            )

        allowed_kinds = {"float", "integer"}
        for slot, facts in enumerate(operand_types):
            if (
                not facts.is_tensor
                or facts.shape != shape
                or facts.kind not in allowed_kinds
                or facts.elem is None
                or not isinstance(facts.width, int)
                or facts.width <= 0
            ):
                raise MetalNonRecoverableError(
                    f"tt.scan native metadata does not prove tensor slot {slot}'s shape and width",
                    op_name="tt.scan",
                )

        result_types = [self._scan_native_value_facts(value_id) for value_id in result_ids]
        for slot, (source, result) in enumerate(zip(operand_types, result_types)):
            if (
                not result.is_tensor
                or result.shape != shape
                or self._native_element_signature(result) != self._native_element_signature(source)
            ):
                raise MetalNonRecoverableError(
                    f"tt.scan result {slot} does not preserve its native slot shape and type",
                    op_name="tt.scan",
                )

        block_types = [self._scan_native_value_facts(value_id) for value_id in block_arg_ids]
        return_ids = list(returns[0].operand_ids or [])
        return_types = [self._scan_native_value_facts(value_id) for value_id in return_ids]
        for slot, source in enumerate(operand_types):
            signature = self._native_element_signature(source)
            for facts in (block_types[slot], block_types[n_values + slot], return_types[slot]):
                if facts.is_tensor or facts.shape != () or self._native_element_signature(facts) != signature:
                    raise MetalNonRecoverableError(
                        f"tt.scan combine region does not preserve scalar slot {slot}'s native type",
                        op_name="tt.scan",
                    )

        # Check every typed body result as well as the values returned from it.
        # This catches a damaged intermediate record before an emitter can use
        # its legacy spelling to select an overload.
        for body_op in ssa.region_ops or []:
            if body_op.op == "tt.scan.return":
                continue
            body_results = list(body_op.result_ids or [])
            if not body_results and body_op.id in self.graph.result_meta:
                body_results = [body_op.id]
            for value_id in body_results:
                facts = self._scan_native_value_facts(value_id)
                if facts.kind not in allowed_kinds or facts.elem is None or facts.width is None:
                    raise MetalNonRecoverableError(
                        "tt.scan combine body has an unproved native result representation",
                        op_name="tt.scan",
                    )

        axis = (ssa.attrs or {}).get("axis", 0)
        if type(axis) is not int or not 0 <= axis < len(shape):
            raise MetalNonRecoverableError(
                "tt.scan native contract has an invalid axis for its operand rank",
                op_name="tt.scan",
            )
        return {
            "shape": tuple(shape),
            "axis": axis,
            "slot_dtypes": [_mlir_to_triton_dtype(facts.elem) for facts in operand_types],
        }

    def _prove_reduce_native_contract(self, ssa):
        """Prove every typed boundary consumed by a reduce.

        Single- and multi-result reductions pair corresponding logical
        elements. Every operand must have the same shape/layout, every result
        must have the axis removed, and both halves of every combine slot must
        preserve that slot's scalar representation through the returned value.
        None of these facts may come from mutable ``env_types``/``env_shapes``
        or printed type strings.
        """
        from triton_msl.errors import MetalNonRecoverableError

        operand_ids = list(ssa.operand_ids or [])
        result_ids = list(ssa.result_ids or [ssa.id])
        block_arg_ids = list((ssa.attrs or {}).get("block_arg_ids") or [])
        return_ids = list((ssa.attrs or {}).get("return_ids") or [])
        n_values = len(operand_ids)
        if (
            n_values < 1
            or len(result_ids) != n_values
            or len(block_arg_ids) != 2 * n_values
            or len(return_ids) != n_values
        ):
            raise MetalNonRecoverableError(
                "tt.reduce native contract requires exact operand, result, block-argument, and return arity",
                op_name="tt.reduce",
            )

        operand_types = [self._native_value_facts(value_id, op_name="tt.reduce") for value_id in operand_ids]
        shape = operand_types[0].shape
        layout = operand_types[0].layout
        allowed_kinds = {"float", "integer"}
        if (
            not operand_types[0].is_tensor
            or shape is None
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise MetalNonRecoverableError(
                "tt.reduce native metadata does not prove a positive-rank tensor shape",
                op_name="tt.reduce",
            )
        for slot, facts in enumerate(operand_types):
            if (
                not facts.is_tensor
                or facts.shape != shape
                or facts.layout != layout
                or facts.kind not in allowed_kinds
                or facts.elem is None
                or not isinstance(facts.width, int)
                or facts.width <= 0
            ):
                raise MetalNonRecoverableError(
                    f"tt.reduce native metadata does not prove tensor slot {slot}'s shape, layout, and representation",
                    op_name="tt.reduce",
                )

        axis = (ssa.attrs or {}).get("axis", 0)
        if type(axis) is not int or not 0 <= axis < len(shape):
            raise MetalNonRecoverableError(
                "tt.reduce native contract has an invalid axis for its operand rank",
                op_name="tt.reduce",
            )
        output_shape = tuple((*shape[:axis], *shape[axis + 1 :]))

        result_types = [self._native_value_facts(value_id, op_name="tt.reduce") for value_id in result_ids]
        result_layout = result_types[0].layout
        for slot, (source, result) in enumerate(zip(operand_types, result_types)):
            if (
                result.shape != output_shape
                or result.is_tensor != bool(output_shape)
                or result.layout != result_layout
                or self._native_element_signature(result) != self._native_element_signature(source)
            ):
                raise MetalNonRecoverableError(
                    f"tt.reduce result {slot} does not preserve its native slot type and reduced shape",
                    op_name="tt.reduce",
                )

        block_types = [self._native_value_facts(value_id, op_name="tt.reduce") for value_id in block_arg_ids]
        return_types = [self._native_value_facts(value_id, op_name="tt.reduce") for value_id in return_ids]
        for slot, source in enumerate(operand_types):
            signature = self._native_element_signature(source)
            for facts in (
                block_types[slot],
                block_types[n_values + slot],
                return_types[slot],
            ):
                if facts.is_tensor or facts.shape != () or self._native_element_signature(facts) != signature:
                    raise MetalNonRecoverableError(
                        f"tt.reduce combine region does not preserve scalar slot {slot}'s native representation",
                        op_name="tt.reduce",
                    )

        for body_op in ssa.region_ops or []:
            body_results = list(body_op.result_ids or [])
            if not body_results and body_op.id in self.graph.result_meta:
                body_results = [body_op.id]
            for value_id in body_results:
                facts = self._native_value_facts(value_id, op_name="tt.reduce")
                if facts.kind not in allowed_kinds or facts.elem is None or facts.width is None:
                    raise MetalNonRecoverableError(
                        "tt.reduce combine body has an unproved native result representation",
                        op_name="tt.reduce",
                    )

        return {
            "shape": tuple(shape),
            "output_shape": output_shape,
            "axis": axis,
            "slot_facts": operand_types,
            "result_facts": result_types,
            "slot_dtypes": [_mlir_to_triton_dtype(facts.elem) for facts in operand_types],
            "result_dtypes": [_mlir_to_triton_dtype(facts.elem) for facts in result_types],
        }

    def _register_bcast_layout_by_native(self, facts, shape, layout_expr):
        """Register a reduce broadcast using its proved native result layout."""
        shape = tuple(shape)
        if facts.layout is not None:
            if not hasattr(self, "_bcast_layouts_by_layout"):
                self._bcast_layouts_by_layout = {}
            registration = (shape, layout_expr)
            existing = self._bcast_layouts_by_layout.get(facts.layout)
            if existing is not None and existing != registration:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "tt.reduce native slice layout was registered by multiple "
                    "stages with conflicting shape or projection",
                    op_name="tt.reduce",
                )
        # Shape is not a stage identity: chained sort reductions legitimately
        # produce the same logical shape with different projections. Index in
        # the direction the binary consumer actually queries, retaining every
        # shape for an expression instead of overwriting shape -> expression.
        if not hasattr(self, "_bcast_shapes_by_expr"):
            self._bcast_shapes_by_expr = {}
        self._bcast_shapes_by_expr.setdefault(layout_expr, set()).add(shape)
        if facts.layout is not None:
            self._bcast_layouts_by_layout[facts.layout] = registration

    @staticmethod
    def _reduce_body_result_id(op):
        result_ids = list(op.result_ids or [])
        if result_ids:
            return result_ids[0] if len(result_ids) == 1 else None
        return op.id

    def _classify_exact_welford_recurrence(self, ssa):
        """Return the source's ratio mode for the recurrence we re-emit.

        Merely finding ``arith.divf`` is not evidence of Welford: a weighted
        mean tuple has a division too, and replaying it as Welford changes its
        independent second result into an M2 accumulator.  Match the complete
        returned value DAG with the same operand order and association the
        template emits; algebraic equivalence alone would not preserve floating
        rounding, signed zero, or NaN payload behavior.
        """
        block_args = list((ssa.attrs or {}).get("block_arg_ids") or [])
        return_ids = list((ssa.attrs or {}).get("return_ids") or [])
        if len(block_args) != 6 or len(return_ids) != 3:
            return None
        mean_a, m2_a, weight_a, mean_b, m2_b, weight_b = block_args
        mean_out, m2_out, weight_out = return_ids

        producers = {}
        for op in ssa.region_ops or []:
            result_id = self._reduce_body_result_id(op)
            if result_id is None or result_id in producers:
                return None
            producers[result_id] = op

        def exact_binary(value_id, op_name, left, right, *, commutative=False):
            op = producers.get(value_id)
            if op is None or op.op != op_name or len(op.operand_ids or []) != 2:
                return False
            operands = tuple(op.operand_ids)
            return operands == (left, right) or (commutative and operands == (right, left))

        def unique_binary(op_name, left, right, *, commutative=False):
            matches = [
                value_id
                for value_id in producers
                if exact_binary(
                    value_id,
                    op_name,
                    left,
                    right,
                    commutative=commutative,
                )
            ]
            return matches[0] if len(matches) == 1 else None

        if not exact_binary(weight_out, "arith.addf", weight_a, weight_b):
            return None
        delta = unique_binary("arith.subf", mean_b, mean_a)
        raw_ratio = unique_binary("arith.divf", weight_b, weight_out)
        if delta is None or raw_ratio is None:
            return None

        ratio = None
        ratio_mode = "unguarded"
        for value_id, select in producers.items():
            if select.op != "arith.select" or len(select.operand_ids or []) != 3:
                continue
            condition_id, zero_id, false_id = select.operand_ids
            if false_id != raw_ratio:
                continue
            condition = producers.get(condition_id)
            zero_facts = self._native_value_facts(zero_id, op_name="tt.reduce")
            if (
                condition is not None
                and condition.op == "arith.cmpf"
                and tuple(condition.operand_ids or []) == (weight_out, zero_id)
                and condition.attrs.get("predicate_name") == "oeq"
                and zero_facts.kind == "float"
                and zero_facts.shape == ()
            ):
                # The external zero constant is part of the returned DAG but
                # not the reduce region. Verify its producer/value directly.
                def find_zero(ops):
                    for op in ops:
                        result_ids = list(op.result_ids or [op.id])
                        if zero_id in result_ids and op.op == "arith.constant":
                            return op
                        found = find_zero(op.region_ops or []) or find_zero(op.else_ops or [])
                        if found is not None:
                            return found
                    return None

                zero_op = find_zero(self.graph.ops)
                if zero_op is not None and zero_op.attrs.get("value") == 0.0:
                    ratio = value_id
                    ratio_mode = "guarded"
                    break
        if ratio is None:
            # Triton's public generic-reduction test spells the same Welford
            # recurrence without the zero-weight select.  Preserve that exact
            # source behavior rather than either refusing it or silently adding
            # a guard: the emitter below is parameterized by this classification.
            ratio = raw_ratio

        mean_correction = unique_binary("arith.mulf", delta, ratio)
        if mean_correction is None or not exact_binary(mean_out, "arith.addf", mean_a, mean_correction):
            return None

        m2_base = unique_binary("arith.addf", m2_a, m2_b)
        delta_squared = unique_binary("arith.mulf", delta, delta)
        weighted_delta = unique_binary("arith.mulf", delta_squared, weight_a) if delta_squared is not None else None
        m2_correction = unique_binary("arith.mulf", weighted_delta, ratio) if weighted_delta is not None else None
        if m2_base is None or m2_correction is None or not exact_binary(m2_out, "arith.addf", m2_base, m2_correction):
            return None
        return ratio_mode

    def _classify_exact_argminmax_recurrence(self, ssa, value_kind):
        """Return ``(is_max, unsigned)`` only for the tuple DAG we emit.

        A body containing a comparison is not necessarily argmin/argmax.  For
        example, lexicographic max-by-key compares slot one and returns its
        associated slot zero; replaying that body as argmax compares slot zero
        instead.  Require the canonical value comparison, equality/index
        tie-break, shared condition, and per-slot selects all the way to the
        region's returned values.
        """
        block_args = list((ssa.attrs or {}).get("block_arg_ids") or [])
        return_ids = list((ssa.attrs or {}).get("return_ids") or [])
        if len(block_args) != 4 or len(return_ids) != 2:
            return None
        value_a, index_a, value_b, index_b = block_args
        value_out, index_out = return_ids

        producers = {}
        for op in ssa.region_ops or []:
            result_id = self._reduce_body_result_id(op)
            if result_id is None or result_id in producers:
                return None
            producers[result_id] = op

        value_select = producers.get(value_out)
        index_select = producers.get(index_out)
        if (
            value_select is None
            or index_select is None
            or value_select.op != "arith.select"
            or index_select.op != "arith.select"
            or len(value_select.operand_ids or []) != 3
            or len(index_select.operand_ids or []) != 3
            or tuple(value_select.operand_ids[1:]) != (value_a, value_b)
            or tuple(index_select.operand_ids[1:]) != (index_a, index_b)
            or value_select.operand_ids[0] != index_select.operand_ids[0]
        ):
            return None

        condition = producers.get(value_select.operand_ids[0])
        if condition is None or condition.op != "arith.ori" or len(condition.operand_ids or []) != 2:
            return None
        condition_children = [producers.get(value_id) for value_id in condition.operand_ids]
        tie = next((op for op in condition_children if op is not None and op.op == "arith.andi"), None)
        winner = next(
            (op for op in condition_children if op is not None and op.op in ("arith.cmpf", "arith.cmpi")), None
        )
        if tie is None or winner is None or len(tie.operand_ids or []) != 2:
            return None

        tie_children = [producers.get(value_id) for value_id in tie.operand_ids]
        if any(op is None for op in tie_children):
            return None
        value_eq = next(
            (op for op in tie_children if tuple(op.operand_ids or []) in ((value_a, value_b), (value_b, value_a))),
            None,
        )
        index_lt = next(
            (op for op in tie_children if tuple(op.operand_ids or []) == (index_a, index_b)),
            None,
        )
        expected_value_op = "arith.cmpf" if value_kind == "float" else "arith.cmpi"
        expected_eq = "oeq" if value_kind == "float" else "eq"
        if (
            value_eq is None
            or index_lt is None
            or value_eq is index_lt
            or value_eq.op != expected_value_op
            or value_eq.attrs.get("predicate_name") != expected_eq
            or index_lt.op != "arith.cmpi"
            or index_lt.attrs.get("predicate_name") != "slt"
            or winner.op != expected_value_op
            or tuple(winner.operand_ids or []) != (value_a, value_b)
        ):
            return None

        predicate = winner.attrs.get("predicate_name")
        if value_kind == "float":
            if predicate not in ("ogt", "olt"):
                return None
            return predicate == "ogt", False
        if predicate not in ("sgt", "slt", "ugt", "ult"):
            return None
        return predicate in ("sgt", "ugt"), predicate in ("ugt", "ult")

    def _mept_reduce_fold(self, arr_name: str, n: int, combine_op: str, msl_type: str) -> str:
        """Phase 4e: fold a per-thread register array to a scalar partial.

        ``arr_name[0..n-1]`` are this thread's elements (single-pass MEPT,
        so they are the thread's full share of the tile). Folding with the
        reduce's combine op produces one partial per thread; the existing
        cross-thread SIMD/threadgroup reduce then finishes the job. This is
        the array-form analogue of the multipass wrap-loop accumulator.
        """
        combine = {
            "sum": lambda a, b: f"{a} + {b}",
            "prod": lambda a, b: f"{a} * {b}",
            "max": lambda a, b: f"max({a}, {b})",
            "min": lambda a, b: f"min({a}, {b})",
            "xor": lambda a, b: f"{a} ^ {b}",
            "and": lambda a, b: f"{a} & {b}",
            "or": lambda a, b: f"{a} | {b}",
            # NaN-PROPAGATING max/min: a NaN in either operand must propagate (max()/min()
            # are NaN-quiet); the cross-thread threadgroup_reduce propagates too.
            "nanmax": lambda a, b: f"(({a} > {b}) || ({a} != {a})) ? {a} : {b}",
            "nanmin": lambda a, b: f"(({a} < {b}) || ({a} != {a})) ? {a} : {b}",
        }.get(combine_op)
        if combine is None:
            # Never default to sum — that would SILENTLY mis-compute an unrecognised combine
            # on the register-array path. Refuse loudly (matches the rest of the surface).
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"register-array reduce: unsupported combine op '{combine_op}' — refusing "
                f"rather than silently folding as a sum.",
                op_name="tt.reduce",
            )

        # Cast each array read to ``msl_type`` so both combine operands have
        # the same type. Without this, an unsigned array (uint8/uint16) mixed
        # with an int ``fold_var`` makes MSL's max/min overload resolution
        # ambiguous (int vs uint candidates). Casting to msl_type also
        # matches the downstream threadgroup_reduce, which receives the
        # folded value at this same type — and mirrors the scalar path,
        # which reduces unsigned narrow ints in signed int32 (they fit).
        def _read(i):
            return f"({msl_type}){arr_name}[{i}]"

        fold_var = self._next_var("fold")
        self.kb.raw_line(f"    {msl_type} {fold_var} = {_read(0)};")
        for i in range(1, n):
            self.kb.raw_line(f"    {fold_var} = {combine(fold_var, _read(i))};")
        return fold_var

    def _cover_inloop_reduce(self, ssa, combine_op, msl_type, total):
        """Stage A: make an in-loop 1-D reduce cover its whole tile.

        Inside an scf.for body where block_size (num_threads) < tile size, a raw
        cross-lane reduce sums only the first num_threads elements. This folds
        each thread's strided share into a scalar accumulator BEFORE the cross-
        thread reduce — the non-register-array analogue of _mept_reduce_fold and
        the in-loop analogue of the top-level _lower_multipass_reduction wrap.

        Re-emits the reduce input's per-element dependency chain inside a
        ``for (_loop_e = lid; _loop_e < total; _loop_e += block_size)`` loop with
        ``_needs_wrapping`` set (so make_range's lid term becomes _loop_e while
        the loop-dependent base, e.g. r = k*BLOCK, is preserved), accumulating
        with the reduce's combine op.

        Returns the accumulator var name, or None if the chain is not safely
        replayable (caller must then keep the loud Stage B refusal).
        """
        iid = ssa.operand_ids[0]
        body_ops = getattr(self, "_current_loop_body_ops", None) or []
        if not body_ops:
            return None

        body_ids = {o.id for o in body_ops}

        # The reduce input itself must be produced inside this body.
        if iid not in body_ids:
            return None

        # Dependency closure of the reduce input within the body.
        deps = self._collect_tensor_deps([ssa], body_ops, set())
        dep_ids = {o.id for o in deps}

        # SAFETY GATE (never silent-wrong): every TENSOR-shaped operand reachable
        # from the reduce input that comes from OUTSIDE the body must be safely
        # replayable. Index/shape ops (make_range, splat, broadcast, expand_dims,
        # addptr, pure arithmetic) are safe because they are re-emitted with
        # _needs_wrapping and produce the same index values. Data-bearing ops
        # (tt.load and any value derived from one) are NOT safe because their
        # content is fixed at the point of the original load and cannot be
        # re-strided differently.
        #
        # Build the set of all IDs that derive from a tt.load in the full
        # graph (and a lookup for safe external ops that need to be replayed).
        _load_derived = set()
        _DATA_OPS = frozenset({"tt.load", "tt.atomic_rmw", "tt.atomic_cas"})
        _all_by_id = {}

        def _scan_all(ops):
            for op in ops:
                _all_by_id[op.id] = op
                if op.region_ops:
                    _scan_all(op.region_ops)
                if op.else_ops:
                    _scan_all(op.else_ops)

        _scan_all(getattr(self, "graph", None) and self.graph.ops or [])
        # BFS: seed with load ops, propagate to their consumers.
        _worklist = [o for o in _all_by_id.values() if o.op in _DATA_OPS]
        for _o in _worklist:
            _load_derived.add(_o.id)
        # Also propagate: any op whose operand is load-derived is also
        # load-derived (conservative but correct; pure index consumers of loads
        # are unusual so over-refusal here is acceptable).
        _changed = True
        while _changed:
            _changed = False
            for _o in _all_by_id.values():
                if _o.id in _load_derived:
                    continue
                if any(op_id in _load_derived for op_id in (_o.operand_ids or [])):
                    _load_derived.add(_o.id)
                    _changed = True

        # Collect external safe deps: ops referenced by body deps that are
        # NOT in the body, NOT load-derived (safe to replay), and ARE tensor-
        # producing. These are index/shape ops (e.g. tt.make_range) that may
        # not yet be in env (e.g. when running inside a multipass outer loop).
        # We include them in the replay so the inner loop can re-emit them
        # with _needs_wrapping=True (giving _loop_e instead of lid).
        # Superset of _SAFE_PREEMIT_OPS in _lowerer_control.py; that set is the
        # index-only subset safe to pre-emit at outer body scope, while this
        # set covers all ops safe to replay inside the _cover_inloop_reduce loop.
        _SAFE_REPLAY_OPS = frozenset(
            {
                "tt.make_range",
                "tt.splat",
                "tt.broadcast",
                "tt.expand_dims",
                "tt.addptr",
                "arith.constant",
                "arith.extf",
                "arith.truncf",
                "arith.sitofp",
                "arith.fptosi",
            }
        )
        external_safe_deps = []
        external_safe_dep_ids = set()
        # BFS over external references from body deps
        _ext_worklist = []
        for d in deps:
            for oid in d.operand_ids:
                if oid not in dep_ids and oid not in body_ids:
                    _ext_worklist.append(oid)
        _visited_ext = set()
        while _ext_worklist:
            oid = _ext_worklist.pop()
            if oid in _visited_ext:
                continue
            _visited_ext.add(oid)
            if oid in _load_derived:
                return None
            # Check if this is a body dep that's already covered
            if oid in dep_ids:
                continue
            ext_op = _all_by_id.get(oid)
            if ext_op is None:
                # Not found in graph — must be a scalar (arg, constant, iter var)
                # that stays in scope. Safe to skip.
                continue
            if ext_op.is_tensor:
                if oid in _load_derived:
                    return None
                # Add to safe external deps if it's a replay-safe op
                if ext_op.op in _SAFE_REPLAY_OPS:
                    external_safe_dep_ids.add(oid)
                    # Walk its operands too
                    for sub_oid in ext_op.operand_ids or []:
                        if sub_oid not in _visited_ext:
                            _ext_worklist.append(sub_oid)
                else:
                    # Tensor external dep that's not a safe replay op → refuse
                    return None
            # Non-tensor external ops (scalars) stay in scope — no action needed

        # Build the ordered list of external safe deps in graph order
        _all_id_order = {oid: idx for idx, oid in enumerate(_all_by_id.keys())}
        external_safe_deps = [
            _all_by_id[oid] for oid in sorted(external_safe_dep_ids, key=lambda i: _all_id_order.get(i, 0))
        ]

        identity, combine_expr = self._reduce_identity_combine(combine_op, msl_type)
        acc = self._next_var("inloop_acc")
        self.kb.raw_line(f"        {msl_type} {acc} = {identity};")
        self.kb.raw_line(f"        for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {self.kb.block_size}u) {{")

        # Save env bindings the replay will overwrite, so later body ops keep
        # their original in-scope vars (a 2nd reduce's shared elementwise would
        # otherwise reference an out-of-scope name → Metal compile error).
        # env_array/env_shapes are included: a replayed make_range writes
        # env_array/env_shapes entries pointing at vars declared INSIDE the
        # now-closed _loop_e loop; those stale descriptors must not leak to
        # later body ops or a second in-loop reduce (wrong shape routing /
        # out-of-scope variable reference → Metal compile error).
        all_replay_ids = list(dep_ids) + list(external_safe_dep_ids)
        saved_env = {rid: self.env.get(rid) for rid in all_replay_ids}
        saved_ty = {rid: self.env_types.get(rid) for rid in all_replay_ids}
        saved_arr = {rid: self.env_array.get(rid) for rid in all_replay_ids}
        saved_shp = {rid: self.env_shapes.get(rid) for rid in all_replay_ids}

        self._needs_wrapping = True
        # Re-emit external safe index ops first (e.g. make_range → _loop_e)
        for d in external_safe_deps:
            self._lower_op(d)
        # Then re-emit the body dependency chain
        for d in deps:
            self._lower_op(d)
        val = self._lookup(iid)
        self.kb.raw_line(
            f"            {{ {msl_type} acc = {acc}; {msl_type} val = ({msl_type}){val}; {acc} = {combine_expr}; }}"
        )
        self._needs_wrapping = False
        self.kb.raw_line(f"        }}")

        for rid in all_replay_ids:
            if saved_env[rid] is not None:
                self.env[rid] = saved_env[rid]
            elif rid in self.env:
                del self.env[rid]
            if saved_ty[rid] is not None:
                self.env_types[rid] = saved_ty[rid]
            elif rid in self.env_types:
                del self.env_types[rid]
            # env_array / env_shapes: restore prior value, or DELETE the entry
            # the replay introduced (so a stale descriptor pointing into the
            # closed _loop_e loop can't leak to later body ops).
            if saved_arr[rid] is not None:
                self.env_array[rid] = saved_arr[rid]
            elif rid in self.env_array:
                del self.env_array[rid]
            if saved_shp[rid] is not None:
                self.env_shapes[rid] = saved_shp[rid]
            elif rid in self.env_shapes:
                del self.env_shapes[rid]

        return acc

    @staticmethod
    def _reduce_is_unsigned_minmax(region_ops):
        """True if a reduce combine is an UNSIGNED max/min: arith.maxui/minui, or a cmp+select
        whose cmpi predicate is unsigned (ugt/ult/uge/ule). Triton types uint32 as i32 but
        emits the unsigned op, so the reduction must compare UNSIGNED — a signed compare reads
        0xFFFFFFFF as -1 and returns the wrong element (Triton-lens re-audit 2026-06-25).
        """
        for b in region_ops or []:
            nm = b.op or ""
            if "maxui" in nm or "minui" in nm:
                return True
            if nm == "arith.cmpi" and (b.attrs.get("predicate_name", "") or "").startswith("u"):
                return True
        return False

    @staticmethod
    def _reduce_acc_msl_type(dtype, unsigned=False):
        """(msl_type, shared_dtype) for a reduce accumulator/staging of `dtype`. Single
        source of truth for the reduce-accumulator type selection (was duplicated across the
        1-D, multipass, and scan paths). `unsigned=True` routes a 32-bit max/min through uint
        — Triton types uint32 as i32, so a signed compare reads 0xFFFFFFFF as -1.
        """
        if dtype.startswith("fp") or dtype.startswith("bf"):
            return "float", "fp32"
        if dtype in ("i64", "u64", "ui64"):
            # Triton types uint64 as the SIGNLESS i64, so an unsigned 64-bit max/min
            # (arith.maxui/minui -> unsigned=True) arrives here as dtype 'i64'. Route it to
            # ulong exactly like the explicit u64/ui64 names — else a signed `long` compare
            # reads 2^64-1 as -1 and silently loses the max (wrong above 2^63). This mirrors
            # the 32-bit `unsigned -> uint` branch below.
            if unsigned or dtype in ("u64", "ui64"):
                return "ulong", "u64"
            return "long", "i64"
        if unsigned:
            return "uint", "u32"
        return "int", "i32"

    # ------------------------------------------------------------------ #
    # Structural reduce-combine classifier (single source of truth).
    #
    # Classifies a 1-input tt.reduce's combine by EXACT STRUCTURE — the
    # yielded op matched against the combine's block arguments — rather than
    # by sniffing op-name substrings across the whole region. This is what
    # lets it distinguish a pure sum (yielded `addf(a,b)`) from a custom
    # `a + relu(b)` (yielded `addf(a, select(...))`, whose operands are NOT
    # the block args): the latter REFUSES. An UNRECOGNISED / custom combine
    # always returns None so the caller refuses loudly — never a silent
    # default-to-sum / plain-max. Replaces the prior twin substring loops +
    # _cmp_select_reduce_kind + _reduce_combine_has_foreign_op +
    # _reduce_is_unsigned_minmax.
    # ------------------------------------------------------------------ #
    def classify_reduce_combine(self, ssa):
        """Return (kind, signed) for a 1-input tt.reduce combine, or None to refuse.

        kind in {'sum','max','min','and','or','xor'}; `signed` is meaningful only for
        max/min (False = unsigned). None means the combine is not a provably-canonical
        reduction (a product, max-by-magnitude, NaN-propagating max/min, custom a+relu(b),
        an identity/first/last pick, ...) — the caller must refuse.
        """
        ops = ssa.region_ops or []
        ba = (ssa.attrs or {}).get("block_arg_ids") or []
        if not ops or len(ba) != 2:
            return None  # tuple combines (argmax/argmin/Welford) use the multi-value path
        a, b = ba[0], ba[1]
        bargs = {a, b}
        top = ops[-1]  # yielded op (the reduce.return terminator is parsed out)
        # Packet 192 (188 §4): the region's LAST op is not necessarily what the reduction RETURNS —
        # a legal region can compute `%s = addf %a, %b` and `tt.reduce.return %a` (a projection,
        # the add dead), which this classifier used to lower as a sum. The walker records the
        # returned ids (packet 154); the returned value must BE the classified op, else refuse.
        rets = (ssa.attrs or {}).get("return_ids")
        if rets != [top.id]:
            return None
        nm = top.op or ""
        # (1) a DIRECT binary op of exactly the two block args.
        if set(top.operand_ids or []) == bargs:
            _DIRECT = {
                "arith.addf": ("sum", True),
                "arith.addi": ("sum", True),
                "arith.mulf": ("prod", True),
                "arith.muli": ("prod", True),
                "arith.maxnumf": ("max", True),
                "arith.maxf": ("max", True),
                "arith.minnumf": ("min", True),
                "arith.minf": ("min", True),
                "arith.maxsi": ("max", True),
                "arith.minsi": ("min", True),
                "arith.maxui": ("max", False),
                "arith.minui": ("min", False),
                "arith.xori": ("xor", True),
                "arith.andi": ("and", True),
                "arith.ori": ("or", True),
                # NaN-PROPAGATING max/min as a direct op — lowered correctly (with NaN
                # propagation) via the nanmax/nanmin path, not as NaN-quiet fmax.
                "arith.maximumf": ("nanmax", True),
                "arith.minimumf": ("nanmin", True),
            }
            # everything else falls through to None -> refuse.
            return _DIRECT.get(nm)
        # (2) a cmp+select max/min: yielded select picking between the two block args,
        #     whose cmp compares exactly those block args. Sign + direction come from the
        #     cmp predicate AND the select operand mapping (where(a>b,b,a) is a MIN).
        if nm == "arith.select" and len(top.operand_ids or []) >= 3:
            return self._classify_cmp_select(ssa, ops, top, a, b)
        return None

    def _classify_cmp_select(self, ssa, ops, sel, a, b):
        """(kind, signed) for a cmp+select max/min, or None. The select must pick between the
        two block args, and its condition must be a cmp of exactly those block args — OR an
        ``ori(cmp, isnan)`` mask (inductor's NaN-propagating triton_helpers.maximum/minimum),
        in which case the kind is the NaN-propagating ``nanmax``/``nanmin``."""
        t, f = sel.operand_ids[1], sel.operand_ids[2]
        if {t, f} != {a, b}:
            return None  # picks value-vs-mask / a derived value -> not a plain max/min
        cond = sel.operand_ids[0]
        by_id = {o.id: o for o in ops}
        cond_op = by_id.get(cond)
        cmp = None
        isnan_value = None
        if cond_op is not None and cond_op.op in ("arith.cmpf", "arith.cmpi"):
            cmp = cond_op
        elif cond_op is not None and cond_op.op == "arith.ori" and len(cond_op.operand_ids or []) == 2:
            # NaN-propagating max/min: mask = (a CMP b) | (x != x), select(mask, a, b),
            # where `x != x` (cmpf une, same operand twice) is isnan(x). Triton inductor's
            # triton_helpers.maximum/minimum emit exactly this.
            parts = [by_id.get(i) for i in cond_op.operand_ids]
            cmp = next(
                (
                    o
                    for o in parts
                    if o is not None and o.op in ("arith.cmpf", "arith.cmpi") and set(o.operand_ids or []) == {a, b}
                ),
                None,
            )
            isnan = next(
                (
                    o
                    for o in parts
                    if o is not None
                    and o.op == "arith.cmpf"
                    and len(o.operand_ids or []) >= 2
                    and o.operand_ids[0] == o.operand_ids[1]
                    and o.operand_ids[0] in (a, b)
                    and "une" in (o.attrs.get("predicate_name", "") or "")
                ),
                None,
            )
            if cmp is None or isnan is None:
                return None
            isnan_value = isnan.operand_ids[0]
        if cmp is None or len(cmp.operand_ids or []) < 2:
            return None
        lhs, rhs = cmp.operand_ids[0], cmp.operand_ids[1]
        if {lhs, rhs} != {a, b}:
            return None
        pred = cmp.attrs.get("predicate_name", "") or ""
        is_gt = "gt" in pred or "ge" in pred
        is_lt = "lt" in pred or "le" in pred
        signed = not (cmp.op == "arith.cmpi" and pred.startswith("u"))
        if t == lhs:
            kind = "max" if is_gt else "min" if is_lt else None
        elif t == rhs:
            kind = "min" if is_gt else "max" if is_lt else None
        else:
            kind = None
        if kind and cmp.op == "arith.cmpf":
            # A float cmp/select recurrence is ordered even when it computes a
            # finite max/min or explicitly propagates NaN: ties and unordered
            # comparisons select an exact operand.  Symmetric fmax/fmin or an
            # any-NaN side channel loses that operand choice and therefore
            # changes signed-zero and NaN/payload semantics. Carry the exact
            # predicate/selection mapping to the ordered reducer. Direct
            # arith.maximumf/minimumf was classified separately above.
            if pred not in ("ogt", "oge", "olt", "ole") or lhs != a or rhs != b:
                return None
            _shape = self._native_shape(ssa.operand_ids[0], op_name="tt.reduce")
            if _shape is None or len(_shape) not in (1, 2):
                return None
            true_side = "a" if t == a else "b"
            nan_side = "none" if isnan_value is None else ("a" if isnan_value == a else "b")
            kind = f"ordered_{pred}_{true_side}_{nan_side}"
        return (kind, signed) if kind else None

    def _get_reduce_combine_info(self, ssa):
        """Extract combine op and identity from a tt.reduce's body region.

        Returns (combine_op, identity_literal) where combine_op is one of
        'sum', 'max', 'min' and identity_literal is the MSL identity value.
        A custom combine this multipass path cannot honor REFUSES (never defaults
        to sum) — the prime directive.
        """
        _res = self.classify_reduce_combine(ssa)
        # The multipass path supports the canonical kinds plus the structurally proved
        # ordered float cmp/select family.  Ordered operands are staged and folded in
        # logical source order by _lower_multipass_reduction; they must never use an
        # invented identity here.
        if _res is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "tl.reduce on the multipass reduction path supports only canonical "
                "sum / max / min / and / or / xor; a custom or NaN-propagating combine is "
                "refused rather than silently mis-computed."
            )
        combine_op = _res[0]
        identities = {
            "sum": "-0.0f",
            "prod": "1.0f",
            "max": "-INFINITY",
            "min": "INFINITY",
            "and": "(~0)",
            "or": "0",
            "xor": "0",
            "nanmax": "-INFINITY",
            "nanmin": "INFINITY",
        }
        return combine_op, identities.get(combine_op, "0.0f")

    @staticmethod
    def _ordered_multipass_shape_is_full(shape, axis, total):
        """True only when one ordered result covers the complete staged tile."""
        shape = tuple(shape or ())
        if axis is None:
            axis = 0 if len(shape) == 1 else None
        if isinstance(axis, int) and axis < 0:
            axis += len(shape)
        if not isinstance(axis, int) or not 0 <= axis < len(shape):
            return False
        nonreduce_extent = 1
        for dim, extent in enumerate(shape):
            if dim != axis:
                nonreduce_extent *= extent
        return shape[axis] == total and nonreduce_extent == 1

    def _reduce_identity_combine(self, combine_op, msl_type):
        """Return (identity, combine_expr) for an `acc`/`val` sequential reduce.

        Branches on the actual MSL accumulator type — float, long (i64),
        ulong (u64), or int (any narrower signed/unsigned int reduced in
        i32). The 64-bit paths must NOT use the 32-bit INT_MIN/INT_MAX
        identities or the float fmax/fmin combine: those would silently
        truncate to 32 bits / pick a float overload on a long argument.

        max identities: float→(-INFINITY); long→LONG_MIN; ulong→0; int→INT_MIN
        min identities: float→INFINITY; long→LONG_MAX; ulong→ULONG_MAX;
                        int→INT_MAX
        sum identity is 0 (-0.0f for float, preserving either source zero sign).
        LONG_MIN/LONG_MAX/ULONG_MAX are
        provided by metal_stdlib (same as the multipass-accumulator path).
        """
        is_float = msl_type == "float"
        if combine_op == "sum":
            identity = "-0.0f" if is_float else "0"
            combine_expr = "acc + val"
        elif combine_op == "prod":
            identity = "1.0f" if is_float else "1"
            combine_expr = "acc * val"
        elif combine_op == "max":
            identity = {
                "float": "(-INFINITY)",
                "long": "LONG_MIN",
                "ulong": "0",
                "uint": "0",
                "int": "INT_MIN",
            }.get(msl_type, "INT_MIN")
            combine_expr = "fmax(acc, val)" if is_float else "max(acc, val)"
        elif combine_op == "min":
            identity = {
                "float": "INFINITY",
                "long": "LONG_MAX",
                "ulong": "ULONG_MAX",
                "uint": "UINT_MAX",
                "int": "INT_MAX",
            }.get(msl_type, "INT_MAX")
            combine_expr = "fmin(acc, val)" if is_float else "min(acc, val)"
        elif combine_op in ("nanmax", "nanmin"):
            # NaN-PROPAGATING max/min (float-only; inductor maximum/minimum). Unlike
            # fmax/fmin (NaN-quiet), a NaN in EITHER operand must propagate: return acc when
            # acc beats val OR acc is NaN, else val (a val=NaN flows through the else since
            # acc>val / acc<val is false for a NaN val). Identity -INF/INF can't mask a NaN.
            cmp = ">" if combine_op == "nanmax" else "<"
            identity = "(-INFINITY)" if combine_op == "nanmax" else "INFINITY"
            combine_expr = f"((acc {cmp} val) || (acc != acc)) ? acc : val"
        elif combine_op == "xor":
            identity = "0"
            combine_expr = "acc ^ val"
        elif combine_op == "and":
            identity = "~0"  # all-ones: AND identity
            combine_expr = "acc & val"
        elif combine_op == "or":
            identity = "0"  # OR identity
            combine_expr = "acc | val"
        else:
            identity = "0.0f" if is_float else "0"
            combine_expr = "acc + val"
        return identity, combine_expr

    def _multipass_scalar_load_ids(self):
        """Prove entry-scalar loads that can live outside the element loops.

        A non-tensor load was previously treated as per-element unconditionally.
        Its scalar consumer was hoisted, however, and read a name whose defining
        loop had closed. Only move ordinary loads through a read-only prefix;
        stores, atomics, assertions, calls, control flow and volatile reads end
        the proof. Pointer/mask/other dependencies must themselves be scalar and
        available outside the loops. This is not general load speculation.
        """
        available = {arg.id for arg in self.graph.args if self._native_shape(arg.id, op_name="tt.load") == ()}
        pure_tt = {
            "tt.make_range",
            "tt.get_program_id",
            "tt.get_num_programs",
            "tt.splat",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
            "tt.trans",
            "tt.addptr",
            "ttg.convert_layout",
        }

        def pure(op):
            if op.op == "tt.load":
                return op.attrs.get("isVolatile") is False
            if op.op == "tt.reduce":
                return all(child.op == "tt.reduce.return" or pure(child) for child in op.region_ops or [])
            if op.region_ops or op.else_ops:
                return False
            return op.op.startswith(("arith.", "math.")) or op.op in pure_tt

        loads = set()
        for op in self.graph.ops:
            if not pure(op):
                break
            ids = self._native_result_ids_for_op(op)
            if (
                len(ids) != 1
                or self._native_shape(ids[0], op_name=op.op) != ()
                or not all(dep in available for dep in op.operand_ids)
            ):
                continue
            if op.op == "tt.load":
                loads.add(op.id)
                available.add(ids[0])
            elif self._is_scalar_op(op):
                available.add(ids[0])
        return loads

    def _lower_multipass_reduction(self, block_size):
        """Emit multi-pass reduction: per-element loops separated by reductions.

        When a kernel has both per-element ops and reductions, we cannot wrap
        everything in a single loop (threadgroup_barrier inside a loop is UB).
        Instead, split into phases:
        - Non-reduce phases: wrap per-element ops in a for-loop with local
          accumulation for the next reduce
        - Reduce phases: emit SIMD + shared memory reduction outside any loop

        Each phase loop re-computes per-element values from scratch (re-loads
        data) because per-element variables from earlier loops are out of scope.
        Scalar ops are hoisted before their phase's loop so that their variables
        remain in scope for later phases.
        """
        total = self._total_elements
        phases = self._split_ops_by_reductions()
        scalar_load_ids = self._multipass_scalar_load_ids()

        def is_scalar_op(op):
            return op.id in scalar_load_ids or self._is_scalar_op(op)

        # Collect all reduce result SSA IDs (scalars available across phases)
        reduce_result_ids = set()
        for ops, is_reduce in phases:
            if is_reduce:
                for ssa in ops:
                    reduce_result_ids.add(ssa.id)
                    if ssa.result_ids:
                        for rid in ssa.result_ids:
                            reduce_result_ids.add(rid)

        # Also include function arg IDs as "always available" scalars
        arg_ids = {a.id for a in self.graph.args}

        # Track all preceding non-reduce ops for dependency resolution
        all_preceding_ops = []
        # Track which scalar ops have already been lowered (by SSA id)
        lowered_scalar_ids = set()

        for phase_idx, (phase_ops, is_reduce) in enumerate(phases):
            if is_reduce:
                # Lower the reduce op outside any loop.
                # The reduce's input is already set to the accumulator variable
                # (overridden in self.env by the preceding phase's accumulation).
                for ssa in phase_ops:
                    self._lower_op(ssa)
                continue

            # Determine if the next phase is a reduce (need accumulation)
            next_reduce = None
            if phase_idx + 1 < len(phases) and phases[phase_idx + 1][1]:
                next_reduce = phases[phase_idx + 1][0][0]
            next_reduce_native = self._prove_reduce_native_contract(next_reduce) if next_reduce is not None else None

            # Separate scalar ops (hoist before loop) from tensor ops (inside loop)
            scalar_ops = [op for op in phase_ops if is_scalar_op(op)]
            tensor_ops = [op for op in phase_ops if not is_scalar_op(op)]

            # A SCALAR (per-program) atomic / store consuming a reduce RESULT must
            # execute EXACTLY ONCE — NOT once per wrap-loop iteration. Emitting it
            # inside the per-element ``for (_loop_e ...)`` loop made thread 0's
            # ``lid==0``-guarded RMW fire once per iteration, over-counting by
            # BLOCK/num_threads (the audit's B3: tl.atomic_add(out, tl.sum(x)) over
            # 1024 ones returned 8192 with 128 threads). These ops are NOT per-element
            # (the value is a SCALAR reduce result, and the pointer is a scalar !tt.ptr),
            # so split them out of ``tensor_ops`` and emit them ONCE after the loop.
            # A PER-ELEMENT store/atomic (the value is a tensor — e.g. the softmax
            # ``result`` written one element per lane) stays in ``tensor_ops`` and is
            # correctly wrapped. NOTE: a tt.store op's own ``is_tensor`` is False (it
            # produces no value), so the discriminator is the VALUE OPERAND's shape, not
            # the op's result flag (the bug that mis-hoisted the softmax tensor store).
            def _is_scalar_terminal(op):
                if op.op not in ("tt.atomic_rmw", "tt.atomic_cas", "tt.store"):
                    return False
                if len(op.operand_ids or []) < 2:
                    return False
                val_id = op.operand_ids[1]
                vshape = self.env_shapes.get(val_id)
                if vshape is not None:
                    # A 0-D / 1-element value is scalar; a multi-element tensor is
                    # per-lane.
                    nel = 1
                    for d in vshape:
                        nel *= d
                    return nel <= 1

                # No recorded shape: fall back to the producing op's tensor flag
                # (a reduce result / scalar arithmetic is not a tensor). Index ALL
                # ops (including scf.for/if region bodies), not just the top level, so
                # a value produced inside a region is still classified — otherwise it
                # defaults to per-element and the op stays wrapped (the B3 backstop
                # would then refuse rather than over-count, so this is safe either way,
                # but indexing regions classifies it correctly).
                def _index_all(ops, acc):
                    for s in ops:
                        acc[s.id] = s
                        if getattr(s, "region_ops", None):
                            _index_all(s.region_ops, acc)
                        if getattr(s, "else_ops", None):
                            _index_all(s.else_ops, acc)

                _all_by_id = {}
                _index_all(self.graph.ops, _all_by_id)
                vop = _all_by_id.get(val_id)
                return bool(vop is not None and not vop.is_tensor)

            scalar_terminal_ops = [op for op in tensor_ops if _is_scalar_terminal(op)]
            if scalar_terminal_ops:
                _term_ids = {op.id for op in scalar_terminal_ops}
                tensor_ops = [op for op in tensor_ops if op.id not in _term_ids]

            # Check if this phase has any tensor ops that need a loop
            has_tensor_ops = len(tensor_ops) > 0

            # Emit scalar ops BEFORE the loop (they stay in function scope)
            for ssa in scalar_ops:
                if ssa.id not in lowered_scalar_ids:
                    self._lower_op(ssa)
                    lowered_scalar_ids.add(ssa.id)

            if not has_tensor_ops and next_reduce is None:
                # Pure scalar phase — no per-element loop needed. Emit any scalar
                # terminal write (atomic/store of a reduce result) exactly once.
                for ssa in scalar_terminal_ops:
                    self._lower_op(ssa)
                all_preceding_ops.extend(phase_ops)
                continue

            # Determine which earlier ops need to be re-emitted in this loop
            # for their per-element values to be available
            replay_ops = self._collect_tensor_deps(
                tensor_ops, all_preceding_ops, reduce_result_ids | arg_ids | lowered_scalar_ids
            )

            # Also hoist scalar deps from replay_ops before the loop
            replay_scalar = [op for op in replay_ops if is_scalar_op(op)]
            replay_tensor = [op for op in replay_ops if not is_scalar_op(op)]
            for ssa in replay_scalar:
                if ssa.id not in lowered_scalar_ids:
                    self._lower_op(ssa)
                    lowered_scalar_ids.add(ssa.id)

            # If this phase precedes a reduce, declare the accumulator
            acc_var = None
            if next_reduce:
                combine_op, identity = self._get_reduce_combine_info(next_reduce)
                acc_var = f"_local_acc_{self._shared_counter}"
                # The accumulator must match the reduce operand's immutable native
                # representation. Mutable replay caches and the result-zero summary
                # cannot authorize a type at this boundary.
                reduce_input_dtype = next_reduce_native["slot_dtypes"][0]
                is_int_reduce = not (reduce_input_dtype.startswith("fp") or reduce_input_dtype.startswith("bf"))
                is_i64_reduce = reduce_input_dtype in ("i64", "u64", "ui64")
                # unsigned 32-bit max/min compares UNSIGNED; the final cross-thread simd
                # reduce casts to reduce_ty in threadgroup_reduce.
                _unsigned = combine_op in ("max", "min") and self._reduce_is_unsigned_minmax(next_reduce.region_ops)
                # uint64 is the SIGNLESS i64 in Triton, so an unsigned 64-bit max/min arrives
                # as dtype 'i64' + _unsigned — treat it as u64 for the ulong/ULONG_MAX path,
                # not just the explicit u64/ui64 names (else signed LONG_MIN/MAX identities
                # mismatch the ulong accumulator and silently mis-reduce above 2^63).
                is_u64_reduce = reduce_input_dtype in ("u64", "ui64") or (is_i64_reduce and _unsigned)
                acc_msl_type, _ = self._reduce_acc_msl_type(reduce_input_dtype, _unsigned)
                ordered_multipass = combine_op.startswith("ordered_")
                ordered_stage = None
                if ordered_multipass:
                    # The ordered comparator is operand-selecting and has no padding
                    # identity.  A full adjacent tree over the logical source sequence
                    # preserves operand order, NaN payload choice and signed-zero ties.
                    # Restrict this bounded recovery to a complete power-of-two tile;
                    # non-power-of-two tails need a separate value+validity design.
                    _full_logical_reduce = self._ordered_multipass_shape_is_full(
                        next_reduce_native.get("shape"),
                        next_reduce_native.get("axis"),
                        total,
                    )
                    if (
                        acc_msl_type != "float"
                        or total < block_size
                        or total <= 0
                        or total & (total - 1)
                        or not _full_logical_reduce
                    ):
                        from triton_msl.errors import MetalNonRecoverableError

                        raise MetalNonRecoverableError(
                            "ordered float cmp/select on the multipass reduction path "
                            "requires a complete power-of-two full logical reduction "
                            "with at least one value per thread and only singleton "
                            "non-reduction dimensions; refusing rather than flattening "
                            "an axis reduction, padding, or reassociating a tail.",
                            op_name="tt.reduce",
                        )
                    ordered_stage = f"shared_ordered_multipass_{self._shared_counter}"
                    self._shared_counter += 1
                    self.kb.declare_threadgroup_array(ordered_stage, dtype="fp32", size=total)
                # bitwise (and/or/xor) identities are width-independent: and = all-ones,
                # or/xor = 0; product identity = 1.
                _bitwise_ident = {"and": "(~0)", "or": "0", "xor": "0", "prod": "1"}
                if is_i64_reduce:
                    # 64-bit identities (LONG_MIN/MAX); ulong min identity is 0.
                    i64_identities = (
                        {"sum": "0", "max": "0", "min": "ULONG_MAX"}
                        if is_u64_reduce
                        else {"sum": "0", "max": "LONG_MIN", "min": "LONG_MAX"}
                    )
                    i64_identities.update(_bitwise_ident)
                    identity = i64_identities.get(combine_op, "0")
                elif is_int_reduce:
                    if _unsigned:
                        identity = "0" if combine_op == "max" else "UINT_MAX"
                    else:
                        identity = {"sum": "0", "max": "INT_MIN", "min": "INT_MAX", **_bitwise_ident}.get(
                            combine_op, "0"
                        )
                # else float: identity preserved from above. Ordered reductions do
                # not declare an identity accumulator: every source element is staged.
                if not ordered_multipass:
                    self.kb.raw_line(f"    {acc_msl_type} {acc_var} = {identity};")

            # Open the per-element loop
            self._needs_wrapping = True
            self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {block_size}u) {{")

            # Re-emit tensor dependency ops from earlier phases
            for ssa in replay_tensor:
                self._lower_op(ssa)

            # Emit this phase's tensor ops inside the loop
            for ssa in tensor_ops:
                self._lower_op(ssa)

            # Accumulate into the local variable for the next reduce
            if next_reduce and acc_var:
                reduce_input_id = next_reduce.operand_ids[0]
                # DOUBLE-COUNT GUARD (re-review 2026-08-25 F2): this phase loop strides
                # ``_loop_e`` over the KERNEL's _total_elements, but a reduce input SMALLER
                # than that wraps (make_range lowers to ``_loop_e % range_size``) and is
                # accumulated MULTIPLE times — e.g. a 256-wide tl.sum inside a 512-element
                # kernel summed every element exactly TWICE, silently. No correct kernel
                # can currently hit this (the result was always wrong); refuse loudly until
                # the phase loop runs per-input extents.
                _rin_shape = next_reduce_native["shape"]
                _rin_total = 1
                for _d in _rin_shape or ():
                    _rin_total *= _d
                if _rin_shape and _rin_total != total:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        f"a {_rin_total}-element reduction inside a {total}-element "
                        "kernel would be accumulated with wrapped (repeated) elements "
                        "by the multipass phase loop — the result would silently "
                        "over-count. Refusing. Make the reduced tensor span the "
                        "kernel's full element count, or reduce in a separate kernel.",
                        op_name="tt.reduce",
                    )
                input_var = self._lookup(reduce_input_id)
                # Cast input to accumulator type to avoid Metal ambiguity
                cast_input = f"({acc_msl_type}){input_var}"
                if ordered_multipass:
                    self.kb.raw_line(f"        {ordered_stage}[_loop_e] = {cast_input};")
                elif combine_op == "sum":
                    self.kb.raw_line(f"        {acc_var} += {cast_input};")
                elif combine_op == "prod":
                    self.kb.raw_line(f"        {acc_var} *= {cast_input};")
                elif combine_op == "max":
                    self.kb.raw_line(f"        {acc_var} = max({acc_var}, {cast_input});")
                elif combine_op == "min":
                    self.kb.raw_line(f"        {acc_var} = min({acc_var}, {cast_input});")
                elif combine_op == "and":
                    self.kb.raw_line(f"        {acc_var} &= {cast_input};")
                elif combine_op == "or":
                    self.kb.raw_line(f"        {acc_var} |= {cast_input};")
                elif combine_op == "xor":
                    self.kb.raw_line(f"        {acc_var} ^= {cast_input};")
                elif combine_op in ("nanmax", "nanmin"):
                    # NaN-PROPAGATING max/min: a NaN in acc OR the element must propagate
                    # (max()/min() are NaN-quiet). The cross-thread simd step propagates too.
                    _c = ">" if combine_op == "nanmax" else "<"
                    self.kb.raw_line(
                        f"        {acc_var} = (({acc_var} {_c} {cast_input}) || "
                        f"({acc_var} != {acc_var})) ? {acc_var} : {cast_input};"
                    )

            # Close the loop
            self.kb.raw_line(f"    }}")
            self._needs_wrapping = False

            if next_reduce and ordered_multipass:
                # Each round combines adjacent, equal-sized contiguous source
                # subranges. All writes in a round are disjoint; the barrier publishes
                # them before the next round. This is the same leaf order as the
                # single-pass ordered tree, without strided per-thread partials.
                self.kb.barrier("threadgroup")
                _span = 1
                while _span < total:
                    _pairs = total // (2 * _span)
                    self.kb.raw_line(f"    for (uint _ord_mp = lid; _ord_mp < {_pairs}u; _ord_mp += {block_size}u) {{")
                    self.kb.raw_line(f"        uint _ord_left = _ord_mp * {2 * _span}u;")
                    self.kb.raw_line(
                        f"        {ordered_stage}[_ord_left] = "
                        + self.kb.ordered_combine_expr(
                            combine_op,
                            f"{ordered_stage}[_ord_left]",
                            f"{ordered_stage}[_ord_left + {_span}u]",
                        )
                        + ";"
                    )
                    self.kb.raw_line("    }")
                    self.kb.barrier("threadgroup")
                    _span *= 2
                self.kb.raw_line(f"    {acc_msl_type} {acc_var} = {ordered_stage}[0];")

            # Emit any scalar terminal write (atomic/store of a reduce result)
            # ONCE, AFTER the per-element loop — never inside it (B3 over-count fix).
            for ssa in scalar_terminal_ops:
                self._lower_op(ssa)

            # Override the reduce's input to point to the accumulator, and
            # record the accumulator's dtype so the downstream reduce dispatch
            # picks the matching (e.g. 64-bit) path even when the original
            # operand's type was dropped by a preceding reshape.
            if next_reduce and acc_var:
                reduce_input_id = next_reduce.operand_ids[0]

                # GUARD (re-audit silent-wrong #3): if this exact input SSA feeds MORE THAN
                # ONE tt.reduce (two INDEPENDENT reductions of the SAME loaded tile, e.g.
                # sum(x) and max(x)), the env rebind below aliases the shared input to THIS
                # phase's accumulator, so the other reduce would silently reduce over this
                # accumulator (wrong whenever BLOCK > num_threads). The multipass path cannot
                # model that — refuse loudly rather than silently mis-compute. (Softmax /
                # layernorm are unaffected: their 2nd reduce consumes a DIFFERENT value, not
                # the same loaded tile.)
                def _all_reduce_ops(ops):
                    for s in ops:
                        if s.op == "tt.reduce":
                            yield s
                        if s.region_ops:
                            yield from _all_reduce_ops(s.region_ops)
                        if s.else_ops:
                            yield from _all_reduce_ops(s.else_ops)

                _sharers = [
                    r for r in _all_reduce_ops(self.graph.ops) if r.operand_ids and r.operand_ids[0] == reduce_input_id
                ]
                if len(_sharers) > 1:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "two or more reductions of the SAME loaded tile (e.g. tl.sum(x) and "
                        "tl.max(x) of the same x) are not supported by the multipass reduce "
                        "lowering: the second silently reduces over the first's accumulator. "
                        "Load the tile separately for each reduction.",
                        op_name="tt.reduce",
                    )
                self.env[reduce_input_id] = acc_var
                if ordered_multipass:
                    self.env_shapes[reduce_input_id] = None
                    if not hasattr(self, "_ordered_multipass_inputs"):
                        self._ordered_multipass_inputs = set()
                    self._ordered_multipass_inputs.add(reduce_input_id)
                _acc_dtype = {"long": "i64", "ulong": "u64", "int": "i32", "float": "fp32"}.get(acc_msl_type)
                if _acc_dtype is not None:
                    self.env_types[reduce_input_id] = _acc_dtype

            # Add this phase's ops to the preceding ops for future phases
            all_preceding_ops.extend(phase_ops)

    def _reduce_result_reaches_1d_store(self, reduce_id, stop_at_convert=True):
        """True iff the 2-D→1-D reduce ``reduce_id`` reaches a 1-D ``tt.store`` as the
        stored VALUE — following forward uses and crossing ``scf.for`` loop-carries —
        without first being re-broadcast to ≥2-D.

        ``stop_at_convert`` selects between the walk's two callers (2026-08-29,
        trifast #6b): the loop-carry silent-wrong GUARD passes True — a value that
        reaches the store through a ``ttg.convert_layout`` is correct-or-refuse at the
        convert itself, so the guard must not double-refuse it. The quant-GEMV
        descriptor MATCHERS pass False — they ask the structural question "does this
        reduce feed a 1-D store at all", for which the convert is a passthrough.

        A 2-D→1-D axis reduce produces its result in the row-broadcast layout (thread
        ``lid`` holds row ``lid/N``). That layout is correct for a 2-D consumer but not
        for a 1-D store, which assumes one-row-per-thread (``lid`` → row ``lid``):

        - **Hazard (GEMV):** ``acc += tl.sum(x, axis=1)`` in a K-loop, then
          ``tl.store(o + offs_n, acc)``. The loop-carried accumulator flows straight
          to a 1-D store → every output row silently collapses to the first (the
          row-broadcast layout is not propagated across the loop-carry).
        - **Safe (online-softmax / FlashAttention):** the carried reduce result (m_i /
          l_i) is re-broadcast to 2-D via ``tt.expand_dims`` (``m[:, None]``) — inside
          the loop OR after it, on the ``scf.for`` result — to rescale/normalize a 2-D
          accumulator, and the 2-D value is what gets stored. The 2-D consumer matches
          the row-broadcast layout, so it is correct.

        We follow forward uses across the loop boundary using the ``scf.for``'s
        ``result_ids`` (positionally matched to the ``scf.yield`` operands). A
        ``tt.expand_dims`` user ends that branch (the value became 2-D → safe). A
        ``tt.store`` whose VALUE operand is the traced value is the hazard. Purely a
        read over the IR graph — no emission state.
        """

        def _iter_all(ops):
            for o in ops:
                yield o
                if getattr(o, "region_ops", None):
                    yield from _iter_all(o.region_ops)
                if getattr(o, "else_ops", None):
                    yield from _iter_all(o.else_ops)

        all_ops = list(_iter_all(self.graph.ops))
        uses = {}
        for op in all_ops:
            for oid in op.operand_ids or []:
                uses.setdefault(oid, []).append(op)
        # scf.yield -> its enclosing scf.for (to cross the loop-carry via result_ids).
        yield_to_for = {}
        for op in all_ops:
            if op.op == "scf.for" and op.region_ops:
                for b in op.region_ops:
                    if b.op == "scf.yield":
                        yield_to_for[b.id] = op

        # Each stack entry carries whether the path has crossed a loop-carry yet.
        # The hazard is specifically the LOOP-CARRIED store: a single-tile reduce that
        # feeds a 1-D store directly (no K-loop) is correct (the store re-reads the row
        # from shared memory), so it must NOT be refused. Only a store reached AFTER
        # crossing an scf.for loop-carry collapses.
        seen = set()
        stack = [(reduce_id, False)]
        while stack:
            vid, crossed = stack.pop()
            if (vid, crossed) in seen:
                continue
            seen.add((vid, crossed))
            for user in uses.get(vid, []):
                if user.op == "tt.expand_dims":
                    continue  # value becomes ≥2-D on this branch → safe, stop
                if user.op == "ttg.convert_layout":
                    if stop_at_convert:
                        # Safe-stop (2026-08-29, trifast #6b): the convert now resolves
                        # the source's thread-to-element mapping across scf.for carries
                        # (declared-layout resolver + carry-aware reduce provenance) and
                        # REFUSES when it cannot prove one — so a 1-D store fed through a
                        # convert is correct-or-refuse there, and refusing here as well
                        # would only forbid the now-working lse pattern.
                        continue
                    stack.append((user.id, crossed))
                    continue
                if user.op == "tt.store":
                    # operand[0] = pointer, operand[1] = stored value. Only a match on
                    # the VALUE operand of a store reached THROUGH a loop-carry is the
                    # hazard (a pointer/index match is address arithmetic; a no-loop
                    # store is correct).
                    if crossed and len(user.operand_ids or []) >= 2 and user.operand_ids[1] == vid:
                        return True
                    continue
                if user.op == "scf.yield":
                    forop = yield_to_for.get(user.id)
                    if forop is not None:
                        # A single-result scf.for has result_ids == None (the walker
                        # collapses len==1), and the result maps to the op's own id.
                        rids = forop.result_ids if getattr(forop, "result_ids", None) else [forop.id]
                        for i, oid in enumerate(user.operand_ids or []):
                            if oid == vid and i < len(rids):
                                stack.append((rids[i], True))  # crossed a loop-carry
                    continue
                # Generic op: follow its result(s) forward, preserving `crossed`.
                if getattr(user, "result_ids", None):
                    stack.extend((rid, crossed) for rid in user.result_ids)
                else:
                    stack.append((user.id, crossed))
        return False

    def _lower_reduce(self, ssa: SSAValue):
        """tt.reduce → SIMD + threadgroup shared memory reduction.

        For 1D: standard full reduction using SIMD intrinsics + shared memory.
        For 2D with axis: reduce along one dimension, keeping the other.
            axis=1 on (M, N): reduce N columns per row → (M,) result
            axis=0 on (M, N): reduce M rows per column → (N,) result
        Multi-value reduces (argmax/argmin) are dispatched to a specialized handler.
        """
        if not ssa.operand_ids:
            return
        if not getattr(self, "_reduce_operand_gathered", False):
            # A tile whose rows are spread wider than its columns is gathered onto the
            # lanes every path below reads (single-value reductions; others refuse).
            _tile = ssa.operand_ids[0]
            _previous = self._compact_spread_tile_operand(_tile, "tt.reduce", gather=len(ssa.operand_ids) == 1)
            if _previous is not None:
                self._reduce_operand_gathered = True
                try:
                    return self._lower_reduce(ssa)
                finally:
                    self._reduce_operand_gathered = False
                    self.env[_tile] = _previous

        # Detect multi-value reduce (argmax/argmin): 2+ inputs, 2+ results
        if len(ssa.operand_ids) >= 2 and ssa.result_ids and len(ssa.result_ids) >= 2:
            self._lower_reduce_multi_value(ssa)
            return

        native = self._prove_reduce_native_contract(ssa)
        input_var = self._lookup(ssa.operand_ids[0])
        axis = native["axis"]

        # Determine the combine via the structural classifier (single source of truth).
        # The 1-D path supports sum/max/min (signed+unsigned) + and/or/xor; a custom /
        # NaN-propagating / unrecognised combine returns None -> refuse loudly.
        _res = self.classify_reduce_combine(ssa)
        if _res is None:
            # None covers a custom combine AND an EMPTY region — a block-arg pick
            # (first/last/identity, `return a`) leaves region_ops empty after the
            # tt.reduce.return terminator is stripped, and must REFUSE (not silently sum,
            # the prior historical default — matches the multipass path). A real
            # single-input reduce always carries a combine with >=1 op.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "tl.reduce with a combine that is not a canonical sum / max / min / and / "
                "or / xor reduction is not supported — a product, max-by-magnitude, NaN-"
                "propagating max/min, a + relu(b), a first/last/identity pick, or other "
                "custom combine is refused rather than silently mis-computed."
            )
        combine_op, _signed = _res
        ordered_cmp = combine_op.startswith("ordered_")
        has_unsigned_minmax = combine_op in ("max", "min") and not _signed

        # The input and result representation/shape are native per-value facts.
        # Multipass may rebind the input's runtime expression to a local
        # accumulator, but it cannot change the logical operand contract.
        input_dtype = native["slot_dtypes"][0]
        is_int_reduce = not (input_dtype.startswith("fp") or input_dtype.startswith("bf"))
        is_i64 = input_dtype in ("i64", "u64", "ui64")
        is_u64 = input_dtype in ("u64", "ui64")
        msl_type, shared_dtype = self._reduce_acc_msl_type(
            input_dtype, has_unsigned_minmax and combine_op in ("max", "min")
        )

        # Check if this is a 2D axis-specific reduction
        input_shape = native["shape"]

        # Ordered float cmp/select has no symmetric intrinsic or generally safe
        # padding identity. Exact one-value-per-lane 1-D and fully staged rank-2
        # axis reductions are handled here. A separately proved top-level
        # multipass full reduction arrives already folded in logical source order;
        # array-fold, nested, tail-padded, or other N-D forms still refuse.
        if ordered_cmp:
            _ordered_multipass = ssa.operand_ids[0] in getattr(self, "_ordered_multipass_inputs", set())
            _rn = input_shape[0] if input_shape and len(input_shape) == 1 else None
            _rank2 = bool(input_shape and len(input_shape) == 2 and self._is_2d)
            _total2 = input_shape[0] * input_shape[1] if _rank2 else None
            _exact_1d = _rn == self.kb.block_size and (
                self.kb.block_size == 16 or (self.kb.block_size >= 32 and self.kb.block_size % 32 == 0)
            )
            _staged_2d = _rank2 and _total2 <= self.kb.block_size
            if (
                self.env_array.get(ssa.operand_ids[0]) is not None
                or self._control_flow_depth > 0
                or not (_exact_1d or _staged_2d or _ordered_multipass)
            ):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "ordered float cmp/select reduction is supported only for an exact "
                    "16-or-more-value 1-D tile or a fully staged rank-2 axis tile, with no "
                    "tail padding, nesting, unproved multipass, or register-array fold; this shape "
                    "is refused rather than changing "
                    "NaN, payload, or signed-zero operand-selection semantics.",
                    op_name="tt.reduce",
                )
            if _ordered_multipass:
                input_shape = None

        # Phase 4e: MEPT array operand. Fold this thread's register array to
        # a scalar partial with the combine op, then run the existing 1-D
        # cross-thread reduce on that partial. Only the 1-D full-reduce case
        # is handled today — MEPT activation never routes multi-dim / axis
        # reduces here (the prescan requires a 1-D reduce for MEPT-reduce
        # eligibility), so folding to 1-D and skipping the multi-dim
        # dispatch is correct.
        mept_arr = self.env_array.get(ssa.operand_ids[0]) if getattr(self, "mept_enabled", False) else None
        if mept_arr is not None:
            arr_name, n_arr, _arr_ty = mept_arr
            input_var = self._mept_reduce_fold(arr_name, n_arr, combine_op, msl_type)
            input_shape = None  # already folded to one element per thread

        if input_shape and len(input_shape) == 3:
            self._lower_reduce_3d(
                ssa,
                input_var,
                axis,
                combine_op,
                msl_type,
                shared_dtype,
                input_shape,
                native["result_facts"][0],
            )
            return

        # N-D axis-specific reduce (n >= 4). Used by e.g. tl.sort's bitonic
        # decomposition, which reshapes to (2,)*n and reduces along a specific
        # axis per compare-and-swap step.
        if input_shape and len(input_shape) >= 4:
            self._lower_reduce_nd(
                ssa,
                input_var,
                axis,
                combine_op,
                msl_type,
                shared_dtype,
                input_shape,
                native["result_facts"][0],
            )
            return

        if self._is_2d and input_shape and len(input_shape) >= 2:
            # For triton_per_* kernels with shape (1, N) or (XBLOCK, R_BLOCK)
            # where dim_0 == 1 and axis == 1, this is really a 1D reduction
            # along the reduction dimension.  Use the efficient SIMD path,
            # not the slow sequential shared memory path.
            # ... unless the operand carries a recorded broadcast index (a tile reshaped
            # and reduced along a third axis holds each element on several lanes): the SIMD
            # path would take every lane, so the staged path, which writes each element to
            # its own slot, is used instead.
            if input_shape[0] == 1 and axis == 1 and ssa.operand_ids[0] not in self._bcast_layout:
                pass  # Fall through to 1D SIMD reduction below
            else:
                # A 2-D axis reduce whose tile exceeds the threadgroup under-covers the
                # same way the 1-D in-loop case does (the Stage-B guard below is 1-D
                # scoped). _lower_reduce_2d then silently reduces only the first
                # block_size lanes (re-audit #11: 2-D tl.sum(axis=1) inside scf.for, and
                # the inductor layernorm-backward weight-grad). Refuse loudly when the
                # whole tile can't be covered one-element-per-thread.
                _tot2d = 1
                for _d in input_shape:
                    _tot2d *= _d
                if _tot2d > self.kb.block_size:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        f"Refusing 2-D reduction: a {tuple(input_shape)} tile exceeds "
                        f"the {self.kb.block_size}-thread threadgroup, so the cross-lane "
                        f"reduce would cover only the first {self.kb.block_size} lanes "
                        f"(silent-wrong). Use BLOCK <= num_threads.",
                        op_name="tt.reduce",
                    )
                # LAYOUT guard: a 2-D→1-D axis reduce whose result is accumulated
                # across a loop (e.g. `acc += tl.sum(x, axis=1)` inside a K-loop)
                # produces the result in the row-broadcast layout (thread lid → row
                # lid/N), but the loop-carried accumulator + 1-D store assume row lid,
                # silently collapsing every output row to the first. The tile FITS
                # (the coverage guard above passes), so this is a distinct silent-wrong;
                # the fix needs loop-carry broadcast-layout propagation. Refuse.
                if self._reduce_result_reaches_1d_store(ssa.id):
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "Refusing a 2-D axis reduce whose result is accumulated across a "
                        "loop, e.g. `acc += tl.sum(x[None,:]*w, axis=1)` inside "
                        "`for k in range(0, K, BK)`: the per-row reduce result is laid out "
                        "one-row-per-simdgroup, but the loop-carried accumulator and 1-D "
                        "store assume one-row-per-thread, which silently collapses every "
                        "output row to the first. Reduce the whole axis in a single tile "
                        "(no K-loop, BLOCK_K == K), or express the row-wise matmul as "
                        "tl.dot(a, w). Correct-or-refuse: refused, not mis-computed.",
                        op_name="tt.reduce",
                    )
                self._lower_reduce_2d(
                    ssa,
                    input_var,
                    axis,
                    combine_op,
                    msl_type,
                    shared_dtype,
                    input_shape,
                    input_dtype,
                )
                return

        # Stage B (in-loop reduction coverage): a 1-D full reduce whose tile
        # exceeds the threadgroup (block_size > num_threads) is only correct
        # when its per-thread input already covers the whole tile — either via
        # the register-array fold (_mept_reduce_fold, applied above, which sets
        # input_shape=None) or the top-level multipass wrap (which rebinds the
        # input to a scalar accumulator before reaching here, depth == 0). An
        # in-loop reduce (inside scf.for/if/while, depth > 0) with a raw block
        # tensor and no array cover would emit a one-element-per-thread cross-
        # lane reduce that SILENTLY sums only the first num_threads elements.
        # Refuse loudly instead of returning a wrong result.
        # Scope: 1-D full reduces only (len(input_shape)==1). The (1,N) axis==1
        # fall-through and the ND reduce paths share the same under-coverage gap
        # but are out of Stage B's 1-D scope (tracked by Task 2 corpus measure).
        if (
            mept_arr is None
            and self._control_flow_depth > 0
            and input_shape is not None
            and len(input_shape) == 1
            and input_shape[0] > self.kb.block_size
        ):
            # Stage A: try to cover the whole tile by folding each thread's
            # strided share before the cross-thread reduce.
            _acc = self._cover_inloop_reduce(ssa, combine_op, msl_type, input_shape[0])
            if _acc is not None:
                input_var = _acc
                input_shape = None  # folded to one scalar per thread
            else:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"Refusing in-loop reduction: a tile of {input_shape[0]} "
                    f"elements exceeds the {self.kb.block_size}-thread threadgroup "
                    f"and is not register-array-covered, so a cross-lane reduce "
                    f"here would sum only the first {self.kb.block_size} elements "
                    f"(silent-wrong). Use the default register-array path "
                    f"(TRITON_MSL_MEPT unset) or BLOCK <= num_threads."
                )

        # Cast bool (i1) to int before reduction — MSL SIMD intrinsics reject bool
        if input_dtype == "i1" or (isinstance(input_var, str) and input_var in ("true", "false", "1", "0")):
            cast_var = self._next_var("bool_to_int")
            self.kb.raw_line(f"    int {cast_var} = (int){input_var};")
            input_var = cast_var

        # The dispatch may run more threads (block_size = num_threads) than the
        # logical reduce length N when the input has no make_range to pin block_size
        # to N — e.g. a reduce over a tt.splat/full. Lanes >= N then hold a value that
        # must NOT contribute (re-audit #6: sum over a splat of N summed num_threads
        # copies — 768 vs 24). Mask the tail to the combine identity. No-op when
        # N == block_size (the common make_range case, where N lanes == all lanes).
        # A one-row (1, N) tile reduced along its row reaches here too, and is the same
        # N-element full reduce: an operand that is itself a reduction result is held on
        # more lanes than its N elements (a kept axis-0 reduction repeats every N lanes).
        _rn = input_shape[0] if (input_shape and len(input_shape) == 1) else None
        if _rn is None and self._is_2d and input_shape and len(input_shape) == 2:
            if input_shape[0] == 1 and axis == 1:
                _rn = input_shape[1]
        if _rn is not None and _rn < self.kb.block_size:
            from triton_msl.errors import MetalNonRecoverableError

            _is_float = msl_type in ("float", "half", "bfloat")
            if combine_op == "sum":
                _ident = "-0.0f" if _is_float else "0"
            elif combine_op == "prod":
                _ident = "1.0f" if _is_float else "1"
            elif combine_op == "max" and _is_float:
                _ident = "-INFINITY"
            elif combine_op == "min" and _is_float:
                _ident = "INFINITY"
            elif combine_op in ("nanmax", "nanmin"):
                # NaN-prop max/min are float-only; -INF/INF tail-pad never spuriously
                # introduces a NaN (so masked lanes don't poison the result).
                _ident = "-INFINITY" if combine_op == "nanmax" else "INFINITY"
            elif combine_op in ("max", "min") and is_i64:
                _ident = self._reduce_identity_combine(combine_op, msl_type)[0]
            elif combine_op == "max":
                # integer max identity (reduce-probe over-refusal fix: int max/min over a
                # splat was wrongly refused).
                _ident = f"metal::numeric_limits<{msl_type}>::lowest()"
            elif combine_op == "min":
                _ident = f"metal::numeric_limits<{msl_type}>::max()"
            elif combine_op == "xor":
                _ident = "0"
            else:
                raise MetalNonRecoverableError(
                    f"reduce ('{combine_op}') over a {_rn}-element value dispatched "
                    f"with {self.kb.block_size} threads (no make_range pins the "
                    f"length): cannot mask the tail lanes for this combine/dtype, so "
                    f"the cross-lane reduce would over-count. Refusing.",
                    op_name="tt.reduce",
                )
            # Which lanes hold the _rn elements. A value of a row-wise (axis=1) reduce is
            # held one row per block of lanes: lanes 0.._rn-1 are all row 0, and taking
            # them reduced _rn copies of the first row. One lane of each block is taken.
            _holds = f"lid < {_rn}u"
            _layout = self._value_1d_layout_of(ssa.operand_ids[0])
            _widths = self._blocked_widths(ssa.operand_ids[0])
            _width = next(iter(_widths)) if len(_widths) == 1 else None
            if getattr(self, "_reduce_operand_gathered", False):
                # A spread tile was just gathered onto lanes 0.._rn-1 (see _lower_reduce):
                # the layout declared for its source no longer describes the operand.
                pass
            elif _layout == "blocked":
                if not _width:
                    raise MetalNonRecoverableError(
                        f"reduce ('{combine_op}') over a row-blocked value whose block width "
                        f"is not one known number. Refusing.",
                        op_name="tt.reduce",
                    )
                if _width > 1:
                    _holds = f"lid % {_width}u == 0u && lid / {_width}u < {_rn}u"
            elif _layout is None and (_widths or self._reduce_provenance(ssa.operand_ids[0])):
                raise MetalNonRecoverableError(
                    f"reduce ('{combine_op}') over a {_rn}-element value computed from a row-wise "
                    f"reduce or a row index, whose thread-to-element mapping is not proved; "
                    f"taking lanes 0..{_rn - 1} could reduce copies of the first row. Refusing.",
                    op_name="tt.reduce",
                )
            _masked = self._next_var("rmask")
            self.kb.raw_line(f"    {msl_type} {_masked} = ({_holds}) ? ({msl_type}){input_var} : ({msl_type}){_ident};")
            input_var = _masked

        # 64-bit integers reduce through a shared-memory tree. It runs AFTER the lane mask:
        # the tree combines every lane of the group, so an operand held on more lanes than
        # it has elements (a row-wise reduce result) was counted once per lane.
        if is_i64:
            self._lower_reduce_1d_i64(ssa, input_var, combine_op, msl_type, shared_dtype)
            return

        # 1D full reduction (original behavior)
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        n_simd_groups = (self.kb.block_size + 31) // 32
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=n_simd_groups)

        result_var = self._next_var("reduced")
        # Reduce IN the element type — integer reductions in float lost precision >2^24.
        self.kb.threadgroup_reduce(combine_op, input_var, shared_name, result_var, reduce_ty=msl_type)

        # Narrow-type masking: when reducing in wider type but output is narrow,
        # apply modular arithmetic (i1 sum = XOR, i8 sum = mod 256, etc.)
        out_elem = native["result_facts"][0].elem
        if out_elem == "i1":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 1);")
            result_var = masked_var
        elif out_elem == "i8":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 0xFF);")
            result_var = masked_var
        elif out_elem == "i16":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 0xFFFF);")
            result_var = masked_var

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        # A full reduce yields a UNIFORM scalar -- every thread holds the same value --
        # so the one thread that stores it is correct under any indexing. Declare DIRECT
        # so the store does not have to fall back to inferring a layout from the size.
        # (Reached for a 2-D tile via the multipass path, which rebinds the reduce input
        # to a per-thread accumulator and re-enters this scalar path.)
        self._register_1d_layout(ssa.id, "direct")
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, "direct")

    def _lower_reduce_1d_i64(self, ssa, input_var, combine_op, msl_type, shared_dtype):
        """1-D full reduce for 64-bit ints via a shared-memory tree (Metal has
        no simd_sum/max/min overload for long/ulong). Each thread writes its
        value to a threadgroup array; a stride-doubling tree (non-power-of-2
        safe via the `lid+s<bs` guard) reduces into slot 0; all threads read it.
        """
        bs = self.kb.block_size
        n = self._shared_counter
        self._shared_counter += 1
        sh = f"red64_{n}"
        self.kb.declare_threadgroup_array(sh, dtype=shared_dtype, size=bs)
        combine = {
            "sum": lambda a, b: f"({a} + {b})",
            "prod": lambda a, b: f"({a} * {b})",
            "max": lambda a, b: f"max({a}, {b})",
            "min": lambda a, b: f"min({a}, {b})",
            "umax": lambda a, b: f"max({a}, {b})",
            "umin": lambda a, b: f"min({a}, {b})",
            "xor": lambda a, b: f"({a} ^ {b})",
            "and": lambda a, b: f"({a} & {b})",
            "or": lambda a, b: f"({a} | {b})",
        }.get(combine_op)
        if combine is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(f"i64 reduce: unsupported combine op {combine_op}")
        kb = self.kb
        kb.raw_line(f"    {sh}[lid] = {input_var};")
        kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        kb.raw_line(f"    for (uint _s = 1u; _s < {bs}u; _s <<= 1u) {{")
        kb.raw_line(f"        if ((lid % (2u*_s)) == 0u && (lid + _s) < {bs}u) {{")
        kb.raw_line(f"            {sh}[lid] = {combine(f'{sh}[lid]', f'{sh}[lid + _s]')};")
        kb.raw_line("        }")
        kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        kb.raw_line("    }")
        result_var = self._next_var("reduced64")
        kb.raw_line(f"    {msl_type} {result_var} = {sh}[0];")
        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        # Uniform scalar (every thread reads sh[0]) -- see the 32-bit twin above.
        self._register_1d_layout(ssa.id, "direct")
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, "direct")

    def _lower_reduce_multi_value(self, ssa: SSAValue):
        """Multi-value reduce: argmax/argmin (2-value) or Welford (3-value).

        For 2 inputs: argmax/argmin (value + index) via SIMD shuffle + shared memory.
        For 3 inputs: Welford online variance (mean + m2 + weight) via SIMD shuffle + shared memory.
        """
        from triton_msl.errors import MetalNonRecoverableError

        native = self._prove_reduce_native_contract(ssa)
        n_values = len(native["slot_facts"])
        if n_values not in (2, 3) or len(native["shape"]) not in (1, 2):
            raise MetalNonRecoverableError(
                "multi-result tt.reduce is implemented only for rank-1/rank-2 argmin/argmax "
                "pairs and rank-1/rank-2 three-slot Welford reductions",
                op_name="tt.reduce",
            )

        # Guard (2026-06-21 audit sibling-divergence): a multi-value reduce
        # (argmax/argmin/Welford) over a 1-D tile WIDER than the threadgroup,
        # INSIDE control flow, would have its tail dropped by the SIMD-shuffle tree
        # (which covers only block_size elements) -> silent-wrong. The scalar
        # single-value reduce folds-or-refuses this (see _lower_reduce); folding a
        # tuple reduce is harder, so mirror the refusal here.
        # This now covers BOTH (a) the in-control-flow case (the SIMD tree would
        # drop the tail -> silent-wrong) AND (b) the top-level N>block_size case,
        # where the multipass reduction path has no argmin/argmax/Welford aggregation
        # and emits uncompilable MSL (an undeclared per-iteration var) -> a cryptic
        # MetalCompilationError instead of a clean refusal (re-audit #3). Either way,
        # a 1-D multi-value reduce wider than the threadgroup is unsupported: refuse.
        _ishape = native["shape"]
        if _ishape is not None and len(_ishape) == 1 and _ishape[0] > self.kb.block_size:
            raise MetalNonRecoverableError(
                f"Multi-value reduce (argmax/argmin/Welford) over a 1-D tile of "
                f"{_ishape[0]} elements exceeds the {self.kb.block_size}-thread "
                f"threadgroup; the multipass path can't aggregate a tuple reduce, so "
                f"this would be silently wrong / uncompilable. Refusing. "
                f"Use BLOCK <= num_threads."
            )

        # Dispatch Welford (3-value) vs argmax/argmin (2-value)
        if n_values == 3:
            if any(facts.kind != "float" or facts.elem != "f32" or facts.width != 32 for facts in native["slot_facts"]):
                raise MetalNonRecoverableError(
                    "the Welford tuple lowering requires three native fp32 slots; a narrower "
                    "or mixed representation would be silently widened by its float temporaries",
                    op_name="tt.reduce",
                )
            # Route to Welford only when the complete returned value DAG is the recurrence
            # emitted below.  A division-presence heuristic silently misclassified a valid
            # weighted-mean tuple and replaced its independent second result with M2.
            welford_ratio_mode = self._classify_exact_welford_recurrence(ssa)
            if welford_ratio_mode is None:
                raise MetalNonRecoverableError(
                    "multi-value tl.reduce with a three-element tuple body whose returned "
                    "value DAG is not the exact Welford online-variance recurrence is not "
                    "supported — replaying a merely division-containing tuple as Welford "
                    "would silently change its results. "
                    "Refusing. Only tl.var/tl.std (Welford) 3-tuples and value+index "
                    "argmax/argmin 2-tuples are handled.",
                    op_name="tt.reduce",
                )
            self._lower_reduce_welford(ssa, native, guarded_ratio=welford_ratio_mode == "guarded")
            return

        value_facts, index_facts = native["slot_facts"]
        value_widths = {16, 32} if value_facts.kind == "float" else {8, 16, 32, 64}
        if (
            value_facts.kind not in {"float", "integer"}
            or value_facts.width not in value_widths
            or index_facts.kind != "integer"
            or index_facts.width != 32
        ):
            raise MetalNonRecoverableError(
                "argmin/argmax tuple lowering requires a native f16/f32 or i8/i16/i32/i64 "
                "value slot and an i32 index slot",
                op_name="tt.reduce",
            )

        argminmax = self._classify_exact_argminmax_recurrence(ssa, value_facts.kind)
        if argminmax is None:
            raise MetalNonRecoverableError(
                "multi-value tl.reduce whose returned value DAG is not the exact "
                "argmin/argmax value comparison plus smaller-index tie-break is not "
                "supported — comparison presence alone cannot prove the tuple semantics. "
                "Refusing rather than replay a different pair reduction as argmin/argmax.",
                op_name="tt.reduce",
            )

        # Check for 2D argmin/argmax
        input_shape = native["shape"]
        if len(input_shape) == 2 and self._is_2d:
            axis = native["axis"]
            # Skip 2D dispatch when first dim is 1 and axis is 1
            # (really a 1D reduction, same logic as _lower_reduce)
            if not (input_shape[0] == 1 and axis == 1):
                self._lower_reduce_2d_argminmax(ssa, axis, input_shape, native, argminmax)
                return

        self._lower_reduce_argminmax(ssa, native, argminmax)

    def _lower_reduce_welford(self, ssa: SSAValue, native, *, guarded_ratio):
        """Welford online variance reduction: (mean, m2, weight) via SIMD shuffle + shared memory."""
        mean_var = self._lookup(ssa.operand_ids[0])
        m2_var = self._lookup(ssa.operand_ids[1])
        weight_var = self._lookup(ssa.operand_ids[2])

        n_simd_groups = (self.kb.block_size + 31) // 32

        # Shared memory for 3 values
        sh_mean = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        sh_m2 = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        sh_w = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(sh_mean, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_m2, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_w, dtype="fp32", size=n_simd_groups)

        wm = self._next_var("wm")  # working mean
        wv = self._next_var("wv")  # working m2
        ww = self._next_var("ww")  # working weight
        rm = self._next_var("rm")  # result mean
        rv = self._next_var("rv")  # result m2
        rw = self._next_var("rw")  # result weight

        self.kb.raw_line(f"    // Welford reduce")
        self.kb.raw_line(f"    float {wm} = {mean_var};")
        self.kb.raw_line(f"    float {wv} = {m2_var};")
        self.kb.raw_line(f"    float {ww} = {weight_var};")

        # SIMD-level tree reduction
        self.kb.raw_line(f"    for (ushort _d = 16; _d >= 1; _d >>= 1) {{")
        self.kb.raw_line(f"        float _om = simd_shuffle_down({wm}, _d);")
        self.kb.raw_line(f"        float _ov = simd_shuffle_down({wv}, _d);")
        self.kb.raw_line(f"        float _ow = simd_shuffle_down({ww}, _d);")
        self.kb.raw_line(f"        float _delta = _om - {wm};")
        self.kb.raw_line(f"        float _nw = {ww} + _ow;")
        ratio_expr = "(_nw == 0.0f) ? 0.0f : _ow / _nw" if guarded_ratio else "_ow / _nw"
        self.kb.raw_line(f"        float _ratio = {ratio_expr};")
        self.kb.raw_line(f"        {wm} = {wm} + _delta * _ratio;")
        self.kb.raw_line(f"        {wv} = {wv} + _ov + _delta * _delta * {ww} * _ratio;")
        self.kb.raw_line(f"        {ww} = _nw;")
        self.kb.raw_line(f"    }}")

        # Write lane 0 of each SIMD group to shared
        self.kb.raw_line(f"    if (lid % 32 == 0) {{")
        self.kb.raw_line(f"        {sh_mean}[lid / 32] = {wm};")
        self.kb.raw_line(f"        {sh_m2}[lid / 32] = {wv};")
        self.kb.raw_line(f"        {sh_w}[lid / 32] = {ww};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Cross-SIMD reduction
        if n_simd_groups > 1:
            self.kb.raw_line(f"    if (lid == 0) {{")
            self.kb.raw_line(f"        {wm} = {sh_mean}[0];")
            self.kb.raw_line(f"        {wv} = {sh_m2}[0];")
            self.kb.raw_line(f"        {ww} = {sh_w}[0];")
            self.kb.raw_line(f"        for (uint _s = 1; _s < {n_simd_groups}u; _s++) {{")
            self.kb.raw_line(f"            float _om = {sh_mean}[_s];")
            self.kb.raw_line(f"            float _ov = {sh_m2}[_s];")
            self.kb.raw_line(f"            float _ow = {sh_w}[_s];")
            self.kb.raw_line(f"            float _delta = _om - {wm};")
            self.kb.raw_line(f"            float _nw = {ww} + _ow;")
            self.kb.raw_line(f"            float _ratio = {ratio_expr};")
            self.kb.raw_line(f"            {wm} = {wm} + _delta * _ratio;")
            self.kb.raw_line(f"            {wv} = {wv} + _ov + _delta * _delta * {ww} * _ratio;")
            self.kb.raw_line(f"            {ww} = _nw;")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {sh_mean}[0] = {wm};")
            self.kb.raw_line(f"        {sh_m2}[0] = {wv};")
            self.kb.raw_line(f"        {sh_w}[0] = {ww};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read the final result
        self.kb.raw_line(f"    float {rm} = {sh_mean}[0];")
        self.kb.raw_line(f"    float {rv} = {sh_m2}[0];")
        self.kb.raw_line(f"    float {rw} = {sh_w}[0];")

        # Store all 3 results in env
        self.env[ssa.id] = rm
        self.env_types[ssa.id] = "fp32"
        if ssa.result_ids and len(ssa.result_ids) >= 3:
            self.env[ssa.result_ids[0]] = rm
            self.env_types[ssa.result_ids[0]] = "fp32"
            self.env[ssa.result_ids[1]] = rv
            self.env_types[ssa.result_ids[1]] = "fp32"
            self.env[ssa.result_ids[2]] = rw
            self.env_types[ssa.result_ids[2]] = "fp32"

    def _lower_reduce_argminmax(self, ssa: SSAValue, native, argminmax):
        """Argmax/argmin: value + index via SIMD shuffle + shared memory."""
        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Determine value type
        val_dtype = native["slot_dtypes"][0]
        is_int = not (val_dtype.startswith("fp") or val_dtype.startswith("bf"))
        # A 64-bit int value cannot go through this SIMD-shuffle reduction — simd_shuffle_down
        # has NO 64-bit overload, and staging as 32-bit would silently truncate the high word
        # and pick the WRONG index (Triton-lens re-audit 2026-06-25: argmax over i64 values
        # > 2^31 returned a wrong index). Refuse cleanly rather than truncate or emit a cryptic
        # 'no matching simd_shuffle_down' compile error.
        if val_dtype in ("i64", "u64", "ui64"):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "argmax/argmin over a 64-bit integer value is not supported (the SIMD-shuffle "
                "reduction has no 64-bit path; a 32-bit staging would silently truncate the "
                "high word and return the wrong index). Refusing. Cast the values to int32 or "
                "float first.",
                op_name="tt.reduce",
            )
        # uint8/16/32 are the SIGNLESS i8/i16/i32 in Triton — a signed compare picks the
        # wrong arg index above the sign bit. Stage + compare as uint when the combine is
        # unsigned (arith.cmpi ugt/ult). 64-bit already refused above.
        is_max, proved_unsigned = argminmax
        _unsigned = is_int and proved_unsigned
        msl_val_type = "uint" if _unsigned else ("int" if is_int else "float")
        val_shared_dtype = "u32" if _unsigned else ("i32" if is_int else "fp32")

        cmp_op = ">" if is_max else "<"

        # Allocate shared memory for values and indices
        n_simd_groups = (self.kb.block_size + 31) // 32
        shared_val = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        shared_idx = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_val, dtype=val_shared_dtype, size=n_simd_groups)
        self.kb.declare_threadgroup_array(shared_idx, dtype="i32", size=n_simd_groups)

        # Unique variable names
        mv = self._next_var("mv")
        mi = self._next_var("mi")
        result_val = self._next_var("rval")
        result_idx = self._next_var("ridx")

        tag = "max" if is_max else "min"
        self.kb.raw_line(f"    // Multi-value reduce: arg{tag}")
        _holds, _idx_expr = self._argminmax_lane_contract(ssa, native, idx_var)
        if _holds is None:
            self.kb.raw_line(f"    {msl_val_type} {mv} = {val_var};")
            self.kb.raw_line(f"    int {mi} = {idx_var};")
        else:
            # Lanes that hold no element of the operand lose every comparison: a lane
            # outside the tile holds a zero, which won an all-negative argmax or an
            # all-positive argmin and returned its (zero) index.
            if msl_val_type == "float":
                _lose = "(-INFINITY)" if is_max else "INFINITY"
            elif msl_val_type == "uint":
                _lose = "0u" if is_max else "0xFFFFFFFFu"
            else:
                _lose = "INT_MIN" if is_max else "INT_MAX"
            self.kb.raw_line(f"    {msl_val_type} {mv} = ({_holds}) ? ({msl_val_type}){val_var} : {_lose};")
            self.kb.raw_line(f"    int {mi} = ({_holds}) ? (int)({_idx_expr}) : INT_MAX;")

        # SIMD-level tree reduction using simd_shuffle_down. The source lane is
        # group-relative (tiisg + _d); its GLOBAL thread index is lid + _d. When the
        # tile is smaller than the SIMD group (block_size < 32, e.g. argmin over N=16)
        # those source lanes are INACTIVE and simd_shuffle_down returns garbage (often
        # 0) — for an all-positive argmin the 0 wins and the index collapses to 0
        # (re-audit #10). Guard the take on the source thread being a real element
        # (lid + _d < block_size). No-op for full groups (lane 0 reads 16,8,..,1, all
        # valid); the upper lanes' results are discarded regardless.
        _bs = self.kb.block_size
        self.kb.raw_line(f"    for (ushort _d = 16; _d >= 1; _d >>= 1) {{")
        self.kb.raw_line(f"        {msl_val_type} _ov = simd_shuffle_down({mv}, _d);")
        self.kb.raw_line(f"        int _oi = simd_shuffle_down({mi}, _d);")
        self.kb.raw_line(
            f"        bool _take = ((lid + _d) < {_bs}u) && ((_ov {cmp_op} {mv}) || (_ov == {mv} && _oi < {mi}));"
        )
        self.kb.raw_line(f"        {mv} = _take ? _ov : {mv};")
        self.kb.raw_line(f"        {mi} = _take ? _oi : {mi};")
        self.kb.raw_line(f"    }}")

        # Write lane 0 of each SIMD group to shared memory
        self.kb.raw_line(f"    if (lid % 32 == 0) {{")
        self.kb.raw_line(f"        {shared_val}[lid / 32] = {mv};")
        self.kb.raw_line(f"        {shared_idx}[lid / 32] = {mi};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Cross-SIMD reduction (thread 0 reduces across SIMD groups)
        if n_simd_groups > 1:
            self.kb.raw_line(f"    if (lid == 0) {{")
            self.kb.raw_line(f"        {mv} = {shared_val}[0];")
            self.kb.raw_line(f"        {mi} = {shared_idx}[0];")
            self.kb.raw_line(f"        for (uint _s = 1; _s < {n_simd_groups}u; _s++) {{")
            self.kb.raw_line(f"            {msl_val_type} _ov = {shared_val}[_s];")
            self.kb.raw_line(f"            int _oi = {shared_idx}[_s];")
            self.kb.raw_line(f"            bool _take = (_ov {cmp_op} {mv}) || (_ov == {mv} && _oi < {mi});")
            self.kb.raw_line(f"            {mv} = _take ? _ov : {mv};")
            self.kb.raw_line(f"            {mi} = _take ? _oi : {mi};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {shared_val}[0] = {mv};")
            self.kb.raw_line(f"        {shared_idx}[0] = {mi};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read the final result
        self.kb.raw_line(f"    {msl_val_type} {result_val} = {shared_val}[0];")
        self.kb.raw_line(f"    int {result_idx} = {shared_idx}[0];")
        # Barrier after the broadcast read — a SECOND argminmax in the same kernel re-stages
        # these pooled shared arrays and would otherwise race this read (twin of the 2-D
        # argminmax + the 1ddac9b reduce fixes).
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store both results in env
        self.env[ssa.id] = result_val
        self.env_types[ssa.id] = val_shared_dtype
        if ssa.result_ids and len(ssa.result_ids) >= 2:
            self.env[ssa.result_ids[0]] = result_val
            self.env_types[ssa.result_ids[0]] = val_shared_dtype
            self.env[ssa.result_ids[1]] = result_idx
            self.env_types[ssa.result_ids[1]] = "i32"
        # Full 1-D reduce to a UNIFORM scalar (every thread reads shared_val[0] /
        # shared_idx[0] above), so any thread's store is correct under any indexing --
        # same reasoning as the scalar full-reduce sites. Declare DIRECT so a 1-D store
        # of this result does not fall back to size-inferring a layout.
        self._register_1d_layout(ssa.id, "direct")
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, "direct")

    def _argminmax_lane_contract(self, ssa, native, idx_var):
        """Which lanes take part in a full argmin/argmax, and the index each one carries.

        Returns ``(None, None)`` when every lane of the group holds its own element (the
        operand fills the threadgroup one element per lane), else ``(predicate, index)``.
        An operand with fewer elements than the group has lanes, or one held a block of
        lanes per element (a row-wise reduce result), needs both: the unmasked tree took
        the zeros of the lanes outside the tile, and the lane number as the index.
        """
        from triton_msl.errors import MetalNonRecoverableError

        shape = native["shape"]
        if len(shape) == 1:
            n = shape[0]
        elif len(shape) == 2 and shape[0] == 1 and native["axis"] == 1:
            n = shape[1]
        else:
            return None, None
        value_id, index_id = ssa.operand_ids[0], ssa.operand_ids[1]
        if value_id in self._bcast_layout:
            raise MetalNonRecoverableError(
                "argmax/argmin of a value held under a recorded broadcast index (a tile reduced "
                "along a third axis): the lanes that hold each element are not one per element, "
                "so the index of the winner cannot be proved. Refusing.",
                op_name="tt.reduce",
            )
        kind, width = self._lanes_of_1d_value(value_id, n, "argmax/argmin")
        if kind == "direct" and n >= self.kb.block_size:
            return None, None
        element = f"lid / {width}u" if kind == "blocked" else "lid"
        holds = f"{element} < {n}u"
        # The index operand of tl.argmax / tl.argmin is tl.arange(0, n) laid out for the
        # operand's declared layout, not for the lanes the value is held on. When it is
        # that range, the element number is the index.
        by_id = self._layout_op_index()
        cur = index_id
        start = None
        for _ in range(8):
            op = by_id.get(cur)
            if op is None:
                break
            if op.op == "tt.make_range":
                _start = op.attrs.get("start", 0)
                _end = op.attrs.get("end", self.graph.block_size)
                if _end - _start == n:
                    start = _start
                break
            if op.op in self._LAYOUT_PASSTHROUGH_OPS and op.operand_ids:
                cur = op.operand_ids[0]
                continue
            break
        if start is not None:
            return holds, f"(int)({element}) + {start}"
        if self._value_1d_layout_of(index_id) == "any":
            return holds, idx_var  # one index on every lane (a one-element operand)
        if kind == "blocked" or self._lanes_of_1d_value(index_id, n, "argmax/argmin index")[0] != "direct":
            raise MetalNonRecoverableError(
                "argmax/argmin of a row-blocked value whose index operand is not the plain "
                "tl.arange of its length: the index each lane carries cannot be proved. Refusing.",
                op_name="tt.reduce",
            )
        return holds, idx_var

    def _lower_reduce_2d_argminmax(self, ssa, axis, input_shape, native, argminmax):
        """Lower 2D argmin/argmax: find min/max value and index along axis.

        For axis=1 on (M, N): each row finds min/max among N values → (M,) values + indices.
        For axis=0 on (M, N): each column finds min/max among M values → (N,) values + indices.
        """
        M, N = input_shape[0], input_shape[1]
        total = M * N

        # (reduce-probe #5/#6, FIXED 2026-08-26) Two refusals used to live here:
        #
        #   * axis=0 on a SQUARE (M==N) tile "mis-broadcasts the index"
        #   * axis=1 with BOTH value and index consumed "broadcasts row-0's index"
        #
        # Neither was a reduce bug -- this function always broadcast correctly
        # (shared[lid % N] for axis=0, shared[lid / N] for axis=1). Both were STORE
        # bugs. The store inferred the 1-D layout from the result's SIZE, which cannot
        # tell an axis=0 result (size N) from an axis=1 one (size M) on a square tile;
        # and it decided "was this converted to simple layout?" from a KERNEL-WIDE flag,
        # so a converted value dragged an unconverted index onto the wrong layout.
        #
        # Both are now resolved structurally: this reduce records its broadcast layout
        # per result id (below) and the store consults that, plus a per-value
        # convert_layout check. See _value_1d_layout_of / _traces_to_converted_layout.

        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        val_dtype = native["slot_dtypes"][0]
        is_int = not (val_dtype.startswith("fp") or val_dtype.startswith("bf"))
        is_i64 = val_dtype in ("i64", "u64", "ui64")
        # uint8/16/32/64 are the SIGNLESS i8/i16/i32/i64 in Triton — a signed compare picks the
        # wrong arg index above the sign bit. The unsigned signal is the combine's arith.cmpi
        # ugt/ult (NOT the dtype name, which is signless i64 even for uint64). Detect it for
        # ALL widths incl. 64-bit (the earlier `not is_i64` gate here was itself the un-fixed
        # 64-bit twin: uint64 argmax/argmin computed signed). The `or val_dtype in (u64,ui64)`
        # keeps any literal-name path unsigned too.
        is_max, proved_unsigned = argminmax
        _unsigned = is_int and proved_unsigned
        _u64 = is_i64 and (_unsigned or val_dtype in ("u64", "ui64"))
        # 64-bit int values compare at full width (else the high word truncates and the
        # arg index is wrong) — Triton-lens re-audit 2026-06-25.
        if not is_int:
            msl_val_type, val_shared_dtype = "float", "fp32"
        elif is_i64:
            msl_val_type, val_shared_dtype = ("ulong", "u64") if _u64 else ("long", "i64")
        elif _unsigned:
            msl_val_type, val_shared_dtype = "uint", "u32"  # uint8/16/32 widen to uint
        else:
            msl_val_type, val_shared_dtype = "int", "i32"

        cmp_op = ">" if is_max else "<"
        identity = "(-INFINITY)" if is_max and not is_int else "INFINITY"
        if is_int:
            if _u64:  # unsigned 64-bit (incl signless i64)
                identity = "0" if is_max else "ULONG_MAX"
            elif is_i64:  # signed 64-bit
                identity = "LONG_MIN" if is_max else "LONG_MAX"
            elif _unsigned:  # uint8/16/32
                identity = "0" if is_max else "UINT_MAX"
            else:
                identity = "INT_MIN" if is_max else "INT_MAX"

        # Shared memory for values and indices
        shared_val = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        shared_idx = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_val, dtype=val_shared_dtype, size=total)
        self.kb.declare_threadgroup_array(shared_idx, dtype="i32", size=total)

        # Stage values and indices
        self.kb.raw_line(f"    if (lid < {total}u) {{")
        self.kb.raw_line(f"        {shared_val}[lid] = {val_var};")
        self.kb.raw_line(f"        {shared_idx}[lid] = {idx_var};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Result arrays
        if axis == 1:
            result_size = M
        else:
            result_size = N
        result_val_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        result_idx_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result_val_shared, dtype=val_shared_dtype, size=result_size)
        self.kb.declare_threadgroup_array(result_idx_shared, dtype="i32", size=result_size)

        result_val_var = self._next_var("rval")
        result_idx_var = self._next_var("ridx")

        self.kb.raw_line(f"    {msl_val_type} {result_val_var} = {identity};")
        self.kb.raw_line(f"    int {result_idx_var} = 0;")

        if axis == 1:
            # Each row: find argmin/max among N columns
            self.kb.raw_line(f"    if (lid < {M}u) {{")
            self.kb.raw_line(f"        {msl_val_type} best_v = {identity};")
            self.kb.raw_line(f"        int best_i = 0;")
            self.kb.raw_line(f"        for (uint j = 0; j < {N}u; j++) {{")
            self.kb.raw_line(f"            {msl_val_type} v = {shared_val}[lid * {N}u + j];")
            # The argmin/max index along axis=1 IS the column position j; the staged
            # shared_idx came back uniformly 0 (re-audit #14) so use the position.
            self.kb.raw_line(f"            int idx = (int)j;")
            self.kb.raw_line(f"            if (v {cmp_op} best_v || (v == best_v && idx < best_i)) {{")
            self.kb.raw_line(f"                best_v = v; best_i = idx;")
            self.kb.raw_line(f"            }}")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_val_shared}[lid] = best_v;")
            self.kb.raw_line(f"        {result_idx_shared}[lid] = best_i;")
            self.kb.raw_line(f"    }}")
        else:
            # Each column: find argmin/max among M rows
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {msl_val_type} best_v = {identity};")
            self.kb.raw_line(f"        int best_i = 0;")
            self.kb.raw_line(f"        for (uint i = 0; i < {M}u; i++) {{")
            self.kb.raw_line(f"            {msl_val_type} v = {shared_val}[i * {N}u + lid];")
            # The argmin/max index along axis=0 IS the row position i (staged shared_idx
            # came back uniformly 0 — re-audit #14).
            self.kb.raw_line(f"            int idx = (int)i;")
            self.kb.raw_line(f"            if (v {cmp_op} best_v || (v == best_v && idx < best_i)) {{")
            self.kb.raw_line(f"                best_v = v; best_i = idx;")
            self.kb.raw_line(f"            }}")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_val_shared}[lid] = best_v;")
            self.kb.raw_line(f"        {result_idx_shared}[lid] = best_i;")
            self.kb.raw_line(f"    }}")

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Broadcast result to all threads.
        # Row-major: threads [0..N-1] in row 0, [N..2N-1] in row 1.
        if axis == 1:
            self.kb.raw_line(f"    {result_val_var} = {result_val_shared}[lid / {N}u];")
            self.kb.raw_line(f"    {result_idx_var} = {result_idx_shared}[lid / {N}u];")
        else:
            self.kb.raw_line(f"    {result_val_var} = {result_val_shared}[lid % {N}u];")
            self.kb.raw_line(f"    {result_idx_var} = {result_idx_shared}[lid % {N}u];")
        # Record the layout these results are broadcast in, so the STORE does not have to
        # infer it from the result's SIZE. Size is ambiguous on a square tile: an axis=0
        # result has size N, an axis=1 result has size M, and when M == N the store
        # matched the axis=1 rule and emitted `lid % N == 0 ? out[lid / N]` -- every
        # thread that stored read column 0, so column 0's index was broadcast to the whole
        # output (reduce-probe #6, previously refused for square tiles).
        _layout = "direct" if axis == 0 else "blocked"
        self._register_1d_layout(ssa.id, _layout, N)
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, _layout, N)
        # Barrier AFTER the broadcast read of the pooled result arrays — without it a SECOND
        # argminmax in the same kernel re-stages those (declare_threadgroup_array-pooled)
        # arrays and races this read, so a tail of rows return their own index. Twin of the
        # 1ddac9b _lower_reduce_2d barrier fix (this argminmax path was missed).
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store results
        self.env[ssa.id] = result_val_var
        self.env_types[ssa.id] = val_shared_dtype
        if ssa.result_ids and len(ssa.result_ids) >= 2:
            self.env[ssa.result_ids[0]] = result_val_var
            self.env_types[ssa.result_ids[0]] = val_shared_dtype
            self.env[ssa.result_ids[1]] = result_idx_var
            self.env_types[ssa.result_ids[1]] = "i32"
        # Set output shape
        out_shape = (M,) if axis == 1 else (N,)
        self.env_shapes[ssa.id] = out_shape
        if ssa.result_ids:
            for rid in ssa.result_ids:
                self.env_shapes[rid] = out_shape

    def _has_combined_2d_reduce(self):
        """True if two DIFFERENT 2-D axis-reduce results are the two operands of one
        binary arithmetic op (e.g. tl.sum(x,1) + tl.max(x,1)).

        That combined result is NOT convert_layout-ed before the store (only a single
        reduce is), so it is stored under the broadcast layout and a tail subset of rows
        reads incorrectly — reduce-probe #7 (in-loop acc+= form) and its no-loop sibling
        (direct store), same convert_layout root cause. FlashAttention's reduces always
        feed an arith op whose OTHER operand is NOT a reduce (max -> qk-max, sum -> li*a),
        so this discriminator does not over-refuse FA.
        """
        # Flatten scf.for/scf.if region bodies too — for the in-loop acc+= form (#7) the
        # reduces and the combining arith op live in the loop region, not at top level.
        all_ops = list(self.graph.ops)
        for s in self.graph.ops:
            if getattr(s, "region_ops", None):
                all_ops.extend(s.region_ops)
        by_id = {s.id: s for s in all_ops}
        reduce_ids = set()
        for s in all_ops:
            if s.op == "tt.reduce" and s.operand_ids:
                _it = self._find_op_type_str(s.operand_ids[0])
                _ish = _extract_shape(_it) if _it else None
                if _ish and len(_ish) == 2:
                    reduce_ids.add(s.id)
                    for _rid in s.result_ids or []:
                        reduce_ids.add(_rid)
        if len(reduce_ids) < 2:
            return False
        _LAYOUT = {"tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape", "tt.splat"}

        def _to_reduce(sid, depth=0, seen=None):
            if seen is None:
                seen = set()
            if depth > 8 or sid in seen:
                return None
            seen.add(sid)
            if sid in reduce_ids:
                return sid
            op = by_id.get(sid)
            if op is None or not op.operand_ids:
                return None
            if op.op in _LAYOUT:
                return _to_reduce(op.operand_ids[0], depth + 1, seen)
            return None

        _BIN = {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.maxnumf",
            "arith.minnumf",
            "arith.maximumf",
            "arith.minimumf",
        }
        for s in all_ops:
            if s.op in _BIN and s.operand_ids and len(s.operand_ids) >= 2:
                r0 = _to_reduce(s.operand_ids[0])
                r1 = _to_reduce(s.operand_ids[1])
                if r0 is not None and r1 is not None and r0 != r1:
                    return True
        return False

    def _lower_reduce_2d(
        self,
        ssa,
        input_var,
        axis,
        combine_op,
        msl_type,
        shared_dtype,
        input_shape,
        input_dtype,
    ):
        """Lower a 2D axis-specific reduction.

        For axis=1 on (M, N): each of M rows sums its N values.
        For axis=0 on (M, N): each of N columns sums its M values.

        Uses shared memory to collect all values, then each result-thread
        performs a sequential reduction over its assigned group.
        """
        M, N = input_shape[0], input_shape[1]
        total = M * N

        # Combined 2-D reduce (reduce-probe #7 + its no-loop sibling): two 2-D axis
        # reduces arithmetically combined in one expression mis-compute a tail subset of
        # rows (the combined result isn't layout-converted before the store). Refuse
        # loudly — both the in-loop acc+= and the no-loop direct-store forms — rather than
        # silently mis-compute. Single reduces, separate stores, and 1-D combines are fine.
        if self._has_combined_2d_reduce():
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "two 2-D axis reductions combined in one arithmetic expression (e.g. "
                "tl.sum(x,1) + tl.max(x,1)) mis-compute a subset of rows — the combined "
                "result is not layout-converted before the store. Refusing. Use separate "
                "stores, or reduce to 1-D.",
                op_name="tt.reduce",
            )

        # UNDER-FILL guard (re-audit #14, twin of the >block_size over-fill guard). Inside
        # control flow the threadgroup is forced to the num_warps*32 minimum, so a small
        # tile (M*N < block_size) under-fills it: the extra threads [total, block_size)
        # corrupt the staged 2-D reduce (uninitialized-read — verified WRONG for M*N<=128
        # in a loop, CORRECT at >=256; the no-loop case has block_size==total so was
        # always correct). Refuse loudly rather than mis-compute. The fused softmax/FA and
        # MEPT reduces don't hit this (their tiles fill the group, or are 1-D).
        # In control flow the threadgroup is dispatched at a 256-thread minimum (8
        # SIMD groups), even when num_warps*32 is smaller — that surplus is what
        # corrupts a small under-filling tile.
        # An in-loop 2-D reduce of fp16/bf16 input is mis-staged ACROSS iterations: the
        # shared-memory staging/result reuse corrupts iterations 2+ so every row collapses
        # to the first row's value (reduce-fuzzer: T=1 fp16 correct, T>=2 WRONG; fp32 fine
        # at any T). Refuse fp16/bf16 in-loop 2-D reduces; fp32 + no-loop are unaffected.
        if getattr(self, "_control_flow_depth", 0) > 0 and input_dtype in ("fp16", "bf16"):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "in-loop 2-D reduction of fp16/bf16 input is mis-staged across iterations "
                "(silent-wrong: rows collapse to the first). Refusing. Reduce in fp32, or "
                "outside the loop.",
                op_name="tt.reduce",
            )

        _tg_threads = max(max(1, getattr(self.graph, "num_warps", 1)) * 32, 256)
        if getattr(self, "_control_flow_depth", 0) > 0 and total < _tg_threads:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing in-loop 2-D reduction of a {tuple(input_shape)} tile that "
                f"under-fills the {_tg_threads}-thread group "
                f"(M*N={total} < block_size): the surplus threads corrupt the staged "
                f"reduce (silent-wrong). Use a tile with M*N >= block_size, or reduce "
                f"outside the loop.",
                op_name="tt.reduce",
            )

        # Check if the input data is already in a shared memory array (e.g.,
        # from a dot result or local_alloc). If so, skip the copy and reuse it.
        input_id = ssa.operand_ids[0] if ssa.operand_ids else None
        existing_shared = None
        if input_id is not None:
            existing_shared = getattr(self, "_shared_mem_descs", {}).get(input_id)

        if existing_shared:
            shared_name = existing_shared[0]
            # Data is already in shared_name[lid] — no copy needed
        else:
            # Allocate shared memory for the full 2D tensor
            shared_name = f"shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        ordered_cmp = combine_op.startswith("ordered_")

        # Identity + combine expression. Must branch on the actual MSL type:
        # long/ulong need 64-bit identities and the integer max/min overload,
        # not the 32-bit INT_MIN/INT_MAX or the float fmax/fmin.
        if ordered_cmp:
            # Zero initializes only lanes outside the logical result. Every
            # real result seeds from its first in-bounds axis element below.
            identity, combine_expr = f"({msl_type})0", None
        else:
            identity, combine_expr = self._reduce_identity_combine(combine_op, msl_type)

        result_var = self._next_var("reduced")

        # Store all values to shared memory (skip if already there).
        # If the input has a tracked broadcast layout, thread `lid` does not
        # hold the value at flat index `lid` — it holds the value at
        # `bcast_layout[ssa.id]` in the logical shape. Use that expression
        # as the write index so that shared_name has a row-major layout
        # matching the reduce's expectations.
        input_bcast_idx = None
        if ssa.operand_ids:
            input_bcast_idx = self._bcast_layout.get(ssa.operand_ids[0])
        if not existing_shared:
            if input_bcast_idx is not None:
                # All threads with lid < block_size participate; threads that
                # map to the same (i, j) write the same value (harmless,
                # last-writer-wins).  Threads whose mapped index is out of
                # range (shouldn't happen for consistent layouts) are guarded.
                bs = self.effective_block_size
                self.kb.raw_line(f"    if (lid < {bs}u) {shared_name}[{input_bcast_idx}] = {input_var};")
            else:
                # A tile with a unit axis is a 1-D value, and a 1-D value that is itself a
                # row-wise reduction (`tl.sum(v, axis=1, keep_dims=True)`) is held one
                # element per block of lanes: lanes 0..total-1 are then all its first
                # element. One lane of each block is staged.
                # (Not after a spread tile was gathered onto lanes 0..total-1 in
                # _lower_reduce: the layout declared for its source no longer holds.)
                _stage_w = 1
                if 1 in (M, N) and total > 1 and not getattr(self, "_reduce_operand_gathered", False):
                    _stage_w = self._lanes_of_1d_value(
                        ssa.operand_ids[0],
                        total,
                        f"reduce ('{combine_op}') of a {tuple(input_shape)} tile",
                        with_own_total=True,
                    )[1]
                if _stage_w > 1:
                    self.kb.raw_line(
                        f"    if (lid % {_stage_w}u == 0u && lid / {_stage_w}u < {total}u) "
                        f"{shared_name}[lid / {_stage_w}u] = {input_var};"
                    )
                else:
                    self.kb.raw_line(f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Use a second shared array to broadcast results to all threads
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1

        if axis == 1:
            # Reduce along columns: each row reduces N values → M results
            result_size = M
            self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=M)
            self.kb.raw_line(f"    {msl_type} {result_var} = {identity};")
            self.kb.raw_line(f"    if (lid < {M}u) {{")
            _seed = f"{shared_name}[lid * {N}u]" if ordered_cmp else identity
            _start = 1 if ordered_cmp else 0
            self.kb.raw_line(f"        {msl_type} acc = {_seed};")
            self.kb.raw_line(f"        for (uint j = {_start}u; j < {N}u; j++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[lid * {N}u + j];")
            if ordered_cmp:
                self.kb.raw_line(f"            acc = {self.kb.ordered_combine_expr(combine_op, 'acc', 'val')};")
            else:
                self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_var} = acc;")
            self.kb.raw_line(f"    }}")
            # The scratch allocator may alias input and result storage. A
            # completed read in ONE row is not completion in every other row.
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            self.kb.raw_line(f"    if (lid < {M}u) {result_shared}[lid] = {result_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their row's result.
            # Row-major: threads [0..N-1] in row 0, [N..2N-1] in row 1.
            if M == 1:
                # One result: every lane holds it (it may be broadcast over a wider tile).
                self.kb.raw_line(f"    {result_var} = {result_shared}[0];")
            else:
                self.kb.raw_line(f"    {result_var} = (lid < {total}u) ? {result_shared}[lid / {N}u] : {identity};")
            # Barrier AFTER the broadcast read: result_shared may alias the input shared
            # array (existing_shared reuse), and a following op that re-stages it (e.g. a
            # fused tt.scan's `shared[lid] = ...`) would otherwise race this read and a tail
            # subset of lanes would read the scan's data instead of the reduce result.
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
        else:
            # Reduce along rows: each column reduces M values → N results
            result_size = N
            self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=N)
            self.kb.raw_line(f"    {msl_type} {result_var} = {identity};")
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            _seed = f"{shared_name}[lid]" if ordered_cmp else identity
            _start = 1 if ordered_cmp else 0
            self.kb.raw_line(f"        {msl_type} acc = {_seed};")
            self.kb.raw_line(f"        for (uint i = {_start}u; i < {M}u; i++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[i * {N}u + lid];")
            if ordered_cmp:
                self.kb.raw_line(f"            acc = {self.kb.ordered_combine_expr(combine_op, 'acc', 'val')};")
            else:
                self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_var} = acc;")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            self.kb.raw_line(f"    if (lid < {N}u) {result_shared}[lid] = {result_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their column's result
            if N == 1:
                # One result: every lane holds it. Lanes 0..M-1 alone would leave the rest
                # of a wider tile it is broadcast over (`v * total`) with the identity.
                self.kb.raw_line(f"    {result_var} = {result_shared}[0];")
            else:
                self.kb.raw_line(f"    {result_var} = (lid < {total}u) ? {result_shared}[lid % {N}u] : {identity};")
            # Barrier after the broadcast read — see the axis==1 branch (guards the
            # result_shared/input-shared alias against a following re-stage, e.g. a scan).
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        # Result shape is the non-reduced dimension
        if axis == 1:
            self.env_shapes[ssa.id] = (M,)
        else:
            self.env_shapes[ssa.id] = (N,)
        # Declare the broadcast layout for the store (see _register_1d_layout): axis=1
        # read result_shared[lid / N] (BLOCKED, per row), axis=0 read
        # result_shared[lid % N] (DIRECT, per column). Without this the store falls back
        # to inferring the layout from the result's size, which is ambiguous.
        _layout = "blocked" if axis == 1 else "direct"
        self._register_1d_layout(ssa.id, _layout, N)
        for _rid in ssa.result_ids or []:
            self._register_1d_layout(_rid, _layout, N)

    def _lower_reduce_3d(
        self,
        ssa,
        input_var,
        axis,
        combine_op,
        msl_type,
        shared_dtype,
        input_shape,
        result_facts,
    ):
        """Lower a 3D axis-specific reduction.

        For (M, N, K) tensor reducing along axis:
          axis=0: result (N, K), loop over M
          axis=1: result (M, K), loop over N
          axis=2: result (M, N), loop over K

        Uses shared memory staging with loop-based loading for cases where
        total elements > block_size. Reads directly from the source X pointer.
        """
        M, N, K = input_shape[0], input_shape[1], input_shape[2]
        total = M * N * K
        block_size = self.kb.block_size

        # Find the source data pointer (first pointer arg = X)
        x_ptr_name = None
        for arg in self.graph.args:
            if arg.is_ptr:
                x_ptr_name = arg.name
                break
        if x_ptr_name is None:
            # Fallback — shouldn't happen
            return

        # Check if input has a tracked broadcast layout from a prior reduce.
        # When it does, thread `lid` holds the element at `input_bcast_idx`
        # in the logical 3D tensor, not at `lid` — we must use that as the
        # write index and as the effective lid for the readback.
        input_bcast_idx = None
        if ssa.operand_ids:
            input_bcast_idx = self._bcast_layout.get(ssa.operand_ids[0])

        # Identity and combine expression (4-way: float/long/ulong/int).
        identity, combine_expr = self._reduce_identity_combine(combine_op, msl_type)

        # Allocate shared memory for the full 3D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage values to shared memory from the already-loaded input_var
        # (not the raw pointer, which ignores strided/computed addresses)
        if total <= block_size:
            if input_bcast_idx is not None:
                # Non-canonical layout: write at bcast position.
                self.kb.raw_line(f"    if (lid < {block_size}u) {shared_name}[{input_bcast_idx}] = {input_var};")
            else:
                self.kb.raw_line(f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
        else:
            # Wrapping loop for large tensors — read from source pointer
            self.kb.raw_line(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            self.kb.raw_line(f"        {shared_name}[_e] = ({msl_type}){x_ptr_name}[_e];")
            self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Compute result dimensions
        if axis == 0:
            result_dims = (N, K)
            result_total = N * K
            axis_size = M
        elif axis == 1:
            result_dims = (M, K)
            result_total = M * K
            axis_size = N
        else:  # axis == 2
            result_dims = (M, N)
            result_total = M * N
            axis_size = K

        # Allocate result shared memory
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=result_total)

        # Reduction loop: each result thread reduces along the axis
        self.kb.raw_line(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        self.kb.raw_line(f"        {msl_type} acc = {identity};")

        # Compute result indices and shared memory indexing based on axis
        if axis == 0:
            # result (j, k) at _r: j = _r/K, k = _r%K. Loop over i.
            self.kb.raw_line(f"        uint _j = _r / {K}u;")
            self.kb.raw_line(f"        uint _k = _r % {K}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            # result (i, k) at _r: i = _r/K, k = _r%K. Loop over j.
            self.kb.raw_line(f"        uint _i = _r / {K}u;")
            self.kb.raw_line(f"        uint _k = _r % {K}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            # result (i, j) at _r: i = _r/N, j = _r%N. Loop over k.
            self.kb.raw_line(f"        uint _i = _r / {N}u;")
            self.kb.raw_line(f"        uint _j = _r % {N}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_i * {N * K}u + _j * {K}u + _a];")

        self.kb.raw_line(f"            acc = {combine_expr};")
        self.kb.raw_line(f"        }}")
        self.kb.raw_line(f"        {result_shared}[_r] = acc;")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read their result. Each thread with linear coord lid =
        # i*N*K + j*K + k reads the reduce output that corresponds to its
        # position in the original 3D tensor with the reduced axis collapsed.
        # This matches the semantics of reduce-then-expand_dims-then-broadcast:
        # after broadcast to the original shape, each position (i,j,k) holds
        # the reduce output at the corresponding coordinates of the collapsed
        # shape.
        #   axis=0 → output (j,k) at idx lid_input % (N*K)
        #   axis=1 → output (i,k) at idx (lid_input/(N*K))*K + lid_input%K
        #   axis=2 → output (i,j) at idx (lid_input/(N*K))*N + (lid_input%(N*K))/K
        # `lid_input` is lid for canonical inputs, input_bcast_idx otherwise.
        lid_input = input_bcast_idx if input_bcast_idx is not None else "lid"
        result_var = self._next_var("reduced")
        if axis == 0:
            read_idx = f"({lid_input} % {N * K}u)"
        elif axis == 1:
            read_idx = f"(({lid_input} / {N * K}u) * {K}u + ({lid_input} % {K}u))"
        else:  # axis == 2
            read_idx = f"(({lid_input} / {N * K}u) * {N}u + (({lid_input} % {N * K}u) / {K}u))"
        self.kb.raw_line(f"    {msl_type} {result_var} = {result_shared}[{read_idx}];")
        # Barrier after the broadcast read — a following op that re-stages this (possibly
        # existing_shared-aliased) result array would otherwise race the read (the recurring
        # shared-memory-race class; cf. _lower_reduce_2d 1ddac9b + the argminmax fix).
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = result_dims
        # Record the broadcast layout: thread `lid` holds the value at flat
        # index `read_idx` of the logical (M, N)-or-(M, K)-or-(N, K) result.
        # Downstream reduces/stores use this to re-stage data correctly when
        # the logical mapping is not lid → lid.
        self._bcast_layout[ssa.id] = f"({read_idx})"
        self._register_bcast_layout_by_native(result_facts, tuple(result_dims), f"({read_idx})")

    def _lower_reduce_nd(
        self,
        ssa,
        input_var,
        axis,
        combine_op,
        msl_type,
        shared_dtype,
        input_shape,
        result_facts,
    ):
        """Lower an axis-specific reduction for N-D tensors (N >= 4).

        Used by tl.sort's bitonic decomposition, which reshapes to (2,)*n and
        reduces along a specific axis per compare-and-swap step.

        Strategy:
          1. Stage the input tensor to shared memory (each thread writes its
             linear-index position).
          2. For each position in the output tensor (shape with axis collapsed),
             a single thread loops over the reduce axis and combines values.
          3. Each thread reads back its result: the output position that
             matches its coords in the original tensor with axis d removed.

        Index math uses strides computed from the shape:
          src_stride[i] = product of input_shape[i+1:]
          res_stride[j] = product of result_shape[j+1:] (result_shape drops axis d)
        """
        # The general N-D (rank>=4) reduce mis-computes for ALL axes — the readback/store
        # compaction writes results to the wrong slots (reduce-probe: (2,2,2,2) sum
        # garbage on every axis). The ONLY correct user of this path is tl.sort's bitonic
        # xor-decomposition, whose xor result feeds the next compare-swap step rather than
        # a direct store. Refuse a non-xor N-D reduce rather than silently mis-compute.
        if combine_op != "xor":
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"N-D (rank {len(input_shape)}) tl.reduce with a '{combine_op}' combine "
                f"is not correctly lowered (the result compaction is wrong for all axes). "
                f"Refusing. Reduce in <= 3 dimensions.",
                op_name="tt.reduce",
            )
        n = len(input_shape)
        total = 1
        for s in input_shape:
            total *= s
        block_size = self.kb.block_size

        # Compute strides for input (row-major)
        src_strides = [1] * n
        for i in range(n - 2, -1, -1):
            src_strides[i] = src_strides[i + 1] * input_shape[i + 1]

        # Result shape = input_shape with axis removed
        result_shape = tuple(input_shape[:axis]) + tuple(input_shape[axis + 1 :])
        result_total = 1
        for s in result_shape:
            result_total *= s
        nr = len(result_shape)
        res_strides = [1] * nr
        for i in range(nr - 2, -1, -1):
            res_strides[i] = res_strides[i + 1] * result_shape[i + 1]

        # Find the source data pointer (first pointer arg = X)
        x_ptr_name = None
        for arg in self.graph.args:
            if arg.is_ptr:
                x_ptr_name = arg.name
                break

        # Identity and combine expression (4-way: float/long/ulong/int).
        identity, combine_expr = self._reduce_identity_combine(combine_op, msl_type)

        # Check if input has a tracked broadcast layout. If so, thread `lid`
        # does not hold the value at flat position `lid` — it holds the value
        # at `input_bcast_idx` in the logical N-D tensor. We use that as the
        # write index so shared memory ends up canonically laid out.
        input_bcast_idx = None
        if ssa.operand_ids:
            input_bcast_idx = self._bcast_layout.get(ssa.operand_ids[0])

        # Check if input is an already-staged shared memory array with the
        # same logical shape. If so, reuse to skip the copy.
        input_id = ssa.operand_ids[0] if ssa.operand_ids else None
        existing_shared = None
        if input_id is not None:
            existing_shared = getattr(self, "_shared_mem_descs", {}).get(input_id)

        # Allocate shared memory for the full N-D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage values to shared memory from the already-computed input_var
        if total <= block_size:
            if input_bcast_idx is not None:
                # Non-canonical layout: thread lid holds element at
                # input_bcast_idx. Write accordingly. Multiple threads may
                # map to the same logical position (broadcast redundancy);
                # they all write the same value so last-writer-wins is safe.
                self.kb.raw_line(f"    if (lid < {block_size}u) {shared_name}[{input_bcast_idx}] = {input_var};")
            else:
                self.kb.raw_line(f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
        else:
            if x_ptr_name is None:
                # Fallback — cannot handle without a source pointer
                return
            self.kb.raw_line(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            self.kb.raw_line(f"        {shared_name}[_e] = ({msl_type}){x_ptr_name}[_e];")
            self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Allocate result shared memory
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=result_total)

        # Per-output-position reduction: thread _r reduces along the reduce axis
        axis_size = input_shape[axis]
        self.kb.raw_line(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        self.kb.raw_line(f"        {msl_type} acc = {identity};")

        # Decompose _r into result coords, then build src base offset by
        # inserting a 0 at position axis and combining with src_strides.
        # Skip decomposition for 1D result (trivial case).
        if nr == 0:
            # All axes reduced to a single scalar (impossible here since
            # we require n >= 4 and we only collapse one axis).
            self.kb.raw_line(f"        uint _base = 0u;")
        else:
            # Decompose _r into coords c_0, c_1, ..., c_{nr-1} of the result
            # Then map to source coords: for i < axis, src_c[i] = res_c[i];
            # for i > axis, src_c[i] = res_c[i-1].
            # Base offset (with axis coord = 0) = sum over i != axis of
            #   res_c[...] * src_strides[i]
            parts = []
            for j in range(nr):
                src_i = j if j < axis else j + 1
                rs = res_strides[j]
                ss = src_strides[src_i]
                if rs == 1:
                    coord = f"(_r % {result_shape[j]}u)"
                else:
                    coord = f"((_r / {rs}u) % {result_shape[j]}u)"
                parts.append(f"{coord} * {ss}u")
            self.kb.raw_line(f"        uint _base = {' + '.join(parts)};")

        self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
        self.kb.raw_line(f"            {msl_type} val = {shared_name}[_base + _a * {src_strides[axis]}u];")
        self.kb.raw_line(f"            acc = {combine_expr};")
        self.kb.raw_line(f"        }}")
        self.kb.raw_line(f"        {result_shared}[_r] = acc;")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Readback: each thread computes the result position corresponding to
        # its own coords in the original N-D tensor, with axis d removed.
        # Extract axis coord using src_strides[axis] then derive result index.
        # For each result axis j (mapping back to input axis src_i):
        #   res_coord[j] = (lid_input / src_strides[src_i]) % input_shape[src_i]
        # then result_idx = sum_j res_coord[j] * res_strides[j].
        #
        # `lid_input` is the LOGICAL N-D flat position of thread lid's input
        # element.  For a canonical input it's just ``lid``; for a post-reduce
        # broadcast-layout input it's the prior reduce's readback (stored in
        # `input_bcast_idx`).  Using lid when the input is non-canonical gives
        # threads the wrong result element (bug in chained tl.sort reduces).
        lid_input = input_bcast_idx if input_bcast_idx is not None else "lid"
        result_var = self._next_var("reduced")
        if nr == 0:
            self.kb.raw_line(f"    {msl_type} {result_var} = {result_shared}[0];")
            result_read_idx = None
        else:
            read_parts = []
            for j in range(nr):
                src_i = j if j < axis else j + 1
                ss = src_strides[src_i]
                size_i = input_shape[src_i]
                rs = res_strides[j]
                if ss == 1:
                    coord = f"({lid_input} % {size_i}u)"
                else:
                    coord = f"(({lid_input} / {ss}u) % {size_i}u)"
                if rs == 1:
                    read_parts.append(coord)
                else:
                    read_parts.append(f"{coord} * {rs}u")
            read_idx = " + ".join(read_parts)
            self.kb.raw_line(f"    {msl_type} {result_var} = {result_shared}[{read_idx}];")
            result_read_idx = read_idx
        # Barrier after the broadcast read — guards the result array against a following op's
        # re-stage (the recurring shared-memory-race class; cf. _lower_reduce_2d / _3d / the
        # argminmax fix).
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = result_shape
        # Record the broadcast layout: thread `lid` holds the value at flat
        # index `read_idx` in the reduced result. Downstream ops (reduce,
        # store, make_range rewrite) use this to re-stage data correctly.
        if result_read_idx is not None and nr >= 2:
            self._bcast_layout[ssa.id] = f"({result_read_idx})"
            self._register_bcast_layout_by_native(result_facts, tuple(result_shape), f"({result_read_idx})")

    def _lower_scan(self, ssa: SSAValue):
        """tt.scan → prefix scan via shared memory.

        For 2D tensors: scan along the specified axis.
            axis=1 on (M, N): each row gets an independent prefix scan along N
            axis=0 on (M, N): each column gets an independent prefix scan along M
        Supports forward and reverse scans, single and multi-value combines.

        NB: a fused reduce + scan in one kernel (inductor's `x.sum(1) + x.cumprod(1)[:,-1]`)
        used to silently mis-compute a tail of rows because the reduce's broadcast read of
        the (aliased) shared array raced this scan's re-stage of it. That is now fixed by a
        barrier after the reduce broadcast read (_lower_reduce_2d), so reduce+scan computes
        correctly and is no longer refused.
        """
        axis = ssa.attrs.get("axis", 0)
        reverse = ssa.attrs.get("reverse", False)
        n_values = len(ssa.operand_ids)

        if not ssa.operand_ids:
            return
        self._compact_spread_tile_operand(ssa.operand_ids[0], "tt.scan")

        # The wide register-array route is authorized by native per-result
        # metadata, never by a first-result type string or mutable env_types.
        # The legacy <=1024 path remains available while the broader consumer
        # migration proceeds one call site at a time.
        native_contract = getattr(self, "_scan_native_contracts", {}).get(ssa.id)

        # Get the legacy-path input shape. For a proved wide scan, replace it
        # with the native contract below before any dimensions are consumed.
        is_1d = False
        input_shape = _extract_shape(ssa.type_str)
        if not input_shape or len(input_shape) < 2:
            input_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        if native_contract is not None:
            input_shape = native_contract["shape"]
            axis = native_contract["axis"]
        if not input_shape or len(input_shape) < 2:
            # 1D tensor: treat as (1, size) and scan along axis=1
            sz = input_shape[0] if input_shape else self.effective_block_size
            input_shape = (1, sz)
            axis = 1  # 1D scan always scans along the data dimension
            is_1d = True

        M, N = input_shape[0], input_shape[1]
        total = M * N

        # Packet 325: the prescan may have proved an exact flattened
        # multi-element-per-thread ownership map for this whole scan cone.
        # The array path validates every remaining dynamic contract itself;
        # the legacy one-thread-per-element path below stays byte-untouched.
        _arrays = (
            [self.env_array.get(o) for o in ssa.operand_ids]
            if getattr(self, "_scan_mept_shape", None) is not None
            else [None] * n_values
        )
        if any(a is not None for a in _arrays):
            self._lower_scan_array(ssa, input_shape, axis, reverse, n_values, _arrays, native_contract)
            return

        # The scan stages every element through threadgroup memory with one thread
        # per element (`if (lid < total) shared[lid] = ...` below) and then reads it
        # back in the prefix sweep. Metal caps a threadgroup at 1024 threads, so for
        # total > 1024 the elements at index >= 1024 are NEVER written — the sweep
        # then reads uninitialized shared memory and returns silently-wrong values
        # (the store/atomic paths already guard this one-thread-per-element regime).
        # The MSL scan has no multi-element-per-thread path, so refuse loudly.
        if total > 1024:
            from triton_msl.errors import MetalNonRecoverableError

            native_error = getattr(self, "_scan_native_errors", {}).get(ssa.id)
            if native_error is not None:
                raise MetalNonRecoverableError(
                    f"Refusing wide tt.scan because its native per-result metadata "
                    f"contract is unproved: {native_error}",
                    op_name="tt.scan",
                ) from None

            raise MetalNonRecoverableError(
                f"Refusing a {total}-element scan (tl.cumsum / associative_scan): "
                f"Metal allows at most 1024 threads per threadgroup and the scan maps "
                f"one element per thread, so elements past 1024 would be left "
                f"uninitialized and the result silently wrong. Reduce the scan tile to "
                f"<= 1024 elements."
            )

        # PER-SLOT dtypes. A multi-value scan's slots are independent tensors and may
        # differ in type -- upstream's cummax is `associative_scan((value, index_i64))`,
        # a float/int32 value beside an int64 index. Staging every slot with operand-0's
        # dtype truncated the others (re-audit #14: an fp32 sum slot staged as i32 came
        # back all zeros), which is why this used to refuse outright. Give each slot its
        # own staging type instead, so the scan COMPUTES rather than refusing; the
        # truncation the refusal guarded is pinned by tests/test_scan_multivalue_dtypes.py.
        #
        # (A single operand-0 `input_dtype`/`msl_type`/`shared_dtype` used to be computed
        # here; the lists below replaced every use, so those scalars were dead and were
        # removed. 64-bit slots still must not truncate to i32 -- cumsum wrapped at 2^31,
        # re-audit #13 -- which is why each slot goes through _reduce_acc_msl_type.)
        slot_dtypes = [self.env_types.get(o, "fp32") for o in ssa.operand_ids]
        slot_msl, slot_shared = [], []
        for _d in slot_dtypes:
            _m, _s = self._reduce_acc_msl_type(_d)
            slot_msl.append(_m)
            slot_shared.append(_s)

        # Allocate shared memory for each input value, in that slot's own dtype
        shared_names = []
        for i in range(n_values):
            shared_name = f"scan_shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype=slot_shared[i], size=total)
            shared_names.append(shared_name)

        # Write input values to shared memory
        for i, operand_id in enumerate(ssa.operand_ids):
            input_var = self._lookup(operand_id)
            # bf16 has no native MSL type (it lives in a float register), so make the
            # narrowing explicit -- per slot, not by operand 0's dtype.
            cast = f"({slot_msl[i]})" if slot_dtypes[i] == "bf16" else ""
            self.kb.raw_line(f"    if (lid < {total}u) {shared_names[i]}[lid] = {cast}{input_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Compute position and base expressions
        if axis == 1:
            scan_size = N
            pos_expr = f"(lid % {N}u)"
            base_expr = f"((lid / {N}u) * {N}u)"
        else:
            scan_size = M
            pos_expr = f"(lid / {N}u)"
            # For axis=0, elements in same column are at stride N

        # Initialize accumulators with first element of scan group
        acc_vars = []
        for i in range(n_values):
            acc_var = self._next_var("scan_acc")
            acc_vars.append(acc_var)
            if axis == 1:
                if not reverse:
                    init_idx = f"{base_expr}"
                else:
                    init_idx = f"({base_expr} + {N - 1}u)"
            else:
                if not reverse:
                    init_idx = f"(lid % {N}u)"
                else:
                    init_idx = f"({(M - 1)}u * {N}u + (lid % {N}u))"
            self.kb.raw_line(f"    {slot_msl[i]} {acc_var} = ({slot_msl[i]}){shared_names[i]}[{init_idx}];")

        # Emit scan loop
        if not reverse:
            self.kb.raw_line(f"    for (uint scan_j = 1u; scan_j <= {pos_expr}; scan_j++) {{")
        else:
            self.kb.raw_line(f"    for (uint scan_j = 1u; scan_j <= ({scan_size - 1}u - {pos_expr}); scan_j++) {{")

        # Load current elements (rhs) from shared memory
        rhs_vars = []
        for i in range(n_values):
            rhs_var = self._next_var("scan_rhs")
            rhs_vars.append(rhs_var)
            if axis == 1:
                if not reverse:
                    idx_expr = f"{base_expr} + scan_j"
                else:
                    idx_expr = f"{base_expr} + ({N - 1}u - scan_j)"
            else:
                if not reverse:
                    idx_expr = f"scan_j * {N}u + (lid % {N}u)"
                else:
                    idx_expr = f"({M - 1}u - scan_j) * {N}u + (lid % {N}u)"
            self.kb.raw_line(f"        {slot_msl[i]} {rhs_var} = ({slot_msl[i]}){shared_names[i]}[{idx_expr}];")

        # Map block args to accumulator (lhs) and current element (rhs) vars
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids and len(block_arg_ids) >= 2 * n_values:
            for i in range(n_values):
                # Per-slot types so the combine body lowers each slot at its OWN width:
                # cummax's `tl.where(gt, i0, i1)` must compare/select longs for the i64
                # index slot while the value slot stays float/int.
                self.env[block_arg_ids[i]] = acc_vars[i]
                self.env_types[block_arg_ids[i]] = slot_shared[i]
                self.env[block_arg_ids[n_values + i]] = rhs_vars[i]
                self.env_types[block_arg_ids[n_values + i]] = slot_shared[i]

        # Lower body ops (combine function) and find scan.return operands
        scan_return_ids = []
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "tt.scan.return":
                    scan_return_ids = body_op.operand_ids
                else:
                    self._lower_op(body_op)

        # Update accumulators from combine results
        for i in range(n_values):
            if i < len(scan_return_ids):
                new_val = self._lookup(scan_return_ids[i])
                self.kb.raw_line(f"        {acc_vars[i]} = {new_val};")

        self.kb.raw_line(f"    }}")

        # Trailing barrier: every thread must finish READING the scan_shared buffer
        # in the loop above before any thread REUSES it. This is critical when the
        # scan runs inside an scf.for (e.g. two tl.cumsum in a loop): the SAME
        # threadgroup buffer is rewritten at the top of the next iteration, and
        # without this fence a fast thread's write clobbered slots a slow thread was
        # still reading -> non-deterministic wrong results (re-audit #6). The
        # write-back branch below adds its own barrier only when total<block_size,
        # so the total==block_size case had NO trailing fence. Cheap + always safe
        # (emitted in the loop body's uniform control flow, after the scan loop).
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # For 1D scans, subsequent reshape+broadcast needs all threads to access
        # the scan results. Write back to shared memory and read with modular index.
        if is_1d and total < self.effective_block_size:
            self.kb.raw_line(f"    if (lid < {total}u) {shared_names[0]}[lid] = {acc_vars[0]};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            result_var = self._next_var("scan_result")
            self.kb.raw_line(f"    {slot_msl[0]} {result_var} = ({slot_msl[0]}){shared_names[0]}[lid % {total}u];")
            for i in range(1, n_values):
                self.kb.raw_line(f"    if (lid < {total}u) {shared_names[i]}[lid] = {acc_vars[i]};")
            if n_values > 1:
                self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            acc_vars_out = [result_var]
            for i in range(1, n_values):
                rv = self._next_var("scan_result")
                self.kb.raw_line(f"    {slot_msl[i]} {rv} = ({slot_msl[i]}){shared_names[i]}[lid % {total}u];")
                acc_vars_out.append(rv)
        else:
            acc_vars_out = acc_vars

        # Map scan results to output variables
        if ssa.result_ids and len(ssa.result_ids) >= n_values:
            for i in range(n_values):
                self.env[ssa.result_ids[i]] = acc_vars_out[i]
                self.env_types[ssa.result_ids[i]] = slot_shared[i]
                self.env_shapes[ssa.result_ids[i]] = input_shape
        else:
            self.env[ssa.id] = acc_vars_out[0]
            self.env_types[ssa.id] = slot_shared[0]
            self.env_shapes[ssa.id] = input_shape

    def _lower_scan_array(self, ssa, input_shape, axis, reverse, n_values, arrays, native_contract):
        """Lower a proved wide scan over contiguous flat register ownership.

        Each physical thread owns ``width`` adjacent flattened elements.
        Values are staged once per slot, then a Hillis-Steele sweep replays
        the source region at power-of-two distances.  All shape, ownership,
        type, arity, uniformity and total-threadgroup-memory checks fail
        closed here before an incomplete shader can be emitted.
        """
        from triton_msl.errors import MetalNonRecoverableError

        if native_contract is None:
            raise MetalNonRecoverableError(
                "tt.scan register-array lowering requires a native per-result contract",
                op_name="tt.scan",
            )

        M, N = input_shape[0], input_shape[1]
        total = M * N
        width = getattr(self, "_scan_mept_width", 0)
        dispatch = self.kb.block_size
        if width <= 1 or dispatch * width != total:
            raise MetalNonRecoverableError(
                f"tt.scan: {total} logical elements are not an exact "
                f"{dispatch}-thread x {width}-slot flattened cover; refusing.",
                op_name="tt.scan",
            )
        proved_shape = tuple(getattr(self, "_scan_mept_shape", ()))
        shape_matches = tuple(input_shape) == proved_shape or (
            len(proved_shape) == 1 and tuple(input_shape) == (1, proved_shape[0])
        )
        if not shape_matches:
            raise MetalNonRecoverableError(
                "tt.scan: the operand shape differs from the shape whose flat ownership was proved; refusing.",
                op_name="tt.scan",
            )
        if axis not in (0, 1):
            raise MetalNonRecoverableError(
                f"tt.scan: wide rank-2 scan axis {axis} is unsupported; refusing.",
                op_name="tt.scan",
            )

        # Every tensor slot must either carry the exact register-array width,
        # or be a syntactic splat (uniform across every logical element).
        for operand, array in zip(ssa.operand_ids, arrays):
            if array is not None and array[1] != width:
                raise MetalNonRecoverableError(
                    "tt.scan: operand register arrays disagree on the proved per-thread width; refusing.",
                    op_name="tt.scan",
                )
            if array is None and operand not in self._is_splat:
                raise MetalNonRecoverableError(
                    "tt.scan: a non-array operand is not proven uniform across the tile; refusing.",
                    op_name="tt.scan",
                )

        block_args = list((ssa.attrs or {}).get("block_arg_ids") or [])
        returns = [b for b in (ssa.region_ops or []) if b.op == "tt.scan.return"]
        if len(block_args) != 2 * n_values or len(returns) != 1:
            raise MetalNonRecoverableError(
                "tt.scan: combine-region block-argument/terminator arity is not exact; refusing.",
                op_name="tt.scan",
            )
        return_ids = list(returns[0].operand_ids or [])
        if len(return_ids) != n_values:
            raise MetalNonRecoverableError(
                "tt.scan: combine-region return arity does not match its slots; refusing.",
                op_name="tt.scan",
            )

        known_dtypes = {
            "fp16",
            "bf16",
            "fp32",
            "fp64",
            "i1",
            "i8",
            "u8",
            "ui8",
            "i16",
            "u16",
            "ui16",
            "i32",
            "u32",
            "ui32",
            "i64",
            "u64",
            "ui64",
        }
        slot_dtypes = []
        slot_msl = []
        slot_shared = []
        for dtype in native_contract["slot_dtypes"]:
            if dtype not in known_dtypes:
                raise MetalNonRecoverableError(
                    f"tt.scan: slot dtype {dtype!r} has no proved staging width; refusing.",
                    op_name="tt.scan",
                )
            msl_type, shared_type = self._reduce_acc_msl_type(dtype)
            slot_dtypes.append(dtype)
            slot_msl.append(msl_type)
            slot_shared.append(shared_type)

        # Static threadgroup arrays coexist for the entire kernel.  Account
        # for arrays already declared by earlier operations and for alignment
        # padding; equality is refused because the 32 KiB limit leaves no
        # margin for an unmodelled compiler allocation.
        type_layout = {
            "bool": (1, 1),
            "char": (1, 1),
            "uchar": (1, 1),
            "i1": (1, 1),
            "i8": (1, 1),
            "u8": (1, 1),
            "ui8": (1, 1),
            "half": (2, 2),
            "short": (2, 2),
            "ushort": (2, 2),
            "fp16": (2, 2),
            "i16": (2, 2),
            "u16": (2, 2),
            "ui16": (2, 2),
            "float": (4, 4),
            "int": (4, 4),
            "uint": (4, 4),
            "fp32": (4, 4),
            "i32": (4, 4),
            "u32": (4, 4),
            "ui32": (4, 4),
            "long": (8, 8),
            "ulong": (8, 8),
            "double": (8, 8),
            "fp64": (8, 8),
            "i64": (8, 8),
            "u64": (8, 8),
            "ui64": (8, 8),
        }

        def _allocation_bytes(entries):
            cursor = 0
            for _name, dtype, count in entries:
                layout = type_layout.get(dtype)
                if layout is None or not isinstance(count, int) or count < 0:
                    raise MetalNonRecoverableError(
                        f"tt.scan: cannot prove threadgroup allocation for {dtype!r}[{count!r}]; refusing.",
                        op_name="tt.scan",
                    )
                size, alignment = layout
                cursor = ((cursor + alignment - 1) // alignment) * alignment
                cursor += size * count
            return cursor

        prospective = list(getattr(self.kb, "_threadgroup_arrays", []))
        prospective.extend((f"scan-slot-{i}", slot_shared[i], total) for i in range(n_values))
        allocation = _allocation_bytes(prospective)
        if allocation >= 32768:
            raise MetalNonRecoverableError(
                f"Refusing a {total}-element scan: all live static threadgroup "
                f"arrays need {allocation} bytes, meeting or exceeding the 32 KiB "
                "budget. Reduce the tile or slot count.",
                op_name="tt.scan",
            )

        shared_names = []
        for i in range(n_values):
            name = f"scan_shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(name, dtype=slot_shared[i], size=total)
            shared_names.append(name)

        for i, array in enumerate(arrays):
            cast = f"({slot_msl[i]})" if slot_dtypes[i] == "bf16" else ""
            source = array[0] if array is not None else self._lookup(ssa.operand_ids[i])
            for k in range(width):
                value = f"{source}[{k}]" if array is not None else source
                self.kb.raw_line(f"    {shared_names[i]}[lid * {width}u + {k}u] = {cast}{value};")
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        if axis == 1:
            length, stride = N, 1
            position = lambda index: f"({index} % {N}u)"
        else:
            length, stride = M, N
            position = lambda index: f"({index} / {N}u)"

        temporary = [self._next_var("scan_tmp") for _ in range(n_values)]
        for i in range(n_values):
            self.kb.raw_line(f"    {slot_msl[i]} {temporary[i]}[{width}];")
        left = [self._next_var("scan_lhs") for _ in range(n_values)]
        right = [self._next_var("scan_rhs") for _ in range(n_values)]

        self.kb.raw_line(f"    for (uint scan_d = 1u; scan_d < {length}u; scan_d <<= 1u) {{")
        self.kb.raw_line(f"        for (uint scan_k = 0u; scan_k < {width}u; scan_k++) {{")
        self.kb.raw_line(f"            uint scan_idx = lid * {width}u + scan_k;")
        pos = position("scan_idx")
        if reverse:
            self.kb.raw_line(f"            bool scan_has = ({length - 1}u - {pos}) >= scan_d;")
            self.kb.raw_line(f"            uint scan_src = scan_idx + scan_d * {stride}u;")
        else:
            self.kb.raw_line(f"            bool scan_has = {pos} >= scan_d;")
            self.kb.raw_line(f"            uint scan_src = scan_idx - scan_d * {stride}u;")
        for i in range(n_values):
            current = self._next_var("scan_cur")
            neighbour = self._next_var("scan_neighbour")
            self.kb.raw_line(f"            {slot_msl[i]} {current} = ({slot_msl[i]}){shared_names[i]}[scan_idx];")
            self.kb.raw_line(
                f"            {slot_msl[i]} {neighbour} = scan_has ? "
                f"({slot_msl[i]}){shared_names[i]}[scan_src] : {current};"
            )
            # Reverse means scan the reversed sequence and flip the result
            # back.  At either direction's Hillis-Steele step the earlier
            # value in traversal order is therefore the neighbour.  This is
            # observable for associative but noncommutative combiners such as
            # get-first (reverse must repeat the original row's last value).
            self.kb.raw_line(f"            {slot_msl[i]} {left[i]} = {neighbour};")
            self.kb.raw_line(f"            {slot_msl[i]} {right[i]} = {current};")
            self.kb.raw_line(f"            {temporary[i]}[scan_k] = {current};")
        self.kb.raw_line("            if (scan_has) {")
        for i in range(n_values):
            self.env[block_args[i]] = left[i]
            self.env_types[block_args[i]] = slot_dtypes[i]
            self.env[block_args[n_values + i]] = right[i]
            self.env_types[block_args[n_values + i]] = slot_dtypes[i]
        for body_op in ssa.region_ops or []:
            if body_op.op != "tt.scan.return":
                self._lower_op(body_op)
        for i, return_id in enumerate(return_ids):
            self.kb.raw_line(f"                {temporary[i]}[scan_k] = {self._lookup(return_id)};")
        self.kb.raw_line("            }")
        self.kb.raw_line("        }")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        for i in range(n_values):
            self.kb.raw_line(
                f"        for (uint scan_k = 0u; scan_k < {width}u; scan_k++) "
                f"{shared_names[i]}[lid * {width}u + scan_k] = {temporary[i]}[scan_k];"
            )
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line("    }")

        outputs = []
        for i in range(n_values):
            expressions = [f"({slot_msl[i]}){shared_names[i]}[lid * {width}u + {k}u]" for k in range(width)]
            outputs.append(self._var_array("scan_res", expressions, slot_msl[i]))
        self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        result_ids = list(ssa.result_ids or [])
        if not result_ids and n_values == 1:
            result_ids = [ssa.id]
        if len(result_ids) != n_values:
            raise MetalNonRecoverableError(
                "tt.scan: result arity does not match the proved slot count; refusing.",
                op_name="tt.scan",
            )
        for i, result_id in enumerate(result_ids):
            self.env[result_id] = outputs[i]
            self.env_array[result_id] = (outputs[i], width, slot_msl[i])
            self.env_types[result_id] = slot_dtypes[i]
            self.env_shapes[result_id] = input_shape

    # -- Shared memory ops (ttg.local_alloc / ttg.local_load) --
