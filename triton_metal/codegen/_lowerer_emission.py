"""Low-level MSL line emitters for ``GenericLowerer``.

The ``_emit_*`` methods are generic op-result builders that wrap
``self.kb.raw_line`` with shape/dtype propagation. They are called by the
specialized ``_lower_*`` methods to emit a single MSL statement and update
``self.env``/``self.env_types``.

Lives separately from ``generic_lowerer.py`` so that the lowerer\'s op
dispatch isn\'t intermixed with the boilerplate that turns a single
binary/unary/cast op into a line of MSL.
"""

from triton_metal.codegen.mlir_walker import SSAValue, _extract_shape
from triton_metal.codegen.msl_emitter import _msl_compute_type
from triton_metal.codegen.msl_types import triton_type_to_msl

from triton_metal.codegen._lowerer_helpers import (
    _UINT_TYPE_MAP,
    _mlir_to_triton_dtype,
    _msl_int_type,
)


class _EmissionMixin:
    """Low-level MSL emit helpers used by ``GenericLowerer``."""

    def _emit_binary(self, ssa: SSAValue, op_str: str, force_unsigned=False):
        """Emit a binary operation: result = a op b.

        When one operand is a shared-memory-backed array (total > block_size,
        from an oversized iter_arg), emits a cooperative strided loop that
        updates the array in-place.  The non-array operand is treated as a
        per-row broadcast: its per-thread value is stored to a temporary
        shared array indexed by row, then read back per-element in the
        strided loop.
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        bs = self.effective_block_size

        # Check if either operand is a shared-memory-backed oversized array.
        smem_descs = getattr(self, '_shared_mem_descs', {})
        a_smem = smem_descs.get(ssa.operand_ids[0])
        b_smem = smem_descs.get(ssa.operand_ids[1])

        smem_info = None  # (smem_name, shape, other_var, other_id)
        if a_smem:
            shape_a = a_smem[1]
            total_a = 1
            for d in shape_a:
                total_a *= d
            if total_a > bs:
                smem_info = (a_smem[0], shape_a, b, ssa.operand_ids[1])
        if smem_info is None and b_smem:
            shape_b = b_smem[1]
            total_b = 1
            for d in shape_b:
                total_b *= d
            if total_b > bs:
                smem_info = (b_smem[0], shape_b, a, ssa.operand_ids[0])

        if smem_info is not None:
            smem_name, shape, other_var, other_id = smem_info
            M = shape[0] if len(shape) >= 2 else 1
            N = shape[1] if len(shape) >= 2 else shape[0]
            total = M * N

            # Determine if other_var is per-row broadcast (from expand_dims
            # of a 1D vector) vs truly per-element.  Per-row values have the
            # same value for all columns in a row.  We check env_shapes: if
            # the original shape was (M, 1) or (M,), it's per-row.
            other_shape = self.env_shapes.get(other_id, ())
            is_per_row = (
                (len(other_shape) == 1 and other_shape[0] == M)
                or (len(other_shape) >= 2 and other_shape[1] == 1)
                or (len(other_shape) >= 2 and other_shape == shape
                    and total > bs)
            )

            if is_per_row or total > bs:
                # Store the per-row value into temp shared memory so the
                # strided loop can access any row's value.
                row_stride = max(1, bs // M)
                temp_smem = f"smem_bcast_{self._shared_counter}"
                self._shared_counter += 1
                self.kb.declare_threadgroup_array(temp_smem, dtype="fp32",
                                                  size=M)
                # Each thread writes its row's value (many threads per row
                # write the same value — harmless).
                self.kb.raw_line(
                    f"    {temp_smem}[lid / {row_stride}u] = {other_var};")
                self.kb.raw_line(
                    f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Strided in-place update
                self.kb.raw_line(
                    f"    for (uint _sb = lid; _sb < {total}u; "
                    f"_sb += {bs}u) {{")
                self.kb.raw_line(
                    f"        {smem_name}[_sb] = {smem_name}[_sb] "
                    f"{op_str} {temp_smem}[_sb / {N}u];")
                self.kb.raw_line(f"    }}")
                self.kb.raw_line(
                    f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

                # Result is the updated smem array
                self.env[ssa.id] = smem_name
                self.env_types[ssa.id] = "fp32"
                self._propagate_shape_elementwise(ssa)
                self._shared_mem_descs[ssa.id] = (smem_name, shape, "fp32")
                return

        var_name = self._next_var("r")
        is_float = self._is_float_op(ssa)
        if is_float:
            ty = "float"
            dtype = "fp32"
        elif force_unsigned:
            # Use the correct unsigned width from elem_type
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
        else:
            # Use the correct signed width from elem_type
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
        if force_unsigned and not is_float:
            # Cast operands to the correct unsigned type for unsigned semantics
            unsigned_ty, _ = _msl_int_type(ssa.elem_type, unsigned=True)
            self.kb.raw_line(f"    {ty} {var_name} = ({unsigned_ty}){a} {op_str} ({unsigned_ty}){b};")
        else:
            self.kb.raw_line(f"    {ty} {var_name} = {a} {op_str} {b};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: element-wise binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)
        # Mask propagation: when an i1-producing op combines two masks
        # (arith.andi / ori / xori on i1), the result is itself a mask.
        # Without this, tt.load / tt.store downstream can't recognize
        # `rmask & xmask` as a mask and emit unmasked memory ops.
        if ssa.elem_type == "i1" and ssa.op in (
            "arith.andi", "arith.ori", "arith.xori"
        ):
            self.env_is_mask[ssa.id] = True
        # Splat-ness propagates when both operands are splat.
        if (ssa.operand_ids
            and ssa.operand_ids[0] in self._is_splat
            and ssa.operand_ids[1] in self._is_splat):
            self._is_splat.add(ssa.id)
        self._propagate_bcast_layout_binary(ssa)


    def _emit_unary(self, ssa: SSAValue, op_str: str):
        """Emit a unary operation: result = op(a)."""
        if not ssa.operand_ids:
            return
        a = self._lookup(ssa.operand_ids[0])
        var_name = self._next_var("r")
        self.kb.raw_line(f"    float {var_name} = {op_str}{a};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: unary inherits shape from its operand
        self._propagate_shape_elementwise(ssa)


    def _emit_builtin_binary(self, ssa: SSAValue, fn_name: str, force_unsigned=False):
        """Emit a builtin binary function: result = fn(a, b)."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        is_float = self._is_float_op(ssa)
        if is_float:
            ty = "float"
            dtype = "fp32"
        elif force_unsigned:
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
        else:
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
        if force_unsigned and not is_float:
            unsigned_ty, _ = _msl_int_type(ssa.elem_type, unsigned=True)
            self.kb.raw_line(f"    {ty} {var_name} = {fn_name}(({unsigned_ty}){a}, ({unsigned_ty}){b});")
        elif not is_float:
            # MSL max/min have separate (int, int) and (uint, uint) overloads.
            # Explicit casts avoid ambiguity when a literal `1` is passed to
            # max/min against a uint operand (signed literal vs unsigned var).
            self.kb.raw_line(f"    {ty} {var_name} = {fn_name}(({ty}){a}, ({ty}){b});")
        else:
            self.kb.raw_line(f"    {ty} {var_name} = {fn_name}({a}, {b});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: element-wise builtin binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)
        self._propagate_bcast_layout_binary(ssa)


    def _emit_nan_propagating_minmax(self, ssa: SSAValue, fn_name: str):
        """Emit NaN-propagating min/max: if either operand is NaN, result is NaN."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        self.kb.raw_line(
            f"    float {var_name} = (isnan({a}) || isnan({b})) "
            f"? NAN : {fn_name}({a}, {b});"
        )
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: element-wise binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)
        self._propagate_bcast_layout_binary(ssa)


    def _emit_passthrough(self, ssa: SSAValue):
        """Emit a type conversion that's a no-op in MSL (extf, truncf, etc.)."""
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Propagate shared_mem_descs for smem-backed oversized arrays
            smem_descs = getattr(self, '_shared_mem_descs', {})
            if src_id in smem_descs:
                smem_descs[ssa.id] = smem_descs[src_id]
            # Propagate shape: passthrough preserves shape from source,
            # unless the result type has a different shape (e.g. reshape).
            if ssa.type_str:
                out_shape = _extract_shape(ssa.type_str)
                if out_shape:
                    self.env_shapes[ssa.id] = out_shape
                elif src_id in self.env_shapes:
                    self.env_shapes[ssa.id] = self.env_shapes[src_id]
            elif src_id in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[src_id]
            # Propagate bcast layout: bitcast / trivial-reshape of a value
            # keeps the same per-thread element identity, so the same
            # (lid → flat-index) mapping applies to the result.  This is
            # critical for chained reduces in tl.sort where reshape-to-slice
            # and bitcast occur between reduce and the next reduce.
            if src_id in self._bcast_layout:
                self._bcast_layout[ssa.id] = self._bcast_layout[src_id]
            # Splat-ness survives passthrough.
            if src_id in self._is_splat:
                self._is_splat.add(ssa.id)
            # Phase 4b: MEPT array storage survives passthrough too. The
            # same per-thread array layout applies to the result. Gated
            # on the env_array entry (which only ever appears when a
            # producer ran with TRITON_METAL_MEPT=1), so there's no flag
            # check needed here: absent producers => absent entries.
            if src_id in self.env_array:
                self.env_array[ssa.id] = self.env_array[src_id]


    def _emit_cast(self, ssa: SSAValue, target_type: str, dtype: str = None):
        """Emit a type cast."""
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        # Resolve output dtype once so both array and scalar paths agree.
        if dtype:
            out_dtype = dtype
        elif target_type == "float":
            out_dtype = "fp32"
        else:
            out_dtype = (_mlir_to_triton_dtype(ssa.elem_type)
                         if ssa.elem_type else "i32")
        # Phase 4b: when MEPT is on and the source SSA carries an array,
        # emit per-element cast into a parallel result array.
        if self.mept_enabled and src_id in self.env_array:
            src_name, n, _src_ty = self.env_array[src_id]
            exprs = [f"static_cast<{target_type}>({src_name}[{i}])"
                     for i in range(n)]
            var_name = self._var_array("r", exprs, target_type)
            self.env[ssa.id] = var_name
            self.env_array[ssa.id] = (var_name, n, target_type)
            self.env_types[ssa.id] = out_dtype
            self._propagate_shape_elementwise(ssa)
            if src_id in self._bcast_layout:
                self._bcast_layout[ssa.id] = self._bcast_layout[src_id]
            return
        a = self._lookup(src_id)
        var_name = self._next_var("r")
        self.kb.raw_line(f"    {target_type} {var_name} = static_cast<{target_type}>({a});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = out_dtype
        # Shape: cast preserves shape from source operand
        self._propagate_shape_elementwise(ssa)
        # Propagate bcast layout — elementwise casts preserve the per-thread
        # identity, so the lid → flat-index mapping carries through.
        if src_id in self._bcast_layout:
            self._bcast_layout[ssa.id] = self._bcast_layout[src_id]


    def _emit_uitofp(self, ssa: SSAValue):
        """Emit unsigned-int-to-float conversion.

        Unlike sitofp, we must first cast the source to its unsigned MSL type
        to prevent sign extension. E.g., for i8 value 241 stored as char(-15),
        static_cast<float>(char(-15)) = -15.0 (wrong), but
        static_cast<float>(static_cast<uchar>(char(-15))) = 241.0 (correct).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        a = self._lookup(src_id)
        var_name = self._next_var("r")
        # Determine the source integer type so we can cast to unsigned first
        src_dtype = self.env_types.get(src_id, "i32")
        if src_dtype in _UINT_TYPE_MAP:
            # Source is already tracked as unsigned — direct cast is fine
            # But the MSL variable may still be signed, so always go through unsigned
            src_unsigned_ty, _ = _UINT_TYPE_MAP[src_dtype]
            self.kb.raw_line(
                f"    float {var_name} = static_cast<float>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        elif src_dtype.startswith("u"):
            # Source is already unsigned (u8, u16, etc.) — direct cast is fine
            self.kb.raw_line(f"    float {var_name} = static_cast<float>({a});")
        else:
            # Source is a signed integer type (i8, i16, etc.) — cast via unsigned
            # to get the correct unsigned interpretation
            src_unsigned_ty, _ = _msl_int_type(src_dtype, unsigned=True)
            self.kb.raw_line(
                f"    float {var_name} = static_cast<float>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: uitofp preserves shape
        self._propagate_shape_elementwise(ssa)


    def _emit_int_cast(self, ssa: SSAValue, unsigned: bool = False):
        """Emit an integer sign-extend, zero-extend, or truncation cast.

        Maps the result type from ssa.elem_type to the correct MSL integer
        type and emits a static_cast. This is needed for arith.extsi,
        arith.extui, and arith.trunci which change integer bitwidths.

        For arith.extui (unsigned=True), we must first cast the source to
        its unsigned equivalent before extending, to prevent sign extension.
        E.g., char(-15) as uchar = 241, then uint(241) = 241 (not 4294967281).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        a = self._lookup(src_id)
        # Determine the target type from the MLIR result type
        msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=unsigned)
        var_name = self._next_var("r")

        if unsigned and ssa.op == "arith.extui":
            # For unsigned extension, first cast source to unsigned of same
            # width to prevent sign extension, then extend to target width.
            src_dtype = self.env_types.get(src_id, "i32")
            # Get unsigned version of source type
            src_unsigned_ty, _ = _msl_int_type(src_dtype, unsigned=True)
            self.kb.raw_line(
                f"    {msl_ty} {var_name} = static_cast<{msl_ty}>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        elif ssa.op == "arith.trunci":
            # For integer truncation to narrow types (char, short), the source
            # may actually be a float (e.g. from simd_sum which always returns
            # float). Direct float→char saturates in Metal. Cast through int
            # first: float→int (truncates) → char (wraps modularly).
            if msl_ty in ("char", "short", "uchar", "ushort"):
                self.kb.raw_line(
                    f"    {msl_ty} {var_name} = static_cast<{msl_ty}>"
                    f"(static_cast<int>({a}));")
            else:
                self.kb.raw_line(f"    {msl_ty} {var_name} = static_cast<{msl_ty}>({a});")
        else:
            self.kb.raw_line(f"    {msl_ty} {var_name} = static_cast<{msl_ty}>({a});")

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Propagate ptr/mask info
        if src_id in self.env_is_mask:
            self.env_is_mask[ssa.id] = True
        if src_id in self.env_is_ptr:
            self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
        # Shape: integer cast preserves shape
        self._propagate_shape_elementwise(ssa)
        # Propagate bcast layout across the integer cast — the per-thread
        # element identity is preserved.
        if src_id in self._bcast_layout:
            self._bcast_layout[ssa.id] = self._bcast_layout[src_id]


    def _emit_cond_br_block(self, blocks, block_order, block_idx, result_var, msl_type):
        """Recursively emit a basic block as structured if/else."""
        if block_idx >= len(block_order):
            return

        bid = block_order[block_idx]
        ops = blocks[bid]

        for op in ops:
            if op.op == "cf.cond_br":
                cond_var = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}") if op.operand_ids else "false"

                # Split operand_ids using walker-parsed arg counts
                n_true = op.attrs.get("n_true_operands", 0)
                n_false = op.attrs.get("n_false_operands", 0)
                true_args = op.operand_ids[1:1 + n_true]
                false_args = op.operand_ids[1 + n_true:1 + n_true + n_false]

                remaining_blocks = block_order[block_idx + 1:]

                if not remaining_blocks:
                    return

                if n_true > 0 and n_false > 0:
                    # Both branches pass values (e.g., both go to return block)
                    true_v = self.env.get(true_args[0], f"v{true_args[0]}")
                    false_v = self.env.get(false_args[0], f"v{false_args[0]}")
                    self.kb.raw_line(f"    {result_var} = {cond_var} ? {true_v} : {false_v};")
                elif n_true > 0 and n_false == 0:
                    # True branch passes value (to return block), false falls through
                    true_v = self.env.get(true_args[0], f"v{true_args[0]}")
                    self.kb.raw_line(f"    if ({cond_var}) {{")
                    self.kb.raw_line(f"        {result_var} = {true_v};")
                    self.kb.raw_line(f"    }} else {{")
                    # Recurse into the next block (false destination)
                    if remaining_blocks:
                        # Find the non-return block to recurse into
                        next_bid = remaining_blocks[0]
                        next_block_idx = block_order.index(next_bid)
                        self._emit_cond_br_block(blocks, block_order, next_block_idx, result_var, msl_type)
                    self.kb.raw_line(f"    }}")
                elif n_true == 0 and n_false > 0:
                    # True branch falls through, false passes value
                    false_v = self.env.get(false_args[0], f"v{false_args[0]}")
                    self.kb.raw_line(f"    if (!{cond_var}) {{")
                    self.kb.raw_line(f"        {result_var} = {false_v};")
                    self.kb.raw_line(f"    }} else {{")
                    if remaining_blocks:
                        next_bid = remaining_blocks[0]
                        next_block_idx = block_order.index(next_bid)
                        self._emit_cond_br_block(blocks, block_order, next_block_idx, result_var, msl_type)
                    self.kb.raw_line(f"    }}")
                return

            elif op.op == "cf.br":
                # Unconditional branch — assign args to result and stop
                if op.operand_ids:
                    val = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}")
                    self.kb.raw_line(f"    {result_var} = {val};")
                return

            elif op.op == "tt.map_elementwise.return":
                # Return block — the block arg was set by cf.cond_br assignments
                # Nothing to emit here since result_var was set in the branches
                pass
            else:
                # Process non-terminator ops (e.g., arith.cmpi, arith.subi) in this block
                self._lower_op_dispatch(op)

    # -- Reductions --


