"""Fail-closed lowering for batch-separable rank-3 ``tt.dot`` kernels.

Round 3 starts with the B in {1, 2, 4, 8} forms emitted by Triton's upstream
``test_dot3d``.
The detector proves the complete graph, each operand/store address, and the
literal-zero accumulator before the template replays the operation.  A rank-3
shape alone is never enough to select this path.
"""

from triton_msl.codegen.msl_emitter import _sanitize_msl_name
from triton_msl.codegen.msl_types import triton_type_to_msl


class _BatchedDotMixin:
    def _batched_dot_dtype(self, value_id, *, pointer=False):
        """Use one checked native dtype authority for admission and emission.

        A legacy element label is not evidence of the native width or of a
        pointer's pointee. Check the complete representation before handing
        its spelling to the existing fragment/type emitters.
        """
        from triton_msl.errors import MetalNonRecoverableError

        facts = self._native_value_facts(value_id, op_name="tt.dot")
        if pointer:
            if (
                facts.kind != "pointer"
                or facts.is_tensor
                or facts.shape != ()
                or facts.address_space != 1
                or facts.pointee is None
            ):
                raise MetalNonRecoverableError(
                    "batched-dot native dtype requires a scalar device pointer",
                    op_name="tt.dot",
                )
            facts = facts.pointee
            if facts.is_tensor or facts.shape != () or facts.unknown_reason is not None:
                raise MetalNonRecoverableError(
                    "batched-dot native dtype has an invalid pointee",
                    op_name="tt.dot",
                )
        signatures = {
            "f16": ("float", 16, None),
            "bf16": ("float", 16, None),
            "f32": ("float", 32, None),
            "f64": ("float", 64, None),
            "i1": ("integer", 1, None),
            "i8": ("integer", 8, None),
            "i16": ("integer", 16, None),
            "i32": ("integer", 32, None),
            "i64": ("integer", 64, None),
        }
        if signatures.get(facts.elem) != (facts.kind, facts.width, facts.signed):
            raise MetalNonRecoverableError(
                "batched-dot native dtype is unsupported or contradictory",
                op_name="tt.dot",
            )
        return facts.elem

    _BATCHED_DOT_ALLOWED_OPS = frozenset(
        {
            "arith.constant",
            "arith.addi",
            "arith.muli",
            "tt.get_program_id",
            "tt.make_range",
            "tt.splat",
            "tt.expand_dims",
            "tt.broadcast",
            "tt.addptr",
            "tt.load",
            "ttg.local_alloc",
            "ttg.local_load",
            "tt.dot",
            "ttg.convert_layout",
            "tt.store",
            "tt.return",
        }
    )

    @staticmethod
    def _batched_dot_all_ops(ops):
        for op in ops:
            yield op
            if op.region_ops:
                yield from _BatchedDotMixin._batched_dot_all_ops(op.region_ops)
            if op.else_ops:
                yield from _BatchedDotMixin._batched_dot_all_ops(op.else_ops)

    def _batched_dot_address(self, ptr_id, arg, op_by_id, *, axis_specs):
        """Prove one canonical rank-3 address and return its three strides.

        ``axis_specs`` maps physical tensor axis to the exact index grammar
        replayed by the template: make-range bounds, pid axis/coefficient, and
        whether the size-one batch term may have been folded away.
        """

        arg_by_id = {a.id: a for a in self.graph.args}

        def _peel(value_id):
            op = op_by_id.get(value_id)
            while op is not None and op.op in ("tt.broadcast", "ttg.convert_layout") and op.operand_ids:
                value_id = op.operand_ids[0]
                op = op_by_id.get(value_id)
            return value_id, op

        terms = []
        current_id, current = ptr_id, op_by_id.get(ptr_id)
        depth = 0
        while current is not None and current.op == "tt.addptr" and len(current.operand_ids or []) == 2:
            if depth >= 12:
                return None
            depth += 1
            terms.append(current.operand_ids[1])
            current_id, current = _peel(current.operand_ids[0])
        if not terms:
            return None
        if current is None:
            if current_id != arg.id:
                return None
        elif not (current.op == "tt.splat" and current.operand_ids and current.operand_ids[0] == arg.id):
            # This rejects residual base arithmetic hidden below a splat, such
            # as ``A + 1`` or ``A + runtime_offset``.
            return None

        found = {}
        for term_id in terms:
            index_id, term = _peel(term_id)
            if term is None:
                return None
            stride = "1"
            if term.op in ("arith.muli", "arith.mul") and len(term.operand_ids or []) == 2:
                sides = [_peel(value_id) for value_id in term.operand_ids]
                stride_side = next(
                    (
                        i
                        for i, (_sid, side) in enumerate(sides)
                        if side is not None
                        and side.op == "tt.splat"
                        and side.operand_ids
                        and side.operand_ids[0] in arg_by_id
                        and not arg_by_id[side.operand_ids[0]].is_ptr
                    ),
                    None,
                )
                if stride_side is None:
                    return None
                stride_arg = arg_by_id[sides[stride_side][1].operand_ids[0]]
                if self._batched_dot_dtype(stride_arg.id) != "i32":
                    # The emitted scalar ABI is an MSL ``int``.  Accepting an
                    # i64 source stride would truncate it before address
                    # replay, so keep that surface fail-closed.
                    return None
                stride = stride_arg.name
                index_id = sides[1 - stride_side][0]
                term = sides[1 - stride_side][1]

            shape = self._native_shape(term.id, op_name="tt.dot")
            if len(shape) != 3:
                return None
            varying = [i for i, extent in enumerate(shape) if extent != 1]
            if len(varying) != 1:
                # A size-one batch term has shape 1x1x1 and Triton normally
                # eliminates it.  Do not guess which axis an explicit scalar
                # term meant; only the complete absence allowed below is safe.
                return None
            axis = varying[0]
            spec = axis_specs.get(axis)
            if spec is None or axis in found or shape[axis] != spec["extent"]:
                return None
            index = self._index_shape(index_id, op_by_id, arg_by_id, None)
            if index["bad"] is not None or index["mod"] is not None or index["pid_div"] is not None:
                return None
            if index["range"] != 1 or index["iv"] != 0 or index["bounds"] != (0, spec["extent"]):
                return None
            if index["pid"] != spec["pid"]:
                return None
            if spec["pid"] is None:
                if index["coef"] is not None:
                    return None
            elif index["coef"] != spec["coef"]:
                return None
            found[axis] = stride

        for axis, spec in axis_specs.items():
            if axis not in found and not spec.get("folded_unit_axis", False):
                return None
        return tuple(found.get(axis, "1") for axis in range(3))

    @staticmethod
    def _batched_dot_numel(shape):
        total = 1
        for extent in shape:
            total *= extent
        return total

    def _flat_contiguous_address(self, ptr_id, arg, op_by_id, total):
        """Prove ``addptr(splat(arg), make_range(0, total))`` exactly."""

        addptr = op_by_id.get(ptr_id)
        if addptr is None or addptr.op != "tt.addptr" or len(addptr.operand_ids or []) != 2:
            return None
        base = op_by_id.get(addptr.operand_ids[0])
        index = op_by_id.get(addptr.operand_ids[1])
        if not (
            base is not None
            and base.op == "tt.splat"
            and base.operand_ids == [arg.id]
            and index is not None
            and index.op == "tt.make_range"
        ):
            return None
        try:
            bounds = (int(index.attrs.get("start")), int(index.attrs.get("end")))
        except (TypeError, ValueError):
            return None
        if bounds != (0, total) or self._native_shape(index.id, op_name="tt.dot") != (total,):
            return None
        return {addptr.id, base.id, index.id}

    def _detect_flattened_batched_dot(self):
        """Prove the rank-3--6 contiguous reshape form from ``test_dot_multidim``.

        This is deliberately a whole-graph recognizer.  It accepts only two
        flat contiguous bf16 loads, value-preserving reshapes, optional exact
        last-two-axis transposes, a literal-zero bf16 dot producing fp32, and
        one flat contiguous store.  Every top-level operation must be claimed;
        any epilogue, residual pointer arithmetic, or extra consumer declines
        to the pre-existing loud refusal.
        """

        if any(op.region_ops or op.else_ops for op in self.graph.ops):
            return None
        ops = list(self.graph.ops)
        op_by_id = {op.id: op for op in ops}
        dots = [op for op in ops if op.op == "tt.dot"]
        loads = [op for op in ops if op.op == "tt.load"]
        stores = [op for op in ops if op.op == "tt.store"]
        allocs = [op for op in ops if op.op == "ttg.local_alloc"]
        if not (len(dots) == 1 and len(loads) == 2 and len(stores) == 1 and len(allocs) == 2):
            return None
        dot = dots[0]
        if len(dot.operand_ids or []) != 3:
            return None
        a_shape = self._native_shape(dot.operand_ids[0], op_name="tt.dot")
        b_shape = self._native_shape(dot.operand_ids[1], op_name="tt.dot")
        c_shape = self._native_shape(dot.id, op_name="tt.dot")
        if (
            len(a_shape) != 3
            or a_shape != b_shape
            or c_shape != a_shape
            or a_shape[0] not in (2, 4, 8, 16)
            or a_shape[1:] != (32, 32)
        ):
            return None
        batch = a_shape[0]
        total = batch * 32 * 32
        ptr_args = [arg for arg in self.graph.args if arg.is_ptr]
        if len(ptr_args) != 3 or len(self.graph.args) != 3:
            return None
        claimed = {dot.id}

        zero = op_by_id.get(dot.operand_ids[2])
        if zero is None or zero.op != "arith.constant":
            return None
        try:
            if float(zero.attrs.get("value")) != 0.0:
                return None
        except (TypeError, ValueError):
            return None
        claimed.add(zero.id)

        original_rank = {2: 3, 4: 4, 8: 5, 16: 6}[batch]
        original_shape = (2,) * (original_rank - 2) + (32, 32)
        permitted_shapes = {(total,), (batch, 32, 32), original_shape}

        def _operand_path(value_id):
            local_load = op_by_id.get(value_id)
            if local_load is None or local_load.op != "ttg.local_load" or len(local_load.operand_ids or []) != 1:
                return None
            alloc = op_by_id.get(local_load.operand_ids[0])
            if alloc is None or alloc.op != "ttg.local_alloc" or len(alloc.operand_ids or []) != 1:
                return None
            path_claimed = {local_load.id, alloc.id}
            current_id = alloc.operand_ids[0]
            transposed = False
            saw_reshape = False
            for _ in range(6):
                current = op_by_id.get(current_id)
                if current is None:
                    return None
                if current.op == "tt.reshape":
                    if len(current.operand_ids or []) != 1:
                        return None
                    src_shape = self._native_shape(current.operand_ids[0], op_name="tt.dot")
                    dst_shape = self._native_shape(current.id, op_name="tt.dot")
                    if (
                        src_shape not in permitted_shapes
                        or dst_shape not in permitted_shapes
                        or self._batched_dot_numel(src_shape) != total
                        or self._batched_dot_numel(dst_shape) != total
                    ):
                        return None
                    saw_reshape = True
                    path_claimed.add(current.id)
                    current_id = current.operand_ids[0]
                    continue
                if current.op == "tt.trans":
                    if transposed or len(current.operand_ids or []) != 1:
                        return None
                    src_shape = self._native_shape(current.operand_ids[0], op_name="tt.dot")
                    dst_shape = self._native_shape(current.id, op_name="tt.dot")
                    rank = len(src_shape)
                    order = self._parse_trans_order(current, rank)
                    expected = list(range(rank - 2)) + [rank - 1, rank - 2]
                    if (
                        rank < 3
                        or src_shape not in permitted_shapes
                        or dst_shape not in permitted_shapes
                        or order != expected
                    ):
                        return None
                    transposed = True
                    path_claimed.add(current.id)
                    current_id = current.operand_ids[0]
                    continue
                if current.op != "tt.load" or len(current.operand_ids or []) != 1:
                    return None
                if not saw_reshape or self._native_shape(current.id, op_name="tt.dot") != (total,):
                    return None
                matches = []
                for arg in ptr_args:
                    address_claimed = self._flat_contiguous_address(current.operand_ids[0], arg, op_by_id, total)
                    if address_claimed is not None:
                        matches.append((arg, address_claimed))
                if len(matches) != 1:
                    return None
                arg, address_claimed = matches[0]
                path_claimed.add(current.id)
                path_claimed.update(address_claimed)
                return arg, transposed, path_claimed
            return None

        a_path = _operand_path(dot.operand_ids[0])
        b_path = _operand_path(dot.operand_ids[1])
        if a_path is None or b_path is None:
            return None
        a_arg, trans_a, a_claimed = a_path
        b_arg, trans_b, b_claimed = b_path
        if (
            a_arg.id == b_arg.id
            or self._batched_dot_dtype(a_arg.id, pointer=True) != "bf16"
            or self._batched_dot_dtype(b_arg.id, pointer=True) != "bf16"
        ):
            return None
        if any(self._batched_dot_dtype(load.id) != "bf16" for load in loads):
            return None
        claimed.update(a_claimed)
        claimed.update(b_claimed)

        store = stores[0]
        if len(store.operand_ids or []) != 2:
            return None
        output_args = [arg for arg in ptr_args if arg.id not in (a_arg.id, b_arg.id)]
        if (
            len(output_args) != 1
            or self._batched_dot_dtype(output_args[0].id, pointer=True) != "f32"
            or self._batched_dot_dtype(dot.id) != "f32"
        ):
            return None
        c_arg = output_args[0]
        address_claimed = self._flat_contiguous_address(store.operand_ids[0], c_arg, op_by_id, total)
        if address_claimed is None:
            return None
        claimed.add(store.id)
        claimed.update(address_claimed)

        current_id = store.operand_ids[1]
        saw_output_reshape = False
        for _ in range(4):
            if current_id == dot.id:
                break
            current = op_by_id.get(current_id)
            if current is None or len(current.operand_ids or []) != 1:
                return None
            if current.op == "tt.reshape":
                src_shape = self._native_shape(current.operand_ids[0], op_name="tt.dot")
                dst_shape = self._native_shape(current.id, op_name="tt.dot")
                if saw_output_reshape or src_shape != (batch, 32, 32) or dst_shape != (total,):
                    return None
                saw_output_reshape = True
            elif current.op == "ttg.convert_layout":
                if self._native_shape(current.id, op_name="tt.dot") != (total,):
                    return None
            else:
                return None
            claimed.add(current.id)
            current_id = current.operand_ids[0]
        if current_id != dot.id or not saw_output_reshape:
            return None

        returns = [op for op in ops if op.op == "tt.return"]
        if len(returns) != 1 or returns[0].operand_ids:
            return None
        claimed.add(returns[0].id)
        if claimed != set(op_by_id):
            return None

        a_strides = (1024, 1, 32) if trans_a else (1024, 32, 1)
        b_strides = (1024, 1, 32) if trans_b else (1024, 32, 1)
        return {
            "batch_dims": (batch,),
            "M": 32,
            "N": 32,
            "K": 32,
            "a_arg": a_arg,
            "b_arg": b_arg,
            "c_arg": c_arg,
            "a_strides": a_strides,
            "b_strides": b_strides,
            "c_strides": (1024, 32, 1),
            "pid_tiled": False,
        }

    def _detect_batched_dot(self):
        """Return a complete supported rank-3 dot plan, else ``None``.

        Returning ``None`` preserves the pre-existing loud rank>=3 refusal.
        Flattened leading batch dimensions remain a later Round-3 slice; this
        proof accepts only the exact single leading dimension in the upstream
        ``test_dot3d`` surface.
        """

        flattened = self._detect_flattened_batched_dot()
        if flattened is not None:
            return flattened

        if any(op.region_ops or op.else_ops for op in self.graph.ops):
            return None
        ops = list(self.graph.ops)
        if any(op.op not in self._BATCHED_DOT_ALLOWED_OPS for op in ops):
            return None
        op_by_id = {op.id: op for op in ops}
        dots = [op for op in ops if op.op == "tt.dot"]
        loads = [op for op in ops if op.op == "tt.load"]
        stores = [op for op in ops if op.op == "tt.store"]
        allocs = [op for op in ops if op.op == "ttg.local_alloc"]
        if not (len(dots) == 1 and len(loads) == 2 and len(stores) == 1 and len(allocs) == 2):
            return None
        dot = dots[0]
        if len(dot.operand_ids or []) != 3:
            return None

        a_shape = self._native_shape(dot.operand_ids[0], op_name="tt.dot")
        b_shape = self._native_shape(dot.operand_ids[1], op_name="tt.dot")
        c_shape = self._native_shape(dot.id, op_name="tt.dot")
        if len(a_shape) != 3 or len(b_shape) != 3 or len(c_shape) != 3:
            return None
        batch_dims = a_shape[:-2]
        if (
            len(batch_dims) != 1
            or batch_dims[0] not in (1, 2, 4, 8)
            or b_shape[:-2] != batch_dims
            or c_shape[:-2] != batch_dims
        ):
            return None
        m, k = a_shape[-2:]
        k2, n = b_shape[-2:]
        if (m, n) != (32, 32) or k not in (32, 64) or k2 != k or c_shape[-2:] != (m, n):
            return None
        if not self._acc_init_is_literal_zero(dot.operand_ids[2], op_by_id):
            return None

        ptr_args = [arg for arg in self.graph.args if arg.is_ptr]
        if len(ptr_args) != 3:
            return None
        roles = self._resolve_dot_ptr_roles(dot, ptr_args)
        if roles is None or len(roles) < 3:
            return None
        a_arg, b_arg, c_arg = roles[:3]
        if len({a_arg.id, b_arg.id, c_arg.id}) != 3:
            return None

        def _operand_load(value_id):
            local_load = op_by_id.get(value_id)
            if local_load is None or local_load.op != "ttg.local_load" or len(local_load.operand_ids or []) != 1:
                return None
            alloc = op_by_id.get(local_load.operand_ids[0])
            if alloc is None or alloc.op != "ttg.local_alloc" or len(alloc.operand_ids or []) != 1:
                return None
            load = op_by_id.get(alloc.operand_ids[0])
            if load is None or load.op != "tt.load" or len(load.operand_ids or []) != 1:
                return None
            return load

        a_load = _operand_load(dot.operand_ids[0])
        b_load = _operand_load(dot.operand_ids[1])
        if a_load is None or b_load is None or {a_load.id, b_load.id} != {load.id for load in loads}:
            return None
        a_dtype = self._batched_dot_dtype(a_arg.id, pointer=True)
        b_dtype = self._batched_dot_dtype(b_arg.id, pointer=True)
        if self._batched_dot_dtype(a_load.id) != a_dtype or self._batched_dot_dtype(b_load.id) != b_dtype:
            return None
        if a_dtype != b_dtype or a_dtype not in ("f16", "f32", "i8"):
            return None

        consumers = {}
        for op in ops:
            for operand in op.operand_ids or []:
                consumers.setdefault(operand, []).append(op)
        current = dot.id
        seen = set()
        while current not in seen:
            seen.add(current)
            uses = consumers.get(current, [])
            if len(uses) != 1:
                return None
            use = uses[0]
            if use.op == "tt.store":
                if use is not stores[0] or len(use.operand_ids or []) < 2 or use.operand_ids[1] != current:
                    return None
                break
            if use.op != "ttg.convert_layout" or not use.operand_ids or use.operand_ids[0] != current:
                return None
            current = use.id
        else:
            return None
        c_dtype = self._batched_dot_dtype(c_arg.id, pointer=True)
        if self._batched_dot_dtype(dot.id) != c_dtype:
            return None
        if a_dtype == "i8" and (c_dtype != "i32" or k * 128 * 128 >= 2**24):
            # Apple has no signed-int8 simdgroup fragment here.  Float MMA is
            # nevertheless bit-exact for the admitted K=32/64 envelope: int8
            # values and products are exactly representable, and even the
            # worst-magnitude partial sum remains below 2**24.  Requiring an
            # i32 result also rejects truncating/narrowing store paths.
            return None

        row_spec = {"extent": m, "pid": 0, "coef": m}
        col_spec = {"extent": n, "pid": 1, "coef": n}
        k_spec = {"extent": k, "pid": None, "coef": None}
        batch_spec = {
            "extent": batch_dims[0],
            "pid": None,
            "coef": None,
            "folded_unit_axis": batch_dims[0] == 1,
        }
        a_strides = self._batched_dot_address(
            a_load.operand_ids[0],
            a_arg,
            op_by_id,
            axis_specs={0: batch_spec, 1: row_spec, 2: k_spec},
        )
        b_strides = self._batched_dot_address(
            b_load.operand_ids[0],
            b_arg,
            op_by_id,
            axis_specs={0: batch_spec, 1: k_spec, 2: col_spec},
        )
        c_strides = self._batched_dot_address(
            stores[0].operand_ids[0],
            c_arg,
            op_by_id,
            axis_specs={0: batch_spec, 1: row_spec, 2: col_spec},
        )
        if a_strides is None or b_strides is None or c_strides is None:
            return None

        return {
            "batch_dims": batch_dims,
            "M": m,
            "N": n,
            "K": k,
            "a_arg": a_arg,
            "b_arg": b_arg,
            "c_arg": c_arg,
            "a_strides": a_strides,
            "b_strides": b_strides,
            "c_strides": c_strides,
            "pid_tiled": True,
        }

    def _lower_batched_dot_template(self, info):
        """Emit the proven per-batch 32x32 simdgroup tile."""

        m, n, k = info["M"], info["N"], info["K"]
        batch = 1
        for extent in info["batch_dims"]:
            batch *= extent
        a_arg, b_arg, c_arg = info["a_arg"], info["b_arg"], info["c_arg"]
        acc_frag, in_frag, tg_type, stage_cast, pad = self._simdgroup_frag_for(
            self._batched_dot_dtype(a_arg.id, pointer=True)
        )
        output_type = triton_type_to_msl(self._batched_dot_dtype(c_arg.id, pointer=True))
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        scalar_names = {arg.name for arg in self.graph.args if not arg.is_ptr}
        arg_indices = {arg.id: i for i, arg in enumerate(self.graph.args)}
        arg_name_indices = {arg.name: i for i, arg in enumerate(self.graph.args)}

        def _stride(value):
            if value in scalar_names:
                # Coordinates are uint, but source strides are signed.  A
                # signed 64-bit product preserves negative i32 strides and
                # avoids overflowing a 32-bit coordinate*stride product.
                return f"(long){source_bindings[value]}"
            return f"{int(value)}l"

        def _stride_ref(value):
            if value in arg_name_indices:
                return ("arg", arg_name_indices[value])
            return ("literal", int(value))

        # Rank-3 dots use a 2-D grid and normally take the host-roundtrip
        # launcher, which mirrors only the tensor-view extent at/after its
        # base. Carry the exact address plan to launch time so runtime strides
        # that escape that mirror refuse instead of reading unwritten bytes.
        self._batched_dot_bounds = (
            "batched_dot_host_bounds_v1",
            batch,
            m,
            n,
            k,
            bool(info.get("pid_tiled", True)),
            arg_indices[a_arg.id],
            arg_indices[b_arg.id],
            arg_indices[c_arg.id],
            tuple(_stride_ref(value) for value in info["a_strides"]),
            tuple(_stride_ref(value) for value in info["b_strides"]),
            tuple(_stride_ref(value) for value in info["c_strides"]),
        )

        ab, am, ak = (_stride(value) for value in info["a_strides"])
        bb, bk, bn = (_stride(value) for value in info["b_strides"])
        cb, cm, cn = (_stride(value) for value in info["c_strides"])
        safe_name = _sanitize_msl_name(self.graph.func_name)

        lines = [
            "#include <metal_stdlib>",
            "#include <metal_simdgroup_matrix>",
            "using namespace metal;",
            "",
            f"kernel void {safe_name}(",
        ]
        declarations = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                msl_type = triton_type_to_msl(self._batched_dot_dtype(arg.id, pointer=True))
                const = "const " if arg.id != c_arg.id else ""
                declarations.append(f"    device {const}{msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                declarations.append(f"    device const int* {source_bindings[arg.name]}_buf [[buffer({i})]]")
        lines.append(",\n".join(declarations) + ",")
        lines.extend(
            [
                "    uint3 pid3 [[threadgroup_position_in_grid]],",
                "    uint sgitg [[simdgroup_index_in_threadgroup]],",
                "    uint tiitg [[thread_index_in_threadgroup]]",
                ") {",
                "    // round3_batched_dot: complete graph and address contract proved",
            ]
        )
        for arg in self.graph.args:
            if not arg.is_ptr:
                lines.append(f"    int {source_bindings[arg.name]} = {source_bindings[arg.name]}_buf[0];")
        if info.get("pid_tiled", True):
            row_base = f"pid3.x * {m}u"
            col_base = f"pid3.y * {n}u"
        else:
            row_base = "0u"
            col_base = "0u"
        lines.extend(
            [
                f"    uint row_base = {row_base};",
                f"    uint col_base = {col_base};",
                f"    threadgroup {tg_type} tg_A[{m * 8}];",
                f"    threadgroup {tg_type} tg_B[{8 * n}];",
                "    threadgroup float tg_store[4u * 64u];",
                "    uint laneid = tiitg % 32u;",
                f"    for (uint batch = 0u; batch < {batch}u; batch++) {{",
                f"        {acc_frag} acc0(0), acc1(0), acc2(0), acc3(0);",
                f"        {in_frag} a_frag, b_frag;",
                f"        for (uint kk = 0u; kk < {k}u; kk += 8u) {{",
                f"            for (uint i = tiitg; i < {m * 8}u; i += 128u) {{",
                "                uint r = i / 8u, q = i % 8u;",
                f"                tg_A[i] = {stage_cast}({source_bindings[a_arg.name]}[batch * {ab} + (row_base + r) * {am} + (kk + q) * {ak}]);",
                "            }",
                f"            for (uint i = tiitg; i < {8 * n}u; i += 128u) {{",
                f"                uint q = i / {n}u, c = i % {n}u;",
                f"                tg_B[i] = {stage_cast}({source_bindings[b_arg.name]}[batch * {bb} + (kk + q) * {bk} + (col_base + c) * {bn}]);",
                "            }",
                "            threadgroup_barrier(mem_flags::mem_threadgroup);",
                "            simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);",
                "            simdgroup_load(a_frag, tg_A, 8);",
                "            simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);",
                "            simdgroup_load(a_frag, tg_A + 64u, 8);",
                "            simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);",
                "            simdgroup_load(a_frag, tg_A + 128u, 8);",
                "            simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);",
                "            simdgroup_load(a_frag, tg_A + 192u, 8);",
                "            simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);",
                "            threadgroup_barrier(mem_flags::mem_threadgroup);",
                "        }",
            ]
        )
        for tile, accumulator in enumerate(("acc0", "acc1", "acc2", "acc3")):
            lines.extend(
                [
                    f"        simdgroup_store({accumulator}, tg_store + sgitg * 64u, 8);",
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);",
                    "        for (uint i = laneid; i < 64u; i += 32u) {",
                    f"            uint r = {tile * 8}u + i / 8u;",
                    "            uint c = sgitg * 8u + i % 8u;",
                    f"            {source_bindings[c_arg.name]}[batch * {cb} + (row_base + r) * {cm} + (col_base + c) * {cn}] = {output_type}(tg_store[sgitg * 64u + i]);",
                    "        }",
                    "        threadgroup_barrier(mem_flags::mem_threadgroup);",
                ]
            )
        lines.extend(["    }", "}"])

        self.effective_block_size = 128
        self._used_pid_axes = {0, 1} if info.get("pid_tiled", True) else set()
        return "\n".join(lines)
