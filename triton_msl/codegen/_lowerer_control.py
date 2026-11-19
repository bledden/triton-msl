"""scf.* control-flow ops + atomic ops for ``GenericLowerer``.

Two op families that share ``self.kb`` block-tracking machinery:

  - ``scf.for`` / ``scf.if`` / ``scf.while`` carry nested regions whose
    bodies need recursive op lowering, plus phi-node rewriting at block
    boundaries (a long-standing source of subtle FA bugs — see
    backend/compiler.py for the wrap-loop phi-rewrite explanation).

  - ``tt.atomic_rmw`` / ``tt.atomic_cas`` synthesize Metal\'s
    ``atomic_*_explicit`` calls or compare-and-swap loops. They aren\'t
    control flow themselves but are equally side-effecting and equally
    rely on careful block scoping.

Mixed into ``GenericLowerer``: the methods recursively call back into
``self._lower_op`` to handle the body, so they need access to the full
op dispatch table, not just a subset.
"""

import re

from triton_msl.codegen.mlir_walker import SSAValue
from triton_msl.codegen.msl_emitter import _msl_compute_type
from triton_msl.codegen.msl_types import triton_type_to_msl

from triton_msl.codegen._lowerer_helpers import _mlir_to_triton_dtype


class _ControlFlowMixin:
    """``scf.*`` and atomic op lowering for ``GenericLowerer``."""

    def _loop_pointer_parts(self, value_id):
        """Resolve an address, never `_lookup` a loaded pointer expression.

        Bare kernel pointer arguments do not enter env_is_ptr until an addptr
        or splat; they are nevertheless valid scalar loop init/yield values.
        """
        info = self.env_is_ptr.get(value_id)
        if info is not None:
            return info
        for arg in self.graph.args:
            if arg.id == value_id and arg.is_ptr:
                return self._lookup(value_id), "0"
        return None

    def _prove_atomic_native_contract(self, ssa: SSAValue):
        """Bind one atomic result to every operand's native representation.

        Atomic emission assumes that the pointer, value/compare operands and
        mask all describe the same logical lanes.  A result-zero ``type_str``
        can neither prove that agreement nor identify a damaged per-result
        record.  Use immutable ResultMeta records only; operation signedness
        remains in ``rmw_op`` because MLIR integer result bits are signless.
        """
        from triton_msl.errors import MetalNonRecoverableError

        op_name = ssa.op
        result_ids = list(ssa.result_ids or [ssa.id])
        if len(result_ids) != 1 or result_ids[0] != ssa.id or len(ssa.operand_ids or []) != 3:
            raise MetalNonRecoverableError(
                "atomic native contract requires exactly one result and three operands",
                op_name=op_name,
            )

        result = self._native_value_facts(ssa.id, op_name=op_name)
        pointer, second, third = [self._native_value_facts(value_id, op_name=op_name) for value_id in ssa.operand_ids]

        shape = result.shape
        tensor_contract = (
            result.is_tensor
            and shape is not None
            and len(shape) > 0
            and all(isinstance(dim, int) and dim > 0 for dim in shape)
            and result.layout is not None
        )
        scalar_contract = not result.is_tensor and shape == () and result.layout is None
        if not (tensor_contract or scalar_contract):
            raise MetalNonRecoverableError(
                "atomic native result does not prove a positive tensor shape or scalar",
                op_name=op_name,
            )

        def same_lanes(facts):
            return facts.is_tensor == result.is_tensor and facts.shape == shape and facts.layout == result.layout

        result_sig = (result.kind, result.elem, result.width, result.signed)
        pointee = pointer.pointee
        if (
            pointer.kind != "pointer"
            or pointer.address_space != 1
            or pointee is None
            or pointee.unknown_reason is not None
            or pointee.is_tensor
            or not same_lanes(pointer)
            or (pointee.kind, pointee.elem, pointee.width, pointee.signed) != result_sig
        ):
            raise MetalNonRecoverableError(
                "atomic native pointer/result representation is contradictory",
                op_name=op_name,
            )

        if ssa.op == "tt.atomic_cas":
            data_operands = (second, third)
        elif ssa.op == "tt.atomic_rmw":
            data_operands = (second,)
            if not same_lanes(third) or (third.kind, third.elem, third.width) != ("integer", "i1", 1):
                raise MetalNonRecoverableError(
                    "atomic native mask representation is contradictory",
                    op_name=op_name,
                )
        else:
            raise MetalNonRecoverableError(f"unsupported atomic native contract for {ssa.op!r}", op_name=op_name)

        for facts in data_operands:
            if not same_lanes(facts) or (facts.kind, facts.elem, facts.width, facts.signed) != result_sig:
                raise MetalNonRecoverableError(
                    "atomic native value/result representation is contradictory",
                    op_name=op_name,
                )
        return result

    def _lower_scf_for(self, ssa: SSAValue):
        """scf.for -> MSL for loop with iter_args.

        scf.for has operands: [start, end, step, init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as iter_args)
        Body block args: [induction_var, iter_arg_0, iter_arg_1, ...]

        For iter_args whose 2D shape total exceeds block_size (e.g. a 32x64
        accumulator with 1024 threads), the value is kept in a persistent
        shared-memory array instead of a per-thread scalar.  Operations on
        those values (arith.mulf, tt.dot, tt.store) use cooperative strided
        loops.  The mapping is tracked in ``_smem_iter_args``.

        For 1-D iter_args carrying more than one element per thread (single-pass
        MEPT, flag-ON only), the value is kept as a per-thread register array
        ``T v[n]`` declared once, seeded from the init value, and updated
        per-element at each yield.  Tracked in ``mept_array_iter_indices`` /
        ``mept_array_iter_n``; the block-arg and result are registered in
        ``env_array`` so the body and the post-loop store read ``v[e]`` (M3a).
        """
        if len(ssa.operand_ids) < 3:
            return

        # i64 loop bounds: the induction lowering assumes 32-bit and does not
        # terminate for 64-bit ranges (the test_for_iv hang). No correct
        # lowering exists yet — refuse loudly rather than hang (Phase 0 T3).
        for bid in ssa.operand_ids[:3]:
            if self.env_types.get(bid) in ("i64", "u64", "ui64"):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "Refusing scf.for with 64-bit loop bounds: the induction "
                    "lowering would not terminate (hang). Use 32-bit bounds."
                )

        start_var = self._lookup(ssa.operand_ids[0])
        end_var = self._lookup(ssa.operand_ids[1])
        step_var = self._lookup(ssa.operand_ids[2])

        # iter_args initial values: operands[3:]
        init_ids = ssa.operand_ids[3:]
        n_iter_args = len(init_ids)

        bs = self.effective_block_size

        # Infer iter_arg types from init values and scf.for result types
        iter_vars = []
        iter_dtypes = []
        # Track which iter_args are oversized and need shared memory
        smem_iter_indices = set()  # indices into iter_vars that are smem-backed
        # MEPT (M3a): indices of iter_args carried as per-thread register
        # arrays ``T v[n]`` (a 1-D multi-element value updated across the
        # loop). Maps index -> (n_elems, msl_type).
        mept_array_iter_indices = set()
        mept_array_iter_n = {}
        # Scalar pointer carries preserve the ACTUAL typed address. Splitting
        # it into base + `uint` offset loses negative signed offsets; keeping a
        # guessed integer offset produced an in-bounds source load as OOB/zero.
        # A loop may also yield another allocation, so a frozen base is unsafe.
        ptr_offset_iter_indices = set()
        ptr_offset_iter_base = {}
        # MEPT (tile > threadgroup) loop-carried pointer: carry a per-thread offset ARRAY
        # ``long off[n]``. Pointer differences are signed; uint would turn a
        # valid negative in-allocation update into an enormous OOB index.
        # index -> (base expr, n).
        ptr_offset_arr_iter_indices = set()
        ptr_offset_arr_iter = {}
        # The scf.for result type tells us the true type of iter_args
        result_elem = ssa.elem_type or "f32"  # First result's type
        for i, init_id in enumerate(init_ids):
            _ptr = self._loop_pointer_parts(init_id)
            if _ptr is not None:
                _base, _off = _ptr
                _address = _base if not _off or _off == "0" else f"({_base} + {_off})"
                _ptr_var = self._next_var("ptr")
                self.kb.raw_line(f"    auto {_ptr_var} = {_address};")
                iter_vars.append(_ptr_var)
                iter_dtypes.append(self._trace_ptr_dtype(init_id))
                ptr_offset_iter_indices.add(i)
                ptr_offset_iter_base[i] = _ptr_var
                continue
            _parr = getattr(self, "env_ptr_array", {}).get(init_id)
            if _parr is not None:
                # MEPT (tile > threadgroup) loop-carried pointer: the offset is a per-thread
                # ARRAY. Carry signed ``long off[n]`` (seeded from the init's offset array); map the
                # block-arg to (base, off, n) so tt.load reads ``base[off[e]]`` and ``p += X``
                # updates ``off[e]`` at the yield. Array analog of the scalar case (#4.2).
                _base, _off_arr, _n = _parr
                _base_var = self._next_var("ptr_base")
                self.kb.raw_line(f"    auto {_base_var} = {_base};")
                _off_var = self._var_array("off", [f"long({_off_arr}[{e}])" for e in range(_n)], "long")
                iter_vars.append(_off_var)
                iter_dtypes.append(self._trace_ptr_dtype(init_id))
                ptr_offset_arr_iter_indices.add(i)
                ptr_offset_arr_iter[i] = (_base_var, _n)
                continue
            init_val = self._lookup(init_id)
            # Prefer result type, fall back to init value type
            init_type = self.env_types.get(init_id, "fp32")
            # Use result type if it's more specific (e.g., i64 vs i32). INTEGER inits only:
            # ``result_elem`` is the FIRST result's element type applied to EVERY iter-arg,
            # so an i64 counter carried FIRST used to "upgrade" a co-carried fp32
            # accumulator to ``long`` — truncating the float every iteration (re-review
            # 2026-08-25 F1: order-dependent silent-wrong, acc_err ~2.8). A float init
            # keeps its own type.
            if result_elem in ("i64",) and init_type == "i32":
                init_type = result_elem

            # Check if this iter_arg is a 2D tensor too large for scalar
            init_shape = self.env_shapes.get(init_id, ())
            init_total = 1
            for d in init_shape:
                init_total *= d
            if len(init_shape) >= 2 and init_total > bs and getattr(self, "_needs_wrapping", False):
                # WRAP regime (GitHub issue #4.4): the whole scf.for body is emitted INSIDE
                # the per-element ``for _loop_e`` loop, so the cooperative smem representation
                # below would re-init + re-accumulate the tile once PER ELEMENT (broken MSL /
                # wrong numbers). The correct representation here is the per-element SCALAR
                # fallback: each ``_loop_e`` iteration is one independent element, so a plain
                # ``float iter_N`` declared inside the wrap loop accumulates exactly that
                # element. That is only valid when the loop uses the accumulator ELEMENTWISE —
                # verify by tainting the block-arg + result and walking all ops: any
                # cross-element consumer (reduce/dot/trans/broadcast/expand_dims/...) of an
                # accumulator-derived value refuses loudly (correct-or-refuse) instead of
                # reaching Metal as a cryptic compile error.
                _ba_ids = ssa.attrs.get("block_arg_ids", [])
                _taint = set()
                if i + 1 < len(_ba_ids):
                    _taint.add(_ba_ids[i + 1])
                if ssa.result_ids and i < len(ssa.result_ids):
                    _taint.add(ssa.result_ids[i])
                elif ssa.id is not None:
                    _taint.add(ssa.id)
                # Ops that keep a value per-element under the wrap loop. arith./math. are
                # elementwise; convert_layout is a layout no-op in this pipeline; addptr/load
                # are lowered per-element; store/yield are the sinks.
                _PASS_OK = {
                    "ttg.convert_layout",
                    "tt.store",
                    "scf.yield",
                    "tt.bitcast",
                    "tt.fp_to_fp",
                    "tt.addptr",
                    "tt.load",
                }

                def _elemwise_taint_walk(ops):
                    for _o in ops:
                        if _o.op == "scf.if":
                            # Walk the branches FIRST (their scf.yields taint via _PASS_OK),
                            # then propagate any tainted branch-yield to the if's results —
                            # otherwise a consumer of the if RESULT escaped the walk
                            # (re-review 2026-08-25 F3; benign today only because every
                            # order-changing consumer also exits the wrap regime).
                            if _o.region_ops and not _elemwise_taint_walk(_o.region_ops):
                                return False
                            if _o.else_ops and not _elemwise_taint_walk(_o.else_ops):
                                return False
                            _yld = [
                                y for y in list(_o.region_ops or []) + list(_o.else_ops or []) if y.op == "scf.yield"
                            ]
                            if any(y.id in _taint or any(x in _taint for x in (y.operand_ids or [])) for y in _yld):
                                _taint.add(_o.id)
                                for _rid in _o.result_ids or []:
                                    _taint.add(_rid)
                            continue
                        if any(x in _taint for x in (_o.operand_ids or [])):
                            if _o.op.startswith(("arith.", "math.")) or _o.op in _PASS_OK:
                                _taint.add(_o.id)
                            elif _o.op == "scf.for":
                                # A NESTED loop carrying the accumulator: its classifier run
                                # diverts the same way (per-element scalar), so taint flows
                                # through — propagate to the matching block-arg + result and
                                # allow. Tainted LOOP BOUNDS (operand 0..2) would be a
                                # cross-element scalar dependency -> refuse.
                                _n_ba = _o.attrs.get("block_arg_ids", []) if _o.attrs else []
                                _n_rid = _o.result_ids or []
                                for _k, _x in enumerate(_o.operand_ids or []):
                                    if _x in _taint:
                                        if _k < 3:
                                            return False  # tainted bound/step
                                        _j = _k - 3
                                        if _j + 1 < len(_n_ba):
                                            _taint.add(_n_ba[_j + 1])
                                        if _j < len(_n_rid):
                                            _taint.add(_n_rid[_j])
                            else:
                                return False
                        if _o.region_ops and not _elemwise_taint_walk(_o.region_ops):
                            return False
                        if _o.else_ops and not _elemwise_taint_walk(_o.else_ops):
                            return False
                    return True

                if not _elemwise_taint_walk(self.graph.ops):
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        f"a {init_shape[0]}x{init_shape[1]} loop-carried accumulator "
                        f"({init_total} elements) with only {bs} threads is consumed by a "
                        "non-elementwise op (reduce/dot/broadcast/...); the fewer-threads-"
                        "than-tile lowering supports elementwise accumulation only. Launch "
                        f"with num_warps = BLOCK/32 so the threadgroup covers the tile, or "
                        "reduce the tile (GitHub issue #4.4).",
                        op_name="scf.for",
                    )
                # Elementwise-safe: FALL THROUGH to the scalar fallback below — a plain
                # per-element scalar inside the wrap loop is the correct lowering.
            elif len(init_shape) >= 2 and init_total > bs:
                # Allocate persistent shared memory for this iter_arg
                smem_name = f"smem_iter_{self._shared_counter}"
                self._shared_counter += 1
                self.kb.declare_threadgroup_array(smem_name, dtype="fp32", size=init_total)
                # Cooperative init from the constant value
                self.kb.raw_line(f"    for (uint _si = lid; _si < {init_total}u; _si += {bs}u) {{")
                self.kb.raw_line(f"        {smem_name}[_si] = {init_val};")
                self.kb.raw_line(f"    }}")
                self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                iter_vars.append(smem_name)
                iter_dtypes.append(init_type)
                smem_iter_indices.add(i)
                # Register in _shared_mem_descs so downstream ops can find it
                if not hasattr(self, "_shared_mem_descs"):
                    self._shared_mem_descs = {}
                # We will register the block_arg_id below after mapping
                continue

            # MEPT register-array iter-arg: a 1-D multi-element value carried
            # across the loop. Each thread owns ``n`` contiguous elements as a
            # mutable register array. Init may be a broadcast scalar (tl.zeros)
            # or already an env_array; declare T v[n] and seed every element.
            # Gated on the single-pass array regime so flag-off / scalar
            # kernels are unaffected.
            if (
                getattr(self, "_mept_single_pass", False)
                and len(init_shape) == 1
                and init_total > bs
                and init_total % bs == 0
            ):
                n = init_total // bs
                if init_type.startswith("f") or init_type.startswith("bf"):
                    msl_type = "float"
                elif init_type in ("i64",):
                    msl_type = "long"
                elif init_type in ("u64", "ui64"):
                    msl_type = "ulong"  # 32-bit uint would truncate (mirrors _lower_scf_if)
                elif init_type.startswith("u"):
                    msl_type = "uint"
                else:
                    msl_type = "int"
                if init_id in self.env_array:
                    src_arr, _src_n, _src_ty = self.env_array[init_id]
                    # Both widths derive from total_elements/num_threads, so they
                    # must agree; refuse loudly if a future producer diverges
                    # rather than emit out-of-bounds src_arr[e] (silent UB).
                    if _src_n != n:
                        from triton_msl.errors import MetalNonRecoverableError

                        raise MetalNonRecoverableError(
                            f"MEPT iter-arg array width mismatch: env_array has "
                            f"{_src_n}, init_total//bs gives {n} (init_id={init_id})"
                        )
                    exprs = [f"{src_arr}[{e}]" for e in range(n)]
                else:
                    exprs = [init_val for _ in range(n)]
                var_name = self._var_array("iter", exprs, msl_type)
                iter_vars.append(var_name)
                iter_dtypes.append(init_type)
                mept_array_iter_indices.add(i)
                mept_array_iter_n[i] = (n, msl_type)
                continue

            var_name = self._next_var("iter")
            if init_type.startswith("f") or init_type.startswith("bf"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type in ("u64", "ui64"):
                msl_type = "ulong"  # 32-bit uint would truncate (mirrors _lower_scf_if)
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self.kb.raw_line(f"    {msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        # Emit for loop -- use long for i64.
        # scf.for semantics: always `iv < ub` (Triton normalizes negative steps).
        start_type = self.env_types.get(ssa.operand_ids[0], "i32")
        is_i64 = start_type == "i64" or "i64" in (ssa.type_str or "")
        loop_type = "long" if is_i64 else "int"
        loop_var = self._next_var("k")

        self.kb.raw_line(
            f"    for ({loop_type} {loop_var} = {start_var}; {loop_var} < {end_var}; {loop_var} += {step_var}) {{"
        )

        # Map block args to MSL variables
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids:
            # First block arg is induction variable
            self.env[block_arg_ids[0]] = loop_var
            self.env_types[block_arg_ids[0]] = start_type
            self.env_shapes[block_arg_ids[0]] = ()  # induction var is scalar
            # Remaining block args are iter_args
            for i, var in enumerate(iter_vars):
                if i + 1 < len(block_arg_ids):
                    ba_id = block_arg_ids[i + 1]
                    self.env[ba_id] = var
                    self.env_types[ba_id] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
                    # Propagate shape from init value to block arg
                    init_id = init_ids[i] if i < len(init_ids) else None
                    if init_id is not None and init_id in self.env_shapes:
                        self.env_shapes[ba_id] = self.env_shapes[init_id]
                    # Propagate splat-ness from init value. If init is
                    # broadcast-redundant (constant / splat), the block_arg
                    # at the start of each iteration is also broadcast-
                    # redundant; combining with a bcast-laid-out value
                    # preserves that layout.
                    if init_id is not None and init_id in self._is_splat:
                        self._is_splat.add(ba_id)
                    # MEPT register-array iter-arg: expose as an env_array so
                    # the body resolves the block arg with per-element ``v[e]``.
                    if i in mept_array_iter_indices:
                        n_arr, mt = mept_array_iter_n[i]
                        self.env_array[ba_id] = (var, n_arr, mt)
                        self.env_n_elems[ba_id] = n_arr
                    # The scalar block arg is the carried typed address itself.
                    if i in ptr_offset_iter_indices:
                        self.env_is_ptr[ba_id] = (var, "0")
                    if i in ptr_offset_arr_iter_indices:
                        _b, _n = ptr_offset_arr_iter[i]
                        self.env_ptr_array[ba_id] = (_b, var, _n)
                        self.env_n_elems[ba_id] = _n
                    # Register shared-memory-backed iter_args
                    if i in smem_iter_indices:
                        init_shape = self.env_shapes.get(init_ids[i], ()) if i < len(init_ids) else ()
                        if not hasattr(self, "_shared_mem_descs"):
                            self._shared_mem_descs = {}
                        self._shared_mem_descs[ba_id] = (var, init_shape, "fp32")
                        # Also track that this block_arg is smem-backed so
                        # that scf.yield can skip the scalar assignment.
                        if not hasattr(self, "_smem_iter_args"):
                            self._smem_iter_args = {}
                        self._smem_iter_args[ba_id] = var

        # Process body ops.  Track the yielded SSA id per iter_arg so we can
        # propagate metadata (e.g., _bcast_layout) from the yielded value to
        # the scf.for result variable below.
        yielded_ids: list = [None] * n_iter_args
        # Expose the body's region_ops so an in-loop reduce (_cover_inloop_reduce)
        # can collect its input dependency chain for body-local multipass
        # coverage. Save/restore for nested loops.
        _prev_body_ops = getattr(self, "_current_loop_body_ops", None)
        self._current_loop_body_ops = list(ssa.region_ops or [])

        # Stage A pre-emission: if any external tensor index ops (make_range,
        # splat, broadcast, etc.) referenced by body ops are NOT yet in env,
        # emit them now (at the outer body scope, _needs_wrapping unchanged)
        # so body ops can reference them without producing UNKNOWN_<id>. This
        # handles the multipass-ordering case where the outer _loop_e loop
        # hasn't yet emitted these ops when the scf.for (scalar) is hoisted
        # before it. We only emit ops that are NOT derived from tt.load (safe
        # to emit at any index — their value is index-only, not data-bearing).
        # Pre-emission block: fires for ALL scf.for (not gated on mept_enabled).
        # A non-eligible MEPT=1 kernel with a top-level multipass + scf.for also
        # needs this — narrowing to MEPT=0 would reintroduce UNKNOWN_<id> there.
        # The `if self.env.get(_oid) is not None: continue` guard makes it a
        # cheap no-op for normal kernels; the _find_op graph scan only runs for
        # genuinely-missing ops.
        if ssa.region_ops:
            _body_ids = {o.id for o in ssa.region_ops}
            # Index-only subset of ops safe to pre-emit at outer body scope.
            # See _SAFE_REPLAY_OPS in _lowerer_reduce.py for the superset used
            # inside _cover_inloop_reduce (which adds addptr + type-conv ops).
            _SAFE_PREEMIT_OPS = frozenset(
                {
                    "tt.make_range",
                    "tt.splat",
                    "tt.broadcast",
                    "tt.expand_dims",
                    "arith.constant",
                }
            )

            # _find_op: recursive op lookup by id across nested regions.
            # Defined once here (not inside the loop) to avoid re-creation per
            # iteration.
            def _find_op(ops, target):
                for _o in ops:
                    if _o.id == target:
                        return _o
                    if _o.region_ops:
                        r = _find_op(_o.region_ops, target)
                        if r:
                            return r
                    if _o.else_ops:
                        r = _find_op(_o.else_ops, target)
                        if r:
                            return r
                return None

            # Collect referenced external safe ops (BFS over body op inputs)
            _ext_needed = []
            _ext_seen = set()
            for _bop in ssa.region_ops:
                for _oid in _bop.operand_ids or []:
                    if _oid in _body_ids or _oid in _ext_seen:
                        continue
                    if self.env.get(_oid) is not None:
                        continue  # already in env
                    _ext_seen.add(_oid)
                    # Find the producing op
                    _prod = _find_op(self.graph.ops, _oid)
                    if _prod is not None and _prod.op in _SAFE_PREEMIT_OPS and _prod.is_tensor:
                        _ext_needed.append(_prod)
            # Emit missing safe external ops before the body starts
            for _ext_op in _ext_needed:
                if self.env.get(_ext_op.id) is None:
                    self._lower_op(_ext_op)

        try:
            if ssa.region_ops:
                for body_op in ssa.region_ops:
                    if body_op.op == "scf.yield":
                        # SSA yields are simultaneous. Snapshot every pointer's
                        # full next state BEFORE updating any carried variable;
                        # p,q = q,p must not read an already-updated p or offset.
                        pointer_next = {}
                        for i, yield_id in enumerate(body_op.operand_ids):
                            if i in ptr_offset_iter_indices:
                                info = self._loop_pointer_parts(yield_id)
                                if info is None:
                                    from triton_msl.errors import MetalNonRecoverableError

                                    raise MetalNonRecoverableError(
                                        "loop-carried pointer yield has no address representation", op_name="scf.for"
                                    )
                                base, off = info
                                address = base if not off or off == "0" else f"({base} + {off})"
                                next_ptr = self._next_var("next_ptr")
                                self.kb.raw_line(f"        auto {next_ptr} = {address};")
                                pointer_next[i] = next_ptr
                            elif i in ptr_offset_arr_iter_indices:
                                info = self.env_ptr_array.get(yield_id)
                                width = ptr_offset_arr_iter[i][1]
                                if info is None or info[2] != width:
                                    from triton_msl.errors import MetalNonRecoverableError

                                    raise MetalNonRecoverableError(
                                        "loop-carried pointer-array yield representation/width mismatch",
                                        op_name="scf.for",
                                    )
                                base, offsets, _ = info
                                next_base = self._next_var("next_base")
                                self.kb.raw_line(f"        auto {next_base} = {base};")
                                next_offsets = self._var_array(
                                    "next_off", [f"long({offsets}[{e}])" for e in range(width)], "long"
                                )
                                pointer_next[i] = (next_base, next_offsets)
                        # Update iter_arg variables from yield operands
                        for i, yield_id in enumerate(body_op.operand_ids):
                            if i < len(iter_vars):
                                yielded_ids[i] = yield_id
                                # Skip scalar assignment for smem-backed iter_args;
                                # the shared memory was already updated in-place by
                                # the dot or strided binary op.
                                if i in smem_iter_indices:
                                    # Check if the yield value has a shared_mem_desc
                                    # pointing to a DIFFERENT array (e.g. the dot
                                    # wrote to smem_dot_X).  If so, copy it over.
                                    yield_smem = getattr(self, "_shared_mem_descs", {}).get(yield_id)
                                    if yield_smem is None:
                                        # A value each thread holds for its own cell. The tile has
                                        # more cells than the group has threads and nothing writes
                                        # the value into it: the next trip and the loop's result
                                        # would read the cells as they stood before.
                                        from triton_msl.errors import MetalNonRecoverableError

                                        raise MetalNonRecoverableError(
                                            "a loop-carried tile held in threadgroup memory is handed on as a "
                                            f"value computed one cell per thread over {bs} threads: writing it "
                                            "back to the tile is not implemented. The kernel is refused.",
                                            op_name="scf.for",
                                        )
                                    if yield_smem[0] != iter_vars[i]:
                                        src_smem = yield_smem[0]
                                        dst_smem = iter_vars[i]
                                        init_shape = self.env_shapes.get(init_ids[i], ()) if i < len(init_ids) else ()
                                        sz = 1
                                        for d in init_shape:
                                            sz *= d
                                        self.kb.raw_line(f"    for (uint _cp = lid; _cp < {sz}u; _cp += {bs}u) {{")
                                        self.kb.raw_line(f"        {dst_smem}[_cp] = {src_smem}[_cp];")
                                        self.kb.raw_line(f"    }}")
                                        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                                    continue
                                if i in ptr_offset_iter_indices:
                                    self.kb.raw_line(f"        {iter_vars[i]} = {pointer_next[i]};")
                                    continue
                                # Pointer offset-ARRAY iter-arg (MEPT): copy the advanced
                                # pointer's per-thread offset array element-wise into off[e].
                                if i in ptr_offset_arr_iter_indices:
                                    _nb, _noff = pointer_next[i]
                                    _base, _nn = ptr_offset_arr_iter[i]
                                    self.kb.raw_line(f"        {_base} = {_nb};")
                                    for _e in range(_nn):
                                        self.kb.raw_line(f"        {iter_vars[i]}[{_e}] = {_noff}[{_e}];")
                                    continue
                                # MEPT register-array iter-arg: update v[e] per
                                # element. The yielded value is itself an env_array
                                # (the elementwise chain); copy element-wise. If it
                                # collapsed to a scalar, splat it into every slot.
                                if i in mept_array_iter_indices:
                                    n_arr, _mt = mept_array_iter_n[i]
                                    ydesc = self.env_array.get(yield_id)
                                    if ydesc is not None:
                                        ysrc, _yn, _yt = ydesc
                                        # Symmetric with the seed-side guard: the
                                        # yielded array must match the iter-arg
                                        # width, else iter_N[e]=ysrc[e] reads OOB.
                                        if _yn != n_arr:
                                            from triton_msl.errors import MetalNonRecoverableError

                                            raise MetalNonRecoverableError(
                                                f"MEPT iter-arg yield width mismatch: "
                                                f"yielded {_yn}, iter-arg {n_arr} "
                                                f"(yield_id={yield_id})"
                                            )
                                        for e in range(n_arr):
                                            self.kb.raw_line(f"        {iter_vars[i]}[{e}] = {ysrc}[{e}];")
                                    else:
                                        yval = self._lookup(yield_id)
                                        for e in range(n_arr):
                                            self.kb.raw_line(f"        {iter_vars[i]}[{e}] = {yval};")
                                    continue
                                yield_val = self._lookup(yield_id)
                                self.kb.raw_line(f"        {iter_vars[i]} = {yield_val};")
                    else:
                        self._lower_op(body_op)
        finally:
            self._current_loop_body_ops = _prev_body_ops

        self.kb.raw_line("    }")

        # Map scf.for results to iter_arg variables using proper result IDs
        if ssa.result_ids:
            for i, var in enumerate(iter_vars):
                if i < len(ssa.result_ids):
                    rid = ssa.result_ids[i]
                    self.env[rid] = var
                    self.env_types[rid] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
                    # Propagate shape from init value to result
                    if i < len(init_ids) and init_ids[i] in self.env_shapes:
                        self.env_shapes[rid] = self.env_shapes[init_ids[i]]
                    # Pointer offset-carry iter-arg: expose the result as a pointer
                    # (base, off_var) so a post-loop tt.load/addptr on the final pointer works.
                    if i in ptr_offset_iter_indices:
                        self.env_is_ptr[rid] = (var, "0")
                    if i in ptr_offset_arr_iter_indices:
                        _b, _n = ptr_offset_arr_iter[i]
                        self.env_ptr_array[rid] = (_b, var, _n)
                        self.env_n_elems[rid] = _n
                    # MEPT register-array iter-arg: expose the result as an
                    # env_array so the post-loop store reads ``v[e]``.
                    if i in mept_array_iter_indices:
                        n_arr, mt = mept_array_iter_n[i]
                        self.env_array[rid] = (var, n_arr, mt)
                        self.env_n_elems[rid] = n_arr
                    # Propagate shared_mem_desc for oversized iter_args
                    if i in smem_iter_indices:
                        init_shape = self.env_shapes.get(init_ids[i], ()) if i < len(init_ids) else ()
                        if not hasattr(self, "_shared_mem_descs"):
                            self._shared_mem_descs = {}
                        self._shared_mem_descs[rid] = (var, init_shape, "fp32")
                    # Propagate broadcast-layout from the yielded value. The
                    # init value is typically a constant (no layout); after
                    # the first iteration, the iter_arg takes on the layout
                    # of `yielded`, which stays invariant for subsequent
                    # iterations (consistent layout in / out of body).
                    if i < len(yielded_ids) and yielded_ids[i] is not None:
                        lay = self._bcast_layout.get(yielded_ids[i])
                        if lay is not None:
                            self._bcast_layout[rid] = lay
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
            if init_ids and init_ids[0] in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[init_ids[0]]
            if 0 in smem_iter_indices:
                # Single-result loops use ssa.id rather than result_ids. They
                # carry the same persistent storage as the multi-result branch.
                self._shared_mem_descs[ssa.id] = (iter_vars[0], self.env_shapes[ssa.id], "fp32")
            if 0 in ptr_offset_iter_indices:
                self.env_is_ptr[ssa.id] = (iter_vars[0], "0")
            if 0 in ptr_offset_arr_iter_indices:
                _b, _n = ptr_offset_arr_iter[0]
                self.env_ptr_array[ssa.id] = (_b, iter_vars[0], _n)
                self.env_n_elems[ssa.id] = _n
            # MEPT register-array iter-arg: a single-result scf.for reports
            # ``result_ids`` as None (mlir_walker collapses len==1), so the
            # result maps to ``ssa.id`` here, not the multi-result loop above.
            # Register the env_array on ssa.id so the post-loop store reads the
            # array rather than treating it as a scalar.
            if 0 in mept_array_iter_indices:
                n_arr, mt = mept_array_iter_n[0]
                self.env_array[ssa.id] = (iter_vars[0], n_arr, mt)
                self.env_n_elems[ssa.id] = n_arr
            if yielded_ids and yielded_ids[0] is not None:
                lay = self._bcast_layout.get(yielded_ids[0])
                if lay is not None:
                    self._bcast_layout[ssa.id] = lay
        elif iter_vars:
            # Fallback: single result maps to first iter_var
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
            if init_ids and init_ids[0] in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[init_ids[0]]
            if 0 in mept_array_iter_indices:
                n_arr, mt = mept_array_iter_n[0]
                self.env_array[ssa.id] = (iter_vars[0], n_arr, mt)
                self.env_n_elems[ssa.id] = n_arr
            if yielded_ids and yielded_ids[0] is not None:
                lay = self._bcast_layout.get(yielded_ids[0])
                if lay is not None:
                    self._bcast_layout[ssa.id] = lay

    def _lower_scf_if(self, ssa: SSAValue):
        """scf.if → MSL if/else block with optional results."""
        if not ssa.operand_ids:
            return

        cond = self._lookup(ssa.operand_ids[0])
        result_ids = ssa.result_ids or ([ssa.id] if ssa.id is not None else [])
        pointer_results = {}
        for i, rid in enumerate(result_ids):
            meta = self.graph.result_meta.get(rid)
            if meta is not None and meta.type.kind == "pointer":
                pointee = meta.type.pointee
                if (
                    meta.schema_version != 1
                    or meta.value_id != rid
                    or meta.kind != "result"
                    or meta.producer_id != ssa.id
                    or meta.result_index != i
                    or meta.type.address_space != 1
                    or pointee is None
                    or pointee.unknown_reason
                    or pointee.elem is None
                ):
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "scf.if pointer result lacks complete native per-result metadata", op_name="scf.if"
                    )
                pointer_results[i] = _mlir_to_triton_dtype(pointee.elem)
        if "!tt.ptr" in (ssa.type_str or "") and not pointer_results:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "scf.if pointer result lacks native per-result identity; refusing address guessing", op_name="scf.if"
            )

        # Check both then and else for yield with operands
        all_body_ops = list(ssa.region_ops or []) + list(ssa.else_ops or [])
        has_results = any(body_op.op == "scf.yield" and body_op.operand_ids for body_op in all_body_ops)

        # For scf.if with results, declare result variables before the if/else
        result_vars = []
        if has_results:
            # Infer result types from yield operands
            yield_types = []
            for body_op in all_body_ops:
                if body_op.op == "scf.yield" and body_op.operand_ids:
                    for yid in body_op.operand_ids:
                        i = len(yield_types)
                        if i in pointer_results:
                            yield_types.append(pointer_results[i])
                            continue
                        # scf.if declares a SCALAR result var below; a yielded MEPT
                        # register array (n>1, block > num_threads) then fails the MSL
                        # compile with 'assigning from incompatible type float[n]' — a
                        # cryptic crash, not a clean refusal (re-audit #14). Refuse loudly.
                        if getattr(self, "env_n_elems", {}).get(yid, 1) > 1:
                            from triton_msl.errors import MetalNonRecoverableError

                            raise MetalNonRecoverableError(
                                "scf.if returning a multi-element-per-thread register "
                                "array (block > num_threads) is not supported. Refusing "
                                "rather than emit a cryptic MSL compile error. Use "
                                "BLOCK <= num_threads or restructure.",
                                op_name="scf.if",
                            )
                        # A value yielded from INSIDE the then/else body (e.g. an inner
                        # scf.for accumulator) is not in env_types yet — the bodies are
                        # lowered AFTER this type-inference pass. Defaulting to fp32 here
                        # silently turns an integer accumulation into float (precision loss
                        # >2^24, i64 truncation). Fall back to the scf.if's IR result
                        # element type; refuse if even that is unknown rather than guess
                        # float. (Triton-lens audit 2026-06-25.)
                        yt = self.env_types.get(yid)
                        if yt is None:
                            _et = getattr(ssa, "elem_type", None)
                            yt = _mlir_to_triton_dtype(_et) if _et else None
                        if yt is None:
                            from triton_msl.errors import MetalNonRecoverableError

                            raise MetalNonRecoverableError(
                                "scf.if result dtype is undeterminable (a value yielded "
                                "from inside the branch with no inferable type). Refusing "
                                "rather than default to float, which would silently "
                                "corrupt an integer result.",
                                op_name="scf.if",
                            )
                        yield_types.append(yt)
                    break

            for i, rid in enumerate(result_ids):
                var_name = f"ifr_{abs(rid)}_{i}"
                result_vars.append((rid, var_name))
                yt = yield_types[i] if i < len(yield_types) else "fp32"
                if i in pointer_results:
                    msl_type = f"volatile device {triton_type_to_msl(yt)}*"
                elif yt.startswith("fp") or yt.startswith("bf") or yt.startswith("f"):
                    msl_type = "float"
                elif yt == "i64":
                    msl_type = "long"
                elif yt == "u64":
                    msl_type = "ulong"
                elif yt.startswith("u"):
                    msl_type = "uint"
                else:
                    msl_type = "int"
                self.kb.raw_line(f"    {msl_type} {var_name};")

        def _assign_result(index, yield_id):
            _, var_name = result_vars[index]
            if index not in pointer_results:
                self.kb.raw_line(f"        {var_name} = {self._lookup(yield_id)};")
                return
            info = self._loop_pointer_parts(yield_id)
            if info is None:
                from triton_msl.errors import MetalNonRecoverableError

                why = "a per-thread pointer array" if yield_id in self.env_ptr_array else "no address representation"
                raise MetalNonRecoverableError(
                    f"scf.if pointer branch has {why}; refusing rather than load/cast it", op_name="scf.if"
                )
            base, offset = info
            expr = base if not offset or offset == "0" else f"({base} + {offset})"
            self.kb.raw_line(f"        {var_name} = {expr};")

        self.kb.raw_line(f"    if ({cond}) {{")

        # Lower "then" body
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            _assign_result(i, yield_id)
                else:
                    self._lower_op(body_op)

        # Lower "else" body
        if ssa.else_ops:
            self.kb.raw_line("    } else {")
            for body_op in ssa.else_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            _assign_result(i, yield_id)
                else:
                    self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map result variables into env with proper types
        for i, (rid, var_name) in enumerate(result_vars):
            self.env[rid] = var_name
            # Propagate type from yield operands
            yt = yield_types[i] if i < len(yield_types) else "fp32"
            self.env_types[rid] = yt
            if i in pointer_results:
                self.env_is_ptr[rid] = (var_name, "0")

    def _lower_scf_while(self, ssa: SSAValue):
        """scf.while → MSL while(true) { condition-check; body; } loop.

        scf.while has operands: [init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as init values)

        Two regions:
          - "before" (region_ops): evaluates condition, terminates with scf.condition
          - "after" (else_ops): loop body, terminates with scf.yield

        The "before" region's scf.condition carries the loop predicate and
        forwarded values to the "after" region's block arguments.
        """
        init_ids = ssa.operand_ids  # Initial values for iter_args
        n_iter_args = len(init_ids)
        result_ids = ssa.result_ids or ([ssa.id] if ssa.id is not None else [])

        # Declare iter_arg variables from init values
        iter_vars = []
        iter_dtypes = []
        ptr_iter = {}
        ptr_array_iter = {}
        for i, init_id in enumerate(init_ids):
            ptr = self._loop_pointer_parts(init_id)
            if ptr is not None:
                base, offset = ptr
                address = base if not offset or offset == "0" else f"({base} + {offset})"
                ptr_var = self._next_var("wh_ptr")
                self.kb.raw_line(f"    auto {ptr_var} = {address};")
                iter_vars.append(ptr_var)
                iter_dtypes.append(self._trace_ptr_dtype(init_id))
                ptr_iter[i] = ptr_var
                continue
            ptr_array = self.env_ptr_array.get(init_id)
            if ptr_array is not None:
                base, offsets, width = ptr_array
                base_var = self._next_var("wh_base")
                self.kb.raw_line(f"    auto {base_var} = {base};")
                off_var = self._var_array("wh_off", [f"long({offsets}[{e}])" for e in range(width)], "long")
                iter_vars.append(off_var)
                iter_dtypes.append(self._trace_ptr_dtype(init_id))
                ptr_array_iter[i] = (base_var, width)
                continue
            var_name = self._next_var("wh")
            init_val = self._lookup(init_id)
            init_type = self.env_types.get(init_id, "i32")
            if init_type.startswith("f") or init_type.startswith("bf") or init_type.startswith("fp"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self.kb.raw_line(f"    {msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        self.kb.raw_line("    for (;;) {")

        # Map "before" region block args to iter_vars
        before_block_args = ssa.attrs.get("block_arg_ids", [])
        for i, var in enumerate(iter_vars):
            if i < len(before_block_args):
                self.env[before_block_args[i]] = var
                self.env_types[before_block_args[i]] = iter_dtypes[i]
                if i in ptr_iter:
                    self.env_is_ptr[before_block_args[i]] = (var, "0")
                elif i in ptr_array_iter:
                    base, width = ptr_array_iter[i]
                    self.env_ptr_array[before_block_args[i]] = (base, var, width)
                    self.env_n_elems[before_block_args[i]] = width

        # Lower "before" region (condition evaluation)
        for body_op in ssa.region_ops or []:
            if body_op.op == "scf.condition":
                # First operand is the condition
                if body_op.operand_ids:
                    cond_var = self._lookup(body_op.operand_ids[0])
                    self.kb.raw_line(f"        if (!({cond_var})) break;")
                # Remaining operands are forwarded values to "after" block args
                after_block_args = ssa.attrs.get("else_block_arg_ids", [])
                for j, fwd_id in enumerate(body_op.operand_ids[1:]):
                    if j < len(after_block_args):
                        after_id = after_block_args[j]
                        fwd_val = self._lookup(fwd_id)
                        self.env[after_id] = fwd_val
                        fwd_type = self.env_types.get(fwd_id, "i32")
                        self.env_types[after_id] = fwd_type
                        fwd_ptr = self._loop_pointer_parts(fwd_id)
                        if fwd_ptr is not None:
                            self.env_is_ptr[after_id] = fwd_ptr
                        elif fwd_id in self.env_ptr_array:
                            self.env_ptr_array[after_id] = self.env_ptr_array[fwd_id]
                            self.env_n_elems[after_id] = self.env_ptr_array[fwd_id][2]
            else:
                self._lower_op(body_op)

        # If "after" block args weren't mapped by scf.condition forwarding,
        # map them to iter_vars directly (they share the same values)
        after_block_args = ssa.attrs.get("else_block_arg_ids", [])
        for i, var in enumerate(iter_vars):
            if i < len(after_block_args) and after_block_args[i] not in self.env:
                self.env[after_block_args[i]] = var
                self.env_types[after_block_args[i]] = iter_dtypes[i]
                if i in ptr_iter:
                    self.env_is_ptr[after_block_args[i]] = (var, "0")
                elif i in ptr_array_iter:
                    base, width = ptr_array_iter[i]
                    self.env_ptr_array[after_block_args[i]] = (base, var, width)
                    self.env_n_elems[after_block_args[i]] = width

        # Lower "after" region (loop body)
        for body_op in ssa.else_ops or []:
            if body_op.op == "scf.yield":
                pointer_next = {}
                for j, yield_id in enumerate(body_op.operand_ids):
                    if j in ptr_iter:
                        info = self._loop_pointer_parts(yield_id)
                        if info is None:
                            from triton_msl.errors import MetalNonRecoverableError

                            raise MetalNonRecoverableError(
                                "scf.while pointer yield has no address representation", op_name="scf.while"
                            )
                        base, offset = info
                        address = base if not offset or offset == "0" else f"({base} + {offset})"
                        next_ptr = self._next_var("wh_next_ptr")
                        self.kb.raw_line(f"        auto {next_ptr} = {address};")
                        pointer_next[j] = next_ptr
                    elif j in ptr_array_iter:
                        info = self.env_ptr_array.get(yield_id)
                        width = ptr_array_iter[j][1]
                        if info is None or info[2] != width:
                            from triton_msl.errors import MetalNonRecoverableError

                            raise MetalNonRecoverableError(
                                "scf.while pointer-array yield representation/width mismatch", op_name="scf.while"
                            )
                        base, offsets, _ = info
                        next_base = self._next_var("wh_next_base")
                        self.kb.raw_line(f"        auto {next_base} = {base};")
                        next_offsets = self._var_array(
                            "wh_next_off", [f"long({offsets}[{e}])" for e in range(width)], "long"
                        )
                        pointer_next[j] = (next_base, next_offsets)
                # Update iter_arg variables from yield operands
                for j, yield_id in enumerate(body_op.operand_ids):
                    if j < len(iter_vars):
                        if j in ptr_iter:
                            self.kb.raw_line(f"        {iter_vars[j]} = {pointer_next[j]};")
                            continue
                        if j in ptr_array_iter:
                            next_base, next_offsets = pointer_next[j]
                            base, width = ptr_array_iter[j]
                            self.kb.raw_line(f"        {base} = {next_base};")
                            for e in range(width):
                                self.kb.raw_line(f"        {iter_vars[j]}[{e}] = {next_offsets}[{e}];")
                            continue
                        yield_val = self._lookup(yield_id)
                        self.kb.raw_line(f"        {iter_vars[j]} = {yield_val};")
            else:
                self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map scf.while results to iter_arg variables
        if len(result_ids) > 1:
            for i, var in enumerate(iter_vars):
                if i < len(result_ids):
                    self.env[result_ids[i]] = var
                    self.env_types[result_ids[i]] = iter_dtypes[i] if i < len(iter_dtypes) else "i32"
                    if i in ptr_iter:
                        self.env_is_ptr[result_ids[i]] = (var, "0")
                    elif i in ptr_array_iter:
                        base, width = ptr_array_iter[i]
                        self.env_ptr_array[result_ids[i]] = (base, var, width)
                        self.env_n_elems[result_ids[i]] = width
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"
            if 0 in ptr_iter:
                self.env_is_ptr[ssa.id] = (iter_vars[0], "0")
            elif 0 in ptr_array_iter:
                base, width = ptr_array_iter[0]
                self.env_ptr_array[ssa.id] = (base, iter_vars[0], width)
                self.env_n_elems[ssa.id] = width
        elif iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"

    def _atomic_lanes(self, ssa, shape):
        """The condition under which a lane performs this atomic, so each element is updated once.

        ``shape`` is the atomic's own (empty for a scalar). A kernel runs one lane per
        element of its widest value, so an atomic over fewer elements is reached by
        several lanes for each of them. Returns ``(guard, one)``: the guard, or None
        when every lane has an element of its own, and whether the atomic is over one
        element and so performed by the first lane alone.
        """
        from triton_msl.errors import MetalNonRecoverableError

        lane, lanes = self._lane_here()
        used, looped = self._graph_facts()
        count = 1
        for extent in shape:
            count *= extent
        if count == 1:
            if lane != "lid" and ssa.id in used and id(ssa) in looped:
                # The first lane's trips all finish before the thread's next lane starts,
                # so one slot cannot hand each trip's old value to the other lanes.
                raise MetalNonRecoverableError(
                    "an atomic on one element, inside a loop, whose returned value is read, in a "
                    "kernel with more lanes than threads: the value each trip returns is not kept "
                    "for the lanes that run later. Use a block no wider than the thread count, or "
                    "do not read the returned value.",
                    op_name=ssa.op,
                )
            self._single_lane_stores.add(id(ssa))
            return f"{lane} == 0u", True
        if len(shape) == 1:
            if count >= lanes:
                return None, False
            # Lanes 0..count-1 take one element each. An address, value or mask held one
            # row per W lanes (computed from a row-wise reduce) would give every one of
            # them the first row's element.
            self._refuse_row_blocked_effect(ssa.op, count, tuple(ssa.operand_ids or ()), op_name=ssa.op)
            # The same question asked of the indices themselves, as a store of fewer
            # elements than lanes asks it. A row index is written ``lane / N`` whether or
            # not it was declared row-blocked, and inside a wrap loop it is not declared:
            # the column of a (R, 1) tile flattened to R elements is held one row per N
            # lanes there, and lanes 0..R-1 hold its first R / N rows.
            # A float maximum or minimum arrives as the bits of its value, one lane's from its own.
            strides, moduli, _, unknown, bounds = self._lane_sources(
                tuple(ssa.operand_ids or ()), through=("tt.bitcast",)
            )
            if strides - {1} or unknown or moduli - {count} or any(bound < count for bound in bounds):
                why = (
                    f"depends on {unknown[0]}, whose place on the lanes is not known"
                    if unknown
                    else "is not held element by lane"
                )
                raise MetalNonRecoverableError(
                    f"an atomic over {count} elements in a kernel that runs {lanes} lanes: its address, "
                    f"value or mask {why}. Which lane holds each element is not proved, so the atomic is "
                    "refused. Give it the shape of the kernel's widest block.",
                    op_name=ssa.op,
                )
            return f"{lane} < {count}u", False
        if self._is_2d and len(shape) == 2:
            if count >= lanes:
                return None, False
            if ssa.id in used:
                raise MetalNonRecoverableError(
                    f"an atomic over {list(shape)} in a kernel that runs {lanes} lanes, whose returned "
                    "value is read: one lane performs it for each element, and the value is not handed "
                    "to the other lanes that hold the element. Do not read the returned value, or give "
                    "the atomic the shape of the kernel's widest block.",
                    op_name=ssa.op,
                )
            # One lane per element, and a lane that holds the element's address and operands.
            guard = self._tile_lanes(tuple(shape), tuple(ssa.operand_ids or ()), lane, lanes, "an atomic", each=True)
            return guard, False
        raise MetalNonRecoverableError(
            f"an atomic over {list(shape)} in a kernel that runs {lanes} lanes: several lanes reach "
            "each element and the update would be applied once for each of them. Give the atomic "
            "the shape of the kernel's widest block, or one dimension.",
            op_name=ssa.op,
        )

    def _atomic_ordering_fences(self, ssa):
        """190/201: fence-bracket ONE logical atomic, with closed source metadata.

        Flags cover the whole kernel conservatively: later lowering stages can add
        threadgroup buffers, so the builder's current declarations cannot exclude them.
        These are thread fences, not barriers or a scheduler-progress guarantee.
        """
        from triton_msl.errors import MetalNonRecoverableError

        sem, scope = ssa.attrs.get("sem"), ssa.attrs.get("scope")
        if sem not in ("relaxed", "acquire", "release", "acq_rel"):
            raise MetalNonRecoverableError(f"atomic has missing/unknown semantics {sem!r}", op_name=ssa.op)
        if scope not in ("gpu", "cta"):
            raise MetalNonRecoverableError(
                f"atomic scope {scope!r} is missing, unknown or unsupported on Metal", op_name=ssa.op
            )
        if ssa.op == "tt.atomic_rmw" and ssa.attrs.get("rmw_op") not in (
            "and",
            "or",
            "xor",
            "add",
            "fadd",
            "max",
            "min",
            "umax",
            "umin",
            "exch",
        ):
            raise MetalNonRecoverableError("atomic has missing/unknown RMW opcode", op_name=ssa.op)
        if sem == "relaxed":
            return None, None
        from triton_msl.backend.device_detect import get_device_info

        device = get_device_info()
        version = getattr(self.options, "target_metal_version", "auto")
        if version == "auto":
            version = device.metal_version
        match = re.fullmatch(r"([0-9]+)\.([0-9]+)", str(version))
        capability = re.fullmatch(r"([0-9]+)\.([0-9]+)", str(device.metal_version))
        if (
            match is None
            or capability is None
            or not (3, 2) <= tuple(map(int, match.groups())) <= tuple(map(int, capability.groups()))
            or re.fullmatch(r"M[1-9][0-9]*", device.chip_family) is None
        ):
            raise MetalNonRecoverableError(
                f"ordered atomic requires Metal >=3.2 on Apple silicon; got {version!r}/{device.chip_family!r}",
                op_name=ssa.op,
            )
        fence = (
            "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, "
            "memory_order_seq_cst, thread_scope_" + ("device" if scope == "gpu" else "threadgroup") + ");"
        )
        return (fence if sem in ("release", "acq_rel") else None, fence if sem in ("acquire", "acq_rel") else None)

    def _emit_atomic_rmw_16bit(self, base_ptr, offsets, val_var, rmw_op, half_type, result_var, indent, n):
        """Neighbor-preserving 16-bit float atomic RMW via a 32-bit word CAS.

        ``half_type`` is "half" (fp16) or "bfloat" (bf16). The element lives in
        one half of an aligned 4-byte word; we CAS the word and preserve the
        other half. The op runs in float domain (always supported) then casts
        back. Returns the OLD value (Triton atomic_rmw semantics).

        Note: ``exch`` still uses the CAS loop (not a bare atomic_exchange) so
        the neighbor half is preserved; it may retry if the neighbor is written
        concurrently — the returned old value is the half immediately before the
        winning CAS. max/min/exch are forward-compatible: Triton's frontend
        currently restricts 16-bit-float atomics to ``add`` only."""
        op_expr = {
            "add": f"({half_type})((float)oldh_{n} + (float){val_var})",
            "fadd": f"({half_type})((float)oldh_{n} + (float){val_var})",
            # fmax/fmin (not max/min): IEEE maxNum/minNum NaN semantics.
            "max": f"({half_type})fmax((float)oldh_{n}, (float){val_var})",
            "min": f"({half_type})fmin((float)oldh_{n}, (float){val_var})",
            "exch": f"({half_type})((float){val_var})",
        }.get(rmw_op)
        if op_expr is None:
            raise ValueError(f"_emit_atomic_rmw_16bit: unsupported op '{rmw_op}'")
        kb = self.kb
        kb.raw_line(f"{indent}uint _eidx_{n} = (uint)({offsets});")
        # base_ptr is typed `device half*`/`device bfloat*` in the emitted MSL,
        # so casting to atomic_uint* and adding (_eidx>>1) yields the 4-byte word
        # containing element _eidx (always 4-byte aligned: buffer bindings are
        # >=16-byte aligned).
        kb.raw_line(f"{indent}device atomic_uint* wptr_{n} = (device atomic_uint*)({base_ptr}) + (_eidx_{n} >> 1);")
        # little-endian: even idx -> low half (shift 0), odd idx -> high half (16).
        kb.raw_line(f"{indent}uint _sh_{n} = 16u * (_eidx_{n} & 1u);")
        kb.raw_line(f"{indent}uint _w_{n} = atomic_load_explicit(wptr_{n}, memory_order_relaxed);")
        kb.raw_line(f"{indent}ushort cur_bits_{n};")
        kb.raw_line(f"{indent}while (true) {{")
        kb.raw_line(f"{indent}    cur_bits_{n} = (ushort)((_w_{n} >> _sh_{n}) & 0xFFFFu);")
        kb.raw_line(f"{indent}    {half_type} oldh_{n} = as_type<{half_type}>(cur_bits_{n});")
        kb.raw_line(f"{indent}    {half_type} new_{n} = {op_expr};")
        kb.raw_line(f"{indent}    ushort nb_{n} = as_type<ushort>(new_{n});")
        kb.raw_line(f"{indent}    uint wn_{n} = (_w_{n} & ~(0xFFFFu << _sh_{n})) | ((uint)nb_{n} << _sh_{n});")
        kb.raw_line(f"{indent}    if (atomic_compare_exchange_weak_explicit(wptr_{n}, &_w_{n}, wn_{n},")
        kb.raw_line(f"{indent}            memory_order_relaxed, memory_order_relaxed)) break;")
        kb.raw_line(f"{indent}}}")
        kb.raw_line(f"{indent}{result_var} = as_type<{half_type}>(cur_bits_{n});")

    def _lower_atomic_rmw(self, ssa: SSAValue):
        """tt.atomic_rmw → MSL atomic read-modify-write.

        Operands: [ptr, val, mask] (mask may be absent for scalar atomics)
        Result: the OLD value at the atomic location.

        For integer atomics: cast to device atomic_int*/atomic_uint* and use
        atomic_fetch_add/max/min/and/or/xor/exchange_explicit.

        For float atomic add: cast to device atomic_uint* and use a CAS loop
        (Metal device pointers are declared as float*, can't use atomic_float*).

        Float max/min: Triton decomposes these into bitcast + integer atomic
        in the TTGIR, so we only see integer atomics for those cases.
        """
        if len(ssa.operand_ids) < 2:
            return

        native_result = self._prove_atomic_native_contract(ssa)
        fence_before, fence_after = self._atomic_ordering_fences(ssa)

        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]
        mask_id = ssa.operand_ids[2] if len(ssa.operand_ids) >= 3 else None

        rmw_op = ssa.attrs["rmw_op"]
        val_var = self._lookup(val_id)

        # Resolve pointer info
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            offsets = "0"

        # Metal has NO 64-bit device atomic. The integer path below casts to
        # device atomic_int* and the value to (int), silently truncating a 64-bit
        # pointer + value to the low 32 bits (re-audit #10: int64 atomic_add wrote 0).
        # Refuse loudly rather than mis-compute.
        if native_result.width != 32 and not (native_result.kind == "float" and native_result.width == 16):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"{native_result.width}-bit atomic is not supported: Metal has no matching "
                "device atomic and emitting a 32-bit operation would silently change the value. Refusing.",
                op_name="tt.atomic_rmw",
            )

        is_float = native_result.kind == "float"

        # 16-bit float atomics: no native Metal 16-bit atomic, but a
        # neighbor-preserving 32-bit word-CAS is correct (Phase 3 feature 1).
        is_16bit_float = is_float and native_result.width == 16
        half_type = None
        if is_16bit_float:
            half_type = "bfloat" if native_result.elem == "bf16" else "half"
            if rmw_op not in ("add", "fadd", "max", "min", "exch"):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"atomic_rmw '{rmw_op}' on 16-bit float not supported (only add/max/min/exch via word-CAS)."
                )
        elif is_float and (native_result.elem != "f32" or rmw_op not in ("add", "fadd", "exch")):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"atomic_rmw '{rmw_op}' has no proved 32-bit floating-point lowering",
                op_name="tt.atomic_rmw",
            )
        elif not is_float and (
            native_result.kind != "integer" or native_result.elem != "i32" or native_result.width != 32
        ):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "atomic_rmw native result is not a supported signless i32 representation",
                op_name="tt.atomic_rmw",
            )

        # Check for mask
        mask_var = None
        if mask_id is not None:
            if mask_id in self.env_is_mask or self._is_mask(mask_id):
                mask_var = self._lookup(mask_id)
            else:
                # Could be a splat of true — check if it's a constant true
                lookup_val = self._lookup(mask_id)
                if lookup_val not in ("true", "1"):
                    mask_var = lookup_val

        # Unique variable suffix
        n = self._var_counter
        self._var_counter += 1

        # Determine if this is an unsigned operation
        is_unsigned = rmw_op in ("umax", "umin")

        # MSL atomic function map for integer atomics
        _RMW_TO_MSL = {
            "add": "atomic_fetch_add_explicit",
            "fadd": None,  # handled separately (CAS loop for float)
            "max": "atomic_fetch_max_explicit",
            "umax": "atomic_fetch_max_explicit",
            "min": "atomic_fetch_min_explicit",
            "umin": "atomic_fetch_min_explicit",
            "and": "atomic_fetch_and_explicit",
            "or": "atomic_fetch_or_explicit",
            "xor": "atomic_fetch_xor_explicit",
            "exch": "atomic_exchange_explicit",
        }

        # Determine result type
        result_var = f"old_{n}"
        if is_16bit_float:
            result_dtype = "bf16" if half_type == "bfloat" else "fp16"
            result_msl_type = half_type
            result_zero = f"({half_type})0"
        elif is_float and rmw_op in ("fadd", "add", "exch"):
            result_dtype = "fp32"
            result_msl_type = "float"
            result_zero = "0.0f"
        elif is_unsigned:
            result_dtype = "u32"
            result_msl_type = "uint"
            result_zero = "0u"
        else:
            result_dtype = "i32"
            result_msl_type = "int"
            result_zero = "0"

        # Scalar atomics (non-tensor): only thread 0 per threadgroup executes.
        # In Triton, a scalar atomic (ptr is !tt.ptr, not tensor<Nx!tt.ptr>)
        # is per-program, not per-thread. Guard with lid == 0.
        is_scalar = not native_result.is_tensor
        atom_shape = native_result.shape
        atomic_result_shared = None
        if is_scalar or all(extent == 1 for extent in atom_shape):
            # The scalar result is logically available to every thread after a later
            # tt.splat, but only lid 0 executes the device atomic below.  A plain local
            # therefore leaves every other thread holding result_zero; labelling that
            # value DIRECT lets a 1-D store silently emit [old, 0, 0, ...].  Broadcast
            # lid 0's old value through threadgroup memory before registering a layout.
            # A tensor<1> has the same physical invariant: the 1-D underfill guard makes
            # only lid 0 execute, and a later tt.broadcast is metadata-only in this
            # lowering. Without the same repair it also emitted [old, 0, 0, ...].
            atomic_result_shared = f"atomic_single_result_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(atomic_result_shared, dtype=result_dtype, size=1)

        # n>1 under-cover guard (mirrors _lower_store): a BLOCK-wide atomic the
        # base path emits as one element per thread (PTR[k+lid]) would silently
        # drop the rest of each tile-stride when block_size > num_threads.
        # Refuse loudly — the only correct n>1 paths are the register-array
        # regime and _loop_e-wrapped emission.
        _val_shape = self.env_shapes.get(val_id)
        _num_threads = self.kb.block_size
        if (
            not self._mept_single_pass
            and not self._needs_wrapping
            and _val_shape is not None
            and len(_val_shape) >= 1
            and _val_shape[0] > _num_threads
        ):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing a {_val_shape[0]}-element atomic with only "
                f"{_num_threads} threads: the base path scatters one element "
                f"per thread, so a tile wider than the threadgroup would "
                f"silently drop the rest. Launch with num_warps = BLOCK/32 "
                f"(so num_threads == BLOCK), or reduce BLOCK."
            )

        # Backstop (B3): a SCALAR (per-program) NON-IDEMPOTENT atomic RMW
        # (add/sub/fadd/xor — value-accumulating) emitted while a wrap loop is
        # ACTIVE would fire once per wrap iteration on thread 0 (lid==0 holds every
        # iteration), over-counting by BLOCK/num_threads. The multipass reduce path
        # now hoists such a post-reduce atomic OUT of the loop (so this is never hit
        # there); this guard refuses any OTHER path that would reach a scalar
        # non-idempotent atomic inside a live wrap loop, rather than silently
        # over-count. (max/min/exch/and/or are idempotent — repeating them is a
        # no-op — so they are exempt.)
        _non_idempotent = rmw_op in ("add", "fadd", "sub", "xor")
        if is_scalar and _non_idempotent and self._needs_wrapping:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing a scalar non-idempotent atomic_{rmw_op} emitted inside a "
                f"multi-element wrap loop (BLOCK > num_threads): thread 0 would apply "
                f"the RMW once per wrap iteration, over-counting by BLOCK/num_threads "
                f"(e.g. tl.atomic_add(out, tl.sum(x)) returned 8x). Launch with "
                f"num_warps = BLOCK/32 so num_threads == BLOCK (no wrap loop), or "
                f"reduce BLOCK.",
                op_name="tt.atomic_rmw",
            )

        # Backstop (re-audit silent-wrong #4): a scalar non-idempotent atomic whose value is
        # accumulated by an IN-LOOP reduce (a tt.reduce inside an scf.for, e.g.
        #   acc=0; for j in range(ITER): acc += tl.sum(load(...)); atomic_add(out, sum(acc)))
        # over-counts by BLOCK/num_threads in the wrapping regime: the per-program scalar
        # accumulation is replayed once per wrap iteration. The B3 hoist above only covers a
        # single NON-looped reduce; this looped form re-adds. Refuse loudly until the scalar
        # accumulation is threaded out of the wrap loop. The single-pass / num_threads==BLOCK
        # regime (_mept_single_pass) has no wrap, so it is exempt (the suggested workaround).
        if is_scalar and _non_idempotent and not self._mept_single_pass and val_id is not None:
            _byid_c = {}

            def _coll(ops):
                for s in ops:
                    _byid_c[s.id] = s
                    if s.region_ops:
                        _coll(s.region_ops)
                    if s.else_ops:
                        _coll(s.else_ops)

            _coll(self.graph.ops)

            def _region_has_reduce(ops):
                for s in ops:
                    if s.op == "tt.reduce":
                        return True
                    if s.region_ops and _region_has_reduce(s.region_ops):
                        return True
                    if s.else_ops and _region_has_reduce(s.else_ops):
                        return True
                return False

            def _traces_to_inloop_reduce(vid, depth=0, seen=None):
                if seen is None:
                    seen = set()
                if vid in seen or depth > 48:
                    return False
                seen.add(vid)
                o = _byid_c.get(vid)
                if o is None:
                    return False
                if o.op == "scf.for" and o.region_ops and _region_has_reduce(o.region_ops):
                    return True
                return any(_traces_to_inloop_reduce(oid, depth + 1, seen) for oid in (o.operand_ids or []))

            if _traces_to_inloop_reduce(val_id):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"scalar non-idempotent atomic_{rmw_op} whose value is accumulated by "
                    f"an IN-LOOP reduce (a tt.reduce inside an scf.for) over-counts by "
                    f"BLOCK/num_threads in the wrap regime (the per-program accumulation is "
                    f"replayed once per wrap iteration). Launch with num_warps = BLOCK/32 "
                    f"(num_threads == BLOCK, no wrap), or accumulate without an in-loop "
                    f"reduce.",
                    op_name="tt.atomic_rmw",
                )

        # Always declare result variable first (needed for mask or not)
        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Build the guard condition
        guard_parts = []
        lane_guard, _ = self._atomic_lanes(ssa, () if is_scalar else atom_shape)
        if lane_guard is not None:
            guard_parts.append(lane_guard)
        if mask_var:
            guard_parts.append(mask_var)

        # Indent prefix — extra indent inside if block
        has_guard = bool(guard_parts)
        indent = "        " if has_guard else "    "

        # Open guard if-block
        if has_guard:
            guard_cond = " && ".join(guard_parts)
            self.kb.raw_line(f"    if ({guard_cond}) {{")

        if fence_before:
            self.kb.raw_line(f"{indent}{fence_before}")

        if is_16bit_float:
            self._emit_atomic_rmw_16bit(base_ptr, offsets, val_var, rmw_op, half_type, result_var, indent, n)
        elif is_float and rmw_op in ("fadd", "add"):
            # Float atomic add via CAS loop
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint old_bits_{n} = atomic_load_explicit(aptr_{n}, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}while (true) {{")
            self.kb.raw_line(f"{indent}    float old_val_{n} = as_type<float>(old_bits_{n});")
            self.kb.raw_line(f"{indent}    float new_val_{n} = old_val_{n} + {val_var};")
            self.kb.raw_line(f"{indent}    uint new_bits_{n} = as_type<uint>(new_val_{n});")
            self.kb.raw_line(
                f"{indent}    if (atomic_compare_exchange_weak_explicit(aptr_{n}, &old_bits_{n}, new_bits_{n},"
            )
            self.kb.raw_line(f"{indent}            memory_order_relaxed, memory_order_relaxed)) break;")
            self.kb.raw_line(f"{indent}}}")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(old_bits_{n});")
        elif is_float and rmw_op == "exch":
            # Float atomic exchange via reinterpret as uint
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint exch_bits_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(
                f"{indent}uint old_bits_{n} = atomic_exchange_explicit(aptr_{n}, exch_bits_{n}, memory_order_relaxed);"
            )
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(old_bits_{n});")
        elif is_unsigned:
            # Unsigned integer atomics
            msl_fn = _RMW_TO_MSL.get(rmw_op, "atomic_fetch_add_explicit")
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}{result_var} = {msl_fn}(aptr_{n}, (uint){val_var}, memory_order_relaxed);")
        else:
            # Signed integer atomics
            msl_fn = _RMW_TO_MSL.get(rmw_op, "atomic_fetch_add_explicit")
            self.kb.raw_line(f"{indent}device atomic_int* aptr_{n} = (device atomic_int*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}{result_var} = {msl_fn}(aptr_{n}, (int){val_var}, memory_order_relaxed);")

        if fence_after:
            self.kb.raw_line(f"{indent}{fence_after}")

        # Close guard if-block
        if has_guard:
            self.kb.raw_line(f"    }}")

        if atomic_result_shared is not None:
            # Written by the one lane that performed the atomic: in a wrap loop the
            # thread's later lanes did not, and must not put their zero in the slot.
            self.kb.raw_line(f"    if ({lane_guard}) {atomic_result_shared}[0] = {result_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            self.kb.raw_line(f"    {result_var} = ({result_msl_type}){atomic_result_shared}[0];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype
        # Tensor atomic: every participating thread's `old_N` is the old value at its
        # own location. Scalar/tensor<1>: the threadgroup broadcast above makes the one
        # logical old value available identically to every thread. Either is DIRECT for
        # a later 1-D store; only register after that invariant is actually true.
        self._register_1d_layout(ssa.id, "direct")

    def _lower_atomic_cas(self, ssa: SSAValue):
        """tt.atomic_cas → MSL atomic compare-and-swap.

        Operands: [ptr, cmp, val]
        Result: the OLD value at the atomic location.

        CAS semantics: if *ptr == cmp, set *ptr = val. Return old *ptr.
        """
        if len(ssa.operand_ids) < 3:
            return

        native_result = self._prove_atomic_native_contract(ssa)
        fence_before, fence_after = self._atomic_ordering_fences(ssa)

        ptr_id = ssa.operand_ids[0]
        cmp_id = ssa.operand_ids[1]
        val_id = ssa.operand_ids[2]

        cmp_var = self._lookup(cmp_id)
        val_var = self._lookup(val_id)

        # Resolve pointer info
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            offsets = "0"

        # Metal has no 64-bit device atomic — refuse rather than truncate to 32 bits
        # (re-audit #10), mirroring _lower_atomic_rmw.
        if native_result.width != 32:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"{native_result.width}-bit atomic CAS is not supported: Metal has no matching "
                "device atomic and a 32-bit CAS would silently change the value. Refusing.",
                op_name="tt.atomic_cas",
            )

        is_float = native_result.kind == "float"
        if is_float:
            if native_result.elem != "f32":
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "atomic_cas native result is not a supported f32 representation",
                    op_name="tt.atomic_cas",
                )
        elif native_result.kind != "integer" or native_result.elem != "i32":
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "atomic_cas native result is not a supported signless i32 representation",
                op_name="tt.atomic_cas",
            )

        # Scalar CAS: only thread 0 per threadgroup should execute.
        is_scalar = not native_result.is_tensor
        atom_shape = native_result.shape

        n = self._var_counter
        self._var_counter += 1

        # Determine result type
        if is_float:
            result_msl_type = "float"
            result_zero = "0.0f"
            result_dtype = "fp32"
        else:
            result_msl_type = "int"
            result_zero = "0"
            result_dtype = "i32"

        result_var = f"old_{n}"

        # Scalar and tensor<1> CAS each have one logical old value. Only lid 0
        # performs that CAS (the scalar guard or the 1-D underfill guard below),
        # so distribute its return before any splat/broadcast consumer. A local
        # initialized to zero on the other lanes previously produced [old,0,...]
        # for scalar CAS and [old,new,new,...] for an unguarded tensor<1> CAS.
        atomic_result_shared = None
        if is_scalar or all(extent == 1 for extent in atom_shape):
            atomic_result_shared = f"atomic_cas_single_result_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(atomic_result_shared, dtype=result_dtype, size=1)

        # n>1 under-cover guard (mirrors _lower_store): a BLOCK-wide atomic the
        # base path emits as one element per thread (PTR[k+lid]) would silently
        # drop the rest of each tile-stride when block_size > num_threads.
        # Refuse loudly — the only correct n>1 paths are the register-array
        # regime and _loop_e-wrapped emission. For CAS the value-tensor whose
        # shape is the tile width is the `val` operand (operand_ids[2]); `cmp`
        # shares that shape, so either resolves the same refusal.
        _val_shape = self.env_shapes.get(val_id)
        _num_threads = self.kb.block_size
        if (
            not self._mept_single_pass
            and not self._needs_wrapping
            and _val_shape is not None
            and len(_val_shape) >= 1
            and _val_shape[0] > _num_threads
        ):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"Refusing a {_val_shape[0]}-element atomic with only "
                f"{_num_threads} threads: the base path scatters one element "
                f"per thread, so a tile wider than the threadgroup would "
                f"silently drop the rest. Launch with num_warps = BLOCK/32 "
                f"(so num_threads == BLOCK), or reduce BLOCK."
            )

        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Scalar / 1-D underfill guard.
        guard, _ = self._atomic_lanes(ssa, () if is_scalar else atom_shape)
        indent = "    "
        if guard is not None:
            self.kb.raw_line(f"    if ({guard}) {{")
            indent = "        "

        if fence_before:
            self.kb.raw_line(f"{indent}{fence_before}")

        # MSL offers only the WEAK compare-exchange, which may fail spuriously: the location still
        # holds `cmp`, `expected` is reloaded with that same value, and a one-shot call would report
        # old == cmp ("swapped") with nothing stored. Emulate the strong CAS: retry while the failure
        # is spurious (reloaded value == cmp, compared as bits, as the CAS itself compares), stop on
        # a genuine mismatch. On success `expected` still holds cmp — the old value either way.
        if is_float:
            # Float CAS: use atomic_uint + as_type casts
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint cmp_bits_{n} = as_type<uint>((float){cmp_var});")
            self.kb.raw_line(f"{indent}uint expected_{n} = cmp_bits_{n};")
            self.kb.raw_line(f"{indent}uint desired_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(
                f"{indent}while (!atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, desired_{n},"
            )
            self.kb.raw_line(f"{indent}        memory_order_relaxed, memory_order_relaxed)) {{")
            self.kb.raw_line(f"{indent}    if (expected_{n} != cmp_bits_{n}) break;  // genuine mismatch: not spurious")
            self.kb.raw_line(f"{indent}}}")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(expected_{n});")
        else:
            # Integer CAS
            self.kb.raw_line(f"{indent}device atomic_int* aptr_{n} = (device atomic_int*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}int cmp_val_{n} = (int){cmp_var};")
            self.kb.raw_line(f"{indent}int expected_{n} = cmp_val_{n};")
            self.kb.raw_line(
                f"{indent}while (!atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, (int){val_var},"
            )
            self.kb.raw_line(f"{indent}        memory_order_relaxed, memory_order_relaxed)) {{")
            self.kb.raw_line(f"{indent}    if (expected_{n} != cmp_val_{n}) break;  // genuine mismatch: not spurious")
            self.kb.raw_line(f"{indent}}}")
            self.kb.raw_line(f"{indent}{result_var} = expected_{n};")

        if fence_after:
            self.kb.raw_line(f"{indent}{fence_after}")

        if guard is not None:
            self.kb.raw_line(f"    }}")

        if atomic_result_shared is not None:
            self.kb.raw_line(f"    if ({guard}) {atomic_result_shared}[0] = {result_var};")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            self.kb.raw_line(f"    {result_var} = ({result_msl_type}){atomic_result_shared}[0];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype
        self._register_1d_layout(ssa.id, "direct")

    # -- Noinline function calls (tt.call) --

    @staticmethod
    def _sanitize_func_name(name: str) -> str:
        """Sanitize a Triton mangled function name for MSL.

        Replaces dots and other invalid chars with underscores.
        """
        return name.replace(".", "_")
