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

from triton_metal.codegen.mlir_walker import SSAValue, _extract_shape
from triton_metal.codegen.msl_emitter import _msl_compute_type
from triton_metal.codegen.msl_types import triton_type_to_msl

from triton_metal.codegen._lowerer_helpers import _mlir_to_triton_dtype


class _ControlFlowMixin:
    """``scf.*`` and atomic op lowering for ``GenericLowerer``."""

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
        """
        if len(ssa.operand_ids) < 3:
            return

        # i64 loop bounds: the induction lowering assumes 32-bit and does not
        # terminate for 64-bit ranges (the test_for_iv hang). No correct
        # lowering exists yet — refuse loudly rather than hang (Phase 0 T3).
        for bid in ssa.operand_ids[:3]:
            if self.env_types.get(bid) in ("i64", "u64", "ui64"):
                from triton_metal.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "Refusing scf.for with 64-bit loop bounds: the induction "
                    "lowering would not terminate (hang). Use 32-bit bounds.")

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
        # The scf.for result type tells us the true type of iter_args
        result_elem = ssa.elem_type or "f32"  # First result's type
        for i, init_id in enumerate(init_ids):
            init_val = self._lookup(init_id)
            # Prefer result type, fall back to init value type
            init_type = self.env_types.get(init_id, "fp32")
            # Use result type if it's more specific (e.g., i64 vs i32)
            if result_elem in ("i64",) and init_type in ("i32", "fp32"):
                init_type = result_elem

            # Check if this iter_arg is a 2D tensor too large for scalar
            init_shape = self.env_shapes.get(init_id, ())
            init_total = 1
            for d in init_shape:
                init_total *= d
            if len(init_shape) >= 2 and init_total > bs:
                # Allocate persistent shared memory for this iter_arg
                smem_name = f"smem_iter_{self._shared_counter}"
                self._shared_counter += 1
                self.kb.declare_threadgroup_array(smem_name, dtype="fp32",
                                                  size=init_total)
                # Cooperative init from the constant value
                self.kb.raw_line(
                    f"    for (uint _si = lid; _si < {init_total}u; "
                    f"_si += {bs}u) {{")
                self.kb.raw_line(
                    f"        {smem_name}[_si] = {init_val};")
                self.kb.raw_line(f"    }}")
                self.kb.raw_line(
                    f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                iter_vars.append(smem_name)
                iter_dtypes.append(init_type)
                smem_iter_indices.add(i)
                # Register in _shared_mem_descs so downstream ops can find it
                if not hasattr(self, '_shared_mem_descs'):
                    self._shared_mem_descs = {}
                # We will register the block_arg_id below after mapping
                continue

            var_name = self._next_var("iter")
            if init_type.startswith("f") or init_type.startswith("bf"):
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

        # Emit for loop -- use long for i64.
        # scf.for semantics: always `iv < ub` (Triton normalizes negative steps).
        start_type = self.env_types.get(ssa.operand_ids[0], "i32")
        is_i64 = start_type == "i64" or "i64" in (ssa.type_str or "")
        loop_type = "long" if is_i64 else "int"
        loop_var = self._next_var("k")

        self.kb.raw_line(
            f"    for ({loop_type} {loop_var} = {start_var}; "
            f"{loop_var} < {end_var}; {loop_var} += {step_var}) {{"
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
                    # Register shared-memory-backed iter_args
                    if i in smem_iter_indices:
                        init_shape = self.env_shapes.get(
                            init_ids[i], ()) if i < len(init_ids) else ()
                        if not hasattr(self, '_shared_mem_descs'):
                            self._shared_mem_descs = {}
                        self._shared_mem_descs[ba_id] = (var, init_shape, "fp32")
                        # Also track that this block_arg is smem-backed so
                        # that scf.yield can skip the scalar assignment.
                        if not hasattr(self, '_smem_iter_args'):
                            self._smem_iter_args = {}
                        self._smem_iter_args[ba_id] = var

        # Process body ops.  Track the yielded SSA id per iter_arg so we can
        # propagate metadata (e.g., _bcast_layout) from the yielded value to
        # the scf.for result variable below.
        yielded_ids: list = [None] * n_iter_args
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
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
                                yield_smem = getattr(self, '_shared_mem_descs', {}).get(yield_id)
                                if yield_smem and yield_smem[0] != iter_vars[i]:
                                    src_smem = yield_smem[0]
                                    dst_smem = iter_vars[i]
                                    init_shape = self.env_shapes.get(
                                        init_ids[i], ()) if i < len(init_ids) else ()
                                    sz = 1
                                    for d in init_shape:
                                        sz *= d
                                    self.kb.raw_line(
                                        f"    for (uint _cp = lid; _cp < {sz}u; "
                                        f"_cp += {bs}u) {{")
                                    self.kb.raw_line(
                                        f"        {dst_smem}[_cp] = {src_smem}[_cp];")
                                    self.kb.raw_line(f"    }}")
                                    self.kb.raw_line(
                                        f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                                continue
                            yield_val = self._lookup(yield_id)
                            self.kb.raw_line(
                                f"        {iter_vars[i]} = {yield_val};"
                            )
                else:
                    self._lower_op(body_op)

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
                    # Propagate shared_mem_desc for oversized iter_args
                    if i in smem_iter_indices:
                        init_shape = self.env_shapes.get(
                            init_ids[i], ()) if i < len(init_ids) else ()
                        if not hasattr(self, '_shared_mem_descs'):
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

        # Check both then and else for yield with operands
        all_body_ops = list(ssa.region_ops or []) + list(ssa.else_ops or [])
        has_results = any(
            body_op.op == "scf.yield" and body_op.operand_ids
            for body_op in all_body_ops
        )

        # For scf.if with results, declare result variables before the if/else
        result_vars = []
        if has_results:
            # Infer result types from yield operands
            yield_types = []
            for body_op in all_body_ops:
                if body_op.op == "scf.yield" and body_op.operand_ids:
                    for yid in body_op.operand_ids:
                        yt = self.env_types.get(yid, "fp32")
                        yield_types.append(yt)
                    break

            for i, rid in enumerate(result_ids):
                var_name = f"ifr_{abs(rid)}_{i}"
                result_vars.append((rid, var_name))
                yt = yield_types[i] if i < len(yield_types) else "fp32"
                if yt.startswith("fp") or yt.startswith("bf") or yt.startswith("f"):
                    msl_type = "float"
                elif yt.startswith("u"):
                    msl_type = "uint"
                else:
                    msl_type = "int"
                self.kb.raw_line(f"    {msl_type} {var_name};")

        self.kb.raw_line(f"    if ({cond}) {{")

        # Lower "then" body
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            yield_val = self._lookup(yield_id)
                            rid, var_name = result_vars[i]
                            self.kb.raw_line(f"        {var_name} = {yield_val};")
                else:
                    self._lower_op(body_op)

        # Lower "else" body
        if ssa.else_ops:
            self.kb.raw_line("    } else {")
            for body_op in ssa.else_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            yield_val = self._lookup(yield_id)
                            rid, var_name = result_vars[i]
                            self.kb.raw_line(f"        {var_name} = {yield_val};")
                else:
                    self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map result variables into env with proper types
        for i, (rid, var_name) in enumerate(result_vars):
            self.env[rid] = var_name
            # Propagate type from yield operands
            yt = yield_types[i] if i < len(yield_types) else "fp32"
            self.env_types[rid] = yt

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
        for i, init_id in enumerate(init_ids):
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

        # Lower "before" region (condition evaluation)
        for body_op in (ssa.region_ops or []):
            if body_op.op == "scf.condition":
                # First operand is the condition
                if body_op.operand_ids:
                    cond_var = self._lookup(body_op.operand_ids[0])
                    self.kb.raw_line(f"        if (!({cond_var})) break;")
                # Remaining operands are forwarded values to "after" block args
                after_block_args = ssa.attrs.get("else_block_arg_ids", [])
                for j, fwd_id in enumerate(body_op.operand_ids[1:]):
                    if j < len(after_block_args):
                        fwd_val = self._lookup(fwd_id)
                        self.env[after_block_args[j]] = fwd_val
                        fwd_type = self.env_types.get(fwd_id, "i32")
                        self.env_types[after_block_args[j]] = fwd_type
            else:
                self._lower_op(body_op)

        # If "after" block args weren't mapped by scf.condition forwarding,
        # map them to iter_vars directly (they share the same values)
        after_block_args = ssa.attrs.get("else_block_arg_ids", [])
        for i, var in enumerate(iter_vars):
            if i < len(after_block_args) and after_block_args[i] not in self.env:
                self.env[after_block_args[i]] = var
                self.env_types[after_block_args[i]] = iter_dtypes[i]

        # Lower "after" region (loop body)
        for body_op in (ssa.else_ops or []):
            if body_op.op == "scf.yield":
                # Update iter_arg variables from yield operands
                for j, yield_id in enumerate(body_op.operand_ids):
                    if j < len(iter_vars):
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
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"
        elif iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"

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

        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]
        mask_id = ssa.operand_ids[2] if len(ssa.operand_ids) >= 3 else None

        rmw_op = ssa.attrs.get("rmw_op", "add")
        val_var = self._lookup(val_id)

        # Resolve pointer info
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            offsets = "0"

        # Determine value type (int vs float)
        val_dtype = self.env_types.get(val_id, "fp32")
        is_float = val_dtype.startswith("fp") or val_dtype.startswith("bf") or val_dtype.startswith("f")

        # Determine the storage element type from the pointer arg
        store_dtype = self._trace_ptr_dtype(ptr_id)
        is_float_ptr = store_dtype.startswith("fp") or store_dtype.startswith("bf")

        # Use float detection: if either the value or the pointer is float
        is_float = is_float or is_float_ptr

        # Refuse 16-bit float atomics (audit C2). The float-atomic path below
        # reinterprets the slot as a 32-bit word (CAS loop on atomic_uint*,
        # result_dtype hardcoded fp32). For an fp16/bf16 element that reads and
        # writes 4 bytes over a 2-byte value — silently corrupting both it and
        # its neighbor. Metal has no 16-bit device atomic, so there is no
        # correct lowering: refuse loudly rather than emit wrong output.
        if (val_dtype in ("fp16", "bf16", "f16")
                or store_dtype in ("fp16", "bf16", "f16")):
            from triton_metal.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "Refusing to emit silently-wrong output: atomic_rmw on a 16-bit "
                f"float ({val_dtype}/{store_dtype}) is not supported — Metal has "
                "no 16-bit device atomic and the 32-bit CAS loop would corrupt "
                "the 2-byte value. Accumulate in fp32 (atomic on an fp32 buffer) "
                "and cast, or restructure to avoid the atomic.")

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
        if is_float and rmw_op in ("fadd", "add", "exch"):
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
        is_scalar = not ssa.is_tensor

        # In 2D kernels, a 1D atomic tensor (e.g. after 2D→1D reduce) must
        # only execute on the first N threads, not all M*N threads.
        atomic_1d_guard = None
        if self._is_2d and ssa.is_tensor:
            atom_shape = _extract_shape(ssa.type_str)
            if len(atom_shape) == 1 and atom_shape[0] < self.effective_block_size:
                atomic_1d_guard = atom_shape[0]

        # Always declare result variable first (needed for mask or not)
        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Build the guard condition
        guard_parts = []
        if is_scalar:
            guard_parts.append("lid == 0")
        elif atomic_1d_guard is not None:
            guard_parts.append(f"lid < {atomic_1d_guard}u")
        if mask_var:
            guard_parts.append(mask_var)

        # Indent prefix — extra indent inside if block
        has_guard = bool(guard_parts)
        indent = "        " if has_guard else "    "

        # Open guard if-block
        if has_guard:
            guard_cond = " && ".join(guard_parts)
            self.kb.raw_line(f"    if ({guard_cond}) {{")

        if is_float and rmw_op in ("fadd", "add"):
            # Float atomic add via CAS loop
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint old_bits_{n} = atomic_load_explicit(aptr_{n}, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}while (true) {{")
            self.kb.raw_line(f"{indent}    float old_val_{n} = as_type<float>(old_bits_{n});")
            self.kb.raw_line(f"{indent}    float new_val_{n} = old_val_{n} + {val_var};")
            self.kb.raw_line(f"{indent}    uint new_bits_{n} = as_type<uint>(new_val_{n});")
            self.kb.raw_line(f"{indent}    if (atomic_compare_exchange_weak_explicit(aptr_{n}, &old_bits_{n}, new_bits_{n},")
            self.kb.raw_line(f"{indent}            memory_order_relaxed, memory_order_relaxed)) break;")
            self.kb.raw_line(f"{indent}}}")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(old_bits_{n});")
        elif is_float and rmw_op == "exch":
            # Float atomic exchange via reinterpret as uint
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint exch_bits_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(f"{indent}uint old_bits_{n} = atomic_exchange_explicit(aptr_{n}, exch_bits_{n}, memory_order_relaxed);")
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

        # Close guard if-block
        if has_guard:
            self.kb.raw_line(f"    }}")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype

    def _lower_atomic_cas(self, ssa: SSAValue):
        """tt.atomic_cas → MSL atomic compare-and-swap.

        Operands: [ptr, cmp, val]
        Result: the OLD value at the atomic location.

        CAS semantics: if *ptr == cmp, set *ptr = val. Return old *ptr.
        """
        if len(ssa.operand_ids) < 3:
            return

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

        # Determine value type
        val_dtype = self.env_types.get(val_id, "i32")
        is_float = val_dtype.startswith("fp") or val_dtype.startswith("bf") or val_dtype.startswith("f")

        # Also check pointer type
        store_dtype = self._trace_ptr_dtype(ptr_id)
        is_float_ptr = store_dtype.startswith("fp") or store_dtype.startswith("bf")
        is_float = is_float or is_float_ptr

        # Scalar CAS: only thread 0 per threadgroup should execute
        is_scalar = not ssa.is_tensor

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
        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Scalar guard
        indent = "    "
        if is_scalar:
            self.kb.raw_line(f"    if (lid == 0) {{")
            indent = "        "

        if is_float:
            # Float CAS: use atomic_uint + as_type casts
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint expected_{n} = as_type<uint>((float){cmp_var});")
            self.kb.raw_line(f"{indent}uint desired_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(f"{indent}atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, desired_{n},")
            self.kb.raw_line(f"{indent}    memory_order_relaxed, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(expected_{n});")
        else:
            # Integer CAS
            self.kb.raw_line(f"{indent}device atomic_int* aptr_{n} = (device atomic_int*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}int expected_{n} = (int){cmp_var};")
            self.kb.raw_line(f"{indent}atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, (int){val_var},")
            self.kb.raw_line(f"{indent}    memory_order_relaxed, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}{result_var} = expected_{n};")

        if is_scalar:
            self.kb.raw_line(f"    }}")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype

    # -- Noinline function calls (tt.call) --

    @staticmethod
    def _sanitize_func_name(name: str) -> str:
        """Sanitize a Triton mangled function name for MSL.

        Replaces dots and other invalid chars with underscores.
        """
        return name.replace(".", "_")

