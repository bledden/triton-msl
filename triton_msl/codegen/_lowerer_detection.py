"""Pattern detection predicates for ``GenericLowerer``.

Each ``_detect_*`` method scans the kernel\'s IRGraph (``self.graph``) and
returns a non-None ``info`` dict if the kernel matches a recognized pattern
that a corresponding ``_lower_*_template`` emitter knows how to handle. The
main ``lower()`` dispatch tries them in order and falls through to the
generic op-by-op lowering when none match.

Conservative by design: any deviation from the canonical pattern returns
None and the generic path is used instead.
"""

import re

from triton_msl.codegen.mlir_walker import SSAValue, _extract_shape

from triton_msl.codegen._lowerer_helpers import _mlir_to_triton_dtype


# Epilogue ops that don't compute a new value — they only reshape the layout or
# round the representation. In the per-element fused-epilogue loop they resolve
# to their operand's expression (no emitted statement). Output dtype casts are
# absorbed by the final store's cast. #158.
_EPI_PASSTHROUGH = frozenset({
    "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape",
    "ttg.convert_layout", "arith.truncf", "arith.extf",
    "arith.sitofp", "arith.fptosi",
})


class _DetectionMixin:
    """Pattern-detection predicates for GenericLowerer.

    All methods read instance state (``self.graph``, ``self.env_types``,
    ``self.ssa_values``, etc.) — they do not define new state.
    """

    def _trace_ptr_source(self, ssa_id, op_by_id=None, depth=0):
        """Walk the value chain from an SSA id back to a kernel ``FuncArg``.

        Follows the typical ``ttg.local_load → ttg.local_alloc →
        ttg.memdesc_trans? → tt.trans? → tt.reshape? → tt.load →
        tt.addptr* → tt.splat → <func-arg>`` chain. Returns the matching
        ``FuncArg`` or ``None`` if no path lands on one (e.g. for
        constant-initialized accumulators).
        """
        if op_by_id is None:
            op_by_id = {}
            def _collect(ops):
                for s in ops:
                    op_by_id[s.id] = s
                    if s.region_ops:
                        _collect(s.region_ops)
                    if s.else_ops:
                        _collect(s.else_ops)
            _collect(self.graph.ops)
        if depth > 32:
            return None
        # Function-arg ids: check against graph.args.
        for arg in self.graph.args:
            if arg.id == ssa_id and arg.is_ptr:
                return arg
        op = op_by_id.get(ssa_id)
        if not op or not op.operand_ids:
            return None
        # ``tt.addptr`` and ``tt.splat`` use operand 0 as the base ptr;
        # ``ttg.local_load`` / ``ttg.local_alloc`` / ``ttg.memdesc_trans``
        # / ``tt.trans`` / ``tt.reshape`` / ``tt.load`` / ``arith.*``
        # all pass through the value (or address) chain via operand 0
        # too, so a single recursive walk handles every step.
        return self._trace_ptr_source(op.operand_ids[0], op_by_id, depth + 1)

    def _resolve_dot_ptr_roles(self, dot_ssa, all_ptr_args):
        """Return ``[A_ptr, B_ptr, C_ptr]`` by tracing dot operands and the
        ``tt.store`` target back to their kernel function args.

        Falls back to the function-arg-declaration order when any leg
        of the trace fails. The caller treats a ``None`` return as
        \"use declaration order\".
        """
        if dot_ssa is None or len(dot_ssa.operand_ids) < 2:
            return None
        # Build op index once.
        op_by_id = {}
        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)
        _collect(self.graph.ops)
        a_arg = self._trace_ptr_source(dot_ssa.operand_ids[0], op_by_id)
        b_arg = self._trace_ptr_source(dot_ssa.operand_ids[1], op_by_id)
        # Find the (single) tt.store and trace its address operand.
        c_arg = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.store" and ssa.operand_ids:
                c_arg = self._trace_ptr_source(ssa.operand_ids[0], op_by_id)
                break
        # All three must resolve to distinct args.
        ptrs = [a_arg, b_arg, c_arg]
        if any(p is None for p in ptrs):
            return None
        if len({p.name for p in ptrs}) != 3:
            return None
        # Append any extra unused ptr args at the end (preserves
        # downstream slicing that may want ``ptr_args[3]`` for a W bias).
        extras = [p for p in all_ptr_args if p.name not in {q.name for q in ptrs}]
        return ptrs + extras

    def _detect_simple_dot(self):
        """Detect a simple dot kernel: load→local_alloc→local_load→dot→store.

        Returns dict with {M, N, K, ptr_args, dot_ssa} if detected, None otherwise.

        Handles two patterns:
        1. Simple (no scf.for): tile fits in one block, no K-loop needed.
        2. K-loop (scf.for wrapping tt.dot): K > BLOCK_K, accumulate across tiles.
           Returns extra fields: has_k_loop=True, BLOCK_K, BLOCK_M, BLOCK_N,
           and scalar_args for M/N/K runtime values.

        Rejects kernels with stride args (those go through the strided template).
        """
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        has_strides = any("stride" in a.name.lower() for a in scalar_args)
        if has_strides:
            return None

        # A single-dot matmul with a trailing compute epilogue the fused template
        # didn't claim (looped dot-in-K-loop, etc.) must REFUSE — the inline simdgroup
        # lowering stores the RAW accumulator and would SILENTLY DROP the epilogue
        # (re-audit #6), and routing to the generic lowerer mis-tiles it. See
        # _has_unhandled_matmul_compute_epilogue. (_detect_matmul_epilogue already
        # claimed the non-looped single-dot case the template CAN emit.)
        if self._has_unhandled_matmul_compute_epilogue():
            from triton_msl.codegen.generic_lowerer import _MATMUL_EPILOGUE_REFUSE_MSG
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(_MATMUL_EPILOGUE_REFUSE_MSG, op_name="tt.dot")

        # Find scf.for and check if it contains tt.dot (K-loop pattern)
        scf_for_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "scf.for":
                scf_for_ssa = ssa
                break

        def _const_scf_iters(scf_op):
            """Return the static iteration count of an scf.for if its
            bounds are compile-time constants (``%c0_i32 to %cN_i32 step %c1_i32``),
            else ``None``."""
            if not scf_op or not scf_op.operand_ids or len(scf_op.operand_ids) < 3:
                return None
            op_by_id = {s.id: s for s in self.graph.ops}
            def _const_val(sid):
                s = op_by_id.get(sid)
                if not s or s.op != "arith.constant":
                    return None
                return s.attrs.get("value")
            lo = _const_val(scf_op.operand_ids[0])
            hi = _const_val(scf_op.operand_ids[1])
            step = _const_val(scf_op.operand_ids[2])
            if lo is None or hi is None or step is None or step == 0:
                return None
            try:
                return max(0, (int(hi) - int(lo) + int(step) - 1) // int(step))
            except (TypeError, ValueError):
                return None

        scf_iters = _const_scf_iters(scf_for_ssa)

        if scf_for_ssa:
            # Check if the scf.for body contains tt.dot
            dot_in_loop = None
            has_loads_in_loop = False
            has_local_alloc_in_loop = False
            has_store_in_loop = False
            if scf_for_ssa.region_ops:
                for body_op in scf_for_ssa.region_ops:
                    if body_op.op == "tt.dot":
                        dot_in_loop = body_op
                    elif body_op.op == "tt.load":
                        has_loads_in_loop = True
                    elif body_op.op == "ttg.local_alloc":
                        has_local_alloc_in_loop = True
                    elif body_op.op == "tt.store":
                        has_store_in_loop = True

            if not dot_in_loop:
                return None  # scf.for without dot — not our pattern

            # A tt.store INSIDE the loop body means this is a TILE-iteration loop (the
            # body computes + writes one output tile per iteration), NOT a K-reduction
            # loop. The inline simdgroup template assumes a K-loop (accumulate, store
            # once after) and mis-computes a tile-loop (re-audit #11). Refuse.
            if has_store_in_loop:
                from triton_msl.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "matmul with a tt.store inside the scf.for loop body is a "
                    "tile-iteration loop, not a K-reduction loop; the inline simdgroup "
                    "template only handles the K-loop form and would mis-compute. "
                    "Refusing.", op_name="tt.dot")

            # A non-zero accumulator INIT (a fused bias / tl.full) is NOT seeded into the
            # simdgroup accumulators by the inline template, so it is silently DROPPED
            # (re-audit #11/#12). In the STANDARD K-loop the accumulator is an scf.for
            # ITER-ARG, so its init is the for-op's initial operand (operand_ids[3:],
            # after lo/hi/step) — NOT a dot operand in graph.ops. The original #11 guard
            # looked at the dot operand and missed every loop-carried accumulator. Check
            # the loop inits: refuse if any is not a zero constant (a non-zero constant,
            # or a non-constant load/broadcast bias).
            by_id_all = {s.id: s for s in self.graph.ops}

            def _init_is_bias(_id):
                # True only when the loop-carried init is CLEARLY a fused bias the inline
                # template drops: a loaded value (tt.load through splat/broadcast) or a
                # NON-ZERO constant. A zero init (the standard tl.zeros accumulator, in
                # whatever lowered form) or an unrecognized form returns False so we never
                # over-refuse a normal matmul.
                _op = by_id_all.get(_id); _seen = set()
                while _op is not None and _op.id not in _seen:
                    _seen.add(_op.id)
                    if _op.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout",
                                  "tt.reshape", "tt.expand_dims"):
                        _op = by_id_all.get(_op.operand_ids[0]) if _op.operand_ids else None
                        continue
                    break
                if _op is None:
                    return False
                if _op.op == "tt.load":
                    return True
                if _op.op == "arith.constant":
                    return any(ch in "123456789"
                               for ch in str(_op.attrs.get("value", "")))
                return False
            for _init_id in (scf_for_ssa.operand_ids[3:]
                             if len(scf_for_ssa.operand_ids) > 3 else []):
                if _init_is_bias(_init_id):
                    from triton_msl.errors import MetalNonRecoverableError
                    raise MetalNonRecoverableError(
                        "K-loop matmul with a non-zero loop-carried accumulator init "
                        "(fused bias / tl.full) is not supported by the inline simdgroup "
                        "template (the init is silently dropped). Refusing. Add the bias "
                        "as a separate kernel after the matmul.", op_name="tt.dot")

            # Extract BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N from dot operands
            a_type = self._find_op_type_str(dot_in_loop.operand_ids[0])
            b_type = self._find_op_type_str(dot_in_loop.operand_ids[1])
            a_shape = _extract_shape(a_type) if a_type else None
            b_shape = _extract_shape(b_type) if b_type else None

            if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
                return None

            BLOCK_M, BLOCK_K = a_shape[0], a_shape[1]
            BLOCK_K2, BLOCK_N = b_shape[0], b_shape[1]

            all_ptr_args = [a for a in self.graph.args if a.is_ptr]
            if len(all_ptr_args) < 3:
                return None

            # Reorder ``ptr_args`` to (A, B, C) by tracing the dot\'s
            # operand_a / operand_b sources and the tt.store target.
            # Falling back to function-arg-declaration order gives wrong
            # results when the kernel lists ``(Z, X, Y)`` like
            # ``test_dot_mulbroadcasted``.
            ptr_args = self._resolve_dot_ptr_roles(
                dot_in_loop, all_ptr_args) or all_ptr_args
            if len(ptr_args) < 3:
                return None

            # Try to find M, N, K scalar args by name
            scalar_arg_map = {a.name: a for a in scalar_args}

            return {
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
                "ptr_args": ptr_args, "dot_ssa": dot_in_loop,
                "has_k_loop": True,
                "scalar_args": scalar_arg_map,
                "all_scalar_args": scalar_args,
                # When K is a constexpr (not a runtime scalar arg) the
                # template can\'t read it from a buffer. Pre-compute the
                # full ``_K = BLOCK_K * scf_iters`` from the scf.for
                # bounds when those are constants (``test_dot_mulbroadcasted``).
                "scf_iters": scf_iters,
            }

        # --- Non-K-loop: simple dot without scf.for ---
        dot_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_ssa = ssa
                break
        if not dot_ssa or len(dot_ssa.operand_ids) < 3:
            return None

        # A non-zero accumulator INIT (a fused bias / tl.full) on the non-looped dot is
        # silently DROPPED by _lower_simple_dot_inline (it emits acc0(0)) — the TWIN of
        # the K-loop guard above (re-audit #12). Here the init IS a dot operand in
        # graph.ops (no scf.for iter-arg). Refuse if it is not a zero constant.
        _by0 = {s.id: s for s in self.graph.ops}

        def _init_is_bias0(_id):
            # True only when the dot's accumulator init is CLEARLY a fused bias (a loaded
            # value or a non-zero constant); zero/unrecognized inits return False so a
            # normal matmul (tl.zeros accumulator) is never over-refused.
            _op = _by0.get(_id); _seen = set()
            while _op is not None and _op.id not in _seen:
                _seen.add(_op.id)
                if _op.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout",
                              "tt.reshape", "tt.expand_dims"):
                    _op = _by0.get(_op.operand_ids[0]) if _op.operand_ids else None
                    continue
                break
            if _op is None:
                return False
            if _op.op == "tt.load":
                return True
            if _op.op == "arith.constant":
                return any(ch in "123456789" for ch in str(_op.attrs.get("value", "")))
            return False
        if _init_is_bias0(dot_ssa.operand_ids[2]):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "non-looped matmul with a non-zero accumulator init (fused bias / "
                "tl.full) is silently dropped by the inline simdgroup template. "
                "Refusing. Add the bias as a separate kernel after the matmul.",
                op_name="tt.dot")

        # Detect a post-dot EPILOGUE on the dot result. _lower_simple_dot_inline
        # emits a bare matmul + store; if the dot result feeds any value-changing
        # op before the store (bias add, activation, scale, ...), that op was
        # SILENTLY DROPPED -> the kernel returned A@B (confirmed: matmul*3+1 and
        # matmul->relu both came back as bare A@B). The softmax epilogue is the
        # one fused form we support, and `lower()` checks _detect_matmul_softmax
        # BEFORE this, so any epilogue still present here is an UNSUPPORTED one;
        # no path computes a matmul-sized dot + arbitrary epilogue correctly
        # (the per-thread generic lowerer is wrong at 16/32/64). Refuse loudly
        # rather than emit wrong numbers. Only layout changes / output dtype
        # casts are value-preserving passthroughs.
        _passthrough = {
            "ttg.convert_layout", "tt.reshape", "tt.trans",
            "arith.truncf", "arith.extf", "arith.bitcast",
            "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui",
            "tt.fp_to_fp",
        }
        _seen = set()
        _frontier = [dot_ssa.id]
        while _frontier:
            _vid = _frontier.pop()
            if _vid in _seen:
                continue
            _seen.add(_vid)
            for _op in self.graph.ops:
                if _vid not in (_op.operand_ids or []):
                    continue
                if _op.op == "tt.store":
                    continue                      # terminal — fine
                if _op.op in _passthrough:
                    _frontier.append(_op.id)      # follow representation change
                else:
                    from triton_msl.errors import MetalNonRecoverableError
                    raise MetalNonRecoverableError(
                        f"matmul with a fused '{_op.op}' epilogue on the dot "
                        "result is not supported (only softmax is fused). The "
                        "simple-dot path would silently drop it. Split the "
                        "epilogue into a separate kernel, or apply it after a "
                        "K-loop matmul store.")

        # Verify the dot operands come from loads (not constants)
        # Trace: dot ← local_load ← local_alloc ← tt.load
        has_loads = False
        for load_op in self.graph.ops:
            if load_op.op == "tt.load":
                has_loads = True
                break
        if not has_loads:
            return None

        # Verify there are local_alloc/local_load ops (the shared memory path)
        has_local_alloc = any(
            ssa.op == "ttg.local_alloc" for ssa in self.graph.ops
        )
        if not has_local_alloc:
            return None

        # Extract shapes from dot operands
        a_type = self._find_op_type_str(dot_ssa.operand_ids[0])
        b_type = self._find_op_type_str(dot_ssa.operand_ids[1])
        a_shape = _extract_shape(a_type) if a_type else None
        b_shape = _extract_shape(b_type) if b_type else None

        if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
            return None

        # tt.dot operates on the two innermost dims; any leading dims form a
        # broadcast batch. Both operands must have the same batch shape.
        if a_shape[:-2] != b_shape[:-2]:
            return None
        batch_dims = list(a_shape[:-2])
        M, K = a_shape[-2], a_shape[-1]
        K2, N = b_shape[-2], b_shape[-1]
        batch_size = 1
        for d in batch_dims:
            batch_size *= d

        all_ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(all_ptr_args) < 3:
            return None

        # Reorder to (A, B, C) by tracing the dot operand sources and the
        # store target. The function-arg-declaration order can put C
        # ahead of A/B (e.g. ``kernel(Z, X, Y, ...)`` in
        # ``test_dot_mulbroadcasted``).
        ptr_args = self._resolve_dot_ptr_roles(
            dot_ssa, all_ptr_args) or all_ptr_args

        # Detect whether each dot operand is transposed. ``tl.trans`` before
        # tt.dot can land in TTGIR three ways depending on rank:
        #   - Rank-2 inputs:  local_alloc → memdesc_trans → local_load → dot
        #     (transpose is folded into the memdesc layout swap).
        #   - Rank-3 inputs:  tt.trans → local_alloc → local_load → dot
        #     (transpose is a tensor op before shared-memory alloc).
        #   - Rank-4+ inputs: tt.trans → tt.reshape → local_alloc → ...
        #     (a reshape collapses leading batch dims into a single batch
        #     after the transpose).
        # We accept the trans only if its ``order`` swaps the last two dims
        # and is identity on the rest — that's the matmul-relevant transpose.
        op_by_id = {op.id: op for op in self.graph.ops}

        def _trans_is_inner_swap(trans_op):
            order = trans_op.attrs.get("order")
            if order is None:
                # The walker doesn\'t always populate ``order``; fall back to
                # shape comparison. tt.trans with inner-2-dim swap maps an
                # input of shape (..., M, K) to (..., K, M).
                # If we can\'t tell, assume yes (matches the common matmul case).
                return True
            order = list(order)
            n = len(order)
            if n < 2:
                return False
            return (order[:n - 2] == list(range(n - 2))
                    and order[n - 2:] == [n - 1, n - 2])

        def _walk_back_to_trans(start_id, max_steps=4):
            """Follow tt.reshape / layout-only ops back from ``start_id``,
            return the first tt.trans found whose order is an inner swap,
            or None.
            """
            current_id = start_id
            for _ in range(max_steps):
                op = op_by_id.get(current_id)
                if not op or not op.operand_ids:
                    return None
                if op.op == "tt.trans":
                    return op if _trans_is_inner_swap(op) else None
                if op.op in ("tt.reshape", "ttg.convert_layout"):
                    current_id = op.operand_ids[0]
                    continue
                return None
            return None

        def _is_trans(operand_id):
            load_op = op_by_id.get(operand_id)
            if not load_op or load_op.op != "ttg.local_load":
                return False
            if not load_op.operand_ids:
                return False
            src = op_by_id.get(load_op.operand_ids[0])
            if not src:
                return False
            if src.op == "ttg.memdesc_trans":
                return True
            if src.op == "ttg.local_alloc" and src.operand_ids:
                return _walk_back_to_trans(src.operand_ids[0]) is not None
            return False

        trans_a = _is_trans(dot_ssa.operand_ids[0])
        trans_b = _is_trans(dot_ssa.operand_ids[1])

        return {
            "M": M, "N": N, "K": K,
            "ptr_args": ptr_args, "dot_ssa": dot_ssa,
            "trans_a": trans_a, "trans_b": trans_b,
            "batch_size": batch_size,
        }


    def _detect_matmul_softmax(self):
        """Detect the matmul → row-softmax → store fused kernel pattern.

        Triton lowers ``tl.dot`` followed by softmax into:
          tt.dot                              # 2-D result, shape (M, N)
          tt.reduce(axis=1, maxnumf)          # row max,   (M,)
          tt.expand_dims + tt.broadcast       # back to (M, N)
          arith.subf                          # subtract max
          math.exp
          tt.reduce(axis=1, addf)             # row sum,   (M,)
          tt.expand_dims + tt.broadcast       # back to (M, N)
          arith.divf
          (ttg.convert_layout)                # optional
          tt.store

        The generic op-by-op lowerer can\'t handle cooperative ops over
        more than 1024 elements (Metal threadgroup cap), and a 64×64 dot
        product is 4096 elements. ``_requires_matmul_template`` refuses
        because ``has_reduce`` is True, so the kernel hits UNSUPPORTED and
        the legacy text parser silently substitutes a bare matmul template
        that drops the softmax. Detecting the full pattern lets us emit a
        single fused kernel that stages the dot result in shared memory
        and does row softmax cooperatively before the store.

        Returns dict with M/N/K/ptr_args/strides/dtypes when matched, else
        None. M, N, K are read from the dot operand shapes; strides come
        from the kernel\'s scalar arg list.
        """
        # Locate the single tt.dot.
        dot_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                if dot_ssa is not None:
                    return None
                dot_ssa = ssa
        if dot_ssa is None or len(dot_ssa.operand_ids) < 2:
            return None

        # Get M, N, K from dot operand and result shapes.
        a_type = self._find_op_type_str(dot_ssa.operand_ids[0])
        b_type = self._find_op_type_str(dot_ssa.operand_ids[1])
        a_shape = _extract_shape(a_type) if a_type else None
        b_shape = _extract_shape(b_type) if b_type else None
        if (not a_shape or not b_shape
                or len(a_shape) != 2 or len(b_shape) != 2):
            return None  # batched dot not handled by this template
        M, K = a_shape
        K2, N = b_shape
        if K != K2:
            return None

        # Walk the post-dot ops looking for the softmax signature. The exact
        # order Triton emits is dot → reduce(max) → expand → broadcast →
        # subf → exp → reduce(sum) → expand → broadcast → divf → store
        # (with an optional convert_layout between divf and store).
        op_index = {ssa.id: i for i, ssa in enumerate(self.graph.ops)}

        def _consumer_of(producer_id, expected_op):
            for ssa in self.graph.ops:
                if ssa.op == expected_op and producer_id in (ssa.operand_ids or []):
                    return ssa
            return None

        def _reduce_op(ssa):
            """Inspect a tt.reduce\'s region to identify maxnumf / addf."""
            if not ssa.region_ops:
                return None
            for body in ssa.region_ops:
                if body.op in ("arith.maxnumf", "arith.maximumf"):
                    return "max"
                if body.op in ("arith.addf",):
                    return "add"
            return None

        max_reduce = _consumer_of(dot_ssa.id, "tt.reduce")
        if max_reduce is None or _reduce_op(max_reduce) != "max":
            return None
        if max_reduce.attrs.get("axis") != 1:
            return None

        # Trace expand_dims → broadcast → subf
        max_expand = _consumer_of(max_reduce.id, "tt.expand_dims")
        if max_expand is None:
            return None
        max_bcast = _consumer_of(max_expand.id, "tt.broadcast")
        if max_bcast is None:
            return None
        sub = _consumer_of(max_bcast.id, "arith.subf")
        if sub is None or dot_ssa.id not in (sub.operand_ids or []):
            return None
        exp_op = _consumer_of(sub.id, "math.exp")
        if exp_op is None:
            return None
        sum_reduce = _consumer_of(exp_op.id, "tt.reduce")
        if sum_reduce is None or _reduce_op(sum_reduce) != "add":
            return None
        if sum_reduce.attrs.get("axis") != 1:
            return None
        sum_expand = _consumer_of(sum_reduce.id, "tt.expand_dims")
        if sum_expand is None:
            return None
        sum_bcast = _consumer_of(sum_expand.id, "tt.broadcast")
        if sum_bcast is None:
            return None
        div = _consumer_of(sum_bcast.id, "arith.divf")
        if div is None or exp_op.id not in (div.operand_ids or []):
            return None

        # The divf feeds the store, possibly through a layout-only chain
        # of ``arith.truncf`` (fp32 → fp16 downcast when out_dtype is half),
        # ``arith.extf`` (fp16 → fp32 upcast), and ``ttg.convert_layout``.
        # Walk forward until we hit the store or run out of layout-only
        # ops; anything else (another reduce, a second math op, …) means
        # this is a richer kernel that the template can\'t reproduce.
        final_id = div.id
        for _ in range(4):
            next_op = None
            for cand_op in ("arith.truncf", "arith.extf", "ttg.convert_layout"):
                cand = _consumer_of(final_id, cand_op)
                if cand is not None:
                    next_op = cand
                    break
            if next_op is None:
                break
            final_id = next_op.id
        store = _consumer_of(final_id, "tt.store")
        if store is None:
            return None

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        if len(ptr_args) < 3:
            return None

        # Identify X / Y / Z pointer args. The upstream test_dot kernel
        # passes (ptr, row_stride, col_stride) triples per matrix in arg
        # order: (X, stride_xm, stride_xk) for A∈ℝ^{M×K},
        # (Y, stride_yk, stride_yn) for B∈ℝ^{K×N}, plus a chain-dot
        # weight (W) and the output (Z, stride_zm, stride_zn). The
        # softmax variant doesn\'t use W, so the store target is the
        # *last* pointer; everything else identifies positionally.
        a_ptr = ptr_args[0]
        b_ptr = ptr_args[1]
        c_ptr = ptr_args[-1]

        # Read the two stride args immediately following each pointer in
        # the kernel signature. Triton\'s naming for the inner dim varies
        # (``stride_xk`` vs ``stride_ym`` vs ``stride_zn``), so positional
        # adjacency is the only convention reliable across matrices.
        def _strides_after(ptr_arg):
            idx = ptr_arg.index
            row = None
            col = None
            for a in scalar_args:
                if a.index == idx + 1:
                    row = a.name
                elif a.index == idx + 2:
                    col = a.name
            return row, col

        a_row_s, a_col_s = _strides_after(a_ptr)
        b_row_s, b_col_s = _strides_after(b_ptr)
        c_row_s, c_col_s = _strides_after(c_ptr)

        return {
            "M": M, "N": N, "K": K,
            "a_ptr": a_ptr.name, "b_ptr": b_ptr.name, "c_ptr": c_ptr.name,
            "a_elem": a_ptr.elem_type, "b_elem": b_ptr.elem_type,
            "c_elem": c_ptr.elem_type,
            "a_row_stride": a_row_s, "a_col_stride": a_col_s,
            "b_row_stride": b_row_s, "b_col_stride": b_col_s,
            "c_row_stride": c_row_s, "c_col_stride": c_col_s,
        }

    # Ops a fused matmul epilogue may apply to the dot result. Pointwise /
    # broadcast / cast / layout only — NO reduce/scan (softmax has its own
    # path; anything else falls through to the #157 refusal). #158.
    _EPILOGUE_ALLOWED = frozenset({
        "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf",
        "arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf",
        "math.exp", "math.exp2", "math.log", "math.log2", "math.sqrt",
        "math.rsqrt", "math.sin", "math.cos", "math.erf", "math.tanh",
        "math.floor", "math.ceil", "math.fma", "math.absf",
        "arith.truncf", "arith.extf", "arith.sitofp", "arith.fptosi",
        "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape",
        "ttg.convert_layout", "arith.constant", "tt.load", "tt.clampf",
    })

    def _detect_matmul_epilogue(self):
        """Detect matmul -> pointwise/broadcast epilogue -> store (#158).

        Same staged vehicle as _detect_matmul_softmax, but the epilogue is a
        GENERAL elementwise/broadcast op chain (scale, bias, activation, ...)
        rather than the hardcoded softmax. Returns the softmax-style info dict
        plus ``epilogue_ops`` (topologically ordered), ``bias_ptr`` (or None),
        and ``store_value_id``; or None if not matched, or if any op on the
        dot->store path is outside _EPILOGUE_ALLOWED (those keep falling
        through to the #157 refusal — never silently dropped).

        Checked AFTER _detect_matmul_softmax in lower(), so a reduce-bearing
        softmax kernel is already claimed; what reaches here has no reduce.
        """
        dot_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                if dot_ssa is not None:
                    return None
                dot_ssa = ssa
        if dot_ssa is None or len(dot_ssa.operand_ids) < 2:
            return None

        a_type = self._find_op_type_str(dot_ssa.operand_ids[0])
        b_type = self._find_op_type_str(dot_ssa.operand_ids[1])
        a_shape = _extract_shape(a_type) if a_type else None
        b_shape = _extract_shape(b_type) if b_type else None
        if (not a_shape or not b_shape
                or len(a_shape) != 2 or len(b_shape) != 2):
            return None
        M, K = a_shape
        K2, N = b_shape
        if K != K2:
            return None

        by_id0 = {ssa.id: ssa for ssa in self.graph.ops}

        # tt.dot's 3rd operand is the accumulator INIT. Triton fuses a trailing
        # `acc + bias` into it, so a non-zero init = a bias added to the matmul
        # result BEFORE the epilogue. Recognise a broadcast-of-load (a (N,) col
        # bias or (M,1) row bias); a zero constant is the plain init (ignore);
        # anything else is an unsupported accumulator -> bail (#157 refuses).
        acc_bias_ptr = None
        acc_bias_dim = None
        if len(dot_ssa.operand_ids) >= 3:
            acc = by_id0.get(dot_ssa.operand_ids[2])

            def _is_zero_const(op):
                if op is None or op.op != "arith.constant":
                    return False
                v = op.attrs.get("value")
                try:
                    return float(str(v).strip()) == 0.0
                except (TypeError, ValueError):
                    return "0.0" in str(v) or str(v) in ("0", "false")

            if acc is not None and not _is_zero_const(acc):
                cur = acc
                axis = None
                for _ in range(6):
                    if cur is None:
                        break
                    if cur.op in ("tt.broadcast", "ttg.convert_layout",
                                  "tt.reshape"):
                        cur = by_id0.get(cur.operand_ids[0]) if cur.operand_ids else None
                        continue
                    if cur.op == "tt.expand_dims":
                        axis = cur.attrs.get("axis")
                        cur = by_id0.get(cur.operand_ids[0]) if cur.operand_ids else None
                        continue
                    break
                if cur is not None and cur.op == "tt.load" and cur.operand_ids:
                    bptr = self._trace_ptr_source(cur.operand_ids[0], by_id0)
                    if bptr is None:
                        return None
                    acc_bias_ptr = bptr.name
                    acc_bias_dim = "col" if str(axis) in ("0",) else "row"
                    # A fused ROW bias (M-length, broadcast over N as the dot accumulator
                    # init) is mis-computed: re-audit #8 found M=64 row 40 grossly wrong
                    # (strip-local bias index), and a direct repro shows even M=32 wrong
                    # (the simdgroup matmul mis-handles a non-zero fused-bias accumulator).
                    # The mechanism isn't reliably fixable here, so REFUSE the row-bias
                    # case loudly for all M. COL bias (the common per-output-feature bias)
                    # is strip-independent and verified correct — unaffected.
                    if acc_bias_dim == "row":
                        from triton_msl.errors import MetalNonRecoverableError
                        raise MetalNonRecoverableError(
                            "fused ROW-bias matmul (per-row bias as the dot accumulator) "
                            "is mis-computed and not reliably lowerable. Refusing rather "
                            "than mis-compute. Use a column bias, or add the row bias in a "
                            "separate kernel after the matmul.", op_name="tt.dot")
                else:
                    return None   # non-zero, non-bias accumulator: unsupported

        # Walk BACKWARD from the single tt.store's value, collecting the
        # epilogue input cone and stopping at the dot (the matmul result is the
        # seed). Backward — not forward from the dot — because a bias enters via
        # an independent tt.load that is NOT a consumer of the dot. Every op in
        # the cone must be allow-listed (else bail -> #157 refuses loudly).
        by_id = {ssa.id: ssa for ssa in self.graph.ops}
        stores = [s for s in self.graph.ops if s.op == "tt.store"]
        if len(stores) != 1 or len(stores[0].operand_ids or []) < 2:
            return None
        store = stores[0]
        # tt.store operands are (ptr, value[, mask]) — the stored value is [1].
        store_value_id = store.operand_ids[1]
        epilogue_ids = set()
        reached_dot = False
        has_compute = False
        seen = set()
        frontier = [store_value_id]
        while frontier:
            vid = frontier.pop()
            if vid in seen:
                continue
            seen.add(vid)
            if vid == dot_ssa.id:
                reached_dot = True
                continue                 # leaf: the matmul result (seeded later)
            op = by_id.get(vid)
            if op is None:
                # A non-dot leaf in the VALUE cone is a kernel/block arg the
                # per-element emitter can't lower (e.g. a runtime scalar scale
                # entering via tt.splat). Resolving it to 0.0f would be silently
                # wrong, so refuse -> the #157 catch-all rejects loudly. (Bias
                # POINTERS never reach here: their address goes through tt.addptr,
                # which isn't allow-listed and bails above.)
                return None
            if op.op == "tt.dot":
                continue
            if op.op not in self._EPILOGUE_ALLOWED:
                return None              # unsupported epilogue op
            epilogue_ids.add(op.id)
            if op.op not in _EPI_PASSTHROUGH and op.op not in (
                    "arith.constant", "tt.load"):
                has_compute = True
            frontier.extend(op.operand_ids or [])
        if not reached_dot or not has_compute:
            return None                  # store doesn't derive from dot, or
                                         # pure matmul (no real epilogue)

        # A bias / extra input enters via a tt.load inside the epilogue. Collect
        # its pointer arg (we index it per element in the template).
        bias_ptr = None
        for eid in epilogue_ids:
            if by_id[eid].op == "tt.load":
                bptr = self._trace_ptr_source(
                    by_id[eid].operand_ids[0], by_id) if by_id[eid].operand_ids else None
                if bptr is not None:
                    if bias_ptr is not None and bias_ptr.name != bptr.name:
                        return None       # >1 extra input not supported yet
                    bias_ptr = bptr

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        if len(ptr_args) < 3:
            return None
        a_ptr = ptr_args[0]
        b_ptr = ptr_args[1]
        c_ptr = ptr_args[-1]

        def _strides_after(ptr_arg):
            row = col = None
            for a in scalar_args:
                if a.index == ptr_arg.index + 1:
                    row = a.name
                elif a.index == ptr_arg.index + 2:
                    col = a.name
            return row, col

        a_row_s, a_col_s = _strides_after(a_ptr)
        b_row_s, b_col_s = _strides_after(b_ptr)
        c_row_s, c_col_s = _strides_after(c_ptr)

        # Topologically-ordered epilogue ops (graph order is topo).
        epilogue_ops = [ssa for ssa in self.graph.ops if ssa.id in epilogue_ids]

        return {
            "M": M, "N": N, "K": K,
            "a_ptr": a_ptr.name, "b_ptr": b_ptr.name, "c_ptr": c_ptr.name,
            "a_elem": a_ptr.elem_type, "b_elem": b_ptr.elem_type,
            "c_elem": c_ptr.elem_type,
            "a_row_stride": a_row_s, "a_col_stride": a_col_s,
            "b_row_stride": b_row_s, "b_col_stride": b_col_s,
            "c_row_stride": c_row_s, "c_col_stride": c_col_s,
            # epilogue-specific:
            "epilogue_ops": epilogue_ops,
            "dot_id": dot_ssa.id,
            "bias_ptr": bias_ptr.name if bias_ptr else None,
            "acc_bias_ptr": acc_bias_ptr,      # bias fused into the dot's init
            "acc_bias_dim": acc_bias_dim,      # "col" (N,) or "row" (M,1)
            "store_value_id": store_value_id,
        }

    def _detect_permute_chained_reduce(self):
        """Detect ``load(In) -> trans(perm) -> sum-reduce* -> store(Out)``.

        This is ``test_chained_reductions``: a large N-D tensor is loaded
        contiguously, permuted, reduced over several axes (sum), and the
        small result is stored contiguously. Materializing the permute is
        infeasible (the tensor exceeds threadgroup memory), so the template
        fuses the permute into the reduction index math: each input element
        maps to exactly one output cell (determined statically from the
        permute + reduce axes), and the kernel cooperatively scatter-adds
        each input into a tiny threadgroup accumulator.

        Returns an info dict (in_shape, surviving original axes, strides,
        In/Out args, elem dtype, totals) or ``None`` if the kernel deviates
        from the canonical pattern. Conservative: integer sum reduce only
        (uses ``atomic_int``); anything else falls through.
        """
        op_by_id = {}
        for s in self.graph.ops:
            op_by_id[s.id] = s

        # 1) Find the single tt.store and the single tt.trans.
        stores = [s for s in self.graph.ops if s.op == "tt.store"]
        transes = [s for s in self.graph.ops if s.op == "tt.trans"]
        loads = [s for s in self.graph.ops if s.op == "tt.load"]
        if len(stores) != 1 or len(transes) != 1 or len(loads) != 1:
            return None
        # No control flow / programs — single-threadgroup cooperative kernel.
        if any(s.op in ("scf.for", "scf.while", "scf.if",
                        "tt.get_num_programs", "tt.get_program_id")
               for s in self.graph.ops):
            return None
        store = stores[0]
        trans = transes[0]
        load = loads[0]
        if len(store.operand_ids) < 2:
            return None

        # 2) The stored value must be a chain of sum-reduces ending at trans.
        def _is_sum_reduce(r):
            if r.op != "tt.reduce" or not r.operand_ids:
                return False
            if r.result_ids and len(r.result_ids) >= 2:
                return False  # argmin/argmax
            body_ops = r.region_ops or []
            adds = [b for b in body_ops
                    if b.op in ("arith.addi", "arith.addf")]
            return len(adds) >= 1 and all(
                b.op in ("arith.addi", "arith.addf", "tt.reduce.return")
                for b in body_ops)

        # Skip dtype/layout passthroughs (e.g. the i32->i64 arith.extsi that
        # an int sum promotes to, or a ttg.convert_layout / reshape) between
        # the store value and the reduce chain.
        _PASS = ("arith.extsi", "arith.extui", "arith.trunci",
                 "ttg.convert_layout", "tt.reshape", "arith.bitcast")

        def _skip_pass(op):
            seen = 0
            while (op is not None and op.op in _PASS and op.operand_ids
                   and seen < 8):
                op = op_by_id.get(op.operand_ids[0])
                seen += 1
            return op

        red_axes = []  # sequential reduce axes (in application order)
        cur = _skip_pass(op_by_id.get(store.operand_ids[1]))
        while cur is not None and cur.op == "tt.reduce":
            if not _is_sum_reduce(cur):
                return None
            red_axes.append(cur.attrs.get("axis", 0))
            cur = _skip_pass(op_by_id.get(cur.operand_ids[0])
                             if cur.operand_ids else None)
        # ``cur`` should now be the trans; reduces were collected innermost
        # last, so reverse to application (outermost-first) order.
        red_axes = red_axes[::-1]
        if cur is None or cur.id != trans.id or not red_axes:
            return None

        # 3) The trans operand must be the load; the load must be a
        #    contiguous identity gather In[i] (addptr(splat(In), reshape(
        #    make_range(0..TOTAL)))).
        if not trans.operand_ids or trans.operand_ids[0] != load.id:
            return None
        in_arg = self._trace_ptr_source(load.operand_ids[0], op_by_id)
        out_arg = self._trace_ptr_source(store.operand_ids[0], op_by_id)
        if in_arg is None or out_arg is None or in_arg.name == out_arg.name:
            return None
        if not self._is_contiguous_range_gather(load.operand_ids[0], op_by_id):
            return None
        if not self._is_contiguous_range_gather(store.operand_ids[0],
                                                op_by_id):
            return None

        # 4) Shapes + permutation.
        in_shape = _extract_shape(self._find_op_type_str(trans.operand_ids[0])
                                  or "")
        if not in_shape or len(in_shape) < 2:
            return None
        rank = len(in_shape)
        order = self._parse_trans_order(trans, rank)
        if order is None or sorted(order) != list(range(rank)):
            return None
        for ax in red_axes:
            if not isinstance(ax, int):
                return None

        # 5) Fuse: track which ORIGINAL axis each current-tensor axis maps to
        #    as the sequential reduces remove axes. permuted axis k -> orig
        #    order[k]; reduces then drop entries.
        cur_axes = list(order)  # current-tensor axis -> original axis id
        for ax in red_axes:
            if ax < 0 or ax >= len(cur_axes):
                return None
            del cur_axes[ax]
        surviving = cur_axes  # original axes, in output order
        if not surviving:
            return None

        # 6) dtype: integer sum only (atomic_int).
        elem = load.elem_type or "i32"
        dtype = _mlir_to_triton_dtype(elem)
        if not (dtype.startswith("i") or dtype.startswith("u")):
            return None

        # Row-major strides over the original shape and the output shape.
        in_strides = [1] * rank
        for i in range(rank - 2, -1, -1):
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1]
        out_shape = [in_shape[a] for a in surviving]
        out_strides = [1] * len(out_shape)
        for i in range(len(out_shape) - 2, -1, -1):
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1]

        total = 1
        for d in in_shape:
            total *= d
        out_total = 1
        for d in out_shape:
            out_total *= d

        return {
            "in_arg": in_arg.name,
            "out_arg": out_arg.name,
            "elem": elem,
            "out_elem": out_arg.elem_type or elem,
            "total": total,
            "out_total": out_total,
            # per surviving output axis k: (input row-major stride of the
            # original axis, that axis's size, output row-major stride)
            "surviving": [
                (in_strides[surviving[k]], in_shape[surviving[k]],
                 out_strides[k])
                for k in range(len(surviving))
            ],
        }

    def _parse_trans_order(self, trans, rank):
        """Parse a ``tt.trans`` permutation order from the module text.

        The walker leaves ``attrs['order'] = None`` (array attrs aren't
        exposed via bindings), so recover it from ``order = array<i32: ...>``
        in ``mod_text``. Returns a list of ``rank`` ints or ``None``.
        """
        o = trans.attrs.get("order")
        if isinstance(o, (list, tuple)) and len(o) == rank:
            return list(o)
        mod_text = getattr(self.graph, "mod_text", "") or ""
        matches = re.findall(r"tt\.trans[^\n]*?order\s*=\s*array<i32:\s*"
                             r"([0-9,\s]+)>", mod_text)
        for m in matches:
            vals = [int(x) for x in m.split(",") if x.strip()]
            if len(vals) == rank:
                return vals
        return None

    def _is_contiguous_range_gather(self, ptr_id, op_by_id):
        """True if ``ptr_id`` is ``addptr(splat(P), reshape?(make_range(0..N)))``
        — i.e. the i-th lane addresses ``P[i]`` (an identity/contiguous gather
        or scatter). Walks the offset operand back to a 0-based tt.make_range.
        """
        op = op_by_id.get(ptr_id)
        if op is None or op.op != "tt.addptr" or len(op.operand_ids) < 2:
            return False
        off = op_by_id.get(op.operand_ids[1])
        seen = 0
        while off is not None and seen < 8:
            seen += 1
            if off.op == "tt.make_range":
                return int(off.attrs.get("start", 0)) == 0
            if off.op in ("tt.reshape", "ttg.convert_layout", "arith.extsi"):
                off = (op_by_id.get(off.operand_ids[0])
                       if off.operand_ids else None)
                continue
            return False
        return False

    def _detect_3d_reduce(self):
        """Detect if this kernel is a simple 3D reduce that needs a template.

        Returns dict with shape/axis info if detected, None otherwise.
        Detects both regular reduce (sum/max/min) and argmin/argmax (2 operands).

        Only triggers for simple kernels (load→reduce→store). Complex kernels
        with scf.for loops, multiple reduces, or multi-axis grids must go
        through the generic op-by-op lowerer instead.
        """
        # Reject complex kernels that need op-by-op lowering
        has_scf_for = False
        has_num_programs = False
        reduce_count = 0
        for ssa in self.graph.ops:
            if ssa.op == "scf.for":
                has_scf_for = True
            elif ssa.op == "tt.get_num_programs":
                has_num_programs = True
            elif ssa.op == "tt.reduce":
                reduce_count += 1
        if has_scf_for or has_num_programs or reduce_count > 1:
            return None

        # Look for tt.reduce with a 3D input
        for ssa in self.graph.ops:
            if ssa.op == "tt.reduce" and ssa.operand_ids:
                # Check input shape
                input_type = self._find_op_type_str(ssa.operand_ids[0])
                if input_type:
                    input_shape = _extract_shape(input_type)
                    if input_shape and len(input_shape) == 3:
                        # The 3D-reduce template reads the RAW load pointer, so any
                        # value-changing op between the load and the reduce (e.g.
                        # tl.sum(a * s) / a.to(f32) / a + b) is SILENTLY DROPPED
                        # (2026-06-21 audit silent-wrong: tl.sum(a*s) returned the
                        # unscaled sum). The op-by-op generic 3D-reduce path
                        # (_lower_reduce_3d) ALSO mis-computes this shape (only the
                        # first row), so there is no correct fall-through — REFUSE
                        # LOUDLY rather than emit silently-wrong output. Only the
                        # DIRECT-load case (reduce input traces to a tt.load through
                        # layout-only ops) is validated and kept.
                        _obid = {s.id: s for s in self.graph.ops}
                        def _is_direct_load(_sid, _depth=0):
                            _o = _obid.get(_sid)
                            if _o is None or _depth > 16:
                                return False
                            if _o.op == "tt.load":
                                return True
                            if (_o.op in ("tt.reshape", "ttg.convert_layout")
                                    and _o.operand_ids):
                                return _is_direct_load(_o.operand_ids[0], _depth + 1)
                            return False
                        if not _is_direct_load(ssa.operand_ids[0]):
                            from triton_msl.errors import MetalNonRecoverableError
                            raise MetalNonRecoverableError(
                                "3D reduce with a pre-reduce elementwise op (e.g. "
                                "tl.sum(a * s, axis=2), a.to(f32), a + b before the "
                                "reduce) is not supported: the 3D-reduce template "
                                "reduces the RAW loaded values and would silently drop "
                                "the op, and the generic 3D-reduce path mis-computes "
                                "this shape. Refusing to emit silently-wrong output. "
                                "Apply the op after the reduce where valid (e.g. "
                                "s * tl.sum(a, axis=2)) or in a separate kernel.")
                        axis = ssa.attrs.get("axis", 0)
                        # Detect argmin/argmax: 2 operands (values, indices)
                        is_argminmax = len(ssa.operand_ids) >= 2
                        # Determine combine op
                        combine_op = "sum"
                        if ssa.region_ops:
                            for body_op in ssa.region_ops:
                                if "max" in body_op.op:
                                    combine_op = "max"
                                elif "min" in body_op.op:
                                    combine_op = "min"
                                elif "addf" in body_op.op or "addi" in body_op.op:
                                    combine_op = "sum"
                                elif body_op.op == "arith.cmpf":
                                    pred = body_op.attrs.get("predicate_name", "")
                                    if "gt" in pred:
                                        combine_op = "max"
                                    elif "lt" in pred:
                                        combine_op = "min"
                        if is_argminmax:
                            # argmin uses cmpf(olt) → detected as "min"
                            # argmax uses cmpf(ogt) → detected as "max"
                            combine_op = "argmin" if combine_op == "min" else "argmax"
                        M, N, K = input_shape
                        # 64-bit integer 3-D reduce / argminmax is not correctly lowered
                        # by ANY path: the TEMPLATE computes in float32 (its dtype switch
                        # is i32->int, ELSE->float), and the generic fallback truncates to
                        # 32 bits (verified: a non-power-of-2 i64 sum = the low-32 sum;
                        # i64 argmax can't distinguish 2^25 from 2^25+1). Refuse loudly
                        # rather than mis-compute (re-audit #12).
                        if input_type and ("i64" in input_type or "u64" in input_type):
                            from triton_msl.errors import MetalNonRecoverableError
                            raise MetalNonRecoverableError(
                                "3-D reduce / argminmax over a 64-bit integer is not "
                                "correctly lowered (the template rounds in float32 and the "
                                "generic path truncates to 32 bits). Refusing. Use a "
                                "32-bit integer, or reduce in 2-D/1-D.", op_name="tt.reduce")
                        total = M * N * K
                        # Use block_size that covers all elements
                        block_size = max(total, self.graph.num_warps * 32)
                        # Cap at 1024 (Metal max threads per threadgroup)
                        block_size = min(block_size, 1024)
                        return {
                            "shape": (M, N, K),
                            "axis": axis,
                            "combine_op": combine_op,
                            "block_size": block_size,
                        }
        return None


    def _detect_flip(self):
        """Detect tl.flip's reshape+xor-reduce+broadcast pattern.

        tl.flip(x, dim) on a 3D tensor (M, N, K) lowers to:
            reshape to higher-dim tensor (flip dim split into 2x2x...x2)
            for each of log2(flip_size) iterations:
                reduce(xor, axis=i, keepdim=True)
                xor with broadcast
            reshape back to (M, N, K)

        Returns dict with {M, N, K, flip_dim, elem_type, x_ptr, z_ptr, off_id,
        block_size} if detected, None otherwise.

        Only matches the exact tl.flip pattern: load → reshape → N reduces →
        reshape → store with the same 3D offset. Other patterns fall through
        to the generic lowerer.
        """
        # Reject complex kernels
        has_scf_for = False
        has_num_programs = False
        for ssa in self.graph.ops:
            if ssa.op in ("scf.for", "scf.while", "scf.if"):
                has_scf_for = True
            elif ssa.op == "tt.get_num_programs":
                has_num_programs = True
        if has_scf_for or has_num_programs:
            return None

        # Find the single tt.load and tt.store
        load_ssa = None
        store_ssa = None
        reshape_ops = []
        reduce_ops = []
        xori_ops = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif ssa.op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif ssa.op == "tt.reshape":
                reshape_ops.append(ssa)
            elif ssa.op == "tt.reduce":
                reduce_ops.append(ssa)
            elif ssa.op == "arith.xori":
                xori_ops.append(ssa)
        if load_ssa is None or store_ssa is None:
            return None
        # Must have at least one xori reduce. Reshapes appear in pairs when the
        # flip dim has size > 2; when size == 2, Triton skips them.
        if len(reduce_ops) < 1:
            return None
        if len(reshape_ops) not in (0, 2):
            return None

        # All reduces must use xori
        for red in reduce_ops:
            if not red.region_ops:
                return None
            has_xori = any("xori" in bop.op for bop in red.region_ops)
            if not has_xori:
                return None

        # Input shape from load: tensor<MxNxK>
        load_shape = _extract_shape(load_ssa.type_str)
        if not load_shape or len(load_shape) != 3:
            return None
        M, N, K = load_shape
        in_shape = (M, N, K)

        if len(reshape_ops) == 2:
            # Two reshapes: (M,N,K) -> higher-dim -> (M,N,K)
            rs1, rs2 = reshape_ops
            rs1_in_shape = _extract_shape(self._find_op_type_str(rs1.operand_ids[0])) \
                if rs1.operand_ids else None
            rs1_out_shape = _extract_shape(rs1.type_str)
            rs2_in_shape = _extract_shape(self._find_op_type_str(rs2.operand_ids[0])) \
                if rs2.operand_ids else None
            rs2_out_shape = _extract_shape(rs2.type_str)
            if tuple(rs1_in_shape or ()) != in_shape:
                return None
            if tuple(rs2_out_shape or ()) != in_shape:
                return None
            if tuple(rs1_out_shape or ()) != tuple(rs2_in_shape or ()):
                return None
            out_shape = tuple(rs1_out_shape)
            # Find flip dim: in_shape[d] = 2^k, replaced with k 2s in out_shape.
            flip_dim = None
            num_steps = None
            for d in range(3):
                dim_size = in_shape[d]
                if dim_size < 2 or (dim_size & (dim_size - 1)) != 0:
                    continue
                steps = dim_size.bit_length() - 1
                expected = in_shape[:d] + (2,) * steps + in_shape[d + 1:]
                if out_shape == expected:
                    flip_dim = d
                    num_steps = steps
                    break
            if flip_dim is None:
                return None
            if len(reduce_ops) != num_steps:
                return None
        else:
            # No reshape: flip dim has size 2 (single xor-reduce step).
            # The single reduce is on the flip dim directly, over 3D input.
            if len(reduce_ops) != 1:
                return None
            red = reduce_ops[0]
            red_axis = red.attrs.get("axis", 0)
            if red_axis not in (0, 1, 2):
                return None
            # Verify the reduce input shape equals in_shape
            red_in_shape = _extract_shape(self._find_op_type_str(red.operand_ids[0])) \
                if red.operand_ids else None
            if tuple(red_in_shape or ()) != in_shape:
                return None
            if in_shape[red_axis] != 2:
                return None
            flip_dim = red_axis
            num_steps = 1

        # Identify pointer args
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 2:
            return None
        x_ptr = ptr_args[0].name
        z_ptr = ptr_args[1].name
        elem_type = ptr_args[0].elem_type

        # Sanity: total elements
        total = M * N * K

        block_size = max(total, self.graph.num_warps * 32)
        block_size = min(block_size, 1024)

        return {
            "M": M,
            "N": N,
            "K": K,
            "flip_dim": flip_dim,
            "elem_type": elem_type,
            "x_ptr": x_ptr,
            "z_ptr": z_ptr,
            "total": total,
            "block_size": block_size,
        }


    def _detect_softmax(self):
        """Detect a row-wise softmax kernel:
            x = tl.load(x_ptr + row * n + offsets, mask, other=-inf)
            x_max = tl.max(x, axis=0)
            x = x - x_max
            x_exp = tl.exp(x)
            x_sum = tl.sum(x_exp, axis=0)
            tl.store(out_ptr + row * n + offsets, x_exp / x_sum, mask)

        The generic phase lowerer would produce 3 wrap-loops over x_ptr (one
        per phase), reading global memory 3x per row and recomputing exp()
        twice. We can do it with a single TG cache and one read.

        Returns a dict if matched, None otherwise. Conservative: any deviation
        falls through to the generic path.
        """
        # No control flow allowed (single-row template only)
        for ssa in self.graph.ops:
            if ssa.op in ("scf.for", "scf.while", "scf.if",
                          "tt.get_num_programs"):
                return None

        load_ssa = None
        store_ssa = None
        reduce_ops = []
        has_exp = False
        has_subf = False
        has_divf = False
        for ssa in self.graph.ops:
            op = ssa.op
            if op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif op == "tt.reduce":
                reduce_ops.append(ssa)
            elif op in ("math.exp", "math.exp2"):
                has_exp = True
            elif op == "arith.subf":
                has_subf = True
            elif op == "arith.divf":
                has_divf = True

        if (load_ssa is None or store_ssa is None
                or len(reduce_ops) != 2
                or not (has_exp and has_subf and has_divf)):
            return None

        # Reduces must be max then sum (in IR order). The combine op lives in
        # the reduce body region; _get_reduce_combine_info inspects that.
        red_ops = [self._get_reduce_combine_info(r)[0] for r in reduce_ops]
        if red_ops != ["max", "sum"]:
            return None

        # Identify ptr args (input vs output) and the n scalar arg.
        input_arg = None
        output_arg = None
        for arg in self.graph.args:
            if not arg.is_ptr:
                continue
            # Output arg is whichever ptr the tt.store writes through. Look
            # for the arg whose name appears in the store op's chain.
            # Heuristic: input is the first ptr arg, output is the second.
            if input_arg is None:
                input_arg = arg.name
            elif output_arg is None:
                output_arg = arg.name
        if input_arg is None or output_arg is None:
            return None

        # Row length / stride: the kernel's single scalar arg. The template
        # uses ONE scalar for BOTH the row stride (row_start = pid * n) and the
        # per-row element count (the stride-loop bound), so it is correct only
        # when those coincide in a single arg. If the kernel has more than one
        # scalar arg we cannot tell which is the reduction-dimension length:
        # an inductor persistent-softmax is (in_ptr, out_ptr, xnumel=row COUNT,
        # r0_numel=row LENGTH), and blindly taking the FIRST scalar (xnumel)
        # makes the loop cover xnumel elements instead of r0_numel -> a silently
        # wrong 1/N-coverage reduction (observed: softmax rows summing to 4
        # instead of 1 through torch.compile). Refuse to the generic (correct)
        # lowering rather than guess. Hand-written softmax kernels carry exactly
        # one scalar arg (n_cols; BLOCK_SIZE is constexpr and not in args).
        scalar_args = [arg.name for arg in self.graph.args if not arg.is_ptr]
        if len(scalar_args) != 1:
            return None
        n_arg = scalar_args[0]

        # Block size = the tensor's dim (we look at make_range end values).
        block_size = self.graph.block_size
        if block_size > 1024:
            # 1024 cap on threadgroup; row larger than 1024 needs a larger
            # TG buffer + more iterations. Skip for safety.
            return None

        return {
            "input_arg": input_arg,
            "output_arg": output_arg,
            "n_arg": n_arg,
            "block_size": block_size,
        }

    def _detect_layer_norm(self):
        """Detect a row-wise layer-norm kernel:
            x = tl.load(x_ptr + row * n + offsets, mask, other=0.0)
            mean = tl.sum(x, axis=0) / n
            diff = x - mean
            var = tl.sum(diff * diff, axis=0) / n
            inv_std = tl.math.rsqrt(var + eps)
            tl.store(out_ptr + ..., (x - mean) * inv_std, mask)

        Like softmax: 3 generic wrap-loops over x_ptr, one read per pass.
        Template caches the row in TG memory, reads once, and uses a
        Welford-style single-pass mean+M2 to fold both reductions into one
        read of the cache.

        Returns a dict if matched, None otherwise.
        """
        # No control flow allowed (single-row template only)
        for ssa in self.graph.ops:
            if ssa.op in ("scf.for", "scf.while", "scf.if",
                          "tt.get_num_programs"):
                return None

        load_ssa = None
        store_ssa = None
        reduce_ops = []
        has_rsqrt = False
        has_subf = False
        has_mulf = False
        has_addf = False
        for ssa in self.graph.ops:
            op = ssa.op
            if op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif op == "tt.reduce":
                reduce_ops.append(ssa)
            elif op == "math.rsqrt":
                has_rsqrt = True
            elif op == "math.sqrt":
                # sqrt followed by 1/sqrt is also a valid normalization shape
                has_rsqrt = True
            elif op == "arith.subf":
                has_subf = True
            elif op == "arith.mulf":
                has_mulf = True
            elif op == "arith.addf":
                has_addf = True

        # Layer norm: 2 sum reduces, normalization math (rsqrt, sub, mul,
        # add for variance + epsilon). Differs from softmax by lacking exp
        # and having two sum reduces (vs max + sum).
        if (load_ssa is None or store_ssa is None
                or len(reduce_ops) != 2
                or not (has_rsqrt and has_subf and has_mulf and has_addf)):
            return None
        red_ops = [self._get_reduce_combine_info(r)[0] for r in reduce_ops]
        if red_ops != ["sum", "sum"]:
            return None

        # Don't fire if softmax pattern matches (max/exp/divf disqualifies
        # layer norm because softmax's _detect_* would fire instead).
        for ssa in self.graph.ops:
            if ssa.op in ("math.exp", "math.exp2"):
                return None

        input_arg = None
        output_arg = None
        for arg in self.graph.args:
            if not arg.is_ptr:
                continue
            if input_arg is None:
                input_arg = arg.name
            elif output_arg is None:
                output_arg = arg.name
        if input_arg is None or output_arg is None:
            return None

        # The row length is the SOLE scalar arg (BLOCK_SIZE is constexpr, not in args).
        # Without this guard the first non-ptr arg was grabbed as the row length, so a
        # layernorm kernel passing BOTH M and N (or eps) as runtime args used the wrong
        # one — normalizing only the first M of N elements (re-audit #11: zeros 448/512).
        # Mirror the sibling _detect_softmax guard; multi-scalar kernels fall through to
        # the generic reduction lowerer, which handles the 2D keep_dims pattern.
        scalar_args = [arg.name for arg in self.graph.args if not arg.is_ptr]
        if len(scalar_args) != 1:
            return None
        n_arg = scalar_args[0]

        block_size = self.graph.block_size
        if block_size > 1024:
            return None

        return {
            "input_arg": input_arg,
            "output_arg": output_arg,
            "n_arg": n_arg,
            "block_size": block_size,
        }


    def _detect_transpose_via_reshape(self):
        """Detect the ``test_trans_reshape``-style transpose kernel:

            x = tl.load(make_block_ptr((M, N), strides=(N, 1), ...))
            x = tl.reshape(x, (M, m, n, 2))   # any 4-D split where m*n*2 == N
            x = tl.permute(x, (1, 2, 3, 0))   # canonical "move row to fastest"
            x = tl.reshape(x, (M*N,))
            tl.store(out + tl.arange(0, M*N), x)

        This is a layout-only transpose: the value at logical 1-D position k
        equals input[k % M, k / M], i.e. ``transpose(input).flat[k]``.

        Without this detector, the kernel falls through to the generic phase
        lowerer + ttg.convert_layout, which doesn\\'t honor the multi-element
        per-thread ``#linear`` source layout and produces wrong values. The
        template below sidesteps the layout shuffle entirely by emitting the
        transpose lookup directly: each output position k reads
        ``input[(k % M) * N + k / M]``.

        Returns dict if matched, None otherwise.
        """
        # Collect the relevant ops in order.
        load_ssa = None
        store_ssa = None
        reshapes = []
        trans_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif ssa.op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif ssa.op == "tt.reshape":
                reshapes.append(ssa)
            elif ssa.op == "tt.trans":
                if trans_ssa is not None:
                    return None
                trans_ssa = ssa
            elif ssa.op in ("scf.for", "scf.while", "scf.if",
                            "tt.reduce", "tt.scan", "tt.dot"):
                return None  # too complex for this template

        if (load_ssa is None or store_ssa is None or trans_ssa is None
                or len(reshapes) != 2):
            return None

        # Extract shapes. Load is 2-D, first reshape goes to 4-D, trans
        # produces a 4-D permuted view, second reshape flattens to 1-D.
        load_shape = _extract_shape(load_ssa.type_str)
        if not load_shape or len(load_shape) != 2:
            return None
        M, N = load_shape

        # First reshape: (M, N) → (M, m, n, 2) where m*n*2 == N
        first_reshape_shape = _extract_shape(reshapes[0].type_str)
        if (not first_reshape_shape or len(first_reshape_shape) != 4
                or first_reshape_shape[0] != M
                or first_reshape_shape[3] != 2):
            return None
        m, n = first_reshape_shape[1], first_reshape_shape[2]
        if m * n * 2 != N:
            return None

        # The trans must apply the (1, 2, 3, 0) permutation: input shape
        # (M, m, n, 2) → output shape (m, n, 2, M). This is the canonical
        # "move axis 0 (size M) to the end" permutation that, combined with
        # the surrounding reshapes, computes a 2-D transpose. Other 4-D
        # permutations don\\'t collapse to a transpose. The walker doesn\\'t
        # populate ``trans_ssa.attrs["order"]`` reliably, so we check the
        # shape transformation instead.
        trans_shape = _extract_shape(trans_ssa.type_str)
        if (not trans_shape or len(trans_shape) != 4
                or trans_shape != (m, n, 2, M)):
            return None

        # Second reshape: must flatten to (M*N,)
        second_reshape_shape = _extract_shape(reshapes[1].type_str)
        if (not second_reshape_shape or len(second_reshape_shape) != 1
                or second_reshape_shape[0] != M * N):
            return None

        # Identify ptr args (input and output).
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 2:
            return None
        input_arg = ptr_args[0].name
        output_arg = ptr_args[1].name
        elem_type = ptr_args[0].elem_type

        return {
            "input_arg": input_arg,
            "output_arg": output_arg,
            "elem_type": elem_type,
            "M": M,
            "N": N,
            "block_size": M * N,
        }


    def _detect_nd_trans(self):
        """Detect a generic N-D transpose: one tt.load of a rank>=3 tensor, one
        tt.trans (any permutation), optional tt.reshape(s), one tt.store to a
        flat pointer, with NO reduce/scan/dot/control-flow. Emits a closed-form
        direct copy (out[k] = in[src_flat(k)]). Returns dict or None.

        More specific transpose templates (_detect_transpose_via_reshape,
        _detect_permute_chained_reduce) run first and return None for anything
        they don't own, so this is the general fallback for test_trans_4d."""
        load_ssa = store_ssa = trans_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif ssa.op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif ssa.op == "tt.trans":
                if trans_ssa is not None:
                    return None
                trans_ssa = ssa
            elif ssa.op == "tt.reshape":
                pass  # allowed (descriptor lowering inserts these)
            elif ssa.op in ("scf.for", "scf.while", "scf.if",
                            "tt.reduce", "tt.scan", "tt.dot"):
                return None
        if load_ssa is None or store_ssa is None or trans_ssa is None:
            return None
        # Data-flow validation (CRITICAL): the template reads straight from the
        # input pointer and writes the permuted index, so it is correct ONLY if
        # the value path is purely load -> [reshape]* -> trans -> [reshape]* ->
        # store. Any intervening compute op (e.g. load -> arith.mulf -> trans)
        # is NOT a tt.reshape, so the chain breaks and we bail — otherwise that
        # op would be silently dropped -> wrong output. (The descriptor lowering
        # inserts layout-only reshapes between load/trans and trans/store, which
        # are fine.) Refuse (return None -> generic path + rank>=3 backstop)
        # unless the clean chain is proven.
        # Value-preserving layout-only ops that may appear in the chain (the
        # descriptor lowering inserts these). NOT arith/math/etc — those change
        # the values and must break the chain (-> bail -> backstop).
        _VALUE_PRESERVING = ("tt.reshape", "ttg.convert_layout")

        def _traces_to(ssa_id, target_id):
            """True if ssa_id is target_id or a value-preserving (reshape/
            convert_layout) chain back to it."""
            cur = ssa_id
            seen = set()
            for _ in range(8):
                if cur == target_id:
                    return True
                if cur in seen:
                    break
                seen.add(cur)
                op = next((s for s in self.graph.ops if s.id == cur), None)
                if (op is None or op.op not in _VALUE_PRESERVING
                        or not op.operand_ids):
                    break
                cur = op.operand_ids[0]
            return False

        if (not trans_ssa.operand_ids
                or not _traces_to(trans_ssa.operand_ids[0], load_ssa.id)):
            return None
        if (len(store_ssa.operand_ids) < 2
                or not _traces_to(store_ssa.operand_ids[1], trans_ssa.id)):
            return None
        # The transpose operates on its INPUT's shape (the N-D tensor).
        src_shape = _extract_shape(self._find_op_type_str(trans_ssa.operand_ids[0]))
        if not src_shape or len(src_shape) < 3:
            return None
        order = self._parse_trans_order(trans_ssa, len(src_shape))
        if order is None or sorted(order) != list(range(len(src_shape))):
            return None
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 2:
            return None
        total = 1
        for s in src_shape:
            total *= s
        return {
            "input_arg": ptr_args[0].name,
            "output_arg": ptr_args[1].name,
            "elem_type": ptr_args[0].elem_type,
            "src_shape": list(src_shape),
            "order": list(order),
            "total": total,
        }


    def _detect_row_wise_sort(self):
        """Detect tl.sort / tl.topk applied to each row of a 2D tensor.

        Pattern (emitted by triton.language.standard.sort_impl):
          - Single tt.load of a 2D tensor shape (M, N) where N is a power of 2.
          - tt.reshape from (M, N) to (2,)*log2(M*N) hypercube.
          - A series of tt.reduce ops with xori combine, where every reduce
            axis corresponds to a bit within the *within-row* range, i.e.,
            axis >= log2(M*N) - log2(N). Additionally, topk has a final
            axis-reduce with a float max/min combine (trimming the extra dims).
          - A final tt.reshape to (M, N) or (M, k) and a tt.store.

        For this pattern, each row is sorted independently, so we can emit
        a kernel where thread `lid` handles row `lid` with a local register
        array. That avoids needing > 1024 threads in a single threadgroup.

        Returns dict with {M, N, k, descending, elem_type, x_ptr, z_ptr,
        stride_xm, stride_zm, block_size} if detected, None otherwise.
        """
        # Only consider kernels with tt.load + tt.store + multiple tt.reduce
        if any(op in {"scf.for", "scf.while", "scf.if", "tt.get_num_programs"}
               for op in (s.op for s in self.graph.ops)):
            return None

        load_ssa = None
        store_ssa = None
        reshape_ops = []
        reduce_ops = []
        const_ops = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif ssa.op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif ssa.op == "tt.reshape":
                reshape_ops.append(ssa)
            elif ssa.op == "tt.reduce":
                reduce_ops.append(ssa)
            elif ssa.op == "arith.constant":
                const_ops.append(ssa)
        if load_ssa is None or store_ssa is None:
            return None
        # Bitonic sort has at least log2(N) xori reduces. Require at least
        # one xori reduce to distinguish from softmax/max-reduce patterns.
        xor_reduce_count = 0
        for red in reduce_ops:
            if red.region_ops and any("xori" in bop.op for bop in red.region_ops):
                xor_reduce_count += 1
        if xor_reduce_count < 1:
            return None

        # Require a 2D -> hypercube reshape (distinctive to tl.sort).
        # Input (M, N) reshapes to (2,)*log2(M*N) with ALL dims size 2.
        has_hypercube_reshape = False
        for rs in reshape_ops:
            out_shape = _extract_shape(rs.type_str)
            if out_shape and len(out_shape) >= 4 and all(d == 2 for d in out_shape):
                has_hypercube_reshape = True
                break
        if not has_hypercube_reshape:
            return None

        # Load shape: 2D tensor<MxNx...>, or 1D tensor<N> (treated as M=1). The 1D
        # case is admitted ONLY to route a 1D tl.topk (K<N) to the template, which
        # REFUSES it (the K<N trim is broken — re-audit #10); a 1D FULL sort (K==N)
        # is correctly handled by the generic path, so it is left unclaimed below.
        load_shape = _extract_shape(load_ssa.type_str)
        if not load_shape or len(load_shape) not in (1, 2):
            return None
        if len(load_shape) == 2:
            M, N = load_shape
        else:
            M, N = 1, load_shape[0]
        # N must be a power of 2
        if N < 1 or (N & (N - 1)) != 0:
            return None

        # Identify the final store shape: (M, K) [2D] or (K,) [1D, M==1].
        store_shape = None
        if store_ssa.operand_ids and len(store_ssa.operand_ids) >= 2:
            val_id = store_ssa.operand_ids[1]
            val_type = self._find_op_type_str(val_id)
            store_shape = _extract_shape(val_type) if val_type else None
        if not store_shape or len(store_shape) not in (1, 2):
            return None
        if len(store_shape) == 2:
            if store_shape[0] != M:
                return None
            K_out = store_shape[1]
        else:
            if M != 1:
                return None
            K_out = store_shape[0]
        # K_out must be a power of 2 and <= N
        if K_out < 1 or (K_out & (K_out - 1)) != 0 or K_out > N:
            return None
        # 1D FULL sort (K==N) is correctly lowered by the generic path — don't claim
        # it (the template's M-row layout is for the 2D case).
        if M == 1 and K_out == N:
            return None

        # topk (K < N): the sort signature is confirmed (>=1 xori reduce + hypercube
        # reshape) but the output is trimmed to K < N. That K<N trim mis-computes in
        # BOTH the template and the generic path (re-audit #10: duplicated values).
        # REFUSE here — before the reduce-axis gate below would otherwise drop a topk
        # to the broken generic path. The full sort (K == N) continues normally.
        if K_out < N:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"tl.topk (k={K_out} < N={N}) is not correctly lowered — the K<N trim "
                f"mis-computes (duplicated values, not the K distinct top elements). "
                f"Refusing rather than return wrong results. Use a full tl.sort and "
                f"slice the top k, or k == N.", op_name="tt.reduce")

        total = M * N
        n_dims = total.bit_length() - 1  # log2(total)
        if (1 << n_dims) != total:
            return None
        log_n = N.bit_length() - 1  # log2(N)

        # Every reduce must have an axis in the within-row range
        # (axes n_dims - log_n .. n_dims - 1 — the last log_n axes)
        min_axis = n_dims - log_n
        for red in reduce_ops:
            axis = red.attrs.get("axis", -1)
            if axis < min_axis or axis >= n_dims:
                return None
            # Must have a reduce body — xori for bitonic compare-swap,
            # or arith.maxf/maximumf/minf/minimumf/cmpf for topk trim.
            if not red.region_ops:
                return None
            body_ops = {bop.op for bop in red.region_ops}
            is_xor = any("xori" in op for op in body_ops)
            is_minmax = any(("max" in op or "min" in op or op == "arith.cmpf")
                            for op in body_ops)
            if not (is_xor or is_minmax):
                return None

        # Identify pointer args (X=input, Z=output) and stride scalars
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        if len(ptr_args) < 2:
            return None
        # Distinguish input vs output via prescan store chain
        self._store_ptr_ids = set()
        self._prescan_stores()
        x_ptr_name = None
        z_ptr_name = None
        for a in ptr_args:
            if a.id in getattr(self, "_output_arg_ids", set()):
                if z_ptr_name is None:
                    z_ptr_name = a.name
            else:
                if x_ptr_name is None:
                    x_ptr_name = a.name
        if x_ptr_name is None or z_ptr_name is None:
            return None
        elem_type = ptr_args[0].elem_type

        # Detect descending by the presence of hypercube-sized `arith.constant
        # dense<1>` tensors at the start of the kernel. triton.language.sort
        # emits them when flipping the compare direction. Shape is
        # (1,)*(n_dims-1) x 2 or similar — any constant dense<1> with only
        # one non-unit dim of size 2, within the within-row range.
        descending = False
        for ssa in const_ops:
            if ssa.attrs.get("value") != 1:
                continue
            shape = _extract_shape(ssa.type_str)
            if not shape:
                continue
            # The inversion constants are 1D-like: all dims size 1 except one
            # axis of size 2. The size-2 axis corresponds to a within-row bit.
            size2_axes = [i for i, s in enumerate(shape) if s == 2]
            other_sizes = [s for s in shape if s != 2]
            if len(size2_axes) == 1 and all(s == 1 for s in other_sizes):
                descending = True
                break
        # Fallback: scan raw IR text for the inversion constants
        if not descending:
            # Look for an arith.constant dense<1> : tensor<...x2xi32 shape
            # at the top (before the first make_range).
            raw = getattr(self.graph, "text", None)
            if raw:
                # Simple heuristic: arith.constant dense<1> : ... occurs
                # BEFORE any tt.make_range.
                mr_pos = raw.find("tt.make_range")
                const_match = re.search(r"arith\.constant\s+dense<1>\s*:\s*tensor<", raw)
                if const_match and (mr_pos == -1 or const_match.start() < mr_pos):
                    descending = True

        # Identify stride scalars: they appear in arith.muli with the
        # make_range offsets. For now, name-based heuristic: first scalar
        # arg = stride_xm, second scalar arg = stride_zm. This matches
        # the sort_kernel signature (X, stride_xm, Z, stride_zm).
        stride_xm_name = None
        stride_zm_name = None
        if len(scalar_args) >= 2:
            stride_xm_name = scalar_args[0].name
            stride_zm_name = scalar_args[1].name
        elif len(scalar_args) >= 1:
            stride_xm_name = stride_zm_name = scalar_args[0].name

        # Require M <= 1024 so each row fits in one thread within the tg
        if M > 1024:
            return None

        # Block size: dispatch enough threads to cover all rows.
        block_size = max(M, self.graph.num_warps * 32)
        block_size = min(block_size, 1024)

        return {
            "M": M,
            "N": N,
            "K": K_out,
            "descending": descending,
            "elem_type": elem_type,
            "x_ptr": x_ptr_name,
            "z_ptr": z_ptr_name,
            "stride_xm": stride_xm_name,
            "stride_zm": stride_zm_name,
            "block_size": block_size,
        }


    def _detect_dot_epilogue(self) -> str:
        """Detect epilogue pattern from IR around tt.dot.

        The Triton compiler folds add-matrix/add-rows/add-cols into the
        tt.dot's 3rd operand (accumulator). So these are detected from the
        accumulator source, not from ops after the dot.

        Ops AFTER the dot indicate softmax (tt.reduce) or chain-dot (tt.dot).

        Returns one of: 'none', 'add-matrix', 'add-rows', 'add-cols',
                         'softmax', 'chain-dot'
        """
        dot_op = None
        dot_idx = None
        for i, ssa in enumerate(self.graph.ops):
            if ssa.op == "tt.dot":
                dot_op = ssa
                dot_idx = i
                break
        if dot_op is None:
            return "none"

        # Check ops AFTER the dot
        after_dot = self.graph.ops[dot_idx + 1:]
        n_dot2 = sum(1 for op in after_dot if op.op == "tt.dot")
        n_reduce = sum(1 for op in after_dot if op.op == "tt.reduce")

        if n_dot2 >= 1:
            return "chain-dot"
        if n_reduce >= 1:
            return "softmax"

        # Check accumulator (3rd operand of tt.dot).
        # If it traces back to a tt.load, it's an add epilogue.
        # If it traces to a zero constant or arith.constant, it's 'none'.
        if len(dot_op.operand_ids or []) >= 3:
            acc_id = dot_op.operand_ids[2]
            acc_source = self._trace_dot_accumulator(acc_id)
            if acc_source in ("add-matrix", "add-rows", "add-cols"):
                return acc_source

        return "none"


    def _detect_dot_constant_inputs(self):
        """Check if tt.dot inputs are compile-time constants (arith.constant).

        Returns (const_a, const_b, M, N, K, dot_elem_type) if both inputs
        are constants, or None otherwise.
        """
        import struct as _struct
        op_by_id = {ssa.id: ssa for ssa in self.graph.ops}

        for ssa in self.graph.ops:
            if ssa.op != "tt.dot":
                continue
            if len(ssa.operand_ids) < 2:
                return None
            a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
            a_op = op_by_id.get(a_id)
            b_op = op_by_id.get(b_id)
            if not (a_op and b_op):
                return None
            if a_op.op != "arith.constant" or b_op.op != "arith.constant":
                return None

            def _get_float_val(op):
                v = op.attrs.get("value")
                if v is None:
                    return 0.0
                if isinstance(v, float):
                    return v
                if isinstance(v, int) and op.elem_type in ("f32", "f16", "bf16"):
                    try:
                        return _struct.unpack('f', _struct.pack('I', v & 0xFFFFFFFF))[0]
                    except _struct.error:
                        return 0.0
                return float(v)

            const_a = _get_float_val(a_op)
            const_b = _get_float_val(b_op)

            dot_shape = _extract_shape(ssa.type_str)
            M = dot_shape[0] if len(dot_shape) >= 1 else 32
            N = dot_shape[1] if len(dot_shape) >= 2 else 32
            a_shape = _extract_shape(a_op.type_str)
            K = a_shape[1] if len(a_shape) >= 2 else 32
            return (const_a, const_b, M, N, K, ssa.elem_type)
        return None


    def _detect_reduce_direction(self, ssa: SSAValue) -> bool:
        """Detect argmax (True) vs argmin (False) from reduce body comparison ops."""
        # Float values: cmpf determines direction unambiguously
        for body_op in (ssa.region_ops or []):
            if body_op.op == "arith.cmpf":
                # Use predicate_name (string) if available, fall back to int code
                pred = body_op.attrs.get("predicate_name", "")
                if not pred:
                    # Integer predicate codes: 1=oeq, 2=ogt, 4=olt
                    code = body_op.attrs.get("predicate", -1)
                    if code == 1:
                        continue  # oeq — tie-break, skip
                    return code == 2  # ogt → max, else min
                if "eq" in pred:
                    continue  # oeq — tie-break, skip
                return "gt" in pred  # ogt → max, olt → min
        # Integer values: sgt/ugt means argmax, absence means argmin
        # (slt is always present for index tie-break, so it's not distinctive)
        for body_op in (ssa.region_ops or []):
            if body_op.op == "arith.cmpi":
                pred = body_op.attrs.get("predicate_name", "")
                if not pred:
                    code = body_op.attrs.get("predicate", -1)
                    if code in (4, 8):  # sgt=4, ugt=8
                        return True
                    continue
                if "sgt" in pred or "ugt" in pred:
                    return True  # argmax
        return False  # default: argmin


