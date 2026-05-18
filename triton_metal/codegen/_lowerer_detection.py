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

from triton_metal.codegen.mlir_walker import SSAValue, _extract_shape

from triton_metal.codegen._lowerer_helpers import _mlir_to_triton_dtype


class _DetectionMixin:
    """Pattern-detection predicates for GenericLowerer.

    All methods read instance state (``self.graph``, ``self.env_types``,
    ``self.ssa_values``, etc.) — they do not define new state.
    """

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

        # Find scf.for and check if it contains tt.dot (K-loop pattern)
        scf_for_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "scf.for":
                scf_for_ssa = ssa
                break

        if scf_for_ssa:
            # Check if the scf.for body contains tt.dot
            dot_in_loop = None
            has_loads_in_loop = False
            has_local_alloc_in_loop = False
            if scf_for_ssa.region_ops:
                for body_op in scf_for_ssa.region_ops:
                    if body_op.op == "tt.dot":
                        dot_in_loop = body_op
                    elif body_op.op == "tt.load":
                        has_loads_in_loop = True
                    elif body_op.op == "ttg.local_alloc":
                        has_local_alloc_in_loop = True

            if not dot_in_loop:
                return None  # scf.for without dot — not our pattern

            # Extract BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N from dot operands
            a_type = self._find_op_type_str(dot_in_loop.operand_ids[0])
            b_type = self._find_op_type_str(dot_in_loop.operand_ids[1])
            a_shape = _extract_shape(a_type) if a_type else None
            b_shape = _extract_shape(b_type) if b_type else None

            if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
                return None

            BLOCK_M, BLOCK_K = a_shape[0], a_shape[1]
            BLOCK_K2, BLOCK_N = b_shape[0], b_shape[1]

            ptr_args = [a for a in self.graph.args if a.is_ptr]
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
            }

        # --- Non-K-loop: simple dot without scf.for ---
        dot_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_ssa = ssa
                break
        if not dot_ssa or len(dot_ssa.operand_ids) < 3:
            return None

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

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 3:
            return None

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

        # n_cols: the first non-ptr scalar arg. The kernel's row stride.
        n_arg = None
        for arg in self.graph.args:
            if not arg.is_ptr:
                n_arg = arg.name
                break
        if n_arg is None:
            return None

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

        n_arg = None
        for arg in self.graph.args:
            if not arg.is_ptr:
                n_arg = arg.name
                break
        if n_arg is None:
            return None

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

        # Load shape must be 2D: tensor<MxNx...>
        load_shape = _extract_shape(load_ssa.type_str)
        if not load_shape or len(load_shape) != 2:
            return None
        M, N = load_shape
        # N must be a power of 2
        if N < 1 or (N & (N - 1)) != 0:
            return None

        # Identify the final store shape: (M, K) where K in {N, user_k}
        store_shape = None
        if store_ssa.operand_ids and len(store_ssa.operand_ids) >= 2:
            val_id = store_ssa.operand_ids[1]
            val_type = self._find_op_type_str(val_id)
            store_shape = _extract_shape(val_type) if val_type else None
        if not store_shape or len(store_shape) != 2:
            return None
        if store_shape[0] != M:
            return None
        K_out = store_shape[1]
        # K_out must be a power of 2 and <= N
        if K_out < 1 or (K_out & (K_out - 1)) != 0 or K_out > N:
            return None

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


