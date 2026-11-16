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
from triton_msl.codegen._normalization_proof import normalization_template_matches


# Epilogue ops that don't compute a new value — they only reshape the layout or
# round the representation. In the per-element fused-epilogue loop they resolve
# to their operand's expression (no emitted statement). Output dtype casts are
# absorbed by the final store's cast. #158.
# packet 162 (160 F-b): layout ops ONLY. The matmul+epilogue emitter substitutes a passthrough op
# with its operand's expression, so a conversion listed here would be dropped, not replayed
# (`z.to(int32).to(f32)` would emit as `z`). Conversions inside an epilogue now make
# `_detect_matmul_epilogue` decline; the K-chunk path claims those kernels and replays the casts.
_EPI_PASSTHROUGH = frozenset(
    {
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "ttg.convert_layout",
    }
)


class _DetectionMixin:
    """Pattern-detection predicates for GenericLowerer.

    All methods read instance state (``self.graph``, ``self.env_types``,
    ``self.ssa_values``, etc.) — they do not define new state.
    """

    def _trace_ptr_source(self, ssa_id, op_by_id=None, depth=0, iter_inits=None):
        """Walk the value chain from an SSA id back to a kernel ``FuncArg``.

        Follows the typical ``ttg.local_load → ttg.local_alloc →
        ttg.memdesc_trans? → tt.trans? → tt.reshape? → tt.load →
        tt.addptr* → tt.splat → <func-arg>`` chain. Returns the matching
        ``FuncArg`` or ``None`` if no path lands on one (e.g. for
        constant-initialized accumulators).

        ``iter_inits`` (opt-in, packet 079): an ``scf.for`` iter-arg id -> init id
        map from ``_scf_iter_inits``. With it the walk crosses a loop-carried
        pointer to the address it was seeded with, so a K-loop's A/B streams trace
        to their args instead of failing. Callers that do not pass it are unchanged.
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
        if op is None and iter_inits and ssa_id in iter_inits:
            return self._trace_ptr_source(iter_inits[ssa_id], op_by_id, depth + 1, iter_inits)
        if not op or not op.operand_ids:
            return None
        # ``tt.addptr`` and ``tt.splat`` use operand 0 as the base ptr;
        # ``ttg.local_load`` / ``ttg.local_alloc`` / ``ttg.memdesc_trans``
        # / ``tt.trans`` / ``tt.reshape`` / ``tt.load`` / ``arith.*``
        # all pass through the value (or address) chain via operand 0
        # too, so a single recursive walk handles every step.
        return self._trace_ptr_source(op.operand_ids[0], op_by_id, depth + 1, iter_inits)

    def _scf_iter_inits(self, op_by_id):
        """Map every ``scf.for`` iter-arg block-arg id to its INIT operand id.

        ``block_arg_ids[0]`` is the induction variable; iter-args ``[1:]`` align
        with the inits at ``operand_ids[3:]`` (after lo/hi/step).
        """
        m = {}
        for o in op_by_id.values():
            if o.op != "scf.for":
                continue
            bargs = list((o.attrs or {}).get("block_arg_ids") or [])
            inits = list(o.operand_ids[3:]) if len(o.operand_ids or []) > 3 else []
            for ba, init in zip(bargs[1:], inits):
                m[ba] = init
        return m

    # Value-path vocabulary of the single-dot matmul templates (packet 079).
    _DOT_LAYOUT_OPS = frozenset({"ttg.convert_layout", "ttg.local_alloc", "ttg.local_load", "ttg.memdesc_trans"})
    _DOT_OUTPUT_CASTS = frozenset({"arith.truncf", "arith.extf"})
    _IDX_CASTS = frozenset(
        {
            "arith.index_cast",
            "arith.index_castui",
            "arith.extsi",
            "arith.extui",
            # packet 139/140: ``arith.trunci`` is NOT transparent — a narrowed tile index
            # (``k.to(tl.int8).to(tl.int32)``) is a different value; MLIR folds the
            # identity ``trunci(extsi(x))`` before we see it, so a surviving trunci is
            # a real narrowing and the decomposer must refuse it by name.
            "builtin.unrealized_conversion_cast",
        }
    )
    _PID_OPS = frozenset({"tt.get_program_id", "tt.program_id"})

    def _scf_upper_bound_arg(self, scf_op):
        """Trace an ``scf.for`` upper bound (the reduction extent) to a runtime scalar
        kernel arg through index/int casts. Returns the arg NAME, else None. Structural
        (any arg name) rather than matching the name ``K`` (GitHub issue #4.1)."""
        if not scf_op or not scf_op.operand_ids or len(scf_op.operand_ids) < 2:
            return None
        arg_by_id = {a.id: a for a in self.graph.args}
        op_by_id = {s.id: s for s in self.graph.ops}
        cur, seen = scf_op.operand_ids[1], set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            a = arg_by_id.get(cur)
            if a is not None:
                return a.name
            o = op_by_id.get(cur)
            if o is not None and o.op in self._IDX_CASTS and o.operand_ids:
                cur = o.operand_ids[0]
                continue
            break
        return None

    def _const_int(self, vid, op_by_id):
        c = op_by_id.get(vid)
        if c is None or c.op != "arith.constant":
            return None
        try:
            return int(str(c.attrs.get("value", "")).split(":")[0].strip())
        except (TypeError, ValueError):
            return None

    def _is_k_offset(self, vid, op_by_id, iv, iv_scale):
        """True iff ``vid`` is the loop's K offset IN ELEMENTS: the induction variable
        itself for an element-stepped loop (``range(0, K, BLOCK_K)``, ``iv_scale`` 1),
        or ``iv * BLOCK_K`` (either order) for an iteration-counted loop
        (``range(0, cdiv(K, BLOCK_K))``, ``iv_scale`` BLOCK_K)."""
        vid = self._strip_wrappers(vid, op_by_id, self._IDX_CASTS)
        if iv is None:
            return False
        if iv_scale == 1:
            return vid == iv
        o = op_by_id.get(vid)
        if o is None or o.op not in ("arith.muli", "arith.mul") or len(o.operand_ids or []) != 2:
            return False
        x, y = (self._strip_wrappers(v, op_by_id, self._IDX_CASTS) for v in o.operand_ids)
        return (x == iv and self._const_int(y, op_by_id) == iv_scale) or (
            y == iv and self._const_int(x, op_by_id) == iv_scale
        )

    def _grouped_pid_expr(self, vid, op_by_id):
        """Recognize the exact grouped tutorial DAG, not arbitrary scalar arithmetic.

        Return an axis and a shared structural key. The caller must ALSO prove the
        two cdiv extents match its actual M/N tiles before any template is admitted.
        """

        def binary(v, name):
            op = op_by_id.get(v)
            if op is not None and op.op == name and len(op.operand_ids or []) == 2:
                from triton_msl.errors import MetalNonRecoverableError

                try:
                    facts = self._native_value_facts(v, op_name="tt.dot")
                except MetalNonRecoverableError:
                    return None  # unknown metadata cannot prove this NEW mapping
                if facts.kind != "integer" or facts.width != 32 or facts.shape != () or facts.is_tensor:
                    return None
                return tuple(op.operand_ids)
            return None

        def scaled(v, scale):
            pair = binary(v, "arith.muli")
            if pair:
                for x, c in (pair, pair[::-1]):
                    if self._const_int(c, op_by_id) == scale:
                        return x
            return None

        root = op_by_id.get(vid)
        if root is None:
            return None
        candidates = []
        add = binary(vid, "arith.addi")
        if add:
            for first, remainder in (add, add[::-1]):
                rem = binary(remainder, "arith.remsi")
                if rem:
                    candidates.append((0, first, *rem))
        div = binary(vid, "arith.divsi")
        if div:
            candidates.append((1, None, *div))
        for axis, row_first, within, group_size in candidates:
            minimum = binary(group_size, "arith.minsi")
            if not minimum:
                continue
            for remaining, constant in (minimum, minimum[::-1]):
                group = self._const_int(constant, op_by_id)
                if group is None or not 1 <= group <= 32:
                    continue
                sub = binary(remaining, "arith.subi")
                if not sub:
                    continue
                npm, first = sub
                if row_first is not None and row_first != first:
                    continue
                group_id = scaled(first, group)
                quotient = binary(group_id, "arith.divsi")
                local = binary(within, "arith.remsi")
                if not quotient or local != quotient:
                    continue
                pid, span = quotient
                npn = scaled(span, group)
                pop = op_by_id.get(pid)
                if npn is None or pop is None or pop.op not in self._PID_OPS:
                    continue
                if str(pop.attrs.get("axis", pop.attrs.get("dim", 0))) != "0":
                    continue
                return axis, ("grouped", group, npm, npn)
        return None

    def _pid_expr(self, vid, op_by_id, *, allow_grouped=False):
        """Classify a scalar as a tile-coordinate expression the templates replay:
        ``program_id(axis)`` -> ``(axis, None)``; the tutorial's 1-D grid mapping
        ``program_id(0) % D`` -> ``(0, D)`` (rows) and ``program_id(0) // D`` ->
        ``(1, D)`` (cols). Else None."""
        if allow_grouped:
            grouped = self._grouped_pid_expr(vid, op_by_id)
            if grouped is not None:
                return grouped
        vid = self._strip_wrappers(vid, op_by_id, self._IDX_CASTS)
        o = op_by_id.get(vid)
        if o is None:
            return None
        if o.op in self._PID_OPS:
            try:
                return (int(str(o.attrs.get("axis", o.attrs.get("dim", 0)))), None)
            except (TypeError, ValueError):
                return None
        if o.op in ("arith.remsi", "arith.divsi") and len(o.operand_ids or []) == 2:
            p = op_by_id.get(self._strip_wrappers(o.operand_ids[0], op_by_id, self._IDX_CASTS))
            if p is None or p.op not in self._PID_OPS:
                return None
            try:
                if int(str(p.attrs.get("axis", p.attrs.get("dim", 0)))) != 0:
                    return None
            except (TypeError, ValueError):
                return None
            return (0 if o.op == "arith.remsi" else 1, o.operand_ids[1])
        return None

    def _index_shape(self, vid, op_by_id, arg_by_id, iv, iv_scale=1, *, allow_grouped=False):
        """Decompose a 1-D tile index into what the matmul templates replay:
        ``[program_id(axis) * BLOCK +] make_range [+ <K offset>]`` where the K offset
        is the induction variable (element-stepped loop) or ``iv * BLOCK_K``
        (iteration-counted loop, ``iv_scale`` = BLOCK_K). Returns ``{"range", "iv",
        "pid", "bad"}``: counts of make_range / K-offset addends, the program_id axis
        (None if absent, -1 if mixed) and the first un-replayed construct (a runtime
        scalar offset, a constant addend, a scaled index, ...) or None."""
        res = {
            "range": 0,
            "iv": 0,
            "pid": None,
            "bad": None,
            "mod": None,
            "pid_div": None,
            "coef": None,
            "bounds": None,
            "index_casts": [],
        }

        def note_pid(axis, div, coef=1):
            # packet 106: the pid's tile COEFFICIENT (``pid * BLOCK``) and the make_range
            # BOUNDS are recorded so the templates can require the exact tile they replay
            if res["pid"] is None:
                res["pid"], res["pid_div"], res["coef"] = axis, div, coef
            elif res["pid"] != axis or res["pid_div"] != div:
                res["pid"] = -1  # mixed
            else:
                res["coef"] = -1  # a second pid term (duplicate)

        def walk(v, depth=0):
            if depth > 24:
                res["bad"] = "index expression exceeds the proved depth"
                return
            if res["bad"] is not None:
                return
            if iv is not None and self._is_k_offset(v, op_by_id, iv, iv_scale):
                res["iv"] += 1
                return
            pe = self._pid_expr(v, op_by_id, allow_grouped=allow_grouped)
            if pe is not None:
                note_pid(*pe)
                return
            o = op_by_id.get(v)
            if o is None:
                if v in arg_by_id:
                    res["bad"] = f"runtime scalar offset '{arg_by_id[v].name}'"
                else:
                    res["bad"] = "unresolved value"
                return
            if o.op in ("arith.remsi", "arith.remui") and len(o.operand_ids or []) == 2:
                # ``index % EXTENT`` (the tutorial's wrap-around load index). Recorded
                # for the caller, which admits it only on a LOAD row/col index whose
                # modulus is the very extent the template clips with: the template
                # zero-pads those rows/cols instead of wrapping them, and both feed only
                # output rows/cols the store drops.
                if res["mod"] is not None or depth != 0:
                    res["bad"] = "nested modulo index"
                    return
                res["mod"] = o.operand_ids[1]
                walk(o.operand_ids[0], depth + 1)
                return
            if o.op == "tt.make_range":
                res["range"] += 1
                try:
                    b = (int(o.attrs.get("start", 0)), int(o.attrs.get("end")))
                except (TypeError, ValueError):
                    b = None
                res["bounds"] = b if res["range"] == 1 else None
                return
            if (
                o.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.expand_dims") or o.op in self._IDX_CASTS
            ) and o.operand_ids:
                if o.op in self._IDX_CASTS:
                    res["index_casts"].append(v)
                walk(o.operand_ids[0], depth + 1)
                return
            if o.op in ("arith.addi", "arith.add"):
                for x in o.operand_ids or []:
                    walk(x, depth + 1)
                return
            if o.op in ("arith.muli", "arith.mul") and len(o.operand_ids or []) == 2:
                # only ``<pid expr> * BLOCK`` (either order) is a replayed product
                sides = [op_by_id.get(x) for x in o.operand_ids]
                ci = next((i for i, s in enumerate(sides) if s is not None and s.op == "arith.constant"), None)
                pe = (
                    self._pid_expr(o.operand_ids[1 - ci], op_by_id, allow_grouped=allow_grouped)
                    if ci is not None
                    else None
                )
                if pe is None:
                    res["bad"] = "scaled index"
                    return
                note_pid(*pe, coef=self._const_int(o.operand_ids[ci], op_by_id))
                return
            res["bad"] = o.op

        walk(vid)
        return res

    def _range_root(self, vid, op_by_id):
        """The single ``tt.make_range`` id an index expression is built on, or None."""
        roots, stack, seen = set(), [vid], set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            o = op_by_id.get(v)
            if o is None:
                continue
            if o.op == "tt.make_range":
                roots.add(o.id)
                continue
            if o.op in self._PID_OPS or o.op == "arith.constant":
                continue
            stack.extend(o.operand_ids or [])
        return next(iter(roots)) if len(roots) == 1 else None

    def _strip_wrappers(self, vid, op_by_id, ops):
        seen = set()
        while vid not in seen:
            seen.add(vid)
            o = op_by_id.get(vid)
            if o is None or o.op not in ops or not o.operand_ids:
                return vid
            vid = o.operand_ids[0]
        return vid

    def _is_k_extent(self, vid, k_extent, op_by_id, arg_by_id):
        """True iff ``vid`` is the K extent the template clips with (the arg by name,
        or its constexpr value)."""
        vid = self._strip_wrappers(
            vid, op_by_id, ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.expand_dims") + tuple(self._IDX_CASTS)
        )
        if isinstance(k_extent, str):
            a = arg_by_id.get(vid)
            return a is not None and a.name == k_extent
        return k_extent is not None and self._const_int(vid, op_by_id) == int(k_extent)

    def _is_k_remaining(self, bound_id, k_for, k_extent, op_by_id, arg_by_id, iv_scale=1):
        """True iff ``bound_id`` is ``K - <K offset>``: an ``arith.subi`` of the loop's K
        extent and the loop's K offset in elements (``_is_k_offset``)."""
        vid = self._strip_wrappers(
            bound_id, op_by_id, ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.expand_dims")
        )
        o = op_by_id.get(vid)
        if o is None or o.op not in ("arith.subi", "arith.sub") or len(o.operand_ids or []) != 2:
            return False
        iv = ((k_for.attrs or {}).get("block_arg_ids") or [None])[0]
        return self._is_k_offset(o.operand_ids[1], op_by_id, iv, iv_scale) and self._is_k_extent(
            o.operand_ids[0], k_extent, op_by_id, arg_by_id
        )

    def _advance_matches(self, vid, block_k, stride, op_by_id, arg_by_id):
        """True iff the per-iteration pointer advance ``vid`` is exactly
        ``BLOCK_K * <stride>`` (either operand order; a bare ``BLOCK_K`` constant when
        the traced stride is the folded unit ``"1"``)."""
        vid = self._strip_wrappers(vid, op_by_id, ("tt.splat", "tt.broadcast", "ttg.convert_layout"))
        o = op_by_id.get(vid)

        def _const(v):
            c = op_by_id.get(v)
            if c is None or c.op != "arith.constant":
                return None
            try:
                return int(str(c.attrs.get("value", "")).split(":")[0].strip())
            except (TypeError, ValueError):
                return None

        def _arg_name(v):
            a = arg_by_id.get(self._strip_wrappers(v, op_by_id, self._IDX_CASTS))
            return a.name if a is not None else None

        if stride is None or block_k is None:
            return False
        if stride == "1":
            return _const(vid) == block_k
        if stride.isdigit():  # constexpr-dim stride folded into one constant
            return _const(vid) == block_k * int(stride)
        if o is None or o.op not in ("arith.muli", "arith.mul") or len(o.operand_ids or []) != 2:
            return False
        x, y = o.operand_ids
        return (_const(x) == block_k and _arg_name(y) == stride) or (_const(y) == block_k and _arg_name(x) == stride)

    def _kloop_load_mask_reason(
        self, load, role, k_for, k_extent, index_ids, mn_names, op_by_id, arg_by_id, iv_scale=1
    ):
        """Reason a K-loop operand load mask is NOT the template's own boundary clip,
        else None. The K-loop template zero-pads exactly ``row < M && k + kk < K`` (A)
        / ``k + kk < K && col < N`` (B) and ignores the IR mask, so the mask must be an
        AND of ``<`` comparisons of THIS operand's own tile indices against those very
        extents (a subset is fine: the template clips at least as much), with a literal
        zero ``other``. The K clip may be spelled ``range < K - k`` or ``k + range < K``
        (``k`` = the loop's K offset in elements, see ``_is_k_offset``); a 1-D compare
        takes its axis from the ``expand_dims`` that broadcasts it. On success returns
        ``(None, has_k_clip)`` so the caller can distinguish exact logical-K masking
        from an unmasked complete final block; failures remain a reason string."""
        if len(load.operand_ids or []) >= 3 and not self._acc_init_is_literal_zero(load.operand_ids[2], op_by_id):
            return f"{role} load mask 'other' is not a literal zero (the template zero-pads)"
        leaves, stack, seen = [], [(load.operand_ids[1], None)], set()
        while stack:
            v, axis_ctx = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            o = op_by_id.get(v)
            if o is None:
                return f"{role} load mask is not built from tile-index comparisons"
            if o.op == "tt.expand_dims":
                try:
                    axis_ctx = int(o.attrs.get("axis")) if axis_ctx is None else axis_ctx
                except (TypeError, ValueError):
                    pass
                stack.extend((x, axis_ctx) for x in (o.operand_ids or []))
                continue
            if o.op in ("arith.andi", "tt.broadcast", "ttg.convert_layout", "tt.reshape"):
                stack.extend((x, axis_ctx) for x in (o.operand_ids or []))
                continue
            if o.op == "arith.cmpi":
                leaves.append((o, axis_ctx))
                continue
            return f"{role} load mask contains '{o.op}' (only an AND of boundary comparisons is replayed)"
        if not leaves:
            return f"{role} load mask has no comparison"
        row_ix, col_ix = index_ids.get(role, (None, None))
        row_root = self._range_root(row_ix, op_by_id) if row_ix is not None else None
        col_root = self._range_root(col_ix, op_by_id) if col_ix is not None else None
        iv = ((k_for.attrs or {}).get("block_arg_ids") or [None])[0]
        wrap = ("tt.expand_dims", "tt.broadcast", "ttg.convert_layout", "tt.reshape")
        has_k_clip = False
        for leaf, axis_ctx in leaves:
            pname = str(leaf.attrs.get("predicate_name") or "")
            pnum = leaf.attrs.get("predicate")
            if pname not in ("slt", "ult") and pnum not in (2, 6):
                return f"{role} load mask comparison is '{pname or pnum}', not '<'"
            l, r = leaf.operand_ids[0], leaf.operand_ids[1]
            li = self._tile_index_info(l, op_by_id)
            ri = self._tile_index_info(r, op_by_id)
            if li is None or ri is not None:
                return f"{role} load mask comparison is not '<tile index> < <bound>'"
            idx_1d = self._strip_wrappers(l, op_by_id, wrap)
            root = self._range_root(idx_1d, op_by_id)
            axis = li[1] if li[1] is not None else axis_ctx
            if axis == 1 and root is not None and root == row_root:
                kind = "row"
            elif axis == 0 and root is not None and root == col_root:
                kind = "col"
            else:
                return f"{role} load mask compares an index that is not this operand's row/col tile index"
            is_k = (role == "A" and kind == "col") or (role == "B" and kind == "row")
            if is_k:
                has_k_clip = True
                sh = self._index_shape(idx_1d, op_by_id, arg_by_id, iv, iv_scale)
                if sh["bad"] or sh["pid"] is not None or sh["range"] != 1 or sh["iv"] > 1:
                    return f"{role} load mask K index is not 'range' or 'k + range'"
                if sh["iv"] == 0:
                    ok_bound = self._is_k_remaining(r, k_for, k_extent, op_by_id, arg_by_id, iv_scale)
                else:
                    ok_bound = self._is_k_extent(r, k_extent, op_by_id, arg_by_id)
                if not ok_bound:
                    return f"{role} load mask K bound is not the loop's K extent (the template clips k + kk < K)"
            else:
                if idx_1d != (row_ix if kind == "row" else col_ix):
                    return f"{role} load mask {kind} index is not the address's {kind} index"
                want = mn_names[0] if role == "A" else mn_names[1]
                bound_arg = self._trace_bound_arg(r, op_by_id, arg_by_id)
                if bound_arg is not None:
                    if want is None or bound_arg.name != want:
                        return (
                            f"{role} load mask {kind} bound '{bound_arg.name}' is not the extent the template "
                            f"clips with ({want or 'the single-tile extent'})"
                        )
                else:
                    c = op_by_id.get(
                        self._strip_wrappers(r, op_by_id, ("tt.splat", "tt.broadcast", "ttg.convert_layout"))
                    )
                    try:
                        cval = (
                            int(str(c.attrs.get("value", "")).split(":")[0].strip())
                            if c is not None and c.op == "arith.constant"
                            else None
                        )
                    except (TypeError, ValueError):
                        cval = None
                    if cval is None or li[0] is None or cval < li[0]:
                        return f"{role} load mask {kind} bound is not a runtime extent or a constant >= the tile"
        return None, has_k_clip

    def _dot_layout_walk(self, vid, stop, op_by_id, ops=None):
        """Follow operand 0 through the templates' layout-only ops (``ops``, default
        ``_DOT_LAYOUT_OPS``). Returns ``("op", o)`` at the first op in ``stop``,
        ``("leaf", vid)`` at a block/func arg, ``("bad", opname)`` at anything the
        templates do not replay."""
        vocab = self._DOT_LAYOUT_OPS if ops is None else ops
        seen = set()
        while vid not in seen:
            seen.add(vid)
            o = op_by_id.get(vid)
            if o is None:
                return ("leaf", vid)
            if o.op in stop:
                return ("op", o)
            if o.op in vocab and o.operand_ids:
                vid = o.operand_ids[0]
                continue
            return ("bad", o.op)
        return ("bad", "<cycle>")

    def _dot_operand_paths_reason(self, dot, a_arg, b_arg, op_by_id, *, allow_masked=True, allow_trans=True):
        """The ONE proof of a dot's A/B operand VALUE paths, shared by the bare-template
        predicate and the fused matmul+softmax / matmul+epilogue detectors (Round 2.1 F1:
        the fused template staged raw fp32 buffers when the kernel cast them to fp16 — the
        079 class on a route P0 had not covered). Each operand must be its load through
        layout-only ops, with ``load.elem_type == operand elem_type == buffer elem_type``,
        both buffers the same type, and (``allow_masked=False``) an unmasked load.
        ``allow_trans=False`` (packet 101/102): a ``ttg.memdesc_trans`` on the path is a
        VALUE transform — the K-loop and fused templates never replay it (GPU: a square
        in-tile ``tl.trans`` on A or B stored the raw product, err 41/49; the fused forms
        produced neither the raw nor the intended result for the canonical [N,K] weight)
        — only the single-tile templates' trans_a/trans_b replay is probe-proven, so only
        they may admit it. Returns ``(reason | None, loads, a_operand_shape)``."""
        vocab = self._DOT_LAYOUT_OPS if allow_trans else (self._DOT_LAYOUT_OPS - {"ttg.memdesc_trans"})
        loads, opnd_shape = {}, None
        for role, idx, arg in (("A", 0, a_arg), ("B", 1, b_arg)):
            kind, x = self._dot_layout_walk(dot.operand_ids[idx], {"tt.load"}, op_by_id, vocab)
            if kind != "op":
                what = x if kind == "bad" else "a loop-carried value"
                if kind == "bad" and x in ("ttg.memdesc_trans", "tt.trans"):
                    return (
                        f"dot operand {role} ({arg.name}) is transposed between its load and the dot; "
                        f"this template does not replay transposes (it stages the raw {arg.elem_type} buffer)",
                        loads,
                        None,
                    )
                return (
                    f"dot operand {role} ({arg.name}) is not its load: '{what}' sits between the load and "
                    f"the dot and is not replayed (the template stages the raw {arg.elem_type} buffer)",
                    loads,
                    None,
                )
            load = x
            opnd = op_by_id.get(dot.operand_ids[idx])
            if load.elem_type != arg.elem_type or (opnd is not None and opnd.elem_type != load.elem_type):
                return (
                    f"dot operand {role} ({arg.name}) element type differs from its buffer "
                    f"({arg.elem_type} buffer, {load.elem_type} loaded, {getattr(opnd, 'elem_type', '?')} at the dot)",
                    loads,
                    None,
                )
            if not allow_masked and len(load.operand_ids or []) >= 2:
                return (f"dot operand {role} ({arg.name}) load is masked; the template ignores masks", loads, None)
            if role == "A" and opnd is not None:
                opnd_shape = _extract_shape(opnd.type_str or "")
            loads[role] = load
        if a_arg.elem_type != b_arg.elem_type:
            return (
                f"A ({a_arg.elem_type}) and B ({b_arg.elem_type}) buffers differ in element type",
                loads,
                opnd_shape,
            )
        return (None, loads, opnd_shape)

    def _dot_template_value_paths(self):
        """P0 (packet 079): prove, PER ROLE, that every value path the single-dot
        matmul templates re-emit is exactly what they replay — nothing more.

        Packet 073 proved each role's ADDRESS; 079 showed the VALUE paths were still
        taken on faith. The positive op allowlist admitted ``arith.truncf`` anywhere
        because the fp16-output cast legitimately sits on the store path — so the same
        opcode between a load and the dot (``tl.load(X).to(tl.float16)`` on an fp32
        buffer) passed while the template staged the raw fp32 buffer: GPU err vs raw
        exactly 0, vs intended 0.065. Opcode presence proves nothing about WHERE a
        transform sits. Same class, all GPU-verified on the 078 tree: the fp16
        round-trip on the store path (``store(C_f32, dot.to(f16))``, 0.06), a
        per-iteration cast on the K-loop accumulator (0.11), a non-canonical K-loop
        load mask (dropped, 26.4), a K-loop pointer advance that is not
        ``BLOCK_K * stride`` (dropped, 43.4), and K-loop A/B roles bound by
        DECLARATION ORDER because loop-carried pointers did not trace
        (``(B, A, C)`` -> 345).

        What the templates replay, hence what is proven:
          A/B  load -> {layout ops} -> dot, element types equal to the buffer's; the
               load reads the traced address of its role (through loop iter-args);
               masks only in the K-loop form and only the template's own boundary
               clip (row < M, k + kk < K, col < N) with a zero ``other``.
          C    dot (or the loop's accumulator result) -> {convert_layout} -> AT MOST
               ONE float cast, to C's element type -> the single store.
          K    accumulator iter-arg -> dot.acc and dot -> yield through layout ops
               only, init literally zero; every other iter-arg is an A/B pointer
               stream advanced by exactly BLOCK_K * its traced K stride, or the loads
               index the K axis with ``iv + range``; every tile index is
               ``[program_id(axis) * BLOCK +] make_range`` with rows on axis 0 and
               cols on axis 1 (the template's grid mapping).

        Returns ``(reason, proven_cast_ids, roles, pid_map)``: ``reason`` None when every
        proof holds — then ``proven_cast_ids`` are the cast ops the template DOES replay
        (admitted by the allowlist by IDENTITY, not by opcode), ``roles`` is ``(A, B, C)``
        and ``pid_map`` is the PROVEN grid mapping (``"2d"`` | ``"1d"`` or a
        coordinate-preserving grouped/N-fastest tuple) the pid-tiled
        templates replay verbatim; otherwise a role-naming reason with the rest None.
        """
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
        dots = [o for o in op_by_id.values() if o.op == "tt.dot"]
        stores = [o for o in op_by_id.values() if o.op == "tt.store"]
        if len(dots) != 1 or len(stores) != 1:
            return (
                f"the matmul template requires exactly one tt.dot and exactly one tt.store in the whole "
                f"kernel (found {len(dots)} dot(s), {len(stores)} store(s)); a second store would be "
                f"silently dropped or mis-bound as the output",
                set(),
                None,
                None,
            )
        dot, store = dots[0], stores[0]
        top_fors = [o for o in self.graph.ops if o.op == "scf.for"]
        all_fors = [o for o in op_by_id.values() if o.op == "scf.for"]
        k_for = next((f for f in top_fors if any(o.id == dot.id for o in (f.region_ops or []))), None)
        if len(all_fors) != (1 if k_for is not None else 0):
            return ("a loop other than the K-reduction loop carrying the dot is not replayed", set(), None, None)
        iter_inits = self._scf_iter_inits(op_by_id)
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        roles = self._resolve_dot_ptr_roles(dot, ptr_args)
        if roles is None or len(roles) < 3:
            return (
                "A/B/C pointer roles could not be resolved from dataflow (dot operands traced "
                "through loop iter-args + the store target); they would otherwise be bound by "
                "argument position",
                set(),
                None,
                None,
            )
        a_arg, b_arg, c_arg = roles[0], roles[1], roles[2]
        # The templates accumulate in float and convert ONCE at the store. A dot whose
        # SSA result type is f16/bf16 (``out_dtype``) therefore has its rounding
        # replayed only when that final store conversion IS the result conversion: the
        # store must be the dot's sole observation and C's element type must equal the
        # result type. Extending the f16 result to an f32 buffer (``arith.extf``) makes
        # the rounding observable and the template would expose its raw f32 accumulator
        # (packets 081/083, GPU: err vs raw 0, vs the IR-ordered oracle 0.06) -> refuse.
        # "More precise" is not semantic equivalence; the reference follows the IR's
        # operations and dtypes in order.
        int_dot = dot.elem_type == "i32"
        if dot.elem_type not in ("f32", "f16", "bf16") and not int_dot:
            return (f"dot result type '{dot.elem_type}' is not an accumulator the template models", set(), None, None)
        src_elem = dot.elem_type
        if src_elem in ("f16", "bf16") and c_arg.elem_type != src_elem:
            return (
                f"an {src_elem} result extended to {c_arg.elem_type} makes the result rounding observable; the "
                f"template would store its raw f32 accumulator instead of the {src_elem}-rounded value",
                set(),
                None,
                None,
            )
        traced = self.infer_dot_strides(with_index=True)
        if traced is None:
            return ("operand addresses could not be traced", set(), None, None)
        strides, index_ids, addr_ids = traced

        def _layout_walk(vid, stop):
            return self._dot_layout_walk(vid, stop, op_by_id)

        iv = ((k_for.attrs or {}).get("block_arg_ids") or [None])[0] if k_for is not None else None
        ptr_wrap = ("ttg.convert_layout",)
        block_k_hint = (
            _extract_shape(op_by_id[dot.operand_ids[0]].type_str or "") if dot.operand_ids[0] in op_by_id else None
        )
        block_k_hint = block_k_hint[-1] if block_k_hint and len(block_k_hint) >= 2 else None
        # The loop's K offset in elements: ``iv`` for ``range(0, K, BLOCK_K)`` (step ==
        # BLOCK_K), ``iv * BLOCK_K`` for ``range(0, cdiv(K, BLOCK_K))`` (step == 1).
        iv_scale = 1
        if k_for is not None:
            _step = self._const_int(k_for.operand_ids[2], op_by_id) if len(k_for.operand_ids or []) > 2 else None
            if _step == 1 and block_k_hint:
                iv_scale = block_k_hint
            elif _step != block_k_hint:
                return (f"the K-loop step ({_step}) is neither BLOCK_K ({block_k_hint}) nor 1", set(), None, None)

        # ---- A / B: load -> layout ops -> dot; types (the shared proof; masks are
        # judged below per loop form, so allow them here) ----
        # Transposes: only the single-tile templates replay them (trans_a/trans_b, probe-proven
        # for in-tile square and canonical [N,K] forms); the K-loop templates refuse the
        # canonical layouts already and silently dropped an in-tile transpose (packet 102).
        _ab_reason, loads, opnd_shape = self._dot_operand_paths_reason(
            dot, a_arg, b_arg, op_by_id, allow_masked=True, allow_trans=(k_for is None)
        )
        if _ab_reason:
            return (_ab_reason, set(), None, None)
        block_k = opnd_shape[-1] if opnd_shape and len(opnd_shape) >= 2 else None
        if int_dot:
            # The templates stage int8 operands as float and accumulate in float, then
            # convert once with ``int(acc)``. That replays an int8 -> i32 dot EXACTLY
            # iff every partial sum stays below 2^24: |a*b| <= 2^14, so K <= 1023 —
            # provable only for the single-tile form (K is the baked tile width) with
            # an i32 output and no cast on the store path (``arith.trunci`` to i8 is
            # modular in Triton but undefined for MSL's float -> char). Anything else
            # integer is not replayed.
            if a_arg.elem_type != "i8" or c_arg.elem_type != "i32":
                return (
                    f"an integer dot is replayed exactly only for int8 operands with an i32 output "
                    f"(got {a_arg.elem_type} operands, {c_arg.elem_type} output)",
                    set(),
                    None,
                    None,
                )
            if k_for is not None or block_k is None or block_k > 1023:
                return (
                    "an integer dot is replayed exactly only as a single tile with K <= 1023 (float "
                    "accumulation of int8 products stays exact below 2^24); a K-loop cannot bound the sum",
                    set(),
                    None,
                    None,
                )

        # The extents the templates clip with (same resolution as _k_extent_line /
        # _matmul_output_extent_args): structural from the store mask, else by name.
        scalar_names = {a.name for a in self.graph.args if not a.is_ptr}
        m_ext, n_ext = self._matmul_output_extent_args()
        mn_names = (m_ext or ("M" if "M" in scalar_names else None), n_ext or ("N" if "N" in scalar_names else None))

        # ---- tile index shapes (grid mapping, no residual addends, K form) ----
        # Packet 106 (the 103 class on the bare templates): each tile index must be EXACTLY
        # the tile the template replays — ``make_range(0, BLOCK)`` of that role/axis's block,
        # a pid coefficient equal to that block, and pid PRESENCE consistent per axis (an
        # axis without a program_id is replayed as tile 0 in every program, never tiled).
        # GPU on the P0 tree: no pid / ``pid * 2BM`` / ``range(BN, 2BN)`` / ``range(BK,
        # 2BK)`` all stored the template's tiled result.
        _b_shape = (
            _extract_shape(op_by_id[dot.operand_ids[1]].type_str or "") if dot.operand_ids[1] in op_by_id else None
        )
        _blk = {}
        if opnd_shape and len(opnd_shape) >= 2 and _b_shape and len(_b_shape) >= 2:
            _bm_t, _bk_t, _bn_t = opnd_shape[-2], opnd_shape[-1], _b_shape[-1]
            _blk = {
                ("A", 0): _bm_t,
                ("A", 1): _bk_t,
                ("B", 0): _bk_t,
                ("B", 1): _bn_t,
                ("C", 0): _bm_t,
                ("C", 1): _bn_t,
            }

        # N-fastest flat grids are a DOT-role proof, not a changed meaning for
        # the shared _pid_expr/_index_shape helpers (quantized callers use them).
        # Establish the output's opposite quotient/remainder orientation first;
        # all A/B/C role checks below must then agree on this exact divisor.
        def i32_index_dag(value):
            pending, seen = [value], set()
            while pending:
                v = pending.pop()
                if v in seen:
                    continue
                seen.add(v)
                from triton_msl.errors import MetalNonRecoverableError

                try:
                    facts = self._native_value_facts(v, op_name="tt.dot")
                except MetalNonRecoverableError:
                    return False
                if facts.kind != "integer" or facts.width != 32:
                    return False
                op = op_by_id.get(v)
                if op is not None:
                    if op.op in self._IDX_CASTS or op.op == "arith.trunci":
                        return False  # no discarded width/signedness conversions
                    pending.extend(op.operand_ids or [])
                elif v not in arg_by_id:
                    return False
            return True

        flat_n_divisor = None
        c_indices = index_ids.get("C", (None, None))
        if all(v is not None for v in c_indices):
            c_shapes = [self._index_shape(v, op_by_id, arg_by_id, iv, iv_scale, allow_grouped=True) for v in c_indices]
            d = c_shapes[0]["pid_div"]
            if (
                c_shapes[0]["pid"] == 1
                and c_shapes[1]["pid"] == 0
                and d is not None
                and not isinstance(d, tuple)
                and c_shapes[1]["pid_div"] == d
            ):
                div = op_by_id.get(d)
                bn = _blk.get(("C", 1))
                if div is not None and div.op == "arith.divsi" and len(div.operand_ids or []) == 2 and bn:
                    numerator, denominator = div.operand_ids
                    add = op_by_id.get(numerator)
                    if (
                        self._const_int(denominator, op_by_id) == bn
                        and add is not None
                        and add.op == "arith.addi"
                        and len(add.operand_ids or []) == 2
                    ):
                        x, y = add.operand_ids
                        if any(
                            v in arg_by_id
                            and not arg_by_id[v].is_ptr
                            and arg_by_id[v].name == mn_names[1]
                            and self._const_int(c, op_by_id) == bn - 1
                            for v, c in ((x, y), (y, x))
                        ) and i32_index_dag(d):
                            flat_n_divisor = d

        def dot_index_shape(value):
            sh = self._index_shape(value, op_by_id, arg_by_id, iv, iv_scale, allow_grouped=True)
            if flat_n_divisor is not None and sh["pid_div"] == flat_n_divisor and sh["pid"] in (0, 1):
                if not i32_index_dag(value):
                    sh["bad"] = "N-fastest coordinate is not an exact cast-free i32 DAG"
                else:
                    sh["pid"] = 1 - sh["pid"]
                    sh["pid_div"] = ("flat_n", flat_n_divisor)
            return sh

        _has_pid = {}
        k_form = {}
        for role, ax, want_pid, is_k in (
            ("A", 0, 0, False),
            ("A", 1, None, True),
            ("B", 0, None, True),
            ("B", 1, 1, False),
            ("C", 0, 0, False),
            ("C", 1, 1, False),
        ):
            axis_name = "row" if ax == 0 else "col"
            vid = index_ids.get(role, (None, None))[ax]
            if vid is None:
                return (
                    f"{role} operand stride could not be inferred: its {axis_name} address term is not a "
                    f"classified tile-index term (a residual base offset or an unrecognized expression)",
                    set(),
                    None,
                    None,
                )
            sh = dot_index_shape(vid)
            if isinstance(sh["pid_div"], tuple) and sh["index_casts"]:
                return (f"{role} grouped tile index contains an unreplayed width cast", set(), None, None)
            if sh["bad"]:
                return (
                    f"{role} {axis_name} index contains {sh['bad']}, which the template does not replay",
                    set(),
                    None,
                    None,
                )
            if sh["range"] != 1:
                return (f"{role} {axis_name} index is not a single tile range", set(), None, None)
            _want_blk = _blk.get((role, ax))
            if _want_blk is None or sh["bounds"] != (0, _want_blk):
                return (
                    f"{role} {axis_name} index range is not make_range(0, {_want_blk}) — the template replays exactly that tile",
                    set(),
                    None,
                    None,
                )
            if sh["pid"] is not None and sh["coef"] != _want_blk:
                return (
                    f"{role} {axis_name} index tiles program_id with coefficient {sh['coef']}, not the tile size {_want_blk}",
                    set(),
                    None,
                    None,
                )
            if not is_k:
                _has_pid[(role, ax)] = sh["pid"] is not None
            if sh["mod"] is not None:
                want = mn_names[0] if (role, ax) == ("A", 0) else (mn_names[1] if (role, ax) == ("B", 1) else None)
                mod_arg = arg_by_id.get(
                    self._strip_wrappers(sh["mod"], op_by_id, ("tt.splat",) + tuple(self._IDX_CASTS))
                )
                if want is None or k_for is None or mod_arg is None or mod_arg.name != want:
                    return (
                        f"{role} {axis_name} index wraps with a modulo the template does not replay "
                        f"(only a K-loop LOAD row/col index modulo the very M/N extent it clips with)",
                        set(),
                        None,
                        None,
                    )
            if is_k:
                if sh["pid"] is not None or sh["iv"] > 1:
                    return (f"{role} K index is not 'range' or 'k + range'", set(), None, None)
                k_form[role] = "iv" if sh["iv"] == 1 else "carried"
            else:
                if sh["iv"]:
                    return (f"{role} {axis_name} index depends on the loop induction variable", set(), None, None)
                if sh["pid"] is not None and sh["pid"] != want_pid:
                    return (
                        f"{role} {axis_name} index tiles program_id({sh['pid']}) but the template maps that axis to program_id({want_pid})",
                        set(),
                        None,
                        None,
                    )
        if k_form["A"] != k_form["B"] and k_for is None:
            return ("A and B K indices disagree", set(), None, None)
        # pid PRESENCE per axis must agree between the operand and the output (the template
        # tiles both from ONE pid_m / pid_n); recorded for the templates and the dispatchers.
        if _has_pid.get(("A", 0)) != _has_pid.get(("C", 0)) or _has_pid.get(("B", 1)) != _has_pid.get(("C", 1)):
            return (
                "the operand and output tile indices disagree on which axes are program_id-tiled",
                set(),
                None,
                None,
            )
        pid_axes = (bool(_has_pid.get(("C", 0))), bool(_has_pid.get(("C", 1))))

        # ---- grid mapping: 2-D (program_id(0)/program_id(1)) or the tutorial's 1-D
        # ``pid % num_pid_m`` / ``pid // num_pid_m`` with ONE consistent divisor that is
        # provably cdiv(<the template's M extent>, BLOCK_M). The pid-tiled templates
        # emit exactly that mapping (``info["pid_map"]``); on the 078 tree the 1-D form
        # was dispatched as if 2-D — 60% of C left unwritten on a 9-tile launch.
        pid_divs = {}
        for role in ("A", "B", "C"):
            for ax in (0, 1):
                vid = index_ids.get(role, (None, None))[ax]
                sh = dot_index_shape(vid) if vid is not None else None
                if sh and sh["pid"] is not None:
                    pid_divs[(role, ax)] = sh["pid_div"]
        divs = set(pid_divs.values())
        if len(divs) > 1:
            return ("tile indices mix grid mappings (2-D program ids with a 1-D pid split)", set(), None, None)
        pid_map = "2d"
        grouped = next(iter(divs)) if len(divs) == 1 else None
        if isinstance(grouped, tuple) and grouped[0] == "flat_n":
            pid_map = ("flat_n", _blk[("C", 1)])
        elif isinstance(grouped, tuple) and grouped[0] == "grouped":

            def cdiv_extent(value, extent, block):
                op = op_by_id.get(value)
                if op is None or not block or len(op.operand_ids or []) != 2:
                    return False
                numerator, denominator = op.operand_ids
                if self._const_int(denominator, op_by_id) != block:
                    return False
                # Replay exactly the i32 add-then-div composition. ceildivsi has
                # different overflow semantics; cast-through-extent matching can
                # silently discard a narrowed dimension. Neither is admitted here.
                add = op_by_id.get(numerator)
                if op.op != "arith.divsi" or add is None or add.op != "arith.addi" or len(add.operand_ids or []) != 2:
                    return False
                x, y = add.operand_ids
                return any(
                    self._const_int(c, op_by_id) == block - 1
                    and v in arg_by_id
                    and not arg_by_id[v].is_ptr
                    and arg_by_id[v].name == extent
                    for v, c in ((x, y), (y, x))
                )

            _, group, npm, npn = grouped
            bm, bn = _blk.get(("C", 0)), _blk.get(("C", 1))
            if None in mn_names or not cdiv_extent(npm, mn_names[0], bm) or not cdiv_extent(npn, mn_names[1], bn):
                return ("grouped grid extents do not match the source M/N tiles", set(), None, None)
            pid_map = ("grouped", group, bn)
        elif divs and divs != {None}:
            d = next(iter(divs))
            block_m = opnd_shape[-2] if opnd_shape and len(opnd_shape) >= 2 else None
            dop = op_by_id.get(self._strip_wrappers(d, op_by_id, self._IDX_CASTS))
            ok = False
            if dop is not None and block_m:
                if dop.op == "arith.ceildivsi" and len(dop.operand_ids or []) == 2:
                    ok = self._const_int(dop.operand_ids[1], op_by_id) == block_m and self._is_k_extent(
                        dop.operand_ids[0], mn_names[0], op_by_id, arg_by_id
                    )
                elif (
                    dop.op == "arith.divsi"
                    and len(dop.operand_ids or []) == 2
                    and self._const_int(dop.operand_ids[1], op_by_id) == block_m
                ):
                    num = op_by_id.get(self._strip_wrappers(dop.operand_ids[0], op_by_id, self._IDX_CASTS))
                    if num is not None and num.op in ("arith.addi", "arith.add") and len(num.operand_ids or []) == 2:
                        x, y = num.operand_ids
                        ok = (
                            self._const_int(y, op_by_id) == block_m - 1
                            and self._is_k_extent(x, mn_names[0], op_by_id, arg_by_id)
                        ) or (
                            self._const_int(x, op_by_id) == block_m - 1
                            and self._is_k_extent(y, mn_names[0], op_by_id, arg_by_id)
                        )
            if not ok or mn_names[0] is None:
                return (
                    "the 1-D grid split divisor is not cdiv(<the M extent the template clips with>, BLOCK_M)",
                    set(),
                    None,
                    None,
                )
            pid_map = "1d"

        proven = set()
        if k_for is not None:
            # ---- K-loop: accumulator <-> iter-arg <-> yield, pointer streams, masks ----
            bargs = list((k_for.attrs or {}).get("block_arg_ids") or [])
            inits = list(k_for.operand_ids[3:]) if len(k_for.operand_ids or []) > 3 else []
            body = k_for.region_ops or []
            ylds = [o for o in body if o.op == "scf.yield"]
            if (
                not bargs
                or len(bargs) != len(inits) + 1
                or len(ylds) != 1
                or len(ylds[0].operand_ids or []) != len(inits)
            ):
                return ("K-loop iter-args, inits and yield could not be aligned", set(), None, None)
            yld = ylds[0]
            kind, x = _layout_walk(dot.operand_ids[2], set())
            if kind != "leaf" or x not in bargs[1:]:
                return (
                    "the dot accumulator is not a loop iter-arg reached through layout ops only "
                    "(a per-iteration op on the accumulator would be dropped)",
                    set(),
                    None,
                    None,
                )
            j = bargs.index(x) - 1
            kind, x = _layout_walk(yld.operand_ids[j], {"tt.dot"})
            if kind != "op" or x.id != dot.id:
                return (
                    "the yielded accumulator is not the dot result through layout ops only "
                    "(a per-iteration op on the accumulator would be dropped)",
                    set(),
                    None,
                    None,
                )
            if not self._acc_init_is_literal_zero(inits[j], op_by_id):
                return ("the K-loop accumulator init is not a literal zero", set(), None, None)
            term_id = k_for.result_ids[j] if k_for.result_ids else k_for.id
            carried = {}
            for i, (ba, init) in enumerate(zip(bargs[1:], inits)):
                if i == j:
                    continue
                y = op_by_id.get(yld.operand_ids[i])
                if y is None or y.op != "tt.addptr" or len(y.operand_ids or []) < 2 or y.operand_ids[0] != ba:
                    return (
                        f"loop-carried value #{i} is not a pointer advanced from its own iter-arg",
                        set(),
                        None,
                        None,
                    )
                src = self._trace_ptr_source(init, op_by_id, iter_inits=iter_inits)
                role = (
                    "A"
                    if (src is not None and src.name == a_arg.name)
                    else ("B" if (src is not None and src.name == b_arg.name) else None)
                )
                if role is None or role in carried:
                    return (f"loop-carried pointer #{i} is not the A or B stream", set(), None, None)
                if init != addr_ids[role]:
                    return (f"the {role} stream is seeded from an address other than the traced one", set(), None, None)
                carried[role] = (ba, y)
            # the template's K extent, in the priority _k_extent_line uses
            k_extent = self._scf_upper_bound_arg(k_for)
            if k_extent is None and "K" in scalar_names:
                k_extent = "K"
            if k_extent is None:
                lo_hi = [op_by_id.get(k_for.operand_ids[i]) for i in (0, 1, 2)]
                try:
                    if all(o is not None and o.op == "arith.constant" for o in lo_hi) and block_k:
                        lo, hi, st = (int(str(o.attrs.get("value", "")).split(":")[0].strip()) for o in lo_hi)
                        k_extent = block_k * max(0, (hi - lo + st - 1) // st)
                except (TypeError, ValueError):
                    k_extent = None
            k_tail_clips = {}
            for role in ("A", "B"):
                load = loads[role]
                ptr = self._strip_wrappers(load.operand_ids[0], op_by_id, ptr_wrap)
                if role in carried:
                    ba, y = carried[role]
                    if ptr != ba:
                        return (
                            f"the {role} load inside the K-loop does not read its loop-carried pointer directly",
                            set(),
                            None,
                            None,
                        )
                    if k_form[role] != "carried":
                        return (
                            f"the {role} address mixes a loop-carried pointer with the induction variable",
                            set(),
                            None,
                            None,
                        )
                    kstride = strides[role][1] if role == "A" else strides[role][0]
                    if not self._advance_matches(y.operand_ids[1], block_k, kstride, op_by_id, arg_by_id):
                        return (
                            f"the {role} pointer advance per iteration is not BLOCK_K ({block_k}) * its K stride "
                            f"({kstride}); the template advances by exactly that",
                            set(),
                            None,
                            None,
                        )
                else:
                    if k_form[role] != "iv" or ptr != addr_ids[role]:
                        return (
                            f"the {role} load inside the K-loop neither reads a loop-carried stream nor the traced "
                            f"'k + range' address",
                            set(),
                            None,
                            None,
                        )
                if len(load.operand_ids or []) >= 2:
                    checked = self._kloop_load_mask_reason(
                        load,
                        role,
                        k_for,
                        k_extent,
                        index_ids,
                        mn_names,
                        op_by_id,
                        arg_by_id,
                        iv_scale,
                    )
                    if not isinstance(checked, tuple):
                        return (checked, set(), None, None)
                    _reason, k_tail_clips[role] = checked
                else:
                    k_tail_clips[role] = False
            if k_tail_clips["A"] != k_tail_clips["B"]:
                return (
                    "A and B must either both zero-mask the K tail or both load the source's full final block; "
                    "one-sided masking is not equivalent for nonfinite values",
                    set(),
                    None,
                    None,
                )
            k_tail_mode = "masked_zero" if k_tail_clips["A"] else "full_blocks"
        else:
            k_tail_mode = None
            term_id = dot.id
            for role in ("A", "B"):
                load = loads[role]
                if len(load.operand_ids or []) >= 2:
                    return (f"the {role} load is masked; the non-looped template ignores masks", set(), None, None)
                if self._strip_wrappers(load.operand_ids[0], op_by_id, ptr_wrap) != addr_ids[role]:
                    return (f"the {role} load does not read the traced {role} address", set(), None, None)

        # ---- C: dot / loop result -> convert_layout -> at most the one output cast -> store ----
        if self._strip_wrappers(store.operand_ids[0], op_by_id, ptr_wrap) != addr_ids["C"]:
            return ("the store does not write the traced C address", set(), None, None)
        vid, casts, seen = store.operand_ids[1], [], set()
        while vid != term_id:
            if vid in seen:
                return ("the stored value path cycles", set(), None, None)
            seen.add(vid)
            o = op_by_id.get(vid)
            if o is None:
                return (
                    "the stored value is not the dot result (it reaches a loop-carried or argument value)",
                    set(),
                    None,
                    None,
                )
            if o.op == "ttg.convert_layout" and o.operand_ids:
                vid = o.operand_ids[0]
                continue
            if o.op in self._DOT_OUTPUT_CASTS and o.operand_ids:
                casts.append(o)
                vid = o.operand_ids[0]
                continue
            return (
                f"the stored value path contains '{o.op}' between the dot and the store; the template stores the raw accumulator",
                set(),
                None,
                None,
            )
        if c_arg.elem_type == src_elem:
            if casts:
                return (
                    f"a cast round-trip on the store path is not replayed (the {src_elem} result is stored as is)",
                    set(),
                    None,
                    None,
                )
        else:
            _cast_in = op_by_id.get(casts[0].operand_ids[0]) if casts and casts[0].operand_ids else None
            _in_elem = (
                src_elem if (casts and casts[0].operand_ids[0] == term_id) else getattr(_cast_in, "elem_type", None)
            )
            if len(casts) != 1 or casts[0].elem_type != c_arg.elem_type or _in_elem != src_elem:
                return (
                    f"the store path must carry exactly one cast {src_elem} -> {c_arg.elem_type} "
                    f"(found {[c.op + '->' + str(c.elem_type) for c in casts]})",
                    set(),
                    None,
                    None,
                )
            proven.add(casts[0].id)
        sv = op_by_id.get(store.operand_ids[1])
        if sv is not None and sv.elem_type != c_arg.elem_type:
            return (
                f"the stored value type {sv.elem_type} differs from C's element type {c_arg.elem_type}",
                set(),
                None,
                None,
            )
        # Sole-observation census (packets 081/083): every consumer of the dot result
        # must be a layout op, the PROVEN output cast, the single store or the K-loop's
        # yield — and so on transitively. Any other observation is not replayed (this
        # is what makes the final f32 -> C conversion the result conversion).
        _pending, _seen_c = [dot.id], set()
        while _pending:
            _v = _pending.pop()
            if _v in _seen_c:
                continue
            _seen_c.add(_v)
            for _c in op_by_id.values():
                if _v not in (_c.operand_ids or []):
                    continue
                if _c.op in ("tt.store", "scf.yield", "tt.dot"):
                    continue  # the store / the loop carry / the K-loop dot's own accumulator use
                if _c.op == "ttg.convert_layout" or _c.id in proven:
                    _pending.append(_c.id)
                    continue
                return (
                    f"the dot result is observed by '{_c.op}', which the template does not replay",
                    set(),
                    None,
                    None,
                )
        if (pid_map == "1d" or isinstance(pid_map, tuple)) and pid_axes != (True, True):
            return ("a 1-D grid split needs program_id on both tile axes", set(), None, None)
        return (None, proven, (a_arg, b_arg, c_arg), pid_map, pid_axes, k_tail_mode)

    def _resolve_load_store_ptr_roles(self, load_ssa, store_ssa):
        """Resolve one-load/one-store template roles from pointer dataflow.

        Returns ``(input_arg, output_arg)`` only when the load address and store
        address trace to two distinct pointer function arguments.  Fast templates
        must never infer these roles from declaration order: output-first signatures
        otherwise make the template read the output sentinel and overwrite the real
        input (GPU-confirmed for flip and both transpose templates), while extra
        pointer arguments create the same ambiguity at either end.
        """
        if not load_ssa.operand_ids or not store_ssa.operand_ids:
            return None
        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)

        _collect(self.graph.ops)
        input_arg = self._trace_ptr_source(load_ssa.operand_ids[0], op_by_id)
        output_arg = self._trace_ptr_source(store_ssa.operand_ids[0], op_by_id)
        if input_arg is None or output_arg is None or input_arg.id == output_arg.id:
            return None
        return input_arg, output_arg

    def _norm_input_output_args(self, load_ssa, store_ssa, first_reduce_ssa):
        """For the softmax / layer-norm fast-template detectors: resolve the INPUT and
        OUTPUT pointer args from ACTUAL DATAFLOW, and validate the reduced tensor is the
        loaded tensor directly.

        - INPUT = the arg the ``tt.load`` reads through; OUTPUT = the arg the ``tt.store``
          writes through (traced via ``_trace_ptr_source``). The old code took the first
          ptr arg as input and the second as output by DECLARATION ORDER, so an
          output-first signature ``(out_ptr, x_ptr, n)`` silently swapped the buffers.
        - The first reduction must operate DIRECTLY on the loaded tensor (only shape/
          layout ops in between). ``softmax(x*scale)`` / ``norm(x*scale)`` puts an
          arithmetic op there which the fast template silently drops; those must fall
          through to the generic (correct) lowering.

        Returns ``(input_name, output_name)`` or ``None`` (refuse the fast template).
        """
        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)

        _collect(self.graph.ops)
        in_arg = self._trace_ptr_source(load_ssa.operand_ids[0], op_by_id) if load_ssa.operand_ids else None
        out_arg = self._trace_ptr_source(store_ssa.operand_ids[0], op_by_id) if store_ssa.operand_ids else None
        if in_arg is None or out_arg is None or in_arg.id == out_arg.id:
            return None

        _TRANSPARENT = {
            "ttg.convert_layout",
            "tt.reshape",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.splat",
            "ttg.local_load",
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "tt.trans",
        }

        def _direct(vid, depth=0):
            if depth > 32:
                return False
            if vid == load_ssa.id:
                return True
            o = op_by_id.get(vid)
            if o is None or o.op not in _TRANSPARENT:
                return False
            return any(_direct(x, depth + 1) for x in (o.operand_ids or []))

        if not (first_reduce_ssa.operand_ids and _direct(first_reduce_ssa.operand_ids[0])):
            return None
        return in_arg.name, out_arg.name

    def _resolve_dot_ptr_roles(self, dot_ssa, all_ptr_args):
        """Return ``[A_ptr, B_ptr, C_ptr]`` by tracing dot operands and the
        ``tt.store`` target back to their kernel function args.

        Returns ``None`` when ANY leg cannot be traced or the three legs are not
        distinct args. There is no declaration-order fallback (packet 079): the
        K-loop's A/B streams are loop-carried, and the old "fill an untraced leg
        from declaration order" bound ``(B_ptr, A_ptr, C)`` as A=B_ptr — GPU err
        345, no exception. Loop-carried pointers now trace THROUGH the loop's
        iter-args to their inits instead.
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
        iter_inits = self._scf_iter_inits(op_by_id)
        a_arg = self._trace_ptr_source(dot_ssa.operand_ids[0], op_by_id, iter_inits=iter_inits)
        b_arg = self._trace_ptr_source(dot_ssa.operand_ids[1], op_by_id, iter_inits=iter_inits)
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

    def _acc_init_is_bias(self, init_id, by_id, dot_shape=None):
        """True iff a tt.dot accumulator init is CLEARLY a fused bias the inline matmul
        template would SILENTLY DROP: a loaded value (traced through
        splat/broadcast/layout/shape/dtype wrappers) or a NON-ZERO constant.
        ``tl.zeros`` and unrecognized inits return False — never over-refuse a normal
        matmul. When ``dot_shape`` is given (the strided-template path), only an init whose
        shape matches the dot-output tile (the accumulator) counts; a load/const of a
        different shape is not the accumulator bias. Single source of truth for the three
        bias-init guards (K-loop iter-arg, non-looped dot-operand, strided template) — was
        3 near-identical closures whose divergence is exactly the twin-drift silent-wrong
        risk the campaign warns about.
        """
        _op = by_id.get(init_id)
        _seen = set()
        while _op is not None and _op.id not in _seen:
            _seen.add(_op.id)
            if _op.op in (
                "tt.splat",
                "tt.broadcast",
                "ttg.convert_layout",
                "tt.reshape",
                "tt.expand_dims",
                # Triton inserts a widening cast when an fp16 accumulator load is
                # passed to a float32-output dot.  The cast does not make the init
                # zero: it is still the loaded bias the inline template drops.  Not
                # seeing through arith.extf left all nine upstream fp16->fp32
                # add-matrix/add-row/add-col cases executing silently wrong.
                "arith.extf",
                "arith.truncf",
                "arith.sitofp",
                "arith.uitofp",
                "tt.fp_to_fp",
            ):
                _op = by_id.get(_op.operand_ids[0]) if _op.operand_ids else None
                continue
            break
        if _op is None:
            return False
        if dot_shape is not None and _extract_shape(_op.type_str) != dot_shape:
            return False
        if _op.op == "tt.load":
            return True
        if _op.op == "arith.constant":
            return any(ch in "123456789" for ch in str(_op.attrs.get("value", "")))
        return False

    def infer_dot_strides(self, with_index=False):
        """General ADDRESS-TRACED stride inference for a single-dot matmul.

        ``with_index=True`` (opt-in, packet 079) additionally returns the 1-D tile
        index SSA id behind each classified row/col term and the traced addptr id
        per role: ``(strides, {"A": (row_ix, col_ix), ...}, {"A": addptr_id, ...})``.
        Every existing caller gets the plain ``strides`` dict unchanged.

        Returns ``{"A": (row, col), "B": (row, col), "C": (row, col)}`` where each
        slot is one of: a stride func-arg NAME (runtime stride), the literal
        ``"1"`` (a unit stride that Triton folded away at compile time), or
        ``None`` (the stride could not be unambiguously inferred -> the caller
        MUST refuse loudly rather than guess; never silently wrong).

        Returns ``None`` for the whole kernel when it is not a single-dot matmul
        we can trace (>1 dot, missing store, unreachable addptr) — caller falls
        back to its existing path / refuses.

        Unlike the name-match ``stride_map`` (which needs args literally named
        ``stride_*`` and bakes in row-major roles) this traces the actual
        ``tt.addptr`` offset arithmetic: each additive offset term is either a
        bare ``tt.expand_dims(range)`` (folded unit stride) or
        ``arith.muli(expand_dims(range), tt.splat(stride_arg))``; the
        ``expand_dims`` axis (1 -> row/dim0, 0 -> col/dim1) classifies the term,
        layout-agnostically (so a transposed/strided operand is read correctly).
        """
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

        dots = [s for s in op_by_id.values() if s.op == "tt.dot"]
        if len(dots) != 1:
            return None
        all_ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(all_ptr_args) < 3:
            return None
        roles = self._resolve_dot_ptr_roles(dots[0], all_ptr_args)
        if roles is None or len(roles) < 3:
            return None  # roles unprovable -> strides un-inferable (never positional)
        a_ptr, b_ptr, c_ptr = roles[0], roles[1], roles[2]

        def _skip(o):
            # Narrowing and integer-to-float conversion change addresses. A
            # stride-only template cannot replay them as transparent wrappers.
            seen_ids = set()
            while o is not None and o.op in ("tt.broadcast", "ttg.convert_layout", "tt.reshape", "arith.extsi"):
                if o.id in seen_ids:
                    return None
                seen_ids.add(o.id)
                if not o.operand_ids:
                    return None
                o = op_by_id.get(o.operand_ids[0])
            return o

        def _addptr_for(ptr_arg):
            # The matmul's per-operand offset lives on the addptr whose BASE
            # traces to this pointer func-arg (the pre-loop addptr for a K-loop;
            # the in-loop advance's base is the loop block-arg, which does NOT
            # trace to the arg, so it is naturally excluded). Pick the one whose
            # offset is a 2D tensor (the row+col offset, not a scalar advance).
            best = None
            for o in op_by_id.values():
                if o.op != "tt.addptr" or len(o.operand_ids) < 2:
                    continue
                base = self._trace_ptr_source(o.operand_ids[0], op_by_id)
                if base is None or base.name != ptr_arg.name:
                    continue
                off = op_by_id.get(o.operand_ids[1])
                if off is not None and off.is_tensor:
                    best = o
            return best

        def _chain_offset_terms(addptr, acc):
            # Triton lowers ``P + r*Sr + c*Sc`` either as ONE addptr with an
            # arith.addi offset, or as a CHAIN ``addptr(addptr(P, r*Sr), c*Sc)``
            # (each Python ``+`` becomes its own addptr when not parenthesized).
            # Walk the chain back toward the base, collecting each link's flattened
            # offset terms, so the row AND col factors are both seen regardless of
            # how the additions were grouped. Stops at the first base that is not an
            # addptr on this same operand (a splat(arg) / loop block-arg / a
            # different chain) — its terms aren't part of this offset.
            seen_ids = set()
            cur = addptr
            while cur is not None and cur.op == "tt.addptr" and len(cur.operand_ids) >= 2:
                if cur.id in seen_ids:
                    acc.append(None)
                    return
                seen_ids.add(cur.id)
                _flatten_addi(op_by_id.get(cur.operand_ids[1]), acc)
                base_id = cur.operand_ids[0]
                base = op_by_id.get(base_id)
                # Follow layout-only wrappers down to the next real op.
                base = _skip(base)
                if base is None:
                    if base_id not in arg_by_id:
                        acc.append(None)
                    return
                # Packet 073: a SCALAR base advance (`X + 1`, `Y + off`) is applied to
                # the raw pointer BEFORE the splat — `addptr(splat(addptr(X, 1)), …)`.
                # `_skip` does not cross tt.splat, so that link was never visited and
                # its offset silently vanished (GPU: X[:-1] @ Y for X[1:] @ Y). Cross
                # the splat here so the scalar link's term reaches the strict
                # classification below and refuses.
                if base is not None and base.op == "tt.splat" and base.operand_ids:
                    _inner = op_by_id.get(base.operand_ids[0])
                    if _inner is not None and _inner.op == "tt.addptr":
                        base = _inner
                cur = base

        # Induction variables of the kernel's scf.for loops (``block_arg_ids[0]``): the
        # ONLY scalar addend the in-loop K addressing ``(range + k)`` may carry.
        _iv_ids = set()
        for _lo in op_by_id.values():
            if _lo.op == "scf.for":
                _ba = list((_lo.attrs or {}).get("block_arg_ids") or [])
                if _ba:
                    _iv_ids.add(_ba[0])

        def _iv_addi(o):
            # ``(range + k)`` spelled in-loop lowers as addi(expand_dims(range), splat(k))
            # (2-D; the row/col expand happens before the add). Return the expand_dims
            # iff the other addend is a splat of a loop induction variable — the index
            # proof then sees ``k + range``. A constant or runtime-scalar addend is NOT
            # this form (it stays a residual and refuses). Packet 102: P0's strict tracer
            # refused this mainstream spelling on the quantized route (base computed it).
            o = _skip(o)
            if o is None or o.op not in ("arith.addi", "arith.add") or len(o.operand_ids or []) != 2:
                return None
            parts = [_skip(op_by_id.get(x)) for x in o.operand_ids]
            eds = [p for p in parts if p is not None and p.op == "tt.expand_dims"]
            ivs = [
                p
                for p in parts
                if p is not None and p.op == "tt.splat" and p.operand_ids and p.operand_ids[0] in _iv_ids
            ]
            return eds[0] if (len(eds) == 1 and len(ivs) == 1) else None

        def _flatten_addi(o, acc):
            # Expand additive DAGs iteratively. Re-expanding a shared add node
            # could require exponential work and represents repeated coefficients
            # that the stride descriptor cannot encode, so refuse it as opaque.
            stack = [(o, False)]
            active = set()
            expanded = set()
            while stack:
                cur, leaving = stack.pop()
                cur = _skip(cur)
                if cur is None:
                    acc.append(None)
                    return
                if leaving:
                    active.discard(cur.id)
                    continue
                if cur.op in ("arith.addi", "arith.add") and _iv_addi(cur) is None:
                    if cur.id in active or cur.id in expanded:
                        acc.append(None)
                        return
                    active.add(cur.id)
                    expanded.add(cur.id)
                    stack.append((cur, True))
                    stack.extend((op_by_id.get(oid), False) for oid in reversed(cur.operand_ids))
                else:
                    acc.append(cur)

        def _const_stride(srcop):
            # If srcop is an integer arith.constant, return its value as a string
            # ("1", "32", ...); else None. A constexpr matrix dim (e.g. K folded
            # to `arith.constant 32`) is a compile-time-known stride, so resolving
            # it lets a constexpr-dim row-major matmul address correctly rather
            # than refuse. (Purely additive: runtime-arg and unit-stride results
            # are unchanged.)
            if srcop is None or srcop.op != "arith.constant":
                return None
            raw = str(srcop.attrs.get("value", "")).strip()
            tok = raw.split(":")[0].strip()  # "32 : i32" -> "32"
            try:
                return str(int(float(tok)))
            except (TypeError, ValueError):
                return None

        def _classify(term):
            # Return (axis, stride, index_id) for an offset term. stride is an arg
            # NAME, an integer literal string ("1" folded unit / "32" constexpr dim),
            # or None (present but unresolvable -> caller refuses). index_id is the
            # 1-D tile index the expand_dims lifts (packet 079: the value-path proof
            # checks its SHAPE and matches mask leaves against it).
            term = _skip(term)
            if term is None:
                return (None, None, None)
            stride = "1"
            index_id = None  # the index the proof checks; the addi itself for ``range + k``
            if term.op in ("arith.muli", "arith.mul") and len(term.operand_ids) == 2:
                o0 = _skip(op_by_id.get(term.operand_ids[0]))
                o1 = _skip(op_by_id.get(term.operand_ids[1]))
                ed = o0 if (o0 and o0.op == "tt.expand_dims") else (o1 if (o1 and o1.op == "tt.expand_dims") else None)
                other = o1 if ed is o0 else o0
                if ed is None:
                    # in-loop ``(range + k) * stride``: the addi carries the index
                    for cand, oth in ((o0, o1), (o1, o0)):
                        _ed = _iv_addi(cand)
                        if _ed is not None:
                            ed, other, index_id = _ed, oth, cand.id
                            break
                if ed is None:
                    return (None, None, None)
                sp = _skip(other)
                if sp is not None and sp.op == "tt.splat" and sp.operand_ids:
                    src = sp.operand_ids[0]
                    if src in arg_by_id:
                        stride = arg_by_id[src].name
                    else:
                        stride = _const_stride(op_by_id.get(src))  # constexpr dim or None
                elif sp is not None and sp.op == "arith.constant":
                    stride = _const_stride(sp)
                else:
                    stride = None
                term = ed
            elif _iv_addi(term) is not None:
                # unit stride folded: the bare ``(range + k)`` addi is the term
                index_id = term.id
                term = _iv_addi(term)
            if term.op == "tt.expand_dims":
                return (
                    term.attrs.get("axis"),
                    stride,
                    index_id if index_id is not None else (term.operand_ids or [None])[0],
                )
            return (None, None, None)

        index_ids = {}

        def _strides(addptr, role):
            index_ids[role] = (None, None)
            if addptr is None or len(addptr.operand_ids) < 2:
                return (None, None)
            terms = []
            _chain_offset_terms(addptr, terms)
            row = col = row_ix = col_ix = None
            row_seen = col_seen = False
            for t in terms:
                if t is None:
                    return (None, None)  # opaque term (runtime-scalar residual)
                axis, stride, ix = _classify(t)
                if axis == 1:
                    if row_seen:
                        return (None, None)  # repeated coefficient is not encoded by the descriptor
                    row_seen = True
                    row, row_ix = stride, ix
                elif axis == 0:
                    if col_seen:
                        return (None, None)  # repeated coefficient is not encoded by the descriptor
                    col_seen = True
                    col, col_ix = stride, ix
                else:
                    # STRICT (packet 073): every additive term of the address must be a
                    # classified row/col term. Anything else — a constant base offset
                    # (`X + 1`), a pid-derived scalar, an unrecognized expression — is
                    # a residual the templates never replay: they emit addresses from
                    # (row_stride, col_stride) only, so the +1 was silently dropped
                    # (GPU: X[:-1] @ Y instead of X[1:] @ Y). Un-inferable -> the loud
                    # "stride could not be inferred" refusal.
                    return (None, None)
            index_ids[role] = (row_ix, col_ix)
            return (row, col)

        a_addr = _addptr_for(a_ptr)
        b_addr = _addptr_for(b_ptr)
        c_addr = None
        for s in op_by_id.values():
            if s.op == "tt.store" and s.operand_ids:
                base = self._trace_ptr_source(s.operand_ids[0], op_by_id)
                if base is not None and base.name == c_ptr.name:
                    c_addr = op_by_id.get(s.operand_ids[0])
                    break
        if a_addr is None or b_addr is None or c_addr is None:
            return None
        strides = {"A": _strides(a_addr, "A"), "B": _strides(b_addr, "B"), "C": _strides(c_addr, "C")}
        if with_index:
            return strides, index_ids, {"A": a_addr.id, "B": b_addr.id, "C": c_addr.id}
        return strides

    def _inferred_stride_descriptors(self):
        """Address-traced 6-tuple of matmul stride descriptors, or None.

        Wraps ``infer_dot_strides()`` into the
        ``(a_row, a_col, b_row, b_col, c_row, c_col)`` shape the matmul
        templates consume. Each element is a runtime stride func-arg NAME or
        the literal string ``"1"`` (a folded unit stride). Returns ``None`` when
        the kernel is not a single-dot matmul we can trace, OR when ANY operand
        stride is un-inferable (``infer_dot_strides`` returned ``None`` for that
        slot) — in which case the caller MUST refuse loudly rather than guess a
        row-major layout. Never returns a partially-guessed descriptor.
        """
        traced = self.infer_dot_strides(with_index=True)
        if traced is None:
            return None
        sd, indices, _ = traced
        # These descriptors feed the non-looped fused makers. Check their index
        # arithmetic as well as coefficients; a cast inside expand_dims is just
        # as significant as a cast around the product. Existing maker-level pid
        # guards still decide whether a recognized program-id mapping is supported.
        by_id = {}

        def collect(ops):
            for op in ops:
                by_id[op.id] = op
                if op.region_ops:
                    collect(op.region_ops)
                if op.else_ops:
                    collect(op.else_ops)

        collect(self.graph.ops)
        args = {a.id: a for a in self.graph.args}
        whole_value_path_proved = None
        for role in ("A", "B", "C"):
            for index in indices[role]:
                shape = self._index_shape(index, by_id, args, None)
                if (
                    shape["bad"] is not None
                    or shape["range"] != 1
                    or shape["iv"] != 0
                    or shape["mod"] is not None
                    or not shape["bounds"]
                    or shape["bounds"][0] != 0
                ):
                    # The local shape check deliberately understands only the
                    # simple fused-maker envelope. Bare matmul's whole-value
                    # proof additionally validates grouped/flat PID mappings
                    # and masked wraparound loads. Reuse that complete proof,
                    # not a relaxed index check, before rejecting those routes.
                    if whole_value_path_proved is None:
                        whole_value_path_proved = self._dot_template_value_paths()[0] is None
                    if not whole_value_path_proved:
                        return None
        try:
            a_row, a_col = sd["A"]
            b_row, b_col = sd["B"]
            c_row, c_col = sd["C"]
        except (KeyError, TypeError, ValueError):
            return None
        # A None in any slot = a present-but-unresolvable stride -> caller refuses.
        if any(s is None for s in (a_row, a_col, b_row, b_col, c_row, c_col)):
            return None
        return (a_row, a_col, b_row, b_col, c_row, c_col)

    def _refuse_batched_matmul_base_offset(self):
        """Refuse loudly (MetalNonRecoverableError) when an A/B/C base pointer is
        advanced by a program_id-derived BATCH offset the simdgroup matmul templates
        drop (BLOCKER 4 — batched MMA is not implemented).

        ROOT CAUSE this closes: an idiomatic batched matmul (grid (Bz, cdiv(M,BM),
        cdiv(N,BN)); ``a += pb*sab`` etc., K-loop over BK) computed batch 0's tile
        region for EVERY batch — the simple/K-loop simdgroup templates map only
        program_id(0/1) to the M/N tiles and IGNORE any batch-pointer offset. The
        old axis>=2 guard ran ONLY in the non-K-loop branch (and only caught a
        program_id(2) batch), so the K-loop twin with batch on program_id(0) slipped
        through. This runs for BOTH branches and traces the actual base-pointer
        offset arithmetic instead of a single pid axis.

        Detection: for each A/B/C pointer func-arg, find any ``tt.addptr`` whose base
        traces to that arg and whose offset is a SCALAR (not the 2-D row+col tensor
        offset ``infer_dot_strides`` already accounts for) and depends on a
        ``tt.get_program_id`` / ``tt.get_num_programs``. Such a term is a batch (or
        otherwise un-modeled) pointer advance -> refuse.
        """
        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)

        _collect(self.graph.ops)

        all_ptr_args = [a for a in self.graph.args if a.is_ptr]
        dots = [s for s in op_by_id.values() if s.op == "tt.dot"]
        if len(dots) != 1 or len(all_ptr_args) < 3:
            return  # only guard the single-dot matmul shape we route here
        roles = self._resolve_dot_ptr_roles(dots[0], all_ptr_args)
        if roles is None or len(roles) < 3:
            return  # unprovable roles refuse at the template boundary; never guess positionally
        ptrs = roles[:3]
        ptr_names = {p.name for p in ptrs}

        _pid_memo = {}

        def _depends_on_pid(start_id):
            """True/False when proven; None for a cycle. Iterative and memoized."""
            if start_id in _pid_memo:
                return _pid_memo[start_id]
            stack = [(start_id, False)]
            active = set()
            while stack:
                vid, leaving = stack.pop()
                if vid in _pid_memo:
                    continue
                op = op_by_id.get(vid)
                if leaving:
                    active.discard(vid)
                    facts = [_pid_memo.get(oid) for oid in (op.operand_ids or [])]
                    _pid_memo[vid] = (
                        True if any(f is True for f in facts) else None if any(f is None for f in facts) else False
                    )
                    continue
                if vid in active:
                    return None
                if op is None:
                    # Func/block arguments are graph leaves, not operations.
                    _pid_memo[vid] = False
                    continue
                if op.op in ("tt.get_program_id", "tt.program_id", "tt.get_num_programs"):
                    _pid_memo[vid] = True
                    continue
                active.add(vid)
                stack.append((vid, True))
                for oid in reversed(op.operand_ids or []):
                    if oid in active:
                        return None
                    if oid not in _pid_memo:
                        stack.append((oid, False))
            return _pid_memo.get(start_id)

        def _skip_wrap(o):
            # Keep value-changing arithmetic visible to the address proof.
            seen_ids = set()
            while o is not None and o.op in ("tt.broadcast", "ttg.convert_layout", "tt.reshape", "arith.extsi"):
                if o.id in seen_ids:
                    return None
                seen_ids.add(o.id)
                if not o.operand_ids:
                    return None
                o = op_by_id.get(o.operand_ids[0])
            return o

        def _flatten_add(o, acc):
            stack = [(o, False)]
            active = set()
            expanded = set()
            while stack:
                cur, leaving = stack.pop()
                cur = _skip_wrap(cur)
                if cur is None:
                    acc.append(None)
                    return
                if leaving:
                    active.discard(cur.id)
                    continue
                if cur.op in ("arith.addi", "arith.add"):
                    if cur.id in active or cur.id in expanded:
                        acc.append(None)
                        return
                    active.add(cur.id)
                    expanded.add(cur.id)
                    stack.append((cur, True))
                    stack.extend((op_by_id.get(oid), False) for oid in reversed(cur.operand_ids))
                else:
                    acc.append(cur)

        def _term_has_range(o, depth=0):
            # True if this offset term is built from a tt.make_range / tt.expand_dims
            # (the modeled per-row/col 2-D offset), i.e. it VARIES across the tile.
            o = _skip_wrap(o)
            if o is None or depth > 24:
                return False
            if o.op in ("tt.make_range", "tt.expand_dims"):
                return True
            return any(_term_has_range(op_by_id.get(oid), depth + 1) for oid in (o.operand_ids or []))

        # 2-D stride arg names (address-traced). The CANONICAL tiled matmul advances an
        # A/B/C base by ``pid_m*BM*stride_row`` — a valid 2-D tiling whose multiplier is the
        # operand's OWN 2-D stride. A genuine BATCH advance (``pid*stride_batch``) multiplies
        # a DIFFERENT stride (the batch dimension). So a program_id-derived advance is a real
        # batch advance (drop -> refuse) ONLY if it references a stride arg that is NOT one of
        # the inferred 2-D strides. (Without this the guard over-refuses the common tile-base
        # advance idiom — re-audit MAJOR.)
        _sd = None
        try:
            _sd = self.infer_dot_strides()
        except Exception:
            _sd = None
        _2d_strides = set()
        if _sd:
            for _rc in _sd.values():
                for _s in _rc:
                    if _s and _s != "1":
                        _2d_strides.add(_s)
        _arg_name_by_id = {a.id: a.name for a in self.graph.args}

        def _argnames(start_id, depth=0, seen=None):
            if seen is None:
                seen = set()
            if start_id in seen or depth > 32:
                return set()
            seen.add(start_id)
            names = set()
            if start_id in _arg_name_by_id:
                names.add(_arg_name_by_id[start_id])
            o = op_by_id.get(start_id)
            if o:
                for oid in o.operand_ids or []:
                    names |= _argnames(oid, depth + 1, seen)
            return names

        # tt.make_range extents = the tile sizes (BM/BN/BK). A folded contiguous tile advance
        # is pid*BM (col stride folded to 1), so its constant multiplier IS a tile extent; a
        # batch advance is pid*(M*K), whose constant is NOT a tile extent.
        _extents = set()
        for _ms in op_by_id.values():
            if _ms.op == "tt.make_range":
                _e = _ms.attrs.get("end")
                if isinstance(_e, int):
                    _extents.add(_e)

        def _const_mult(o, depth=0):
            # Product of the arith.constant int factors in a muli chain (the constexpr part of
            # a pid term). None if a non-constant, non-program_id factor (a func-arg) is present
            # — those are classified by the 2-D-stride check instead.
            o = _skip_wrap(o)
            if o is None or depth > 24:
                return 1
            if o.op == "arith.constant":
                try:
                    return int(str(o.attrs.get("value", "")).split(":")[0].strip())
                except Exception:
                    return None
            if o.op in ("tt.get_program_id", "tt.program_id", "tt.get_num_programs"):
                return 1
            if o.op in ("arith.muli", "arith.mul"):
                p = 1
                for oid in o.operand_ids or []:
                    m = _const_mult(op_by_id.get(oid), depth + 1)
                    if m is None:
                        return None
                    p *= m
                return p
            return None

        def _is_batch_advance(off_id):
            # A program_id-derived advance into an A/B/C base pointer is a BATCH advance — which
            # the simdgroup template DROPS (silently computing only batch 0) -> must REFUSE —
            # unless EVERY uniform pid term is provably a TILE offset. Works for a scalar advance
            # (a += pid*stride_batch) AND a tensor offset that broadcast-ADDS the batch term.
            # A uniform pid term (depends on a program_id, no make_range/expand_dims factor) is a
            # valid tile advance iff: (a) it multiplies an inferred 2-D stride ARG (runtime tile
            # advance pid_m*BM*stride_row), OR (b) it is a pure constexpr multiple whose constant
            # equals a make_range EXTENT (folded contiguous tile advance pid_m*BM, col stride
            # folded to 1). Anything else — a batch-stride arg (pid*sab), or a constexpr
            # multiplier that is NOT a tile extent (pid*M*K, the batch stride) — is a batch
            # advance. AMBIGUITY = REFUSE. (re-audit: the constexpr-folded spelling z*(M*K)
            # slipped the prior arg-only gate and silently computed only batch 0.)
            terms = []
            _flatten_add(op_by_id.get(off_id), terms)
            for t in terms:
                if t is None:
                    return True  # malformed/cyclic proof graph: ambiguity must refuse
                pid_fact = _depends_on_pid(t.id)
                if pid_fact is None:
                    return True  # incomplete proof is not affirmative evidence of no batch offset
                if not pid_fact or _term_has_range(t):
                    continue
                _names = _argnames(t.id)
                if _names and not (_names - _2d_strides):
                    continue  # runtime 2-D-stride tile advance -> allow
                if not _names:
                    _m = _const_mult(t)
                    if _m is not None and _m in _extents:
                        continue  # folded tile advance (multiplier == a tile extent) -> allow
                return True  # batch / ambiguous -> refuse
            return False

        for o in op_by_id.values():
            if o.op != "tt.addptr" or len(o.operand_ids) < 2:
                continue
            base = self._trace_ptr_source(o.operand_ids[0], op_by_id)
            if base is None or base.name not in ptr_names:
                continue
            off = op_by_id.get(o.operand_ids[1])
            if off is None:
                continue
            # A program_id-derived advance into a matmul base pointer is a BATCH offset the
            # simdgroup template drops. _is_batch_advance flattens the offset (scalar OR tensor)
            # and refuses any uniform pid term that is not a provable 2-D tile advance.
            if _is_batch_advance(o.operand_ids[1]):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "batched matmul (an A/B/C base pointer advanced by a "
                    "program_id-derived batch offset, e.g. a += pid*stride_batch on "
                    "a 3-D grid, whether a separate addptr or broadcast-added into the "
                    "tile offset) is not supported by the simdgroup matmul template, "
                    "which maps only program_id(0/1) to the M/N output tile and would "
                    "compute batch 0's region for every batch. Batched MMA is not "
                    "implemented. Refusing rather than mis-compute. Loop the batch "
                    "dimension on the host, or use a separate kernel per batch.",
                    op_name="tt.dot",
                )

    # Wrappers a tile index can pass through on the way to its make_range. Kept in sync
    # with the ``_IDX_WRAP`` closure inside ``_template_output_mask_nontrivial`` (the mask
    # guard) -- ``_tile_index_info`` and that closure MUST agree, since the guard's
    # acceptance and this resolver's ``_M``/``_N`` must pick the SAME arg (GitHub issue #4.5).
    _MASK_IDX_WRAP = (
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "tt.broadcast",
        "ttg.convert_layout",
        "tt.reshape",
        "tt.splat",
        "tt.expand_dims",
    )

    def _tile_index_info(self, start_id, by_id):
        """Trace an SSA value to a ``tt.make_range`` tile index. Returns
        ``(extent, axis, has_pid)`` -- axis = 1 for a ``[:, None]`` ROW index, 0 for a
        ``[None, :]`` COL index, None if 1-D -- or None if no make_range is reachable (the
        operand is a scalar bound, not a tile index). Mirrors the guard's ``_index_info``."""
        seen, stack = set(), [start_id]
        has_range = has_pid = False
        extent = axis = None
        while stack:
            vid = stack.pop()
            if vid in seen:
                continue
            seen.add(vid)
            o = by_id.get(vid)
            if o is None:
                continue
            if o.op == "tt.make_range":
                has_range = True
                try:
                    extent = int(o.attrs.get("end")) - int(o.attrs.get("start"))
                except (TypeError, ValueError):
                    extent = None
                continue
            if o.op in ("tt.get_program_id", "tt.program_id", "tt.get_num_programs"):
                has_pid = True
                continue
            if o.op == "tt.expand_dims":
                try:
                    axis = int(o.attrs.get("axis"))
                except (TypeError, ValueError):
                    pass
                stack.extend(o.operand_ids or [])
                continue
            if o.op in self._MASK_IDX_WRAP:
                stack.extend(o.operand_ids or [])
                continue
        return (extent, axis, has_pid) if has_range else None

    def _trace_bound_arg(self, bound_id, by_id, arg_by_id):
        """Trace a mask comparison's BOUND through layout-only wrappers to the kernel arg it
        is; returns the arg (``.name``/``.index``) or None (a constant / expression). Same
        wrapper set as the guard's ``_bound_ok`` trace -- they MUST agree."""
        seen, cur = set(), bound_id
        o = by_id.get(cur)
        while (
            o is not None
            and o.id not in seen
            and o.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.reshape")
            and o.operand_ids
        ):
            seen.add(o.id)
            cur = o.operand_ids[0]
            o = by_id.get(cur)
        return arg_by_id.get(cur)

    def _matmul_output_extent_args(self):
        """Resolve the output row/col extent ARGS structurally from the matmul's output
        ``tt.store`` mask (GitHub issue #4.5). A square ``N x N`` matmul that clips both axes
        by one runtime arg then lowers with ``_M``/``_N`` = that arg, instead of the template
        guessing ``_M = BLOCK_M`` off the missing name 'M' (which drops rows past the first
        tile). Returns ``(m_extent_arg, n_extent_arg)`` -- the arg NAME bounding rows / cols,
        or None. Only a clean single runtime-arg bound resolves; const bounds / value masks
        return None so the mask guard keeps its refuse behavior for those."""

        def _flat(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _flat(s.region_ops)
                if s.else_ops:
                    yield from _flat(s.else_ops)

        allops = list(_flat(self.graph.ops))
        by_id = {s.id: s for s in allops}
        arg_by_id = {a.id: a for a in self.graph.args}
        m_arg = n_arg = None
        for st in allops:
            if st.op != "tt.store" or len(st.operand_ids or []) < 3:
                continue
            stack, seen = [st.operand_ids[2]], set()
            while stack:
                mid = stack.pop()
                if mid in seen:
                    continue
                seen.add(mid)
                o = by_id.get(mid)
                if o is None:
                    continue
                if o.op in ("arith.andi", "tt.broadcast", "ttg.convert_layout", "tt.reshape", "tt.expand_dims"):
                    stack.extend(o.operand_ids or [])
                    continue
                if o.op != "arith.cmpi" or len(o.operand_ids or []) < 2:
                    continue
                op0, op1 = o.operand_ids[0], o.operand_ids[1]
                i0 = self._tile_index_info(op0, by_id)
                i1 = self._tile_index_info(op1, by_id)
                if i0 is not None and i1 is None:
                    axis, bound_id = i0[1], op1
                elif i1 is not None and i0 is None:
                    axis, bound_id = i1[1], op0
                else:
                    continue
                arg = self._trace_bound_arg(bound_id, by_id, arg_by_id)
                if arg is None:
                    continue
                if axis == 0:  # [None, :] col index -> N
                    n_arg = arg.name
                else:  # [:, None] row index (axis 1) or 1-D -> M
                    m_arg = arg.name
        return m_arg, n_arg

    def _template_output_mask_nontrivial(self, is_fa, fa_ctx_index=None):
        """True iff a dot-bearing kernel carries an output ``tt.store`` mask that
        RESTRICTS the output WITHIN the computed tile — a non-tile-boundary mask the
        matmul / FlashAttention TEMPLATES would SILENTLY DROP. They gate writes only on
        the hard-coded tile boundary (``gr < M && gc < N`` for matmul, ``qr < N_CTX`` for
        FA — and, for aligned float, a fully-unmasked simdgroup_store) and never thread
        the store's mask operand into the emitted kernel.

        Because the templates compute the FULL baked output tile, a mask that is
        TRIVIALLY-TRUE over that tile is HARMLESS to drop and returns False here:
          * no mask at all, or
          * the canonical tile-boundary clip ``(rm < M) & (rn < N)`` / ``om < N_CTX``
            (the template's own boundary already enforces exactly this), or
          * a comparison against a compile-time constant >= the tile extent.
        A mask that actually RESTRICTS the output WITHIN the tile — a tighter
        ``make_range`` bound such as ``rm < BOUND`` with ``BOUND < M``, or a value mask
        like ``acc > 0`` — returns True (it would be SILENTLY DROPPED, clobbering the
        masked-off elements with finite values; latent OOB if C is under-allocated).

        ``is_fa`` selects the boundary-arg names the template's OWN clip references (FA
        clips output ROWS by N_CTX; matmul clips rows by M and cols by N) — the caller
        passes the template kind the kernel routes to.

        SINGLE SOURCE OF TRUTH for the triviality classification, shared by the matmul
        dispatch chokepoint (``_refuse_nontrivial_template_output_mask``) AND the
        head_dim=128 simdgroup/tiled FA self-refusal (``_lower_flash_attention_template``)
        so the two cannot diverge.

        Structural trivially-true proof (naming mirrors the template's OWN boundary
        clip, which references the kernel arg literally named M / N / N_CTX): the mask
        must be an AND-tree whose every leaf is ``cmpi`` of a ``make_range``-rooted tile
        index against the matching output-extent arg (row index < M / N_CTX, col index
        < N) or a constant >= the tile extent. Anything else (a different/extra arg
        bound, a value comparison, OR / XOR, an un-modelled expression) is NOT
        provably-trivial -> returns True.
        """

        def _flatten(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _flatten(s.region_ops)
                if s.else_ops:
                    yield from _flatten(s.else_ops)

        allops = list(_flatten(self.graph.ops))
        dots = [s for s in allops if s.op == "tt.dot"]
        if not dots:
            return False  # not a matmul / FA kernel — the mask-dropping templates never fire

        # Only the stores that could route to a template matter, and a template only
        # fires for a dot kernel. Classify every masked output store in such a kernel.
        masked_stores = [s for s in allops if s.op == "tt.store" and len(s.operand_ids or []) >= 3]
        if not masked_stores:
            return False  # no output mask -> nothing dropped

        by_id = {s.id: s for s in allops}
        arg_by_id = {a.id: a for a in self.graph.args}

        # FA clips output ROWS by N_CTX; matmul clips rows by M and cols by N.
        ok_row_names = {"N_CTX"} if is_fa else {"M"}
        ok_col_names = set() if is_fa else {"N"}

        _IDX_WRAP = (
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "tt.broadcast",
            "ttg.convert_layout",
            "tt.reshape",
            "tt.splat",
            "tt.expand_dims",
        )

        def _index_info(start_id):
            """Trace an SSA value back toward a ``tt.make_range`` (a tile-position
            index). Return ``(extent, axis)`` if a make_range is reachable (extent =
            its end-start span; axis = the ``tt.expand_dims`` axis on the path, 1 for a
            ``[:, None]`` ROW index / 0 for a ``[None, :]`` COL index / None if 1-D),
            else ``None`` (this operand is NOT a tile index — e.g. the scalar bound)."""
            seen = set()
            stack = [start_id]
            has_range = False
            has_pid = False
            extent = None
            axis = None
            while stack:
                vid = stack.pop()
                if vid in seen:
                    continue
                seen.add(vid)
                o = by_id.get(vid)
                if o is None:
                    continue  # func arg / external — not an index by itself
                if o.op == "tt.make_range":
                    has_range = True
                    try:
                        extent = int(o.attrs.get("end")) - int(o.attrs.get("start"))
                    except (TypeError, ValueError):
                        extent = None
                    continue
                if o.op in ("tt.get_program_id", "tt.program_id", "tt.get_num_programs"):
                    has_pid = True  # multi-block tile offset pid*BLOCK+range
                    continue
                if o.op == "tt.expand_dims":
                    try:
                        axis = int(o.attrs.get("axis"))
                    except (TypeError, ValueError):
                        pass
                    stack.extend(o.operand_ids or [])
                    continue
                if o.op in _IDX_WRAP:
                    stack.extend(o.operand_ids or [])
                    continue
                # tt.get_program_id / arith.constant: scalar leaves of pid*BLOCK+range;
                # ignore (they don't disqualify a reachable make_range). Any other op
                # leaves has_range as-is.
            return (extent, axis, has_pid) if has_range else None

        def _bound_ok(bound_id, ok_names, idx_extent, has_pid):
            """True iff the comparison's BOUND is the template's output-extent arg for
            this axis (matching name) or a constant >= the tile extent."""
            seen = set()
            cur = bound_id
            o = by_id.get(cur)
            while (
                o is not None
                and o.id not in seen
                and o.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.reshape")
                and o.operand_ids
            ):
                seen.add(o.id)
                cur = o.operand_ids[0]
                o = by_id.get(cur)
            arg = arg_by_id.get(cur)
            if arg is not None:
                # A runtime-arg bound is trivially droppable for MATMUL: the template now
                # clips _M/_N at exactly this arg (structural, _matmul_output_extent_args),
                # so the mask is applied, not dropped -- for ANY arg name (issue #4.5). FA's
                # template clips output ROWS at exactly its resolved N_CTX arg, so the mask is
                # honored (not dropped) when the store bound IS that same arg -- matched by
                # NAME ("N_CTX") or, when the seq-len was resolved STRUCTURALLY from a renamed
                # arg (seqlen, L, ...), by that arg's INDEX (fa_ctx_index).
                if not is_fa:
                    return True
                return arg.name in ok_names or (
                    fa_ctx_index is not None and getattr(arg, "index", None) == fa_ctx_index
                )
            if o is not None and o.op == "arith.constant":
                try:
                    val = int(o.attrs.get("value"))
                except (TypeError, ValueError):
                    return False
                # Equal-to-one specialization removes the FA N_CTX argument,
                # while the emitted template still clips query rows at the exact
                # literal 1.  Only that row-axis boundary is therefore harmless
                # to drop; a column mask or any other constant remains subject to
                # the ordinary multi-block refusal below.
                if is_fa and fa_ctx_index == "c1" and ok_names == {"N_CTX"}:
                    return val == 1
                # a CONSTANT bound on a MULTI-BLOCK index (pid*BLOCK+range) can't be
                # proven trivial: the make_range extent is the per-block span, the index
                # runs to a RUNTIME total -> a const between BLOCK and total clips later
                # blocks. Only the arg-name case (matching the template's own clip) is
                # trivial for multi-block. Re-audit 2026-06-28.
                return (not has_pid) and idx_extent is not None and val >= idx_extent
            return False

        def _leaf_is_trivial(cmp_op):
            if cmp_op.op != "arith.cmpi":
                return False  # cmpf value mask, etc. -> not a tile-boundary clip
            pred = str(cmp_op.attrs.get("predicate_name", ""))
            if pred not in ("slt", "ult", "sle", "ule", "sgt", "ugt", "sge", "uge"):
                return False  # eq/ne -> not an upper-bound tile clip
            if len(cmp_op.operand_ids or []) < 2:
                return False
            op0, op1 = cmp_op.operand_ids[0], cmp_op.operand_ids[1]
            info0, info1 = _index_info(op0), _index_info(op1)
            if info0 is not None and info1 is None:
                idx_info, bound_id, idx_left = info0, op1, True
            elif info1 is not None and info0 is None:
                idx_info, bound_id, idx_left = info1, op0, False
            else:
                return False  # neither (or both) is a tile index -> can't prove trivial
            # Must be an UPPER bound on the index (idx < bound):
            #   slt/ult/sle/ule with idx on the LEFT, or sgt/ugt/sge/uge with idx RIGHT.
            upper = (pred in ("slt", "ult", "sle", "ule") and idx_left) or (
                pred in ("sgt", "ugt", "sge", "uge") and not idx_left
            )
            if not upper:
                return False
            extent, axis, has_pid = idx_info
            ok_names = ok_col_names if axis == 0 else ok_row_names
            return _bound_ok(bound_id, ok_names, extent, has_pid)

        def _mask_is_trivial(mask_id, depth=0):
            if depth > 64:
                return False
            o = by_id.get(mask_id)
            if o is None:
                return False
            if o.op == "arith.andi":
                ops = o.operand_ids or []
                return bool(ops) and all(_mask_is_trivial(c, depth + 1) for c in ops)
            if o.op in ("tt.broadcast", "ttg.convert_layout", "tt.reshape", "tt.expand_dims"):
                return bool(o.operand_ids) and _mask_is_trivial(o.operand_ids[0], depth + 1)
            if o.op == "arith.cmpi":
                return _leaf_is_trivial(o)
            return False  # arith.ori/xori, cmpf (value mask), splat(true), unknown -> refuse

        for st in masked_stores:
            if not _mask_is_trivial(st.operand_ids[2]):
                return True
        return False

    def _refuse_nontrivial_template_output_mask(self):
        """SYSTEMIC anti-silent-wrong gate at the SINGLE matmul dispatch chokepoint in
        ``lower()`` for every route that has not already proved its own stores: a MATMUL
        with a non-tile-boundary output ``tt.store`` mask has NO
        correct lowering on this backend — the simdgroup simple-dot / K-loop / strided /
        matmul+softmax templates compute the FULL tile and gate writes only on the tile
        boundary (silently DROPPING a tighter user mask), AND the generic per-element
        matmul fallback is itself wrong at 16/32/64 — so we REFUSE LOUDLY rather than emit
        silently-wrong output.

        Backward FlashAttention runs its whole-value and exact-store-mask proof before
        this gate and returns only after that proof succeeds; an unrecognized or
        ambiguous backward-shaped graph still reaches this fail-closed check. Forward
        FlashAttention is DELIBERATELY EXEMPTED here. An FA kernel routes to a mask-
        HONORING path: head_dim<=64 lowers through the scalar-tiled / generic
        ``_lower_store``, which clips the user's output mask correctly. Refusing FA at
        this chokepoint would OVER-REFUSE a clip the honoring path computes (the
        regression this method's split repairs). The ONE FA path that DROPS the mask — the
        head_dim=128 simdgroup/tiled template, which takes no mask operand — refuses the
        non-tile-boundary mask ITSELF at emission (``_lower_flash_attention_template``),
        using the SAME shared triviality check (``_template_output_mask_nontrivial``) so
        the two cannot diverge. The no-mask / provable tile-boundary cases are untouched.
        """

        def _flatten(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _flatten(s.region_ops)
                if s.else_ops:
                    yield from _flatten(s.else_ops)

        allops = list(_flatten(self.graph.ops))
        dots = [s for s in allops if s.op == "tt.dot"]
        if not dots:
            return  # not a matmul / FA kernel — the mask-dropping templates never fire

        # Classify FA vs matmul exactly as lower() routes it: >= 2 dots with both an exp
        # and a max between them (online-softmax). FA is exempt here (it routes to a
        # honoring path, or the head_dim=128 template self-refuses at emission).
        def _is_exp(op):
            return op in ("math.exp", "math.exp2", "tt.exp")

        is_fa = len(dots) >= 2 and any(_is_exp(s.op) for s in allops) and any("max" in (s.op or "") for s in allops)
        if is_fa:
            return

        # A specialized one-iteration dot can have literal boundary masks
        # (e.g. M == 1). The complete budgeted masked-dot graph is lowered by
        # the existing generic staged-fill path, which replays those masks;
        # it must not be refused by a guard for mask-dropping templates.
        # _detect_simple_dot still rejects its masked operand loads, and
        # _requires_matmul_template uses this same predicate to exclude it.
        if self._masked_dot_generic_eligible():
            return

        if self._template_output_mask_nontrivial(is_fa=False):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "matmul with a non-tile-boundary output store mask is not supported: "
                "the simdgroup / K-loop / strided / matmul+softmax templates compute the "
                "FULL output tile and gate writes only on the tile boundary (M/N), "
                "silently DROPPING any tighter store mask (e.g. tl.store(c, acc, "
                "mask=rm < BOUND) with BOUND < M, or a value mask such as acc > 0) -> the "
                "masked-off elements would be clobbered with finite values, and the "
                "generic per-element matmul fallback is itself wrong at 16/32/64. "
                "Refusing to emit silently-wrong output. Use a full-tile store (the "
                "tile-boundary mask (rm < M) & (rn < N) is honored automatically by the "
                "template boundary), or apply the partial-output mask in a separate "
                "elementwise kernel.",
                op_name="tt.dot",
            )

    def _detect_quantized_dot(self):
        """True iff the (single) tt.dot's B operand is an INTEGER weight dequantized
        to float — an ``arith.sitofp`` / ``arith.uitofp`` anywhere in B's value chain
        (e.g. ``(w_i8.to(float) - zero) * scale``). The simdgroup matmul templates
        load B directly as float/half from its pointer, so a quantized (int) weight
        can only be mis-compiled or read raw (silently wrong); the caller refuses.
        Fail-safe: over-detection refuses, never silent-wrong.
        """
        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if getattr(s, "region_ops", None):
                    _collect(s.region_ops)
                if getattr(s, "else_ops", None):
                    _collect(s.else_ops)

        _collect(self.graph.ops)
        dots = [s for s in op_by_id.values() if s.op == "tt.dot"]
        if len(dots) != 1 or len(dots[0].operand_ids) < 2:
            return False
        seen = set()

        def _has_int_to_float(sid, depth=0):
            if sid in seen or depth > 40:
                return False
            seen.add(sid)
            o = op_by_id.get(sid)
            if o is None:
                return False
            if o.op in ("arith.sitofp", "arith.uitofp"):
                return True
            return any(_has_int_to_float(oid, depth + 1) for oid in (o.operand_ids or []))

        return _has_int_to_float(dots[0].operand_ids[1])

    def _acc_init_is_literal_zero(self, acc_id, by_id):
        """POSITIVE proof that a dot accumulator init is a literal zero constant
        (through ``ttg.convert_layout`` only). Anything else — a loaded bias, a
        ``tl.full``, a splat of a reduce, or UNKNOWN — is NOT zero. The bare matmul
        templates emit ``acc0(0)`` and so may proceed only on this proof; the old
        ``not _acc_init_is_bias(...)`` test admitted UNKNOWN and silently dropped a
        ``tt.dot(x, y, splat(tl.sum(...)))`` accumulator (packet 065, P0)."""
        cur = acc_id
        for _ in range(6):
            op = by_id.get(cur)
            if op is None:
                return False
            # Value-preserving shape/layout wrappers only: tl.zeros lowers to a
            # scalar constant behind tt.splat on the simdgroup routes.
            if op.op in ("ttg.convert_layout", "tt.splat", "tt.broadcast", "tt.expand_dims") and op.operand_ids:
                cur = op.operand_ids[0]
                continue
            if op.op != "arith.constant":
                return False
            try:
                return float(op.attrs.get("value")) == 0.0
            except (TypeError, ValueError):
                return False
        return False

    def _detect_simple_dot(self):
        """Detect a simple dot kernel: load→local_alloc→local_load→dot→store.

        RESHAPE-provenance decline (dot-recovery 2B, 2026-09-01): an operand
        staged from ``load(1-D) -> tt.reshape -> local_alloc`` carries no 2-D
        addptr arithmetic, so this route's stride decision can only refuse
        ("operand stride could not be inferred"). Inside the generic envelope
        the generic path stages that chain correctly (probe-verified: all four
        upstream test_dot_multidim rank-2 variants, bf16 + memdesc_trans, err
        0.0) — decline so lower() falls through. Outside the envelope the loud
        refusal remains the authority.

        Returns dict with {M, N, K, ptr_args, dot_ssa} if detected, None otherwise.

        Handles two patterns:
        1. Simple (no scf.for): tile fits in one block, no K-loop needed.
        2. K-loop (scf.for wrapping tt.dot): K > BLOCK_K, accumulate across tiles.
           Returns extra fields: has_k_loop=True, BLOCK_K, BLOCK_M, BLOCK_N,
           and scalar_args for M/N/K runtime values.

        Naming-independent: a kernel whose stride args are literally named
        ``stride_*`` is NOT blanket-rejected (BLOCKER 2). The old code did
        ``if has_strides: return None`` for any "stride"-substring arg name, so a
        canonical Triton-tutorial matmul (standard ``stride_am, stride_ak, ...``
        names) NEVER reached the address-traced ``infer_dot_strides`` /
        ``_matmul_stride_decision`` safety path — it routed to
        ``_lower_dot_via_prebuilt_template``, whose name-match ``stride_map``
        + row-major assumption silently mis-computed a transposed operand
        (x @ w.t() → err 86). Now the matmul lowering is naming-independent: the
        address-traced stride decision (downstream in ``_lower_simple_dot_inline``)
        routes any non-contiguous-inner operand to the correct scalar matmul and
        refuses when un-inferable, regardless of arg naming.

        Still defers to ``_lower_dot_via_prebuilt_template`` (returns None) for the
        forms that path models and the address-traced single-dot decision does NOT:
        a CHAIN-dot (>1 tt.dot, e.g. a fused 2-matmul kernel with a W operand) and
        a 3-D batched-via-strides dot (rank>=3 tt.dot result). Those are recognized
        below before any stride-based routing.
        """
        if self._dot_has_reshape_provenance() and self._dot_generic_eligible(allow_flat=True):
            return None

        scalar_args = [a for a in self.graph.args if not a.is_ptr]

        # A "simple dot" is exactly ONE 2-D tt.dot. A MULTI-dot kernel is NOT this
        # pattern and must NOT be processed here (the old ``has_strides`` early-return
        # happened to shield these stride-named shapes; this restores that without the
        # naming dependency that was BLOCKER 2):
        #   - pure chain-dot ``(A@B)@W`` (no reduce) → the prebuilt strided template.
        #   - multi-dot WITH a reduce → FlashAttention (Q@Kᵀ + softmax + P@V), handled by
        #     the generic / FA path (with its own gates). Returning a partial single-dot
        #     info would mis-tile / wrongly refuse it (e.g. a head_dim 32/64 FA refused
        #     via the simple-dot epilogue gate).
        #   - 3-D batched-via-strides dot → the prebuilt strided template.
        # A SINGLE-dot kernel that carries a reduce epilogue (``acc - sum(acc, axis=1)``)
        # is NOT short-circuited here — it must still reach the epilogue/softmax refusal
        # logic below (no correct simple-dot lowering exists; it must refuse loudly —
        # test_matmul_rowreduce_epilogue_refuses).
        def _all_ops(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all_ops(s.region_ops)
                if s.else_ops:
                    yield from _all_ops(s.else_ops)

        _all = list(_all_ops(self.graph.ops))
        _dots = [s for s in _all if s.op == "tt.dot"]
        if len(_dots) > 1:
            return None  # chain-dot / FA → not a simple dot
        if _dots:
            _dshape = _extract_shape(self._find_op_type_str(_dots[0].id) or _dots[0].type_str or "")
            if _dshape and len(_dshape) >= 3:
                return None  # 3-D batched-via-strides → prebuilt

        # A single-dot matmul with a trailing compute epilogue the fused template
        # didn't claim (looped dot-in-K-loop, etc.) must REFUSE — the inline simdgroup
        # lowering stores the RAW accumulator and would SILENTLY DROP the epilogue
        # (re-audit #6), and routing to the generic lowerer mis-tiles it. See
        # _has_unhandled_matmul_compute_epilogue. (_detect_matmul_epilogue already
        # claimed the non-looped single-dot case the template CAN emit.)
        if self._has_unhandled_matmul_compute_epilogue():
            # Dot-recovery stage 1a (2026-08-30): inside the generic dot path's proven
            # envelope, decline instead of refusing — the generic lowerer computes the
            # epilogue correctly there (probe-verified all tiers at S in {16, 32}).
            if self._dot_generic_eligible():
                return None
            from triton_msl.codegen.generic_lowerer import _MATMUL_EPILOGUE_REFUSE_MSG
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(_MATMUL_EPILOGUE_REFUSE_MSG, op_name="tt.dot")

        # BLOCKER 4: refuse a batched matmul (program_id-derived batch offset on an
        # A/B/C base pointer) for BOTH the K-loop and non-K-loop branches, BEFORE either
        # returns its info dict. Batched MMA is not implemented; the templates drop the
        # batch offset and would compute batch 0's region for every batch.
        self._refuse_batched_matmul_base_offset()

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
        k_extent_const = None
        if scf_for_ssa is not None and len(scf_for_ssa.operand_ids or []) >= 2:
            upper = {s.id: s for s in self.graph.ops}.get(scf_for_ssa.operand_ids[1])
            if upper is not None and upper.op == "arith.constant":
                try:
                    k_extent_const = int(str(upper.attrs.get("value")).split(":")[0].strip())
                except (TypeError, ValueError):
                    k_extent_const = None

        # Reduction extent traced STRUCTURALLY (any arg name) rather than by matching
        # the name ``K`` -- renaming it must not silently drop the K-loop (issue #4.1).
        k_extent_arg = self._scf_upper_bound_arg(scf_for_ssa)

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
                    "Refusing.",
                    op_name="tt.dot",
                )

            # A non-zero accumulator INIT (a fused bias / tl.full) is NOT seeded into the
            # simdgroup accumulators by the inline template, so it is silently DROPPED
            # (re-audit #11/#12). In the STANDARD K-loop the accumulator is an scf.for
            # ITER-ARG, so its init is the for-op's initial operand (operand_ids[3:],
            # after lo/hi/step) — NOT a dot operand in graph.ops. The original #11 guard
            # looked at the dot operand and missed every loop-carried accumulator. Check
            # the loop inits: refuse if any is not a zero constant (a non-zero constant,
            # or a non-constant load/broadcast bias).
            by_id_all = {s.id: s for s in self.graph.ops}

            # P0 (packet 065): the POSITIVE literal-zero proof applies to the iter-arg
            # that IS the dot's accumulator (its region block-arg feeds the dot's third
            # operand); other loop-carried values (advanced pointers) keep the old
            # negative "not a loaded bias" test so they are not over-refused. If the
            # accumulator cannot be mapped to an iter-arg, refuse — the template
            # accumulates across iterations and cannot replay an unmapped init.
            _inits = list(scf_for_ssa.operand_ids[3:]) if len(scf_for_ssa.operand_ids) > 3 else []
            _block_args = list((scf_for_ssa.attrs or {}).get("block_arg_ids") or [])
            _acc_src = dot_in_loop.operand_ids[2] if len(dot_in_loop.operand_ids or []) > 2 else None
            _body_by_id = {o.id: o for o in (scf_for_ssa.region_ops or [])}
            # Layout passthroughs ONLY (packet 079): a dtype cast between the iter-arg
            # and the dot is a per-iteration transform the float accumulator never
            # replays (GPU err 0.11) — it must leave the accumulator unmapped -> refuse.
            for _ in range(4):
                _o = _body_by_id.get(_acc_src)
                if _o is not None and _o.op == "ttg.convert_layout" and _o.operand_ids:
                    _acc_src = _o.operand_ids[0]
                    continue
                break
            # block_arg_ids[0] is the INDUCTION VARIABLE (see _lowerer_control's
            # `env[block_arg_ids[0]] = loop_var`); iter-args are block_arg_ids[1:],
            # aligned with the inits at operand_ids[3:].
            _acc_idx = (_block_args.index(_acc_src) - 1) if _acc_src in _block_args else None
            if _acc_idx is None or _acc_idx < 0 or _acc_idx >= len(_inits):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "K-loop matmul: the dot accumulator could not be mapped to a "
                    "loop-carried iter-arg, so its init cannot be proven zero. Refusing "
                    "rather than let the inline simdgroup template drop it.",
                    op_name="tt.dot",
                )
            for _i, _init_id in enumerate(_inits):
                _bad = (
                    not self._acc_init_is_literal_zero(_init_id, by_id_all)
                    if _i == _acc_idx
                    else self._acc_init_is_bias(_init_id, by_id_all)
                )
                if _bad:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "K-loop matmul with a non-zero loop-carried accumulator init "
                        "(fused bias / tl.full) is not supported by the inline simdgroup "
                        "template (the init is silently dropped). Refusing. Add the bias "
                        "as a separate kernel after the matmul.",
                        op_name="tt.dot",
                    )

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
            ptr_args = self._resolve_dot_ptr_roles(dot_in_loop, all_ptr_args) or all_ptr_args
            if len(ptr_args) < 3:
                return None

            # Try to find M, N, K scalar args by name
            scalar_arg_map = {a.name: a for a in scalar_args}

            # P0 (packet 079): every value path the K-loop template replays must be
            # proven per role (operand casts, the store cast, the accumulator carry,
            # loop-carried pointer advances, load masks, grid mapping). A K-loop is
            # never generic-eligible, so an unproven path refuses loudly here.
            _vp = self._dot_template_value_paths()
            _reason, _pid_map = _vp[0], _vp[3]
            _pid_axes = _vp[4] if len(_vp) > 4 else None  # packet 106: per-axis program_id presence
            if _reason:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"K-loop matmul: {_reason}. Refusing (correct-or-refuse).",
                    op_name="tt.dot",
                )

            return {
                # PROVEN grid mapping ("2d" | "1d") from the value-path predicate, replayed
                # verbatim by the pid-tiled templates (no default: a K-loop route without
                # a proven mapping does not reach a template).
                "pid_map": _pid_map,
                "pid_axes": _pid_axes,
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "ptr_args": ptr_args,
                "dot_ssa": dot_in_loop,
                "has_k_loop": True,
                "scalar_args": scalar_arg_map,
                "all_scalar_args": scalar_args,
                # Reduction extent traced structurally from the scf.for upper bound
                # (arg name, or None). Used by the template so K/DIM/depth/... all work
                # and an unresolvable extent refuses instead of reducing one tile.
                "k_extent_arg": k_extent_arg,
                "k_extent_width": next(
                    (a.elem_type for a in self.graph.args if a.name == k_extent_arg),
                    None,
                ),
                "k_extent_const": k_extent_const,
                "k_tail_mode": _vp[5],
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

        if not self._acc_init_is_literal_zero(dot_ssa.operand_ids[2], _by0):
            # P0 (packet 065): POSITIVE literal-zero proof — UNKNOWN is not evidence.
            # Dot-recovery stage 1a (2026-08-30): inside the generic dot path's proven
            # envelope, decline (the generic _lower_dot seeds its accumulator from the
            # init value, probe-verified); outside it, keep refusing.
            if self._dot_generic_eligible():
                return None
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "non-looped matmul with a non-zero accumulator init (fused bias / "
                "tl.full) is silently dropped by the inline simdgroup template. "
                "Refusing. Add the bias as a separate kernel after the matmul.",
                op_name="tt.dot",
            )

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
                    continue  # terminal — fine
                if _op.op in _passthrough:
                    _frontier.append(_op.id)  # follow representation change
                else:
                    # Dot-recovery stage 1a (2026-08-30): inside the generic dot
                    # path's proven envelope, decline instead of refusing — the
                    # generic lowerer computes the epilogue correctly there.
                    if self._dot_generic_eligible():
                        return None
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        f"matmul with a fused '{_op.op}' epilogue on the dot "
                        "result is not supported (only softmax is fused). The "
                        "simple-dot path would silently drop it. Split the "
                        "epilogue into a separate kernel, or apply it after a "
                        "K-loop matmul store."
                    )

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
        has_local_alloc = any(ssa.op == "ttg.local_alloc" for ssa in self.graph.ops)
        if not has_local_alloc:
            return None

        # (#6 re-audit #14) The simple/K-loop simdgroup templates map program_id(0)->M
        # tile and program_id(1)->N tile and IGNORE any further program_id axis. A 3-D
        # grid batched matmul uses program_id(2) (or a pid axis feeding a batch pointer
        # offset) which the template silently DROPS -> wrong batch. Refuse when any
        # program_id with axis >= 2 is present.
        for _pop in self.graph.ops:
            if _pop.op in ("tt.get_program_id", "tt.program_id"):
                try:
                    _ax = int(str(_pop.attrs.get("axis", _pop.attrs.get("dim", 0))))
                except (TypeError, ValueError):
                    _ax = 0
                if _ax >= 2:
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "batched matmul using program_id(2) (a 3-D grid batch axis) is "
                        "not supported by the simdgroup matmul template, which maps only "
                        "program_id(0/1) to the M/N tiles and would drop the batch "
                        "offset. Refusing.",
                        op_name="tt.dot",
                    )

        # (#5 re-audit #14) A masked dot input load signals a PADDED dimension (e.g. K
        # padded to a power of 2 with mask k<K, so the real A row stride is the runtime
        # K, not the BLOCK_K tile width the simple-dot template bakes in). The template
        # ignores the mask and strides by the tile width -> wrong. Refuse a masked
        # simple-dot rather than mis-stride. A complete graph that the generic
        # staged-fill path can replay declines this template instead; unresolvable
        # masks still refuse at the structural rebuild boundary.
        for _lop in self.graph.ops:
            if _lop.op == "tt.load" and len(_lop.operand_ids or []) >= 2:
                if self._masked_dot_generic_eligible():
                    return None
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "matmul with a masked input load (a padded M/N/K dimension) is not "
                    "supported by the inline simdgroup template — it ignores the mask and "
                    "strides by the tile width, not the runtime dimension. Refusing.",
                    op_name="tt.dot",
                )

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
        ptr_args = self._resolve_dot_ptr_roles(dot_ssa, all_ptr_args) or all_ptr_args

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
                return False
            order = list(order)
            n = len(order)
            if n < 2:
                return False
            return order[: n - 2] == list(range(n - 2)) and order[n - 2 :] == [n - 1, n - 2]

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

        def _descriptor_orientation(descriptor_id, seen=None, depth=0):
            from triton_msl.errors import MetalNonRecoverableError

            if depth > 64:
                raise MetalNonRecoverableError(
                    "non-looped matmul descriptor chain exceeds the 64-operation proof bound",
                    op_name="ttg.memdesc_trans",
                )
            seen = set() if seen is None else seen
            if descriptor_id in seen:
                raise MetalNonRecoverableError(
                    "non-looped matmul descriptor chain contains a cycle",
                    op_name="ttg.memdesc_trans",
                )
            seen.add(descriptor_id)
            src = op_by_id.get(descriptor_id)
            if src is None:
                return False
            if src.op == "ttg.local_alloc":
                if src.operand_ids:
                    return _walk_back_to_trans(src.operand_ids[0]) is not None
                return False
            if src.op != "ttg.memdesc_trans" or not src.operand_ids:
                return False
            incoming = _descriptor_orientation(src.operand_ids[0], seen, depth + 1)
            order = src.attrs.get("order")
            if not isinstance(order, (list, tuple)) or sorted(order) != list(range(len(order))):
                raise MetalNonRecoverableError(
                    "non-looped matmul requires a proven operation-owned ttg.memdesc_trans permutation",
                    op_name="ttg.memdesc_trans",
                )
            order = list(order)
            if order == list(range(len(order))):
                return incoming
            inner_swap = list(range(len(order)))
            if len(order) >= 2:
                inner_swap[-2:] = [len(order) - 1, len(order) - 2]
            if order == inner_swap:
                return not incoming
            raise MetalNonRecoverableError(
                f"non-looped matmul does not reproduce non-inner ttg.memdesc_trans permutation {order!r}",
                op_name="ttg.memdesc_trans",
            )

        def _is_trans(operand_id):
            load_op = op_by_id.get(operand_id)
            if not load_op or load_op.op != "ttg.local_load":
                return False
            if not load_op.operand_ids:
                return False
            src = op_by_id.get(load_op.operand_ids[0])
            if not src:
                return False
            return _descriptor_orientation(src.id)

        trans_a = _is_trans(dot_ssa.operand_ids[0])
        trans_b = _is_trans(dot_ssa.operand_ids[1])

        # P0 (packet 079): every value path the bare template replays must be proven
        # per role (operand casts, the store cast, roles, addresses). Refuse — no
        # decline to the generic path: unlike the stage-1a forms above, these have
        # not been probe-verified on the generic lowerer, and the prebuilt template
        # would claim and refuse them at its boundary regardless.
        _vp = self._dot_template_value_paths()
        _reason, _pid_map = _vp[0], _vp[3]
        _pid_axes = _vp[4] if len(_vp) > 4 else None  # packet 106: per-axis program_id presence
        if _reason:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"non-looped matmul: {_reason}. Refusing (correct-or-refuse).",
                op_name="tt.dot",
            )

        return {
            "pid_map": _pid_map,
            "pid_axes": _pid_axes,
            "M": M,
            "N": N,
            "K": K,
            "ptr_args": ptr_args,
            "dot_ssa": dot_ssa,
            "trans_a": trans_a,
            "trans_b": trans_b,
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
        if not a_shape or not b_shape or len(a_shape) != 2 or len(b_shape) != 2:
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
            """max / add for the softmax pattern, via the shared structural classifier."""
            _k = self.classify_reduce_combine(ssa)
            if _k is None:
                return None
            return "max" if _k[0] == "max" else "add" if _k[0] == "sum" else None

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
        # The template emits exactly ONE store; a second store would be silently
        # DROPPED, and _resolve_dot_ptr_roles below traces C from the first store —
        # both require single-store (same guard _detect_matmul_epilogue already has;
        # added 2026-08-30 with the dataflow role fix).
        if sum(1 for _s in self.graph.ops if _s.op == "tt.store") != 1:
            return None

        # This maker seeds the dot with zero and emits exp(dot - row_max) /
        # row_sum. Recognizing the consumers is not a proof of either the
        # accumulator value or the order of noncommutative operands.
        from triton_msl.errors import MetalNonRecoverableError

        by_id = {op.id: op for op in self.graph.ops}
        if len(dot_ssa.operand_ids) != 3 or not self._acc_init_is_literal_zero(dot_ssa.operand_ids[2], by_id):
            raise MetalNonRecoverableError(
                "fused matmul+softmax requires a proven zero dot accumulator; "
                "the template cannot discard a bias or unknown accumulator.",
                op_name="tt.dot",
            )
        if sub.operand_ids != [dot_ssa.id, max_bcast.id] or div.operand_ids != [exp_op.id, sum_bcast.id]:
            raise MetalNonRecoverableError(
                "fused matmul+softmax requires dot minus row maximum and "
                "exponential divided by row sum; reversed operands change the source operation.",
                op_name="tt.dot",
            )

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
        # DATAFLOW-resolved pointer roles (2026-08-30, dot-recovery recon): the old
        # positional pick (`a=ptr[0], b=ptr[1], c=ptr[-1]`) silently READ/WROTE the
        # WRONG BUFFERS for any arg order it didn't anticipate — a kernel with an
        # extra pointer arg after the output stored its entire (numerically perfect)
        # result into that unrelated buffer and left the real output untouched
        # (GPU-verified: M32N64 softmax landed in the unused 4th arg at 5.6e-09).
        # _resolve_dot_ptr_roles traces A/B from the dot operands and C from the
        # tt.store target; a failed/ambiguous trace returns None -> decline, so the
        # kernel falls to the loud #157 catch-all instead of a positional guess.
        _roles = self._resolve_dot_ptr_roles(dot_ssa, ptr_args)
        if _roles is None or len(_roles) < 3:
            return None
        a_ptr, b_ptr, c_ptr = _roles[0], _roles[1], _roles[2]
        # Round 2.1 F1: the fused template stages A/B from their BUFFER types; a value
        # transform between a load and the dot (an fp32 -> fp16 cast, GPU: raw result
        # stored, err 1.8e-3 / 2e-4 vs the IR-ordered oracle on clean ba21e9d) was dropped.
        # Same proof as the bare templates (shared method); inside the generic envelope
        # decline to the op-by-op lowerer, outside it refuse.
        _by_all = {}

        def _collect_all(ops):
            for _s in ops:
                _by_all[_s.id] = _s
                if _s.region_ops:
                    _collect_all(_s.region_ops)
                if _s.else_ops:
                    _collect_all(_s.else_ops)

        _collect_all(self.graph.ops)
        _ab_reason = self._dot_operand_paths_reason(
            dot_ssa, a_ptr, b_ptr, _by_all, allow_masked=False, allow_trans=False
        )[0]
        if _ab_reason:
            if self._dot_generic_eligible():
                return None
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"fused matmul template: {_ab_reason}. Refusing (correct-or-refuse).",
                op_name="tt.dot",
            )

        # ADDRESS-TRACED strides ONLY (mechanism A — same contract as
        # _detect_matmul_epilogue, which already refuses on un-inferable strides).
        # The old positional ``_strides_after`` reader (mechanism B) GUESSED the row/col
        # stride from arg ORDER: ``stride[ptr.index + 1/+2]``. For a canonical
        # ``(a, b, c, M, N, K, stride...)`` signature that landed on the M/N DIM args
        # and silently mis-addressed a transposed/strided operand (the same BLOCKER-2
        # class the matmul path was hardened against). infer_dot_strides() reads the
        # real tt.addptr offset arithmetic, so a transposed-operand fused-softmax matmul
        # is read correctly; when ANY operand stride is un-inferable the tracer returns
        # None and we REFUSE here (return None -> the kernel falls through to the loud
        # #157 catch-all) rather than emit a row-major guess. Mechanism B is DELETED:
        # an instrumented full-suite + audit-repro run (TRIPWIRE_B) hit the positional
        # fallback ZERO times, while a forced positive control proved the tripwire fires
        # when it IS reached.
        descriptors = self._inferred_stride_descriptors()
        if descriptors is None:
            return None
        a_row_s, a_col_s, b_row_s, b_col_s, c_row_s, c_col_s = descriptors

        return {
            "M": M,
            "N": N,
            "K": K,
            "a_ptr": a_ptr.name,
            "b_ptr": b_ptr.name,
            "c_ptr": c_ptr.name,
            "a_elem": a_ptr.elem_type,
            "b_elem": b_ptr.elem_type,
            "c_elem": c_ptr.elem_type,
            "a_row_stride": a_row_s,
            "a_col_stride": a_col_s,
            "b_row_stride": b_row_s,
            "b_col_stride": b_col_s,
            "c_row_stride": c_row_s,
            "c_col_stride": c_col_s,
        }

    # Ops a fused matmul epilogue may apply to the dot result. Pointwise /
    # broadcast / cast / layout only — NO reduce/scan (softmax has its own
    # path; anything else falls through to the #157 refusal). #158.
    _EPILOGUE_ALLOWED = frozenset(
        {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
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
            "math.fma",
            "math.absf",
            # packet 162: no conversions — see _EPI_PASSTHROUGH
            "tt.splat",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
            "ttg.convert_layout",
            "arith.constant",
            "tt.load",
            "tt.clampf",
        }
    )

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
        if not a_shape or not b_shape or len(a_shape) != 2 or len(b_shape) != 2:
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

            if not self._acc_init_is_literal_zero(dot_ssa.operand_ids[2], by_id0):
                cur = acc
                axis = None
                accumulator_wrappers = []
                for _ in range(6):
                    if cur is None:
                        break
                    if cur.op in ("tt.broadcast", "ttg.convert_layout", "tt.reshape"):
                        accumulator_wrappers.append(cur)
                        cur = by_id0.get(cur.operand_ids[0]) if cur.operand_ids else None
                        continue
                    if cur.op == "tt.expand_dims":
                        accumulator_wrappers.append(cur)
                        axis = cur.attrs.get("axis")
                        cur = by_id0.get(cur.operand_ids[0]) if cur.operand_ids else None
                        continue
                    break
                if cur is not None and cur.op == "tt.load" and cur.operand_ids:
                    bptr = self._trace_ptr_source(cur.operand_ids[0], by_id0)
                    if bptr is None:
                        return None
                    acc_bias_ptr = bptr.name
                    # Distinguish the three accumulator shapes by the expand_dims axis the
                    # trace recorded (a 1-D bias is loaded then broadcast via expand_dims; a
                    # full 2-D accumulator has NO expand_dims, so axis stays None):
                    #   axis==0  -> COL bias (per-output-feature, broadcast over rows):
                    #               strip-independent, verified correct -> COMPUTE.
                    #   axis==1  -> ROW bias (per-row, M-length, broadcast over N): the
                    #               simdgroup matmul's per-strip output indexing mis-handles
                    #               it (re-audit #8, even M=32) -> REFUSE.
                    #   axis None -> a FULL 2-D accumulator C = A@B + C: the simdgroup epilogue
                    #               adds only a 1-D row/col bias, not a per-element M×N tile.
                    #               (Was previously mislabeled a 'row bias' -> REFUSE accurately.)
                    from triton_msl.errors import MetalNonRecoverableError

                    if str(axis) in ("0",):
                        # This template emits exactly bias[col], without replaying
                        # the source address, mask or shape transforms. Prove that
                        # exact load before admitting it. A pointer role and an
                        # expand_dims axis alone do not prove unit column stride.
                        expands = [s for s in accumulator_wrappers if s.op == "tt.expand_dims"]
                        col_load_proven = (
                            len(cur.operand_ids) == 1
                            and len(expands) == 1
                            and not any(s.op == "tt.reshape" for s in accumulator_wrappers)
                            and self._flat_contiguous_address(cur.operand_ids[0], bptr, by_id0, N) is not None
                        )
                        if not col_load_proven:
                            if self._dot_generic_eligible():
                                return None
                            raise MetalNonRecoverableError(
                                "fused column-bias matmul requires an unmasked unit-stride "
                                "column load; its source address or value path is not reproduced "
                                "by the template. Refusing (correct-or-refuse).",
                                op_name="tt.dot",
                            )
                        acc_bias_dim = "col"
                    elif str(axis) in ("1",):
                        # Dot-recovery stage 1a (2026-08-30): inside the generic dot
                        # path's proven envelope, decline instead of refusing — the
                        # generic _lower_dot seeds each output element's accumulator
                        # from the per-element init (probe-verified add-rows).
                        if self._dot_generic_eligible():
                            return None
                        raise MetalNonRecoverableError(
                            "fused ROW-bias matmul (per-row bias as the dot accumulator) "
                            "is mis-computed and not reliably lowerable. Refusing rather "
                            "than mis-compute. Use a column bias, or add the row bias in a "
                            "separate kernel after the matmul.",
                            op_name="tt.dot",
                        )
                    else:
                        # Same envelope fall-through (probe-verified add-matrix).
                        if self._dot_generic_eligible():
                            return None
                        raise MetalNonRecoverableError(
                            "fused 2-D accumulator matmul (C = A@B + C, a full M×N tile as "
                            "the dot's 3rd operand) is not supported: the simdgroup matmul "
                            "epilogue adds only a 1-D row/col bias, not a per-element "
                            "accumulator. Add C in a separate elementwise kernel after the "
                            "matmul (out = A @ B; out += C).",
                            op_name="tt.dot",
                        )
                else:
                    return None  # non-zero, non-bias accumulator: unsupported

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
                continue  # leaf: the matmul result (seeded later)
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
                return None  # unsupported epilogue op
            epilogue_ids.add(op.id)
            if op.op not in _EPI_PASSTHROUGH and op.op not in ("arith.constant", "tt.load"):
                has_compute = True
            frontier.extend(op.operand_ids or [])
        if not reached_dot or not has_compute:
            return None  # store doesn't derive from dot, or
            # pure matmul (no real epilogue)

        # A bias / extra input enters via a tt.load inside the epilogue. Collect
        # its pointer arg (we index it per element in the template).
        bias_ptr = None
        for eid in epilogue_ids:
            if by_id[eid].op == "tt.load":
                bptr = self._trace_ptr_source(by_id[eid].operand_ids[0], by_id) if by_id[eid].operand_ids else None
                if bptr is not None:
                    if bias_ptr is not None and bias_ptr.name != bptr.name:
                        return None  # >1 extra input not supported yet
                    bias_ptr = bptr

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 3:
            return None
        # DATAFLOW-resolved pointer roles (2026-08-30, dot-recovery recon): the old
        # positional pick (`a=ptr[0], b=ptr[1], c=ptr[-1]`) silently READ/WROTE the
        # WRONG BUFFERS for any arg order it didn't anticipate — a kernel with an
        # extra pointer arg after the output stored its entire (numerically perfect)
        # result into that unrelated buffer and left the real output untouched
        # (GPU-verified: M32N64 softmax landed in the unused 4th arg at 5.6e-09).
        # _resolve_dot_ptr_roles traces A/B from the dot operands and C from the
        # tt.store target; a failed/ambiguous trace returns None -> decline, so the
        # kernel falls to the loud #157 catch-all instead of a positional guess.
        _roles = self._resolve_dot_ptr_roles(dot_ssa, ptr_args)
        if _roles is None or len(_roles) < 3:
            return None
        a_ptr, b_ptr, c_ptr = _roles[0], _roles[1], _roles[2]
        # Round 2.1 F1: the fused template stages A/B from their BUFFER types; a value
        # transform between a load and the dot (an fp32 -> fp16 cast, GPU: raw result
        # stored, err 1.8e-3 / 2e-4 vs the IR-ordered oracle on clean ba21e9d) was dropped.
        # Same proof as the bare templates (shared method); inside the generic envelope
        # decline to the op-by-op lowerer, outside it refuse.
        _by_all = {}

        def _collect_all(ops):
            for _s in ops:
                _by_all[_s.id] = _s
                if _s.region_ops:
                    _collect_all(_s.region_ops)
                if _s.else_ops:
                    _collect_all(_s.else_ops)

        _collect_all(self.graph.ops)
        _ab_reason = self._dot_operand_paths_reason(
            dot_ssa, a_ptr, b_ptr, _by_all, allow_masked=False, allow_trans=False
        )[0]
        if _ab_reason:
            if self._dot_generic_eligible():
                return None
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"fused matmul template: {_ab_reason}. Refusing (correct-or-refuse).",
                op_name="tt.dot",
            )

        # ADDRESS-TRACED strides (not the old positional index+1/+2 read, which
        # landed on the M/N dim args for a `(a,b,c,M,N,K,sam,sak,...)` signature
        # and silently mis-addressed a transposed/strided operand — BLOCKER 2).
        # infer_dot_strides() reads the actual tt.addptr offset arithmetic, so a
        # transposed B (col stride = a runtime arg) is read correctly. When any
        # operand stride is un-inferable, bail to None so the kernel refuses via
        # the #157 catch-all rather than emit a row-major guess.
        descriptors = self._inferred_stride_descriptors()
        if descriptors is None:
            return None
        a_row_s, a_col_s, b_row_s, b_col_s, c_row_s, c_col_s = descriptors

        # Topologically-ordered epilogue ops (graph order is topo).
        epilogue_ops = [ssa for ssa in self.graph.ops if ssa.id in epilogue_ids]

        return {
            "M": M,
            "N": N,
            "K": K,
            "a_ptr": a_ptr.name,
            "b_ptr": b_ptr.name,
            "c_ptr": c_ptr.name,
            "a_elem": a_ptr.elem_type,
            "b_elem": b_ptr.elem_type,
            "c_elem": c_ptr.elem_type,
            "a_row_stride": a_row_s,
            "a_col_stride": a_col_s,
            "b_row_stride": b_row_s,
            "b_col_stride": b_col_s,
            "c_row_stride": c_row_s,
            "c_col_stride": c_col_s,
            # epilogue-specific:
            "epilogue_ops": epilogue_ops,
            "dot_id": dot_ssa.id,
            "bias_ptr": bias_ptr.name if bias_ptr else None,
            "acc_bias_ptr": acc_bias_ptr,  # bias fused into the dot's init
            "acc_bias_dim": acc_bias_dim,  # "col" (N,) or "row" (M,1)
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
        if any(
            s.op in ("scf.for", "scf.while", "scf.if", "tt.get_num_programs", "tt.get_program_id")
            for s in self.graph.ops
        ):
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
            _k = self.classify_reduce_combine(r)
            return _k is not None and _k[0] == "sum"

        # Skip dtype/layout passthroughs (e.g. the i32->i64 arith.extsi that
        # an int sum promotes to, or a ttg.convert_layout / reshape) between
        # the store value and the reduce chain.
        _PASS = ("arith.extsi", "arith.extui", "arith.trunci", "ttg.convert_layout", "tt.reshape", "arith.bitcast")

        def _skip_pass(op):
            seen = 0
            while op is not None and op.op in _PASS and op.operand_ids and seen < 8:
                op = op_by_id.get(op.operand_ids[0])
                seen += 1
            return op

        red_axes = []  # sequential reduce axes (in application order)
        cur = _skip_pass(op_by_id.get(store.operand_ids[1]))
        while cur is not None and cur.op == "tt.reduce":
            if not _is_sum_reduce(cur):
                return None
            red_axes.append(cur.attrs.get("axis", 0))
            cur = _skip_pass(op_by_id.get(cur.operand_ids[0]) if cur.operand_ids else None)
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
        if not self._is_contiguous_range_gather(store.operand_ids[0], op_by_id):
            return None

        # 4) Shapes + permutation.
        in_shape = _extract_shape(self._find_op_type_str(trans.operand_ids[0]) or "")
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
                (in_strides[surviving[k]], in_shape[surviving[k]], out_strides[k]) for k in range(len(surviving))
            ],
        }

    def _parse_trans_order(self, trans, rank):
        """Return the operation-owned, walker-validated transpose order."""
        o = trans.attrs.get("order")
        if isinstance(o, (list, tuple)) and len(o) == rank and sorted(o) == list(range(rank)):
            return list(o)
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
                off = op_by_id.get(off.operand_ids[0]) if off.operand_ids else None
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
        has_dot = False
        reduce_count = 0
        for ssa in self.graph.ops:
            if ssa.op == "scf.for":
                has_scf_for = True
            elif ssa.op == "tt.get_num_programs":
                has_num_programs = True
            elif ssa.op == "tt.dot":
                has_dot = True
            elif ssa.op == "tt.reduce":
                reduce_count += 1
        # A pure 3-D reduce never contains a tt.dot. Rejecting dot-kernels stops a
        # FlashAttention kernel from being MIS-DETECTED as a 3-D reduce (it has 2-D
        # softmax reduces + dots) and routed to / refused by the 3-D template instead of
        # reaching its own head_dim/dtype guards — which also unblocks adding correct
        # multi-program (program_id) support to the 3-D template below.
        if has_scf_for or has_num_programs or has_dot or reduce_count > 1:
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
                            if _o.op in ("tt.reshape", "ttg.convert_layout") and _o.operand_ids:
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
                                "s * tl.sum(a, axis=2)) or in a separate kernel."
                            )
                        axis = ssa.attrs.get("axis", 0)
                        # Detect argmin/argmax: 2 operands (values, indices)
                        is_argminmax = len(ssa.operand_ids) >= 2
                        # Reuse the returned-DAG proof from the ordinary reduction
                        # path, not comparison presence (which also mistook integer
                        # argmin for argmax). Unknown combines must never become sum.
                        from triton_msl.errors import MetalNonRecoverableError

                        if is_argminmax:
                            value_op = _obid[ssa.operand_ids[0]]
                            kind = "integer" if value_op.elem_type.startswith(("i", "u")) else "float"
                            proof = self._classify_exact_argminmax_recurrence(ssa, kind)
                            if proof is None or proof[1]:
                                raise MetalNonRecoverableError(
                                    "3-D argmin/argmax requires the exact signed-value comparison "
                                    "and smaller-index tie-break; this tuple combine is not supported.",
                                    op_name="tt.reduce",
                                )
                            combine_op = "argmax" if proof[0] else "argmin"
                        else:
                            _k = self.classify_reduce_combine(ssa)
                            if _k is None or _k[0] not in {"sum", "max", "min"} or not _k[1]:
                                raise MetalNonRecoverableError(
                                    "3-D reduce template supports only proven sum or signed/float "
                                    "max/min combines; refusing rather than substitute a sum "
                                    "for a different reduction.",
                                    op_name="tt.reduce",
                                )
                            combine_op = _k[0]
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
                                "32-bit integer, or reduce in 2-D/1-D.",
                                op_name="tt.reduce",
                            )
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
        # The direct maker replaces the WHOLE kernel, not only the stored value.
        # Admit only the closed, side-effect-free vocabulary used to construct
        # the canonical flip plus its one load/store. In particular a retained
        # atomic/assert/call/barrier must not disappear merely because its result
        # is dead. Value/address proofs below remain independently required.
        effect_vocabulary = {
            "arith.constant",
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
            "arith.shli",
            "arith.shrsi",
            "arith.shrui",
            "arith.cmpi",
            "arith.select",
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "arith.index_cast",
            "arith.index_castui",
            "arith.bitcast",
            "tt.addptr",
            "tt.bitcast",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.load",
            "tt.make_range",
            "tt.reduce",
            "tt.reshape",
            "tt.return",
            "tt.splat",
            "tt.store",
            "ttg.convert_layout",
        }

        def _all(ops):
            for op in ops:
                yield op
                if op.region_ops:
                    yield from _all(op.region_ops)
                if op.else_ops:
                    yield from _all(op.else_ops)

        if any(op.op not in effect_vocabulary for op in _all(self.graph.ops)):
            return None

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
        if load_ssa is None or store_ssa is None:
            return None
        # The direct maker has no load/store predicate, ``other`` value, or
        # volatile access. Do not erase those source semantics.
        if (
            len(load_ssa.operand_ids or []) != 1
            or len(store_ssa.operand_ids or []) != 2
            or bool((load_ssa.attrs or {}).get("isVolatile", False))
        ):
            return None
        # Must have at least one xori reduce. Reshapes appear in pairs when the
        # flip dim has size > 2; when size == 2, Triton skips them.
        if len(reduce_ops) < 1:
            return None
        if len(reshape_ops) not in (0, 2):
            return None

        # Every reducer must YIELD exactly xor(block_arg0, block_arg1). Merely
        # containing an xori is insufficient: ``yield (a ^ b) + 1`` is a
        # different reduction even though its region contains xor.
        for red in reduce_ops:
            block_args = list((red.attrs or {}).get("block_arg_ids") or [])
            returned = list((red.attrs or {}).get("return_ids") or [])
            body = list(red.region_ops or [])
            if len(block_args) != 2 or len(returned) != 1 or len(body) != 1:
                return None
            combine = body[0]
            if (
                combine.op != "arith.xori"
                or combine.id != returned[0]
                or len(combine.operand_ids or []) != 2
                or set(combine.operand_ids) != set(block_args)
            ):
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
            rs1_in_shape = _extract_shape(self._find_op_type_str(rs1.operand_ids[0])) if rs1.operand_ids else None
            rs1_out_shape = _extract_shape(rs1.type_str)
            rs2_in_shape = _extract_shape(self._find_op_type_str(rs2.operand_ids[0])) if rs2.operand_ids else None
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
                expected = in_shape[:d] + (2,) * steps + in_shape[d + 1 :]
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
            red_in_shape = _extract_shape(self._find_op_type_str(red.operand_ids[0])) if red.operand_ids else None
            if tuple(red_in_shape or ()) != in_shape:
                return None
            if in_shape[red_axis] != 2:
                return None
            flip_dim = red_axis
            num_steps = 1

        # Prove the complete value chain the direct template replaces:
        #   current -> reduce(xor) -> expand(axis) -> broadcast -> current^broadcast
        # for every split bit, followed only by the final reshape (when present)
        # and store. This rejects both a valid reducer with extra output
        # arithmetic and an unrelated/dead xori used to fool op-name sniffing.
        by_id = {o.id: o for o in self.graph.ops}
        current = load_ssa.id
        first = reshape_ops[0] if reshape_ops else reduce_ops[0]
        if first.operand_ids != [current]:
            cast = by_id.get(first.operand_ids[0]) if first.operand_ids else None
            if cast is None or cast.op != "tt.bitcast" or cast.operand_ids != [current]:
                return None
            current = cast.id
        if reshape_ops:
            if reshape_ops[0].operand_ids != [current]:
                return None
            current = reshape_ops[0].id
        expected_axes = list(range(flip_dim, flip_dim + num_steps))
        for red, expected_axis in zip(reduce_ops, expected_axes):
            if red.operand_ids != [current] or red.attrs.get("axis") != expected_axis:
                return None
            expands = [
                o
                for o in self.graph.ops
                if o.op == "tt.expand_dims" and o.operand_ids == [red.id] and o.attrs.get("axis") == expected_axis
            ]
            if len(expands) != 1:
                return None
            broadcasts = [o for o in self.graph.ops if o.op == "tt.broadcast" and o.operand_ids == [expands[0].id]]
            if len(broadcasts) != 1:
                return None
            combines = [
                o
                for o in self.graph.ops
                if o.op == "arith.xori"
                and len(o.operand_ids or []) == 2
                and set(o.operand_ids) == {current, broadcasts[0].id}
            ]
            if len(combines) != 1:
                return None
            current = combines[0].id
        if reshape_ops:
            if reshape_ops[1].operand_ids != [current]:
                return None
            current = reshape_ops[1].id
        if store_ssa.operand_ids[1] != current:
            cast = by_id.get(store_ssa.operand_ids[1])
            if cast is None or cast.op != "tt.bitcast" or cast.operand_ids != [current]:
                return None
            current = cast.id

        # The template reads/writes dense row-major ``_e`` and therefore may
        # claim only the matching source address expression. Load and store
        # must share the exact index SSA, and that index must be the sum of the
        # three 0-based ranges with row-major coefficients (N*K, K, 1).
        load_ptr = by_id.get(load_ssa.operand_ids[0]) if load_ssa.operand_ids else None
        store_ptr = by_id.get(store_ssa.operand_ids[0]) if store_ssa.operand_ids else None
        if (
            load_ptr is None
            or store_ptr is None
            or load_ptr.op != "tt.addptr"
            or store_ptr.op != "tt.addptr"
            or len(load_ptr.operand_ids or []) != 2
            or len(store_ptr.operand_ids or []) != 2
            or load_ptr.operand_ids[1] != store_ptr.operand_ids[1]
        ):
            return None
        arg_ids = {a.id for a in self.graph.args if a.is_ptr}
        load_base = by_id.get(load_ptr.operand_ids[0])
        store_base = by_id.get(store_ptr.operand_ids[0])
        if (
            load_base is None
            or store_base is None
            or load_base.op != "tt.splat"
            or store_base.op != "tt.splat"
            or len(load_base.operand_ids or []) != 1
            or len(store_base.operand_ids or []) != 1
            or load_base.operand_ids[0] not in arg_ids
            or store_base.operand_ids[0] not in arg_ids
        ):
            return None

        import re

        def _literal(vid):
            op = by_id.get(vid)
            if op is None or op.op != "arith.constant":
                return None
            text = str((op.attrs or {}).get("value", ""))
            match = re.fullmatch(r"(?:dense<)?\s*(-?\d+)\s*>?", text)
            return int(match.group(1)) if match else None

        def _affine(vid, depth=0):
            if depth > 32:
                return None
            op = by_id.get(vid)
            if op is None:
                return None
            if op.op in {"tt.broadcast", "ttg.convert_layout"} and len(op.operand_ids or []) == 1:
                return _affine(op.operand_ids[0], depth + 1)
            if op.op == "tt.expand_dims" and len(op.operand_ids or []) == 1:
                terms = _affine(op.operand_ids[0], depth + 1)
                try:
                    inserted = int(op.attrs.get("axis"))
                except (TypeError, ValueError):
                    return None
                if terms is None:
                    return None
                # A range begins as rank-1 coordinate axis zero. Each
                # expand_dims inserts a singleton coordinate before/after it;
                # carry that ownership through later broadcasts rather than
                # inferring an axis from equal extents or sorted coefficients.
                return [(axis + (inserted <= axis), start, end, coef) for axis, start, end, coef in terms]
            if op.op == "tt.make_range":
                try:
                    start, end = int(op.attrs.get("start")), int(op.attrs.get("end"))
                except (TypeError, ValueError):
                    return None
                return [(0, start, end, 1)]
            if op.op == "arith.addi" and len(op.operand_ids or []) == 2:
                left, right = (_affine(x, depth + 1) for x in op.operand_ids)
                return None if left is None or right is None else left + right
            if op.op == "arith.muli" and len(op.operand_ids or []) == 2:
                for value, other in ((op.operand_ids[0], op.operand_ids[1]), (op.operand_ids[1], op.operand_ids[0])):
                    factor = _literal(value)
                    terms = _affine(other, depth + 1)
                    if factor is not None and terms is not None:
                        return [(axis, start, end, coef * factor) for axis, start, end, coef in terms]
            return None

        terms = _affine(load_ptr.operand_ids[1])
        required = sorted(((0, 0, M, N * K), (1, 0, N, K), (2, 0, K, 1)))
        if terms is None or sorted(terms) != required:
            return None

        # Pointer roles and the source element type come from DATAFLOW, never
        # declaration order (packet 043 sibling audit, 2026-08-30).
        roles = self._resolve_load_store_ptr_roles(load_ssa, store_ssa)
        if roles is None:
            return None
        input_arg, output_arg = roles
        x_ptr = input_arg.name
        z_ptr = output_arg.name
        elem_type = input_arg.elem_type

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

    def _row_stride_is_constexpr(self):
        """True if the kernel's per-row stride (the program_id multiplier in the
        load/store address) is a COMPILE-TIME CONSTANT rather than a runtime scalar.

        The softmax/layernorm templates compute ``row_start = pid * n_arg`` using the
        sole runtime scalar as the row length. That is correct only when the kernel's
        own per-row stride IS that scalar. A kernel ``softmax(x, out, M, N: constexpr)``
        strides by the CONSTEXPR N (baked as an arith.constant) while its sole runtime
        scalar M is the row COUNT — so n_arg=M is the wrong span (re-audit #12: rows
        summed to M, not 1). When the stride is constant the row length is constexpr and
        the template cannot represent it; the caller declines to the generic lowering.
        """
        by_id = {s.id: s for s in self.graph.ops}
        pid_ids = {s.id for s in self.graph.ops if "program_id" in s.op}
        if not pid_ids:
            return False
        for o in self.graph.ops:
            if o.op in ("arith.muli", "arith.mul") and o.operand_ids and len(o.operand_ids) >= 2:
                if any(x in pid_ids for x in o.operand_ids):
                    # the operand that is NOT the program_id is the row stride
                    for x in o.operand_ids:
                        if x in pid_ids:
                            continue
                        so = by_id.get(x)
                        if so is not None and so.op == "arith.constant":
                            return True
        return False

    def _stored_value_roots_at(self, store_ssa, root_pred, what):
        """Packet 162 (160 F-h): the value a template's single store writes must BE the
        computation the template replays, through layout ops only — plus at most ONE
        ``arith.truncf`` whose result element type equals the store pointer's element type
        (exactly what the template's own store cast replays). Anything else between the
        computation and the store (a round trip, an integer cast, arithmetic) is not replayed:
        the softmax template stored the raw division for ``(y*4).to(int32).to(f32)`` (GPU err
        2.24) and for ``y.to(f16).to(f32)`` (err 1.4e-4). Returns the root op or None."""
        by_id = {o.id: o for o in self.graph.ops}
        args = {a.id: a for a in self.graph.args}
        if store_ssa is None or len(store_ssa.operand_ids or []) < 2:
            return None
        ptr_op = by_id.get(store_ssa.operand_ids[0])
        # the store pointer's element type: walk addptr/splat/broadcast to the pointer arg
        cur, seen = store_ssa.operand_ids[0], set()
        ptr_elem = None
        while cur not in seen:
            seen.add(cur)
            if cur in args:
                ptr_elem = str(args[cur].elem_type)
                break
            o = by_id.get(cur)
            if o is None or not (o.operand_ids or []):
                break
            cur = o.operand_ids[0]
        cur, seen, casts = store_ssa.operand_ids[1], set(), 0
        while cur not in seen:
            seen.add(cur)
            o = by_id.get(cur)
            if o is None:
                return None
            if o.op in ("ttg.convert_layout", "tt.reshape") and o.operand_ids:
                cur = o.operand_ids[0]
                continue
            if o.op == "arith.truncf" and o.operand_ids and casts == 0:
                t = o.type_str or ""
                el = t.rsplit("x", 1)[-1].split(",")[0].split(">")[0].strip() if "x" in t else t.strip()
                norm = {"f16": ("f16", "fp16", "half"), "bf16": ("bf16", "bfloat16"), "f32": ("f32", "fp32", "float")}
                ok = ptr_elem is not None and any(el in v and ptr_elem in v for v in norm.values())
                if not ok:
                    return None
                casts += 1
                cur = o.operand_ids[0]
                continue
            return o if root_pred(o, by_id) else None
        return None

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
            if ssa.op in ("scf.for", "scf.while", "scf.if", "tt.get_num_programs"):
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

        if load_ssa is None or store_ssa is None or len(reduce_ops) != 2 or not (has_exp and has_subf and has_divf):
            return None

        # Reduces must be max then sum (in IR order). The combine op lives in
        # the reduce body region; _get_reduce_combine_info inspects that.
        red_ops = [self._get_reduce_combine_info(r)[0] for r in reduce_ops]
        if red_ops != ["max", "sum"]:
            return None

        # Identify ptr args (input vs output) from ACTUAL DATAFLOW (traced from the
        # load/store), not declaration order, and require the max-reduce to operate
        # directly on the loaded tensor (else a softmax(x*scale) silently drops the
        # scale). reduce_ops[0] is the max reduce (the [max, sum] order verified above).
        io = self._norm_input_output_args(load_ssa, store_ssa, reduce_ops[0])
        if io is None:
            return None
        input_arg, output_arg = io

        # packet 162 (160 F-h): the STORED value must be the division itself (exp / sum),
        # through layout ops and at most one output cast the template's store replays.
        def _is_softmax_div(o, by_id):
            if o.op != "arith.divf" or len(o.operand_ids or []) != 2:
                return False
            a, b = (by_id.get(x) for x in o.operand_ids)
            return (
                a is not None
                and a.op in ("math.exp", "math.exp2")
                and b is not None
                and b.op in ("tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.splat")
            )

        if self._stored_value_roots_at(store_ssa, _is_softmax_div, "softmax") is None:
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
        # The sole scalar is the row length ONLY if the per-row stride is that runtime
        # value, not a constexpr. Otherwise it is the row COUNT and the template would
        # normalize the wrong span (re-audit #12) — decline to the generic lowering.
        if self._row_stride_is_constexpr():
            return None

        # Block size = the tensor's dim (we look at make_range end values).
        block_size = self.graph.block_size
        if block_size > 1024:
            # 1024 cap on threadgroup; row larger than 1024 needs a larger
            # TG buffer + more iterations. Skip for safety.
            return None

        info = {
            "input_arg": input_arg,
            "output_arg": output_arg,
            "n_arg": n_arg,
            "block_size": block_size,
        }
        if not normalization_template_matches(self.graph, info, "softmax"):
            return None
        return info

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
            if ssa.op in ("scf.for", "scf.while", "scf.if", "tt.get_num_programs"):
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
        if (
            load_ssa is None
            or store_ssa is None
            or len(reduce_ops) != 2
            or not (has_rsqrt and has_subf and has_mulf and has_addf)
        ):
            return None
        red_ops = [self._get_reduce_combine_info(r)[0] for r in reduce_ops]
        if red_ops != ["sum", "sum"]:
            return None

        # Don't fire if softmax pattern matches (max/exp/divf disqualifies
        # layer norm because softmax's _detect_* would fire instead).
        for ssa in self.graph.ops:
            if ssa.op in ("math.exp", "math.exp2"):
                return None

        # Input/output ptrs from ACTUAL DATAFLOW (not declaration order), and the mean
        # reduce (reduce_ops[0], first sum) must operate directly on the loaded tensor
        # (else a norm(x*scale) silently drops the scale). Mirrors _detect_softmax.
        io = self._norm_input_output_args(load_ssa, store_ssa, reduce_ops[0])
        if io is None:
            return None
        input_arg, output_arg = io

        # packet 162 (160 F-h): the STORED value must be the normalisation itself —
        # mulf(centred x, broadcast(rsqrt(var + eps))) — through layout ops and at most one
        # output cast the template's store replays.
        def _peel_b(x, by_id):
            seen = set()
            while x not in seen:
                seen.add(x)
                o = by_id.get(x)
                if (
                    o is None
                    or o.op not in ("tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.splat")
                    or not o.operand_ids
                ):
                    return o
                x = o.operand_ids[0]
            return None

        def _is_ln_norm(o, by_id):
            if o.op != "arith.mulf" or len(o.operand_ids or []) != 2:
                return False
            sides = [_peel_b(x, by_id) for x in o.operand_ids]
            has_rsqrt = any(x is not None and x.op in ("math.rsqrt",) for x in sides)
            has_centred = any(x is not None and x.op in ("arith.subf", "arith.select") for x in sides)
            return has_rsqrt and has_centred

        if self._stored_value_roots_at(store_ssa, _is_ln_norm, "layer norm") is None:
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
        # Sole scalar must be the row length (per-row stride), not the row COUNT — else
        # the template normalizes the wrong span (re-audit #12). Decline to the generic.
        if self._row_stride_is_constexpr():
            return None

        block_size = self.graph.block_size
        if block_size > 1024:
            return None

        # --- eps: the fast template must use the KERNEL's epsilon from rsqrt(var + eps),
        #     NOT a hardcoded default. A wrong eps is a silent-wrong (worst on low-
        #     variance rows). Extract the float constant; if it can't be resolved, refuse
        #     to the generic lowering (which applies the real eps). ---
        _obid = {o.id: o for o in self.graph.ops}

        def _const_float(oid, depth=0):
            if depth > 8:
                return None
            o = _obid.get(oid)
            if o is None:
                return None
            if o.op == "arith.constant":
                try:
                    return float(str(o.attrs.get("value", "")).split(":")[0].strip())
                except (TypeError, ValueError):
                    return None
            if (
                o.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout", "arith.extf", "arith.truncf")
                and o.operand_ids
            ):
                return _const_float(o.operand_ids[0], depth + 1)
            return None

        eps_val = None
        for ssa in self.graph.ops:
            if ssa.op in ("math.rsqrt", "math.sqrt") and ssa.operand_ids:
                inner = _obid.get(ssa.operand_ids[0])
                if inner is not None and inner.op == "arith.addf":
                    for oid in inner.operand_ids:
                        c = _const_float(oid)
                        if c is not None:
                            eps_val = c
                break
        if eps_val is None:
            return None

        info = {
            "input_arg": input_arg,
            "output_arg": output_arg,
            "n_arg": n_arg,
            "block_size": block_size,
            "eps": eps_val,
        }
        if not normalization_template_matches(self.graph, info, "layer_norm"):
            return None
        return info

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
            elif ssa.op in ("scf.for", "scf.while", "scf.if", "tt.reduce", "tt.scan", "tt.dot"):
                return None  # too complex for this template

        if load_ssa is None or store_ssa is None or trans_ssa is None or len(reshapes) != 2:
            return None

        # Extract shapes. Load is 2-D, first reshape goes to 4-D, trans
        # produces a 4-D permuted view, second reshape flattens to 1-D.
        load_shape = _extract_shape(load_ssa.type_str)
        if not load_shape or len(load_shape) != 2:
            return None
        M, N = load_shape

        # First reshape: (M, N) → (M, m, n, 2) where m*n*2 == N
        first_reshape_shape = _extract_shape(reshapes[0].type_str)
        if (
            not first_reshape_shape
            or len(first_reshape_shape) != 4
            or first_reshape_shape[0] != M
            or first_reshape_shape[3] != 2
        ):
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
        if not trans_shape or len(trans_shape) != 4 or trans_shape != (m, n, 2, M):
            return None

        # Second reshape: must flatten to (M*N,)
        second_reshape_shape = _extract_shape(reshapes[1].type_str)
        if not second_reshape_shape or len(second_reshape_shape) != 1 or second_reshape_shape[0] != M * N:
            return None

        # Resolve input/output from the actual load/store address chains. The old
        # positional pick swapped both buffers for output-first signatures.
        roles = self._resolve_load_store_ptr_roles(load_ssa, store_ssa)
        if roles is None:
            return None
        input_ptr, output_ptr = roles
        input_arg = input_ptr.name
        output_arg = output_ptr.name
        elem_type = input_ptr.elem_type

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
            elif ssa.op in ("scf.for", "scf.while", "scf.if", "tt.reduce", "tt.scan", "tt.dot"):
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
                if op is None or op.op not in _VALUE_PRESERVING or not op.operand_ids:
                    break
                cur = op.operand_ids[0]
            return False

        if not trans_ssa.operand_ids or not _traces_to(trans_ssa.operand_ids[0], load_ssa.id):
            return None
        if len(store_ssa.operand_ids) < 2 or not _traces_to(store_ssa.operand_ids[1], trans_ssa.id):
            return None
        # The transpose operates on its INPUT's shape (the N-D tensor).
        src_shape = _extract_shape(self._find_op_type_str(trans_ssa.operand_ids[0]))
        if not src_shape or len(src_shape) < 3:
            return None
        order = self._parse_trans_order(trans_ssa, len(src_shape))
        if order is None or sorted(order) != list(range(len(src_shape))):
            return None
        roles = self._resolve_load_store_ptr_roles(load_ssa, store_ssa)
        if roles is None:
            return None
        input_ptr, output_ptr = roles
        total = 1
        for s in src_shape:
            total *= s
        if any(op.op in self._PID_OPS for op in self.graph.ops):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "N-D transpose template does not reproduce program-id addressing; "
                "refusing rather than transpose the first block in every program.",
                op_name="tt.trans",
            )
        return {
            "input_arg": input_ptr.name,
            "output_arg": output_ptr.name,
            "elem_type": input_ptr.elem_type,
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
        if any(
            op in {"scf.for", "scf.while", "scf.if", "tt.get_num_programs"} for op in (s.op for s in self.graph.ops)
        ):
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
                f"slice the top k, or k == N.",
                op_name="tt.reduce",
            )

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
            is_minmax = any(("max" in op or "min" in op or op == "arith.cmpf") for op in body_ops)
            if not (is_xor or is_minmax):
                return None

        # Identify pointer args (X=input, Z=output) and stride scalars
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        # Preserve the prescan side effect used for driver copy-back metadata, but
        # resolve the template's input/output roles from this detector's exact load
        # and store. `_output_arg_ids` alone cannot identify the input when unrelated
        # pointer args are present.
        self._store_ptr_ids = set()
        self._prescan_stores()
        roles = self._resolve_load_store_ptr_roles(load_ssa, store_ssa)
        if roles is None:
            return None
        input_arg, output_arg = roles
        x_ptr_name = input_arg.name
        z_ptr_name = output_arg.name
        # Compute comparisons in the INPUT domain. A mixed-dtype output-first
        # signature previously picked the output dtype from ptr_args[0], narrowing
        # values BEFORE sorting (28/32 wrong stores in the i32->i8 discriminator).
        elem_type = input_arg.elem_type

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

        # Identify the row-stride scalars. They appear as a factor of a make_range
        # (the row offset) in an arith.muli. The canonical sort signature carries ONLY
        # the strides as runtime scalars (X, stride_xm, Z, stride_zm), so a POSITIONAL
        # pick would silently mis-assign if a NON-stride scalar (e.g. a runtime row count)
        # were present. Guard it structurally: verify EVERY runtime scalar arg is actually
        # a row-stride (a make_range coefficient); if any is not, the positional pick is
        # ambiguous -> REFUSE rather than guess (correct-or-refuse). Constexpr strides
        # (no runtime scalars) stay the safe None case (template uses the baked N).
        def _flatten_ops(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _flatten_ops(s.region_ops)
                if s.else_ops:
                    yield from _flatten_ops(s.else_ops)

        _all_ops = list(_flatten_ops(self.graph.ops))
        _obid = {s.id: s for s in _all_ops}

        def _is_row_stride(arg):
            for o in _all_ops:
                if o.op == "arith.muli" and arg.id in (o.operand_ids or []):
                    for other in o.operand_ids:
                        if other != arg.id and self._trace_to_make_range(other, _all_ops, _obid) is not None:
                            return True
            return False

        stride_xm_name = None
        stride_zm_name = None
        if len(scalar_args) >= 1:
            if not all(_is_row_stride(a) for a in scalar_args):
                return None  # a non-stride runtime scalar is present -> can't positionally pick
            if len(scalar_args) > 2:
                return None  # more stride-like scalars than the 2 rows (xm, zm) -> ambiguous
            stride_xm_name = scalar_args[0].name
            stride_zm_name = scalar_args[1].name if len(scalar_args) >= 2 else scalar_args[0].name
        else:
            # Constexpr strides (no runtime stride scalars): the row stride is baked
            # into the IR as an integer constant multiplying the M-range
            # (row_off = expand(make_range(0,M)) * C). Extract C separately for the
            # load (x) and store (z) pointer chains — a padded-row layout (C != N)
            # must use the REAL C, not an assumed N. If either constant cannot be
            # proven unambiguously, bail to the generic path (correct-or-refuse).
            # (2026-08-29, trifast #6b fallout: this branch became reachable when
            # _prescan_stores learned to resolve broadcast/expand store chains; the
            # template previously interpolated Python `None` into the MSL here.)
            def _baked_row_stride(root_id):
                consts = set()
                seen = set()
                stack = [root_id]
                while stack:
                    cur = stack.pop()
                    if cur is None or cur in seen:
                        continue
                    seen.add(cur)
                    o = _obid.get(cur)
                    if o is None:
                        continue
                    if o.op == "arith.muli" and len(o.operand_ids or []) == 2:
                        for mr_side, c_side in ((0, 1), (1, 0)):
                            # Peel shape wrappers locally (the shared tracer does not
                            # follow expand_dims, and widening it would loosen every
                            # other detection that relies on it).
                            _mr_root = o.operand_ids[mr_side]
                            for _ in range(6):
                                _w = _obid.get(_mr_root)
                                if (
                                    _w is not None
                                    and _w.op in ("tt.expand_dims", "tt.broadcast", "ttg.convert_layout")
                                    and _w.operand_ids
                                ):
                                    _mr_root = _w.operand_ids[0]
                                    continue
                                break
                            mr_id = self._trace_to_make_range(_mr_root, _all_ops, _obid)
                            if mr_id is None:
                                continue
                            mr_op = _obid.get(mr_id)
                            try:
                                extent = int(mr_op.attrs.get("end")) - int(mr_op.attrs.get("start", 0))
                            except (TypeError, ValueError):
                                continue
                            if extent != M:
                                continue
                            # Other side must trace to an integer constant
                            c_id = o.operand_ids[c_side]
                            for _ in range(6):
                                c_op = _obid.get(c_id)
                                if c_op is None:
                                    break
                                if c_op.op == "arith.constant":
                                    v = c_op.attrs.get("value")
                                    if isinstance(v, int):
                                        consts.add(v)
                                    break
                                if (
                                    c_op.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "arith.extsi")
                                    and c_op.operand_ids
                                ):
                                    c_id = c_op.operand_ids[0]
                                    continue
                                break
                    for opd in o.operand_ids or []:
                        stack.append(opd)
                if len(consts) == 1:
                    return consts.pop()
                return None  # none found, or ambiguous (e.g. M == N with an unfolded col muli)

            _sx = _baked_row_stride(load_ssa.operand_ids[0]) if load_ssa.operand_ids else None
            _sz = _baked_row_stride(store_ssa.operand_ids[0]) if store_ssa.operand_ids else None
            if _sx is None or _sz is None or _sx < N or _sz < K_out:
                return None  # can't prove the baked strides -> generic path handles it
            stride_xm_name = f"{_sx}"
            stride_zm_name = f"{_sz}"

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
                        return _struct.unpack("f", _struct.pack("I", v & 0xFFFFFFFF))[0]
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
        for body_op in ssa.region_ops or []:
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
        for body_op in ssa.region_ops or []:
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
