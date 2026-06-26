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

import re

from triton_msl.codegen.mlir_walker import SSAValue, _extract_shape
from triton_msl.codegen.msl_emitter import _msl_compute_type
from triton_msl.codegen.msl_types import triton_type_to_msl

from triton_msl.codegen._lowerer_helpers import _mlir_to_triton_dtype


class _ReduceScanMixin:
    """``tt.reduce`` and ``tt.scan`` lowering for ``GenericLowerer``."""

    def _mept_reduce_fold(self, arr_name: str, n: int, combine_op: str,
                          msl_type: str) -> str:
        """Phase 4e: fold a per-thread register array to a scalar partial.

        ``arr_name[0..n-1]`` are this thread's elements (single-pass MEPT,
        so they are the thread's full share of the tile). Folding with the
        reduce's combine op produces one partial per thread; the existing
        cross-thread SIMD/threadgroup reduce then finishes the job. This is
        the array-form analogue of the multipass wrap-loop accumulator.
        """
        combine = {
            "sum": lambda a, b: f"{a} + {b}",
            "max": lambda a, b: f"max({a}, {b})",
            "min": lambda a, b: f"min({a}, {b})",
            "xor": lambda a, b: f"{a} ^ {b}",
            "and": lambda a, b: f"{a} & {b}",
            "or":  lambda a, b: f"{a} | {b}",
        }.get(combine_op, lambda a, b: f"{a} + {b}")
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
            self.kb.raw_line(
                f"    {fold_var} = {combine(fold_var, _read(i))};")
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
        _SAFE_REPLAY_OPS = frozenset({
            "tt.make_range", "tt.splat", "tt.broadcast", "tt.expand_dims",
            "tt.addptr", "arith.constant", "arith.extf", "arith.truncf",
            "arith.sitofp", "arith.fptosi",
        })
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
                    for sub_oid in (ext_op.operand_ids or []):
                        if sub_oid not in _visited_ext:
                            _ext_worklist.append(sub_oid)
                else:
                    # Tensor external dep that's not a safe replay op → refuse
                    return None
            # Non-tensor external ops (scalars) stay in scope — no action needed

        # Build the ordered list of external safe deps in graph order
        _all_id_order = {oid: idx for idx, oid in enumerate(_all_by_id.keys())}
        external_safe_deps = [
            _all_by_id[oid]
            for oid in sorted(external_safe_dep_ids,
                              key=lambda i: _all_id_order.get(i, 0))
        ]

        identity, combine_expr = self._reduce_identity_combine(combine_op, msl_type)
        acc = self._next_var("inloop_acc")
        self.kb.raw_line(f"        {msl_type} {acc} = {identity};")
        self.kb.raw_line(
            f"        for (uint _loop_e = lid; _loop_e < {total}u; "
            f"_loop_e += {self.kb.block_size}u) {{")

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
            f"            {{ {msl_type} acc = {acc}; "
            f"{msl_type} val = ({msl_type}){val}; {acc} = {combine_expr}; }}")
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
        for b in (region_ops or []):
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
            return ("ulong", "u64") if dtype in ("u64", "ui64") else ("long", "i64")
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
            return None   # tuple combines (argmax/argmin/Welford) use the multi-value path
        a, b = ba[0], ba[1]
        bargs = {a, b}
        top = ops[-1]                          # yielded op (the reduce.return terminator is parsed out)
        nm = top.op or ""
        # (1) a DIRECT binary op of exactly the two block args.
        if set(top.operand_ids or []) == bargs:
            _DIRECT = {
                "arith.addf": ("sum", True), "arith.addi": ("sum", True),
                "arith.maxnumf": ("max", True), "arith.maxf": ("max", True),
                "arith.minnumf": ("min", True), "arith.minf": ("min", True),
                "arith.maxsi": ("max", True), "arith.minsi": ("min", True),
                "arith.maxui": ("max", False), "arith.minui": ("min", False),
                "arith.xori": ("xor", True), "arith.andi": ("and", True),
                "arith.ori": ("or", True),
            }
            # arith.maximumf/minimumf (NaN-PROPAGATING) and everything else fall through
            # to None -> refuse (would lower to NaN-quiet fmax / would be silently wrong).
            return _DIRECT.get(nm)
        # (2) a cmp+select max/min: yielded select picking between the two block args,
        #     whose cmp compares exactly those block args. Sign + direction come from the
        #     cmp predicate AND the select operand mapping (where(a>b,b,a) is a MIN).
        if nm == "arith.select" and len(top.operand_ids or []) >= 3:
            return self._classify_cmp_select(ops, top, a, b)
        return None

    @staticmethod
    def _classify_cmp_select(ops, sel, a, b):
        """(kind, signed) for a cmp+select max/min, or None. The select must pick between the
        two block args, and its condition must be a cmp of exactly those block args."""
        t, f = sel.operand_ids[1], sel.operand_ids[2]
        if {t, f} != {a, b}:
            return None   # picks value-vs-mask / a derived value -> not a plain max/min
        cond = sel.operand_ids[0]
        cmp = None
        for o in ops:
            if o.id == cond and o.op in ("arith.cmpf", "arith.cmpi"):
                cmp = o
                break
        if cmp is None or len(cmp.operand_ids or []) < 2:
            return None
        lhs, rhs = cmp.operand_ids[0], cmp.operand_ids[1]
        if {lhs, rhs} != {a, b}:
            return None
        pred = (cmp.attrs.get("predicate_name", "") or "")
        is_gt = "gt" in pred or "ge" in pred
        is_lt = "lt" in pred or "le" in pred
        signed = not (cmp.op == "arith.cmpi" and pred.startswith("u"))
        if t == lhs:
            kind = "max" if is_gt else "min" if is_lt else None
        elif t == rhs:
            kind = "min" if is_gt else "max" if is_lt else None
        else:
            kind = None
        return (kind, signed) if kind else None

    def _get_reduce_combine_info(self, ssa):
        """Extract combine op and identity from a tt.reduce's body region.

        Returns (combine_op, identity_literal) where combine_op is one of
        'sum', 'max', 'min' and identity_literal is the MSL identity value.
        A custom combine this multipass path cannot honor REFUSES (never defaults
        to sum) — the prime directive.
        """
        _res = self.classify_reduce_combine(ssa)
        # The multipass path supports the canonical kinds (sum/max/min/and/or/xor — the
        # same set the single-pass path does); a custom / NaN-propagating combine refuses.
        if _res is None:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "tl.reduce on the multipass reduction path supports only canonical "
                "sum / max / min / and / or / xor; a custom or NaN-propagating combine is "
                "refused rather than silently mis-computed.")
        combine_op = _res[0]
        identities = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY",
                      "and": "(~0)", "or": "0", "xor": "0"}
        return combine_op, identities.get(combine_op, "0.0f")

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
        sum identity is 0 (0.0f for float). LONG_MIN/LONG_MAX/ULONG_MAX are
        provided by metal_stdlib (same as the multipass-accumulator path).
        """
        is_float = msl_type == "float"
        if combine_op == "sum":
            identity = "0.0f" if is_float else "0"
            combine_expr = "acc + val"
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
        elif combine_op == "xor":
            identity = "0"
            combine_expr = "acc ^ val"
        elif combine_op == "and":
            identity = "~0"            # all-ones: AND identity
            combine_expr = "acc & val"
        elif combine_op == "or":
            identity = "0"             # OR identity
            combine_expr = "acc | val"
        else:
            identity = "0.0f" if is_float else "0"
            combine_expr = "acc + val"
        return identity, combine_expr


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

            # Separate scalar ops (hoist before loop) from tensor ops (inside loop)
            scalar_ops = [op for op in phase_ops if self._is_scalar_op(op)]
            tensor_ops = [op for op in phase_ops if not self._is_scalar_op(op)]

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
            replay_scalar = [op for op in replay_ops if self._is_scalar_op(op)]
            replay_tensor = [op for op in replay_ops if not self._is_scalar_op(op)]
            for ssa in replay_scalar:
                if ssa.id not in lowered_scalar_ids:
                    self._lower_op(ssa)
                    lowered_scalar_ids.add(ssa.id)

            # If this phase precedes a reduce, declare the accumulator
            acc_var = None
            if next_reduce:
                combine_op, identity = self._get_reduce_combine_info(next_reduce)
                acc_var = f"_local_acc_{self._shared_counter}"
                # Determine accumulator type from the reduce input. The operand
                # may not be in env_types after multipass replay/reordering (a
                # reshape between load and reduce can drop the type), so fall
                # back to the reduce op's own element type (reliable from IR).
                reduce_input_dtype = self.env_types.get(
                    next_reduce.operand_ids[0]) if next_reduce.operand_ids else None
                if reduce_input_dtype is None:
                    _et = getattr(next_reduce, "elem_type", None)
                    reduce_input_dtype = (
                        _mlir_to_triton_dtype(_et) if _et else "fp32")
                is_int_reduce = not (
                    reduce_input_dtype.startswith("fp") or reduce_input_dtype.startswith("bf")
                )
                is_i64_reduce = reduce_input_dtype in ("i64", "u64", "ui64")
                is_u64_reduce = reduce_input_dtype in ("u64", "ui64")
                # unsigned 32-bit max/min compares UNSIGNED; the final cross-thread simd
                # reduce casts to reduce_ty in threadgroup_reduce.
                _unsigned = (combine_op in ("max", "min")
                             and self._reduce_is_unsigned_minmax(next_reduce.region_ops))
                acc_msl_type, _ = self._reduce_acc_msl_type(reduce_input_dtype, _unsigned)
                # bitwise (and/or/xor) identities are width-independent: and = all-ones,
                # or/xor = 0.
                _bitwise_ident = {"and": "(~0)", "or": "0", "xor": "0"}
                if is_i64_reduce:
                    # 64-bit identities (LONG_MIN/MAX); ulong min identity is 0.
                    i64_identities = ({"sum": "0", "max": "0", "min": "ULONG_MAX"}
                                      if is_u64_reduce
                                      else {"sum": "0", "max": "LONG_MIN", "min": "LONG_MAX"})
                    i64_identities.update(_bitwise_ident)
                    identity = i64_identities.get(combine_op, "0")
                elif is_int_reduce:
                    if _unsigned:
                        identity = "0" if combine_op == "max" else "UINT_MAX"
                    else:
                        identity = {"sum": "0", "max": "INT_MIN", "min": "INT_MAX",
                                    **_bitwise_ident}.get(combine_op, "0")
                # else float: identity preserved from above
                self.kb.raw_line(f"    {acc_msl_type} {acc_var} = {identity};")

            # Open the per-element loop
            self._needs_wrapping = True
            self.kb.raw_line(
                f"    for (uint _loop_e = lid; _loop_e < {total}u; "
                f"_loop_e += {block_size}u) {{"
            )

            # Re-emit tensor dependency ops from earlier phases
            for ssa in replay_tensor:
                self._lower_op(ssa)

            # Emit this phase's tensor ops inside the loop
            for ssa in tensor_ops:
                self._lower_op(ssa)

            # Accumulate into the local variable for the next reduce
            if next_reduce and acc_var:
                reduce_input_id = next_reduce.operand_ids[0]
                input_var = self._lookup(reduce_input_id)
                # Cast input to accumulator type to avoid Metal ambiguity
                cast_input = f"({acc_msl_type}){input_var}"
                if combine_op == "sum":
                    self.kb.raw_line(f"        {acc_var} += {cast_input};")
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

            # Close the loop
            self.kb.raw_line(f"    }}")
            self._needs_wrapping = False

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
                _sharers = [r for r in _all_reduce_ops(self.graph.ops)
                            if r.operand_ids and r.operand_ids[0] == reduce_input_id]
                if len(_sharers) > 1:
                    from triton_msl.errors import MetalNonRecoverableError
                    raise MetalNonRecoverableError(
                        "two or more reductions of the SAME loaded tile (e.g. tl.sum(x) and "
                        "tl.max(x) of the same x) are not supported by the multipass reduce "
                        "lowering: the second silently reduces over the first's accumulator. "
                        "Load the tile separately for each reduction.", op_name="tt.reduce")
                self.env[reduce_input_id] = acc_var
                _acc_dtype = {"long": "i64", "ulong": "u64",
                              "int": "i32", "float": "fp32"}.get(acc_msl_type)
                if _acc_dtype is not None:
                    self.env_types[reduce_input_id] = _acc_dtype

            # Add this phase's ops to the preceding ops for future phases
            all_preceding_ops.extend(phase_ops)


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

        # Detect multi-value reduce (argmax/argmin): 2+ inputs, 2+ results
        if (len(ssa.operand_ids) >= 2 and ssa.result_ids
                and len(ssa.result_ids) >= 2):
            self._lower_reduce_multi_value(ssa)
            return

        input_var = self._lookup(ssa.operand_ids[0])
        axis = ssa.attrs.get("axis", 0)

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
                "custom combine is refused rather than silently mis-computed.")
        combine_op, _signed = _res
        has_unsigned_minmax = combine_op in ("max", "min") and not _signed

        # Determine type from input operand. After a multipass wrap-loop the
        # operand is rebound to a freshly-typed accumulator whose env_type is
        # set; but if the operand is missing from env_types (e.g. a reshape
        # between load and reduce dropped it), fall back to the reduce op's own
        # element type so 64-bit reduces still route to the i64 tree.
        input_dtype = self.env_types.get(ssa.operand_ids[0])
        if input_dtype is None:
            _et = getattr(ssa, "elem_type", None)
            input_dtype = _mlir_to_triton_dtype(_et) if _et else "fp32"
        is_int_reduce = not (
            input_dtype.startswith("fp") or input_dtype.startswith("bf")
        )
        is_i64 = input_dtype in ("i64", "u64", "ui64")
        is_u64 = input_dtype in ("u64", "ui64")
        msl_type, shared_dtype = self._reduce_acc_msl_type(
            input_dtype, has_unsigned_minmax and combine_op in ("max", "min"))

        # Check if this is a 2D axis-specific reduction
        input_shape = self.env_shapes.get(ssa.operand_ids[0])
        if not input_shape:
            input_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))

        # Phase 4e: MEPT array operand. Fold this thread's register array to
        # a scalar partial with the combine op, then run the existing 1-D
        # cross-thread reduce on that partial. Only the 1-D full-reduce case
        # is handled today — MEPT activation never routes multi-dim / axis
        # reduces here (the prescan requires a 1-D reduce for MEPT-reduce
        # eligibility), so folding to 1-D and skipping the multi-dim
        # dispatch is correct.
        mept_arr = (self.env_array.get(ssa.operand_ids[0])
                    if getattr(self, "mept_enabled", False) else None)
        if mept_arr is not None:
            arr_name, n_arr, _arr_ty = mept_arr
            input_var = self._mept_reduce_fold(
                arr_name, n_arr, combine_op, msl_type)
            input_shape = None  # already folded to one element per thread

        if input_shape and len(input_shape) == 3:
            self._lower_reduce_3d(ssa, input_var, axis, combine_op,
                                  msl_type, shared_dtype, input_shape)
            return

        # N-D axis-specific reduce (n >= 4). Used by e.g. tl.sort's bitonic
        # decomposition, which reshapes to (2,)*n and reduces along a specific
        # axis per compare-and-swap step.
        if input_shape and len(input_shape) >= 4:
            self._lower_reduce_nd(ssa, input_var, axis, combine_op,
                                  msl_type, shared_dtype, input_shape)
            return

        if self._is_2d and input_shape and len(input_shape) >= 2:
            # For triton_per_* kernels with shape (1, N) or (XBLOCK, R_BLOCK)
            # where dim_0 == 1 and axis == 1, this is really a 1D reduction
            # along the reduction dimension.  Use the efficient SIMD path,
            # not the slow sequential shared memory path.
            if input_shape[0] == 1 and axis == 1:
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
                        f"(silent-wrong). Use BLOCK <= num_threads.", op_name="tt.reduce")
                self._lower_reduce_2d(ssa, input_var, axis, combine_op,
                                      msl_type, shared_dtype, input_shape)
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
        if (mept_arr is None
                and self._control_flow_depth > 0
                and input_shape is not None
                and len(input_shape) == 1
                and input_shape[0] > self.kb.block_size):
            # Stage A: try to cover the whole tile by folding each thread's
            # strided share before the cross-thread reduce.
            _acc = self._cover_inloop_reduce(
                ssa, combine_op, msl_type, input_shape[0])
            if _acc is not None:
                input_var = _acc
                input_shape = None   # folded to one scalar per thread
            else:
                from triton_msl.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    f"Refusing in-loop reduction: a tile of {input_shape[0]} "
                    f"elements exceeds the {self.kb.block_size}-thread threadgroup "
                    f"and is not register-array-covered, so a cross-lane reduce "
                    f"here would sum only the first {self.kb.block_size} elements "
                    f"(silent-wrong). Use the default register-array path "
                    f"(TRITON_MSL_MEPT unset) or BLOCK <= num_threads.")

        # Cast bool (i1) to int before reduction — MSL SIMD intrinsics reject bool
        if input_dtype == "i1" or (isinstance(input_var, str) and input_var in ("true", "false", "1", "0")):
            cast_var = self._next_var("bool_to_int")
            self.kb.raw_line(f"    int {cast_var} = (int){input_var};")
            input_var = cast_var

        if is_i64:
            self._lower_reduce_1d_i64(ssa, input_var, combine_op,
                                      msl_type, shared_dtype)
            return

        # The dispatch may run more threads (block_size = num_threads) than the
        # logical reduce length N when the input has no make_range to pin block_size
        # to N — e.g. a reduce over a tt.splat/full. Lanes >= N then hold a value that
        # must NOT contribute (re-audit #6: sum over a splat of N summed num_threads
        # copies — 768 vs 24). Mask the tail to the combine identity. No-op when
        # N == block_size (the common make_range case, where N lanes == all lanes).
        _rn = input_shape[0] if (input_shape and len(input_shape) == 1) else None
        if _rn is not None and _rn < self.kb.block_size:
            from triton_msl.errors import MetalNonRecoverableError
            _is_float = msl_type in ("float", "half", "bfloat")
            if combine_op == "sum":
                _ident = "0"
            elif combine_op == "max" and _is_float:
                _ident = "-INFINITY"
            elif combine_op == "min" and _is_float:
                _ident = "INFINITY"
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
                    op_name="tt.reduce")
            _masked = self._next_var("rmask")
            self.kb.raw_line(
                f"    {msl_type} {_masked} = (lid < {_rn}u) ? "
                f"({msl_type}){input_var} : ({msl_type}){_ident};")
            input_var = _masked

        # 1D full reduction (original behavior)
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        n_simd_groups = (self.kb.block_size + 31) // 32
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=n_simd_groups)

        result_var = self._next_var("reduced")
        # Reduce IN the element type — integer reductions in float lost precision >2^24.
        self.kb.threadgroup_reduce(combine_op, input_var, shared_name, result_var,
                                   reduce_ty=msl_type)

        # Narrow-type masking: when reducing in wider type but output is narrow,
        # apply modular arithmetic (i1 sum = XOR, i8 sum = mod 256, etc.)
        out_elem = ssa.elem_type
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


    def _lower_reduce_1d_i64(self, ssa, input_var, combine_op,
                             msl_type, shared_dtype):
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
            raise MetalNonRecoverableError(
                f"i64 reduce: unsupported combine op {combine_op}")
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


    def _lower_reduce_multi_value(self, ssa: SSAValue):
        """Multi-value reduce: argmax/argmin (2-value) or Welford (3-value).

        For 2 inputs: argmax/argmin (value + index) via SIMD shuffle + shared memory.
        For 3 inputs: Welford online variance (mean + m2 + weight) via SIMD shuffle + shared memory.
        """
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
        _ishape = self.env_shapes.get(ssa.operand_ids[0]) or _extract_shape(
            self._find_op_type_str(ssa.operand_ids[0]))
        if (_ishape is not None
                and len(_ishape) == 1 and _ishape[0] > self.kb.block_size):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Multi-value reduce (argmax/argmin/Welford) over a 1-D tile of "
                f"{_ishape[0]} elements exceeds the {self.kb.block_size}-thread "
                f"threadgroup; the multipass path can't aggregate a tuple reduce, so "
                f"this would be silently wrong / uncompilable. Refusing. "
                f"Use BLOCK <= num_threads.")

        # Dispatch Welford (3-value) vs argmax/argmin (2-value)
        if len(ssa.operand_ids) >= 3 and ssa.result_ids and len(ssa.result_ids) >= 3:
            # Route to Welford ONLY if the body is the online-variance recurrence (it divides
            # by the running count — arith.divf). A custom 3-tuple combine (triple-max,
            # triple-sum, ...) has no division and would be SILENTLY computed as the Welford
            # mean/m2/weight math (Triton-lens re-audit 2026-06-25: a triple-max returned
            # Welford output). Refuse it rather than mis-compute, mirroring the 2-value
            # comparison-presence guard below.
            _has_div = any((bop.op or "").startswith("arith.div")
                           for bop in (ssa.region_ops or []))
            if not _has_div:
                from triton_msl.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "multi-value tl.reduce with a 3+ element tuple body that is not the "
                    "Welford online-variance recurrence (it does not divide by a running "
                    "count) is not supported — it would be silently computed as variance. "
                    "Refusing. Only tl.var/tl.std (Welford) 3-tuples and value+index "
                    "argmax/argmin 2-tuples are handled.", op_name="tt.reduce")
            self._lower_reduce_welford(ssa)
            return

        # A 2-value reduce is handled ONLY as argmax/argmin (value+index). Verify the body
        # has a COMPARISON (cmpf/cmpi) — argmin/argmax compare values. A custom 2-value
        # combine (e.g. (x+y, i+j)) has no comparison and would be SILENTLY mis-computed
        # by the argminmax path (reduce-probe finding). Refuse it.
        if (len(ssa.operand_ids) >= 2
                and not any(bop.op in ("arith.cmpf", "arith.cmpi")
                            for bop in (ssa.region_ops or []))):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "multi-value tl.reduce whose body is not argmax/argmin (no comparison) "
                "is not supported — only value+index argmax/argmin 2-tuples are handled. "
                "Refusing.", op_name="tt.reduce")

        # Check for 2D argmin/argmax
        if len(ssa.operand_ids) >= 2:
            input_shape = self.env_shapes.get(ssa.operand_ids[0])
            if not input_shape:
                input_shape = _extract_shape(
                    self._find_op_type_str(ssa.operand_ids[0]))
            if input_shape and len(input_shape) == 2 and self._is_2d:
                axis = ssa.attrs.get("axis", 0)
                # Skip 2D dispatch when first dim is 1 and axis is 1
                # (really a 1D reduction, same logic as _lower_reduce)
                if not (input_shape[0] == 1 and axis == 1):
                    self._lower_reduce_2d_argminmax(ssa, axis, input_shape)
                    return

        self._lower_reduce_argminmax(ssa)


    def _lower_reduce_welford(self, ssa: SSAValue):
        """Welford online variance reduction: (mean, m2, weight) via SIMD shuffle + shared memory."""
        mean_var = self._lookup(ssa.operand_ids[0])
        m2_var = self._lookup(ssa.operand_ids[1])
        weight_var = self._lookup(ssa.operand_ids[2])

        n_simd_groups = (self.kb.block_size + 31) // 32

        # Shared memory for 3 values
        sh_mean = f"shared_{self._shared_counter}"; self._shared_counter += 1
        sh_m2 = f"shared_{self._shared_counter}"; self._shared_counter += 1
        sh_w = f"shared_{self._shared_counter}"; self._shared_counter += 1
        self.kb.declare_threadgroup_array(sh_mean, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_m2, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_w, dtype="fp32", size=n_simd_groups)

        wm = self._next_var("wm")   # working mean
        wv = self._next_var("wv")    # working m2
        ww = self._next_var("ww")    # working weight
        rm = self._next_var("rm")    # result mean
        rv = self._next_var("rv")    # result m2
        rw = self._next_var("rw")    # result weight

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
        self.kb.raw_line(f"        float _ratio = (_nw == 0.0f) ? 0.0f : _ow / _nw;")
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
            self.kb.raw_line(f"            float _ratio = (_nw == 0.0f) ? 0.0f : _ow / _nw;")
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


    def _lower_reduce_argminmax(self, ssa: SSAValue):
        """Argmax/argmin: value + index via SIMD shuffle + shared memory."""
        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Determine value type
        val_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
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
                "float first.", op_name="tt.reduce")
        msl_val_type = "int" if is_int else "float"
        val_shared_dtype = "i32" if is_int else "fp32"

        # Detect argmax vs argmin from body ops
        is_max = self._detect_reduce_direction(ssa)
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
        self.kb.raw_line(f"    {msl_val_type} {mv} = {val_var};")
        self.kb.raw_line(f"    int {mi} = {idx_var};")

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
        self.kb.raw_line(f"        bool _take = ((lid + _d) < {_bs}u) && "
                         f"((_ov {cmp_op} {mv}) || (_ov == {mv} && _oi < {mi}));")
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

        # Store both results in env
        self.env[ssa.id] = result_val
        self.env_types[ssa.id] = val_shared_dtype
        if ssa.result_ids and len(ssa.result_ids) >= 2:
            self.env[ssa.result_ids[0]] = result_val
            self.env_types[ssa.result_ids[0]] = val_shared_dtype
            self.env[ssa.result_ids[1]] = result_idx
            self.env_types[ssa.result_ids[1]] = "i32"


    def _lower_reduce_2d_argminmax(self, ssa, axis, input_shape):
        """Lower 2D argmin/argmax: find min/max value and index along axis.

        For axis=1 on (M, N): each row finds min/max among N values → (M,) values + indices.
        For axis=0 on (M, N): each column finds min/max among M values → (N,) values + indices.
        """
        M, N = input_shape[0], input_shape[1]
        total = M * N

        # (#6 reduce-probe) axis=0 argmin/argmax on a SQUARE (M==N) tile mis-broadcasts
        # the index (column-0's index to every column); rectangular tiles are correct.
        if axis == 0 and M == N:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "2-D argmin/argmax along axis=0 on a square (M==N) tile mis-broadcasts "
                "the index. Refusing. Use a non-square tile or reduce along axis=1.",
                op_name="tt.reduce")
        # (#5 reduce-probe) axis=1, BOTH value AND index consumed: the value is moved to
        # simple layout (convert_layout) before its store but the index is not, so the
        # index store broadcasts row-0's index. Index-only / value-only is correct.
        if axis == 1 and ssa.result_ids and len(ssa.result_ids) >= 2:
            _used_ids = set()
            for o in self.graph.ops:
                _used_ids.update(o.operand_ids or [])
            if ssa.result_ids[0] in _used_ids and ssa.result_ids[1] in _used_ids:
                from triton_msl.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "2-D argmin/argmax along axis=1 with BOTH the value and the index "
                    "consumed mis-stores the index (broadcasts row-0's index). Refusing. "
                    "Use the value or the index alone, or separate kernels.",
                    op_name="tt.reduce")

        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        val_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (val_dtype.startswith("fp") or val_dtype.startswith("bf"))
        is_i64 = val_dtype in ("i64", "u64", "ui64")
        # 64-bit int values compare at full width (else the high word truncates and the
        # arg index is wrong) — Triton-lens re-audit 2026-06-25.
        if not is_int:
            msl_val_type, val_shared_dtype = "float", "fp32"
        elif is_i64:
            msl_val_type = "long" if val_dtype == "i64" else "ulong"
            val_shared_dtype = "i64" if val_dtype == "i64" else "u64"
        else:
            msl_val_type, val_shared_dtype = "int", "i32"

        is_max = self._detect_reduce_direction(ssa)
        cmp_op = ">" if is_max else "<"
        identity = "(-INFINITY)" if is_max and not is_int else "INFINITY"
        if is_int:
            if val_dtype == "i64":
                identity = "LONG_MIN" if is_max else "LONG_MAX"
            elif is_i64:                                   # u64/ui64
                identity = "0" if is_max else "ULONG_MAX"
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
                    for _rid in (s.result_ids or []):
                        reduce_ids.add(_rid)
        if len(reduce_ids) < 2:
            return False
        _LAYOUT = {"tt.broadcast", "tt.expand_dims", "ttg.convert_layout",
                   "tt.reshape", "tt.splat"}

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
        _BIN = {"arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                "arith.addi", "arith.subi", "arith.muli", "arith.maxnumf",
                "arith.minnumf", "arith.maximumf", "arith.minimumf"}
        for s in all_ops:
            if s.op in _BIN and s.operand_ids and len(s.operand_ids) >= 2:
                r0 = _to_reduce(s.operand_ids[0])
                r1 = _to_reduce(s.operand_ids[1])
                if r0 is not None and r1 is not None and r0 != r1:
                    return True
        return False

    def _lower_reduce_2d(self, ssa, input_var, axis, combine_op,
                         msl_type, shared_dtype, input_shape):
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
                "stores, or reduce to 1-D.", op_name="tt.reduce")

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
        _in_dt = (self.env_types.get(ssa.operand_ids[0], "fp32")
                  if ssa.operand_ids else "fp32")
        if (getattr(self, "_control_flow_depth", 0) > 0
                and _in_dt in ("fp16", "bf16")):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "in-loop 2-D reduction of fp16/bf16 input is mis-staged across iterations "
                "(silent-wrong: rows collapse to the first). Refusing. Reduce in fp32, or "
                "outside the loop.", op_name="tt.reduce")

        _tg_threads = max(max(1, getattr(self.graph, "num_warps", 1)) * 32, 256)
        if (getattr(self, "_control_flow_depth", 0) > 0
                and total < _tg_threads):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Refusing in-loop 2-D reduction of a {tuple(input_shape)} tile that "
                f"under-fills the {_tg_threads}-thread group "
                f"(M*N={total} < block_size): the surplus threads corrupt the staged "
                f"reduce (silent-wrong). Use a tile with M*N >= block_size, or reduce "
                f"outside the loop.", op_name="tt.reduce")

        # Check if the input data is already in a shared memory array (e.g.,
        # from a dot result or local_alloc). If so, skip the copy and reuse it.
        input_id = ssa.operand_ids[0] if ssa.operand_ids else None
        existing_shared = None
        if input_id is not None:
            existing_shared = getattr(self, '_shared_mem_descs', {}).get(input_id)

        if existing_shared:
            shared_name = existing_shared[0]
            # Data is already in shared_name[lid] — no copy needed
        else:
            # Allocate shared memory for the full 2D tensor
            shared_name = f"shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Identity + combine expression. Must branch on the actual MSL type:
        # long/ulong need 64-bit identities and the integer max/min overload,
        # not the 32-bit INT_MIN/INT_MAX or the float fmax/fmin.
        identity, combine_expr = self._reduce_identity_combine(
            combine_op, msl_type)

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
                self.kb.raw_line(
                    f"    if (lid < {bs}u) {shared_name}[{input_bcast_idx}] = {input_var};"
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
            self.kb.raw_line(f"        {msl_type} acc = {identity};")
            self.kb.raw_line(f"        for (uint j = 0; j < {N}u; j++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[lid * {N}u + j];")
            self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_shared}[lid] = acc;")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their row's result.
            # Row-major: threads [0..N-1] in row 0, [N..2N-1] in row 1.
            self.kb.raw_line(f"    {result_var} = {result_shared}[lid / {N}u];")
        else:
            # Reduce along rows: each column reduces M values → N results
            result_size = N
            self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=N)
            self.kb.raw_line(f"    {msl_type} {result_var} = {identity};")
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {msl_type} acc = {identity};")
            self.kb.raw_line(f"        for (uint i = 0; i < {M}u; i++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[i * {N}u + lid];")
            self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_shared}[lid] = acc;")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their column's result
            self.kb.raw_line(f"    {result_var} = {result_shared}[lid % {N}u];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        # Result shape is the non-reduced dimension
        if axis == 1:
            self.env_shapes[ssa.id] = (M,)
        else:
            self.env_shapes[ssa.id] = (N,)


    def _lower_reduce_3d(self, ssa, input_var, axis, combine_op,
                         msl_type, shared_dtype, input_shape):
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
        identity, combine_expr = self._reduce_identity_combine(
            combine_op, msl_type)

        # Allocate shared memory for the full 3D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage values to shared memory from the already-loaded input_var
        # (not the raw pointer, which ignores strided/computed addresses)
        if total <= block_size:
            if input_bcast_idx is not None:
                # Non-canonical layout: write at bcast position.
                self.kb.raw_line(
                    f"    if (lid < {block_size}u) {shared_name}[{input_bcast_idx}] = {input_var};"
                )
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

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = result_dims
        # Record the broadcast layout: thread `lid` holds the value at flat
        # index `read_idx` of the logical (M, N)-or-(M, K)-or-(N, K) result.
        # Downstream reduces/stores use this to re-stage data correctly when
        # the logical mapping is not lid → lid.
        self._bcast_layout[ssa.id] = f"({read_idx})"
        self._register_bcast_layout_by_type(ssa.type_str, tuple(result_dims),
                                            f"({read_idx})")


    def _lower_reduce_nd(self, ssa, input_var, axis, combine_op,
                         msl_type, shared_dtype, input_shape):
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
                f"Refusing. Reduce in <= 3 dimensions.", op_name="tt.reduce")
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
        result_shape = tuple(input_shape[:axis]) + tuple(input_shape[axis + 1:])
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
        identity, combine_expr = self._reduce_identity_combine(
            combine_op, msl_type)

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
            existing_shared = getattr(self, '_shared_mem_descs', {}).get(input_id)

        # Allocate shared memory for the full N-D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(
            shared_name, dtype=shared_dtype, size=total)

        # Stage values to shared memory from the already-computed input_var
        if total <= block_size:
            if input_bcast_idx is not None:
                # Non-canonical layout: thread lid holds element at
                # input_bcast_idx. Write accordingly. Multiple threads may
                # map to the same logical position (broadcast redundancy);
                # they all write the same value so last-writer-wins is safe.
                self.kb.raw_line(
                    f"    if (lid < {block_size}u) {shared_name}[{input_bcast_idx}] = {input_var};"
                )
            else:
                self.kb.raw_line(
                    f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
        else:
            if x_ptr_name is None:
                # Fallback — cannot handle without a source pointer
                return
            self.kb.raw_line(
                f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            self.kb.raw_line(
                f"        {shared_name}[_e] = ({msl_type}){x_ptr_name}[_e];")
            self.kb.raw_line(f"    }}")
        self.kb.raw_line(
            f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Allocate result shared memory
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(
            result_shared, dtype=shared_dtype, size=result_total)

        # Per-output-position reduction: thread _r reduces along the reduce axis
        axis_size = input_shape[axis]
        self.kb.raw_line(
            f"    for (uint _r = lid; _r < {result_total}u; "
            f"_r += {block_size}u) {{")
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

        self.kb.raw_line(
            f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
        self.kb.raw_line(
            f"            {msl_type} val = {shared_name}"
            f"[_base + _a * {src_strides[axis]}u];")
        self.kb.raw_line(f"            acc = {combine_expr};")
        self.kb.raw_line(f"        }}")
        self.kb.raw_line(f"        {result_shared}[_r] = acc;")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(
            f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

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
            self.kb.raw_line(
                f"    {msl_type} {result_var} = {result_shared}[0];")
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
            self.kb.raw_line(
                f"    {msl_type} {result_var} = {result_shared}[{read_idx}];")
            result_read_idx = read_idx

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = result_shape
        # Record the broadcast layout: thread `lid` holds the value at flat
        # index `read_idx` in the reduced result. Downstream ops (reduce,
        # store, make_range rewrite) use this to re-stage data correctly.
        if result_read_idx is not None and nr >= 2:
            self._bcast_layout[ssa.id] = f"({result_read_idx})"
            self._register_bcast_layout_by_type(ssa.type_str, tuple(result_shape),
                                                f"({result_read_idx})")


    def _lower_scan(self, ssa: SSAValue):
        """tt.scan → prefix scan via shared memory.

        For 2D tensors: scan along the specified axis.
            axis=1 on (M, N): each row gets an independent prefix scan along N
            axis=0 on (M, N): each column gets an independent prefix scan along M
        Supports forward and reverse scans, single and multi-value combines.
        """
        axis = ssa.attrs.get("axis", 0)
        reverse = ssa.attrs.get("reverse", False)
        n_values = len(ssa.operand_ids)

        if not ssa.operand_ids:
            return

        # Get input shape from type string
        is_1d = False
        input_shape = _extract_shape(ssa.type_str)
        if not input_shape or len(input_shape) < 2:
            input_shape = _extract_shape(
                self._find_op_type_str(ssa.operand_ids[0]))
        if not input_shape or len(input_shape) < 2:
            # 1D tensor: treat as (1, size) and scan along axis=1
            sz = input_shape[0] if input_shape else self.effective_block_size
            input_shape = (1, sz)
            axis = 1  # 1D scan always scans along the data dimension
            is_1d = True

        M, N = input_shape[0], input_shape[1]
        total = M * N

        # The scan stages every element through threadgroup memory with one thread
        # per element (`if (lid < total) shared[lid] = ...` below) and then reads it
        # back in the prefix sweep. Metal caps a threadgroup at 1024 threads, so for
        # total > 1024 the elements at index >= 1024 are NEVER written — the sweep
        # then reads uninitialized shared memory and returns silently-wrong values
        # (the store/atomic paths already guard this one-thread-per-element regime).
        # The MSL scan has no multi-element-per-thread path, so refuse loudly.
        if total > 1024:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Refusing a {total}-element scan (tl.cumsum / associative_scan): "
                f"Metal allows at most 1024 threads per threadgroup and the scan maps "
                f"one element per thread, so elements past 1024 would be left "
                f"uninitialized and the result silently wrong. Reduce the scan tile to "
                f"<= 1024 elements.")

        # Determine element type and MSL type
        input_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (input_dtype.startswith("fp") or input_dtype.startswith("bf"))
        is_i64 = input_dtype in ("i64", "u64", "ui64")
        is_u64 = input_dtype in ("u64", "ui64")
        # 64-bit ints must NOT truncate to i32 (cumsum/scan wrapped at 2^31, re-audit #13);
        # the shared accumulator-type helper handles it (scan has no unsigned max/min).
        msl_type, shared_dtype = self._reduce_acc_msl_type(input_dtype)

        # A multi-value scan stages EVERY slot with operand-0's dtype (single
        # shared_dtype), so a mixed-dtype scan (e.g. i32 count + fp32 sum) silently
        # truncates the other slots (re-audit #14: the fp32 sum slot became i32 -> all
        # zeros). Refuse mixed-dtype multi-value scans; same-dtype scans proceed.
        if n_values > 1:
            _slot_dtypes = {self.env_types.get(o, "fp32") for o in ssa.operand_ids}
            if len(_slot_dtypes) > 1:
                from triton_msl.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "multi-value tl.associative_scan with mixed operand dtypes is not "
                    "supported — all slots would be staged with the first operand's "
                    "dtype, silently truncating the others. Refusing.", op_name="tt.scan")

        # Allocate shared memory for each input value
        shared_names = []
        for i in range(n_values):
            shared_name = f"scan_shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype,
                                              size=total)
            shared_names.append(shared_name)

        # Write input values to shared memory
        for i, operand_id in enumerate(ssa.operand_ids):
            input_var = self._lookup(operand_id)
            cast = f"({msl_type})" if input_dtype == "bf16" else ""
            self.kb.raw_line(
                f"    if (lid < {total}u) {shared_names[i]}[lid] = {cast}{input_var};")
        self.kb.raw_line(
            f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

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
            self.kb.raw_line(
                f"    {msl_type} {acc_var} = ({msl_type}){shared_names[i]}[{init_idx}];")

        # Emit scan loop
        if not reverse:
            self.kb.raw_line(
                f"    for (uint scan_j = 1u; scan_j <= {pos_expr}; scan_j++) {{")
        else:
            self.kb.raw_line(
                f"    for (uint scan_j = 1u; scan_j <= ({scan_size - 1}u - {pos_expr}); scan_j++) {{")

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
            self.kb.raw_line(
                f"        {msl_type} {rhs_var} = ({msl_type}){shared_names[i]}[{idx_expr}];")

        # Map block args to accumulator (lhs) and current element (rhs) vars
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids and len(block_arg_ids) >= 2 * n_values:
            for i in range(n_values):
                self.env[block_arg_ids[i]] = acc_vars[i]
                self.env_types[block_arg_ids[i]] = shared_dtype
                self.env[block_arg_ids[n_values + i]] = rhs_vars[i]
                self.env_types[block_arg_ids[n_values + i]] = shared_dtype

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
            self.kb.raw_line(
                f"    if (lid < {total}u) {shared_names[0]}[lid] = {acc_vars[0]};")
            self.kb.raw_line(
                f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            result_var = self._next_var("scan_result")
            self.kb.raw_line(
                f"    {msl_type} {result_var} = ({msl_type}){shared_names[0]}[lid % {total}u];")
            for i in range(1, n_values):
                self.kb.raw_line(
                    f"    if (lid < {total}u) {shared_names[i]}[lid] = {acc_vars[i]};")
            if n_values > 1:
                self.kb.raw_line(
                    f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            acc_vars_out = [result_var]
            for i in range(1, n_values):
                rv = self._next_var("scan_result")
                self.kb.raw_line(
                    f"    {msl_type} {rv} = ({msl_type}){shared_names[i]}[lid % {total}u];")
                acc_vars_out.append(rv)
        else:
            acc_vars_out = acc_vars

        # Map scan results to output variables
        if ssa.result_ids and len(ssa.result_ids) >= n_values:
            for i in range(n_values):
                self.env[ssa.result_ids[i]] = acc_vars_out[i]
                self.env_types[ssa.result_ids[i]] = shared_dtype
                self.env_shapes[ssa.result_ids[i]] = input_shape
        else:
            self.env[ssa.id] = acc_vars_out[0]
            self.env_types[ssa.id] = shared_dtype
            self.env_shapes[ssa.id] = input_shape

    # -- Shared memory ops (ttg.local_alloc / ttg.local_load) --


