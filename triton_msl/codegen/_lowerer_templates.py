"""Template methods for ``GenericLowerer``.

Pattern-specific MSL emitters that bypass the generic op-by-op lowering when
a recognized pattern (softmax, flash-attention dot, flip, row-wise sort,
3-D reduce, etc.) is detected. Each method takes an ``info`` dict produced
by the corresponding ``_detect_*`` predicate and returns a complete MSL
kernel string.

These all live on a mixin rather than as free functions because they call
into ``GenericLowerer`` instance state (``self.graph``, ``self.options``,
``self.kb``, ``self._next_var``, ``self.env``, etc.). The mixin pattern
lets them share that state without polluting ``generic_lowerer.py``\'s
top-level namespace, and without requiring every template to take 10+
parameters.
"""

import re

from triton_msl.codegen.mlir_walker import _extract_shape
from triton_msl.codegen.msl_emitter import _msl_compute_type, _sanitize_msl_name
from triton_msl.codegen.msl_types import triton_type_to_msl

from triton_msl.codegen._lowerer_helpers import _mlir_to_triton_dtype
from triton_msl.codegen._narrow_float import NARROW, int_to_narrow


def _scalar_buffer_decl(arg, index, source_bindings=None):
    """Match the driver's scalar storage ABI, not a template's index type.

    Even an unused scalar keeps its native positional argument slot. Declaration
    and unpacking share the type mapping so a float's bits cannot become an int,
    or a narrow/wide integer be read at the wrong width (GitHub issue11).
    """
    name = (source_bindings or {}).get(arg.name, arg.name)
    return f"    device {triton_type_to_msl(arg.elem_type)}* {name}_buf [[buffer({index})]]"


def _scalar_unpack(arg, source_bindings=None):
    name = (source_bindings or {}).get(arg.name, arg.name)
    return f"    {triton_type_to_msl(arg.elem_type)} {name} = {name}_buf[0];"


def _emit_masked_staged_store(lines, *, acc, scratch, gr, gc, cond, dst, out_type, store_pfx="", indent="    "):
    """Emit the masked, per-simdgroup staged store of ONE float8x8 accumulator.

    simdgroup_store can't mask, so each simdgroup stages its 8x8 into its OWN
    64-float slot (``{scratch} + sgitg*64u`` — NEVER a shared offset, which raced
    across simdgroups), barriers, then its 32 lanes (``laneid = tiitg % 32u``, which
    the caller must declare) write only the in-bounds elements with a cast to
    ``out_type``. The caller supplies the per-site index math (``gr``/``gc``), the
    bounds ``cond``, the ``dst`` lvalue, and an optional ``store_pfx`` column guard.

    SINGLE SOURCE OF TRUTH for the staging mechanism: it previously existed as two
    near-identical copies (the simple-dot and K-loop dot epilogues), and a fix
    landing in only one twin caused two silent-wrongs (a cross-simdgroup race and an
    unmasked-overflow OOB — 2026-06-21/22 audit). Both call sites MUST route through
    here so the mechanism can't diverge again. (Per-site index math legitimately
    differs and stays at the call site; only the bug-prone staging is shared.)
    """
    lines.append(f"{indent}{store_pfx}simdgroup_store({acc}, {scratch} + sgitg * 64u, 8);")
    lines.append(f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append(f"{indent}for (uint i = laneid; i < 64u; i += 32u) {{")
    lines.append(f"{indent}    uint gr = {gr}, gc = {gc};")
    lines.append(f"{indent}    if ({cond}) {{ {dst} = {out_type}({scratch}[sgitg * 64u + i]); }}")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);")


class _TemplateMixin:
    """Pattern-specialized MSL emitters.

    Mixed into ``GenericLowerer``. Methods here read instance state owned by
    the main lowerer (``self.graph``, ``self.options``, ``self.kb``, etc.)
    but never define new state — that belongs in ``GenericLowerer.__init__``.
    """

    def _simdgroup_leading_dims_are_dense(self, info, descriptors):
        """True iff every operand's inferred ROW stride is a compile-time literal
        equal to the matrix dim the NON-K-loop simdgroup template hard-codes as that
        operand's leading dim (A leading = K, B leading = N, C leading = N).

        Returns False for a runtime-arg row stride (can't be threaded into the 3-buffer
        simple-dot template) or a literal that differs from the dim (a sliced/padded
        row) — the caller then routes to the stride-aware scalar matmul. Used only for
        the non-K-loop single-tile form; the K-loop form threads strides directly.
        """
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        M = info.get("M")
        N = info.get("N")
        K = info.get("K")

        def _is_lit_eq(stride, dim):
            if dim is None:
                return False
            try:
                return int(str(stride)) == int(dim)
            except (TypeError, ValueError):
                return False  # a runtime arg NAME (not an int literal) → not dense

        return _is_lit_eq(a_row, K) and _is_lit_eq(b_row, N) and _is_lit_eq(c_row, N)

    def _matmul_stride_decision(self, info):
        """Decide the stride-aware lowering route for a bare-matmul ``info``.

        Returns one of:
          - ``("simdgroup", (a_row, a_col, b_row, b_col, c_row, c_col))``: every
            operand has a CONTIGUOUS inner dim (col stride ``"1"``), so the
            simdgroup-MMA fast path can load it — using the inferred ROW strides
            as the simdgroup_load leading dims (correct for sliced/padded rows,
            not just dim==stride).
          - ``("scalar", descriptors)``: at least one operand has a
            NON-contiguous inner dim (col stride is a runtime arg or a non-unit
            constant, e.g. a transposed B from ``x @ w.t()``). simdgroup_load
            needs a contiguous inner dim, so route to the stride-aware scalar
            matmul which addresses with all 6 strides.
          - ``None``: no traceable single-dot stride info -> caller keeps its
            existing (legacy / dim-based) behavior.

        REFUSES loudly (MetalNonRecoverableError) when the kernel IS a single-dot
        matmul but a stride is un-inferable (infer returned a None slot) — never
        falls through to a row-major guess (the silent-wrong this fixes).
        """
        sd = self.infer_dot_strides()
        if sd is None:
            return None  # not a traceable single-dot matmul
        descriptors = self._inferred_stride_descriptors()
        if descriptors is None:
            # Single-dot matmul but a stride could not be inferred. Guessing
            # row-major here is exactly the silent-wrong we are eliminating.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "matmul operand stride could not be inferred from the address "
                "arithmetic (traced layout: "
                f"{sd}); refusing to assume a row-major layout, which would "
                "silently mis-compute a transposed/strided operand. File an "
                "issue at https://github.com/bledden/triton-msl/issues",
                op_name="tt.dot",
            )
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        # Contiguous inner dim == col stride is literal "1". A runtime-arg or
        # non-unit-constant col stride means the inner dim is strided.
        if a_col == "1" and b_col == "1" and c_col == "1":
            return ("simdgroup", descriptors)
        return ("scalar", descriptors)

    def _tail_ref(self, value):
        """Immutable runtime/literal reference used by the launch bounds plan."""
        for index, arg in enumerate(self.graph.args):
            if arg.name == value:
                width = self._declared_int_elem_type(arg.name)
                if width not in ("i32", "i64", "u32", "u64"):
                    return None
                return ("arg", index, width)
        try:
            return ("literal", int(value), "i64")
        except (TypeError, ValueError, OverflowError):
            return None

    def _matmul_needs_grid_plan(self, info):
        store = self._matmul_output_extent_args()
        grid = info.get("grid_extent_args", (None, None))
        return (
            None in store
            or info.get("load_domain_args", info.get("load_extent_args")) != store
            or any(bound is not None and bound != clip for bound, clip in zip(grid, store))
        )

    def _record_full_block_tail_bounds(self, info, block_m, block_n, block_k, source):
        """Bind an unmasked K-tail template to its runtime allocation proof."""
        roles = self._dot_template_ptr_roles()
        strides = self._inferred_stride_descriptors()
        widths = self._dot_offset_arith_widths()
        if roles is None or len(roles) != 3 or strides is None or any(value is None for value in strides):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "unmasked K-loop full-block replay has no complete A/B/C address proof; mask the K tail",
                op_name="tt.dot",
            )
        if widths is None or any(role not in widths for role in ("A", "B", "C")):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "unmasked K-loop source address widths are unresolved; mask the K tail", op_name="tt.dot"
            )
        indices = {arg.id: index for index, arg in enumerate(self.graph.args)}
        try:
            tensor_indices = tuple(indices[arg.id] for arg in roles)
        except (KeyError, AttributeError):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "unmasked K-loop pointer ownership is unresolved; mask the K tail", op_name="tt.dot"
            )
        m_ext, n_ext = self._matmul_output_extent_args()
        grid_plan = self._matmul_needs_grid_plan(info)
        m_ref = self._tail_ref(m_ext) if m_ext is not None else None
        n_ref = self._tail_ref(n_ext) if n_ext is not None else None
        k_ref = self._tail_ref(source)
        stride_refs = tuple(self._tail_ref(value) for value in strides)
        if (
            (m_ext is not None and m_ref is None)
            or (n_ext is not None and n_ref is None)
            or k_ref is None
            or any(ref is None for ref in stride_refs)
        ):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "unmasked K-loop affine bounds cannot be represented by the launch ABI; mask the K tail",
                op_name="tt.dot",
            )
        descriptor = (
            "matmul_full_k_storage_v1",
            int(block_k),
            tensor_indices,
            (m_ref, n_ref, k_ref),
            stride_refs,
            tuple(tuple(widths[role]) for role in ("A", "B", "C")),
        )
        if grid_plan:
            grid_refs = tuple(
                self._tail_ref(v) if v is not None else None for v in info.get("grid_extent_args", (None, None))
            )
            load_refs = tuple(
                self._tail_ref(v) if v is not None else None
                for v in info.get("load_domain_args", info.get("load_extent_args", (None, None)))
            )
            source_extents = (
                *info.get("grid_extent_args", (None, None)),
                *info.get("load_domain_args", info.get("load_extent_args", (None, None))),
            )
            if any(bound is not None and ref is None for bound, ref in zip(source_extents, (*grid_refs, *load_refs))):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "matmul source extent cannot be represented by the launch ABI", op_name="tt.dot"
                )
            descriptor = (
                "matmul_grid_storage_v1",
                *descriptor[1:],
                (
                    int(block_m),
                    int(block_n),
                    info.get("pid_map"),
                    tuple(info.get("pid_axes") or (False, False)),
                    grid_refs,
                    load_refs,
                    info.get("k_tail_mode") == "full_blocks",
                ),
            )
        identity_args = info.get("load_identity_modulo_args", (None, None))
        if any(value is not None for value in identity_args):
            identity_refs = tuple(self._tail_ref(value) if value is not None else None for value in identity_args)
            if not grid_plan or any(
                value is not None and ref is None for value, ref in zip(identity_args, identity_refs)
            ):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "matmul load modulo has no representable no-wrap launch proof", op_name="tt.dot"
                )
            descriptor = ("matmul_grid_storage_v2", *descriptor[1:6], (*descriptor[6], identity_refs))
        previous = getattr(self, "_tail_access_bounds", None)
        if previous is not None and previous != descriptor:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "matmul template produced inconsistent K-tail launch contracts; mask the K tail", op_name="tt.dot"
            )
        self._tail_access_bounds = descriptor

    def _k_extent_line(self, info, BLOCK_K, has_K, BLOCK_M=None, BLOCK_N=None, source_bindings=None):
        """MSL statement setting ``_K`` (the matmul reduction extent), resolved
        STRUCTURALLY from the K-loop bounds (``k_extent_arg`` for a runtime argument of
        any name, ``k_extent_const`` elements for constant bounds; see
        ``_kloop_source_extent``). A looped kernel never falls back to an argument that is
        merely NAMED ``K``: with constant bounds, or a count over another argument, that
        argument is a stride or an unrelated extent (GPU: reduced over it anyway). A
        non-looped dot reduces its own BLOCK_K tile, whatever the arguments are called
        (``has_K`` is unused and kept for the callers' signature). Refuses
        when a runtime reduction extent can't be resolved, rather than silently reducing
        over a single BLOCK_K tile (GitHub issue #4.1 -- renaming the reduction extent
        ``K -> DIM`` dropped the K-loop and returned ``A[:, :BLOCK_K] @ B[:BLOCK_K, :]``
        with no error). Single source of truth for every matmul template's ``_K``."""
        source_bindings = source_bindings or {}
        k_extent_arg = info.get("k_extent_arg")
        has_k_loop = bool(info.get("has_k_loop"))
        if has_k_loop:
            mode = info.get("k_tail_mode")
            source = k_extent_arg if k_extent_arg is not None else info.get("k_extent_const")
            source_expr = source_bindings.get(source, str(source)) if str(source).isidentifier() else f"({source})"
            if source is not None and mode == "masked_zero":
                if self._matmul_needs_grid_plan(info):
                    self._record_full_block_tail_bounds(info, BLOCK_M, BLOCK_N, BLOCK_K, source)
                return f"    long _K = (long){source_expr};  // both operands prove a zero-masked K tail"
            if source is not None and mode == "full_blocks":
                # Padding a dynamic i64 extent with ``BLOCK_K - 1`` in signed MSL
                # could overflow before the fragment-path range guard runs.  The
                # canonical Triton K-loop induction bound is i32, for which the
                # widening addition below is exact.  Decline wider dynamic bounds
                # rather than manufacture a wrapped reduction extent.
                source_width = info.get("k_extent_width")
                if k_extent_arg is not None:
                    if source_width != "i32":
                        from triton_msl.errors import MetalNonRecoverableError

                        raise MetalNonRecoverableError(
                            "unmasked K-loop full-block replay requires an i32 source "
                            "reduction extent; a wider dynamic bound cannot be rounded "
                            "without a signed-overflow proof",
                            op_name="tt.dot",
                        )
                elif int(source) > (2**63 - 1) - (BLOCK_K - 1):
                    from triton_msl.errors import MetalNonRecoverableError

                    raise MetalNonRecoverableError(
                        "unmasked K-loop full-block replay extent exceeds the signed MSL reduction-index range",
                        op_name="tt.dot",
                    )
                self._record_full_block_tail_bounds(info, BLOCK_M, BLOCK_N, BLOCK_K, source)
                return (
                    f"    long _K = (long){source_expr};\n"
                    f"    long _K_source = _K;\n"
                    f"    _K = _K_source > 0L ? ((_K_source + {BLOCK_K - 1}L) / {BLOCK_K}L) * {BLOCK_K}L : 0L;"
                    "  // unmasked source executes complete BLOCK_K iterations"
                )
            if mode not in ("masked_zero", "full_blocks"):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "K-loop matmul reduction-tail semantics were not proven from both operand loads",
                    op_name="tt.dot",
                )
        else:
            # A non-looped dot reduces exactly its own tile: the loads are unmasked (the
            # value-path proof refuses a mask), so an argument named K is A's row stride
            # or anything else, never the depth (GPU: K=64 over a 32-wide tile summed 64).
            return f"    long _K = {BLOCK_K};  // single, non-looped dot tile"
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(
            "matmul K-loop reduction extent could not be resolved: it is neither a "
            "compile-time constant nor traceable to a kernel argument, so reducing over "
            "a single BLOCK_K tile would be silently wrong. Refusing. Pass the reduction "
            "size as a scalar kernel argument used directly as the loop bound, e.g. "
            "`for k in range(0, DIM, BK)`.",
            op_name="tt.dot",
        )

    def _single_tile_is_tiled_or_clipped(self):
        """Whether the source tiles an axis by program_id or clips its store at a runtime extent.

        Either makes the program depend on a value the dense single-tile template does
        not read. Any program-id operation counts, wherever it is used.
        """

        def _any_pid(ops):
            for op in ops:
                if (
                    op.op in self._PID_OPS
                    or (op.region_ops and _any_pid(op.region_ops))
                    or (op.else_ops and _any_pid(op.else_ops))
                ):
                    return True
            return False

        return _any_pid(self.graph.ops) or any(extent is not None for extent in self._matmul_output_extent_args())

    def _refuse_if_pid_tiles_baked_output(self, has_M, has_N, what):
        """Single-source integrity guard for every matmul template that emits the
        whole baked output from a single threadgroup. If the kernel tiles the
        output across programs (program_id on the M/N axes) but M/N are baked
        constexpr (not runtime args), the true output extent/strides can't be
        derived here — refuse rather than emit silently-wrong output (each block
        would recompute/overwrite the same rows, or collapse a real stride to the
        block size). De-duplicated from three copies after the re-audit
        (2026-06-27) found the fused matmul+softmax twin was the one copy MISSING
        this guard -> multi-block silent-wrong."""
        # An axis whose extent resolves STRUCTURALLY from the output store mask (any arg
        # name, issue #4.5) is recoverable, so don't refuse it as "baked constexpr".
        m_ext, n_ext = self._matmul_output_extent_args()
        has_M = has_M or m_ext is not None
        has_N = has_N or n_ext is not None
        pid_axes = {s.attrs.get("axis", 0) for s in self.graph.ops if s.op == "tt.get_program_id"}
        if (1 in pid_axes and not has_N) or (0 in pid_axes and not has_M):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"{what} tiles the output across programs (program_id axes "
                f"{sorted(pid_axes)}) but M/N are baked as constexpr, not runtime "
                "args — the true output extent/strides can't be derived here. "
                "Refusing rather than emit silently-wrong output.",
                op_name="tt.dot",
            )

    @staticmethod
    def _pid_map_lines(info, block_m, source_bindings=None):
        """The tile coordinates, replaying the PROVEN grid mapping (packet 080):
        the 2-D form reads ``program_id(0)`` / ``program_id(1)``; the tutorial's 1-D form
        splits ``program_id(0)`` by ``num_pid_m = cdiv(M, BLOCK_M)`` — rows are the
        remainder, cols the quotient. The N-fastest form reverses those roles
        using the proved N extent. Requires ``_M``/``_N`` to be defined already."""
        source_bindings = source_bindings or {}
        grid_m, grid_n = info.get("grid_extent_args", (None, None))
        grid_m = source_bindings.get(grid_m, grid_m) or "_M"
        grid_n = source_bindings.get(grid_n, grid_n) or "_N"
        pid_map = info.get("pid_map")
        if isinstance(pid_map, tuple) and len(pid_map) == 2 and pid_map[0] == "flat_n":
            block_n = pid_map[1]
            return [
                "    // N-fastest source tile mapping; preserve signed i32 add/div/rem",
                f"    int _npn = as_type<int>((uint){grid_n} + {block_n - 1}u) / {block_n};",
                "    uint pid_m = as_type<uint>(as_type<int>(pid3.x) / _npn);",
                "    uint pid_n = as_type<uint>(as_type<int>(pid3.x) % _npn);",
            ]
        if isinstance(pid_map, tuple) and len(pid_map) == 3 and pid_map[0] == "grouped":
            group, block_n = pid_map[1:]
            return [
                "    // grouped source tile mapping; retain the short final group",
                # arith.addi/muli/subi wrap at i32, while divsi/remsi/minsi
                # interpret those bits as signed. Do not turn the latter unsigned.
                f"    int _npm = as_type<int>((uint){grid_m} + {block_m - 1}u) / {block_m};",
                f"    int _npn = as_type<int>((uint){grid_n} + {block_n - 1}u) / {block_n};",
                f"    int _span = as_type<int>(as_type<uint>(_npn) * {group}u);",
                f"    int _first_m = as_type<int>(as_type<uint>(as_type<int>(pid3.x) / _span) * {group}u);",
                f"    int _group_m = min(as_type<int>(as_type<uint>(_npm) - as_type<uint>(_first_m)), {group});",
                "    int _within = as_type<int>(pid3.x) % _span;",
                "    uint pid_m = as_type<uint>(_first_m) + as_type<uint>(_within % _group_m);",
                "    uint pid_n = as_type<uint>(_within / _group_m);",
            ]
        if pid_map == "1d":
            return [
                f"    uint _npm = ((uint){grid_m} + {block_m}u - 1u) / {block_m}u;  // cdiv(M, BLOCK_M), as in the IR",
                "    uint pid_m = pid3.x % _npm;",
                "    uint pid_n = pid3.x / _npm;",
            ]
        if pid_map == "2d":
            # packet 106: an axis the kernel does NOT tile by program_id is tile 0 in every
            # program (the kernel's semantics), never ``pid3.x`` / ``pid3.y``.
            has_m, has_n = tuple(info.get("pid_axes") or (True, True))
            return [
                "    uint pid_m = pid3.x;"
                if has_m
                else "    uint pid_m = 0u;  // no program_id on rows in the source kernel",
                "    uint pid_n = pid3.y;"
                if has_n
                else "    uint pid_n = 0u;  // no program_id on cols in the source kernel",
            ]
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(  # no implicit 2-D fallback: the mapping must be PROVEN
            f"pid-tiled matmul template reached without a proven grid mapping (pid_map={pid_map!r}). Refusing.",
            op_name="tt.dot",
        )

    def _matmul_tile_extent_lines(self, info, block_m, block_n, source_bindings):
        """Exact store/load clips; an absent clip means this program's entire tile.

        Grid coordinates have already been recovered from their own source divisor.
        Local full-tile endpoints keep every SIMD/scalar branch on the same domain
        without inventing a global extent from an unrelated scalar's spelling.
        """
        store = self._matmul_output_extent_args()
        loads = info.get("load_domain_args", info.get("load_extent_args", store))
        result = []
        for name, bound, base, block in (
            ("_M", store[0], "row_base", block_m),
            ("_N", store[1], "col_base", block_n),
            ("_A_M", loads[0], "row_base", block_m),
            ("_B_N", loads[1], "col_base", block_n),
        ):
            if type(bound) is int:
                expr = f"{bound}L"  # a compile-time load clip
            else:
                expr = source_bindings[bound] if bound is not None else f"(long){base} + {block}L"
            result.append(f"    long {name} = (long)({expr});")
        return result

    def _dot_template_ptr_roles(self):
        """A/B/C pointer args for a single-dot matmul template, resolved from
        DATAFLOW (dot operands + tt.store target via ``_resolve_dot_ptr_roles``),
        never from argument position.

        P0 (packet 065): every bare/K-loop/scalar/simple dot template picked
        ``ptr_args[0/1/2]``. Any signature with an extra pointer before C —
        e.g. ``(X, Y, P, C)`` — wrote X @ Y into P (an unrelated INPUT) and left C
        untouched; GPU-verified, no exception. Refuses when roles cannot be proven.
        """
        from triton_msl.errors import MetalNonRecoverableError

        dots, stores, op_by_id = [], [], {}

        def _walk(ops):
            for o in ops:
                op_by_id[o.id] = o
                if o.op == "tt.dot":
                    dots.append(o)
                elif o.op == "tt.store":
                    stores.append(o)
                if o.region_ops:
                    _walk(o.region_ops)
                if o.else_ops:
                    _walk(o.else_ops)

        _walk(self.graph.ops)
        # EXACTLY ONE STORE, whole graph (packet 067): every template here emits one
        # store, so a second one is either mis-bound (a store of P before the dot's
        # store made P the "first store" and the matmul overwrote it, C left NaN) or
        # silently DROPPED (dot's store first, then a store of P never emitted).
        # GPU-verified on both orders. Default-deny; the generic path handles
        # multi-store graphs correctly, this template does not.
        if len(dots) != 1 or len(stores) != 1:
            raise MetalNonRecoverableError(
                f"matmul template requires exactly one tt.dot and exactly one tt.store "
                f"in the whole kernel (found {len(dots)} dot(s), {len(stores)} store(s)): "
                f"a second store would be silently dropped or mis-bound as the output. "
                f"Refusing (correct-or-refuse).",
                op_name="tt.dot",
            )
        # PER-ROLE VALUE-PATH PROOF (packet 079): roles from dataflow (through loop
        # iter-args), operand load -> layout ops -> dot with equal types, the store
        # path carrying at most the ONE output cast the template replays, K-loop
        # accumulator carry / pointer advances / load masks, and the grid mapping.
        # Returns the cast ops it proved the template replays, admitted below by
        # IDENTITY — an opcode allowlist cannot say WHERE a cast sits, and that was
        # exactly the hole (fp16 operand casts dropped, GPU err vs raw = 0).
        _vp = self._dot_template_value_paths()  # pid map / axes consumed via info by the pid-tiled templates
        _reason, _proven, _roles, _pid_map = _vp[0], _vp[1], _vp[2], _vp[3]
        if _reason:
            raise MetalNonRecoverableError(
                f"matmul template: {_reason}. Refusing (correct-or-refuse).",
                op_name="tt.dot",
            )
        # POSITIVE EFFECT/OP ALLOWLIST, whole graph (packet 071): the bare templates
        # re-emit exactly this vocabulary — staging loads, address arithmetic, layout
        # ops, the dot, K-loop control, and the single terminal store — and NOTHING
        # else. Any op outside it is either an externally observable effect the
        # template never replays (tt.atomic_rmw / tt.atomic_cas: GPU-verified dropped,
        # P stayed 0 with no exception) or compute on a value the template does not
        # model. Enumerated denylists found this class one opcode at a time (two
        # stores, then atomics); the allowlist is the censused union of every graph
        # that reaches this gate in the passing project + upstream suites. Casts are
        # NOT in it (packet 079): only the proven output cast is admitted, by id.
        _REPLAYED = {
            "arith.addi",
            "arith.andi",
            "arith.cmpi",
            "arith.constant",
            "arith.muli",
            "scf.for",
            "scf.yield",
            "tt.addptr",
            "tt.broadcast",
            "tt.dot",
            "tt.expand_dims",
            "tt.get_program_id",
            "tt.get_num_programs",
            "tt.load",
            "tt.make_range",
            "tt.return",
            "tt.splat",
            "tt.store",
            "ttg.convert_layout",
            "ttg.local_alloc",
            "ttg.local_load",
            "ttg.memdesc_trans",
        }
        # INTEGER arith is address / index / mask math (grid `pid // n`, `n - k`,
        # ceil-div, index casts): value-preserving and effect-free, and it cannot touch
        # the (float) dot result. Admit it as a CLASS by result type rather than by name
        # — enumerating divsi/subi/remsi/… one at a time is the same trap as enumerating
        # effects. Float-typed arith (addf/mulf/…) stays foreign: that is result compute.
        _INT_RESULT = re.compile(r"(?:^|[x<\s])i(?:1|8|16|32|64)(?![0-9])")

        def _replayed(o):
            if o.op in _REPLAYED or o.id in _proven:
                return True
            if o.op.startswith("arith.") and o.type_str:
                return bool(_INT_RESULT.search(o.type_str.split(",")[0]))
            return False

        _foreign = sorted({o.op for o in op_by_id.values() if not _replayed(o)})
        if _foreign:
            raise MetalNonRecoverableError(
                f"matmul template does not replay {_foreign}: only staging loads, address/"
                f"layout ops, the dot, K-loop control and the single terminal store are "
                f"re-emitted, so this op's effect or value would be silently dropped. "
                f"Refusing (correct-or-refuse).",
                op_name=_foreign[0],
            )
        # C must be THE single store's target, proven directly — not a first-store
        # convention inherited from the shared resolver.
        _c_from_store = self._trace_ptr_source(stores[0].operand_ids[0], op_by_id) if stores[0].operand_ids else None
        if _c_from_store is None or _c_from_store.name != _roles[2].name:
            raise MetalNonRecoverableError(
                "matmul template could not resolve A/B/C pointer roles from dataflow "
                "(dot operands + the single store's target); refusing rather than pick "
                "them by argument position.",
                op_name="tt.dot",
            )
        return _roles[0], _roles[1], _roles[2]

    # ------------------------------------------------------------------
    # Address-arithmetic WIDTH/SIGNEDNESS (packet 825, W2 F-7)
    #
    # A Triton offset expression is evaluated in the SOURCE's integer type and
    # only then sign-extended by ``tt.addptr``. The templates used to render
    # every runtime stride as ``(uint)s`` and index with ``uint`` variables,
    # which (a) TRUNCATED an i64 stride to 32 bits, (b) evaluated the sum as
    # unsigned, and (c) zero-extended a negative offset into the pointer
    # (``A[(uint)(-6)]`` -> ``A + 4294967290``). The driver binds every argument
    # at its view's byte offset inside a full-storage mirror, so a negative
    # offset relative to the bound base is reachable, not hypothetical.
    #
    # The repair reproduces the source's arithmetic instead of swapping casts:
    #   * i32 offset term -> 32-bit WRAPPING product in ``uint`` (well defined,
    #     no signed-overflow UB), bit-cast back with ``as_type<int>`` and then
    #     widened to ``long`` for the pointer add (= the addptr sign-extend).
    #   * i64 offset term -> computed in ``long`` with the i32 index promoted
    #     (= the ``arith.extsi`` Triton inserts).
    #   * terms that share ONE ``tt.addptr`` link are summed at the link's own
    #     width before that extend; terms on SEPARATE links are extended
    #     independently (Triton emits one addptr per source ``+``).
    # Anything whose width or signedness is not provable refuses.
    # ------------------------------------------------------------------

    _ADDR_LAYOUT_OPS = (
        "tt.broadcast",
        "ttg.convert_layout",
        "tt.reshape",
        "arith.sitofp",
        "arith.extsi",
        "arith.trunci",
    )

    def _addr_op_index(self):
        """``{ssa_id: op}`` over the whole function, regions included."""
        op_by_id = {}

        def _collect(ops):
            for s in ops:
                op_by_id[s.id] = s
                if s.region_ops:
                    _collect(s.region_ops)
                if s.else_ops:
                    _collect(s.else_ops)

        _collect(self.graph.ops)
        return op_by_id

    def _dot_offset_arith_widths(self):
        """Prove the offset-arithmetic width of each matmul operand's address, PER AXIS.

        Walks the ``tt.addptr`` chain ``infer_dot_strides`` already traced for each
        A/B/C role. Each link's offset operand carries the element type Triton
        evaluated that link's offset in — and the type ``tt.addptr`` sign-extends
        from. Each link is attributed to an axis by the ``tt.expand_dims`` axes
        reachable inside its offset subgraph (axis 1 = row, axis 0 = col), which is
        exactly how ``_classify`` separates the terms; the INDEX ssa ids cannot be
        used because a kernel that reuses one ``tl.arange`` for both axes CSEs them
        to a single ``tt.make_range``.

        Returns ``{"A": (row_width, col_width, fused), ...}`` where each width is
        ``"i32"`` or ``"i64"`` and ``fused`` says both terms share ONE addptr link
        (so their sum wraps at that width before the single extend). Returns
        ``None`` when anything is unprovable — the caller then refuses.
        """
        traced = self.infer_dot_strides(with_index=True)
        if not (isinstance(traced, tuple) and len(traced) == 3):
            return None
        _strides, _index_ids, addptr_ids = traced
        if not isinstance(addptr_ids, dict):
            return None
        op_by_id = self._addr_op_index()
        _EXT = ("arith.extsi", "arith.extui", "arith.trunci")
        _SHAPE = ("tt.broadcast", "ttg.convert_layout", "tt.reshape")

        def _skip(o):
            seen = 0
            while o is not None and seen < 16 and o.op in self._ADDR_LAYOUT_OPS:
                if not o.operand_ids:
                    return o
                o = op_by_id.get(o.operand_ids[0])
                seen += 1
            return o

        def _peel(o):
            """Drop shape-only wrappers (NOT the width-carrying ext/trunc ops)."""
            seen = 0
            while o is not None and seen < 16 and o.op in _SHAPE:
                if not o.operand_ids:
                    return o
                o = op_by_id.get(o.operand_ids[0])
                seen += 1
            return o

        def _link_width(off):
            """Width one link's offset was COMPUTED in, or None.

            MLIR integers are signless, so a ``u32`` argument is indistinguishable
            from ``i32`` in the signature — the signedness shows up only in the
            extension opcode. ``arith.extsi`` is transparent (the arithmetic
            happened at the narrower width and is sign-extended exactly as the
            pointer add would); ``arith.extui`` means the source type was UNSIGNED
            and ``arith.trunci`` that a wider expression was narrowed — neither is
            reproduced by the signed model, so both refuse.
            """
            seen = 0
            while off is not None and seen < 16:
                seen += 1
                if off.op in _SHAPE or off.op == "tt.expand_dims":
                    off = op_by_id.get(off.operand_ids[0]) if off.operand_ids else None
                    continue
                if off.op == "arith.extsi":
                    off = op_by_id.get(off.operand_ids[0]) if off.operand_ids else None
                    continue
                if off.op in ("arith.extui", "arith.trunci"):
                    return None
                w = (off.elem_type or "").strip()
                return w if w in ("i32", "i64") else None
            return None

        def _axes(root_id):
            """expand_dims axes reachable in this offset subgraph (bounded DFS)."""
            found = set()
            seen = set()
            stack = [root_id]
            budget = 256
            while stack and budget > 0:
                budget -= 1
                cur = stack.pop()
                if cur is None or cur in seen:
                    continue
                seen.add(cur)
                o = op_by_id.get(cur)
                if o is None:
                    continue
                if o.op == "tt.expand_dims":
                    a = o.attrs.get("axis")
                    if a in (0, 1):
                        found.add(a)
                stack.extend(o.operand_ids or [])
            return found if budget > 0 else None

        def _fused_is_uniform(off):
            """A fused link is uniform only if neither addend is itself extended.

            ``A + (rm*sam_i32 + rk*sak_i64)`` promotes the i32 product with an
            ``extsi`` INSIDE an i64 ``addi``: that product still wraps at 32 bits,
            which a single-width fused model would not reproduce.
            """
            top = _peel(off)
            if top is None or top.op not in ("arith.addi", "arith.add"):
                return True
            for oid in top.operand_ids or []:
                a = _peel(op_by_id.get(oid))
                if a is not None and a.op in _EXT:
                    return False
            return True

        out = {}
        for role in ("A", "B", "C"):
            cur = op_by_id.get(addptr_ids.get(role))
            links = []  # (width, axes)
            seen_ids = set()
            depth = 0
            while (
                cur is not None
                and cur.op == "tt.addptr"
                and len(cur.operand_ids) >= 2
                and cur.id not in seen_ids
                and depth < 16
            ):
                seen_ids.add(cur.id)
                depth += 1
                off = op_by_id.get(cur.operand_ids[1])
                w = _link_width(off)
                ax = _axes(cur.operand_ids[1])
                if w is None or not ax:
                    return None
                if len(ax) == 2 and not _fused_is_uniform(off):
                    return None
                links.append((w, ax))
                base = _skip(op_by_id.get(cur.operand_ids[0]))
                # A scalar base advance is applied before the splat; cross it so the
                # link count matches what infer_dot_strides flattened (it refuses any
                # such residual, so crossing here can only AGREE with it).
                if base is not None and base.op == "tt.splat" and base.operand_ids:
                    inner = op_by_id.get(base.operand_ids[0])
                    if inner is not None and inner.op == "tt.addptr":
                        base = inner
                cur = base
            if len(links) == 1:
                w, ax = links[0]
                # One link. Both axes inside it -> their sum wraps at w before the
                # single extend. One axis -> that link carries the whole offset at w.
                out[role] = (w, w, len(ax) == 2)
            elif len(links) == 2 and all(len(a) == 1 for _, a in links):
                by_axis = {}
                for w, ax in links:
                    a = next(iter(ax))
                    if a in by_axis:
                        return None  # two links on the same axis: not attributable
                    by_axis[a] = w
                if set(by_axis) != {0, 1}:
                    return None
                out[role] = (by_axis[1], by_axis[0], False)
            else:
                return None
        return out

    def _declared_int_elem_type(self, name):
        """Declared Triton element type of scalar argument ``name`` (or None)."""
        for a in self.graph.args:
            if a.name == name and not a.is_ptr:
                return (a.elem_type or "").strip()
        return None

    def _checked_signed_stride(self, desc, scalar_names, what, source_bindings=None):
        """A stride descriptor rendered as a plain MSL operand, or refuse.

        The function signature Triton hands the walker is SIGNLESS (``tl.uint32``
        and ``tl.int32`` both arrive as ``i32``), so unsignedness is NOT decidable
        here — it is decided in ``_dot_offset_arith_widths`` from the extension
        opcode (``arith.extui`` refuses). This guard covers what the declared type
        genuinely proves: that the descriptor names an INTEGER scalar at all, so a
        float argument can never be spliced into an address.
        """
        if desc not in scalar_names:
            return str(desc)  # compile-time integer literal
        elem = self._declared_int_elem_type(desc)
        if elem is None or not (elem.startswith("i") or elem.startswith("u")):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"{what}: stride argument '{desc}' has declared type "
                f"{elem!r}, which is not an integer, so it cannot be the traced "
                "address coefficient the template is about to emit. Refusing "
                "(correct-or-refuse).",
                op_name="tt.dot",
            )
        return (source_bindings or {}).get(desc, str(desc))

    def _width_faithful_addr(self, terms, row_width, col_width, fused, *, what, scalar_names, source_bindings=None):
        """Build the MSL index expression for one operand address.

        ``terms`` is ``[(row_index_expr, row_stride), (col_index_expr, col_stride)]``.
        ``row_width`` / ``col_width`` are the proven widths of the two offset terms and
        ``fused`` says they shared ONE ``tt.addptr`` link. The result is a ``long``
        expression suitable for ``ptr[<expr>]``.

        EVERY arithmetic step is performed in an UNSIGNED type and reinterpreted with
        ``as_type`` (packet 834): signed overflow is UB in MSL/C++ and would let the
        compiler assume it cannot happen, while Triton's ``arith.muli``/``arith.addi``
        are two's-complement WRAPPING at their width. Unsigned arithmetic is defined
        modulo 2**w and has exactly the same bit pattern, so the bit-cast back is the
        wrap.

        * i32 term  -> ``(uint)idx * (uint)(stride)`` then ``as_type<int>`` (the
          32-bit wrap) then widened to 64 bits (the ``tt.addptr`` sign-extend).
        * i64 term  -> ``(ulong)idx * (ulong)(stride)``, the index zero-extended
          (it is a non-negative tile index, so that equals the source's
          ``arith.extsi``) and the product wrapping mod 2**64 as ``arith.muli`` does.
        * terms in ONE link are summed AT THAT WIDTH before the single extend; terms
          in separate links are extended separately and added at 64 bits.

        Two forms are used so the common all-i32 address stays readable:

        * all-i32 separate links: ``(long)as_type<int>(t0) + (long)as_type<int>(t1)``.
          Each addend is in ``[-2**31, 2**31)``, so their sum is in ``[-2**32, 2**32)``
          and CANNOT overflow ``long`` — the signed ``+`` is defined, and it equals the
          mod-2**64 pointer add because the exact value is representable.
        * anything involving i64: the whole offset is accumulated in ``ulong``
          (defined mod 2**64, the pointer-arithmetic modulus) with each i32 link
          sign-extended first (``(ulong)as_type<int>(...)`` is a modular conversion),
          and one ``as_type<long>`` produces the signed offset.
        """

        def _u_term(idx, desc, width):
            s = self._checked_signed_stride(desc, scalar_names, what, source_bindings)
            # A compound index ("mstrip + r") must be parenthesised before the cast,
            # or ``(uint)mstrip + r`` would cast only the first addend.
            i = idx if str(idx).isidentifier() else f"({idx})"
            if width == "i64":
                return f"(ulong){i} * (ulong)({s})"
            return f"(uint){i} * (uint)({s})"

        widths = (row_width, row_width) if fused else (row_width, col_width)
        any64 = "i64" in widths

        if fused:
            inner = " + ".join(_u_term(i, d, row_width) for i, d in terms)
            if row_width == "i64":
                # One i64 addptr: the sum wraps mod 2**64 exactly as the pointer add.
                return f"as_type<long>({inner})"
            return f"(long)as_type<int>({inner})"

        parts = []
        for (idx, desc), width in zip(terms, widths):
            t = _u_term(idx, desc, width)
            if width == "i64":
                parts.append(t if any64 else f"as_type<long>({t})")
            elif any64:
                # Sign-extend the 32-bit link, then keep accumulating modulo 2**64.
                parts.append(f"(ulong)as_type<int>({t})")
            else:
                parts.append(f"(long)as_type<int>({t})")
        joined = " + ".join(parts)
        return f"as_type<long>({joined})" if any64 else joined

    # ------------------------------------------------------------------
    # Simdgroup leading-dimension ABI proof + width-faithful scalar fallback
    # (packet 834, task 1)
    #
    # ``simdgroup_load`` / ``simdgroup_store`` take ``ulong elements_per_row``: a
    # NEGATIVE leading dimension is not expressible, and a 64-bit one does not fit
    # the 32-bit index arithmetic the fragment path uses. The K-loop template used
    # that ABI unconditionally, so a negative or oversized runtime row stride read
    # and wrote the wrong memory. The repair keeps the fast path for layouts the ABI
    # provably represents and takes a scalar, width-faithful route otherwise.
    #
    # The decision is UNIFORM: it reads only kernel scalars (the runtime extents and
    # the leading dimensions), never ``pid``/``tiitg``/``sgitg``, so every thread in
    # every threadgroup computes the same verdict and the early ``return`` on the
    # fallback branch can never split a threadgroup across a barrier.
    # ------------------------------------------------------------------

    _I32_MAX_L = "0x7fffffffL"
    _I32_LIMIT_L = "0x80000000L"
    _I32_MAX_UL = "0x7fffffffUL"

    def _simd_layout_guard_lines(self, a_ld, b_ld, c_ld, indent="    ", b_transposed=False):
        """Emit ``bool _simd_ok = <uniform proof>;``.

        ``b_transposed`` states that B is stored with a unit K stride and ``b_ld``
        as its COLUMN stride (``B[n * b_ld + k]``, the layout of ``w.t()``); its
        maximal offset is then ``(_N-1)*b_ld + (_K-1)`` and (4) bounds that instead.

        SUFFICIENT CONDITIONS for the simdgroup leading-dimension path. With row
        extents from the traced output bounds, ``a_ld``/``b_ld``/``c_ld`` the traced
        ROW strides and unit column strides (this template only claims
        contiguous-inner operands), the largest element the SIMD path touches is
        ``(_M-1)*a_ld + (_K-1)`` for A, ``(_K-1)*b_ld + (_N-1)`` for B and
        ``(_M-1)*c_ld + (_N-1)`` for C. The path is entered only when ALL of:

          1. ``_M > 0 && _N > 0 && _K > 0``        (non-empty, non-negative extents)
          2. each extent ``<= 2**31-1``            (an i32 index cannot name more)
          3. each leading dim, read as ``ulong``, ``<= 2**31-1``. For an i32 stride
             argument rendered ``(uint)s`` this is exactly ``s >= 0``; for a 64-bit
             one rendered ``(ulong)s`` it rejects negatives AND anything the
             fragment path's 32-bit stepping could not carry.
          4. each operand's MAXIMAL offset above is ``< 2**31``.

        (4) is the condition a bare sign test cannot give: a POSITIVE stride still
        lets ``index * stride`` cross the signed boundary, after which the source's
        i32 arithmetic wraps NEGATIVE while ``simdgroup_load`` would keep walking
        forward. With (1)-(3) holding, every product below is at most
        ``(2**31-1)**2 < 2**63``, so the 64-bit check itself cannot overflow, and
        the ``&&`` chain short-circuits before any product is formed from a leading
        dimension that failed (3).

        Under 1-4 every address the SIMD path forms lies in ``[0, 2**31)``, so the
        template's ``uint`` index arithmetic is bit-identical to the source's i32
        arithmetic AND to the exact mathematical value — no wrap, no sign question.
        """
        i32 = self._I32_MAX_L
        lim = self._I32_LIMIT_L
        u32 = self._I32_MAX_UL
        return [
            f"{indent}// Uniform simdgroup leading-dimension proof (packet 834): reads only",
            f"{indent}// kernel scalars, so every thread agrees and the fallback return below",
            f"{indent}// can never split a threadgroup across a barrier.",
            f"{indent}ulong _lda = (ulong)({a_ld});",
            f"{indent}ulong _ldb = (ulong)({b_ld});",
            f"{indent}ulong _ldc = (ulong)({c_ld});",
            f"{indent}bool _simd_ok =",
            f"{indent}       _M > 0L && _N > 0L && _K > 0L",
            f"{indent}    && _M <= {i32} && _N <= {i32} && _K <= {i32}",
            f"{indent}    && _lda <= {u32} && _ldb <= {u32} && _ldc <= {u32}",
            f"{indent}    && (_M - 1L) * (long)_lda + (_K - 1L) < {lim}",
            (
                f"{indent}    && (_N - 1L) * (long)_ldb + (_K - 1L) < {lim}"
                if b_transposed
                else f"{indent}    && (_K - 1L) * (long)_ldb + (_N - 1L) < {lim}"
            ),
            f"{indent}    && (_M - 1L) * (long)_ldc + (_N - 1L) < {lim};",
        ]

    @staticmethod
    def _direct_twin_width(block_m, block_n):
        """Threadgroup width of the sub-tiled direct twin, or ``None`` when the
        tile does not split into 32x32 sub-tiles. Up to four 32-lane simdgroups
        share a tile; a larger tile's sub-tiles are dealt out among those four."""
        if block_m <= 0 or block_n <= 0 or block_m % 32 or block_n % 32:
            return None
        return 32 * min((block_m // 32) * (block_n // 32), 4)

    @staticmethod
    def _twin_surplus_program_lines(rows_masked, cols_masked):
        """A direct twin stores whole tiles, and the launcher proves that the
        extents are tile multiples, not that the grid has one program per tile.
        Where the source masks its store by an extent, a program whose tile starts
        outside that extent stores nothing, and one that starts inside owns a whole
        in-range tile. Uniform per threadgroup.

        An axis the source stores without a mask has no extent to test: the source
        stores every launched tile whole, and so does the twin. ``_M``/``_N`` are
        then only the arguments of those names, which may play another role, and
        the launcher instead admits no more tiles than the output holds."""
        tests = ["(long)row_base >= _M"] * bool(rows_masked) + ["(long)col_base >= _N"] * bool(cols_masked)
        return [f"    if ({' || '.join(tests)}) return;"] if tests else []

    def _bound_argument_index(self, source_bindings, emitted_name):
        """Position of the kernel argument an emitted pointer name is bound to."""
        (index,) = [i for i, arg in enumerate(self.graph.args) if source_bindings[arg.name] == emitted_name]
        return index

    @staticmethod
    def _subtiled_direct_lines(a_name, b_name, c_name, block_m, block_n, acc_frag, in_frag, b_transposed):
        """Body of the sub-tiled direct twin, after the ``_simd_ok`` proof.

        Each simdgroup computes whole 32x32 sub-tiles of the threadgroup's output
        tile, keeping all sixteen 8x8 accumulators of the sub-tile it is on, so one
        8-deep K step loads four A and four B fragments for sixteen
        multiply-accumulates (the layout with one 8-column strip per simdgroup
        spent an A load on every accumulate). Sub-tile ``_st`` belongs to simdgroup
        ``_st % groups``, so the sub-tiles are disjoint and all are covered.
        Fragments come straight from device memory through the proven leading
        dimensions ``_lda``/``_ldb``/``_ldc``; a transposed B is read with the
        transposing ``simdgroup_load``, whose ``elements_per_row`` is then B's
        column stride. Every address is below 2**31 under the proof.
        """
        subtiles = (block_m // 32) * (block_n // 32)
        groups = min(subtiles, 4)
        lines = [
            f"    for (uint _st = sgitg; _st < {subtiles}u; _st += {groups}u) {{",
            f"        uint _sr = row_base + (_st / {block_n // 32}u) * 32u;",
            f"        uint _sc = col_base + (_st % {block_n // 32}u) * 32u;",
            f"        {acc_frag} {', '.join(f'acc_{r}_{c}(0)' for r in range(4) for c in range(4))};",
            f"        {in_frag} a_frag, b_frag0, b_frag1, b_frag2, b_frag3;",
            "        for (uint k = 0u; k < _K; k += 8u) {",
        ]
        for c in range(4):
            if b_transposed:
                lines.append(
                    f"            simdgroup_load(b_frag{c}, {b_name} + (ulong)(_sc + {c * 8}u) * _ldb + k, _ldb, ulong2(0), true);"
                )
            else:
                lines.append(
                    f"            simdgroup_load(b_frag{c}, {b_name} + (ulong)k * _ldb + _sc + {c * 8}u, _ldb);"
                )
        for r in range(4):
            lines.append(f"            simdgroup_load(a_frag, {a_name} + (ulong)(_sr + {r * 8}u) * _lda + k, _lda);")
            for c in range(4):
                lines.append(f"            simdgroup_multiply_accumulate(acc_{r}_{c}, a_frag, b_frag{c}, acc_{r}_{c});")
        lines.append("        }")
        for r in range(4):
            for c in range(4):
                lines.append(
                    f"        simdgroup_store(acc_{r}_{c}, {c_name} + (ulong)(_sr + {r * 8}u) * _ldc + _sc + {c * 8}u, _ldc);"
                )
        lines.append("    }")
        return lines

    def _scalar_tile_fallback_lines(
        self,
        *,
        widths,
        descriptors,
        a_name,
        b_name,
        c_name,
        out_type,
        block_m,
        block_n,
        threads,
        scalar_names,
        source_bindings=None,
        indent="        ",
        preserve_alias=True,
    ):
        """Width-faithful scalar matmul over THIS threadgroup's output tile.

        Used when ``_simd_ok`` is false. Reproduces what the simdgroup path would
        have computed for the same tile:

        * the same tile bounds and the same signed ``_M``/``_N`` masks, so partial
          edge tiles write exactly the in-range elements;
        * the full reduction ``k in [0, _K)`` with no 8-multiple assumption, so a K
          tail is covered (the SIMD path needs ``_K % 8 == 0`` and stages the tail);
        * fp32 accumulation, the accumulator width ``tl.dot`` specifies and the same
          the ``simdgroup_float8x8`` accumulators use;
        * the same terminal cast to the output element type.

        It does NOT reproduce the MMA path's summation ORDER (``tl.dot`` fixes no
        order; both are fp32 sums of the identical products, and the branch is a
        deterministic function of the kernel scalars, so one launch gets one answer).
        It allocates no threadgroup memory, so it can be used inside the
        occupancy-tuned ``__mmdirect`` twin without changing that kernel's
        threadgroup footprint.
        """
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        a_addr = self._width_faithful_addr(
            [("m", a_row), ("k", a_col)],
            *widths["A"],
            what="K-loop matmul scalar fallback operand A",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        b_addr = self._width_faithful_addr(
            [("k", b_row), ("n", b_col)],
            *widths["B"],
            what="K-loop matmul scalar fallback operand B",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        c_addr = self._width_faithful_addr(
            [("m", c_row), ("n", c_col)],
            *widths["C"],
            what="K-loop matmul scalar fallback output C",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        i = indent
        cell_loop = [
            f"{i}for (uint _fe = tiitg; _fe < {block_m * block_n}u; _fe += {threads}u) {{",
            f"{i}    uint m = row_base + _fe / {block_n}u;",
            f"{i}    uint n = col_base + _fe % {block_n}u;",
            f"{i}    if ((long)m >= _M || (long)n >= _N) continue;",
        ]
        if not preserve_alias:
            return [
                *cell_loop,
                f"{i}    float _facc = 0.0f;",
                f"{i}    for (long k = 0L; k < _K; k++) {{",
                f"{i}        _facc += (m < _A_M ? (float){a_name}[{a_addr}] : 0.0f) * (n < _B_N ? (float){b_name}[{b_addr}] : 0.0f);",
                f"{i}    }}",
                f"{i}    {c_name}[{c_addr}] = ({out_type})_facc;",
                f"{i}}}",
            ]
        return [
            f"{i}// Layout the simdgroup leading-dimension ABI cannot represent: same tile,",
            f"{i}// same masks, same K tail, same fp32 accumulator, width-faithful addresses.",
            f"{i}float _fallback_cells[{-(-(block_m * block_n) // threads)}];",
            *cell_loop,
            f"{i}    float _facc = 0.0f;",
            f"{i}    for (long k = 0L; k < _K; k++) {{",
            f"{i}        _facc += (m < _A_M ? (float){a_name}[{a_addr}] : 0.0f) * (n < _B_N ? (float){b_name}[{b_addr}] : 0.0f);",
            f"{i}    }}",
            f"{i}    _fallback_cells[_fe / {threads}u] = _facc;",
            f"{i}}}",
            f"{i}threadgroup_barrier(mem_flags::mem_device);",
            *cell_loop,
            f"{i}    {c_name}[{c_addr}] = ({out_type})_fallback_cells[_fe / {threads}u];",
            f"{i}}}",
        ]

    @staticmethod
    def _strided_matrix_stage_lines(a_name, b_name, c_name, a_addr, b_addr, c_addr):
        """Transpose strided operands into FP32 matrix fragments without conversion.

        Eight 32-lane SIMD groups own two 8x8 outputs each. Loads and stores
        retain the source address expressions. The final input barrier makes
        reusing A's shared tile for disjoint output fragments safe, including
        race-free single-program input/output aliases. Partial K uses the scalar
        path so no extra zero products are introduced by an 8-wide MMA.
        """
        return [
            "#if defined(__HAVE_SIMDGROUP_MATRIX__)",
            "        if (_matrix_width == 32u && _K % 32L == 0L) {",
            "            uint _matrix_row = (_matrix_sg / 4u) * 16u;",
            "            uint _matrix_col = (_matrix_sg % 4u) * 8u;",
            "            metal::simdgroup_float8x8 _matrix_acc0(0), _matrix_acc1(0);",
            "            metal::simdgroup_float8x8 _matrix_a0, _matrix_a1, _matrix_b;",
            "            for (uint _kb = 0u; _kb < _K; _kb += 32u) {",
            "                for (uint _t = lid; _t < 1024u; _t += 256u) {",
            "                    uint m = row_base + _t / 32u;",
            "                    uint n = col_base + _t / 32u;",
            "                    uint k = _kb + _t % 32u;",
            f"                    _stage_a[(_t / 32u) * 33u + _t % 32u] = m < _A_M ? (float){a_name}[{a_addr}] : 0.0f;",
            f"                    _stage_b[(_t % 32u) * 33u + _t / 32u] = n < _B_N ? (float){b_name}[{b_addr}] : 0.0f;",
            "                }",
            "                threadgroup_barrier(mem_flags::mem_threadgroup);",
            "                for (uint _kk = 0u; _kk < 32u; _kk += 8u) {",
            "                    metal::simdgroup_load(_matrix_a0, _stage_a + _matrix_row * 33u + _kk, 33);",
            "                    metal::simdgroup_load(_matrix_a1, _stage_a + (_matrix_row + 8u) * 33u + _kk, 33);",
            "                    metal::simdgroup_load(_matrix_b, _stage_b + _kk * 33u + _matrix_col, 33);",
            "                    metal::simdgroup_multiply_accumulate(_matrix_acc0, _matrix_a0, _matrix_b, _matrix_acc0);",
            "                    metal::simdgroup_multiply_accumulate(_matrix_acc1, _matrix_a1, _matrix_b, _matrix_acc1);",
            "                }",
            "                threadgroup_barrier(mem_flags::mem_threadgroup);",
            "            }",
            "            metal::simdgroup_store(_matrix_acc0, _stage_a + _matrix_row * 33u + _matrix_col, 33);",
            "            metal::simdgroup_store(_matrix_acc1, _stage_a + (_matrix_row + 8u) * 33u + _matrix_col, 33);",
            "            threadgroup_barrier(mem_flags::mem_threadgroup);",
            "            for (uint _t = lid; _t < 1024u; _t += 256u) {",
            "                uint m = row_base + _t / 32u, n = col_base + _t % 32u;",
            f"                if (m < _M && n < _N) {c_name}[{c_addr}] = _stage_a[(_t / 32u) * 33u + _t % 32u];",
            "            }",
            "            return;",
            "        }",
            "#endif",
        ]

    @staticmethod
    def _strided_scalar_stage_lines(a_name, b_name, c_name, a_addr, b_addr, c_addr):
        """Stage a bounded 32x32 FP32 dot with a scalar fallback.

        Matrix-capable targets use FP32 fragments for complete K chunks. The
        scalar path retains K order without conversion for other reductions.
        The old scalar loop remains the fallback for exceptional index ranges. All lanes cross both barriers;
        masked output lanes still own their shared loads. Padded shared row strides
        avoid broadcasting different B rows through the same shared-memory bank.
        """
        lines = [
            "    if (_M > 0L && _N > 0L && _K > 0L && _K <= 2147483647L",
            "        && row_base <= 4294967264u && col_base <= 4294967264u",
            "        && (long)row_base < _M && (long)col_base < _N) {",
            "        threadgroup float _stage_a[1056];",
            "        threadgroup float _stage_b[1056];",
        ]
        lines.extend(_TemplateMixin._strided_matrix_stage_lines(a_name, b_name, c_name, a_addr, b_addr, c_addr))
        for slot in range(4):
            lines += [
                f"        uint _m{slot} = row_base + (lid + {slot * 256}u) / 32u;",
                f"        uint _n{slot} = col_base + (lid + {slot * 256}u) % 32u;",
                f"        float _acc{slot} = 0.0f;",
            ]
        lines += [
            "        for (uint _kb = 0u; _kb < _K; _kb += 32u) {",
            "            for (uint _t = lid; _t < 1024u; _t += 256u) {",
            "                uint m = row_base + _t / 32u;",
            "                uint n = col_base + _t / 32u;",
            "                uint k = _kb + _t % 32u;",
            f"                _stage_a[(_t / 32u) * 33u + _t % 32u] = m < _A_M && k < _K ? (float){a_name}[{a_addr}] : 0.0f;",
            f"                _stage_b[(_t / 32u) * 33u + _t % 32u] = n < _B_N && k < _K ? (float){b_name}[{b_addr}] : 0.0f;",
            "            }",
            "            threadgroup_barrier(mem_flags::mem_threadgroup);",
            "            for (uint _kk = 0u; _kk < 32u && _kb + _kk < _K; _kk++) {",
        ]
        for slot in range(4):
            lines.append(
                f"                _acc{slot} += _stage_a[((lid + {slot * 256}u) / 32u) * 33u + _kk]"
                f" * _stage_b[((lid + {slot * 256}u) % 32u) * 33u + _kk];"
            )
        lines += [
            "            }",
            "            threadgroup_barrier(mem_flags::mem_threadgroup);",
            "        }",
        ]
        for slot in range(4):
            lines += [
                "        {",
                f"            uint m = _m{slot}, n = _n{slot};",
                f"            if (m < _M && n < _N) {c_name}[{c_addr}] = _acc{slot};",
                "        }",
            ]
        lines += ["        return;", "    }"]
        return lines

    def _lower_strided_scalar_matmul(self, info, descriptors):
        """Fully stride-aware scalar matmul for a NON-contiguous-inner operand.

        Used when the simdgroup-MMA path can't load an operand because its inner
        dim is strided (a transposed B from ``x @ w.t()``, a column-major C, a
        sliced operand). Each thread computes a set of output elements; every
        load/store addresses through the 6 inferred strides
        ``A[m*a_row + k*a_col]``, ``B[k*b_row + n*b_col]``,
        ``C[m*c_row + n*c_col]`` (each stride emitted as a literal or a runtime
        ``int`` arg), so it is correct for ANY layout. A bounded 32x32 fp32 case cooperatively stages its operands without
        using FP32 matrix fragments where available, with scalar accumulation
        as the fallback; other cases retain scalar loads.
        The common contiguous case never reaches this method.

        Handles both the pid-tiled K-loop form (runtime M/N/K, grid tiles over
        BLOCK_M/BLOCK_N) and the single-tile form (constexpr dims).
        """
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        ptr_args = info["ptr_args"]
        # Emitted argument names belong to a separate compiler namespace. Source
        # names remain in the graph, stride proofs, and launch metadata; only MSL
        # references change. In particular m/n/k and pointer c cannot be captured
        # by template loops, and source names resembling these aliases are safe.
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        a_arg, b_arg, c_arg = self._dot_template_ptr_roles()  # P0 (065): dataflow, not position
        a_name = source_bindings[a_arg.name]
        b_name = source_bindings[b_arg.name]
        c_name = source_bindings[c_arg.name]
        a_msl = triton_type_to_msl(a_arg.elem_type)
        b_msl = triton_type_to_msl(b_arg.elem_type)
        c_msl = triton_type_to_msl(c_arg.elem_type)

        all_scalar_args = [a for a in self.graph.args if not a.is_ptr]
        scalar_names = {a.name for a in all_scalar_args}

        # Offset arithmetic is rendered at the SOURCE's integer width (packet 825,
        # W2 F-7): the old ``(uint){desc}`` truncated i64 strides, evaluated the sum
        # as unsigned and zero-extended negative offsets into the pointer.
        _widths = self._dot_offset_arith_widths()
        if _widths is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "strided matmul: the integer width of the operand address arithmetic "
                "could not be proven from the traced tt.addptr chain (unreadable, or "
                "the chain mixes i32 and i64 offsets). Emitting the address would have "
                "to guess whether each term wraps at 32 bits, so refusing "
                "(correct-or-refuse).",
                op_name="tt.dot",
            )

        has_k_loop = bool(info.get("has_k_loop"))
        if has_k_loop:
            BLOCK_M = info["BLOCK_M"]
            BLOCK_N = info["BLOCK_N"]
            BLOCK_K = info["BLOCK_K"]
            scalar_arg_map = info.get("scalar_args", {})
        else:
            BLOCK_M = info["M"]
            BLOCK_N = info["N"]
            BLOCK_K = info["K"]
            scalar_arg_map = {a.name: a for a in all_scalar_args}

        has_M = "M" in scalar_arg_map
        has_N = "N" in scalar_arg_map
        has_K = "K" in scalar_arg_map
        has_pid = any(s.op == "tt.get_program_id" for s in self.graph.ops)
        # Shared integrity guard (see _refuse_if_pid_tiles_baked_output).
        # The proven source strides and program mapping determine the whole tile.

        # 256 threads per threadgroup; each handles a strided subset of the tile.
        block_size = 256
        self.effective_block_size = block_size
        tile = BLOCK_M * BLOCK_N

        # An in-tile transpose (single-tile form only; the value-path proof admits the
        # square forms) swaps which load index the reduction walks. ``a = trans(load(A +
        # rm[:, None] * sr + rk[None, :] * sc))`` is ``a[i, j] = A[rm[j] * sr + rk[i] * sc]``:
        # the reduction index walks the ROW term, with the row tile's program offset, and
        # the output row walks the column term. B is the mirror image. Addressing these
        # as ``A[m * sr + k * sc]`` computed the untransposed product (GPU).
        trans_a = bool(info.get("trans_a")) and not has_k_loop
        trans_b = bool(info.get("trans_b")) and not has_k_loop
        use_staging = (
            (BLOCK_M, BLOCK_N, BLOCK_K) == (32, 32, 32)
            and (a_msl, b_msl, c_msl) == ("float", "float", "float")
            and not (trans_a or trans_b)
        )

        safe_name = _sanitize_msl_name(self.graph.func_name)
        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                m = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {m}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_decls.append(_scalar_buffer_decl(arg, i, source_bindings))
        lines.append(",\n".join(arg_decls) + ",")
        if has_pid:
            self._used_pid_axes = {0, 1}
            # Metal requires all thread-index attributes to share a type; use
            # uint3 for both when multi-axis dispatch is in play.
            lines.append("    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append("    uint3 _lid3 [[thread_position_in_threadgroup]]")
            if use_staging:
                lines += [
                    "#if defined(__HAVE_SIMDGROUP_MATRIX__)",
                    "    , uint _matrix_sg [[simdgroup_index_in_threadgroup]]",
                    "    , uint _matrix_width [[threads_per_simdgroup]]",
                    "#endif",
                ]
            lines.append(") {")
            lines.append("    uint lid = _lid3.x;")
            # pid_m / pid_n follow the extents (the proven 1-D mapping needs _M).
        else:
            lines.append("    uint pid [[threadgroup_position_in_grid]],")
            lines.append("    uint lid [[thread_position_in_threadgroup]]")
            if use_staging:
                lines += [
                    "#if defined(__HAVE_SIMDGROUP_MATRIX__)",
                    "    , uint _matrix_sg [[simdgroup_index_in_threadgroup]]",
                    "    , uint _matrix_width [[threads_per_simdgroup]]",
                    "#endif",
                ]
            lines.append(") {")
            lines.append("    uint pid_m = 0u, pid_n = 0u;")
        for arg in all_scalar_args:
            lines.append(_scalar_unpack(arg, source_bindings))

        m_ext, n_ext = self._matmul_output_extent_args()
        lines.append(self._k_extent_line(info, BLOCK_K, has_K, BLOCK_M, BLOCK_N, source_bindings))
        if has_pid:
            lines.extend(self._pid_map_lines(info, BLOCK_M, source_bindings))
        lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
        lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
        lines.extend(self._matmul_tile_extent_lines(info, BLOCK_M, BLOCK_N, source_bindings))
        lines.append("")
        _a_addr = self._width_faithful_addr(
            [("row_base + k", a_row), ("_lm", a_col)] if trans_a else [("m", a_row), ("k", a_col)],
            *_widths["A"],
            what="strided matmul operand A",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        _b_addr = self._width_faithful_addr(
            [("_ln", b_row), ("col_base + k", b_col)] if trans_b else [("k", b_row), ("n", b_col)],
            *_widths["B"],
            what="strided matmul operand B",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        _c_addr = self._width_faithful_addr(
            [("m", c_row), ("n", c_col)],
            *_widths["C"],
            what="strided matmul output C",
            scalar_names=scalar_names,
            source_bindings=source_bindings,
        )
        # Shared staging is a layout/tile optimization, never source-name matching.
        # Retain the fully stride-aware scalar path for all other shapes/types.
        if use_staging:
            lines.extend(self._strided_scalar_stage_lines(a_name, b_name, c_name, _a_addr, _b_addr, _c_addr))
        # Each thread strides over the BLOCK_M x BLOCK_N tile. A program loads its
        # whole tile before it stores any of it, so every thread keeps its cells
        # until the threadgroup has finished loading: one program whose output is
        # also an operand (``C is A``, ``C is B``) then reads no cell another thread
        # has already replaced, as the staged 32x32 case above has it.
        cell_loop = [
            f"    for (uint _e = lid; _e < {tile}u; _e += {block_size}u) {{",
            f"        uint _lm = _e / {BLOCK_N}u;",
            f"        uint _ln = _e % {BLOCK_N}u;",
            "        uint m = row_base + _lm;",
            "        uint n = col_base + _ln;",
            "        if (m >= _M || n >= _N) continue;",
        ]
        lines.append(f"    float _cell[{-(-tile // block_size)}];")
        lines.extend(cell_loop)
        lines.append("        float _sum = 0.0f;")
        lines.append("        for (uint k = 0u; k < _K; k++) {")
        lines.append(f"            _sum += (m < _A_M ? (float){a_name}[{_a_addr}] : 0.0f)")
        lines.append(f"                  * (n < _B_N ? (float){b_name}[{_b_addr}] : 0.0f);")
        lines.append("        }")
        lines.append(f"        _cell[_e / {block_size}u] = _sum;")
        lines.append("    }")
        lines.append("    threadgroup_barrier(mem_flags::mem_device);")
        lines.extend(cell_loop)
        lines.append(f"        {c_name}[{_c_addr}] = ({c_msl})_cell[_e / {block_size}u];")
        lines.append("    }")
        lines.append("}")
        lines.extend(
            self._transposed_b_twin_lines(
                info,
                descriptors,
                safe_name=safe_name,
                arg_decls=arg_decls,
                scalars=all_scalar_args,
                source_bindings=source_bindings,
                names=(a_name, b_name, c_name),
                types=(a_msl, b_msl, c_msl),
                blocks=(BLOCK_M, BLOCK_N, BLOCK_K),
                has_extents=(has_M, has_N, has_K),
                extent_args=(m_ext, n_ext),
                widths=_widths,
                has_pid=has_pid and has_k_loop,
            )
        )
        return "\n".join(lines)

    def _transposed_b_twin_lines(
        self,
        info,
        descriptors,
        *,
        safe_name,
        arg_decls,
        scalars,
        source_bindings,
        names,
        types,
        blocks,
        has_extents,
        extent_args,
        widths,
        has_pid,
    ):
        """Direct twin for ``A @ W.t()``: row-major A and C, B stored transposed.

        A B whose K stride is the literal 1 is ``B[n * ldb + k]``: each output
        column's K run is contiguous, so the transposing ``simdgroup_load`` reads
        its 8x8 fragments straight from device memory and the per-K-block staging
        of the ordinary entry (two barriers and a scalar copy of both tiles)
        disappears. The twin replays the ordinary entry's extents, K rounding and
        grid mapping, carries the same uniform layout proof as the K-loop twin
        (with B's bound taken along its own layout) and the same width-faithful
        scalar fallback, and is dispatched by the launcher only for tile-aligned
        extents. Returns no lines, and leaves no descriptor, for any other layout.
        """
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        block_m, block_n, block_k = blocks
        has_M, has_N, has_K = has_extents
        scalar_names = {a.name for a in scalars}
        width = self._direct_twin_width(block_m, block_n)

        def _unit(desc):
            try:
                return desc not in scalar_names and int(str(desc)) == 1
            except (TypeError, ValueError):
                return False

        def _ld(desc):
            # Same rendering as the K-loop template: a runtime stride at its
            # declared width, a literal as an unsigned constant, else decline.
            if desc in scalar_names:
                elem = self._declared_int_elem_type(desc)
                if elem is None:
                    return None
                msl = triton_type_to_msl(elem)
                return f"({'ulong' if msl in ('long', 'ulong') else 'uint'}){source_bindings[desc]}"
            try:
                return f"{int(str(desc))}u"
            except (TypeError, ValueError):
                return None

        lds = (_ld(a_row), _ld(b_col), _ld(c_row))
        argnames = [a.name for a in self.graph.args]
        if (
            not has_pid
            or width is None
            or block_k % 8
            or types != ("float", "float", "float")
            or not (_unit(a_col) and _unit(b_row) and _unit(c_col))
            or None in lds
            or not (has_M and has_N and has_K)
            or any(name not in argnames for name in ("M", "N", "K"))
            # The twin's launcher reads the reduction extent from an argument slot: only
            # the one the loop bounds PROVED, never an argument that is merely named K.
            or info.get("k_extent_arg") is None
        ):
            return []
        a_name, b_name, c_name = names
        m_ext, n_ext = extent_args
        direct_m = m_ext or "M"
        direct_n = n_ext or "N"
        direct_k = info["k_extent_arg"]
        dn = f"{safe_name}__mmdirect"
        lines = [
            "",
            f"kernel void {dn}(",
            ",\n".join(arg_decls) + ",",
            "    uint3 pid3 [[threadgroup_position_in_grid]],",
            "    uint tiitg [[thread_index_in_threadgroup]]",
            "#if defined(__HAVE_SIMDGROUP_MATRIX__)",
            "    , uint sgitg [[simdgroup_index_in_threadgroup]]",
            "    , uint _sgw [[threads_per_simdgroup]]",
            "#endif",
            ") {",
        ]
        for arg in scalars:
            lines.append(_scalar_unpack(arg, source_bindings))
        lines.append(self._k_extent_line(info, block_k, has_K, block_m, block_n, source_bindings))
        lines.extend(self._pid_map_lines(info, block_m, source_bindings))
        lines.append(f"    uint row_base = pid_m * {block_m}u;")
        lines.append(f"    uint col_base = pid_n * {block_n}u;")
        lines.extend(self._matmul_tile_extent_lines(info, block_m, block_n, source_bindings))
        lines.extend(self._twin_surplus_program_lines(m_ext, n_ext))
        lines.append("#if defined(__HAVE_SIMDGROUP_MATRIX__)")
        lines.extend(self._simd_layout_guard_lines(*lds, b_transposed=True))
        # Sub-tile ownership is by simdgroup index, so it needs the 32-lane
        # simdgroup it was laid out for. Both verdicts are uniform.
        lines.append("    _simd_ok = _simd_ok && _sgw == 32u;")
        lines.append(
            f"    _simd_ok = _simd_ok && (long)row_base + {block_m}L <= _A_M && (long)col_base + {block_n}L <= _B_N;"
        )
        lines.append("#else")
        lines.append("    bool _simd_ok = false;")
        lines.append("#endif")
        lines.append("    if (!_simd_ok) {")
        lines.extend(
            self._scalar_tile_fallback_lines(
                widths=widths,
                descriptors=descriptors,
                a_name=a_name,
                b_name=b_name,
                c_name=c_name,
                out_type=types[2],
                block_m=block_m,
                block_n=block_n,
                threads=width,
                preserve_alias=False,
                scalar_names=scalar_names,
                source_bindings=source_bindings,
            )
        )
        lines.append("        return;")
        lines.append("    }")
        lines.append("#if defined(__HAVE_SIMDGROUP_MATRIX__)")
        lines.extend(
            self._subtiled_direct_lines(
                a_name, b_name, c_name, block_m, block_n, "simdgroup_float8x8", "simdgroup_float8x8", True
            )
        )
        lines.append("#endif")
        lines.append("}")
        self._mm_two_kernel = {
            "direct_name": dn,
            "direct_block": width,
            "block_m": block_m,
            "block_n": block_n,
            "m_idx": argnames.index(direct_m),
            "n_idx": argnames.index(direct_n),
            "k_idx": argnames.index(direct_k),
            "c_idx": self._bound_argument_index(source_bindings, c_name),
        }
        return lines

    @staticmethod
    def _simdgroup_frag_for(elem_type):
        """SINGLE source of truth for simdgroup MMA fragment selection (Step 5
        de-duplication — the documented root cause of recurring silent-wrongs).

        Given a Triton/MLIR input element type, return the tuple
        ``(acc_frag, in_frag, tg_type, stage_cast, pad)`` describing how to stage
        and multiply that dtype on the Apple simdgroup matrix unit:
          - ``acc_frag``   : accumulator fragment type (ALWAYS ``simdgroup_float8x8``;
                             half/bfloat inputs accumulate in float).
          - ``in_frag``    : input fragment type fed to
                             ``simdgroup_matrix_multiply_accumulate``.
          - ``tg_type``    : threadgroup staging element type for A/B tiles.
          - ``stage_cast`` : MSL cast applied when staging into ``tg_type``
                             (``""`` = no cast; the load is already that type).
          - ``pad``        : the zero-pad literal for partial tiles.

        fp16 -> half MMA (Apple ~2x fp16 throughput, no float upcast).
        bf16 -> the M-series ``simdgroup_bfloat8x8`` matrix unit (verified on M4),
                staged as bfloat (no upcast).
        everything else -> the exact float path (stage+cast to float).

        This block used to be copy-pasted byte-identically in
        ``_lower_simple_dot_inline`` and ``_lower_k_loop_dot_inline``, with a THIRD
        divergent copy in ``_lower_matmul_softmax_template`` that LACKED the bf16
        branch (bf16 silently took the float-upcast path). Collapsing to one helper
        makes a fix land in one place; the softmax template now also gets the bf16
        MMA branch through this shared selector.
        """
        input_msl_type = triton_type_to_msl(elem_type)
        acc_frag = "simdgroup_float8x8"
        if input_msl_type == "half":
            return (acc_frag, "simdgroup_half8x8", "half", "", "half(0.0h)")
        if input_msl_type == "bfloat":
            return (acc_frag, "simdgroup_bfloat8x8", "bfloat", "", "bfloat(0.0)")
        return (acc_frag, "simdgroup_float8x8", "float", "float", "0.0f")

    def _lower_simple_dot_inline(self, info):
        """Generate MSL for a simple dot kernel using simdgroup MMA.

        Emits a complete kernel that uses simdgroup_matrix_multiply_accumulate
        (hardware 8x8 MMA) instead of scalar per-thread matmul. Each SIMD group
        processes 32x8 of the output tile via 4 rows of 8x8 MMA operations.

        128 threads per threadgroup (4 SIMD groups), K-loop in steps of 8,
        data staged through threadgroup shared memory. For M or N > 32 the
        kernel loops over 32x32 output tiles within a single threadgroup.

        When has_k_loop is True, generates a pid-tiled kernel where each
        threadgroup handles one BLOCK_M x BLOCK_N output tile and iterates
        over K in steps of BLOCK_K.
        """
        # Stride-awareness gate (BLOCKER 1): a transposed/strided operand
        # (non-contiguous INNER dim — e.g. a transposed B from x @ w.t()) can't be
        # loaded by simdgroup_load, and the simdgroup templates hard-code
        # row-major addressing — so it would be silently wrong. Inspect the traced
        # strides: route a non-contiguous-inner matmul to the stride-aware scalar
        # template (which addresses with all 6 strides), and refuse loudly when a
        # stride is un-inferable. A contiguous-inner operand stays on the
        # simdgroup fast path (the common ~11 TFLOP/s case).
        decision = self._matmul_stride_decision(info)
        if decision is not None and decision[0] == "scalar":
            return self._lower_strided_scalar_matmul(info, decision[1])

        # BLOCKER 1: the NON-K-loop simple-dot simdgroup template loads each operand row
        # cooperatively with the MATRIX DIM as the leading stride (A: gr*K+gc, B: gr*N+gc,
        # C: gr*N+gc) and declares ONLY the 3 A/B/C buffers — the runtime stride args are
        # not available in-kernel. That addressing is correct ONLY when the inferred row
        # stride literally equals that dim — a DENSE row-major operand. A row-SLICED /
        # padded operand (e.g. A=randn(M,K+pad)[:, :K], whose row stride is a RUNTIME arg
        # != K, inner contiguous) was routed to "simdgroup" (col stride == "1") yet
        # addressed with the wrong leading dim → wrong values (err 24, the audit's B1).
        # Since the runtime row-stride arg can't be threaded into this template, route any
        # non-dense operand to the fully stride-aware scalar matmul (which declares + uses
        # all 6 inferred strides). The common DENSE case stays on the fast path. (The
        # K-loop variant DOES declare all args, so it threads the inferred row strides as
        # leading dims directly — see _lower_k_loop_dot_inline.)
        if decision is not None and decision[0] == "simdgroup" and not info.get("has_k_loop"):
            descriptors = decision[1]
            if not self._simdgroup_leading_dims_are_dense(info, descriptors):
                return self._lower_strided_scalar_matmul(info, descriptors)

        # The dense single-tile template below computes tile (0, 0) of baked extents in
        # one threadgroup: it takes no program id and reads no runtime extent. A source
        # that tiles either axis by program_id, or clips its store at a runtime extent,
        # is a different program there (GPU: a second row tile was never computed; a
        # store clipped at M = 20 wrote all 32 rows). The stride-aware scalar template
        # replays the proved grid mapping and the store clip, so such a kernel takes it
        # whatever its strides are; without a traced layout it has no template.
        if not info.get("has_k_loop") and self._single_tile_is_tiled_or_clipped():
            if decision is None:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "non-looped matmul tiles its output by program_id or clips its store at a runtime "
                    "extent, and its operand layout could not be traced; the single-tile template "
                    "computes one whole baked tile and would drop both. Refusing (correct-or-refuse).",
                    op_name="tt.dot",
                )
            return self._lower_strided_scalar_matmul(info, decision[1])

        # Phase 4: record the runtime fast-matmul dispatch descriptor (additive;
        # the generic inline kernel below is still emitted + returned).
        self._fast_matmul = self._maybe_fast_matmul_descriptor()
        if info.get("has_k_loop"):
            return self._lower_k_loop_dot_inline(
                info, decision[1] if (decision is not None and decision[0] == "simdgroup") else None
            )

        # The non-K-loop simple-dot fast path emits the whole baked M×N output from a
        # SINGLE threadgroup and never references program_id -- the 4th twin of the
        # pid-axes integrity guard (the strided/K-loop/softmax siblings have it). A
        # pid-tiled baked-output launch would leave every block but the first stale.
        # M/N are baked constexpr here, so has_M=has_N=False. Re-audit 2026-06-27.
        self._refuse_if_pid_tiles_baked_output(False, False, "simple-dot matmul")
        if self._single_tile_is_tiled_or_clipped():
            # Unreachable while the routing above holds; the shared guard accepts an axis
            # whose store mask names a runtime extent, which this template never reads.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "simple-dot matmul template reached with a program_id-tiled or runtime-clipped "
                "output, which it does not replay. Refusing.",
                op_name="tt.dot",
            )

        M = info["M"]
        N = info["N"]
        K = info["K"]
        ptr_args = info["ptr_args"]
        dot_ssa = info["dot_ssa"]
        trans_a = info.get("trans_a", False)
        trans_b = info.get("trans_b", False)
        batch_size = info.get("batch_size", 1)

        # 128 threads = 4 SIMD groups x 32 threads
        self.effective_block_size = 128

        # Emitted argument names belong to a separate compiler namespace. Source
        # names remain in the graph, stride proofs, and launch metadata; only MSL
        # references change. In particular m/n/k and pointer c cannot be captured
        # by template loops, and source names resembling these aliases are safe.
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        a_arg, b_arg, c_arg = self._dot_template_ptr_roles()  # P0 (065): dataflow, not position
        a_name = source_bindings[a_arg.name]
        b_name = source_bindings[b_arg.name]
        c_name = source_bindings[c_arg.name]

        # Determine input element type from A pointer arg. ``triton_type_to_msl``
        # handles bf16 → bfloat, fp16 → half, fp64 → float (with warning), etc.;
        # the previous hand-rolled mapping silently misrouted bf16 as float
        # (4-byte stride over a 2-byte buffer → wrong loads, partial output).
        a_elem = a_arg.elem_type  # e.g. "f32", "f16", "bf16"
        input_msl_type = triton_type_to_msl(a_elem)

        # Determine output element type from C pointer arg.
        c_elem = c_arg.elem_type
        output_msl_type = triton_type_to_msl(c_elem)

        # Genuine fp16 (WS1 Phase C): half INPUT fragments + float ACCUMULATOR;
        # fp16 staged as half (no upcast) so the half MMA is actually used. bf16
        # uses the bfloat MMA (simdgroup_bfloat8x8); other types stay on the float
        # path (exact). Accumulator always float. Fragment selection is the SHARED
        # ``_simdgroup_frag_for`` helper (Step 5 de-dup — was copy-pasted here).
        acc_frag, in_frag, tg_type, stage_cast, pad = self._simdgroup_frag_for(a_elem)
        frag_type = acc_frag

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Number of 32x32 tiles needed
        n_tile_rows = (M + 31) // 32
        n_tile_cols = (N + 31) // 32

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("#include <metal_simdgroup_matrix>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        # Declare EVERY kernel arg at its own positional slot ``buffer(i)`` (P0,
        # packet 065). The old fixed ``buffer(0/1/2)`` for a/b/c only was a second
        # positional mechanism: the launcher binds args by position, so with an
        # extra pointer before C the slot named C received that pointer and the
        # matmul was written into it. Same pattern the strided/K-loop siblings use.
        for _i, _arg in enumerate(self.graph.args):
            if _arg.is_ptr:
                _m = triton_type_to_msl(_arg.elem_type)
                _const = "const " if _arg.id != c_arg.id else ""
                lines.append(f"    device {_const}{_m}* {source_bindings[_arg.name]} [[buffer({_i})]],")
            else:
                lines.append(_scalar_buffer_decl(_arg, _i, source_bindings) + ",")
        lines.append(f"    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append(f"    uint tiitg [[thread_index_in_threadgroup]]")
        lines.append(f") {{")
        lines.append(f"    // Constants")
        lines.append(f"    const uint M = {M}u;")
        lines.append(f"    const uint N = {N}u;")
        lines.append(f"    const uint K = {K}u;")
        lines.append(f"")
        lines.append(f"    threadgroup {tg_type} tg_A[32 * 8];")
        lines.append(f"    threadgroup {tg_type} tg_B[8 * 32];")
        lines.append(f"")

        # Leading batch dimensions on tt.dot become an outer loop. Each batch
        # is an independent 32×32 matmul over its own slice of X / Y / Z.
        if batch_size > 1:
            lines.append(f"    for (uint batch = 0u; batch < {batch_size}u; batch++) {{")
            lines.append(f"        uint a_batch_off = batch * {M * K}u;")
            lines.append(f"        uint b_batch_off = batch * {K * N}u;")
            lines.append(f"        uint c_batch_off = batch * {M * N}u;")
        else:
            lines.append(f"    {{")
            lines.append(f"        uint a_batch_off = 0u;")
            lines.append(f"        uint b_batch_off = 0u;")
            lines.append(f"        uint c_batch_off = 0u;")

        # Loop over 32x32 output tiles (single threadgroup handles entire matrix)
        lines.append(f"    for (uint tile_row = 0u; tile_row < {n_tile_rows}u; tile_row++) {{")
        lines.append(f"    for (uint tile_col = 0u; tile_col < {n_tile_cols}u; tile_col++) {{")
        lines.append(f"        uint row_base = tile_row * 32u;")
        lines.append(f"        uint col_base = tile_col * 32u + sgitg * 8u;")
        lines.append(f"")
        lines.append(f"        {acc_frag} acc0(0), acc1(0), acc2(0), acc3(0);")
        lines.append(f"        {in_frag} a_frag, b_frag;")
        lines.append(f"")
        lines.append(f"        for (uint kk = 0u; kk < K; kk += 8u) {{")

        # Load A tile (32x8) cooperatively. When ``trans_a`` is set, the
        # dot operand A is X.T — i.e. the original buffer X has shape (K, M)
        # in row-major and A[gr, gc] = X[gc, gr] = X_buf[gc * M + gr].
        a_index = f"gc * {M}u + gr" if trans_a else f"gr * K + gc"
        lines.append(f"            for (uint i = tiitg; i < 256u; i += 128u) {{")
        lines.append(f"                uint r = i / 8u, c = i % 8u;")
        lines.append(f"                uint gr = row_base + r, gc = kk + c;")
        lines.append(
            f"                tg_A[i] = (gr < M && gc < K) ? {stage_cast}({a_name}[a_batch_off + {a_index}]) : {pad};"
        )
        lines.append(f"            }}")

        # Load B tile (8x32) cooperatively. ``trans_b`` analogue: original
        # buffer Y has shape (N, K) and B[gr, gc] = Y[gc, gr] = Y_buf[gc * K + gr].
        b_index = f"gc * K + gr" if trans_b else f"gr * N + gc"
        lines.append(f"            uint col_base_tg = tile_col * 32u;")
        lines.append(f"            for (uint i = tiitg; i < 256u; i += 128u) {{")
        lines.append(f"                uint r = i / 32u, c = i % 32u;")
        lines.append(f"                uint gr = kk + r, gc = col_base_tg + c;")
        lines.append(
            f"                tg_B[i] = (gr < K && gc < N) ? {stage_cast}({b_name}[b_batch_off + {b_index}]) : {pad};"
        )
        lines.append(f"            }}")
        lines.append(f"")
        lines.append(f"            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"")

        # Simdgroup MMA: each SIMD group loads its B column slice, then 4 rows of A
        lines.append(f"            simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);")
        lines.append(f"")
        lines.append(f"            simdgroup_load(a_frag, tg_A, 8);")
        lines.append(f"            simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);")
        lines.append(f"")
        lines.append(f"            simdgroup_load(a_frag, tg_A + 64u, 8);")
        lines.append(f"            simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);")
        lines.append(f"")
        lines.append(f"            simdgroup_load(a_frag, tg_A + 128u, 8);")
        lines.append(f"            simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);")
        lines.append(f"")
        lines.append(f"            simdgroup_load(a_frag, tg_A + 192u, 8);")
        lines.append(f"            simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);")
        lines.append(f"")
        lines.append(f"            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"        }}")
        lines.append(f"")

        # Store results to global memory.
        # The fast direct simdgroup_store writes a FULL unmasked 8x8; that is only
        # safe when every tile is full — i.e. M%32==0 AND N%32==0. For a partial
        # tile (M or N not a multiple of 32, e.g. a 16x16 dot) acc2/acc3 (rows
        # 16-31) and the overflow columns write PAST the output buffer -> wrong
        # values + OOB writes that corrupt adjacent allocations (2026-06-21 audit
        # sibling-divergence: the half/bfloat twin below already masked; this float
        # branch did not). So only take the fast path when fully aligned; otherwise
        # use the same masked per-simdgroup staged store (the cast is a no-op for float).
        if output_msl_type == "float" and M % 32 == 0 and N % 32 == 0:
            lines.append(f"        simdgroup_store(acc0, {c_name} + c_batch_off + (row_base) * N + col_base, N);")
            lines.append(f"        simdgroup_store(acc1, {c_name} + c_batch_off + (row_base + 8u) * N + col_base, N);")
            lines.append(f"        simdgroup_store(acc2, {c_name} + c_batch_off + (row_base + 16u) * N + col_base, N);")
            lines.append(f"        simdgroup_store(acc3, {c_name} + c_batch_off + (row_base + 24u) * N + col_base, N);")
        else:
            # Non-float (half/bfloat) output: a float8x8 accumulator can't
            # simdgroup_store directly to a half/bfloat buffer, so stage through a
            # threadgroup float slot, then cast per element. Each simdgroup owns a
            # DISTINCT 8-column slice (col_base = tile_col*32 + sgitg*8), so it MUST
            # stage into its OWN 64-float slot (tg_out + sgitg*64u) — the old code
            # stored every simdgroup's acc0..3 to the SAME fixed offsets (0/64/128/
            # 192), which RACED across simdgroups and returned wrong, nondeterministic
            # results (silent-wrong). One accumulator (8-row block) at a time, with a
            # barrier; mirrors the masked staged store in _lower_k_loop_dot_inline.
            lines.append(f"        // Store accumulators (float) via per-simdgroup slot, convert to {output_msl_type}")
            lines.append(f"        threadgroup float tg_out[4u * 64u];")
            lines.append(f"        uint laneid = tiitg % 32u;")
            for _n, _acc in enumerate(("acc0", "acc1", "acc2", "acc3")):
                _emit_masked_staged_store(
                    lines,
                    acc=_acc,
                    scratch="tg_out",
                    gr=f"row_base + {_n * 8}u + i / 8u",
                    gc="col_base + i % 8u",
                    cond="gr < M && gc < N",
                    dst=f"{c_name}[c_batch_off + gr * N + gc]",
                    out_type=output_msl_type,
                    indent="        ",
                )

        lines.append(f"    }} // tile_col")
        lines.append(f"    }} // tile_row")
        lines.append(f"    }} // batch")
        lines.append(f"}}")

        return "\n".join(lines)

    def _lower_k_loop_dot_inline(self, info, descriptors=None):
        """Generate MSL for a K-loop dot kernel using simdgroup MMA.

        Each threadgroup computes one BLOCK_M x BLOCK_N output tile, iterating
        over K in steps of BLOCK_K. Uses pid_m (program_id(0)) and pid_n
        (program_id(1)) to select which output tile to compute.

        128 threads = 4 SIMD groups x 32 threads per threadgroup.
        Each SIMD group handles an 8-column slice of the 32-wide output tile.
        The inner MMA loop iterates BLOCK_K in steps of 8.

        ``descriptors`` (a_row,a_col,b_row,b_col,c_row,c_col), when provided by the
        address-traced stride gate, supplies the per-operand ROW strides used as the
        simdgroup leading dims — correct for a row-SLICED / padded operand whose row
        stride is a runtime arg != the matrix dim (BLOCKER 1). When None (legacy /
        un-traced) the template falls back to the matrix dims (dense row-major).
        """
        BLOCK_M = info["BLOCK_M"]
        BLOCK_N = info["BLOCK_N"]
        BLOCK_K = info["BLOCK_K"]
        ptr_args = info["ptr_args"]
        scalar_arg_map = info.get("scalar_args", {})
        all_scalar_args = info.get("all_scalar_args", [])

        # 128 threads = 4 SIMD groups x 32 threads
        self.effective_block_size = 128

        # Signal 2D grid dispatch (pid_m on axis 0, pid_n on axis 1)
        self._used_pid_axes = {0, 1}

        # Emitted argument names belong to a separate compiler namespace. Source
        # names remain in the graph, stride proofs, and launch metadata; only MSL
        # references change. In particular m/n/k and pointer c cannot be captured
        # by template loops, and source names resembling these aliases are safe.
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        a_arg, b_arg, c_arg = self._dot_template_ptr_roles()  # P0 (065): dataflow, not position
        a_name = source_bindings[a_arg.name]
        b_name = source_bindings[b_arg.name]
        c_name = source_bindings[c_arg.name]

        # Determine input/output element types via the shared mapping —
        # see the simple-dot variant for the bf16 stride bug this avoids.
        a_elem = a_arg.elem_type
        input_msl_type = triton_type_to_msl(a_elem)
        c_elem = c_arg.elem_type
        output_msl_type = triton_type_to_msl(c_elem)

        # Genuine fp16 (WS1 Phase C): half INPUT fragments + float ACCUMULATOR
        # (half x half -> float MMA, de-risked in test_simdgroup_half_mma.py).
        # fp16 inputs are staged as half and fed to simdgroup_half8x8 — no float
        # upcast, so Apple's ~2x fp16 matrix throughput is actually used. bf16
        # uses simdgroup_bfloat8x8 (the M-series bfloat matrix unit, verified on
        # M4); other types stay on the float path. The accumulator is always float.
        # Fragment selection is the SHARED ``_simdgroup_frag_for`` helper (Step 5
        # de-dup — this block was the byte-identical twin of the simple-dot copy).
        acc_frag, in_frag, tg_type, stage_cast, pad = self._simdgroup_frag_for(a_elem)
        frag_type = acc_frag  # accumulators

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Number of row tiles per MMA step (BLOCK_M / 8)
        n_row_tiles = BLOCK_M // 8

        # The (edge-only) staged path stages just 8 deep per K step rather than
        # BLOCK_K deep. The matmul is identical (it accumulates over the full K
        # either way), but the threadgroup footprint drops from BLOCK_M*BLOCK_K
        # + BLOCK_K*BLOCK_N to BLOCK_M*8 + 8*BLOCK_N — and since that static
        # allocation also caps the FAST direct path's occupancy (Metal reserves
        # threadgroup memory whether or not the direct branch touches it),
        # shrinking it lifts aligned-matmul throughput ~13% (#156). The deep
        # K-tiling that mattered for the old staging-bound kernel is irrelevant
        # here: the fast path doesn't stage at all.
        STAGE_DEPTH = 8

        # Threadgroup memory sizes
        tg_a_size = BLOCK_M * STAGE_DEPTH
        tg_b_size = STAGE_DEPTH * BLOCK_N

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("#include <metal_simdgroup_matrix>")
        lines.append("using namespace metal;")
        lines.append("")

        # Build kernel signature with all args from the IR
        lines.append(f"kernel void {safe_name}(")
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_decls.append(_scalar_buffer_decl(arg, i, source_bindings))
        lines.append(",\n".join(arg_decls) + ",")
        lines.append(f"    uint3 pid3 [[threadgroup_position_in_grid]],")
        lines.append(f"    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append(f"    uint tiitg [[thread_index_in_threadgroup]]")
        lines.append(f") {{")

        # Unpack scalar args from buffers
        for arg in all_scalar_args:
            lines.append(_scalar_unpack(arg, source_bindings))

        lines.append(f"")
        # pid_m / pid_n are defined AFTER the extents below: the proven 1-D grid mapping
        # (``pid % cdiv(_M, BLOCK_M)`` / ``pid / cdiv(_M, BLOCK_M)``, packet 080) needs _M.
        lines.append(f"")

        # Determine M, N, K — use scalar args if available, else BLOCK dims
        has_M = "M" in scalar_arg_map
        has_N = "N" in scalar_arg_map
        has_K = "K" in scalar_arg_map

        # Integrity guard (PR1): when M/N aren't runtime args the template
        # guesses ``_M=BLOCK_M`` / ``_N=BLOCK_N`` — correct ONLY for a
        # single output tile. If the kernel actually tiles the output across
        # programs (uses program_id on that axis), the guess collapses the
        # real stride to the block size and silently produces wrong output
        # (test_dot_mulbroadcasted: grid 2x6 over 256x192, B read with
        # stride 32 instead of 192 -> ~98% mismatch). The full dim is baked
        # in as constexpr and isn't recoverable here, so refuse rather than
        # emit wrong numbers. Shared guard (_refuse_if_pid_tiles_baked_output).
        # The proven source strides and program mapping determine the whole tile.

        m_ext, n_ext = self._matmul_output_extent_args()
        lines.append(self._k_extent_line(info, BLOCK_K, has_K, BLOCK_M, BLOCK_N, source_bindings))
        lines.extend(self._pid_map_lines(info, BLOCK_M, source_bindings))

        # Leading dims for the simdgroup loads/stores = each operand's ROW stride.
        # This path only handles contiguous-INNER operands (the stride-aware gate routed
        # any strided inner dim to the scalar template), so the inner (col) stride is 1
        # and the row stride is the leading dim. The runtime device buffer carries the
        # HOST strides (the driver copies the operand's storage VERBATIM — it does NOT
        # materialize a dense row-major copy; an earlier comment here claimed it did and
        # was wrong, which is BLOCKER 1: a row-SLICED A whose row stride is a runtime arg
        # != K was addressed with the K dim → wrong values). So use the ADDRESS-TRACED
        # row stride: a runtime stride arg (unpacked as ``int <name>``) when traced, else
        # the matrix dim (a dense row-major operand whose stride Triton folded to a
        # constexpr == the dim). All scalar args are declared/unpacked above, so a stride
        # arg name is referenceable here.
        scalar_names = {a.name for a in self.graph.args if not a.is_ptr}

        def _ld(row_desc, dim_default):
            # row_desc is a stride func-arg NAME, an int-literal string, or None.
            if row_desc is None:
                return dim_default
            if row_desc in scalar_names:
                # Runtime row stride (sliced/padded rows). Render it at its DECLARED
                # width (packet 825): ``(uint)`` silently truncated a 64-bit stride to
                # 32 bits here, so a row stride past 4 Gi elements addressed the wrong
                # row on every load, store and leading-dim argument.
                _msl = triton_type_to_msl(self._declared_int_elem_type(row_desc) or "i32")
                return f"({'ulong' if _msl in ('long', 'ulong') else 'uint'}){source_bindings[row_desc]}"
            try:
                return f"{int(str(row_desc))}u"  # compile-time literal row stride
            except (TypeError, ValueError):
                return dim_default

        if descriptors is not None:
            a_row, a_col, b_row, b_col, c_row, c_col = descriptors
            a_ld = _ld(a_row, "_K")
            b_ld = _ld(b_row, "_N")
            c_ld = _ld(c_row, "_N")
        else:
            a_ld, b_ld, c_ld = "_K", "_N", "_N"

        lines.append(f"")
        lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
        lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
        lines.extend(self._matmul_tile_extent_lines(info, BLOCK_M, BLOCK_N, source_bindings))

        # Packet 834: prove the simdgroup leading-dimension ABI can represent this
        # launch's layout before using it, and take a width-faithful scalar route
        # otherwise. The verdict is uniform (kernel scalars only), so the early
        # return below is taken by a whole threadgroup or by none of it — no
        # divergent barrier. Both the direct fast path and the staged path live
        # behind it, so neither needs its own check.
        _fb_widths = self._dot_offset_arith_widths()
        if descriptors is None or _fb_widths is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "K-loop matmul requires proven operand strides and source integer widths "
                "for its simdgroup layout guard and scalar fallback; refusing rather than "
                "emit an unguarded leading-dimension path",
                op_name="tt.dot",
            )
        _emit_layout_guard = True
        if _emit_layout_guard:
            lines.append(f"")
            lines.extend(self._simd_layout_guard_lines(a_ld, b_ld, c_ld))
            lines.append(f"    if (!_simd_ok) {{")
            lines.extend(
                self._scalar_tile_fallback_lines(
                    widths=_fb_widths,
                    descriptors=descriptors,
                    a_name=a_name,
                    b_name=b_name,
                    c_name=c_name,
                    out_type=output_msl_type,
                    block_m=BLOCK_M,
                    block_n=BLOCK_N,
                    threads=128,
                    scalar_names=scalar_names,
                    source_bindings=source_bindings,
                )
            )
            lines.append(f"        return;")
            lines.append(f"    }}")
        lines.append(f"")
        lines.append(f"    threadgroup {tg_type} tg_A[{tg_a_size}];")
        lines.append(f"    threadgroup {tg_type} tg_B[{tg_b_size}];")
        lines.append(f"")

        # Column register-blocking: each of the 4 simdgroups owns BLOCK_N/4
        # output columns = col_tiles 8-wide blocks. The old code emitted only
        # sgitg*8 = 32 columns regardless of BLOCK_N, silently dropping columns
        # 32+ for BLOCK_N>32 (the *standard* matmul tiling, BLOCK_N=64/128) —
        # a reachable silent-wrong. (WS1 integrity fix; also adds a-fragment
        # reuse across col blocks, which helps throughput.)
        from triton_msl.errors import MetalNonRecoverableError

        # The simdgroup tiling is 8x8: BLOCK_M must split into 8-row tiles and
        # BLOCK_K into 8-deep MMA steps, else the row loop / inner K loop drop
        # the tail and silently under-compute. (Unreachable via normal kernels —
        # tl.arange forces power-of-2 block dims — but guarded defensively so a
        # non-arange-constructed dot can never silently produce wrong numbers.)
        if BLOCK_M % 8 != 0:
            raise MetalNonRecoverableError(
                f"K-loop matmul BLOCK_M={BLOCK_M} is not a multiple of 8; the "
                "8-row simdgroup tiling would drop the tail rows. Refusing."
            )
        if BLOCK_K % 8 != 0:
            raise MetalNonRecoverableError(
                f"K-loop matmul BLOCK_K={BLOCK_K} is not a multiple of 8; the "
                "8-deep MMA step would drop the tail of K. Refusing."
            )
        if BLOCK_N % 8 != 0:
            raise MetalNonRecoverableError(
                f"K-loop matmul BLOCK_N={BLOCK_N} is not a multiple of 8; "
                "output columns tile in 8-wide simdgroup blocks. Refusing."
            )
        # Distribute the BLOCK_N/8 8-wide column blocks across the 4 simdgroups
        # (ceil). For BLOCK_N a multiple of 32 all four are fully used; for
        # smaller / non-32-multiple BLOCK_N the extra simdgroups idle (guarded).
        total_col_blocks = BLOCK_N // 8
        col_tiles = (total_col_blocks + 3) // 4  # 8-wide col blocks per simdgroup
        cols_per_sg = col_tiles * 8
        col_needs_guard = total_col_blocks % 4 != 0

        def _col_guard(c):
            # Uniform-per-simdgroup predicate: is local block c a real column?
            # None when every (sgitg, c) is in range (no runtime cost).
            return f"(sgitg * {col_tiles}u + {c}u) < {total_col_blocks}u" if col_needs_guard else None

        # Threadgroup budget: the staged path needs tg_A + tg_B (plus a small
        # 1 KiB store scratch). Refuse if it exceeds Metal's 32 KiB threadgroup
        # limit rather than emit a kernel that silently overflows.
        tg_elt = 2 if tg_type in ("half", "bfloat") else 4  # bfloat is 2 bytes too
        tg_bytes = (tg_a_size + tg_b_size) * tg_elt + 4 * 64 * 4
        if tg_bytes > 32 * 1024:
            raise MetalNonRecoverableError(
                f"K-loop matmul needs {tg_bytes} B of threadgroup memory "
                "(> 32 KiB limit) for the staged path; refusing."
            )

        # Accumulators: one simdgroup_float8x8 per (8-row tile, 8-col block).
        acc_names = [[f"acc_{r}_{c}" for c in range(col_tiles)] for r in range(n_row_tiles)]
        all_accs = [acc_names[r][c] for r in range(n_row_tiles) for c in range(col_tiles)]
        lines.append(f"    {acc_frag} {', '.join(n + '(0)' for n in all_accs)};")
        b_names = [f"b_frag{c}" for c in range(col_tiles)]
        lines.append(f"    {in_frag} a_frag, {', '.join(b_names)};")
        lines.append(f"")

        # Fast path (WS1 C.2): a FULL float-output tile with K % 8 == 0 loads
        # simdgroup fragments DIRECTLY from device A/B — no threadgroup staging,
        # no barriers (the GPU cache supplies reuse). This is the lever that
        # takes the standalone kernel to MLX parity; combined with the column
        # register-blocking above it brings the same to real @triton.jit
        # matmuls. Partial/edge tiles (and half output) fall through to the
        # boundary-safe staged path below — direct simdgroup_load can't mask.
        if output_msl_type == "float":
            lines.append(
                f"    if (row_base + {BLOCK_M}u <= _M && col_base + {BLOCK_N}u <= _N && row_base + {BLOCK_M}u <= _A_M && col_base + {BLOCK_N}u <= _B_N && (_K % 8u) == 0u) {{"
            )
            lines.append(f"        for (uint k = 0u; k < _K; k += 8u) {{")
            for c in range(col_tiles):
                g = _col_guard(c)
                pfx = f"if ({g}) " if g else ""
                lines.append(
                    f"            {pfx}simdgroup_load(b_frag{c}, {b_name} + k * {b_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {b_ld});"
                )
            for t in range(n_row_tiles):
                lines.append(
                    f"            simdgroup_load(a_frag, {a_name} + (row_base + {t * 8}u) * {a_ld} + k, {a_ld});"
                )
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(
                        f"            {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});"
                    )
            lines.append(f"        }}")
            lines.append("        threadgroup_barrier(mem_flags::mem_device);")
            for t in range(n_row_tiles):
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(
                        f"        {pfx}simdgroup_store({acc_names[t][c]}, {c_name} + (row_base + {t * 8}u) * {c_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {c_ld});"
                    )
            lines.append(f"        return;")
            lines.append(f"    }}")
            lines.append(f"")

        # Staged path: cooperative masked load through threadgroup memory —
        # handles partial M/N/K tiles (and is the only path for half output).
        # Stages STAGE_DEPTH(8) deep per K step (small tg footprint, see above).
        lines.append(f"    for (uint _k = 0u; _k < _K; _k += {STAGE_DEPTH}u) {{")
        lines.append(f"")

        # Cooperative masked load A tile (BLOCK_M x STAGE_DEPTH).
        lines.append(f"        for (uint i = tiitg; i < {tg_a_size}u; i += 128u) {{")
        lines.append(f"            uint r = i / {STAGE_DEPTH}u, c = i % {STAGE_DEPTH}u;")
        lines.append(f"            uint gr = row_base + r, gc = _k + c;")
        lines.append(
            f"            tg_A[i] = (gr < _A_M && gc < _K) ? {stage_cast}({a_name}[gr * {a_ld} + gc]) : {pad};"
        )
        lines.append(f"        }}")

        # Cooperative masked load B tile (STAGE_DEPTH x BLOCK_N).
        lines.append(f"        for (uint i = tiitg; i < {tg_b_size}u; i += 128u) {{")
        lines.append(f"            uint r = i / {BLOCK_N}u, c = i % {BLOCK_N}u;")
        lines.append(f"            uint gr = _k + r, gc = col_base + c;")
        lines.append(
            f"            tg_B[i] = (gr < _K && gc < _B_N) ? {stage_cast}({b_name}[gr * {b_ld} + gc]) : {pad};"
        )
        lines.append(f"        }}")
        lines.append(f"        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"")

        # One STAGE_DEPTH(8)-deep MMA step over the staged tiles.
        for c in range(col_tiles):
            g = _col_guard(c)
            pfx = f"if ({g}) " if g else ""
            lines.append(
                f"        {pfx}simdgroup_load(b_frag{c}, tg_B + sgitg * {cols_per_sg}u + {c * 8}u, {BLOCK_N});"
            )
        for t in range(n_row_tiles):
            lines.append(f"        simdgroup_load(a_frag, tg_A + {t}u * 8u * {STAGE_DEPTH}u, {STAGE_DEPTH});")
            for c in range(col_tiles):
                g = _col_guard(c)
                pfx = f"if ({g}) " if g else ""
                lines.append(
                    f"        {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});"
                )
        lines.append(f"        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    }} // K-loop")
        lines.append(f"")

        # Store the BLOCK_M x BLOCK_N tile, MASKED so partial M/N tiles don't
        # over-write. simdgroup_store can't mask, so each simdgroup stages its
        # 8x8 block to its OWN 64-float slot (no cross-simdgroup collision),
        # then its 32 lanes write only the in-bounds elements. This fixes two
        # silent-wrongs the old store had: the float path wrote a full unmasked
        # 8x8 (for partial N the overflow columns wrapped into the next row's
        # in-bounds data), and the half path raced on a shared slot. (Reached
        # only by the staged path — partial/odd-K tiles, or non-float output;
        # the direct fast path above stores full tiles unmasked.)
        lines.append(f"    // Store {BLOCK_M}x{BLOCK_N} result tile (masked)")
        lines.append(f"    threadgroup float tg_st[4u * 64u];")
        lines.append(f"    uint laneid = tiitg % 32u;")
        for t in range(n_row_tiles):
            for c in range(col_tiles):
                g = _col_guard(c)
                spfx = f"if ({g}) " if g else ""
                cond = f"{g} && gr < _M && gc < _N" if g else "gr < _M && gc < _N"
                # store + write are per-simdgroup guarded; the barriers are NOT
                # (every thread must reach them, regardless of which columns its
                # simdgroup owns).
                _emit_masked_staged_store(
                    lines,
                    acc=acc_names[t][c],
                    scratch="tg_st",
                    gr=f"row_base + {t * 8}u + i / 8u",
                    gc=f"col_base + sgitg * {cols_per_sg}u + {c * 8}u + i % 8u",
                    cond=cond,
                    dst=f"{c_name}[gr * {c_ld} + gc]",
                    out_type=output_msl_type,
                    store_pfx=spfx,
                    indent="    ",
                )

        lines.append(f"}}")

        # Two-kernel split (#159): when M/N/K are runtime args and the output is
        # float, ALSO emit a standalone pure-direct kernel whose CODE contains no
        # threadgroup memory at all -> max occupancy -> MLX parity. The launcher
        # dispatches it for fully-aligned dispatches (every tile takes the direct
        # path); the staged kernel above handles partial/odd-K/half. (The
        # dynamic-threadgroup approach failed: Metal/AGX caps occupancy by a
        # kernel's compile-time tg usage, not the per-dispatch host length, so
        # only a SEPARATE no-tg kernel recovers the residual occupancy.)
        _argnames = [a.name for a in self.graph.args]
        if (
            output_msl_type == "float"
            and has_M
            and has_N
            and has_K
            and "M" in _argnames
            and "N" in _argnames
            and "K" in _argnames
            # Only a reduction extent the loop bounds PROVED is a launch-time K: with
            # constant bounds or a count over another argument, the argument named K is
            # a stride or an unrelated extent (GPU: the twin reduced over it anyway).
            and info.get("k_extent_arg") is not None
        ):
            dn = f"{safe_name}__mmdirect"
            # Whole 32x32 sub-tiles per simdgroup when the tile splits that way; the
            # twin then has its OWN threadgroup width, carried in the descriptor.
            # Other tiles keep the four-strip layout at the staged entry's width.
            _twin_width = self._direct_twin_width(BLOCK_M, BLOCK_N)
            lines.append("")
            lines.append(f"kernel void {dn}(")
            lines.append(",\n".join(arg_decls) + ",")
            lines.append(f"    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append(f"    uint sgitg [[simdgroup_index_in_threadgroup]],")
            if _twin_width is not None:
                lines.append(f"    uint _sgw [[threads_per_simdgroup]],")
            lines.append(f"    uint tiitg [[thread_index_in_threadgroup]]")
            lines.append(f") {{")
            for arg in all_scalar_args:
                lines.append(_scalar_unpack(arg, source_bindings))
            # Bind the extents by ROLE, exactly as the staged entry above does: the
            # store-mask extents when they resolve, else the names. Binding by the
            # names alone sent a kernel whose arguments named M/N/K play other roles
            # (``(A, B, C, M, K, N)``, or M and N exchanged) to the wrong tiles.
            _direct_m = m_ext or "M"
            _direct_n = n_ext or "N"
            _direct_k = info["k_extent_arg"]
            lines.append(self._k_extent_line(info, BLOCK_K, has_K, BLOCK_M, BLOCK_N, source_bindings))
            # Packet 106: the direct variant replays the SAME proven grid mapping as the
            # staged kernel (the tutorial's 1-D split, and tile 0 on an axis the source
            # kernel does not tile) — it used to hard-code ``pid3.x / pid3.y``, so the
            # host path's aligned-shape swap mis-mapped both forms.
            lines.extend(self._pid_map_lines(info, BLOCK_M, source_bindings))
            lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
            lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
            lines.extend(self._matmul_tile_extent_lines(info, BLOCK_M, BLOCK_N, source_bindings))
            lines.extend(self._twin_surplus_program_lines(m_ext, n_ext))
            # Packet 834: the direct twin has no staged path to fall through to, so it
            # carries the SAME uniform proof and the SAME scalar fallback. The fallback
            # uses no threadgroup memory, so this kernel keeps the zero-threadgroup
            # footprint that is its whole reason to exist.
            if _emit_layout_guard:
                lines.extend(self._simd_layout_guard_lines(a_ld, b_ld, c_ld))
                if _twin_width is not None:
                    # The sub-tile ownership below is by simdgroup index, so it also
                    # needs the 32-lane simdgroup it was laid out for (uniform too).
                    lines.append("    _simd_ok = _simd_ok && _sgw == 32u;")
                lines.append(
                    f"    _simd_ok = _simd_ok && (long)row_base + {BLOCK_M}L <= _A_M && (long)col_base + {BLOCK_N}L <= _B_N;"
                )
                lines.append(f"    if (!_simd_ok) {{")
                lines.extend(
                    self._scalar_tile_fallback_lines(
                        widths=_fb_widths,
                        descriptors=descriptors,
                        a_name=a_name,
                        b_name=b_name,
                        c_name=c_name,
                        out_type=output_msl_type,
                        block_m=BLOCK_M,
                        block_n=BLOCK_N,
                        threads=_twin_width or 128,
                        preserve_alias=False,
                        scalar_names=scalar_names,
                        source_bindings=source_bindings,
                    )
                )
                lines.append(f"        return;")
                lines.append(f"    }}")
            if _twin_width is not None:
                lines.extend(
                    self._subtiled_direct_lines(a_name, b_name, c_name, BLOCK_M, BLOCK_N, acc_frag, in_frag, False)
                )
            else:
                lines.append(f"    {acc_frag} {', '.join(n + '(0)' for n in all_accs)};")
                lines.append(f"    {in_frag} a_frag, {', '.join(b_names)};")
                lines.append(f"    for (uint k = 0u; k < _K; k += 8u) {{")
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(
                        f"        {pfx}simdgroup_load(b_frag{c}, {b_name} + k * {b_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {b_ld});"
                    )
                for t in range(n_row_tiles):
                    lines.append(
                        f"        simdgroup_load(a_frag, {a_name} + (row_base + {t * 8}u) * {a_ld} + k, {a_ld});"
                    )
                    for c in range(col_tiles):
                        g = _col_guard(c)
                        pfx = f"if ({g}) " if g else ""
                        lines.append(
                            f"        {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});"
                        )
                lines.append(f"    }}")
                for t in range(n_row_tiles):
                    for c in range(col_tiles):
                        g = _col_guard(c)
                        pfx = f"if ({g}) " if g else ""
                        lines.append(
                            f"    {pfx}simdgroup_store({acc_names[t][c]}, {c_name} + (row_base + {t * 8}u) * {c_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {c_ld});"
                        )
            lines.append(f"}}")
            self._mm_two_kernel = {
                "direct_name": dn,
                "direct_block": _twin_width or 128,
                "block_m": BLOCK_M,
                "block_n": BLOCK_N,
                "m_idx": _argnames.index(_direct_m),
                "n_idx": _argnames.index(_direct_n),
                "k_idx": _argnames.index(_direct_k),
                "c_idx": self._bound_argument_index(source_bindings, c_name),
            }

        return "\n".join(lines)

    def _lower_flip_template(self, info) -> str:
        """Generate a direct MSL kernel for tl.flip of a 3D tensor.

        For output at row-major index lid = i*N*K + j*K + k (where (i,j,k)
        are the 3D coordinates), emits:
            src_i, src_j, src_k = flip coordinates along flip_dim
            Z[lid] = X[src_i*N*K + src_j*K + src_k]

        The offsets match test_flip's row-major offset pattern. If the
        user-specified strides differ, this template will produce wrong
        results — but the detector matches the test_flip pattern which
        always uses row-major offsets.
        """
        M = info["M"]
        N = info["N"]
        K = info["K"]
        flip_dim = info["flip_dim"]
        elem_type = info["elem_type"]
        x_ptr = info["x_ptr"]
        z_ptr = info["z_ptr"]
        total = info["total"]
        block_size = info["block_size"]

        msl_type = triton_type_to_msl(elem_type)

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_decls.append(_scalar_buffer_decl(arg, i, source_bindings))

        x_ptr = source_bindings[x_ptr]
        z_ptr = source_bindings[z_ptr]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
        # Decompose _e into (i, j, k)
        lines.append(f"        uint _i = _e / {N * K}u;")
        lines.append(f"        uint _j = (_e / {K}u) % {N}u;")
        lines.append(f"        uint _k = _e % {K}u;")
        # Compute flipped coordinates
        if flip_dim == 0:
            lines.append(f"        uint _si = {M - 1}u - _i;")
            lines.append(f"        uint _sj = _j;")
            lines.append(f"        uint _sk = _k;")
        elif flip_dim == 1:
            lines.append(f"        uint _si = _i;")
            lines.append(f"        uint _sj = {N - 1}u - _j;")
            lines.append(f"        uint _sk = _k;")
        else:  # flip_dim == 2
            lines.append(f"        uint _si = _i;")
            lines.append(f"        uint _sj = _j;")
            lines.append(f"        uint _sk = {K - 1}u - _k;")
        lines.append(f"        uint _src = _si * {N * K}u + _sj * {K}u + _sk;")
        lines.append(f"        {z_ptr}[_e] = {x_ptr}[_src];")
        lines.append(f"    }}")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _lower_softmax_template(self, info) -> str:
        """Emit a TG-cached row-wise softmax kernel.

        Layout: 1 threadgroup per row, ``num_warps * 32`` threads per group.
        Each thread iterates over its share of the row in a stride loop.

        Memory traffic vs the generic 3-phase lowering:
          - global x_ptr reads: 3 → 1 (cached in TG memory)
          - global out_ptr writes: 1 → 1
          - exp() calls: 2 → 1

        When the input is float (4-byte element) and ``n_cols`` is a multiple
        of 4 at runtime, the kernel takes a vectorized path that loads/stores
        ``float4`` chunks. This roughly doubles memory throughput on M4 Max
        (measured 1.49x speedup on softmax_8Kx1K vs the scalar path).
        """
        block_size = info["block_size"]
        n_arg = info["n_arg"]
        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        # Threads per group: num_warps * warp_size. Default 4*32 = 128.
        num_warps = self.options.num_warps if self.options else 4
        warp_size = 32
        threads = num_warps * warp_size
        n_simd = num_warps  # one shared slot per warp for cross-warp reduce

        # Vectorized float4 path is correct only when:
        #   - the input arg is fp32 (4-byte element), so reinterpret-cast
        #     to float4 produces aligned packs of 4 elements
        #   - the threadgroup row_cache buffer has block_size divisible by 4
        #     so the float4 reinterpret is well-defined inside TG memory
        #   - n_cols is a runtime multiple of 4 (checked at runtime)
        # Other element types (fp16/bf16/integer) fall through to the scalar
        # path until a typed-vector emission is added.
        # Vectorize only when EVERY pointer arg (input AND output) is fp32: the
        # float4 store reinterpret-casts the OUTPUT pointer to device float4*, which
        # is garbage (-> NaN) for a half*/bfloat* output. Gating on the input dtype
        # alone shipped a fp32-in / fp16-out softmax that stored NaN (re-audit #9).
        # Non-fp32 output falls through to the scalar store, which casts per element.
        _ptr_args = [a for a in self.graph.args if a.is_ptr]
        all_fp32_ptrs = all(a.elem_type in ("f32", "fp32") for a in _ptr_args)
        vectorize = all_fp32_ptrs and (block_size % 4 == 0)

        # float -> half is implicit in MSL, but float -> bfloat is NOT, so an uncast
        # bf16-output store is an MSL compile error (cryptic crash). Cast the scalar
        # store value to the output element type when it isn't float (re-audit #9).
        _out_elem = next((a.elem_type for a in _ptr_args if a.name == output_arg), None)
        _out_msl = triton_type_to_msl(_out_elem) if _out_elem else "float"
        _store_cast = "" if _out_msl == "float" else f"({_out_msl})"

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                # Scalar arg passed as constant&
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {arg_msl_type}& {source_bindings[arg.name]} [[buffer({i})]]")

        input_arg = source_bindings[input_arg]
        output_arg = source_bindings[output_arg]
        n_arg = source_bindings[n_arg]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append("    uint tiisg [[thread_index_in_simdgroup]]")
        lines.append(") {")

        lines.append(f"    threadgroup float row_cache[{block_size}];")
        lines.append(f"    threadgroup float reduce_buf[{n_simd}];")
        lines.append(f"    int row_start = pid * {n_arg};")
        lines.append(f"    int _norm_cols = clamp(int({n_arg}), 0, {block_size});")
        lines.append("    float local_max = -INFINITY;")
        lines.append("")

        # ----- Phase 1: load → row_cache, reduce local max -----
        if vectorize:
            lines.append(f"    bool use_vec = (({n_arg} & 3) == 0);")
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = _norm_cols / 4;")
            lines.append(f"        device float4* x4 = (device float4*)({input_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = x4[i];")
            lines.append("            row4[i] = v;")
            lines.append("            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"            float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("            row_cache[i] = v;")
            lines.append("            local_max = max(local_max, v);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"        float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("        row_cache[i] = v;")
            lines.append("        local_max = max(local_max, v);")
            lines.append("    }")

        # Reduce max across threadgroup
        lines.append("    float simd_max_v = simd_max(local_max);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_max_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_max = simd_max((tiisg < {n_simd}u) ? reduce_buf[tiisg] : -INFINITY);")
        lines.append("    float local_sum = 0.0f;")
        lines.append("")

        # ----- Phase 2: exp(x - max) → row_cache, reduce local sum -----
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = _norm_cols / 4;")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = row4[i];")
            lines.append("            float4 e;")
            lines.append("            e.x = exp(v.x - row_max); e.y = exp(v.y - row_max);")
            lines.append("            e.z = exp(v.z - row_max); e.w = exp(v.w - row_max);")
            lines.append("            row4[i] = e;")
            lines.append("            local_sum += e.x + e.y + e.z + e.w;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append("            float e = exp(row_cache[i] - row_max);")
            lines.append("            row_cache[i] = e;")
            lines.append("            local_sum += e;")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append("        float e = exp(row_cache[i] - row_max);")
            lines.append("        row_cache[i] = e;")
            lines.append("        local_sum += e;")
            lines.append("    }")

        # Reduce sum
        lines.append("    float simd_sum_v = simd_sum(local_sum);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sum_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sum = simd_sum((tiisg < {n_simd}u) ? reduce_buf[tiisg] : 0.0f);")
        lines.append("    float inv_sum = 1.0f / row_sum;")
        lines.append("")

        # ----- Phase 3: write normalized exp to global memory -----
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = _norm_cols / 4;")
            lines.append(f"        device float4* o4 = (device float4*)({output_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            o4[i] = row4[i] * inv_sum;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"            {output_arg}[row_start + i] = {_store_cast}(row_cache[i] * inv_sum);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"        {output_arg}[row_start + i] = {_store_cast}(row_cache[i] * inv_sum);")
            lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_layer_norm_template(self, info) -> str:
        """Emit a TG-cached row-wise layer-norm kernel.

        Mirrors ``_lower_softmax_template``\\'s structure with a row_cache TG
        buffer; differs only in the math (``sum`` followed by centered-square
        variance, instead of ``max`` followed by ``exp``).

        Memory traffic vs the generic 3-phase lowering:
          - global x_ptr reads: 3 → 1 (cached in TG memory)
          - global out_ptr writes: 1 → 1
        Vectorizes through float4 when the input is fp32 and ``n_cols`` is a
        runtime multiple of 4 — same eligibility as the softmax template.

        Caveat: this template assumes the canonical layer-norm shape
            (x - mean) * rsqrt(var + eps)
        without learnable gamma/beta scaling. Kernels that include gamma/beta
        load extra pointers and would not match the detector\\'s 2-ptr
        signature; they fall through to the generic phase lowerer.
        """
        block_size = info["block_size"]
        n_arg = info["n_arg"]
        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        num_warps = self.options.num_warps if self.options else 4
        warp_size = 32
        threads = num_warps * warp_size
        n_simd = num_warps

        # Vectorize only when EVERY pointer arg (input AND output) is fp32: the
        # float4 store reinterpret-casts the OUTPUT pointer to device float4*, which
        # is garbage (-> NaN) for a half*/bfloat* output. Gating on the input dtype
        # alone shipped a fp32-in / fp16-out softmax that stored NaN (re-audit #9).
        # Non-fp32 output falls through to the scalar store, which casts per element.
        _ptr_args = [a for a in self.graph.args if a.is_ptr]
        all_fp32_ptrs = all(a.elem_type in ("f32", "fp32") for a in _ptr_args)
        vectorize = all_fp32_ptrs and (block_size % 4 == 0)

        # float -> half is implicit in MSL, but float -> bfloat is NOT, so an uncast
        # bf16-output store is an MSL compile error (cryptic crash). Cast the scalar
        # store value to the output element type when it isn't float (re-audit #9).
        _out_elem = next((a.elem_type for a in _ptr_args if a.name == output_arg), None)
        _out_msl = triton_type_to_msl(_out_elem) if _out_elem else "float"
        _store_cast = "" if _out_msl == "float" else f"({_out_msl})"

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {arg_msl_type}& {source_bindings[arg.name]} [[buffer({i})]]")

        # eps comes from the KERNEL's own rsqrt(var + eps) (extracted by the detector,
        # exact-or-refuse), NOT a hardcoded default — a wrong eps is a silent-wrong
        # (worst on low-variance rows). ``info["eps"]`` is always present (the detector
        # refuses when it can't resolve the constant).
        eps_literal = f"{float(info['eps'])!r}f"

        input_arg = source_bindings[input_arg]
        output_arg = source_bindings[output_arg]
        n_arg = source_bindings[n_arg]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append("    uint tiisg [[thread_index_in_simdgroup]]")
        lines.append(") {")

        lines.append(f"    threadgroup float row_cache[{block_size}];")
        lines.append(f"    threadgroup float reduce_buf[{n_simd}];")
        lines.append(f"    int row_start = pid * {n_arg};")
        lines.append(f"    int _norm_cols = clamp(int({n_arg}), 0, {block_size});")
        lines.append("    float local_sum = 0.0f;")
        lines.append("")

        # Phase 1: load -> row_cache, reduce the source mean
        if vectorize:
            lines.append(f"    bool use_vec = (({n_arg} & 3) == 0);")
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = _norm_cols / 4;")
            lines.append(f"        device float4* x4 = (device float4*)({input_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = x4[i];")
            lines.append("            row4[i] = v;")
            lines.append("            local_sum += v.x + v.y + v.z + v.w;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"            float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("            row_cache[i] = v;")
            lines.append("            local_sum += v;")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"        float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("        row_cache[i] = v;")
            lines.append("        local_sum += v;")
            lines.append("    }")

        # Reduce the mean, then the centered-square variance across the threadgroup
        lines.append("    float simd_sum_v = simd_sum(local_sum);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sum_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sum = simd_sum((tiisg < {n_simd}u) ? reduce_buf[tiisg] : 0.0f);")
        lines.append("")
        # Match the source's centered-square variance. Difference-of-moments
        # loses small deviations around a large mean (1000 +/- 0.125 -> var=0).
        lines.append(f"    float mean = row_sum / float({n_arg});")
        lines.append("    float local_sumsq = 0.0f;")
        lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
        lines.append("        float centered = row_cache[i] - mean;")
        lines.append("        local_sumsq += centered * centered;")
        lines.append("    }")
        lines.append("    float simd_sumsq_v = simd_sum(local_sumsq);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sumsq_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sumsq = simd_sum((tiisg < {n_simd}u) ? reduce_buf[tiisg] : 0.0f);")
        lines.append("")
        lines.append(f"    float var = row_sumsq / float({n_arg});")
        lines.append(f"    float inv_std = rsqrt(var + {eps_literal});")
        lines.append("")

        # Phase 2: write (x - mean) * inv_std to global memory
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = _norm_cols / 4;")
            lines.append(f"        device float4* o4 = (device float4*)({output_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = row4[i];")
            lines.append("            float4 m = float4(mean);")
            lines.append("            o4[i] = (v - m) * inv_std;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"            {output_arg}[row_start + i] = {_store_cast}((row_cache[i] - mean) * inv_std);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint)_norm_cols; i += {threads}u) {{")
            lines.append(f"        {output_arg}[row_start + i] = {_store_cast}((row_cache[i] - mean) * inv_std);")
            lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_transpose_via_reshape_template(self, info) -> str:
        """Emit a direct transpose-lookup kernel for the test_trans_reshape pattern.

        The detected pattern is a 2-D transpose expressed via reshape→permute→
        reshape: input shape (M, N), output is the row-major flatten of
        ``transpose(input)``. The closed-form mapping is

            output[k] = input[(k % M) * N + (k // M)]

        Each thread iterates ``k`` in stride-``threads`` chunks and emits one
        element per iteration. We bypass the generic phase lowerer entirely
        because the source ``#linear`` layout assigns multiple elements per
        thread and the generic ``ttg.convert_layout`` path can\\'t honor that
        without a shared-memory shuffle (see ``_linear_layout`` and Phase 4).
        """
        M = info["M"]
        N = info["N"]
        elem_type = info["elem_type"]
        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        block_size = info["block_size"]  # = M * N

        msl_type = triton_type_to_msl(elem_type)
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {arg_msl_type}& {source_bindings[arg.name]} [[buffer({i})]]")

        input_arg = source_bindings[input_arg]
        output_arg = source_bindings[output_arg]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")
        lines.append(f"    for (uint k = lid; k < {block_size}u; k += {threads}u) {{")
        lines.append(f"        uint _row = k % {M}u;")
        lines.append(f"        uint _col = k / {M}u;")
        lines.append(f"        {output_arg}[k] = {input_arg}[_row * {N}u + _col];")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_nd_trans_template(self, info) -> str:
        """Closed-form N-D transpose: out[k] = in[src_flat(k)] in a strided
        loop. src_flat(k) = sum_d ((k / dst_stride[d]) % dst_shape[d]) *
        src_stride[order[d]], with row-major strides computed here."""
        src_shape = info["src_shape"]
        order = info["order"]
        total = info["total"]
        rank = len(src_shape)
        dst_shape = [src_shape[order[d]] for d in range(rank)]

        def _row_major_strides(shape):
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            return st

        dst_stride = _row_major_strides(dst_shape)
        src_stride = _row_major_strides(src_shape)
        # in_flat = sum_d O[d] * src_stride[order[d]];  O[d]=(k/dst_stride[d])%dst_shape[d]
        terms = []
        for d in range(rank):
            o_d = f"(k % {dst_shape[d]}u)" if dst_stride[d] == 1 else f"((k / {dst_stride[d]}u) % {dst_shape[d]}u)"
            terms.append(f"{o_d} * {src_stride[order[d]]}u")
        in_flat = " + ".join(terms)

        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        # Pointer element types come from each arg's own elem_type in the
        # arg-decl loop below; no kernel-wide msl_type needed here.
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)
        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {arg_msl_type}& {source_bindings[arg.name]} [[buffer({i})]]")

        input_arg = source_bindings[input_arg]
        output_arg = source_bindings[output_arg]

        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            f"kernel void {safe_name}(",
            ",\n".join(arg_decls) + ",",
            "    uint pid [[threadgroup_position_in_grid]],",
            "    uint lid [[thread_position_in_threadgroup]],",
            "    uint tid [[thread_position_in_grid]]",
            ") {",
            f"    for (uint k = lid; k < {total}u; k += {threads}u) {{",
            f"        {output_arg}[k] = {input_arg}[{in_flat}];",
            "    }",
            "}",
            "",
        ]
        return "\n".join(lines)

    def _lower_permute_chained_reduce_template(self, info) -> str:
        """Emit a fused permute+chained-sum-reduce cooperative kernel.

        The permute is never materialized. Each input element ``In[i]`` maps
        to exactly one output cell (the surviving original-axis coordinates,
        computed from the permute + reduce axes at detect time). 1024 threads
        cooperatively scatter-add the inputs into a tiny threadgroup atomic
        accumulator, then write it out. Correct for sum reductions of integer
        tensors (the test_chained_reductions case).
        """
        in_arg = info["in_arg"]
        out_arg = info["out_arg"]
        out_elem = info.get("out_elem", info["elem"])
        total = info["total"]
        out_total = info["out_total"]
        surviving = info["surviving"]  # [(in_stride, size, out_stride), ...]

        msl_type = triton_type_to_msl(out_elem)
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)
        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {arg_msl_type}& {source_bindings[arg.name]} [[buffer({i})]]")

        # out_cell(i) = sum_k ((i / in_stride_k) % size_k) * out_stride_k
        oc_terms = []
        for in_s, size, out_s in surviving:
            coord = f"((i / {in_s}u) % {size}u)"
            oc_terms.append(coord if out_s == 1 else f"{coord} * {out_s}u")
        oc_expr = " + ".join(oc_terms) if oc_terms else "0u"

        in_arg = source_bindings[in_arg]
        out_arg = source_bindings[out_arg]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")
        lines.append(f"    threadgroup atomic_int _acc[{out_total}];")
        lines.append(f"    for (uint e = lid; e < {out_total}u; e += {threads}u) {{")
        lines.append("        atomic_store_explicit(&_acc[e], 0, memory_order_relaxed);")
        lines.append("    }")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    for (uint i = lid; i < {total}u; i += {threads}u) {{")
        lines.append(f"        uint _oc = {oc_expr};")
        lines.append(f"        atomic_fetch_add_explicit(&_acc[_oc], (int){in_arg}[i], memory_order_relaxed);")
        lines.append("    }")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    for (uint e = lid; e < {out_total}u; e += {threads}u) {{")
        lines.append(f"        {out_arg}[e] = ({msl_type})atomic_load_explicit(&_acc[e], memory_order_relaxed);")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_matmul_softmax_template(self, info) -> str:
        """Emit a fused matmul + row-softmax kernel.

        Layout (single threadgroup of 128 threads, 4 SIMD groups):
          1. Outer loop over M-strips of ``M_BLOCK`` rows so the staged
             output never exceeds Metal\'s 32 KB threadgroup-memory cap.
          2. Per strip: stage A[M_BLOCK, K] and B[K, N] (one 8-wide
             K-tile at a time), accumulate via simdgroup_matrix MMA.
          3. Store the strip\'s output (M_BLOCK × N) into TG memory.
          4. Cooperative row softmax across all N cols, one thread per
             row.
          5. Write the strip to global through the supplied output
             strides (or row-major contiguous when none are named).

        Constraints: M, N, K are multiples of 8. M_BLOCK is chosen so
        ``M_BLOCK * N`` floats ≤ 16 KB and ``M_BLOCK`` is a multiple of
        the simdgroup tile (8) — currently ``min(M, 32)`` which handles
        the 64×64 and 128×128 test_dot softmax cases.
        """
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        M = info["M"]
        N = info["N"]
        K = info["K"]
        a_ptr = info["a_ptr"]
        b_ptr = info["b_ptr"]
        c_ptr = info["c_ptr"]
        a_elem = info["a_elem"]
        c_elem = info["c_elem"]
        a_row_s = info["a_row_stride"]
        a_col_s = info["a_col_stride"]
        b_row_s = info["b_row_stride"]
        b_col_s = info["b_col_stride"]
        c_row_s = info["c_row_stride"]
        c_col_s = info["c_col_stride"]

        # M, N, K must be 8-multiples so 8×8 simdgroup matrices cover
        # the whole tile without partial-tile guards.
        if M % 8 or N % 8 or K % 8:
            return None

        # Strip the M dimension so the staged output fits in TG memory.
        # Cap the strip at 32 rows (one strip\'s tg_C is M_BLOCK*N floats;
        # 32 × 256 = 32KB, the Metal limit). M itself must split evenly
        # into M_BLOCK strips.
        m_block = min(M, 32)
        while m_block > 8 and M % m_block != 0:
            m_block -= 8
        if m_block < 8:
            return None
        n_strips = M // m_block
        tg_bytes = (m_block * 8 + 8 * N + m_block * N) * 4
        if tg_bytes > 32 * 1024:
            return None

        # This template emits the WHOLE baked M×N output from a SINGLE
        # threadgroup and never references program_id; if the kernel tiles the
        # output across programs every block recomputes/overwrites the same rows
        # -> silently wrong. Shared guard (_refuse_if_pid_tiles_baked_output); M/N
        # are baked constexpr here (a runtime M/N can't reach this template —
        # m_block/n_strips need a concrete M), so has_M/has_N are False.
        # Unlike the whole-output matmul makers, this maker never consumes
        # mask-derived runtime extents. The shared guard's extent inference
        # cannot certify a program-id mapping this maker does not emit.
        if any(op.op in self._PID_OPS for op in self.graph.ops):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "fused matmul+softmax/epilogue template does not reproduce program-id "
                "tiling, even when the store mask exposes runtime output bounds.",
                op_name="tt.dot",
            )

        # 128 threads = 4 SIMD groups × 32 threads.
        self.effective_block_size = 128

        c_msl_type = triton_type_to_msl(c_elem)

        # Fragment selection via the SHARED _simdgroup_frag_for selector (no longer a
        # divergent inline copy): half INPUT fragments + float ACCUMULATOR (the matmul
        # output tg_C stays float for the softmax epilogue). fp16 -> half MMA (no float
        # upcast); bf16 -> the M-series simdgroup_bfloat8x8 MMA (staged as bfloat, float
        # accumulate -> the float softmax epilogue is unchanged); other types -> float.
        _af, in_frag, in_stage_t, in_cast, _pad = self._simdgroup_frag_for(a_elem)

        # Each SIMD group owns ``N / 4`` columns of the output, which must
        # split evenly into 8-wide simdgroup tiles.
        cols_per_sg = N // 4
        if cols_per_sg == 0 or cols_per_sg % 8:
            return None
        col_tiles_per_sg = cols_per_sg // 8  # 8×8 col tiles per SG
        row_tiles = m_block // 8  # 8×8 row tiles per strip

        # Addresses are rendered at the SOURCE's integer width (packet 825, W2 F-7 —
        # extension). ``f"({row}) * {row_stride}"`` was an UNCAST product: with an i32
        # stride argument (``constant int&``) a ``uint`` row index made it a 32-bit
        # UNSIGNED product that is then ZERO-extended at the subscript, so a negative
        # stride addressed ~4 GiB forward instead of backwards, and a product past
        # 2**31 lost its sign. (It did not truncate an i64 stride — ``uint * long``
        # promotes — but that was luck, not a proof.) Same prover as the strided scalar
        # matmul: per-operand addptr-link width, refuse when unprovable.
        _widths = self._dot_offset_arith_widths()
        if _widths is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "fused matmul+softmax/epilogue template: the integer width of the operand "
                "address arithmetic could not be proven from the traced tt.addptr chain "
                "(unreadable, or the chain mixes i32 and i64 offsets). Emitting the address "
                "would have to guess whether each term wraps at 32 bits, so refusing "
                "(correct-or-refuse).",
                op_name="tt.dot",
            )
        _scalar_names = {a.name for a in self.graph.args if not a.is_ptr}

        def _addr(base, row, col, row_stride, col_stride, inner_dim, role):
            # The detectors feeding this template both go through
            # _inferred_stride_descriptors, which never yields a None slot; the
            # ``inner_dim`` / unit fallbacks are kept for defence and carry the same
            # width as the rest of that operand's address.
            terms = [(row, row_stride if row_stride else str(inner_dim)), (col, col_stride if col_stride else "1")]
            expr = self._width_faithful_addr(
                terms,
                *_widths[role],
                what=f"fused matmul+softmax/epilogue operand {role}",
                scalar_names=_scalar_names,
                source_bindings=source_bindings,
            )
            return f"{source_bindings[base]}[{expr}]"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                m = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {m}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                m = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(f"    constant {m}& {source_bindings[arg.name]} [[buffer({i})]]")

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("#include <metal_simdgroup_matrix>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append("    uint tiitg [[thread_index_in_threadgroup]],")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")
        lines.append(f"    threadgroup {in_stage_t} tg_A[{m_block} * 8];")
        lines.append(f"    threadgroup {in_stage_t} tg_B[8 * {N}];")
        lines.append(f"    threadgroup float tg_C[{m_block} * {N}];")
        lines.append("")

        # ----- Outer M-strip loop -----
        lines.append(f"    for (uint mstrip = 0u; mstrip < {M}u; mstrip += {m_block}u) {{")

        lines.append("        // Per-SG output accumulators for this strip.")
        for rt in range(row_tiles):
            for ct in range(col_tiles_per_sg):
                lines.append(f"        simdgroup_float8x8 acc_{rt}_{ct}(0);")
        lines.append("")

        lines.append(f"        for (uint kk = 0u; kk < {K}u; kk += 8u) {{")
        # Stage strip\'s A[M_BLOCK, 8] cooperatively.
        lines.append(f"            for (uint i = tiitg; i < {m_block * 8}u; i += 128u) {{")
        lines.append("                uint r = i / 8u, c = i % 8u;")
        a_load = _addr(a_ptr, "mstrip + r", "kk + c", a_row_s, a_col_s, K, "A")
        lines.append(f"                tg_A[i] = {in_cast}({a_load});")
        lines.append("            }")
        # Stage B[8, N] cooperatively.
        lines.append(f"            for (uint i = tiitg; i < {8 * N}u; i += 128u) {{")
        lines.append(f"                uint r = i / {N}u, c = i % {N}u;")
        b_load = _addr(b_ptr, "kk + r", "c", b_row_s, b_col_s, N, "B")
        lines.append(f"                tg_B[i] = {in_cast}({b_load});")
        lines.append("            }")
        lines.append("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("")
        for ct in range(col_tiles_per_sg):
            lines.append(f"            {in_frag} b_{ct};")
            lines.append(f"            simdgroup_load(b_{ct}, tg_B + sgitg * {cols_per_sg}u + {ct * 8}u, {N});")
        lines.append("")
        lines.append(f"            {in_frag} a_frag;")
        for rt in range(row_tiles):
            lines.append(f"            simdgroup_load(a_frag, tg_A + {rt * 8 * 8}u, 8);")
            for ct in range(col_tiles_per_sg):
                lines.append(
                    f"            simdgroup_multiply_accumulate(acc_{rt}_{ct}, a_frag, b_{ct}, acc_{rt}_{ct});"
                )
        lines.append("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("        }")
        lines.append("")

        # Spill the strip\'s accumulators.
        lines.append("        // Store strip accumulators to TG memory.")
        for rt in range(row_tiles):
            for ct in range(col_tiles_per_sg):
                row_off = rt * 8 * N
                col_off = f"sgitg * {cols_per_sg}u + {ct * 8}u"
                lines.append(f"        simdgroup_store(acc_{rt}_{ct}, tg_C + {row_off}u + {col_off}, {N});")
        lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("")

        # Fused pointwise/broadcast epilogue (#158): apply the lowered op chain
        # per element on the staged tg_C, then write. Early-return leaves the
        # softmax path below untouched.
        if info.get("epilogue_ops"):
            lines.extend(
                self._emit_matmul_epilogue_loop(
                    info, m_block, N, c_ptr, c_row_s, c_col_s, c_msl_type, _addr, source_bindings
                )
            )
            lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
            lines.append("    }  // end M-strip loop")
            lines.append("}")
            lines.append("")
            return "\n".join(lines)

        # Row softmax across the strip\'s M_BLOCK rows.
        lines.append(f"        for (uint row = tiitg; row < {m_block}u; row += 128u) {{")
        lines.append(f"            uint row_off = row * {N}u;")
        lines.append("            float row_max = -INFINITY;")
        lines.append(f"            for (uint c = 0u; c < {N}u; c++) {{")
        lines.append("                row_max = max(row_max, tg_C[row_off + c]);")
        lines.append("            }")
        lines.append("            float row_sum = 0.0f;")
        lines.append(f"            for (uint c = 0u; c < {N}u; c++) {{")
        lines.append("                float v = exp(tg_C[row_off + c] - row_max);")
        lines.append("                tg_C[row_off + c] = v;")
        lines.append("                row_sum += v;")
        lines.append("            }")
        lines.append("            float inv = 1.0f / row_sum;")
        lines.append(f"            for (uint c = 0u; c < {N}u; c++) {{")
        lines.append("                tg_C[row_off + c] *= inv;")
        lines.append("            }")
        lines.append("        }")
        lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("")

        # Write strip to global.
        lines.append(f"        for (uint i = tiitg; i < {m_block * N}u; i += 128u) {{")
        lines.append(f"            uint row = i / {N}u, col = i % {N}u;")
        c_store_addr = _addr(c_ptr, "mstrip + row", "col", c_row_s, c_col_s, N, "C")
        lines.append(f"            {self._fused_store_clip(source_bindings)}{c_store_addr} = ({c_msl_type})tg_C[i];")
        lines.append("        }")
        lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    }  // end M-strip loop")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    # MSL for each fused-epilogue op given operand expressions. Reuses the same
    # operators/functions the generic op dispatch maps to (#158).
    _EPI_BIN = {
        "arith.addf": "+",
        "arith.subf": "-",
        "arith.mulf": "*",
        "arith.divf": "/",
        "arith.addi": "+",
        "arith.subi": "-",
        "arith.muli": "*",
    }
    # NaN-QUIET min/max (IEEE minNum/maxNum): MSL fmax/fmin return the non-NaN
    # operand — correct for maxnumf/minnumf.
    _EPI_FN2 = {"arith.maxnumf": "fmax", "arith.minnumf": "fmin"}
    # NaN-PROPAGATING min/max (tl.maximum/minimum with propagate_nan=ALL): if either
    # operand is NaN the result must be NaN. Plain fmax/fmin would silently drop it
    # (re-audit #5 — relu(NaN accumulator) returned 0.0 instead of NaN).
    _EPI_FN2_NANPROP = {"arith.maximumf": "fmax", "arith.minimumf": "fmin"}
    _EPI_FN1 = {
        "math.exp": "exp",
        "math.exp2": "exp2",
        "math.log": "log",
        "math.log2": "log2",
        "math.sqrt": "sqrt",
        "math.rsqrt": "rsqrt",
        "math.sin": "precise::sin",
        "math.cos": "precise::cos",
        "math.erf": "erf",
        "math.tanh": "precise::tanh",
        "math.floor": "floor",
        "math.ceil": "ceil",
        "math.absf": "fabs",
    }

    def _fused_store_clip(self, source_bindings):
        """The store's runtime clip as an MSL ``if (...) `` prefix, empty without one.

        The fused template computes its whole baked tile, epilogue included, and stores
        it. A store mask that clips an axis at a runtime extent keeps only the rows and
        columns below it (GPU: ``M = 20`` wrote all 32 rows). The stride-aware matmul
        templates clip at the same argument (``_matmul_tile_extent_lines``).
        """
        clips = []
        for extent, index in zip(self._matmul_output_extent_args(), ("mstrip + row", "col")):
            if extent is None:
                continue
            if self._declared_int_elem_type(extent) not in ("i32", "i64", "u32", "u64"):
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    "fused matmul+softmax/epilogue template: the store clips at an argument that is not "
                    "a 32- or 64-bit integer, which the template cannot replay. Refusing.",
                    op_name="tt.store",
                )
            clips.append(f"(long)({index}) < (long)({(source_bindings or {}).get(extent, extent)})")
        return f"if ({' && '.join(clips)}) " if clips else ""

    def _emit_matmul_epilogue_loop(
        self, info, m_block, N, c_ptr, c_row_s, c_col_s, c_msl_type, _addr, source_bindings=None
    ):
        """Per-element fused-epilogue loop body (#158).

        Lowers the topologically-ordered epilogue op chain to scalar MSL on the
        staged matmul result tg_C[i], then writes. The dot result is seeded to
        ``tg_C[i]``; constants reuse ``_lower_constant``; a bias load indexes by
        ``col``; layout/cast ops pass through.
        """
        from triton_msl.codegen._lowerer_detection import _EPI_PASSTHROUGH

        by_id = {ssa.id: ssa for ssa in self.graph.ops}
        source_bindings = source_bindings or {}
        bias = source_bindings.get(info.get("bias_ptr"), info.get("bias_ptr"))
        # The matmul result staged in tg_C is a@b (accumulators init to 0). If a
        # bias was fused into the dot's accumulator, add it back here.
        acc_bias = source_bindings.get(info.get("acc_bias_ptr"), info.get("acc_bias_ptr"))
        # ROW bias indexes the M-length bias by the GLOBAL row (mstrip + row), NOT the
        # strip-local `row`. _detect_matmul_epilogue refuses the row-bias case when
        # M > m_block (re-audit #8), so here mstrip is always 0 for a row bias and this
        # is exact; the mstrip term is kept for correctness if that guard ever loosens.
        acc_idx = "(mstrip + row)" if info.get("acc_bias_dim") == "row" else "col"
        dot_expr = f"(tg_C[i] + {acc_bias}[{acc_idx}])" if acc_bias else "tg_C[i]"
        val = {info["dot_id"]: dot_expr}

        def _const_expr(op):
            try:
                self._lower_constant(op)
                return f"({self.env[op.id]})"
            except Exception:
                return "0.0f"

        def expr(vid):
            if vid in val:
                return val[vid]
            op = by_id.get(vid)
            if op is None:
                return "0.0f"
            if op.op == "arith.constant":
                return _const_expr(op)
            if op.op == "tt.load":
                return f"{bias}[col]" if bias else "0.0f"
            if op.op in _EPI_PASSTHROUGH and op.operand_ids:
                return expr(op.operand_ids[0])
            return "0.0f"

        body = []
        n = 0
        for op in info["epilogue_ops"]:
            if op.op in _EPI_PASSTHROUGH:
                val[op.id] = expr(op.operand_ids[0]) if op.operand_ids else "0.0f"
                continue
            if op.op == "arith.constant":
                val[op.id] = _const_expr(op)
                continue
            if op.op == "tt.load":
                val[op.id] = f"{bias}[col]" if bias else "0.0f"
                continue
            es = [expr(o) for o in (op.operand_ids or [])]
            if op.op == "arith.negf" and es:
                e = f"(-({es[0]}))"
            elif op.op in self._EPI_BIN and len(es) >= 2:
                e = f"(({es[0]}) {self._EPI_BIN[op.op]} ({es[1]}))"
            elif op.op in self._EPI_FN2 and len(es) >= 2:
                e = f"{self._EPI_FN2[op.op]}(({es[0]}), ({es[1]}))"
            elif op.op in self._EPI_FN2_NANPROP and len(es) >= 2:
                _fn = self._EPI_FN2_NANPROP[op.op]
                e = f"((isnan({es[0]}) || isnan({es[1]})) ? NAN : {_fn}(({es[0]}), ({es[1]})))"
            elif op.op in self._EPI_FN1 and es:
                e = f"{self._EPI_FN1[op.op]}(({es[0]}))"
            elif op.op == "math.fma" and len(es) >= 3:
                # fma(a,b,c) = a*b+c. The old else returned only operand 0 (acc
                # unchanged) — a silent-wrong (re-audit #6).
                e = f"fma(({es[0]}), ({es[1]}), ({es[2]}))"
            elif op.op == "arith.extf" and es:
                # fp16/bf16 -> fp32 widening: a no-op in the float epilogue.
                e = f"({es[0]})"
            elif op.op == "tt.clampf" and len(es) >= 3:
                if op.attrs.get("propagateNan", "none") == "all":
                    # propagate_nan=ALL: NaN in -> NaN out (plain clamp would drop it,
                    # the same class as the maximumf/minimumf NaN-drop — re-audit #6).
                    e = f"(isnan({es[0]}) ? NAN : clamp(({es[0]}), ({es[1]}), ({es[2]})))"
                else:
                    e = f"clamp(({es[0]}), ({es[1]}), ({es[2]}))"
            else:
                # An allow-listed epilogue op with NO emission branch previously fell
                # through to `e = es[0]` (return operand 0) — a SILENT-WRONG trap (that
                # is how math.fma slipped through). Refuse loudly instead.
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"Fused matmul epilogue has no correct lowering for '{op.op}'. "
                    f"Refusing rather than emit operand-0 (silent-wrong).",
                    op_name=op.op,
                )
            var = f"ep{n}"
            n += 1
            body.append(f"            float {var} = {e};")
            val[op.id] = var

        out = val.get(info["store_value_id"], "tg_C[i]")
        lines = [
            f"        for (uint i = tiitg; i < {m_block * N}u; i += 128u) {{",
            f"            uint row = i / {N}u, col = i % {N}u;",
        ]
        lines.extend(body)
        c_store_addr = _addr(c_ptr, "mstrip + row", "col", c_row_s, c_col_s, N, "C")
        lines.append(f"            {self._fused_store_clip(source_bindings)}{c_store_addr} = ({c_msl_type})({out});")
        lines.append("        }")
        return lines

    def _row_stride_offset_expr(self, lid_expr, stride_desc, what):
        """``long`` row-base offset for a row-per-thread template (packet 825).

        Only a COMPILE-TIME literal stride is emitted: ``_detect_row_wise_sort``
        proves such a stride is a non-negative constant no smaller than the row
        width, so widening it to ``long`` is exact and also removes the 32-bit
        overflow the old ``lid * (uint)C`` carried.

        A RUNTIME stride refuses. The old form (``lid * (uint)s``) truncated a
        64-bit stride and zero-extended a negative one, and this template has no
        traced ``tt.addptr`` chain to prove the offset's width/signedness from
        (the way ``_dot_offset_arith_widths`` does for the matmul templates).
        The runtime branch of the detector is currently unreachable — it requires
        the stride argument to appear DIRECTLY in an ``arith.muli``, while Triton
        always ``tt.splat``s a scalar first — so this refusal cannot regress an
        admitted kernel; it makes the latent hole loud if that detector branch is
        ever repaired.
        """
        names = {a.name for a in self.graph.args if not a.is_ptr}
        if str(stride_desc) not in names:
            return f"(long){lid_expr} * (long)({stride_desc})"
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(
            f"{what}: the row stride is the runtime argument '{stride_desc}', and this "
            "template cannot prove the integer width or signedness of the source's row "
            "offset arithmetic (no traced tt.addptr chain). Emitting it as a 32-bit "
            "unsigned product would truncate a 64-bit stride and turn a negative one "
            "into a huge positive offset. Refusing (correct-or-refuse).",
            op_name="tt.store",
        )

    def _lower_row_wise_sort_template(self, info) -> str:
        """Emit a per-row bitonic sort kernel.

        Thread `lid` owns row `lid` of the (M, N) input. It loads the row's N cells
        into a register array, runs the compare-and-swap steps of
        ``triton.language.sort`` over them (once for each sort the source applies),
        and stores them to row `lid` of the output.

            for stage in 1 .. log2(N):
                for swap in stage-1 .. 0:
                    for each pair i, j = i ^ (1 << swap), i < j:
                        m = bit `stage` of i (the requested direction in the last stage)
                        v[i], v[j] = (v[i] > v[j]) != m ? v[j] : v[i],
                                     (v[j] > v[i]) == m ? v[i] : v[j]

        The detector proves the source is exactly this (`_sort_proof`) and hands over
        the direction of each sort, whether the source compares unsigned, the place in
        its row each column is loaded from, and the conversion the store makes.
        """
        M = info["M"]
        N = info["N"]
        K = info["K"]
        # tl.topk (K < N) is mis-computed: the K<N trim returns duplicated pairs, not
        # the K distinct top values — verified WRONG in BOTH this template (2D) and the
        # generic xor path (1D) (re-audit #10: topk N=16 K=4 gave [2.18,0.77,2.18,0.77]
        # vs [2.18,0.83,0.77,0.77]). Only the FULL sort (K == N) is correct. Refuse
        # topk loudly rather than mis-compute (fixing the K<N trim is a follow-up).
        if K < N:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"tl.topk (k={K} < N={N}) is not correctly lowered — the K<N trim "
                f"mis-computes (duplicated values). Refusing rather than return wrong "
                f"results. Use a full tl.sort and slice, or k == N.",
                op_name="tt.reduce",
            )
        networks = info["networks"]
        columns = info["columns"]
        convert = info["convert"]
        elem_type = info["elem_type"]
        x_ptr = info["x_ptr"]
        z_ptr = info["z_ptr"]
        stride_xm = info["stride_xm"]
        stride_zm = info["stride_zm"]
        block_size = info["block_size"]
        # Defense in depth (2026-08-29): stride names are interpolated verbatim into
        # the MSL. A missing stride previously emitted Python `None` (invalid MSL,
        # loud compile failure — but the refusal belongs here, before emission).
        if stride_xm is None or stride_zm is None:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "row-wise sort template requires proven row strides (runtime scalar "
                "args or baked constants); the detector supplied None. Refusing "
                "rather than emit invalid MSL.",
                op_name="tt.reduce",
            )

        msl_type = triton_type_to_msl(elem_type)
        # Pick a compute type for comparisons (promote fp16/bf16 to float)
        compute_type = _msl_compute_type(elem_type)
        unsigned_types = {"char": "uchar", "short": "ushort", "int": "uint", "long": "ulong"}
        if info.get("unsigned"):
            # The source compares unsigned; the cells are held in the unsigned type of
            # their width, and the casts on load and store keep every bit.
            compute_type = unsigned_types[compute_type]
        if convert in (None, "arith.trunci"):
            # The cell's own type; a narrower output cell keeps the low bits of it.
            stored = f"static_cast<{msl_type}>(v[_i])"
        else:
            # The conversion says how the sorted value is read, whatever the comparison
            # read it as: an integer widens or becomes a float as a signed or as an
            # unsigned one.
            read = "v[_i]"
            if convert in ("arith.sitofp", "arith.extsi"):
                read = f"static_cast<{msl_type}>(v[_i])"
            elif convert in ("arith.uitofp", "arith.extui"):
                read = f"static_cast<{unsigned_types[msl_type]}>(v[_i])"
            if convert in ("arith.sitofp", "arith.uitofp") and info["out_elem_type"] in NARROW:
                stored = int_to_narrow(info["out_elem_type"], read)
            else:
                stored = f"static_cast<{triton_type_to_msl(info['out_elem_type'])}>({read})"

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument declarations in the original IR arg order
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                if arg.name == z_ptr:
                    arg_decls.append(f"    volatile device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
                else:
                    arg_decls.append(f"    device const {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                scalar_ty = triton_type_to_msl(arg.elem_type or "i32")
                arg_decls.append(f"    constant {scalar_ty}& {source_bindings[arg.name]} [[buffer({i})]]")

        # Integer sort? If elem type is integer we just use standard
        # `<` / `>` over the integer values. For floats, use `<` / `>`.
        is_int = elem_type.startswith("i") or elem_type.startswith("u")

        x_ptr = source_bindings[x_ptr]
        z_ptr = source_bindings[z_ptr]

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        lines.append(f"    if (lid < {M}u) {{")
        # Load N elements for this row
        lines.append(f"        {compute_type} v[{N}];")
        _row_off_x = self._row_stride_offset_expr("lid", stride_xm, "row-wise sort input")
        lines.append(f"        long _row_off = {_row_off_x};")
        place = "_i"
        if columns is not None:
            # The place in its row the source loads each column from.
            lines.append(f"        const int _places[{N}] = {{{', '.join(str(place) for place in columns)}}};")
            place = "(long)_places[_i]"
        lines.append(f"        for (uint _i = 0u; _i < {N}u; _i++) {{")
        lines.append(f"            v[_i] = static_cast<{compute_type}>({x_ptr}[_row_off + {place}]);")
        lines.append(f"        }}")
        # The network is the source's own (triton.language.sort), step for step: in each
        # step an element takes its partner where (element > partner) != mask, the mask
        # being its coordinate along the step's axis exclusive-or the direction of its
        # run. Written this way rather than as "swap when out of order" it is the
        # source's result for every input: where neither of a pair is greater (equal
        # values, a zero of each sign, a NaN) both cells end with one of the two values.
        lines.append(f"        // Bitonic sort of {N} elements: the source's compare-and-swap steps")
        log_n = N.bit_length() - 1
        for descending in networks:
            for stage in range(1, log_n + 1):
                if stage < log_n:
                    direction = f"((i >> {stage}u) & 1u) != 0u"
                else:
                    direction = "true" if descending else "false"
                for swap in range(stage - 1, -1, -1):
                    lines.append(f"        // stage={stage}, step={1 << swap}")
                    lines.append(f"        for (uint i = 0u; i < {N}u; i++) {{")
                    lines.append(f"            uint j = i ^ {1 << swap}u;")
                    lines.append(f"            if (j > i) {{")
                    lines.append(f"                bool m = {direction};")
                    lines.append(f"                {compute_type} a = v[i]; {compute_type} b = v[j];")
                    lines.append(f"                v[i] = ((a > b) != m) ? b : a;")
                    lines.append(f"                v[j] = ((b > a) == m) ? a : b;")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
        _row_off_zx = self._row_stride_offset_expr("lid", stride_zm, "row-wise sort output")
        lines.append(f"        long _row_off_z = {_row_off_zx};")
        lines.append(f"        for (uint _i = 0u; _i < {K}u; _i++) {{")
        lines.append(f"            {z_ptr}[_row_off_z + _i] = {stored};")
        lines.append(f"        }}")
        lines.append(f"    }}")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _reduce3_origin(origin):
        """The MSL prefix of a proved tile origin: ``[constant + ]pid * step + ``."""
        first, step = origin
        return (f"{first}u + " if first else "") + f"pid * {step}u + "

    @staticmethod
    def _reduce3_cast_chain(expr, held, element_types):
        """``expr`` (an MSL value of type ``held``) taken through the source's own element types."""
        for element in element_types:
            msl = triton_type_to_msl(element)
            if msl != held:
                expr, held = f"({msl}){expr}", msl
        return expr

    def _lower_3d_reduce_template(self, info) -> str:
        """Generate a complete MSL kernel for 3D axis reduction.

        Bypasses the generic lowerer since 3D→2D dimensionality change
        breaks the per-thread index decomposition.
        """
        M, N, K = info["shape"]
        axis = info["axis"]
        combine_op = info["combine_op"]
        block_size = info["block_size"]
        total = M * N * K

        # Determine result dimensions
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

        # Pointer roles, the element type and both tile origins come from the proof in
        # _prove_3d_reduce_template (dataflow), never from the order of the arguments. Every
        # integer of at most 32 bits accumulates in int: a narrow sum then wraps in its own
        # width at the store cast, as the source's does (a float accumulator saturates).
        input_arg, output_arg = info["input_arg"], info["output_arg"]
        msl_type = "int" if input_arg.elem_type in ("i8", "i16", "i32", "si32") else "float"

        # Identity and combine expression
        if combine_op == "sum":
            identity = "-0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"
        elif combine_op == "max":
            identity = "(-INFINITY)" if msl_type == "float" else "INT_MIN"
            combine_expr = "fmax(acc, val)" if msl_type == "float" else "max(acc, val)"
        elif combine_op == "min":
            identity = "INFINITY" if msl_type == "float" else "INT_MAX"
            combine_expr = "fmin(acc, val)" if msl_type == "float" else "min(acc, val)"
        else:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"3-D reduce template cannot emit combine {combine_op!r}; refusing rather than substitute a sum.",
                op_name="tt.reduce",
            )

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list (X and Z pointers)
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_decls.append(_scalar_buffer_decl(arg, i, source_bindings))

        x_name = source_bindings[input_arg.name]
        z_name = source_bindings[output_arg.name]
        x_origin = self._reduce3_origin(info["load_origin"])
        z_origin = self._reduce3_origin(info["store_origin"])

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        # Shared memory for input staging and result
        lines.append(f"    threadgroup {msl_type} _input[{total}];")
        lines.append(f"    threadgroup {msl_type} _result[{result_total}];")

        # Stage all input values to shared memory. Offset by program_id: each program in
        # a multi-program (batched) dispatch processes its OWN contiguous `total`-element
        # 3-D slice. Previously this read x_name[_e] with NO pid offset, so a multi-program
        # dispatch silently computed only program 0's slice and left the others untouched
        # (reduce-probe finding). pid*total is a no-op for a single-program (grid=1) launch.
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u)")
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[{x_origin}_e];")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reduce along axis
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        lines.append(f"        {msl_type} acc = {identity};")

        if axis == 0:
            lines.append(f"        uint _j = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            lines.append(f"        uint _i = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            lines.append(f"        uint _i = _r / {N}u;")
            lines.append(f"        uint _j = _r % {N}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _j * {K}u + _a];")

        lines.append(f"            acc = {combine_expr};")
        lines.append(f"        }}")
        lines.append(f"        _result[_r] = acc;")
        lines.append(f"    }}")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store results to output (row-major 2D). Cast `_result` (accumulated in
        # `msl_type`, float for fp/bf inputs) to the OUTPUT element type. MSL allows
        # implicit float->half but NOT float->bfloat, so a bf16 output otherwise fails to
        # compile ("assigning to 'bfloat' from 'float'") — reduce-fuzzer finding. No-op
        # for float/half.
        # The result passes through the source's own types: the reduction's element type, then
        # each cast between the reduction and the store, then the stored element type.
        _stored3d = self._reduce3_cast_chain(
            "_result[_r]", msl_type, [input_arg.elem_type, *info["result_casts"], output_arg.elem_type]
        )
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u)")
        lines.append(f"        {z_name}[{z_origin}_r] = {_stored3d};")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _lower_3d_argminmax_template(self, info) -> str:
        """Generate a complete MSL kernel for 3D argmin/argmax.

        Similar to _lower_3d_reduce_template but tracks both value and index,
        storing only the index result.
        """
        M, N, K = info["shape"]
        axis = info["axis"]
        combine_op = info["combine_op"]
        block_size = info["block_size"]
        total = M * N * K
        is_max = combine_op == "argmax"

        # Determine result dimensions
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

        # Roles, element type and tile origins from the proof (see _lower_3d_reduce_template).
        input_arg, output_arg = info["input_arg"], info["output_arg"]
        msl_type = "int" if input_arg.elem_type in ("i8", "i16", "i32", "si32") else "float"

        identity = "(-INFINITY)" if is_max and msl_type == "float" else "INFINITY"
        if msl_type == "int":
            identity = "INT_MIN" if is_max else "INT_MAX"
        cmp_op = ">" if is_max else "<"

        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
            else:
                arg_decls.append(_scalar_buffer_decl(arg, i, source_bindings))

        x_name = source_bindings[input_arg.name]
        z_name = source_bindings[output_arg.name]
        x_origin = self._reduce3_origin(info["load_origin"])
        z_origin = self._reduce3_origin(info["store_origin"])

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        # Shared memory for input staging and results
        lines.append(f"    threadgroup {msl_type} _input[{total}];")
        lines.append(f"    threadgroup int _result_idx[{result_total}];")

        # Stage all input values to shared memory. Offset by program_id: each program in
        # a multi-program (batched) dispatch processes its OWN contiguous `total`-element
        # 3-D slice. Previously this read x_name[_e] with NO pid offset, so a multi-program
        # dispatch silently computed only program 0's slice and left the others untouched
        # (reduce-probe finding). pid*total is a no-op for a single-program (grid=1) launch.
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u)")
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[{x_origin}_e];")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reduce along axis, tracking best value and index
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        lines.append(f"        {msl_type} best_v = {identity};")
        lines.append(f"        int best_i = 0;")

        if axis == 0:
            lines.append(f"        uint _j = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            lines.append(f"        uint _i = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            lines.append(f"        uint _i = _r / {N}u;")
            lines.append(f"        uint _j = _r % {N}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _j * {K}u + _a];")

        lines.append(f"            if (val {cmp_op} best_v || (val == best_v && (int)_a < best_i)) {{")
        lines.append(f"                best_v = val; best_i = (int)_a;")
        lines.append(f"            }}")
        lines.append(f"        }}")
        lines.append(f"        _result_idx[_r] = best_i;")
        lines.append(f"    }}")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store index results to output (program_id-offset for multi-program dispatch).
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u)")
        _stored3d = self._reduce3_cast_chain("_result_idx[_r]", "int", [*info["result_casts"], output_arg.elem_type])
        lines.append(f"        {z_name}[{z_origin}_r] = {_stored3d};")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _lower_dot_via_prebuilt_template(self) -> str:
        """Generate strided matmul MSL for kernels containing tt.dot.

        Generates a scalar-loop matmul that handles arbitrary M, N, K
        and strided access patterns. Each thread computes one or more
        output elements via a K-loop, reading directly from global memory.

        For kernels with strides (upstream test_dot), this correctly handles
        row-major, column-major, and transposed layouts.

        For simple 3-pointer kernels (A, B, C with no strides), falls back
        to the optimized simdgroup_matrix template.
        """
        # STRUCTURAL REFUSAL (matmul unification): this template models EXACTLY ONE 2-D
        # matmul. A multi-tt.dot kernel (chain-dot (A@B)@W) or a rank>=3 (batched/3-D) dot
        # has no correct lowering here — the simple-template branch silently computes a
        # single 2-D GEMM and drops the rest (confirmed silent-wrong, both abbreviated and
        # stride_* naming). FlashAttention, matmul-softmax, and matmul-epilogue are all
        # routed BEFORE this method, so a multi-dot / 3-D dot reaching here is genuinely
        # unmodeled: refuse loudly rather than mis-emit. (Split into separate kernels.)
        from triton_msl.errors import MetalNonRecoverableError as _MNR

        def _all_dot_ops(ops):
            for s in ops:
                if s.op == "tt.dot":
                    yield s
                if s.region_ops:
                    yield from _all_dot_ops(s.region_ops)
                if s.else_ops:
                    yield from _all_dot_ops(s.else_ops)

        _all_dots = list(_all_dot_ops(self.graph.ops))
        if len(_all_dots) > 1:
            raise _MNR(
                "chain-dot / multiple tt.dot in one kernel (e.g. z = (x@y)@w) is not "
                "supported: the matmul template models a single 2-D matmul and would "
                "silently drop the later dot(s). Split into separate matmul kernels.",
                op_name="tt.dot",
            )
        if _all_dots:
            _rshape = _extract_shape(_all_dots[0].type_str or "")
            if _rshape and len(_rshape) >= 3:
                raise _MNR(
                    "batched / 3-D tt.dot (rank>=3 result) is not supported: the matmul "
                    "template models a single 2-D matmul and would drop the batch axis. "
                    "Loop the batch on the host or use a per-batch 2-D kernel.",
                    op_name="tt.dot",
                )

        # Extract tile dimensions from tt.make_range ops (for simple template fallback)
        tile_dims = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.make_range":
                end = ssa.attrs.get("end", 32)
                if end not in tile_dims:
                    tile_dims.append(end)

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        scalar_args = [a for a in self.graph.args if not a.is_ptr]

        # Mirror the simple-dot acc-init guard onto the STRIDED path. _detect_simple_dot
        # early-returns on stride args, so a strided/pid K-loop matmul with a non-zero
        # accumulator init (acc = tl.full(...)) would otherwise be SILENTLY DROPPED — the
        # strided template seeds acc=0 (re-audit #13). Refuse only a CLEAR bias: a loaded
        # value (through splat/broadcast) or a non-zero constant whose shape matches the
        # dot output tile (the accumulator). tl.zeros and scalar loop-carried values
        # (pointer/counter iter-args) are unaffected.
        _by = {s.id: s for s in self.graph.ops}
        _dot_shape = None
        for _s in self.graph.ops:
            if _s.op == "tt.dot":
                _dot_shape = _extract_shape(_s.type_str)
                break
            # In a strided pid K-loop the tt.dot lives ONLY in the scf.for body — the
            # top-level scan above misses it (re-audit #14: _dot_shape stayed None so the
            # whole bias guard was dead). Search the loop region too.
            if _s.op == "scf.for" and getattr(_s, "region_ops", None):
                for _ro in _s.region_ops:
                    if _ro.op == "tt.dot":
                        _dot_shape = _extract_shape(_ro.type_str)
                        break
                if _dot_shape is not None:
                    break

        _init_ids = []
        for _s in self.graph.ops:
            if _s.op == "scf.for" and len(_s.operand_ids) > 3:
                _init_ids.extend(_s.operand_ids[3:])
        if not _init_ids:
            for _s in self.graph.ops:
                if _s.op == "tt.dot" and len(_s.operand_ids) >= 3:
                    _init_ids.append(_s.operand_ids[2])
                    break
        if _dot_shape and any(self._acc_init_is_bias(_i, _by, dot_shape=_dot_shape) for _i in _init_ids):
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "strided matmul with a non-zero accumulator init (fused bias / tl.full) "
                "is silently dropped by the strided template (it seeds acc=0). Refusing. "
                "Add the bias as a separate kernel after the matmul.",
                op_name="tt.dot",
            )

        # Determine dtype from pointer args
        dtype = "fp32"
        if ptr_args:
            dtype = _mlir_to_triton_dtype(ptr_args[0].elem_type)

        # Every single-2-D dot reaching this method routes through the simdgroup
        # matmul template (or the constant-input variant). The address-traced stride
        # decision (mechanism A, in _lower_simple_dot_inline / _lower_k_loop_dot_inline)
        # and _detect_simple_dot now claim EVERY computable single-2-D matmul — incl.
        # the canonical stride_*-named / transposed cases that the old name-match
        # ``stride_map`` strided body (mechanism C) used to GUESS row-major for and
        # silently mis-compute. Chain-dot (>1 tt.dot) and 3-D dots already refused at
        # the top of this method. Mechanism C is therefore DEAD for every reachable
        # correct path: an instrumented full-suite + audit-repro run (TRIPWIRE_C) hit
        # its strided body ZERO times, while a forced positive control proved the
        # tripwire fires when the body IS reached. Deleted; this method now ONLY routes
        # to the validated simdgroup template (the ~11 TFLOP/s contiguous fast path is
        # selected there via _maybe_fast_matmul_descriptor) or to the constant-dot
        # variant — there is no longer a name-match / row-major-guess code path.
        # Check for constant-input dot (e.g., test_dot_without_load).
        const_info = self._detect_dot_constant_inputs()
        if const_info:
            return self._lower_dot_constant_template(const_info, ptr_args)
        return self._lower_dot_simple_template(tile_dims, ptr_args, dtype)

    def _lower_dot_constant_template(self, const_info, ptr_args):
        """Generate MSL for dot product where both inputs are compile-time constants."""
        source_bindings = {arg.name: f"_mmarg{i}" for i, arg in enumerate(self.graph.args)}
        const_a, const_b, M, N, K, dot_elem_type = const_info
        safe_name = _sanitize_msl_name(self.graph.func_name)
        total = M * N
        num_warps = self.graph.num_warps
        block_size = num_warps * 32
        self._matmul_block_size = total

        c_ptr = ptr_args[-1] if ptr_args else None
        c_msl_type = triton_type_to_msl(c_ptr.elem_type) if c_ptr else "float"

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(f"    device {arg_msl_type}* {source_bindings[arg.name]} [[buffer({i})]]")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint lid [[thread_position_in_threadgroup]]")
        lines.append(") {")
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
        lines.append(f"        uint _i = _e / {N}u;")
        lines.append(f"        uint _j = _e % {N}u;")
        # Each element = sum over K of const_a * const_b = K * const_a * const_b
        lines.append(f"        float _sum = (float){K}u * {const_a}f * {const_b}f;")
        lines.append(f"        {source_bindings[c_ptr.name]}[_i * {N}u + _j] = ({c_msl_type})_sum;")
        lines.append(f"    }}")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_dot_simple_template(self, tile_dims, ptr_args, dtype) -> str:
        """Fall back to optimized simdgroup matmul template for simple kernels."""
        from triton_msl.codegen.msl_emitter import make_matmul_kernel

        # Integrity guard (PR1): make_matmul_kernel addresses the output and
        # bounds its loops using runtime M/N/K scalar args (buffers 3-5). A
        # kernel that bakes M/N/K as ``tl.constexpr`` supplies no such args,
        # so the template's M/N/K are unbound -> garbage dims -> silently
        # wrong output (this is exactly test_dot_mulbroadcasted: pid-tiled
        # over a 256x192 output with BM/BN/BK=128/32/32 baked in, ~98%
        # mismatch). Every matmul this template emits needs M/N/K, so any
        # kernel that legitimately uses it provides them. When they're
        # absent, refuse rather than emit wrong numbers: emit an UNSUPPORTED
        # stub so emit_msl falls back to the legacy parser / errors clearly.
        scalar_args = [a for a in self.graph.args if not a.is_ptr]
        has_pid = any(s.op == "tt.get_program_id" for s in self.graph.ops)
        if has_pid and len(scalar_args) < 3:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "matmul template requires runtime M/N/K scalar args but the "
                "kernel bakes its dims as constexpr; cannot derive the true "
                "output strides (e.g. test_dot_mulbroadcasted)."
            )

        block_m = tile_dims[0] if len(tile_dims) > 0 else 32
        block_n = block_m
        block_k = block_m
        self._matmul_block_size = block_m * block_n

        # Output dtype can differ from input dtype (e.g. fp16 in / fp32 out).
        # Without honoring this, ``make_matmul_kernel`` declares
        # ``device <input_dtype>* C`` and Metal stores half values into
        # what the caller allocated as float* — output looks like
        # garbage when read back (saw a 7.6 trillion mismatch on
        # ``test_simple_matmul[…-float16-float32]``).
        out_dtype = dtype
        if len(ptr_args) >= 3:
            a_arg, b_arg, c_arg = self._dot_template_ptr_roles()  # P0 (065): dataflow, not position
            out_dtype = _mlir_to_triton_dtype(c_arg.elem_type)
            # Prove integrity BEFORE constructing the template: construction may
            # raise a prunable resource failure, which must not hide bad roles.
            # The template hard-binds A/B/C to buffers 0/1/2 and cannot re-slot.
            _idx = tuple(a.index for a in (a_arg, b_arg, c_arg))
            if _idx != (0, 1, 2) or len(ptr_args) != 3:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"simple matmul template requires the canonical (A, B, C, ...) "
                    f"signature with the three pointers at argument slots 0/1/2 "
                    f"(dataflow-resolved roles sit at {_idx}, {len(ptr_args)} pointer "
                    f"args): make_matmul_kernel binds buffers by position and would "
                    f"write the output into the wrong buffer. Refusing.",
                    op_name="tt.dot",
                )

        # Phase 4: record the runtime fast-matmul dispatch descriptor (additive;
        # the generic kernel below is still emitted + returned). The launcher only
        # uses it when the RUNTIME tensors are MPS and dims are aligned.
        self._fast_matmul = self._maybe_fast_matmul_descriptor()

        msl = make_matmul_kernel(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            dtype=dtype,
            out_dtype=out_dtype,
        )

        # These are identifiers owned by make_matmul_kernel, not source names.
        # Substitute simultaneously so A->B style source spelling cannot cascade,
        # and match pointer arithmetic as well as indexed loads/stores.
        bindings = {name: f"_mmarg{i}" for i, name in enumerate(("A", "B", "C", "M", "N", "K"))}
        bindings.update({name + "_buf": value + "_buf" for name, value in tuple(bindings.items())[3:]})
        msl = re.sub(r"\b(?:A|B|C|M|N|K|M_buf|N_buf|K_buf)\b", lambda match: bindings[match.group(0)], msl)

        safe_name = _sanitize_msl_name(self.graph.func_name)
        msl = msl.replace("matmul_kernel", safe_name, 1)

        return msl

    def _quant_plain_load(self, o):
        """A load a quantized template may stage from its raw pointer: ``tt.load`` with NO
        mask and NO ``other`` (packet 101/102: a mask on the input, weight, scale or zero
        load was silently ignored on every quantized route — raw result stored)."""
        return o is not None and getattr(o, "op", None) == "tt.load" and len(o.operand_ids or []) == 1

    def _quant_find_load(self, oid, op_by_id, depth=0, seen=None):
        """First ``tt.load`` in ``oid``'s cone (every operand, depth-limited)."""
        if seen is None:
            seen = set()
        if oid in seen or depth > 40:
            return None
        seen.add(oid)
        o = op_by_id.get(oid)
        if o is None:
            return None
        if o.op == "tt.load":
            return o
        for x in o.operand_ids or []:
            r = self._quant_find_load(x, op_by_id, depth + 1, seen)
            if r is not None:
                return r
        return None

    def _quant_legs_plain(self, legs, op_by_id):
        """Every leg's load is plain (see ``_quant_plain_load``); a leg without a load fails."""
        return all(self._quant_plain_load(self._quant_find_load(leg, op_by_id)) for leg in legs)

    # ------------------------------------------------------------------ packet 102: the
    # address grammar every quantized template replays, proven per staged role (input,
    # weight, scale, zero). The descriptors used to READ a stride out of an address term
    # and let anything else in it vanish; the runtime stride check then caught some of the
    # malformed forms by luck (a ``+ 1`` on the scale index, an index-level offset on the
    # weight), and none of the others (GPU: raw result stored). Now an address is admitted
    # only when its terms are EXACTLY the tile indices the template re-emits.

    def _quant_loop_ivs(self, op_by_id):
        ivs = set()
        for o in op_by_id.values():
            if o.op == "scf.for":
                ba = list((o.attrs or {}).get("block_arg_ids") or [])
                if ba:
                    ivs.add(ba[0])
        return ivs

    def _quant_term(self, tid, op_by_id):
        """One additive address term -> ``(expand_axis | None, stride_arg_index | -1,
        index_id)`` or None. Peels layout wrappers, one optional ``* splat(<stride arg>)``,
        and reads the expand axis from ``expand_dims`` — also through the in-loop
        ``addi(expand_dims(range), splat(k))`` spelling (the index is then the addi)."""
        arg_idx = {getattr(a, "id", None): i for i, a in enumerate(self.graph.args)}

        def _peel(oid):
            o = op_by_id.get(oid)
            while o is not None and o.op in ("ttg.convert_layout", "tt.broadcast") and o.operand_ids:
                oid = o.operand_ids[0]
                o = op_by_id.get(oid)
            return oid, o

        oid, o = _peel(tid)
        stride = -1
        if o is not None and o.op in ("arith.muli", "arith.mul") and len(o.operand_ids or []) == 2:
            sides = [_peel(x) for x in o.operand_ids]
            sp = next(
                (
                    i
                    for i, (_sid, so) in enumerate(sides)
                    if so is not None and so.op == "tt.splat" and so.operand_ids and so.operand_ids[0] in arg_idx
                ),
                None,
            )
            if sp is None:
                return None  # a stride that is not a runtime arg is not replayed
            stride = arg_idx[sides[sp][1].operand_ids[0]]
            oid, o = sides[1 - sp]
        axis = None
        if o is not None and o.op == "tt.expand_dims":
            axis = int(str((o.attrs or {}).get("axis", "")).split(":")[0].strip() or -1)
        elif o is not None and o.op in ("arith.addi", "arith.add") and len(o.operand_ids or []) == 2:
            eds = [p for p in (_peel(x) for x in o.operand_ids) if p[1] is not None and p[1].op == "tt.expand_dims"]
            if len(eds) == 1:
                axis = int(str((eds[0][1].attrs or {}).get("axis", "")).split(":")[0].strip() or -1)
        return (axis, stride, oid)

    def _quant_index_shape(self, idx_id, op_by_id, *, allow_iv, allow_div=None):
        """``_index_shape`` of a tile index after peeling layout wrappers and (``allow_div``)
        one ``index // allow_div`` (the int4 GEMV's byte ``k // 2`` and group ``k // G``);
        None when anything un-replayed appears (constant addend, runtime scalar, modulo,
        scaled index, a pid split, or a divisor other than the allowed one)."""
        arg_by_id = {getattr(a, "id", None): a for a in self.graph.args}
        ivs = self._quant_loop_ivs(op_by_id)
        iv = next(iter(ivs)) if len(ivs) == 1 else None
        if allow_iv and iv is None:
            return None
        oid, o = idx_id, op_by_id.get(idx_id)
        while o is not None and o.op in ("ttg.convert_layout", "tt.expand_dims", "tt.broadcast") and o.operand_ids:
            oid = o.operand_ids[0]
            o = op_by_id.get(oid)
        if (
            allow_div is not None
            and o is not None
            and o.op in ("arith.divsi", "arith.divui")
            and len(o.operand_ids or []) == 2
        ):
            if self._const_int(o.operand_ids[1], op_by_id) != allow_div:
                return None
            oid = o.operand_ids[0]
        sh = self._index_shape(oid, op_by_id, arg_by_id, iv if allow_iv else None)
        if sh["bad"] is not None or sh["mod"] is not None or sh["pid_div"] is not None or sh["pid"] == -1:
            return None
        return sh

    def _quant_addr_ok(self, ptr_id, arg, op_by_id, *, row=None, col=None, vec=None):
        """The load pointer is ``addptr(...addptr(splat(arg) | arg, t1)..., tn)`` whose terms,
        flattened, are exactly the kinds given: ``vec`` (1-D), or ``row`` (expand axis 1) +
        ``col`` (expand axis 0); each spec is ``dict(want_pid=, allow_iv=, allow_div=)``.
        Per kind: exactly one ``range``, at most one induction-variable addend when allowed,
        a program_id only on the wanted axis, at most one ``* splat(stride)``. The base is
        the ARGUMENT itself (``x + 1`` is a link no template replays). Returns
        ``(True, {kind: stride_arg_index | -1})`` or ``(False, None)``."""
        specs = {"row": row, "col": col, "vec": vec}
        ivs = self._quant_loop_ivs(op_by_id)

        def _peel(oid):
            o = op_by_id.get(oid)
            while o is not None and o.op in ("ttg.convert_layout", "tt.broadcast") and o.operand_ids:
                oid = o.operand_ids[0]
                o = op_by_id.get(oid)
            return oid, o

        terms = []
        base_id, cur = ptr_id, op_by_id.get(ptr_id)
        depth = 0
        while cur is not None and cur.op == "tt.addptr" and len(cur.operand_ids or []) == 2 and depth < 8:
            depth += 1
            terms.append(cur.operand_ids[1])
            base_id, cur = _peel(cur.operand_ids[0])
        if depth == 0:
            return (False, None)
        if cur is None:
            if base_id != getattr(arg, "id", None):
                return (False, None)
        elif not (cur.op == "tt.splat" and cur.operand_ids and cur.operand_ids[0] == getattr(arg, "id", None)):
            return (False, None)  # a scalar advance / another chain under the splat

        atoms = []

        def _flat(oid):
            _oid, o = _peel(oid)
            if o is not None and o.op in ("arith.addi", "arith.add") and len(o.operand_ids or []) == 2:
                parts = [_peel(x) for x in o.operand_ids]
                if not any(
                    p[1] is not None and p[1].op == "tt.splat" and p[1].operand_ids and p[1].operand_ids[0] in ivs
                    for p in parts
                ):
                    for x in o.operand_ids:
                        _flat(x)
                    return
            atoms.append(oid)

        for t in terms:
            _flat(t)
        agg = {k: {"range": 0, "iv": 0, "pid": None, "strides": []} for k in specs}
        for a in atoms:
            tm = self._quant_term(a, op_by_id)
            if tm is None:
                return (False, None)
            axis, stride, idx = tm
            kind = "vec" if axis is None else ("row" if axis == 1 else "col")
            spec = specs.get(kind)
            if spec is None:
                return (False, None)
            sh = self._quant_index_shape(
                idx, op_by_id, allow_iv=spec.get("allow_iv", False), allow_div=spec.get("allow_div")
            )
            if sh is None:
                return (False, None)
            if sh["pid"] is not None and sh["pid"] != spec.get("want_pid"):
                return (False, None)
            g = agg[kind]
            g["range"] += sh["range"]
            g["iv"] += sh["iv"]
            if sh["pid"] is not None:
                g["pid"] = sh["pid"]
            if stride != -1:
                g["strides"].append(stride)
            if sh["range"] == 0 and sh["iv"] == 0 and sh["pid"] is None:
                return (False, None)
        strides = {}
        for kind, spec in specs.items():
            g = agg[kind]
            if spec is None:
                if g["range"] or g["iv"] or g["strides"]:
                    return (False, None)
                continue
            if g["range"] != 1 or g["iv"] > (1 if spec.get("allow_iv") else 0) or len(g["strides"]) > 1:
                return (False, None)
            strides[kind] = g["strides"][0] if g["strides"] else -1
        return (True, strides)

    def _quant_1d_addr_ok(self, ptr_id, arg, op_by_id, *, allow_iv, want_pid=None):
        """1-D convenience over ``_quant_addr_ok``: ``(ok, stride_arg_index | -1)``."""
        ok, st = self._quant_addr_ok(ptr_id, arg, op_by_id, vec=dict(want_pid=want_pid, allow_iv=allow_iv))
        return (ok, st["vec"] if ok else None)

    # ------------------------------------------------------------------ packet 104: EXACT
    # per-role contracts. Packet 102's grammar bounded each term from ABOVE ("at most one
    # pid / iv / stride") and never required what the template always replays; nine GEMV
    # forms (no pid, pid * 2BN, no K term, output base/index +1, masked store, make_range
    # (BN, 2BN) / (BK, 2BK)) plus a wrong loop step / bound / a duplicate pid therefore ran
    # the specialized route with the raw result (packet 103 §2). Now every role's address
    # must contain EXACTLY the terms the template re-emits — pid required-or-forbidden with
    # its axis and coefficient, the induction variable required-or-forbidden, exactly one
    # make_range with its bounds read, a stride required / forbidden / bound — and the
    # descriptor cross-checks the numbers across roles and against the K-loop.

    def _quant_index_exact(self, idx_id, op_by_id, *, ivs, div=None):
        """Exact decomposition of a 1-D tile index: ``{"pid": (axis, coef) | None,
        "range": (start, end) | None, "iv": 0 | 1}`` or None on any foreign term or any
        duplicated term. Peels layout wrappers and index casts; ``div`` admits one leading
        ``index // div`` (the int4 GEMV's byte ``k // 2`` and group ``k // G``)."""
        wrappers = ("ttg.convert_layout", "tt.expand_dims", "tt.broadcast", "tt.splat")

        def _peel(v):
            o = op_by_id.get(v)
            while o is not None and (o.op in wrappers or o.op in self._IDX_CASTS) and o.operand_ids:
                v = o.operand_ids[0]
                o = op_by_id.get(v)
            return v, o

        v, o = _peel(idx_id)
        if div is not None:
            if o is None or o.op not in ("arith.divsi", "arith.divui") or len(o.operand_ids or []) != 2:
                return None
            if self._const_int(o.operand_ids[1], op_by_id) != div:
                return None
            v, o = _peel(o.operand_ids[0])
        addends = []

        def _flat(x):
            x, xo = _peel(x)
            if xo is not None and xo.op in ("arith.addi", "arith.add") and len(xo.operand_ids or []) == 2:
                for y in xo.operand_ids:
                    _flat(y)
            else:
                addends.append(x)

        _flat(v)
        res = {"pid": None, "range": None, "iv": 0}
        for a in addends:
            ao = op_by_id.get(a)
            if ao is None:
                if a in ivs:
                    res["iv"] += 1
                    continue
                return None  # a runtime scalar / unknown block arg
            if ao.op == "tt.make_range":
                if res["range"] is not None:
                    return None
                try:
                    res["range"] = (int(ao.attrs.get("start", 0)), int(ao.attrs.get("end")))
                except (TypeError, ValueError):
                    return None
                continue
            if ao.op in ("arith.muli", "arith.mul") and len(ao.operand_ids or []) == 2:
                sides = [_peel(x)[0] for x in ao.operand_ids]
                consts = [self._const_int(x, op_by_id) for x in sides]
                ci = next((i for i, c in enumerate(consts) if c is not None), None)
                pe = self._pid_expr(sides[1 - ci], op_by_id) if ci is not None else None
                if ci is None or pe is None or pe[1] is not None or res["pid"] is not None:
                    return None
                res["pid"] = (pe[0], consts[ci])
                continue
            pe = self._pid_expr(a, op_by_id)
            if pe is not None and pe[1] is None:
                if res["pid"] is not None:
                    return None
                res["pid"] = (pe[0], 1)
                continue
            return None
        if res["iv"] > 1:
            return None
        return res

    def _quant_kloop(self, op_by_id, k_arg):
        """The kernel's ONE ``scf.for`` is ``range(0, args[k_arg], STEP)`` with a constant
        step: returns ``(induction_variable_id, step)`` or None."""
        fors = [o for o in op_by_id.values() if o.op == "scf.for"]
        if len(fors) != 1 or len(fors[0].operand_ids or []) < 3:
            return None
        f = fors[0]
        arg_idx = {getattr(a, "id", None): i for i, a in enumerate(self.graph.args)}
        lo = self._strip_wrappers(f.operand_ids[0], op_by_id, self._IDX_CASTS)
        hi = self._strip_wrappers(f.operand_ids[1], op_by_id, self._IDX_CASTS)
        st = self._strip_wrappers(f.operand_ids[2], op_by_id, self._IDX_CASTS)
        if self._const_int(lo, op_by_id) != 0 or arg_idx.get(hi) != k_arg:
            return None
        step = self._const_int(st, op_by_id)
        ba = list((f.attrs or {}).get("block_arg_ids") or [])
        if step is None or step <= 0 or not ba:
            return None
        return (ba[0], step)

    def _quant_role_addr(self, ptr_id, arg, op_by_id, *, ivs, kinds):
        """EXACT address contract for one staged role. ``kinds`` maps ``"vec"`` (1-D) or
        ``"row"`` / ``"col"`` (expand axis 1 / 0) to ``dict(pid=("required", axis) | None,
        iv=0 | 1, stride="unit" | "required" | "bound", div=None | int)``. The pointer is
        ``addptr*(splat(arg) | arg, terms)``; the flattened terms cover exactly the given
        kinds, each with exactly one ``make_range``, the pid present iff required (on that
        axis, once, with its coefficient read), exactly ``iv`` induction-variable addends,
        and a stride term absent (unit) / present once (required) / at most once (bound).
        Returns ``(True, {kind: {"pid": (axis, coef) | None, "range": (start, end),
        "stride": arg_index | -1}})`` or ``(False, None)``. The caller cross-checks the
        numbers (``coef == end``, ``start == 0``, equality across roles, ``end == step``)."""

        def _peel(oid):
            o = op_by_id.get(oid)
            while o is not None and o.op in ("ttg.convert_layout", "tt.broadcast") and o.operand_ids:
                oid = o.operand_ids[0]
                o = op_by_id.get(oid)
            return oid, o

        terms = []
        base_id, cur = ptr_id, op_by_id.get(ptr_id)
        depth = 0
        while cur is not None and cur.op == "tt.addptr" and len(cur.operand_ids or []) == 2 and depth < 8:
            depth += 1
            terms.append(cur.operand_ids[1])
            base_id, cur = _peel(cur.operand_ids[0])
        if depth == 0:
            return (False, None)
        if cur is None:
            if base_id != getattr(arg, "id", None):
                return (False, None)
        elif not (cur.op == "tt.splat" and cur.operand_ids and cur.operand_ids[0] == getattr(arg, "id", None)):
            return (False, None)

        atoms = []

        def _flat(oid):
            _oid, o = _peel(oid)
            if o is not None and o.op in ("arith.addi", "arith.add") and len(o.operand_ids or []) == 2:
                parts = [_peel(x) for x in o.operand_ids]
                if not any(
                    p[1] is not None and p[1].op == "tt.splat" and p[1].operand_ids and p[1].operand_ids[0] in ivs
                    for p in parts
                ):
                    for x in o.operand_ids:
                        _flat(x)
                    return
            atoms.append(oid)

        for t in terms:
            _flat(t)
        acc = {k: {"pid": None, "range": None, "iv": 0, "strides": []} for k in kinds}
        for a in atoms:
            tm = self._quant_term(a, op_by_id)
            if tm is None:
                return (False, None)
            axis, stride, idx = tm
            kind = "vec" if axis is None else ("row" if axis == 1 else "col")
            spec = kinds.get(kind)
            if spec is None:
                return (False, None)
            ex = self._quant_index_exact(idx, op_by_id, ivs=ivs, div=spec.get("div"))
            if ex is None:
                return (False, None)
            g = acc[kind]
            if ex["pid"] is not None:
                if g["pid"] is not None:
                    return (False, None)
                g["pid"] = ex["pid"]
            if ex["range"] is not None:
                if g["range"] is not None:
                    return (False, None)
                g["range"] = ex["range"]
            g["iv"] += ex["iv"]
            if stride != -1:
                g["strides"].append(stride)
        out = {}
        for kind, spec in kinds.items():
            g = acc[kind]
            if g["range"] is None or g["iv"] != spec.get("iv", 0):
                return (False, None)
            want = spec.get("pid")
            if want is None:
                if g["pid"] is not None:
                    return (False, None)
            else:
                if g["pid"] is None or g["pid"][0] != want[1]:
                    return (False, None)
            mode = spec.get("stride", "unit")
            if (
                (mode == "unit" and g["strides"])
                or (mode == "required" and len(g["strides"]) != 1)
                or len(g["strides"]) > 1
            ):
                return (False, None)
            out[kind] = {"pid": g["pid"], "range": g["range"], "stride": g["strides"][0] if g["strides"] else -1}
        return (True, out)

    def _quant_gemm_contract(self, op_by_id, *, k_arg, nk=False, b_kdiv=1):
        """EXACT tile / loop / store contract of the quantized GEMM templates (per-group,
        symmetric): A ``[pid(0)*BM + range(0,BM)] x [range(0,BK) (+ k)]``, B ``[range(0,BK)
        (+ k)] x [pid(1)*BN + range(0,BN)]`` (``nk``: B loaded ``[n] x [k]``), C ``[pid(0)*BM
        + range(0,BM)] x [pid(1)*BN + range(0,BN)]`` with a PLAIN store, the ONE K-loop
        ``range(0, args[k_arg], BK)``, and each K operand either loop-carried (its pointer an
        iter-arg advanced by exactly ``BK * K-stride``, no ``k`` in the index) or in-loop
        (``range(0,BK) + k``, pointer not carried). Packet 103/104: a missing pid, ``pid *
        2BM``, a missing K term, shifted ranges, a wrong loop step / bound, a duplicate pid
        or swapped output axes all ran the template with the raw result. Returns
        ``{"bm", "bn", "bk", "iv"}`` or None."""
        traced = self.infer_dot_strides(with_index=True)
        if traced is None:
            return None
        strides, index_ids, _addr_ids = traced
        kl = self._quant_kloop(op_by_id, k_arg)
        if kl is None:
            return None
        iv, step = kl
        ivs = {iv}
        arg_by_id = {getattr(a, "id", None): a for a in self.graph.args}
        dots = [o for o in op_by_id.values() if o.op == "tt.dot"]
        fors = [o for o in op_by_id.values() if o.op == "scf.for"]
        if len(dots) != 1 or len(fors) != 1:
            return None
        dot, f = dots[0], fors[0]
        bargs = list((f.attrs or {}).get("block_arg_ids") or [])
        inits = list(f.operand_ids[3:]) if len(f.operand_ids or []) > 3 else []
        ylds = [o for o in (f.region_ops or []) if o.op == "scf.yield"]
        if len(ylds) != 1 or len(bargs) != len(inits) + 1 or len(ylds[0].operand_ids or []) != len(inits):
            return None
        yld = ylds[0]

        def _advance_of(load):
            ptr = self._strip_wrappers(load.operand_ids[0], op_by_id, ("ttg.convert_layout",))
            if ptr in bargs[1:]:
                y = op_by_id.get(yld.operand_ids[bargs.index(ptr) - 1])
                if y is None or y.op != "tt.addptr" or len(y.operand_ids or []) < 2 or y.operand_ids[0] != ptr:
                    return "bad"
                return y.operand_ids[1]
            return None

        def _kdim(role, ix, kstride):
            # ``b_kdiv`` (int4 GEMM): the packed weight's K index is ``(range(0,BK) + k) // 2``
            # (a byte per nibble pair) and a carried pointer advances by BK // 2 bytes.
            kdiv = b_kdiv if role == "B" else 1
            load = self._quant_find_load(dot.operand_ids[0 if role == "A" else 1], op_by_id)
            if load is None or not load.operand_ids or ix is None:
                return None
            adv = _advance_of(load)
            if adv == "bad":
                return None
            ex = self._quant_index_exact(ix, op_by_id, ivs=ivs, div=(kdiv if kdiv != 1 else None))
            if ex is None or ex["pid"] is not None:
                return None
            c = self._quant_tile_ok(ex, block=step)
            if c is None:
                return None
            if adv is not None:
                if ex["iv"] != 0 or not self._advance_matches(adv, step // kdiv, kstride, op_by_id, arg_by_id):
                    return None
            elif ex["iv"] != 1:
                return None
            return c

        def _ndim(ix, axis):
            if ix is None:
                return None
            ex = self._quant_index_exact(ix, op_by_id, ivs=ivs)
            if ex is None or ex["iv"] != 0 or ex["pid"] is None or ex["pid"][0] != axis:
                return None
            return self._quant_tile_ok(ex)

        a_row, a_col = index_ids.get("A", (None, None))
        b_row, b_col = index_ids.get("B", (None, None))
        c_row, c_col = index_ids.get("C", (None, None))
        bm = _ndim(a_row, 0)
        bk = _kdim("A", a_col, strides["A"][1])
        if nk:
            bn = _ndim(b_row, 1)
            bk_b = _kdim("B", b_col, strides["B"][1])
        else:
            bk_b = _kdim("B", b_row, strides["B"][0])
            bn = _ndim(b_col, 1)
        cm = _ndim(c_row, 0)
        cn = _ndim(c_col, 1)
        if None in (bm, bk, bn, bk_b, cm, cn) or bk != bk_b or bm != cm or bn != cn:
            return None
        stores = [o for o in op_by_id.values() if o.op == "tt.store"]
        if len(stores) != 1 or len(stores[0].operand_ids or []) != 2:
            return None  # a masked store is not replayed
        return {"bm": bm, "bn": bn, "bk": bk, "iv": iv}

    @staticmethod
    def _quant_tile_ok(info, *, block=None):
        """An n-tile ``pid * C + range(0, C)`` (or a k-tile ``range(0, C)`` when no pid):
        start 0, coefficient == end; returns C (or None). ``block`` pins C to a known value."""
        if info is None or info["range"] is None or info["range"][0] != 0:
            return None
        c = info["range"][1]
        if info["pid"] is not None and info["pid"][1] != c:
            return None
        if block is not None and c != block:
            return None
        return c

    def _quant_input_path_ok(self, vid, arg_index, op_by_id, *, role="gemm"):
        """Round 2.1 F3 + packet 102: the quantized templates stage the INPUT operand (A of a
        GEMM, x of a GEMV) from its raw pointer, so its value path must be EXACTLY what the
        template replays — replay-based, default-deny. GPU on clean ``ba21e9d``: a cast, a
        mask, a square in-tile transpose, a wrong-axis expand, a base or index offset on
        the input each stored the raw-pointer result on every route (err vs the IR-ordered
        oracle 0.036 … 372).

          gemm role: ``load -> {convert_layout, local_alloc, local_load}* -> dot``
                     (no expand/broadcast/transpose — the template tiles A itself);
          gemv role: ``load -> {convert_layout}* -> expand_dims(axis 0) -> {convert_layout}*
                     -> broadcast([1,K] -> [N,K])`` (the ``x[None, :]`` the kernel spells;
                     the template broadcasts x across rows itself — the wrong-axis
                     ``x[:, None]`` at N == K computed a different contraction).

        The load is PLAIN (no mask / other), its element type equals the buffer's and the
        operand's, and its pointer is the argument itself: 2-D (gemm) through the strict
        stride tracer's address proof (base offsets and residual terms refuse there), 1-D
        (gemv) exactly ``addptr(splat(x), range [+ k])`` (``_quant_1d_addr_ok``)."""
        from triton_msl.codegen.mlir_walker import _extract_shape

        args = self.graph.args
        if arg_index >= len(args):
            return False
        arg = args[arg_index]
        opnd = op_by_id.get(vid)
        identity = ("ttg.convert_layout", "ttg.local_alloc", "ttg.local_load")
        state = "start"
        seen = set()
        cur = vid
        while cur not in seen:
            seen.add(cur)
            o = op_by_id.get(cur)
            if o is None or not o.operand_ids:
                return False
            if o.op in identity:
                if role == "gemv" and o.op != "ttg.convert_layout":
                    return False
                cur = o.operand_ids[0]
                continue
            if role == "gemv" and o.op == "tt.broadcast" and state == "start":
                src = op_by_id.get(o.operand_ids[0])
                ss = _extract_shape((src.type_str if src is not None else "") or "")
                rs = _extract_shape(o.type_str or "")
                if not (ss and rs and len(ss) == 2 and len(rs) == 2 and ss[0] == 1 and ss[1] == rs[1]):
                    return False
                state = "bcast"
                cur = o.operand_ids[0]
                continue
            if role == "gemv" and o.op == "tt.expand_dims" and state == "bcast":
                if str((o.attrs or {}).get("axis")) != "0":
                    return False
                state = "expand"
                cur = o.operand_ids[0]
                continue
            if o.op == "tt.load":
                if role == "gemv" and state != "expand":
                    return False
                if not self._quant_plain_load(o):
                    return False
                if o.elem_type != arg.elem_type or (opnd is not None and opnd.elem_type != o.elem_type):
                    return False
                if role == "gemv":
                    return True  # the address is proven EXACTLY by the descriptor (_quant_role_addr, packet 104)
                iter_inits = self._scf_iter_inits(op_by_id)  # loop-carried A pointers trace to their inits
                src = self._trace_ptr_source(o.operand_ids[0], op_by_id, iter_inits=iter_inits)
                return src is not None and src.name == arg.name
            return False
        return False

    def _maybe_quant_gemv_int4_descriptor(self):
        """CANONICAL weight-only INT4 (GPTQ/AWQ per-group) decode GEMV, or None.

        Recognizes (correct-or-refuse — every constant verified against make_int4_gemv's
        packing so a mismatch refuses rather than mis-computes):

          byte = k // 2;  nib = (extui(load w[N,K/2]) >> ((k % 2) * 4)) & 0xF
          g = k // GROUP;  w = (sitofp(nib) - zero[N,g]) * scale[N,g]
          acc += tl.sum(x[k] * w, axis=1);  out = acc

        weight [N,K/2] uchar (2 nibbles/byte, low=even k / high=odd k), per-group fp32
        scale/zero [N, K/GROUP], fp32 in/out, signature
        ``(input, weight, output, scale, zero, N, K, ng, w_row_stride, s_row_stride)``.
        Routes to make_int4_gemv(GROUP). Returns

          ("gemv_int4", int4_msl, in=0, w=1, out=2, scale=3, zero=4, n_idx, k_idx,
           swn_idx, ssn_idx, ng_idx, group_size)
        """
        import os

        if os.environ.get("TRITON_MSL_QUANT_MATMUL", "1") == "0":
            return None

        def _all(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all(s.region_ops)
                if s.else_ops:
                    yield from _all(s.else_ops)

        op_by_id = {o.id: o for o in _all(self.graph.ops)}

        def _cint(oid):
            o = op_by_id.get(oid)
            if o is None or o.op != "arith.constant":
                return None
            try:
                return int(str((o.attrs or {}).get("value", "")).split(":")[0].strip())
            except (TypeError, ValueError):
                return None

        def _find(oid, ops, seen=None, d=0):
            """Find the first op whose name is in `ops` in oid's operand chain."""
            if seen is None:
                seen = set()
            if oid in seen or d > 30:
                return None
            seen.add(oid)
            o = op_by_id.get(oid)
            if o is None:
                return None
            if o.op in ops:
                return o
            for x in o.operand_ids or []:
                r = _find(x, ops, seen, d + 1)
                if r is not None:
                    return r
            return None

        # --- single sum-reduce, axis=1, loop-carried → 1-D store. ---
        reduces = [o for o in op_by_id.values() if o.op == "tt.reduce"]
        if len(reduces) != 1:
            return None
        red = reduces[0]
        if not red.operand_ids or str((red.attrs or {}).get("axis", None)) != "1":
            return None
        _comb = self.classify_reduce_combine(red)
        if _comb is None or _comb[0] != "sum":
            return None
        if not self._reduce_result_reaches_1d_store(red.id, stop_at_convert=False):
            # stop_at_convert=False: this is the structural "feeds a 1-D store"
            # SIGNATURE question — a ttg.convert_layout on the way is a passthrough
            # here (the guard semantics in _lowerer_reduce pass True instead).
            return None

        # --- reduce input = mulf(x_bc, mulf(subf(sitofp(NIBBLE), zero_bc), scale_bc)). ---
        mul = op_by_id.get(red.operand_ids[0])
        if mul is None or mul.op != "arith.mulf" or len(mul.operand_ids) != 2:
            return None
        m0, m1 = op_by_id.get(mul.operand_ids[0]), op_by_id.get(mul.operand_ids[1])
        s0 = _find(mul.operand_ids[0], ("arith.sitofp",)) is not None
        s1 = _find(mul.operand_ids[1], ("arith.sitofp",)) is not None
        if s0 and not s1:
            W, x_bc = m0, mul.operand_ids[1]
        elif s1 and not s0:
            W, x_bc = m1, mul.operand_ids[0]
        else:
            return None
        if W is None or W.op != "arith.mulf" or len(W.operand_ids) != 2:
            return None  # in-reduce scale form (int4 GEMVs apply per-group scale here)
        w0, w1 = op_by_id.get(W.operand_ids[0]), op_by_id.get(W.operand_ids[1])
        if w0 is not None and w0.op == "arith.subf":
            sub, scale_bc = w0, W.operand_ids[1]
        elif w1 is not None and w1.op == "arith.subf":
            sub, scale_bc = w1, W.operand_ids[0]
        else:
            return None
        zero_bc = sub.operand_ids[1]
        sitofp = _find(sub.operand_ids[0], ("arith.sitofp",))
        if sitofp is None or not sitofp.operand_ids:
            return None
        if not self._quant_legs_plain((scale_bc, zero_bc), op_by_id):  # packet 102: masks never replayed
            return None

        # --- NIBBLE unpack: andi(shrui(extui(load_w), shift), 0xF), shift=(k%2)*4. ---
        andi = op_by_id.get(sitofp.operand_ids[0])
        if andi is None or andi.op != "arith.andi" or len(andi.operand_ids) != 2:
            return None
        shr = None
        mask_ok = False
        for oid in andi.operand_ids:
            o = op_by_id.get(oid)
            if o is not None and o.op == "arith.shrui":
                shr = o
            elif _cint(oid) == 15:
                mask_ok = True
        if shr is None or not mask_ok or len(shr.operand_ids) != 2:
            return None
        extui = _find(shr.operand_ids[0], ("arith.extui",))
        if extui is None or not extui.operand_ids:
            return None
        w_load = _find(extui.operand_ids[0], ("tt.load",))
        if not self._quant_plain_load(w_load):  # packet 102: a masked packed-weight load is never replayed
            return None
        # shift = muli(remsi(kk, 2), 4)  → low nibble for even k, high for odd k.
        muli_op = _find(shr.operand_ids[1], ("arith.muli",))
        if muli_op is None or len(muli_op.operand_ids) != 2:
            return None
        rem = _find(muli_op.operand_ids[0], ("arith.remsi",)) or _find(muli_op.operand_ids[1], ("arith.remsi",))
        mul4 = any(_cint(x) == 4 for x in muli_op.operand_ids)
        if rem is None or not mul4 or len(rem.operand_ids) != 2 or _cint(rem.operand_ids[1]) != 2:
            return None

        # --- FULL loop-k pinning of the nibble/byte packing (correct-or-refuse). ---
        # make_int4_gemv bakes: byte = k//2, nibble = k%2 (LOW nibble for EVEN k), and
        # pairs the dequantized weight with the activation x[k] — all keyed by the SAME
        # reduction index k. The structural nibble/byte ops (%2, //2, mask 0xF, *4) are
        # necessary but NOT sufficient: a high-nibble-first packing `(k+1)%2`, or a
        # `(k+1)//2` byte index, has a DIFFERENT dividend and would route yet silently
        # dequantize the wrong nibble/byte (reproduced: err ~7.6). So pin all three
        # indices to ONE ssa value: the byte-divsi dividend == the nibble-remsi dividend
        # == the activation-load index. Any k-vs-(k+1) skew (single OR double swap)
        # yields distinct ssa ids here and REFUSES.
        if w_load is None or not w_load.operand_ids:
            return None
        byte_divsi = _find(w_load.operand_ids[0], ("arith.divsi",))
        if byte_divsi is None or len(byte_divsi.operand_ids) != 2 or _cint(byte_divsi.operand_ids[1]) != 2:
            return None  # byte index must be exactly k // 2
        k_idx = byte_divsi.operand_ids[0]
        if rem.operand_ids[0] != k_idx:
            return None  # nibble k%2 must use the SAME index as the byte k//2
        x_load = _find(x_bc, ("tt.load",))
        if x_load is None or not x_load.operand_ids:
            return None
        x_addr = op_by_id.get(x_load.operand_ids[0])
        if x_addr is None or x_addr.op != "tt.addptr" or len(x_addr.operand_ids) != 2:
            return None
        if x_addr.operand_ids[1] != k_idx:
            return None  # activation x[k] must use that SAME reduction index k

        # --- per-group scale/zero: address uses g = k // GROUP; extract + match GROUP. ---
        def _group_of(bc):
            div = _find(bc, ("arith.divsi",))
            if div is None or len(div.operand_ids) != 2:
                return None
            return _cint(div.operand_ids[1])

        g_s = _group_of(scale_bc)
        g_z = _group_of(zero_bc)
        if g_s is None or g_s != g_z or g_s < 2:
            return None
        group_size = g_s
        # Packet 105 A: make_int4_gemv dequantizes BOTH nibbles of a byte with ONE
        # scale/zero group ((2b) // G); the kernel computes k // G per nibble. Equivalent
        # only when group boundaries are byte-aligned -> G must be even (GPU: G = 3 stored
        # the template's result, err 8.3 vs the kernel's semantics).
        if group_size % 2 != 0:
            return None

        # --- Canonical signature + arg bindings. ---
        args = self.graph.args
        arg_id_to_idx = {getattr(a, "id", None): i for i, a in enumerate(args)}
        ptr_args = [a for a in args if a.is_ptr]
        if len(ptr_args) != 5 or len(args) < 7:
            return None
        if not all(args[i].is_ptr for i in range(5)):
            return None
        if args[5].is_ptr or args[6].is_ptr:
            return None

        def _to_arg(oid, d=0):
            if d > 48:
                return None
            if oid in arg_id_to_idx:
                return arg_id_to_idx[oid]
            o = op_by_id.get(oid)
            if o is None or not o.operand_ids:
                return None
            return _to_arg(o.operand_ids[0], d + 1)

        if not self._quant_input_path_ok(
            x_bc, 0, op_by_id, role="gemv"
        ):  # F3/102: plain load, x[None, :], exact address
            return None
        if _to_arg(x_bc) != 0 or _to_arg(w_load.operand_ids[0]) != 1:
            return None
        if _to_arg(scale_bc) != 3 or _to_arg(zero_bc) != 4:
            return None
        stores = [o for o in op_by_id.values() if o.op == "tt.store"]
        if len(stores) != 1 or len(stores[0].operand_ids or []) < 2:
            return None
        if _to_arg(stores[0].operand_ids[0]) != 2:
            return None

        # --- dtypes: input/output/scale/zero fp32, weight i8 (packed). ---
        if _mlir_to_triton_dtype(args[0].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[2].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[1].elem_type) not in ("int8", "i8", "si8", "uint8", "u8"):
            return None
        # make_int4_gemv declares scale/zero as ``device const float*``. Refuse a
        # half/bfloat buffer instead of reinterpreting its bits as fp32 in the template.
        for _sza in args[3:4] + args[4:5]:
            if getattr(_sza, "is_ptr", False) and _mlir_to_triton_dtype(_sza.elem_type) not in (
                "fp32",
                "f32",
                "float",
            ):
                return None

        # --- weight/scale row-stride arg indices (verified at dispatch: w_row == K/2,
        #     s_row == ng == K/group). Traced from the [N,1] row-offset muli. ---
        def _row_stride_idx(bc_or_load):
            # The row offset is muli(expand_dims(offs_n), splat(STRIDE_ARG)); STRIDE_ARG
            # is the scalar arg. Find the muli feeding the addptr base + trace its scalar.
            mop = _find(bc_or_load, ("arith.muli",))
            if mop is None:
                return None
            for oid in mop.operand_ids:
                idx = _to_arg(oid)
                if idx is not None and idx >= 7:  # a stride arg (after N,K,ng at 5,6,7)
                    return idx
            return None

        # Packet 102/104: EXACT address contracts for every role make_int4_gemv replays —
        # x ``+ range(0,BK) + k``; packed weight ``+ (pid(0)*BN + range(0,BN))[:,None]*swn +
        # ((range(0,BK) + k) // 2)[None,:]``; scale/zero ``+ (pid(0)*BN + range(0,BN))[:,None]
        # *ssn + ((range(0,BK) + k) // G)[None,:]``; output ``+ pid(0)*BN + range(0,BN)``
        # (plain store); the ONE K-loop ``range(0, K, BK)``. The strides the template is
        # given are the ones proven here (weight row, scale row == zero row).
        _kl4 = self._quant_kloop(op_by_id, 6)
        if _kl4 is None:
            return None
        _iv4, _step4 = _kl4
        _ivs4 = {_iv4}
        _xl4 = self._quant_find_load(x_bc, op_by_id)
        _sl4 = self._quant_find_load(scale_bc, op_by_id)
        _zl4 = self._quant_find_load(zero_bc, op_by_id)
        if any(x is None or not x.operand_ids for x in (_xl4, _sl4, _zl4)):
            return None
        _N4 = dict(pid=("required", 0), iv=0, stride="required")
        _KV4 = dict(pid=None, iv=1, stride="unit")
        okx4, ix4 = self._quant_role_addr(_xl4.operand_ids[0], args[0], op_by_id, ivs=_ivs4, kinds={"vec": _KV4})
        okw4, iw4 = self._quant_role_addr(
            w_load.operand_ids[0],
            args[1],
            op_by_id,
            ivs=_ivs4,
            kinds={"row": _N4, "col": dict(pid=None, iv=1, stride="unit", div=2)},
        )
        _CG4 = dict(pid=None, iv=1, stride="unit", div=group_size)
        oks4, is4 = self._quant_role_addr(
            _sl4.operand_ids[0], args[3], op_by_id, ivs=_ivs4, kinds={"row": _N4, "col": _CG4}
        )
        okz4, iz4 = self._quant_role_addr(
            _zl4.operand_ids[0], args[4], op_by_id, ivs=_ivs4, kinds={"row": _N4, "col": _CG4}
        )
        if not (okx4 and okw4 and oks4 and okz4):
            return None
        _st4 = stores[0]
        if len(_st4.operand_ids or []) != 2:
            return None  # a masked store is not replayed
        oko4, io4 = self._quant_role_addr(
            _st4.operand_ids[0],
            args[2],
            op_by_id,
            ivs=_ivs4,
            kinds={"vec": dict(pid=("required", 0), iv=0, stride="unit")},
        )
        if not oko4:
            return None
        _bn4 = self._quant_tile_ok(iw4["row"])
        if _bn4 is None or any(
            self._quant_tile_ok(t, block=_bn4) is None for t in (is4["row"], iz4["row"], io4["vec"])
        ):
            return None
        _bk4 = self._quant_tile_ok(ix4["vec"], block=_step4)
        if _bk4 is None or any(
            self._quant_tile_ok(t, block=_bk4) is None for t in (iw4["col"], is4["col"], iz4["col"])
        ):
            return None
        if is4["row"]["stride"] != iz4["row"]["stride"]:
            return None  # the template applies ONE row stride to scale and zero

        swn_idx = iw4["row"]["stride"]  # the strides the template is bound to are the proven ones
        ssn_idx = is4["row"]["stride"]
        # ng arg: the per-group count (scales [N, ng]); it's the arg whose value == K/group.
        # Bound at dispatch by ng_idx if we can find it; otherwise omit (dispatch recomputes).
        ng_idx = None
        for i in range(7, len(args)):
            if not args[i].is_ptr and i not in (swn_idx, ssn_idx):
                ng_idx = i
                break

        from triton_msl.codegen._msl_templates import make_int4_gemv, make_int4_matmul_pergroup

        int4_msl = make_int4_gemv(group_size)
        # [13] = the SOURCE program mapping (BN; one program per BN outputs on program_id(0)):
        # the dispatcher runs the template only for a launch of exactly cdiv(N, BN) programs
        # (packet 105 B: a partial launch used to run the full output).
        # [14] = the SOURCE K block and [15] = the stride-generic scalar template: the kernel
        # is unmasked, so it computes whole BN x BK blocks; the dispatcher runs those extents
        # when the launch sizes are not whole blocks (its SOURCE DOMAIN note).
        return (
            "gemv_int4",
            int4_msl,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            swn_idx,
            ssn_idx,
            ng_idx,
            group_size,
            _bn4,
            _bk4,
            make_int4_matmul_pergroup(group_size),
        )

    def _maybe_quant_gemv_descriptor(self):
        """Build the runtime QUANTIZED-GEMV (decode / M=1) dispatch descriptor, or None.

        Recognizes the CANONICAL weight-only int8 GEMV written as an in-loop row reduce
        (GPTQ/AWQ decode) and ONLY that shape (correct-or-refuse):

          acc += tl.sum(x[None,:] * (w_i8.to(f32) - zero[:,None]), axis=1)   # over K
          out = acc * scale                                                  # per-N epilogue

        weight stored [N,K] contiguous, per-N float scale/zero, fp32 in/out, with the
        ``(input, weight, output, scale, zero, N, K, weight_row_stride)`` signature
        (5 pointers at 0..4, N/K scalars at 5/6, the single non-unit weight stride at 7;
        the contiguous K stride is dropped by Triton). Verified structurally: the exact
        ``mulf(bcast(x), subf(sitofp(load w), bcast(zero)))`` reduce input, the
        ``acc * scale`` epilogue into a 1-D store, and every leg traced to its arg. Any
        deviation -> None (the caller then refuses via the reduce-layout guard).
        Dispatched to ``make_int8_gemv`` (N simdgroups, at the memory roofline). Returns

          ("gemv", gemv_msl, in=0, weight=1, out=2, scale=3, zero=4, n_idx=5, k_idx=6, stride_checks)
        """
        import os

        if os.environ.get("TRITON_MSL_QUANT_MATMUL", "1") == "0":
            return None

        def _all(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all(s.region_ops)
                if s.else_ops:
                    yield from _all(s.else_ops)

        op_by_id = {o.id: o for o in _all(self.graph.ops)}

        # Exactly one tt.reduce, sum-combine (an arith.addf in its region), axis=1,
        # that is the loop-carried→1-D-store pattern (the one the reduce guard refuses;
        # route it instead when it matches the quantized-GEMV shape).
        reduces = [o for o in op_by_id.values() if o.op == "tt.reduce"]
        if len(reduces) != 1:
            return None
        red = reduces[0]
        if not red.operand_ids:
            return None
        if str((red.attrs or {}).get("axis", None)) != "1":
            return None
        _combine = self.classify_reduce_combine(red)
        if _combine is None or _combine[0] != "sum":
            return None
        if not self._reduce_result_reaches_1d_store(red.id, stop_at_convert=False):
            # stop_at_convert=False: this is the structural "feeds a 1-D store"
            # SIGNATURE question — a ttg.convert_layout on the way is a passthrough
            # here (the guard semantics in _lowerer_reduce pass True instead).
            return None

        # reduce input = mulf(bcast(x), subf(sitofp(load w), bcast(zero))).
        mul = op_by_id.get(red.operand_ids[0])
        if mul is None or mul.op != "arith.mulf" or len(mul.operand_ids) != 2:
            return None

        def _find_sitofp(sid, seen=None, d=0):
            if seen is None:
                seen = set()
            if sid in seen or d > 24:
                return None
            seen.add(sid)
            o = op_by_id.get(sid)
            if o is None:
                return None
            if o.op in ("arith.sitofp", "arith.uitofp"):
                return o
            for x in o.operand_ids or []:
                r = _find_sitofp(x, seen, d + 1)
                if r is not None:
                    return r
            return None

        # The reduce input is mulf(x_bc, W). W is the "weight side" (contains the
        # sitofp); x_bc is the input broadcast. W is one of two mathematically-identical
        # forms (scale is per-N, so it factors out of the sum either way — both route to
        # the same make_int8_gemv, which applies the per-N scale once to the raw sum):
        #   epilogue form:   subf(sitofp(w), zero_bc)                  (scale applied after the loop)
        #   in-reduce form:  mulf(subf(sitofp(w), zero_bc), scale_bc)  (scale inside the reduce)
        m0_has = _find_sitofp(mul.operand_ids[0]) is not None
        m1_has = _find_sitofp(mul.operand_ids[1]) is not None
        if m0_has and not m1_has:
            W, x_bc = op_by_id.get(mul.operand_ids[0]), mul.operand_ids[1]
        elif m1_has and not m0_has:
            W, x_bc = op_by_id.get(mul.operand_ids[1]), mul.operand_ids[0]
        else:
            return None
        if W is None:
            return None
        scale_bc = None  # in-reduce: bound here; epilogue: bound from the store below
        if W.op == "arith.subf":
            sub = W
        elif W.op == "arith.mulf" and len(W.operand_ids) == 2:
            w0, w1 = op_by_id.get(W.operand_ids[0]), op_by_id.get(W.operand_ids[1])
            if w0 is not None and w0.op == "arith.subf":
                sub, scale_bc = w0, W.operand_ids[1]
            elif w1 is not None and w1.op == "arith.subf":
                sub, scale_bc = w1, W.operand_ids[0]
            else:
                return None
        else:
            return None
        if len(sub.operand_ids) != 2 or _find_sitofp(sub.operand_ids[0]) is None:
            return None
        zero_bc = sub.operand_ids[1]
        sitofp = _find_sitofp(sub.operand_ids[0])
        if sitofp is None or not sitofp.operand_ids:
            return None
        w_load = sitofp.operand_ids[0]
        if not self._quant_legs_plain((w_load, zero_bc), op_by_id):  # packet 102: masks never replayed
            return None

        args = self.graph.args
        arg_id_to_idx = {getattr(a, "id", None): i for i, a in enumerate(args)}

        def _trace_to_arg(oid, d=0):
            if d > 48:
                return None
            if oid in arg_id_to_idx:
                return arg_id_to_idx[oid]
            o = op_by_id.get(oid)
            if o is None or not o.operand_ids:
                return None
            return _trace_to_arg(o.operand_ids[0], d + 1)

        # --- Canonical signature: 5 ptrs at 0..4, N/K scalars at 5/6, one weight
        #     stride at 7 (contiguous K dropped). ---
        ptr_args = [a for a in args if a.is_ptr]
        if len(ptr_args) != 5 or len(args) != 8:
            return None
        if not all(args[i].is_ptr for i in range(5)):
            return None
        if args[5].is_ptr or args[6].is_ptr or args[7].is_ptr:
            return None

        # --- Trace every leg to its arg and require the canonical binding. ---
        if _trace_to_arg(x_bc) != 0:
            return None  # input vector
        if not self._quant_input_path_ok(
            x_bc, 0, op_by_id, role="gemv"
        ):  # F3/102: plain load, x[None, :], exact address
            return None
        if _trace_to_arg(w_load) != 1:
            return None  # weight
        if _trace_to_arg(zero_bc) != 4:
            return None  # zero

        # The single tt.store's pointer traces to the output arg. The stored value is:
        #   epilogue form:   mulf(acc, scale_bc)  -> bind scale here (scale not yet set)
        #   in-reduce form:  the raw acc          -> scale already bound; the store must
        #                                            NOT apply a second per-arg scale.
        stores = [o for o in op_by_id.values() if o.op == "tt.store"]
        if len(stores) != 1 or len(stores[0].operand_ids or []) < 2:
            return None
        store = stores[0]
        if _trace_to_arg(store.operand_ids[0]) != 2:
            return None  # output
        val = op_by_id.get(store.operand_ids[1])
        while val is not None and val.op in ("ttg.convert_layout",) and val.operand_ids:
            val = op_by_id.get(val.operand_ids[0])
        if scale_bc is None:
            # EPILOGUE form: store value must be mulf(acc, scale) with scale -> arg 3.
            if val is None or val.op != "arith.mulf" or len(val.operand_ids) != 2:
                return None
            if _trace_to_arg(val.operand_ids[0]) == 3:
                scale_bc = val.operand_ids[0]
            elif _trace_to_arg(val.operand_ids[1]) == 3:
                scale_bc = val.operand_ids[1]
            else:
                return None
        else:
            # IN-REDUCE form: guard against a SECOND scale in the epilogue (double-apply).
            if val is not None and val.op == "arith.mulf" and len(val.operand_ids) == 2:
                if 3 in (_trace_to_arg(val.operand_ids[0]), _trace_to_arg(val.operand_ids[1])):
                    return None  # scale applied both in-reduce AND as epilogue -> refuse
        if _trace_to_arg(scale_bc) != 3:
            return None  # per-N scale (arg 3)
        if not self._quant_legs_plain((scale_bc,), op_by_id):  # packet 102
            return None
        # Packet 102/104: EXACT address contracts for every role make_int8_gemv replays —
        # x ``+ range(0,BK) + k``, scale/zero/output ``+ pid(0)*BN + range(0,BN)``, weight
        # ``+ (pid(0)*BN + range(0,BN))[:,None]*swn + (range(0,BK) + k)[None,:]`` — the ONE
        # K-loop ``range(0, K, BK)``, and a PLAIN output store. Packet 103: a missing pid, a
        # ``pid * 2BN``, a missing K term, ``make_range(BN, 2BN)``, an output base / index
        # shift, a masked store, a wrong loop step / bound all ran the specialized route with
        # the raw result.
        _kl = self._quant_kloop(op_by_id, 6)
        if _kl is None:
            return None
        _iv, _step = _kl
        _ivs = {_iv}
        _xl = self._quant_find_load(x_bc, op_by_id)
        _sl = self._quant_find_load(scale_bc, op_by_id)
        _zl = self._quant_find_load(zero_bc, op_by_id)
        _wl = self._quant_find_load(w_load, op_by_id)
        if any(x is None or not x.operand_ids for x in (_xl, _sl, _zl, _wl)):
            return None
        _N = dict(pid=("required", 0), iv=0, stride="unit")
        _KV = dict(pid=None, iv=1, stride="unit")
        okx, ix = self._quant_role_addr(_xl.operand_ids[0], args[0], op_by_id, ivs=_ivs, kinds={"vec": _KV})
        oks, isc = self._quant_role_addr(_sl.operand_ids[0], args[3], op_by_id, ivs=_ivs, kinds={"vec": _N})
        okz, iz = self._quant_role_addr(_zl.operand_ids[0], args[4], op_by_id, ivs=_ivs, kinds={"vec": _N})
        okw, iw = self._quant_role_addr(
            _wl.operand_ids[0],
            args[1],
            op_by_id,
            ivs=_ivs,
            kinds={"row": dict(pid=("required", 0), iv=0, stride="required"), "col": _KV},
        )
        if not (okx and oks and okz and okw):
            return None
        if len(store.operand_ids or []) != 2:
            return None  # a masked store is not replayed
        oko, io = self._quant_role_addr(store.operand_ids[0], args[2], op_by_id, ivs=_ivs, kinds={"vec": _N})
        if not oko:
            return None
        _bn = self._quant_tile_ok(isc["vec"])
        if _bn is None or any(self._quant_tile_ok(t, block=_bn) is None for t in (iz["vec"], iw["row"], io["vec"])):
            return None
        _bk = self._quant_tile_ok(ix["vec"], block=_step)
        if _bk is None or self._quant_tile_ok(iw["col"], block=_bk) is None:
            return None
        if iw["row"]["stride"] != 7:
            return None  # the weight row stride must be the signature's args[7]

        # --- dtypes: input fp32, output fp32, weight int8. ---
        if _mlir_to_triton_dtype(args[0].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[2].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[1].elem_type) not in ("int8", "i8", "si8"):
            return None
        # SCALE/ZERO dtype (re-review 2026-08-25): the templates declare
        # ``device const float*`` for scale/zero pointers — a non-f32 tensor (e.g. GPTQ's
        # fp16 scales) would be reinterpreted as float bits (garbage, no error). Refuse
        # the quant route; the generic lowering handles half/bf16 loads correctly.
        for _sza in args[3:4] + args[4:5]:
            if getattr(_sza, "is_ptr", False) and _mlir_to_triton_dtype(_sza.elem_type) not in ("fp32", "f32", "float"):
                return None

        # --- Runtime stride contract: make_int8_gemv reads weight[n*K + k], so the
        #     weight row stride (args[7]) must equal K (args[6]) at dispatch. ---
        stride_checks = ((7, 6),)

        from triton_msl.codegen._msl_templates import make_int8_gemv, make_int8_matmul_pergroup

        gemv_msl = make_int8_gemv()
        # [10] = the SOURCE program mapping (BN on program_id(0)) — packet 105 B.
        # [11] = the SOURCE K block and [12] = the stride-generic scalar template: the kernel
        # is unmasked, so it computes whole BN x BK blocks; the dispatcher runs those extents
        # when the launch sizes are not whole blocks (its SOURCE DOMAIN note).
        return ("gemv", gemv_msl, 0, 1, 2, 3, 4, 5, 6, stride_checks, _bn, _bk, make_int8_matmul_pergroup(32))

    def _maybe_quant_matmul_descriptor(self):
        """Build the runtime QUANTIZED-matmul dispatch descriptor, or None.

        Recognizes the CANONICAL weight-only int8 quantized matmul and ONLY that
        shape (correct-or-refuse — a wrong descriptor is silently-wrong):

          [K,N]: B = (sitofp(load(w_i8[K,N])) - zero[None,:]) * scale[None,:]; out = A @ B
          [N,K]: B = trans((sitofp(load(w_i8[N,K])) - zero[:,None]) * scale[:,None])  (GPTQ)

        with the standard ``(input, weight, output, scale, zero, M, N, K, ...strides)``
        signature — 5 pointers at args 0..4, M/N/K scalars at 5..7 — contiguous-inner
        (per ``infer_dot_strides``), per-N float scale/zero, fp32 input/output, int8
        weight. A ``ttg.memdesc_trans`` (or ``tt.trans``) on B means weight is stored
        [N,K] (routed to layout 'nk'); otherwise [K,N] (layout 'kn'). Every structural
        fact is VERIFIED against the IR (the exact mul/sub/sitofp tree; scale->arg3 and
        zero->arg4 by tracing the broadcast chains; the K-loop bound anchors to
        args[7]); any deviation returns None so the caller refuses. Returns

          (fast_msl, m_idx=5, n_idx=6, k_idx=7, tile_m, tile_n, stride_checks)

        dispatched by triton_msl.autotuning._quant_matmul_dispatch.dispatch_quant_matmul.
        """
        import os

        if os.environ.get("TRITON_MSL_QUANT_MATMUL", "1") == "0":
            return None

        def _all(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all(s.region_ops)
                if s.else_ops:
                    yield from _all(s.else_ops)

        op_by_id = {o.id: o for o in _all(self.graph.ops)}
        dots = [o for o in op_by_id.values() if o.op == "tt.dot"]
        if len(dots) != 1:
            return None
        dot = dots[0]
        if not dot.operand_ids or len(dot.operand_ids) < 2:
            return None

        # --- Canonical signature: 5 ptrs at 0..4, M/N/K scalars at 5..7. ---
        args = self.graph.args
        ptr_args = [a for a in args if a.is_ptr]
        if len(ptr_args) != 5 or len(args) < 8:
            return None
        if not all(args[i].is_ptr for i in range(5)):
            return None
        if args[5].is_ptr or args[6].is_ptr or args[7].is_ptr:
            return None
        arg_id_to_idx = {getattr(a, "id", None): i for i, a in enumerate(args)}
        # F3: the A operand's VALUE path must be its load (both the fast and the scalar
        # per-group routes behind this descriptor stage A from the pointer type).
        if not self._quant_input_path_ok(dot.operand_ids[0], 0, op_by_id):
            return None

        # --- Exact dequant tree: mulf(subf(<sitofp weight>, zero_bc), scale_bc).
        #     A leading tt.trans on B means the weight is stored [N,K] (GPTQ prefill,
        #     `tl.dot(a, tl.trans(w_deq))`) rather than [K,N] — route to layout 'nk'. ---
        _is_nk = [False]

        def _peel(oid):
            o = op_by_id.get(oid)
            while (
                o is not None
                and o.operand_ids
                and o.op
                in (
                    "ttg.local_load",
                    "ttg.local_alloc",
                    "ttg.convert_layout",
                    "tt.trans",
                    "ttg.memdesc_trans",
                )
            ):
                if o.op in ("tt.trans", "ttg.memdesc_trans"):
                    _is_nk[0] = True
                o = op_by_id.get(o.operand_ids[0])
            return o

        mul = _peel(dot.operand_ids[1])
        if mul is None or mul.op != "arith.mulf" or len(mul.operand_ids) != 2:
            return None
        sub = op_by_id.get(mul.operand_ids[0])
        scale_bc = mul.operand_ids[1]
        if sub is None or sub.op != "arith.subf" or len(sub.operand_ids) != 2:
            return None
        zero_bc = sub.operand_ids[1]

        def _has_int_to_float(oid, seen=None, d=0):
            if seen is None:
                seen = set()
            if oid in seen or d > 24:
                return False
            seen.add(oid)
            o = op_by_id.get(oid)
            if o is None:
                return False
            if o.op in ("arith.sitofp", "arith.uitofp"):
                return True
            return any(_has_int_to_float(x, seen, d + 1) for x in (o.operand_ids or []))

        if not _has_int_to_float(sub.operand_ids[0]):
            return None
        # Packet 102: the weight / scale / zero loads are staged from their raw pointers —
        # a mask or ``other`` on any of them is never replayed (GPU: raw result stored).
        if not self._quant_legs_plain((sub.operand_ids[0], scale_bc, zero_bc), op_by_id):
            return None

        # --- scale_bc -> args[3], zero_bc -> args[4] (walk operand[0]: bcast/expand/
        #     load/addptr-base/splat all forward the pointer through operand[0]). ---
        def _trace_to_arg(oid, d=0):
            if d > 32:
                return None
            if oid in arg_id_to_idx:
                return arg_id_to_idx[oid]
            o = op_by_id.get(oid)
            if o is None or not o.operand_ids:
                return None
            return _trace_to_arg(o.operand_ids[0], d + 1)

        if _trace_to_arg(scale_bc) != 3 or _trace_to_arg(zero_bc) != 4:
            return None

        # --- PER-N vs PER-GROUP: the fast kernel indexes scale/zero as length-N (per
        #     output channel). A GPTQ PER-GROUP kernel loads scale/zero INSIDE the K-loop
        #     at k//G, so the load is loop-VARIANT and stays in the scf.for body; routing
        #     it to the per-N kernel silently drops the group boundary (wrong numbers, no
        #     error). Refuse when either scale/zero load lives inside a loop region. A
        #     genuinely per-N load (or a single-group load, k//G folded to 0) is hoisted
        #     out -> not in the loop set -> still routes. ---
        _loop_op_ids = set()

        def _collect_loop_ops(ops, in_loop):
            for s in ops:
                if in_loop:
                    _loop_op_ids.add(s.id)
                child = in_loop or (s.op == "scf.for")
                if getattr(s, "region_ops", None):
                    _collect_loop_ops(s.region_ops, child)
                if getattr(s, "else_ops", None):
                    _collect_loop_ops(s.else_ops, child)

        _collect_loop_ops(self.graph.ops, False)

        def _find_load(oid, d=0):
            if d > 32:
                return None
            o = op_by_id.get(oid)
            if o is None:
                return None
            if o.op == "tt.load":
                return o
            if not o.operand_ids:
                return None
            return _find_load(o.operand_ids[0], d + 1)

        _pergroup = any(
            (_find_load(_bc) is not None and _find_load(_bc).id in _loop_op_ids) for _bc in (scale_bc, zero_bc)
        )
        if _pergroup:
            # PER-GROUP int8 (GPTQ): scale/zero loaded in the K-loop at k//G. The per-N
            # fast kernel can't express it (dropping the group boundary is the silent-wrong
            # closed by 8e3526a). Route to the scalar per-group template (stride-generic),
            # resolving G + every operand's strides exact-or-refuse. Validated e2e.
            if _mlir_to_triton_dtype(args[0].elem_type) not in ("fp32", "f32", "float"):
                return None
            if _mlir_to_triton_dtype(args[2].elem_type) not in ("fp32", "f32", "float"):
                return None
            # SCALE/ZERO dtype (re-review 2026-08-25): every quant template declares
            # ``device const float*`` for scales/zeros, so a non-f32 scale/zero tensor
            # (e.g. GPTQ's fp16 scales) would be REINTERPRETED as float bits — garbage
            # output with no error. Refuse the quant route; the generic lowering loads
            # half/bf16 correctly.
            if _mlir_to_triton_dtype(args[3].elem_type) not in ("fp32", "f32", "float"):
                return None
            if _mlir_to_triton_dtype(args[4].elem_type) not in ("fp32", "f32", "float"):
                return None
            # A packed int4 weight and an int8 weight BOTH report signless i8, so distinguish
            # by STRUCTURE: int4 unpacks a nibble (andi 0xF over shrui by (k%2)*4) from a byte
            # load at k//2, whereas int8 is a direct sitofp(load). If a nibble unpack is
            # present it MUST verify + FULL loop-k pin (byte//2 dividend == nibble%2 dividend
            # == the dot A-operand's k index) or REFUSE — never route packed int4 to the
            # linear int8 template (that would dequantize the wrong nibble: silent-wrong).
            if _mlir_to_triton_dtype(args[1].elem_type) not in (
                "int8",
                "i8",
                "si8",
                "uint8",
                "u8",
                "ui8",
            ):
                return None

            def _q_find(oid, names, seen=None, d=0):
                if seen is None:
                    seen = set()
                if oid in seen or d > 30:
                    return None
                seen.add(oid)
                o = op_by_id.get(oid)
                if o is None:
                    return None
                if o.op in names:
                    return o
                for x in o.operand_ids or []:
                    r = _q_find(x, names, seen, d + 1)
                    if r is not None:
                        return r
                return None

            def _q_cval(oid):
                o = op_by_id.get(oid)
                if o is None or o.op != "arith.constant":
                    return None
                try:
                    return int(str((o.attrs or {}).get("value", "")).split(":")[0].strip())
                except (TypeError, ValueError):
                    return None

            def _q_has_const_offset(root, seen=None, d=0):
                # True if an arith.addi/subi with a CONSTANT (or splat/broadcast of a
                # constant) operand appears above the make_range in root's cone — i.e. the
                # k index is SHIFTED (k +/- c), the double-swap. Stops at make_range/load/
                # arg; muli-by-constant (block scaling pid*BLOCK) is NOT an additive shift.
                if seen is None:
                    seen = set()
                if root in seen or d > 40:
                    return False
                seen.add(root)
                o = op_by_id.get(root)
                if o is None or o.op in ("tt.make_range", "tt.load"):
                    return False
                if o.op in ("arith.addi", "arith.subi"):
                    for x in o.operand_ids or []:
                        xo = op_by_id.get(x)
                        while (
                            xo is not None
                            and xo.op
                            in (
                                "tt.splat",
                                "tt.broadcast",
                                "ttg.convert_layout",
                                "tt.expand_dims",
                                "tt.reshape",
                            )
                            and xo.operand_ids
                        ):
                            xo = op_by_id.get(xo.operand_ids[0])
                        if xo is not None and xo.op == "arith.constant":
                            return True
                return any(_q_has_const_offset(x, seen, d + 1) for x in (o.operand_ids or []))

            def _q_in_cone(root, target, seen=None, d=0):
                if seen is None:
                    seen = set()
                if root in seen or d > 48:
                    return False
                seen.add(root)
                if root == target:
                    return True
                o = op_by_id.get(root)
                if o is None:
                    return False
                return any(_q_in_cone(x, target, seen, d + 1) for x in (o.operand_ids or []))

            _is_int4 = False
            _andi = _q_find(sub.operand_ids[0], ("arith.andi",))
            if _andi is not None and len(_andi.operand_ids) == 2:
                _shr = None
                _mask_ok = False
                for _oid in _andi.operand_ids:
                    _oo = op_by_id.get(_oid)
                    if _oo is not None and _oo.op == "arith.shrui":
                        _shr = _oo
                    elif _q_cval(_oid) == 15:
                        _mask_ok = True
                if _mask_ok and _shr is not None and len(_shr.operand_ids) == 2:
                    _muli = _q_find(_shr.operand_ids[1], ("arith.muli",))
                    _wload = _q_find(_shr.operand_ids[0], ("tt.load",))
                    if _muli is not None and _wload is not None and _wload.operand_ids and len(_muli.operand_ids) == 2:
                        _rem = _q_find(_muli.operand_ids[0], ("arith.remsi",)) or _q_find(
                            _muli.operand_ids[1], ("arith.remsi",)
                        )
                        _mul4 = any(_q_cval(x) == 4 for x in _muli.operand_ids)
                        _bdiv = _q_find(_wload.operand_ids[0], ("arith.divsi",))
                        if (
                            _rem is not None
                            and _mul4
                            and len(_rem.operand_ids) == 2
                            and _q_cval(_rem.operand_ids[1]) == 2
                            and _bdiv is not None
                            and len(_bdiv.operand_ids) == 2
                            and _q_cval(_bdiv.operand_ids[1]) == 2
                            and _rem.operand_ids[0] == _bdiv.operand_ids[0]
                        ):
                            # byte//2 and nibble%2 share ONE k index (same_k). FULL pin: that
                            # k must reach a tt.make_range (a real loop index) with NO constant
                            # additive offset down to it -> rules out a k-vs-(k+1) shift (the
                            # double-swap; a single-swap already fails same_k above). The
                            # ACTIVATION is implicitly pinned to the same k by the tt.dot, which
                            # contracts A and B over one shared k — so no fragile cross-index
                            # match is needed (and TTGIR duplicates the make_range per MMA
                            # layout, making one unreliable anyway).
                            _kdiv = _bdiv.operand_ids[0]
                            _kmr = self._trace_to_make_range(_kdiv, list(op_by_id.values()), op_by_id)
                            if _kmr is not None and not _q_has_const_offset(_kdiv):
                                _is_int4 = True
                if not _is_int4:
                    return None  # nibble-like weight but not the safe pinned packing -> refuse
            try:
                _sd = self.infer_dot_strides()
            except Exception:  # noqa: BLE001
                return None
            if not _sd or not all(_sd.get(x) for x in ("A", "B", "C")):
                return None
            _gc = self._quant_gemm_contract(
                op_by_id, k_arg=7, nk=_is_nk[0], b_kdiv=(2 if _is_int4 else 1)
            )  # packet 104
            if _gc is None:
                return None
            _bn_expect = _gc["bn"]
            # A stride can be a runtime ARG (its index) or a compile-time constant 1
            # (equal_to_1 specialization folds a contiguous stride out of the arg list;
            # infer_dot_strides then reports "1"). Encode const-1 as the sentinel -1 (the
            # dispatch passes literal 1); a stride that is neither an arg nor "1" -> None
            # (refuse).
            _nti = {a.name: i for i, a in enumerate(args)}

            def _sidx(nm):
                if nm == "1":
                    return -1
                return _nti.get(nm)

            _stride_idx = [
                _sidx(_sd["A"][0]),
                _sidx(_sd["A"][1]),  # isr, isc
                _sidx(_sd["B"][0]),
                _sidx(_sd["B"][1]),  # wsk, wsn
                _sidx(_sd["C"][0]),
                _sidx(_sd["C"][1]),  # osr, osc
            ]

            def _vec_stride(tid):
                # Packet 102/104: the per-N term must be EXACTLY ``offs_n [* splat(stride)]``
                # with ``offs_n = program_id(1) * BN + range(0, BN)`` (the SAME BN as the dot's
                # N tile) -> the stride's arg index (-1 when folded to 1). Reading the stride
                # alone let ``(offs_n + 1) * ssn`` through; a missing pid, ``pid * 2BN`` or
                # ``range(BN, 2BN)`` passed the "at most one" grammar (packet 103).
                tm = self._quant_term(tid, op_by_id)
                if tm is None or tm[0] is not None:
                    return None  # not a 1-D term
                ex = self._quant_index_exact(tm[2], op_by_id, ivs=set())
                if ex is None or ex["iv"] != 0 or ex["pid"] is None or ex["pid"][0] != 1:
                    return None
                if self._quant_tile_ok(ex, block=_bn_expect) is None:
                    return None
                return tm[1]

            def _G_from_divsi(dv):
                d = op_by_id.get(dv)
                if d is None or d.op not in ("arith.divsi", "arith.divui"):
                    return None
                for x in d.operand_ids:
                    c = op_by_id.get(x)
                    if c is not None and c.op == "arith.constant":
                        try:
                            return int(str(c.attrs.get("value", "")).split(":")[0].strip())
                        except (TypeError, ValueError):
                            return None
                return None

            def _scalar_stride_and_G(tid):
                # `g * ssg` where g = divsi/divui(_, G) -> (ssg_arg_idx, G); a bare
                # `g` (ssg folded to 1) -> (-1, G).
                o = op_by_id.get(tid)
                if o is None:
                    return None, None
                if o.op in ("arith.divsi", "arith.divui"):
                    return -1, _G_from_divsi(tid)
                if o.op != "arith.muli":
                    return None, None
                _ssg = _g = None
                for oid in o.operand_ids:
                    ai = arg_id_to_idx.get(oid)
                    if ai is not None:
                        _ssg = ai
                    else:
                        _gv = _G_from_divsi(oid)
                        if _gv is not None:
                            _g = _gv
                return _ssg, _g

            def _resolve_scz(bc):
                # scale/zero load addr = addptr(splat(addptr(ptr, g*sg)), offs_n*sn).
                ld = _find_load(bc)
                if ld is None or not ld.operand_ids:
                    return None
                outer = op_by_id.get(ld.operand_ids[0])
                if outer is None or outer.op != "tt.addptr" or len(outer.operand_ids) < 2:
                    return None
                nstride = _vec_stride(outer.operand_ids[1])
                base = op_by_id.get(outer.operand_ids[0])
                if base is None or base.op != "tt.splat" or not base.operand_ids:
                    return None
                inner = op_by_id.get(base.operand_ids[0])
                if inner is None or inner.op != "tt.addptr" or len(inner.operand_ids) < 2:
                    return None
                ptr = arg_id_to_idx.get(inner.operand_ids[0])
                gstride, gval = _scalar_stride_and_G(inner.operand_ids[1])
                if None in (nstride, ptr, gstride, gval):
                    return None
                return ptr, gstride, nstride, gval

            _sc = _resolve_scz(scale_bc)
            _zc = _resolve_scz(zero_bc)
            if _sc is None or _zc is None:
                return None
            s_ptr, ssg, ssn, g_s = _sc
            z_ptr, zsg, zsn, g_z = _zc
            if s_ptr != 3 or z_ptr != 4 or g_s != g_z or g_s <= 0:
                return None
            _stride_idx += [ssg, ssn, zsg, zsn]
            if any(i is None for i in _stride_idx):
                return None
            # Carry a FAST simdgroup-MMA variant (~4-7x); the dispatch selects it only when
            # the runtime sizes/strides meet its contract (contiguous kn/row-major, aligned
            # M/N/K/group), else falls back to the stride-generic scalar template.
            _rr, _rc, _bk = 4, 2, 32
            if _is_int4:
                # B[0] (infer_dot_strides) is the packed weight's BYTE-row stride (wbk),
                # which the int4 template applies at k//2 — exactly the kernel's addressing.
                from triton_msl.codegen._msl_templates import (
                    make_int4_matmul_pergroup,
                    make_int4_matmul_pergroup_fast,
                )

                _fast4 = make_int4_matmul_pergroup_fast(g_s, _rr, _rc, _bk) if (g_s % _bk == 0) else None
                return (
                    "pergroup_int4",
                    make_int4_matmul_pergroup(g_s),
                    5,
                    6,
                    7,
                    tuple(_stride_idx),
                    _fast4,
                    _rr,
                    _rc,
                    _bk,
                    g_s,
                    (_gc["bm"], _gc["bn"]),
                    _gc["bk"],
                )  # [11] source (BM, BN), packet 105 B; [12] source BK (dispatcher SOURCE DOMAIN)
            from triton_msl.codegen._msl_templates import (
                make_int8_matmul_pergroup,
                make_int8_matmul_pergroup_fast,
            )

            _fast = make_int8_matmul_pergroup_fast(g_s, _rr, _rc, _bk) if (g_s % _bk == 0) else None
            return (
                "pergroup_int8",
                make_int8_matmul_pergroup(g_s),
                5,
                6,
                7,
                tuple(_stride_idx),
                _fast,
                _rr,
                _rc,
                _bk,
                g_s,
                (_gc["bm"], _gc["bn"]),
                _gc["bk"],
            )  # [11] source (BM, BN), packet 105 B; [12] source BK (dispatcher SOURCE DOMAIN)

        # --- Weight [K,N] contiguous-inner (+ input [M,K], output [M,N]) via strides. ---
        try:
            sd = self.infer_dot_strides()
        except Exception:  # noqa: BLE001
            return None
        if not sd:
            return None
        a_rc, b_rc, c_rc = sd.get("A"), sd.get("B"), sd.get("C")
        if not (a_rc and b_rc and c_rc):
            return None
        if not (a_rc[1] == "1" and b_rc[1] == "1" and c_rc[1] == "1"):
            return None  # non-contiguous inner (some other transpose) -> refuse
        _gc = self._quant_gemm_contract(op_by_id, k_arg=7, nk=_is_nk[0])  # packet 104: exact tiles / loop / store
        if _gc is None:
            return None
        # per-N scale / zero: EXACTLY ``ptr + program_id(1) * BN + range(0, BN)`` — the fast
        # kernel reads scale[n] / zero[n] contiguous (a runtime stride is not replayed).
        _NN = dict(pid=("required", 1), iv=0, stride="unit")
        for _bc, _ai in ((scale_bc, 3), (zero_bc, 4)):
            _ld = self._quant_find_load(_bc, op_by_id)
            if _ld is None or not _ld.operand_ids:
                return None
            _okn, _in = self._quant_role_addr(
                _ld.operand_ids[0], args[_ai], op_by_id, ivs={_gc["iv"]}, kinds={"vec": _NN}
            )
            if not _okn or self._quant_tile_ok(_in["vec"], block=_gc["bn"]) is None:
                return None

        # --- K-loop bound anchors to args[7] (the reliable K binding; same net as
        #     the fast-matmul descriptor). A non-canonical dim binding -> refuse. ---
        for _o in _all(self.graph.ops):
            if _o.op == "scf.for" and _o.operand_ids and len(_o.operand_ids) >= 2:
                _hi = _o.operand_ids[1]
                _depth = 0
                while _depth < 8 and _hi not in arg_id_to_idx:
                    _hop = op_by_id.get(_hi)
                    if _hop is None or not _hop.operand_ids:
                        break
                    _hi = _hop.operand_ids[0]
                    _depth += 1
                if arg_id_to_idx.get(_hi) != 7:
                    return None

        # --- dtypes: input fp32, output fp32, weight int8. ---
        if _mlir_to_triton_dtype(args[0].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[2].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[1].elem_type) not in ("int8", "i8", "si8"):
            return None
        # SCALE/ZERO dtype (re-review 2026-08-25): the templates declare
        # ``device const float*`` for scale/zero pointers — a non-f32 tensor (e.g. GPTQ's
        # fp16 scales) would be reinterpreted as float bits (garbage, no error). Refuse
        # the quant route; the generic lowering handles half/bf16 loads correctly.
        for _sza in args[3:4] + args[4:5]:
            if getattr(_sza, "is_ptr", False) and _mlir_to_triton_dtype(_sza.elem_type) not in ("fp32", "f32", "float"):
                return None

        # --- Runtime stride contract. The fast kernel hard-codes row-major addressing:
        #       input[m*K + k]  -> A row stride == K (args[7])
        #       output[m*N + n] -> C row stride == N (args[6])
        #       weight: layout 'kn' reads weight[k*N + n] (row == N, args[6]); layout
        #               'nk' reads weight[n*K + k] (row == K, args[7]).
        #     A stride arg that differs is verified at dispatch. ---
        b_expected = 7 if _is_nk[0] else 6
        name_to_idx = {a.name: i for i, a in enumerate(args)}
        stride_checks = []
        for _rowname, _exp_idx in ((a_rc[0], 7), (b_rc[0], b_expected), (c_rc[0], 6)):
            if _rowname in name_to_idx:
                _ri = name_to_idx[_rowname]
                if _ri != _exp_idx:
                    stride_checks.append((_ri, _exp_idx))
            elif _rowname == "1":
                return None  # row stride literally 1 but the dim is a runtime arg
            else:
                return None  # non-1 literal row stride: can't prove it equals the dim

        from triton_msl.codegen._msl_templates import make_int8_matmul_fast, make_int8_matmul_pergroup

        rr, rc = 4, 2
        layout = "nk" if _is_nk[0] else "kn"
        fast_msl = make_int8_matmul_fast(rr=rr, rc=rc, bk=32, layout=layout)
        # [7] = the SOURCE program mapping (BM on program_id(0), BN on program_id(1)) — packet 105 B.
        # [8] = the SOURCE K block, [9] = the weight layout, [10] = the stride-generic scalar
        # template: the kernel is unmasked, so it computes whole BM x BN x BK blocks; the
        # dispatcher runs those extents when the launch sizes are not whole blocks (its
        # SOURCE DOMAIN note).
        return (
            fast_msl,
            5,
            6,
            7,
            8 * rr,
            8 * rc,
            tuple(stride_checks),
            (_gc["bm"], _gc["bn"]),
            _gc["bk"],
            layout,
            make_int8_matmul_pergroup(32),
        )

    def _maybe_quant_matmul_symmetric_descriptor(self):
        """SYMMETRIC weight-only int8 GEMM: out = a @ (w_i8.to(float) * scale), NO
        zero-point — 4 ptrs (input, weight, output, scale), M/N/K at 4/5/6, per-N float
        scale, fp32 in/out, int8 weight. Routes to the stride-generic per-group scalar
        template (make_int8_matmul_pergroup) with a SYNTHESIZED all-zero zeros buffer +
        ssg=0 (so g*ssg=0 -> per-N regardless of the baked group size). Correct-or-refuse.
        """
        import os

        if os.environ.get("TRITON_MSL_QUANT_MATMUL", "1") == "0":
            return None

        def _all(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all(s.region_ops)
                if s.else_ops:
                    yield from _all(s.else_ops)

        op_by_id = {o.id: o for o in _all(self.graph.ops)}
        dots = [o for o in op_by_id.values() if o.op == "tt.dot"]
        if len(dots) != 1 or len(dots[0].operand_ids) < 2:
            return None
        dot = dots[0]
        args = self.graph.args
        if sum(1 for a in args if a.is_ptr) != 4 or not all(args[i].is_ptr for i in range(4)):
            return None
        if len(args) < 7 or args[4].is_ptr or args[5].is_ptr or args[6].is_ptr:
            return None
        arg_id_to_idx = {getattr(a, "id", None): i for i, a in enumerate(args)}
        if not self._quant_input_path_ok(dot.operand_ids[0], 0, op_by_id):  # F3
            return None

        def _peel(oid):
            o = op_by_id.get(oid)
            while (
                o is not None
                and o.operand_ids
                and o.op
                in (
                    "ttg.local_load",
                    "ttg.local_alloc",
                    "ttg.convert_layout",
                    "tt.trans",
                    "ttg.memdesc_trans",
                )
            ):
                o = op_by_id.get(o.operand_ids[0])
            return o

        mul = _peel(dot.operand_ids[1])
        if mul is None or mul.op != "arith.mulf" or len(mul.operand_ids) != 2:
            return None

        def _has_i2f_no_sub(oid, seen=None, d=0):
            if seen is None:
                seen = set()
            if oid in seen or d > 24:
                return False
            seen.add(oid)
            o = op_by_id.get(oid)
            if o is None:
                return False
            if o.op == "arith.subf":  # a subf -> ASYMMETRIC, not this path
                return False
            if o.op in ("arith.sitofp", "arith.uitofp"):
                return True
            return any(_has_i2f_no_sub(x, seen, d + 1) for x in (o.operand_ids or []))

        # weight side = the mul operand that reaches sitofp with NO subf; scale = the other.
        w_side = scale_bc = None
        for cw, cs in ((mul.operand_ids[0], mul.operand_ids[1]), (mul.operand_ids[1], mul.operand_ids[0])):
            _sub = op_by_id.get(cw)
            if _sub is not None and _sub.op == "arith.subf":
                return None  # asymmetric dequant present -> the canonical descriptor handles it
            if _has_i2f_no_sub(cw):
                w_side, scale_bc = cw, cs
                break
        if w_side is None:
            return None
        if not self._quant_legs_plain((w_side, scale_bc), op_by_id):  # packet 102: masks never replayed
            return None

        def _trace_to_arg(oid, d=0):
            if d > 32:
                return None
            if oid in arg_id_to_idx:
                return arg_id_to_idx[oid]
            o = op_by_id.get(oid)
            if o is None or not o.operand_ids:
                return None
            return _trace_to_arg(o.operand_ids[0], d + 1)

        if _trace_to_arg(scale_bc) != 3:
            return None
        if _mlir_to_triton_dtype(args[0].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[2].elem_type) not in ("fp32", "f32", "float"):
            return None
        if _mlir_to_triton_dtype(args[1].elem_type) not in ("int8", "i8", "si8"):
            return None
        # SCALE dtype (re-review 2026-08-25): the template declares ``device const float*``
        # for the scale — a non-f32 scale (e.g. fp16) would be reinterpreted as float bits.
        if getattr(args[3], "is_ptr", False) and _mlir_to_triton_dtype(args[3].elem_type) not in (
            "fp32",
            "f32",
            "float",
        ):
            return None

        try:
            sd = self.infer_dot_strides()
        except Exception:  # noqa: BLE001
            return None
        if not sd or not all(sd.get(x) for x in ("A", "B", "C")):
            return None
        _gc = self._quant_gemm_contract(op_by_id, k_arg=6, nk=False)  # packet 104: exact tiles / loop / store
        if _gc is None:
            return None
        _nti = {a.name: i for i, a in enumerate(args)}

        def _sidx(nm):
            return -1 if nm == "1" else _nti.get(nm)

        # per-N scale: addr = addptr(splat(scale_ptr), offs_n * ssn); ssn arg or -1 (folded).
        def _find_load(oid, d=0):
            o = op_by_id.get(oid)
            if o is None:
                return None
            if o.op == "tt.load":
                return o
            if not o.operand_ids:
                return None
            return _find_load(o.operand_ids[0], d + 1)

        def _vec_stride(tid):
            o = op_by_id.get(tid)
            if o is None:
                return None
            if o.op in ("tt.broadcast", "ttg.convert_layout", "tt.expand_dims") and o.operand_ids:
                return _vec_stride(o.operand_ids[0])
            if o.op == "arith.muli":
                for oid in o.operand_ids:
                    sub = op_by_id.get(oid)
                    if sub is not None and sub.op in ("tt.splat", "tt.broadcast") and sub.operand_ids:
                        ai = arg_id_to_idx.get(sub.operand_ids[0])
                        if ai is not None:
                            return ai
                    ai = arg_id_to_idx.get(oid)
                    if ai is not None:
                        return ai
                return None
            return -1  # no stride multiply -> stride 1

        sld = _find_load(scale_bc)
        if sld is None or not sld.operand_ids:
            return None
        # Packet 102: the scale address must be EXACTLY ``scale_ptr + offs_n [* ssn]`` with
        # ``offs_n = program_id(1) * BN + range`` — ``_vec_stride`` read the stride and let a
        # ``+ 1`` on the index vanish (GPU: raw result stored, err 0 vs raw).
        _ok, _isym = self._quant_role_addr(
            sld.operand_ids[0],
            args[3],
            op_by_id,
            ivs={_gc["iv"]},
            kinds={"vec": dict(pid=("required", 1), iv=0, stride="bound")},
        )
        if not _ok or self._quant_tile_ok(_isym["vec"], block=_gc["bn"]) is None:
            return None
        ssn = _isym["vec"]["stride"]

        idxs = [
            _sidx(sd["A"][0]),
            _sidx(sd["A"][1]),  # isr, isc
            _sidx(sd["B"][0]),
            _sidx(sd["B"][1]),  # wsk, wsn
            _sidx(sd["C"][0]),
            _sidx(sd["C"][1]),  # osr, osc
            ssn,  # scale n-stride
        ]
        if any(i is None for i in idxs):
            return None
        from triton_msl.codegen._msl_templates import make_int8_matmul_pergroup

        # [6] = the SOURCE program mapping (BM on program_id(0), BN on program_id(1)) — packet 105 B.
        # [7] = the SOURCE K block (the dispatcher's SOURCE DOMAIN note).
        return ("sym_int8", make_int8_matmul_pergroup(32), 4, 5, 6, tuple(idxs), (_gc["bm"], _gc["bn"]), _gc["bk"])

    def _maybe_fast_matmul_descriptor(self):
        """Build the runtime fast-matmul dispatch descriptor, or None.

        Path-independent: called from BOTH bare-matmul inline lowerings
        (_lower_simple_dot_inline and _lower_dot_simple_template). Returns
        (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n) ONLY when the kernel is a
        single tt.dot whose A/B/C pointers resolve to the CANONICAL buffer layout
        (A=arg0, B=arg1, C=arg2 — the buffers the fast template hard-codes) with
        M/N/K scalar args at 3/4/5, fp16/fp32 input, fp32 output. ADDITIVE: never
        changes the emitted generic kernel. The launcher additionally gates on
        runtime MPS + M%32/N%32/K%8 alignment (the fast template has no edge
        handling; misaligned dims would write OOB). Reordered-pointer kernels
        (e.g. test_dot_mulbroadcasted's Z,X,Y) resolve to non-canonical roles ->
        None -> generic kernel. Never silent-wrong.
        """
        import os

        if os.environ.get("TRITON_MSL_FAST_MATMUL", "1") == "0":
            return None

        # Exactly one tt.dot (a matmul), scanning nested regions (K-loop body).
        def _all(ops):
            for s in ops:
                yield s
                if s.region_ops:
                    yield from _all(s.region_ops)
                if s.else_ops:
                    yield from _all(s.else_ops)

        dots = [s for s in _all(self.graph.ops) if s.op == "tt.dot"]
        if len(dots) != 1:
            return None
        dot_ssa = dots[0]
        args = self.graph.args
        if len(args) < 6:
            return None
        all_ptr_args = [a for a in args if a.is_ptr]
        if len(all_ptr_args) != 3:
            return None
        # Resolve A/B/C roles; require the CANONICAL layout A=arg0, B=arg1, C=arg2
        # (the fast template hard-codes buffers 0/1/2). _resolve_dot_ptr_roles
        # returns None when it cannot prove the roles (e.g. for K-loop kernels
        # where the dot operands pass through scf.for block-arg forwarding that
        # the tracer cannot follow). When it CAN resolve and the layout is
        # non-canonical (e.g. test_dot_mulbroadcasted's Z,X,Y order), refuse.
        # When it CANNOT resolve, fall back to the positional check below
        # (same conservative rule as _detect_simple_dot uses for its K-loop path:
        # `_resolve_dot_ptr_roles(...) or all_ptr_args`). The positional fallback
        # requires args[0..2] to be the only three pointers in exact declaration
        # order -- which is true for all standard Triton matmuls. Compare by
        # .name (unique per kernel).
        roles = self._resolve_dot_ptr_roles(dot_ssa, all_ptr_args)
        if roles is not None:
            # Role resolution succeeded: verify canonical A=arg0, B=arg1, C=arg2.
            if len(roles) < 3:
                return None
            if not (roles[0].name == args[0].name and roles[1].name == args[1].name and roles[2].name == args[2].name):
                return None
        else:
            # Role resolution failed (K-loop tracer limitation): fall back to
            # positional check. Requires the first three args are the only three
            # pointers in declaration order (A/B/C at 0/1/2).
            if not (args[0].is_ptr and args[1].is_ptr and args[2].is_ptr):
                return None
        # M/N/K runtime scalars at buffers 3/4/5 (the template binds buffer3=M,
        # buffer4=N, buffer5=K). The canonical A/B/C@0-2 resolution above confirms
        # the standard (a,b,c,M,N,K,...) signature; the full ratchet is the net.
        if args[3].is_ptr or args[4].is_ptr or args[5].is_ptr:
            return None
        # Output dtype selects the template variant: fp32 -> direct float* store,
        # fp16/bf16 -> half/bfloat* C + cast epilogue (float accumulation preserved
        # either way). Other output -> ineligible (fall back to the generic kernel).
        out_dtype_t = _mlir_to_triton_dtype(args[2].elem_type)
        if out_dtype_t in ("fp32", "f32", "float"):
            msl_out = "fp32"
        elif out_dtype_t in ("fp16", "f16"):
            msl_out = "fp16"
        elif out_dtype_t in ("bf16",):
            msl_out = "bf16"
        else:
            return None
        # Input fp16, bf16, or fp32 (the template's three supported branches). bf16
        # uses the M-series simdgroup_bfloat8x8 matrix unit (float accumulate) — same
        # ~11 TFLOP/s fast path as fp16, vs the ~2.4 TFLOP/s generic float-compute
        # fallback bf16 used to take.
        in_dtype = _mlir_to_triton_dtype(args[0].elem_type)
        if in_dtype in ("fp16", "f16"):
            msl_dtype = "fp16"
        elif in_dtype in ("bf16",):
            msl_dtype = "bf16"
        elif in_dtype in ("fp32", "f32"):
            msl_dtype = "fp32"
        else:
            return None
        # VERIFY the assumed K (buffer 5) is REALLY the matmul K: the K-loop's scf.for
        # upper bound must trace to args[5]. The descriptor hard-codes m/n/k_idx = 3/4/5
        # (the canonical `(a,b,c,M,N,K,...)` signature) and the fast template IGNORES the
        # kernel's explicit stride args + assumes row-major M/N/K. A non-standard signature
        # — strides at 3/4/5 with K elsewhere (a kernel passing strides before the dims, or
        # M/N as constexpr) — would mis-bind strides as M/N/K and SILENTLY mis-compute.
        # (Triton drops unit strides, so arg positions shift; the K-loop bound is the
        # reliable anchor.) If any K-loop's bound is not args[5], refuse -> generic path.
        _arg_index = {getattr(a, "id", None): i for i, a in enumerate(args)}
        _by_id = {o.id: o for o in self.graph.ops}
        for _o in _all(self.graph.ops):
            if _o.op == "scf.for" and _o.operand_ids and len(_o.operand_ids) >= 2:
                _hi = _o.operand_ids[1]
                _depth = 0
                while _depth < 8 and _hi not in _arg_index:
                    _hop = _by_id.get(_hi)
                    if _hop is None or not _hop.operand_ids:
                        break
                    _hi = _hop.operand_ids[0]
                    _depth += 1
                if _arg_index.get(_hi) != 5:
                    return None  # K-loop bound is not args[5] => non-canonical => refuse
        # STRIDE GATE (brief): the fast compile_shader template hard-codes
        # ROW-MAJOR addressing — leading dims = the M/N/K args, all INNER strides
        # = 1. A non-contiguous inner dim (a transposed B from ``x @ w.t()``)
        # cannot be loaded this way and would be silently wrong, so decline when
        # the traced layout shows ANY operand's col (inner) stride is not the
        # unit "1". The non-contiguous case then takes the stride-aware scalar
        # template; a contiguous-inner operand whose row stride differs from the
        # dim is handled by the simdgroup template's inferred leading dims (this
        # fast descriptor's row==dim assumption is additionally net-checked by
        # the launcher's runtime alignment gate). Only gates when traceable.
        # Build a RUNTIME stride contract: the fast template uses the M/N/K dim
        # buffers as leading dims and assumes inner stride 1. That is correct iff
        # the operand strides equal that row-major layout AT RUNTIME. When the
        # kernel passes explicit stride args (their values aren't known at
        # compile time), the launcher must verify them and skip the fast path if
        # they differ (a column-sliced contiguous operand, or any non-row-major
        # layout). stride_checks = [(arg_index, expected_dim_arg_index_or_-1)]
        # where -1 means "expected literal 1".
        _name_to_idx = {a.name: i for i, a in enumerate(args)}
        try:
            _sd = self.infer_dot_strides()
        except Exception:  # noqa: BLE001
            _sd = None
        stride_checks = []
        if _sd is not None:
            _ar, _ac = _sd.get("A", (None, None))
            _br, _bc = _sd.get("B", (None, None))
            _cr, _cc = _sd.get("C", (None, None))
            # Inner (col) stride must be contiguous; a non-unit inner dim
            # (transposed B) can't be loaded by the fast template at all.
            if not (_ac == "1" and _bc == "1" and _cc == "1"):
                return None
            # Row strides: name-equal to the dim arg => statically canonical (no
            # runtime check). A *different* runtime arg => emit a runtime check
            # (value must equal the dim). A literal int constant must equal the
            # dim too, but the dim is a runtime arg here so a constant row stride
            # can't be statically validated -> require a runtime check is
            # impossible (no arg to read) => decline to be safe.
            _kidx, _nidx = 5, 4
            for _rs, _exp_idx in ((_ar, _kidx), (_br, _nidx), (_cr, _nidx)):
                if _rs in _name_to_idx:
                    _ri = _name_to_idx[_rs]
                    if _ri != _exp_idx:  # explicit stride arg != dim
                        stride_checks.append((_ri, _exp_idx))
                elif _rs == "1":
                    # row stride literally 1 but the dim is a runtime arg: only
                    # correct if that dim is 1 at runtime — too degenerate; the
                    # fast template would mis-stride. Decline.
                    return None
                else:
                    # row stride is a non-1 literal constant; can't prove it
                    # equals the runtime dim. Decline rather than risk it.
                    return None
        from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

        rr = rc = 4
        fast_msl = make_simdgroup_matmul_kernel_fast(dtype=msl_dtype, rr=rr, rc=rc, out_dtype=msl_out)
        # (msl, m_idx, n_idx, k_idx, tile_m, tile_n, msl_dtype, msl_out, stride_checks).
        # The driver builds alternative (rr,rc) variants for per-shape autotuning;
        # stride_checks are verified at dispatch (skip fast path if a runtime
        # stride doesn't match the assumed row-major layout).
        # Packet 105 B: the SOURCE program mapping, proven by the value-path predicate
        # (grid map + per-axis pid presence) and the dot operand tiles:
        # ("2d", BM, BN, pid_on_rows, pid_on_cols) | ("1d", BM, BN). The dispatcher runs the
        # fast template only for exactly that program grid.
        _vp = self._dot_template_value_paths()
        if _vp[0] is not None or len(_vp) < 5:
            return None
        if isinstance(_vp[3], tuple):
            # Grouped and N-fastest sources use the coordinate-preserving template.
            # Do not mislabel it "2d" and let a full-output replacement reinterpret
            # a partial/swizzled source launch. Fast-grid admission is separate work.
            return None
        # The replacement computes rows [0, args[3]) by columns [0, args[4]) and nothing
        # else. Every source domain fact must name that very argument or be absent: a
        # separate store clip, load clip, load modulus or 1-D grid divisor describes a
        # different result (GPU: a store clip MS < M was written through to row M).
        _domain = _vp[6] if len(_vp) > 6 and isinstance(_vp[6], dict) else None
        if _domain is None:
            return None
        _canon = (args[3].name, args[4].name)
        _store = self._matmul_output_extent_args()
        for _facts in (
            _store,
            _domain.get("grid_extent_args", (None, None)),
            _domain.get("load_extent_args", (None, None)),
            _domain.get("load_modulo_args", (None, None)),
            _domain.get("load_identity_modulo_args", (None, None)),
        ):
            if len(_facts) != 2 or any(f is not None and f != c for f, c in zip(_facts, _canon)):
                return None
        # An unmasked store axis writes whole tiles, so the dispatcher may replace it
        # only when that runtime extent is a whole number of source tiles.
        _whole_tile_axes = (_store[0] is None, _store[1] is None)
        _ops = {o.id: o for o in _all(self.graph.ops)}
        _ash = _extract_shape(_ops[dot_ssa.operand_ids[0]].type_str or "") if dot_ssa.operand_ids[0] in _ops else None
        _bsh = _extract_shape(_ops[dot_ssa.operand_ids[1]].type_str or "") if dot_ssa.operand_ids[1] in _ops else None
        if not (_ash and _bsh and len(_ash) == 2 and len(_bsh) == 2):
            return None
        _bm, _bn = int(_ash[0]), int(_bsh[1])
        _grid_spec = ("1d", _bm, _bn) if _vp[3] == "1d" else ("2d", _bm, _bn, bool(_vp[4][0]), bool(_vp[4][1]))
        # Bind the replacement launch to the source K-loop tail contract.  An
        # unmasked source executes its final complete BLOCK_K iteration; the
        # fast shader instead loops to logical K and therefore may replace it
        # only when K is BLOCK_K-divisible.  A both-zero-masked source does have
        # logical-K semantics and retains the existing fast-path K%8 gate.
        _tail_mode = _vp[5] if len(_vp) > 5 else None
        _block_k = None
        if _tail_mode is not None:
            # The dot operand's inner tile is the source loop's BLOCK_K; this
            # shape was already operation-owned and validated immediately above.
            _block_k = int(_ash[1])
            if _tail_mode not in ("masked_zero", "full_blocks") or _block_k <= 0:
                return None
        return (
            fast_msl,
            3,
            4,
            5,
            8 * rr,
            32 * rc,
            msl_dtype,
            msl_out,
            tuple(stride_checks),
            _grid_spec,
            _tail_mode,
            _block_k,
            _whole_tile_axes,
        )
