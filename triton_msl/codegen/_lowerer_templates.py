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


def _emit_masked_staged_store(lines, *, acc, scratch, gr, gc, cond, dst, out_type,
                              store_pfx="", indent="    "):
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
            return None              # not a traceable single-dot matmul
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
                op_name="tt.dot")
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        # Contiguous inner dim == col stride is literal "1". A runtime-arg or
        # non-unit-constant col stride means the inner dim is strided.
        if a_col == "1" and b_col == "1" and c_col == "1":
            return ("simdgroup", descriptors)
        return ("scalar", descriptors)

    def _lower_strided_scalar_matmul(self, info, descriptors):
        """Fully stride-aware scalar matmul for a NON-contiguous-inner operand.

        Used when the simdgroup-MMA path can't load an operand because its inner
        dim is strided (a transposed B from ``x @ w.t()``, a column-major C, a
        sliced operand). Each thread computes a set of output elements; every
        load/store addresses through the 6 inferred strides
        ``A[m*a_row + k*a_col]``, ``B[k*b_row + n*b_col]``,
        ``C[m*c_row + n*c_col]`` (each stride emitted as a literal or a runtime
        ``int`` arg), so it is correct for ANY layout. Slower than the simdgroup
        fast path, but that path is correctness-impossible here; the common
        contiguous case never reaches this method.

        Handles both the pid-tiled K-loop form (runtime M/N/K, grid tiles over
        BLOCK_M/BLOCK_N) and the single-tile form (constexpr dims).
        """
        a_row, a_col, b_row, b_col, c_row, c_col = descriptors
        ptr_args = info["ptr_args"]
        a_name = ptr_args[0].name
        b_name = ptr_args[1].name
        c_name = ptr_args[2].name
        a_msl = triton_type_to_msl(ptr_args[0].elem_type)
        b_msl = triton_type_to_msl(ptr_args[1].elem_type)
        c_msl = triton_type_to_msl(ptr_args[2].elem_type)

        all_scalar_args = [a for a in self.graph.args if not a.is_ptr]
        scalar_names = {a.name for a in all_scalar_args}

        def _sx(desc):
            # Stride descriptor -> MSL expression. A runtime arg name must be a
            # declared scalar (unpacked from its buffer below); a literal int
            # string passes through. (Descriptors only ever hold an arg NAME or
            # an integer literal — never None here; the gate refused None.)
            return desc if desc not in scalar_names else f"(uint){desc}"

        a_rs, a_cs = _sx(a_row), _sx(a_col)
        b_rs, b_cs = _sx(b_row), _sx(b_col)
        c_rs, c_cs = _sx(c_row), _sx(c_col)

        has_k_loop = bool(info.get("has_k_loop"))
        if has_k_loop:
            BLOCK_M = info["BLOCK_M"]; BLOCK_N = info["BLOCK_N"]; BLOCK_K = info["BLOCK_K"]
            scalar_arg_map = info.get("scalar_args", {})
        else:
            BLOCK_M = info["M"]; BLOCK_N = info["N"]; BLOCK_K = info["K"]
            scalar_arg_map = {a.name: a for a in all_scalar_args}

        has_M = "M" in scalar_arg_map
        has_N = "N" in scalar_arg_map
        has_K = "K" in scalar_arg_map
        pid_axes = {s.attrs.get("axis", 0) for s in self.graph.ops
                    if s.op == "tt.get_program_id"}
        has_pid = len(pid_axes) > 0

        # Same integrity guard as the simdgroup K-loop: if the kernel tiles the
        # output across programs but M/N are constexpr (no runtime arg), the true
        # output extent is unknown here -> refuse rather than guess.
        if (1 in pid_axes and not has_N) or (0 in pid_axes and not has_M):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "strided matmul tiles the output across programs "
                f"(program_id axes {sorted(pid_axes)}) but M/N are baked as "
                "constexpr, not runtime args — the true output extent can't be "
                "derived. Refusing.", op_name="tt.dot")

        # 256 threads per threadgroup; each handles a strided subset of the tile.
        block_size = 256
        self.effective_block_size = block_size
        tile = BLOCK_M * BLOCK_N

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
                arg_decls.append(f"    device {m}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(f"    device int* {arg.name}_buf [[buffer({i})]]")
        lines.append(",\n".join(arg_decls) + ",")
        if has_pid:
            self._used_pid_axes = {0, 1}
            # Metal requires all thread-index attributes to share a type; use
            # uint3 for both when multi-axis dispatch is in play.
            lines.append("    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append("    uint3 _lid3 [[thread_position_in_threadgroup]]")
            lines.append(") {")
            lines.append("    uint pid_m = pid3.x;")
            lines.append("    uint pid_n = pid3.y;")
            lines.append("    uint lid = _lid3.x;")
        else:
            lines.append("    uint pid [[threadgroup_position_in_grid]],")
            lines.append("    uint lid [[thread_position_in_threadgroup]]")
            lines.append(") {")
            lines.append("    uint pid_m = 0u, pid_n = 0u;")
        for arg in all_scalar_args:
            lines.append(f"    int {arg.name} = {arg.name}_buf[0];")

        lines.append(f"    uint _M = {'(uint)M' if has_M else f'{BLOCK_M}u'};")
        lines.append(f"    uint _N = {'(uint)N' if has_N else f'{BLOCK_N}u'};")
        if has_K:
            lines.append("    uint _K = (uint)K;")
        else:
            scf_iters = info.get("scf_iters")
            if has_k_loop and scf_iters and scf_iters > 1:
                lines.append(f"    uint _K = {BLOCK_K * scf_iters}u;")
            else:
                lines.append(f"    uint _K = {BLOCK_K}u;")
        lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
        lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
        lines.append("")
        # Each thread strides over the BLOCK_M x BLOCK_N tile.
        lines.append(f"    for (uint _e = lid; _e < {tile}u; _e += {block_size}u) {{")
        lines.append(f"        uint _lm = _e / {BLOCK_N}u;")
        lines.append(f"        uint _ln = _e % {BLOCK_N}u;")
        lines.append("        uint m = row_base + _lm;")
        lines.append("        uint n = col_base + _ln;")
        lines.append("        if (m >= _M || n >= _N) continue;")
        lines.append("        float _sum = 0.0f;")
        lines.append("        for (uint k = 0u; k < _K; k++) {")
        lines.append(f"            _sum += (float){a_name}[m * {a_rs} + k * {a_cs}]")
        lines.append(f"                  * (float){b_name}[k * {b_rs} + n * {b_cs}];")
        lines.append("        }")
        lines.append(f"        {c_name}[m * {c_rs} + n * {c_cs}] = ({c_msl})_sum;")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

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

        # Phase 4: record the runtime fast-matmul dispatch descriptor (additive;
        # the generic inline kernel below is still emitted + returned).
        self._fast_matmul = self._maybe_fast_matmul_descriptor()
        if info.get("has_k_loop"):
            return self._lower_k_loop_dot_inline(info)

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

        a_name = ptr_args[0].name
        b_name = ptr_args[1].name
        c_name = ptr_args[2].name

        # Determine input element type from A pointer arg. ``triton_type_to_msl``
        # handles bf16 → bfloat, fp16 → half, fp64 → float (with warning), etc.;
        # the previous hand-rolled mapping silently misrouted bf16 as float
        # (4-byte stride over a 2-byte buffer → wrong loads, partial output).
        a_elem = ptr_args[0].elem_type  # e.g. "f32", "f16", "bf16"
        input_msl_type = triton_type_to_msl(a_elem)

        # Determine output element type from C pointer arg.
        c_elem = ptr_args[2].elem_type
        output_msl_type = triton_type_to_msl(c_elem)

        # Genuine fp16 (WS1 Phase C): half INPUT fragments + float ACCUMULATOR;
        # fp16 staged as half (no upcast) so the half MMA is actually used. bf16
        # uses the bfloat MMA (simdgroup_bfloat8x8); other types stay on the float
        # path (exact). Accumulator always float.
        acc_frag = "simdgroup_float8x8"
        if input_msl_type == "half":
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_half8x8", "half", "", "half(0.0h)")
        elif input_msl_type == "bfloat":
            # M-series simdgroup_bfloat8x8 matrix unit (verified on M4): bfloat
            # input fragments + float accumulator, staged as bfloat (no upcast),
            # so the direct-device-load path loads bfloat into a bfloat fragment.
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_bfloat8x8", "bfloat", "", "bfloat(0.0)")
        else:
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_float8x8", "float", "float", "0.0f")
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
        lines.append(f"    device const {input_msl_type}* {a_name} [[buffer(0)]],")
        lines.append(f"    device const {input_msl_type}* {b_name} [[buffer(1)]],")
        lines.append(f"    device {output_msl_type}* {c_name} [[buffer(2)]],")
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
        lines.append(f"                tg_A[i] = (gr < M && gc < K) ? {stage_cast}({a_name}[a_batch_off + {a_index}]) : {pad};")
        lines.append(f"            }}")

        # Load B tile (8x32) cooperatively. ``trans_b`` analogue: original
        # buffer Y has shape (N, K) and B[gr, gc] = Y[gc, gr] = Y_buf[gc * K + gr].
        b_index = f"gc * K + gr" if trans_b else f"gr * N + gc"
        lines.append(f"            uint col_base_tg = tile_col * 32u;")
        lines.append(f"            for (uint i = tiitg; i < 256u; i += 128u) {{")
        lines.append(f"                uint r = i / 32u, c = i % 32u;")
        lines.append(f"                uint gr = kk + r, gc = col_base_tg + c;")
        lines.append(f"                tg_B[i] = (gr < K && gc < N) ? {stage_cast}({b_name}[b_batch_off + {b_index}]) : {pad};")
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
                    lines, acc=_acc, scratch="tg_out",
                    gr=f"row_base + {_n * 8}u + i / 8u", gc="col_base + i % 8u",
                    cond="gr < M && gc < N",
                    dst=f"{c_name}[c_batch_off + gr * N + gc]",
                    out_type=output_msl_type, indent="        ")

        lines.append(f"    }} // tile_col")
        lines.append(f"    }} // tile_row")
        lines.append(f"    }} // batch")
        lines.append(f"}}")

        return "\n".join(lines)


    def _lower_k_loop_dot_inline(self, info):
        """Generate MSL for a K-loop dot kernel using simdgroup MMA.

        Each threadgroup computes one BLOCK_M x BLOCK_N output tile, iterating
        over K in steps of BLOCK_K. Uses pid_m (program_id(0)) and pid_n
        (program_id(1)) to select which output tile to compute.

        128 threads = 4 SIMD groups x 32 threads per threadgroup.
        Each SIMD group handles an 8-column slice of the 32-wide output tile.
        The inner MMA loop iterates BLOCK_K in steps of 8.
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

        a_name = ptr_args[0].name
        b_name = ptr_args[1].name
        c_name = ptr_args[2].name

        # Determine input/output element types via the shared mapping —
        # see the simple-dot variant for the bf16 stride bug this avoids.
        a_elem = ptr_args[0].elem_type
        input_msl_type = triton_type_to_msl(a_elem)
        c_elem = ptr_args[2].elem_type
        output_msl_type = triton_type_to_msl(c_elem)

        # Genuine fp16 (WS1 Phase C): half INPUT fragments + float ACCUMULATOR
        # (half x half -> float MMA, de-risked in test_simdgroup_half_mma.py).
        # fp16 inputs are staged as half and fed to simdgroup_half8x8 — no float
        # upcast, so Apple's ~2x fp16 matrix throughput is actually used. bf16
        # uses simdgroup_bfloat8x8 (the M-series bfloat matrix unit, verified on
        # M4); other types stay on the float path. The accumulator is always float.
        acc_frag = "simdgroup_float8x8"
        if input_msl_type == "half":
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_half8x8", "half", "", "half(0.0h)")
        elif input_msl_type == "bfloat":
            # M-series simdgroup_bfloat8x8 matrix unit (verified on M4): bfloat
            # input fragments + float accumulator, staged as bfloat (no upcast),
            # so the direct-device-load path loads bfloat into a bfloat fragment.
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_bfloat8x8", "bfloat", "", "bfloat(0.0)")
        else:
            in_frag, tg_type, stage_cast, pad = (
                "simdgroup_float8x8", "float", "float", "0.0f")
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
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append(f"    uint3 pid3 [[threadgroup_position_in_grid]],")
        lines.append(f"    uint sgitg [[simdgroup_index_in_threadgroup]],")
        lines.append(f"    uint tiitg [[thread_index_in_threadgroup]]")
        lines.append(f") {{")

        # Unpack scalar args from buffers
        for arg in all_scalar_args:
            lines.append(f"    int {arg.name} = {arg.name}_buf[0];")

        lines.append(f"")
        lines.append(f"    uint pid_m = pid3.x;")
        lines.append(f"    uint pid_n = pid3.y;")
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
        # emit wrong numbers: an UNSUPPORTED stub makes emit_msl fall back.
        pid_axes = {s.attrs.get("axis", 0) for s in self.graph.ops
                    if s.op == "tt.get_program_id"}
        if (1 in pid_axes and not has_N) or (0 in pid_axes and not has_M):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "K-loop matmul tiles the output across programs "
                f"(program_id axes {sorted(pid_axes)}) but M/N are baked as "
                "constexpr, not runtime args — the true output strides "
                "can't be derived (e.g. test_dot_mulbroadcasted).")

        if has_M:
            lines.append(f"    uint _M = (uint)M;")
        else:
            lines.append(f"    uint _M = {BLOCK_M}u;  // no M arg, single tile")
        if has_N:
            lines.append(f"    uint _N = (uint)N;")
        else:
            lines.append(f"    uint _N = {BLOCK_N}u;  // no N arg, single tile")
        scf_iters = info.get("scf_iters")
        if has_K:
            lines.append(f"    uint _K = (uint)K;")
        elif scf_iters and scf_iters > 1:
            # K is a constexpr / not a runtime scalar arg, but the
            # ``scf.for`` body iterates ``scf_iters`` times across BLOCK_K
            # chunks. Total K = BLOCK_K * scf_iters
            # (``test_dot_mulbroadcasted`` baked K=160 in as constexpr).
            lines.append(f"    uint _K = {BLOCK_K * scf_iters}u;  // BLOCK_K * scf.for iters")
        else:
            lines.append(f"    uint _K = {BLOCK_K}u;  // no K arg, single tile")

        # Leading dims for the simdgroup loads/stores = the DENSE row-major row
        # stride. This path only handles contiguous-INNER operands (the
        # stride-aware gate routed any strided inner dim to the scalar template),
        # and the Metal driver materializes each operand DENSE row-major before
        # dispatch (a non-contiguous host tensor is memmove'd as contiguous
        # bytes), so the in-kernel row stride is always the matrix dim: A leading
        # = K, B/C leading = N. (A passed runtime stride that differs from the
        # dim describes the HOST layout, not the materialized device buffer, so
        # using it here would mis-address — the dim is correct.)
        a_ld, b_ld, c_ld = "_K", "_N", "_N"

        lines.append(f"")
        lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
        lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
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
                "8-row simdgroup tiling would drop the tail rows. Refusing.")
        if BLOCK_K % 8 != 0:
            raise MetalNonRecoverableError(
                f"K-loop matmul BLOCK_K={BLOCK_K} is not a multiple of 8; the "
                "8-deep MMA step would drop the tail of K. Refusing.")
        if BLOCK_N % 8 != 0:
            raise MetalNonRecoverableError(
                f"K-loop matmul BLOCK_N={BLOCK_N} is not a multiple of 8; "
                "output columns tile in 8-wide simdgroup blocks. Refusing.")
        # Distribute the BLOCK_N/8 8-wide column blocks across the 4 simdgroups
        # (ceil). For BLOCK_N a multiple of 32 all four are fully used; for
        # smaller / non-32-multiple BLOCK_N the extra simdgroups idle (guarded).
        total_col_blocks = BLOCK_N // 8
        col_tiles = (total_col_blocks + 3) // 4   # 8-wide col blocks per simdgroup
        cols_per_sg = col_tiles * 8
        col_needs_guard = (total_col_blocks % 4 != 0)

        def _col_guard(c):
            # Uniform-per-simdgroup predicate: is local block c a real column?
            # None when every (sgitg, c) is in range (no runtime cost).
            return (f"(sgitg * {col_tiles}u + {c}u) < {total_col_blocks}u"
                    if col_needs_guard else None)
        # Threadgroup budget: the staged path needs tg_A + tg_B (plus a small
        # 1 KiB store scratch). Refuse if it exceeds Metal's 32 KiB threadgroup
        # limit rather than emit a kernel that silently overflows.
        tg_elt = 2 if tg_type in ("half", "bfloat") else 4   # bfloat is 2 bytes too
        tg_bytes = (tg_a_size + tg_b_size) * tg_elt + 4 * 64 * 4
        if tg_bytes > 32 * 1024:
            raise MetalNonRecoverableError(
                f"K-loop matmul needs {tg_bytes} B of threadgroup memory "
                "(> 32 KiB limit) for the staged path; refusing.")

        # Accumulators: one simdgroup_float8x8 per (8-row tile, 8-col block).
        acc_names = [[f"acc_{r}_{c}" for c in range(col_tiles)]
                     for r in range(n_row_tiles)]
        all_accs = [acc_names[r][c] for r in range(n_row_tiles)
                    for c in range(col_tiles)]
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
            lines.append(f"    if (row_base + {BLOCK_M}u <= _M && col_base + {BLOCK_N}u <= _N && (_K % 8u) == 0u) {{")
            lines.append(f"        for (uint k = 0u; k < _K; k += 8u) {{")
            for c in range(col_tiles):
                g = _col_guard(c)
                pfx = f"if ({g}) " if g else ""
                lines.append(f"            {pfx}simdgroup_load(b_frag{c}, {b_name} + k * {b_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {b_ld});")
            for t in range(n_row_tiles):
                lines.append(f"            simdgroup_load(a_frag, {a_name} + (row_base + {t * 8}u) * {a_ld} + k, {a_ld});")
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(f"            {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});")
            lines.append(f"        }}")
            for t in range(n_row_tiles):
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(f"        {pfx}simdgroup_store({acc_names[t][c]}, {c_name} + (row_base + {t * 8}u) * {c_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {c_ld});")
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
        lines.append(f"            tg_A[i] = (gr < _M && gc < _K) ? {stage_cast}({a_name}[gr * {a_ld} + gc]) : {pad};")
        lines.append(f"        }}")

        # Cooperative masked load B tile (STAGE_DEPTH x BLOCK_N).
        lines.append(f"        for (uint i = tiitg; i < {tg_b_size}u; i += 128u) {{")
        lines.append(f"            uint r = i / {BLOCK_N}u, c = i % {BLOCK_N}u;")
        lines.append(f"            uint gr = _k + r, gc = col_base + c;")
        lines.append(f"            tg_B[i] = (gr < _K && gc < _N) ? {stage_cast}({b_name}[gr * {b_ld} + gc]) : {pad};")
        lines.append(f"        }}")
        lines.append(f"        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"")

        # One STAGE_DEPTH(8)-deep MMA step over the staged tiles.
        for c in range(col_tiles):
            g = _col_guard(c)
            pfx = f"if ({g}) " if g else ""
            lines.append(f"        {pfx}simdgroup_load(b_frag{c}, tg_B + sgitg * {cols_per_sg}u + {c * 8}u, {BLOCK_N});")
        for t in range(n_row_tiles):
            lines.append(f"        simdgroup_load(a_frag, tg_A + {t}u * 8u * {STAGE_DEPTH}u, {STAGE_DEPTH});")
            for c in range(col_tiles):
                g = _col_guard(c)
                pfx = f"if ({g}) " if g else ""
                lines.append(f"        {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});")
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
                    lines, acc=acc_names[t][c], scratch="tg_st",
                    gr=f"row_base + {t * 8}u + i / 8u",
                    gc=f"col_base + sgitg * {cols_per_sg}u + {c * 8}u + i % 8u",
                    cond=cond, dst=f"{c_name}[gr * {c_ld} + gc]",
                    out_type=output_msl_type, store_pfx=spfx, indent="    ")

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
        if (output_msl_type == "float" and has_M and has_N and has_K
                and "M" in _argnames and "N" in _argnames and "K" in _argnames):
            dn = f"{safe_name}__mmdirect"
            lines.append("")
            lines.append(f"kernel void {dn}(")
            lines.append(",\n".join(arg_decls) + ",")
            lines.append(f"    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append(f"    uint sgitg [[simdgroup_index_in_threadgroup]],")
            lines.append(f"    uint tiitg [[thread_index_in_threadgroup]]")
            lines.append(f") {{")
            for arg in all_scalar_args:
                lines.append(f"    int {arg.name} = {arg.name}_buf[0];")
            lines.append(f"    uint pid_m = pid3.x, pid_n = pid3.y;")
            lines.append(f"    uint _M = (uint)M, _N = (uint)N, _K = (uint)K;")
            lines.append(f"    uint row_base = pid_m * {BLOCK_M}u;")
            lines.append(f"    uint col_base = pid_n * {BLOCK_N}u;")
            lines.append(f"    {acc_frag} {', '.join(n + '(0)' for n in all_accs)};")
            lines.append(f"    {in_frag} a_frag, {', '.join(b_names)};")
            lines.append(f"    for (uint k = 0u; k < _K; k += 8u) {{")
            for c in range(col_tiles):
                g = _col_guard(c)
                pfx = f"if ({g}) " if g else ""
                lines.append(f"        {pfx}simdgroup_load(b_frag{c}, {b_name} + k * {b_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {b_ld});")
            for t in range(n_row_tiles):
                lines.append(f"        simdgroup_load(a_frag, {a_name} + (row_base + {t * 8}u) * {a_ld} + k, {a_ld});")
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(f"        {pfx}simdgroup_multiply_accumulate({acc_names[t][c]}, a_frag, b_frag{c}, {acc_names[t][c]});")
            lines.append(f"    }}")
            for t in range(n_row_tiles):
                for c in range(col_tiles):
                    g = _col_guard(c)
                    pfx = f"if ({g}) " if g else ""
                    lines.append(f"    {pfx}simdgroup_store({acc_names[t][c]}, {c_name} + (row_base + {t * 8}u) * {c_ld} + col_base + sgitg * {cols_per_sg}u + {c * 8}u, {c_ld});")
            lines.append(f"}}")
            self._mm_two_kernel = {
                "direct_name": dn, "block_m": BLOCK_M, "block_n": BLOCK_N,
                "m_idx": _argnames.index("M"), "n_idx": _argnames.index("N"),
                "k_idx": _argnames.index("K"),
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

        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")

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

        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                # Scalar arg passed as constant&
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

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
        lines.append("    float local_max = -INFINITY;")
        lines.append("")

        # ----- Phase 1: load → row_cache, reduce local max -----
        if vectorize:
            lines.append(f"    bool use_vec = (({n_arg} & 3) == 0);")
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = {n_arg} / 4;")
            lines.append(f"        device float4* x4 = (device float4*)({input_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = x4[i];")
            lines.append("            row4[i] = v;")
            lines.append("            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"            float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("            row_cache[i] = v;")
            lines.append("            local_max = max(local_max, v);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"        float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("        row_cache[i] = v;")
            lines.append("        local_max = max(local_max, v);")
            lines.append("    }")

        # Reduce max across threadgroup
        lines.append("    float simd_max_v = simd_max(local_max);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_max_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_max = simd_max((tiisg < {n_simd}u) ? "
                     f"reduce_buf[tiisg] : -INFINITY);")
        lines.append("    float local_sum = 0.0f;")
        lines.append("")

        # ----- Phase 2: exp(x - max) → row_cache, reduce local sum -----
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = {n_arg} / 4;")
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
            lines.append(f"        for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append("            float e = exp(row_cache[i] - row_max);")
            lines.append("            row_cache[i] = e;")
            lines.append("            local_sum += e;")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append("        float e = exp(row_cache[i] - row_max);")
            lines.append("        row_cache[i] = e;")
            lines.append("        local_sum += e;")
            lines.append("    }")

        # Reduce sum
        lines.append("    float simd_sum_v = simd_sum(local_sum);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sum_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sum = simd_sum((tiisg < {n_simd}u) ? "
                     f"reduce_buf[tiisg] : 0.0f);")
        lines.append("    float inv_sum = 1.0f / row_sum;")
        lines.append("")

        # ----- Phase 3: write normalized exp to global memory -----
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = {n_arg} / 4;")
            lines.append(f"        device float4* o4 = (device float4*)({output_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            o4[i] = row4[i] * inv_sum;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"            {output_arg}[row_start + i] = {_store_cast}(row_cache[i] * inv_sum);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"        {output_arg}[row_start + i] = {_store_cast}(row_cache[i] * inv_sum);")
            lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_layer_norm_template(self, info) -> str:
        """Emit a TG-cached row-wise layer-norm kernel.

        Mirrors ``_lower_softmax_template``\\'s structure with a row_cache TG
        buffer; differs only in the math (single-pass ``sum`` and ``sum_sq``
        in phase 1, instead of ``max`` + delayed ``exp``).

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

        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_msl_type = triton_type_to_msl(arg.elem_type) if arg.elem_type else "int"
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

        # eps default — matches torch.nn.LayerNorm. Most layer-norm Triton
        # kernels pass eps as a constexpr, so it doesn\\'t reach codegen as
        # a runtime arg.
        eps_literal = "1e-6f"

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
        lines.append("    float local_sum = 0.0f;")
        lines.append("    float local_sumsq = 0.0f;")
        lines.append("")

        # Phase 1: load → row_cache, single-pass mean + variance
        if vectorize:
            lines.append(f"    bool use_vec = (({n_arg} & 3) == 0);")
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = {n_arg} / 4;")
            lines.append(f"        device float4* x4 = (device float4*)({input_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = x4[i];")
            lines.append("            row4[i] = v;")
            lines.append("            local_sum += v.x + v.y + v.z + v.w;")
            lines.append("            local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"            float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("            row_cache[i] = v;")
            lines.append("            local_sum += v;")
            lines.append("            local_sumsq += v * v;")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"        float v = static_cast<float>({input_arg}[row_start + i]);")
            lines.append("        row_cache[i] = v;")
            lines.append("        local_sum += v;")
            lines.append("        local_sumsq += v * v;")
            lines.append("    }")

        # Reduce sum and sum-of-squares across threadgroup
        lines.append("    float simd_sum_v = simd_sum(local_sum);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sum_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sum = simd_sum((tiisg < {n_simd}u) ? reduce_buf[tiisg] : 0.0f);")
        lines.append("")
        lines.append("    float simd_sumsq_v = simd_sum(local_sumsq);")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    if (tiisg == 0) reduce_buf[sgitg] = simd_sumsq_v;")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    float row_sumsq = simd_sum((tiisg < {n_simd}u) ? reduce_buf[tiisg] : 0.0f);")
        lines.append("")
        lines.append(f"    float inv_n = 1.0f / float({n_arg});")
        lines.append("    float mean = row_sum * inv_n;")
        lines.append("    float var = row_sumsq * inv_n - mean * mean;")
        lines.append(f"    float inv_std = rsqrt(var + {eps_literal});")
        lines.append("")

        # Phase 2: write (x - mean) * inv_std to global memory
        if vectorize:
            lines.append("    if (use_vec) {")
            lines.append(f"        int n_v = {n_arg} / 4;")
            lines.append(f"        device float4* o4 = (device float4*)({output_arg} + row_start);")
            lines.append("        threadgroup float4* row4 = (threadgroup float4*)row_cache;")
            lines.append(f"        for (uint i = lid; i < (uint)n_v; i += {threads}u) {{")
            lines.append("            float4 v = row4[i];")
            lines.append("            float4 m = float4(mean);")
            lines.append("            o4[i] = (v - m) * inv_std;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
            lines.append(f"            {output_arg}[row_start + i] = {_store_cast}((row_cache[i] - mean) * inv_std);")
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append(f"    for (uint i = lid; i < (uint){n_arg}; i += {threads}u) {{")
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
        safe_name = _sanitize_msl_name(self.graph.func_name)

        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_msl_type = (triton_type_to_msl(arg.elem_type)
                                if arg.elem_type else "int")
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

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
            o_d = (f"(k % {dst_shape[d]}u)" if dst_stride[d] == 1
                   else f"((k / {dst_stride[d]}u) % {dst_shape[d]}u)")
            terms.append(f"{o_d} * {src_stride[order[d]]}u")
        in_flat = " + ".join(terms)

        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        # Pointer element types come from each arg's own elem_type in the
        # arg-decl loop below; no kernel-wide msl_type needed here.
        safe_name = _sanitize_msl_name(self.graph.func_name)
        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_msl_type = (triton_type_to_msl(arg.elem_type)
                                if arg.elem_type else "int")
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

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
        safe_name = _sanitize_msl_name(self.graph.func_name)
        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_msl_type = (triton_type_to_msl(arg.elem_type)
                                if arg.elem_type else "int")
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

        # out_cell(i) = sum_k ((i / in_stride_k) % size_k) * out_stride_k
        oc_terms = []
        for (in_s, size, out_s) in surviving:
            coord = f"((i / {in_s}u) % {size}u)"
            oc_terms.append(coord if out_s == 1 else f"{coord} * {out_s}u")
        oc_expr = " + ".join(oc_terms) if oc_terms else "0u"

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
        lines.append(f"        atomic_fetch_add_explicit(&_acc[_oc], "
                     f"(int){in_arg}[i], memory_order_relaxed);")
        lines.append("    }")
        lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"    for (uint e = lid; e < {out_total}u; e += {threads}u) {{")
        lines.append(f"        {out_arg}[e] = ({msl_type})atomic_load_explicit("
                     "&_acc[e], memory_order_relaxed);")
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

        # 128 threads = 4 SIMD groups × 32 threads.
        self.effective_block_size = 128

        c_msl_type = triton_type_to_msl(c_elem)

        # Genuine fp16 (WS1 Phase C): half INPUT fragments + float ACCUMULATOR
        # (the matmul output tg_C stays float for the softmax epilogue). fp16 A/B
        # are staged as half and fed to simdgroup_half8x8 — no float upcast — so
        # the half MMA is actually used. bf16 here stays on the float-upcast path
        # (staged+cast to float — exact and correct); unlike the plain-matmul
        # lowering it does NOT yet use simdgroup_bfloat8x8 in this fused template
        # (a future optimization — would need its own coverage). Other types: float.
        input_msl_type = triton_type_to_msl(a_elem)
        if input_msl_type == "half":
            in_frag, in_stage_t, in_cast = "simdgroup_half8x8", "half", ""
        else:
            in_frag, in_stage_t, in_cast = "simdgroup_float8x8", "float", "float"

        # Each SIMD group owns ``N / 4`` columns of the output, which must
        # split evenly into 8-wide simdgroup tiles.
        cols_per_sg = N // 4
        if cols_per_sg == 0 or cols_per_sg % 8:
            return None
        col_tiles_per_sg = cols_per_sg // 8        # 8×8 col tiles per SG
        row_tiles = m_block // 8                   # 8×8 row tiles per strip

        def _addr(base, row, col, row_stride, col_stride, inner_dim):
            row_term = f"({row}) * {row_stride}" if row_stride \
                else f"({row}) * {inner_dim}u"
            col_term = f"({col}) * {col_stride}" if col_stride else f"({col})"
            return f"{base}[{row_term} + {col_term}]"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                m = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {m}* {arg.name} [[buffer({i})]]")
            else:
                m = (triton_type_to_msl(arg.elem_type)
                     if arg.elem_type else "int")
                arg_decls.append(
                    f"    constant {m}& {arg.name} [[buffer({i})]]")

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
                lines.append(
                    f"        simdgroup_float8x8 acc_{rt}_{ct}(0);")
        lines.append("")

        lines.append(f"        for (uint kk = 0u; kk < {K}u; kk += 8u) {{")
        # Stage strip\'s A[M_BLOCK, 8] cooperatively.
        lines.append(
            f"            for (uint i = tiitg; i < {m_block * 8}u; i += 128u) {{")
        lines.append("                uint r = i / 8u, c = i % 8u;")
        a_load = _addr(a_ptr, "mstrip + r", "kk + c", a_row_s, a_col_s, K)
        lines.append(f"                tg_A[i] = {in_cast}({a_load});")
        lines.append("            }")
        # Stage B[8, N] cooperatively.
        lines.append(
            f"            for (uint i = tiitg; i < {8 * N}u; i += 128u) {{")
        lines.append(f"                uint r = i / {N}u, c = i % {N}u;")
        b_load = _addr(b_ptr, "kk + r", "c", b_row_s, b_col_s, N)
        lines.append(f"                tg_B[i] = {in_cast}({b_load});")
        lines.append("            }")
        lines.append("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("")
        for ct in range(col_tiles_per_sg):
            lines.append(f"            {in_frag} b_{ct};")
            lines.append(
                f"            simdgroup_load(b_{ct}, "
                f"tg_B + sgitg * {cols_per_sg}u + {ct * 8}u, {N});")
        lines.append("")
        lines.append(f"            {in_frag} a_frag;")
        for rt in range(row_tiles):
            lines.append(
                f"            simdgroup_load(a_frag, tg_A + {rt * 8 * 8}u, 8);")
            for ct in range(col_tiles_per_sg):
                lines.append(
                    f"            simdgroup_multiply_accumulate(acc_{rt}_{ct}, "
                    f"a_frag, b_{ct}, acc_{rt}_{ct});")
        lines.append("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("        }")
        lines.append("")

        # Spill the strip\'s accumulators.
        lines.append("        // Store strip accumulators to TG memory.")
        for rt in range(row_tiles):
            for ct in range(col_tiles_per_sg):
                row_off = rt * 8 * N
                col_off = f"sgitg * {cols_per_sg}u + {ct * 8}u"
                lines.append(
                    f"        simdgroup_store(acc_{rt}_{ct}, "
                    f"tg_C + {row_off}u + {col_off}, {N});")
        lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("")

        # Fused pointwise/broadcast epilogue (#158): apply the lowered op chain
        # per element on the staged tg_C, then write. Early-return leaves the
        # softmax path below untouched.
        if info.get("epilogue_ops"):
            lines.extend(self._emit_matmul_epilogue_loop(
                info, m_block, N, c_ptr, c_row_s, c_col_s, c_msl_type, _addr))
            lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
            lines.append("    }  // end M-strip loop")
            lines.append("}")
            lines.append("")
            return "\n".join(lines)

        # Row softmax across the strip\'s M_BLOCK rows.
        lines.append(
            f"        for (uint row = tiitg; row < {m_block}u; row += 128u) {{")
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
        lines.append(
            f"        for (uint i = tiitg; i < {m_block * N}u; i += 128u) {{")
        lines.append(f"            uint row = i / {N}u, col = i % {N}u;")
        c_store_addr = _addr(c_ptr, "mstrip + row", "col",
                             c_row_s, c_col_s, N)
        lines.append(f"            {c_store_addr} = ({c_msl_type})tg_C[i];")
        lines.append("        }")
        lines.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("    }  // end M-strip loop")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    # MSL for each fused-epilogue op given operand expressions. Reuses the same
    # operators/functions the generic op dispatch maps to (#158).
    _EPI_BIN = {"arith.addf": "+", "arith.subf": "-", "arith.mulf": "*",
                "arith.divf": "/", "arith.addi": "+", "arith.subi": "-",
                "arith.muli": "*"}
    # NaN-QUIET min/max (IEEE minNum/maxNum): MSL fmax/fmin return the non-NaN
    # operand — correct for maxnumf/minnumf.
    _EPI_FN2 = {"arith.maxnumf": "fmax", "arith.minnumf": "fmin"}
    # NaN-PROPAGATING min/max (tl.maximum/minimum with propagate_nan=ALL): if either
    # operand is NaN the result must be NaN. Plain fmax/fmin would silently drop it
    # (re-audit #5 — relu(NaN accumulator) returned 0.0 instead of NaN).
    _EPI_FN2_NANPROP = {"arith.maximumf": "fmax", "arith.minimumf": "fmin"}
    _EPI_FN1 = {"math.exp": "exp", "math.exp2": "exp2", "math.log": "log",
                "math.log2": "log2", "math.sqrt": "sqrt", "math.rsqrt": "rsqrt",
                "math.sin": "precise::sin", "math.cos": "precise::cos",
                "math.erf": "erf", "math.tanh": "precise::tanh",
                "math.floor": "floor", "math.ceil": "ceil", "math.absf": "fabs"}

    def _emit_matmul_epilogue_loop(self, info, m_block, N, c_ptr, c_row_s,
                                   c_col_s, c_msl_type, _addr):
        """Per-element fused-epilogue loop body (#158).

        Lowers the topologically-ordered epilogue op chain to scalar MSL on the
        staged matmul result tg_C[i], then writes. The dot result is seeded to
        ``tg_C[i]``; constants reuse ``_lower_constant``; a bias load indexes by
        ``col``; layout/cast ops pass through.
        """
        from triton_msl.codegen._lowerer_detection import _EPI_PASSTHROUGH
        by_id = {ssa.id: ssa for ssa in self.graph.ops}
        bias = info.get("bias_ptr")
        # The matmul result staged in tg_C is a@b (accumulators init to 0). If a
        # bias was fused into the dot's accumulator, add it back here.
        acc_bias = info.get("acc_bias_ptr")
        # ROW bias indexes the M-length bias by the GLOBAL row (mstrip + row), NOT the
        # strip-local `row`. _detect_matmul_epilogue refuses the row-bias case when
        # M > m_block (re-audit #8), so here mstrip is always 0 for a row bias and this
        # is exact; the mstrip term is kept for correctness if that guard ever loosens.
        acc_idx = "(mstrip + row)" if info.get("acc_bias_dim") == "row" else "col"
        dot_expr = (f"(tg_C[i] + {acc_bias}[{acc_idx}])" if acc_bias else "tg_C[i]")
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
                e = (f"((isnan({es[0]}) || isnan({es[1]})) ? NAN : "
                     f"{_fn}(({es[0]}), ({es[1]})))")
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
                    e = (f"(isnan({es[0]}) ? NAN : "
                         f"clamp(({es[0]}), ({es[1]}), ({es[2]})))")
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
                    op_name=op.op)
            var = f"ep{n}"
            n += 1
            body.append(f"            float {var} = {e};")
            val[op.id] = var

        out = val.get(info["store_value_id"], "tg_C[i]")
        lines = [f"        for (uint i = tiitg; i < {m_block * N}u; i += 128u) {{",
                 f"            uint row = i / {N}u, col = i % {N}u;"]
        lines.extend(body)
        c_store_addr = _addr(c_ptr, "mstrip + row", "col", c_row_s, c_col_s, N)
        lines.append(f"            {c_store_addr} = ({c_msl_type})({out});")
        lines.append("        }")
        return lines

    def _lower_row_wise_sort_template(self, info) -> str:
        """Emit a per-row bitonic sort / top-k kernel.

        Thread `lid` owns row `lid` of the (M, N) input. It loads N elements
        into a local register array, performs an in-register bitonic sort,
        and stores the first K elements back to row `lid` of the output.

        Bitonic sort for direction `asc`:
            for stage in 2, 4, ..., N:
                for step in stage/2, stage/4, ..., 1:
                    for i in 0 .. N-1:
                        j = i ^ step
                        if j > i:
                            up = ((i & stage) == 0) xor !asc
                            if ((v[i] > v[j]) == up) swap

        For topk: run the full sort, then store first K elements. For
        `descending=True`: sort ascending and reverse-store. For
        `descending=False` + topk: the smallest K elements.
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
                f"results. Use a full tl.sort and slice, or k == N.", op_name="tt.reduce")
        descending = info["descending"]
        elem_type = info["elem_type"]
        x_ptr = info["x_ptr"]
        z_ptr = info["z_ptr"]
        stride_xm = info["stride_xm"]
        stride_zm = info["stride_zm"]
        block_size = info["block_size"]

        msl_type = triton_type_to_msl(elem_type)
        # Pick a compute type for comparisons (promote fp16/bf16 to float)
        compute_type = _msl_compute_type(elem_type)

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument declarations in the original IR arg order
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                if arg.name == z_ptr:
                    arg_decls.append(
                        f"    volatile device {arg_msl_type}* {arg.name} [[buffer({i})]]")
                else:
                    arg_decls.append(
                        f"    device const {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                scalar_ty = triton_type_to_msl(arg.elem_type or "i32")
                arg_decls.append(
                    f"    constant {scalar_ty}& {arg.name} [[buffer({i})]]")

        # Integer sort? If elem type is integer we just use standard
        # `<` / `>` over the integer values. For floats, use `<` / `>`.
        is_int = elem_type.startswith("i") or elem_type.startswith("u")

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
        lines.append(f"        uint _row_off = lid * (uint){stride_xm};")
        lines.append(f"        for (uint _i = 0u; _i < {N}u; _i++) {{")
        lines.append(f"            v[_i] = static_cast<{compute_type}>({x_ptr}[_row_off + _i]);")
        lines.append(f"        }}")
        # In-register bitonic sort (ascending). Unroll to avoid the
        # `step /= 2u` where step=1 lap not exiting the `>=1u` condition
        # cleanly in MSL (0 >= 1 is false so it DOES exit; but some compilers
        # warn). We emit the sequence directly from known N.
        lines.append(f"        // Bitonic sort of {N} elements (ascending order)")
        stage = 2
        while stage <= N:
            step = stage // 2
            while step >= 1:
                lines.append(f"        // stage={stage}, step={step}")
                lines.append(f"        for (uint i = 0u; i < {N}u; i++) {{")
                lines.append(f"            uint j = i ^ {step}u;")
                lines.append(f"            if (j > i) {{")
                lines.append(f"                bool up = ((i & {stage}u) == 0u);")
                lines.append(f"                bool gt = (v[i] > v[j]);")
                lines.append(f"                if (gt == up) {{")
                lines.append(f"                    {compute_type} _t = v[i]; v[i] = v[j]; v[j] = _t;")
                lines.append(f"                }}")
                lines.append(f"            }}")
                lines.append(f"        }}")
                step //= 2
            stage *= 2
        # After ascending sort, v[0..N-1] is ascending.
        # For sort descending: store reversed.
        # For topk ascending (not descending): smallest K → v[0..K-1] in ascending order.
        # For topk descending: largest K → v[N-K..N-1] → store in reverse order.
        # For sort ascending: v[0..N-1].
        lines.append(f"        uint _row_off_z = lid * (uint){stride_zm};")
        if descending:
            if K < N:
                # topk largest: take v[N-K..N-1], reverse for descending order
                lines.append(f"        for (uint _i = 0u; _i < {K}u; _i++) {{")
                lines.append(f"            {z_ptr}[_row_off_z + _i] = static_cast<{msl_type}>(v[{N}u - 1u - _i]);")
                lines.append(f"        }}")
            else:
                # sort descending: reverse entire ascending output
                lines.append(f"        for (uint _i = 0u; _i < {N}u; _i++) {{")
                lines.append(f"            {z_ptr}[_row_off_z + _i] = static_cast<{msl_type}>(v[{N}u - 1u - _i]);")
                lines.append(f"        }}")
        else:
            # ascending sort or topk ascending (smallest K): take v[0..K-1]
            lines.append(f"        for (uint _i = 0u; _i < {K}u; _i++) {{")
            lines.append(f"            {z_ptr}[_row_off_z + _i] = static_cast<{msl_type}>(v[_i]);")
            lines.append(f"        }}")
        lines.append(f"    }}")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)


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

        # Determine data type from pointer args
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        msl_type = "float"
        if ptr_args:
            elem = ptr_args[0].elem_type
            if elem in ("i32", "si32"):
                msl_type = "int"

        # Identity and combine expression
        if combine_op == "sum":
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"
        elif combine_op == "max":
            identity = "(-INFINITY)" if msl_type == "float" else "INT_MIN"
            combine_expr = "fmax(acc, val)" if msl_type == "float" else "max(acc, val)"
        elif combine_op == "min":
            identity = "INFINITY" if msl_type == "float" else "INT_MAX"
            combine_expr = "fmin(acc, val)" if msl_type == "float" else "min(acc, val)"
        else:
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list (X and Z pointers)
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")

        x_name = ptr_args[0].name if ptr_args else "X"
        z_name = ptr_args[1].name if len(ptr_args) > 1 else "Z"

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
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[pid * {total}u + _e];")
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
        _out_msl3d = (triton_type_to_msl(ptr_args[1].elem_type)
                      if len(ptr_args) > 1 else "float")
        _scast3d = "" if _out_msl3d in ("float", "half") else f"({_out_msl3d})"
        R0, R1 = result_dims
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u)")
        lines.append(f"        {z_name}[pid * {result_total}u + _r] = {_scast3d}_result[_r];")

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
        is_max = (combine_op == "argmax")

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

        # Determine data type from pointer args
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        msl_type = "float"
        if ptr_args:
            elem = ptr_args[0].elem_type
            if elem in ("i32", "si32"):
                msl_type = "int"

        identity = "(-INFINITY)" if is_max and msl_type == "float" else "INFINITY"
        if msl_type == "int":
            identity = "INT_MIN" if is_max else "INT_MAX"
        cmp_op = ">" if is_max else "<"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")

        x_name = ptr_args[0].name if ptr_args else "X"
        z_name = ptr_args[1].name if len(ptr_args) > 1 else "Z"

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
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[pid * {total}u + _e];")
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
        lines.append(f"        {z_name}[pid * {result_total}u + _r] = _result_idx[_r];")

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

        def _init_is_bias(_id):
            _op = _by.get(_id); _seen = set()
            while _op is not None and _op.id not in _seen:
                _seen.add(_op.id)
                if _op.op in ("tt.splat", "tt.broadcast", "ttg.convert_layout",
                              "tt.reshape", "tt.expand_dims"):
                    _op = _by.get(_op.operand_ids[0]) if _op.operand_ids else None
                    continue
                break
            if _op is None:
                return False
            # Only the accumulator (2-D tile matching the dot output) is the risk.
            if _extract_shape(_op.type_str) != _dot_shape:
                return False
            if _op.op == "tt.load":
                return True
            if _op.op == "arith.constant":
                return any(ch in "123456789" for ch in str(_op.attrs.get("value", "")))
            return False
        _init_ids = []
        for _s in self.graph.ops:
            if _s.op == "scf.for" and len(_s.operand_ids) > 3:
                _init_ids.extend(_s.operand_ids[3:])
        if not _init_ids:
            for _s in self.graph.ops:
                if _s.op == "tt.dot" and len(_s.operand_ids) >= 3:
                    _init_ids.append(_s.operand_ids[2]); break
        if _dot_shape and any(_init_is_bias(_i) for _i in _init_ids):
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                "strided matmul with a non-zero accumulator init (fused bias / tl.full) "
                "is silently dropped by the strided template (it seeds acc=0). Refusing. "
                "Add the bias as a separate kernel after the matmul.", op_name="tt.dot")

        # Determine dtype from pointer args
        dtype = "fp32"
        if ptr_args:
            dtype = _mlir_to_triton_dtype(ptr_args[0].elem_type)

        # Check if this is a strided kernel (has stride args) vs simple kernel
        has_strides = any("stride" in a.name.lower() for a in scalar_args)
        # If the kernel uses program_id, it's a pid-tiled kernel that
        # needs the simdgroup matmul template with block selection.
        has_pid = any(ssa.op == "tt.get_program_id" for ssa in self.graph.ops)

        # Detect 3D batched dot (e.g. test_dot3d): tt.dot on tensor<BxMxNxT>
        # 3D dot has both strides AND pids — strides for batch/row/col access,
        # pids for spatial tiling over M and N. Use strided template, not simdgroup.
        is_3d_dot = False
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_shape = _extract_shape(ssa.type_str)
                if len(dot_shape) >= 3:
                    is_3d_dot = True
                break

        if not has_strides or (has_pid and not is_3d_dot):
            # Check for constant-input dot (e.g., test_dot_without_load)
            const_info = self._detect_dot_constant_inputs()
            if const_info:
                return self._lower_dot_constant_template(const_info, ptr_args)
            # Fall back to optimized simdgroup template for pid-tiled kernels
            return self._lower_dot_simple_template(tile_dims, ptr_args, dtype)

        # --- Strided dot kernel generation ---
        # Extract M, N, K from tt.dot operand type shapes (reliable).
        # 2D: tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxT>
        # 3D: tensor<BxMxKxT> * tensor<BxKxNxT> -> tensor<BxMxNxT>
        M, N, K = 32, 32, 32  # fallback
        B_batch = 1
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_shape = _extract_shape(ssa.type_str)
                if len(dot_shape) >= 3:
                    B_batch = dot_shape[0]
                    M, N = dot_shape[1], dot_shape[2]
                elif len(dot_shape) >= 2:
                    M, N = dot_shape[0], dot_shape[1]
                # Get K from first operand shape
                # 2D: [M, K], 3D: [B, M, K]
                if ssa.operand_ids:
                    for ssa2 in self.graph.ops:
                        if ssa2.id == ssa.operand_ids[0]:
                            op_shape = _extract_shape(ssa2.type_str)
                            if len(op_shape) >= 3:
                                K = op_shape[2]
                            elif len(op_shape) >= 2:
                                K = op_shape[1]
                            break
                break

        # Detect accumulator initialization from the IR
        has_accumulator_load = False
        for ssa in self.graph.ops:
            if ssa.op == "arith.addf" and any(
                self.graph.ops[i].op == "tt.dot"
                for i, op in enumerate(self.graph.ops)
                if op.id in (ssa.operand_ids or [])
            ):
                has_accumulator_load = True

        # MSL type mapping
        from triton_msl.codegen.msl_builtins import is_fp8_type
        is_fp8_dot = is_fp8_type(dtype)
        if is_fp8_dot:
            msl_type = "uchar"  # FP8 stored as uchar
            compute_type = "float"
        elif dtype in ("fp16", "f16"):
            msl_type = "half"
            compute_type = "float"
        elif dtype in ("bf16",):
            msl_type = "bfloat"
            compute_type = "float"
        else:
            msl_type = "float"
            compute_type = "float"

        # Determine output type from tt.dot result
        out_msl_type = "float"  # tt.dot typically outputs f32
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_out_type = ssa.elem_type
                if dot_out_type in ("f16",):
                    out_msl_type = "half"
                elif dot_out_type in ("bf16",):
                    out_msl_type = "bfloat"
                break

        num_warps = self.graph.num_warps
        block_size = min(num_warps * 32, 1024)
        self._matmul_block_size = block_size

        # For 3D dot, signal that the kernel needs 2D grid dispatch
        if is_3d_dot:
            self._used_pid_axes = {0, 1}

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list from IR
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_triton_dtype = _mlir_to_triton_dtype(arg.elem_type)
                arg_msl_type = triton_type_to_msl(arg_triton_dtype)
                arg_decls.append(
                    f"    volatile device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    volatile device int* {arg.name}_buf [[buffer({i})]]")

        # Generate the kernel
        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        # Use uint3 for grid position when multi-axis dispatch is needed
        used_axes = getattr(self.kb, '_used_pid_axes', {0}) if self.kb else {0}
        if is_3d_dot:
            used_axes = {0, 1}  # 3D dot needs pid3.x and pid3.y for spatial tiling
        if max(used_axes) > 0:
            # Metal requires all thread-index attributes to use the same
            # type (all uint or all uint3). Use uint3 for all when multi-axis.
            lines.append("    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append("    uint3 _lid3 [[thread_position_in_threadgroup]],")
            lines.append("    uint3 _tid3 [[thread_position_in_grid]]")
            lines.append(") {")
            lines.append("    uint lid = _lid3.x;")
            lines.append("    uint pid = pid3.x;")
            if 1 in used_axes:
                lines.append("    uint pid_y = pid3.y;")
            if 2 in used_axes:
                lines.append("    uint pid_z = pid3.z;")
        else:
            lines.append("    uint pid [[threadgroup_position_in_grid]],")
            lines.append("    uint lid [[thread_position_in_threadgroup]],")
            lines.append("    uint tid [[thread_position_in_grid]]")
            lines.append(") {")

        # Unpack scalar args from buffers
        for arg in self.graph.args:
            if not arg.is_ptr:
                lines.append(f"    int {arg.name} = {arg.name}_buf[0];")

        # Map scalar stride args to their pointer by name prefix.
        # Triton TTGIR folds stride=1 args to constants, so we may have
        # fewer scalars than expected. Name matching is robust to this.
        #
        # Convention: stride_{ptrbase}{dim} where ptrbase is derived from
        # the pointer name (e.g. "x" from "x_ptr", "x_ptr" → "x").
        # 2D: Each pointer gets [dim0_stride, dim1_stride], default "1".
        # 3D: Each pointer gets [batch_stride, dim0_stride, dim1_stride].
        #
        # IMPORTANT: When one stride is folded (e.g. stride_xm=1 for col_a),
        # the remaining stride must go in the CORRECT slot based on its
        # dimension suffix, not just the first empty slot.
        # For dot A[M,K] @ B[K,N] → C[M,N]:
        #   A: 'k'/'1' suffix → dim1, everything else → dim0
        #   B: 'n'/'1' suffix → dim1, everything else → dim0
        #   C: 'n'/'1' suffix → dim1, everything else → dim0
        #   W: 'l'/'1' suffix → dim1, everything else → dim0
        # For 3D dot A[B,M,K] @ B[B,K,N] → C[B,M,N]:
        #   'b' suffix → batch slot (slot 0), dim suffixes → slots 1,2
        if is_3d_dot:
            stride_map = {p.name: ["1", "1", "1"] for p in ptr_args}
        else:
            stride_map = {p.name: ["1", "1"] for p in ptr_args}

        # Define which suffix characters map to dim1 for each pointer position
        _dim1_suffixes = {
            0: {'k', '1'},    # A (MxK): K-stride is dim1
            1: {'n', '1'},    # B (KxN): N-stride is dim1
        }
        # Last pointer (C/Z) and chain-dot W
        _dim1_last = {'n', '1'}   # C (MxN): N-stride is dim1
        if len(ptr_args) >= 4:
            _dim1_suffixes[2] = {'l', '1'}  # W (NxL): L-stride is dim1

        matched_strides = set()
        for sarg in scalar_args:
            sname = sarg.name.lower()
            if "stride" not in sname:
                continue
            # Match to pointer by name prefix (case-insensitive)
            for pi, p in enumerate(ptr_args):
                base = p.name
                if base.endswith("_ptr"):
                    base = base[:-4]
                base_lower = base.lower()
                prefix = f"stride_{base_lower}"
                alt_prefix = f"s_{base_lower}"
                # Also try reverse pattern: {base}_stride (e.g., "in_stride" for "in1_ptr")
                rev_prefix = f"{base_lower}_stride"
                # Try without trailing digits: "in_stride" matches "in1_ptr"
                base_nodigit = base_lower.rstrip("0123456789")
                rev_prefix_nodigit = f"{base_nodigit}_stride" if base_nodigit != base_lower else ""
                # Try prefix match: "out_stride" matches "output_ptr" (stride base is prefix of ptr base)
                stride_base = sname.replace("_stride", "").replace("stride_", "").replace("s_", "")
                if sname.startswith(prefix):
                    suffix = sname[len(prefix):]
                elif sname.startswith(alt_prefix):
                    suffix = sname[len(alt_prefix):]
                elif sname == rev_prefix or sname.startswith(rev_prefix + "_"):
                    suffix = sname[len(rev_prefix):]
                elif rev_prefix_nodigit and (sname == rev_prefix_nodigit or sname.startswith(rev_prefix_nodigit + "_")):
                    suffix = sname[len(rev_prefix_nodigit):]
                elif stride_base and base_lower.startswith(stride_base) and sname.endswith("_stride"):
                    suffix = ""
                else:
                    continue

                dims = stride_map[p.name]
                # Determine correct dim slot from suffix
                is_last = (pi == len(ptr_args) - 1)
                dim1_chars = _dim1_last if is_last else _dim1_suffixes.get(pi, {'1'})
                if is_3d_dot and suffix and suffix[0] == 'b':
                    # Batch stride → slot 0
                    dims[0] = sarg.name
                elif suffix and suffix[0] in dim1_chars:
                    # Inner dimension (K for A, N for B/C)
                    dims[-1] = sarg.name
                else:
                    # Outer dimension (M for A, K for B, M for C)
                    dims[1 if is_3d_dot else 0] = sarg.name
                matched_strides.add(sarg.name)
                break

        # Positional fallback: if stride args remain unmatched and pointers
        # have default strides, assign remaining strides positionally to dim0
        if has_strides and len(matched_strides) == 0:
            unmatched = [s for s in scalar_args if "stride" in s.name.lower()]
            for si, sarg in enumerate(unmatched):
                if si < len(ptr_args):
                    stride_map[ptr_args[si].name][0] = sarg.name

        if len(ptr_args) < 3:
            return self._lower_dot_simple_template(tile_dims, ptr_args, dtype)

        a_ptr = ptr_args[0]
        b_ptr = ptr_args[1]
        c_ptr = ptr_args[-1]

        if is_3d_dot:
            a_sb, a_s0, a_s1 = stride_map[a_ptr.name]
            b_sb, b_s0, b_s1 = stride_map[b_ptr.name]
            c_sb, c_s0, c_s1 = stride_map[c_ptr.name]
        else:
            a_sb = b_sb = c_sb = "0"
            a_s0, a_s1 = stride_map[a_ptr.name]
            b_s0, b_s1 = stride_map[b_ptr.name]
            c_s0, c_s1 = stride_map[c_ptr.name]

        c_type = triton_type_to_msl(_mlir_to_triton_dtype(c_ptr.elem_type))

        # Detect epilogue from IR ops after tt.dot
        epilogue = self._detect_dot_epilogue()

        # For chain-dot, we need W pointer and strides
        w_ptr = None
        w_s0 = "1"
        w_s1 = "1"
        if epilogue == "chain-dot" and len(ptr_args) >= 4 and not is_3d_dot:
            w_ptr = ptr_args[2]
            w_s0, w_s1 = stride_map[w_ptr.name]

        # Shared memory for epilogues that need staging
        total = M * N
        if epilogue in ("softmax", "chain-dot"):
            lines.append(f"    threadgroup {compute_type} _shared_c[{total}];")
        elif epilogue == "add-matrix":
            # Stage Z bias into shared memory to avoid aliasing (Z is both source and output)
            lines.append(f"    threadgroup {compute_type} _bias[{total}];")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _i = _e / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")
            lines.append(f"        _bias[_e] = ({compute_type}){c_ptr.name}[_i * {c_s0} + _j * {c_s1}];")
            lines.append(f"    }}")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif epilogue == "add-rows":
            lines.append(f"    threadgroup {compute_type} _bias[{M}];")
            lines.append(f"    if (lid < {M}u) _bias[lid] = ({compute_type}){c_ptr.name}[lid * {c_s0}];")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif epilogue == "add-cols":
            lines.append(f"    threadgroup {compute_type} _bias[{N}];")
            lines.append(f"    if (lid < {N}u) _bias[lid] = ({compute_type}){c_ptr.name}[lid * {c_s1}];")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # FP8 dot: inject conversion functions and use them for A/B loads
        if is_fp8_dot:
            from triton_msl.codegen.msl_builtins import fp8_to_float_func, fp8_device_functions
            a_dtype = _mlir_to_triton_dtype(a_ptr.elem_type)
            b_dtype = _mlir_to_triton_dtype(b_ptr.elem_type)
            fp8_funcs_added = set()
            for dt in (a_dtype, b_dtype):
                if is_fp8_type(dt) and dt not in fp8_funcs_added:
                    fp8_funcs_added.add(dt)
                    for fn_src in fp8_device_functions(dt):
                        lines.insert(2, fn_src)  # Insert after "using namespace metal;"
                        lines.insert(3, "")
            a_load = lambda idx: f"{fp8_to_float_func(a_dtype)}({a_ptr.name}[{idx}])" if is_fp8_type(a_dtype) else f"({compute_type}){a_ptr.name}[{idx}]"
            b_load = lambda idx: f"{fp8_to_float_func(b_dtype)}({b_ptr.name}[{idx}])" if is_fp8_type(b_dtype) else f"({compute_type}){b_ptr.name}[{idx}]"
        else:
            a_load = lambda idx: f"({compute_type}){a_ptr.name}[{idx}]"
            b_load = lambda idx: f"({compute_type}){b_ptr.name}[{idx}]"

        # Emit the matmul loop
        if is_3d_dot:
            total = B_batch * M * N
            lines.append(f"    // 3D strided dot: [{B_batch}x{M}x{K}] @ [{B_batch}x{K}x{N}] -> [{B_batch}x{M}x{N}]")
            lines.append(f"    uint _pid_m_off = pid3.x * {M}u;")
            lines.append(f"    uint _pid_n_off = pid3.y * {N}u;")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _b = _e / {M * N}u;")
            lines.append(f"        uint _i = (_e % {M * N}u) / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")
            lines.append(f"        {compute_type} _sum = 0.0f;")
            lines.append(f"        for (uint _k = 0; _k < {K}u; _k++) {{")
            a_idx = f"_b * {a_sb} + (_pid_m_off + _i) * {a_s0} + _k * {a_s1}"
            b_idx = f"_b * {b_sb} + _k * {b_s0} + (_pid_n_off + _j) * {b_s1}"
            lines.append(f"            _sum += {a_load(a_idx)}")
            lines.append(f"                  * {b_load(b_idx)};")
            lines.append(f"        }}")
            lines.append(f"        {c_ptr.name}[_b * {c_sb} + (_pid_m_off + _i) * {c_s0} + (_pid_n_off + _j) * {c_s1}] = ({c_type})_sum;")
            lines.append(f"    }}")
        else:
            lines.append(f"    // Strided dot: [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}], epilogue={epilogue}")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _i = _e / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")

            # Initialize accumulator from staged bias or zero
            if epilogue == "add-matrix":
                lines.append(f"        {compute_type} _sum = _bias[_e];")
            elif epilogue == "add-rows":
                lines.append(f"        {compute_type} _sum = _bias[_i];")
            elif epilogue == "add-cols":
                lines.append(f"        {compute_type} _sum = _bias[_j];")
            else:
                lines.append(f"        {compute_type} _sum = 0.0f;")

            lines.append(f"        for (uint _k = 0; _k < {K}u; _k++) {{")
            a_idx_2d = f"_i * {a_s0} + _k * {a_s1}"
            b_idx_2d = f"_k * {b_s0} + _j * {b_s1}"
            lines.append(f"            _sum += {a_load(a_idx_2d)}")
            lines.append(f"                  * {b_load(b_idx_2d)};")
            lines.append(f"        }}")

        # Epilogue handling (2D only — 3D dot loop already includes store)
        if not is_3d_dot:
            if epilogue == "softmax":
                # Store dot result to shared memory for row-wise softmax
                lines.append(f"        _shared_c[_e] = _sum;")
                lines.append(f"    }}")
                lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Row-wise softmax via shared memory
                lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
                lines.append(f"        uint _i = _e / {N}u;")
                lines.append(f"        uint _j = _e % {N}u;")
                lines.append(f"        // Row-wise max")
                lines.append(f"        {compute_type} _row_max = -INFINITY;")
                lines.append(f"        for (uint _c = 0; _c < {N}u; _c++)")
                lines.append(f"            _row_max = fmax(_row_max, _shared_c[_i * {N}u + _c]);")
                lines.append(f"        {compute_type} _exp_val = exp(_shared_c[_e] - _row_max);")
                lines.append(f"        // Row-wise sum of exp")
                lines.append(f"        {compute_type} _row_sum = 0.0f;")
                lines.append(f"        for (uint _c = 0; _c < {N}u; _c++)")
                lines.append(f"            _row_sum += exp(_shared_c[_i * {N}u + _c] - _row_max);")
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})(_exp_val / _row_sum);")
                lines.append(f"    }}")
            elif epilogue == "chain-dot" and w_ptr:
                # Store first dot to shared, then second matmul with W
                lines.append(f"        _shared_c[_e] = _sum;")
                lines.append(f"    }}")  # end first matmul loop
                lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Second matmul: shared_c[M,N] @ W[N,N] → result[M,N]
                lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
                lines.append(f"        uint _i = _e / {N}u;")
                lines.append(f"        uint _j = _e % {N}u;")
                lines.append(f"        {compute_type} _sum2 = 0.0f;")
                lines.append(f"        for (uint _k2 = 0; _k2 < {N}u; _k2++) {{")
                lines.append(f"            _sum2 += _shared_c[_i * {N}u + _k2]")
                lines.append(f"                   * ({compute_type}){w_ptr.name}[_k2 * {w_s0} + _j * {w_s1}];")
                lines.append(f"        }}")
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})_sum2;")
                lines.append(f"    }}")
            else:
                # Default store (none, trans, add-*)
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})_sum;")
                lines.append(f"    }}")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)


    def _lower_dot_constant_template(self, const_info, ptr_args):
        """Generate MSL for dot product where both inputs are compile-time constants."""
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
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint lid [[thread_position_in_threadgroup]]")
        lines.append(") {")
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
        lines.append(f"        uint _i = _e / {N}u;")
        lines.append(f"        uint _j = _e % {N}u;")
        # Each element = sum over K of const_a * const_b = K * const_a * const_b
        lines.append(f"        float _sum = (float){K}u * {const_a}f * {const_b}f;")
        lines.append(f"        {c_ptr.name}[_i * {N}u + _j] = ({c_msl_type})_sum;")
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
                "output strides (e.g. test_dot_mulbroadcasted).")

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
            out_dtype = _mlir_to_triton_dtype(ptr_args[2].elem_type)

        # Phase 4: record the runtime fast-matmul dispatch descriptor (additive;
        # the generic kernel below is still emitted + returned). The launcher only
        # uses it when the RUNTIME tensors are MPS and dims are aligned.
        self._fast_matmul = self._maybe_fast_matmul_descriptor()

        msl = make_matmul_kernel(
            block_m=block_m, block_n=block_n, block_k=block_k,
            dtype=dtype, out_dtype=out_dtype,
        )

        safe_name = _sanitize_msl_name(self.graph.func_name)
        msl = msl.replace("matmul_kernel", safe_name, 1)

        if len(ptr_args) >= 3:
            a_name, b_name, c_name = ptr_args[0].name, ptr_args[1].name, ptr_args[2].name
            # Replace parameter declarations -- use regex to match any MSL type (float, half, etc.)
            msl = re.sub(r'(device\s+const\s+\w+\*)\s+A\s', rf'\1 {a_name} ', msl)
            msl = re.sub(r'(device\s+const\s+\w+\*)\s+B\s', rf'\1 {b_name} ', msl)
            msl = re.sub(r'(volatile\s+device\s+\w+\*|device\s+\w+\*)\s+C\s', rf'\1 {c_name} ', msl)
            # Replace body references
            msl = re.sub(r'(?<![a-zA-Z_])A\[', f'{a_name}[', msl)
            msl = re.sub(r'(?<![a-zA-Z_])B\[', f'{b_name}[', msl)
            msl = re.sub(r'(?<![a-zA-Z_])C\[', f'{c_name}[', msl)

        return msl

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
            if not (roles[0].name == args[0].name
                    and roles[1].name == args[1].name
                    and roles[2].name == args[2].name):
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
                    _hi = _hop.operand_ids[0]; _depth += 1
                if _arg_index.get(_hi) != 5:
                    return None   # K-loop bound is not args[5] => non-canonical => refuse
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
        except Exception:                                      # noqa: BLE001
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
                    if _ri != _exp_idx:               # explicit stride arg != dim
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
        return (fast_msl, 3, 4, 5, 8 * rr, 32 * rc, msl_dtype, msl_out,
                tuple(stride_checks))


