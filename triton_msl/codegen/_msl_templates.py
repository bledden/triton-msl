"""Pre-baked MSL kernel templates.

Each ``make_*_kernel`` function builds a complete MSL kernel for a specific
op (vector_add, softmax, matmul, layer_norm, attention variants, etc.) and
returns a built kernel via ``KernelBuilder.build()``. They\'re used by:

  - ``ttgir_parser.py``: pattern-match a TTGIR kernel and substitute one of
    these prebuilt templates when applicable.
  - ``benchmarks/bench_all.py``: standalone benchmarks of representative
    kernels.

These were originally inlined in ``msl_emitter.py`` but at 65 functions and
~4.7 kLOC they dwarfed the actual emitter (KernelBuilder + MSLCodeGen +
emit_msl) by a 5:1 ratio. Splitting them out makes ``msl_emitter.py``
focused on the emit infrastructure and gives templates their own home.

Each function is independent. Cross-template dependencies (when one
template uses another) go through this same module.
"""

from triton_msl.codegen.msl_types import triton_type_to_msl
from triton_msl.codegen._fa_fragments import (
    row_loop as _fa_row_loop,
    scope_reason as _fa_fragment_scope,
    validate_mode as _fa_fragment_mode,
)
# NOTE: KernelBuilder, _msl_compute_type, _msl_zero, _sanitize_msl_name are
# imported from msl_emitter at the BOTTOM of this module, not here. msl_emitter
# re-exports this module via `from _msl_templates import *`, so importing it at
# the top creates a circular import: when _msl_templates is imported first, the
# star-import runs while this module is only partially initialized and silently
# drops make_matmul_kernel et al. Deferring our import to the end (after all
# defs, none of which run at import time) lets both orders resolve. See #152.


# ---------------------------------------------------------------------------
# High-level kernel generators
# ---------------------------------------------------------------------------


def make_elementwise_kernel(name, n_inputs, op, block_size=256, dtype="fp32"):
    """Generate an elementwise kernel: output[i] = op(input_0[i], ..., input_n[i]).

    Args:
        name: Kernel function name.
        n_inputs: Number of input buffers.
        op: Operation name ("add", "mul", "sub", "silu", "gelu", etc.)
        block_size: Elements per threadgroup.
        dtype: Data type.

    Returns:
        MSL source code string.
    """
    kb = KernelBuilder(name, block_size=block_size)

    # Register arguments
    input_names = []
    for i in range(n_inputs):
        input_names.append(kb.add_ptr_arg(f"input{i}", dtype=dtype, const=True))
    out_name = kb.add_ptr_arg("output", dtype=dtype, const=False)
    n_name = kb.add_scalar_arg("n_elements", dtype="u32")

    # Compute offsets and mask
    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, n_name, "mask")

    # Load inputs (promoted to float for FP16/BF16)
    val_names = []
    for i, inp in enumerate(input_names):
        val = kb.load(inp, offsets, mask, out_var=f"val{i}", dtype=dtype)
        val_names.append(val)

    # Apply operation (always in float compute precision)
    if n_inputs == 1:
        # Unary or fused unary
        if op in ("silu", "gelu", "gelu_tanh"):
            result = kb.fused_op(op, val_names, "result")
        else:
            result = kb.unary_op(op, val_names[0], "result")
    elif n_inputs == 2:
        result = kb.binary_op(op, val_names[0], val_names[1], "result")
    elif n_inputs == 3 and op == "fma":
        result = kb.fused_op("fma", val_names, "result")
    else:
        raise ValueError(f"Unsupported: {n_inputs} inputs with op '{op}'")

    # Store result (cast back to half for FP16/BF16)
    kb.store(out_name, offsets, result, mask, dtype=dtype)

    return kb.build()


def make_vector_add_kernel(block_size=256, dtype="fp32"):
    """Generate a vector add kernel: output = a + b."""
    return make_elementwise_kernel("vector_add", 2, "add", block_size, dtype)


def make_silu_kernel(block_size=256, dtype="fp32"):
    """Generate a SiLU activation kernel: output = x * sigmoid(x)."""
    return make_elementwise_kernel("silu_kernel", 1, "silu", block_size, dtype)


def make_gelu_kernel(block_size=256, dtype="fp32"):
    """Generate a GELU activation kernel."""
    return make_elementwise_kernel("gelu_kernel", 1, "gelu", block_size, dtype)


def make_swiglu_kernel(block_size=256, dtype="fp32"):
    """Generate a fused SwiGLU activation kernel.

    SwiGLU(x, gate) = SiLU(gate) * x = (gate / (1 + exp(-gate))) * x

    Used in LLaMA, Mistral, and Gemma FFN layers. Fuses the gate
    activation and element-wise multiply into one kernel for memory efficiency.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    kb = KernelBuilder("swiglu_kernel", block_size=block_size)

    kb.add_ptr_arg("x", dtype=dtype, const=True)
    kb.add_ptr_arg("gate", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    x_val = kb.load("x", offsets, mask, out_var="x_val", dtype=dtype)
    gate_val = kb.load("gate", offsets, mask, out_var="gate_val", dtype=dtype)

    # SiLU(gate) * x
    silu_gate = kb.fused_op("silu", [gate_val], "silu_gate")
    result = kb.binary_op("mul", silu_gate, x_val, "result")

    kb.store("output", offsets, result, mask, dtype=dtype)

    return kb.build()


def make_embedding_kernel(block_size=256, dtype="fp32"):
    """Generate an embedding lookup kernel.

    output[i, :] = table[indices[i], :]

    Each threadgroup handles one token (one row of the output).
    Threads within the group cooperatively copy the embedding vector.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        table: [vocab_size, embed_dim] embedding table
        indices: [batch_size] int32 token indices
        output: [batch_size, embed_dim] output
        embed_dim: embedding dimension
    """
    kb = KernelBuilder("embedding_kernel", block_size=block_size)

    kb.add_ptr_arg("table", dtype=dtype, const=True)
    kb.add_ptr_arg("indices", dtype="i32", const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("embed_dim", dtype="u32")

    # pid = token index in the batch
    kb._var("token_idx", "indices[pid]", ty="int")
    kb._var("src_offset", "uint(token_idx) * embed_dim", ty="uint")
    kb._var("dst_offset", "pid * embed_dim", ty="uint")

    # Each thread copies one or more elements of the embedding vector
    kb.raw_line(f"for (uint i = lid; i < embed_dim; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("output[dst_offset + i] = table[src_offset + i];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_scalar_mul_kernel(block_size=256, dtype="fp32"):
    """Generate a scalar multiply kernel: output = input * scalar.

    Note: scalar is passed as a separate buffer argument.
    """
    kb = KernelBuilder("scalar_mul", block_size=block_size)

    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("scalar", dtype="fp32")
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    val = kb.load("input", offsets, mask, out_var="val", dtype=dtype)
    result = kb.binary_op("mul", val, "scalar", "result")
    kb.store("output", offsets, result, mask, dtype=dtype)

    return kb.build()


# ---------------------------------------------------------------------------
# Reduction kernel generators
# ---------------------------------------------------------------------------


def make_reduce_kernel(name, op, block_size=256, dtype="fp32"):
    """Generate a 1D reduction kernel: output[group] = reduce(input[group*N:...]).

    Each threadgroup reduces block_size elements. For inputs larger than
    block_size, launch multiple threadgroups and reduce the partial results.

    Uses two-level reduction: SIMD intrinsics + threadgroup shared memory.

    Args:
        name: Kernel function name.
        op: Reduction operation ("sum", "max", "min").
        block_size: Elements per threadgroup.
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder(name, block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    # Shared memory for cross-SIMD-group reduction
    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    identity = {"sum": "(n_elements == 0u ? 0.0f : -0.0f)", "max": "-INFINITY", "min": "INFINITY"}[op]
    combine = {"sum": "+", "max": "max", "min": "min"}[op]

    # Each thread accumulates over strided elements
    kb._var("acc", identity, ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_elements; i += {block_size}) {{")
    kb.indent()
    kb._var("idx", f"pid * n_elements + i", ty="uint")
    if combine in ("+",):
        kb.raw_line("acc += float(input[idx]);")
    else:
        # acc is float; cast the (possibly half/bfloat) input so max/min isn't an
        # ambiguous mixed-type MSL overload (compile error for fp16/bf16 reduce).
        kb.raw_line(f"acc = {combine}(acc, float(input[idx]));")
    kb.dedent()
    kb.raw_line("}")

    # Two-level threadgroup reduction
    kb.threadgroup_reduce(op, "acc", "shared", "total")

    # Thread 0 writes result
    kb.begin_if("lid == 0")
    if dtype in ("fp16", "bf16"):
        store_ty = triton_type_to_msl(dtype)
        kb.raw_line(f"output[pid] = {store_ty}(total);")
    else:
        kb.raw_line("output[pid] = total;")
    kb.end_block()

    return kb.build()


def make_row_reduce_kernel(name, op, block_size=256, dtype="fp32"):
    """Generate a row-wise 2D reduction: output[row] = reduce(input[row, :]).

    Each threadgroup processes one row. Dispatch n_rows threadgroups.

    Args:
        name: Kernel function name.
        op: Reduction operation ("sum", "max", "min").
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder(name, block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_rows", dtype="u32")
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    identity = {"sum": "(n_cols == 0u ? 0.0f : -0.0f)", "max": "-INFINITY", "min": "INFINITY"}[op]
    combine = {"sum": "+", "max": "max", "min": "min"}[op]

    kb._var("row", "pid", ty="uint")
    kb.begin_if("row >= n_rows")
    kb.raw_line("return;")
    kb.end_block()

    kb._var("acc", identity, ty="float")
    kb.raw_line(f"for (uint c = lid; c < n_cols; c += {block_size}u) {{")
    kb.indent()
    kb._var("idx", "row * n_cols + c", ty="uint")
    if combine in ("+",):
        kb.raw_line("acc += float(input[idx]);")
    else:
        kb.raw_line(f"acc = {combine}(acc, float(input[idx]));")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce(op, "acc", "shared", "total")

    kb.begin_if("lid == 0")
    if dtype in ("fp16", "bf16"):
        store_ty = triton_type_to_msl(dtype)
        kb.raw_line(f"output[row] = {store_ty}(total);")
    else:
        kb.raw_line("output[row] = total;")
    kb.end_block()

    return kb.build()


def make_col_reduce_kernel(name, op, block_size=256, dtype="fp32"):
    """Generate a column-wise 2D reduction: output[col] = reduce(input[:, col]).

    Each threadgroup processes one column. Dispatch n_cols threadgroups.

    Args:
        name: Kernel function name.
        op: Reduction operation ("sum", "max", "min").
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder(name, block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_rows", dtype="u32")
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    identity = {"sum": "(n_rows == 0u ? 0.0f : -0.0f)", "max": "-INFINITY", "min": "INFINITY"}[op]
    combine = {"sum": "+", "max": "max", "min": "min"}[op]

    kb._var("col", "pid", ty="uint")
    kb.begin_if("col >= n_cols")
    kb.raw_line("return;")
    kb.end_block()

    kb._var("acc", identity, ty="float")
    kb.raw_line(f"for (uint r = lid; r < n_rows; r += {block_size}u) {{")
    kb.indent()
    kb._var("idx", "r * n_cols + col", ty="uint")
    if combine in ("+",):
        kb.raw_line("acc += float(input[idx]);")
    else:
        kb.raw_line(f"acc = {combine}(acc, float(input[idx]));")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce(op, "acc", "shared", "total")

    kb.begin_if("lid == 0")
    if dtype in ("fp16", "bf16"):
        store_ty = triton_type_to_msl(dtype)
        kb.raw_line(f"output[col] = {store_ty}(total);")
    else:
        kb.raw_line("output[col] = total;")
    kb.end_block()

    return kb.build()


def make_softmax_kernel(block_size=256, dtype="fp32"):
    """Generate a fused row-wise softmax kernel.

    Each threadgroup processes one row:
    1. Find max(row) — for numerical stability
    2. Compute exp(x - max) for each element
    3. Sum the exponentials
    4. Divide each by the sum

    Args:
        block_size: Threads per threadgroup (should be >= row length or will stride).
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32
    needs_cast = dtype in ("fp16", "bf16")
    # Wrap buffer reads with float() for half types to avoid ambiguous overloads
    read_input = "float(input[row_start + i])" if needs_cast else "input[row_start + i]"
    store_ty = triton_type_to_msl(dtype)

    kb = KernelBuilder("softmax_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    # Two shared arrays: one for max reduction, one for sum reduction
    kb.declare_threadgroup_array("shared_max", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_sum", dtype=dtype, size=n_simd_groups)

    # Row base pointer: each threadgroup handles one row
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Find row max (strided accumulation)
    kb._var("local_max", "-INFINITY", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    kb.raw_line(f"local_max = max(local_max, {read_input});")
    kb.dedent()
    kb.raw_line("}")

    # Reduce max across threadgroup
    kb.threadgroup_reduce("max", "local_max", "shared_max", "row_max")

    # Broadcast row_max to all threads via shared memory
    kb.begin_if("lid == 0")
    kb.raw_line("shared_max[0] = row_max;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("max_val", "shared_max[0]", ty="float")

    # Pass 2: Compute exp(x - max) and accumulate sum
    kb._var("local_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    kb._var("e", f"exp({read_input} - max_val)", ty="float")
    if needs_cast:
        kb.raw_line(f"output[row_start + i] = {store_ty}(e);")
    else:
        kb.raw_line("output[row_start + i] = e;")
    kb.raw_line("local_sum += e;")
    kb.dedent()
    kb.raw_line("}")

    # Reduce sum across threadgroup
    kb.threadgroup_reduce("sum", "local_sum", "shared_sum", "row_sum")

    # Broadcast row_sum
    kb.begin_if("lid == 0")
    kb.raw_line("shared_sum[0] = row_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("sum_val", "shared_sum[0]", ty="float")

    # Pass 3: Normalize
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    if needs_cast:
        kb.raw_line(f"output[row_start + i] = {store_ty}(float(output[row_start + i]) / sum_val);")
    else:
        kb.raw_line("output[row_start + i] /= sum_val;")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_matmul_kernel(block_m=32, block_n=32, block_k=32, dtype="fp32", out_dtype=None):
    """Generate a tiled matrix multiplication kernel: C = A @ B.

    A is (M, K), B is (K, N), C is (M, N).
    Each threadgroup computes a BLOCK_M x BLOCK_N tile of C.

    Uses threadgroup shared memory for A and B tiles to enable
    coalesced global memory access and data reuse.

    Constrained by Metal's 32KB threadgroup memory limit:
    - 32x32 fp32 tile = 4KB, two tiles = 8KB (well within limit)

    Apple GPUs cap threadgroup size at 1024 threads, so tiles larger
    than 32×32 distribute multiple output elements per thread:
    ``elements_per_thread = (block_m * block_n) / threads_per_tg``.

    Args:
        block_m: Tile height (rows of A/C per threadgroup).
        block_n: Tile width (cols of B/C per threadgroup).
        block_k: Tile depth (inner dimension chunk).
        dtype: Data type.
    """
    from triton_msl.codegen.msl_builtins import is_fp8_type, fp8_to_float_func, fp8_device_functions

    fp8_input = is_fp8_type(dtype)

    # Apple GPU threadgroup-thread cap.
    TG_MAX = 1024
    tile_elems = block_m * block_n
    threads_per_tg = min(tile_elems, TG_MAX)
    elements_per_thread = max(1, tile_elems // threads_per_tg)

    kb = KernelBuilder("matmul_kernel", block_size=threads_per_tg)
    kb.add_ptr_arg("A", dtype=dtype, const=True)
    kb.add_ptr_arg("B", dtype=dtype, const=True)
    # Output is always fp32 for FP8 inputs; otherwise honor the
    # caller-supplied ``out_dtype`` (defaults to the input dtype for
    # back-compat).
    if fp8_input:
        out_dtype = "fp32"
    elif out_dtype is None:
        out_dtype = dtype
    kb.add_ptr_arg("C", dtype=out_dtype, const=False)
    # The accumulator is float; the C store must cast to the output element type.
    # MSL permits implicit float->half but NOT float->bfloat, so the cast is
    # required for bf16 output (and harmless/correct for fp16/fp32).
    out_msl = triton_type_to_msl(out_dtype)
    kb.add_scalar_arg("M", dtype="u32")
    kb.add_scalar_arg("N", dtype="u32")
    kb.add_scalar_arg("K", dtype="u32")

    # For FP8, inject conversion device functions
    if fp8_input:
        for fn_src in fp8_device_functions(dtype):
            kb._device_functions.append(fn_src)

    # Threadgroup budget: tileA + tileB are fp32 (4 B each). Refuse if they exceed
    # Metal's 32 KiB limit rather than emit a kernel that crashes cryptically at
    # pipeline-state creation (e.g. block 128 -> two 64 KiB tiles = 128 KiB).
    _tg_bytes = (block_m * block_k + block_k * block_n) * 4
    if _tg_bytes > 32 * 1024:
        from triton_msl.errors import MetalResourceError

        raise MetalResourceError(_tg_bytes, 32 * 1024, "threadgroup memory bytes")

    # Shared memory tiles (always float for computation)
    kb.declare_threadgroup_array("tileA", dtype="fp32", size=block_m * block_k)
    kb.declare_threadgroup_array("tileB", dtype="fp32", size=block_k * block_n)

    # Global tile position: pid encodes the 2D tile index (grid is flattened 1D).
    kb._var("n_tile_cols", f"(N + {block_n}u - 1u) / {block_n}u", ty="uint")
    kb._var("tile_row", "pid / n_tile_cols", ty="uint")
    kb._var("tile_col", "pid % n_tile_cols", ty="uint")

    # Tile loop over K dimension
    kb._var("n_tiles_k", f"(K + {block_k}u - 1u) / {block_k}u", ty="uint")

    # Per-thread accumulator array. When the BM×BN tile exceeds 1024
    # output elements, each thread accumulates a strided sub-set of
    # the tile (positions ``lid``, ``lid + threads_per_tg``, …).
    # An array (vs N unrolled scalars) keeps generated MSL small enough
    # for Metal\'s shader compiler at large tiles — 128×128 with EPT=16
    # otherwise blows the complexity budget.
    if elements_per_thread == 1:
        kb._var("acc", "0.0f", ty="float")
    else:
        kb.raw_line(f"float acc[{elements_per_thread}];")
        kb.raw_line(f"for (uint _i = 0; _i < {elements_per_thread}u; _i++) acc[_i] = 0.0f;")

    # A-tile load amount per K-iter: block_m * block_k entries split
    # across ``threads_per_tg`` threads.
    a_tile_elems = block_m * block_k
    a_loads_per_thread = max(1, (a_tile_elems + threads_per_tg - 1) // threads_per_tg)
    b_tile_elems = block_k * block_n
    b_loads_per_thread = max(1, (b_tile_elems + threads_per_tg - 1) // threads_per_tg)

    if fp8_input:
        to_float = fp8_to_float_func(dtype)
        a_load_expr = f"{to_float}(A[a_gr * K + a_gc])"
        b_load_expr = f"{to_float}(B[b_gr * N + b_gc])"
    else:
        a_load_expr = "A[a_gr * K + a_gc]"
        b_load_expr = "B[b_gr * N + b_gc]"

    kb.raw_line("for (uint tk = 0; tk < n_tiles_k; tk++) {")
    kb.indent()

    # --- Load A tile cooperatively (loop instead of unroll) ---
    kb.raw_line(f"for (uint _i = 0; _i < {a_loads_per_thread}u; _i++) {{")
    kb.indent()
    kb.raw_line(f"uint a_idx = _i * {threads_per_tg}u + lid;")
    kb.raw_line(f"if (a_idx < {a_tile_elems}u) {{")
    kb.indent()
    kb.raw_line(f"uint a_lr = a_idx / {block_k}u;")
    kb.raw_line(f"uint a_lc = a_idx % {block_k}u;")
    kb.raw_line(f"uint a_gr = tile_row * {block_m}u + a_lr;")
    kb.raw_line(f"uint a_gc = tk * {block_k}u + a_lc;")
    kb.raw_line(f"tileA[a_idx] = (a_gr < M && a_gc < K) ? (float)({a_load_expr}) : 0.0f;")
    kb.dedent()
    kb.raw_line("}")
    kb.dedent()
    kb.raw_line("}")

    # --- Load B tile cooperatively ---
    kb.raw_line(f"for (uint _i = 0; _i < {b_loads_per_thread}u; _i++) {{")
    kb.indent()
    kb.raw_line(f"uint b_idx = _i * {threads_per_tg}u + lid;")
    kb.raw_line(f"if (b_idx < {b_tile_elems}u) {{")
    kb.indent()
    kb.raw_line(f"uint b_lr = b_idx / {block_n}u;")
    kb.raw_line(f"uint b_lc = b_idx % {block_n}u;")
    kb.raw_line(f"uint b_gr = tk * {block_k}u + b_lr;")
    kb.raw_line(f"uint b_gc = tile_col * {block_n}u + b_lc;")
    kb.raw_line(f"tileB[b_idx] = (b_gr < K && b_gc < N) ? (float)({b_load_expr}) : 0.0f;")
    kb.dedent()
    kb.raw_line("}")
    kb.dedent()
    kb.raw_line("}")

    kb.barrier("threadgroup")

    # --- Compute partial dot products ---
    if elements_per_thread == 1:
        kb.raw_line(f"uint lr = lid / {block_n}u;")
        kb.raw_line(f"uint lc = lid % {block_n}u;")
        kb.raw_line(f"for (uint kk = 0; kk < {block_k}u; kk++) {{")
        kb.indent()
        kb.raw_line(f"acc += tileA[lr * {block_k}u + kk] * tileB[kk * {block_n}u + lc];")
        kb.dedent()
        kb.raw_line("}")
    else:
        kb.raw_line(f"for (uint _e = 0; _e < {elements_per_thread}u; _e++) {{")
        kb.indent()
        kb.raw_line(f"uint out_idx = _e * {threads_per_tg}u + lid;")
        kb.raw_line(f"uint lr = out_idx / {block_n}u;")
        kb.raw_line(f"uint lc = out_idx % {block_n}u;")
        kb.raw_line(f"for (uint kk = 0; kk < {block_k}u; kk++) {{")
        kb.indent()
        kb.raw_line(f"acc[_e] += tileA[lr * {block_k}u + kk] * tileB[kk * {block_n}u + lc];")
        kb.dedent()
        kb.raw_line("}")
        kb.dedent()
        kb.raw_line("}")

    kb.barrier("threadgroup")

    kb.dedent()
    kb.raw_line("}")  # end K-tile loop

    # --- Write results ---
    if elements_per_thread == 1:
        kb.raw_line(f"uint olr = lid / {block_n}u;")
        kb.raw_line(f"uint olc = lid % {block_n}u;")
        kb.raw_line(f"uint gr = tile_row * {block_m}u + olr;")
        kb.raw_line(f"uint gc = tile_col * {block_n}u + olc;")
        kb.raw_line("if (gr < M && gc < N) {")
        kb.indent()
        kb.raw_line(f"C[gr * N + gc] = {out_msl}(acc);")
        kb.dedent()
        kb.raw_line("}")
    else:
        kb.raw_line(f"for (uint _e = 0; _e < {elements_per_thread}u; _e++) {{")
        kb.indent()
        kb.raw_line(f"uint out_idx = _e * {threads_per_tg}u + lid;")
        kb.raw_line(f"uint olr = out_idx / {block_n}u;")
        kb.raw_line(f"uint olc = out_idx % {block_n}u;")
        kb.raw_line(f"uint gr = tile_row * {block_m}u + olr;")
        kb.raw_line(f"uint gc = tile_col * {block_n}u + olc;")
        kb.raw_line("if (gr < M && gc < N) {")
        kb.indent()
        kb.raw_line(f"C[gr * N + gc] = {out_msl}(acc[_e]);")
        kb.dedent()
        kb.raw_line("}")
        kb.dedent()
        kb.raw_line("}")

    return kb.build()


def make_matmul_2d_kernel(block_m=32, block_n=32, block_k=32, dtype="fp32"):
    """Generate a tiled matmul kernel using native 2D threadgroup dispatch.

    Instead of linearizing the 2D tile grid into 1D (pid = row*cols + col),
    this kernel uses Metal's 2D dispatch directly:
    - threadgroup_position_in_grid.x → tile column
    - threadgroup_position_in_grid.y → tile row

    Dispatch with: MTLSizeMake(n_tile_cols, n_tile_rows, 1)

    This enables better hardware scheduling and is the foundation for
    swizzled dispatch patterns.

    Args:
        block_m: Tile height (rows of C per threadgroup).
        block_n: Tile width (cols of C per threadgroup).
        block_k: Tile depth (K dimension chunk).
        dtype: Data type ("fp32" or "fp16").
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)
    threads_per_tg = block_m * block_n

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void matmul_2d(
    device const {msl_ty}* A [[buffer(0)]],
    device const {msl_ty}* B [[buffer(1)]],
    device {msl_ty}* C [[buffer(2)]],
    device const uint* M_buf [[buffer(3)]],
    device const uint* N_buf [[buffer(4)]],
    device const uint* K_buf [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint M = M_buf[0];
    const uint N = N_buf[0];
    const uint K = K_buf[0];

    // 2D tile position from Metal's native 2D dispatch
    const uint tile_col = gid.x;
    const uint tile_row = gid.y;

    // Local thread position within tile
    const uint local_row = lid / {block_n}u;
    const uint local_col = lid % {block_n}u;

    // Global position
    const uint global_row = tile_row * {block_m}u + local_row;
    const uint global_col = tile_col * {block_n}u + local_col;

    // Shared memory tiles
    threadgroup {msl_ty} tileA[{block_m * block_k}];
    threadgroup {msl_ty} tileB[{block_k * block_n}];

    {compute_ty} acc = 0.0f;
    const uint n_tiles_k = (K + {block_k}u - 1u) / {block_k}u;

    for (uint tk = 0; tk < n_tiles_k; tk++) {{
        // Load A tile
        uint a_col = tk * {block_k}u + local_col;
        if (global_row < M && a_col < K) {{
            tileA[local_row * {block_k}u + local_col] = A[global_row * K + a_col];
        }} else {{
            tileA[local_row * {block_k}u + local_col] = 0.0f;
        }}

        // Load B tile
        uint b_row = tk * {block_k}u + local_row;
        if (b_row < K && global_col < N) {{
            tileB[local_row * {block_n}u + local_col] = B[b_row * N + global_col];
        }} else {{
            tileB[local_row * {block_n}u + local_col] = 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint kk = 0; kk < {block_k}u; kk++) {{
            acc += tileA[local_row * {block_k}u + kk] * tileB[kk * {block_n}u + local_col];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = static_cast<{msl_ty}>(acc);
    }}
}}
"""
    return msl


def make_matmul_swizzled_kernel(block_m=32, block_n=32, block_k=32, group_size=4, dtype="fp32"):
    """Generate a tiled matmul kernel with swizzled tile ordering.

    Uses a grouped swizzle pattern to improve L2 cache hit rate on large
    matrices. Instead of row-major tile traversal, tiles are grouped and
    columns within each group are traversed before moving to the next group.

    The swizzle pattern:
        group_id = tile_y / GROUP_SIZE
        first_tile_m = group_id * GROUP_SIZE
        group_size_m = min(n_tile_rows - first_tile_m, GROUP_SIZE)
        tile_y = first_tile_m + (linear_id % group_size_m)
        tile_x = (linear_id / group_size_m)

    This causes adjacent threadgroups to access adjacent rows of B,
    maximizing L2 cache reuse.

    Dispatch: 1D with n_groups = n_tile_rows * n_tile_cols

    Args:
        block_m: Tile height.
        block_n: Tile width.
        block_k: Tile depth.
        group_size: Number of tile rows per swizzle group.
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)
    threads_per_tg = block_m * block_n

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void matmul_swizzled(
    device const {msl_ty}* A [[buffer(0)]],
    device const {msl_ty}* B [[buffer(1)]],
    device {msl_ty}* C [[buffer(2)]],
    device const uint* M_buf [[buffer(3)]],
    device const uint* N_buf [[buffer(4)]],
    device const uint* K_buf [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint M = M_buf[0];
    const uint N = N_buf[0];
    const uint K = K_buf[0];

    const uint n_tile_cols = (N + {block_n}u - 1u) / {block_n}u;
    const uint n_tile_rows = (M + {block_m}u - 1u) / {block_m}u;

    // Swizzled tile assignment for L2 cache optimization
    const uint GROUP_SIZE = {group_size}u;
    const uint group_id = pid / (GROUP_SIZE * n_tile_cols);
    const uint first_tile_m = group_id * GROUP_SIZE;
    const uint group_size_m = min(n_tile_rows - first_tile_m, GROUP_SIZE);
    const uint linear_in_group = pid % (group_size_m * n_tile_cols);
    const uint tile_row = first_tile_m + (linear_in_group % group_size_m);
    const uint tile_col = linear_in_group / group_size_m;

    const uint local_row = lid / {block_n}u;
    const uint local_col = lid % {block_n}u;
    const uint global_row = tile_row * {block_m}u + local_row;
    const uint global_col = tile_col * {block_n}u + local_col;

    threadgroup {msl_ty} tileA[{block_m * block_k}];
    threadgroup {msl_ty} tileB[{block_k * block_n}];

    {compute_ty} acc = 0.0f;
    const uint n_tiles_k = (K + {block_k}u - 1u) / {block_k}u;

    for (uint tk = 0; tk < n_tiles_k; tk++) {{
        uint a_col = tk * {block_k}u + local_col;
        if (global_row < M && a_col < K) {{
            tileA[local_row * {block_k}u + local_col] = A[global_row * K + a_col];
        }} else {{
            tileA[local_row * {block_k}u + local_col] = 0.0f;
        }}

        uint b_row = tk * {block_k}u + local_row;
        if (b_row < K && global_col < N) {{
            tileB[local_row * {block_n}u + local_col] = B[b_row * N + global_col];
        }} else {{
            tileB[local_row * {block_n}u + local_col] = 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < {block_k}u; kk++) {{
            acc += tileA[local_row * {block_k}u + kk] * tileB[kk * {block_n}u + local_col];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = static_cast<{msl_ty}>(acc);
    }}
}}
"""
    return msl


def make_activation_kernel(activation="tanh", block_size=256, dtype="fp32"):
    """Generate an activation function kernel.

    Supports tanh, sigmoid, silu, elu, leaky_relu, and hardswish activations
    with optimized implementations.

    Args:
        activation: One of "tanh", "sigmoid", "silu", "elu", "leaky_relu", "hardswish".
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    act_map = {
        "tanh": "tanh(x)",
        "sigmoid": "1.0f / (1.0f + exp(-x))",
        "silu": "x / (1.0f + exp(-x))",
        "elu": "x >= 0.0f ? x : (exp(x) - 1.0f)",
        "leaky_relu": "x >= 0.0f ? x : 0.01f * x",
        "hardswish": "x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f)",
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation: {activation}. Supported: {list(act_map.keys())}")

    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)
    act_expr = act_map[activation]

    kb = KernelBuilder(f"{activation}_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n", "mask")
    kb.load("input", offsets, mask, out_var="x", dtype=dtype)
    kb._var("result", act_expr, ty=compute_ty)
    kb.store("output", offsets, "result", mask, dtype=dtype)

    return kb.build()


def make_rms_norm_kernel(block_size=256, dtype="fp32", eps=1e-6):
    """Generate a fused RMS normalization kernel.

    Used in LLaMA, Mistral, Gemma, and other modern LLMs.
    For each row: output = x * rsqrt(mean(x^2) + eps) * weight

    Each threadgroup processes one row. Three passes:
    1. Compute sum of squares (strided accumulation + threadgroup reduce)
    2. Compute RMS = rsqrt(mean_sq + eps)
    3. Apply normalization: output[i] = input[i] * rms * weight[i]

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("rms_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("weight", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared_sq", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Sum of squares (strided)
    kb._var("sq_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb._var("v", "input[row_start + i]", ty="float")
    kb.raw_line("sq_sum += v * v;")
    kb.dedent()
    kb.raw_line("}")

    # Reduce sum of squares across threadgroup
    kb.threadgroup_reduce("sum", "sq_sum", "shared_sq", "total_sq")

    # Broadcast and compute RMS
    kb.begin_if("lid == 0")
    kb.raw_line("shared_sq[0] = total_sq;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_sq", "shared_sq[0] / float(n_cols)", ty="float")
    kb._var("rms", f"rsqrt(mean_sq + {eps}f)", ty="float")

    # Pass 2: Apply normalization
    needs_cast = dtype in ("fp16", "bf16")
    if needs_cast:
        store_ty = triton_type_to_msl(dtype)
        norm_expr = f"{store_ty}(float(input[row_start + i]) * rms * float(weight[i]))"
    else:
        norm_expr = "input[row_start + i] * rms * weight[i]"
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"output[row_start + i] = {norm_expr};")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_rope_kernel(block_size=256, dtype="fp32"):
    """Generate a fused RoPE (rotary position embedding) kernel.

    Applies rotary position embeddings to pairs of elements:
    For each pair (x0, x1) at position pos with frequency freq:
        out0 = x0 * cos(theta) - x1 * sin(theta)
        out1 = x0 * sin(theta) + x1 * cos(theta)
    where theta = pos * freq

    Each threadgroup processes elements for one position.
    Frequencies are pre-computed: freq[i] = 1 / (10000^(2i/dim)).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        input: [seq_len, dim] tensor
        freqs: [dim/2] pre-computed inverse frequencies
        output: [seq_len, dim] tensor
        dim: hidden dimension (must be even)
        pos_offset: starting position index
    """
    kb = KernelBuilder("rope_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("freqs", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("dim", dtype="u32")
    kb.add_scalar_arg("pos_offset", dtype="u32")

    # Each threadgroup handles one position (pid = position index)
    kb._var("pos", "pid + pos_offset", ty="uint")
    kb._var("row_start", "pid * dim", ty="uint")

    # Each thread handles a pair of elements
    kb.raw_line(f"for (uint i = lid; i < dim / 2u; i += {block_size}u) {{")
    kb.indent()
    needs_cast = dtype in ("fp16", "bf16")
    read_freq = "float(freqs[i])" if needs_cast else "freqs[i]"
    kb._var("theta", f"float(pos) * {read_freq}", ty="float")
    kb._var("cos_t", "cos(theta)", ty="float")
    kb._var("sin_t", "sin(theta)", ty="float")
    kb._var("x0", "float(input[row_start + 2u * i])", ty="float")
    kb._var("x1", "float(input[row_start + 2u * i + 1u])", ty="float")
    if needs_cast:
        store_ty = triton_type_to_msl(dtype)
        kb.raw_line(f"output[row_start + 2u * i] = {store_ty}(x0 * cos_t - x1 * sin_t);")
        kb.raw_line(f"output[row_start + 2u * i + 1u] = {store_ty}(x0 * sin_t + x1 * cos_t);")
    else:
        kb.raw_line("output[row_start + 2u * i] = x0 * cos_t - x1 * sin_t;")
        kb.raw_line("output[row_start + 2u * i + 1u] = x0 * sin_t + x1 * cos_t;")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_layer_norm_kernel(block_size=256, dtype="fp32", eps=1e-6):
    """Generate a fused layer normalization kernel.

    Standard layer norm used in transformers:
    output = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    Each threadgroup processes one row. Three passes:
    1. Compute mean (strided accumulation + threadgroup reduce)
    2. Compute variance (strided + reduce)
    3. Normalize: (x - mean) * rsqrt(var + eps) * gamma + beta

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("layer_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("gamma", dtype=dtype, const=True)
    kb.add_ptr_arg("beta", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared_mean", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_var", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Compute mean
    needs_cast = dtype in ("fp16", "bf16")
    read_expr = "float(input[row_start + i])" if needs_cast else "input[row_start + i]"
    kb._var("local_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"local_sum += {read_expr};")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_sum", "shared_mean", "total_sum")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_mean[0] = total_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_val", "shared_mean[0] / float(n_cols)", ty="float")

    # Pass 2: Compute variance
    kb._var("local_var", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb._var("diff", f"{read_expr} - mean_val", ty="float")
    kb.raw_line("local_var += diff * diff;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_var", "shared_var", "total_var")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_var[0] = total_var;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("var_val", "shared_var[0] / float(n_cols)", ty="float")
    kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

    # Pass 3: Normalize
    needs_cast = dtype in ("fp16", "bf16")
    store_ty = triton_type_to_msl(dtype) if needs_cast else None
    compute_expr = (
        "(float(input[row_start + i]) - mean_val) * inv_std * float(gamma[i]) + float(beta[i])"
        if needs_cast
        else "(input[row_start + i] - mean_val) * inv_std * gamma[i] + beta[i]"
    )
    store_expr = f"{store_ty}({compute_expr})" if needs_cast else compute_expr
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"output[row_start + i] = {store_expr};")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_fused_residual_norm_kernel(block_size=256, dtype="fp32", eps=1e-6):
    """Generate a fused residual connection + layer normalization kernel.

    output = LayerNorm(input + residual, gamma, beta)

    Combines residual add and layer norm in a single kernel, avoiding
    a separate elementwise kernel and the intermediate materialization.
    This is the standard pattern in every transformer block.

    Each threadgroup processes one row. Three passes:
    1. Compute x = input + residual, accumulate sum for mean
    2. Compute (x - mean)^2 for variance
    3. Normalize: (x - mean) * rsqrt(var + eps) * gamma + beta

    Also optionally writes the pre-norm value (input + residual) for
    use in the next residual connection (common in pre-norm architectures).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("fused_residual_norm", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("residual", dtype=dtype, const=True)
    kb.add_ptr_arg("gamma", dtype=dtype, const=True)
    kb.add_ptr_arg("beta", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_ptr_arg("residual_out", dtype=dtype, const=False)  # pre-norm output
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared_sum", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_var", dtype=dtype, size=n_simd_groups)
    # Shared buffer for the fused input+residual (needed for pass 2 & 3)
    kb.declare_threadgroup_array("tg_x", dtype=dtype, size=block_size)

    # Row base pointer
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Half-precision handling
    needs_cast = dtype in ("fp16", "bf16")
    store_ty = triton_type_to_msl(dtype) if needs_cast else None

    def _read(buf_expr):
        return f"float({buf_expr})" if needs_cast else buf_expr

    def _store(val_expr):
        return f"{store_ty}({val_expr})" if needs_cast else val_expr

    # Pass 1: Compute x = input + residual, write residual_out, accumulate sum
    kb._var("local_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"float x_val = {_read('input[row_start + i]')} + {_read('residual[row_start + i]')};")
    kb.raw_line(f"residual_out[row_start + i] = {_store('x_val')};")
    kb.raw_line("local_sum += x_val;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_sum", "shared_sum", "total_sum")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_sum[0] = total_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_val", "shared_sum[0] / float(n_cols)", ty="float")

    # Pass 2: Compute variance
    kb._var("local_var", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"float x_val = {_read('input[row_start + i]')} + {_read('residual[row_start + i]')};")
    kb.raw_line("float diff = x_val - mean_val;")
    kb.raw_line("local_var += diff * diff;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_var", "shared_var", "total_var")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_var[0] = total_var;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("var_val", "shared_var[0] / float(n_cols)", ty="float")
    kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

    # Pass 3: Normalize
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line(f"float x_val = {_read('input[row_start + i]')} + {_read('residual[row_start + i]')};")
    norm_val = f"(x_val - mean_val) * inv_std * {_read('gamma[i]')} + {_read('beta[i]')}"
    kb.raw_line(f"output[row_start + i] = {_store(norm_val)};")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_variance_kernel(block_size=256, dtype="fp32"):
    """Generate a row-wise variance kernel.

    Computes var(x) = mean((x - mean(x))^2) for each row.
    Two-pass approach: first compute mean, then compute mean of squared diffs.

    Each threadgroup processes one row (dispatch one group per row).

    Layout:
        input: [n_rows, n_cols]
        output: [n_rows] — variance for each row
        n_cols: number of columns

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void variance_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device {msl_ty}* output [[buffer(1)]],
    device const uint* ncols_buf [[buffer(2)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint n_cols = ncols_buf[0];
    const uint row = pid;
    const uint row_offset = row * n_cols;

    // Pass 1: compute mean via strided sum
    {compute_ty} partial_sum = 0.0f;
    for (uint col = lid; col < n_cols; col += {block_size}u) {{
        partial_sum += static_cast<{compute_ty}>(input[row_offset + col]);
    }}

    // SIMD reduce for sum
    partial_sum = simd_sum(partial_sum);

    threadgroup {compute_ty} tg_sum[{(block_size + 31) // 32}];
    if (tiisg == 0) {{
        tg_sum[sgitg] = partial_sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {compute_ty} mean_val = 0.0f;
    if (lid == 0) {{
        {compute_ty} total = 0.0f;
        for (uint i = 0; i < {(block_size + 31) // 32}u; i++) {{
            total += tg_sum[i];
        }}
        mean_val = total / static_cast<{compute_ty}>(n_cols);
        tg_sum[0] = mean_val;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean_val = tg_sum[0];

    // Pass 2: compute mean of squared differences
    {compute_ty} partial_var = 0.0f;
    for (uint col = lid; col < n_cols; col += {block_size}u) {{
        {compute_ty} diff = static_cast<{compute_ty}>(input[row_offset + col]) - mean_val;
        partial_var += diff * diff;
    }}

    partial_var = simd_sum(partial_var);
    if (tiisg == 0) {{
        tg_sum[sgitg] = partial_var;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {{
        {compute_ty} total_var = 0.0f;
        for (uint i = 0; i < {(block_size + 31) // 32}u; i++) {{
            total_var += tg_sum[i];
        }}
        output[row] = static_cast<{msl_ty}>(total_var / static_cast<{compute_ty}>(n_cols));
    }}
}}
"""
    return msl


def make_batch_norm_kernel(block_size=256, dtype="fp32", eps=1e-5):
    """Generate a batch normalization kernel (inference mode).

    For each channel c:
        output[b, c, h, w] = gamma[c] * (input[b, c, h, w] - mean[c]) / sqrt(var[c] + eps) + beta[c]

    Uses pre-computed running mean and variance (eval mode).
    Each threadgroup processes one spatial position across channels.

    Layout:
        input: [N * C * HW] flattened, stored as [N, C, HW]
        output: [N * C * HW]
        gamma: [C] — scale parameters
        beta: [C] — shift parameters
        running_mean: [C]
        running_var: [C]
        n_channels: scalar
        spatial_size: HW (height * width)

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device {msl_ty}* output [[buffer(1)]],
    device const {msl_ty}* gamma [[buffer(2)]],
    device const {msl_ty}* beta [[buffer(3)]],
    device const {msl_ty}* running_mean [[buffer(4)]],
    device const {msl_ty}* running_var [[buffer(5)]],
    device const uint* n_channels_buf [[buffer(6)]],
    device const uint* spatial_size_buf [[buffer(7)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {{
    const uint n_channels = n_channels_buf[0];
    const uint spatial_size = spatial_size_buf[0];
    const uint total = n_channels * spatial_size;

    // Each thread processes one element: index = batch * C*HW + channel * HW + spatial
    // We iterate over the flattened [N*C*HW] with standard 1D dispatch
    uint idx = tid;
    if (idx >= total) return;

    // Determine which channel this element belongs to
    uint channel = (idx / spatial_size) % n_channels;

    {compute_ty} x = static_cast<{compute_ty}>(input[idx]);
    {compute_ty} mean = static_cast<{compute_ty}>(running_mean[channel]);
    {compute_ty} var = static_cast<{compute_ty}>(running_var[channel]);
    {compute_ty} g = static_cast<{compute_ty}>(gamma[channel]);
    {compute_ty} b = static_cast<{compute_ty}>(beta[channel]);

    {compute_ty} normalized = (x - mean) * rsqrt(var + {eps}f);
    output[idx] = static_cast<{msl_ty}>(g * normalized + b);
}}
"""
    return msl


def make_causal_attention_kernel(n_heads=8, head_dim=64, block_size=256):
    """Generate a multi-head attention kernel with causal (triangular) mask.

    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d) + causal_mask) @ V

    The causal mask sets entries where key_pos > query_pos to -inf,
    preventing attention to future tokens (autoregressive).

    Each threadgroup processes one query position for one head.
    For small sequences only (seq_len <= block_size).

    Layout:
        Q: [n_heads, seq_len, head_dim]
        K: [n_heads, seq_len, head_dim]
        V: [n_heads, seq_len, head_dim]
        output: [n_heads, seq_len, head_dim]
        seq_len: scalar

    Args:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Threads per threadgroup.
    """
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void causal_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* seq_len_buf [[buffer(4)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint seq_len = seq_len_buf[0];
    const uint head = pid / seq_len;
    const uint query_pos = pid % seq_len;
    const float scale = rsqrt(float({head_dim}));

    const uint head_offset = head * seq_len * {head_dim}u;
    const uint q_offset = head_offset + query_pos * {head_dim}u;

    // Each thread computes attention score for one key position
    threadgroup float tg_scores[{block_size}];
    threadgroup float tg_max[{(block_size + 31) // 32}];
    threadgroup float tg_sum[{(block_size + 31) // 32}];

    // Step 1: compute Q @ K^T with causal mask
    float score = -INFINITY;
    if (lid < seq_len && lid <= query_pos) {{
        // Dot product Q[query_pos] . K[lid]
        float dot = 0.0f;
        for (uint d = 0; d < {head_dim}u; d++) {{
            dot += Q[q_offset + d] * K[head_offset + lid * {head_dim}u + d];
        }}
        score = dot * scale;
    }}
    tg_scores[lid] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: online softmax - find max
    float local_max = score;
    local_max = simd_max(local_max);
    if (tiisg == 0) tg_max[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {{
        float m = -INFINITY;
        for (uint i = 0; i < {(block_size + 31) // 32}u; i++) m = max(m, tg_max[i]);
        tg_max[0] = m;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = tg_max[0];

    // Step 3: exp(score - max)
    float exp_score = (lid < seq_len && lid <= query_pos) ? exp(score - row_max) : 0.0f;
    tg_scores[lid] = exp_score;

    // Step 4: sum
    float local_sum = exp_score;
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) tg_sum[sgitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {{
        float s = 0.0f;
        for (uint i = 0; i < {(block_size + 31) // 32}u; i++) s += tg_sum[i];
        tg_sum[0] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_sum = tg_sum[0];

    // Normalize
    float weight = (lid < seq_len && lid <= query_pos) ? tg_scores[lid] / row_sum : 0.0f;

    // Step 5: weighted sum of V
    // Each thread contributes weight * V[lid]
    for (uint d = lid; d < {head_dim}u; d += {block_size}u) {{
        float val = 0.0f;
        for (uint k = 0; k <= query_pos && k < seq_len; k++) {{
            float w = tg_scores[k] / row_sum;
            val += w * V[head_offset + k * {head_dim}u + d];
        }}
        output[q_offset + d] = val;
    }}
}}
"""
    return msl


def make_online_softmax_kernel(block_size=256, dtype="fp32"):
    """Generate a single-pass online softmax kernel.

    Uses the online softmax trick to compute softmax in one pass:
    - Track running max and running sum of exp(x - max)
    - Correct when a new max is found: multiply old sum by exp(old_max - new_max)

    More efficient than two-pass softmax for long sequences.

    Layout:
        input: [n_rows, n_cols]
        output: [n_rows, n_cols]
        n_cols: number of columns

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void online_softmax_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device {msl_ty}* output [[buffer(1)]],
    device const uint* ncols_buf [[buffer(2)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint n_cols = ncols_buf[0];
    const uint row = pid;
    const uint row_offset = row * n_cols;

    // Single-pass: each thread processes a strided chunk
    // Use -1e30 instead of -INFINITY to avoid NaN in exp(-inf - (-inf))
    {compute_ty} thread_max = -1e30f;
    {compute_ty} thread_sum = 0.0f;

    for (uint col = lid; col < n_cols; col += {block_size}u) {{
        {compute_ty} x = static_cast<{compute_ty}>(input[row_offset + col]);
        if (x > thread_max) {{
            thread_sum = thread_sum * exp(thread_max - x) + 1.0f;
            thread_max = x;
        }} else {{
            thread_sum += exp(x - thread_max);
        }}
    }}

    // SIMD-level online reduce: combine (max, sum) pairs
    for (uint offset = 16; offset > 0; offset >>= 1) {{
        {compute_ty} other_max = simd_shuffle_down(thread_max, offset);
        {compute_ty} other_sum = simd_shuffle_down(thread_sum, offset);
        {compute_ty} new_max = max(thread_max, other_max);
        {compute_ty} scale1 = exp(thread_max - new_max);
        {compute_ty} scale2 = exp(other_max - new_max);
        thread_sum = thread_sum * scale1 + other_sum * scale2;
        thread_max = new_max;
    }}

    // Cross-simdgroup reduce
    threadgroup {compute_ty} tg_max[{(block_size + 31) // 32}];
    threadgroup {compute_ty} tg_sum[{(block_size + 31) // 32}];
    if (tiisg == 0) {{
        tg_max[sgitg] = thread_max;
        tg_sum[sgitg] = thread_sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {{
        {compute_ty} final_max = tg_max[0];
        {compute_ty} final_sum = tg_sum[0];
        for (uint i = 1; i < {(block_size + 31) // 32}u; i++) {{
            {compute_ty} m = tg_max[i];
            {compute_ty} s = tg_sum[i];
            {compute_ty} new_max = max(final_max, m);
            final_sum = final_sum * exp(final_max - new_max) + s * exp(m - new_max);
            final_max = new_max;
        }}
        tg_max[0] = final_max;
        tg_sum[0] = final_sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {compute_ty} row_max = tg_max[0];
    {compute_ty} row_sum = tg_sum[0];

    // Write normalized output
    for (uint col = lid; col < n_cols; col += {block_size}u) {{
        {compute_ty} x = static_cast<{compute_ty}>(input[row_offset + col]);
        output[row_offset + col] = static_cast<{msl_ty}>(exp(x - row_max) / row_sum);
    }}
}}
"""
    return msl


def make_cross_entropy_kernel(block_size=256, dtype="fp32"):
    """Generate a fused cross-entropy loss kernel.

    For each sample (row):
    1. Compute max(logits) for numerical stability
    2. Compute log_sum_exp = max + log(sum(exp(logits - max)))
    3. loss = log_sum_exp - logits[target]

    Each threadgroup processes one sample. Outputs per-sample losses.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        logits: [batch, vocab_size] tensor
        targets: [batch] int32 tensor (class indices)
        losses: [batch] float output (per-sample losses)
        vocab_size: number of classes
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("cross_entropy_kernel", block_size=block_size)
    kb.add_ptr_arg("logits", dtype=dtype, const=True)
    kb.add_ptr_arg("targets", dtype="i32", const=True)
    kb.add_ptr_arg("losses", dtype=dtype, const=False)
    kb.add_scalar_arg("vocab_size", dtype="u32")

    kb.declare_threadgroup_array("shared_max", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_sum", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * vocab_size", ty="uint")

    # Pass 1: Find max logit for numerical stability
    kb._var("local_max", "-INFINITY", ty="float")
    kb.raw_line(f"for (uint i = lid; i < vocab_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("local_max = max(local_max, logits[row_start + i]);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("max", "local_max", "shared_max", "row_max")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_max[0] = row_max;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("max_val", "shared_max[0]", ty="float")

    # Pass 2: Compute sum(exp(logits - max))
    kb._var("local_exp_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < vocab_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("local_exp_sum += exp(logits[row_start + i] - max_val);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_exp_sum", "shared_sum", "total_exp_sum")

    # Thread 0 computes final loss
    kb.begin_if("lid == 0")
    kb._var("target_idx", "targets[pid]", ty="int")
    kb._var("log_sum_exp", "max_val + log(total_exp_sum)", ty="float")
    kb._var("target_logit", "logits[row_start + uint(target_idx)]", ty="float")
    kb.raw_line("losses[pid] = log_sum_exp - target_logit;")
    kb.end_block()

    return kb.build()


def make_flash_attention_kernel(head_dim=64, Br=16, Bc=16, block_size=256, causal=False):
    """Generate a fused Flash Attention kernel for Metal.

    Implements the FlashAttention-2 algorithm with online softmax:
    For each query block:
        O = 0, l = 0, m = -inf
        For each KV block:
            S = Q @ K^T           (Br x Bc scores)
            m_new = max(m, rowmax(S))
            P = exp(S - m_new)    (unnormalized attention)
            l = exp(m - m_new) * l + rowsum(P)
            O = exp(m - m_new) * O + P @ V
            m = m_new
        O = O / l

    When causal=True, attention scores where key position > query position
    are masked to -infinity, implementing autoregressive causal attention.

    Threadgroup memory budget (head_dim=64, Br=Bc=16, fp32):
        Q:  16x64x4 =  4KB
        K:  16x64x4 =  4KB
        V:  16x64x4 =  4KB
        S:  16x16x4 =  1KB
        O:  16x64x4 =  4KB
        l,m: 16x4x2 = 128B
        Total: ~17KB (well within 32KB)

    Args:
        head_dim: Head dimension (d). Must be multiple of 8.
        Br: Query block size (rows of Q per threadgroup).
        Bc: KV block size (rows of K/V loaded per inner loop step).
        block_size: Threads per threadgroup.
        causal: If True, apply causal mask (key pos <= query pos only).

    Kernel args:
        Q: [n_heads * seq_len, head_dim]
        K: [n_heads * seq_len, head_dim]
        V: [n_heads * seq_len, head_dim]
        O: [n_heads * seq_len, head_dim]
        seq_len: sequence length
        scale: 1/sqrt(head_dim)
    """
    # Causal masking: mask out S[i][j] where kv_pos > q_pos
    causal_mask = ""
    if causal:
        causal_mask = """
            uint q_pos = q_start + r;
            uint kv_pos_check = kv_start + c;
            tg_S[i] = (kv_pos < seq_len && kv_pos_check <= q_pos) ? dot * scale : -INFINITY;"""
    else:
        causal_mask = """
            tg_S[i] = (kv_pos < seq_len) ? dot * scale : -INFINITY;"""

    return f"""#include <metal_stdlib>
using namespace metal;

kernel void flash_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    // pid encodes (head_idx * n_q_blocks + q_block_idx)
    // Each threadgroup handles Br={Br} query rows
    const uint BR = {Br}u;
    const uint BC = {Bc}u;
    const uint D = {head_dim}u;

    uint n_q_blocks = (seq_len + BR - 1u) / BR;
    uint head_idx = pid / n_q_blocks;
    uint q_block = pid % n_q_blocks;
    uint q_start = q_block * BR;
    uint head_offset = head_idx * seq_len * D;

    // Threadgroup memory
    threadgroup float tg_Q[{Br} * {head_dim}];    // Br x D
    threadgroup float tg_K[{Bc} * {head_dim}];    // Bc x D
    threadgroup float tg_V[{Bc} * {head_dim}];    // Bc x D
    threadgroup float tg_S[{Br} * {Bc}];          // Br x Bc
    threadgroup float tg_O[{Br} * {head_dim}];    // Br x D
    threadgroup float tg_m[{Br}];                  // row max
    threadgroup float tg_l[{Br}];                  // row sum

    // Load Q block into threadgroup memory
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        uint c = i % D;
        uint global_r = q_start + r;
        tg_Q[i] = (global_r < seq_len) ? Q[head_offset + global_r * D + c] : 0.0f;
    }}

    // Initialize O, m, l
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        tg_O[i] = 0.0f;
    }}
    if (lid < BR) {{
        tg_m[lid] = -INFINITY;
        tg_l[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over KV blocks
    uint n_kv_blocks = (seq_len + BC - 1u) / BC;
    for (uint kv_block = 0u; kv_block < n_kv_blocks; kv_block++) {{
        uint kv_start = kv_block * BC;

        // Load K block
        for (uint i = lid; i < BC * D; i += {block_size}u) {{
            uint r = i / D;
            uint c = i % D;
            uint global_r = kv_start + r;
            tg_K[i] = (global_r < seq_len) ? K[head_offset + global_r * D + c] : 0.0f;
        }}

        // Load V block
        for (uint i = lid; i < BC * D; i += {block_size}u) {{
            uint r = i / D;
            uint c = i % D;
            uint global_r = kv_start + r;
            tg_V[i] = (global_r < seq_len) ? V[head_offset + global_r * D + c] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T (Br x Bc) — each thread computes one element
        for (uint i = lid; i < BR * BC; i += {block_size}u) {{
            uint r = i / BC;
            uint c = i % BC;
            float dot = 0.0f;
            for (uint d = 0u; d < D; d++) {{
                dot += tg_Q[r * D + d] * tg_K[c * D + d];
            }}
            // Mask out-of-bounds KV positions (and causal mask if enabled)
            uint kv_pos = kv_start + c;{causal_mask}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // For each query row: update online softmax and accumulate output
        // Each thread handles one query row
        if (lid < BR) {{
            uint r = lid;
            float m_prev = tg_m[r];
            float l_prev = tg_l[r];

            // Row max of S[r, :]
            float m_new = m_prev;
            for (uint c = 0u; c < BC; c++) {{
                m_new = max(m_new, tg_S[r * BC + c]);
            }}

            // Compute P[r, :] = exp(S[r, :] - m_new) and sum
            float exp_scale = exp(m_prev - m_new);
            float l_new = l_prev * exp_scale;
            for (uint c = 0u; c < BC; c++) {{
                float p = exp(tg_S[r * BC + c] - m_new);
                tg_S[r * BC + c] = p;  // store P in place of S
                l_new += p;
            }}

            // Rescale existing O and accumulate P @ V
            for (uint d = 0u; d < D; d++) {{
                float o_val = tg_O[r * D + d] * exp_scale;
                for (uint c = 0u; c < BC; c++) {{
                    o_val += tg_S[r * BC + c] * tg_V[c * D + d];
                }}
                tg_O[r * D + d] = o_val;
            }}

            tg_m[r] = m_new;
            tg_l[r] = l_new;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Final normalization: O = O / l
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        float l_val = tg_l[r];
        if (l_val > 0.0f) {{
            tg_O[i] /= l_val;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        uint c = i % D;
        uint global_r = q_start + r;
        if (global_r < seq_len) {{
            O[head_offset + global_r * D + c] = tg_O[i];
        }}
    }}
}}
"""


def _fa_scale_chain_expr(expr, scale_chain, round_q):
    """Replay scalar conversions independently of the Q product's precision.

    The canonical single conversion is already replayed by Q staging. Keep that
    spelling unchanged. Other chains round the uniform before either use of scale.
    Software bf16 RNE preserves NaNs and avoids Metal's uniform-bfloat cast bug.
    """
    chain = tuple(scale_chain)
    if chain not in ((), ("half",), ("bfloat",), ("half", "bfloat")):
        raise ValueError(f"unproved forward scalar conversion chain {chain!r}")
    if not chain or chain == (round_q,):
        return expr
    for target in chain:
        if target == "half":
            expr = f"float(half({expr}))"
        else:
            expr = (
                f"(isnan({expr}) ? {expr} : as_type<float>((as_type<uint>({expr}) + "
                f"0x7FFFu + ((as_type<uint>({expr}) >> 16u) & 1u)) & 0xFFFF0000u))"
            )
    return expr


def _fa_source_tail_guard(source_kv_block, staging_block):
    """Score extent after a proved zero-filled source KV loop, not a load mask.

    The physical tile may combine several source iterations. Unsigned subtraction
    is guarded by the first disjunct; unlike ceil(N/B)*B, it does not overflow at
    the length boundary. Defaults on all other maker callers remain unchanged.
    """
    if type(source_kv_block) is not int or source_kv_block <= 0 or staging_block % source_kv_block != 0:
        raise ValueError("source KV block must be a positive divisor of the staging block")
    b = source_kv_block
    return f"((kv_row < N_CTX) || (kv_row - N_CTX < (({b}u - N_CTX % {b}u) % {b}u))) /* source KV tail: {b} */"


def _fa_initial_values(values, source_kv_block):
    """Finite f32 numerator/denominator initializers, opt-in from the dense proof."""
    import math

    if values is None:
        return "0.0f", "0.0f"
    if (
        source_kv_block is None
        or not isinstance(values, tuple)
        or len(values) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
    ):
        raise ValueError("attention initial values require the proved dense source and two finite constants")
    return tuple(f"{float(v)!r}f" for v in values)


def make_flash_attention_kernel_simdgroup(
    head_dim=128,
    BLOCK_M=32,
    BLOCK_N=64,
    causal=False,
    out_dtype="fp32",
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention",
    scale=None,
    v_head_dim=None,
    half_accumulate=False,
    bias=False,
    mask=False,
    lse=False,
    runtime_scale=False,
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    neg_inf_unit=1.0,
    round_q=None,
    round_p=None,
    scale_chain=(),
    source_kv_block=None,
    mask_tail_scores=True,
    unmasked_kv=(),
    source_causal_stop=None,
    initial_values=None,
    fragment_lowering="off",
):
    """simdgroup_matrix FlashAttention-2 (fp32/fp16, causal/non-causal, head_dim=128).

    packet 164 (item F-a): ``round_q`` / ``round_p`` in {None, "half", "bfloat"} replay the SOURCE's
    narrowing — ``q * tl.full([1], scale, dtype)`` (the scale rounded to the input dtype, each product
    rounded; the score tile is then NOT scaled again) and ``p.to(dtype)`` before P @ V. A half×half
    or bfloat×bfloat product is exact in fp32, so ``elem(float(q) * float(elem(scale)))`` is the
    source's single rounding. fp32 sources (None) emit byte-identical MSL to before.

    Device-direct simdgroup MMA: QK^T via transpose-load, register-resident O
    accumulator with elementwise alpha-rescale + 1/l-normalize, Q staged once,
    V loaded once-per-(col-tile,k) and prefetched. 256 threads / 8 SIMD groups,
    q-tile BLOCK_M=32, kv-tile BLOCK_N=64. LOOP-SPLIT: full kv-blocks run a
    branchless device-direct fast path; the single partial tail block (if
    N_CTX %% BN != 0) runs a separate staged+masked path (keeps the hot loop free
    of barrier-in-conditional, which otherwise ~halved throughput). Partial
    q-blocks are handled by the Q-staging zero-pad + the guarded element-wise
    final store. Correct for ANY N_CTX and contiguous innermost strides; the
    routing contiguity gate sends non-contiguous FA to the scalar template.
    Same buffer ABI as make_flash_attention_kernel_tiled.

    half_accumulate (OPT-IN, fp16 only): use fp16 (half8x8) MMA accumulators for
    the QK^T and P@V matmuls + the online-softmax rescale/normalize instead of the
    default fp32 accumulate. ~4% faster at ~1% max-abs error vs ~0.01% for
    fp32-accumulate — the same accuracy/speed trade the int8/int4 quant paths ship.
    (M4's half MMA is ~1.38x the fp pipeline, but this kernel is latency-bound on
    device loads, so the MMA is only ~14% of runtime — hence ~4%, not ~38%. It does
    NOT close the full-attention MLX gap.) OFF by default: the accumulators stay fp32
    and the emitted MSL is byte-identical to before, so the default correctness
    contract is unchanged. Enabling it on an fp32 kernel is a hard error (there is no
    half input to accumulate) — correct-or-refuse.
    """
    import math as _math

    _fa_fragment_mode(fragment_lowering)
    _expanded = fragment_lowering != "off"
    if _expanded:
        reason = _fa_fragment_scope(
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            out_dtype=out_dtype,
            causal=causal,
            half_accumulate=half_accumulate,
            source_kv_block=source_kv_block,
            mask_tail_scores=mask_tail_scores,
            unmasked_kv=unmasked_kv,
            source_causal_stop=source_causal_stop,
            initial_values=initial_values,
            bias=bias,
            mask=mask,
            lse=lse,
            runtime_scale=runtime_scale,
            grid_3d=grid_3d,
            mask_batch_div=mask_batch_div,
            runtime_neg_inf=runtime_neg_inf,
            round_q=round_q,
            round_p=round_p,
            scale_chain=scale_chain,
            scale=scale,
        )
        if reason is not None:
            raise ValueError("fragment lowering outside prepared scope: " + reason)
    D, BM, BN, NT = head_dim, BLOCK_M, BLOCK_N, 256
    _initial_acc, _initial_den = _fa_initial_values(initial_values, source_kv_block)
    # ASYMMETRIC head dims (MLA / DeepSeek-style): the QK CONTRACTION runs over
    # head_dim (Dqk), but the OUTPUT / V tiling runs over v_head_dim (Dv). `D` drives
    # tgQ + the QK loop (Dqk); the O accumulator, V loads, and final store depend ONLY
    # on TPG. So decoupling is exactly TPG <- Dv (not D). v_head_dim=None => symmetric
    # (Dv == head_dim), byte-identical to before. Registers scale with Dv, so
    # MLA (Dqk=192, Dv=128) has the SAME footprint as hd128 despite the 192-wide QK.
    Dv = v_head_dim if v_head_dim is not None else D
    if not (BM == 32 and BN == 64 and D > 0 and Dv > 0 and D % 8 == 0 and Dv % 8 == 0):
        raise ValueError("simd FA requires BLOCK_M=32, BLOCK_N=64, positive head_dim/v_head_dim divisible by 8")
    n_groups = NT // 32
    # TPG = output col-tiles (8-wide) each simdgroup owns. When Dv/8 < n_groups (a
    # SMALL head_dim, e.g. Dv=32 -> 4 tiles < 8 groups), the plain floor gives 0
    # (a zero-length o[4][TPG] -> compile error). Round up and, when there are
    # then MORE (group,tile) slots than real col-tiles, gate the V-load / PV-MMA /
    # store on `ct*8 < Dv` so the surplus groups idle. For Dv a multiple of 64
    # (64/128/192) TPG*n_groups*8 == Dv exactly -> no guard, byte-identical to before.
    # Round up also for partial later rounds (e.g. Dv=96 needs two rounds).
    TPG = (Dv + n_groups * 8 - 1) // (n_groups * 8)
    need_ct_guard = TPG * n_groups * 8 > Dv
    # ct-guard fragments (value-level, NOT wrapping any barrier -> no divergent-barrier
    # UB). Empty when not needed -> byte-identical for Dv multiple of 64. In the biased
    # path Dv==D so retain its existing text; asymmetric callers need the V
    # width, not the QK contraction width, at all three memory boundaries.
    _v_bound = "D" if Dv == D else f"{Dv}u"
    _ct_off = f"(ct*8u < {_v_bound} ? ct*8u : 0u)" if need_ct_guard else "ct*8u"  # full V-load clamp
    _ct_tail_g = f" && (ct*8u + cc < {_v_bound})" if need_ct_guard else ""  # tail V-staging value
    _ct_store_g = f" && (dc2 < {_v_bound})" if need_ct_guard else ""  # final store value
    SCALE = float(scale) if scale is not None else 1.0 / _math.sqrt(float(D))
    if out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
        p_load_t, kv_frag = "simdgroup_half8x8", "simdgroup_half8x8"
    elif out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
        p_load_t, kv_frag = "simdgroup_float8x8", "simdgroup_float8x8"
    else:
        raise ValueError(f"out_dtype must be fp32/f32/fp16/f16 (got {out_dtype!r})")
    # OPT-IN half-accumulate: fp16 MMA accumulators (QK s0-s3, PV o, rescale/normalize)
    # + their backing threadgroup scratch (tg_S scores, on_scratch). fp16 only —
    # accumulating fp32 inputs in half would silently truncate the inputs themselves.
    if half_accumulate and elem != "half":
        raise ValueError(
            f"half_accumulate=True requires fp16/f16 out_dtype (half MMA accumulators); got out_dtype={out_dtype!r}"
        )
    if half_accumulate:
        acc_frag, acc_zero, acc_buf = "simdgroup_half8x8", "0.0h", "half"
    else:
        acc_frag, acc_zero, acc_buf = "simdgroup_float8x8", "0.0f", "float"
    Dc_tail = 8
    # Threadgroup-memory budget (32KB on M-series). tgQ (BM*head_dim) is the
    # head_dim-dependent term; the rest is fixed by BM/BN. Raise a CLEAR error early
    # instead of a cryptic AGX pipeline-state failure at load time. Empirically: fp16
    # fits head_dim<=~248, fp32<=~148 (qk=256 fp16 -> 33152B; qk=192 fp32 -> 38272B).
    _eb = 2 if elem == "half" else 4
    _tg_bytes = (
        BM * D * _eb  # tgQ
        + BM * BN * 4  # tg_S (float)
        + (BM * BN * 2 if elem == "half" else 0)  # tgP (fp16 only)
        + n_groups * 64 * 4  # on_scratch (float)
        + BM * 4 * 3  # tg_m / tg_l / tg_alpha
        + BM * 8 * 4 * 2  # tg_pmax / tg_psum (parallel softmax)
        + BN * Dc_tail * _eb  # tgKV
    )
    if _tg_bytes > 32768:
        raise ValueError(
            f"simd FA threadgroup memory {_tg_bytes}B exceeds the 32KB M-series limit "
            f"for head_dim={D} v_head_dim={Dv} {elem}. Reduce head_dim "
            f"(fp16 head_dim<=192, fp32 head_dim<=128 are validated)."
        )
    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "o_sz",
        "o_sh",
        "o_sm",
        "o_sk",
        "Z",
        "H",
        "N_CTX",
    ]
    # Biased/triangle-attention extensions (bias/mask/lse/runtime_scale/grid_3d/
    # mask_batch_div): all no-ops when off, so a plain-FA emission is byte-identical.
    # These mirror the tiled template's fusions (bias into the score, mask -> -inf,
    # per-row lse, runtime scale, 3-D grid, cross-head mask divisor) but on the MMA
    # kernel — the fast path for trifast. Extra uint stride aliases append to _LOGICAL;
    # the Bias/Mask/Lse pointers come from the routed arg_decls (route-only path).
    _bias_uint = []
    if bias:
        _bias_uint += ["b_sz", "b_sh", "b_sm", "b_sn"]
    if mask:
        _bias_uint += ["mask_sz", "mask_sh", "mask_sn"]
    if lse:
        _bias_uint += ["lse_sz", "lse_sh", "lse_sm"]
    _LOGICAL = _LOGICAL + _bias_uint
    if (arg_decls is None) != (bindings is None):
        raise ValueError("arg_decls and bindings must be provided together")
    if (bias or mask or lse or runtime_scale or grid_3d) and arg_decls is None:
        raise ValueError("simd FA biased/3-D extensions are route-only (need arg_decls/bindings)")
    if arg_decls is None:
        arg_decls = [
            f"    device const {elem}* Q [[buffer(0)]]",
            f"    device const {elem}* K [[buffer(1)]]",
            f"    device const {elem}* V [[buffer(2)]]",
            f"    device {elem}* Out [[buffer(3)]]",
        ]
        for i, nm in enumerate(_LOGICAL):
            arg_decls.append(f"    constant uint& arg_{nm} [[buffer({4 + i})]]")
        bindings = {nm: f"arg_{nm}" for nm in _LOGICAL}
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {nm} = {bindings[nm]};" for nm in _LOGICAL)

    # ---- biased/3-D helper strings (interpolated as VALUES -> single braces) ----
    _scale_value = bindings["scale"] if runtime_scale else f"{SCALE!r}f"
    scale_decl = f"const float scale = {_fa_scale_chain_expr(_scale_value, scale_chain, round_q)};"
    if round_q:
        scale_decl += (
            "\n    const float s_scale = 1.0f;  // packet 164: the scale is applied to Q in the source's dtype"
        )
    if runtime_neg_inf:
        if not bindings or "neg_inf" not in bindings:
            raise ValueError("simd FA: runtime_neg_inf requires bindings['neg_inf']")
        # packet 114: the source sentinel (its exp domain, after the c_eff multiply) in
        # this template's natural units; every masked/tail cell holds it and counts in the sum.
        scale_decl += f"\n    const float neg_inf = ({bindings['neg_inf']}) * {float(neg_inf_unit)!r}f;"
    _NI = "neg_inf" if runtime_neg_inf else "-INFINITY"
    if grid_3d:
        grid_decode = "uint q_block = pid3.x, h = pid3.y, z = pid3.z;"
    else:
        grid_decode = "uint q_block = pid3.x, zh = pid3.y;\n    uint z = zh / H, h = zh % H;"
    _bb = []
    if bias:
        _bb.append("    uint bias_base = z*b_sz + h*b_sh;")
    if mask:
        _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"
        _bb.append(f"    uint mask_base = {_mz}*mask_sz + h*mask_sh;")
    if lse:
        _bb.append("    uint lse_base = z*lse_sz + h*lse_sh;")
    biased_base_lines = "\n".join(_bb)
    if lse:
        _lse_value = (
            "!isfinite(lv) ? NAN : ((lv > 0.0f) ? (tg_m[lid] + log(lv)) : -INFINITY)"
            if runtime_neg_inf
            else "(lv > 0.0f) ? (tg_m[lid] + log(lv)) : -INFINITY"
        )
        lse_store = (
            "\n    // ---- per-query log-sum-exp: lse = m + log(l) (natural log) ----\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (lid < BM) {\n"
            "        uint qr = q_start + lid;\n"
            "        if (qr < N_CTX) {\n"
            "            float lv = tg_l[lid];\n"
            f"            Lse[lse_base + qr*lse_sm] = {_lse_value};\n"
            "        }\n"
            "    }"
        )
    else:
        lse_store = ""

    # bias/mask cell fragments (guard-independent; combined with the guard in block()).
    _bias_add = "\n                    s += float(Bias[bias_base + qrow_b*b_sm + kvrow_b*b_sn]);" if bias else ""
    _mask_apply = "\n                    if (Mask[mask_base + kvrow_b*mask_sn] != 0) s = " + _NI + ";" if mask else ""

    if round_q not in (None, "half", "bfloat") or round_p not in (None, "half", "bfloat"):
        raise ValueError("round_q / round_p must be None, 'half' or 'bfloat'")
    # packet 164: the source's rounding points. round_q: stage elem(float(Q) * float(elem(scale)))
    # and do NOT scale the score tile again (the `* scale` below becomes `* 1.0f` — written as a
    # separate constant so the fp32 path stays byte-identical). round_p: round P before P @ V.
    _q_raw = "Q[q_base + qr*q_sm + (i%D)*q_sk]"
    # (elem == round_q on the half maker: one conversion rounds and stages)
    _q_stage = (
        (
            f"{round_q}(float({_q_raw}) * float({round_q}(scale)))"
            if elem == round_q
            else f"{elem}({round_q}(float({_q_raw}) * float({round_q}(scale))))"
        )
        if round_q
        else _q_raw
    )
    _s_scale = "s_scale" if round_q else "scale"
    if elem == "half":
        p_buffers = f"    threadgroup half tgP[{BM} * {BN}];\n    threadgroup {acc_buf} on_scratch[{n_groups} * 64];"
        p_write, p_src = "tgP[r*BN+cj] = half(p);", "tgP"  # fp16 P is rounded by construction (== round_p 'half')
        p_zero = "tgP[pr*BN+cj] = half(0.0f);"
        if round_p == "bfloat":
            raise ValueError("bf16 P rounding on the half simdgroup maker is not a source shape")
    else:
        p_buffers = f"    threadgroup float on_scratch[{n_groups} * 64];"
        p_write, p_src = (f"tg_S[r*BN+cj] = float({round_p}(p));" if round_p else "tg_S[r*BN+cj] = p;"), "tg_S"
        p_zero = "tg_S[pr*BN+cj] = 0.0f;"
    _score_store_update = (
        "                tg_S[pr*BN+cj] = s;\n                pm = (isnan(pm) || isnan(s)) ? NAN : max(pm, s);"
        if runtime_neg_inf
        else "                tg_S[pr*BN+cj] = s; pm = max(pm, s);"
    )
    _mnew_loop = (
        "                for (uint j = 0u; j < 8u; j++) {\n"
        "                    float pj = tg_pmax[r*8u+j];\n"
        "                    m_new = (isnan(m_new) || isnan(pj)) ? NAN : max(m_new, pj);\n"
        "                }"
        if runtime_neg_inf
        else "                for (uint j = 0u; j < 8u; j++) m_new = max(m_new, tg_pmax[r*8u+j]);"
    )
    # A nonfinite row is valid source data on every route, not only the biased
    # runtime-sentinel spelling.  Never feed its P/alpha through the diagonal
    # 8x8 MMA: IEEE 0*NaN contaminates the seven sibling rows.  Carry a finite
    # internal zero for the bad row and restore its NaN at the final store.
    _nonfinite_sanitize = (
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "            // Keep a nonfinite source row local across the 8x8 MMAs.\n"
        "            if (!isfinite(tg_l[pr])) {\n"
        f"                for (uint cj = cA; cj < cB; cj++) {{ {p_zero} }}\n"
        "            }"
    )
    # A diagonal MMA is not an elementwise scale for nonfinite O: its
    # off-diagonal zero products introduce 0*Inf / 0*NaN from sibling rows.
    # Materialize each fragment in its SIMD group's private scratch instead;
    # this also avoids depending on an undocumented fragment-to-lane mapping.
    _alpha_rescale = f"""for (uint rb=0u;rb<4u;rb++) {{
                simdgroup_store(o[rb][t], on_scratch + sgitg*64u, 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e=lid%32u;e<64u;e+=32u) {{
                    float a=tg_alpha[rb*8u+e/8u];
                    on_scratch[sgitg*64u+e] = {acc_buf}(on_scratch[sgitg*64u+e] * {acc_buf}(isfinite(a) ? a : 0.0f));
                }}
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(o[rb][t], on_scratch + sgitg*64u, 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }}"""

    # Batched scratch is valid only when P lives separately in tgP and the
    # score tile has float storage. Other variants retain the original path.
    # The unconditional threadgroup barrier immediately before ALPHA_RESCALE
    # completes every score reader. Each SIMD group then owns 256 disjoint
    # float slots; 8 groups * 4 fragments * 64 slots == BM * BN == 2048.
    # The end-of-key-block threadgroup barrier protects the next score store.
    if elem == "half" and not half_accumulate:
        _alpha_rescale = (
            "// Batch four row fragments in the now-dead float score tile.\n            "
            + _fa_row_loop(
                "\n                simdgroup_store(o[@ROW@][t], tg_S + sgitg*256u + @ROW@*64u, 8u);\n            ",
                _expanded,
            )
            + "\n            simdgroup_barrier(mem_flags::mem_threadgroup);\n            "
            + _fa_row_loop(
                "\n                for (uint e=lid%32u;e<64u;e+=32u) {\n                    float a=tg_alpha[@ROW@*8u+e/8u];\n                    tg_S[sgitg*256u+@ROW@*64u+e] = float(tg_S[sgitg*256u+@ROW@*64u+e] * float(isfinite(a) ? a : 0.0f));\n                }\n            ",
                _expanded,
            )
            + "\n            simdgroup_barrier(mem_flags::mem_threadgroup);\n            "
            + _fa_row_loop(
                "\n                simdgroup_load(o[@ROW@][t], tg_S + sgitg*256u + @ROW@*64u, 8u);\n            ",
                _expanded,
            )
            + "\n            simdgroup_barrier(mem_flags::mem_threadgroup);"
        )

    _before_fragments = _after_fragments = ""
    if fragment_lowering in ("private", "private-output"):
        _before_fragments = (
            "        // CPU prototype: broadcast each row's alpha into an identically typed fragment.\n"
            "        // Each SIMD group owns 256 slots; no cross-group scratch reader exists here.\n"
            "        "
            + _fa_row_loop(
                "\n            for (uint e=lid%32u;e<64u;e+=32u) {\n"
                "                float a=tg_alpha[@ROW@*8u+e/8u];\n"
                "                tg_S[sgitg*256u+@ROW@*64u+e] = isfinite(a) ? a : 0.0f;\n"
                "            }\n        ",
                True,
            )
            + "\n"
            "        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        )
        _alpha_rescale = (
            "// Same-type storage multiplication: no diagonal MMA and no lane-row formula.\n            "
            + _fa_row_loop(
                "\n                simdgroup_float8x8 alpha_fragment;\n"
                "                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + @ROW@*64u, 8u);\n"
                "                o[@ROW@][t].thread_elements() = o[@ROW@][t].thread_elements() * alpha_fragment.thread_elements();\n            ",
                True,
            )
        )
    elif fragment_lowering in ("shared", "shared-output"):
        _after_fragments = (
            "        // Replicate each row coefficient across one 8-wide matrix row.\n"
            "        float row_alpha = tg_alpha[lid/8u];\n"
            "        on_scratch[lid] = isfinite(row_alpha) ? row_alpha : 0.0f;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        )
        _alpha_rescale = (
            "// Same-type fragments pair elements without a lane-coordinate formula.\n            "
            + _fa_row_loop(
                "\n                simdgroup_float8x8 coefficient;\n"
                "                simdgroup_load(coefficient, on_scratch + @ROW@*64u, 8u);\n"
                "                o[@ROW@][t].thread_elements() *= coefficient.thread_elements();\n            ",
                True,
            )
        )

    # An explicit dense-source proof may retain zero-K scores beyond N_CTX.
    # Keep this opt-in: biased/MLA/standalone contracts must not inherit it.
    _source_tail = None
    if source_kv_block is not None:
        if bias or mask or runtime_neg_inf or (causal and not mask_tail_scores):
            raise ValueError("source KV tail replay requires plain dense attention")
        _source_tail = _fa_source_tail_guard(source_kv_block, BLOCK_N)
    if unmasked_kv and (_source_tail is None or not set(unmasked_kv) <= {"K", "V"}):
        raise ValueError("unmasked KV roles require the proved source block")
    if source_causal_stop is not None and (_source_tail is None or type(source_causal_stop) is not bool):
        raise ValueError("source loop stop requires a proved dense source block")
    _stop = "source_kv_stop" if source_causal_stop else "N_CTX"
    _visited = (
        _source_tail.replace("N_CTX", _stop).replace("kv_row", "kvr").replace("source KV tail:", "source KV visit:")
        if _source_tail
        else None
    )
    _k_tail = _visited if "K" in unmasked_kv else "(kvr < N_CTX)"
    _v_tail = _visited if "V" in unmasked_kv else "kvr < N_CTX"
    if source_causal_stop:
        _k_tail = f"({_k_tail} && {_visited})" if "K" not in unmasked_kv else _k_tail
        _v_tail = f"({_v_tail} && {_visited})" if "V" not in unmasked_kv else _v_tail
    _k_tail, _v_tail = _k_tail.replace("%", "%%"), _v_tail.replace("%", "%%")

    # ---- per-block body emitter (mode in {'full','tail'}) ----
    def guard_decls(mode):
        if mode == "full" and not causal:
            return "true", "", ""
        if mode == "full" and causal:
            return "(kv_row <= q_row)", "uint q_row = q_start + r;", "uint kv_row = kv_start + cj;"
        if mode == "tail" and not causal:
            return (_source_tail if not mask_tail_scores else "(kv_row < N_CTX)"), "", "uint kv_row = kv_start + cj;"
        return "((kv_row < N_CTX) && (kv_row <= q_row))", "uint q_row = q_start + r;", "uint kv_row = kv_start + cj;"

    def block(mode):
        g, qd, kd = guard_decls(mode)
        # Score cell: plain FA keeps the original ternary (byte-identical); the biased
        # path adds bias into the score + a mask -> -inf inside the valid branch. qrow_b
        # / kvrow_b index the bias/mask (q_start/kv_start/pr/cj all in scope here).
        if bias or mask:
            score_cell = (
                "uint qrow_b = q_start + pr; uint kvrow_b = kv_start + cj;\n"
                "                float s;\n"
                "                if (" + g + ") {\n"
                "                    s = tg_S[pr*BN+cj]*" + _s_scale + ";" + _bias_add + _mask_apply + "\n"
                "                } else { s = " + _NI + "; }"
            )
        else:
            score_cell = "float s = " + g + " ? (tg_S[pr*BN+cj]*" + _s_scale + ") : " + _NI + ";"
        if mode == "full":
            kload = "            simdgroup_load(kf, K + k_base + (kv_start + sgitg*8u)*k_sn + kc*k_sk, k_sn, 0, true);"
        else:
            kload = (
                "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "            for (uint e = lid; e < %(BN)du*%(Dct)du; e += NT) {\n"
                "                uint rr = e / %(Dct)du, cc = e %% %(Dct)du; uint kvr = kv_start + rr;\n"
                "                tgKV[rr*%(Dct)du + cc] = "
                + _k_tail
                + " ? K[k_base + kvr*k_sn + (kc+cc)*k_sk] : %(elem)s(0);\n"
                "            }\n"
                "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "            simdgroup_load(kf, tgKV + (sgitg*8u)*%(Dct)du, %(Dct)du, 0, true);\n"
                "            threadgroup_barrier(mem_flags::mem_threadgroup);"
            ) % {"BN": BN, "Dct": Dc_tail, "elem": elem}
        if mode == "full":
            vload = (
                "            for (uint kk=0u;kk<BN;kk+=8u)\n"
                "                simdgroup_load(vfs[kk/8u], V + v_base + (kv_start + kk)*v_sn + ("
                + _ct_off
                + ")*v_sk, v_sn);"
            )
        else:
            vload = (
                "            for (uint kk=0u;kk<BN;kk+=8u) {\n"
                "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "                for (uint e = lid%%32u; e < 64u; e += 32u) {\n"
                "                    uint rr = e / 8u, cc = e %% 8u; uint kvr = kv_start + kk + rr;\n"
                "                    tgKV[sgitg*64u + rr*8u + cc] = ("
                + _v_tail
                + _ct_tail_g
                + ") ? V[v_base + kvr*v_sn + (ct*8u + cc)*v_sk] : %(elem)s(0);\n"
                "                }\n"
                "                simdgroup_barrier(mem_flags::mem_threadgroup);\n"
                "                simdgroup_load(vfs[kk/8u], tgKV + sgitg*64u, 8u);\n"
                "            }\n"
                "            threadgroup_barrier(mem_flags::mem_threadgroup);"
            ) % {"elem": elem}
        tmpl = """    {
        %(ACC)s s0(%(AZ)s), s1(%(AZ)s), s2(%(AZ)s), s3(%(AZ)s);
        %(KVFRAG)s qf, kf;
        for (uint kc = 0u; kc < D; kc += %(DCT)du) {
%(KLOAD)s
            simdgroup_load(qf, tgQ + 0u*D + kc, D);  simdgroup_multiply_accumulate(s0, qf, kf, s0);
            simdgroup_load(qf, tgQ + 8u*D + kc, D);  simdgroup_multiply_accumulate(s1, qf, kf, s1);
            simdgroup_load(qf, tgQ + 16u*D + kc, D); simdgroup_multiply_accumulate(s2, qf, kf, s2);
            simdgroup_load(qf, tgQ + 24u*D + kc, D); simdgroup_multiply_accumulate(s3, qf, kf, s3);
        }
        simdgroup_store(s0, tg_S + 0u*BN + sgitg*8u, BN);
        simdgroup_store(s1, tg_S + 8u*BN + sgitg*8u, BN);
        simdgroup_store(s2, tg_S + 16u*BN + sgitg*8u, BN);
        simdgroup_store(s3, tg_S + 24u*BN + sgitg*8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Parallel online-softmax: all 256 threads (8 per row, row=lid/8) scan
            // BN/8 cols each and reduce partial max/sum via tg_pmax/tg_psum scratch —
            // vs the old 32-thread serial BN-col scan (was ~14%% of block time).
            // Masked cols get -INFINITY in the max pass, so exp() = 0 in the sum pass:
            // the causal/tail GUARD is preserved without a second guard.
            uint pr = lid / 8u, pc = lid %% 8u; %(QD_PR)s
            uint cA = pc * (BN / 8u), cB = cA + (BN / 8u);
            float pm = -INFINITY;
            for (uint cj = cA; cj < cB; cj++) {
                %(KD)s
                %(SCORE_CELL)s
%(SCORE_STORE_UPDATE)s
            }
            tg_pmax[pr*8u + pc] = pm;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float m_prev = tg_m[r], l_prev = tg_l[r], m_new = m_prev;
%(MNEW_LOOP)s
                tg_m[r] = m_new; tg_alpha[r] = exp(m_prev - m_new); tg_l[r] = l_prev * tg_alpha[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float ps = 0.0f; float mr = tg_m[pr];
            for (uint cj = cA; cj < cB; cj++) {
                float p = exp(tg_S[pr*BN+cj] - mr); %(PWRITE_PR)s ps += p;
            }
            tg_psum[pr*8u + pc] = ps;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float l = tg_l[r];
                for (uint j = 0u; j < 8u; j++) l += tg_psum[r*8u+j];
                tg_l[r] = l;
            }
%(NONFINITE_SANITIZE)s        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
%(BEFORE_FRAGMENTS)s        %(PLOADT)s pf, vf, vfs[%(BN)d/8];
%(AFTER_FRAGMENTS)s        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*%(NG)du;
            %(ALPHA_RESCALE)s
%(VLOAD)s
            for (uint kk=0u;kk<BN;kk+=8u) {
                vf = vfs[kk/8u];
                simdgroup_load(pf, %(PSRC)s + 0u*BN + kk, BN);  simdgroup_multiply_accumulate(o[0][t], pf, vf, o[0][t]);
                simdgroup_load(pf, %(PSRC)s + 8u*BN + kk, BN);  simdgroup_multiply_accumulate(o[1][t], pf, vf, o[1][t]);
                simdgroup_load(pf, %(PSRC)s + 16u*BN + kk, BN); simdgroup_multiply_accumulate(o[2][t], pf, vf, o[2][t]);
                simdgroup_load(pf, %(PSRC)s + 24u*BN + kk, BN); simdgroup_multiply_accumulate(o[3][t], pf, vf, o[3][t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }"""
        return tmpl % {
            "BEFORE_FRAGMENTS": _before_fragments,
            "AFTER_FRAGMENTS": _after_fragments,
            "ACC": acc_frag,
            "AZ": acc_zero,
            "KVFRAG": kv_frag,
            "DCT": Dc_tail,
            "KLOAD": kload,
            "QD": qd,
            "QD_PR": qd.replace("q_start + r", "q_start + pr"),  # q_row for row = pr
            "KD": kd,
            "GUARD": g,
            "SCORE_CELL": score_cell,
            "PWRITE": p_write,
            "PWRITE_PR": p_write.replace("[r*BN", "[pr*BN"),  # prob store for row = pr
            "SCORE_STORE_UPDATE": _score_store_update,
            "MNEW_LOOP": _mnew_loop,
            "NONFINITE_SANITIZE": _nonfinite_sanitize,
            "ALPHA_RESCALE": _alpha_rescale,
            "PLOADT": p_load_t,
            "BN": BN,
            "NG": n_groups,
            "VLOAD": vload,
            "PSRC": p_src,
        }

    block_full, block_tail = block("full"), block("tail")
    # Route-lowered biased kernels always set runtime_neg_inf: their source
    # recurrence distinguishes a nonfinite denominator from an empty row.  The
    # standalone biased maker predates that source contract and deliberately
    # maps a fully masked row to zero, so preserve it when no runtime sentinel
    # is present.  Plain/MLA routes cannot form an empty masked row; a
    # nonfinite denominator there necessarily came from source data.
    _preserve_nonfinite = runtime_neg_inf or not bias
    _normalized = (
        f"{acc_buf}(on_scratch[sgitg*64u+e] * {acc_buf}((tg_l[rb*8u+dr]>0.0f) ? (1.0f/tg_l[rb*8u+dr]) : 0.0f))"
    )
    _final_value = f"!isfinite(tg_l[rb*8u+dr]) ? NAN : {_normalized}" if _preserve_nonfinite else _normalized
    final_store = (
        "            simdgroup_store(o[rb][t], on_scratch + sgitg*64u, 8u);\n"
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "            for (uint e=lid%32u;e<64u;e+=32u) {\n"
        "                uint dr=e/8u, dc=e%8u;\n"
        "                uint qr2=(q_start+rb*8u+dr), dc2=(ct*8u+dc);\n"
        "                if (qr2 < N_CTX" + _ct_store_g + ")\n"
        f"                    Out[o_base + qr2*o_sm + dc2*o_sk] = {store_cast(_final_value)};\n"
        "            }\n"
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )

    if fragment_lowering in ("output", "private-output", "shared-output"):
        # on_scratch is group-private here. The end-of-block group barrier
        # already retires all shared coefficient readers before this reuse.
        final_store = final_store.replace("threadgroup_barrier(", "simdgroup_barrier(")
    _initialization = _fa_row_loop(
        f"for (uint t=0u;t<TPG;t++) o[@ROW@][t]={acc_frag}({acc_zero if _initial_acc == '0.0f' else _initial_acc});",
        _expanded,
        single=True,
    )
    _final_rows = _fa_row_loop(
        "\n        for (uint t=0u;t<TPG;t++) {\n"
        f"            uint ct = sgitg + t*{n_groups}u;\n" + final_store.replace("rb", "@ROW@") + "\n        }\n    ",
        _expanded,
    )

    # Causal work-skipping: a q-block covers rows [q_start, q_start+BM); any kv-block
    # starting past the last row (kv_start > q_start+BM-1) is FULLY masked, so the
    # kv-loop can stop there (and the tail block runs only if it can hold an unmasked
    # element). Removes the wasted upper-triangle work -> ~1.3-1.7x on causal, pushing
    # every N>=1024 case above SDPA. Only prunes provably-all-masked blocks, so the
    # per-element mask still handles the partially-masked diagonal block -> identical
    # numerics. Empty for non-causal (every block is needed).
    causal_break = (
        "        if (kv_start > q_start + BM - 1u) break;  // causal: skip fully-masked blocks\n" if causal else ""
    )
    causal_tail_guard = " && n_full * BN <= q_start + BM - 1u" if causal else ""
    _source_stop_decl = ""
    if source_causal_stop is not None:
        # A score mask cannot justify skipping source loads: 0 * NaN is NaN.
        # Conversely, a wider physical tile must not read beyond the SOURCE's
        # visited KV iterations. The tail staging predicates above enforce it.
        causal_break = causal_tail_guard = ""
        if source_causal_stop:
            _source_stop_decl = "\n    const uint source_kv_stop = min(q_start + BM, N_CTX); // source KV loop stop"

    head = f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// simdgroup-MMA FlashAttention-2 ({elem} in/out, {"fp16-accumulate" if half_accumulate else "fp32"} compute, {"causal" if causal else "non-causal"}).
// Loop-split: branchless full-block fast loop + separate staged/masked tail block.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {{
    const uint BM = {BM}u, BN = {BN}u, D = {D}u, NT = {NT}u, TPG = {TPG}u;
    {scale_decl}
{bind_lines}
    {grid_decode}
    uint q_start = q_block * BM;{_source_stop_decl}
    uint q_base = z*q_sz+h*q_sh, k_base = z*k_sz+h*k_sh, v_base = z*v_sz+h*v_sh, o_base = z*o_sz+h*o_sh;
{biased_base_lines}

    threadgroup {elem} tgQ[{BM} * {D}];
    threadgroup {acc_buf}  tg_S[{BM} * {BN}];
{p_buffers}
    threadgroup float  tg_m[{BM}], tg_l[{BM}], tg_alpha[{BM}];
    threadgroup float  tg_pmax[{BM} * 8], tg_psum[{BM} * 8];  // parallel-softmax partials
    threadgroup {elem} tgKV[{BN} * {Dc_tail}u];

    {acc_frag} o[4][TPG];
    {_initialization}

    for (uint i = lid; i < BM*D; i += NT) {{
        uint qr = q_start + i/D;
        tgQ[i] = (qr < N_CTX) ? {_q_stage} : {elem}(0);
    }}
    if (lid < BM) {{ tg_m[lid]=-INFINITY; tg_l[lid]={_initial_den}; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_full = {_stop} / BN;            // floor: count of FULL kv-blocks
    for (uint kv_block = 0u; kv_block < n_full; kv_block++) {{
        uint kv_start = kv_block * BN;
{causal_break}{block_full}
    }}
    if (n_full * BN < {_stop}{causal_tail_guard}) {{          // one partial tail block (masked staging)
        uint kv_start = n_full * BN;
{block_tail}
    }}
{lse_store}

    {_final_rows}
}}
"""
    return head


def _bwd_kv_subtile_size(head_dim, block_j, block_k):
    """Largest power-of-two divisor KS of block_k such that the scalar bwd_kv kernel
    fits Metal's limits: TPG = KS*head_dim <= 1024 threads AND the threadgroup footprint
    (4*KS*D staged K/V/dK/dV + BJ*D staged Q + 2*BJ*KS scores/dS, fp32) < 32 KB. Returns
    KS, or None if even KS=1 does not fit. KS == block_k for head_dim<=32 (single pass)."""
    D, BJ = head_dim, block_j

    def _fits(ks):
        tpg = ks * D
        mem = (4 * ks * D + BJ * D + 2 * BJ * ks) * 4 + (2 * BJ + ks + 16) * 4
        return tpg <= 1024 and mem <= 32768

    # KS must DIVIDE block_k — the subtile loop steps by KS through block_k, so a KS that
    # does not divide it would overshoot the tile and double-write dK/dV for the next
    # k-block's rows. Triton's tl.arange already forces power-of-2 blocks (for which the
    # largest fitting divisor is a power of 2), but iterate TRUE divisors descending so the
    # helper's own invariant holds regardless of caller — never return an overshooting KS.
    for ks in range(block_k, 0, -1):
        if block_k % ks == 0 and _fits(ks):
            return ks
    return None


def _bwd_output_casts(output_casts, roles, legacy_cast):
    """Closed per-store replay contract; None preserves the direct-maker legacy API.

    The compiler always supplies a complete map proved against each store pointer.
    Do not fill a partial map from another output's dtype: that was the 174 bug.
    """
    if output_casts is None:
        return tuple(legacy_cast for _ in roles)
    if not isinstance(output_casts, dict) or set(output_casts) != set(roles):
        raise ValueError(f"backward output casts require exactly the roles {roles!r}")
    if any(t not in (None, "half", "bfloat") for t in output_casts.values()):
        raise ValueError("backward output casts must be None, half or bfloat")
    return tuple((lambda e, t=output_casts[r]: f"{t}({e})" if t else e) for r in roles)


def make_flash_attention_bwd_kv_kernel(
    head_dim=32,
    BLOCK_J=32,
    BLOCK_K=32,
    out_dtype="fp32",
    round_k=None,
    round_p=None,
    round_ds=None,
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention_bwd_kv",
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    output_casts=None,
):
    """Tiled FA-2 BACKWARD dK/dV for biased/triangle attention (route-only ABI).

    Per (k-block, i, batch*heads) threadgroup: recompute P from the saved lse, then
    accumulate over j-blocks
        dV[k,d] += sum_j P[j,k] * dO[j,d]
        dP[j,k]  = sum_d dO[j,d] * V[k,d];  dS[j,k] = P[j,k] * (dP[j,k] - delta[j])
        dK[k,d] += sum_j dS[j,k] * Q[j,d]
    and finally dK *= sm_scale. P[j,k] = exp2((sm_scale*Q@Kᵀ + bias - lse[j]) * inv_ln2)
    (== softmax prob; row_max stored by the fwd IS the natural lse). dO is read from
    device (its 4 KB stays cached) so the threadgroup budget is 7*[BLOCK*head_dim]
    fp32 buffers = ~28 KB < 32 KB. delta[j] = rowsum(O*dO) is precomputed by _bwd_q.
    Grid: (n_k_blocks, I, Z*H); bias shared across i (0 stride) + cross-head mask
    (z/H_heads) handled exactly as the forward biased template.
    """
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"bwd_kv out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    if arg_decls is None or bindings is None:
        raise ValueError("make_flash_attention_bwd_kv_kernel is route-only (needs arg_decls/bindings)")

    store_cast_k, store_cast_v = _bwd_output_casts(output_casts, ("dk", "dv"), store_cast)
    D, BJ, BK = head_dim, BLOCK_J, BLOCK_K
    # packet 174: the source's rounding points (None = fp32 source, text unchanged).
    _rd = lambda t: (lambda e: f"float({t}({e}))") if t else (lambda e: e)
    _scale_r = (
        "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))"
        if round_k == "bfloat"
        else f"float({round_k}(scale))"
    )
    _k_stage = (lambda e: f"float({round_k}(float({e}) * {_scale_r}))") if round_k else (lambda e: f"float({e})")
    _s_scaled = "s" if round_k else "s*scale"
    _p_rd, _ds_rd = _rd(round_p), _rd(round_ds)
    # K-SUB-TILE so a large head_dim fits Metal's limits: TPG = KS*D <= 1024 threads
    # and the threadgroup footprint < 32 KB (see _bwd_kv_subtile_size). KS == BK for the
    # validated head_dim<=32 case (a SINGLE subtile) -> emitted kernel semantically
    # identical to the pre-subtile template; the j-loop re-runs per k-subtile otherwise.
    KS = _bwd_kv_subtile_size(D, BJ, BK)
    if KS is None:
        raise ValueError(
            f"bwd_kv: head_dim={D} too large to tile within 32 KB / 1024 threads (BLOCK_J={BJ}, BLOCK_K={BK})"
        )
    TPG = KS * D
    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "b_sz",
        "b_sh",
        "b_sm",
        "b_sn",
        "mask_sz",
        "mask_sh",
        "mask_sn",
        "lse_sz",
        "lse_sh",
        "lse_sm",
        "dlt_sz",
        "dlt_sh",
        "dlt_sm",
        "do_sz",
        "do_sh",
        "do_sm",
        "do_sk",
        "dk_sz",
        "dk_sh",
        "dk_sn",
        "dk_sk",
        "dv_sz",
        "dv_sh",
        "dv_sn",
        "dv_sk",
        "H",
        "N_CTX",
        "scale",
    ]
    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"bwd_kv bindings missing {missing}")
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {n} = {bindings[n]};" for n in _LOGICAL if n != "scale")
    scale_decl = f"const float scale = {bindings['scale']};"
    if runtime_neg_inf:
        # packet 134: the source applies ONE sentinel to masked / out-of-range score
        # cells and then subtracts the row lse; replay it on in-bounds masked cells so
        # a NaN lse propagates exactly as in the source (finite data: exp2 underflows
        # to 0, bit-identical to the old hard-coded 0).
        if not bindings or "neg_inf" not in bindings:
            raise ValueError(f"{kernel_name}: runtime_neg_inf requires bindings['neg_inf']")
        scale_decl += f"\n    const float neg_inf = {bindings['neg_inf']};"
    _masked_cond = "jrow < N_CTX && krow < N_CTX" if runtime_neg_inf else "false"
    _masked_p = "exp2((neg_inf - tg_lse[jj]) * inv_ln2)" if runtime_neg_inf else "0.0f"
    if grid_3d:
        grid_decode = "uint h = pid3.y; uint z = pid3.z;"
    else:
        grid_decode = "uint zh = pid3.y; uint z = zh / H; uint h = zh % H;"
    _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"

    return f"""#include <metal_stdlib>
using namespace metal;

// Tiled FA-2 backward dK/dV ({elem} in/out, fp32 compute) for biased attention.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BJ = {BJ}u, BK = {BK}u, KS = {KS}u, D = {D}u, TPG = {TPG}u;
    const float inv_ln2 = 1.4426950408889634f;
    {scale_decl}
{bind_lines}
    uint k_block = pid3.x; {grid_decode}
    uint k_start = k_block * BK;

    uint q_base   = z*q_sz  + h*q_sh;
    uint k_base   = z*k_sz  + h*k_sh;
    uint v_base   = z*v_sz  + h*v_sh;
    uint do_base  = z*do_sz + h*do_sh;
    uint dk_base  = z*dk_sz + h*dk_sh;
    uint dv_base  = z*dv_sz + h*dv_sh;
    uint lse_base = z*lse_sz + h*lse_sh;
    uint dlt_base = z*dlt_sz + h*dlt_sh;
    uint bias_base = z*b_sz + h*b_sh;
    uint mask_base = {_mz}*mask_sz + h*mask_sh;

    threadgroup float tg_K[{KS} * {D}];
    threadgroup float tg_V[{KS} * {D}];
    threadgroup float dk_acc[{KS} * {D}];
    threadgroup float dv_acc[{KS} * {D}];
    threadgroup float tg_Q[{BJ} * {D}];
    threadgroup float tg_P[{BJ} * {KS}];
    threadgroup float tg_dS[{BJ} * {KS}];
    threadgroup float tg_lse[{BJ}];
    threadgroup float tg_delta[{BJ}];
    threadgroup uchar tg_mask[{KS}];

    uint n_j = (N_CTX + BJ - 1u) / BJ;
    // Process the BK k-rows in KS-sized subtiles (KS == BK -> single pass for
    // head_dim<=32). Each subtile stages its own K/V, re-runs the j-loop, stores DK/DV.
    for (uint ksub = 0u; ksub < BK; ksub += KS) {{
    uint ks_start = k_start + ksub;
    for (uint i = lid; i < KS*D; i += TPG) {{ dk_acc[i] = 0.0f; dv_acc[i] = 0.0f; }}
    for (uint i = lid; i < KS*D; i += TPG) {{
        uint kk = i / D, dd = i % D; uint krow = ks_start + kk;
        tg_K[i] = (krow < N_CTX) ? {_k_stage("K[k_base + krow*k_sn + dd*k_sk]")} : 0.0f;
        tg_V[i] = (krow < N_CTX) ? float(V[v_base + krow*v_sn + dd*v_sk]) : 0.0f;
    }}
    for (uint i = lid; i < KS; i += TPG) {{
        uint krow = ks_start + i;
        tg_mask[i] = (krow < N_CTX) ? Mask[mask_base + krow*mask_sn] : (uchar)1;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint jb = 0u; jb < n_j; jb++) {{
        uint j_start = jb * BJ;
        for (uint i = lid; i < BJ*D; i += TPG) {{
            uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
            tg_Q[i] = (jrow < N_CTX) ? float(Q[q_base + jrow*q_sm + dd*q_sk]) : 0.0f;
        }}
        for (uint i = lid; i < BJ; i += TPG) {{
            uint jrow = j_start + i;
            tg_lse[i]   = (jrow < N_CTX) ? Lse[lse_base + jrow*lse_sm] : 0.0f;
            tg_delta[i] = (jrow < N_CTX) ? Delta[dlt_base + jrow*dlt_sm] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scores -> P[j,k] = exp2((scale*Q@Kᵀ + bias - lse) * inv_ln2)
        for (uint i = lid; i < BJ*KS; i += TPG) {{
            uint jj = i / KS, kk = i % KS; uint jrow = j_start + jj, krow = ks_start + kk;
            if (jrow < N_CTX && krow < N_CTX && tg_mask[kk] == 0) {{
                float s = 0.0f;
                for (uint d = 0u; d < D; d++) s += tg_Q[jj*D + d] * tg_K[kk*D + d];
                s = {_s_scaled} + float(Bias[bias_base + jrow*b_sm + krow*b_sn]);
                tg_P[i] = exp2((s - tg_lse[jj]) * inv_ln2);
            }} else if ({_masked_cond}) {{
                tg_P[i] = {_masked_p};
            }} else {{
                tg_P[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dV[k,d] += sum_j P[j,k] * dO[j,d]   (dO read from device)
        for (uint i = lid; i < KS*D; i += TPG) {{
            uint kk = i / D, dd = i % D;
            float acc = 0.0f;
            for (uint jj = 0u; jj < BJ; jj++) {{
                uint jrow = j_start + jj;
                if (jrow < N_CTX) acc += {_p_rd("tg_P[jj*KS + kk]")} * float(dO[do_base + jrow*do_sm + dd*do_sk]);
            }}
            dv_acc[i] += acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dP[j,k] = sum_d dO[j,d]*V[k,d];  dS = P*(dP - delta[j])
        for (uint i = lid; i < BJ*KS; i += TPG) {{
            uint jj = i / KS, kk = i % KS; uint jrow = j_start + jj;
            // packet 134: no P != 0 fast-skip — the source computes dS = P*(dP - delta) on
            // every cell, and 0 * (NaN - delta) is NaN (a NaN dO row must poison masked
            // columns too). For finite data 0 * finite == 0, bit-identical to the skip.
            {{
                float dp = 0.0f;
                // packet 136: the source's dO load is masked (other = 0) beyond N_CTX; a physical
                // padding row must never be read (0 * NaN is NaN). In-range cells stay unconditional.
                if (jrow < N_CTX) for (uint d = 0u; d < D; d++) dp += float(dO[do_base + jrow*do_sm + d*do_sk]) * tg_V[kk*D + d];
                tg_dS[i] = {_ds_rd("tg_P[i] * (dp - tg_delta[jj])")};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dK[k,d] += sum_j dS[j,k] * Q[j,d]
        for (uint i = lid; i < KS*D; i += TPG) {{
            uint kk = i / D, dd = i % D;
            float acc = 0.0f;
            for (uint jj = 0u; jj < BJ; jj++) acc += tg_dS[jj*KS + kk] * tg_Q[jj*D + dd];
            dk_acc[i] += acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint i = lid; i < KS*D; i += TPG) {{
        uint kk = i / D, dd = i % D; uint krow = ks_start + kk;
        if (krow < N_CTX) {{
            DK[dk_base + krow*dk_sn + dd*dk_sk] = {store_cast_k("dk_acc[i] * scale")};
            DV[dv_base + krow*dv_sn + dd*dv_sk] = {store_cast_v("dv_acc[i]")};
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""


def make_flash_attention_bwd_kv_kernel_simd(
    head_dim=32,
    BLOCK_J=32,
    BLOCK_K=32,
    out_dtype="fp32",
    round_k=None,
    round_p=None,
    round_ds=None,
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention_bwd_kv",
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    output_casts=None,
):
    """simdgroup-MMA FA-2 backward dK/dV (same ABI/bindings as the scalar template).

    NG=4 simdgroups (TPG=128) tile the 4 matmuls with 8x8 hardware fragments:
        S[j,k]  = Q[j,:] @ K[k,:]ᵀ            (simdgroup owns k-col-band sgitg)
        dV[k,d] += P[j,k]ᵀ @ dO[j,d]          (owns d-col-band sgitg, reg-accumulated)
        dP[j,k]  = dO[j,:] @ V[k,:]ᵀ           (owns k-col-band sgitg)
        dK[k,d] += dS[j,k]ᵀ @ Q[j,d]          (owns d-col-band sgitg, reg-accumulated)
    P/dS recompute is scalar over the [BJ,BK] tile. dO is STAGED per j-block (guarded
    → tail-safe, no OOB) into tg_dO; dV/dK accumulate in registers (dv_f/dk_f) across
    j-blocks, freeing the acc buffers so tg_dO fits (6 fp32 [32²] tiles ~24 KB). Gated
    to BJ,BK,head_dim all %8 and <=32 (else the scalar template). fp32 interior compute.
    """
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"bwd_kv(simd) out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    if arg_decls is None or bindings is None:
        raise ValueError("make_flash_attention_bwd_kv_kernel_simd is route-only (needs arg_decls/bindings)")

    store_cast_k, store_cast_v = _bwd_output_casts(output_casts, ("dk", "dv"), store_cast)
    D, BJ, BK = head_dim, BLOCK_J, BLOCK_K
    # packet 174: the source's rounding points (None = fp32 source, text unchanged).
    _rd = lambda t: (lambda e: f"float({t}({e}))") if t else (lambda e: e)
    _scale_r = (
        "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))"
        if round_k == "bfloat"
        else f"float({round_k}(scale))"
    )
    _k_stage = (lambda e: f"float({round_k}(float({e}) * {_scale_r}))") if round_k else (lambda e: f"float({e})")
    _s_scaled = "s" if round_k else "s*scale"
    _p_rd, _ds_rd = _rd(round_p), _rd(round_ds)
    if not (D % 8 == 0 and BJ % 8 == 0 and BK % 8 == 0 and D <= 32 and BJ <= 32 and BK <= 32):
        raise ValueError("bwd_kv(simd) requires BJ,BK,head_dim %8 and <=32")
    # With round_p the dV dot must read the ROUNDED P while dS uses the fp32 P: dS is computed
    # first (its dP dot does not depend on P), P is rounded in place, then dV = Pᵀ @ dO. Without
    # round_p the original order (dV before dP) is kept so the fp32 text is unchanged.
    _dv_block = """        if (d_active) {
            for (uint rk = 0u; rk < NK; rk++) {
                simdgroup_float8x8 pf, dof;
                for (uint jc = 0u; jc < BJ; jc += 8u) {
                    simdgroup_load(pf, tg_P + jc*BK + rk*8u, BK, 0, true);      // Pᵀ [8k,8j]
                    simdgroup_load(dof, tg_dO + jc*D + (sgitg*8u), D);          // dO [8j,8d]
                    simdgroup_multiply_accumulate(dv_f[rk], pf, dof, dv_f[rk]);
                }
            }
        }
"""
    _dv_before = "" if round_p else _dv_block
    _dv_after = (_dv_block + "        threadgroup_barrier(mem_flags::mem_threadgroup);\n") if round_p else ""
    _p_round_line = f"            tg_P[i] = {_p_rd('tg_P[i]')};\n" if round_p else ""
    NJ, NK, ND = BJ // 8, BK // 8, D // 8
    TPG = 128  # NG=4 simdgroups

    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "b_sz",
        "b_sh",
        "b_sm",
        "b_sn",
        "mask_sz",
        "mask_sh",
        "mask_sn",
        "lse_sz",
        "lse_sh",
        "lse_sm",
        "dlt_sz",
        "dlt_sh",
        "dlt_sm",
        "do_sz",
        "do_sh",
        "do_sm",
        "do_sk",
        "dk_sz",
        "dk_sh",
        "dk_sn",
        "dk_sk",
        "dv_sz",
        "dv_sh",
        "dv_sn",
        "dv_sk",
        "H",
        "N_CTX",
        "scale",
    ]
    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"bwd_kv(simd) bindings missing {missing}")
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {n} = {bindings[n]};" for n in _LOGICAL if n != "scale")
    scale_decl = f"const float scale = {bindings['scale']};"
    if runtime_neg_inf:
        # packet 134: the source applies ONE sentinel to masked / out-of-range score
        # cells and then subtracts the row lse; replay it on in-bounds masked cells so
        # a NaN lse propagates exactly as in the source (finite data: exp2 underflows
        # to 0, bit-identical to the old hard-coded 0).
        if not bindings or "neg_inf" not in bindings:
            raise ValueError(f"{kernel_name}: runtime_neg_inf requires bindings['neg_inf']")
        scale_decl += f"\n    const float neg_inf = {bindings['neg_inf']};"
    _masked_cond = "jrow < N_CTX && krow < N_CTX" if runtime_neg_inf else "false"
    _masked_p = "exp2((neg_inf - tg_lse[jj]) * inv_ln2)" if runtime_neg_inf else "0.0f"
    if grid_3d:
        grid_decode = "uint h = pid3.y; uint z = pid3.z;"
    else:
        grid_decode = "uint zh = pid3.y; uint z = zh / H; uint h = zh % H;"
    _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// simdgroup-MMA FA-2 backward dK/dV ({elem} in/out, fp32 compute) for biased attention.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BJ = {BJ}u, BK = {BK}u, D = {D}u, TPG = {TPG}u;
    const uint NJ = {NJ}u, NK = {NK}u, ND = {ND}u;
    const float inv_ln2 = 1.4426950408889634f;
    {scale_decl}
{bind_lines}
    uint sgitg = lid / 32u;
    uint k_block = pid3.x; {grid_decode}
    uint k_start = k_block * BK;
    bool k_active = (sgitg * 8u < BK);
    bool d_active = (sgitg * 8u < D);

    uint q_base   = z*q_sz  + h*q_sh;
    uint k_base   = z*k_sz  + h*k_sh;
    uint v_base   = z*v_sz  + h*v_sh;
    uint do_base  = z*do_sz + h*do_sh;
    uint dk_base  = z*dk_sz + h*dk_sh;
    uint dv_base  = z*dv_sz + h*dv_sh;
    uint lse_base = z*lse_sz + h*lse_sh;
    uint dlt_base = z*dlt_sz + h*dlt_sh;
    uint bias_base = z*b_sz + h*b_sh;
    uint mask_base = {_mz}*mask_sz + h*mask_sh;

    threadgroup float tg_K[{BK} * {D}];
    threadgroup float tg_V[{BK} * {D}];
    threadgroup float tg_Q[{BJ} * {D}];
    threadgroup float tg_dO[{BJ} * {D}];
    threadgroup float tg_P[{max(BJ * BK, BK * D)}];   // scores S -> P [BJ,BK]; reused as dV store scratch [BK,D] (packet 148: sized for BOTH)
    threadgroup float tg_dS[{max(BJ * BK, BK * D)}];  // dP -> dS [BJ,BK];   reused as dK store scratch [BK,D]
    threadgroup float tg_lse[{BJ}];
    threadgroup float tg_delta[{BJ}];
    threadgroup uchar tg_mask[{BK}];

    for (uint i = lid; i < BK*D; i += TPG) {{
        uint kk = i / D, dd = i % D; uint krow = k_start + kk;
        tg_K[i] = (krow < N_CTX) ? {_k_stage("K[k_base + krow*k_sn + dd*k_sk]")} : 0.0f;
        tg_V[i] = (krow < N_CTX) ? float(V[v_base + krow*v_sn + dd*v_sk]) : 0.0f;
    }}
    for (uint i = lid; i < BK; i += TPG) {{
        uint krow = k_start + i;
        tg_mask[i] = (krow < N_CTX) ? Mask[mask_base + krow*mask_sn] : (uchar)1;
    }}

    // dV/dK accumulators for this simdgroup's d-col-band (cols sgitg*8..+8), one per
    // k-row-band; accumulate across ALL j-blocks in registers.
    simdgroup_float8x8 dv_f[{NK}], dk_f[{NK}];
    for (uint r = 0u; r < NK; r++) {{ dv_f[r] = simdgroup_float8x8(0.0f); dk_f[r] = simdgroup_float8x8(0.0f); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_j = (N_CTX + BJ - 1u) / BJ;
    for (uint jb = 0u; jb < n_j; jb++) {{
        uint j_start = jb * BJ;
        for (uint i = lid; i < BJ*D; i += TPG) {{
            uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
            tg_Q[i]  = (jrow < N_CTX) ? float(Q[q_base + jrow*q_sm + dd*q_sk]) : 0.0f;
            tg_dO[i] = (jrow < N_CTX) ? float(dO[do_base + jrow*do_sm + dd*do_sk]) : 0.0f;
        }}
        for (uint i = lid; i < BJ; i += TPG) {{
            uint jrow = j_start + i;
            tg_lse[i]   = (jrow < N_CTX) ? Lse[lse_base + jrow*lse_sm] : 0.0f;
            tg_delta[i] = (jrow < N_CTX) ? Delta[dlt_base + jrow*dlt_sm] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[j,k] = Q @ Kᵀ  (this simdgroup: k-col-band sgitg, all NJ j-row-bands)
        if (k_active) {{
            simdgroup_float8x8 sf[{NJ}];
            for (uint r = 0u; r < NJ; r++) sf[r] = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 qf, kf;
            for (uint dc = 0u; dc < D; dc += 8u) {{
                simdgroup_load(kf, tg_K + (sgitg*8u)*D + dc, D, 0, true);  // Kᵀ [8d,8k]
                for (uint rj = 0u; rj < NJ; rj++) {{
                    simdgroup_load(qf, tg_Q + (rj*8u)*D + dc, D);          // Q  [8j,8d]
                    simdgroup_multiply_accumulate(sf[rj], qf, kf, sf[rj]);
                }}
            }}
            for (uint rj = 0u; rj < NJ; rj++)
                simdgroup_store(sf[rj], tg_P + (rj*8u)*BK + sgitg*8u, BK);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // recompute P[j,k] = exp2((S*scale + bias - lse) * inv_ln2), masked -> 0
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
            if (jrow < N_CTX && krow < N_CTX && tg_mask[kk] == 0) {{
                float s = {"tg_P[i]" if round_k else "tg_P[i]*scale"} + float(Bias[bias_base + jrow*b_sm + krow*b_sn]);
                tg_P[i] = exp2((s - tg_lse[jj]) * inv_ln2);
            }} else if ({_masked_cond}) {{
                tg_P[i] = {_masked_p};
            }} else {{
                tg_P[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dV[k,d] += Pᵀ @ dO   (d-col-band sgitg)   AND   dP[j,k] = dO @ Vᵀ (k-col-band)
{_dv_before}        if (k_active) {{
            simdgroup_float8x8 dpf[{NJ}];
            for (uint r = 0u; r < NJ; r++) dpf[r] = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 dof2, vf;
            for (uint dc = 0u; dc < D; dc += 8u) {{
                simdgroup_load(vf, tg_V + (sgitg*8u)*D + dc, D, 0, true);       // Vᵀ [8d,8k]
                for (uint rj = 0u; rj < NJ; rj++) {{
                    simdgroup_load(dof2, tg_dO + (rj*8u)*D + dc, D);            // dO [8j,8d]
                    simdgroup_multiply_accumulate(dpf[rj], dof2, vf, dpf[rj]);
                }}
            }}
            for (uint rj = 0u; rj < NJ; rj++)
                simdgroup_store(dpf[rj], tg_dS + (rj*8u)*BK + sgitg*8u, BK);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dS[j,k] = P * (dP - delta[j])
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK;
            tg_dS[i] = {_ds_rd("tg_P[i] * (tg_dS[i] - tg_delta[jj])")};
{_p_round_line}        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
{_dv_after}
        // dK[k,d] += dSᵀ @ Q   (d-col-band sgitg)
        if (d_active) {{
            for (uint rk = 0u; rk < NK; rk++) {{
                simdgroup_float8x8 sf2, qf2;
                for (uint jc = 0u; jc < BJ; jc += 8u) {{
                    simdgroup_load(sf2, tg_dS + jc*BK + rk*8u, BK, 0, true);    // dSᵀ [8k,8j]
                    simdgroup_load(qf2, tg_Q + jc*D + (sgitg*8u), D);           // Q   [8j,8d]
                    simdgroup_multiply_accumulate(dk_f[rk], sf2, qf2, dk_f[rk]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // store registers -> tg scratch (tg_P for dV, tg_dS for dK) -> strided device write
    if (d_active) {{
        for (uint rk = 0u; rk < NK; rk++) {{
            simdgroup_store(dv_f[rk], tg_P  + (rk*8u)*D + sgitg*8u, D);
            simdgroup_store(dk_f[rk], tg_dS + (rk*8u)*D + sgitg*8u, D);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < BK*D; i += TPG) {{
        uint kk = i / D, dd = i % D; uint krow = k_start + kk;
        if (krow < N_CTX) {{
            DK[dk_base + krow*dk_sn + dd*dk_sk] = {store_cast_k("tg_dS[i] * scale")};
            DV[dv_base + krow*dv_sn + dd*dv_sk] = {store_cast_v("tg_P[i]")};
        }}
    }}
}}
"""


def _bwd_q_subtile_size(head_dim, block_j, block_k):
    """Largest power-of-two divisor JS of block_j such that the scalar bwd_q kernel fits
    Metal's limits: TPG = JS*head_dim <= 1024 threads AND the threadgroup footprint
    (2*JS*D staged Q/dQ + 2*BK*D staged K/V + 2*JS*BK scores/dS, fp32) < 32 KB. Note the
    K/V staging (2*BK*D) is J-INDEPENDENT, so head_dim large enough that 2*BK*D alone
    exceeds the budget (e.g. head_dim>=128 at BLOCK_K=32) does NOT fit at any JS -> None
    (dQ refuses; dK/dV still lower via the K-subtiled bwd_kv). JS == block_j for
    head_dim<=32 (single pass)."""
    D, BK = head_dim, block_k

    def _fits(js):
        tpg = js * D
        mem = (2 * js * D + 2 * BK * D + 2 * js * BK) * 4 + (2 * js + BK + 16) * 4
        return tpg <= 1024 and mem <= 32768

    # JS must DIVIDE block_j (the subtile loop steps by JS); iterate true divisors
    # descending so a non-power-of-2 block can never yield an overshooting JS (Triton's
    # tl.arange forces power-of-2 blocks, but keep the invariant caller-independent).
    for js in range(block_j, 0, -1):
        if block_j % js == 0 and _fits(js):
            return js
    return None


def make_flash_attention_bwd_q_kernel(
    head_dim=32,
    BLOCK_J=32,
    BLOCK_K=32,
    out_dtype="fp32",
    round_k=None,
    round_ds=None,
    round_dprod=None,
    round_delta=None,
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention_bwd_q",
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    output_casts=None,
):
    """Tiled FA-2 BACKWARD dQ + delta for biased/triangle attention (route-only ABI).

    Per (j-block, i, batch*heads): delta[j] = rowsum(O[j]*dO[j]) (stored for _bwd_kv),
    then recompute P and accumulate over k-blocks
        dP[j,k] = sum_d dO[j,d]*V[k,d];  dS[j,k] = P[j,k]*(dP[j,k] - delta[j])
        dQ[j,d] += sum_k dS[j,k]*K[k,d]      (dQ *= sm_scale at store)
    O and dO read from device (cached); the threadgroup budget is tg_Q + dq_acc +
    tg_K + tg_V + tg_P + tg_dS = 6 fp32 [BLOCK*head_dim] buffers ~24 KB. Grid:
    (n_j_blocks, I, Z*H); bias shared across i + cross-head mask as in the forward.
    """
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"bwd_q out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    if arg_decls is None or bindings is None:
        raise ValueError("make_flash_attention_bwd_q_kernel is route-only (needs arg_decls/bindings)")
    if round_delta is not None:
        raise ValueError("narrow delta reduction requires the proved D=32 simdgroup binary/XOR route")

    store_cast_q, store_cast_delta = _bwd_output_casts(output_casts, ("dq", "delta"), store_cast)

    D, BJ, BK = head_dim, BLOCK_J, BLOCK_K
    # packet 174: the source's rounding points (None = fp32 source, text unchanged). With round_k
    # the staged K IS the source's rounded k * scale, which also feeds the dQ dot: no store-time scale.
    _rd = lambda t: (lambda e: f"float({t}({e}))") if t else (lambda e: e)
    _scale_r = (
        "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))"
        if round_k == "bfloat"
        else f"float({round_k}(scale))"
    )
    _k_stage = (lambda e: f"float({round_k}(float({e}) * {_scale_r}))") if round_k else (lambda e: f"float({e})")
    _s_scaled = "s" if round_k else "s*scale"
    _dq_out = "dq_acc[i]" if round_k else "dq_acc[i] * scale"
    _dq_out_simd = "tg_P[i]" if round_k else "tg_P[i] * scale"
    _ds_rd = _rd(round_ds)
    _dp_rd = _rd(round_dprod)  # each O*dO product rounded before the fp32 rowsum (delta)
    # J-SUB-TILE so a large head_dim fits: TPG = JS*D <= 1024 threads and the threadgroup
    # footprint < 32 KB (see _bwd_q_subtile_size). JS == BJ for head_dim<=32 (single
    # subtile) -> emitted kernel semantically identical to the pre-subtile template; the
    # k-loop re-runs per j-subtile otherwise.
    JS = _bwd_q_subtile_size(D, BJ, BK)
    if JS is None:
        raise ValueError(
            f"bwd_q: head_dim={D} too large to tile within 32 KB / 1024 threads "
            f"(BLOCK_J={BJ}, BLOCK_K={BK}); the J-independent K/V staging (2*BK*D) exceeds "
            f"the budget — dK/dV still lower via the K-subtiled bwd_kv."
        )
    TPG = JS * D
    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "b_sz",
        "b_sh",
        "b_sm",
        "b_sn",
        "mask_sz",
        "mask_sh",
        "mask_sn",
        "lse_sz",
        "lse_sh",
        "lse_sm",
        "o_sz",
        "o_sh",
        "o_sm",
        "o_sk",
        "do_sz",
        "do_sh",
        "do_sm",
        "do_sk",
        "dq_sz",
        "dq_sh",
        "dq_sm",
        "dq_sk",
        "dlt_sz",
        "dlt_sh",
        "dlt_sm",
        "H",
        "N_CTX",
        "scale",
    ]
    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"bwd_q bindings missing {missing}")
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {n} = {bindings[n]};" for n in _LOGICAL if n != "scale")
    scale_decl = f"const float scale = {bindings['scale']};"
    if runtime_neg_inf:
        # packet 134: the source applies ONE sentinel to masked / out-of-range score
        # cells and then subtracts the row lse; replay it on in-bounds masked cells so
        # a NaN lse propagates exactly as in the source (finite data: exp2 underflows
        # to 0, bit-identical to the old hard-coded 0).
        if not bindings or "neg_inf" not in bindings:
            raise ValueError(f"{kernel_name}: runtime_neg_inf requires bindings['neg_inf']")
        scale_decl += f"\n    const float neg_inf = {bindings['neg_inf']};"
    _masked_cond = "jrow < N_CTX && krow < N_CTX" if runtime_neg_inf else "false"
    _masked_p = "exp2((neg_inf - tg_lse[jj]) * inv_ln2)" if runtime_neg_inf else "0.0f"
    if grid_3d:
        grid_decode = "uint h = pid3.y; uint z = pid3.z;"
    else:
        grid_decode = "uint zh = pid3.y; uint z = zh / H; uint h = zh % H;"
    _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"

    return f"""#include <metal_stdlib>
using namespace metal;

// Tiled FA-2 backward dQ + delta ({elem} in/out, fp32 compute) for biased attention.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BJ = {BJ}u, BK = {BK}u, JS = {JS}u, D = {D}u, TPG = {TPG}u;
    const float inv_ln2 = 1.4426950408889634f;
    {scale_decl}
{bind_lines}
    uint j_block = pid3.x; {grid_decode}
    uint j_block_start = j_block * BJ;

    uint q_base   = z*q_sz  + h*q_sh;
    uint k_base   = z*k_sz  + h*k_sh;
    uint v_base   = z*v_sz  + h*v_sh;
    uint o_base   = z*o_sz  + h*o_sh;
    uint do_base  = z*do_sz + h*do_sh;
    uint dq_base  = z*dq_sz + h*dq_sh;
    uint lse_base = z*lse_sz + h*lse_sh;
    uint dlt_base = z*dlt_sz + h*dlt_sh;
    uint bias_base = z*b_sz + h*b_sh;
    uint mask_base = {_mz}*mask_sz + h*mask_sh;

    threadgroup float tg_Q[{JS} * {D}];
    threadgroup float dq_acc[{JS} * {D}];
    threadgroup float tg_K[{BK} * {D}];
    threadgroup float tg_V[{BK} * {D}];
    threadgroup float tg_P[{JS} * {BK}];
    threadgroup float tg_dS[{JS} * {BK}];
    threadgroup float tg_lse[{JS}];
    threadgroup float tg_delta[{JS}];
    threadgroup uchar tg_mask[{BK}];

    uint n_k = (N_CTX + BK - 1u) / BK;
    // Process the BJ j-rows in JS-sized subtiles (JS == BJ -> single pass for
    // head_dim<=32). Each subtile stages its own Q/delta, re-runs the k-loop, stores dQ.
    for (uint jsub = 0u; jsub < BJ; jsub += JS) {{
    uint j_start = j_block_start + jsub;
    for (uint i = lid; i < JS*D; i += TPG) {{
        uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
        tg_Q[i] = (jrow < N_CTX) ? float(Q[q_base + jrow*q_sm + dd*q_sk]) : 0.0f;
        dq_acc[i] = 0.0f;
    }}
    // delta[j] = rowsum(O[j]*dO[j]); store it; load lse
    for (uint i = lid; i < JS; i += TPG) {{
        uint jrow = j_start + i;
        float dl = 0.0f;
        if (jrow < N_CTX) {{
            for (uint d = 0u; d < D; d++)
                dl += {_dp_rd("float(O[o_base + jrow*o_sm + d*o_sk]) * float(dO[do_base + jrow*do_sm + d*do_sk])")};
            Delta[dlt_base + jrow*dlt_sm] = {store_cast_delta("dl")};
        }}
        tg_delta[i] = dl;
        tg_lse[i] = (jrow < N_CTX) ? Lse[lse_base + jrow*lse_sm] : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kb = 0u; kb < n_k; kb++) {{
        uint k_start = kb * BK;
        for (uint i = lid; i < BK*D; i += TPG) {{
            uint kk = i / D, dd = i % D; uint krow = k_start + kk;
            tg_K[i] = (krow < N_CTX) ? {_k_stage("K[k_base + krow*k_sn + dd*k_sk]")} : 0.0f;
            tg_V[i] = (krow < N_CTX) ? float(V[v_base + krow*v_sn + dd*v_sk]) : 0.0f;
        }}
        for (uint i = lid; i < BK; i += TPG) {{
            uint krow = k_start + i;
            tg_mask[i] = (krow < N_CTX) ? Mask[mask_base + krow*mask_sn] : (uchar)1;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P[j,k]
        for (uint i = lid; i < JS*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
            if (jrow < N_CTX && krow < N_CTX && tg_mask[kk] == 0) {{
                float s = 0.0f;
                for (uint d = 0u; d < D; d++) s += tg_Q[jj*D + d] * tg_K[kk*D + d];
                s = {_s_scaled} + float(Bias[bias_base + jrow*b_sm + krow*b_sn]);
                tg_P[i] = exp2((s - tg_lse[jj]) * inv_ln2);
            }} else if ({_masked_cond}) {{
                tg_P[i] = {_masked_p};
            }} else {{
                tg_P[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dP -> dS
        for (uint i = lid; i < JS*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj;
            // packet 134: no P != 0 fast-skip — the source computes dS = P*(dP - delta) on
            // every cell, and 0 * (NaN - delta) is NaN (a NaN dO row must poison masked
            // columns too). For finite data 0 * finite == 0, bit-identical to the skip.
            {{
                float dp = 0.0f;
                // packet 136: the source's dO load is masked (other = 0) beyond N_CTX; a physical
                // padding row must never be read (0 * NaN is NaN). In-range cells stay unconditional.
                if (jrow < N_CTX) for (uint d = 0u; d < D; d++) dp += float(dO[do_base + jrow*do_sm + d*do_sk]) * tg_V[kk*D + d];
                tg_dS[i] = {_ds_rd("tg_P[i] * (dp - tg_delta[jj])")};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dQ[j,d] += sum_k dS[j,k] * K[k,d]
        for (uint i = lid; i < JS*D; i += TPG) {{
            uint jj = i / D, dd = i % D;
            float acc = 0.0f;
            for (uint kk = 0u; kk < BK; kk++) acc += tg_dS[jj*BK + kk] * tg_K[kk*D + dd];
            dq_acc[i] += acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint i = lid; i < JS*D; i += TPG) {{
        uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
        if (jrow < N_CTX) DQ[dq_base + jrow*dq_sm + dd*dq_sk] = {store_cast_q(_dq_out)};
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""


def make_flash_attention_bwd_q_kernel_simd(
    head_dim=32,
    BLOCK_J=32,
    BLOCK_K=32,
    out_dtype="fp32",
    round_k=None,
    round_ds=None,
    round_dprod=None,
    round_delta=None,
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention_bwd_q",
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    output_casts=None,
):
    """simdgroup-MMA FA-2 backward dQ + delta (same ABI/bindings as the scalar template).

    NG=4 simdgroups (TPG=128). delta[j]=rowsum(O*dO) is a scalar reduce (stored); then
    per k-block: S=Q@Kᵀ, P, dP=dO@Vᵀ, dS=P*(dP-delta), dQ+=dS@K (all MMA except the
    P/dS recompute). Q/dO staged once (j-block fixed); dQ accumulates in registers
    (dq_f, d-col-band per simdgroup) across k-blocks; O read from device for delta.
    Gated to BJ,BK,head_dim %8 and <=32; fp32 interior.
    """
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"bwd_q(simd) out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    if arg_decls is None or bindings is None:
        raise ValueError("make_flash_attention_bwd_q_kernel_simd is route-only (needs arg_decls/bindings)")

    store_cast_q, store_cast_delta = _bwd_output_casts(output_casts, ("dq", "delta"), store_cast)
    D, BJ, BK = head_dim, BLOCK_J, BLOCK_K
    # packet 174: the source's rounding points (None = fp32 source, text unchanged). With round_k
    # the staged K IS the source's rounded k * scale, which also feeds the dQ dot: no store-time scale.
    _rd = lambda t: (lambda e: f"float({t}({e}))") if t else (lambda e: e)
    _scale_r = (
        "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))"
        if round_k == "bfloat"
        else f"float({round_k}(scale))"
    )
    _k_stage = (lambda e: f"float({round_k}(float({e}) * {_scale_r}))") if round_k else (lambda e: f"float({e})")
    _s_scaled = "s" if round_k else "s*scale"
    _dq_out = "dq_acc[i]" if round_k else "dq_acc[i] * scale"
    _dq_out_simd = "tg_P[i]" if round_k else "tg_P[i] * scale"
    _ds_rd = _rd(round_ds)
    _dp_rd = _rd(round_dprod)  # each O*dO product rounded before the fp32 rowsum (delta)
    if round_delta not in (None, "half", "bfloat"):
        raise ValueError("round_delta must be None, 'half' or 'bfloat'")
    if round_delta is None:
        # Exact pre-319 block: the fp32 emitted shader must remain byte-identical.
        _delta_block = f"""    // delta[j] = rowsum(O[j]*dO[j]); store it; load lse
    for (uint i = lid; i < BJ; i += TPG) {{
        uint jrow = j_start + i;
        float dl = 0.0f;
        if (jrow < N_CTX) {{
            for (uint d = 0u; d < D; d++)
                dl += {_dp_rd("float(O[o_base + jrow*o_sm + d*o_sk]) * tg_dO[i*D + d]")};
            Delta[dlt_base + jrow*dlt_sm] = {store_cast_delta("dl")};
        }}
        tg_delta[i] = dl;
        tg_lse[i] = (jrow < N_CTX) ? Lse[lse_base + jrow*lse_sm] : 0.0f;
    }}"""
    else:

        def _delta_round(expr):
            if round_delta == "half":
                return f"float(half({expr}))"
            return _fa_scale_chain_expr(expr, ("bfloat",), None)

        products = "\n".join(
            f"        float dl{d} = "
            + _dp_rd(f"float(O[o_base + jrow*o_sm + (delta_d_base + {d}u)*o_sk]) * tg_dO[i*D + delta_d_base + {d}u]")
            + ";"
            for d in range(8)
        )
        pairs = "\n".join(f"        float ds{i // 2} = {_delta_round(f'dl{i} + dl{i + 1}')};" for i in range(0, 8, 2))
        quarters = "\n".join(
            f"        float dt{i // 2} = {_delta_round(f'ds{i} + ds{i + 1}')};" for i in range(0, 4, 2)
        )
        _delta_block = f"""    // Narrow delta runtime layout: eight contiguous D values per lane,
    // four adjacent lanes per row, eight rows per simdgroup. Replay the
    // binary in-thread tree then XOR 2,1, rounding after every source addf.
    uint delta_d_lane = lid % 4u;
    uint delta_row = (lid % 32u) / 4u;
    uint i = sgitg * 8u + delta_row;
    uint jrow = j_start + i;
    float dl = 0.0f;
    if (jrow < N_CTX) {{
        uint delta_d_base = delta_d_lane * 8u;
{products}
{pairs}
{quarters}
        dl = {_delta_round("dt0 + dt1")};
        float delta_peer2 = simd_shuffle_xor(dl, 2u);
        dl = {_delta_round("dl + delta_peer2")};
        float delta_peer1 = simd_shuffle_xor(dl, 1u);
        dl = {_delta_round("dl + delta_peer1")};
        if (delta_d_lane == 0u)
            Delta[dlt_base + jrow*dlt_sm] = {store_cast_delta("dl")};
    }}
    if (delta_d_lane == 0u) {{
        tg_delta[i] = dl;
        tg_lse[i] = (jrow < N_CTX) ? Lse[lse_base + jrow*lse_sm] : 0.0f;
    }}"""
    if not (D % 8 == 0 and BJ % 8 == 0 and BK % 8 == 0 and D <= 32 and BJ <= 32 and BK <= 32):
        raise ValueError("bwd_q(simd) requires BJ,BK,head_dim %8 and <=32")
    NJ, NK, ND = BJ // 8, BK // 8, D // 8
    TPG = 128
    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "b_sz",
        "b_sh",
        "b_sm",
        "b_sn",
        "mask_sz",
        "mask_sh",
        "mask_sn",
        "lse_sz",
        "lse_sh",
        "lse_sm",
        "o_sz",
        "o_sh",
        "o_sm",
        "o_sk",
        "do_sz",
        "do_sh",
        "do_sm",
        "do_sk",
        "dq_sz",
        "dq_sh",
        "dq_sm",
        "dq_sk",
        "dlt_sz",
        "dlt_sh",
        "dlt_sm",
        "H",
        "N_CTX",
        "scale",
    ]
    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"bwd_q(simd) bindings missing {missing}")
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {n} = {bindings[n]};" for n in _LOGICAL if n != "scale")
    scale_decl = f"const float scale = {bindings['scale']};"
    if runtime_neg_inf:
        # packet 134: the source applies ONE sentinel to masked / out-of-range score
        # cells and then subtracts the row lse; replay it on in-bounds masked cells so
        # a NaN lse propagates exactly as in the source (finite data: exp2 underflows
        # to 0, bit-identical to the old hard-coded 0).
        if not bindings or "neg_inf" not in bindings:
            raise ValueError(f"{kernel_name}: runtime_neg_inf requires bindings['neg_inf']")
        scale_decl += f"\n    const float neg_inf = {bindings['neg_inf']};"
    _masked_cond = "jrow < N_CTX && krow < N_CTX" if runtime_neg_inf else "false"
    _masked_p = "exp2((neg_inf - tg_lse[jj]) * inv_ln2)" if runtime_neg_inf else "0.0f"
    if grid_3d:
        grid_decode = "uint h = pid3.y; uint z = pid3.z;"
    else:
        grid_decode = "uint zh = pid3.y; uint z = zh / H; uint h = zh % H;"
    _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// simdgroup-MMA FA-2 backward dQ + delta ({elem} in/out, fp32 compute) for biased attention.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BJ = {BJ}u, BK = {BK}u, D = {D}u, TPG = {TPG}u;
    const uint NJ = {NJ}u, NK = {NK}u, ND = {ND}u;
    const float inv_ln2 = 1.4426950408889634f;
    {scale_decl}
{bind_lines}
    uint sgitg = lid / 32u;
    uint j_block = pid3.x; {grid_decode}
    uint j_start = j_block * BJ;
    bool k_active = (sgitg * 8u < BK);
    bool d_active = (sgitg * 8u < D);

    uint q_base   = z*q_sz  + h*q_sh;
    uint k_base   = z*k_sz  + h*k_sh;
    uint v_base   = z*v_sz  + h*v_sh;
    uint o_base   = z*o_sz  + h*o_sh;
    uint do_base  = z*do_sz + h*do_sh;
    uint dq_base  = z*dq_sz + h*dq_sh;
    uint lse_base = z*lse_sz + h*lse_sh;
    uint dlt_base = z*dlt_sz + h*dlt_sh;
    uint bias_base = z*b_sz + h*b_sh;
    uint mask_base = {_mz}*mask_sz + h*mask_sh;

    threadgroup float tg_Q[{BJ} * {D}];
    threadgroup float tg_dO[{BJ} * {D}];
    threadgroup float tg_K[{BK} * {D}];
    threadgroup float tg_V[{BK} * {D}];
    threadgroup float tg_P[{max(BJ * BK, BJ * D)}];   // scores S -> P [BJ,BK]; reused as dQ store scratch [BJ,D] (packet 148: sized for BOTH)
    threadgroup float tg_dS[{BJ} * {BK}];  // dP -> dS
    threadgroup float tg_lse[{BJ}];
    threadgroup float tg_delta[{BJ}];
    threadgroup uchar tg_mask[{BK}];

    for (uint i = lid; i < BJ*D; i += TPG) {{
        uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
        tg_Q[i]  = (jrow < N_CTX) ? float(Q[q_base + jrow*q_sm + dd*q_sk]) : 0.0f;
        tg_dO[i] = (jrow < N_CTX) ? float(dO[do_base + jrow*do_sm + dd*do_sk]) : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
{_delta_block}

    // dQ accumulator: this simdgroup's d-col-band, one per j-row-band; across k-blocks.
    simdgroup_float8x8 dq_f[{NJ}];
    for (uint r = 0u; r < NJ; r++) dq_f[r] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_k = (N_CTX + BK - 1u) / BK;
    for (uint kb = 0u; kb < n_k; kb++) {{
        uint k_start = kb * BK;
        for (uint i = lid; i < BK*D; i += TPG) {{
            uint kk = i / D, dd = i % D; uint krow = k_start + kk;
            tg_K[i] = (krow < N_CTX) ? {_k_stage("K[k_base + krow*k_sn + dd*k_sk]")} : 0.0f;
            tg_V[i] = (krow < N_CTX) ? float(V[v_base + krow*v_sn + dd*v_sk]) : 0.0f;
        }}
        for (uint i = lid; i < BK; i += TPG) {{
            uint krow = k_start + i;
            tg_mask[i] = (krow < N_CTX) ? Mask[mask_base + krow*mask_sn] : (uchar)1;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[j,k] = Q @ Kᵀ  (k-col-band sgitg)
        if (k_active) {{
            simdgroup_float8x8 sf[{NJ}];
            for (uint r = 0u; r < NJ; r++) sf[r] = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 qf, kf;
            for (uint dc = 0u; dc < D; dc += 8u) {{
                simdgroup_load(kf, tg_K + (sgitg*8u)*D + dc, D, 0, true);
                for (uint rj = 0u; rj < NJ; rj++) {{
                    simdgroup_load(qf, tg_Q + (rj*8u)*D + dc, D);
                    simdgroup_multiply_accumulate(sf[rj], qf, kf, sf[rj]);
                }}
            }}
            for (uint rj = 0u; rj < NJ; rj++)
                simdgroup_store(sf[rj], tg_P + (rj*8u)*BK + sgitg*8u, BK);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // recompute P[j,k]
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
            if (jrow < N_CTX && krow < N_CTX && tg_mask[kk] == 0) {{
                float s = {"tg_P[i]" if round_k else "tg_P[i]*scale"} + float(Bias[bias_base + jrow*b_sm + krow*b_sn]);
                tg_P[i] = exp2((s - tg_lse[jj]) * inv_ln2);
            }} else if ({_masked_cond}) {{
                tg_P[i] = {_masked_p};
            }} else {{
                tg_P[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dP[j,k] = dO @ Vᵀ  (k-col-band sgitg)
        if (k_active) {{
            simdgroup_float8x8 dpf[{NJ}];
            for (uint r = 0u; r < NJ; r++) dpf[r] = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 dof, vf;
            for (uint dc = 0u; dc < D; dc += 8u) {{
                simdgroup_load(vf, tg_V + (sgitg*8u)*D + dc, D, 0, true);
                for (uint rj = 0u; rj < NJ; rj++) {{
                    simdgroup_load(dof, tg_dO + (rj*8u)*D + dc, D);
                    simdgroup_multiply_accumulate(dpf[rj], dof, vf, dpf[rj]);
                }}
            }}
            for (uint rj = 0u; rj < NJ; rj++)
                simdgroup_store(dpf[rj], tg_dS + (rj*8u)*BK + sgitg*8u, BK);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dS[j,k] = P * (dP - delta[j])
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK;
            tg_dS[i] = {_ds_rd("tg_P[i] * (tg_dS[i] - tg_delta[jj])")};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dQ[j,d] += dS @ K  (d-col-band sgitg; contraction over k)
        if (d_active) {{
            for (uint rj = 0u; rj < NJ; rj++) {{
                simdgroup_float8x8 dsf, kf2;
                for (uint kc = 0u; kc < BK; kc += 8u) {{
                    simdgroup_load(dsf, tg_dS + (rj*8u)*BK + kc, BK);       // dS [8j,8k]
                    simdgroup_load(kf2, tg_K + kc*D + (sgitg*8u), D);       // K  [8k,8d]
                    simdgroup_multiply_accumulate(dq_f[rj], dsf, kf2, dq_f[rj]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (d_active) {{
        for (uint rj = 0u; rj < NJ; rj++)
            simdgroup_store(dq_f[rj], tg_P + (rj*8u)*D + sgitg*8u, D);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < BJ*D; i += TPG) {{
        uint jj = i / D, dd = i % D; uint jrow = j_start + jj;
        if (jrow < N_CTX) DQ[dq_base + jrow*dq_sm + dd*dq_sk] = {store_cast_q(_dq_out_simd)};
    }}
}}
"""


def _bwd_b_config(head_dim, block_j, block_k):
    """(stage_q, fits) for the scalar bwd_b (dbias) kernel. TPG = BJ*BK is D-independent
    (fits 1024), but the staged Q/K/V (=(BJ+2*BK)*D) scale with head_dim. tg_Q is used
    only in the score dot and is reused across k, so for a head_dim where full staging
    overflows 32 KB we DE-STAGE Q (read from device) — that alone fits head_dim=64.
    head_dim large enough that the J-independent K/V staging (2*BK*D) + the BJ*BK*3
    score/bias/db scratch already exceed 32 KB (e.g. head_dim>=128 at BLOCK_K=32) does
    NOT fit -> (False, False) -> refuse. stage_q True (head_dim<=32) -> byte-identical."""
    D, BJ, BK = head_dim, block_j, block_k
    common = (3 * BJ * BK + 2 * BJ) * 4 + BK  # db_acc+tg_bias+tg_P + lse+delta + mask
    kv = 2 * BK * D * 4
    q = BJ * D * 4
    if common + kv + q <= 32768:
        return True, True
    if common + kv <= 32768:
        return False, True
    return False, False


def make_flash_attention_bwd_b_kernel(
    head_dim=32,
    BLOCK_J=32,
    BLOCK_K=32,
    out_dtype="fp32",
    round_q=None,
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention_bwd_b",
    mask_batch_div=None,
    runtime_neg_inf=False,
    output_casts=None,
):
    """Tiled FA-2 BACKWARD dbias for biased/triangle attention (route-only ABI).

    bias is SHARED across the triangle-i axis, so its gradient SUMS over i:
        db[j,k] = sum_i dS[i,j,k]    where dS[i,j,k] = P*(dO[i]@V[i]ᵀ - delta[i])
    Grid: (n_j_blocks, n_k_blocks, Z*H); each threadgroup loops over ALL i (trifast
    assumes I == N, matching its ``for start_i in range(0, N)``). Q/K/V/lse/delta/mask
    advance by their i-stride each iteration; bias/db do not depend on i. dO read from
    device. Budget: db_acc + tg_bias + tg_Q + tg_K + tg_V + tg_P = 6 fp32 buffers.
    """
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"bwd_b out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    if arg_decls is None or bindings is None:
        raise ValueError("make_flash_attention_bwd_b_kernel is route-only (needs arg_decls/bindings)")

    (store_cast_b,) = _bwd_output_casts(output_casts, ("db",), store_cast)

    D, BJ, BK = head_dim, BLOCK_J, BLOCK_K
    # packet 174: the source's rounding point q * full(scale, dtype) (None = fp32 source, text unchanged).
    _scale_r = (
        "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))"
        if round_q == "bfloat"
        else f"float({round_q}(scale))"
    )
    _q_stage = (lambda e: f"float({round_q}(float({e}) * {_scale_r}))") if round_q else (lambda e: f"float({e})")
    _s_scaled = "s" if round_q else "s*scale"
    TPG = BJ * BK
    # For a large head_dim, DE-STAGE Q (read from device in the score dot) so the
    # threadgroup footprint fits 32 KB; stage_q True for head_dim<=32 (byte-identical).
    stage_q, _fits = _bwd_b_config(D, BJ, BK)
    if not _fits:
        raise ValueError(
            f"bwd_b: head_dim={D} too large to fit dbias scratch within 32 KB "
            f"(BLOCK_J={BJ}, BLOCK_K={BK}); the J-independent K/V staging exceeds the budget"
        )
    _LOGICAL = [
        "q_sh",
        "q_si",
        "q_sm",
        "q_sk",
        "k_sh",
        "k_si",
        "k_sn",
        "k_sk",
        "v_sh",
        "v_si",
        "v_sn",
        "v_sk",
        "b_sh",
        "b_sm",
        "b_sn",
        "mask_sz",
        "mask_si",
        "mask_sn",
        "lse_sh",
        "lse_si",
        "lse_sm",
        "dlt_sh",
        "dlt_si",
        "dlt_sm",
        "do_sh",
        "do_si",
        "do_sm",
        "do_sk",
        "db_sh",
        "db_sm",
        "db_sn",
        "H",
        "N_CTX",
        "scale",
    ]
    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"bwd_b bindings missing {missing}")
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {n} = {bindings[n]};" for n in _LOGICAL if n != "scale")
    scale_decl = f"const float scale = {bindings['scale']};"
    if runtime_neg_inf:
        # packet 134: the source applies ONE sentinel to masked / out-of-range score
        # cells and then subtracts the row lse; replay it on in-bounds masked cells so
        # a NaN lse propagates exactly as in the source (finite data: exp2 underflows
        # to 0, bit-identical to the old hard-coded 0).
        if not bindings or "neg_inf" not in bindings:
            raise ValueError(f"{kernel_name}: runtime_neg_inf requires bindings['neg_inf']")
        scale_decl += f"\n    const float neg_inf = {bindings['neg_inf']};"
    _masked_cond = "jrow < N_CTX && krow < N_CTX" if runtime_neg_inf else "false"
    _masked_p = "exp2((neg_inf - tg_lse[jj]) * inv_ln2)" if runtime_neg_inf else "0.0f"
    _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"

    # Conditional Q staging (de-staged for large head_dim). These fragments carry SINGLE
    # braces (interpolated into the f-string verbatim, not re-escaped).
    if stage_q:
        tg_q_decl = f"    threadgroup float tg_Q[{BJ} * {D}];"
        tg_q_load = (
            "        for (uint i = lid; i < BJ*D; i += TPG) {\n"
            "            uint jj = i / D, dd = i % D; uint jrow = j_start + jj;\n"
            "            tg_Q[i] = (jrow < N_CTX) ? "
            + _q_stage("Q[q_hbase + ii*q_si + jrow*q_sm + dd*q_sk]")
            + " : 0.0f;\n"
            "        }"
        )
        q_score = "tg_Q[jj*D + d]"
    else:
        tg_q_decl = "    // tg_Q de-staged (read from device) to fit the threadgroup budget"
        tg_q_load = "        // Q read from device in the score dot (de-staged)"
        q_score = _q_stage("Q[q_hbase + ii*q_si + jrow*q_sm + d*q_sk]")

    return f"""#include <metal_stdlib>
using namespace metal;

// Tiled FA-2 backward dbias ({elem} in/out, fp32 compute) for biased attention.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BJ = {BJ}u, BK = {BK}u, D = {D}u, TPG = {TPG}u;
    const float inv_ln2 = 1.4426950408889634f;
    {scale_decl}
{bind_lines}
    uint j_block = pid3.x, k_block = pid3.y, z = pid3.z;
    uint j_start = j_block * BJ, k_start = k_block * BK;

    uint q_hbase  = z*q_sh;
    uint k_hbase  = z*k_sh;
    uint v_hbase  = z*v_sh;
    uint do_hbase = z*do_sh;
    uint lse_hbase = z*lse_sh;
    uint dlt_hbase = z*dlt_sh;
    uint bias_base = z*b_sh;
    uint db_base   = z*db_sh;
    uint mask_hbase = {_mz}*mask_sz;

    threadgroup float db_acc[{BJ} * {BK}];
    threadgroup float tg_bias[{BJ} * {BK}];
{tg_q_decl}
    threadgroup float tg_K[{BK} * {D}];
    threadgroup float tg_V[{BK} * {D}];
    threadgroup float tg_P[{BJ} * {BK}];
    threadgroup float tg_lse[{BJ}];
    threadgroup float tg_delta[{BJ}];
    threadgroup uchar tg_mask[{BK}];

    for (uint i = lid; i < BJ*BK; i += TPG) {{
        db_acc[i] = 0.0f;
        uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
        tg_bias[i] = (jrow < N_CTX && krow < N_CTX)
            ? float(Bias[bias_base + jrow*b_sm + krow*b_sn]) : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ii = 0u; ii < N_CTX; ii++) {{
{tg_q_load}
        for (uint i = lid; i < BK*D; i += TPG) {{
            uint kk = i / D, dd = i % D; uint krow = k_start + kk;
            tg_K[i] = (krow < N_CTX) ? float(K[k_hbase + ii*k_si + krow*k_sn + dd*k_sk]) : 0.0f;
            tg_V[i] = (krow < N_CTX) ? float(V[v_hbase + ii*v_si + krow*v_sn + dd*v_sk]) : 0.0f;
        }}
        for (uint i = lid; i < BJ; i += TPG) {{
            uint jrow = j_start + i;
            tg_lse[i]   = (jrow < N_CTX) ? Lse[lse_hbase + ii*lse_si + jrow*lse_sm] : 0.0f;
            tg_delta[i] = (jrow < N_CTX) ? Delta[dlt_hbase + ii*dlt_si + jrow*dlt_sm] : 0.0f;
        }}
        for (uint i = lid; i < BK; i += TPG) {{
            uint krow = k_start + i;
            tg_mask[i] = (krow < N_CTX) ? Mask[mask_hbase + ii*mask_si + krow*mask_sn] : (uchar)1;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P[j,k]
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
            if (jrow < N_CTX && krow < N_CTX && tg_mask[kk] == 0) {{
                float s = 0.0f;
                for (uint d = 0u; d < D; d++) s += {q_score} * tg_K[kk*D + d];
                s = {_s_scaled} + tg_bias[i];
                tg_P[i] = exp2((s - tg_lse[jj]) * inv_ln2);
            }} else if ({_masked_cond}) {{
                tg_P[i] = {_masked_p};
            }} else {{
                tg_P[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dS[j,k] = P*(dO[i]@V[i]ᵀ - delta[j]); db_acc += dS
        for (uint i = lid; i < BJ*BK; i += TPG) {{
            uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj;
            {{  // packet 134: no P != 0 fast-skip (see the kv/q makers)
                float dp = 0.0f;
                if (jrow < N_CTX) for (uint d = 0u; d < D; d++)  // packet 136: masked dO load beyond N_CTX
                    dp += float(dO[do_hbase + ii*do_si + jrow*do_sm + d*do_sk]) * tg_V[kk*D + d];
                db_acc[i] += tg_P[i] * (dp - tg_delta[jj]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint i = lid; i < BJ*BK; i += TPG) {{
        uint jj = i / BK, kk = i % BK; uint jrow = j_start + jj, krow = k_start + kk;
        if (jrow < N_CTX && krow < N_CTX)
            DB[db_base + jrow*db_sm + krow*db_sn] = {store_cast_b("db_acc[i]")};
    }}
}}
"""


def flash_attention_tiled_threads(block_m, block_n):
    """One dispatch/ownership contract for the tiled maker and its callers.

    Every cooperative phase strides by this width. Row-only work uses lid < BM,
    so every query row must also have a physical owner. The runtime caps Metal
    threadgroups at 1024; logical score tiles may be larger, physical strides may
    not. This does not change or relax the template's shared-memory requirement.
    """
    if type(block_m) is not int or type(block_n) is not int or not 1 <= block_m <= 1024 or block_n < 1:
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(
            "tiled FlashAttention requires positive integer tile sizes and at most "
            "1024 query rows so each row has a physical thread owner"
        )
    return min(block_m * block_n, 1024)


def make_flash_attention_kernel_tiled(
    head_dim=128,
    BLOCK_M=32,
    BLOCK_N=32,
    Dc=64,
    causal=False,
    out_dtype="fp32",
    arg_decls=None,
    bindings=None,
    kernel_name="flash_attention",
    scale=None,
    bias=False,
    mask=False,
    lse=False,
    runtime_scale=False,
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    neg_inf_unit=1.0,
    round_q=None,
    round_p=None,
    scale_chain=(),
    source_kv_block=None,
    mask_tail_scores=True,
    unmasked_kv=(),
    source_causal_stop=None,
    query_tile=None,
    initial_values=None,
    query_length=None,
    transpose_k=False,
):
    """Generate a HEAD-DIM-TILED FlashAttention-2 kernel for Metal (fp32/fp16).

    packet 164 (item F-a): ``round_q`` / ``round_p`` in {None, "half", "bfloat"} replay the SOURCE's
    narrowing (see make_flash_attention_kernel_simdgroup). None keeps the MSL byte-identical.

    Same online-softmax FA2 algorithm as ``make_flash_attention_kernel`` but with
    two structural changes that let it run at large ``head_dim`` (e.g. 128) on
    Apple GPUs (32 KB threadgroup-memory limit), where the un-tiled template
    OOMs / OutOfResources:

      (a) GENERAL STRIDED addressing for a ``[Z, H, N_CTX, head_dim]`` layout
          using 16 strides (Q,K,V,O x stride(0..3)), replacing the collapsed
          ``head_offset + row*D + col`` indexing.

      (b) HEAD-DIM TILING: a ``for (uint dc = 0; dc < D; dc += Dc)`` loop that
          stages only ``[BLOCK, Dc]``-wide K/V slices at a time. The output
          accumulator ``acc[BLOCK_M*D]`` and scores ``tg_S[BLOCK_M*BLOCK_N]``
          stay resident; per KV-block the score ``S[i,j] += sum_{c<Dc} Q[i,c]*Kc[j,c]``
          is accumulated across chunks, then ``acc[i,d] += sum_j P[i,j]*Vc[j,c]``
          is accumulated chunk-by-chunk (each chunk owns a disjoint slice of D,
          so the online-softmax alpha-rescale of acc happens exactly once per
          element per KV-block).

    Threadgroup-memory budget (BLOCK_M=BLOCK_N=32, D=128, Dc=64, fp32):
        tg_S   : 32*32*4      =  4 KB
        acc    : 32*128*4     = 16 KB
        tg_KV  : 32*64*4      =  8 KB   (shared: Kc in the QK phase, Vc in the PV phase)
        tg_m/l : 32*4*2       = ~0.25 KB
        Total  : ~28.25 KB    (< 32 KB)

    Q is read directly from device memory in the QK phase (it cannot also be
    staged at Dc=64 without exceeding 32 KB: acc 16 + S 4 + Qc 8 + Kc 8 = 36 KB).
    Staging K (and reusing that buffer for V) keeps the high-reuse operand resident
    while fitting the budget; this is correctness-first, not bandwidth-optimal.

    Buffer ABI (binding order) — matches the REAL ``_flash_attn_fwd`` kernel's
    non-constexpr argument order (the launcher binds ``kargs[N]`` -> ``buffer(N)``,
    where ``kargs`` is the flat non-constexpr arg list in declaration order), so a
    routed @triton.jit FA kernel binds correctly without any reordering:
        Q, K, V, Out            device pointers (float)   buffers 0..3
        16 strides              uint, order: Q.{z,h,m,k}, K.{z,h,n,k},
                                V.{z,h,n,k}, O.{z,h,m,k}              buffers 4..19
        Z, H, N_CTX             uint                                 buffers 20..22
    The softmax ``scale = 1/sqrt(head_dim)`` is BAKED in as a compile-time
    constant — the real kernel computes ``qk_scale`` internally and has NO scale
    arg, so the template must not expect one.

    Grid: ``(N_CTX // BLOCK_M, Z*H)`` threadgroups; the kernel uses
    ``program_id(0)`` (q-block, ``pid.x``) and ``program_id(1)`` (z*h, ``pid.y``),
    so the dispatch preserves the 2-D threadgroup grid.
    Threads/threadgroup: ``min(BLOCK_M * BLOCK_N, 1024)``. All cooperative
    loops cover larger logical tiles using this actual dispatch width.

    Args:
        head_dim: Head dimension D. Must be a multiple of Dc.
        BLOCK_M:  Query rows per threadgroup.
        BLOCK_N:  KV rows staged per inner loop step.
        Dc:       Head-dim chunk width (default 64).
        transpose_k: Experimental K-only shared layout. Stage K[r,c] at
            c*BLOCK_N+r and read it with the matching score index. V keeps
            its row-major layout. The caller owns device/performance qualification;
            False preserves the original source text.
        query_tile: Optional resident query-row width. The proved dense D64
            32/64 source envelope uses 32 rows, preserving the source query
            grid, causal loop stop and KV block width. No query or KV source
            work is skipped; the storage is reused behind a uniform barrier.
        causal:   If True, apply causal mask: kv positions where kv_row > q_row
            are masked to -INFINITY in the online-softmax loop (before exp),
            so they contribute zero to numerator and denominator. Works
            identically for fp32 and fp16 (mask is on the fp32 score).
        out_dtype: Input/output element type. "fp32"/"f32" (default) emits
            float Q/K/V/Out; "fp16"/"f16" emits half Q/K/V/Out, promotes to
            float on every load, keeps ALL compute + acc + softmax in fp32, and
            casts to half on the final Out store (the proven cast-epilogue from
            ``make_simdgroup_matmul_kernel_fast``). The scale stays fp32.
        arg_decls: Optional list of MSL ``[[buffer(N)]]`` declaration lines for
            the kernel signature, in binding order. When ``None`` the canonical
            full ABI is emitted (Q,K,V,Out, 16 strides, Z,H,N_CTX — used by the
            standalone test). The ROUTED path passes the REAL kernel's present
            args (some strides / Z / H are ``equal_to_1``-specialized away), so
            the [[buffer(N)]] order matches the launcher's binding exactly.
        bindings: Optional dict mapping each logical name
            (``q_sz q_sh q_sm q_sk k_sz k_sh k_sn k_sk v_sz v_sh v_sn v_sk
            o_sz o_sh o_sm o_sk Z H N_CTX``) to its MSL expression — the buffer
            arg name when present at runtime, or a baked literal (e.g. ``1u``)
            when the arg was specialized out. Required iff ``arg_decls`` is given.
        scale: The softmax scale to bake as a compile-time constant.  When
            ``None`` (default), ``1/sqrt(head_dim)`` is computed and used —
            matching the canonical FA kernel.  When provided (e.g. from
            ``_detect_flash_attention``'s ``info["scale"]``), the EXACT
            detected value is baked, so a kernel that uses a non-standard scale
            (e.g. a learned temperature) computes correctly.
    """
    if type(transpose_k) is not bool:
        raise ValueError("transpose_k must be an exact bool")
    _k_stage_index = "c * BN + r" if transpose_k else "i"
    _k_score_index = "c * BN + cj" if transpose_k else "cj * DC + c"
    # Element type of the Q/K/V/Out device buffers. fp16 inputs are promoted to
    # float on load and the result is cast back to half on store (cast epilogue,
    # mirroring make_simdgroup_matmul_kernel_fast); all interior compute stays
    # fp32. Accept both the "fpNN" and MLIR-style "fNN" spellings so the routed
    # path (which passes info["out_dtype"], e.g. "f16") and the standalone path
    # ("fp16") agree.
    if out_dtype in ("fp32", "f32"):
        elem_t = "float"
        store_cast = lambda expr: expr  # noqa: E731  (float -> float, no-op)
    elif out_dtype in ("fp16", "f16"):
        elem_t = "half"
        store_cast = lambda expr: f"half({expr})"  # noqa: E731  (float -> half)
    elif out_dtype in ("bf16", "bfloat16"):
        # bf16 in/out, fp32 interior compute (loads promote bfloat->float implicitly,
        # the store casts float->bfloat). Same accuracy contract as the fp16 path.
        elem_t = "bfloat"
        store_cast = lambda expr: f"bfloat({expr})"  # noqa: E731  (float -> bfloat)
    else:
        raise ValueError(
            f"make_flash_attention_kernel_tiled: out_dtype must be fp32/f32/fp16/f16/bf16 (got {out_dtype!r})"
        )
    # A query-storage subtile changes residency, not the source program's grid
    # or KV iteration range. Each query row has an independent recurrence;
    # keep the complete source BLOCK_N and reuse the storage only after a
    # uniform barrier. This deliberately narrow envelope is exercised by the
    # larger dense autotuner configurations, not the biased/MLA callers.
    _initial_acc, _initial_den = _fa_initial_values(initial_values, source_kv_block)
    source_block_m = BLOCK_M
    if query_length is not None and (
        head_dim != 64 or BLOCK_M not in (8, 16) or BLOCK_N != 32 or bias or mask or lse or query_tile is not None
    ):
        raise ValueError("separate query bounds require the proved small-query D64 dense route")
    _query_bound = query_length if query_length is not None else "N_CTX"
    if query_tile is not None:
        if (
            type(query_tile) is not int
            or query_tile != 32
            or head_dim != 64
            or BLOCK_M not in (32, 64)
            or BLOCK_N not in (32, 64)
            or bias
            or mask
            or lse
            or source_kv_block != BLOCK_N
        ):
            raise ValueError("query subtiles require the proved dense D64, 32/64 source envelope")
        BLOCK_M = query_tile
        # This matches every declaration below, including max(BM, BN) staging.
        _smem_bytes = 4 * (BLOCK_M * BLOCK_N + BLOCK_M * head_dim + max(BLOCK_M, BLOCK_N) * Dc + 2 * BLOCK_M)
        if _smem_bytes > 32768:
            raise ValueError("query-subtile working set exceeds the 32768-byte staging budget")
    # Causal mask expression: emitted into the online-softmax row loop.
    # When causal=True the mask excludes kv positions *after* the query position
    # (kv_row > q_row), mapping those scores to -INFINITY before exp/sum.
    # q_row and kv_row are already in scope at the insertion point.
    if causal:
        score_guard = "(kv_row < N_CTX && kv_row <= q_row)"
        prob_guard = "(kv_row < N_CTX && kv_row <= q_row)"
    else:
        score_guard = "(kv_row < N_CTX)"
        prob_guard = "(kv_row < N_CTX)"
    _source_tail = None
    if source_kv_block is not None:
        if bias or mask or runtime_neg_inf or source_kv_block != BLOCK_N or (causal and not mask_tail_scores):
            raise ValueError("tiled source KV tail replay requires its exact plain dense source block")
        _source_tail = _fa_source_tail_guard(source_kv_block, BLOCK_N)
        if not mask_tail_scores:
            score_guard = prob_guard = _source_tail
    if unmasked_kv and (_source_tail is None or not set(unmasked_kv) <= {"K", "V"}):
        raise ValueError("unmasked KV roles require the proved source block")
    _k_tail = _source_tail if "K" in unmasked_kv else "(kv_row < N_CTX)"
    _v_tail = _source_tail if "V" in unmasked_kv else "(kv_row < N_CTX)"
    _dot_tail = _source_tail if "K" in unmasked_kv else "kv_row < N_CTX"
    if source_causal_stop is not None and (_source_tail is None or type(source_causal_stop) is not bool):
        raise ValueError("source loop stop requires a proved dense source block")
    _stop = "source_kv_stop" if source_causal_stop else "N_CTX"
    _source_stop_decl = (
        "    const uint source_kv_stop = min(q_start + BM, N_CTX); // source KV loop stop\n"
        if source_causal_stop
        else ""
    )
    if query_tile is not None and source_causal_stop:
        _source_stop_decl = (
            "    const uint source_kv_stop = min((q_block + 1u) * SOURCE_BM, N_CTX); // source KV loop stop\n"
        )
    if runtime_neg_inf:
        # packet 114: every cell (masked, tail) holds the source's FINITE sentinel and
        # contributes exp(sentinel - m) to the sum exactly as the source does.
        prob_guard = "true"
    if head_dim % Dc != 0:
        raise ValueError(f"make_flash_attention_kernel_tiled: head_dim ({head_dim}) must be a multiple of Dc ({Dc})")
    if (arg_decls is None) != (bindings is None):
        raise ValueError(
            "make_flash_attention_kernel_tiled: arg_decls and bindings must be provided together (or both omitted)"
        )

    TPG = flash_attention_tiled_threads(BLOCK_M, BLOCK_N)
    KV_STAGE = max(BLOCK_M, BLOCK_N) * Dc  # shared K/V staging buffer size
    import math as _math

    # When ``scale`` is provided (e.g. from the detector's info["scale"]), bake
    # THAT exact value so a kernel with a non-standard temperature computes
    # correctly.  When None, fall back to the canonical 1/sqrt(head_dim).
    SCALE = float(scale) if scale is not None else 1.0 / _math.sqrt(float(head_dim))
    if round_q not in (None, "half", "bfloat") or round_p not in (None, "half", "bfloat"):
        raise ValueError("round_q / round_p must be None, 'half' or 'bfloat'")
    # packet 164: the source's rounding points (fp32 sources: None -> byte-identical MSL)
    _q_raw = "Q[q_base + q_row * q_sm + (dc + c) * q_sk]"
    _qv = f"float({round_q}(float({_q_raw}) * float({round_q}(scale))))" if round_q else _q_raw
    _p_store = f"float({round_p}(p))" if round_p else "p"
    _s_scale = "s_scale" if round_q else "scale"

    # --- Biased/triangle-attention extensions (bias/mask/lse/runtime_scale). ---
    # All are no-ops when their flag is off, so a plain-FA emission is byte-for-byte
    # identical to the pre-extension template (guarded interpolations below). This
    # lets trifast-style attention — additive bias into the scores, a loaded mask →
    # −inf, a per-row log-sum-exp output, and a *runtime* softmax scale — reuse the
    # validated online-softmax loop instead of the generic op-by-op path (which
    # cannot express a dot-result epilogue). See generic_lowerer _dot_result_scaled.
    _bias_uint = []
    if bias:
        _bias_uint += ["b_sz", "b_sh", "b_sm", "b_sn"]
    if mask:
        _bias_uint += ["mask_sz", "mask_sh", "mask_sn"]
    if lse:
        _bias_uint += ["lse_sz", "lse_sh", "lse_sm"]

    _LOGICAL = [
        "q_sz",
        "q_sh",
        "q_sm",
        "q_sk",
        "k_sz",
        "k_sh",
        "k_sn",
        "k_sk",
        "v_sz",
        "v_sh",
        "v_sn",
        "v_sk",
        "o_sz",
        "o_sh",
        "o_sm",
        "o_sk",
        "Z",
        "H",
        "N_CTX",
    ] + _bias_uint
    if arg_decls is None:
        # Canonical full ABI: Q,K,V,Out (0..3), then the uint strides/dims (4..),
        # then (when enabled) the optional runtime scale + Bias/Mask/Lse pointers.
        _ptr_decls = [
            f"    device const {elem_t}* Q [[buffer(0)]]",
            f"    device const {elem_t}* K [[buffer(1)]]",
            f"    device const {elem_t}* V [[buffer(2)]]",
            f"    device {elem_t}* Out [[buffer(3)]]",
        ]
        arg_decls = list(_ptr_decls)
        for i, name in enumerate(_LOGICAL):
            arg_decls.append(f"    constant uint& arg_{name} [[buffer({4 + i})]]")
        bindings = {name: f"arg_{name}" for name in _LOGICAL}
        _next = 4 + len(_LOGICAL)
        if runtime_scale:
            arg_decls.append(f"    constant float& arg_scale [[buffer({_next})]]")
            bindings["scale"] = "arg_scale"
            _next += 1
        if bias:
            arg_decls.append(f"    device const {elem_t}* Bias [[buffer({_next})]]")
            _next += 1
        if mask:
            arg_decls.append(f"    device const uchar* Mask [[buffer({_next})]]")
            _next += 1
        if lse:
            arg_decls.append(f"    device float* Lse [[buffer({_next})]]")
            _next += 1

    missing = [n for n in _LOGICAL if n not in bindings]
    if missing:
        raise ValueError(f"make_flash_attention_kernel_tiled: bindings missing {missing}")
    if runtime_scale and "scale" not in bindings:
        raise ValueError("make_flash_attention_kernel_tiled: runtime_scale requires bindings['scale']")
    if runtime_neg_inf and "neg_inf" not in bindings:
        raise ValueError("make_flash_attention_kernel_tiled: runtime_neg_inf requires bindings['neg_inf']")
    sig = ",\n".join(arg_decls)
    # Local aliases so the body references uniform names regardless of which
    # strides/dims are real buffer args vs baked constants.
    bind_lines = "\n".join(f"    const uint {name} = {bindings[name]};" for name in _LOGICAL)

    # Scale: baked compile-time constant, or a runtime buffer arg (trifast passes
    # sm_scale at runtime and folds it into Q; we apply it to the QK score instead).
    _scale_value = bindings["scale"] if runtime_scale else f"{SCALE!r}f"
    scale_decl = f"const float scale = {_fa_scale_chain_expr(_scale_value, scale_chain, round_q)};"
    if round_q:
        scale_decl += (
            "\n    const float s_scale = 1.0f;  // packet 164: the scale is applied to Q in the source's dtype"
        )
    if runtime_neg_inf:
        # the source sentinel lives in its exp domain (after the c_eff multiply); convert
        # to this template's natural units: sentinel_nat = sentinel * (1 / c_eff)
        scale_decl += f"\n    const float neg_inf = ({bindings['neg_inf']}) * {float(neg_inf_unit)!r}f;"
    _NI = "neg_inf" if runtime_neg_inf else "-INFINITY"
    # Grid decode. 2-D (default): pid3.y = z*H + h -> z = zh/H, h = zh%H. 3-D
    # (grid_3d, trifast's pid_j/pid_i/pid_h): the two batch axes are SEPARATE grid
    # dims -> h = pid3.y (the triangle-i axis), z = pid3.z (batch*heads). H is unused.
    if grid_3d:
        grid_decode = "uint h = pid3.y;\n    uint z = pid3.z;"
    else:
        grid_decode = "uint zh      = pid3.y;\n    uint z = zh / H;\n    uint h = zh % H;"

    # Per-(z,h) base offsets for the optional bias/mask/lse tensors (z,h in scope
    # in the body). Shared-across-an-axis layouts (e.g. trifast's bias, which does
    # not depend on the triangle-i axis) are expressed by a 0 stride from the detector.
    # ``mask_batch_div`` (trifast: H_heads): the mask is shared across heads, indexed
    # by (z / div) on its batch axis (mask_start_h = pid_h // H_heads).
    _bb = []
    if bias:
        _bb.append("    uint bias_base = z * b_sz + h * b_sh;")
    if mask:
        _mz = f"(z / {mask_batch_div})" if mask_batch_div is not None else "z"
        _bb.append(f"    uint mask_base = {_mz} * mask_sz + h * mask_sh;")
    if lse:
        _bb.append("    uint lse_base = z * lse_sz + h * lse_sh;")
    biased_base_lines = "\n".join(_bb)

    # Score-cell compute: plain FA emits the original ternary (byte-identical); the
    # biased path adds `+ Bias[...]` and a `Mask[...] -> -inf` inside the valid branch.
    # These strings are interpolated as VALUES into the outer f-string, so they use
    # single braces (real MSL), not f-string-doubled braces.
    if bias or mask:
        _bias_add = "\n                    s += float(Bias[bias_base + q_row * b_sm + kv_row * b_sn]);" if bias else ""
        _mask_apply = (
            "\n                    if (Mask[mask_base + kv_row * mask_sn] != 0) s = " + _NI + ";" if mask else ""
        )
        score_compute_block = (
            "float s;\n"
            "                if (" + score_guard + ") {\n"
            "                    s = tg_S[r * BN + cj] * " + _s_scale + ";" + _bias_add + _mask_apply + "\n"
            "                } else {\n"
            "                    s = " + _NI + ";\n"
            "                }"
        )
    else:
        score_compute_block = (
            "float s = " + score_guard + " ? (tg_S[r * BN + cj] * " + _s_scale + ")\n"
            "                                           : " + _NI + ";"
        )

    # Log-sum-exp per query row (natural log): lse = running_max + log(running_sum).
    if lse:
        _lse_value = (
            "!isfinite(l_val) ? NAN : ((l_val > 0.0f) ? (tg_m[r] + log(l_val)) : -INFINITY)"
            if runtime_neg_inf
            else "(l_val > 0.0f) ? (tg_m[r] + log(l_val)) : -INFINITY"
        )
        lse_store_block = (
            "\n    // ---- Log-sum-exp per query row (lse = m + log(l), natural log) ----\n"
            "    if (lid < BM) {\n"
            "        uint r = lid;\n"
            "        uint q_row = q_start + r;\n"
            "        if (q_row < N_CTX) {\n"
            "            float l_val = tg_l[r];\n"
            "            // Preserve a nonfinite source recurrence as NaN; only a genuine\n"
            "            // finite empty denominator maps to logsumexp(empty) = -inf.\n"
            f"            Lse[lse_base + q_row * lse_sm] = {_lse_value};\n"
            "        }\n"
            "    }"
        )
    else:
        lse_store_block = ""

    _mnew_update = (
        "m_new = (isnan(m_new) || isnan(s)) ? NAN : max(m_new, s);" if runtime_neg_inf else "m_new = max(m_new, s);"
    )
    # Route-lowered biased kernels always set runtime_neg_inf: their source
    # recurrence distinguishes a nonfinite denominator from an empty row.  The
    # standalone biased maker predates that source contract and deliberately
    # maps a fully masked row to zero, so preserve it when no runtime sentinel
    # is present.  Plain/MLA routes cannot form an empty masked row; a
    # nonfinite denominator there necessarily came from source data.
    _preserve_nonfinite = runtime_neg_inf or not bias
    _out_value = (
        "!isfinite(l_val) ? NAN : ((l_val > 0.0f) ? (acc[i] / l_val) : 0.0f)"
        if _preserve_nonfinite
        else "(l_val > 0.0f) ? (acc[i] / l_val) : 0.0f"
    )

    _query_start = "uint q_start = q_block * BM;"
    _query_open = _query_close = ""
    if query_tile is not None:
        _query_start = f"const uint SOURCE_BM = {source_block_m}u;"
        _query_open = (
            "    // Query storage subtiles preserve the source grid and KV recurrence.\n"
            "    for (uint query_offset = 0u; query_offset < SOURCE_BM; query_offset += BM) {\n"
            "    uint q_start = q_block * SOURCE_BM + query_offset;\n\n"
        )
        _query_close = (
            "    threadgroup_barrier(mem_flags::mem_threadgroup); // finish stores before reusing query storage\n"
            "    }\n"
        )

    return f"""#include <metal_stdlib>
using namespace metal;

// Head-dim-tiled FlashAttention-2 ({elem_t} in/out, fp32 compute, {"causal" if causal else "non-causal"}).
// Layout: Q,K,V,Out are [Z, H, N_CTX, D]; addressing via the (possibly
// specialization-folded) strides. Buffer ABI matches the kernel's REAL present
// arg order. The softmax scale (1/sqrt(D)) is baked (kernel has no scale arg).
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint BM = {BLOCK_M}u;
    const uint BN = {BLOCK_N}u;
    const uint D  = {head_dim}u;
    const uint DC = {Dc}u;
    const uint TPG = {TPG}u;
    {scale_decl}

    // Logical stride / dim aliases (buffer arg or baked constant).
{bind_lines}

    // Grid: program_id(0) = q-block (pid3.x); batch decode below (2-D or 3-D).
    uint q_block = pid3.x;
    {grid_decode}
    {_query_start}

    // Per-(z,h) base offsets into each tensor.
    uint q_base = z * q_sz + h * q_sh;
    uint k_base = z * k_sz + h * k_sh;
    uint v_base = z * v_sz + h * v_sh;
    uint o_base = z * o_sz + h * o_sh;
{biased_base_lines}

    // Threadgroup memory.
    threadgroup float tg_S[{BLOCK_M} * {BLOCK_N}];   // BM x BN scores / probs
    threadgroup float acc[{BLOCK_M} * {head_dim}];   // BM x D output accumulator
    threadgroup float tg_KV[{KV_STAGE}];             // shared: Kc (QK) then Vc (PV)
    threadgroup float tg_m[{BLOCK_M}];               // running row max
    threadgroup float tg_l[{BLOCK_M}];               // running row sum

{_query_open}    // Init acc, m, l.
    for (uint i = lid; i < BM * D; i += TPG) {{
        acc[i] = {_initial_acc};
    }}
    if (lid < BM) {{
        tg_m[lid] = -INFINITY;
        tg_l[lid] = {_initial_den};
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{_source_stop_decl}    uint n_kv_blocks = ({_stop} + BN - 1u) / BN;
    for (uint kv_block = 0u; kv_block < n_kv_blocks; kv_block++) {{
        uint kv_start = kv_block * BN;

        // ---- Score phase: tg_S = (scale * Q) @ K^T, accumulated over head-dim chunks ----
        for (uint i = lid; i < BM * BN; i += TPG) {{
            tg_S[i] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dc = 0u; dc < D; dc += DC) {{
            // Stage Kc[BN, DC] for this chunk into tg_KV.
            for (uint i = lid; i < BN * DC; i += TPG) {{
                uint r = i / DC;          // kv row within block
                uint c = i % DC;          // head-dim col within chunk
                uint kv_row = kv_start + r;
                tg_KV[{_k_stage_index}] = {_k_tail}
                    ? K[k_base + kv_row * k_sn + (dc + c) * k_sk]
                    : 0.0f;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate this chunk's contribution to S. Q read from device.
            for (uint i = lid; i < BM * BN; i += TPG) {{
                uint r = i / BN;          // query row within block
                uint cj = i % BN;         // kv row within block
                uint q_row = q_start + r;
                uint kv_row = kv_start + cj;
                if (q_row < {_query_bound} && {_dot_tail}) {{
                    float dot = 0.0f;
                    for (uint c = 0u; c < DC; c++) {{
                        float qv = {_qv};
                        dot += qv * tg_KV[{_k_score_index}];
                    }}
                    tg_S[i] += dot;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // ---- Online-softmax update (one thread per query row) ----
        if (lid < BM) {{
            uint r = lid;
            uint q_row = q_start + r;
            if (q_row < {_query_bound}) {{
                float m_prev = tg_m[r];
                float l_prev = tg_l[r];

                // Row max over valid KV positions (apply scale; mask OOB/causal -> -inf).
                float m_new = m_prev;
                for (uint cj = 0u; cj < BN; cj++) {{
                    uint kv_row = kv_start + cj;
                    {score_compute_block}
                    tg_S[r * BN + cj] = s;   // store scaled score in place
                    {_mnew_update}
                }}

                // P = exp(S - m_new); rescale running sum.
                float alpha = exp(m_prev - m_new);
                float l_new = l_prev * alpha;
                for (uint cj = 0u; cj < BN; cj++) {{
                    uint kv_row = kv_start + cj;
                    float p = {prob_guard} ? exp(tg_S[r * BN + cj] - m_new) : 0.0f;
                    tg_S[r * BN + cj] = {_p_store};   // store P in place of S
                    l_new += p;
                }}

                // Stash alpha in tg_m so all threads can rescale acc in the PV phase.
                tg_m[r] = m_new;
                tg_l[r] = l_new;
                // Reuse acc-rescale: scale this row's resident acc by alpha now.
                for (uint d = 0u; d < D; d++) {{
                    acc[r * D + d] *= alpha;
                }}
            }} else {{
                // Padding row: zero its probabilities so it contributes nothing.
                for (uint cj = 0u; cj < BN; cj++) {{
                    tg_S[r * BN + cj] = 0.0f;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Output phase: acc[i, dc+c] += sum_j P[i,j] * Vc[j, c], per chunk ----
        for (uint dc = 0u; dc < D; dc += DC) {{
            // Stage Vc[BN, DC] for this chunk into tg_KV (reusing the K buffer).
            for (uint i = lid; i < BN * DC; i += TPG) {{
                uint r = i / DC;
                uint c = i % DC;
                uint kv_row = kv_start + r;
                tg_KV[i] = {_v_tail}
                    ? V[v_base + kv_row * v_sn + (dc + c) * v_sk]
                    : 0.0f;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // acc update for this chunk's D-slice.
            for (uint i = lid; i < BM * DC; i += TPG) {{
                uint r = i / DC;
                uint c = i % DC;
                float o_val = 0.0f;
                for (uint cj = 0u; cj < BN; cj++) {{
                    o_val += tg_S[r * BN + cj] * tg_KV[cj * DC + c];
                }}
                acc[r * D + (dc + c)] += o_val;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}

    // ---- Final normalize + write out: Out = acc / l ----
    for (uint i = lid; i < BM * D; i += TPG) {{
        uint r = i / D;
        uint c = i % D;
        uint q_row = q_start + r;
        if (q_row < {_query_bound}) {{
            float l_val = tg_l[r];
            float o = {_out_value};
            Out[o_base + q_row * o_sm + c * o_sk] = {store_cast("o")};
        }}
    }}
{lse_store_block}
{_query_close}}}
"""


def make_rope_attention_kernel(head_dim=64, block_size=256):
    """Generate a fused RoPE + single-query attention kernel.

    Applies rotary position embeddings to Q and K on-the-fly during
    attention computation. This avoids materializing the rotated Q/K
    tensors, saving memory bandwidth.

    For autoregressive inference (single query token):
        1. Apply RoPE to Q at position `q_pos`
        2. For each cached K[j]: apply RoPE at position j, compute dot(Q_rot, K_rot)
        3. Softmax over scores
        4. Weighted sum of V

    Layout:
        Q: [head_dim] — single query vector
        K_cache: [max_seq_len, head_dim] — cached keys (NOT rotated)
        V_cache: [max_seq_len, head_dim]
        freqs: [head_dim/2] — RoPE frequency table (1/10000^(2i/d))
        O: [head_dim] — output

    Args:
        head_dim: Dimension per head.
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void rope_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device const float* freqs [[buffer(3)]],
    device float* O [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& q_pos [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint D = {head_dim}u;
    const uint HALF_D = D / 2u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    // Phase 1: Apply RoPE to Q and compute attention scores
    // Pre-rotate Q at q_pos
    threadgroup float tg_q_rot[{head_dim}];
    for (uint d = lid; d < HALF_D; d += BLOCK) {{
        float theta = float(q_pos) * freqs[d];
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float q_r = Q[2u * d];
        float q_i = Q[2u * d + 1u];
        tg_q_rot[2u * d] = q_r * cos_t - q_i * sin_t;
        tg_q_rot[2u * d + 1u] = q_r * sin_t + q_i * cos_t;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute attention scores with on-the-fly RoPE on K
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < HALF_D; d++) {{
            float theta = float(j) * freqs[d];
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            float k_r = K_cache[j * D + 2u * d];
            float k_i = K_cache[j * D + 2u * d + 1u];
            float k_rot_r = k_r * cos_t - k_i * sin_t;
            float k_rot_i = k_r * sin_t + k_i * cos_t;
            dot += tg_q_rot[2u * d] * k_rot_r + tg_q_rot[2u * d + 1u] * k_rot_i;
        }}
        local_max = max(local_max, dot * scale);
    }}

    // Reduce max
    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = shared_max[tiisg];
    float global_max = simd_max(rd_max);

    // Phase 3: exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < HALF_D; d++) {{
            float theta = float(j) * freqs[d];
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            float k_r = K_cache[j * D + 2u * d];
            float k_i = K_cache[j * D + 2u * d + 1u];
            float k_rot_r = k_r * cos_t - k_i * sin_t;
            float k_rot_i = k_r * sin_t + k_i * cos_t;
            dot += tg_q_rot[2u * d] * k_rot_r + tg_q_rot[2u * d + 1u] * k_rot_i;
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = shared_sum[tiisg];
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 4: Weighted V sum (V is NOT rotated)
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint j = 0u; j < seq_len; j++) {{
                // Recompute attention weight
                float dot = 0.0f;
                for (uint dd = 0u; dd < HALF_D; dd++) {{
                    float theta = float(j) * freqs[dd];
                    float cos_t = cos(theta);
                    float sin_t = sin(theta);
                    float k_r = K_cache[j * D + 2u * dd];
                    float k_i = K_cache[j * D + 2u * dd + 1u];
                    float k_rot_r = k_r * cos_t - k_i * sin_t;
                    float k_rot_i = k_r * sin_t + k_i * cos_t;
                    dot += tg_q_rot[2u * dd] * k_rot_r + tg_q_rot[2u * dd + 1u] * k_rot_i;
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * V_cache[j * D + d];
            }}
            O[d] = o_val;
        }}
    }}
}}
"""


def make_residual_add_kernel(block_size=256, dtype="fp32", has_bias=True):
    """Generate a fused residual connection kernel.

    output = input + residual + bias (or input + residual if has_bias=False)

    Common in transformer blocks after attention and FFN layers.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        has_bias: Whether to include a bias term.
    """
    name = "residual_add_kernel"
    kb = KernelBuilder(name, block_size=block_size)

    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("residual", dtype=dtype, const=True)
    if has_bias:
        kb.add_ptr_arg("bias", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    in_val = kb.load("input", offsets, mask, out_var="in_val", dtype=dtype)
    res_val = kb.load("residual", offsets, mask, out_var="res_val", dtype=dtype)

    if has_bias:
        bias_val = kb.load("bias", offsets, mask, out_var="bias_val", dtype=dtype)
        kb._var("sum_val", f"{in_val} + {res_val} + {bias_val}", ty="float")
        kb.store("output", offsets, "sum_val", mask, dtype=dtype)
    else:
        kb._var("sum_val", f"{in_val} + {res_val}", ty="float")
        kb.store("output", offsets, "sum_val", mask, dtype=dtype)

    return kb.build()


def make_kv_cache_attention_kernel(head_dim=64, block_size=256):
    """Generate a KV-cache attention kernel for autoregressive inference.

    Single query token attending to a cached K,V sequence:
    score[j] = Q[0,:] . K[j,:] * scale
    attn = softmax(scores)
    output = sum(attn[j] * V[j,:])

    Each threadgroup handles one attention head.
    Iterates over the KV cache in chunks.

    Args:
        head_dim: Dimension of each attention head.
        block_size: Threads per threadgroup.

    Kernel args:
        Q: [n_heads, head_dim] — single query token per head
        K_cache: [n_heads, max_seq_len, head_dim] — key cache
        V_cache: [n_heads, max_seq_len, head_dim] — value cache
        O: [n_heads, head_dim] — output
        seq_len: current sequence length (how much of cache is valid)
        scale: 1/sqrt(head_dim)
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void kv_cache_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    // pid = head index
    const uint D = {head_dim}u;
    uint head_offset_q = pid * D;
    uint head_offset_kv = pid * seq_len * D;

    // Shared memory for online softmax
    threadgroup float tg_scores[{block_size}];  // attention scores buffer

    // Phase 1: Compute all attention scores and find max
    // Each thread handles one or more KV positions
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        float score = dot * scale;
        tg_scores[j % {block_size}u] = score;  // partial storage
        local_max = max(local_max, score);
    }}

    // Reduce max across threadgroup
    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = shared_max[tiisg];
    float global_max = simd_max(rd_max);

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        float score = dot * scale;
        float p = exp(score - global_max);
        local_sum += p;
    }}

    // Reduce sum
    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = shared_sum[tiisg];
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Compute weighted V sum
    // Each thread accumulates over its KV positions for all D dimensions
    // To avoid excessive register pressure, process D in chunks
    for (uint d_start = 0u; d_start < D; d_start += {block_size}u) {{
        float o_val = 0.0f;
        uint d = d_start + lid;
        if (d < D) {{
            for (uint j = 0u; j < seq_len; j++) {{
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[head_offset_q + dd] * K_cache[head_offset_kv + j * D + dd];
                }}
                float score = dot * scale;
                float attn_weight = exp(score - global_max) * inv_sum;
                o_val += attn_weight * V_cache[head_offset_kv + j * D + d];
            }}
            O[head_offset_q + d] = o_val;
        }}
    }}
}}
"""


def make_gqa_attention_kernel(head_dim=64, n_q_per_kv=4, block_size=256):
    """Generate a Grouped Query Attention kernel for inference.

    GQA: multiple query heads share fewer KV heads. Used in LLaMA 3,
    Mistral, Gemma 2. Each threadgroup handles one query head and maps
    to the correct KV head via integer division.

    Layout:
        Q: [n_q_heads, head_dim] — single query token per head
        K_cache: [n_kv_heads, max_seq_len, head_dim]
        V_cache: [n_kv_heads, max_seq_len, head_dim]
        O: [n_q_heads, head_dim]

    Args:
        head_dim: Dimension per head.
        n_q_per_kv: Number of query heads per KV head (e.g., 4 or 8).
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void gqa_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    // pid = query head index
    const uint D = {head_dim}u;
    const uint N_Q_PER_KV = {n_q_per_kv}u;

    uint q_head = pid;
    uint kv_head = q_head / N_Q_PER_KV;
    uint head_offset_q = q_head * D;
    uint head_offset_kv = kv_head * seq_len * D;

    // Phase 1: Compute all attention scores
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        local_max = max(local_max, dot * scale);
    }}

    // Reduce max
    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = shared_max[tiisg];
    float global_max = simd_max(rd_max);

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        float p = exp(dot * scale - global_max);
        local_sum += p;
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = shared_sum[tiisg];
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum
    for (uint d_start = 0u; d_start < D; d_start += {block_size}u) {{
        float o_val = 0.0f;
        uint d = d_start + lid;
        if (d < D) {{
            for (uint j = 0u; j < seq_len; j++) {{
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[head_offset_q + dd] * K_cache[head_offset_kv + j * D + dd];
                }}
                float attn_weight = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_weight * V_cache[head_offset_kv + j * D + d];
            }}
            O[head_offset_q + d] = o_val;
        }}
    }}
}}
"""


def make_batched_kv_decode_kernel(n_heads=8, head_dim=64, block_size=256):
    """Generate a batched multi-head KV-cache decode kernel.

    For autoregressive inference: each batch item has one query token per head,
    attending to cached K,V of varying sequence lengths.

    Layout:
        Q: [batch, n_heads, head_dim] — current query tokens
        K_cache: [batch, n_heads, max_seq_len, head_dim]
        V_cache: [batch, n_heads, max_seq_len, head_dim]
        O: [batch, n_heads, head_dim]
        seq_lens: [batch] — actual sequence length per batch item

    Dispatch: one threadgroup per (batch_item, head) pair.
    pid = batch_idx * n_heads + head_idx.

    Args:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void batched_kv_decode(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    device const uint* seq_lens [[buffer(4)]],
    constant uint& max_seq_len [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint N_HEADS = {n_heads}u;
    const uint D = {head_dim}u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    uint batch_idx = pid / N_HEADS;
    uint head_idx = pid % N_HEADS;
    if (batch_idx >= batch_size) return;

    uint seq_len = seq_lens[batch_idx];

    // Offsets into contiguous memory
    uint q_offset = (batch_idx * N_HEADS + head_idx) * D;
    uint kv_head_offset = (batch_idx * N_HEADS + head_idx) * max_seq_len * D;
    uint o_offset = q_offset;

    // Phase 1: Find max attention score
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[q_offset + d] * K_cache[kv_head_offset + j * D + d];
        }}
        local_max = max(local_max, dot * scale);
    }}

    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = shared_max[tiisg];
    float global_max = simd_max(rd_max);

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[q_offset + d] * K_cache[kv_head_offset + j * D + d];
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = shared_sum[tiisg];
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum — each thread computes one output dimension
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint j = 0u; j < seq_len; j++) {{
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[q_offset + dd] * K_cache[kv_head_offset + j * D + dd];
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * V_cache[kv_head_offset + j * D + d];
            }}
            O[o_offset + d] = o_val;
        }}
    }}
}}
"""


def make_paged_attention_kernel(head_dim=64, page_size=16, block_size=256):
    """Generate a paged attention kernel for variable-length KV-cache.

    Instead of contiguous KV-cache, keys and values are stored in fixed-size
    pages (blocks) of `page_size` tokens each. A page table maps logical
    block indices to physical block indices in a shared page pool.

    This is critical for production LLM serving (vLLM-style) where sequences
    have variable lengths and memory must be managed efficiently.

    Layout:
        Q: [head_dim] — single query vector for one head
        K_pages: [n_physical_pages, page_size, head_dim] — KV page pool (keys)
        V_pages: [n_physical_pages, page_size, head_dim] — KV page pool (values)
        page_table: [max_pages_per_seq] — maps logical block idx → physical block idx
        O: [head_dim] — output
        seq_len: actual sequence length (may not fill last page)
        n_pages: number of pages for this sequence

    Dispatch: 1 threadgroup per query head.

    Args:
        head_dim: Dimension per head.
        page_size: Tokens per page (16 typical for vLLM).
        block_size: Threads per threadgroup.
    """
    n_simdgroups = (block_size + 31) // 32
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void paged_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_pages [[buffer(1)]],
    device const float* V_pages [[buffer(2)]],
    device const uint* page_table [[buffer(3)]],
    device float* O [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& n_pages [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint D = {head_dim}u;
    const uint PAGE_SIZE = {page_size}u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    // Phase 1: Compute attention scores over all paged tokens
    // Each thread handles a subset of token positions
    float local_max = -INFINITY;
    for (uint pos = lid; pos < seq_len; pos += BLOCK) {{
        // Map logical position to physical page + offset
        uint page_idx = pos / PAGE_SIZE;
        uint page_offset = pos % PAGE_SIZE;
        uint phys_page = page_table[page_idx];
        uint k_base = (phys_page * PAGE_SIZE + page_offset) * D;

        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * K_pages[k_base + d];
        }}
        local_max = max(local_max, dot * scale);
    }}

    // Reduce max across threads
    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{n_simdgroups}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = (tiisg < {n_simdgroups}u) ? shared_max[tiisg] : -INFINITY;
    float global_max = simd_max(rd_max);

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint pos = lid; pos < seq_len; pos += BLOCK) {{
        uint page_idx = pos / PAGE_SIZE;
        uint page_offset = pos % PAGE_SIZE;
        uint phys_page = page_table[page_idx];
        uint k_base = (phys_page * PAGE_SIZE + page_offset) * D;

        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * K_pages[k_base + d];
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{n_simdgroups}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = (tiisg < {n_simdgroups}u) ? shared_sum[tiisg] : 0.0f;
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum — each thread computes one output dimension
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint pos = 0u; pos < seq_len; pos++) {{
                uint page_idx = pos / PAGE_SIZE;
                uint page_offset = pos % PAGE_SIZE;
                uint phys_page = page_table[page_idx];
                uint k_base = (phys_page * PAGE_SIZE + page_offset) * D;
                uint v_base = k_base;  // V_pages same layout as K_pages

                // Recompute attention weight
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[dd] * K_pages[k_base + dd];
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * V_pages[v_base + d];
            }}
            O[d] = o_val;
        }}
    }}
}}
"""


def make_multi_head_paged_attention_kernel(n_heads=8, head_dim=64, page_size=16, block_size=256):
    """Generate a multi-head paged attention kernel.

    Extends paged attention to handle multiple attention heads simultaneously.
    Each threadgroup handles one head. The page table is shared across heads
    but K/V pages are stored per-head.

    Layout:
        Q: [n_heads, head_dim] — one query per head
        K_pages: [n_physical_pages, page_size, n_heads, head_dim]
        V_pages: [n_physical_pages, page_size, n_heads, head_dim]
        page_table: [max_pages_per_seq] — logical → physical page mapping
        O: [n_heads, head_dim] — output per head
        seq_len: actual sequence length
        n_pages: number of logical pages

    Dispatch: n_heads threadgroups (one per head).

    Args:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        page_size: Tokens per page.
        block_size: Threads per threadgroup.
    """
    n_simdgroups = (block_size + 31) // 32
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void multi_head_paged_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_pages [[buffer(1)]],
    device const float* V_pages [[buffer(2)]],
    device const uint* page_table [[buffer(3)]],
    device float* O [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& n_pages [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint N_HEADS = {n_heads}u;
    const uint D = {head_dim}u;
    const uint PAGE_SIZE = {page_size}u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    uint head_idx = pid;
    if (head_idx >= N_HEADS) return;

    uint q_offset = head_idx * D;

    // Phase 1: Compute attention scores
    float local_max = -INFINITY;
    for (uint pos = lid; pos < seq_len; pos += BLOCK) {{
        uint page_idx = pos / PAGE_SIZE;
        uint page_offset = pos % PAGE_SIZE;
        uint phys_page = page_table[page_idx];
        // K layout: [phys_page, page_offset, head_idx, d]
        uint k_base = ((phys_page * PAGE_SIZE + page_offset) * N_HEADS + head_idx) * D;

        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[q_offset + d] * K_pages[k_base + d];
        }}
        local_max = max(local_max, dot * scale);
    }}

    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{n_simdgroups}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = (tiisg < {n_simdgroups}u) ? shared_max[tiisg] : -INFINITY;
    float global_max = simd_max(rd_max);

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (uint pos = lid; pos < seq_len; pos += BLOCK) {{
        uint page_idx = pos / PAGE_SIZE;
        uint page_offset = pos % PAGE_SIZE;
        uint phys_page = page_table[page_idx];
        uint k_base = ((phys_page * PAGE_SIZE + page_offset) * N_HEADS + head_idx) * D;

        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[q_offset + d] * K_pages[k_base + d];
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{n_simdgroups}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = (tiisg < {n_simdgroups}u) ? shared_sum[tiisg] : 0.0f;
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint pos = 0u; pos < seq_len; pos++) {{
                uint page_idx = pos / PAGE_SIZE;
                uint page_offset = pos % PAGE_SIZE;
                uint phys_page = page_table[page_idx];
                uint k_base = ((phys_page * PAGE_SIZE + page_offset) * N_HEADS + head_idx) * D;
                uint v_base = k_base;

                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[q_offset + dd] * K_pages[k_base + dd];
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * V_pages[v_base + d];
            }}
            O[q_offset + d] = o_val;
        }}
    }}
}}
"""


def make_fp16_kv_attention_kernel(head_dim=64, block_size=256):
    """Generate a mixed-precision KV-cache attention kernel.

    Q is float32, K and V caches are stored in float16 (half precision).
    Computation is done in float32 for accuracy, but reads K/V as half,
    reducing memory bandwidth by 2x for the KV-cache.

    Layout:
        Q: [head_dim] float32 — single query vector
        K_cache: [max_seq_len, head_dim] float16
        V_cache: [max_seq_len, head_dim] float16
        O: [head_dim] float32 — output
        seq_len: actual sequence length

    Dispatch: 1 threadgroup.

    Args:
        head_dim: Dimension per head.
        block_size: Threads per threadgroup.
    """
    n_simdgroups = (block_size + 31) // 32
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void fp16_kv_attention(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint D = {head_dim}u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    // Phase 1: max score
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * float(K_cache[j * D + d]);
        }}
        local_max = max(local_max, dot * scale);
    }}

    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{n_simdgroups}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = (tiisg < {n_simdgroups}u) ? shared_max[tiisg] : -INFINITY;
    float global_max = simd_max(rd_max);

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += BLOCK) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * float(K_cache[j * D + d]);
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{n_simdgroups}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = (tiisg < {n_simdgroups}u) ? shared_sum[tiisg] : 0.0f;
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum (V is half, compute in float)
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint j = 0u; j < seq_len; j++) {{
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[dd] * float(K_cache[j * D + dd]);
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * float(V_cache[j * D + d]);
            }}
            O[d] = o_val;
        }}
    }}
}}
"""


def make_int8_matmul_kernel():
    """Generate a weight-only INT8 quantized matmul kernel.

    output = input(float) @ dequant(weight_int8, scale, zero_point)

    Per-row quantization: each row of the weight matrix has its own
    float scale and zero_point. Dequantization happens on-the-fly:
    w_float = (w_int8 - zero_point) * scale

    Layout:
        input: [M, K] float
        weight: [N, K] int8 (signed char)
        scales: [N] float (per-output-channel scale)
        zeros: [N] float (per-output-channel zero point)
        output: [M, N] float

    Uses tiled approach for memory efficiency. Each threadgroup
    computes a block of the output.
    """
    return """#include <metal_stdlib>
using namespace metal;

kernel void int8_matmul(
    device const float* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid / N;
    uint col = gid % N;
    if (row >= M || col >= N) return;

    float s = scales[col];
    float z = zeros[col];
    float acc = 0.0f;

    // Dot product with on-the-fly dequantization
    for (uint k = 0u; k < K; k++) {
        float w = (float(weight[col * K + k]) - z) * s;
        acc += input[row * K + k] * w;
    }

    output[row * N + col] = acc;
}
"""


def make_int8_matmul_fast(rr=4, rc=2, bk=32, layout="nk"):
    """Fast weight-only INT8 matmul: tiled simdgroup MMA with on-the-fly dequant.

    Adapts the fp16 fast-matmul to quantized weights. The int8 weight tile is
    staged into threadgroup memory as float with the per-channel zero folded in
    (``bdeq[k][n] = float(w[.,.]) - zero[n]``); the FLOAT simdgroup MMA runs on
    the dequantized tile; the epilogue multiplies by ``scale[n]``. Near-bit-exact
    vs the scalar ``make_int8_matmul_kernel`` reference (float accumulator).

    ``layout`` selects the weight storage:
      - ``"nk"``: weight [N,K] int8 (GPTQ/AWQ style, ``out = a @ w.T``), read
        ``weight[n*K + k]``.
      - ``"kn"``: weight [K,N] int8 (the natural ``tl.dot(a, w)`` Triton pattern),
        read ``weight[k*N + n]``. The staging read stays coalesced across the BN
        row; only the k stride differs.

    Layout: input [M,K] float, scales/zeros [N], out [M,N] (both weight layouts).
    Each simdgroup (32 threads) computes an (8*rr) x (8*rc) output block.
    DISPATCH: n_groups = ceil(M/(8*rr)) * ceil(N/(8*rc)); 32 threads/group.
    SIZE CONTRACT: M % (8*rr) == 0, N % (8*rc) == 0, K % bk == 0 (bk % 8 == 0).
    Measured ~13-18x over the scalar kernel on M4 Max; best (rr,rc,bk)=(4,2,32).
    """
    if layout not in ("nk", "kn"):
        raise ValueError(f"make_int8_matmul_fast: layout must be 'nk' or 'kn', got {layout!r}")
    # The ONLY layout-dependent expression: the int8 weight element index.
    w_index = "(n0 + nn) * K + (k0 + kk)" if layout == "nk" else "(k0 + kk) * N + (n0 + nn)"
    BN = 8 * rc
    accs = "\n    ".join(f"simdgroup_float8x8 c{r}_{c}(0.0f);" for r in range(rr) for c in range(rc))
    afrags = " ".join(f"simdgroup_float8x8 a{r};" for r in range(rr))
    bfrags = " ".join(f"simdgroup_float8x8 b{c};" for c in range(rc))
    sub = []
    for kk in range(0, bk, 8):
        for r in range(rr):
            sub.append(f"simdgroup_load(a{r}, input + (m0 + {r * 8}u) * K + k0 + {kk}u, K);")
        for c in range(rc):
            sub.append(f"simdgroup_load(b{c}, bdeq + {kk * BN}u + {c * 8}u, {BN}u);")
        for r in range(rr):
            for c in range(rc):
                sub.append(f"simdgroup_multiply_accumulate(c{r}_{c}, a{r}, b{c}, c{r}_{c});")
    submma = "\n        ".join(sub)
    epi = "\n    ".join(
        f"simdgroup_store(c{r}_{c}, cbuf, 8);\n    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        f"    for (uint e = tid; e < 64u; e += 32u) {{ uint mm=e/8u, nn=e%8u; "
        f"output[(m0 + {r * 8}u + mm) * N + (n0 + {c * 8}u + nn)] = scales[n0 + {c * 8}u + nn] * cbuf[e]; }}\n"
        f"    simdgroup_barrier(mem_flags::mem_threadgroup);"
        for r in range(rr)
        for c in range(rc)
    )
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void int8_matmul_fast(
    device const float* input  [[buffer(0)]],
    device const char*  weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros  [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {{
    uint ntn = N / {BN}u;
    uint m0 = (tgid / ntn) * {8 * rr}u;
    uint n0 = (tgid % ntn) * {BN}u;

    threadgroup float bdeq[{bk * BN}];
    {accs}
    {afrags} {bfrags}

    for (uint k0 = 0u; k0 < K; k0 += {bk}u) {{
        for (uint e = tid; e < {bk * BN}u; e += 32u) {{
            uint kk = e / {BN}u, nn = e % {BN}u;
            bdeq[kk * {BN}u + nn] = float(weight[{w_index}]) - zeros[n0 + nn];
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {submma}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}

    threadgroup float cbuf[64];
    {epi}
}}
"""


def make_int8_matmul_pergroup_fast(group_size=128, rr=4, rc=2, bk=32):
    """FAST per-group int8 GEMM: tiled simdgroup MMA, ~4-7x the scalar per-group template.

    The per-N fast kernel folds the per-channel zero into the staged tile and applies
    scale once in the epilogue (scale/zero constant along K). For PER-GROUP, scale AND
    zero change every ``group_size`` along K — but a ``bk``-tile sits within ONE group
    (bk <= group_size, group_size % bk == 0), so fold BOTH per group into the staged
    dequant tile ``bdeq[kk][nn] = (float(w) - zero_g) * scale_g`` and drop the epilogue
    scale. Float simdgroup MMA on the fully-dequantized tile; exact vs the scalar kernel.

    Weight [K,N] kn CONTIGUOUS (weight[(k0+kk)*N + n]); input [M,K] / output [M,N]
    contiguous; scales/zeros [n_groups, N] with runtime strides (ssg, ssn). Each simdgroup
    computes an (8*rr) x (8*rc) block. DISPATCH: ceil(M/(8*rr))*ceil(N/(8*rc)) groups,
    32 threads. SIZE CONTRACT: M % (8*rr) == 0, N % (8*rc) == 0, K % bk == 0, bk % 8 == 0,
    group_size % bk == 0 (routed only when all hold + contiguous inner strides).
    """
    if group_size % bk != 0 or bk % 8 != 0:
        raise ValueError(f"make_int8_matmul_pergroup_fast: need group_size%bk==0, bk%8==0 (G={group_size}, bk={bk})")
    BN = 8 * rc
    accs = "\n    ".join(f"simdgroup_float8x8 c{r}_{c}(0.0f);" for r in range(rr) for c in range(rc))
    afrags = " ".join(f"simdgroup_float8x8 a{r};" for r in range(rr))
    bfrags = " ".join(f"simdgroup_float8x8 b{c};" for c in range(rc))
    sub = []
    for kk in range(0, bk, 8):
        for r in range(rr):
            sub.append(f"simdgroup_load(a{r}, input + (m0 + {r * 8}u) * K + k0 + {kk}u, K);")
        for c in range(rc):
            sub.append(f"simdgroup_load(b{c}, bdeq + {kk * BN}u + {c * 8}u, {BN}u);")
        for r in range(rr):
            for c in range(rc):
                sub.append(f"simdgroup_multiply_accumulate(c{r}_{c}, a{r}, b{c}, c{r}_{c});")
    submma = "\n        ".join(sub)
    epi = "\n    ".join(
        f"simdgroup_store(c{r}_{c}, cbuf, 8);\n    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        f"    for (uint e = tid; e < 64u; e += 32u) {{ uint mm=e/8u, nn=e%8u; "
        f"output[(m0 + {r * 8}u + mm) * N + (n0 + {c * 8}u + nn)] = cbuf[e]; }}\n"
        f"    simdgroup_barrier(mem_flags::mem_threadgroup);"
        for r in range(rr)
        for c in range(rc)
    )
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void int8_matmul_pergroup_fast(
    device const float* input  [[buffer(0)]],
    device const char*  weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros  [[buffer(4)]],
    constant uint& M [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& ssg [[buffer(8)]], constant uint& ssn [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {{
    const uint GROUP = {group_size}u;
    uint ntn = N / {BN}u;
    uint m0 = (tgid / ntn) * {8 * rr}u;
    uint n0 = (tgid % ntn) * {BN}u;
    threadgroup float bdeq[{bk * BN}];
    {accs}
    {afrags} {bfrags}
    for (uint k0 = 0u; k0 < K; k0 += {bk}u) {{
        uint g = k0 / GROUP;
        for (uint e = tid; e < {bk * BN}u; e += 32u) {{
            uint kk = e / {BN}u, nn = e % {BN}u;
            float sc = scales[g * ssg + (n0 + nn) * ssn];
            float ze = zeros[g * ssg + (n0 + nn) * ssn];
            bdeq[kk * {BN}u + nn] = (float(weight[(k0 + kk) * N + (n0 + nn)]) - ze) * sc;
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {submma}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
    threadgroup float cbuf[64];
    {epi}
}}
"""


def make_varlen_flash_attention(
    head_dim=64,
    causal=False,
    out_dtype="fp32",
    arg_decls=None,
    bindings=None,
    kernel_name="varlen_fa",
    scale=None,
):
    """VARLEN FlashAttention-2 (packed cu_seqlens, route-only ABI). One thread per query
    row; K/V tiles staged in threadgroup memory; online softmax with a register acc.

    The dominant real-world FA shape (vLLM / flash-attn / HF): sequences are PACKED into
    [total_tokens, H, head_dim] and indexed per batch b by cu_seqlens (cu_q[b] gives the
    row offset, cu_q[b+1]-cu_q[b] the length). The dense [Z,H,N,D] FA templates can't
    express this; the generic per-element path is correct but ~0.04 TF. This tiled kernel
    is ~8-9x that, EXACT vs the per-sequence reference.

    ``bindings`` (from the detector): pointer names Q/K/V/O/CUQ/CUK, the heads count H, and
    the token/head/dim strides per tensor (q_st,q_sh,q_sk, k_.., v_.., o_..). Grid is 2-D
    (tgpos.x = q-block, tgpos.y = batch*H); BLOCK_M threads/threadgroup; dispatched by the
    native-grid ``dispatch_flash_attention`` (descriptor ("flash_attention", msl, BLOCK_M)).
    """
    if arg_decls is None or bindings is None:
        raise ValueError("make_varlen_flash_attention is route-only (needs arg_decls/bindings)")
    if out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
    elif out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
    elif out_dtype in ("bf16", "bfloat16"):
        elem, store_cast = "bfloat", lambda e: f"bfloat({e})"
    else:
        raise ValueError(f"varlen FA out_dtype must be fp32/fp16/bf16 (got {out_dtype!r})")
    D = head_dim
    BM = BN = 32
    # The detector BAKES the kernel's own constant Q-scale (refuses if absent); default to
    # the canonical 1/sqrt(head_dim) only for the route-free template smoke test.
    scale = float(D) ** -0.5 if scale is None else float(scale)
    inv_ln2 = 1.4426950408889634
    _need = [
        "Q",
        "K",
        "V",
        "O",
        "CUQ",
        "CUK",
        "H",
        "q_st",
        "q_sh",
        "q_sk",
        "k_st",
        "k_sh",
        "k_sk",
        "v_st",
        "v_sh",
        "v_sk",
        "o_st",
        "o_sh",
        "o_sk",
    ]
    missing = [n for n in _need if n not in bindings]
    if missing:
        raise ValueError(f"varlen FA bindings missing {missing}")
    b = bindings
    # MAXS = the kernel's own kv-loop bound (max_seqlen arg): the kernel attends
    # min(seqlen_k, max_seqlen) keys, so the template must clamp identically (re-review
    # 2026-08-25: looping to seqlen_k silently attended MORE keys when max_seqlen was
    # smaller). Route-only default (no MAXS binding) = seqlen_k, the old behavior.
    _maxs = b.get("MAXS")
    kmax_expr = f"min(seqlen_k, uint({_maxs}))" if _maxs else "seqlen_k"
    causal_guard = "&& (qrow >= krow)" if causal else ""
    sig = ",\n".join(arg_decls)
    return f"""#include <metal_stdlib>
using namespace metal;

// Varlen FlashAttention-2 ({elem} in/out, fp32 compute), packed cu_seqlens.
kernel void {kernel_name}(
{sig},
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint D = {D}u, BM = {BM}u, BN = {BN}u;
    const float scale = {scale}f, inv_ln2 = {inv_ln2}f;
    const uint nheads = {b["H"]};
    const uint q_st = {b["q_st"]}, q_sh = {b["q_sh"]}, q_sk = {b["q_sk"]};
    const uint k_st = {b["k_st"]}, k_sh = {b["k_sh"]}, k_sk = {b["k_sk"]};
    const uint v_st = {b["v_st"]}, v_sh = {b["v_sh"]}, v_sk = {b["v_sk"]};
    const uint o_st = {b["o_st"]}, o_sh = {b["o_sh"]}, o_sk = {b["o_sk"]};
    uint m_block = tgpos.x, bh = tgpos.y;
    uint bb = bh / nheads, h = bh % nheads;

    uint q_start = uint({b["CUQ"]}[bb]);
    uint seqlen_q = uint({b["CUQ"]}[bb + 1u]) - q_start;
    if (m_block * BM >= seqlen_q) return;
    uint k_start = uint({b["CUK"]}[bb]);
    uint seqlen_k = uint({b["CUK"]}[bb + 1u]) - k_start;

    uint row = lid;
    uint qrow = m_block * BM + row;
    bool valid = (qrow < seqlen_q);

    float q_reg[{D}];
    float acc[{D}];
    for (uint d = 0u; d < D; d++) {{
        q_reg[d] = valid ? float({b["Q"]}[(q_start + qrow) * q_st + h * q_sh + d * q_sk]) : 0.0f;
        acc[d] = 0.0f;
    }}
    float m_i = -INFINITY, l_i = 0.0f;

    threadgroup float tg_K[{BN} * {D}];
    threadgroup float tg_V[{BN} * {D}];

    uint kmax = {kmax_expr};
    for (uint kt = 0u; kt < kmax; kt += BN) {{
        for (uint e = lid; e < BN * D; e += BM) {{
            uint kk = e / D, dd = e % D; uint krow = kt + kk;
            bool kv = (krow < kmax);
            tg_K[e] = kv ? float({b["K"]}[(k_start + krow) * k_st + h * k_sh + dd * k_sk]) : 0.0f;
            tg_V[e] = kv ? float({b["V"]}[(k_start + krow) * v_st + h * v_sh + dd * v_sk]) : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {{
            for (uint kk = 0u; kk < BN; kk++) {{
                uint krow = kt + kk;
                if (krow < kmax {causal_guard}) {{
                    float s = 0.0f;
                    for (uint d = 0u; d < D; d++) s += q_reg[d] * tg_K[kk * D + d];
                    s *= scale;
                    float m_new = max(m_i, s);
                    float alpha = exp2((m_i - m_new) * inv_ln2);
                    float p = exp2((s - m_new) * inv_ln2);
                    l_i = l_i * alpha + p;
                    for (uint d = 0u; d < D; d++) acc[d] = acc[d] * alpha + p * tg_V[kk * D + d];
                    m_i = m_new;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (valid) {{
        for (uint d = 0u; d < D; d++)
            {b["O"]}[(q_start + qrow) * o_st + h * o_sh + d * o_sk] = {store_cast("acc[d] / l_i")};
    }}
}}
"""


def make_varlen_flash_attention_mma(
    head_dim=64,
    causal=False,
    out_dtype="fp16",
    arg_decls=None,
    bindings=None,
    kernel_name="varlen_fa_mma",
    scale=None,
):
    """FAST varlen FlashAttention-2 via simdgroup_matrix MMA (fp16 in/out, fp32 accumulate).

    BM=32 query rows = 4 simdgroups (128 threads); each simdgroup owns 8 rows. QK^T and P@V
    run through 8x8 simdgroup MMA; the online softmax is scalar through threadgroup memory.
    The O accumulator is REGISTER-RESIDENT (o_acc[D/8] float8x8 per simdgroup) and the online
    per-row rescale is elementwise through SIMD-private scratch, not a diagonal MMA:
    off-diagonal zero products must not contaminate Inf/NaN output rows. Q/K/V
    tiles are staged into threadgroup memory with bounds-masking (packed cu_seqlens), so ANY
    global strides work (the MMA reads the contiguous tg tiles). Measured on a ragged fp16
    batch: D=64 0.45ms (~7.6x the scalar template, ~4x pad+SDPA); D=128 1.37ms (1.4x pad+SDPA).
    DTYPES: fp16/bf16 stage 2-byte tiles (BN=32, head_dim<=128); fp32 stages TRUE float
    tiles (float8x8 MMA, BN=16, head_dim<=~112). Accumulation is fp32 in every case
    (simdgroup_float8x8 S/o_acc). Ineligible dims fall to the scalar template. Same buffer
    ABI / bindings as make_varlen_flash_attention; 128 threads/tg.
    """
    if arg_decls is None or bindings is None:
        raise ValueError("make_varlen_flash_attention_mma is route-only (needs arg_decls/bindings)")
    # Staged-tile type + kv-block width. fp16/bf16 use 2-byte tiles (BN=32); fp32 uses TRUE
    # float tiles (float8x8 MMA) at BN=16 so the doubled tile bytes still fit the 32KB tg
    # budget. All accumulate in fp32, including the elementwise rescale.
    if out_dtype in ("fp16", "f16"):
        TILE, SGT, TB, BN = "half", "simdgroup_half8x8", 2, 32
    elif out_dtype in ("bf16", "bfloat16"):
        TILE, SGT, TB, BN = "bfloat", "simdgroup_bfloat8x8", 2, 32
    elif out_dtype in ("fp32", "f32"):
        TILE, SGT, TB, BN = "float", "simdgroup_float8x8", 4, 16
    else:
        raise ValueError(f"varlen MMA FA dtype must be fp16/bf16/fp32 (got {out_dtype!r})")
    D, BM, NT = head_dim, 32, 128
    NG = NT // 32
    if D % 8 != 0:
        raise ValueError(f"varlen MMA FA needs head_dim %% 8 == 0 (got {D})")
    # register-O budget (no tg_O): tg_Q + tg_K + tg_V + tg_P (TB bytes each) + tg_S (float,
    # reused as SIMD-private rescale/output scratch) + tg_m/tg_l/tg_alpha.
    tg_bytes = BM * D * TB + BN * D * TB * 2 + BM * BN * TB + BM * BN * 4 + BM * 4 * 3
    if tg_bytes > 32768:
        raise ValueError(
            f"varlen MMA FA threadgroup memory {tg_bytes}B > 32KB for head_dim={D}; use the scalar template"
        )
    _need = [
        "Q",
        "K",
        "V",
        "O",
        "CUQ",
        "CUK",
        "H",
        "q_st",
        "q_sh",
        "q_sk",
        "k_st",
        "k_sh",
        "k_sk",
        "v_st",
        "v_sh",
        "v_sk",
        "o_st",
        "o_sh",
        "o_sk",
    ]
    missing = [n for n in _need if n not in bindings]
    if missing:
        raise ValueError(f"varlen MMA FA bindings missing {missing}")
    b = bindings
    # MAXS: clamp to the kernel's own kv-loop bound (see make_varlen_flash_attention).
    _maxs = b.get("MAXS")
    kmax_expr = f"min(seqlen_k, uint({_maxs}))" if _maxs else "seqlen_k"
    SCALE = float(D) ** -0.5 if scale is None else float(scale)
    causal_expr = "(qrow >= kr)" if causal else "true"
    sig = ",\n".join(arg_decls)
    return f"""#include <metal_stdlib>
using namespace metal;

// Varlen FlashAttention-2 ({TILE} in/out, fp32 accumulate) via simdgroup_matrix MMA.
kernel void {kernel_name}(
{sig},
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {{
    const uint D = {D}u, BM = {BM}u, BN = {BN}u, NT = {NT}u;
    const float SCALE = {SCALE}f;
    const uint nheads = {b["H"]};
    const uint q_st = {b["q_st"]}, q_sh = {b["q_sh"]}, q_sk = {b["q_sk"]};
    const uint k_st = {b["k_st"]}, k_sh = {b["k_sh"]}, k_sk = {b["k_sk"]};
    const uint v_st = {b["v_st"]}, v_sh = {b["v_sh"]}, v_sk = {b["v_sk"]};
    const uint o_st = {b["o_st"]}, o_sh = {b["o_sh"]}, o_sk = {b["o_sk"]};
    threadgroup {TILE} tg_Q[BM*D];
    threadgroup {TILE} tg_K[BN*D];
    threadgroup {TILE} tg_V[BN*D];
    threadgroup {TILE} tg_P[BM*BN];
    threadgroup float tg_S[BM*BN];
    threadgroup float tg_m[BM];
    threadgroup float tg_l[BM];
    threadgroup float tg_alpha[BM];

    uint m_block = tgpos.x, bh = tgpos.y;
    uint bb = bh / nheads, h = bh % nheads;
    uint q_start = uint({b["CUQ"]}[bb]);   uint seqlen_q = uint({b["CUQ"]}[bb+1u]) - q_start;
    uint k_start = uint({b["CUK"]}[bb]);   uint seqlen_k = uint({b["CUK"]}[bb+1u]) - k_start;
    if (m_block * BM >= seqlen_q) return;
    uint kmax = {kmax_expr};

    for (uint i = lid; i < BM*D; i += NT) {{
        uint r = i / D, c = i % D; uint qr = m_block*BM + r;
        tg_Q[i] = qr < seqlen_q ? {TILE}({b["Q"]}[(q_start+qr)*q_st + h*q_sh + c*q_sk]) : {TILE}(0);
    }}
    for (uint i = lid; i < BM; i += NT) {{ tg_m[i] = -INFINITY; tg_l[i] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg = sgid, lane = lid % 32u;
    simdgroup_float8x8 o_acc[D/8];
    for (uint et = 0u; et < D/8u; et++) o_acc[et] = simdgroup_float8x8(0.0f);

    for (uint kt = 0u; kt < kmax; kt += BN) {{
        for (uint i = lid; i < BN*D; i += NT) {{
            uint r = i / D, c = i % D; uint kr = kt + r; bool ok = kr < kmax;
            tg_K[i] = ok ? {TILE}({b["K"]}[(k_start+kr)*k_st + h*k_sh + c*k_sk]) : {TILE}(0);
            tg_V[i] = ok ? {TILE}({b["V"]}[(k_start+kr)*v_st + h*v_sh + c*v_sk]) : {TILE}(0);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint nt = 0u; nt < BN/8u; nt++) {{
            simdgroup_float8x8 S = simdgroup_float8x8(0.0f);
            for (uint dt = 0u; dt < D/8u; dt++) {{
                {SGT} qf, kf;
                simdgroup_load(qf, tg_Q + (sg*8u)*D + dt*8u, D);
                simdgroup_load(kf, tg_K + (nt*8u)*D + dt*8u, D, ulong2(0,0), true);
                simdgroup_multiply_accumulate(S, qf, kf, S);
            }}
            simdgroup_store(S, tg_S + (sg*8u)*BN + nt*8u, BN);
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8u) {{
            uint r = sg*8u + lane; uint qrow = m_block*BM + r;
            float m_old = tg_m[r], l_old = tg_l[r], m_blk = -INFINITY;
            for (uint j = 0u; j < BN; j++) {{
                uint kr = kt + j;
                float s = (kr < kmax && {causal_expr}) ? tg_S[r*BN+j]*SCALE : -INFINITY;
                tg_S[r*BN+j] = s; m_blk = max(m_blk, s);
            }}
            float m_new = max(m_old, m_blk);
            float alpha = exp(m_old - m_new), lsum = 0.0f;
            for (uint j = 0u; j < BN; j++) {{ float p = exp(tg_S[r*BN+j]-m_new); tg_P[r*BN+j]={TILE}(p); lsum += p; }}
            tg_l[r] = l_old*alpha + lsum; tg_m[r] = m_new;
            bool finite_row = isfinite(tg_l[r]);
            // Keep a nonfinite row out of both MMAs.  Even an off-diagonal
            // zero contaminates sibling rows when multiplied by NaN.
            if (!finite_row)
                for (uint j = 0u; j < BN; j++) tg_P[r*BN+j] = {TILE}(0);
            tg_alpha[r] = finite_row ? alpha : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scalar row rescale: no invented off-diagonal 0*Inf or 0*NaN.
        // tg_S is dead after softmax; each group owns a disjoint 64-cell tile.
        for (uint et = 0u; et < D/8u; et++) {{
            simdgroup_store(o_acc[et], tg_S + sg*64u, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e=lane; e<64u; e+=32u)
                tg_S[sg*64u+e] *= tg_alpha[sg*8u+e/8u];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o_acc[et], tg_S + sg*64u, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint nt = 0u; nt < BN/8u; nt++) {{
                {SGT} pf, vf;
                simdgroup_load(pf, tg_P + (sg*8u)*BN + nt*8u, BN);
                simdgroup_load(vf, tg_V + (nt*8u)*D + et*8u, D);
                simdgroup_multiply_accumulate(o_acc[et], pf, vf, o_acc[et]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Normalize each output element using only its own row's denominator.
    for (uint et = 0u; et < D/8u; et++) {{
        simdgroup_store(o_acc[et], tg_S + sg*64u, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {{
            uint r = sg*8u + lane; uint qrow = m_block*BM + r;
            if (qrow < seqlen_q)
                for (uint c = 0u; c < 8u; c++)
                    {b["O"]}[(q_start+qrow)*o_st + h*o_sh + (et*8u+c)*o_sk] =
                        (!isfinite(tg_l[r]) || tg_l[r] == 0.0f)
                            ? {TILE}(NAN) : {TILE}(tg_S[sg*64u + lane*8u + c] * (1.0f/tg_l[r]));
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""


def make_int4_matmul_pergroup_fast(group_size=128, rr=4, rc=2, bk=32):
    """FAST per-group INT4 GEMM: the int8 fast kernel with the nibble unpack folded into
    the staged dequant tile. Packed uchar weight [K/2, N] contiguous (LOW nibble for EVEN
    k); per-group scale/zero folded per group into ``bdeq``; float simdgroup MMA. Same
    size contract as make_int8_matmul_pergroup_fast + weight [K/2, N] contiguous (wbk==N).
    """
    if group_size % bk != 0 or bk % 8 != 0:
        raise ValueError(f"make_int4_matmul_pergroup_fast: need group_size%bk==0, bk%8==0 (G={group_size}, bk={bk})")
    BN = 8 * rc
    accs = "\n    ".join(f"simdgroup_float8x8 c{r}_{c}(0.0f);" for r in range(rr) for c in range(rc))
    afrags = " ".join(f"simdgroup_float8x8 a{r};" for r in range(rr))
    bfrags = " ".join(f"simdgroup_float8x8 b{c};" for c in range(rc))
    sub = []
    for kk in range(0, bk, 8):
        for r in range(rr):
            sub.append(f"simdgroup_load(a{r}, input + (m0 + {r * 8}u) * K + k0 + {kk}u, K);")
        for c in range(rc):
            sub.append(f"simdgroup_load(b{c}, bdeq + {kk * BN}u + {c * 8}u, {BN}u);")
        for r in range(rr):
            for c in range(rc):
                sub.append(f"simdgroup_multiply_accumulate(c{r}_{c}, a{r}, b{c}, c{r}_{c});")
    submma = "\n        ".join(sub)
    epi = "\n    ".join(
        f"simdgroup_store(c{r}_{c}, cbuf, 8);\n    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        f"    for (uint e = tid; e < 64u; e += 32u) {{ uint mm=e/8u, nn=e%8u; "
        f"output[(m0 + {r * 8}u + mm) * N + (n0 + {c * 8}u + nn)] = cbuf[e]; }}\n"
        f"    simdgroup_barrier(mem_flags::mem_threadgroup);"
        for r in range(rr)
        for c in range(rc)
    )
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void int4_matmul_pergroup_fast(
    device const float* input  [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros  [[buffer(4)]],
    constant uint& M [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& K [[buffer(7)]],
    constant uint& ssg [[buffer(8)]], constant uint& ssn [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {{
    const uint GROUP = {group_size}u;
    uint ntn = N / {BN}u;
    uint m0 = (tgid / ntn) * {8 * rr}u;
    uint n0 = (tgid % ntn) * {BN}u;
    threadgroup float bdeq[{bk * BN}];
    {accs}
    {afrags} {bfrags}
    for (uint k0 = 0u; k0 < K; k0 += {bk}u) {{
        uint g = k0 / GROUP;
        for (uint e = tid; e < {bk * BN}u; e += 32u) {{
            uint kk = e / {BN}u, nn = e % {BN}u;
            uint kki = k0 + kk;
            float sc = scales[g * ssg + (n0 + nn) * ssn];
            float ze = zeros[g * ssg + (n0 + nn) * ssn];
            uchar packed = weight[(kki / 2u) * N + (n0 + nn)];
            uint w4 = (uint(packed) >> ((kki % 2u) * 4u)) & 0xFu;   // low nibble for even k
            bdeq[kk * {BN}u + nn] = (float(w4) - ze) * sc;
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {submma}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
    threadgroup float cbuf[64];
    {epi}
}}
"""


def make_int8_matmul_pergroup(group_size=128):
    """Weight-only INT8 matmul with PER-GROUP quantization (GPTQ-style): every
    ``group_size`` elements along K share a (scale, zero). Scalar (one thread per
    output) — correct-first coverage for the per-group case the per-N fast kernel
    (``make_int8_matmul_fast``) cannot express; it applies ``scale``/``zero`` per K
    group inside the accumulation. A fast per-group MMA variant is a perf follow-up.

    Fully STRIDE-GENERIC (every operand's strides are runtime buffers), so it works
    for any weight layout ([K,N] or [N,K]) and any scale/zero layout ([n_groups,N]
    or [N,n_groups]) — the router forwards the kernel's own stride args. ``dequant:
    w = (float(weight) - zero_g) * scale_g``; ``out = input @ w``.

    Buffers: input,weight,output,scales,zeros, M,N,K, then the row/col strides
    isr,isc (input) / wsk,wsn (weight, as fn of k,n) / osr,osc (output) / ssg,ssn
    (scale, as fn of group,n) / zsg,zsn (zero).
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void int8_matmul_pergroup(
    device const float* input   [[buffer(0)]],
    device const char*  weight  [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    device const float* scales  [[buffer(3)]],
    device const float* zeros   [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& isr [[buffer(8)]],  constant uint& isc [[buffer(9)]],
    constant uint& wsk [[buffer(10)]], constant uint& wsn [[buffer(11)]],
    constant uint& osr [[buffer(12)]], constant uint& osc [[buffer(13)]],
    constant uint& ssg [[buffer(14)]], constant uint& ssn [[buffer(15)]],
    constant uint& zsg [[buffer(16)]], constant uint& zsn [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint GROUP = {group_size}u;
    uint row = gid / N;
    uint col = gid % N;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0u; k < K; k++) {{
        uint g = k / GROUP;
        float s = scales[g * ssg + col * ssn];
        float z = zeros[g * zsg + col * zsn];
        float w = (float(weight[k * wsk + col * wsn]) - z) * s;
        acc += input[row * isr + k * isc] * w;
    }}
    output[row * osr + col * osc] = acc;
}}
"""


def make_int4_matmul_pergroup(group_size=128):
    """Weight-only INT4 matmul with PER-GROUP quantization (GPTQ/AWQ), the M>1 GEMM
    companion to ``make_int4_gemv``. Packed weight is ``uchar`` [K/2, N] — 2 nibbles per
    byte, LOW nibble for EVEN k / HIGH for odd (the ONLY packing the int4 descriptor
    routes, pinned to the loop k). Scalar (one thread per output); a fast per-group MMA
    variant is a perf follow-up. ``w = (float(nibble) - zero_g) * scale_g; out = input @ w``.

    Fully STRIDE-GENERIC. Buffers mirror ``make_int8_matmul_pergroup`` EXCEPT buffer 10 is
    the BYTE-row (k/2) stride ``wbk`` (weight is [K/2, N], indexed by the byte k/2), and
    the weight buffer is ``uchar``.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void int4_matmul_pergroup(
    device const float* input   [[buffer(0)]],
    device const uchar* weight  [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    device const float* scales  [[buffer(3)]],
    device const float* zeros   [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& isr [[buffer(8)]],  constant uint& isc [[buffer(9)]],
    constant uint& wbk [[buffer(10)]], constant uint& wsn [[buffer(11)]],
    constant uint& osr [[buffer(12)]], constant uint& osc [[buffer(13)]],
    constant uint& ssg [[buffer(14)]], constant uint& ssn [[buffer(15)]],
    constant uint& zsg [[buffer(16)]], constant uint& zsn [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint GROUP = {group_size}u;
    uint row = gid / N;
    uint col = gid % N;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0u; k < K; k++) {{
        uint g = k / GROUP;
        float s = scales[g * ssg + col * ssn];
        float z = zeros[g * zsg + col * zsn];
        uchar packed = weight[(k / 2u) * wbk + col * wsn];
        uint w4 = (uint(packed) >> ((k % 2u) * 4u)) & 0xFu;   // low nibble for even k
        float w = (float(w4) - z) * s;
        acc += input[row * isr + k * isc] * w;
    }}
    output[row * osr + col * osc] = acc;
}}
"""


def make_int8_gemv():
    """Fast weight-only INT8 GEMV (M=1 decode) — the dominant LLM-inference shape.

    One SIMDGROUP per output column: the 32 lanes stride the K dimension in
    char4/float4 (coalesced within the simdgroup), then ``simd_sum`` reduces.
    Memory-bandwidth-bound; measured ~11x over the scalar kernel at the roofline.

    Layout: input [K] float, weight [N,K] int8, scales/zeros [N], output [N].
    DISPATCH: N simdgroups = N*32 threads; K % 4 == 0.
    """
    return """#include <metal_stdlib>
using namespace metal;
kernel void int8_gemv(
    device const float* input  [[buffer(0)]],
    device const char*  weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros  [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint n = gid / 32u;
    if (n >= N) return;
    float z = zeros[n];
    device const char4*  w4  = (device const char4*)(weight + n * K);
    device const float4* in4 = (device const float4*)input;
    uint K4 = K / 4u;
    float acc = 0.0f;
    for (uint k = lane; k < K4; k += 32u) {
        char4 wv = w4[k]; float4 iv = in4[k];
        acc += iv.x * (float(wv.x) - z) + iv.y * (float(wv.y) - z)
             + iv.z * (float(wv.z) - z) + iv.w * (float(wv.w) - z);
    }
    acc = simd_sum(acc);
    if (lane == 0u) output[n] = scales[n] * acc;
}
"""


def make_int4_gemv(group_size=128):
    """Fast weight-only INT4 decode GEMV (M=1), GPTQ/AWQ per-group quantized.

    Layout: input [K] float, weight [N, K/2] uchar (2 int4/byte, low nibble = even k,
    high nibble = odd k), scales/zeros [N, K/group_size] float (per-group along K),
    output [N]. out[n] = sum_k input[k] * (nibble(w,n,k) - zero[n,k/g]) * scale[n,k/g].

    One SIMDGROUP per output column; each lane strides bytes (2 nibbles) coalesced,
    then simd_sum. Validated ~1.47x faster than int8 (half the weight bytes),
    ~320 GB/s packed. DISPATCH: N simdgroups = N*32 threads; K % 4 == 0 (2 nibbles/byte
    x 2 = the char2 stride). Buffers: (input, weight, output, scales, zeros, K, N).
    """
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void int4_gemv(
    device const float* input  [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros  [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    uint n = gid / 32u;
    if (n >= N) return;
    const uint G = {group_size}u;
    uint ng = (K + G - 1u) / G;
    device const uchar* wrow = weight + n * (K / 2u);
    device const float* srow = scales + n * ng;
    device const float* zrow = zeros  + n * ng;
    float acc = 0.0f;
    // Each lane owns bytes b = lane, lane+32, ...  each byte = 2 nibbles (k=2b, 2b+1).
    for (uint b = lane; b < K / 2u; b += 32u) {{
        uchar packed = wrow[b];
        uint k0 = 2u * b;
        uint g = k0 / G;                 // 2b and 2b+1 share a group (G even)
        float s = srow[g], z = zrow[g];
        float w0 = (float(packed & 0x0Fu) - z) * s;
        float w1 = (float((packed >> 4u) & 0x0Fu) - z) * s;
        acc += input[k0] * w0 + input[k0 + 1u] * w1;
    }}
    acc = simd_sum(acc);
    if (lane == 0u) output[n] = acc;
}}
"""


def make_int4_matmul_kernel(group_size=128):
    """Generate a weight-only INT4 quantized matmul kernel (GPTQ/AWQ style).

    output = input(float) @ dequant(weight_int4, scale, zeros)

    Per-group quantization: every `group_size` elements in the K dimension
    share a scale and zero_point. Two int4 values are packed per byte
    (low nibble = even index, high nibble = odd index).

    Layout:
        input: [M, K] float
        weight: [N, K/2] uchar (2 int4s per byte, unsigned)
        scales: [N, K/group_size] float
        zeros: [N, K/group_size] float
        output: [M, N] float

    Args:
        group_size: Number of K elements per quantization group (typically 128).
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void int4_matmul(
    device const float* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* zeros [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint GROUP_SIZE = {group_size}u;
    uint row = gid / N;
    uint col = gid % N;
    if (row >= M || col >= N) return;

    uint n_groups = (K + GROUP_SIZE - 1u) / GROUP_SIZE;
    float acc = 0.0f;

    for (uint k = 0u; k < K; k++) {{
        // Unpack int4 from byte: even indices in low nibble, odd in high
        uint byte_idx = col * (K / 2u) + k / 2u;
        uchar packed = weight[byte_idx];
        uint w4;
        if (k % 2u == 0u) {{
            w4 = uint(packed & 0x0Fu);  // low nibble (0-15)
        }} else {{
            w4 = uint((packed >> 4u) & 0x0Fu);  // high nibble (0-15)
        }}

        // Dequantize: w_float = (w4 - zero) * scale
        uint group_idx = k / GROUP_SIZE;
        float s = scales[col * n_groups + group_idx];
        float z = zeros[col * n_groups + group_idx];
        float w = (float(w4) - z) * s;

        acc += input[row * K + k] * w;
    }}

    output[row * N + col] = acc;
}}
"""


def make_concat_kernel(n_inputs=2, block_size=256):
    """Generate a kernel that concatenates tensors along axis 0.

    output = concat(input_0, input_1, ...) along the first dimension.
    Each input is a flat 1D buffer. The kernel copies all inputs
    sequentially into the output buffer.

    For 2 inputs: output[0:n0] = input_0, output[n0:n0+n1] = input_1.

    Args:
        n_inputs: Number of input tensors to concatenate (2, 3, or 4).
        block_size: Threads per threadgroup.
    """
    # Generate buffer params for each input + its size
    input_params = []
    for i in range(n_inputs):
        input_params.append(f"    device const float* input_{i} [[buffer({i})]]")
    out_idx = n_inputs
    input_params.append(f"    device float* output [[buffer({out_idx})]]")
    # Size params for each input
    size_params = []
    for i in range(n_inputs):
        size_params.append(f"    constant uint& n_{i} [[buffer({out_idx + 1 + i})]]")

    all_params = ",\n".join(input_params + size_params + ["    uint gid [[thread_position_in_grid]]"])

    # Build the copy logic
    copy_logic = ""
    offset_expr = "0u"
    for i in range(n_inputs):
        if i == 0:
            copy_logic += f"""
    if (gid < n_0) {{
        output[gid] = input_0[gid];
        return;
    }}"""
            offset_expr = "n_0"
        else:
            prev_offset = offset_expr
            offset_expr = f"{prev_offset} + n_{i}"
            copy_logic += f"""
    if (gid < {offset_expr}) {{
        output[gid] = input_{i}[gid - ({prev_offset})];
        return;
    }}"""

    return f"""#include <metal_stdlib>
using namespace metal;

kernel void concat_kernel(
{all_params}
) {{{copy_logic}
}}
"""


def make_split_kernel(n_outputs=2, block_size=256):
    """Generate a kernel that splits a tensor into equal chunks along axis 0.

    Given input of size N, splits into n_outputs chunks of size N/n_outputs.
    Each thread copies one element from input to the appropriate output.

    Args:
        n_outputs: Number of output chunks.
        block_size: Threads per threadgroup.
    """
    out_params = []
    for i in range(n_outputs):
        out_params.append(f"    device float* output_{i} [[buffer({i + 1})]]")

    all_params = ",\n".join(
        ["    device const float* input [[buffer(0)]]"]
        + out_params
        + [f"    constant uint& chunk_size [[buffer({n_outputs + 1})]]", "    uint gid [[thread_position_in_grid]]"]
    )

    # Build dispatch logic
    dispatch = ""
    for i in range(n_outputs):
        if i == 0:
            dispatch += f"""
    if (gid < chunk_size) {{
        output_0[gid] = input[gid];
        return;
    }}"""
        else:
            dispatch += f"""
    if (gid < {i + 1}u * chunk_size) {{
        uint local_idx = gid - {i}u * chunk_size;
        output_{i}[local_idx] = input[gid];
        return;
    }}"""

    return f"""#include <metal_stdlib>
using namespace metal;

kernel void split_kernel(
{all_params}
) {{{dispatch}
}}
"""


def make_top_k_kernel(k=50, block_size=256):
    """Generate a top-k sampling kernel for LLM inference.

    Given logits of shape [vocab_size], finds the top-k largest values
    and their indices. Uses a parallel partial-sort approach:
    - Each thread scans a chunk of the vocabulary, maintaining a local
      top-k candidate (value + index).
    - Candidates are gathered into shared memory.
    - A single-threadgroup reduction picks the final top-k.

    For simplicity and correctness, this uses a heap-free approach:
    each thread finds its single best candidate, then we iteratively
    collect the top-k from all threads via shared memory reductions.

    Args:
        k: Number of top elements to select.
        block_size: Threads per threadgroup (one threadgroup per row).
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void top_k(
    device const float* logits [[buffer(0)]],
    device float* out_values [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& K_val [[buffer(4)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // pid = batch row index, each threadgroup handles one row
    const uint BLOCK = {block_size}u;
    const uint row_offset = pid * vocab_size;

    // Phase 1: Each thread finds its top-{k} candidates using a min-heap-like array
    // For simplicity, each thread tracks its best single candidate per iteration
    // We use shared memory to collect all thread-best candidates

    threadgroup float tg_vals[{block_size}];
    threadgroup uint tg_idxs[{block_size}];

    // Scratch to mark "already picked" indices
    threadgroup uint picked[{k}];

    for (uint ki = 0u; ki < K_val && ki < {k}u; ki++) {{
        // Each thread scans its chunk for the best unpicked element
        float best_val = -INFINITY;
        uint best_idx = 0u;

        for (uint v = lid; v < vocab_size; v += BLOCK) {{
            // Check if already picked
            bool is_picked = false;
            for (uint p = 0u; p < ki; p++) {{
                if (picked[p] == v) {{ is_picked = true; break; }}
            }}
            if (!is_picked) {{
                float val = logits[row_offset + v];
                if (val > best_val) {{
                    best_val = val;
                    best_idx = v;
                }}
            }}
        }}

        tg_vals[lid] = best_val;
        tg_idxs[lid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction to find the global best
        for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
            if (lid < stride) {{
                if (tg_vals[lid + stride] > tg_vals[lid]) {{
                    tg_vals[lid] = tg_vals[lid + stride];
                    tg_idxs[lid] = tg_idxs[lid + stride];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Thread 0 writes the k-th result
        if (lid == 0u) {{
            uint out_offset = pid * K_val + ki;
            out_values[out_offset] = tg_vals[0];
            out_indices[out_offset] = tg_idxs[0];
            picked[ki] = tg_idxs[0];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""


def make_top_p_kernel(max_k=256, block_size=256):
    """Generate a top-p (nucleus) sampling kernel.

    Given logits for a single row, applies temperature scaling, computes
    softmax, sorts by descending probability, and returns the nucleus set
    (tokens whose cumulative probability <= p, plus the first token that
    exceeds the threshold).

    Output:
        out_values: softmax probabilities of selected tokens (descending)
        out_indices: vocabulary indices of selected tokens
        out_count: number of tokens in the nucleus (single uint)

    Uses the top-k infrastructure internally: first finds the top max_k
    by probability, then applies the cumulative threshold.

    Args:
        max_k: Maximum number of candidates to consider.
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void top_p(
    device const float* logits [[buffer(0)]],
    device float* out_values [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device uint* out_count [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    constant float& p_threshold [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    const uint BLOCK = {block_size}u;
    const uint MAX_K = {max_k}u;
    const uint row_offset = pid * vocab_size;

    // Phase 1: Find global max for numerical stability
    threadgroup float tg_max[{block_size}];
    float local_max = -INFINITY;
    for (uint v = lid; v < vocab_size; v += BLOCK) {{
        local_max = max(local_max, logits[row_offset + v]);
    }}
    tg_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
        if (lid < stride) {{
            tg_max[lid] = max(tg_max[lid], tg_max[lid + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float global_max = tg_max[0];

    // Phase 2: Compute exp sum for softmax denominator
    threadgroup float tg_sum[{block_size}];
    float local_sum = 0.0f;
    for (uint v = lid; v < vocab_size; v += BLOCK) {{
        local_sum += exp((logits[row_offset + v] - global_max) / temperature);
    }}
    tg_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
        if (lid < stride) {{
            tg_sum[lid] += tg_sum[lid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float inv_sum = 1.0f / tg_sum[0];

    // Phase 3: Find top MAX_K candidates by softmax probability
    threadgroup float cand_vals[{max_k}];
    threadgroup uint cand_idxs[{max_k}];
    threadgroup float tg_vals[{block_size}];
    threadgroup uint tg_idxs[{block_size}];
    threadgroup uint picked[{max_k}];

    for (uint ki = 0u; ki < MAX_K; ki++) {{
        float best_val = -INFINITY;
        uint best_idx = 0u;

        for (uint v = lid; v < vocab_size; v += BLOCK) {{
            bool is_picked = false;
            for (uint pp = 0u; pp < ki; pp++) {{
                if (picked[pp] == v) {{ is_picked = true; break; }}
            }}
            if (!is_picked) {{
                float prob = exp((logits[row_offset + v] - global_max) / temperature) * inv_sum;
                if (prob > best_val) {{
                    best_val = prob;
                    best_idx = v;
                }}
            }}
        }}

        tg_vals[lid] = best_val;
        tg_idxs[lid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
            if (lid < stride) {{
                if (tg_vals[lid + stride] > tg_vals[lid]) {{
                    tg_vals[lid] = tg_vals[lid + stride];
                    tg_idxs[lid] = tg_idxs[lid + stride];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (lid == 0u) {{
            cand_vals[ki] = tg_vals[0];
            cand_idxs[ki] = tg_idxs[0];
            picked[ki] = tg_idxs[0];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Phase 4: Thread 0 applies cumulative probability threshold
    if (lid == 0u) {{
        float cum = 0.0f;
        uint count = 0u;
        for (uint i = 0u; i < MAX_K; i++) {{
            cum += cand_vals[i];
            uint out_offset = pid * MAX_K + i;
            out_values[out_offset] = cand_vals[i];
            out_indices[out_offset] = cand_idxs[i];
            count++;
            if (cum >= p_threshold) break;
        }}
        out_count[pid] = count;
    }}
}}
"""


def make_speculative_decode_kernel(block_size=256):
    """Generate a speculative decoding verification kernel.

    Given draft model logits and target model logits for a sequence of
    speculated tokens, computes acceptance probabilities and determines
    which tokens to accept using the standard speculative decoding algorithm:

        For each token i:
            p = target_prob[i][draft_token[i]]
            q = draft_prob[i][draft_token[i]]
            accept if random[i] < min(1, p/q)

    If token i is rejected, all subsequent tokens are also rejected.
    The kernel outputs the number of accepted tokens and the first
    rejected position's adjusted distribution (target - draft, renormalized)
    for resampling.

    Layout:
        draft_probs: [n_tokens, vocab_size] — draft model softmax probs
        target_probs: [n_tokens, vocab_size] — target model softmax probs
        draft_tokens: [n_tokens] — token IDs chosen by draft model (uint)
        rand_vals: [n_tokens] — uniform random values in [0,1] for acceptance
        n_accepted: [1] — output: number of accepted tokens (uint)
        adjusted_probs: [vocab_size] — output: resampling dist at rejection point
        n_tokens: number of speculated tokens
        vocab_size: vocabulary size

    Dispatch: 1 threadgroup (verification is sequential per-token).

    Args:
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void speculative_decode(
    device const float* draft_probs [[buffer(0)]],
    device const float* target_probs [[buffer(1)]],
    device const uint* draft_tokens [[buffer(2)]],
    device const float* rand_vals [[buffer(3)]],
    device uint* n_accepted [[buffer(4)]],
    device float* adjusted_probs [[buffer(5)]],
    constant uint& n_tokens [[buffer(6)]],
    constant uint& vocab_size [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    const uint BLOCK = {block_size}u;

    // Thread 0 does the sequential acceptance check
    threadgroup uint accepted_count;
    if (lid == 0u) {{
        accepted_count = n_tokens;  // assume all accepted
        for (uint i = 0u; i < n_tokens; i++) {{
            uint tok = draft_tokens[i];
            float p = target_probs[i * vocab_size + tok];
            float q = draft_probs[i * vocab_size + tok];
            float ratio = (q > 0.0f) ? (p / q) : 1.0f;
            float accept_prob = min(1.0f, ratio);
            if (rand_vals[i] >= accept_prob) {{
                accepted_count = i;
                break;
            }}
        }}
        n_accepted[0] = accepted_count;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rej_pos = accepted_count;

    // Compute adjusted distribution at the rejection point
    // If all accepted (rej_pos == n_tokens), output the target dist at last token
    uint dist_pos = (rej_pos < n_tokens) ? rej_pos : (n_tokens - 1u);

    // adjusted[v] = max(0, target[v] - draft[v]) then normalize
    // Each thread handles a slice of the vocab
    threadgroup float tg_sum[{block_size}];
    float local_sum = 0.0f;
    for (uint v = lid; v < vocab_size; v += BLOCK) {{
        float t = target_probs[dist_pos * vocab_size + v];
        float d = draft_probs[dist_pos * vocab_size + v];
        float adj = max(0.0f, t - d);
        adjusted_probs[v] = adj;
        local_sum += adj;
    }}
    tg_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum
    for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
        if (lid < stride) {{
            tg_sum[lid] += tg_sum[lid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float total = tg_sum[0];

    // Normalize
    float inv_total = (total > 0.0f) ? (1.0f / total) : 0.0f;
    for (uint v = lid; v < vocab_size; v += BLOCK) {{
        adjusted_probs[v] *= inv_total;
    }}
}}
"""


def make_beam_search_kernel(beam_width=4, block_size=256):
    """Generate a beam search step kernel for LLM inference.

    Given current beam scores and next-token log-probabilities, finds
    the top beam_width candidates across all beams * vocab_size options.

    For each beam b and vocabulary token v:
        candidate_score = beam_scores[b] + log_probs[b * vocab_size + v]

    Then selects the top beam_width candidates globally.

    Uses iterative selection (like top-k): find the global best, mark it,
    repeat beam_width times.

    Layout:
        beam_scores: [beam_width] — current cumulative log-probs per beam
        log_probs: [beam_width, vocab_size] — next-token log-probs from model
        out_scores: [beam_width] — selected beam scores
        out_beam_ids: [beam_width] — which beam each selection came from (uint)
        out_token_ids: [beam_width] — which token was selected (uint)
        vocab_size: vocabulary size

    Dispatch: 1 threadgroup.

    Args:
        beam_width: Number of beams.
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void beam_search(
    device const float* beam_scores [[buffer(0)]],
    device const float* log_probs [[buffer(1)]],
    device float* out_scores [[buffer(2)]],
    device uint* out_beam_ids [[buffer(3)]],
    device uint* out_token_ids [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    const uint BLOCK = {block_size}u;
    const uint BEAM_WIDTH = {beam_width}u;
    const uint total = BEAM_WIDTH * vocab_size;

    // Use threadgroup memory for iterative selection
    threadgroup float tg_best_val[{block_size}];
    threadgroup uint tg_best_idx[{block_size}];
    threadgroup float picked_threshold[1];

    // For each of beam_width selections
    for (uint sel = 0u; sel < BEAM_WIDTH; sel++) {{
        // Each thread finds its local best candidate
        float local_best = -INFINITY;
        uint local_idx = 0u;

        for (uint i = lid; i < total; i += BLOCK) {{
            uint beam = i / vocab_size;
            uint tok = i % vocab_size;
            float score = beam_scores[beam] + log_probs[i];

            // Skip already-picked candidates (marked with -INFINITY)
            if (sel > 0u) {{
                bool is_picked = false;
                for (uint p = 0u; p < sel; p++) {{
                    if (out_beam_ids[p] == beam && out_token_ids[p] == tok) {{
                        is_picked = true;
                        break;
                    }}
                }}
                if (is_picked) continue;
            }}

            if (score > local_best) {{
                local_best = score;
                local_idx = i;
            }}
        }}

        tg_best_val[lid] = local_best;
        tg_best_idx[lid] = local_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction to find global best
        for (uint stride = BLOCK / 2u; stride > 0u; stride >>= 1u) {{
            if (lid < stride) {{
                if (tg_best_val[lid + stride] > tg_best_val[lid]) {{
                    tg_best_val[lid] = tg_best_val[lid + stride];
                    tg_best_idx[lid] = tg_best_idx[lid + stride];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Thread 0 writes the selection
        if (lid == 0u) {{
            uint best_flat = tg_best_idx[0];
            uint best_beam = best_flat / vocab_size;
            uint best_tok = best_flat % vocab_size;
            out_scores[sel] = tg_best_val[0];
            out_beam_ids[sel] = best_beam;
            out_token_ids[sel] = best_tok;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""


def make_fused_mlp_kernel(block_size=256):
    """Generate a fused SwiGLU MLP kernel (LLaMA-style).

    output[i] = silu(gate[i]) * up[i]

    This is the element-wise fusion of the gate and up projections after
    the linear layers. The full LLaMA MLP is:
        gate = W_gate @ x
        up = W_up @ x
        output = silu(gate) * up

    This kernel fuses the silu activation with the element-wise multiply.
    The linear projections (W_gate, W_up) are handled separately.

    Layout:
        gate: [n] — output of W_gate @ x
        up: [n] — output of W_up @ x
        output: [n] — silu(gate) * up
        n_elements: total element count

    Args:
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void fused_mlp(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * up[gid];
}}
"""


def make_sliding_window_attention_kernel(head_dim=64, window_size=256, block_size=256):
    """Generate a sliding window attention kernel (Mistral-style).

    Each query token only attends to the last `window_size` tokens,
    reducing the O(n^2) attention to O(n * window_size).

    For a single query at position q_pos:
        window_start = max(0, q_pos - window_size + 1)
        Attend only to K[window_start:q_pos+1]

    Layout:
        Q: [head_dim] — single query vector
        K_cache: [max_seq_len, head_dim] — full KV-cache
        V_cache: [max_seq_len, head_dim]
        O: [head_dim] — output
        q_pos: query position
        seq_len: total sequence length

    Dispatch: 1 threadgroup.

    Args:
        head_dim: Dimension per head.
        window_size: Number of tokens to attend to.
        block_size: Threads per threadgroup.
    """
    n_simdgroups = (block_size + 31) // 32
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void sliding_window_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& q_pos [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    const uint D = {head_dim}u;
    const uint WINDOW = {window_size}u;
    const uint BLOCK = {block_size}u;
    const float scale = 1.0f / sqrt(float(D));

    // Window bounds
    uint win_end = min(q_pos + 1u, seq_len);
    uint win_start = (q_pos + 1u > WINDOW) ? (q_pos + 1u - WINDOW) : 0u;
    uint win_len = win_end - win_start;

    // Phase 1: max score within window
    float local_max = -INFINITY;
    for (uint w = lid; w < win_len; w += BLOCK) {{
        uint j = win_start + w;
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * K_cache[j * D + d];
        }}
        local_max = max(local_max, dot * scale);
    }}

    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{n_simdgroups}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = (tiisg < {n_simdgroups}u) ? shared_max[tiisg] : -INFINITY;
    float global_max = simd_max(rd_max);

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (uint w = lid; w < win_len; w += BLOCK) {{
        uint j = win_start + w;
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[d] * K_cache[j * D + d];
        }}
        local_sum += exp(dot * scale - global_max);
    }}

    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{n_simdgroups}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = (tiisg < {n_simdgroups}u) ? shared_sum[tiisg] : 0.0f;
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Weighted V sum
    for (uint d_start = 0u; d_start < D; d_start += BLOCK) {{
        uint d = d_start + lid;
        if (d < D) {{
            float o_val = 0.0f;
            for (uint w = 0u; w < win_len; w++) {{
                uint j = win_start + w;
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[dd] * K_cache[j * D + dd];
                }}
                float attn_w = exp(dot * scale - global_max) * inv_sum;
                o_val += attn_w * V_cache[j * D + d];
            }}
            O[d] = o_val;
        }}
    }}
}}
"""


def make_repeat_kv_kernel(block_size=256):
    """Generate a kernel that repeats KV heads for GQA inference.

    For Grouped Query Attention, the number of KV heads is less than
    the number of query heads. This kernel expands KV by repeating
    each KV head n_rep times.

    output[h, s, d] = input[h // n_rep, s, d]

    Layout:
        input: [n_kv_heads, seq_len, head_dim] — original KV
        output: [n_q_heads, seq_len, head_dim] — expanded KV
        n_kv_heads: number of KV heads
        seq_len: sequence length
        head_dim: dimension per head
        n_rep: repetition factor (n_q_heads / n_kv_heads)

    Each thread handles one element in the output.

    Args:
        block_size: Threads per threadgroup.
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void repeat_kv(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n_kv_heads [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& n_rep [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {{
    uint n_q_heads = n_kv_heads * n_rep;
    uint total = n_q_heads * seq_len * head_dim;
    if (gid >= total) return;

    // Decompose flat index: output[q_head, s, d]
    uint d = gid % head_dim;
    uint remainder = gid / head_dim;
    uint s = remainder % seq_len;
    uint q_head = remainder / seq_len;

    // Map query head to KV head
    uint kv_head = q_head / n_rep;

    // Read from input[kv_head, s, d]
    uint in_idx = (kv_head * seq_len + s) * head_dim + d;
    output[gid] = input[in_idx];
}}
"""


def make_matmul_splitk_fp32(rr=4, rc=2, bk=32):
    """Deterministic two-pass split-K fp32 matmul for SKINNY/DEEP shapes.

    Skinny/deep (small M/N, large K) starves occupancy: few output tiles → few
    threadgroups, each running the whole K-loop serially (measured ~0.1-0.4x torch).
    Split-K assigns G threadgroups per output tile (each computes 1/G of K), each
    writing its partial to ``P[G, M, N]`` (NO atomics — deterministic run-to-run),
    then a reduce kernel sums over G. Measured 10x over the serial path (1.1-1.7x
    over torch). Two kernels: ``mm_sk_partial`` + ``mm_sk_reduce``.

    DISPATCH: grid_partial = (ceil(M/(8rr)) * ceil(N/(8rc))) * G threadgroups, 32
    threads each; grid_reduce = M*N threads. CONTRACT: M % (8rr) == 0, N % (8rc) == 0,
    K % (G*bk) == 0 (each chunk a whole number of bk-steps).
    """
    BN = 8 * rc
    accs = "\n    ".join(f"simdgroup_float8x8 c{r}_{c}(0.0f);" for r in range(rr) for c in range(rc))
    afr = " ".join(f"simdgroup_float8x8 a{r};" for r in range(rr))
    bfr = " ".join(f"simdgroup_float8x8 b{c};" for c in range(rc))
    sub = []
    for kk in range(0, bk, 8):
        for r in range(rr):
            sub.append(f"simdgroup_load(a{r}, A + (m0 + {r * 8}u) * K + k + {kk}u, K);")
        for c in range(rc):
            sub.append(f"simdgroup_load(b{c}, B + (k + {kk}u) * N + n0 + {c * 8}u, N);")
        for r in range(rr):
            for c in range(rc):
                sub.append(f"simdgroup_multiply_accumulate(c{r}_{c}, a{r}, b{c}, c{r}_{c});")
    submma = "\n        ".join(sub)
    epi = "\n    ".join(
        f"simdgroup_store(c{r}_{c}, cbuf, 8);\n    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        f"    for (uint e = tid; e < 64u; e += 32u) {{ uint mm=e/8u, nn=e%8u; "
        f"P[kc*M*N + (m0+{r * 8}u+mm)*N + (n0+{c * 8}u+nn)] = cbuf[e]; }}\n"
        f"    simdgroup_barrier(mem_flags::mem_threadgroup);"
        for r in range(rr)
        for c in range(rc)
    )
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void mm_sk_partial(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* P [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]], constant uint& G [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {{
    uint ntn = N / {BN}u; uint tile = tgid / G; uint kc = tgid % G;
    uint m0 = (tile / ntn) * {8 * rr}u; uint n0 = (tile % ntn) * {BN}u;
    uint kchunk = K / G; uint kstart = kc * kchunk; uint kend = kstart + kchunk;
    {accs}
    {afr} {bfr}
    for (uint k = kstart; k < kend; k += {bk}u) {{ {submma} }}
    threadgroup float cbuf[64];
    {epi}
}}
kernel void mm_sk_reduce(
    device const float* P [[buffer(0)]], device float* C [[buffer(1)]],
    constant uint& MN [[buffer(2)]], constant uint& G [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= MN) return;
    float acc = 0.0f;
    for (uint g = 0u; g < G; g++) acc += P[g*MN + gid];
    C[gid] = acc;
}}
"""


def make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4, out_dtype="fp32"):
    """Fast matmul: direct-device simdgroup_load + register blocking (WS1 C.2).

    Much faster than the staged template by (a) loading simdgroup fragments
    DIRECTLY from device A/B (no threadgroup staging, no barriers — the GPU
    cache provides reuse) and (b) register blocking: each simdgroup computes an
    (8*rr) x (8*rc) output block (rr*rc accumulators), so each loaded a-/b-
    fragment feeds rc / rr MMAs.

    Measured throughput (2026-06, M4 Max — 546 GB/s, 18.4/36.9 TFLOP/s fp32/fp16
    peak), end-to-end at 2048^3 vs the ~2.8 TFLOP/s staged fallback:
        fp32  ~9.6-11.5 TFLOP/s  (~55-62% of fp32 peak; competitive with MLX/MPS GEMM)
        fp16  ~7.8-12   TFLOP/s  (~33% of fp16 peak)
    NOT "MLX parity": fp16 runs at roughly the fp32 rate because the accumulator
    is float (simdgroup_float8x8) for precision, so the hardware's ~2x fp16
    matrix throughput is deliberately forgone (a half-accumulate variant could
    approach fp16 peak at the cost of accuracy).

    INPUT fragments are half/float per `dtype`; the ACCUMULATOR is always float
    (precision). `out_dtype` selects the C store: "fp32" -> direct `simdgroup_store`
    to float* C; "fp16" -> half* C via a per-block cast epilogue.

    DISPATCH CONTRACT (the caller MUST honour this — different from the staged
    template's grid):
        threads/threadgroup = 128 (4 simdgroups)
        n_groups            = ceil(M / (8*rr)) * ceil(N / (32*rc))

    SIZE CONTRACT (required for correctness — direct simdgroup_load does not
    mask, so the caller must guarantee or refuse otherwise):
        M % (8*rr) == 0        (row tile; grid guarantees in-bounds rows)
        N % 32 == 0            (col strips align to 32; partial column tiles
                                beyond N are guarded off per-simdgroup, so any
                                multiple of 32 — incl. non-multiples of 32*rc —
                                is fine, but NOT arbitrary N)
        K % 8 == 0             (the K loop reads 8-deep fragments)
    With rr=rc=4: row tile 32, col tile 128, 16 accumulators/simdgroup.
    """
    if dtype in ("fp16", "f16"):
        in_t, in_frag, pad = "half", "simdgroup_half8x8", "half(0.0h)"
    elif dtype in ("bf16",):
        # M-series simdgroup matrix units DO support bfloat (verified
        # simdgroup_bfloat8x8 on M4); bf16 input + float accumulator, like fp16.
        in_t, in_frag, pad = "bfloat", "simdgroup_bfloat8x8", "bfloat(0.0)"
    elif dtype in ("fp32", "f32"):
        in_t, in_frag, pad = "float", "simdgroup_float8x8", "0.0f"
    else:
        raise ValueError(f"fast matmul supports fp16/bf16/fp32, got {dtype}")

    if out_dtype in ("fp32", "f32"):
        out_t = "float"
    elif out_dtype in ("fp16", "f16"):
        out_t = "half"
    elif out_dtype in ("bf16",):
        out_t = "bfloat"
    else:
        raise ValueError(f"fast matmul out_dtype supports fp16/bf16/fp32, got {out_dtype}")

    accs = "\n    ".join("simdgroup_float8x8 " + ", ".join(f"c{r}_{c}(0)" for c in range(rc)) + ";" for r in range(rr))
    bdecl = ", ".join(f"b{c}" for c in range(rc))
    loads_b = "\n        ".join(f"simdgroup_load(b{c}, B + k * N + col0 + {c * 8}u, N);" for c in range(rc))
    inner = []
    for r in range(rr):
        inner.append(f"simdgroup_load(a_frag, A + (row_base + {r * 8}u) * K + k, K);")
        for c in range(rc):
            inner.append(f"simdgroup_multiply_accumulate(c{r}_{c}, a_frag, b{c}, c{r}_{c});")
    inner = "\n        ".join(inner)
    if out_dtype in ("fp16", "f16", "bf16"):
        # float8x8 accumulator can't simdgroup_store directly to a half/bfloat
        # buffer -> store to a float scratch, then cast per-element to out_t.
        _tiisg_param = ",\n    uint tiisg [[thread_index_in_simdgroup]]"
        _scratch_line = "    threadgroup float scratch[4 * 64];\n"
        _epi = []
        for r in range(rr):
            for c in range(rc):
                _epi.append(f"simdgroup_store(c{r}_{c}, scratch + sgitg*64u, 8);")
                _epi.append("simdgroup_barrier(mem_flags::mem_threadgroup);")
                _epi.append(
                    f"for (uint i = tiisg; i < 64u; i += 32u) {{ "
                    f"C[(row_base + {r * 8}u + i / 8u) * N + col0 + {c * 8}u + i % 8u] "
                    f"= {out_t}(scratch[sgitg*64u + i]); }}"
                )
                _epi.append("simdgroup_barrier(mem_flags::mem_threadgroup);")
        stores = "\n    ".join(_epi)
    else:
        _tiisg_param = ""
        _scratch_line = ""
        stores = "\n    ".join(
            f"simdgroup_store(c{r}_{c}, C + (row_base + {r * 8}u) * N + col0 + {c * 8}u, N);"
            for r in range(rr)
            for c in range(rc)
        )

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul_fast(
    device const {in_t}* A [[buffer(0)]],
    device const {in_t}* B [[buffer(1)]],
    device {out_t}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]]{_tiisg_param}
) {{
    uint ntc = (N + {32 * rc - 1}u) / {32 * rc}u;
    uint row_base = (pid / ntc) * {8 * rr}u;
    uint col0 = (pid % ntc) * {32 * rc}u + sgitg * {8 * rc}u;
    if (col0 >= N) return;   // partial column tile: this simdgroup is OOB (uniform)
{_scratch_line}    {accs}
    {in_frag} a_frag, {bdecl};
    for (uint k = 0u; k < K; k += 8u) {{
        {loads_b}
        {inner}
    }}
    {stores}
}}
"""


def make_simdgroup_matmul_kernel(dtype="fp32"):
    """Generate a matmul kernel using Apple's simdgroup_matrix hardware.

    C[M,N] = A[M,K] * B[K,N]

    Uses simdgroup_matrix for hardware-accelerated 8x8 matrix
    multiply-accumulate. Each threadgroup (128 threads = 4 SIMD groups)
    computes a 32x32 output tile. Data is staged through threadgroup memory.

    Dispatch: block_size=128, n_groups = ceil(M/32) * ceil(N/32).
    Requires M, N to be multiples of 32 (no boundary masking on simdgroup_store).

    Args:
        dtype: "fp32" or "fp16". FP16 uses half inputs/outputs with float accumulation.
    """
    if dtype in ("fp32", "f32"):
        elem_type = "float"
        tg_type = "float"
        frag_type = "simdgroup_float8x8"
        zero = "0.0f"
        cast_load = "float"
        cast_store = ""
    elif dtype in ("fp16", "f16"):
        elem_type = "half"
        tg_type = "float"  # stage through float for precision
        frag_type = "simdgroup_float8x8"
        zero = "0.0f"
        cast_load = "float"
        cast_store = ""  # accumulator is float, output is float
    else:
        raise ValueError(f"simdgroup_matmul supports fp32 and fp16, got {dtype}")

    # FP16: half inputs, float accumulation, float output
    # (simdgroup_store with float accumulators requires float* destination)
    if dtype in ("fp16", "f16"):
        return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    // GENUINE fp16 + deeper K-tiling (WS1 Phase C). Half INPUT fragments,
    // float ACCUMULATOR (half x half -> float MMA, de-risked in
    // tests/test_simdgroup_half_mma.py). A 32-DEEP K-tile is staged per
    // barrier and the inner loop issues 16 MMAs (4 K-substeps x 4 row-blocks)
    // vs 4 in the old 8-deep template — amortizing the barrier so the kernel
    // is MMA-bound rather than sync-bound (the ~7 TFLOP/s plateau). fp16 =
    // INPUT precision; accumulator and C output stay float.
    uint n_tile_cols = (N + 31u) / 32u;
    uint row_base = (pid / n_tile_cols) * 32u;
    uint col_base_tg = (pid % n_tile_cols) * 32u;

    threadgroup half tg_A[32 * 32];   // [row(32) x k(32)]
    threadgroup half tg_B[32 * 32];   // [k(32)   x col(32)]
    simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);
    simdgroup_half8x8 a_frag, b_frag;

    for (uint k = 0u; k < K; k += 32u) {{
        for (uint i = tiitg; i < 1024u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0.0h);
        }}
        for (uint i = tiitg; i < 1024u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;
            uint gr = k + r, gc = col_base_tg + c;
            tg_B[i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0.0h);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {{
            simdgroup_load(b_frag, tg_B + kk * 32u + sgitg * 8u, 32);
            simdgroup_load(a_frag, tg_A + 0u * 8u * 32u + kk, 32);
            simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);
            simdgroup_load(a_frag, tg_A + 1u * 8u * 32u + kk, 32);
            simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);
            simdgroup_load(a_frag, tg_A + 2u * 8u * 32u + kk, 32);
            simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);
            simdgroup_load(a_frag, tg_A + 3u * 8u * 32u + kk, 32);
            simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    uint col = col_base_tg + sgitg * 8u;
    simdgroup_store(acc0, C + (row_base) * N + col, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col, N);
}}
"""

    # FP32 path
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul(
    device const {elem_type}* A [[buffer(0)]],
    device const {elem_type}* B [[buffer(1)]],
    device {elem_type}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    uint n_tile_cols = (N + 31u) / 32u;
    uint tile_row = pid / n_tile_cols;
    uint tile_col = pid % n_tile_cols;
    uint row_base = tile_row * 32u;
    uint col_base = tile_col * 32u + sgitg * 8u;

    {frag_type} acc0(0), acc1(0), acc2(0), acc3(0);
    {frag_type} a_frag, b_frag;

    threadgroup {tg_type} tg_A[32 * 8];
    threadgroup {tg_type} tg_B[8 * 32];

    for (uint k = 0u; k < K; k += 8u) {{
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 8u, c = i % 8u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? {cast_load}(A[gr * K + gc]) : {zero};
        }}
        uint col_base_tg = tile_col * 32u;
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;
            uint gr = k + r, gc = col_base_tg + c;
            tg_B[i] = (gr < K && gc < N) ? {cast_load}(B[gr * N + gc]) : {zero};
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);

        simdgroup_load(a_frag, tg_A, 8);
        simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);

        simdgroup_load(a_frag, tg_A + 64u, 8);
        simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);

        simdgroup_load(a_frag, tg_A + 128u, 8);
        simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);

        simdgroup_load(a_frag, tg_A + 192u, 8);
        simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(acc0, C + (row_base) * N + col_base, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col_base, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col_base, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col_base, N);
}}
"""


# ---------------------------------------------------------------------------
def make_instance_norm_kernel(block_size=256, dtype="fp32", eps=1e-5):
    """Generate an instance normalization kernel.

    Normalizes each channel independently across spatial dimensions.
    Used in style transfer, image generation, and some GANs.

    Input layout: [batch * channels, spatial_size]
    Each threadgroup handles one (batch, channel) pair.

    output = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("instance_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("weight", dtype=dtype, const=True)
    kb.add_ptr_arg("bias", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("spatial_size", dtype="u32")
    kb.add_scalar_arg("n_channels", dtype="u32")

    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    # pid indexes (batch * channel) pairs
    kb._var("ch_idx", "pid % n_channels", ty="uint")
    kb._var("base_offset", "pid * spatial_size", ty="uint")

    # Pass 1: Compute sum (for mean)
    kb._var("sum_val", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < spatial_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("sum_val += float(input[base_offset + i]);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "sum_val", "shared", "total_sum")

    # Broadcast mean
    kb.begin_if("lid == 0")
    kb.raw_line("shared[0] = total_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_val", "shared[0] / float(spatial_size)", ty="float")

    # Pass 2: Compute variance
    kb._var("sq_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < spatial_size; i += {block_size}u) {{")
    kb.indent()
    kb._var("diff", "float(input[base_offset + i]) - mean_val", ty="float")
    kb.raw_line("sq_sum += diff * diff;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "sq_sum", "shared", "total_sq")

    # Broadcast variance and compute inv_std
    kb.begin_if("lid == 0")
    kb.raw_line("shared[0] = total_sq;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("var_val", "shared[0] / float(spatial_size)", ty="float")
    kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

    # Pass 3: Normalize with per-channel affine transform
    kb.raw_line(f"for (uint i = lid; i < spatial_size; i += {block_size}u) {{")
    kb.indent()
    kb._var("normalized", "(float(input[base_offset + i]) - mean_val) * inv_std", ty="float")
    kb.raw_line("output[base_offset + i] = normalized * weight[ch_idx] + bias[ch_idx];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_fused_dropout_kernel(block_size=256, dtype="fp32", p=0.5):
    """Generate a fused dropout + scale kernel.

    Uses a simple counter-based hash (Philox-like) for random number generation.
    Fuses the masking and scaling into a single pass for efficiency.

    output = (x * mask) / (1 - p)  where mask = (hash(seed, idx) > p)

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        p: Dropout probability (fraction to zero out).
    """
    compute_ty = "half" if dtype == "fp16" else "float"
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

    kb = KernelBuilder("fused_dropout_kernel", block_size=block_size)
    msl = f"""#include <metal_stdlib>
using namespace metal;

// Simple hash for pseudo-random dropout mask
inline uint hash_philox(uint seed, uint idx) {{
    uint key = seed ^ 0xDEADBEEFu;
    uint counter = idx;
    // 4 rounds of Philox-like mixing
    for (int i = 0; i < 4; i++) {{
        counter ^= key;
        counter *= 0x9E3779B9u;
        counter ^= (counter >> 16u);
        key += 0x6C078965u;
    }}
    return counter;
}}

kernel void fused_dropout_kernel(
    device const {compute_ty}* input [[buffer(0)]],
    device {compute_ty}* output [[buffer(1)]],
    device {compute_ty}* mask_out [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    constant uint& seed [[buffer(4)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    uint gid = pid * {block_size}u + lid;
    if (gid >= n_elements) return;

    uint h = hash_philox(seed, gid);
    // Convert hash to uniform [0, 1) float
    float rnd = float(h) / 4294967296.0f;
    bool keep = rnd >= {p}f;

    {compute_ty} val = input[gid];
    {compute_ty} result = keep ? val * {compute_ty}({scale}f) : {compute_ty}(0.0f);
    output[gid] = result;
    mask_out[gid] = keep ? {compute_ty}(1.0f) : {compute_ty}(0.0f);
}}
"""
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_group_norm_kernel(n_groups=32, block_size=256, dtype="fp32", eps=1e-5):
    """Generate a group normalization kernel.

    Common in diffusion models (Stable Diffusion, DALL-E). Divides channels
    into groups and normalizes within each group.

    Input layout: [batch, channels, spatial] flattened to [batch, channels * spatial]
    Each threadgroup handles one (batch, group) pair.

    output = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        n_groups: Number of channel groups (typically 32).
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("group_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("weight", dtype=dtype, const=True)
    kb.add_ptr_arg("bias", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_channels", dtype="u32")
    kb.add_scalar_arg("spatial_size", dtype="u32")

    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    # pid indexes (batch, group) pairs
    kb._var("channels_per_group", f"n_channels / {n_groups}u", ty="uint")
    kb._var("group_size", "channels_per_group * spatial_size", ty="uint")
    kb._var("batch_idx", f"pid / {n_groups}u", ty="uint")
    kb._var("group_idx", f"pid % {n_groups}u", ty="uint")
    kb._var("base_offset", "batch_idx * n_channels * spatial_size + group_idx * group_size", ty="uint")

    # Pass 1: Compute sum (for mean)
    kb._var("sum_val", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < group_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("sum_val += float(input[base_offset + i]);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "sum_val", "shared", "total_sum")

    # Broadcast mean
    kb.begin_if("lid == 0")
    kb.raw_line("shared[0] = total_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_val", "shared[0] / float(group_size)", ty="float")

    # Pass 2: Compute sum of squared diffs (for variance)
    kb._var("sq_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < group_size; i += {block_size}u) {{")
    kb.indent()
    kb._var("diff", "float(input[base_offset + i]) - mean_val", ty="float")
    kb.raw_line("sq_sum += diff * diff;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "sq_sum", "shared", "total_sq")

    # Broadcast variance
    kb.begin_if("lid == 0")
    kb.raw_line("shared[0] = total_sq;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("var_val", f"shared[0] / float(group_size)", ty="float")
    kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

    # Pass 3: Normalize with affine transform
    kb.raw_line(f"for (uint i = lid; i < group_size; i += {block_size}u) {{")
    kb.indent()
    # Channel index within the group for weight/bias lookup
    kb._var("ch_in_group", "i / spatial_size", ty="uint")
    kb._var("ch_idx", "group_idx * channels_per_group + ch_in_group", ty="uint")
    kb._var("normalized", "(float(input[base_offset + i]) - mean_val) * inv_std", ty="float")
    kb.raw_line("output[base_offset + i] = normalized * weight[ch_idx] + bias[ch_idx];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_gather_kernel(block_size=256, dtype="fp32"):
    """Generate a gather kernel: output[i] = input[indices[i]].

    Supports arbitrary indexed reads (embedding lookup, KV-cache gather).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    compute_ty = _msl_compute_type(dtype)
    kb = KernelBuilder("gather_kernel", block_size=block_size)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void gather_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {msl_ty}* output [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    int idx = indices[gid];
    output[gid] = input[idx];
}}
"""
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_scatter_kernel(block_size=256, dtype="fp32"):
    """Generate a scatter kernel: output[indices[i]] = input[i].

    Atomic-free version (assumes no duplicate indices).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    kb = KernelBuilder("scatter_kernel", block_size=block_size)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void scatter_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {msl_ty}* output [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    int idx = indices[gid];
    output[idx] = input[gid];
}}
"""
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_transpose_kernel(block_size=256, tile_size=16, dtype="fp32"):
    """Generate a 2D matrix transpose kernel with threadgroup memory.

    Uses tiled transpose with shared memory for coalesced reads and writes.
    Each threadgroup handles a TILE_SIZE x TILE_SIZE tile.

    Args:
        block_size: Threads per threadgroup (should be tile_size * tile_size).
        tile_size: Tile dimension (16 for 16x16 tiles).
        dtype: Data type.
    """
    msl_ty = triton_type_to_msl(dtype)
    actual_block = tile_size * tile_size
    # Pad shared memory to avoid bank conflicts
    pad = 1

    kb = KernelBuilder("transpose_kernel", block_size=actual_block)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void transpose_kernel(
    device const {msl_ty}* input [[buffer(0)]],
    device {msl_ty}* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {{
    const uint TILE = {tile_size}u;
    threadgroup {msl_ty} tile[{tile_size}][{tile_size + pad}];

    uint tx = lid % TILE;
    uint ty = lid / TILE;

    // Read tile (coalesced along cols)
    uint read_row = gid.y * TILE + ty;
    uint read_col = gid.x * TILE + tx;
    if (read_row < rows && read_col < cols) {{
        tile[ty][tx] = input[read_row * cols + read_col];
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write transposed tile (coalesced along rows of output = cols of input)
    uint write_row = gid.x * TILE + ty;
    uint write_col = gid.y * TILE + tx;
    if (write_row < cols && write_col < rows) {{
        output[write_row * rows + write_col] = tile[tx][ty];
    }}
}}
"""
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_fused_linear_kernel(has_bias=True):
    """Generate a fused linear layer kernel: output = input @ weight^T + bias.

    Uses simdgroup_matrix for hardware-accelerated matmul with optional
    bias addition fused in. Each threadgroup computes a 32x32 output tile.

    This is the most common operation in transformers — used for all
    attention projections (Q, K, V, O) and FFN layers.

    Layout:
        input:  [M, K]
        weight: [N, K] (stored as row-major, transposed during compute)
        bias:   [N] (optional, broadcast across M dimension)
        output: [M, N]

    Dispatch: block_size=128, n_groups = ceil(M/32) * ceil(N/32).

    Args:
        has_bias: Whether to include bias addition.
    """
    bias_buffer = ""
    bias_param = ""
    bias_add = ""

    if has_bias:
        bias_param = "    device const float* bias [[buffer(3)]],\n"
        bias_buffer = ""
        bias_add = """
    // Add bias
    for (uint i = tiitg; i < 32u; i += 128u) {
        uint r = i / 32u;
        uint c = i % 32u;
        uint gc = col_base + c;
        // Bias is broadcast along M dimension — load once per column
        if (gc < N) {
            // For each row in the tile that this thread handles
            for (uint rr = 0u; rr < 32u; rr++) {
                uint gr = row_base + rr;
                if (gr < M) {
                    C[gr * N + gc] += bias[gc];
                }
            }
        }
    }"""
    else:
        bias_param = ""
        bias_add = ""

    # Buffer indices shift based on whether bias is present
    if has_bias:
        m_idx, n_idx, k_idx = 4, 5, 6
    else:
        m_idx, n_idx, k_idx = 3, 4, 5

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void fused_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* C [[buffer(2)]],
{bias_param}    constant uint& M [[buffer({m_idx})]],
    constant uint& N [[buffer({n_idx})]],
    constant uint& K [[buffer({k_idx})]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    uint n_tile_cols = (N + 31u) / 32u;
    uint tile_row = pid / n_tile_cols;
    uint tile_col = pid % n_tile_cols;
    uint row_base = tile_row * 32u;
    uint col_base = tile_col * 32u + sgitg * 8u;

    simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);
    simdgroup_float8x8 a_frag, b_frag;

    threadgroup float tg_A[32 * 8];
    threadgroup float tg_B[8 * 32];

    for (uint k = 0u; k < K; k += 8u) {{
        // Load input tile: input[row_base:row_base+32, k:k+8]
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 8u, c = i % 8u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }}
        // Load weight tile transposed: weight[col_base:col_base+32, k:k+8]
        // weight is [N, K], we want W^T, so we load weight[n, k] as B[k, n]
        uint col_base_tg = tile_col * 32u;
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;  // r is the K index, c is the N index
            uint gk = k + r, gn = col_base_tg + c;
            tg_B[i] = (gk < K && gn < N) ? weight[gn * K + gk] : 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);

        simdgroup_load(a_frag, tg_A, 8);
        simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);

        simdgroup_load(a_frag, tg_A + 64u, 8);
        simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);

        simdgroup_load(a_frag, tg_A + 128u, 8);
        simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);

        simdgroup_load(a_frag, tg_A + 192u, 8);
        simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(acc0, C + (row_base) * N + col_base, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col_base, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col_base, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col_base, N);
{bias_add}
}}
"""


def make_reduce_scatter_kernel(n_buffers=2, block_size=256, dtype="fp32"):
    """Generate a reduce-scatter kernel for multi-buffer reduction.

    Reduces n_buffers input buffers element-wise (sum), then scatters
    the result: each segment of the output gets a contiguous chunk of
    the reduced result.

    For single-GPU use, this simulates the collective operation by summing
    multiple input buffers and writing output segments. Useful as a building
    block for pipeline-parallel inference.

    Args:
        n_buffers: Number of input buffers to reduce.
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        input0..inputN-1: [total_elements] input buffers
        output: [total_elements] reduced output
        n_elements: total element count
    """
    msl_ty = triton_type_to_msl(dtype)

    # Build buffer parameters
    buffer_params = []
    for i in range(n_buffers):
        buffer_params.append(f"    device const {msl_ty}* input{i} [[buffer({i})]]")
    out_idx = n_buffers
    n_idx = n_buffers + 1
    buffer_params.append(f"    device {msl_ty}* output [[buffer({out_idx})]]")
    buffer_params.append(f"    constant uint& n_elements [[buffer({n_idx})]]")
    params_str = ",\n".join(buffer_params)

    # Build sum expression
    sum_parts = [f"input{i}[gid]" for i in range(n_buffers)]
    sum_expr = " + ".join(sum_parts)

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void reduce_scatter(
{params_str},
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    output[gid] = {sum_expr};
}}
"""
    kb = KernelBuilder("reduce_scatter", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_all_reduce_kernel(n_buffers=2, block_size=256, dtype="fp32", op="sum"):
    """Generate an all-reduce kernel for multi-buffer reduction.

    Reduces n_buffers input buffers element-wise and writes the full
    result to the output buffer (all ranks get the complete result).

    Supports sum, max, and min reductions.

    Args:
        n_buffers: Number of input buffers to reduce.
        block_size: Threads per threadgroup.
        dtype: Data type.
        op: Reduction operation ("sum", "max", "min").

    Kernel args:
        input0..inputN-1: [n_elements] input buffers
        output: [n_elements] reduced output
        n_elements: total element count
    """
    msl_ty = triton_type_to_msl(dtype)

    buffer_params = []
    for i in range(n_buffers):
        buffer_params.append(f"    device const {msl_ty}* input{i} [[buffer({i})]]")
    out_idx = n_buffers
    n_idx = n_buffers + 1
    buffer_params.append(f"    device {msl_ty}* output [[buffer({out_idx})]]")
    buffer_params.append(f"    constant uint& n_elements [[buffer({n_idx})]]")
    params_str = ",\n".join(buffer_params)

    # Build reduction expression based on op
    if op == "sum":
        sum_parts = [f"input{i}[gid]" for i in range(n_buffers)]
        reduce_expr = " + ".join(sum_parts)
    elif op == "max":
        reduce_expr = f"input0[gid]"
        for i in range(1, n_buffers):
            reduce_expr = f"max({reduce_expr}, input{i}[gid])"
    elif op == "min":
        reduce_expr = f"input0[gid]"
        for i in range(1, n_buffers):
            reduce_expr = f"min({reduce_expr}, input{i}[gid])"
    else:
        raise ValueError(f"Unsupported all-reduce op: {op}")

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void all_reduce(
{params_str},
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    output[gid] = {reduce_expr};
}}
"""
    kb = KernelBuilder("all_reduce", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_cumsum_kernel(block_size=256, dtype="fp32"):
    """Generate a parallel prefix sum (inclusive scan) kernel.

    Uses the Hillis-Steele algorithm within a threadgroup for in-place
    prefix sum. Each threadgroup processes block_size elements.
    For inputs larger than block_size, launch multiple passes.

    The kernel processes one row per threadgroup (pid indexes rows).

    Args:
        block_size: Elements per threadgroup (also max row length).
        dtype: Data type.
    """
    msl_type = triton_type_to_msl(dtype)
    needs_cast = dtype in ("fp16", "bf16")
    read_cast = f"float({msl_type}(" if needs_cast else ""
    read_end = "))" if needs_cast else ""

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void cumsum_kernel(
    device const {msl_type}* input [[buffer(0)]],
    device {msl_type}* output [[buffer(1)]],
    constant uint& n_cols [[buffer(2)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // Each threadgroup handles one row
    threadgroup float shared[{block_size}];

    uint row_start = pid * n_cols;
    // Load into shared memory
    if (lid < n_cols) {{
        shared[lid] = float(input[row_start + lid]);
    }} else {{
        shared[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint stride = 1; stride < {block_size}u; stride <<= 1) {{
        float val = 0.0f;
        if (lid >= stride) {{
            val = shared[lid - stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid >= stride) {{
            shared[lid] += val;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write result
    if (lid < n_cols) {{
        output[row_start + lid] = {msl_type}(shared[lid]);
    }}
}}
"""
    kb = KernelBuilder("cumsum_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_bitonic_sort_kernel(block_size=256, ascending=True):
    """Generate a bitonic sort kernel for in-threadgroup sorting.

    Sorts block_size elements per threadgroup using the bitonic merge
    sort network. Returns both sorted values and original indices
    (argsort).

    Each threadgroup handles one segment of block_size elements,
    indexed by pid.

    Args:
        block_size: Number of elements to sort per threadgroup (must be power of 2).
        ascending: Sort order.
    """
    cmp = "<" if ascending else ">"
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void bitonic_sort_kernel(
    device const float* input [[buffer(0)]],
    device float* output_vals [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    threadgroup float vals[{block_size}];
    threadgroup uint idxs[{block_size}];

    uint base = pid * {block_size}u;
    uint gid = base + lid;

    // Load or pad
    if (gid < n_elements) {{
        vals[lid] = input[gid];
        idxs[lid] = gid;
    }} else {{
        vals[lid] = {"INFINITY" if ascending else "-INFINITY"};
        idxs[lid] = 0xFFFFFFFFu;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort network
    for (uint size = 2; size <= {block_size}u; size <<= 1) {{
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {{
            uint pos = lid;
            uint partner = pos ^ stride;
            if (partner > pos) {{
                bool ascending_pair = ((pos & size) == 0);
                bool should_swap;
                if (ascending_pair) {{
                    should_swap = vals[pos] > vals[partner];
                }} else {{
                    should_swap = vals[pos] < vals[partner];
                }}
                if (should_swap) {{
                    float tmp_v = vals[pos];
                    vals[pos] = vals[partner];
                    vals[partner] = tmp_v;
                    uint tmp_i = idxs[pos];
                    idxs[pos] = idxs[partner];
                    idxs[partner] = tmp_i;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}

    // Write sorted output
    if (gid < n_elements) {{
        output_vals[gid] = vals[lid];
        output_indices[gid] = idxs[lid];
    }}
}}
"""
    kb = KernelBuilder("bitonic_sort_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_atomic_add_kernel(block_size=256, dtype="fp32"):
    """Generate a kernel that performs atomic addition to a global accumulator.

    Each thread atomically adds its input element to the corresponding
    output location. Used for gradient accumulation, histogram computation,
    and scatter-add operations.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type (fp32 only — Metal atomics are int32/uint32/float).
    """
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void atomic_add_kernel(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    uint idx = indices[gid];
    atomic_fetch_add_explicit(&output[idx], input[gid], memory_order_relaxed);
}}
"""
    kb = KernelBuilder("atomic_add_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_conv2d_kernel(
    in_channels=3, out_channels=64, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, block_size=256
):
    """Generate a 2D convolution kernel using direct computation.

    Each thread computes one output element by iterating over the filter
    window. Suitable for small kernels (3x3, 1x1). For larger kernels,
    im2col + matmul is more efficient.

    Dispatch: n_threadgroups = ceil(out_h * out_w * out_channels / block_size)
    Scalar args: batch_size, in_h, in_w, out_h, out_w

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_h, kernel_w: Filter dimensions.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Zero-padding.
        block_size: Threads per threadgroup.
    """
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void conv2d_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& in_h [[buffer(5)]],
    constant uint& in_w [[buffer(6)]],
    constant uint& out_h [[buffer(7)]],
    constant uint& out_w [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {{
    // Total output elements per batch: out_channels * out_h * out_w
    uint out_spatial = out_h * out_w;
    uint out_total = {out_channels}u * out_spatial;
    uint total = batch_size * out_total;
    if (gid >= total) return;

    uint b = gid / out_total;
    uint rem = gid % out_total;
    uint oc = rem / out_spatial;
    uint spatial = rem % out_spatial;
    uint oh = spatial / out_w;
    uint ow = spatial % out_w;

    float acc = bias[oc];

    for (uint ic = 0; ic < {in_channels}u; ic++) {{
        for (uint kh = 0; kh < {kernel_h}u; kh++) {{
            for (uint kw = 0; kw < {kernel_w}u; kw++) {{
                int ih = int(oh * {stride_h}u + kh) - {pad_h};
                int iw = int(ow * {stride_w}u + kw) - {pad_w};
                if (ih >= 0 && ih < int(in_h) && iw >= 0 && iw < int(in_w)) {{
                    uint in_idx = b * ({in_channels}u * in_h * in_w) + ic * (in_h * in_w) + uint(ih) * in_w + uint(iw);
                    uint w_idx = oc * ({in_channels}u * {kernel_h}u * {kernel_w}u) + ic * ({kernel_h}u * {kernel_w}u) + kh * {kernel_w}u + kw;
                    acc += input[in_idx] * weight[w_idx];
                }}
            }}
        }}
    }}

    uint out_idx = b * out_total + oc * out_spatial + oh * out_w + ow;
    output[out_idx] = acc;
}}
"""
    kb = KernelBuilder("conv2d_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_max_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, pad_h=0, pad_w=0, block_size=256):
    """Generate a 2D max pooling kernel.

    Each thread computes one output element by scanning the pooling window.
    Dispatch: ceil(batch * channels * out_h * out_w / block_size) threadgroups.

    Args:
        kernel_h, kernel_w: Pooling window size.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Zero-padding.
        block_size: Threads per threadgroup.
    """
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_h [[buffer(6)]],
    constant uint& out_w [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {{
    uint out_spatial = out_h * out_w;
    uint out_per_batch = channels * out_spatial;
    uint total = batch_size * out_per_batch;
    if (gid >= total) return;

    uint b = gid / out_per_batch;
    uint rem = gid % out_per_batch;
    uint c = rem / out_spatial;
    uint spatial = rem % out_spatial;
    uint oh = spatial / out_w;
    uint ow = spatial % out_w;

    float max_val = -INFINITY;
    for (uint kh = 0; kh < {kernel_h}u; kh++) {{
        for (uint kw = 0; kw < {kernel_w}u; kw++) {{
            int ih = int(oh * {stride_h}u + kh) - {pad_h};
            int iw = int(ow * {stride_w}u + kw) - {pad_w};
            if (ih >= 0 && ih < int(in_h) && iw >= 0 && iw < int(in_w)) {{
                uint idx = b * (channels * in_h * in_w) + c * (in_h * in_w) + uint(ih) * in_w + uint(iw);
                max_val = max(max_val, input[idx]);
            }}
        }}
    }}

    output[gid] = max_val;
}}
"""
    kb = KernelBuilder("max_pool2d_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_avg_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, pad_h=0, pad_w=0, block_size=256):
    """Generate a 2D average pooling kernel.

    Each thread computes one output element by averaging the pooling window.

    Args:
        kernel_h, kernel_w: Pooling window size.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Zero-padding.
        block_size: Threads per threadgroup.
    """
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void avg_pool2d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_h [[buffer(6)]],
    constant uint& out_w [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {{
    uint out_spatial = out_h * out_w;
    uint out_per_batch = channels * out_spatial;
    uint total = batch_size * out_per_batch;
    if (gid >= total) return;

    uint b = gid / out_per_batch;
    uint rem = gid % out_per_batch;
    uint c = rem / out_spatial;
    uint spatial = rem % out_spatial;
    uint oh = spatial / out_w;
    uint ow = spatial % out_w;

    float sum_val = 0.0f;
    uint count = 0;
    for (uint kh = 0; kh < {kernel_h}u; kh++) {{
        for (uint kw = 0; kw < {kernel_w}u; kw++) {{
            int ih = int(oh * {stride_h}u + kh) - {pad_h};
            int iw = int(ow * {stride_w}u + kw) - {pad_w};
            if (ih >= 0 && ih < int(in_h) && iw >= 0 && iw < int(in_w)) {{
                uint idx = b * (channels * in_h * in_w) + c * (in_h * in_w) + uint(ih) * in_w + uint(iw);
                sum_val += input[idx];
                count++;
            }}
        }}
    }}

    output[gid] = (count > 0) ? (sum_val / float(count)) : 0.0f;
}}
"""
    kb = KernelBuilder("avg_pool2d_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_index_select_kernel(block_size=256, dtype="fp32"):
    """Generate an index_select kernel: output[i] = input[indices[i]].

    Gather elements from input at positions specified by indices.
    More general than the existing gather kernel (which operates on 2D with axis).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_type = triton_type_to_msl(dtype)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void index_select_kernel(
    device const {msl_type}* input [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device {msl_type}* output [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    output[gid] = input[indices[gid]];
}}
"""
    kb = KernelBuilder("index_select_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_where_kernel(block_size=256, dtype="fp32"):
    """Generate a where kernel: output[i] = cond[i] ? x[i] : y[i].

    Conditional element-wise selection between two tensors based on a
    boolean mask. The mask is stored as uint (0 or 1).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_type = triton_type_to_msl(dtype)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void where_kernel(
    device const uint* condition [[buffer(0)]],
    device const {msl_type}* x [[buffer(1)]],
    device const {msl_type}* y [[buffer(2)]],
    device {msl_type}* output [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    output[gid] = condition[gid] ? x[gid] : y[gid];
}}
"""
    kb = KernelBuilder("where_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_clamp_kernel(block_size=256, dtype="fp32"):
    """Generate a clamp kernel: output[i] = clamp(input[i], min_val, max_val).

    Clips values to [min_val, max_val] range. Used in gradient clipping,
    ReLU6, and numerical stability.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_type = triton_type_to_msl(dtype)
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void clamp_kernel(
    device const {msl_type}* input [[buffer(0)]],
    device {msl_type}* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    float val = float(input[gid]);
    val = max(min_val, min(max_val, val));
    output[gid] = {msl_type}(val);
}}
"""
    kb = KernelBuilder("clamp_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_compare_kernel(op="eq", block_size=256, dtype="fp32"):
    """Generate an element-wise comparison kernel: output[i] = (a[i] op b[i]) ? 1 : 0.

    Args:
        op: Comparison operator ("eq", "ne", "lt", "le", "gt", "ge").
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    msl_type = triton_type_to_msl(dtype)
    ops = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    msl_op = ops[op]
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void compare_{op}_kernel(
    device const {msl_type}* a [[buffer(0)]],
    device const {msl_type}* b [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant uint& n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    output[gid] = (float(a[gid]) {msl_op} float(b[gid])) ? 1u : 0u;
}}
"""
    kb = KernelBuilder(f"compare_{op}_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


# Deferred import (see the note near the top): placed AFTER all defs so the
# circular re-export with msl_emitter resolves regardless of import order (#152).
# These names are only referenced inside the functions above, never at module
# load time, so binding them here (before any function is called) is sufficient.
from triton_msl.codegen.msl_emitter import (  # noqa: E402
    KernelBuilder,
    _msl_compute_type,
    _msl_zero,
    _sanitize_msl_name,
)


def make_kda_kernel(fp16=False):
    """Gated DeltaNet / Kimi Delta Attention (KDA) — chunked PREFILL, MMA-optimized.

    ``fp16=True`` makes the I/O (q,k,v,a,beta,out) half; MSL auto-promotes half to float in
    arithmetic so the accumulate and state stay fp32, and only the MMA output store converts
    float8x8 -> half8x8. Validated fp16 rel ~6e-4 vs an fp64 reference.


    The linear/delta-rule attention the 2026 frontier models (Kimi, DeltaNet family)
    use; NOT standard softmax attention, so the simdgroup FlashAttention path does not
    cover it. Recurrence: ``Sg = diag(a_t) S; u = k_t^T Sg; S = Sg + b_t k_t (v_t-u)^T;
    o_t = q_t^T S`` (per-key-dim gate ``a`` + delta-rule correction, scalar ``b`` = beta).

    Chunked form (C=8): ``B = cumprod(a)`` within chunk; gate q~=q*B, split keys
    k~=k/B (accumulation) and k^=k*B (read); ``M = tril(diag(b) k^ k~^T, -1)``; solve
    ``(I+M) W = diag(b)(V - k^ S)`` by forward substitution (the UT transform); output
    ``o = q~ S + tril(q~ k~^T) W``; carry ``S = diag(B_last)(S + k~^T W)``.

    The four D-contraction matmuls run on simdgroup_float8x8; the triangular solve and the
    (cheap) C-contraction state update stay scalar. fp32 accumulate + state (models feed
    fp16 I/O; a half-I/O variant casts on load). Validated rel ~3e-7 vs the recurrent
    ground truth; ~7-12x over the naive scalar kernel; 0.69ms T=512 8-head D=64 on M4 Max.

    Layout: q,k,v [ZH,T,64] float, a [ZH,T,64] float in (0,1), beta [ZH,T] float; out
    [ZH,T,64]. DISPATCH: one threadgroup per head, 256 threads (8 simdgroups); grid
    threads=(ZH*256, 1, 1), group=(256,1,1). Constant: T (buffer 6). T % 8 == 0.

    Non-finite inputs, non-normal cumulative gates, non-finite transformed operands,
    or non-finite chunk state/output trigger a whole-head recurrent replay in the same
    dispatch. This preserves causal exceptional-value behavior without inventing the
    chunk transform's zero-times-nonfinite terms. Finite inputs alone do not establish
    safety: their gate products can underflow. The finite chunk path remains an
    approximate, reordered evaluation, not a bit-exact IEEE recurrence proof.
    """
    src = r"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void kda_prefill(
  device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],
  device const float* beta [[buffer(4)]], device float* Out [[buffer(5)]],
  constant uint& T [[buffer(6)]],
  uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const uint D=64u, C=8u, NT=256u;
  uint head=tg.x; uint hb=head*T*D; uint bb=head*T;
  threadgroup float S[64*64]; threadgroup float B[8*64];
  threadgroup float QT[8*64]; threadgroup float KT[8*64]; threadgroup float KH[8*64];
  threadgroup float M[8*8]; threadgroup float W[8*64];
  // Chunk algebra introduces non-source terms (Inf*zero old state, or
  // zero triangular entries times future NaNs). Replay the documented
  // recurrence for this head when that algebra leaves its finite domain.
  threadgroup atomic_uint replay;
  if(lid==0u) atomic_store_explicit(&replay,0u,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint idx=lid;idx<T*D;idx+=NT){
    uint p=hb+idx;
    if(!isfinite(float(q[p])) || !isfinite(float(k[p])) ||
       !isfinite(float(v[p])) || !isfinite(float(a[p])))
      atomic_store_explicit(&replay,1u,memory_order_relaxed);
  }
  for(uint i=lid;i<T;i+=NT)
    if(!isfinite(float(beta[bb+i]))) atomic_store_explicit(&replay,1u,memory_order_relaxed);
  for(uint i=lid;i<D*D;i+=NT) S[i]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint nch=T/C;
  for(uint ch=0u; ch<nch; ch++){
    // Every branch on replay follows a uniform threadgroup barrier.
    if(atomic_load_explicit(&replay,memory_order_relaxed)) break;
    uint base=ch*C;
    if(lid<D){ uint l=lid; float b=1.0f; for(uint i=0u;i<C;i++){
      b*=float(a[hb+(base+i)*D+l]); B[i*D+l]=b;
      // Do not divide by a zero/subnormal/overflowed cumulative gate.
      if(!isnormal(b)) atomic_store_explicit(&replay,1u,memory_order_relaxed);
    } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(atomic_load_explicit(&replay,memory_order_relaxed)) break;
    for(uint idx=lid; idx<C*D; idx+=NT){ uint i=idx/D, l=idx%D; float bl=B[i*D+l]; float kk=k[hb+(base+i)*D+l];
      QT[idx]=q[hb+(base+i)*D+l]*bl; KT[idx]=kk/bl; KH[idx]=kk*bl;
      if(!isfinite(QT[idx]) || !isfinite(KT[idx]) || !isfinite(KH[idx]))
        atomic_store_explicit(&replay,1u,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(atomic_load_explicit(&replay,memory_order_relaxed)) break;
    if(sgitg==0u){ simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<D/8u;dt++){ simdgroup_load(af,KH+dt*8u,D); simdgroup_load(bf,KT+dt*8u,D,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,M,8u); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint idx=lid; idx<C*C; idx+=NT){ uint i=idx/C, j=idx%C; M[idx]=(j<i)?beta[bb+base+i]*M[idx]:0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    { simdgroup_float8x8 acc(0.0f),af,sf;
      for(uint dt=0u;dt<D/8u;dt++){ simdgroup_load(af,KH+dt*8u,D); simdgroup_load(sf,S+(dt*8u)*D+sgitg*8u,D); simdgroup_multiply_accumulate(acc,af,sf,acc); }
      simdgroup_store(acc,W+sgitg*8u,D); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint idx=lid; idx<C*D; idx+=NT){ uint i=idx/D, j=idx%D; W[idx]=beta[bb+base+i]*(v[hb+(base+i)*D+j]-W[idx]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=0u;i<C;i++){ for(uint jc=lid; jc<D; jc+=NT){ float acc=W[i*D+jc]; for(uint j=0u;j<i;j++) acc-=M[i*C+j]*W[j*D+jc]; W[i*D+jc]=acc; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if(sgitg==0u){ simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<D/8u;dt++){ simdgroup_load(af,QT+dt*8u,D); simdgroup_load(bf,KT+dt*8u,D,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,M,8u); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint idx=lid; idx<C*C; idx+=NT){ uint i=idx/C, j=idx%C; if(j>i) M[idx]=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    { simdgroup_float8x8 acc(0.0f),af,sf;
      for(uint dt=0u;dt<D/8u;dt++){ simdgroup_load(af,QT+dt*8u,D); simdgroup_load(sf,S+(dt*8u)*D+sgitg*8u,D); simdgroup_multiply_accumulate(acc,af,sf,acc); }
      simdgroup_float8x8 pf,wf; simdgroup_load(pf,M,8u); simdgroup_load(wf,W+sgitg*8u,D); simdgroup_multiply_accumulate(acc,pf,wf,acc);
      simdgroup_store(acc,Out+hb+base*D+sgitg*8u,D); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint idx=lid; idx<D*D; idx+=NT){ uint l=idx/D, j=idx%D; float d=0.0f;
      for(uint i=0u;i<C;i++) d += KT[i*D+l]*W[i*D+j];
      float Bl=B[(C-1u)*D+l]; S[l*D+j]=Bl*(S[l*D+j]+d);
      if(!isfinite(S[l*D+j])) atomic_store_explicit(&replay,1u,memory_order_relaxed); }
    // The diagnostic reads Out through different lanes than the MMA store.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for(uint idx=lid;idx<C*D;idx+=NT)
      if(!isfinite(W[idx]) || !isfinite(float(Out[hb+base*D+idx])))
        atomic_store_explicit(&replay,1u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(atomic_load_explicit(&replay,memory_order_relaxed)){
    // KDA_SOURCE_RECURRENCE_REPLAY: restart from the specified zero state;
    // never reuse partially transformed state or output. Reuse dead W as u.
    for(uint idx=lid;idx<D*D;idx+=NT) S[idx]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=0u;t<T;t++){
      uint row=hb+t*D;
      for(uint idx=lid;idx<D*D;idx+=NT)
        S[idx]=float(a[row+idx/D])*S[idx];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(lid<D){
        float u=0.0f;
        for(uint l=0u;l<D;l++) u+=float(k[row+l])*S[l*D+lid];
        W[lid]=u;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint idx=lid;idx<D*D;idx+=NT){
        uint l=idx/D, j=idx%D;
        float delta=float(k[row+l])*(float(v[row+j])-W[j]);
        S[idx]=S[idx]+float(beta[bb+t])*delta;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(lid<D){
        float o=0.0f;
        for(uint l=0u;l<D;l++) o+=float(q[row+l])*S[l*D+lid];
        Out[row+lid]=o;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
"""
    if fp16:
        src = src.replace(
            "device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],\n"
            "  device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],\n"
            "  device const float* beta [[buffer(4)]], device float* Out [[buffer(5)]],",
            "device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],\n"
            "  device const half* v [[buffer(2)]], device const half* a [[buffer(3)]],\n"
            "  device const half* beta [[buffer(4)]], device half* Out [[buffer(5)]],",
        )
        src = src.replace(
            "      simdgroup_store(acc,Out+hb+base*D+sgitg*8u,D); }",
            "      simdgroup_half8x8 oh; oh.thread_elements()[0]=half(acc.thread_elements()[0]);"
            " oh.thread_elements()[1]=half(acc.thread_elements()[1]);"
            " simdgroup_store(oh,Out+hb+base*D+sgitg*8u,D); }",
        )
    return src


def make_kda_decode_kernel():
    """KDA (gated DeltaNet) single-step recurrent DECODE — the autoregressive inference step.

    One token per call: ``Sg = diag(a) S; u = k^T Sg; S = Sg + beta k (v-u)^T; o = q^T S``.
    The state ``S`` [ZH,64,64] is read and written in place (device buffer 5) so the caller
    threads it across steps. Memory-bound (reads/writes the D*D state). Validated forward
    T steps vs the recurrent ground truth (rel ~3e-7). See ``make_kda_kernel`` for prefill.

    Layout: q,k,v,a [ZH,64] float, beta [ZH] float, S [ZH,64,64] float (in/out), out [ZH,64]
    float. DISPATCH: one threadgroup per head, 256 threads; threads=(ZH*256,1,1), group=(256,1,1).
    """
    return r"""#include <metal_stdlib>
using namespace metal;
kernel void kda_decode(
  device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],
  device const float* beta [[buffer(4)]], device float* S [[buffer(5)]],
  device float* Out [[buffer(6)]],
  uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  const uint D=64u, NT=256u;
  uint head=tg.x; uint hd=head*D; uint hS=head*D*D;
  threadgroup float u[64];
  for(uint j=lid; j<D; j+=NT){ float acc=0.0f; for(uint l=0u;l<D;l++) acc += k[hd+l]*a[hd+l]*S[hS+l*D+j]; u[j]=acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  float bt=beta[head];
  for(uint idx=lid; idx<D*D; idx+=NT){ uint l=idx/D, j=idx%D;
    S[hS+idx] = a[hd+l]*S[hS+idx] + bt*k[hd+l]*(v[hd+j]-u[j]); }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for(uint j=lid; j<D; j+=NT){ float acc=0.0f; for(uint l=0u;l<D;l++) acc += q[hd+l]*S[hS+l*D+j]; Out[hd+j]=acc; }
}
"""


def make_fa_backward_dkv_kernel(causal=False):
    """FlashAttention-2 BACKWARD dK/dV — tiled, MMA-optimized (head_dim=64).

    One threadgroup per (kv-block, head); accumulates dKj/dVj over query blocks in
    simdgroup registers (no threadgroup reload). Recomputes P from the saved logsumexp
    ``L``; ``D = rowsum(dO*O)`` is precomputed. The four D-contraction matmuls run on
    simdgroup_float8x8; P/dS is scalar in between. Br=Bc=16 (measured MMA optimum). Enables
    TRAINING (the forward path was inference-only). Validated vs torch autograd, rel ~1e-6.

    Layout: Q,K,V,dO [ZH,N,64] float, L,Dr [ZH,N] float; dK,dV [ZH,N,64] float out.
    DISPATCH: (N/16) x ZH threadgroups, 256 threads; threads=((N/16)*256, ZH, 1). N % 16 == 0.
    """
    skip = "jb" if causal else "0u"
    mask = "((i0+r)<(j0+c)) ? 0.0f : " if causal else ""
    return (
        r"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void fa_bwd_dkv(
  device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
  device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* L [[buffer(4)]],
  device const float* Dr [[buffer(5)]], device float* dK [[buffer(6)]], device float* dV [[buffer(7)]],
  constant uint& N [[buffer(8)]], constant float& scale [[buffer(9)]],
  uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const uint Bc=16u,Br=16u,Dh=64u,NT=256u; uint jb=tg.x,head=tg.y; uint hb=head*N*Dh,hl=head*N,j0=jb*Bc;
  threadgroup float Kj[16*64],Vj[16*64],Qi[16*64],dOi[16*64],Ss[16*16],Ds[16*16];
  simdgroup_float8x8 dV0(0.0f),dV1(0.0f),dK0(0.0f),dK1(0.0f);
  for(uint x=lid;x<Bc*Dh;x+=NT){ Kj[x]=K[hb+(j0+x/Dh)*Dh+x%Dh]; Vj[x]=V[hb+(j0+x/Dh)*Dh+x%Dh]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint ib="""
        + skip
        + r"""; ib<N/Br; ib++){ uint i0=ib*Br;
    for(uint x=lid;x<Br*Dh;x+=NT){ Qi[x]=Q[hb+(i0+x/Dh)*Dh+x%Dh]; dOi[x]=dO[hb+(i0+x/Dh)*Dh+x%Dh]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sgitg<4u){ uint ro=sgitg/2u, co=sgitg%2u; simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<Dh/8u;dt++){ simdgroup_load(af,Qi+ro*8u*Dh+dt*8u,Dh); simdgroup_load(bf,Kj+co*8u*Dh+dt*8u,Dh,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,Ss+ro*8u*16u+co*8u,16u); }
    else { uint s2=sgitg-4u, ro=s2/2u, co=s2%2u; simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<Dh/8u;dt++){ simdgroup_load(af,dOi+ro*8u*Dh+dt*8u,Dh); simdgroup_load(bf,Vj+co*8u*Dh+dt*8u,Dh,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,Ds+ro*8u*16u+co*8u,16u); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint x=lid;x<Br*Bc;x+=NT){ uint r=x/Bc, c=x%Bc; float p="""
        + mask
        + r"""exp(scale*Ss[x]-L[hl+i0+r]); Ss[x]=p; Ds[x]=p*(Ds[x]-Dr[hl+i0+r]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sgitg<Dh/8u){ simdgroup_float8x8 pf,df,qf,of;
      for(uint rt=0u;rt<Br/8u;rt++){
        simdgroup_load(of,dOi+rt*8u*Dh+sgitg*8u,Dh); simdgroup_load(qf,Qi+rt*8u*Dh+sgitg*8u,Dh);
        simdgroup_load(pf,Ss+rt*8u*16u+0u*8u,16u,0,true); simdgroup_load(df,Ds+rt*8u*16u+0u*8u,16u,0,true);
        simdgroup_multiply_accumulate(dV0,pf,of,dV0); simdgroup_multiply_accumulate(dK0,df,qf,dK0);
        simdgroup_load(pf,Ss+rt*8u*16u+1u*8u,16u,0,true); simdgroup_load(df,Ds+rt*8u*16u+1u*8u,16u,0,true);
        simdgroup_multiply_accumulate(dV1,pf,of,dV1); simdgroup_multiply_accumulate(dK1,df,qf,dK1);
      } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(sgitg<Dh/8u){ simdgroup_store(dV0,Vj+0u*8u*Dh+sgitg*8u,Dh); simdgroup_store(dV1,Vj+1u*8u*Dh+sgitg*8u,Dh);
    simdgroup_store(dK0,Kj+0u*8u*Dh+sgitg*8u,Dh); simdgroup_store(dK1,Kj+1u*8u*Dh+sgitg*8u,Dh); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint x=lid;x<Bc*Dh;x+=NT){ dV[hb+(j0+x/Dh)*Dh+x%Dh]=Vj[x]; dK[hb+(j0+x/Dh)*Dh+x%Dh]=scale*Kj[x]; }
}
"""
    )


def make_fa_backward_dq_kernel(causal=False):
    """FlashAttention-2 BACKWARD dQ — tiled, MMA-optimized (head_dim=64).

    One threadgroup per (query-block, head); accumulates dQi over kv-blocks in registers.
    Split from dK/dV because dQ accumulates over KV while dK/dV accumulate over Q (the
    standard FA-2 two-kernel split, avoiding atomics). See ``make_fa_backward_dkv_kernel``.

    Layout: Q,K,V,dO [ZH,N,64] float, L,Dr [ZH,N] float; dQ [ZH,N,64] float out.
    DISPATCH: (N/16) x ZH threadgroups, 256 threads. N % 16 == 0.
    """
    lim = "ib+1u" if causal else "N/Bc"
    mask = "((i0+r)<(j0+c)) ? 0.0f : " if causal else ""
    return (
        r"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void fa_bwd_dq(
  device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
  device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* L [[buffer(4)]],
  device const float* Dr [[buffer(5)]], device float* dQ [[buffer(6)]],
  constant uint& N [[buffer(7)]], constant float& scale [[buffer(8)]],
  uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const uint Bc=16u,Br=16u,Dh=64u,NT=256u; uint ib=tg.x,head=tg.y; uint hb=head*N*Dh,hl=head*N,i0=ib*Br;
  threadgroup float Qi[16*64],dOi[16*64],Kj[16*64],Vj[16*64],Ss[16*16],Ds[16*16];
  simdgroup_float8x8 dQ0(0.0f),dQ1(0.0f);
  for(uint x=lid;x<Br*Dh;x+=NT){ Qi[x]=Q[hb+(i0+x/Dh)*Dh+x%Dh]; dOi[x]=dO[hb+(i0+x/Dh)*Dh+x%Dh]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint jb=0u; jb<"""
        + lim
        + r"""; jb++){ uint j0=jb*Bc;
    for(uint x=lid;x<Bc*Dh;x+=NT){ Kj[x]=K[hb+(j0+x/Dh)*Dh+x%Dh]; Vj[x]=V[hb+(j0+x/Dh)*Dh+x%Dh]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sgitg<4u){ uint ro=sgitg/2u, co=sgitg%2u; simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<Dh/8u;dt++){ simdgroup_load(af,Qi+ro*8u*Dh+dt*8u,Dh); simdgroup_load(bf,Kj+co*8u*Dh+dt*8u,Dh,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,Ss+ro*8u*16u+co*8u,16u); }
    else { uint s2=sgitg-4u, ro=s2/2u, co=s2%2u; simdgroup_float8x8 acc(0.0f),af,bf;
      for(uint dt=0u;dt<Dh/8u;dt++){ simdgroup_load(af,dOi+ro*8u*Dh+dt*8u,Dh); simdgroup_load(bf,Vj+co*8u*Dh+dt*8u,Dh,0,true); simdgroup_multiply_accumulate(acc,af,bf,acc); }
      simdgroup_store(acc,Ds+ro*8u*16u+co*8u,16u); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint x=lid;x<Br*Bc;x+=NT){ uint r=x/Bc, c=x%Bc; float p="""
        + mask
        + r"""exp(scale*Ss[x]-L[hl+i0+r]); Ss[x]=p; Ds[x]=p*(Ds[x]-Dr[hl+i0+r]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sgitg<Dh/8u){ simdgroup_float8x8 dsf,kf;
      for(uint ct=0u;ct<Bc/8u;ct++){ simdgroup_load(kf,Kj+ct*8u*Dh+sgitg*8u,Dh);
        simdgroup_load(dsf,Ds+0u*8u*16u+ct*8u,16u); simdgroup_multiply_accumulate(dQ0,dsf,kf,dQ0);
        simdgroup_load(dsf,Ds+1u*8u*16u+ct*8u,16u); simdgroup_multiply_accumulate(dQ1,dsf,kf,dQ1); } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(sgitg<Dh/8u){ simdgroup_store(dQ0,Qi+0u*8u*Dh+sgitg*8u,Dh); simdgroup_store(dQ1,Qi+1u*8u*Dh+sgitg*8u,Dh); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint x=lid;x<Br*Dh;x+=NT) dQ[hb+(i0+x/Dh)*Dh+x%Dh]=scale*Qi[x];
}
"""
    )


def make_fa_logsumexp_kernel(causal=False):
    """Flash log-sum-exp — the softmax normalizer the FA-2 backward needs, WITHOUT the N*N
    materialization (tiled, online max/sum over kv-blocks; no V, no output).

    Used by the ``flash_attention`` backward so it does not recompute ``S`` in torch. One
    threadgroup per (query-block, head); L_i = m_i + log(l_i). Validated rel ~1e-7 vs
    torch.logsumexp, full and causal.

    Layout: Q,K [ZH,N,64] float; Lout [ZH,N] float. DISPATCH: (N/16) x ZH threadgroups. N % 16 == 0.
    """
    lim = "ib+1u" if causal else "N/Bc"
    mask = "if((i0+r)<(j0+c)) s=-INFINITY;" if causal else ""
    return (
        r"""#include <metal_stdlib>
using namespace metal;
kernel void fa_lse(
  device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
  device float* Lout [[buffer(2)]], constant uint& N [[buffer(3)]], constant float& scale [[buffer(4)]],
  uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  const uint Bc=16u,Br=16u,Dh=64u,NT=256u; uint ib=tg.x,head=tg.y; uint hb=head*N*Dh,hl=head*N,i0=ib*Br;
  threadgroup float Qi[16*64],Kj[16*64],Sb[16*16],mrow[16],lrow[16];
  for(uint x=lid;x<Br*Dh;x+=NT) Qi[x]=Q[hb+(i0+x/Dh)*Dh+x%Dh];
  if(lid<Br){ mrow[lid]=-INFINITY; lrow[lid]=0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint jb=0u; jb<"""
        + lim
        + r"""; jb++){ uint j0=jb*Bc;
    for(uint x=lid;x<Bc*Dh;x+=NT) Kj[x]=K[hb+(j0+x/Dh)*Dh+x%Dh];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint x=lid;x<Br*Bc;x+=NT){ uint r=x/Bc,c=x%Bc; float s=0.0f;
      for(uint d=0u;d<Dh;d++) s+=Qi[r*Dh+d]*Kj[c*Dh+d]; s*=scale; """
        + mask
        + r""" Sb[x]=s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lid<Br){ uint r=lid; float bm=-INFINITY;
      for(uint c=0u;c<Bc;c++) bm=max(bm,Sb[r*Bc+c]);
      float mnew=max(mrow[r],bm); float ls=0.0f;
      for(uint c=0u;c<Bc;c++) ls+=exp(Sb[r*Bc+c]-mnew);
      lrow[r]=lrow[r]*exp(mrow[r]-mnew)+ls; mrow[r]=mnew; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(lid<Br) Lout[hl+i0+lid]=mrow[lid]+log(lrow[lid]);
}
"""
    )
