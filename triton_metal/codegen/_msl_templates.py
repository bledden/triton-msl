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

from triton_metal.codegen.msl_types import triton_type_to_msl
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

    identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[op]
    combine = {"sum": "+", "max": "max", "min": "min"}[op]

    # Each thread accumulates over strided elements
    kb._var("acc", identity, ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_elements; i += {block_size}) {{")
    kb.indent()
    kb._var("idx", f"pid * n_elements + i", ty="uint")
    if combine in ("+",):
        kb.raw_line("acc += input[idx];")
    else:
        kb.raw_line(f"acc = {combine}(acc, input[idx]);")
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

    identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[op]
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

    identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[op]
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
    from triton_metal.codegen.msl_builtins import is_fp8_type, fp8_to_float_func, fp8_device_functions

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
    kb.add_scalar_arg("M", dtype="u32")
    kb.add_scalar_arg("N", dtype="u32")
    kb.add_scalar_arg("K", dtype="u32")

    # For FP8, inject conversion device functions
    if fp8_input:
        for fn_src in fp8_device_functions(dtype):
            kb._device_functions.append(fn_src)

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
        kb.raw_line("C[gr * N + gc] = acc;")
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
        kb.raw_line("C[gr * N + gc] = acc[_e];")
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
        raise ValueError(f"Unknown activation: {activation}. "
                         f"Supported: {list(act_map.keys())}")

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
    compute_expr = "(float(input[row_start + i]) - mean_val) * inv_std * float(gamma[i]) + float(beta[i])" if needs_cast else "(input[row_start + i] - mean_val) * inv_std * gamma[i] + beta[i]"
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

    all_params = ",\n".join(input_params + size_params + [
        "    uint gid [[thread_position_in_grid]]"
    ])

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
        ["    device const float* input [[buffer(0)]]"] +
        out_params +
        [f"    constant uint& chunk_size [[buffer({n_outputs + 1})]]",
         "    uint gid [[thread_position_in_grid]]"]
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


def make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4):
    """Fast matmul: direct-device simdgroup_load + register blocking (WS1 C.2).

    Reaches MLX parity (~13.8 TFLOP/s fp16 @ 2048, vs ~7.7 for the staged
    template) by (a) loading simdgroup fragments DIRECTLY from device A/B (no
    threadgroup staging, no barriers — the GPU cache provides reuse) and
    (b) register blocking: each simdgroup computes an (8*rr) x (8*rc) output
    block (rr*rc accumulators), so each loaded a-/b-fragment feeds rc / rr MMAs.

    Genuine fp16: half INPUT fragments (simdgroup_half8x8) with a float
    ACCUMULATOR; fp16 = input precision, C output is float.

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
    elif dtype in ("fp32", "f32"):
        in_t, in_frag, pad = "float", "simdgroup_float8x8", "0.0f"
    else:
        raise ValueError(f"fast matmul supports fp16/fp32, got {dtype}")

    accs = "\n    ".join(
        "simdgroup_float8x8 " + ", ".join(f"c{r}_{c}(0)" for c in range(rc))
        + ";" for r in range(rr))
    bdecl = ", ".join(f"b{c}" for c in range(rc))
    loads_b = "\n        ".join(
        f"simdgroup_load(b{c}, B + k * N + col0 + {c * 8}u, N);"
        for c in range(rc))
    inner = []
    for r in range(rr):
        inner.append(f"simdgroup_load(a_frag, A + (row_base + {r * 8}u) * K + k, K);")
        for c in range(rc):
            inner.append(
                f"simdgroup_multiply_accumulate(c{r}_{c}, a_frag, b{c}, c{r}_{c});")
    inner = "\n        ".join(inner)
    stores = "\n    ".join(
        f"simdgroup_store(c{r}_{c}, C + (row_base + {r * 8}u) * N + col0 + {c * 8}u, N);"
        for r in range(rr) for c in range(rc))

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul_fast(
    device const {in_t}* A [[buffer(0)]],
    device const {in_t}* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {{
    uint ntc = (N + {32 * rc - 1}u) / {32 * rc}u;
    uint row_base = (pid / ntc) * {8 * rr}u;
    uint col0 = (pid % ntc) * {32 * rc}u + sgitg * {8 * rc}u;
    if (col0 >= N) return;   // partial column tile: this simdgroup is OOB (uniform)
    {accs}
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
        buffer_params.append(
            f"    device const {msl_ty}* input{i} [[buffer({i})]]"
        )
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
        buffer_params.append(
            f"    device const {msl_ty}* input{i} [[buffer({i})]]"
        )
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


def make_atomic_max_kernel(block_size=256):
    """Generate a kernel that performs atomic max to a global accumulator.

    Each thread atomically computes max of its input with the output location.
    Uses atomic_fetch_max on int representation (reinterpret cast).

    Args:
        block_size: Threads per threadgroup.
    """
    # Metal doesn't have atomic_fetch_max for float, so use int reinterpret trick
    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void atomic_max_kernel(
    device const float* input [[buffer(0)]],
    device atomic_int* output [[buffer(1)]],
    constant uint& n_elements [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= n_elements) return;
    float val = input[gid];
    int int_val = as_type<int>(val);
    // For positive floats, int comparison preserves order.
    // Handle negative: flip all bits if negative, else flip sign bit only.
    int_val = (int_val >= 0) ? int_val : (int_val ^ 0x7FFFFFFF);
    atomic_fetch_max_explicit(&output[0], int_val, memory_order_relaxed);
}}
"""
    kb = KernelBuilder("atomic_max_kernel", block_size=block_size)
    kb.set_prebuilt_msl(msl)
    return kb.build()


def make_conv2d_kernel(in_channels=3, out_channels=64, kernel_h=3, kernel_w=3,
                       stride_h=1, stride_w=1, pad_h=1, pad_w=1, block_size=256):
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


def make_max_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2,
                           pad_h=0, pad_w=0, block_size=256):
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


def make_avg_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2,
                           pad_h=0, pad_w=0, block_size=256):
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
from triton_metal.codegen.msl_emitter import (  # noqa: E402
    KernelBuilder,
    _msl_compute_type,
    _msl_zero,
    _sanitize_msl_name,
)
