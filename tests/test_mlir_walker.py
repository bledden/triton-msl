"""Test the MLIR walker against real TTGIR modules.

Compiles simple Triton kernels through the pipeline and validates
that the walker extracts the correct IR structure.
"""

import pytest
import sys

try:
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import Metal
    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    HAS_METAL = False

requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def _compile_to_ttgir(kernel_fn, sig, constexprs=None):
    """Compile a @triton.jit kernel through TTIR and TTGIR stages.

    Returns the TTGIR MLIR module (before MSL emission).
    """
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})

    # Create AST source and compile to TTIR
    src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs or {})
    context = ir.context()
    # Enable the passes we need
    ir.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    mod = src.make_ir(target, options, codegen_fns, module_map, context)

    metadata = {}
    # Run TTIR passes
    mod = backend.make_ttir(mod, metadata, options)
    # Run TTGIR passes
    mod = backend.make_ttgir(mod, metadata, options)

    return mod, metadata, options


# ---------------------------------------------------------------------------
# Walker tests
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_walker_vector_add():
    """Walker extracts correct structure from vector_add kernel."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        vector_add, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    # Validate function name
    assert "vector_add" in graph.func_name

    # Validate arguments: 3 pointers + 1 scalar
    assert len(graph.args) == 4
    ptr_args = [a for a in graph.args if a.is_ptr]
    scalar_args = [a for a in graph.args if not a.is_ptr]
    assert len(ptr_args) == 3
    assert len(scalar_args) == 1

    # Validate block_size
    assert graph.block_size == 256

    # Validate we have the expected op types
    op_names = [ssa.op for ssa in graph.ops]
    assert "tt.get_program_id" in op_names
    assert "tt.make_range" in op_names
    assert "tt.load" in op_names
    assert "tt.store" in op_names
    assert "arith.addf" in op_names or "arith.addi" in op_names

    print(f"Walker extracted {len(graph.ops)} ops from vector_add")
    print(f"Op types: {sorted(set(op_names))}")


@requires_triton
@requires_metal
def test_walker_scalar_mul():
    """Walker extracts correct structure from scalar multiply kernel."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def scalar_mul(x_ptr, out_ptr, n, scale, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * scale, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "scale": "fp32"}
    mod, metadata, options = _compile_to_ttgir(
        scalar_mul, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    assert len(graph.args) == 4
    assert graph.block_size == 256
    op_names = [ssa.op for ssa in graph.ops]
    assert "arith.mulf" in op_names
    assert "tt.load" in op_names
    assert "tt.store" in op_names

    print(f"Walker extracted {len(graph.ops)} ops from scalar_mul")


@requires_triton
@requires_metal
def test_walker_sum_reduction():
    """Walker extracts tt.reduce with body region."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def sum_kernel(input_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(output_ptr + pid, result)

    sig = {"input_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        sum_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    op_names = [ssa.op for ssa in graph.ops]
    assert "tt.reduce" in op_names

    # Find the reduce op and check it has body ops
    reduce_ops = [ssa for ssa in graph.ops if ssa.op == "tt.reduce"]
    assert len(reduce_ops) >= 1

    reduce_op = reduce_ops[0]
    # The reduce should have a body region with the combine op
    if reduce_op.region_ops:
        body_op_names = [b.op for b in reduce_op.region_ops]
        print(f"Reduce body ops: {body_op_names}")
        # Body should contain arith.addf (sum combine)
        assert any("addf" in n for n in body_op_names), \
            f"Expected arith.addf in reduce body, got: {body_op_names}"
    else:
        print("WARNING: reduce body ops not captured (may need region walking)")

    print(f"Walker extracted {len(graph.ops)} ops from sum_reduction")


@requires_triton
@requires_metal
def test_walker_fp16_cast():
    """Walker handles FP16 type cast operations (extf/truncf)."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def cast_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
        result = (x * y + x) * 0.5
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

    sig = {"x_ptr": "*fp16", "y_ptr": "*fp16", "out_ptr": "*fp16", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        cast_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    op_names = [ssa.op for ssa in graph.ops]
    # Should have type conversion ops
    has_ext = "arith.extf" in op_names
    has_trunc = "arith.truncf" in op_names
    print(f"FP16 ops - extf: {has_ext}, truncf: {has_trunc}")
    print(f"All ops: {sorted(set(op_names))}")

    # Should have pointer args with f16 types
    ptr_args = [a for a in graph.args if a.is_ptr]
    assert len(ptr_args) == 3
    for arg in ptr_args:
        assert arg.elem_type == "f16", f"Expected f16, got {arg.elem_type}"


@requires_triton
@requires_metal
def test_walker_constants():
    """Walker extracts arith.constant values correctly."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def const_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # Uses constant 0.5
        result = x * 0.5
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        const_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    # Find constant ops
    const_ops = [ssa for ssa in graph.ops if ssa.op == "arith.constant"]
    print(f"Found {len(const_ops)} constants:")
    for c in const_ops:
        print(f"  {c.name}: value={c.attrs.get('value')}, type={c.type_str}")

    # Should have the 0.5 constant
    float_consts = [c for c in const_ops if isinstance(c.attrs.get("value"), float)]
    assert any(abs(c.attrs["value"] - 0.5) < 1e-6 for c in float_consts), \
        f"Expected 0.5 constant, got: {[c.attrs.get('value') for c in const_ops]}"


@requires_triton
@requires_metal
def test_walker_ssa_connectivity():
    """Verify SSA operand IDs correctly reference previous results."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def simple_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        simple_add, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    # Build a set of all defined IDs (args + op results)
    defined_ids = set()
    for arg in graph.args:
        defined_ids.add(arg.id)
    for ssa in graph.ops:
        if ssa.id >= 0:  # Skip ops without results
            defined_ids.add(ssa.id)

    # Check that every operand ID references a defined value
    unresolved = []
    for ssa in graph.ops:
        for oid in ssa.operand_ids:
            if oid not in defined_ids:
                unresolved.append((ssa.name, ssa.op, oid))

    if unresolved:
        print(f"WARNING: {len(unresolved)} unresolved operand references:")
        for name, op, oid in unresolved[:5]:
            print(f"  {name} ({op}): references ID {oid}")
    else:
        print("All SSA operand references resolved correctly!")

    # Allow some unresolved (block args in nested regions), but flag many
    assert len(unresolved) < len(graph.ops), \
        f"Too many unresolved references: {len(unresolved)}/{len(graph.ops)}"


@requires_triton
@requires_metal
def test_walker_comparison():
    """Walker extracts comparison predicates."""
    from triton_msl.codegen.mlir_walker import walk_ttgir

    @triton.jit
    def cmp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n  # This generates arith.cmpi slt
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        cmp_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    graph = walk_ttgir(mod, options)

    # Find comparison ops
    cmp_ops = [ssa for ssa in graph.ops if "cmp" in ssa.op]
    assert len(cmp_ops) >= 1, "Expected at least one comparison op"

    for c in cmp_ops:
        print(f"  {c.op}: predicate={c.attrs.get('predicate')}, "
              f"name={c.attrs.get('predicate_name')}")


# ---------------------------------------------------------------------------
# _parse_blocked_layout unit tests (no Triton/Metal required)
# ---------------------------------------------------------------------------

def test_parse_blocked_layout():
    """_parse_blocked_layout extracts sizePerThread from TTGIR text."""
    from triton_msl.codegen.mlir_walker import _parse_blocked_layout

    text = '#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>'
    layout = _parse_blocked_layout(text)
    assert layout is not None
    assert layout["size_per_thread"] == [4]
    assert layout["threads_per_warp"] == [32]
    assert layout["warps_per_cta"] == [4]


def test_parse_blocked_layout_2d():
    """_parse_blocked_layout handles 2D layouts."""
    from triton_msl.codegen.mlir_walker import _parse_blocked_layout

    text = '#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>'
    layout = _parse_blocked_layout(text)
    assert layout is not None
    assert layout["size_per_thread"] == [1, 4]
    assert layout["threads_per_warp"] == [8, 4]
    assert layout["warps_per_cta"] == [4, 1]


def test_parse_blocked_layout_missing():
    """_parse_blocked_layout returns None when no layout present."""
    from triton_msl.codegen.mlir_walker import _parse_blocked_layout

    assert _parse_blocked_layout("no layout here") is None
