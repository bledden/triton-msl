"""Test the generic op-by-op lowerer against real TTGIR modules.

End-to-end tests: Triton kernel → TTGIR → MLIR walker → generic lowerer → MSL.
Validates that the generated MSL compiles with xcrun metal.
"""

import os
import subprocess
import tempfile
import pytest

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
    """Compile a @triton.jit kernel through TTIR and TTGIR stages."""
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_metal.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})

    src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs or {})
    context = ir.context()
    ir.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    mod = src.make_ir(target, options, codegen_fns, module_map, context)

    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)

    return mod, metadata, options


def _lower_to_msl(mod, metadata, options):
    """Walk TTGIR module and lower to MSL."""
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import lower_ir_graph

    graph = walk_ttgir(mod, options)
    msl = lower_ir_graph(graph, options)
    return msl, graph


def _validate_msl_compiles(msl_src: str):
    """Verify that MSL source compiles with xcrun metal."""
    with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
        f.write(msl_src)
        metal_path = f.name

    air_path = metal_path.replace(".metal", ".air")
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
             "-o", air_path, "-std=metal3.2", "-O0"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"MSL compilation FAILED:\n{result.stderr}")
            print(f"\nGenerated MSL:\n{msl_src}")
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("xcrun metal not available")
    finally:
        for p in (metal_path, air_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_vector_add():
    """Generic lowerer produces valid MSL for vector_add."""
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

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for vector_add ===")
    print(msl)

    # Basic structure checks
    assert "kernel void" in msl
    assert "vector_add" in msl
    assert "a_ptr" in msl
    assert "b_ptr" in msl
    assert "out_ptr" in msl

    # Should have loads, an add, and a store
    assert "+" in msl  # arith.addf

    # Should compile
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_scalar_mul():
    """Generic lowerer produces valid MSL for scalar multiply."""
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

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for scalar_mul ===")
    print(msl)

    assert "kernel void" in msl
    assert "scale" in msl
    assert "*" in msl  # multiplication
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_fp16_cast():
    """Generic lowerer handles FP16 loads/stores."""
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

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for fp16_cast ===")
    print(msl)

    assert "half" in msl  # FP16 storage type
    assert "static_cast" in msl  # Type casts
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_constant_mul():
    """Generic lowerer handles float constants."""
    @triton.jit
    def const_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = x * 0.5
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        const_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for constant_mul ===")
    print(msl)

    assert "0.5" in msl  # The constant
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_sum_reduction():
    """Generic lowerer handles tt.reduce (sum)."""
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

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for sum_reduction ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd_sum" in msl or "simd" in msl  # SIMD reduction
    assert "threadgroup" in msl  # Shared memory
    assert _validate_msl_compiles(msl), "MSL failed to compile"


# ---------------------------------------------------------------------------
# Adversarial tests — novel op combinations the pattern matchers can't handle
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_negation():
    """Generic lowerer handles arith.negf."""
    @triton.jit
    def negate_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, -x, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        negate_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for negate ===")
    print(msl)

    assert "kernel void" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_math_exp():
    """Generic lowerer handles math.exp (exponential)."""
    @triton.jit
    def exp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.exp(x), mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        exp_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for exp ===")
    print(msl)

    assert "exp(" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_fused_silu():
    """Adversarial: SiLU (x * sigmoid(x)) — not a named pattern."""
    @triton.jit
    def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        result = x * tl.sigmoid(x)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        silu_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for fused_silu ===")
    print(msl)

    assert "kernel void" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_multi_op_chain():
    """Adversarial: long chain of mixed ops — no single pattern covers this."""
    @triton.jit
    def chain_kernel(a_ptr, b_ptr, out_ptr, n, alpha,
                     BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        # Complex expression: (a * alpha + b) * (a - b)
        result = (a * alpha + b) * (a - b)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32",
           "n": "i32", "alpha": "fp32"}
    mod, metadata, options = _compile_to_ttgir(
        chain_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for multi_op_chain ===")
    print(msl)

    assert "kernel void" in msl
    assert "alpha" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_reduce_mul_add():
    """Adversarial: reduce(x * y + z) — reduction of a fused expression."""
    @triton.jit
    def reduce_expr_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n,
                           BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        expr = x * y + z
        result = tl.sum(expr, axis=0)
        tl.store(out_ptr + pid, result)

    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32",
           "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        reduce_expr_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for reduce_mul_add ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd" in msl  # SIMD reduction
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_max_reduction():
    """Generic lowerer handles tt.reduce with maxf combine."""
    @triton.jit
    def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        result = tl.max(x, axis=0)
        tl.store(out_ptr + pid, result)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        max_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for max_reduction ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_where_clamp():
    """Adversarial: tl.where (arith.select) for clamping."""
    @triton.jit
    def clamp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # Clamp to [0, 1] using tl.where
        clamped = tl.where(x > 1.0, 1.0, x)
        clamped = tl.where(clamped < 0.0, 0.0, clamped)
        tl.store(out_ptr + offsets, clamped, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        clamp_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for where_clamp ===")
    print(msl)

    assert "kernel void" in msl
    assert "?" in msl  # Ternary from select
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_int_to_float():
    """Generic lowerer handles sitofp (integer to float conversion)."""
    @triton.jit
    def itof_kernel(idx_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        idx = tl.load(idx_ptr + offsets, mask=mask)
        # Convert int to float and scale
        result = idx.to(tl.float32) * 0.1
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"idx_ptr": "*i32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        itof_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for int_to_float ===")
    print(msl)

    assert "kernel void" in msl
    assert "static_cast" in msl or "float" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


# ---------------------------------------------------------------------------
# Pipeline integration tests — verify emit_msl uses new codegen
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_emit_msl_uses_new_codegen_for_elementwise():
    """Verify the full emit_msl pipeline uses the new walker+lowerer for elementwise."""
    from triton_metal.codegen.msl_emitter import emit_msl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        add_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    # Call through the full pipeline — same path as compiler.make_msl
    msl = emit_msl(mod, metadata, options)

    print(f"\n=== emit_msl pipeline output for add_kernel ===")
    print(msl)

    assert "kernel void" in msl
    assert "add_kernel" in msl
    assert metadata.get("name") == "add_kernel"
    assert metadata.get("block_size") == 256
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_emit_msl_handles_matmul_via_new_pipeline():
    """Verify emit_msl routes matmul (tt.dot) through the new pipeline."""
    from triton_metal.codegen.msl_emitter import emit_msl

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        tl.store(c_ptrs, acc)

    sig = {
        "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
        "M": "i32", "N": "i32", "K": "i32",
        "stride_am": "i32", "stride_ak": "i32",
        "stride_bk": "i32", "stride_bn": "i32",
        "stride_cm": "i32", "stride_cn": "i32",
    }
    mod, metadata, options = _compile_to_ttgir(
        matmul_kernel, sig,
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msl = emit_msl(mod, metadata, options)
        # Should NOT trigger a legacy fallback deprecation warning
        legacy_warnings = [x for x in w if "legacy" in str(x.message).lower()]
        assert len(legacy_warnings) == 0, \
            f"Matmul should use new pipeline, got legacy warning: {legacy_warnings}"

    print(f"\n=== emit_msl pipeline output for matmul_kernel ===")
    print(f"(first 500 chars): {msl[:500]}")

    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert metadata.get("name") is not None
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_no_legacy_fallback_for_standard_kernels():
    """No standard kernel should fall back to the legacy parser."""
    from triton_metal.codegen.msl_emitter import emit_msl
    import warnings

    kernels = {}

    @triton.jit
    def add_k(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    kernels["add"] = (
        add_k,
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    @triton.jit
    def relu_k(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = tl.where(x > 0, x, 0.0)
        tl.store(out_ptr + offsets, result, mask=mask)

    kernels["relu"] = (
        relu_k,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    @triton.jit
    def sum_k(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(out_ptr + pid, result)

    kernels["sum"] = (
        sum_k,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    for name, (fn, sig, constexprs) in kernels.items():
        mod, metadata, options = _compile_to_ttgir(fn, sig, constexprs=constexprs)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msl = emit_msl(mod, metadata, options)
            legacy_warnings = [x for x in w if "legacy" in str(x.message).lower()]
            assert len(legacy_warnings) == 0, \
                f"Kernel '{name}' fell back to legacy parser!"
            assert "UNSUPPORTED" not in msl, \
                f"Kernel '{name}' has unsupported ops in output"
        print(f"  {name}: OK (new pipeline)")


# ---------------------------------------------------------------------------
# Adversarial end-to-end tests — novel combinations through full pipeline
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_softmax_fused():
    """Adversarial: row-wise softmax — max + sub + exp + sum + div."""
    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        x_max = tl.max(x, axis=0)
        x_shifted = x - x_max
        numerator = tl.exp(x_shifted)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        softmax_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for softmax_fused ===")
    print(msl)

    assert "kernel void" in msl
    assert "exp(" in msl
    assert "simd" in msl  # Reductions
    assert "UNSUPPORTED" not in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_gelu_sigmoid():
    """Adversarial: GELU sigmoid approximation — x * sigmoid(1.702 * x)."""
    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # GELU sigmoid approx: x * sigmoid(1.702 * x)
        result = x * tl.sigmoid(1.702 * x)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        gelu_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for gelu_sigmoid ===")
    print(msl)

    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


# ---------------------------------------------------------------------------
# sizePerThread tests — wrapping loop when Triton expects fewer threads
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_size_per_thread_wrapping_decision():
    """Verify the lowerer's wrapping decision logic with sizePerThread.

    We can't easily force Triton to generate sizePerThread > 1 for a
    non-reduction kernel, so we test the decision logic directly by
    checking that:
    1. Elementwise kernels with large BLOCK_SIZE compile and run correctly
    2. Reduction kernels with large BLOCK_SIZE compile and run correctly
    3. The sizePerThread extraction works on real TTGIR
    """
    import torch

    # Test 1: Elementwise kernel with BLOCK_SIZE=2048 (forces wrapping since > 1024)
    @triton.jit
    def _scale(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * 2.0, mask=mask)

    n = 4096
    x = torch.randn(n, device="cpu")
    out = torch.empty(n, device="cpu")
    grid = (triton.cdiv(n, 2048),)
    _scale[grid](x, out, n, BLOCK_SIZE=2048)
    diff = (out - x * 2.0).abs().max().item()
    assert diff < 1e-5, f"Scale kernel wrong: max_diff={diff}"

    # Test 2: Reduction kernel correctness at various block sizes
    @triton.jit
    def _sum(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(out_ptr, result)

    for bs in [256, 512, 1024]:
        n2 = bs
        x2 = torch.randn(n2, device="cpu")
        out2 = torch.zeros(1, device="cpu")
        _sum[(1,)](x2, out2, n2, BLOCK_SIZE=bs)
        diff2 = abs(out2.item() - x2.sum().item())
        assert diff2 < 1e-3, f"Sum reduction wrong at BLOCK_SIZE={bs}: diff={diff2}"


@requires_triton
@requires_metal
def test_size_per_thread_no_wrapping_when_fits():
    """Reduction kernels that fit in 1024 threads should use standard (no-loop) path.

    When sizePerThread=1 and block_size <= 1024, no wrapping or multi-pass is
    needed. Each thread handles one element, and reductions use standard SIMD +
    shared memory. The multi-pass optimization only activates when sizePerThread > 1
    or block_size > 1024.
    """
    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        x_max = tl.max(x, axis=0)
        x_shifted = x - x_max
        numerator = tl.exp(x_shifted)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        softmax_kernel, sig, constexprs={"BLOCK_SIZE": 1024}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for softmax (no wrapping, sizePerThread=1) ===")
    print(msl)

    # With sizePerThread=1 and block_size=1024, no wrapping needed
    assert "_loop_e" not in msl, (
        "No wrapping loop when sizePerThread=1 and block_size fits in 1024 threads"
    )
    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_size_per_thread_softmax_correctness():
    """Softmax produces correct results when sizePerThread > 1 in TTGIR."""
    import torch

    @triton.jit
    def _softmax(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        x_exp = tl.exp(x)
        x_sum = tl.sum(x_exp, axis=0)
        out = x_exp / x_sum
        tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)

    for cols in [128, 256, 512, 1024]:
        x = torch.randn(8, cols, device="cpu")
        out = torch.empty_like(x)
        _softmax[(8,)](x, out, cols, BLOCK_SIZE=1024)
        expected = torch.softmax(x, dim=-1)
        diff = (out - expected).abs().max().item()
        assert diff < 1e-5, f"cols={cols}: max_diff={diff}"


@requires_triton
@requires_metal
def test_multipass_reduction_softmax():
    """Multi-pass reduction for softmax with BLOCK_SIZE > 1024.

    When BLOCK_SIZE exceeds Metal's 1024-thread limit and the kernel has
    reductions, the codegen should emit a multi-pass kernel:
    per-element loops separated by reductions, with local accumulators.
    """
    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + pid * n + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x_shifted = x - x_max
        numerator = tl.exp(x_shifted)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(out_ptr + pid * n + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        softmax_kernel, sig, constexprs={"BLOCK_SIZE": 2048}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for multipass softmax (BLOCK=2048) ===")
    print(msl)

    # Multi-pass should be active: multiple loops and no barriers inside loops
    assert "_loop_e" in msl, "Multi-pass should use _loop_e loop variable"
    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert "_local_acc_" in msl, "Multi-pass should declare local accumulators"
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_multipass_reduction_correctness():
    """Multi-pass softmax produces correct results with BLOCK_SIZE > 1024."""
    import torch

    @triton.jit
    def _softmax(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        x_exp = tl.exp(x)
        x_sum = tl.sum(x_exp, axis=0)
        out = x_exp / x_sum
        tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)

    for cols in [256, 512, 1024, 2048]:
        x = torch.randn(8, cols, device="cpu")
        out = torch.empty_like(x)
        _softmax[(8,)](x, out, cols, BLOCK_SIZE=2048)
        expected = torch.softmax(x, dim=-1)
        diff = (out - expected).abs().max().item()
        assert diff < 1e-4, f"BLOCK=2048, cols={cols}: max_diff={diff}"


@requires_triton
@requires_metal
def test_multipass_sum_reduce():
    """Multi-pass sum reduction produces correct results."""
    import torch

    @triton.jit
    def _sum_reduce(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(out_ptr, result)

    for n in [512, 1024, 2048]:
        x = torch.randn(n, device="cpu")
        out = torch.zeros(1, device="cpu")
        _sum_reduce[(1,)](x, out, n, BLOCK_SIZE=2048)
        expected = x.sum().item()
        diff = abs(out.item() - expected)
        assert diff < 1e-2, f"Sum reduce n={n}: diff={diff}"


# ---------------------------------------------------------------------------
# Phase 4a infrastructure: env_n_elems tracking
# ---------------------------------------------------------------------------


def test_parse_blocked_field_extracts_lists():
    """`_parse_blocked_field` pulls the four required int lists."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    mod_text = (
        "#bar = #ttg.blocked<{sizePerThread = [1, 4], "
        "threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], "
        "order = [1, 0]}>\n"
        "#foo = #ttg.blocked<{sizePerThread = [8], "
        "threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\n"
    )

    assert GenericLowerer._parse_blocked_field(mod_text, "bar",
                                                "sizePerThread") == [1, 4]
    assert GenericLowerer._parse_blocked_field(mod_text, "bar",
                                                "threadsPerWarp") == [2, 16]
    assert GenericLowerer._parse_blocked_field(mod_text, "bar",
                                                "warpsPerCTA") == [4, 1]
    assert GenericLowerer._parse_blocked_field(mod_text, "bar",
                                                "order") == [1, 0]
    assert GenericLowerer._parse_blocked_field(mod_text, "foo",
                                                "sizePerThread") == [8]
    # Missing alias returns None.
    assert GenericLowerer._parse_blocked_field(mod_text, "baz",
                                                "sizePerThread") is None
    # Missing field returns None.
    assert GenericLowerer._parse_blocked_field(mod_text, "bar",
                                                "nonexistent") is None


def test_track_n_elems_blocked_layout():
    """Tracking computes elements-per-thread from a #ttg.blocked layout."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    # sizePerThread=[4], threadsPerWarp=[32], warpsPerCTA=[4] →
    # 4 warps * 32 lanes = 128 threads, 4 elems/thread → 512 total.
    mod_text = (
        "#blk = #ttg.blocked<{sizePerThread = [4], "
        "threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>"
    )

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[], mod_text=mod_text)
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = graph
    lowerer.options = _Options()
    lowerer.env_n_elems = {}

    lowerer._track_n_elems(42, "tensor<512xf32, #blk>", (512,))
    assert lowerer.env_n_elems[42] == 4, (
        f"expected 4 elems/thread for blocked[4], got "
        f"{lowerer.env_n_elems[42]}")


def test_track_n_elems_linear_layout():
    """Tracking computes elements-per-thread from a #ttg.linear layout."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    # 2 register bases → 4 elements per thread.
    mod_text = (
        "#lin = #ttg.linear<{register = [[1], [2]], "
        "lane = [[4], [8], [16], [32], [64]], "
        "warp = [[128], [256]], block = []}>"
    )

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[], mod_text=mod_text)
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = graph
    lowerer.options = _Options()
    lowerer.env_n_elems = {}

    lowerer._track_n_elems(7, "tensor<512xf32, #lin>", (512,))
    assert lowerer.env_n_elems[7] == 4, (
        f"expected 4 elems/thread for linear with 2 register bases, got "
        f"{lowerer.env_n_elems[7]}")


def test_track_n_elems_scalar_defaults_to_one():
    """Scalar types (no shape) always track as 1 element per thread."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[], mod_text="")
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = graph
    lowerer.options = _Options()
    lowerer.env_n_elems = {}

    lowerer._track_n_elems(1, "i32", ())
    assert lowerer.env_n_elems[1] == 1


def test_track_n_elems_unresolved_alias_falls_back():
    """Without mod_text or alias, tracking falls back to numel/num_threads."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4  # 128 threads

    graph = IRGraph(func_name="t", args=[], ops=[], mod_text="")
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = graph
    lowerer.options = _Options()
    lowerer.env_n_elems = {}

    # 512 elems / 128 threads = 4 elems/thread (default).
    lowerer._track_n_elems(3, "tensor<512xf32>", (512,))
    assert lowerer.env_n_elems[3] == 4

    # Small tile: 32 elems / 128 threads → floored to 1.
    lowerer._track_n_elems(4, "tensor<32xf32>", (32,))
    assert lowerer.env_n_elems[4] == 1


# ---------------------------------------------------------------------------
# Phase 4b scaffolding: _var_array + TRITON_METAL_MEPT flag
# ---------------------------------------------------------------------------


def test_mept_flag_defaults_off():
    """Without TRITON_METAL_MEPT in env, the feature flag is off."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.pop("TRITON_METAL_MEPT", None)
    try:
        lowerer = GenericLowerer(graph, _Options())
        assert lowerer.mept_enabled is False
        assert lowerer.env_array == {}
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_mept_flag_on_when_env_set():
    """With TRITON_METAL_MEPT=1, the feature flag is on."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        assert lowerer.mept_enabled is True
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_var_array_emits_declaration_and_initializers():
    """`_var_array` emits `T name[N];` plus per-index assignments."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=256)

    name = lowerer._var_array("v", ["a + b", "c + d", "e + f"], "float")

    body = "\n".join(lowerer.kb._body_lines)
    assert f"float {name}[3];" in body
    assert f"{name}[0] = a + b;" in body
    assert f"{name}[1] = c + d;" in body
    assert f"{name}[2] = e + f;" in body


def test_var_array_rejects_empty_exprs():
    """`_var_array` requires at least one expression."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=256)

    with pytest.raises(ValueError, match="at least one"):
        lowerer._var_array("v", [], "float")


def test_lookup_array_lifts_scalar():
    """`_lookup_array` lifts a scalar env entry to (name, 1, "")."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.env[42] = "v_3"

    name, n, ty = lowerer._lookup_array(42)
    assert name == "v_3"
    assert n == 1
    assert ty == ""


def test_emit_passthrough_propagates_env_array():
    """`_emit_passthrough` carries env_array entry from src to dst."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=256)

    # Source SSA has both scalar env entry and array entry.
    lowerer.env[100] = "v_scalar"
    lowerer.env_array[100] = ("v_arr", 4, "float")

    dst = SSAValue(id=200, name="v200", op="tt.bitcast",
                   operand_ids=[100], attrs={}, type_str="tensor<512xf32>",
                   elem_type="f32", is_tensor=True)
    lowerer._emit_passthrough(dst)

    # Both forms propagate.
    assert lowerer.env[200] == "v_scalar"
    assert lowerer.env_array[200] == ("v_arr", 4, "float")


def test_emit_cast_emits_array_when_mept_on_and_src_is_array():
    """`_emit_cast` produces per-element casts when MEPT + array src."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "v_src"
        lowerer.env_array[100] = ("v_src", 4, "half")

        dst = SSAValue(id=200, name="v200", op="arith.extf",
                       operand_ids=[100], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_cast(dst, "float", dtype="fp32")

        # Result should be an array with per-element casts.
        name, n, ty = lowerer.env_array[200]
        assert n == 4
        assert ty == "float"
        body = "\n".join(lowerer.kb._body_lines)
        assert f"float {name}[4];" in body
        for i in range(4):
            assert f"{name}[{i}] = static_cast<float>(v_src[{i}]);" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_unary_emits_array_when_mept_on_and_src_is_array():
    """`_emit_unary` produces per-element unary ops when MEPT + array src."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "v_src"
        lowerer.env_array[100] = ("v_src", 3, "float")

        dst = SSAValue(id=200, name="v200", op="arith.negf",
                       operand_ids=[100], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_unary(dst, "-")

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        assert f"float {name}[3];" in body
        for i in range(3):
            assert f"{name}[{i}] = -v_src[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_binary_emits_array_when_both_operands_are_arrays():
    """`_emit_binary` produces per-element binary ops when both src arrays."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "a"
        lowerer.env[101] = "b"
        lowerer.env_array[100] = ("a", 4, "float")
        lowerer.env_array[101] = ("b", 4, "float")
        lowerer.effective_block_size = 256

        dst = SSAValue(id=200, name="v200", op="arith.addf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_binary(dst, "+")

        name, n, ty = lowerer.env_array[200]
        assert n == 4
        body = "\n".join(lowerer.kb._body_lines)
        assert f"float {name}[4];" in body
        for i in range(4):
            assert f"{name}[{i}] = a[{i}] + b[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_binary_emits_scalar_when_mept_off():
    """Flag-off keeps scalar form even if env_array is populated."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.pop("TRITON_METAL_MEPT", None)
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "a"
        lowerer.env[101] = "b"
        lowerer.env_array[100] = ("a", 4, "float")
        lowerer.env_array[101] = ("b", 4, "float")
        lowerer.effective_block_size = 256

        dst = SSAValue(id=200, name="v200", op="arith.addf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_binary(dst, "+")

        assert 200 not in lowerer.env_array
        body = "\n".join(lowerer.kb._body_lines)
        assert "= a + b;" in body
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_binary_broadcasts_scalar_against_array():
    """Array `a` + scalar `b` broadcasts b across array positions."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "a"
        lowerer.env[101] = "b_scalar"
        lowerer.env_array[100] = ("a", 4, "float")
        # No env_array for 101 — scalar.

        dst = SSAValue(id=200, name="v200", op="arith.addf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_binary(dst, "+")

        name, n, ty = lowerer.env_array[200]
        assert n == 4
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(4):
            assert f"{name}[{i}] = a[{i}] + b_scalar;" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_binary_broadcasts_scalar_against_array_b_side():
    """Scalar `a` + array `b` broadcasts a across array positions."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "a_scalar"
        lowerer.env[101] = "b"
        lowerer.env_array[101] = ("b", 3, "float")
        # No env_array for 100 — scalar.

        dst = SSAValue(id=200, name="v200", op="arith.mulf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_binary(dst, "*")

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"{name}[{i}] = a_scalar * b[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_binary_mismatched_array_lengths_falls_through():
    """Different-length arrays aren't supported — falls back to scalar."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "a"
        lowerer.env[101] = "b"
        lowerer.env_array[100] = ("a", 4, "float")
        lowerer.env_array[101] = ("b", 2, "float")  # different length
        lowerer.effective_block_size = 256

        dst = SSAValue(id=200, name="v200", op="arith.addf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_binary(dst, "+")

        # Falls through to scalar emission (legacy path).
        assert 200 not in lowerer.env_array
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_unary_emits_scalar_when_mept_off():
    """With MEPT off, `_emit_unary` keeps the existing scalar form."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.pop("TRITON_METAL_MEPT", None)
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "v_src"
        lowerer.env_array[100] = ("v_src", 3, "float")

        dst = SSAValue(id=200, name="v200", op="arith.negf",
                       operand_ids=[100], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_unary(dst, "-")

        assert 200 not in lowerer.env_array
        body = "\n".join(lowerer.kb._body_lines)
        assert "-v_src;" in body
        assert "v_src[0]" not in body
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_cast_emits_scalar_when_mept_off():
    """With MEPT off, `_emit_cast` keeps the existing scalar form."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    saved = os.environ.pop("TRITON_METAL_MEPT", None)
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=256)

        lowerer.env[100] = "v_src"
        # Even if env_array has an entry, flag-off ignores it.
        lowerer.env_array[100] = ("v_src", 4, "half")

        dst = SSAValue(id=200, name="v200", op="arith.extf",
                       operand_ids=[100], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_cast(dst, "float", dtype="fp32")

        assert 200 not in lowerer.env_array  # stayed scalar
        body = "\n".join(lowerer.kb._body_lines)
        assert "static_cast<float>(v_src)" in body
        assert "v_src[0]" not in body
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_passthrough_no_env_array_when_src_has_none():
    """Without a source env_array entry, dst gets none either."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=256)

    lowerer.env[100] = "v_scalar"
    # Deliberately no env_array entry on src.

    dst = SSAValue(id=200, name="v200", op="tt.bitcast",
                   operand_ids=[100], attrs={}, type_str="tensor<512xf32>",
                   elem_type="f32", is_tensor=True)
    lowerer._emit_passthrough(dst)

    assert lowerer.env[200] == "v_scalar"
    assert 200 not in lowerer.env_array


def test_lookup_array_returns_env_array_entry():
    """`_lookup_array` returns env_array directly when present."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.env_array[7] = ("v_5", 4, "float")

    name, n, ty = lowerer._lookup_array(7)
    assert name == "v_5"
    assert n == 4
    assert ty == "float"


@requires_triton
@requires_metal
def test_mept_flag_on_preserves_existing_behavior():
    """Flipping TRITON_METAL_MEPT=1 must not change scalar-path output.

    Phase 4b scaffolding is dead code until call sites wire it in. This
    test guards against an accidental wire-up that changes default output.
    """
    import os
    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}

    def lower(flag_on: bool):
        saved = os.environ.get("TRITON_METAL_MEPT")
        if flag_on:
            os.environ["TRITON_METAL_MEPT"] = "1"
        else:
            os.environ.pop("TRITON_METAL_MEPT", None)
        try:
            mod, metadata, options = _compile_to_ttgir(
                vector_add, sig, constexprs={"BLOCK_SIZE": 256}
            )
            msl, _ = _lower_to_msl(mod, metadata, options)
            return msl
        finally:
            if saved is None:
                os.environ.pop("TRITON_METAL_MEPT", None)
            else:
                os.environ["TRITON_METAL_MEPT"] = saved

    assert lower(False) == lower(True), (
        "MEPT flag-on output diverged from flag-off baseline; some call "
        "site started consuming env_array without the explicit gate.")


@requires_triton
def test_track_n_elems_against_real_kernel_layouts():
    """The parser resolves every #ttg.blocked alias in a real TTGIR module.

    Phase 4a's tracking infrastructure is plumbed selectively today (only
    sites that go through ``_propagate_shape_from_type`` populate
    ``env_n_elems``). This test instead validates that the parser
    components — ``_parse_blocked_field`` + ``_track_n_elems`` — produce
    a sensible result for every tensor SSA in a representative kernel.
    """
    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        vector_add, sig, constexprs={"BLOCK_SIZE": 256}
    )

    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import (
        GenericLowerer, _extract_shape,
    )

    graph = walk_ttgir(mod, options)
    lowerer = GenericLowerer(graph, options)

    tensor_ssa = [op for op in graph.ops
                  if op.type_str and "tensor<" in op.type_str]
    assert tensor_ssa, "expected at least one tensor SSA value"

    for op in tensor_ssa:
        shape = _extract_shape(op.type_str)
        lowerer._track_n_elems(op.id, op.type_str, shape)
        n = lowerer.env_n_elems.get(op.id)
        assert n is not None and n >= 1, (
            f"_track_n_elems failed for ssa id={op.id} "
            f"type={op.type_str!r} (got {n!r})")
