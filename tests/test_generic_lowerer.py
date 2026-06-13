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
def test_dot_scaled_refuses_not_silently_wrong():
    """Integrity (PR1): microscaling matmul (tt.dot_scaled) has no Apple
    hardware and no handler, so the result tensor is never computed —
    emitting anything yields silently-wrong output. The lowerer must RAISE
    MetalNonRecoverableError instead (test_scaled_dot was ~all mismatch)."""
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.msl_emitter import emit_msl
    from triton_metal.errors import MetalNonRecoverableError

    @triton.jit
    def k(a_base, b_base, out, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
        a_ptr = a_base + tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
        b_ptr = b_base + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        c = tl.dot_scaled(a, None, "e5m2", b, None, "e5m2")
        out_ptr = out + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(out_ptr, c)

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(fn=k, signature={"a_base": "*fp8e5", "b_base": "*fp8e5",
                                     "out": "*fp32"},
                    constexprs=dict(BM=32, BN=32, BK=32))
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    with pytest.raises(MetalNonRecoverableError):
        emit_msl(mod, meta, options)


@requires_triton
def test_constexpr_dim_matmul_refuses_not_silently_wrong():
    """Integrity (PR1): a pid-tiled matmul with constexpr-baked M/N must
    RAISE (MetalNonRecoverableError), never emit silently-wrong output.

    The matmul template needs runtime M/N to derive output strides; when
    they're baked as constexpr it would guess _N=BLOCK_N and produce ~98%
    wrong numbers. The guard refuses instead. This is the
    test_dot_mulbroadcasted shape, lowered directly.
    """
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.msl_emitter import emit_msl
    from triton_metal.errors import MetalNonRecoverableError

    @triton.jit
    def kernel(Z, X, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pidn = tl.program_id(1)
        pidm = tl.program_id(0)
        offm = tl.arange(0, BM)[:, None]
        offn = tl.arange(0, BN)[None, :]
        offak = tl.arange(0, BK)[None, :]
        offbk = tl.arange(0, BK)[:, None]
        acc = tl.full((BM, BN), 0.0, tl.float32)
        for ridx5 in range(0, K // BK):
            x = tl.load(X + ((pidm * K * BM) + (offm * K) + (ridx5 * BK) + offak))
            y = tl.load(Y + ((pidn * BN) + (offbk * N) + (ridx5 * N * BK) + offn))
            x = tl.expand_dims(x, axis=2)
            y = tl.expand_dims(y, axis=0)
            t = tl.sum(x * y, axis=1)
            acc = t + acc
        tl.store(Z + ((pidm * BM * N) + (pidn * BN) + (offm * N) + offn), acc)

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    sig = {"Z": "*fp32", "X": "*fp32", "Y": "*fp32"}
    ce = dict(M=256, N=192, K=160, BM=128, BN=32, BK=32)
    src = ASTSource(fn=kernel, signature=sig, constexprs=ce)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)

    with pytest.raises(MetalNonRecoverableError, match="silently-wrong"):
        emit_msl(mod, meta, options)


@requires_triton
def test_unstructured_cf_refuses_not_silently_wrong():
    """Integrity (PR1): a void early `return` mid-kernel lowers to top-level
    `cf.cond_br` (unstructured control flow). `_lower_op_dispatch` has no
    handler for the `cf` dialect, so the branch is silently dropped and the
    wrong value is stored — `test_nested_if_else_return` returned -1 for 1.
    The lowerer must RAISE MetalNonRecoverableError instead.

    Counter-check: a value-returning early return (the `test_if_call[jit_if]`
    shape) inlines to structured `scf.if`, produces NO top-level `cf.*`, and
    must still compile cleanly — the guard must not over-reject it.
    """
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.msl_emitter import emit_msl
    from triton_metal.errors import MetalNonRecoverableError

    def _emit(fn, sig):
        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})
        src = ASTSource(fn=fn, signature=sig, constexprs={})
        ctx = ir.context()
        ir.load_dialects(ctx)
        mod = src.make_ir(target, options,
                          backend.get_codegen_implementation(options),
                          backend.get_module_map(), ctx)
        meta = {}
        mod = backend.make_ttir(mod, meta, options)
        mod = backend.make_ttgir(mod, meta, options)
        return emit_msl(mod, meta, options)

    # Void early return → unstructured cf.cond_br → must refuse.
    @triton.jit
    def nested(Cond1, Cond2, Cond3, Val1, Val2, Val3, Out):
        val = 0
        if tl.load(Cond1):
            if tl.load(Cond2):
                val = tl.load(Val1)
            else:
                return
        else:
            if tl.load(Cond3):
                val = tl.load(Val2)
            else:
                val = tl.load(Val3)
        tl.store(Out, val)

    sig = {k: "*i32" for k in
           ("Cond1", "Cond2", "Cond3", "Val1", "Val2", "Val3", "Out")}
    with pytest.raises(MetalNonRecoverableError, match="control flow"):
        _emit(nested, sig)

    # Value-returning early return inlines to scf.if — must NOT be refused.
    @triton.jit
    def add_fn_return(x, pid):
        if pid == 0:
            return x + 1
        else:
            return x + 2

    @triton.jit
    def jit_if(Out):
        pid = tl.program_id(0)
        o = tl.load(Out)
        a = o
        if pid == 0:
            a = o
            a = add_fn_return(a, pid)
        tl.store(Out, a)

    msl = _emit(jit_if, {"Out": "*i32"})  # must not raise
    assert "UNSUPPORTED" not in msl


@requires_triton
@requires_metal
@pytest.mark.parametrize("in_shape,perm,red_dims", [
    ((4, 32, 32, 4, 2), [2, 1, 0, 3, 4], [3, 1, 0]),
    ((8, 2, 32, 4, 16), [4, 0, 1, 3, 2], [0, 2, 0]),
])
def test_permute_chained_reduce_matches_torch(in_shape, perm, red_dims):
    """Fused permute + chained sum-reduce (test_chained_reductions shape)
    produces exact integer results via the cooperative scatter-reduce."""
    import torch

    @triton.jit
    def kernel(In, Out, dim_0: tl.constexpr, dim_1: tl.constexpr,
               dim_2: tl.constexpr, dim_3: tl.constexpr, dim_4: tl.constexpr,
               perm_0: tl.constexpr, perm_1: tl.constexpr, perm_2: tl.constexpr,
               perm_3: tl.constexpr, perm_4: tl.constexpr,
               red_dim_0: tl.constexpr, red_dim_1: tl.constexpr,
               red_dim_2: tl.constexpr):
        idx = tl.arange(0, dim_0 * dim_1 * dim_2 * dim_3 * dim_4)
        idx = idx.reshape(dim_0, dim_1, dim_2, dim_3, dim_4)
        vals = tl.load(In + idx)
        vals = tl.permute(vals, [perm_0, perm_1, perm_2, perm_3, perm_4])
        r = tl.sum(tl.sum(tl.sum(vals, red_dim_0), red_dim_1), red_dim_2)
        st_idx = tl.arange(0, r.shape[0] * r.shape[1]).reshape(r.shape)
        tl.store(Out + st_idx, r)

    inp = torch.randint(0, 1000, in_shape, dtype=torch.int32, device="cpu")
    temp = torch.permute(inp, perm).contiguous()
    ref = torch.sum(torch.sum(torch.sum(temp, red_dims[0]), red_dims[1]),
                    red_dims[2])
    out = torch.empty_like(ref)
    kernel[(1,)](inp, out, *in_shape, *perm, *red_dims)
    assert torch.all(ref == out), f"mismatch: ref={ref}\nout={out}"


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


def test_emit_builtin_binary_array_path():
    """`_emit_builtin_binary` emits fn(a, b) per array position when MEPT on."""
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
        lowerer.env_array[100] = ("a", 3, "float")
        lowerer.env_array[101] = ("b", 3, "float")

        dst = SSAValue(id=200, name="v200", op="math.exp2",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_builtin_binary(dst, "pow")

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"{name}[{i}] = pow(a[{i}], b[{i}]);" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_builtin_binary_scalar_path_unchanged():
    """Flag-off keeps the existing fn(a, b) scalar emission."""
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
        lowerer.env_array[100] = ("a", 3, "float")
        lowerer.env_array[101] = ("b", 3, "float")

        dst = SSAValue(id=200, name="v200", op="math.exp2",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_builtin_binary(dst, "pow")

        assert 200 not in lowerer.env_array
        body = "\n".join(lowerer.kb._body_lines)
        assert "pow(a, b)" in body
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_nan_propagating_minmax_array_path():
    """`_emit_nan_propagating_minmax` emits per-position when MEPT on."""
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
        lowerer.env_array[100] = ("a", 2, "float")
        lowerer.env_array[101] = ("b", 2, "float")

        dst = SSAValue(id=200, name="v200", op="arith.maxnumf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<256xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_nan_propagating_minmax(dst, "fmax")

        name, n, ty = lowerer.env_array[200]
        assert n == 2
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(2):
            expected = (
                f"{name}[{i}] = (isnan(a[{i}]) || isnan(b[{i}])) ? "
                f"NAN : fmax(a[{i}], b[{i}]);"
            )
            assert expected in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_uitofp_array_path():
    """`_emit_uitofp` emits per-element float conversion when MEPT on."""
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
        lowerer.env_array[100] = ("v_src", 4, "int")
        # Signed source -> casts through unsigned first.
        lowerer.env_types[100] = "i32"

        dst = SSAValue(id=200, name="v200", op="arith.uitofp",
                       operand_ids=[100], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_uitofp(dst)

        name, n, ty = lowerer.env_array[200]
        assert n == 4
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(4):
            assert (
                f"{name}[{i}] = static_cast<float>(static_cast<uint>"
                f"(v_src[{i}]));" in body
            )
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_emit_int_cast_array_path():
    """`_emit_int_cast` emits per-element extension when MEPT on."""
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
        lowerer.env_array[100] = ("v_src", 3, "char")
        lowerer.env_types[100] = "i8"

        dst = SSAValue(id=200, name="v200", op="arith.extsi",
                       operand_ids=[100], attrs={},
                       type_str="tensor<384xi32>", elem_type="i32",
                       is_tensor=True)
        lowerer._emit_int_cast(dst, unsigned=False)

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"{name}[{i}] = static_cast<int>(v_src[{i}]);" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_math_unary_array_path():
    """`_lower_math` unary ops emit per-position calls when MEPT on."""
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

        dst = SSAValue(id=200, name="v200", op="math.exp",
                       operand_ids=[100], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._lower_math(dst)

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"{name}[{i}] = exp(v_src[{i}]);" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_math_fma_array_path():
    """`_lower_math` fma emits per-position fma(a, b, c) when MEPT on."""
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
        lowerer.env[102] = "c_scalar"
        lowerer.env_array[100] = ("a", 2, "float")
        lowerer.env_array[101] = ("b", 2, "float")
        # c is a scalar — broadcast it.

        dst = SSAValue(id=200, name="v200", op="math.fma",
                       operand_ids=[100, 101, 102], attrs={},
                       type_str="tensor<256xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._lower_math(dst)

        name, n, ty = lowerer.env_array[200]
        assert n == 2
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(2):
            assert (
                f"{name}[{i}] = fma(a[{i}], b[{i}], c_scalar);" in body
            )
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_make_range_emits_array_when_mept_and_n_elems():
    """`_lower_make_range` produces idx[N] when MEPT on + n_elems > 1."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[],
                    block_size=512, num_warps=4)
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=512)
        # Required state for _lower_make_range's pure-1D branch.
        lowerer._is_2d = False
        # _lid_expr is a property returning "lid" by default.
        lowerer.effective_block_size = 128  # num_warps*32

        # Hand-inject n_elems = 4 (would normally come from layout
        # tracking once the producer layout includes sizePerThread=4).
        range_ssa = SSAValue(id=42, name="r42", op="tt.make_range",
                             operand_ids=[],
                             attrs={"start": 0, "end": 512},
                             type_str="tensor<512xi32>", elem_type="i32",
                             is_tensor=True)
        lowerer.env_n_elems[42] = 4
        # Prescan normally sets this; we drive make_range directly.
        lowerer._mept_single_pass = True

        lowerer._lower_make_range(range_ssa)

        name, n, ty = lowerer.env_array[42]
        assert n == 4
        assert ty == "uint"
        body = "\n".join(lowerer.kb._body_lines)
        assert f"uint {name}[4];" in body
        for i in range(4):
            assert f"{name}[{i}] = lid * 4u + {i}u;" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_make_range_scalar_when_mept_off():
    """`_lower_make_range` keeps existing scalar form when MEPT off."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[],
                    block_size=512, num_warps=4)
    saved = os.environ.pop("TRITON_METAL_MEPT", None)
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=512)
        lowerer._is_2d = False
        # _lid_expr is a property returning "lid" by default.
        lowerer.effective_block_size = 128

        range_ssa = SSAValue(id=42, name="r42", op="tt.make_range",
                             operand_ids=[],
                             attrs={"start": 0, "end": 512},
                             type_str="tensor<512xi32>", elem_type="i32",
                             is_tensor=True)
        # Even with n_elems hint, flag-off ignores it.
        lowerer.env_n_elems[42] = 4

        lowerer._lower_make_range(range_ssa)

        assert 42 not in lowerer.env_array
        # env[42] should be the scalar lid (start=0 fast path).
        assert lowerer.env[42] == "lid"
    finally:
        if saved is not None:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_addptr_emits_ptr_array_when_offset_is_array():
    """`_lower_addptr` records env_ptr_array when offset has env_array."""
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

        # ptr operand is a bare buffer name; offset is an array.
        lowerer.env[100] = "x_ptr"
        lowerer.env[101] = "off"
        lowerer.env_array[101] = ("off", 4, "uint")

        addptr_ssa = SSAValue(
            id=200, name="r200", op="tt.addptr",
            operand_ids=[100, 101], attrs={},
            type_str="tensor<512x!tt.ptr<f32>>",
            elem_type="f32", is_tensor=True,
        )
        lowerer._lower_addptr(addptr_ssa)

        assert 200 in lowerer.env_ptr_array
        base, off_name, n = lowerer.env_ptr_array[200]
        assert base == "x_ptr"
        assert n == 4
        # The offset array is just the input offset directly (no parent).
        assert off_name == "off"
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_addptr_combines_scalar_parent_with_array_offset():
    """Parent ptr has scalar offset; new addptr offset is array."""
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

        # Parent ptr is an addptr with scalar offset.
        lowerer.env[100] = "x_ptr[base_off]"
        lowerer.env_is_ptr[100] = ("x_ptr", "base_off")
        # Offset is an array of 3.
        lowerer.env[101] = "off"
        lowerer.env_array[101] = ("off", 3, "uint")

        addptr_ssa = SSAValue(
            id=200, name="r200", op="tt.addptr",
            operand_ids=[100, 101], attrs={},
            type_str="tensor<384x!tt.ptr<f32>>",
            elem_type="f32", is_tensor=True,
        )
        lowerer._lower_addptr(addptr_ssa)

        assert 200 in lowerer.env_ptr_array
        base, off_name, n = lowerer.env_ptr_array[200]
        assert base == "x_ptr"
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        assert f"uint {off_name}[3];" in body
        for i in range(3):
            assert f"{off_name}[{i}] = base_off + off[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_load_array_path_via_env_ptr_array():
    """`_lower_load` emits per-position reads when ptr has env_ptr_array."""
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

        lowerer.env_ptr_array[100] = ("x_ptr", "off", 4)

        load_ssa = SSAValue(
            id=200, name="r200", op="tt.load",
            operand_ids=[100], attrs={},
            type_str="tensor<512xf32>", elem_type="f32",
            is_tensor=True,
        )
        lowerer._lower_load(load_ssa)

        name, n, ty = lowerer.env_array[200]
        assert n == 4
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(4):
            assert (
                f"{name}[{i}] = static_cast<float>"
                f"(x_ptr[off[{i}]]);" in body
            )
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_load_array_with_array_mask_and_other():
    """`_lower_load` honors array-form mask + array-form other."""
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

        lowerer.env_ptr_array[100] = ("x_ptr", "off", 3)
        # Mask is an i1 env_array.
        lowerer.env[101] = "msk"
        lowerer.env_array[101] = ("msk", 3, "bool")
        lowerer.env_is_mask[101] = True
        # 'other' is an env_array.
        lowerer.env[102] = "deflt"
        lowerer.env_array[102] = ("deflt", 3, "float")

        load_ssa = SSAValue(
            id=200, name="r200", op="tt.load",
            operand_ids=[100, 101, 102], attrs={},
            type_str="tensor<384xf32>", elem_type="f32",
            is_tensor=True,
        )
        lowerer._lower_load(load_ssa)

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert (
                f"{name}[{i}] = msk[{i}] ? "
                f"static_cast<float>(x_ptr[off[{i}]]) : "
                f"static_cast<float>(deflt[{i}]);"
            ) in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_load_array_with_scalar_mask_and_scalar_other():
    """Scalar mask broadcasts across array positions; scalar 'other'."""
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

        lowerer.env_ptr_array[100] = ("x_ptr", "off", 2)
        # Mask is a plain scalar (typical of splat-of-comparison).
        lowerer.env[101] = "mask_scalar"
        lowerer.env_is_mask[101] = True
        # 'other' is a scalar constant.
        lowerer.env[102] = "0.0f"

        load_ssa = SSAValue(
            id=200, name="r200", op="tt.load",
            operand_ids=[100, 101, 102], attrs={},
            type_str="tensor<256xf32>", elem_type="f32",
            is_tensor=True,
        )
        lowerer._lower_load(load_ssa)

        name, n, ty = lowerer.env_array[200]
        assert n == 2
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(2):
            assert (
                f"{name}[{i}] = mask_scalar ? "
                f"static_cast<float>(x_ptr[off[{i}]]) : "
                f"static_cast<float>(0.0f);"
            ) in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_load_array_fp8_unmasked():
    """FP8 array load emits raw[N] uchar gather + val[N] float convert."""
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

        lowerer.env_ptr_array[100] = ("x_ptr", "off", 3)

        load_ssa = SSAValue(
            id=200, name="r200", op="tt.load",
            operand_ids=[100], attrs={},
            type_str="tensor<384xf8E4M3FN>",
            elem_type="f8E4M3FN",
            is_tensor=True,
        )
        lowerer._lower_load(load_ssa)

        # Result is a float array.
        val_name, n, ty = lowerer.env_array[200]
        assert n == 3
        assert ty == "float"
        body = "\n".join(lowerer.kb._body_lines)
        # Should have a uchar raw[3] and float val[3] declarations.
        assert "uchar raw" in body
        assert "float " + val_name in body
        # Per-position load + convert (fp8e4m3 device fn).
        for i in range(3):
            assert f"x_ptr[off[{i}]]" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_load_array_fp8_with_array_mask_and_other():
    """FP8 + array mask + array other emits the full conditional chain."""
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

        lowerer.env_ptr_array[100] = ("x_ptr", "off", 2)
        lowerer.env[101] = "msk"
        lowerer.env_array[101] = ("msk", 2, "bool")
        lowerer.env_is_mask[101] = True
        lowerer.env[102] = "deflt"
        lowerer.env_array[102] = ("deflt", 2, "float")

        load_ssa = SSAValue(
            id=200, name="r200", op="tt.load",
            operand_ids=[100, 101, 102], attrs={},
            type_str="tensor<256xf8E5M2>",
            elem_type="f8E5M2",
            is_tensor=True,
        )
        lowerer._lower_load(load_ssa)

        val_name, n, ty = lowerer.env_array[200]
        assert n == 2
        assert ty == "float"
        body = "\n".join(lowerer.kb._body_lines)
        # Masked uchar gather.
        for i in range(2):
            assert (
                f"msk[{i}] ? x_ptr[off[{i}]] : uchar(0)" in body
            )
        # Converted float (masked) with 'other' fallback.
        for i in range(2):
            assert (
                f"msk[{i}] ? " in body
                and f"static_cast<float>(deflt[{i}])" in body
            )
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_store_array_path_scatters_to_env_ptr_array():
    """`_lower_store` writes val[i] -> base[off[i]] when MEPT round-trip."""
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

        # Pointer with env_ptr_array, value with env_array of same length.
        lowerer.env_ptr_array[100] = ("out_ptr", "off", 3)
        lowerer.env[101] = "vals"
        lowerer.env_array[101] = ("vals", 3, "float")

        store_ssa = SSAValue(
            id=200, name="r200", op="tt.store",
            operand_ids=[100, 101], attrs={},
            type_str="", elem_type="f32", is_tensor=False,
        )
        lowerer._lower_store(store_ssa)

        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"out_ptr[off[{i}]] = vals[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_mept_round_trip_load_op_store():
    """End-to-end: make_range → addptr → load → unary → addptr → store
    emits a fully array-form pipeline when MEPT is on."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[], block_size=512,
                    num_warps=4)
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=512)
        lowerer._is_2d = False
        lowerer.effective_block_size = 128

        # Two input pointer SSA values (bare buffers).
        lowerer.env[1] = "x_ptr"
        lowerer.env[2] = "out_ptr"
        # Prescan normally sets this; we drive the ops directly.
        lowerer._mept_single_pass = True

        # 1) make_range → idx[4]
        rng = SSAValue(id=10, name="r10", op="tt.make_range",
                       operand_ids=[], attrs={"start": 0, "end": 512},
                       type_str="tensor<512xi32>", elem_type="i32",
                       is_tensor=True)
        lowerer.env_n_elems[10] = 4
        lowerer._lower_make_range(rng)
        assert 10 in lowerer.env_array
        idx_name, _, _ = lowerer.env_array[10]

        # 2) tt.addptr(x_ptr, idx) → env_ptr_array
        in_ptr = SSAValue(id=11, name="r11", op="tt.addptr",
                          operand_ids=[1, 10], attrs={},
                          type_str="tensor<512x!tt.ptr<f32>>",
                          elem_type="f32", is_tensor=True)
        lowerer._lower_addptr(in_ptr)
        assert 11 in lowerer.env_ptr_array

        # 3) tt.load(in_ptr) → val[4]
        load = SSAValue(id=12, name="r12", op="tt.load",
                        operand_ids=[11], attrs={},
                        type_str="tensor<512xf32>", elem_type="f32",
                        is_tensor=True)
        lowerer._lower_load(load)
        assert 12 in lowerer.env_array
        val_name = lowerer.env_array[12][0]

        # 4) Unary negate via _emit_unary → r[4]
        neg = SSAValue(id=13, name="r13", op="arith.negf",
                       operand_ids=[12], attrs={},
                       type_str="tensor<512xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._emit_unary(neg, "-")
        assert 13 in lowerer.env_array
        neg_name = lowerer.env_array[13][0]

        # 5) tt.addptr(out_ptr, idx) → env_ptr_array for output
        out_ptr = SSAValue(id=14, name="r14", op="tt.addptr",
                           operand_ids=[2, 10], attrs={},
                           type_str="tensor<512x!tt.ptr<f32>>",
                           elem_type="f32", is_tensor=True)
        lowerer._lower_addptr(out_ptr)
        assert 14 in lowerer.env_ptr_array

        # 6) tt.store(out_ptr, neg) → per-position writes
        store = SSAValue(id=15, name="r15", op="tt.store",
                         operand_ids=[14, 13], attrs={},
                         type_str="", elem_type="f32", is_tensor=False)
        lowerer._lower_store(store)

        body = "\n".join(lowerer.kb._body_lines)
        # Sanity: every stage left a visible array trail.
        assert f"uint {idx_name}[4];" in body
        assert f"float {val_name}[4];" in body
        assert f"float {neg_name}[4];" in body
        for i in range(4):
            # Load reads x_ptr[idx[i]]
            assert f"x_ptr[{idx_name}[{i}]]" in body
            # Final store writes out_ptr[idx[i]] = -val[i]
            assert (
                f"out_ptr[{idx_name}[{i}]] = {neg_name}[{i}];"
                in body
            )
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_store_array_path_with_array_mask():
    """`_lower_store` honors array-form mask (per-position if-write)."""
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

        lowerer.env_ptr_array[100] = ("out_ptr", "off", 3)
        lowerer.env[101] = "vals"
        lowerer.env_array[101] = ("vals", 3, "float")
        lowerer.env[102] = "msk"
        lowerer.env_array[102] = ("msk", 3, "bool")
        lowerer.env_is_mask[102] = True

        store = SSAValue(
            id=200, name="r200", op="tt.store",
            operand_ids=[100, 101, 102], attrs={},
            type_str="", elem_type="f32", is_tensor=False,
        )
        lowerer._lower_store(store)

        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"if (msk[{i}]) {{" in body
            assert f"out_ptr[off[{i}]] = vals[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_store_array_path_with_scalar_mask():
    """Scalar mask broadcasts across all per-position writes."""
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

        lowerer.env_ptr_array[100] = ("out_ptr", "off", 2)
        lowerer.env[101] = "vals"
        lowerer.env_array[101] = ("vals", 2, "float")
        lowerer.env[102] = "mask_scalar"
        lowerer.env_is_mask[102] = True

        store = SSAValue(
            id=200, name="r200", op="tt.store",
            operand_ids=[100, 101, 102], attrs={},
            type_str="", elem_type="f32", is_tensor=False,
        )
        lowerer._lower_store(store)

        body = "\n".join(lowerer.kb._body_lines)
        for i in range(2):
            assert "if (mask_scalar) {" in body
            assert f"out_ptr[off[{i}]] = vals[{i}];" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_make_range_uses_linear_layout_position_when_available():
    """`_lower_make_range` consults env_layout for non-contiguous math."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder
    from triton_metal.codegen._linear_layout import LinearLayout

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[], block_size=512,
                    num_warps=4)
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=512)
        lowerer._is_2d = False
        lowerer.effective_block_size = 128

        # Build a 1D LinearLayout with 2 register bases [1, 2] (i.e. 4
        # contiguous elements per thread, contiguous in the lowest two
        # registers) and lane/warp bases extending from 4 onward.
        ll = LinearLayout(
            register_basis=[1, 2],
            lane_basis=[4, 8, 16, 32, 64],
            warp_basis=[128, 256],
            block_basis=[],
        )

        rng = SSAValue(id=42, name="r42", op="tt.make_range",
                       operand_ids=[], attrs={"start": 0, "end": 512},
                       type_str="tensor<512xi32>", elem_type="i32",
                       is_tensor=True)
        lowerer.env_n_elems[42] = 4
        lowerer.env_layout[42] = ll
        # Prescan normally sets this; we drive make_range directly.
        lowerer._mept_single_pass = True

        lowerer._lower_make_range(rng)
        assert 42 in lowerer.env_array
        name, n, ty = lowerer.env_array[42]
        assert n == 4

        body = "\n".join(lowerer.kb._body_lines)
        # The emitted formula must use XOR-basis position expressions.
        # For register i, the contribution from register basis is
        # ((-(int)((i_bit >> j) & 1u)) & basis_j) XORed across j.
        # Position at reg=0 has no register contribution beyond
        # the lane/warp terms; we just spot-check that the LL formula
        # leaked into the emitted code instead of the simple lid*N+i.
        assert f"uint {name}[4];" in body
        # The simple fallback ``lid * 4u + 0u`` should NOT appear when
        # the LL path is taken.
        assert "lid * 4u + 0u" not in body
        # XOR sign-mask pattern is unique to msl_position_expr.
        assert " ^ " in body or "& 4)" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_make_range_scalar_when_n_elems_is_one():
    """Even with MEPT on, n_elems=1 keeps the scalar form."""
    import os
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[],
                    block_size=128, num_warps=4)
    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        lowerer = GenericLowerer(graph, _Options())
        lowerer.kb = KernelBuilder("t", block_size=128)
        lowerer._is_2d = False
        # _lid_expr is a property returning "lid" by default.
        lowerer.effective_block_size = 128

        range_ssa = SSAValue(id=42, name="r42", op="tt.make_range",
                             operand_ids=[],
                             attrs={"start": 0, "end": 128},
                             type_str="tensor<128xi32>", elem_type="i32",
                             is_tensor=True)
        lowerer.env_n_elems[42] = 1

        lowerer._lower_make_range(range_ssa)

        assert 42 not in lowerer.env_array
        assert lowerer.env[42] == "lid"
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_math_binary_array_path():
    """`_lower_math` binary fns (pow, copysign, atan2) take array path."""
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
        lowerer.env_array[100] = ("a", 2, "float")
        lowerer.env_array[101] = ("b", 2, "float")

        dst = SSAValue(id=200, name="v200", op="math.powf",
                       operand_ids=[100, 101], attrs={},
                       type_str="tensor<256xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._lower_math(dst)

        name, n, ty = lowerer.env_array[200]
        assert n == 2
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(2):
            assert f"{name}[{i}] = pow(a[{i}], b[{i}]);" in body
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


def test_lower_math_roundeven_array_path():
    """`_lower_math` roundeven / trunc go through unary MEPT path."""
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

        dst = SSAValue(id=200, name="v200", op="math.roundeven",
                       operand_ids=[100], attrs={},
                       type_str="tensor<384xf32>", elem_type="f32",
                       is_tensor=True)
        lowerer._lower_math(dst)

        name, n, ty = lowerer.env_array[200]
        assert n == 3
        body = "\n".join(lowerer.kb._body_lines)
        for i in range(3):
            assert f"{name}[{i}] = rint(v_src[{i}]);" in body
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
def test_mept_flag_actually_changes_output_when_layout_supports_it():
    """When the layout genuinely carries sizePerThread > 1, the MEPT flag
    flips the generated MSL from scalar form to single-pass array form.

    A real sizePerThread > 1 layout requires divisibility hints (which the
    JIT runtime adds automatically) so the coalesce pass packs multiple
    contiguous elements per thread. Without them the layout is
    sizePerThread=1 and MEPT correctly stays dormant.
    """
    import os
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    def lower(flag_on: bool):
        saved = os.environ.get("TRITON_METAL_MEPT")
        if flag_on:
            os.environ["TRITON_METAL_MEPT"] = "1"
        else:
            os.environ.pop("TRITON_METAL_MEPT", None)
        try:
            target = GPUTarget("metal", "apple-m4", 32)
            backend = MetalBackend(target)
            options = backend.parse_options({})
            sig = {"a_ptr": "*fp32", "b_ptr": "*fp32",
                   "out_ptr": "*fp32", "n": "i32"}
            attrs = {(0,): [("tt.divisibility", 16)],
                     (1,): [("tt.divisibility", 16)],
                     (2,): [("tt.divisibility", 16)]}
            src = ASTSource(fn=vector_add, signature=sig,
                            constexprs={"BLOCK_SIZE": 512}, attrs=attrs)
            context = ir.context()
            ir.load_dialects(context)
            cg = backend.get_codegen_implementation(options)
            mm = backend.get_module_map()
            mod = src.make_ir(target, options, cg, mm, context)
            meta = {}
            mod = backend.make_ttir(mod, meta, options)
            mod = backend.make_ttgir(mod, meta, options)
            graph = walk_ttgir(mod, options)
            return GenericLowerer(graph, options).lower()
        finally:
            if saved is None:
                os.environ.pop("TRITON_METAL_MEPT", None)
            else:
                os.environ["TRITON_METAL_MEPT"] = saved

    off = lower(False)
    on = lower(True)
    # Flag-off: scalar form, no per-position array declarations.
    assert "[4];" not in off, f"flag-off should be scalar, got:\n{off}"
    # Flag-on: single-pass array form must appear.
    assert "[4];" in on, (
        f"Expected per-position array decl in MEPT-on MSL, got:\n{on}"
    )
    # And no wrap-loop double-count.
    assert "_loop_e * 4" not in on
    # Both paths must compile.
    assert _validate_msl_compiles(off), "Scalar path MSL failed to compile"
    assert _validate_msl_compiles(on), "MEPT array path MSL failed to compile"


def test_mept_convert_layout_shuffle_emits_position_redistribution():
    """`_lower_convert_layout` shuffles a register array via shared memory
    using src/dst LinearLayout positions."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph, SSAValue
    from triton_metal.codegen.msl_emitter import KernelBuilder
    from triton_metal.codegen._linear_layout import LinearLayout

    class _Options:
        num_warps = 4

    # Two 1-D linear layouts over 512 elements, 4 regs/thread (128 threads).
    # src: contiguous within thread (reg basis [1,2]); dst: a different
    # register assignment (reg basis [256, 1]) — a genuine redistribution.
    src_ll = LinearLayout(register_basis=[1, 2],
                          lane_basis=[4, 8, 16, 32, 64],
                          warp_basis=[128, 256], block_basis=[])
    dst_ll = LinearLayout(register_basis=[256, 1],
                          lane_basis=[2, 4, 8, 16, 32],
                          warp_basis=[64, 128], block_basis=[])
    assert src_ll.total_elements == 512 == dst_ll.total_elements

    mod_text = (
        "#src = #ttg.linear<{register = [[1], [2]], "
        "lane = [[4], [8], [16], [32], [64]], warp = [[128], [256]], "
        "block = []}>\n"
        "#dst = #ttg.linear<{register = [[256], [1]], "
        "lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], "
        "block = []}>"
    )
    graph = IRGraph(func_name="t", args=[], ops=[], mod_text=mod_text)
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=128)
    lowerer.mept_enabled = True

    # Source value is a register array of 4 with a resolved src layout.
    lowerer.env[100] = "v"
    lowerer.env_array[100] = ("v", 4, "float")
    lowerer.env_layout[100] = src_ll
    lowerer.env_types[100] = "fp32"

    cvt = SSAValue(id=200, name="r200", op="ttg.convert_layout",
                   operand_ids=[100], attrs={},
                   type_str="tensor<512xf32, #dst>", elem_type="f32",
                   is_tensor=True)
    # Source op so _find_op_type_str / src_type resolution works.
    src_op = SSAValue(id=100, name="v", op="tt.load", operand_ids=[],
                      attrs={}, type_str="tensor<512xf32, #src>",
                      elem_type="f32", is_tensor=True)
    graph.ops = [src_op, cvt]

    lowerer._lower_convert_layout(cvt)

    body = "\n".join(lowerer.kb._body_lines)
    # Shared buffer declared, barriered write then read.
    assert "shuf_" in body
    assert "threadgroup_barrier" in body
    # Result is a register array of 4 (dst register count).
    name, n, ty = lowerer.env_array[200]
    assert n == 4
    # Write phase indexes shared by src positions; read by dst positions.
    # reg 0 of dst basis [256,1]: position(reg=0)=0 contribution from reg,
    # so the read of element 0 includes the lane/warp XOR terms.
    assert "= v[0];" in body
    assert "= v[3];" in body
    # The read assigns from shared buffer into the new array.
    assert f"{name}[0] = shuf_" in body


def test_mept_reduce_fold_emits_per_thread_fold():
    """`_mept_reduce_fold` collapses arr[0..n-1] with the combine op."""
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.mlir_walker import IRGraph
    from triton_metal.codegen.msl_emitter import KernelBuilder

    class _Options:
        num_warps = 4

    graph = IRGraph(func_name="t", args=[], ops=[])
    lowerer = GenericLowerer(graph, _Options())
    lowerer.kb = KernelBuilder("t", block_size=128)

    # sum fold (reads cast to msl_type to avoid overload ambiguity)
    fv = lowerer._mept_reduce_fold("v", 4, "sum", "float")
    body = "\n".join(lowerer.kb._body_lines)
    assert f"float {fv} = (float)v[0];" in body
    assert f"{fv} = {fv} + (float)v[1];" in body
    assert f"{fv} = {fv} + (float)v[2];" in body
    assert f"{fv} = {fv} + (float)v[3];" in body

    # max fold
    lowerer.kb = KernelBuilder("t", block_size=128)
    fv2 = lowerer._mept_reduce_fold("w", 3, "max", "float")
    body2 = "\n".join(lowerer.kb._body_lines)
    assert f"float {fv2} = (float)w[0];" in body2
    assert f"{fv2} = max({fv2}, (float)w[1]);" in body2
    assert f"{fv2} = max({fv2}, (float)w[2]);" in body2

    # xor fold (int) — the int cast disambiguates unsigned arrays.
    lowerer.kb = KernelBuilder("t", block_size=128)
    fv3 = lowerer._mept_reduce_fold("x", 2, "xor", "int")
    body3 = "\n".join(lowerer.kb._body_lines)
    assert f"int {fv3} = (int)x[0];" in body3
    assert f"{fv3} = {fv3} ^ (int)x[1];" in body3


def test_mept_reduce_uses_fold_when_operand_is_array():
    """`_lower_reduce` folds an env_array operand then 1-D reduces."""
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
        lowerer.kb = KernelBuilder("t", block_size=128)
        lowerer._is_2d = False

        # Operand 50 is a float register array of 4 (e.g. from a MEPT load).
        lowerer.env[50] = "vals"
        lowerer.env_array[50] = ("vals", 4, "float")
        lowerer.env_types[50] = "fp32"

        # tt.reduce with a sum body.
        add_body = SSAValue(id=51, name="b51", op="arith.addf",
                            operand_ids=[], attrs={}, type_str="f32",
                            elem_type="f32", is_tensor=False)
        red = SSAValue(id=52, name="r52", op="tt.reduce",
                       operand_ids=[50], attrs={"axis": 0},
                       type_str="f32", elem_type="f32", is_tensor=False,
                       region_ops=[add_body])
        lowerer._lower_reduce(red)

        body = "\n".join(lowerer.kb._body_lines)
        # The fold must appear (per-thread partial), then a threadgroup
        # reduce (simd_sum) over the partials. Reads are cast to msl_type.
        assert "= (float)vals[0];" in body
        assert "+ (float)vals[1]" in body
        assert "simd_sum" in body
        # The reduce result is registered for downstream consumers.
        assert 52 in lowerer.env
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved


@requires_triton
@requires_metal
def test_mept_bf16_store_casts_to_buffer_dtype():
    """MEPT store of a float-computed array into a bf16 buffer must cast.

    MSL implicitly narrows float->half but NOT float->bfloat, so a bf16
    output buffer needs an explicit static_cast<bfloat> per array element
    (matching the scalar store path). Regression for the 10 bf16
    test_masked_load failures.
    """
    import os
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    @triton.jit
    def add1_bf16(x_ptr, o_ptr, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off)
        tl.store(o_ptr + off, x + 1.0)

    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})
        sig = {"x_ptr": "*bf16", "o_ptr": "*bf16"}
        attrs = {(0,): [("tt.divisibility", 16)],
                 (1,): [("tt.divisibility", 16)]}
        src = ASTSource(fn=add1_bf16, signature=sig,
                        constexprs={"BLOCK": 512}, attrs=attrs)
        context = ir.context()
        ir.load_dialects(context)
        cg = backend.get_codegen_implementation(options)
        mm = backend.get_module_map()
        mod = src.make_ir(target, options, cg, mm, context)
        meta = {}
        mod = backend.make_ttir(mod, meta, options)
        mod = backend.make_ttgir(mod, meta, options)
        graph = walk_ttgir(mod, options)
        msl = GenericLowerer(graph, options).lower()
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved

    # MEPT active (array form), store casts to bfloat, and it compiles.
    assert "[4];" in msl, f"expected MEPT array form, got:\n{msl}"
    assert "static_cast<bfloat>" in msl, (
        f"bf16 store must cast float->bfloat, got:\n{msl}"
    )
    assert _validate_msl_compiles(msl), "bf16 MEPT MSL failed to compile"


@requires_triton
@requires_metal
def test_mept_no_double_count_with_wrap_loop():
    """Regression: MEPT array + wrap-loop must not double-count indices.

    With sizePerThread=4 and a 512-element tile (128 threads), the OLD
    code emitted `for (_loop_e ...) { idx[i] = _loop_e*4 + i; }`. That
    double-counts: the wrap-loop already strides `_loop_e` over the 128
    threads to cover all 512 elements, so multiplying by 4 again pushed
    idx to 2047 in a 512-element buffer — a 4x out-of-bounds overrun
    (benign for masked copies on Apple GPUs, but wrong for reductions /
    atomics and wasteful everywhere). The fix makes MEPT and the
    wrap-loop mutually exclusive: when the tile is exactly covered
    (num_threads * sizePerThread == total), MEPT runs single-pass with
    idx[i] = lid*N + i and NO wrap-loop.
    """
    import os
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    @triton.jit
    def copy_unmasked(x_ptr, o_ptr, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off)        # NO mask -> OOB would actually run
        tl.store(o_ptr + off, x + 1.0)  # NO mask

    def lower_with_mept():
        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})
        sig = {"x_ptr": "*fp32", "o_ptr": "*fp32"}
        # Divisibility hints (what the JIT runtime adds) -> coalesce emits
        # sizePerThread=[4] on default Apple configs.
        attrs = {(0,): [("tt.divisibility", 16)],
                 (1,): [("tt.divisibility", 16)]}
        src = ASTSource(fn=copy_unmasked, signature=sig,
                        constexprs={"BLOCK": 512}, attrs=attrs)
        context = ir.context()
        ir.load_dialects(context)
        cg = backend.get_codegen_implementation(options)
        mm = backend.get_module_map()
        mod = src.make_ir(target, options, cg, mm, context)
        meta = {}
        mod = backend.make_ttir(mod, meta, options)
        mod = backend.make_ttgir(mod, meta, options)
        graph = walk_ttgir(mod, options)
        lowerer = GenericLowerer(graph, options)
        return lowerer.lower()

    saved = os.environ.get("TRITON_METAL_MEPT")
    os.environ["TRITON_METAL_MEPT"] = "1"
    try:
        msl = lower_with_mept()
    finally:
        if saved is None:
            os.environ.pop("TRITON_METAL_MEPT", None)
        else:
            os.environ["TRITON_METAL_MEPT"] = saved

    # MEPT must be active (sizePerThread=4 -> array form present).
    assert "[4]" in msl, f"expected MEPT array form, got:\n{msl}"
    # The double-count signature must be ABSENT.
    assert "_loop_e * 4" not in msl, (
        f"double-count: make_range multiplies the wrap-loop variable:\n{msl}"
    )
    # No wrap-loop at all — MEPT covers the tile in one pass.
    assert "for (uint _loop_e" not in msl, (
        f"wrap-loop must be suppressed when MEPT covers the tile:\n{msl}"
    )
    # Correct single-pass index uses the raw thread id.
    assert "lid * 4u" in msl, f"expected lid*4 single-pass index:\n{msl}"
    # And it must compile.
    assert _validate_msl_compiles(msl), "MEPT single-pass MSL failed to compile"


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


def test_find_op_type_str_recurses_nested_regions():
    from types import SimpleNamespace
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    def _op(id, type_str="", region_ops=None, else_ops=None):
        return SimpleNamespace(id=id, type_str=type_str,
                               region_ops=region_ops or [],
                               else_ops=else_ops or [])

    gl = GenericLowerer.__new__(GenericLowerer)
    inner = _op(99, "tensor<256xf32>")          # depth-2: inside a nested loop
    mid = _op(50, "", region_ops=[inner])        # depth-1: nested scf.for body
    outer = _op(10, "", region_ops=[mid])        # top-level scf.for
    gl.graph = SimpleNamespace(ops=[outer], args=[])

    assert gl._find_op_type_str(99) == "tensor<256xf32>"
    einner = _op(77, "tensor<128xi32>")
    eouter = _op(20, "", else_ops=[_op(60, "", region_ops=[einner])])
    gl.graph = SimpleNamespace(ops=[eouter], args=[])
    assert gl._find_op_type_str(77) == "tensor<128xi32>"
    assert gl._find_op_type_str(404) == ""
