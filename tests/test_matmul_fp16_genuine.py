"""WS1 Phase C.1: fp16 matmul must be GENUINE.

The "fp16" simdgroup matmul template used to upcast halves to float and run
`simdgroup_float8x8` MMA — i.e. it ran fp32 regardless of dtype (the harness
measured fp16==fp32==7.00 TFLOP/s). These tests assert on the GENERATED MSL so
"fp16" can never silently regress to fp32 again. The accumulator stays float
(fp32 accumulation for precision); only the INPUT fragments become half.
"""
import re

# Import via the public surface (msl_emitter), NOT _msl_templates directly:
# importing _msl_templates before msl_emitter triggers a pre-existing
# circular/star-import fragility that breaks `make_matmul_kernel` (tracked
# follow-up). msl_emitter re-exports make_simdgroup_matmul_kernel.
from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel


def test_fp16_matmul_uses_half_input_fragments():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_half8x8" in msl, \
        "fp16 MMA must use simdgroup_half8x8 INPUT fragments, not float8x8"


def test_fp16_matmul_does_not_upcast_inputs_before_mma():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    # staging buffers must be half (no float staging), and no float(A[/float(B[
    assert "threadgroup half" in msl, "fp16 must stage inputs as half"
    assert not re.search(r"float\(\s*A\[", msl), "no float() upcast of A"
    assert not re.search(r"float\(\s*B\[", msl), "no float() upcast of B"


def test_fp16_accumulator_stays_float_for_precision():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_float8x8 acc" in msl, \
        "accumulator must stay simdgroup_float8x8 (fp32 accumulation)"


def test_fp32_matmul_unchanged_still_float_fragments():
    # the fp32 path must remain float fragments (regression guard).
    msl = make_simdgroup_matmul_kernel(dtype="fp32")
    assert "simdgroup_float8x8 a_frag" in msl
    assert "simdgroup_half8x8" not in msl


def _emit_jit_matmul_msl(dtype_sig):
    """Compile a @triton.jit fp16/fp32 matmul through the Metal pipeline and
    return the emitted MSL — exercises the INLINE dot path (real kernels), not
    the standalone template."""
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.msl_emitter import emit_msl

    @triton.jit
    def mm(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        ok = tl.arange(0, BK)
        a = tl.load(A + om[:, None] * BK + ok[None, :])
        b = tl.load(B + ok[:, None] * BN + on[None, :])
        c = tl.dot(a, b)
        tl.store(C + om[:, None] * BN + on[None, :], c)

    t = GPUTarget("metal", "apple-m4", 32)
    be = MetalBackend(t)
    o = be.parse_options({})
    src = ASTSource(fn=mm, signature={"A": dtype_sig, "B": dtype_sig, "C": "*fp32"},
                    constexprs=dict(BM=32, BN=32, BK=32))
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(t, o, be.get_codegen_implementation(o),
                      be.get_module_map(), ctx)
    meta = {}
    mod = be.make_ttir(mod, meta, o)
    mod = be.make_ttgir(mod, meta, o)
    return emit_msl(mod, meta, o)


def test_inline_jit_fp16_matmul_is_genuine():
    """Real @triton.jit fp16 matmuls (inline dot path) must also use genuine
    half fragments — not just the standalone harness template."""
    try:
        import triton  # noqa
    except Exception:
        pytest.skip("triton not importable")
    msl = _emit_jit_matmul_msl("*fp16")
    assert "simdgroup_half8x8" in msl, "inline fp16 matmul must use half frags"
    assert "simdgroup_multiply_accumulate" in msl
    assert not re.search(r"float\(\s*A\[", msl), "no float upcast of A in fp16"


def test_inline_jit_fp32_matmul_stays_float():
    try:
        import triton  # noqa
    except Exception:
        pytest.skip("triton not importable")
    msl = _emit_jit_matmul_msl("*fp32")
    assert "simdgroup_half8x8" not in msl  # fp32 stays on the float path


def test_import_order_msl_templates_first_reexports_matmul(tmp_path):
    """#152: importing _msl_templates BEFORE msl_emitter must not break
    msl_emitter's re-export of make_matmul_kernel (circular-import regression).
    Run in a subprocess so the import order is actually fresh."""
    import subprocess
    import sys
    code = (
        "import triton_metal.codegen._msl_templates\n"
        "import triton_metal.codegen.msl_emitter as e\n"
        "assert hasattr(e, 'make_matmul_kernel'), 'make_matmul_kernel missing'\n"
        "assert 'kernel' in e.make_matmul_kernel().lower()\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
