"""MEPT parity gate: flag-ON emitted MSL must equal flag-OFF on the scalar
corpus. The ratchet invariant -- the unified model must reproduce today's
output byte-for-byte until a milestone deliberately unlocks new behavior."""
import importlib
import os

import pytest
import triton  # noqa: F401
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


def _emit(fn, sig, cst, mept):
    os.environ["TRITON_MSL_FORCE_PYTHON"] = "1"
    os.environ["TRITON_MSL_MEPT"] = "1" if mept else "0"
    import triton_msl.codegen.generic_lowerer as G
    import triton_msl.codegen.msl_emitter as M
    importlib.reload(G)
    importlib.reload(M)
    from triton_msl.backend.compiler import MetalBackend
    t = GPUTarget("metal", "apple-m4", 32)
    be = MetalBackend(t)
    o = be.parse_options({})
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(t, o, be.get_codegen_implementation(o), be.get_module_map(), ctx)
    meta = {}
    mod = be.make_ttir(mod, meta, o)
    mod = be.make_ttgir(mod, meta, o)
    return M.emit_msl(mod, meta, o)


@triton.jit
def _vadd(X, Y, O, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(O + i, tl.load(X + i) + tl.load(Y + i))


@triton.jit
def _vmul_scalar(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(O + i, tl.load(X + i) * 3.0 + 1.0)


@pytest.mark.parametrize("fn,sig,cst", [
    (_vadd, {"X": "*fp32", "Y": "*fp32", "O": "*fp32"}, dict(N=256)),
    (_vmul_scalar, {"X": "*fp32", "O": "*fp32"}, dict(N=256)),
])
def test_mept_flag_parity_scalar_corpus(fn, sig, cst):
    off = _emit(fn, sig, cst, mept=False)
    on = _emit(fn, sig, cst, mept=True)
    assert on == off, (
        "MEPT flag changed scalar MSL:\n--- OFF ---\n%s\n--- ON ---\n%s" % (off, on))


def teardown_module(module):
    # _emit mutates these process-global env vars; the lowerer reads them
    # per-compile, so leaving a stale value would flip later test files onto
    # the wrong path. Remove both so the process returns to its true defaults
    # (MEPT is default-ON since M5; popping it restores that default).
    os.environ.pop("TRITON_MSL_MEPT", None)
    os.environ.pop("TRITON_MSL_FORCE_PYTHON", None)
