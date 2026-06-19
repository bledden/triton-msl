"""MEPT M3a: an scf.for carrying a multi-element register-array iter-arg
emits array-indexed MSL (no UNKNOWN_) under flag-ON. CPU emission only."""
import importlib
import os

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


@triton.jit
def _vec_accumulate(X, OUT, n_tiles, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_tiles):
        acc = acc + tl.load(X + i * BLOCK + offs)
    tl.store(OUT + offs, acc)


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
    o = be.parse_options({"num_warps": 4})
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(t, o, be.get_codegen_implementation(o),
                      be.get_module_map(), ctx)
    meta = {}
    mod = be.make_ttir(mod, meta, o)
    mod = be.make_ttgir(mod, meta, o)
    return M.emit_msl(mod, meta, o)


_SIG = {"X": "*fp32", "OUT": "*fp32", "n_tiles": "i32"}


def test_vec_accumulate_no_unknown():
    on = _emit(_vec_accumulate, _SIG, dict(BLOCK=256), mept=True)
    assert "UNKNOWN_" not in on, on


def teardown_module(module):
    os.environ.pop("TRITON_MSL_MEPT", None)
    os.environ.pop("TRITON_MSL_FORCE_PYTHON", None)
