"""MEPT M2: a data-dependent scf.for carrying a hoisted multi-element value
emits the register-array form (no UNKNOWN_) under flag-ON. CPU emission only
(no GPU launch) — the fast signal for the eligibility extension. GPU numerical
correctness lives in tests/test_mept_m2_bug2_gpu.py.
"""
import importlib
import os

import pytest
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


@triton.jit
def _sum_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)            # hoisted outside the runtime loop
    total = 0.0
    for i in range(n_tiles):
        idx = i * BLOCK + offs
        v = tl.load(X + idx, mask=idx < N, other=0.0)
        total += tl.sum(v)
    tl.store(OUT, total)


def _emit(fn, sig, cst, mept):
    os.environ["TRITON_METAL_FORCE_PYTHON"] = "1"
    os.environ["TRITON_METAL_MEPT"] = "1" if mept else "0"
    import triton_metal.codegen.generic_lowerer as G
    import triton_metal.codegen.msl_emitter as M
    importlib.reload(G)
    importlib.reload(M)
    from triton_metal.backend.compiler import MetalBackend
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


_SIG = {"X": "*fp32", "OUT": "*fp32", "N": "i32", "n_tiles": "i32"}


def test_sum_in_loop_block256_emits_array_no_unknown():
    on = _emit(_sum_in_loop, _SIG, dict(BLOCK=256), mept=True)
    assert "UNKNOWN_" not in on, (
        "hoisted arange/other still unresolved inside the loop:\n%s" % on)


def teardown_module(module):
    os.environ.pop("TRITON_METAL_MEPT", None)
    os.environ.pop("TRITON_METAL_FORCE_PYTHON", None)
