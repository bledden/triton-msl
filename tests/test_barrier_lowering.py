"""ttg.barrier must emit a real threadgroup_barrier, not be silently dropped.

Triton renamed the barrier op to `ttg.barrier` (was `tt.debug_barrier`); the
lowerer only knew the old spelling, so `tl.debug_barrier()` compiled to MSL with
NO barrier — racy cross-SIMD-group kernels with no error (downstream tridec bug
1, 2026-06-10). This pins the new spelling emits the barrier.
"""
import os
import pytest

import triton  # noqa: F401


@pytest.fixture
def emit(monkeypatch):
    # monkeypatch.setenv auto-restores at teardown — a bare os.environ[...]=...
    # here leaked FORCE_PYTHON=1 into later test files (e.g. test_cpp_backend),
    # which routes their C++ kernels to Python; harmless flag-off but turns into
    # a MEPT-array refusal under TRITON_METAL_MEPT=1.
    monkeypatch.setenv("TRITON_METAL_FORCE_PYTHON", "1")
    import triton, triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.msl_emitter import emit_msl

    def _emit(fn, sig, cst):
        t = GPUTarget("metal", "apple-m4", 32)
        be = MetalBackend(t); o = be.parse_options({})
        src = ASTSource(fn=fn, signature=sig, constexprs=cst)
        ctx = ir.context(); ir.load_dialects(ctx)
        mod = src.make_ir(t, o, be.get_codegen_implementation(o),
                          be.get_module_map(), ctx)
        meta = {}
        mod = be.make_ttir(mod, meta, o); mod = be.make_ttgir(mod, meta, o)
        assert "ttg.barrier" in str(mod), "expected ttg.barrier in TTGIR"
        return emit_msl(mod, meta, o)
    return _emit


def test_debug_barrier_emits_threadgroup_barrier(emit):
    import triton, triton.language as tl

    @triton.jit
    def kbar(X, O, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(X + i)
        tl.debug_barrier()
        tl.store(O + i, v)

    msl = emit(kbar, {"X": "*fp32", "O": "*fp32"}, dict(N=128))
    assert "threadgroup_barrier" in msl


import torch  # noqa: E402


@pytest.mark.skipif(
    not __import__("Metal").MTLCreateSystemDefaultDevice(),
    reason="Metal device needed")
def test_barrier_orders_cross_lane_gather_deterministic():
    """The barrier must make a write->gather across SIMD groups (BLOCK=128 =
    4 groups) correct and deterministic; without it the kernel is racy."""
    import numpy as np
    import triton, triton.language as tl

    @triton.jit
    def race(S, O, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(S + i, i.to(tl.float32))
        tl.debug_barrier()
        tl.store(O + i, tl.load(S + ((i + 1) % N)))

    N = 128
    ref = ((np.arange(N) + 1) % N).astype(np.float32)
    outs = []
    for _ in range(5):
        s = torch.zeros(N); o = torch.zeros(N)
        race[(1,)](s, o, N=N)
        outs.append(o.numpy().copy())
    assert all(np.array_equal(o, ref) for o in outs)
    assert all(np.array_equal(o, outs[0]) for o in outs)
