"""Per-family op coverage for the C++ MLIR path (Phase 1 spec).

Routing contract:
- Default-on (no env vars): kernels route through C++ only when every op
  in their TTGIR belongs to a family in ``ENABLED``. Phase 1 ships only
  the ``elementwise`` family — it is validated by the differential
  harness (tests/test_diff_cpp_python.py) including multi-dim grids.
- ``TRITON_METAL_USE_CPP=1`` (legacy explicit opt-in): the full C++
  surface — union of ALL families — is admitted, preserving the
  pre-Phase-1 behavior that tests/test_cpp_backend.py exercises
  (reductions, dot, flash attention). The dot path is only validated
  for the single-tile grids those tests use; it is NOT default-on
  (a 2D-grid tt.dot kernel produces wrong output: pid_n tiles never
  written — see test_integration.py::test_triton_jit_matmul).
- ``TRITON_METAL_FORCE_PYTHON=1`` bypasses C++ entirely (compiler.py).
"""
import os

FAMILIES = {
    # Validated default-on: differential harness + project suite.
    "elementwise": {
        # -- Triton ops (custom patterns in ElementwiseOpToLLVM.cpp) --
        'tt.get_program_id', 'tt.get_num_programs',
        'tt.make_range', 'tt.splat', 'tt.broadcast',
        'tt.expand_dims', 'tt.reshape',
        'tt.addptr', 'tt.load', 'tt.store',
        'tt.func', 'tt.return',
        'tt.extern_elementwise',
        # -- Arith ops (standard arith-to-LLVM + custom constant) --
        'arith.constant',
        'arith.addf', 'arith.addi',
        'arith.subf', 'arith.subi',
        'arith.mulf', 'arith.muli',
        'arith.divf', 'arith.divsi', 'arith.divui',
        'arith.remf', 'arith.remsi', 'arith.remui',
        'arith.negf',
        'arith.andi', 'arith.ori', 'arith.xori',
        'arith.shli', 'arith.shrsi', 'arith.shrui',
        'arith.cmpi', 'arith.cmpf',
        'arith.select',
        'arith.sitofp', 'arith.fptosi',
        'arith.uitofp', 'arith.fptoui',
        'arith.extf', 'arith.truncf',
        'arith.extsi', 'arith.extui', 'arith.trunci',
        'arith.bitcast', 'arith.index_cast',
        'arith.maxnumf', 'arith.minnumf',
        'arith.maximumf', 'arith.minimumf',
        'arith.maxsi', 'arith.minsi',
        'arith.maxui', 'arith.minui',
        # -- Math ops (standard math-to-LLVM) --
        'math.exp', 'math.exp2',
        'math.log', 'math.log2', 'math.log10',
        'math.sqrt', 'math.rsqrt',
        'math.absf', 'math.abs',
        'math.sin', 'math.cos', 'math.tan',
        'math.tanh', 'math.erf',
        'math.ceil', 'math.floor', 'math.round',
        'math.powf', 'math.fma',
        'math.copysign',
        # -- Control flow (standard cf-to-LLVM) --
        'cf.br', 'cf.cond_br',
        # -- SCF (structured control flow) --
        'scf.for', 'scf.yield', 'scf.if',
        # Layout conversions are passthrough in the per-thread model
        # (_strip_ttg_annotations) when no tt.dot is present.
        'ttg.convert_layout',
    },
    # Opt-in only (TRITON_METAL_USE_CPP=1): SIMD-then-crossSIMD scan
    # over threadgroup memory; validated by test_cpp_backend.py.
    "reduction": {
        'tt.reduce', 'tt.reduce.return',
        'ttg.local_alloc', 'ttg.local_load', 'ttg.local_store',
        'ttg.local_dealloc',
    },
    # Opt-in only (TRITON_METAL_USE_CPP=1): simdgroup-matrix dot;
    # validated only for single-tile grids (test_cpp_backend.py, FA).
    # Known-broken for multi-tile 2D grids — do not enable by default.
    "dot": {
        'tt.dot',
        'ttg.local_alloc', 'ttg.local_load', 'ttg.local_store',
        'ttg.local_dealloc',
        'ttg.memdesc_subview', 'ttg.memdesc_trans',
        'ttg.async_copy_global_to_local', 'ttg.async_wait',
    },
}

# Families safe to route through C++ without explicit opt-in.
ENABLED = {"elementwise"}


def enabled_ops():
    """Union of allowed ops across the families currently admitted.

    Default: only ``ENABLED`` families (Phase 1: elementwise).
    With ``TRITON_METAL_USE_CPP=1`` the legacy opt-in surface (all
    families) is preserved for the explicit C++ test suite.
    """
    if os.environ.get("TRITON_METAL_USE_CPP", "") == "1":
        families = set(FAMILIES)
    else:
        families = ENABLED
    out = set()
    for fam in families:
        out |= FAMILIES[fam]
    return out


# Dtypes the C++ AIR pipeline miscompiles or AGX rejects today (Phase 1 burn-in:
# int8+float16 mixes crash AGXMetalG16X with "internal error" at pipeline
# creation, repeated crashes wedge the corpus run). Kernels whose TTGIR mentions
# any of these dtypes route to Python until the C++ lowering is fixed per dtype.
UNSAFE_DTYPE_PAT = ("f16", "i8", "i16", "i1,", "i1>")  # coarse: over-routing safe


def cpp_safe_text(ttgir_text):
    return not any(p in ttgir_text for p in UNSAFE_DTYPE_PAT)
