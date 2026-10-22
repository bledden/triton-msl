"""Per-family op coverage for the C++ MLIR path (Phase 1 spec).

Routing contract:
- C++ is OFF by default; the attempted elementwise default flip was reverted.
  ENABLED is the family-table selection, not a switch enabling C++ compilation.
- TRITON_MSL_USE_CPP=1 is an unaudited development opt-in. The known-broken
  dot route is refused even under that opt-in; the independently generated MSL
  route may compute it instead, with an explicit disposition/warning.
- Runtime grid dimensions are unavailable at compilation. The old single-tile
  observation cannot prove that a produced C++ binary will only get one tile.
  Until that contract is implemented/audited, no C++ dot capability is claimed.
- ``TRITON_MSL_FORCE_PYTHON=1`` bypasses C++ entirely (compiler.py).
- Load volatility is preserved as LLVM volatile, including on masked loads.
  NVIDIA cache/eviction hints on loads and stores are accepted but ignored;
  they do not select an Apple cache policy or establish memory ordering.
"""

import os
import re


KNOWN_UNSAFE_OPS = frozenset({"tt.dot"})


def cpp_refusal_reason(ttgir_text):
    # Also covers quoted generic MLIR spelling. This is deliberately a
    # conservative known-broken-op guard, NOT a proof of all other C++ semantics.
    if re.search(r"(?<![\w.])tt\.dot(?![\w.])", ttgir_text):
        return "tt.dot has known-broken multi-tile/grid and loop-carry semantics; a single-program launch is unproved"
    # The C++ annotation/block-size rewrites and operation census understand
    # custom assembly, not generic quoted operation syntax. An empty census is
    # not evidence that every operation is allowed. Decline this entire spelling
    # envelope until it is structurally parsed, rather than guessing its operands.
    # Quoted module attribute keys/values do not have an operation's following '('.
    # Restrict the match to an operation position. Named debug locations such
    # as loc("x_ptr"(#loc)) also contain a quoted name followed by '(' but
    # are not operations; rejecting them would de-route ordinary vector add.
    if re.search(r'(?m)(?:^|[={])\s*"[^"\n]+"\s*\(', ttgir_text):
        return "generic quoted operation spelling is outside the proved C++ text-rewrite envelope"
    # Native run_to_llvm asserts on a scalar load. Keep this at the COMMON
    # boundary so direct make_llir callers cannot bypass the ordinary router.
    if re.search(r"tt\.load\b[^\n]*:\s*!tt\.ptr<", ttgir_text):
        return "scalar tt.load is unsupported by C++ native lowering; use the MSL path"
    return None


FAMILIES = {
    # Historical default-on candidate; actual compilation is opt-in only.
    "elementwise": {
        # -- Triton ops (custom patterns in ElementwiseOpToLLVM.cpp) --
        "tt.get_program_id",
        "tt.get_num_programs",
        "tt.make_range",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.func",
        "tt.return",
        "tt.extern_elementwise",
        # -- Arith ops (standard arith-to-LLVM + custom constant) --
        "arith.constant",
        "arith.addf",
        "arith.addi",
        "arith.subf",
        "arith.subi",
        "arith.mulf",
        "arith.muli",
        "arith.divf",
        "arith.divsi",
        "arith.divui",
        "arith.remf",
        "arith.remsi",
        "arith.remui",
        "arith.negf",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrsi",
        "arith.shrui",
        "arith.cmpi",
        "arith.cmpf",
        "arith.select",
        "arith.sitofp",
        "arith.fptosi",
        "arith.uitofp",
        "arith.fptoui",
        "arith.extf",
        "arith.truncf",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.bitcast",
        "arith.index_cast",
        "arith.maxnumf",
        "arith.minnumf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.maxsi",
        "arith.minsi",
        "arith.maxui",
        "arith.minui",
        # -- Math ops (standard math-to-LLVM) --
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.log10",
        "math.sqrt",
        "math.rsqrt",
        "math.absf",
        "math.abs",
        "math.sin",
        "math.cos",
        "math.tan",
        "math.tanh",
        "math.erf",
        "math.ceil",
        "math.floor",
        "math.round",
        "math.powf",
        "math.fma",
        "math.copysign",
        # -- Control flow (standard cf-to-LLVM) --
        "cf.br",
        "cf.cond_br",
        # -- SCF (structured control flow) --
        "scf.for",
        "scf.yield",
        "scf.if",
        # Layout conversions are passthrough in the per-thread model
        # (_strip_ttg_annotations) when no tt.dot is present.
        "ttg.convert_layout",
    },
    # Opt-in only (TRITON_MSL_USE_CPP=1): SIMD-then-crossSIMD scan
    # over threadgroup memory; validated by test_cpp_backend.py.
    "reduction": {
        "tt.reduce",
        "tt.reduce.return",
        "ttg.local_alloc",
        "ttg.local_load",
        "ttg.local_store",
        "ttg.local_dealloc",
    },
    # Historical simdgroup-dot coverage, retained for the recovery census.
    # tt.dot itself is excluded below even under opt-in until its launch and
    # loop-carry contracts are proved. Shared-memory ops also serve reductions.
    "dot": {
        "tt.dot",
        "ttg.local_alloc",
        "ttg.local_load",
        "ttg.local_store",
        "ttg.local_dealloc",
        "ttg.memdesc_subview",
        "ttg.memdesc_trans",
        "ttg.async_copy_global_to_local",
        "ttg.async_wait",
    },
}

# Historical default candidate; compiler.add_stages still requires explicit opt-in.
ENABLED = {"elementwise"}


def enabled_ops():
    """Union of allowed ops across the families currently admitted.

    Default: only ``ENABLED`` families (Phase 1: elementwise).
    With TRITON_MSL_USE_CPP=1 all historical families are considered, minus
    explicitly known-unsafe operations. This table is not a correctness proof.
    """
    if os.environ.get("TRITON_MSL_USE_CPP", "") == "1":
        families = set(FAMILIES)
    else:
        families = ENABLED
    out = set()
    for fam in families:
        out |= FAMILIES[fam]
    return out - KNOWN_UNSAFE_OPS


# Dtypes the C++ AIR pipeline miscompiles or AGX rejects today (Phase 1 burn-in:
# int8+float16 mixes crash AGXMetalG16X with "internal error" at pipeline
# creation, repeated crashes wedge the corpus run). Kernels whose semantic TTGIR
# mentions any of these dtypes route to Python until the C++ lowering is fixed.
# Quoted strings and comments are not semantic type evidence: source locations
# contain arbitrary paths (including worktrees named ``bf16``), and scanning the
# raw text made routing depend on the checkout directory.
UNSAFE_DTYPE_RE = re.compile(r"(?:(?<=x)|(?<![A-Za-z0-9_]))(?:bf16|f16|i1|i8|i16)(?![A-Za-z0-9_])")


def _without_mlir_strings_and_comments(ttgir_text):
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', ttgir_text)
    return re.sub(r"//[^\n]*", "", without_strings)


def cpp_safe_text(ttgir_text):
    return UNSAFE_DTYPE_RE.search(_without_mlir_strings_and_comments(ttgir_text)) is None
