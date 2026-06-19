# Phase 1 (C++ port) — definitive blocker: AGX-compiler internal errors (2026-06-11)

## Conclusion
The C++ MLIR → LLVM IR → metallib path cannot be made default-on without an
unbounded, impractical debugging effort. It triggers **AGXMetalG16X Code=3
"Compiler encountered an internal error"** at pipeline-state creation on real
corpus kernels (e.g. test_bin_op[1-int32-float16-+], failing on kernel_scalar_rhs:
int32 vector + a Python-float constexpr). C++ stays OPT-IN (TRITON_MSL_USE_CPP=1).

## Evidence (this session)
- Default (MSL) path: int32-float16 binary ops pass 24/24.
- USE_CPP=1: the same test fails, in isolation (1.1s, not a wedge), with AGX Code=3.
- The crash is at load_binary (pipeline-state creation) on the produced metallib;
  no C++ exception is logged (TRITON_MSL_CPP_TRACE=1), so the C++ path produces
  a metallib that the AGX *compiler* rejects internally — below our code.
- NOT REPRODUCIBLE STANDALONE: structurally identical kernels (same dtypes,
  num_warps=4, output dtype, C++ route confirmed via _has_complex_ops=False) all
  RUN CORRECTLY on the C++ path. The trigger is some exact value/metadata detail
  in the full test harness that could not be isolated after extensive effort.

## Why this is a stop, not a fix
- The error is an Apple-compiler internal error (Code=3), not malformed IR we can
  see — we can only AVOID emitting whatever triggers it, by trial.
- Non-reproducible-standalone + a slow debug loop (reproduce → edit C++ → rebuild
  → run full corpus ~15min) makes each unknown AGX-trigger a multi-hour cycle.
- The dtype gate (route f16/i8/i16/i1 → MSL) is already this kind of avoidance;
  closing the remaining crashes is open-ended whack-a-mole against AGX internals.

## Decision
Phase 1 "C++ primary" is shelved. C++ remains an opt-in experimental accelerator
behind the dtype gate. The register-array spine (Phase 2, the thing tridec needs
for BLOCK>=256, and the >1024 ceiling) and 1.0 work proceed on the validated
Python/MSL path. This empirically confirms the pre-1.0 audit + the Phase-1
flip-revert: the working backend, the actual downstream user, and the tractable
path all live on Python/MSL.
