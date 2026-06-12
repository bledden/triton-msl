# Decision: implementation language + IR target (long-term) — 2026-06-11

> Researched adversarially (5 angles × verify). Confidence: HIGH on the decision,
> MEDIUM on rationale (two popular sub-claims are wrong; the AGX blocker is
> undiagnosed, not proven-unfixable).

## Decision
**Primary/production: Python emitter → MSL source text → `xcrun metal` → metallib.**
Keep the C++ MLIR → LLVM-IR/AIR plugin **opt-in/experimental** (`TRITON_METAL_USE_CPP=1`),
**frozen, not deleted**. The destination is Python+MSL.

## The real variable: IR target, not language
- The kernel runs on the AGX GPU at the same speed regardless of what emitted the
  text. Python vs C++ vs Rust is orthogonal to kernel speed.
- What is NOT orthogonal is the IR handed to Apple. **MSL source** goes through
  Apple's hardened public front-end (`xcrun metal`). **LLVM-IR/AIR** goes through
  the under-exercised `metal -c -x ir` path — where our AGX `Code=3` internal
  error fires. Same compiler, different front-end; the fragile one is `-x ir`.
- You could write the emitter in C++ and STILL emit MSL (the Halide model). C++
  buys nothing toward the AIR target.

## Unanimous precedent (language-invariant, all stop at MSL source)
tinygrad (Python, MetalRenderer) ships MSL as production; IREE (MLIR, like Triton)
SPIRV-Cross → MSL; Halide (an LLVM compiler) source-to-source CodeGen_Metal_Dev;
wgpu/Naga (Rust) → MSL; PyTorch TorchInductor's native Mac path → MSL codegen.
Every serious Apple-GPU compiler emits MSL source, whatever its impl language.

## Corrections to the prior framing (intellectual honesty)
1. **"MSL is immune / Python just works" — FALSE.** The AGX backend JIT also runs
   at `newComputePipelineState` for MSL (MoltenVK #2363, wgpu #4817 hit "internal
   error" via MSL too). MSL is more *workaroundable* (source-level, caught at
   xcrun-compile), not immune. It carries a real toolchain-drift tax (MLX #3337:
   Apple moved `vec`/dropped a bf16 include on macOS 26) → needs CI vs current+next xcrun.
2. **"AGX blocker = shelve forever" — OVERSTATED.** It is UNDIAGNOSED, not proven
   Apple-unfixable. The `-opaque-pointers` workaround is already in and it still
   fires, so not a one-liner — but this error genus is sometimes emitter-side /
   register-pressure (MoltenVK precedent). Right move: time-boxed minimal repro +
   bisect + Apple Feedback, then decide. Don't chase endlessly; don't claim
   permanence.
3. **"Python can't go upstream" — REFINED.** Triton's hardware-BACKEND contract IS
   a Python contract (BaseBackend; NVIDIA/AMD/Intel are Python subclasses; the
   final stage may return metallib bytes). C++/MLIR is only for optional pass
   PLUGINS, and PR 8401 (2025-11) exposes out-of-tree passes. Realistic target:
   "advertised out-of-tree" plugin (Intel-XPU precedent). Only IN-TREE merge is
   foreclosed (would need a C++ rewrite AND the AGX fix).

## Perf note
Orthogonality holds, with one caveat: MSL *emission quality* still matters (AGX
-O2 is IR-form-sensitive). The one AIR-unique primitive (`simdgroup_async_copy`,
device→threadgroup) is NOT M5-gated (M1+), but the TMA pattern using it was
prototyped and SHELVED as a net slowdown on M4. Re-benchmark on M5 before
concluding. Don't chase AIR for perf on M1–M4 (no measured upside).

## Next investments (ranked)
1. Harden the MSL path: CI compiling emitted MSL vs current+next xcrun toolchains.
2. Upstream hygiene: per-version `add_stages`/Language shim + CI matrix; verify
   entry-point discovery as an out-of-tree plugin.
3. Time-boxed AGX diagnosis: minimal standalone `Code=3` repro + bisect the AIR
   metadata invariant + Apple Feedback. Convert "undiagnosed" → "diagnosed."
4. Perf: measure the ~0.15ms/launch buffer-copy; pursue zero-copy MPS/MLX dispatch.
   Keep the MEPT/register-array spine moving (retires detector debt; unblocks tridec).
5. Freeze (don't delete) the C++ plugin. Revisit AIR-as-target only if the repro
   proves the bug is ours+cheap, M5 async-copy is a real win, Apple ships an open
   IR→metallib path, or triton-ext becomes the sole sanctioned Apple path in-window.
