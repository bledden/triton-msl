# Phase-5 readiness audit — dual lens (2026-06-16)

> Adversarial pre-1.0 audit of `triton-metal` from two expert perspectives: the team that
> owns OpenAI Triton (NVIDIA/Triton-core) and Apple's GPU/MLX team. Grounded in current
> (2026-06) published reality + the project's own claims + the codebase at the merged tip.
> Severity: **[BLOCKER]** (fix before a credible public 1.0) / **[MAJOR]** / **[MINOR]**.

## Current-reality grounding (web, 2026-06; sources at end)
- **M4 Max (40-core) peak: 18.4 TFLOP/s fp32, 36.9 fp16, 546 GB/s.**
- **No official OpenAI-Triton Apple/Metal backend exists in 2026.** Third-party backends are
  the precedent (Intel `intel-xpu-backend-for-triton`, AMD ROCm, `triton-cpu`); Metal is an
  acknowledged community gap. `triton-metal` is a genuine **first-mover** on the `TRITON_EXT`
  plugin path (PR #9783) — no official competitor.
- Apple GEMM tops out ~60-63% of peak in practice (MPS/MLX). On-device LLM inference is
  **bandwidth-gated, not flops-gated** — so the memory-bound zero-copy win targets the right
  bottleneck.

## 🟦 NVIDIA / Triton-core lens — "would the Triton team sign off?"
**Verdict: a credible, unusually-honest 0.x — not a Triton-team 1.0 yet. What's missing is
honesty/reproducibility/coverage, not correctness.**
- **Strength:** the never-silent-wrong integrity contract is better-articulated than most
  third-party backends; fp32 matmul (~55-62% of peak) is competitive with MLX/MPS; the
  memory-bound zero-copy win targets the right bottleneck for Apple inference.
- **[BLOCKER] Inconsistent/stale conformance numbers across docs.** README **5,335 passed /
  0 failed / ~4,007 skip** (Triton 3.7.0); CHANGELOG **4,326 / 5,016 skip** ("as of
  2026-05-30", and it still references the *abandoned* C++ MLIR→LLVM FlashAttention path +
  507 project tests); CONTRIBUTING **~4,279**. A public 1.0 needs ONE dated, reproducible
  figure across all docs, with the `--device cpu` methodology stated.
- **[BLOCKER] The "never silent-wrong" headline needs a systematic refusal-coverage test.**
  This session alone found+fixed multiple real silent-wrongs (tridec in-loop-reduce; n>1
  store/atomic under-coverage; the CHANGELOG even documents `test_constexpr_if_return`
  emitting `pid+0` garbage that a weak assert masked). The refusal catalog
  (`_refuse_unsafe_unsupported_ops`) is empirically unproven-complete; back it with a test
  that enumerates unsupported ops/shapes and asserts each refuses loudly.
- **[MAJOR]** fp16 matmul ≈ fp32 (~33% of fp16 peak) — float accumulation forgoes the 2×
  fp16 matrix rate; FlashAttention only HEAD_DIM=32 (real models use 64/128); no fp8/
  `dot_scaled` (correctly refused); pure-Python pattern-matched lowering is structurally
  gap-prone (integrity contract is the mitigation).

## 🟧 MLX / Apple-GPU lens — "does it do the hardware justice?"
**Verdict: kernels are credible and useful, but not MLX-class — and one public claim
oversells it.**
- **Strength:** the fast matmul template is genuinely good Metal (direct `simdgroup_load`,
  register blocking, no threadgroup staging); UMA zero-copy via `compile_shader` is the right
  Apple-Silicon move; float accumulation is numerically correct; the fp16-output cast
  epilogue is sound (post-K-loop, measured 12.3 TFLOP/s).
- **[BLOCKER → FIXED 2026-06-16] Overstated perf claim.** The
  `make_simdgroup_matmul_kernel_fast` docstring claimed "~13.8 TFLOP/s MLX parity"; measured
  is ~8-12 and only ~33% of fp16 peak — well below MLX's hand-tuned kernels. Corrected in
  this audit to measured %-of-peak with the no-MLX-parity caveat.
- **[MAJOR]** memory-bound 64% vs an ~75-80% ceiling (vectorized loads — a documented
  follow-up); bf16 *input* unsupported by the fast template; the generic/staged fallback
  (serial `tg_st` epilogue, ~34 barriers) is poor Metal (but correct). (The "fp16 2× left
  on the table" claim was retracted — see the EXPLORED+DECLINED note in the synthesis: the
  matrix unit isn't 2× for half accumulation, so fp16≈fp32 is near-optimal here.)
- **Honest positioning:** not beating MLX on raw kernel perf (won't — it's hand-tuned by the
  HW team). The value prop is *the Triton programming model on Apple GPUs*, which MLX doesn't
  offer. Lead with that; don't claim parity.

## 🎯 Synthesis — true publish-blockers (both lenses converge on HONESTY)
1. **[BLOCKER] Reconcile + pin the conformance number** — one reproducible, dated figure
   across README/CHANGELOG/CONTRIBUTING; refresh the stale CHANGELOG (numbers + remove the
   abandoned-C++-FA reference); state the `--device cpu` method.
2. **[BLOCKER → FIXED] Overstated perf claims** — the "~13.8 TFLOP/s MLX parity" docstring is
   corrected to measured numbers + honest %-of-peak.
3. **[BLOCKER → FIXED 2026-06-16] Systematic refusal-coverage test** — added
   `tests/test_refusal_coverage.py`: a consolidated guard asserting representative
   integrity-prescan-catalog patterns refuse loudly. First case (verified PASSED): a
   K-loop matmul tiling the output across programs with M/N baked as constexpr (the
   `test_dot_mulbroadcasted` silent-wrong class) raises `MetalNonRecoverableError` rather
   than emitting wrong output. Joins the existing per-feature refusal tests
   (`test_nover_store_refusal`, `test_atomic_nover_refusal`, `test_inloop_reduce_coverage`);
   extend with more catalog entries (dot_scaled, rank≥3 trans, rank≥2 cat/join) over time.

**MAJOR / roadmap (not 1.0 blockers):** large-head_dim FlashAttention (>64); fp8/`dot_scaled`
(refused — fine).

**published op/dtype support matrix — DONE 2026-06-16** (`docs/SUPPORTED_OPS.md`, linked from
README).

**FlashAttention head_dim — ADDRESSED 2026-06-16 (and uncovered + closed a silent-wrong).**
The audit's "FA only at head_dim 32" was based on an imprecise README — head_dim **64 was
already supported + tested** (causal + non-causal). Probing head_dim=128 revealed a genuine
**silent-wrong**: at BLOCK_M=16/8 the attention lowering compiled + ran + produced garbage
(max error ~1000) with no error (only BLOCK=32 failed loudly, by accident, via OutOfResources).
Fixed: an FA-pattern prescan guard in `GenericLowerer.lower()` (≥2 dots + exp + max + a dot
tile dim > 64) now refuses head_dim > 64 loudly (`MetalNonRecoverableError`); README corrected
(32 **and** 64), `SUPPORTED_OPS.md` updated, test added (`test_head_dim_over_64_refuses`),
full project suite 720 passed. True large-head_dim FA support (tiled threadgroup memory) is
the remaining roadmap item.

**~~vectorized loads (memory-BW ceiling 64% → ~75-80%)~~ — EXPLORED + DECLINED 2026-06-16.**
The "75-80%" headroom doesn't exist on this hardware. Probe (M4 Max, vector_add @ 16M):
at a tuned grid-stride config, `float4` gives only **~3-5%** over scalar (scalar 293-304,
float4 308-314 GB/s — both ~56-58% of 546), and BOTH are *below* the current production MEPT
path (347 GB/s, 64%). The 546 GB/s is a raw LPDDR5X figure; the practical memory-bound
ceiling for elementwise compute is ~58-64% (torch.add itself ≈ 58%), and the current path
is already at/above it. Vectorized loads add ~3-5% at best for a real codegen change with an
alignment/contiguity silent-wrong surface — poor ROI. Like the fp16-2× MAJOR, this rested on
a headline-peak comparison the execution units can't reach; the kernels are already near the
practical ceiling.

**~~fp16 half-accumulate opt-in (2× fp16)~~ — EXPLORED + DECLINED 2026-06-16.** This MAJOR
rested on a flawed peak-ratio inference. Empirical probe (M4 Max, 2048³ fp16): half
accumulators (`simdgroup_half8x8`) give only **~6% speedup (12.6 vs 11.9 TFLOP/s) at a real
accuracy cost (relerr 1.75e-2 vs 0.0)**. Apple's simdgroup-matrix unit runs at ~the same
rate regardless of float-vs-half accumulator — its throughput is gated by the matrix unit,
not accumulator precision — so the 36.9 TFLOP/s "fp16 peak" is a vector-ALU FMA figure the
matrix path cannot reach by changing accumulators. **Correction to the audit:** fp16 matmul
≈ fp32 matmul (~60% of the *fp32* matrix ceiling) is the expected, near-optimal behavior;
the current float accumulation is the right choice (same speed, full precision). Not worth a
speed/accuracy knob for ~6%.

**Through-line:** the *engineering* is largely 1.0-ready (integrity contract + fp32/memory
perf are genuinely good); the **public claims weren't yet matched to measured reality** —
the single biggest risk when an NVIDIA- or Apple-caliber reader looks. Fixing #1-#3 closes it.

## Sources
- M4 Max specs (18.4 fp32 / 36.9 fp16 TFLOP/s, 546 GB/s): https://flopper.io/gpu/apple-m4-max/spec-sheet
- No official Triton Metal backend / third-party precedent: https://aiwiki.ai/wiki/openai_triton , https://github.com/intel/intel-xpu-backend-for-triton , https://ai-blog.it/blog/infrastrutture-software-assenti-ma-necessarie-per-l-ai-su-apple-silicon-il-linguaggio-triton
- Triton 3.x feature surface: https://www.spheron.network/blog/openai-triton-kernel-gpu-cloud-2026/
- Apple GEMM ~60-63% of peak / bandwidth-gated inference: https://arxiv.org/pdf/2502.05317 , https://petronellatech.com/blog/mlx-exo-unlocking-apple-silicon-s-ml-performance/
