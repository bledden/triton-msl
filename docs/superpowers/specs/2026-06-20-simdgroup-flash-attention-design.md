# simdgroup-MMA FlashAttention integration

**Date:** 2026-06-20
**Status:** design (spike validated; pre-implementation)
**Spike:** `/tmp/fa_v3.py` (+ `fa_fp16.py`); findings in memory `simdgroup_fa_spike.md`.

## Goal

Replace the marketed head_dim=128 FlashAttention's scalar-FMA GEMMs (matrix units
idle — the dual-lens audit's #1 perf finding) with a `simdgroup_matrix` MMA kernel.
The spike proved **~7.4× (fp32) / ~8.5× (fp16) over the scalar template, ~8.2/9.6
TFLOP/s, correct to torch within 1e-3 (fp32) / ~4e-5 (fp16)** — ~70–80% of the matmul
template's peak. This is a real, honest win over our own baseline; it is **not**
MFA-competitive in absolute terms, and the docs/claims must say so.

Hard constraint: the routed FlashAttention path is **prime-directive** (never
silent-wrong). The integration must be correct for all shapes the scalar template
handles, or fall back to it — never silently mis-compute.

## Non-goals

- Beating Apple metal-flash-attention / MLX (would need a ground-up rewrite: larger
  tiles + register management; block-level pipelining was tried in the spike and was
  a wash — this is the structural ceiling).
- head_dim ≠ 128, BLOCK ≠ 32×(BN) — those keep using the scalar template.
- Changing the FA detector / ABI-reconciliation machinery (reused as-is).

## Kernel: `make_flash_attention_kernel_simdgroup`

New sibling of `make_flash_attention_kernel_tiled` in `triton_msl/codegen/_msl_templates.py`.
Same buffer ABI (Q,K,V,Out @ 0..3; 16 strides; Z,H,N_CTX; scale baked) so the routed
`_lower_flash_attention_template` binds it identically. Validated structure (spike v3+fp16):

- **Register-resident O accumulator** (no 16KB threadgroup acc). The online-softmax
  per-row α-rescale and final 1/l-normalize are done as **diagonal-matrix MMAs**
  (`diag(α)@O`, `diag(1/l)@O`) — sidesteps the opaque simdgroup thread-element layout.
- **QK^T** via `simdgroup_load(..., transpose=true)` on K (no separate transpose pass).
- **Q staged once** into threadgroup (invariant across the kv-loop); the staging loop
  **zero-pads OOB q-rows**, which also handles the partial last q-block.
- **P@V**: V tile loaded once per (col-tile, k-substep) and reused across all 4
  row-blocks (kills a 4× redundant device load); all V tiles **prefetched** up front
  (memory-level parallelism — the "async double-buffering" effect on hardware without
  cp.async).
- **Config:** q-tile BLOCK_M=32 (= the detected `block_m`), internal **kv-tile
  BN=64** (the kernel's OWN choice — independent of the user kernel's `block_n`,
  since the template fully replaces the kv-loop), head_dim=128, 256 threads / 8 SIMD
  groups.
- **Variants:** fp32 and fp16 (half Q/K/V/P, fp32 accumulate, half Out via the cast
  epilogue — the proven `make_simdgroup_matmul_kernel_fast` pattern); causal and
  non-causal (causal = score-mask `kv_row <= q_row` before exp, identical to the
  scalar template's `score_guard`/`prob_guard`).

## Correctness: boundaries, strides, fallback

The fast path is **device-direct** simdgroup_load (the source of the speedup); it
cannot per-row-mask, and `simdgroup_load` needs a unit innermost stride. Two limits,
each handled so nothing is ever silently wrong:

1. **Non-contiguous head-dim** (innermost stride ≠ 1): detectable at **compile time**
   — the detector reports `q_sk/k_sk/v_sk/o_sk` as `c1` (folded to 1 = contiguous).
   Gate: simd only when all four are `c1`; otherwise emit the **scalar template**
   (handles general strides). Rare — standard attention tensors are contiguous.

2. **Runtime N_CTX boundaries** (N_CTX not a multiple of BN/BM):
   - Partial **q-block** (`q_start + BM > N_CTX`): handled by Q-staging zero-pad +
     the `q_row < N_CTX` guard on the final store. No extra path.
   - Partial **kv-block** (only the last: `kv_start + BN > N_CTX`): in-kernel branch
     — full blocks take the fast device-direct MMA; the one partial block takes a
     **staged+masked** path (Dc-chunk stage K/V into threadgroup with `kv_row<N_CTX`
     zero-pad, MMA from threadgroup; softmax masks `kv_row>=N_CTX` to −∞). Runs at
     most once per kernel, so the aligned bulk stays fast.

**Scalar template stays in the tree** as (a) the differential test **oracle** and
(b) the fallback for non-contiguous / head_dim≠128 / block≠32.

## Routing

`_lower_flash_attention_template(info)` (generic_lowerer.py): when the SAME validated
gate as today — `head_dim==128 ∧ block_m==32 ∧ block_n==32 ∧ out_dtype∈{f32,f16}` —
PLUS `all innermost strides c1` (the new contiguity check) → emit
`make_flash_attention_kernel_simdgroup` (passing causal/scale from `info`; the kernel
uses its own internal kv-tile BN=64); else the existing
`make_flash_attention_kernel_tiled`. The detector already refuses on any ABI/stride
ambiguity (unchanged).

## Tests (never-silent-wrong gate)

- **Differential simd == scalar** (the primary regression guard) across the matrix:
  {fp32, fp16} × {causal, non-causal} × {aligned N_CTX, **unaligned** e.g. 100, 192} ×
  several (Z,H,N_CTX) shapes — both within fp tol of the torch reference.
- **Fallback coverage**: non-contiguous strides → scalar template emitted (assert the
  emitted MSL is the scalar one); head_dim 64 unaffected.
- **Routed `@triton.jit` end-to-end**: the real FA kernel lowers + runs + matches eager
  (extends the existing FA tiled tests).
- **Budget/compile**: assert the simd kernel compiles within 32KB threadgroup + that
  register-O + the partial-kv staged path coexist (the spike hit a 32KB wall when
  double-buffering — validate the final layout).

## Risks

- **Register pressure**: register-O (8 O frags/group) + the staged partial path's
  temporaries. Spike validated register-O alone compiles; the added partial path must
  be checked. Mitigation: chunk the partial-block staging (Dc=32) to bound both
  threadgroup (≤32KB) and registers.
- **The partial-kv staged path is new code** (spike was aligned-only) — highest-risk
  correctness area; the unaligned-N_CTX differential tests are the guard.
- **fp16 half-out cast epilogue** (spike used float out) — small, mirrors the matmul
  template.

## Rollout

Additive: new template + a routing branch; scalar path untouched as fallback/oracle.
Default-on once the differential gate is green (no env flag needed — it's correctness-
gated by the diff tests, like the compile_shader ON==OFF gate).
