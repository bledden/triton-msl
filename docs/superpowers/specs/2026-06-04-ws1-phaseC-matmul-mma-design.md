# WS1 Phase C — matmul MMA: fix the fp16 lie, then close the gap (design)

> First phase of WS1 (the perf/gap trunk), chosen by the measure-first data:
> matmul is the biggest gap. Approach (user-approved): **fix the MSL template
> first, reassess C++ after.** Overriding emphasis from the user:
> **"fix the lie and make sure things are genuine first-off."**

## The measured problem (WS0/C6 harness, M4 Max, stable large sizes)

| matmul | our TFLOP/s | % of roof | vs MLX |
|---|---|---|---|
| 2048 fp32 simd | 7.00 | 49% | 1.60× slower |
| 2048 fp16 simd | 7.00 | 24.5% | 1.75× slower |
| 4096 fp16 simd | 7.01 | 24.5% | 2.10× slower |

Two root causes, both in the MSL templates (read from
`_lowerer_templates.py` + `_msl_templates.py`):

1. **The fp16 path is a lie.** The "fp16" template stages inputs through a
   `threadgroup float` buffer, upcasts each `half` to `float` on load, and
   uses `simdgroup_float8x8` fragments. The MMA runs in **fp32 regardless of
   dtype** — `simdgroup_half8x8` is never used. That is why fp16 and fp32
   produce the *identical* 7.00 TFLOP/s. Apple's ~2× fp16 matrix throughput
   is completely unused.
2. **Barrier-every-8-K with tiny tiles → sync-bound plateau.** The K-loop
   steps by 8; each step re-stages a 32×8 + 8×32 tile, hits a
   `threadgroup_barrier`, then issues only **4** 8×8×8 MMAs. For K=2048 that
   is 256 barriers feeding 4 MMAs each — the MMA units idle on staging+sync,
   capping throughput at ~25–49% of roof, not scaling with size.

**Duplication:** the same naive pattern is replicated across THREE paths —
`make_simdgroup_matmul_kernel` (harness proxy), `_lower_simple_dot_inline`,
and `_lower_k_loop_dot_inline` (what real `@triton.jit` matmuls use) — plus a
scalar `make_matmul_kernel` fallback that uses no MMA at all. Fixing the bug
in one place must not leave the other two lying.

## Goals (in priority order)

1. **C.1 — Make fp16 genuine (the lie), proven.** fp16 matmul must actually
   use `simdgroup_half8x8` input fragments. "Genuine" is not a claim — it is
   verified by: (a) the generated MSL *contains* `simdgroup_half8x8` and does
   *not* upcast halves to float before the MMA; (b) numerical correctness vs
   numpy; (c) the harness showing fp16 throughput **genuinely exceeding** fp32
   (the 7.00==7.00 tie is broken upward). Land this FIRST.
2. **C.2 — Close the perf gap.** Deeper K-tiling + larger output tiles +
   register blocking so each barrier feeds many MMAs; drive the MLX gap from
   1.6–2.1× toward parity, measured per variant in the harness.
3. **Throughout — stay correct and honest.** `test_core` dot tests stay
   **4326/0**; integrity guards (constexpr-dim refusal, etc.) untouched; no
   path left claiming a precision it doesn't deliver.

Non-goal for this phase: the C++ generic-MMA rebuild (reassessed *after* we
see how far the MSL fix gets — per the approved approach). Double-buffering is
deferred unless the harness shows we are still staging-bound after C.2.

## Components

### A single shared MMA-tile emitter

`_emit_simdgroup_mma_tile(...)` — one well-tested core that emits the
threadgroup-staging + simdgroup-MMA inner loop, parameterized by:
- `dtype` ("fp32" | "fp16") → selects `simdgroup_float8x8` vs
  `simdgroup_half8x8` *input* fragments; the accumulator is **always**
  `simdgroup_float8x8` (fp32 accumulation for precision, even for fp16
  inputs — the half×half→float MMA form).
- tile dims (output rows/cols per threadgroup) and `K_depth` (how deep a
  K-tile is staged per barrier).

The three existing paths (`make_simdgroup_matmul_kernel`,
`_lower_simple_dot_inline`, `_lower_k_loop_dot_inline`) call this core instead
of each carrying its own copy. This both fixes the bug once and pays down the
duplication the maintainer review flagged. The scalar `make_matmul_kernel`
fallback stays as the boundary/edge fallback but is no longer the primary
compute path.

**Interface contract.** Inputs: A/B/C device buffers, M/N/K, dtype, tile
config. Output: a complete, correct MSL kernel that handles M/N/K not a
multiple of the tile (boundary masking), and for fp16 uses half input
fragments + float accumulation. Consumers don't need to know the staging
internals; the tile config is the only knob.

## Implementation phases

### C.1 — Fix the fp16 lie (genuine fp16), FIRST and PROVEN
This is the priority deliverable — the lie is fixed before anything else,
including consolidation and perf tuning.

Step 1 (de-risk): prove the half×half→float MMA form compiles and is correct
on the Metal target, in a minimal standalone kernel. This validates the
central technical risk *before* building on it. If unsupported, fall back
(see Risks) and document honestly.

Step 2 (fix + consolidate together): build the shared
`_emit_simdgroup_mma_tile` core **with genuine fp16** — fp16 inputs load into
`simdgroup_half8x8` fragments and MMA into a `simdgroup_float8x8` accumulator,
no float upcast before the MMA — and route the three duplicated paths
(`make_simdgroup_matmul_kernel`, `_lower_simple_dot_inline`,
`_lower_k_loop_dot_inline`) through it. Consolidation is the *vehicle* for
fixing the lie once, not a separate step that delays it. fp32 behavior stays
equivalent (gated by 4326/0 + unchanged fp32 harness numbers).

Genuineness gate (all required before C.1 is "done"):
- **MSL inspection test:** generated fp16 kernel contains `simdgroup_half8x8`
  and contains no `float(A[` / `float(B[` upcast in the staging path.
- **Compile test:** the half×half→float MMA form compiles on the Metal target
  (validated early — this is the main technical risk).
- **Correctness:** fp16 matmul matches numpy (fp16 tol) across sizes incl.
  non-32-multiples.
- **Perf proof the lie is gone:** harness shows fp16 TFLOP/s **>** fp32
  TFLOP/s (the 7.00==7.00 tie broken upward). This is the user's
  "make it genuine" bar, measured.
- `test_core` dot tests stay 4326/0.

### C.2 — Close the gap (deeper tiling), measured per variant
Stage K-deep tiles (start 32) per barrier, inner-loop the 8×8 MMAs; widen the
output tile + register-block for reuse. Tune depth/size empirically with the
harness as the oracle. Success: MLX gap closed substantially (target toward
parity), fp32 and fp16 both scaling with size (no flat ~7 TFLOP/s ceiling).

### C.3 — Double-buffer (only if still staging-bound)
Add threadgroup double-buffering to overlap staging with compute, *only if*
the harness shows C.2 is still sync/staging-limited. YAGNI otherwise.

## Validation

- **Correctness (every phase):** numpy/PyTorch numerical match for fp32 and
  fp16 across sizes including non-tile-multiples; upstream `test_core` dot
  tests stay **4326/0**; full sweep unaffected.
- **Genuineness (C.1):** the MSL-inspection + fp16>fp32-throughput gates
  above. The point is to make "fp16" *true*, provably.
- **Performance (C.2):** `python benchmarks/hw_harness.py matmul_*` re-run;
  per-variant TFLOP/s, % of roof, and MLX ratio recorded; compared against
  the committed baseline. A variant is kept only if the harness shows it
  helps.
- **Integrity:** the matmul refusal guards (constexpr-dim, etc.) and the
  refusal catalog stay intact; no kernel silently returns wrong numbers.

## Risks

- **`simdgroup_half8x8` mixed-precision MMA support** on the Metal target is
  the central unknown — validated by a compile+correctness test at the very
  start of C.1 (before building on it). If half×half→float isn't supported,
  fall back to half×half→half with periodic float accumulation, or document
  the limitation honestly.
- **Consolidation regressions:** the three paths may have subtle differences;
  C.0 must keep 4326/0 and unchanged harness numbers before C.1 changes
  behavior.
- **Boundary correctness** for M/N/K not multiples of the (now larger) tile.
- **Register pressure** from bigger tiles / deeper K could drop occupancy —
  watch the harness reflection occupancy hint; tune.

## Success criteria

1. fp16 matmul genuinely uses `simdgroup_half8x8` (MSL-verified) and the
   harness shows fp16 throughput meaningfully above fp32 — **the lie is
   fixed and proven.**
2. The MLX gap (1.6–2.1×) is closed substantially, with throughput scaling
   with size (no flat ~7 TFLOP/s ceiling).
3. One shared MMA emitter; the three duplicated paths retired into it.
4. `test_core` dot tests stay 4326/0; integrity guards intact.
5. Whether the C++ generic-MMA rebuild is still warranted is decided *after*,
   from the post-fix harness numbers.

## C.1 result (2026-06-04) — lie fixed and genuine; perf is sync-bound (latent)

The fp16 lie is fixed in the standalone template, proven genuine:
- MSL contains `simdgroup_half8x8` input fragments, no `float(A[`/`float(B[`
  upcast, half threadgroup staging, `simdgroup_float8x8` accumulator.
- Numerically correct end-to-end: full 64×64×128 fp16 template vs numpy
  (`test_genuine_fp16_full_template_is_numerically_correct`), plus the de-risk
  K=8 and K=256 cases.
- fp32 path unchanged; existing emitter matmul/simdgroup tests stay green.

But the **numeric floor (fp16 ≥15% over fp32, ≥8.5 TFLOP/s) is NOT met** —
harness after the fix:

| matmul | TFLOP/s | vs MLX |
|---|---|---|
| 2048 fp32 | 6.59 | 1.69× |
| 2048 fp16 | 6.73 | 1.83× |
| 4096 fp16 | 6.88 | 2.13× |

fp16 and fp32 are the same *absolute* throughput (~6.7 vs ~6.6). This is the
honest, useful finding: **the kernel is sync/staging-bound (root cause #2),
not MMA-bound**, so doubling MMA precision throughput does not move the
needle — the MMA units idle on the barrier-every-8-K. The fp16 benefit is
real but **latent**: it materializes only once C.2 (deeper K-tiling) makes the
kernel MMA-bound. C.1 delivered correctness + honesty (no more lie); C.2
delivers the speed, and is where fp16 should finally pull ahead of fp32.

## C.2 result (2026-06-05) — MLX parity reached; C++ rebuild NOT needed

Measure-and-keep sweep on the 2048^3 fp16 matmul (M4 Max), each variant
validated vs numpy first:

| approach | TFLOP/s | note |
|---|---|---|
| staged, k_depth=8 (genuine fp16) | ~6.7 | sync-bound (the C.1 plateau) |
| staged, k_depth=32 | 7.73 | barrier amortized 4->16 MMAs; **committed** |
| staged, 64x64 tile | 7.12 | REGRESS — occupancy (8 acc + smem) |
| staged, double-buffered | 7.45 | REGRESS — occupancy (2x smem) |
| **direct device load (no staging/barriers)** | 8.62 | removing the barrier beats amortizing it |
| direct + 2-col register blocking | 10.34 | a-fragment reuse |
| **direct + 4x4 register blocking (16 acc)** | **13.76** | **MLX parity (MLX ~12.9 @ 2048 fp16)** |
| direct + 8x4 (32 acc) | 11.09 | too many registers -> occupancy drop |

Findings:
- The staged-template levers (bigger tiles, double-buffering) REGRESS on
  occupancy. The win is the opposite: **drop threadgroup staging entirely**
  (`simdgroup_load` directly from device A/B, relying on the GPU cache for
  reuse) and **maximize fragment reuse via register blocking** (each simdgroup
  computes a 32x32 output block = 16 accumulators; each a-/b-fragment feeds 4
  MMAs).
- The 4x4 register-blocked direct-load kernel hits **13.76 TFLOP/s = MLX
  parity**, ~2x the original ~7 plateau.
- **C++ decision (spec success criterion #5): the MSL path reaches MLX
  parity, so the C++ generic-MMA rebuild is NOT warranted for matmul
  performance.** The remaining work is productionization, not a rewrite.

Productionization constraint (next task): the direct-load kernel requires
tile alignment (M % (8*RR), N % (32*RC), K % 8) — `simdgroup_load` from device
does not mask, so edge tiles read OOB. Shipping needs boundary handling (a
masked-staging fallback for partial tiles, or refuse-when-unaligned + keep the
boundary-safe staged template for those), and the same treatment applied to
the inline JIT dot paths so real @triton.jit matmuls get it — gated on the
full 4326/0 sweep. Deferred to a focused pass rather than rushed.

## C.2 productionized (2026-06-07) — MLX-parity matmul shipped (harness path)

`make_simdgroup_matmul_kernel_fast` (direct-load + 4x4 register blocking)
is in the harness, boundary-correct, MLX parity/better:

| matmul | TFLOP/s | % of roof | vs MLX |
|---|---|---|---|
| 2048 fp16 | 13.75 | 48% | 0.90x (beats MLX) |
| 4096 fp16 | 14.28 | 50% | 1.02x (parity) |
| 2048 fp32 | 12.75 | **89%** | 0.90x (beats MLX) |

- Dispatch contract: 128 threads, n_groups = ceil(M/(8*rr)) * ceil(N/(32*rc)).
- Size contract: M%(8*rr)==0, N%32==0 (partial col tiles guarded per-simdgroup),
  K%8==0. 18 correctness tests vs numpy across boundary cases (32x32 w/ 3
  simdgroups guarded; N=96/160 %32-not-%128; rectangular; fp16+fp32).
- Fixed a pre-existing harness bug (fp16 C is FLOAT output; specs under-allocated
  it as half).

Remaining (the inline JIT paths): real @triton.jit `tl.dot` matmuls use
`_lower_simple_dot_inline`/`_lower_k_loop_dot_inline`, whose dispatch grid is
tied to the Triton program model (not a free choice), plus transpose/batch
support and the constexpr-dim refusal. Applying the fast kernel there needs a
careful dispatch-grid mapping + the full 4326/0 sweep — a focused follow-on,
not rushed at session end. The harness path proves the technique and locks in
the perf; the inline integration ships it to user kernels.
