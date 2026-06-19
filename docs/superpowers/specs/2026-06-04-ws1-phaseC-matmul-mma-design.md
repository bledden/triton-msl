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

## C.2 inline-path integrity excavation (2026-06-07)

Bringing the matmul work to real `@triton.jit` kernels (the inline dot paths)
surfaced two REACHABLE silent-wrong bugs in `_lower_k_loop_dot_inline` (the
pid-tiled large-matmul path) — both latent because the test corpus only used
BLOCK_N=32, fp32-output K-loop matmuls:

1. **BLOCK_N>32 → only 32 columns computed.** The kernel emitted output at
   `col_base + sgitg*8` (4 simdgroups x 8 = 32 cols) regardless of BLOCK_N. Any
   tiled matmul with BLOCK_N=64/128 (the *standard* Triton tiling) silently
   dropped columns 32+. Confirmed: BLOCK_N=64 maxdiff 0.324 on cols 32-63.
   Fixed with column register-blocking (each simdgroup owns BLOCK_N/4 cols;
   a-fragment reused across col blocks). Commit 6012445.

2. **fp16-output raced on shared `tg_out`.** The non-float-output convert path
   stored every simdgroup's accumulator to the same `tg_out + t*64` slot (no
   per-simdgroup column offset) → race → all columns corrupted (maxdiff 0.337
   even at BLOCK_N=32). Fixed with full-tile staging (each accumulator to its
   own slot). Commit 2cb58bd.

Geometry that the 8x8-simdgroup tiling can't represent now REFUSES loudly
(MetalNonRecoverableError) instead of computing wrong numbers: BLOCK_N%32!=0,
BLOCK_M%8!=0, BLOCK_K%8!=0 (the last two are unreachable via normal kernels —
tl.arange forces power-of-2 block dims — but guarded defensively), and
non-float output whose full-tile staging would exceed the 32 KiB threadgroup
limit.

Method note: a stale `~/.cache/triton_msl` (keyed by AST, not codegen
version) initially masked the fix during development — every codegen-change
verification must clear it. Each fix was gated on a FRESH-cache full test_core
sweep at 4326 passed / 0 failed.

The MLX-parity direct-load + register-blocking config (13.76 TFLOP/s) is
shipped to the harness/standalone path; the inline path is now correct for all
representable geometries and genuine-fp16, but does not yet use direct loads
(it keeps the boundary-safe staged structure). Direct-load for inline remains
#153.

## C.2 inline-path: direct-load perf + full correctness (2026-06-07, cont.)

Direct-load fast path added to `_lower_k_loop_dot_inline`: a FULL float-output
tile with K%8==0 loads simdgroup fragments straight from device A/B (no staging,
no barriers), like the standalone kernel. Real @triton.jit fp16 matmul (2048^3,
wall-clock vs MLX 12.7): best config 4.70 -> 8.10 TFLOP/s (0.37x -> 0.64x MLX).
Not full parity — the staged tg_A/tg_B are allocated even on the direct path,
capping occupancy; closing that needs a two-kernel split (host picks direct vs
staged by runtime alignment). That is the remaining #153 perf item.

The excavation found FOUR reachable correctness bugs in this one path, all
latent because the corpus used only aligned, BLOCK_N=32, fp32-output K-loop
matmuls:
  1. BLOCK_N>32 computed only 32 columns (6012445).
  2. fp16-output raced on a shared tg_out slot (2cb58bd).
  3. partial-N float store wrote unmasked -> overflow columns wrapped into the
     next row's in-bounds data (fe8e5a6).
  4. BLOCK_N<32 (e.g. 16) over-wrote a <32-wide tile; was a false-pass
     (cf472f0).
The path now: column-register-blocks any BLOCK_N%8 (idle simdgroups for small
tiles, guarded only when not /4); direct-loads aligned float tiles; masked
per-simdgroup store for partial/edge/half; refuses BLOCK_M/K%8, BLOCK_N%8, and
>32KiB threadgroup configs. Genuine fp16 throughout. Each step gated on a
FRESH-cache (~/.cache/triton_msl) full test_core sweep at 4326/0.

## C.2 fourth path + a fifth silent-wrong (2026-06-07, #154)

Genuine fp16 applied to `_lower_matmul_softmax_template` (the fused
matmul→row-softmax path): half INPUT fragments + float accumulator; the matmul
result tg_C stays float for the softmax epilogue. bf16/other stay on the float
path.

Trying to exercise it exposed a FIFTH reachable silent-wrong: a simple
matmul→softmax kernel routed to `_detect_simple_dot` (checked first in
`lower()`), which emits a BARE matmul and silently drops the softmax — output
== A@B, row sums != 1 (confirmed: fp32 maxdiff vs softmax 2.05, matches A@B
exactly). _detect_simple_dot matches any load→dot→store and ignores the
epilogue. Fix: check `_detect_matmul_softmax` BEFORE `_detect_simple_dot` —
matmul_softmax only matches the full dot→max→sub→exp→sum→div pattern, so pure
matmuls still fall through to simple_dot. After the fix both fp32 and fp16
matmul→softmax are exact (maxdiff 0.0000, row sums = 1), and the fp16 path is
genuine (simdgroup_half8x8).

Broader latent concern tracked (#157): _detect_simple_dot also drops NON-softmax
epilogues (e.g. matmul+bias) the same way; needs a general "dot result must feed
only the store" check.

Genuine fp16 is now true on ALL FOUR matmul paths (standalone template, simple
dot inline, K-loop inline, matmul→softmax). Five reachable silent-wrongs found
and fixed across the inline matmul lowering this session.

## C.2 sixth silent-wrong: simple_dot dropped ALL epilogues (2026-06-07, #157)

The #154 softmax-routing fix exposed the general case: `_detect_simple_dot`
matches any load->dot->store and emits a bare matmul, dropping ANY value-changing
epilogue on the dot result — not just softmax. Confirmed: matmul*3+1 and
matmul->relu both returned bare A@B. (matmul+softmax was the special case already
routed to matmul_softmax.)

No path computes a matmul-sized dot + arbitrary elementwise epilogue correctly:
simple_dot drops it; the per-thread generic lowerer is wrong at 16/32/64 (a dot
that large exceeds the cooperative cap — the original reason matmul_softmax was
written). Fix: _detect_simple_dot now traces the dot result's consumers; only
layout changes / output dtype casts are value-preserving passthroughs to the
store, and any other consumer (arith/math/reduce/broadcast) raises
MetalNonRecoverableError — a loud refusal instead of silent wrong numbers.
matmul_softmax is checked first, so the supported fused-softmax form is
unaffected. Users hit the refusal with a clear message (split the epilogue into
a separate kernel). General fused epilogues remain a possible future feature,
not an integrity gap.

Six reachable silent-wrong/false-pass bugs found and fixed across the inline
matmul lowering this session (BLOCK_N>32, fp16-output race, partial-N store
wrap, BLOCK_N<32 false-pass, softmax-drop routing, epilogue-drop).

## C.2 #156: inline matmul occupancy — low-risk 8-deep staging

Profiling refined the inline-vs-MLX gap: at GPU-bound sizes the inline path was
~0.84x MLX (not the wall-clock 0.64x). Root cause confirmed by the standalone
comparison — the standalone fast kernel (SAME 16 accumulators / register
pressure, but NO threadgroup memory) hits 14.28 TFLOP/s @ 4096 vs the inline
path's 11-12; the differentiator is the static threadgroup allocation, which
Metal reserves and which caps occupancy even on the FAST direct branch (the
branch never touches tg, but the kernel still declares it).

A shrink-tg experiment quantified it (4096^3): 11 KB tg -> 11.18 TFLOP/s;
~0 tg -> 13.09; standalone 0 tg -> 14.28. So most of the gap is recoverable by
shrinking the static footprint — no launch-path change.

Fix (low risk): the edge-only staged path now stages STAGE_DEPTH=8 deep per K
step instead of BLOCK_K deep. The matmul is identical (it accumulates over the
full K either way); tg drops from BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N to
BLOCK_M*8 + 8*BLOCK_N (~11 KB -> ~3.5 KB). Result: inline 4096^3 11.18 ->
12.68 TFLOP/s (+13%, ~0.87x MLX), correctness preserved (partial tiles /
fp16-output / small BLOCK_N all green), test_core 4326/0.

Residual to full parity (~0.87x -> ~0.98x) is the remaining 3.5 KB tg, only
fully removable by zero-tg-for-aligned (dynamic threadgroup length or a
two-kernel split selected in the driver by runtime alignment) — a high-risk
launch-path change for ~13% more. Deferred unless that last bit is needed.

## C.2 #159: inline matmul to MLX parity — two-kernel split (after dynamic-tg dead end)

The residual to parity (0.87x -> ~1.0x) is the ~3.5KB static tg capping the
direct path's occupancy. Two attempts:

1. DYNAMIC THREADGROUP MEMORY (reverted). Bind the staging as one host-sized
   [[threadgroup(0)]] buffer; launcher sets length 0 for fully-aligned float
   dispatches (which take the direct path and never touch tg). Correct, but NO
   perf gain (12.70 vs 12.68): Metal/AGX caps occupancy by a kernel's
   compile-time tg usage (the staged path's accesses), not the per-dispatch
   host length. Setting length 0 doesn't free occupancy when the kernel CODE
   still contains the staged path. Reverted (no benefit, launcher risk).

2. TWO-KERNEL SPLIT (kept). Emit a SEPARATE pure-direct kernel `<name>__mmdirect`
   whose CODE has zero threadgroup memory (no staged path, no full-tile guard —
   the launcher only dispatches it when aligned). Plumbing: lowerer emits both +
   a dispatch descriptor (block_m/n, M/N/K arg indices); compiler threads it as
   kernel_metadata[6]; load_binary resolves the direct pipeline and stashes it
   keyed by id(primary); the launcher reads M/N/K from flat_args, and for a
   fully-aligned float matmul dispatches the direct kernel (same grid), else the
   staged kernel. Safety: any uncertainty (size mismatch, non-aligned, read
   error) falls back to the staged kernel — never wrong.

Result (fp16, thermally-throttled bench so absolute TFLOP/s low; RATIO is the
signal): 4096^3 reaches ~1.03x MLX (parity+), up from 0.87x. The dynamic-tg
result PROVED only a separate no-tg kernel recovers the occupancy — confirmed.
Correctness: aligned -> direct kernel, partial/odd-K/half -> staged kernel, both
verified (test_matmul_block_n) + full test_core sweep.

## Benchmark honesty correction (audit #164, 2026-06-08)

An external (MLX-team-lens) audit found the perf claims rested on a flawed
comparison. Corrected:

1. ASYMMETRIC TIMER (the load-bearing flaw). The harness timed OURS by GPU-only
   time (GPUEndTime-GPUStartTime) but MLX by wall-clock (perf_counter+
   synchronize). Comparing our kernel-only time to MLX's full wall-clock made us
   look faster than we are: e.g. matmul_2048_fp16 reported 0.90x ("beats MLX")
   but is ~1.04x (≈4% SLOWER) when both are timed wall-clock. FIXED: _time_dispatch
   now also records wall-clock and the MLX ratio is wall/wall; GPU-only TFLOP/s is
   kept as the kernel-throughput / roofline metric, and a gpu_only ratio is
   reported but labelled not-apples-to-apples.

2. HONEST NUMBERS (symmetric wall-clock both sides, M4 Max; thermal-sensitive,
   so a BAND not a point):
   - Standalone fast kernel (make_simdgroup_matmul_kernel_fast): ≈ MLX parity,
     0.95–1.05x wall-clock across 1024–4096 (fp16) and 1.00x at 2048 fp32.
   - Inline @triton.jit two-kernel-split path: ~0.96x at 2048, ~0.82–0.87x at
     4096 (ours 11.9–12.8 TFLOP/s vs MLX ~14.6) — competitive but below MLX on
     large GEMM, not parity.
   - The #159 "1.03x parity+" was a THROTTLED-run artifact (MLX read 12.47 that
     run vs its real ~14.6); it does not hold on a thermally-stable machine.

3. ROOFLINE. "% of roof" (esp. the fp16 89%/50% figures) is soft: the compute
   roof uses a guessed ~1.4 GHz clock and a flat 2x-fp16 ALU assumption that does
   NOT reflect simdgroup-matrix (MMA) throughput. Do not quote it as precise.

4. WHAT STANDS: "genuine fp16" is verified — half inputs + fp32 accumulation,
   matching MLX/MPS semantics to ~0 error (distinguishable from fp16-accumulate).
   The two-kernel split is a correct, safe occupancy fix (falls back to staged on
   any misalignment). The honest one-line claim: "competitive with MLX — standalone
   GEMM ≈parity, inline @triton.jit ~0.85–0.96x — measured wall-clock both sides;
   genuine fp16 (fp32-accumulate)."
