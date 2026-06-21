# Safe matmul rr/rc autotuning (runtime-tuned + cached)

> **SUPERSEDED (2026-06-20).** This describes the GPU-timed + disk-cached autotuner we
> explored. We built it, then **measured it through the production dispatch: no real win**
> (−2% to +3%, within noise — the "+102%" below was a micro-benchmark artifact) and it
> introduced a silent-wrong (an N-contract miss). The genuinely-reachable win turned out
> to be a *different* mechanism — **fast-path enablement for unaligned M** — delivered
> **deterministically** (no GPU timing), so the autotuner was replaced by a deterministic,
> occupancy-gated tile selector (`triton_msl/autotuning/matmul_tuner.py`). This doc is
> kept to record the approach considered and why it was abandoned. The scoping data below
> is real; the conclusions it implied did not survive integrated measurement.

**Date:** 2026-06-20
**Status:** SUPERSEDED by the deterministic occupancy-gated selector (see note above).
**Audit item:** "no autotuning" (the `num_stages` half is resolved separately — measured no-op, now documented).

## Goal

Pick the fast-matmul register-blocking `(rr, rc)` per shape at dispatch time instead of the fixed `(4,4)`, chosen by GPU-timed benchmark on first sight of a shape and **disk-cached** so selection is stable across runs. Measured win: **up to +102% on small matmuls (512³), ~+3–6% mid, 0% large** (already near peak). Bonus: `(2,*)` candidates (tile_m=16) let the fast path serve `M%16==0` shapes that the `(4,4)`-only path (needs `M%32==0`) currently drops to the generic lowering.

## Why this is safe (vs the #3 inductor-autotuning silent-wrong)

Every `(rr,rc)` in the candidate set computes a **correct** matmul (the kernel is correct for any blocking meeting its size contract). So a noisy or even nondeterministic *selection* can only ever pick a slightly-slower config — **never a wrong result**. This is the opposite of the #3 case (where autotuning explored configs the MSL lowering didn't cover → silent-wrong). Selection is further stabilized by the disk cache (tune once per shape, reuse).

## Candidate set

`CANDIDATES = [(4,4), (2,4), (4,2), (2,8), (8,2), (4,8), (8,4)]` — register budget `rr*rc ≤ 32` accumulators (8×8 float fragments); all measured-correct. `(4,4)` is the default/fallback (the current behavior) and the deterministic tie-break winner.

A candidate is **valid for a shape** iff `M % (8*rr) == 0` (the existing fast-path gate already requires `N%32==0` and `K%8==0`, which are config-independent).

## Architecture

1. **Descriptor** (`_maybe_fast_matmul_descriptor`, `_lowerer_templates.py`): instead of returning one baked `(4,4)` MSL, return enough to generate variants at dispatch — carry `msl_dtype` and `msl_out` alongside the existing `(m_idx,n_idx,k_idx)`. Keep a `(4,4)` MSL as the default so the non-autotuned path is unchanged.
2. **Selector** (new, e.g. `triton_msl/autotuning/matmul_tuner.py`): `best_rrrc(msl_dtype, msl_out, M, N, K, runtime) -> (rr,rc)`.
   - shape-key = `(msl_dtype, msl_out, M, N, K)`.
   - In-memory + disk cache (reuse `MetalAutotuner`'s cache_dir + JSON pattern): `{key: [rr,rc]}`.
   - Cache miss → **tune**: for each valid candidate, `runtime.get_library(make_simdgroup_matmul_kernel_fast(...))` (compiled+cached by source) and time it (warmup + median wall-clock with `torch.mps.synchronize`, on the real A/B/C buffers); pick fastest; deterministic tie-break = lowest index in CANDIDATES (so `(4,4)` wins ties). Persist.
   - Opt-out: `TRITON_MSL_MATMUL_AUTOTUNE=0` → always `(4,4)`.
3. **Dispatch** (`driver.py`, the fast_matmul block ~line 618): compute `(rr,rc) = best_rrrc(...)`, generate/`get_library` that variant, dispatch with `tile_m=8*rr, tile_n=32*rc, n_groups=ceil(M/tile_m)*ceil(N/tile_n)`. On ANY exception in the tuned path → fall back to the fixed `(4,4)` descriptor MSL (then generic) — never fail a dispatch.

## Correctness

All candidates are correct by construction + covered by the existing matmul tests across configs; no per-dispatch correctness check needed. The differential anchor is parity vs `torch @`.

## Tests

- **Parity:** autotuned matmul == `torch @` across shapes that select different configs (512, 1024, 2048, non-square, and an `M%16==0, M%32!=0` shape that only `(2,*)` serves).
- **Selector unit:** `best_rrrc` returns a valid candidate, caches it (2nd call hits cache, no re-tune), tie-breaks to `(4,4)`, and only ever returns size-contract-valid configs.
- **Opt-out:** `TRITON_MSL_MATMUL_AUTOTUNE=0` → always `(4,4)`.
- **Fallback:** a forced selector exception → dispatch still succeeds (fixed path).
- **Full-suite regression** + a before/after perf sanity (512³ should improve, 2048³ unchanged).

## Risks

- **First-call tuning latency** (~7 compiles+timings once per shape-key, amortized by the disk cache; opt-out exists). For one-shot shapes the tuning cost isn't recovered — acceptable (and `torch.compile` reuses shapes).
- **Timing noise** picks a near-optimal (still correct) config; the disk cache makes it stable across runs.

## Non-goals

Re-enabling inductor's pointwise/reduction autotuning (correctly disabled — the #3 silent-wrong); tuning anything where configs aren't all-correct.
