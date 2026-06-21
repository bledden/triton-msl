# Safe matmul rr/rc autotuning — Implementation Plan

> **SUPERSEDED (2026-06-20).** This plan built a GPU-timed + disk-cached autotuner.
> Integrated measurement showed it gave no real win (within noise) and it caused a
> silent-wrong (N-contract); it was replaced by a deterministic, occupancy-gated tile
> selector that captures the genuine win (fast-path enablement for unaligned M) without
> GPU timing. Kept as a record of the approach explored. See the design doc's superseded
> note and `triton_msl/autotuning/matmul_tuner.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Pick the fast-matmul register-blocking `(rr,rc)` per shape at dispatch time (GPU-timed on first sight, disk-cached), instead of the fixed `(4,4)` — safe-by-construction (all configs correct), opt-outable. Measured win: up to +102% small, ~+3–6% mid, 0 large.

**Architecture:** A selector module (`matmul_tuner.py`) caches the best `(rr,rc)` per shape-key (in-memory + disk JSON); on a miss it GPU-times the size-contract-valid candidates via the compile_shader runtime and persists the winner. The fast-matmul descriptor carries `msl_dtype`/`msl_out` so the driver dispatch can build the chosen variant; on any error it falls back to the fixed `(4,4)` path.

**Tech Stack:** Python; `triton_msl.codegen._msl_templates.make_simdgroup_matmul_kernel_fast(dtype, rr, rc, out_dtype)`; `CompileShaderRuntime.get_library/dispatch`; `torch.mps` timing; pytest.

## Global Constraints

- Work in `/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread`. Clear caches before correctness/perf checks: `rm -rf ~/.cache/triton_msl ~/.triton/cache`.
- Prime directive: NEVER silent-wrong. SAFETY here = only ever consider `(rr,rc)` that (a) are in the validated CANDIDATES set and (b) satisfy the size contract `M % (8*rr) == 0` (plus the config-independent `N%32==0`, `K%8==0`). All such configs compute a correct matmul; selection only affects perf. On ANY error in the tuned path, fall back to the fixed `(4,4)` dispatch — never fail or mis-dispatch.
- Commits LOCAL only; do NOT push/PR without explicit confirmation. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- `CANDIDATES = [(4,4), (2,4), (4,2), (2,8), (8,2), (4,8), (8,4)]` (register budget rr*rc ≤ 32). `(4,4)` is the default + deterministic tie-break winner (lowest index).
- Opt-out env var: `TRITON_MSL_MATMUL_AUTOTUNE` (default "1"; "0" → always `(4,4)`).
- Dispatch math: `tile_m = 8*rr`, `tile_n = 32*rc`, `n_groups = ceil(M/tile_m) * ceil(N/tile_n)`, `threads = n_groups*128`, `group_size = 128`.

---

### Task 1: Selector module `matmul_tuner.py`

**Files:**
- Create: `triton_msl/autotuning/matmul_tuner.py`
- Test: `tests/test_matmul_tuner.py`

**Interfaces:**
- Produces: `CANDIDATES: list[tuple[int,int]]`; `valid_candidates(M,N,K) -> list[tuple]`; `best_rrrc(msl_dtype, msl_out, M, N, K, runtime, cache_dir=None) -> tuple[int,int]`. `runtime` is a `CompileShaderRuntime`-like object exposing `get_library(msl)` and `dispatch(lib, name, args, threads=, group_size=)`. `best_rrrc` returns a size-contract-valid `(rr,rc)`, GPU-times the valid candidates on a cache miss, persists the winner (per `(msl_dtype,msl_out,M,N,K)` key) to a JSON cache, and returns the cached winner on a hit without re-timing. `TRITON_MSL_MATMUL_AUTOTUNE=0` → return `(4,4)` (or, if `(4,4)` is invalid for the shape, the lowest-index valid candidate). If no candidate is valid, return `None`.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_matmul_tuner.py
"""Unit tests for the safe matmul rr/rc selector (no GPU needed for these)."""
import os
import pytest

from triton_msl.autotuning.matmul_tuner import CANDIDATES, valid_candidates, best_rrrc


def test_candidates_are_register_safe_and_include_default():
    assert (4, 4) == CANDIDATES[0]               # default + tie-break winner
    assert all(rr * rc <= 32 for rr, rc in CANDIDATES)


def test_valid_candidates_respects_size_contract():
    # M=2048 divisible by all 8*rr; N=2048%32==0; K=2048%8==0 -> all valid.
    assert set(valid_candidates(2048, 2048, 2048)) == set(CANDIDATES)
    # M=48 (=16*3) divisible by 8*rr for rr in {2} (16) but NOT rr in {4,8} (32,64).
    v = valid_candidates(48, 64, 64)
    assert all(48 % (8 * rr) == 0 for rr, rc in v)
    assert (4, 4) not in v and (2, 4) in v
    # N not %32 -> no candidate valid.
    assert valid_candidates(64, 40, 64) == []


def test_optout_returns_default(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_MATMUL_AUTOTUNE", "0")
    assert best_rrrc("fp32", "fp32", 2048, 2048, 2048, runtime=None) == (4, 4)
    # opt-out but (4,4) invalid -> lowest-index valid candidate, still no runtime use.
    assert best_rrrc("fp32", "fp32", 48, 64, 64, runtime=None) == (2, 4)


def test_no_valid_candidate_returns_none(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_MATMUL_AUTOTUNE", "0")
    assert best_rrrc("fp32", "fp32", 64, 40, 64, runtime=None) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_matmul_tuner.py -q -p no:cacheprovider`
Expected: FAIL — module/functions don't exist.

- [ ] **Step 3: Implement the selector**

```python
# triton_msl/autotuning/matmul_tuner.py
"""Safe, deterministic, cached selection of the fast-matmul register blocking
(rr, rc) per shape. All candidates compute a CORRECT matmul (the kernel is correct
for any blocking meeting its size contract), so selection only ever affects perf —
never correctness (unlike inductor's #3 autotuning silent-wrong). Selection is
GPU-timed once per shape-key and disk-cached for stable, reproducible choices."""
import json
import os
import statistics
import time

# Register budget rr*rc <= 32 (8x8 float accumulators). (4,4) first = default +
# deterministic tie-break winner.
CANDIDATES = [(4, 4), (2, 4), (4, 2), (2, 8), (8, 2), (4, 8), (8, 4)]

_MEM_CACHE = {}   # (msl_dtype, msl_out, M, N, K) -> (rr, rc)


def _autotune_on():
    return os.environ.get("TRITON_MSL_MATMUL_AUTOTUNE", "1") != "0"


def valid_candidates(M, N, K):
    """Candidates whose size contract this shape satisfies. N%32 and K%8 are
    config-independent gates; per-candidate it's M % (8*rr) == 0."""
    if N % 32 != 0 or K % 8 != 0:
        return []
    return [(rr, rc) for (rr, rc) in CANDIDATES if M % (8 * rr) == 0]


def _cache_dir(cache_dir):
    d = cache_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "triton_msl", "matmul_tuner")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(cache_dir, key):
    safe = "_".join(str(x) for x in key)
    return os.path.join(_cache_dir(cache_dir), f"{safe}.json")


def best_rrrc(msl_dtype, msl_out, M, N, K, runtime, cache_dir=None):
    """Best (rr,rc) for this shape: opt-out/default-aware, cached, GPU-timed on miss.
    Returns a size-contract-valid (rr,rc), or None if no candidate is valid."""
    valid = valid_candidates(M, N, K)
    if not valid:
        return None
    default = (4, 4) if (4, 4) in valid else valid[0]
    if not _autotune_on() or runtime is None:
        return default

    key = (msl_dtype, msl_out, M, N, K)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    path = _cache_path(cache_dir, key)
    if os.path.exists(path):
        try:
            rr, rc = json.load(open(path))["best"]
            if (rr, rc) in valid:
                _MEM_CACHE[key] = (rr, rc)
                return (rr, rc)
        except Exception:
            pass

    best = _tune(msl_dtype, msl_out, M, N, K, valid, runtime)
    if best is None:
        best = default
    _MEM_CACHE[key] = best
    try:
        json.dump({"best": list(best)}, open(path, "w"))
    except Exception:
        pass
    return best


def _tune(msl_dtype, msl_out, M, N, K, valid, runtime, warmup=8, reps=20):
    """GPU-time each valid candidate on real buffers; return the fastest (or None
    on any failure -> caller uses the default). Deterministic tie-break: CANDIDATES
    order (valid is built in that order; strict < keeps the earliest on ties)."""
    import math
    import torch
    from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast
    try:
        in_dt = torch.float16 if msl_dtype in ("fp16", "f16") else torch.float32
        out_dt = torch.float16 if msl_out in ("fp16", "f16") else torch.float32
        A = torch.randn(M, K, device="mps", dtype=in_dt)
        B = torch.randn(K, N, device="mps", dtype=in_dt)
        C = torch.empty(M, N, device="mps", dtype=out_dt)
    except Exception:
        return None
    best_cfg, best_ms = None, float("inf")
    for (rr, rc) in valid:
        try:
            lib = runtime.get_library(
                make_simdgroup_matmul_kernel_fast(msl_dtype, rr, rc, msl_out))
            tile_m, tile_n = 8 * rr, 32 * rc
            ng = math.ceil(M / tile_m) * math.ceil(N / tile_n)

            def run():
                runtime.dispatch(lib, "simdgroup_matmul_fast", [A, B, C, M, N, K],
                                 threads=ng * 128, group_size=128)
            for _ in range(warmup):
                run()
            torch.mps.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                run()
                torch.mps.synchronize()
                ts.append(time.perf_counter() - t0)
            ms = statistics.median(ts)
            if ms < best_ms:           # strict < -> earliest (CANDIDATES order) wins ties
                best_ms, best_cfg = ms, (rr, rc)
        except Exception:
            continue
    return best_cfg
```

- [ ] **Step 4: Run unit tests to verify pass**

Run: `PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_matmul_tuner.py -q -p no:cacheprovider`
Expected: PASS (4 tests). These don't touch the GPU (opt-out / pure-logic paths).

- [ ] **Step 5: Commit**

```bash
git add triton_msl/autotuning/matmul_tuner.py tests/test_matmul_tuner.py
git commit -m "feat(autotune): safe cached matmul rr/rc selector (all configs correct)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire the selector into the descriptor + driver dispatch

**Files:**
- Modify: `triton_msl/codegen/_lowerer_templates.py` (`_maybe_fast_matmul_descriptor`, ~line 2449-2453)
- Modify: `triton_msl/backend/driver.py` (the `fast_matmul` dispatch block, ~line 618-648)
- Test: `tests/test_matmul_autotune_dispatch.py`

**Interfaces:**
- Consumes: `best_rrrc`, `valid_candidates` (Task 1); `make_simdgroup_matmul_kernel_fast`.
- Produces: the `fast_matmul` descriptor tuple GAINS two trailing fields → `(fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n, msl_dtype, msl_out)` (back-compat: the driver unpacks defensively — if the 2 new fields are absent it uses the fixed path). Driver picks `(rr,rc)` via `best_rrrc` per shape and dispatches that variant; any exception → fixed `(4,4)` MSL.

- [ ] **Step 1: Write the failing test** (dispatch parity across shapes that select different configs)

```python
# tests/test_matmul_autotune_dispatch.py
"""End-to-end: a torch.compiled / @triton.jit matmul routed through the fast path
matches torch @ across shapes, with rr/rc autotuning on (default) and off."""
import os
import platform
import pytest
import torch

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()
         and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader")


def _mm(M, K, N):
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a_ptr, b_ptr, c_ptr, M, N, K,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0); pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps")
    c = torch.empty(M, N, device="mps")
    grid = (M // 32, N // 32)
    mm[grid](a, b, c, M, N, K, BM=32, BN=32, BK=32)
    torch.mps.synchronize()
    return c, a @ b


@requires_mps
@pytest.mark.parametrize("M,K,N", [(512, 512, 512), (1024, 1024, 1024), (2048, 512, 2048)])
def test_autotuned_matmul_matches_torch(M, K, N):
    os.environ.pop("TRITON_MSL_MATMUL_AUTOTUNE", None)   # default ON
    c, ref = _mm(M, K, N)
    assert (c - ref).abs().max().item() < 1e-1   # fp32 matmul over K, generous abs tol


@requires_mps
def test_optout_matches_torch():
    os.environ["TRITON_MSL_MATMUL_AUTOTUNE"] = "0"
    try:
        c, ref = _mm(1024, 1024, 1024)
        assert (c - ref).abs().max().item() < 1e-1
    finally:
        os.environ.pop("TRITON_MSL_MATMUL_AUTOTUNE", None)
```

- [ ] **Step 2: Run to verify it passes pre-change, then becomes the regression anchor**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache; PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_matmul_autotune_dispatch.py -q -p no:cacheprovider`
Expected: with the CURRENT fixed `(4,4)` code these already PASS (the fast path is correct at (4,4)). They are the regression anchor — they must STILL pass after wiring autotuning (now exercising non-(4,4) configs for 512/2048×512). Note (2048,512,2048): N=512%32==0, so it routes; 512³ exercises small-shape config selection.

- [ ] **Step 3: Extend the descriptor**

In `triton_msl/codegen/_lowerer_templates.py`, change the return of `_maybe_fast_matmul_descriptor` (currently `return (fast_msl, 3, 4, 5, 8 * rr, 32 * rc)`):

```python
        from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast
        rr = rc = 4
        fast_msl = make_simdgroup_matmul_kernel_fast(dtype=msl_dtype, rr=rr, rc=rc, out_dtype=msl_out)
        # (msl, m_idx, n_idx, k_idx, tile_m, tile_n, msl_dtype, msl_out). The last two
        # let the driver build alternative (rr,rc) variants for per-shape autotuning;
        # the baked (4,4) msl stays the default/fallback.
        return (fast_msl, 3, 4, 5, 8 * rr, 32 * rc, msl_dtype, msl_out)
```

- [ ] **Step 4: Wire selection into the driver dispatch**

In `triton_msl/backend/driver.py`, the fast_matmul block. Replace the unpack + dispatch (currently `fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n = fast_matmul` then the fixed dispatch) with a version that selects `(rr,rc)` per shape and dispatches that variant, falling back to the fixed path on any error:

```python
                    if (fast_matmul is not None and all_mps
                            and _os.environ.get("TRITON_MSL_FAST_MATMUL", "1") != "0"):
                        try:
                            fast_msl = fast_matmul[0]
                            m_idx, n_idx, k_idx = fast_matmul[1], fast_matmul[2], fast_matmul[3]
                            tile_m, tile_n = fast_matmul[4], fast_matmul[5]
                            # New trailing fields enable per-shape autotuning; absent -> fixed.
                            msl_dtype = fast_matmul[6] if len(fast_matmul) > 6 else None
                            msl_out = fast_matmul[7] if len(fast_matmul) > 7 else None
                        except (TypeError, ValueError, IndexError):
                            fast_msl = None
                        if fast_msl is not None and not _rt.is_unsupported(fast_msl):
                            try:
                                M = int(kargs[m_idx]); N = int(kargs[n_idx]); K = int(kargs[k_idx])
                                # Per-shape autotune (safe: all configs correct). Any
                                # failure -> fixed (4,4) tile_m/tile_n + fast_msl below.
                                sel_msl = fast_msl; sel_tm, sel_tn = tile_m, tile_n
                                if msl_dtype is not None and msl_out is not None:
                                    try:
                                        from triton_msl.autotuning.matmul_tuner import best_rrrc
                                        from triton_msl.codegen._msl_templates import (
                                            make_simdgroup_matmul_kernel_fast)
                                        rrrc = best_rrrc(msl_dtype, msl_out, M, N, K, _rt)
                                        if rrrc is not None and rrrc != (4, 4):
                                            rr, rc = rrrc
                                            sel_msl = make_simdgroup_matmul_kernel_fast(
                                                msl_dtype, rr, rc, msl_out)
                                            sel_tm, sel_tn = 8 * rr, 32 * rc
                                    except Exception:
                                        sel_msl = fast_msl; sel_tm, sel_tn = tile_m, tile_n
                                if (M > 0 and N > 0 and K > 0
                                        and M % sel_tm == 0 and N % 32 == 0 and K % 8 == 0):
                                    import math as _math
                                    n_groups = _math.ceil(M / sel_tm) * _math.ceil(N / sel_tn)
                                    lib = _rt.get_library(sel_msl)
                                    _rt.dispatch(lib, "simdgroup_matmul_fast", kargs[:6],
                                                 threads=n_groups * 128, group_size=128)
                                    if launch_exit_hook:
                                        launch_exit_hook(launch_metadata)
                                    return
                            except Exception:
                                try:
                                    _rt.mark_unsupported(fast_msl)
                                except Exception:
                                    pass
```

(Keep the surrounding indentation/context exactly as in the existing block; only the unpack + selection are added. The existing fixed-path guard `M % sel_tm == 0` now uses the SELECTED tile_m so a chosen (2,*) config that needs only M%16 dispatches correctly.)

- [ ] **Step 5: Run the dispatch tests + a focused matmul regression**

Run:
```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_matmul_autotune_dispatch.py tests/test_fast_matmul*.py tests/test_matmul*.py -q -p no:cacheprovider
```
Expected: PASS (parity holds with autotuning on AND off; existing matmul tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add triton_msl/codegen/_lowerer_templates.py triton_msl/backend/driver.py tests/test_matmul_autotune_dispatch.py
git commit -m "feat(autotune): per-shape rr/rc selection in the fast-matmul dispatch (fallback-safe)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Full-suite regression + measured before/after + honest docs

**Files:**
- Modify: `docs/SUPPORTED_OPS.md` (matmul row), `reports/perf_baseline.json` (note the autotuning)

**Interfaces:** Consumes Tasks 1–2.

- [ ] **Step 1: Full-suite regression**

Run:
```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache /var/folders/*/T/torchinductor_* 2>/dev/null
PYTHONPATH=$(pwd) python3.14 -m pytest tests/ -q -p no:cacheprovider --deselect "tests/test_fast_matmul_perf.py::test_fast_matmul_throughput[dtype1]"
```
Expected: green (the known transformer-convergence test is now robust; any other failure is a real regression to fix).

- [ ] **Step 2: Measure the SHIPPED integrated path, before/after (do NOT trust projections)**

Write a throwaway bench that drives the production dispatch (a `@triton.jit` matmul as in Task 2's `_mm`) at 512³ (should improve) and 2048³ (should be unchanged), timing with `torch.mps.synchronize` warmup+median, with `TRITON_MSL_MATMUL_AUTOTUNE` on vs `=0`. Record the real numbers.
Expected: 512³ measurably faster with autotuning ON; 2048³ within noise. If 512³ is NOT faster through the integrated path, the wiring isn't selecting the better config — debug Task 2 before claiming the win.

- [ ] **Step 3: Honest docs**

Update `docs/SUPPORTED_OPS.md` matmul row + a `reports/perf_baseline.json` note: "fast matmul autotunes register-blocking (rr,rc) per shape (safe: all configs correct; disk-cached; `TRITON_MSL_MATMUL_AUTOTUNE=0` to disable) — measured +<X>% at 512³, neutral at 2048³ (already near peak)." Use the REAL measured `<X>` from Step 2; do not overstate.

- [ ] **Step 4: Commit**

```bash
git add docs/SUPPORTED_OPS.md reports/perf_baseline.json
git commit -m "docs(autotune): document + honest measured numbers for matmul rr/rc autotuning

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Self-review

**Spec coverage:** selector w/ cache + opt-out + tie-break + size-contract validity → Task 1; descriptor + dispatch wiring + fallback → Task 2; regression + measured-not-projected perf + honest docs → Task 3. ✓
**Placeholder scan:** none — selector + dispatch code are complete; Task 3 Step 2 is a measurement step (numbers filled at execution, explicitly). ✓
**Type consistency:** `best_rrrc(msl_dtype, msl_out, M, N, K, runtime, cache_dir=None)`, `valid_candidates(M,N,K)`, `CANDIDATES`, the 8-tuple descriptor, `tile_m=8*rr`/`tile_n=32*rc` — consistent across tasks. ✓
**Safety:** every dispatched config is CANDIDATES-set + size-contract valid (all correct); any tuned-path error falls back to fixed `(4,4)`; opt-out env var. The measured-before-claiming step (Task 3 Step 2) is the anti-overstatement guard (the FA lesson).
