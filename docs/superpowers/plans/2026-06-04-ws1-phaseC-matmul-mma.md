# WS1 Phase C — matmul MMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Metal matmul fast and honest — fix the fp16 lie (genuine `simdgroup_half8x8` MMA, proven), then close the ~2× MLX gap via deeper K-tiling, through one consolidated MMA emitter.

**Architecture:** The three duplicated matmul emitters (`make_simdgroup_matmul_kernel`, `_lower_simple_dot_inline`, `_lower_k_loop_dot_inline`) all upcast fp16→fp32 and barrier every 8 K-steps. We fix the fp16 lie first (in the standalone template for a fast proof, then consolidate all three through a shared `_emit_simdgroup_mma_tile` core), verify genuineness with hard evidence, then deepen K-tiling measured by the WS0/C6 harness.

**Tech Stack:** Metal Shading Language (`simdgroup_matrix`), Python MSL emitters, pyobjc Metal, numpy/torch reference, `benchmarks/hw_harness.py` as the perf oracle.

**Spec:** `docs/superpowers/specs/2026-06-04-ws1-phaseC-matmul-mma-design.md`

**Reference (read before starting):**
- `triton_metal/codegen/_msl_templates.py:3546` — `make_simdgroup_matmul_kernel` (the fp16-lie + barrier-per-8 template).
- `triton_metal/codegen/_lowerer_templates.py:34` — `_lower_simple_dot_inline`; `:209` — `_lower_k_loop_dot_inline` (same pattern; what real `@triton.jit` matmuls use).
- `tests/test_emitter.py:1251` — `test_simdgroup_matmul_32x32` (the runner-fixture correctness pattern to mirror).
- `benchmarks/hw_harness.py` — `matmul_*` specs (perf oracle; run with the `.venv` python, `TRITON_DEFAULT_BACKEND=metal`).

**Conventions for every command below:**
```bash
VPY=/Users/bledden/Documents/triton-metal/.venv/bin/python
export PYTHONPATH=/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
export TRITON_DEFAULT_BACKEND=metal
```
No PRs at any point (standing instruction — local worktree commits only).

---

## Phase C.1 — Fix the fp16 lie (genuine fp16), proven

### Task 1: De-risk — prove `simdgroup_half8x8` mixed-precision MMA compiles & is correct

**Files:**
- Test: `tests/test_simdgroup_half_mma.py` (Create)

This validates the central risk (does half×half→float MMA compile and compute correctly on this Metal target?) BEFORE we build the emitter on it.

- [ ] **Step 1: Write the de-risk test** (a minimal standalone half-MMA kernel, compiled + dispatched + compared to numpy)

```python
"""De-risk: simdgroup_half8x8 inputs with a float accumulator (half x half ->
float MMA) must compile on this Metal target and compute an 8x8x8 matmul
correctly. This is the foundation of the genuine-fp16 fix (WS1 Phase C.1)."""
import os, struct
import numpy as np
import pytest

try:
    import Metal, Foundation
    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS_METAL = False
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

_HALF_MMA_MSL = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void half_mma(device const half* A [[buffer(0)]],
                     device const half* B [[buffer(1)]],
                     device float* C [[buffer(2)]],
                     uint tiitg [[thread_index_in_threadgroup]]) {
    threadgroup half tgA[64], tgB[64];
    for (uint i = tiitg; i < 64u; i += 32u) { tgA[i] = A[i]; tgB[i] = B[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 a, b;          // GENUINE half input fragments
    simdgroup_float8x8 acc(0);        // float accumulator (precision)
    simdgroup_load(a, tgA, 8);
    simdgroup_load(b, tgB, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);   // half x half -> float
    simdgroup_store(acc, C, 8);
}
"""

@requires_metal
def test_half_mma_compiles_and_is_correct(tmp_path):
    import subprocess
    dev = Metal.MTLCreateSystemDefaultDevice()
    metal_p = str(tmp_path / "k.metal"); air = str(tmp_path / "k.air")
    lib_p = str(tmp_path / "k.metallib")
    open(metal_p, "w").write(_HALF_MMA_MSL)
    # Compile — this is the make-or-break for the genuine-fp16 path.
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", metal_p,
                           "-o", air, "-std=metal3.2", "-O2"])
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", air, "-o", lib_p])
    lib, _ = dev.newLibraryWithURL_error_(
        Foundation.NSURL.fileURLWithPath_(lib_p), None)
    fn = lib.newFunctionWithName_("half_mma")
    pso, _ = dev.newComputePipelineStateWithFunction_error_(fn, None)

    a = np.arange(64, dtype=np.float16).reshape(8, 8) * 0.1
    b = (np.arange(64, dtype=np.float16).reshape(8, 8) * 0.1)[::-1].copy()
    shared = Metal.MTLResourceStorageModeShared
    def hbuf(arr):
        flat = arr.astype(np.float16).flatten()
        buf = dev.newBufferWithLength_options_(flat.nbytes, shared)
        buf.contents().as_buffer(flat.nbytes)[:] = flat.tobytes(); return buf
    A, B = hbuf(a), hbuf(b)
    C = dev.newBufferWithLength_options_(64 * 4, shared)
    q = dev.newCommandQueue(); cmd = q.commandBuffer()
    enc = cmd.computeCommandEncoder(); enc.setComputePipelineState_(pso)
    for i, bf in enumerate([A, B, C]):
        enc.setBuffer_offset_atIndex_(bf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(32, 1, 1))
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    got = np.frombuffer(C.contents().as_buffer(64 * 4), dtype=np.float32).reshape(8, 8)
    ref = (a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 2: Run it; confirm half×half→float MMA works on this target**

Run: `$VPY -m pytest tests/test_simdgroup_half_mma.py -q -p no:cacheprovider`
Expected: PASS. If the `xcrun metal` compile FAILS on `simdgroup_half8x8` /
the mixed `simdgroup_multiply_accumulate`, STOP and record the exact error —
the fallback (half×half→half with periodic float accumulation, or documenting
the limitation) is chosen here before any emitter work, per the spec Risks.

- [ ] **Step 3: Commit**

```bash
git add tests/test_simdgroup_half_mma.py
git commit -m "test(WS1-C1): de-risk simdgroup_half8x8 mixed-precision MMA"
```

### Task 2: Failing genuineness test for the standalone template

**Files:**
- Test: `tests/test_matmul_fp16_genuine.py` (Create)

- [ ] **Step 1: Write the failing genuineness test**

```python
"""WS1 Phase C.1: fp16 matmul must be GENUINE — use simdgroup_half8x8 input
fragments and NOT upcast halves to float before the MMA. Asserted on the
generated MSL so 'fp16' can never silently be fp32 again."""
import re
from triton_metal.codegen._msl_templates import make_simdgroup_matmul_kernel

def test_fp16_matmul_uses_half_fragments():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_half8x8" in msl, "fp16 MMA must use half input fragments"

def test_fp16_matmul_does_not_upcast_before_mma():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    # the staging buffers must be half, not float; no float(A[/float(B[ upcast
    assert "threadgroup half" in msl
    assert not re.search(r"float\(\s*A\[", msl)
    assert not re.search(r"float\(\s*B\[", msl)

def test_fp16_accumulator_is_float_for_precision():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_float8x8 acc" in msl  # acc stays float
```

- [ ] **Step 2: Run it; confirm it FAILS against the current lying template**

Run: `$VPY -m pytest tests/test_matmul_fp16_genuine.py -q -p no:cacheprovider`
Expected: FAIL — current fp16 template has no `simdgroup_half8x8` and stages
through `threadgroup float` with `float(A[...])` upcasts.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_matmul_fp16_genuine.py
git commit -m "test(WS1-C1): failing genuineness test for fp16 matmul (currently fp32 in disguise)"
```

### Task 3: Make the standalone fp16 template genuine

**Files:**
- Modify: `triton_metal/codegen/_msl_templates.py` — the `dtype in ("fp16","f16")` branch of `make_simdgroup_matmul_kernel` (around `:3580`).

- [ ] **Step 1: Replace the fp16 branch's staging + fragments**

In the fp16 return block: change the threadgroup buffers to `half`, drop the
`float(...)` upcasts (store the raw `half`), use `simdgroup_half8x8` for
`a_frag`/`b_frag`, keep `simdgroup_float8x8` for `acc0..3`. Concretely the
fragment declarations and staging become:

```c
    threadgroup half tg_A[32 * 8];
    threadgroup half tg_B[8 * 32];
    simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);  // float accum
    simdgroup_half8x8 a_frag, b_frag;                        // half inputs
    ...
        // staging: store raw half (no float() upcast)
        tg_A[i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0.0h);
        ...
        tg_B[i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0.0h);
```
`simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0)` now MMAs
half×half→float (validated in Task 1). The store stays
`simdgroup_store(accN, C + ..., N)` into the float `C`.

- [ ] **Step 2: Run the genuineness test; confirm it PASSES**

Run: `$VPY -m pytest tests/test_matmul_fp16_genuine.py -q -p no:cacheprovider`
Expected: PASS (all three).

- [ ] **Step 3: Run the de-risk + emitter correctness suite**

Run: `$VPY -m pytest tests/test_simdgroup_half_mma.py tests/test_emitter.py -q -p no:cacheprovider -k "matmul or simdgroup"`
Expected: PASS — fp16 numerical correctness preserved (the existing
`test_simdgroup_matmul_*` and matmul tests stay green).

- [ ] **Step 4: Commit**

```bash
git add triton_metal/codegen/_msl_templates.py
git commit -m "feat(WS1-C1): genuine fp16 matmul — simdgroup_half8x8 inputs + float accumulator"
```

### Task 4: Prove the lie is dead in the harness (fp16 > fp32)

**Files:** none (measurement).

- [ ] **Step 1: Re-measure the matmul suite**

Run: `$VPY benchmarks/hw_harness.py matmul_2048_fp32_simd matmul_2048_fp16_simd matmul_4096_fp16_simd --no-disasm`
Expected: **fp16 TFLOP/s now meaningfully exceeds fp32** (the 7.00==7.00 tie is
broken upward). Record the numbers. If fp16 is NOT faster than fp32, the fix
isn't genuine yet — return to Task 3 (the staging or fragment type is still
forcing fp32).

- [ ] **Step 2: Record the result in the spec's success log**

Append the measured fp16-vs-fp32 numbers to the spec file under a new
"## C.1 result" heading (so the genuineness is documented with evidence).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-04-ws1-phaseC-matmul-mma-design.md
git commit -m "docs(WS1-C1): record genuine-fp16 harness evidence (fp16 > fp32)"
```

### Task 5: Consolidate the three emitter paths through one core

**Files:**
- Create: `triton_metal/codegen/_mma_tile.py` — `emit_simdgroup_mma_tile(dtype, ...)` returning the staging+MMA inner-loop MSL.
- Modify: `triton_metal/codegen/_msl_templates.py` (`make_simdgroup_matmul_kernel`), `triton_metal/codegen/_lowerer_templates.py` (`_lower_simple_dot_inline:34`, `_lower_k_loop_dot_inline:209`) to call the shared core.

- [ ] **Step 1: Write a failing equivalence test**

```python
"""All three matmul emitters must share the genuine-fp16 core: every path
that emits a simdgroup matmul uses half fragments for fp16, none upcasts."""
import re, pytest
from triton_metal.codegen._msl_templates import make_simdgroup_matmul_kernel

def test_jit_dot_paths_are_genuine_fp16():
    # the JIT inline paths must, for fp16, also use simdgroup_half8x8.
    from triton_metal.codegen._mma_tile import emit_simdgroup_mma_tile
    msl = emit_simdgroup_mma_tile(dtype="fp16")
    assert "simdgroup_half8x8" in msl
    assert not re.search(r"float\(\s*A\[", msl)
```

- [ ] **Step 2: Run it; confirm FAIL (module doesn't exist yet)**

Run: `$VPY -m pytest tests/test_matmul_fp16_genuine.py::test_jit_dot_paths_are_genuine_fp16 -q`
Expected: FAIL — `_mma_tile` not importable.

- [ ] **Step 3: Extract the shared core**

Create `_mma_tile.py::emit_simdgroup_mma_tile(dtype, m_tile=32, n_tile=32, k_depth=8)`
holding the staging + simdgroup-MMA inner loop (genuine half fragments for
fp16, float accumulator). Replace the bodies of the three emitters with calls
to it (fp32 output byte-identical or provably equivalent; fp16 now genuine).

- [ ] **Step 4: Run the genuineness, emitter, and equivalence tests**

Run: `$VPY -m pytest tests/test_matmul_fp16_genuine.py tests/test_emitter.py -q -p no:cacheprovider -k "matmul or simdgroup"`
Expected: PASS.

- [ ] **Step 5: Full upstream dot-test sweep stays green**

Run: `scripts/run_upstream_test.sh unit/language/test_core.py -q -p no:cacheprovider -k "dot or matmul"`
Expected: same pass count as baseline, 0 failed. (Then a full `test_core`
sweep before the phase closes — must stay 4326/0.)

- [ ] **Step 6: Commit**

```bash
git add triton_metal/codegen/_mma_tile.py triton_metal/codegen/_msl_templates.py triton_metal/codegen/_lowerer_templates.py tests/test_matmul_fp16_genuine.py
git commit -m "refactor(WS1-C1): one shared simdgroup-MMA emitter; genuine fp16 across all 3 paths"
```

---

## Phase C.2 — Close the MLX gap (deeper K-tiling), measured

This phase is **empirical**: each variant is implemented, then kept only if
the harness shows it helps. The "winning" tile config is discovered by
measurement, not pre-invented.

### Task 6: Deeper K-tiling (amortize the barrier)

**Files:**
- Modify: `triton_metal/codegen/_mma_tile.py` — add a `k_depth` that stages a
  K-deep tile per barrier and inner-loops the 8×8 MMAs over it.

- [ ] **Step 1: Add a correctness test at the new tiling for fp16 and fp32**

```python
# tests/test_mma_tile_tiling.py — compile emit_simdgroup_mma_tile(k_depth=32)
# for fp16 and fp32, dispatch a 256x256x256 matmul, assert np.allclose vs
# numpy (rtol 1e-2 fp16, 1e-4 fp32). (Mirror the Task 1 dispatch harness.)
```

- [ ] **Step 2: Run it; confirm FAIL (k_depth not yet supported)**

Run: `$VPY -m pytest tests/test_mma_tile_tiling.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement K-deep staging**

Stage `tg_A[m_tile * k_depth]`, `tg_B[k_depth * n_tile]` per barrier; inner
loop `for (kk = 0; kk < k_depth; kk += 8)` issuing the 8×8 MMAs; outer loop
`for (k = 0; k < K; k += k_depth)`. Keep boundary masking for K not a multiple
of `k_depth`.

- [ ] **Step 4: Correctness passes**

Run: `$VPY -m pytest tests/test_mma_tile_tiling.py tests/test_emitter.py -q -k "matmul or simdgroup or tile"`
Expected: PASS.

- [ ] **Step 5: Measure — keep only if it helps**

Run: `$VPY benchmarks/hw_harness.py matmul_2048_fp16_simd matmul_4096_fp16_simd matmul_2048_fp32_simd --no-disasm`
Compare TFLOP/s + MLX ratio vs the C.1 numbers. Try `k_depth` ∈ {16, 32, 64}
and (separately) larger output tiles; record each in a table. Keep the config
that maximizes throughput without dropping occupancy (watch the harness
reflection `occupancy_hint`). If a variant regresses, discard it.

- [ ] **Step 6: Commit the winning config**

```bash
git add triton_metal/codegen/_mma_tile.py tests/test_mma_tile_tiling.py
git commit -m "perf(WS1-C2): deeper K-tiling for simdgroup matmul (<best config>, <Nx> vs C.1)"
```

### Task 7: Larger output tiles + register blocking (if Task 6 still gap-bound)

- [ ] **Step 1:** Add a correctness test for a 64×64 output tile per threadgroup (8 simdgroups), mirroring Task 6's harness.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the wider tile (more accumulators per simdgroup; map simdgroups to a 64×64 region).
- [ ] **Step 4:** Correctness passes (numpy match).
- [ ] **Step 5:** Measure vs Task 6; keep only if it helps; watch occupancy.
- [ ] **Step 6:** Commit the winning config.

---

## Phase C.3 — Double-buffer (ONLY if still staging-bound)

### Task 8: Threadgroup double-buffering (conditional)

- [ ] **Step 1:** Decide via the harness: is C.2's best still sync/staging-bound (throughput well under MLX, occupancy not the limiter)? If NOT, skip C.3 entirely (YAGNI) and go to the closeout.
- [ ] **Step 2:** If yes: add a correctness test, then implement two threadgroup buffers ping-ponged so staging of tile k+1 overlaps the MMAs of tile k; barrier discipline updated.
- [ ] **Step 3:** Correctness passes.
- [ ] **Step 4:** Measure; keep only if it helps.
- [ ] **Step 5:** Commit.

---

## Closeout

### Task 9: Full verification + reassess C++

- [ ] **Step 1: Full upstream sweep stays green**

Run: `scripts/run_upstream_test.sh unit/language/test_core.py -q -p no:cacheprovider`
Expected: **4326 passed / 0 failed** (fresh cache; the established baseline).

- [ ] **Step 2: Project suite stays green**

Run: `$VPY -m pytest tests/ -q -m "not stress and not benchmark" --ignore=tests/test_models.py --ignore=tests/test_torch_compile.py -p no:cacheprovider`
Expected: 0 failed.

- [ ] **Step 3: Final harness baseline + summary**

Run: `$VPY benchmarks/hw_harness.py --no-disasm` and record the new matmul
numbers (fp16 genuinely > fp32; gap to MLX). Update the spec's "C.1 result"
section with the final C.2 numbers.

- [ ] **Step 4: Decide C++ generic-MMA**

Per the spec success criterion #5: from the post-fix harness numbers, decide
whether the C++ generic-MMA rebuild is still warranted (gap still large) or
deferred (MSL fix closed it). Record the decision in the roadmap doc.

- [ ] **Step 5: CHANGELOG + commit**

```bash
git add CHANGELOG.md docs/superpowers/specs/2026-06-04-ws1-phaseC-matmul-mma-design.md
git commit -m "docs(WS1-C): Phase C closeout — matmul perf results + C++ decision"
```

---

## Self-review notes (filled by writing-plans)

- **Spec coverage:** C.1 genuine-fp16 (Tasks 1–5) ✓; C.2 deeper tiling (6–7) ✓;
  C.3 double-buffer conditional (8) ✓; consolidation of 3 paths (Task 5) ✓;
  validation/correctness/4326-0 (Tasks 3,5,9) ✓; C++ reassessment (Task 9) ✓.
- **Empirical parts are honest:** C.2/C.3 are measure-and-keep loops, not
  pre-invented tile code — the harness is the oracle and bad variants are
  discarded. The genuinely-discoverable artifact (the half-MMA syntax) is
  produced by Task 1 before anything depends on it.
- **No silent caps:** boundary masking carried through every tiling change;
  K/M/N non-multiples handled.
