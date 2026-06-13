# MEPT Milestone 2 — scf.for register-array carry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hoisted multi-element values (a `tl.arange`, a masked-load `other=`) resolve inside a data-dependent `scf.for` loop, flipping tridec Bug 2 (`BLOCK≥256` reduction-in-loop) from a clean refusal to a correct computation — behind `TRITON_METAL_MEPT`, with the scalar parity gate held green.

**Architecture:** Milestone 1 already built the register-array machinery: `RegVal`/`region_needs_arrays` (`regval.py`), `_lookup_regval`, the array-aware `make_range`/`addptr`/`load`/`reduce` handlers, threadgroup-array hoisting, and the in-loop-reduce leading barrier. The ONLY reason Bug 2 still refuses is the eligibility gate: `_mept_single_pass` (the flag that makes `make_range` emit a register array) is set only when every op passes `_op_mept_ok`, and control-flow ops (`scf.for`/`scf.while`/`scf.if`) are not in `_MEPT_SAFE_OPS`. So a kernel with a runtime loop never enters the array form; the hoisted `arange` stays scalar, the re-execution wrap-loop can't carry it into the loop body, and `_lookup` falls back to `UNKNOWN_<addr>` → the integrity backstop refuses. **M2 extends eligibility:** a kernel whose data-dependent control flow carries multi-element values, and whose every op is array-wired (or control-flow / reduce / yield / condition), becomes single-pass MEPT-eligible. Register arrays declared once before the loop then persist into the body naturally (`env_array` is instance state, not loop-scoped), so the existing body-op handlers resolve them. No change to `_lower_scf_for` is required for Bug 2 — its iter-args here are scalar (`i`, `total`); the multi-element values (`offs`, `idx`, `v`) are loop-invariant or body-local and resolve through `env_array`.

**Tech Stack:** Python lowerer (`triton_metal/codegen/`), MSL source emission, pytest (CPU emission tests + serial GPU correctness tests), Apple Metal.

---

## Scope

**In scope (M2):** the eligibility extension that lets control-flow kernels enter the single-pass register-array form, validated end-to-end by the tridec Bug-2 kernel (`_sum_in_loop`) computing correctly at `BLOCK=256/512` under flag-ON.

**Deferred to M3 (explicitly, not an omission):** *array iter-arg carry* — a multi-element value that **changes per iteration** and is threaded through `scf.yield` as a loop-carried register array (declare `T v[n]` before the loop, body writes `v[e]`, yield updates `v`). Bug 2 does **not** need this: its design data-flow (`docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md` lines 56-64) carries only a **scalar** partial (`total`) across iterations; the array (`offs`) is loop-invariant and merely *referenced* inside the loop, not carried. Chained reductions (M3) are the first case that carries a per-element array across iterations, so the iter-arg array path lands there with a test that exercises it. Building it now would be untested-by-real-kernel dead code (YAGNI).

**Invariant held every task:** flag-OFF `test_core` stays 5,335/0; flag-ON `tests/test_mept_parity.py` stays green (the array form must never perturb scalar kernels — guaranteed because the new eligibility branch requires `any(control-flow op)`, which scalar kernels lack).

## File Structure

- `triton_metal/codegen/regval.py` (modify): add `tensor_value_ids(ops, is_multi_fn)` — a pure tree-walk that collects the SSA ids of multi-element values, recursing into control-flow regions. Pairs with the existing `region_needs_arrays`. No GPU, no lowerer dependency — unit-testable with mock ops.
- `triton_metal/codegen/generic_lowerer.py` (modify, two edits in `_decide_block_size`-region, ~lines 859-994): (1) compute `mept_arrayform_eligible` after the existing `mept_reduce_eligible`; (2) add the eligible branch inside `elif size_per_thread > 1 and block_size > num_threads:` that sets `_mept_single_pass`.
- `tests/test_regval.py` (modify): unit tests for `tensor_value_ids`.
- `tests/test_mept_m2_arrayform.py` (create): CPU emission test — flag-ON `_sum_in_loop` at `BLOCK=256` emits array-form MSL with no `UNKNOWN_` (fast signal; RED before the wiring).
- `tests/test_unknown_value_backstop.py` (modify): keep the flag-OFF refusal test; the existing test already runs flag-OFF (default) and must continue to refuse.
- `tests/test_mept_m2_bug2_gpu.py` (create): serial GPU correctness — flag-ON `_sum_in_loop` at `BLOCK=256/512/1024` computes `X.sum()` within tolerance. The authoritative validation.

---

### Task 1: `tensor_value_ids` pure helper

**Files:**
- Modify: `triton_metal/codegen/regval.py`
- Test: `tests/test_regval.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regval.py` (use the same mock-op style already in that file — a tiny object with `.op`, `.id`, `.operand_ids`, `.result_ids`, `.region_ops`):

```python
from triton_metal.codegen.regval import tensor_value_ids


class _Op:
    def __init__(self, op, id=None, operand_ids=None, result_ids=None,
                 region_ops=None, multi=False):
        self.op = op
        self.id = id
        self.operand_ids = operand_ids or []
        self.result_ids = result_ids or []
        self.region_ops = region_ops or []
        self._multi = multi


def _is_multi(op):
    return getattr(op, "_multi", False)


def test_tensor_value_ids_collects_multi_top_level():
    rng = _Op("tt.make_range", id="offs", result_ids=["offs"], multi=True)
    pid = _Op("tt.get_program_id", id="pid", result_ids=["pid"], multi=False)
    ids = tensor_value_ids([rng, pid], _is_multi)
    assert ids == {"offs"}


def test_tensor_value_ids_recurses_into_control_flow():
    body_load = _Op("tt.load", id="v", result_ids=["v"], multi=True)
    loop = _Op("scf.for", id="loop", result_ids=["loop"],
               region_ops=[body_load], multi=False)
    ids = tensor_value_ids([loop], _is_multi)
    assert ids == {"v"}


def test_tensor_value_ids_empty_when_no_multi():
    pid = _Op("tt.get_program_id", id="pid", result_ids=["pid"], multi=False)
    ids = tensor_value_ids([pid], _is_multi)
    assert ids == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regval.py -k tensor_value_ids -v`
Expected: FAIL with `ImportError: cannot import name 'tensor_value_ids'`

- [ ] **Step 3: Write minimal implementation**

Add to `triton_metal/codegen/regval.py` (after `region_needs_arrays`):

```python
def tensor_value_ids(ops, is_multi_fn) -> set:
    """Collect SSA ids of values that are multi-element-per-thread.

    ``is_multi_fn(op)`` returns True if ``op``'s result holds >1 element per
    thread (the caller decides, using the kernel's thread count + the value's
    tensor shape). Recurses into control-flow regions so a multi-element value
    produced inside a loop body is captured too. Pairs with
    ``region_needs_arrays`` to decide whether a region needs the array form.
    """
    ids = set()

    def _walk(op_list):
        for op in op_list:
            if is_multi_fn(op):
                oid = getattr(op, "id", None)
                if oid is not None:
                    ids.add(oid)
                for rid in (getattr(op, "result_ids", None) or []):
                    ids.add(rid)
            body = getattr(op, "region_ops", None)
            if body:
                _walk(body)

    _walk(ops)
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_regval.py -k tensor_value_ids -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add triton_metal/codegen/regval.py tests/test_regval.py
git commit -m "feat(mept-m2): tensor_value_ids — pure multi-element id collector

Pairs with region_needs_arrays to decide when a control-flow region
needs the register-array form. Pure tree-walk, unit-tested with mock ops.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Eligibility wiring + CPU emission test

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (after the `mept_reduce_eligible` block, ~line 878; and the `elif size_per_thread > 1 and block_size > num_threads:` cascade, ~line 951)
- Test: `tests/test_mept_m2_arrayform.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_mept_m2_arrayform.py`. This reuses the exact CPU emission mechanism from `tests/test_mept_parity.py` (`ASTSource` → `make_ttir` → `make_ttgir` → `emit_msl`, no GPU launch). The Bug-2 kernel is the one in `tests/test_unknown_value_backstop.py`:

```python
"""MEPT M2: a data-dependent scf.for carrying a hoisted multi-element value
emits the register-array form (no UNKNOWN_) under flag-ON. CPU emission only
(no GPU launch) — the fast signal for the eligibility extension. GPU numerical
correctness lives in tests/test_mept_m2_bug2_gpu.py.
"""
import importlib
import os

import pytest
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


@triton.jit
def _sum_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)            # hoisted outside the runtime loop
    total = 0.0
    for i in range(n_tiles):
        idx = i * BLOCK + offs
        v = tl.load(X + idx, mask=idx < N, other=0.0)
        total += tl.sum(v)
    tl.store(OUT, total)


def _emit(fn, sig, cst, mept):
    os.environ["TRITON_METAL_FORCE_PYTHON"] = "1"
    os.environ["TRITON_METAL_MEPT"] = "1" if mept else "0"
    import triton_metal.codegen.generic_lowerer as G
    import triton_metal.codegen.msl_emitter as M
    importlib.reload(G)
    importlib.reload(M)
    from triton_metal.backend.compiler import MetalBackend
    t = GPUTarget("metal", "apple-m4", 32)
    be = MetalBackend(t)
    o = be.parse_options({"num_warps": 4})
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(t, o, be.get_codegen_implementation(o),
                      be.get_module_map(), ctx)
    meta = {}
    mod = be.make_ttir(mod, meta, o)
    mod = be.make_ttgir(mod, meta, o)
    return M.emit_msl(mod, meta, o)


_SIG = {"X": "*fp32", "OUT": "*fp32", "N": "i32", "n_tiles": "i32"}


def test_sum_in_loop_block256_emits_array_no_unknown():
    on = _emit(_sum_in_loop, _SIG, dict(BLOCK=256), mept=True)
    assert "UNKNOWN_" not in on, (
        "hoisted arange/other still unresolved inside the loop:\n%s" % on)


def teardown_module(module):
    os.environ.pop("TRITON_METAL_MEPT", None)
    os.environ.pop("TRITON_METAL_FORCE_PYTHON", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mept_m2_arrayform.py -v`
Expected: FAIL — `emit_msl` raises `MetalNonRecoverableError` (the `UNKNOWN_` backstop) OR the emitted MSL contains `UNKNOWN_`. Either way the assertion/call fails. This is the RED that proves the kernel is still on the refusing path.

> If instead it PASSES at this step, the `ASTSource`→`make_ttgir` layout for this shape produced `sizePerThread=[1]` (no MEPT regime via ASTSource), so this CPU test cannot exercise the fix — in that case treat **Task 3 (GPU)** as the authoritative RED/GREEN and keep this test as a no-`UNKNOWN_` smoke check. Record which case occurred in the commit message.

- [ ] **Step 3: Write the eligibility predicate**

In `triton_metal/codegen/generic_lowerer.py`, immediately after the `mept_reduce_eligible = (...)` assignment (ends ~line 878), insert:

```python
        # Phase 4f (MEPT M2): control-flow kernels that carry a multi-element
        # value across a data-dependent scf.for/while/if. The re-execution
        # wrap-loop cannot carry per-element state across the control-flow
        # boundary, so values hoisted before the loop (a tl.arange register
        # array, a masked-load `other=` constant) fall back to UNKNOWN_ inside
        # it and the integrity backstop refuses (tridec Bug 2, BLOCK>=256).
        # Register arrays declared once before the loop persist into the body
        # naturally (env_array is instance state), so the existing array-wired
        # body handlers resolve them. Eligible iff: MEPT on; a control-flow op
        # is present; the region references/carries a multi-element value;
        # every op is array-wired (or control-flow / reduce / yield /
        # condition); every reduce is a 1-D full reduce; the tile cover is
        # exact; no fp8. See
        # docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md
        from triton_metal.codegen.regval import (
            region_needs_arrays as _region_needs_arrays,
            tensor_value_ids as _tensor_value_ids,
            _CONTROL_OPS as _CF_OPS,
        )

        def _arrayform_op_ok(s):
            if s.op in _CF_OPS:
                return all(_arrayform_op_ok(b) for b in (s.region_ops or []))
            if s.op in ("scf.yield", "scf.condition", "tt.reduce"):
                return True
            return _op_mept_ok(s)

        def _all_reduces(op_list):
            for s in op_list:
                if s.op == "tt.reduce":
                    yield s
                if s.region_ops:
                    yield from _all_reduces(s.region_ops)

        def _value_is_multi(s):
            shp = _extract_shape(getattr(s, "type_str", "") or "")
            if not shp:
                return False
            tot = 1
            for d in shp:
                tot *= d
            return tot > num_threads

        _multi_ids = _tensor_value_ids(_top_ops, _value_is_multi)
        mept_arrayform_eligible = (
            self.mept_enabled
            and any(s.op in _CF_OPS for s in _top_ops)
            and _region_needs_arrays(_top_ops, _multi_ids)
            and all(_arrayform_op_ok(s) for s in _top_ops)
            and all(_reduce_is_1d_full(r) for r in _all_reduces(_top_ops))
            and not any(_op_is_fp8(s) for s in all_ops_iter)
        )
```

- [ ] **Step 4: Add the eligible branch to the wrapping-strategy cascade**

In the same file, find `elif size_per_thread > 1 and block_size > num_threads:` (~line 951). Its first child is currently `if (mept_reduce_eligible ...`. Insert a new FIRST child above it:

```python
        elif size_per_thread > 1 and block_size > num_threads:
            if (mept_arrayform_eligible
                    and num_threads * size_per_thread == block_size):
                # MEPT M2 single-pass register-array form for control-flow
                # kernels. Each thread owns size_per_thread contiguous
                # elements as a register array (idx[i] = lid*N + i). The array
                # IS the per-thread multiplicity, so there is NO wrap-loop
                # (_needs_wrapping stays False). Arrays declared before a
                # data-dependent scf.for persist into its body, so hoisted
                # values (arange, masked-load `other`) resolve inside the loop.
                self._total_elements = block_size
                block_size = num_threads
                self._mept_single_pass = True
            elif (mept_reduce_eligible
                    and num_threads * size_per_thread == block_size):
```

(The `elif (mept_reduce_eligible ...` line is the EXISTING line — leave its body unchanged; you are only converting the existing leading `if` into an `elif` and prepending the new `if`. Verify by re-reading the block after editing: the order is now arrayform → reduce → multipass → kernel_safe → wrap.)

- [ ] **Step 5: Run the emission test to verify it passes**

Run: `python -m pytest tests/test_mept_m2_arrayform.py -v`
Expected: PASS — no `UNKNOWN_` in the emitted MSL (the hoisted `arange`/`other` now resolve via `env_array` inside the loop). (If Step 2 hit the "passes-already" branch, this remains green; rely on Task 3.)

- [ ] **Step 6: Run the parity gate (must stay green)**

Run: `python -m pytest tests/test_mept_parity.py tests/test_regval.py -v`
Expected: PASS — scalar corpus byte-identical flag-ON vs flag-OFF (the new branch requires a control-flow op, which these kernels lack, so nothing changes for them).

- [ ] **Step 7: Commit**

```bash
git add triton_metal/codegen/generic_lowerer.py tests/test_mept_m2_arrayform.py
git commit -m "feat(mept-m2): single-pass array form for control-flow kernels

Extend MEPT eligibility so a kernel whose data-dependent scf.for/while/if
carries a multi-element value enters the single-pass register-array path.
make_range then emits arrays that persist into the loop body, so hoisted
values (arange offsets, masked-load other=) resolve inside the loop instead
of falling back to UNKNOWN_. No change for scalar/straight-line kernels
(branch requires a control-flow op); parity gate stays green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: tridec Bug 2 — GPU correctness (RED → GREEN)

**Files:**
- Create: `tests/test_mept_m2_bug2_gpu.py`
- Modify: `tests/test_unknown_value_backstop.py` (keep flag-OFF refusal; add a comment that flag-ON now computes — see Step 5)

> **GPU discipline (mandatory):** run GPU tests SERIALLY (never a parallel sweep — it wedges the Metal command queue). Clear BOTH caches before any flag-ON run because codegen changed. If a run hangs, `pkill -f pytest`, then sanity-check GPU recovery with a tiny kernel before retrying.

- [ ] **Step 1: Write the failing GPU test**

Create `tests/test_mept_m2_bug2_gpu.py`:

```python
"""MEPT M2 GPU correctness: the tridec Bug-2 reduction-in-loop kernel computes
correctly at BLOCK>=256 under flag-ON (previously refused with
MetalNonRecoverableError). Run with TRITON_METAL_MEPT=1. Serial only.
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _sum_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = 0.0
        for i in range(n_tiles):
            idx = i * BLOCK + offs
            v = tl.load(X + idx, mask=idx < N, other=0.0)
            total += tl.sum(v)
        tl.store(OUT, total)


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_sum_in_loop_computes_flag_on(BLOCK):
    N = 4096
    X = torch.randn(N)
    OUT = torch.zeros(1)
    n_tiles = (N + BLOCK - 1) // BLOCK
    _sum_in_loop[(1,)](X, OUT, N, n_tiles, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-1, (
        f"BLOCK={BLOCK}: got {float(OUT[0])}, want {X.sum().item()}")
```

- [ ] **Step 2: Run to verify it fails (flag-ON, fresh cache)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 python -m pytest tests/test_mept_m2_bug2_gpu.py -v
```
Expected: BEFORE Task 2's wiring this would refuse; AFTER Task 2 it should already PASS (the fix is the eligibility extension from Task 2). If it FAILS here with a numerical mismatch or a fresh `UNKNOWN_`/compile error, that is a real M2 gap — diagnose before proceeding (likely: the masked-load `other=` constant, or the contiguous `idx[i]=lid*N+i` mapping vs the tile layout). Do NOT mark GREEN until all three BLOCK sizes compute within tolerance.

- [ ] **Step 3: If a gap surfaces — diagnose, do not paper over**

If Step 2 mismatches, dump the emitted MSL to see what is wrong:

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 TRITON_METAL_DEBUG=1 python -c "
import torch, triton, triton.language as tl
@triton.jit
def k(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK); total = 0.0
    for i in range(n_tiles):
        idx = i*BLOCK + offs
        v = tl.load(X+idx, mask=idx<N, other=0.0)
        total += tl.sum(v)
    tl.store(OUT, total)
X=torch.randn(4096); OUT=torch.zeros(1)
k[(1,)](X, OUT, 4096, 16, BLOCK=256)
print('got', float(OUT[0]), 'want', X.sum().item())
"
```
Per the "no shortcuts" rule: fully resolve any mismatch (it is a real correctness bug in the array path), do not loosen the tolerance to hide it. Re-run Step 2 until all three pass.

- [ ] **Step 4: Run to verify it passes**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 python -m pytest tests/test_mept_m2_bug2_gpu.py -v
```
Expected: PASS (3 passed) — BLOCK 256/512/1024 all within tolerance.

- [ ] **Step 5: Confirm flag-OFF still refuses (unchanged backstop)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
python -m pytest tests/test_unknown_value_backstop.py -v
```
Expected: PASS — `test_sum_in_loop_block256_refuses_not_compile_error` still raises `MetalNonRecoverableError` with MEPT off (default). Add this one-line comment above that test so the flag dependency is explicit:

```python
# NOTE: refuses only with MEPT OFF (default). Under TRITON_METAL_MEPT=1 this
# kernel computes correctly — see tests/test_mept_m2_bug2_gpu.py (M2).
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_mept_m2_bug2_gpu.py tests/test_unknown_value_backstop.py
git commit -m "test(mept-m2): tridec Bug 2 computes at BLOCK 256/512/1024 (flag-on)

The reduction-in-loop kernel that previously refused (UNKNOWN_ on the
hoisted arange/other inside a runtime scf.for) now computes X.sum()
correctly under TRITON_METAL_MEPT=1. Flag-off still refuses cleanly.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Ratchet gate + memory/design update

**Files:**
- Modify: `docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md` (milestone status)
- Modify: memory `project_mept_spine.md` (M2 progress)

- [ ] **Step 1: Flag-OFF full regression (the 5,335/0 invariant)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
python -m pytest tests/test_core.py -q 2>&1 | tail -5
```
Expected: 5,335 passed, 0 failed (flag-OFF is the integrity reference; M2 must not perturb it). If any regression: STOP, the eligibility branch leaked into a flag-OFF path — re-check that `mept_arrayform_eligible` short-circuits on `self.mept_enabled`.

- [ ] **Step 2: Project suite (flag-OFF)**

```bash
python -m pytest tests/ -q -k "not gpu" 2>&1 | tail -8
```
Expected: project suite green (the 625/0 project tests + the new CPU tests). Note: `test_mept_m2_bug2_gpu.py` is skipped here unless `TRITON_METAL_MEPT=1` and Metal are present; that is fine — it ran in Task 3.

- [ ] **Step 3: Flag-ON parity gate (final confirmation)**

```bash
python -m pytest tests/test_mept_parity.py tests/test_mept_m2_arrayform.py tests/test_regval.py -v
```
Expected: PASS — scalar parity byte-identical, array-form emission clean.

- [ ] **Step 4: Update the design doc milestone status**

In `docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md`, edit the Milestones list (line 88-94) to mark M2 done:

```markdown
2. `scf.for`/`while` array-carry -> Bug-2 BLOCK>=256 correct. **DONE (M2):
   eligibility extended so control-flow kernels enter the single-pass array
   form; hoisted values persist into the loop body via env_array. tridec Bug 2
   computes at BLOCK 256/512/1024 (flag-on); flag-off 5,335/0 unchanged. Array
   iter-arg carry (per-element state across iterations) deferred to M3 where
   chained reductions require it.**
```

- [ ] **Step 5: Update memory**

Edit the memory file `project_mept_spine.md` — change the `M2 NEXT` bullet to `M2 DONE` with the commit range, and set `M3 NEXT`. Keep it one fact, no duplication.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md
git commit -m "docs(mept-m2): mark Milestone 2 done — Bug 2 computes at BLOCK>=256

Eligibility extension lands the register-array form for control-flow
kernels. Flag-off 5,335/0 held; array iter-arg carry deferred to M3.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (`2026-06-11-mept-register-array-spine-design.md` M2 = "scf.for/while array-carry → Bug-2 BLOCK≥256 correct"):
- "Bug-2 BLOCK≥256 correct" → Task 3 (GPU, BLOCK 256/512/1024). ✓
- "scf.for/while array-carry" → the design's own Bug-2 dataflow (lines 56-64) carries a scalar partial and references a loop-invariant array; that is delivered by the eligibility extension (Tasks 1-2) + env_array persistence. True loop-carried *array iter-args* are explicitly deferred to M3 with rationale (Scope section). ✓ (consistent with the design's example).
- "Flag-gated, parity stays green" → Task 2 Step 6, Task 4 Step 3. ✓
- "UNKNOWN_ backstop stays" (design line 69) → unchanged; flag-OFF still refuses (Task 3 Step 5). ✓
- Register-budget guard (design line 71) → not triggered at M2's `spt`=2-8; the guard is an M3+ concern when `spt` grows. Noted, not built (YAGNI). 

**Placeholder scan:** every code step has complete code; every run step has an exact command + expected output. No TBD/TODO. The one conditional ("if Step 2 passes already") gives a concrete fallback (rely on Task 3) rather than a vague instruction. ✓

**Type/name consistency:** `tensor_value_ids(ops, is_multi_fn)` defined in Task 1, called in Task 2 (as `_tensor_value_ids(_top_ops, _value_is_multi)`). `region_needs_arrays` and `_CONTROL_OPS` are pre-existing M1 exports. `_mept_single_pass`, `_total_elements`, `num_threads`, `size_per_thread`, `_top_ops`, `all_ops_iter`, `_op_mept_ok`, `_reduce_is_1d_full`, `_op_is_fp8`, `_extract_shape` all confirmed in scope at the insertion point (generic_lowerer.py lines 774-994). `_sum_in_loop` signature identical across the CPU and GPU tests and matches `tests/test_unknown_value_backstop.py`. ✓
