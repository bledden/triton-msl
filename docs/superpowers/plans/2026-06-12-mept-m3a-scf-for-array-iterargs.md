# MEPT Milestone 3a — loop-carried register arrays in scf.for Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an `scf.for` carry a multi-element value as a per-thread register array iter-arg (declare `T v[n]` once, body reads/writes `v[e]`, yield updates `v[e]`, result is the array), so a per-element accumulator carried across loop iterations computes correctly — completing the deferred M2 work. Plus the two M2-review predicate-completeness fixes that this relies on. Behind `TRITON_METAL_MEPT`, parity gate green.

**Architecture:** M2 made control-flow kernels enter the single-pass register-array form, but only *loop-invariant* multi-element values (referenced inside the loop) and *scalar* iter-args (the accumulator in Bug 2) were handled. A value that is itself a multi-element loop-carried iter-arg — e.g. `acc = tl.zeros((BLOCK,)); for i: acc = acc + load(...)` — still falls into `_lower_scf_for`'s scalar iter-arg path: it declares one MSL scalar for a value that needs `T v[n]`, so the yield assigns an array name to a scalar → invalid MSL / `UNKNOWN_` → refuse. M3a adds a register-array iter-arg branch to `_lower_scf_for` (declare/init `T v[n]`, map `env_array` for the block-arg and the result, yield per-element). Two precondition fixes land first: `_find_op_type_str` must recurse into nested scf regions (it stops one level deep, so type lookups inside nested loops falsely return `""` and disqualify MEPT), and `region_needs_arrays` must match all `result_ids` not just `.id` (asymmetric with `tensor_value_ids`).

**Tech Stack:** Python lowerer (`triton_metal/codegen/`), MSL source emission, pytest (CPU unit/emission + serial GPU correctness), Apple Metal.

---

## Scope

**In scope (M3a):** the register-array iter-arg path in `_lower_scf_for` (the deferred M2 work) + the two precondition fixes (`_find_op_type_str` recursion, `region_needs_arrays` result_ids symmetry).

**Deferred to later M3 plans (explicit, not omissions):**
- **M3b — `tt.dot` on the array form** (dot is not in `_MEPT_SAFE_OPS`; independent capability).
- **M3c — >1024 1D-ceiling audit** (may be a zero-code confirm-with-test) **+ convert_layout shuffle GPU hardening** (the shuffle is already implemented; M3-0 here removes one false-negative that blocked it).
These are independent per the code-exploration map and get their own plans/tests so each lands as working, testable software.

**Invariant held every task:** flag-OFF upstream `test_core` stays **5335 passed / 4007 skipped**; flag-OFF project suite stays green (**616/0** baseline post-M2); flag-ON `tests/test_mept_parity.py` stays byte-identical on the scalar corpus. All M3a changes are behind `self.mept_enabled` / `self._mept_single_pass`, so flag-OFF is provably untouched.

## File Structure

- `triton_metal/codegen/generic_lowerer.py` (modify): `_find_op_type_str` (~line 4223) → recurse into nested `region_ops`/`else_ops` at any depth, preserving "found-but-empty-type" vs "not-found" semantics.
- `triton_metal/codegen/regval.py` (modify): `region_needs_arrays` (~line 74) → also match `b.result_ids` against the multi set, symmetric with `tensor_value_ids`.
- `triton_metal/codegen/_lowerer_control.py` (modify): `_lower_scf_for` (lines 32–263) → add a register-array iter-arg branch (declare `T v[n]`, init per-element, map `env_array` for block-arg + result, yield per-element).
- `tests/test_generic_lowerer.py` (modify): unit test for `_find_op_type_str` nested recursion.
- `tests/test_regval.py` (modify): unit test for `region_needs_arrays` result_ids matching.
- `tests/test_mept_m3a_arrayiter.py` (create): CPU emission test (flag-ON, array iter-arg MSL, no `UNKNOWN_`).
- `tests/test_mept_m3a_itercarry_gpu.py` (create): serial GPU correctness (a per-element accumulator carried across a loop computes the column-sum).

---

### Task 1: `_find_op_type_str` recurses nested scf regions (M3-0)

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py:4223-4240`
- Test: `tests/test_generic_lowerer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generic_lowerer.py` (uses `GenericLowerer.__new__` to bypass `__init__` and a duck-typed graph — `_find_op_type_str` only touches `self.graph.ops`, `self.graph.args`, and each op's `.id`/`.type_str`/`.region_ops`/`.else_ops`):

```python
def test_find_op_type_str_recurses_nested_regions():
    from types import SimpleNamespace
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    def _op(id, type_str="", region_ops=None, else_ops=None):
        return SimpleNamespace(id=id, type_str=type_str,
                               region_ops=region_ops or [],
                               else_ops=else_ops or [])

    gl = GenericLowerer.__new__(GenericLowerer)
    inner = _op(99, "tensor<256xf32>")          # depth-2: inside a nested loop
    mid = _op(50, "", region_ops=[inner])        # depth-1: nested scf.for body
    outer = _op(10, "", region_ops=[mid])        # top-level scf.for
    gl.graph = SimpleNamespace(ops=[outer], args=[])

    # depth-2 id must be found (the bug: search stopped at depth 1)
    assert gl._find_op_type_str(99) == "tensor<256xf32>"
    # else_ops branch at depth 2 also reachable
    einner = _op(77, "tensor<128xi32>")
    eouter = _op(20, "", else_ops=[_op(60, "", region_ops=[einner])])
    gl.graph = SimpleNamespace(ops=[eouter], args=[])
    assert gl._find_op_type_str(77) == "tensor<128xi32>"
    # missing id returns "" (not an exception)
    assert gl._find_op_type_str(404) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_generic_lowerer.py -k find_op_type_str_recurses -v`
Expected: FAIL — `_find_op_type_str(99)` returns `""` (the current code searches only one region level deep), so the first assert fails.

- [ ] **Step 3: Write the recursive implementation**

Replace `_find_op_type_str` (lines 4223-4240) with:

```python
    def _find_op_type_str(self, ssa_id: int) -> str:
        """Find the type_str for an SSA value by searching ops, recursing
        into nested scf regions (region_ops = body/then, else_ops = while
        body/else) at any depth. Returns "" if not found."""
        def _search(ops):
            for ssa in ops:
                if ssa.id == ssa_id:
                    return ssa.type_str or ""
                body = list(ssa.region_ops or []) + list(ssa.else_ops or [])
                if body:
                    r = _search(body)
                    if r is not None:
                        return r
            return None
        r = _search(self.graph.ops)
        if r is not None:
            return r
        # Check args
        for arg in self.graph.args:
            if arg.id == ssa_id:
                return arg.type_str
        return ""
```

(The `None`-vs-`""` distinction preserves the original "found the id, its type_str happens to be empty" behavior — `_search` returns `""` on an id match and `None` only when the id is absent, so an op with an empty `type_str` still short-circuits the search instead of falling through to args.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_generic_lowerer.py -k find_op_type_str_recurses -v`
Expected: PASS.

- [ ] **Step 5: Quick regression — the file's existing tests still pass**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_generic_lowerer.py -q 2>&1 | tail -5`
Expected: all pass (no behavior change for already-found ids; only previously-unfound nested ids now resolve).

- [ ] **Step 6: Commit**

```bash
git add triton_metal/codegen/generic_lowerer.py tests/test_generic_lowerer.py
git commit -m "fix(mept-m3a): _find_op_type_str recurses nested scf regions

It searched only one region level deep, so type lookups for values defined
inside nested scf.for/if bodies returned '' and falsely disqualified the
kernel from MEPT (via _reduce_is_1d_full / _convert_resolves / _reshape).
Now recurses to any depth, preserving found-but-empty vs not-found.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `region_needs_arrays` matches all result_ids (M3-1)

**Files:**
- Modify: `triton_metal/codegen/regval.py` (the `region_needs_arrays` body, ~line 74)
- Test: `tests/test_regval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regval.py` (reuse the `_TVOp` mock added in M2 — it has `.op`, `.id`, `.operand_ids`, `.result_ids`, `.region_ops`, `.else_ops`):

```python
def test_region_needs_arrays_matches_result_ids_not_just_id():
    from triton_metal.codegen.regval import region_needs_arrays
    # A body op whose multi-element value is its result_ids[0]=200, while its
    # .id is a different number (300). tensor_value_ids would add 200 to the
    # multi set; region_needs_arrays must detect that 200 is produced here.
    producer = _TVOp("tt.load", id=300, operand_ids=[], result_ids=[200])
    loop = _TVOp("scf.for", id=1, operand_ids=[], result_ids=[],
                 region_ops=[producer])
    assert region_needs_arrays([loop], {200}) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_regval.py -k matches_result_ids -v`
Expected: FAIL — `region_needs_arrays` checks only `b.id` (300), which is not in `{200}`, so it returns False.

- [ ] **Step 3: Write the fix**

In `triton_metal/codegen/regval.py`, find the body-op id check inside `region_needs_arrays` (the line `if getattr(b, "id", None) in multi: return True`) and extend it to also match `result_ids`, mirroring how `tensor_value_ids` collects ids:

```python
                if getattr(b, "id", None) in multi:
                    return True
                for rid in (getattr(b, "result_ids", None) or []):
                    if rid in multi:
                        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_regval.py -k matches_result_ids -v`
Expected: PASS.

- [ ] **Step 5: Full regval suite still green**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_regval.py -q 2>&1 | tail -4`
Expected: all pass (the M1/M2 tests + the new one).

- [ ] **Step 6: Commit**

```bash
git add triton_metal/codegen/regval.py tests/test_regval.py
git commit -m "fix(mept-m3a): region_needs_arrays matches result_ids, not just .id

tensor_value_ids adds every result_id to the multi set, but
region_needs_arrays checked only b.id — so a multi-element value carried as
a body op's result_ids[0] (distinct from its .id) was missed, and the kernel
fell to the wrap-loop instead of the array form. Now symmetric.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: register-array iter-arg in `_lower_scf_for` (M3-2)

**Files:**
- Modify: `triton_metal/codegen/_lowerer_control.py:32-263` (`_lower_scf_for`)
- Test: `tests/test_mept_m3a_arrayiter.py` (create), `tests/test_mept_m3a_itercarry_gpu.py` (create)

This is the core of M3a. The change adds a third iter-arg form (register array) alongside the existing scalar and oversized-2D-smem forms.

- [ ] **Step 1: Write the failing GPU correctness test**

Create `tests/test_mept_m3a_itercarry_gpu.py` (run with `TRITON_METAL_MEPT=1`; serial only):

```python
"""MEPT M3a GPU correctness: a multi-element value carried as an scf.for
iter-arg (per-element accumulator) computes the column-sum correctly under
flag-ON. Previously the array iter-arg was emitted as a scalar -> invalid
MSL / refusal. Run with TRITON_METAL_MEPT=1. Serial only.
"""
import os
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
requires_mept = pytest.mark.skipif(
    os.environ.get("TRITON_METAL_MEPT") != "1",
    reason="requires TRITON_METAL_MEPT=1 (M3 register-array iter-arg)")

if HAS:
    @triton.jit
    def _vec_accumulate(X, OUT, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)   # per-element array iter-arg
        for i in range(n_tiles):
            acc = acc + tl.load(X + i * BLOCK + offs)
        tl.store(OUT + offs, acc)


@requires_metal
@requires_mept
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_vec_accumulate_column_sum(BLOCK):
    n_tiles = 8
    X = torch.randn(n_tiles * BLOCK)
    OUT = torch.zeros(BLOCK)
    _vec_accumulate[(1,)](X, OUT, n_tiles, BLOCK=BLOCK)
    want = X.view(n_tiles, BLOCK).sum(0)
    assert torch.allclose(OUT, want, atol=1e-2), (
        f"BLOCK={BLOCK}: max|diff|={float((OUT-want).abs().max()):.4g}")
```

- [ ] **Step 2: Run flag-ON, fresh cache — observe the RED (refuse or wrong)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 /opt/homebrew/bin/python3 -m pytest tests/test_mept_m3a_itercarry_gpu.py -v 2>&1 | tail -15
```
Expected: FAIL — the array iter-arg currently takes the scalar path; the kernel either raises `MetalNonRecoverableError` (UNKNOWN_ backstop / invalid MSL) or computes wrong values. Record which. (Per project rule: do NOT loosen tolerance; a wrong result must be made correct, not masked.)

- [ ] **Step 3: Write the register-array iter-arg branch**

In `triton_metal/codegen/_lowerer_control.py`, inside `_lower_scf_for`:

(3a) Near the iter-arg tracking init (right after `smem_iter_indices = set()`, ~line 72), add:

```python
        # MEPT (M3a): indices of iter_args carried as per-thread register
        # arrays ``T v[n]`` (a 1-D multi-element value updated across the
        # loop). Maps index -> (n_elems, msl_type).
        mept_array_iter_indices = set()
        mept_array_iter_n = {}
```

(3b) In the per-init_id loop, insert a new branch BEFORE the scalar fallback (i.e. before the `var_name = self._next_var("iter")` block at ~line 112), and AFTER the oversized-2D-smem `if len(init_shape) >= 2 and init_total > bs:` block:

```python
            # MEPT register-array iter-arg: a 1-D multi-element value carried
            # across the loop. Each thread owns ``n`` contiguous elements as a
            # mutable register array. Init may be a broadcast scalar (tl.zeros)
            # or already an env_array; declare T v[n] and seed every element.
            # Gated on the single-pass array regime so flag-off / scalar
            # kernels are unaffected.
            if (getattr(self, "_mept_single_pass", False)
                    and len(init_shape) == 1
                    and init_total > bs and init_total % bs == 0):
                n = init_total // bs
                if init_type.startswith("f") or init_type.startswith("bf"):
                    msl_type = "float"
                elif init_type in ("i64",):
                    msl_type = "long"
                elif init_type.startswith("u"):
                    msl_type = "uint"
                else:
                    msl_type = "int"
                if init_id in self.env_array:
                    src_arr, _src_n, _src_ty = self.env_array[init_id]
                    exprs = [f"{src_arr}[{e}]" for e in range(n)]
                else:
                    # broadcast scalar init (e.g. tl.zeros -> 0.0) to all elems
                    exprs = [init_val for _ in range(n)]
                var_name = self._var_array("iter", exprs, msl_type)
                iter_vars.append(var_name)
                iter_dtypes.append(init_type)
                mept_array_iter_indices.add(i)
                mept_array_iter_n[i] = (n, msl_type)
                continue
```

(3c) In the block-arg mapping loop (the `for i, var in enumerate(iter_vars):` at ~line 145), register the array view so the loop body resolves the iter-arg as an array. Add, inside that loop body (e.g. right after the `self.env_shapes[ba_id] = ...` propagation, alongside the `if i in smem_iter_indices:` handling):

```python
                    if i in mept_array_iter_indices:
                        n_arr, mt = mept_array_iter_n[i]
                        self.env_array[ba_id] = (var, n_arr, mt)
                        self.env_n_elems[ba_id] = n_arr
```

(3d) In the `scf.yield` handler (~line 182), add a per-element update for array iter-args BEFORE the scalar `yield_val = self._lookup(...)` assignment. Insert after the `if i in smem_iter_indices: ... continue` block:

```python
                            if i in mept_array_iter_indices:
                                n_arr, _mt = mept_array_iter_n[i]
                                ydesc = self.env_array.get(yield_id)
                                if ydesc is not None:
                                    ysrc, _yn, _yt = ydesc
                                    for e in range(n_arr):
                                        self.kb.raw_line(
                                            f"        {iter_vars[i]}[{e}] = "
                                            f"{ysrc}[{e}];")
                                else:
                                    yval = self._lookup(yield_id)
                                    for e in range(n_arr):
                                        self.kb.raw_line(
                                            f"        {iter_vars[i]}[{e}] = "
                                            f"{yval};")
                                continue
```

(3e) In the result-mapping loop (~line 221), register the result as an env_array so the post-loop store reads the array. Add inside that loop, alongside the `if i in smem_iter_indices:` handling:

```python
                    if i in mept_array_iter_indices:
                        n_arr, mt = mept_array_iter_n[i]
                        self.env_array[rid] = (var, n_arr, mt)
                        self.env_n_elems[rid] = n_arr
```

**Verification notes for the implementer (the intricate parts — confirm, don't guess):**
- Confirm that in the single-pass regime `bs = self.effective_block_size == num_threads` (the eligibility cascade sets `block_size = num_threads`), so `n = init_total // bs` equals the per-thread element count that `_lower_make_range` uses (`env_n_elems`). If `effective_block_size` is NOT num_threads here, derive `n` from `self._total_elements // num_threads` (or the same source make_range uses) so the iter-arg array width matches the loaded arrays it combines with. Add a `print`/assert during Step 2 debugging if unsure.
- Confirm the `env_array` tuple convention is `(name, n_elems, msl_type_string)` (as `_lower_make_range` writes `(var_name, n_per_thread, "uint")` and `_lower_reduce` unpacks `arr_name, n_arr, _ty`). Match it exactly.
- Confirm `init_shape = self.env_shapes.get(init_id, ())` is populated for the `tl.zeros((BLOCK,))` init (a 1-D shape like `(256,)`). If a `tl.zeros` init has no env_shapes entry, fall back to the result shape via `_extract_shape(self._find_op_type_str(ssa.result_ids[i]))`.

- [ ] **Step 4: Re-run flag-ON GPU correctness — fresh cache, until GREEN**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 /opt/homebrew/bin/python3 -m pytest tests/test_mept_m3a_itercarry_gpu.py -v 2>&1 | tail -12
```
Expected: 3 passed (BLOCK 256/512/1024), `OUT == X.view(n_tiles,BLOCK).sum(0)` within tol. If a mismatch persists, dump the MSL (`TRITON_METAL_DEBUG=1` or the codebase's MSL-dump mechanism — grep `msl_emitter.py`/`compiler.py` for `DEBUG`/`dump`) and fix the array width / yield indexing. Do not stop until all three are correct.

- [ ] **Step 5: Add the CPU emission smoke test**

Create `tests/test_mept_m3a_arrayiter.py` (CPU only, mirrors `tests/test_mept_m2_arrayform.py`'s `_emit` reload mechanism):

```python
"""MEPT M3a: an scf.for carrying a multi-element register-array iter-arg
emits array-indexed MSL (no UNKNOWN_) under flag-ON. CPU emission only."""
import importlib
import os

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


@triton.jit
def _vec_accumulate(X, OUT, n_tiles, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_tiles):
        acc = acc + tl.load(X + i * BLOCK + offs)
    tl.store(OUT + offs, acc)


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


_SIG = {"X": "*fp32", "OUT": "*fp32", "n_tiles": "i32"}


def test_vec_accumulate_no_unknown():
    on = _emit(_vec_accumulate, _SIG, dict(BLOCK=256), mept=True)
    assert "UNKNOWN_" not in on, on


def teardown_module(module):
    os.environ.pop("TRITON_METAL_MEPT", None)
    os.environ.pop("TRITON_METAL_FORCE_PYTHON", None)
```

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_mept_m3a_arrayiter.py -v`
Expected: PASS (no `UNKNOWN_`). If ASTSource collapses to `sizePerThread=[1]` for this kernel (the M2 contingency), this stays a smoke check and the GPU test (Step 4) is authoritative — record which case occurred.

- [ ] **Step 6: Parity gate green**

Run: `/opt/homebrew/bin/python3 -m pytest tests/test_mept_parity.py tests/test_regval.py -q 2>&1 | tail -4`
Expected: all pass — the new branch is gated on `_mept_single_pass`, so scalar/straight-line kernels (no array iter-arg) emit identical MSL flag-ON vs flag-OFF.

- [ ] **Step 7: Commit**

```bash
git add triton_metal/codegen/_lowerer_control.py tests/test_mept_m3a_arrayiter.py tests/test_mept_m3a_itercarry_gpu.py
git commit -m "feat(mept-m3a): scf.for carries per-thread register-array iter-args

A multi-element value carried as an scf.for iter-arg (e.g. a per-element
accumulator acc = tl.zeros((BLOCK,)); for i: acc += load(...)) is now declared
as T v[n], its block-arg and result registered in env_array, and yields update
v[e] per element. Previously it took the scalar iter-arg path -> invalid MSL /
UNKNOWN_ refusal. Column-sum kernel computes correctly at BLOCK 256/512/1024
(flag-on). Gated on _mept_single_pass; parity gate byte-identical.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: ratchet gate + memory/design update

**Files:**
- Modify: `docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md` (milestone status)
- (controller updates memory `project_mept_spine.md` separately)

> GPU discipline: SERIAL ONLY (no xdist), dual-cache-clear before flag-sensitive runs, `pkill` + recovery check on any hang.

- [ ] **Step 1: Flag-OFF upstream test_core (the 5,335/0 invariant)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
unset TRITON_METAL_MEPT
bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
```
Expected: **5335 passed, 4007 skipped** (the baseline). Any failure or different pass count → STOP, do not fix blindly; capture the failing names (all M3a changes are flag-gated, so a flag-OFF regression would indicate a leak — re-check the `_mept_single_pass`/`mept_enabled` guards). This run takes ~15 min; run it FOREGROUND/background and wait.

- [ ] **Step 2: Flag-OFF project suite**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/ -q -k "not gpu" 2>&1 | tail -8
```
Expected: green (the post-M2 baseline was 616 passed / 42 skipped; M3a adds unit + emission tests, so the passed count rises). Note: the flag-ON GPU iter-carry test self-skips here (no flag) — fine. Capture any FAILED names.

- [ ] **Step 3: Flag-ON gates (serial, fresh cache)**

```bash
/opt/homebrew/bin/python3 -m pytest tests/test_mept_parity.py tests/test_regval.py tests/test_mept_m3a_arrayiter.py -q 2>&1 | tail -4
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_MEPT=1 /opt/homebrew/bin/python3 -m pytest tests/test_mept_m3a_itercarry_gpu.py tests/test_mept_m2_bug2_gpu.py -q 2>&1 | tail -6
```
Expected: parity+units pass; flag-ON GPU iter-carry (3) + M2 Bug 2 (3) all pass (M3a must not regress M2).

- [ ] **Step 4: Update the design doc milestone status**

In `docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md`, update milestone 3 to record the M3a slice as done and the M3b/M3c remainder:

```markdown
3. Cooperative ops (reduce/dot/convert_layout) on the array form -> >1024 ceiling
   + chained reductions. **M3a DONE: scf.for carries per-thread register-array
   iter-args (the deferred M2 work) — a per-element accumulator across a loop
   computes (BLOCK 256/512/1024, flag-on); plus the two M2-review predicate
   fixes (_find_op_type_str recursion, region_needs_arrays result_ids). Flag-off
   test_core 5,335/0 held. Remaining: M3b (tt.dot on array form), M3c (>1024 1D
   ceiling audit + convert_layout shuffle GPU hardening).**
```

- [ ] **Step 5: Commit (design doc only; controller updates memory)**

```bash
git add docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md
git commit -m "docs(mept-m3a): mark M3a done — scf.for register-array iter-args

Loop-carried per-element accumulators compute (flag-on); flag-off test_core
5,335/0 held. M3b (dot) and M3c (>1024 audit + convert_layout) remain.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (umbrella design M3 = "cooperative ops … → >1024 + chained reductions"):
- "chained reductions" (loop-carried per-element array, the deferred M2 work) → Task 3. ✓
- The two M2-review carry-overs (preconditions named in `project_mept_spine.md`) → Tasks 1 & 2. ✓
- dot / >1024 / convert_layout → explicitly deferred to M3b/M3c with rationale (independent per the exploration map). ✓ (scope decision, not a gap.)
- Register-budget guard (design line 71): at M3a's `n`=2–8 per thread, no spill risk; the guard remains an M3b+/large-`n` concern. Noted, not built (YAGNI).

**Placeholder scan:** every code step has complete code; every run step an exact command + expected output. The Task 3 "verification notes" give concrete fallbacks (derive `n` from `_total_elements // num_threads`; result-shape fallback for missing env_shapes) rather than vague instructions — the one genuinely intricate area, called out explicitly. No TBD/TODO.

**Type/name consistency:** `mept_array_iter_indices` (set) and `mept_array_iter_n` (dict idx→`(n, msl_type)`) introduced in 3a, used in 3b/3c/3d/3e. `env_array` tuple `(name, n_elems, msl_type)` matches `_lower_make_range` (writes `(var, n, "uint")`) and `_lower_reduce` (unpacks `arr_name, n_arr, _ty`). `_var_array(prefix, exprs, ty)` signature matches `generic_lowerer.py:199` (mutable `ty name[N];` + per-element assigns). `_find_op_type_str` return contract (str, "" if absent) preserved. `_TVOp` mock reused from M2's `tests/test_regval.py`. The GPU and CPU tests share an identical `_vec_accumulate` definition and the `_SIG` matches its params.
