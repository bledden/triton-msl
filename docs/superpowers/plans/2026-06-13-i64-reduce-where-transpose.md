# i64/u64 reduce / where / transpose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `tl.sum`/`max`/`min` (reduce), `tl.where`, and transpose work on int64/uint64 by threading the 64-bit MSL type (`long`/`ulong`) through those paths and replacing the unsupported `simd_sum(long)` 1-D reduction with a shared-memory tree reduction — closing the `_I64_UNIMPLEMENTED` skips.

**Architecture:** The 2-D reduce (`_lower_reduce_2d`) is already type-parameterized + sequential, so it works once given the 64-bit dtype. Only the 1-D full reduce uses `simd_sum` (no `long` overload) and needs a new shared-memory tree branch. Transpose just needs the 64-bit shared dtype. `where` is likely pure type-plumbing (verify). The 64-bit branches are gated on the 64-bit dtype, so float/i32 paths are byte-unchanged.

**Tech Stack:** Python lowerer (`triton_metal/codegen/_lowerer_reduce.py`, `generic_lowerer.py`), MSL, pytest (serial GPU), `scripts/conftest_metal.py`.

> Run from the worktree with `/opt/homebrew/bin/python3`. GPU: SERIAL ONLY, dual-cache-clear before codegen-sensitive runs, `pkill`+recovery on hang.

---

### Task 1: 64-bit reduce/transpose + project test (RED→GREEN)

**Files:**
- Modify: `triton_metal/codegen/_lowerer_reduce.py` (`_lower_reduce` type detection + new `_lower_reduce_1d_i64`)
- Modify: `triton_metal/codegen/generic_lowerer.py` (`_lower_tt_trans` 64-bit dtype)
- Modify: `triton_metal/codegen/msl_emitter.py` (if `declare_threadgroup_array` doesn't map i64/u64)
- Test: `tests/test_i64_ops.py` (create)

- [ ] **Step 1: Write the project test (RED — i64 reduce/transpose/where with values >2^40 to prove no truncation)**

Create `tests/test_i64_ops.py`:

```python
"""int64/uint64 reduce / where / transpose. Values exceed 2^32 to prove no
truncation. Run with METAL_TEST_INT64=1 for the upstream corpus; the project
tests here exercise the paths directly. Serial GPU."""
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

if HAS:
    @triton.jit
    def _sum_i64(X, OUT, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(OUT, tl.sum(x))

    @triton.jit
    def _max_i64(X, OUT, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(OUT, tl.max(x))

    @triton.jit
    def _where_i64(C, A, B, OUT, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        c = tl.load(C + i) != 0
        tl.store(OUT + i, tl.where(c, tl.load(A + i), tl.load(B + i)))


@requires_metal
def test_i64_sum():
    BLOCK = 256
    X = torch.randint(2**40, 2**41, (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(1, dtype=torch.int64)
    _sum_i64[(1,)](X, OUT, BLOCK=BLOCK)
    assert int(OUT[0]) == int(X.sum()), f"got {int(OUT[0])} want {int(X.sum())}"


@requires_metal
def test_i64_max():
    BLOCK = 256
    X = torch.randint(-(2**41), 2**41, (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(1, dtype=torch.int64)
    _max_i64[(1,)](X, OUT, BLOCK=BLOCK)
    assert int(OUT[0]) == int(X.max()), f"got {int(OUT[0])} want {int(X.max())}"


@requires_metal
def test_i64_where():
    BLOCK = 256
    C = torch.randint(0, 2, (BLOCK,), dtype=torch.int64)
    A = torch.randint(2**40, 2**41, (BLOCK,), dtype=torch.int64)
    B = torch.randint(-(2**41), -(2**40), (BLOCK,), dtype=torch.int64)
    OUT = torch.zeros(BLOCK, dtype=torch.int64)
    _where_i64[(1,)](C, A, B, OUT, BLOCK=BLOCK)
    want = torch.where(C != 0, A, B)
    assert torch.equal(OUT, want)
```

- [ ] **Step 2: Run, observe RED**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_i64_ops.py -v 2>&1 | tail -15
```
Expected: `test_i64_sum`/`test_i64_max` FAIL (MSL compile error — `simd_sum`/`simd_max` no `long` overload, or i32 truncation). `test_i64_where` may already pass (then it's a guard for no-regression). Record which fail.

- [ ] **Step 3: 64-bit type detection in `_lower_reduce`**

In `triton_metal/codegen/_lowerer_reduce.py`, in `_lower_reduce` where `is_int_reduce`/`shared_dtype`/`msl_type` are set (~lines 288-292), add 64-bit detection:

```python
        input_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int_reduce = not (
            input_dtype.startswith("fp") or input_dtype.startswith("bf"))
        is_i64 = input_dtype in ("i64", "u64", "ui64")
        is_u64 = input_dtype in ("u64", "ui64")
        if is_i64:
            shared_dtype = "u64" if is_u64 else "i64"
            msl_type = "ulong" if is_u64 else "long"
        elif is_int_reduce:
            shared_dtype = "i32"
            msl_type = "int"
        else:
            shared_dtype = "fp32"
            msl_type = "float"
```
(Replace the existing 2-line shared_dtype/msl_type assignment with this.)

- [ ] **Step 4: Route the 1-D full reduce to a 64-bit tree when i64**

Find the `# 1D full reduction (original behavior)` block (~line 345). Add a branch BEFORE it:

```python
        if is_i64:
            self._lower_reduce_1d_i64(ssa, input_var, combine_op,
                                      msl_type, shared_dtype)
            return
        # 1D full reduction (original behavior)
        ...  # (existing simd path unchanged)
```

- [ ] **Step 5: Implement `_lower_reduce_1d_i64`**

Add to `_lowerer_reduce.py` (the `_ReduceScanMixin`):

```python
    def _lower_reduce_1d_i64(self, ssa, input_var, combine_op,
                             msl_type, shared_dtype):
        """1-D full reduce for 64-bit ints via a shared-memory tree (Metal has
        no simd_sum/max/min overload for long/ulong). Each thread writes its
        value to a threadgroup array; a stride-doubling tree (non-power-of-2
        safe via the `lid+s<bs` guard) reduces into slot 0; all threads read it.
        """
        bs = self.kb.block_size
        n = self._shared_counter
        self._shared_counter += 1
        sh = f"red64_{n}"
        self.kb.declare_threadgroup_array(sh, dtype=shared_dtype, size=bs)
        combine = {
            "sum": lambda a, b: f"({a} + {b})",
            "max": lambda a, b: f"max({a}, {b})",
            "min": lambda a, b: f"min({a}, {b})",
            "xor": lambda a, b: f"({a} ^ {b})",
        }[combine_op]
        kb = self.kb
        kb.raw_line(f"    {sh}[lid] = {input_var};")
        kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        kb.raw_line(f"    for (uint _s = 1u; _s < {bs}u; _s <<= 1u) {{")
        kb.raw_line(f"        if ((lid % (2u*_s)) == 0u && (lid + _s) < {bs}u) {{")
        kb.raw_line(f"            {sh}[lid] = {combine(f'{sh}[lid]', f'{sh}[lid + _s]')};")
        kb.raw_line("        }")
        kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        kb.raw_line("    }")
        result_var = self._next_var("reduced64")
        kb.raw_line(f"    {msl_type} {result_var} = {sh}[0];")
        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
```
NOTE: confirm `declare_threadgroup_array` maps `dtype="i64"`→`long` and `"u64"`→`ulong` in the emitted MSL. If it doesn't (only knows fp32/i32), extend its dtype→MSL map in `msl_emitter.py`. Also confirm `combine_op` for umax/umin: if `_lower_reduce` produces `combine_op="max"/"min"` and the type is `ulong`, `max`/`min` on ulong is unsigned — correct. (If the op set uses "umax"/"umin", add them to the `combine` dict mapping to max/min.)

- [ ] **Step 6: 64-bit transpose dtype**

In `generic_lowerer.py` `_lower_tt_trans` (~line 4187-4190), extend the dtype pick to 64-bit:

```python
        input_dtype = self.env_types.get(src_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        if input_dtype in ("i64", "u64", "ui64"):
            msl_type = "ulong" if input_dtype != "i64" else "long"
            shared_dtype = "u64" if input_dtype != "i64" else "i64"
        elif is_float:
            msl_type, shared_dtype = "float", "fp32"
        else:
            msl_type, shared_dtype = "int", "i32"
```
(Replace the existing 3-line is_float/msl_type/shared_dtype block.)

- [ ] **Step 7: Run the project test — GREEN; fix `where` only if it failed**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_i64_ops.py -v 2>&1 | tail -12
```
Expected: all pass (exact integer equality, values >2^40 proving no truncation). If `test_i64_where` fails, find where `tl.where`/`arith.select` emits its temporary type and ensure it uses `long`/`ulong` for an i64 result (likely a one-line type fix in the select lowering — grep `_lower_select`/`arith.select`/`_emit_select`). Re-run.

- [ ] **Step 8: Commit**

```bash
git add triton_metal/codegen/_lowerer_reduce.py triton_metal/codegen/generic_lowerer.py tests/test_i64_ops.py
# include msl_emitter.py / the select fix if touched
git commit -m "feat: int64/uint64 reduce, transpose (+ where) via 64-bit type-plumbing

1-D reduce uses a shared-memory tree (Metal has no simd_sum/max/min(long));
2-D reduce + transpose thread the long/ulong shared dtype. Values >2^40 verify
no 32-bit truncation. Gated on the 64-bit dtype; float/i32 paths unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Relax the conftest gate + ratchet (corpus reveals 2D/where coverage)

**Files:**
- Modify: `scripts/conftest_metal.py`

- [ ] **Step 1: Run the i64 corpus under the gate-override first (see what passes)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
METAL_TEST_INT64=1 bash scripts/run_upstream_test.sh "unit/language/test_core.py::test_reduce1d unit/language/test_core.py::test_reduce2d unit/language/test_core.py::test_where unit/language/test_core.py::test_transpose" -q 2>&1 | tail -10
```
This runs the i64 variants (the gate is overridden by METAL_TEST_INT64=1). Report pass/fail per family. If `test_reduce2d` or others FAIL (e.g. the 2-D path needs more than the dtype, or argmin/argmax i64), diagnose + fix following the same 64-bit-type pattern, OR narrow `_I64_UNIMPLEMENTED` to keep only the still-failing family skipped (be honest — only un-skip what passes).

- [ ] **Step 2: Relax `_I64_UNIMPLEMENTED` to the families that now pass**

In `scripts/conftest_metal.py` (~line 480), remove from `_I64_UNIMPLEMENTED` the families that pass (likely all four). If a family still fails (e.g. an argmin/argmax i64 sub-case), keep just that one and document why. Add a dated rationale referencing the design spec.

- [ ] **Step 3: Verify the now-ungated i64 corpus passes by DEFAULT (no override)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
bash scripts/run_upstream_test.sh "unit/language/test_core.py::test_reduce1d unit/language/test_core.py::test_reduce2d unit/language/test_core.py::test_where unit/language/test_core.py::test_transpose" -q 2>&1 | tail -8
```
Expected: the i64/u64 variants pass without METAL_TEST_INT64. Report the count delta.

- [ ] **Step 4: Ratchet — project suite green**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/ -q -k "not test_mept_m2_bug2_gpu and not test_mept_m3a_itercarry_gpu and not test_mept_m3c_gt1024_gpu" 2>&1 | tail -4
```
Expected: 0 failed (the known autotuner flake aside — re-run it alone if it appears).

- [ ] **Step 5: Commit**

```bash
git add scripts/conftest_metal.py
git commit -m "test(phase3): un-skip i64/u64 reduce/where/transpose — now pass

64-bit reduce (shared-mem tree) + transpose/where type-plumbing land these
without METAL_TEST_INT64. test_for_iv (i64 loop hang) + i64 atomics stay
skipped. Upstream test_core ratchets up.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** 64-bit type detection (Task 1 Step 3); 1-D tree reduce (Steps 4-5); 2-D reduce works via the type-parameterized path (given the dtype — Step 3 feeds it); transpose 64-bit dtype (Step 6); where verify-then-fix (Step 7); un-gate + ratchet (Task 2); test_for_iv/atomics stay out of scope. ✓

**Placeholder scan:** the tree-reduce + type-detection + transpose code is complete; the `where` step is conditional-on-failure with a concrete grep target; Task 2 Step 1 reveals 2D/argmin coverage empirically. The `declare_threadgroup_array` i64 mapping is a "confirm/extend" instruction (concrete). ✓

**Type consistency:** `is_i64`/`is_u64`/`shared_dtype`/`msl_type` consistent across `_lower_reduce` and `_lower_reduce_1d_i64`; `combine_op` keys (sum/max/min/xor) match the dict; `_lower_tt_trans` uses the same i64/u64→long/ulong + i64/u64 shared_dtype mapping. The tree's non-power-of-2 guard (`lid+_s < bs`) matches the textbook safe reduction. ✓
