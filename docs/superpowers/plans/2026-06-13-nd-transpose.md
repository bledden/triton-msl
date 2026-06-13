# Generic N-D transpose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close `test_trans_4d` (96 cases) by detecting the `load(N-D) → trans → [reshape] → store` pattern and emitting a closed-form direct-copy kernel — `out[k] = in[src_flat(k)]` in a strided loop that handles the >1024-element shapes the shared-memory path can't.

**Architecture:** A new detector `_detect_nd_trans` + template `_lower_nd_trans_template` (mirroring the existing `_detect_transpose_via_reshape` / `_lower_transpose_via_reshape_template` pair). The detector matches the pattern and extracts src_shape + permutation `order` + the input/output pointer args; the template computes row-major strides at build time and emits the closed-form output→input index map in a `for(k=lid; k<total; k+=threads)` loop. No shared memory, no barrier. The generic `_lower_tt_trans` rank≥3 refusal stays as the backstop for un-matched cases.

**Tech Stack:** Python lowerer (`triton_metal/codegen/_lowerer_detection.py`, `_lowerer_templates.py`, `generic_lowerer.py`), MSL, pytest (serial GPU), `scripts/conftest_metal.py`.

> Run from the worktree dir with `/opt/homebrew/bin/python3`. GPU: SERIAL ONLY, dual-cache-clear before codegen-sensitive runs, `pkill`+recovery on hang.

---

### Task 1: Detector + template + wiring (RED→GREEN)

**Files:**
- Modify: `triton_metal/codegen/_lowerer_detection.py` (add `_detect_nd_trans`)
- Modify: `triton_metal/codegen/_lowerer_templates.py` (add `_lower_nd_trans_template`)
- Modify: `triton_metal/codegen/generic_lowerer.py` (`lower()` — call the detector)
- Test: `tests/test_nd_transpose.py` (create)

- [ ] **Step 1: Write the project correctness test (RED)**

Create `tests/test_nd_transpose.py`. It mirrors upstream `test_trans_4d` (make_tensor_descriptor → load N-D → permute → reshape → store), including the allocator setup the upstream conftest provides:

```python
"""Generic N-D transpose (tt.trans rank>=3) via the closed-form direct-copy
template. Mirrors upstream test_trans_4d. Serial GPU."""
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
    def _trans4d(In, Out, s1, s2, s3, s4,
                 t1: tl.constexpr, t2: tl.constexpr,
                 t3: tl.constexpr, t4: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            In, shape=[s1, s2, s3, s4],
            strides=[s2 * s3 * s4, s3 * s4, s4, 1],
            block_shape=[s1, s2, s3, s4])
        total = s1 * s2 * s3 * s4
        out_desc = tl.make_tensor_descriptor(
            Out, shape=[total], strides=[1], block_shape=[total])
        val = in_desc.load([0, 0, 0, 0]).permute((t1, t2, t3, t4))
        out_desc.store([0], val.reshape(out_desc.block_shape))

    def _alloc(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cpu")


@requires_metal
@pytest.mark.parametrize("shape,perm", [
    ((4, 4, 4, 16), (3, 1, 0, 2)),     # 1024, non-trivial perm
    ((4, 4, 4, 16), (0, 2, 1, 3)),     # 1024, mid-axis swap
    ((2, 2, 8, 64), (1, 0, 3, 2)),     # 2048 (>1024) — exercises the strided loop
    ((2, 2, 8, 64), (3, 2, 1, 0)),     # 2048, full reverse
])
@pytest.mark.parametrize("dt", [torch.int32, torch.int8])
def test_nd_transpose(shape, perm, dt):
    triton.set_allocator(_alloc)
    s1, s2, s3, s4 = shape
    total = s1 * s2 * s3 * s4
    hi = 127 if dt == torch.int8 else 100000
    In = torch.randint(-hi, hi, shape, dtype=dt)
    Out = torch.zeros(total, dtype=dt)
    _trans4d[(1,)](In.reshape(-1), Out, s1, s2, s3, s4, *perm)
    want = In.permute(perm).reshape(-1)
    assert torch.equal(Out, want), (
        f"shape={shape} perm={perm} dt={dt}: mismatch")
```

- [ ] **Step 2: Run, verify it FAILS (refusal today)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_nd_transpose.py -x 2>&1 | tail -12
```
Expected: FAIL — the rank-4 non-identity `tt.trans` raises `MetalNonRecoverableError` ("rank-4 tt.trans with a non-identity permutation ... not supported"). This is the RED. (If instead it errors on `make_tensor_descriptor`/allocator setup, fix the test harness first — the allocator must be set before the launch.)

- [ ] **Step 3: Add `_detect_nd_trans` to `_lowerer_detection.py`**

Model on `_detect_transpose_via_reshape` (same file). Add:

```python
    def _detect_nd_trans(self):
        """Detect a generic N-D transpose: one tt.load of a rank>=3 tensor, one
        tt.trans (any permutation), optional tt.reshape(s), one tt.store to a
        flat pointer, with NO reduce/scan/dot/control-flow. Emits a closed-form
        direct copy (out[k] = in[src_flat(k)]). Returns dict or None.

        More specific transpose templates (_detect_transpose_via_reshape,
        _detect_permute_chained_reduce) run first and return None for anything
        they don't own, so this is the general fallback for test_trans_4d."""
        load_ssa = store_ssa = trans_ssa = None
        for ssa in self.graph.ops:
            if ssa.op == "tt.load":
                if load_ssa is not None:
                    return None
                load_ssa = ssa
            elif ssa.op == "tt.store":
                if store_ssa is not None:
                    return None
                store_ssa = ssa
            elif ssa.op == "tt.trans":
                if trans_ssa is not None:
                    return None
                trans_ssa = ssa
            elif ssa.op == "tt.reshape":
                pass  # allowed (descriptor lowering inserts these)
            elif ssa.op in ("scf.for", "scf.while", "scf.if",
                            "tt.reduce", "tt.scan", "tt.dot"):
                return None
        if load_ssa is None or store_ssa is None or trans_ssa is None:
            return None
        # The transpose operates on its INPUT's shape (the N-D tensor).
        src_shape = _extract_shape(self._find_op_type_str(trans_ssa.operand_ids[0]))
        if not src_shape or len(src_shape) < 3:
            return None
        order = self._parse_trans_order(trans_ssa, len(src_shape))
        if order is None or sorted(order) != list(range(len(src_shape))):
            return None
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        if len(ptr_args) < 2:
            return None
        total = 1
        for s in src_shape:
            total *= s
        return {
            "input_arg": ptr_args[0].name,
            "output_arg": ptr_args[1].name,
            "elem_type": ptr_args[0].elem_type,
            "src_shape": list(src_shape),
            "order": list(order),
            "total": total,
        }
```

- [ ] **Step 4: Add `_lower_nd_trans_template` to `_lowerer_templates.py`**

Model on `_lower_transpose_via_reshape_template`. Add:

```python
    def _lower_nd_trans_template(self, info) -> str:
        """Closed-form N-D transpose: out[k] = in[src_flat(k)] in a strided
        loop. src_flat(k) = sum_d ((k / dst_stride[d]) % dst_shape[d]) *
        src_stride[order[d]], with row-major strides computed here."""
        src_shape = info["src_shape"]
        order = info["order"]
        total = info["total"]
        rank = len(src_shape)
        dst_shape = [src_shape[order[d]] for d in range(rank)]

        def _row_major_strides(shape):
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            return st

        dst_stride = _row_major_strides(dst_shape)
        src_stride = _row_major_strides(src_shape)
        # in_flat = sum_d O[d] * src_stride[order[d]];  O[d]=(k/dst_stride[d])%dst_shape[d]
        terms = []
        for d in range(rank):
            o_d = (f"(k % {dst_shape[d]}u)" if dst_stride[d] == 1
                   else f"((k / {dst_stride[d]}u) % {dst_shape[d]}u)")
            terms.append(f"{o_d} * {src_stride[order[d]]}u")
        in_flat = " + ".join(terms)

        elem_type = info["elem_type"]
        input_arg = info["input_arg"]
        output_arg = info["output_arg"]
        msl_type = triton_type_to_msl(elem_type)  # noqa: F841 (typed via args)
        safe_name = _sanitize_msl_name(self.graph.func_name)
        num_warps = self.options.num_warps if self.options else 4
        threads = num_warps * 32

        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_msl_type = (triton_type_to_msl(arg.elem_type)
                                if arg.elem_type else "int")
                arg_decls.append(
                    f"    constant {arg_msl_type}& {arg.name} [[buffer({i})]]")

        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            f"kernel void {safe_name}(",
            ",\n".join(arg_decls) + ",",
            "    uint pid [[threadgroup_position_in_grid]],",
            "    uint lid [[thread_position_in_threadgroup]],",
            "    uint tid [[thread_position_in_grid]]",
            ") {",
            f"    for (uint k = lid; k < {total}u; k += {threads}u) {{",
            f"        {output_arg}[k] = {input_arg}[{in_flat}];",
            "    }",
            "}",
            "",
        ]
        return "\n".join(lines)
```
(Confirm `triton_type_to_msl` and `_sanitize_msl_name` are already imported in `_lowerer_templates.py` — they are used by `_lower_transpose_via_reshape_template` in the same file.)

- [ ] **Step 5: Wire the detector into `lower()`**

In `generic_lowerer.py`, find where `_detect_transpose_via_reshape` is called (~line 689) and add the new detector right after it (so the more-specific one wins first):

```python
        nd_trans_info = self._detect_nd_trans()
        if nd_trans_info:
            return self._lower_nd_trans_template(nd_trans_info)
```
(Match the surrounding return style — the existing detector does `msl = self._lower_..._template(info); return ...`. If `lower()` returns the MSL string directly, mirror that exactly.)

- [ ] **Step 6: Run the test — verify GREEN**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_nd_transpose.py -v 2>&1 | tail -15
```
Expected: all 8 cases pass (4 perms × 2 dtypes), INCLUDING the (2,2,8,64)=2048 cases (the strided loop). `torch.equal` exact match. If a perm is wrong, the closed-form index map has a bug — print the emitted MSL (the `in_flat` expression) and check against `In.permute(perm)`; do NOT loosen to allclose (it's an exact integer permutation).

- [ ] **Step 7: Commit**

```bash
git add triton_metal/codegen/_lowerer_detection.py triton_metal/codegen/_lowerer_templates.py triton_metal/codegen/generic_lowerer.py tests/test_nd_transpose.py
git commit -m "feat: generic N-D transpose via closed-form direct-copy template

Detect load(N-D)->trans->[reshape]->store and emit out[k]=in[src_flat(k)] in a
strided loop (handles >1024 elements the shared-mem path can't). Closes
test_trans_4d's rank-4 permutations; reduces to the 2-D formula. The generic
_lower_tt_trans rank>=3 refusal stays as the backstop for un-matched cases.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Un-skip test_trans_4d + ratchet

**Files:**
- Modify: `scripts/conftest_metal.py`

- [ ] **Step 1: Un-skip test_trans_4d**

Find the `test_trans_4d` skip entry (the audit cited conftest_metal.py ~line 204, in `UNIMPLEMENTED_FEATURES` or `SKIP_TESTS`). Comment it out with a dated rationale referencing the design spec, mirroring the file's `# Enabled (date): ...` convention.

- [ ] **Step 2: Run the upstream corpus (serial)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
bash scripts/run_upstream_test.sh "unit/language/test_core.py::test_trans_4d" -q 2>&1 | tail -8
```
Expected: 96 passed, 0 failed (all 24 perms × 2 shapes × 2 dtypes). Report the exact count. If any fail, diagnose the index map for that perm/shape; do NOT re-skip to hide a failure.

- [ ] **Step 3: Ratchet — project suite green**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/ -q -k "not test_mept_m2_bug2_gpu and not test_mept_m3a_itercarry_gpu and not test_mept_m3c_gt1024_gpu" 2>&1 | tail -4
```
Expected: 0 failed (the new test_nd_transpose tests included). Note: the autotuner cache-pollution flake is pre-existing (passes in isolation) — if it appears, re-run it alone to confirm it's not this change.

- [ ] **Step 4: Commit**

```bash
git add scripts/conftest_metal.py
git commit -m "test(phase3): un-skip test_trans_4d — generic N-D transpose now passes

96 cases (24 perms x 2 shapes x int32/int8) via the closed-form copy template.
Upstream test_core ratchets up.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** detector for `load(N-D)→trans→[reshape]→store` no-reduce/no-cf (Task 1 Step 3); closed-form template with strided loop for >1024 (Step 4); wired after the specific detectors (Step 5); `_lower_tt_trans` refusal untouched as backstop; un-skip + ratchet (Task 2). ✓

**Placeholder scan:** detector + template code complete; the index-map math is concrete (Python computes literal strides). Task 2 Step 1 requires reading the actual conftest entry text (a read instruction). ✓

**Type consistency:** `info` dict keys (`input_arg`/`output_arg`/`elem_type`/`src_shape`/`order`/`total`) produced by `_detect_nd_trans` and consumed by `_lower_nd_trans_template` match. `_parse_trans_order(trans_ssa, rank)` signature matches its definition. `_row_major_strides`/`dst_shape`/`order` consistent. The template's arg-decl + loop structure mirrors `_lower_transpose_via_reshape_template` exactly. The 2-D reduction check (order=[1,0] → `(k%M)*N+(k/M)`) confirms the formula. ✓
