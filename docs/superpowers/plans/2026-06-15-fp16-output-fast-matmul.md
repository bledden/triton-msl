# fp16-output Fast Matmul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shipped fast-matmul routing to fp16 output (`half* C`) so fp16-in→fp16-out matmuls (the common transformer pattern) get the zero-copy fast path instead of the copy-bound generic fallback.

**Architecture:** Add an `out_dtype` param to `make_simdgroup_matmul_kernel_fast`; `out_dtype="fp16"` emits `half* C` + a per-block cast epilogue (store float accumulator → threadgroup scratch → cast→half → write), float accumulation preserved; `out_dtype="fp32"` (default) is byte-identical to today. The detector's output-dtype gate accepts fp16 (in addition to fp32) and passes `out_dtype` through. The launcher is unchanged (same descriptor shape + dispatch contract). Probe-confirmed: relerr 0.0, 11.9 TFLOP/s @ fp16 2048³.

**Tech Stack:** Python, Metal Shading Language (`simdgroup_matrix`), PyTorch MPS (`torch.mps.compile_shader`), pytest.

**Key facts (do not re-derive):**
- The current template (`triton_msl/codegen/_msl_templates.py:3547`) accumulates in `simdgroup_float8x8` and stores via `simdgroup_store(c{r}_{c}, C /*float* */, N)`. Metal CANNOT down-convert `float8x8→half*` (both a direct store and a `simdgroup_half8x8(float8x8)` cast fail to compile — verified). The result must be read out via `simdgroup_store` to threadgroup memory, then cast per-element.
- Proven epilogue (relerr 0.0 @ 512³/2048³/non-square; 11.9 TFLOP/s @ fp16 2048³): per accumulator block `c{r}_{c}`, `simdgroup_store(c{r}_{c}, scratch + sgitg*64u, 8)` → `simdgroup_barrier(threadgroup)` → lanes write `C[(row_base+{r*8}+i/8)*N + col0 + {c*8} + i%8] = half(scratch[sgitg*64 + i])` for `i=tiisg; i<64; i+=32` → `simdgroup_barrier(threadgroup)`.
- The detector `_maybe_fast_matmul_descriptor` (`_lowerer_templates.py`) already reads `out_dtype = _mlir_to_triton_dtype(args[2].elem_type)` and currently returns None unless it's fp32. Descriptor tuple shape `(fast_msl, 3, 4, 5, 32, 128)` stays the same — only the embedded MSL string differs by output dtype.
- The launcher (`driver.py`) needs NO change: same `simdgroup_matmul_fast` entry name, same `n_groups`/128-threads/`M%32 N%32 K%8` contract; `compile_shader` binds the half C tensor to the `half*` buffer zero-copy. The variant is fixed at compile time from the IR's output dtype.

**Operational rules (all tasks):**
- Run every command from the worktree root `/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread` — NEVER the main repo. Use `python3` (NOT `python`).
- Before any GPU test RUN: `rm -rf ~/.cache/triton_msl ~/.triton/cache`. Serial GPU (no xdist).
- The fp32-out path MUST stay provably untouched — Task 1's golden test + the existing fp32 tests are the proof.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `out_dtype` param on the fast template + golden characterization test

**Files:**
- Modify: `triton_msl/codegen/_msl_templates.py` (`make_simdgroup_matmul_kernel_fast`, ~lines 3547-3625)
- Create: `tests/golden/simdgroup_matmul_fast_fp16in_fp32out.msl`, `tests/golden/simdgroup_matmul_fast_fp32in_fp32out.msl`
- Test: `tests/test_fast_matmul_template.py`

- [ ] **Step 1: Capture the pre-change golden MSL (proves fp32-out stays byte-identical)**

Before touching the template, run this to snapshot the CURRENT output for both input dtypes (fp32 output is the only output today):
```bash
cd /Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
mkdir -p tests/golden
python3 - <<'PY'
from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast
open("tests/golden/simdgroup_matmul_fast_fp16in_fp32out.msl","w").write(
    make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4))
open("tests/golden/simdgroup_matmul_fast_fp32in_fp32out.msl","w").write(
    make_simdgroup_matmul_kernel_fast(dtype="fp32", rr=4, rc=4))
print("golden written")
PY
```
Expected: `golden written`, two files created.

- [ ] **Step 2: Write the failing test**

Create `tests/test_fast_matmul_template.py`:
```python
"""Template-level tests for make_simdgroup_matmul_kernel_fast's out_dtype param.
The fp32-out path must stay BYTE-IDENTICAL to the pre-change golden (no regression);
the fp16-out path must declare half* C + the cast epilogue. No GPU needed."""
import os
from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

GOLD = os.path.join(os.path.dirname(__file__), "golden")


def test_fp32_out_byte_identical_to_golden():
    # Default out_dtype (and explicit "fp32") reproduce the pre-change output exactly.
    for in_dt, fname in [("fp16", "simdgroup_matmul_fast_fp16in_fp32out.msl"),
                         ("fp32", "simdgroup_matmul_fast_fp32in_fp32out.msl")]:
        golden = open(os.path.join(GOLD, fname)).read()
        assert make_simdgroup_matmul_kernel_fast(dtype=in_dt, rr=4, rc=4) == golden
        assert make_simdgroup_matmul_kernel_fast(dtype=in_dt, rr=4, rc=4, out_dtype="fp32") == golden


def test_fp16_out_has_half_C_and_cast_epilogue():
    msl = make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4, out_dtype="fp16")
    assert "device half* C [[buffer(2)]]" in msl
    assert "threadgroup float scratch[4 * 64];" in msl
    assert "uint tiisg [[thread_index_in_simdgroup]]" in msl
    assert "half(scratch[sgitg*64u + i])" in msl
    # accumulators stay float (precision); no direct float-store to C remains.
    assert "simdgroup_float8x8 c0_0(0)" in msl
    assert "simdgroup_store(c0_0, C +" not in msl


def test_bad_out_dtype_raises():
    import pytest
    with pytest.raises(ValueError):
        make_simdgroup_matmul_kernel_fast(dtype="fp16", out_dtype="bf16")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fast_matmul_template.py -v`
Expected: FAIL — `make_simdgroup_matmul_kernel_fast()` has no `out_dtype` kwarg (TypeError) on the fp16-out + fp32-explicit cases.

- [ ] **Step 4: Add `out_dtype` to the template (fp32 byte-identical, fp16 = cast epilogue)**

In `triton_msl/codegen/_msl_templates.py`, change the signature and body of `make_simdgroup_matmul_kernel_fast`. Replace the signature line:
```python
def make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4):
```
with:
```python
def make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4, out_dtype="fp32"):
```
After the existing `in_t/in_frag/pad` dtype block (the `if dtype in (...)` block ending with the `raise ValueError`), add the output-dtype block:
```python
    if out_dtype in ("fp32", "f32"):
        out_t = "float"
    elif out_dtype in ("fp16", "f16"):
        out_t = "half"
    else:
        raise ValueError(f"fast matmul out_dtype supports fp16/fp32, got {out_dtype}")
```
Keep `accs`, `bdecl`, `loads_b`, `inner` EXACTLY as they are. Replace the `stores = ...` assignment and the final `return f"""..."""` with output-dtype-aware construction. The fp32 branch must reproduce the current strings exactly; the fp16 branch adds the param, the scratch, and the epilogue:
```python
    if out_dtype in ("fp16", "f16"):
        _tiisg_param = ",\n    uint tiisg [[thread_index_in_simdgroup]]"
        _scratch_line = "    threadgroup float scratch[4 * 64];\n"
        _epi = []
        for r in range(rr):
            for c in range(rc):
                _epi.append(f"simdgroup_store(c{r}_{c}, scratch + sgitg*64u, 8);")
                _epi.append("simdgroup_barrier(mem_flags::mem_threadgroup);")
                _epi.append(
                    f"for (uint i = tiisg; i < 64u; i += 32u) {{ "
                    f"C[(row_base + {r * 8}u + i / 8u) * N + col0 + {c * 8}u + i % 8u] "
                    f"= half(scratch[sgitg*64u + i]); }}")
                _epi.append("simdgroup_barrier(mem_flags::mem_threadgroup);")
        stores = "\n    ".join(_epi)
    else:
        _tiisg_param = ""
        _scratch_line = ""
        stores = "\n    ".join(
            f"simdgroup_store(c{r}_{c}, C + (row_base + {r * 8}u) * N + col0 + {c * 8}u, N);"
            for r in range(rr) for c in range(rc))

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul_fast(
    device const {in_t}* A [[buffer(0)]],
    device const {in_t}* B [[buffer(1)]],
    device {out_t}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]]{_tiisg_param}
) {{
    uint ntc = (N + {32 * rc - 1}u) / {32 * rc}u;
    uint row_base = (pid / ntc) * {8 * rr}u;
    uint col0 = (pid % ntc) * {32 * rc}u + sgitg * {8 * rc}u;
    if (col0 >= N) return;   // partial column tile: this simdgroup is OOB (uniform)
{_scratch_line}    {accs}
    {in_frag} a_frag, {bdecl};
    for (uint k = 0u; k < K; k += 8u) {{
        {loads_b}
        {inner}
    }}
    {stores}
}}
"""
```
CRITICAL byte-identical check: for `out_dtype="fp32"`, `out_t="float"` → `device float* C`; `_tiisg_param=""` → the sgitg line is unchanged; `_scratch_line=""` → `{_scratch_line}    {accs}` becomes `    {accs}` with the SAME single leading newline as before. If the golden test fails on whitespace, diff and align (the prior body had exactly one `\n` before `    {accs}`).

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fast_matmul_template.py -v`
Expected: PASS (3 tests). If `test_fp32_out_byte_identical_to_golden` fails, the refactor introduced a whitespace diff in the fp32 path — diff `make_simdgroup_matmul_kernel_fast(dtype="fp16")` against the golden file and fix the f-string until identical. Do NOT regenerate the golden (it is the pre-change reference).

- [ ] **Step 6: Commit**

```bash
git add triton_msl/codegen/_msl_templates.py tests/test_fast_matmul_template.py tests/golden/
git commit -m "feat(fast-matmul): out_dtype param — fp16 output via cast epilogue

out_dtype='fp16' emits half* C + per-block store->scratch->cast->write epilogue
(float accumulation preserved); out_dtype='fp32' (default) byte-identical to prior
output (golden characterization test). Entry name + dispatch contract unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Detector accepts fp16 output + CODEGEN_VERSION bump

**Files:**
- Modify: `triton_msl/codegen/_lowerer_templates.py` (`_maybe_fast_matmul_descriptor` output-dtype gate)
- Modify: `triton_msl/__init__.py:6` (CODEGEN_VERSION)
- Modify: `tests/test_fast_matmul_detect.py` (update the fp16-output test)

- [ ] **Step 1: Update the detector test (TDD — the behavior flips)**

In `tests/test_fast_matmul_detect.py`, REPLACE the existing `test_fp16_output_no_descriptor` with a test asserting the descriptor IS now emitted for fp16 output, with the half-output variant MSL:
```python
@requires
def test_fp16_output_emits_half_variant_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT
    _run(_build(".to(tl.float16)"), A, B, C, M, N, K)
    descs = _descriptors()
    assert descs, "fp16-output matmul must now emit a fast_matmul descriptor"
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = descs[0]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "device half* C [[buffer(2)]]" in msl          # the fp16-output variant
    assert "half(scratch[sgitg*64u + i])" in msl
```
Keep `test_eligible_fp32_emits_descriptor`, `test_abbreviated_name_emits_descriptor`, and `test_flag_off_no_descriptor` as-is. (If a `test_bf16_output_no_descriptor` does not already exist, add one to lock the bf16-out fallback — see Step 2's note.)

Add a bf16-output fall-back guard test (bf16 out stays ineligible):
```python
@requires
def test_bf16_output_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="mps", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="mps", dtype=torch.bfloat16)
    _run(_build(".to(tl.bfloat16)"), A, B, C, M, N, K)
    assert not _descriptors(), "bf16-output matmul must NOT emit a descriptor (deferred)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache && python3 -m pytest tests/test_fast_matmul_detect.py::test_fp16_output_emits_half_variant_descriptor -v`
Expected: FAIL — no descriptor emitted yet for fp16 output (the gate still returns None for non-fp32 output).

- [ ] **Step 3: Extend the detector's output-dtype gate**

In `triton_msl/codegen/_lowerer_templates.py`, in `_maybe_fast_matmul_descriptor`, find the output-dtype check (currently):
```python
        # Output must be fp32 (template always declares `device float* C`).
        out_dtype = _mlir_to_triton_dtype(args[2].elem_type)
        if out_dtype not in ("fp32", "f32", "float"):
            return None
```
Replace it with a mapping to the template's `out_dtype` (fp32 or fp16; anything else → None):
```python
        # Output dtype selects the template variant: fp32 -> direct float* store,
        # fp16 -> half* C + cast epilogue (float accumulation preserved either way).
        # bf16 / other output -> ineligible (fall back to the generic kernel).
        out_dtype_t = _mlir_to_triton_dtype(args[2].elem_type)
        if out_dtype_t in ("fp32", "f32", "float"):
            msl_out = "fp32"
        elif out_dtype_t in ("fp16", "f16"):
            msl_out = "fp16"
        else:
            return None
```
Then find the template build call (currently):
```python
        fast_msl = make_simdgroup_matmul_kernel_fast(dtype=msl_dtype, rr=rr, rc=rc)
```
and pass the output dtype:
```python
        fast_msl = make_simdgroup_matmul_kernel_fast(dtype=msl_dtype, rr=rr, rc=rc, out_dtype=msl_out)
```
(The `in_dtype → msl_dtype` block between these stays unchanged.)

- [ ] **Step 4: Bump CODEGEN_VERSION**

In `triton_msl/__init__.py:6`, change `CODEGEN_VERSION = "2026.06.15.1"` to `CODEGEN_VERSION = "2026.06.15.2"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache && python3 -m pytest tests/test_fast_matmul_detect.py -v`
Expected: PASS (all — fp32 + abbreviated + fp16-output-now-emits + bf16-no-descriptor + flag-off).

- [ ] **Step 6: Commit**

```bash
git add triton_msl/codegen/_lowerer_templates.py triton_msl/__init__.py tests/test_fast_matmul_detect.py
git commit -m "feat(fast-matmul): detector accepts fp16 output (half variant), CODEGEN bump

_maybe_fast_matmul_descriptor maps output dtype to the template variant: fp32->
direct store, fp16-> half* + cast epilogue. bf16/other out still falls back.
Descriptor shape unchanged; launcher unchanged. CODEGEN_VERSION .1->.2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Numeric parity for fp16-output

**Files:**
- Modify: `tests/test_fast_matmul_parity.py`

- [ ] **Step 1: Add the fp16-output parity test**

Append to `tests/test_fast_matmul_parity.py` a test that runs an fp16-in→fp16-out matmul (half C tensor) through the fast path (flag on) and the generic kernel (flag off), asserting both match torch and each other. Reuse the file's existing `mm` kernel and seeded `_run` if compatible; if the existing `_run` hardcodes a float32 C, add an `out_dtype` arg to it. Concretely add:
```python
def _run_f16out(M, N, K, flag, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", flag)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
             C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (512, 512, 512),
                                   (256, 2080, 256), (1024, 512, 256)])
def test_fp16_output_parity(M, N, K, monkeypatch):
    A, B, C_on = _run_f16out(M, N, K, "1", monkeypatch)
    ref = (A.float() @ B.float()).half()
    torch.testing.assert_close(C_on, ref, rtol=2e-2, atol=2e-2)
    _, _, C_off = _run_f16out(M, N, K, "0", monkeypatch)
    torch.testing.assert_close(C_on, C_off, rtol=2e-2, atol=2e-2)
```
IMPORTANT: the `mm` kernel here must store fp16 output. If the file's existing module-level `mm` stores fp32 (`tl.store(c_ptrs, acc)`), add a SECOND kernel `mm_f16` that does `tl.store(c_ptrs, acc.to(tl.float16))` and call it in `_run_f16out`. Verify which by reading the file; do not assume.

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/test_fast_matmul_parity.py -k fp16_output -v`
Expected: PASS (4). If any case exceeds fp16 tol (2e-2), STOP and report the actual max error — do NOT loosen the tolerance (it would mean the epilogue is wrong).

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_matmul_parity.py
git commit -m "test(fast-matmul): fp16-output numeric parity (vs torch + flag on==off)

fp16-in->fp16-out across aligned square + non-square; fast path matches torch
(fp16 tol) and the generic kernel.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Gate-logic for fp16-output (fires when aligned, falls back when not)

**Files:**
- Modify: `tests/test_fast_matmul_gate.py`

- [ ] **Step 1: Add fp16-output gate tests (dispatch spy)**

Append to `tests/test_fast_matmul_gate.py` (it already has `_spy`, and an `mm` kernel). Add an fp16-output kernel + tests that the fast path fires for aligned fp16-out MPS matmuls and falls back when misaligned:
```python
@triton.jit
def mm_f16(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc.to(tl.float16))


def _launch_f16(M, N, K):
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm_f16[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
def test_fp16out_aligned_fires_fast(monkeypatch):
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch_f16(256, 256, 256)
    assert "simdgroup_matmul_fast" in seen
    torch.testing.assert_close(C, (A.float() @ B.float()).half(), rtol=2e-2, atol=2e-2)


@requires
@pytest.mark.parametrize("M,N,K", [(258, 256, 256), (256, 258, 256), (256, 256, 252)])
def test_fp16out_misaligned_falls_back(monkeypatch, M, N, K):
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch_f16(M, N, K)
    assert "simdgroup_matmul_fast" not in seen
    torch.testing.assert_close(C, (A.float() @ B.float()).half(), rtol=2e-2, atol=2e-2)
```
(If `_spy` / imports differ in the actual file, adapt to match — read the file first.)

- [ ] **Step 2: Run the test**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache && python3 -m pytest tests/test_fast_matmul_gate.py -k fp16out -v`
Expected: PASS (1 + 3). Aligned fp16-out dispatches `simdgroup_matmul_fast`; misaligned does not; all results match torch.

- [ ] **Step 3: Run the FULL gate + parity + detect + template suite (no regression)**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache && python3 -m pytest tests/test_fast_matmul_gate.py tests/test_fast_matmul_parity.py tests/test_fast_matmul_detect.py tests/test_fast_matmul_template.py tests/test_compile_shader_parity.py -q`
Expected: PASS (all — fp32 cases unchanged, fp16-out cases new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fast_matmul_gate.py
git commit -m "test(fast-matmul): fp16-output gate logic (fires aligned, falls back misaligned)

Dispatch spy: aligned fp16-out MPS matmul now dispatches simdgroup_matmul_fast
(was generic fallback); M%32/N%32/K%8 misalignment falls back. Results match torch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Correctness gate — real test_core dot/matmul subset, ON == OFF (THE gate)

**Files:** none (verification). No perf claim until green.

- [ ] **Step 1: Run the real dot/matmul subset, fp16-output feature on vs off**

The subset takes ~2.4h per config; run in the background. Uses the project harness semantics (raw pytest hits CUDA asserts — see the [[reference_upstream_ratchet]] memory): `--device cpu`, `TRITON_DEFAULT_BACKEND=metal`, worktree `PYTHONPATH`.
```bash
cd /Users/bledden/Documents/triton/python/test
WT=/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
for FAST in 1 0; do
  rm -rf ~/.cache/triton_msl ~/.triton/cache
  echo "=== dot/matmul subset (--device cpu, backend=metal) FAST=$FAST ==="
  TRITON_DEFAULT_BACKEND=metal PYTHONPATH="$WT:$PYTHONPATH" TRITON_MSL_FAST_MATMUL=$FAST TRITON_MSL_MEPT=1 \
    python3 -m pytest unit/language/test_core.py --device cpu -k "dot or matmul" --tb=line -q -rs 2>&1 | tail -4
done
```
Expected: FAST=1 and FAST=0 produce IDENTICAL counts (passed/failed/skipped). The shipped baseline was `593 passed / 1916 failed / 105 skipped`; this run must remain identical FAST=1 vs FAST=0 (the fp16-output matmuls in test_core must not change — they either newly take the fast path with identical results, or were already passing/failing the same way). 0 NEW failures vs FAST=0.

- [ ] **Step 2: If FAST=1 != FAST=0, STOP and diagnose**

A difference means the fp16-output fast path changed a test_core result — a regression. Diagnose the differing test IDs (re-run with `-rA` to list), do NOT proceed to perf. If identical, the gate is GREEN.

- [ ] **Step 3: Project suite (fp16-output feature on)**

```bash
cd /Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
rm -rf ~/.cache/triton_msl ~/.triton/cache
TRITON_MSL_FAST_MATMUL=1 python3 -m pytest tests/ -q -rs 2>&1 | tail -8
```
Expected: 0 failed.

---

### Task 6: Perf gate + baseline record

**Files:**
- Modify: `tests/test_fast_matmul_perf.py`, `reports/perf_baseline.json`

- [ ] **Step 1: Add the fp16-output perf test**

Append to `tests/test_fast_matmul_perf.py` a 2048³ fp16-in→fp16-out throughput test, recording `matmul_2048_fp16out`:
```python
@requires
def test_fast_matmul_fp16out_throughput(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    M = N = K = 2048
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    def fn():
        mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    fn(); torch.mps.synchronize()
    ms = min(do_bench(fn, warmup=25, rep=100, return_mode="min") for _ in range(3))
    tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
    try:
        with open("reports/perf_baseline.json") as f:
            base = json.load(f)
    except Exception:
        base = {}
    base["matmul_2048_fp16out"] = {"name": "matmul_2048_fp16out", "min_ms": round(ms, 4), "tflops": round(tflops, 2)}
    with open("reports/perf_baseline.json", "w") as f:
        json.dump(base, f, indent=2)
    assert tflops >= 7.0, "fp16-out matmul %.2f TFLOP/s < 7.0 floor (>=2x generic ~2.8)" % tflops
```
IMPORTANT: the `mm` kernel used here must store fp16 (`acc.to(tl.float16)` into a half C). Reuse the file's kernel if it does; otherwise add an `mm_f16` kernel (same body, fp16 store) and call it. Read the file to confirm. `import json` if not already imported.

- [ ] **Step 2: Run the perf test**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache && python3 -m pytest tests/test_fast_matmul_perf.py -k fp16out -v`
Expected: PASS. fp16-out ≈ 11-12 TFLOP/s (probe was 11.9). If near the ~2.8 generic floor, the fast path did NOT fire — diagnose with the gate spy, do NOT lower the threshold.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_matmul_perf.py reports/perf_baseline.json
git commit -m "test(fast-matmul): fp16-output perf gate (~11.9 TFLOP/s; >=2x generic)

Records matmul_2048_fp16out. Asserts the fp16-output fast path beats the generic
~2.8 floor by >=2x. Run only after the Task 5 correctness gate is green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Template `out_dtype` param + fp16 cast epilogue + fp32 byte-identical → Task 1. ✓
- Detector output-dtype gate accepts fp16, passes `out_dtype`, bf16 still None → Task 2. ✓
- Launcher unchanged → no task (explicitly: not modified). ✓
- CODEGEN_VERSION bump → Task 2 Step 4. ✓
- Template unit test (fp32 golden + fp16 epilogue) → Task 1. ✓
- Detect test update (fp16-out now emits) + bf16 fallback → Task 2 Step 1. ✓
- Parity fp16-out → Task 3. ✓
- Gate-logic fp16-out fires/falls-back → Task 4. ✓
- Real ratchet ON==OFF → Task 5 (before perf). ✓
- Perf fp16-out + record → Task 6. ✓

**2. Placeholder scan:** No TBD/vague steps. The "read the file to confirm which `mm` kernel stores fp16" instructions (Tasks 3/6) are explicit verification steps with a concrete fallback (add `mm_f16`), not placeholders.

**3. Type consistency:** `out_dtype` param name consistent (template signature, detector call). `msl_out` ∈ {"fp32","fp16"} maps to template `out_dtype`. Descriptor tuple `(fast_msl, 3, 4, 5, 32, 128)` unchanged across tasks. Entry name `simdgroup_matmul_fast` consistent (template, gate spy, launcher — unchanged). Marker strings (`device half* C [[buffer(2)]]`, `half(scratch[sgitg*64u + i])`, `threadgroup float scratch[4 * 64];`) match between the template emission (Task 1) and the assertions (Tasks 1, 2). Alignment gate `M%32/N%32/K%8` consistent with the shipped launcher (unchanged).
