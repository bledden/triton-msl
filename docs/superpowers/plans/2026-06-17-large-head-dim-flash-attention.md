# Large-head_dim FlashAttention-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FlashAttention compute correctly at `HEAD_DIM=128, BLOCK_M=BLOCK_N=32` (fp32 + fp16, causal + non-causal) on the Apple Metal backend, by routing the recognized FA pattern to a new head-dim-tiled FA2 MSL template instead of refusing.

**Architecture:** A fresh head-dim-tiled FA2 MSL template (general strided addressing, fp32-accumulate, fp16 cast epilogue) lives in `_msl_templates.py`. `GenericLowerer.lower()`'s FA prescan stops refusing `head_dim>64` and instead *attempts to route*: it structurally recognizes the canonical FA kernel, extracts Q/K/V/Out pointer roles + their strides + the constexprs, and emits the template for the gated config (`BLOCK=32, HEAD_DIM=128`). Any ambiguity → `MetalNonRecoverableError` (integrity preserved). All other out-of-range configs keep their existing refusals.

**Tech Stack:** Python codegen (`triton_metal/codegen/`), Metal Shading Language (MSL) templates, `torch.mps.compile_shader` for standalone kernel testing, pytest, PyTorch reference.

## Global Constraints

- Never silent-wrong: a routed kernel is correct, or the prescan raises `MetalNonRecoverableError`. Never emit a guessed/approximate kernel. (verbatim project prime directive)
- fp32 accumulation always (softmax + PV), regardless of in/out dtype.
- Validated range only: route `BLOCK_M==BLOCK_N==32 && HEAD_DIM==128`. `BLOCK<32`, `head_dim>128`, and any non-canonical shape keep refusing.
- Existing behavior unchanged: head_dim ≤ 64 @ BLOCK=32 stays on the existing generic path; FA suite, full project suite, and `test_core` ratchet (5,559 / 0) must not regress.
- Bump `CODEGEN_VERSION` (`triton_metal/__init__.py`) on any emitter/lowerer change; clear `~/.cache/triton_metal` + `~/.triton/cache` (via `find -delete`, not `rm -rf`) when verifying codegen.
- Run upstream tests via `scripts/run_upstream_tests.py` (loads `-p conftest_metal`, `--device cpu`).
- Worktree commits are fine; do NOT push without explicit user confirmation.
- Reference layout: real kernel is `_flash_attn_fwd` in `tests/test_flash_attention.py` — Q/K/V/Out `[Z,H,N_CTX,head_dim]`, 16 strides, `Z/H/N_CTX`, constexprs `BLOCK_M/BLOCK_N/HEAD_DIM/IS_CAUSAL`, grid `(N_CTX//BLOCK_M, Z*H)`, `qk_scale = 1/sqrt(HEAD_DIM)`.

---

## File Structure

- `triton_metal/codegen/_msl_templates.py` — **add** `make_flash_attention_kernel_tiled(head_dim, BLOCK_M, BLOCK_N, Dc, causal, out_dtype)`. New tiled FA2 template; does not touch the existing `make_flash_attention_kernel` demo.
- `triton_metal/codegen/generic_lowerer.py` — **modify** the FA prescan in `lower()` (currently `triton_metal/codegen/generic_lowerer.py` ~line 643–675, the `_fa_maxdim/_fa_mindim` guard) to route-or-refuse; **add** `_detect_flash_attention()` and `_lower_flash_attention_template()` (mirror `_detect_matmul_softmax` / `_lower_matmul_softmax_template`; reuse `_resolve_dot_ptr_roles` for pointer roles).
- `tests/test_fa_tiled_template.py` — **create**. Standalone template parity (compile the raw MSL via `torch.mps.compile_shader`, compare to torch). Golden-structure checks.
- `tests/test_flash_attention.py` — **modify**. Add `head_dim=128 @ BLOCK=32` parametrized correctness tests (fp32/fp16 × causal/non-causal) + a near-miss-refuses test.
- `triton_metal/__init__.py` — **modify** `CODEGEN_VERSION`.
- `docs/SUPPORTED_OPS.md`, `README.md` — **modify** the FA row/refusal-catalog/Attention claims.

Precedents to read before implementing (do not reinvent):
- `_msl_templates.py:1614` `make_flash_attention_kernel` (FA2 structure, threadgroup-mem idioms, causal mask block).
- `_msl_templates.py:~3547` `make_simdgroup_matmul_kernel_fast` (fp16 cast epilogue: `out_dtype` param + `half*` buffers + scratch cast).
- The `matmul_softmax` detection + `_lower_matmul_softmax_template` (pattern → emit full MSL kernel; how buffers/strides are bound for a generic kernel).
- `_resolve_dot_ptr_roles` (pointer-role resolution + positional fallback + refuse-on-failure) used by the fast-matmul detector.
- `tests/test_fast_matmul_parity.py` and `tests/test_fast_matmul_template.py` (compile_shader parity + golden-marker test patterns).

---

## Task 1: Standalone tiled FA2 template — fp32, non-causal

Develop the template test-first: a concrete parity test pins correctness; the MSL body is iterated until it passes (legitimate TDD for a GPU kernel).

**Files:**
- Create: `triton_metal/codegen/_msl_templates.py` (add `make_flash_attention_kernel_tiled`)
- Test: `tests/test_fa_tiled_template.py`

**Interfaces:**
- Produces: `make_flash_attention_kernel_tiled(head_dim:int, BLOCK_M:int, BLOCK_N:int, Dc:int=64, causal:bool=False, out_dtype:str="fp32") -> str` — returns MSL source for an FA2 kernel. Buffer ABI (binding order): `Q,K,V,Out` (pointers), then `Z,H,N_CTX` (uint), then the 16 strides (uint, in Q,K,V,O × z,h,m/n,k order), then `scale` (float). Grid `(N_CTX//BLOCK_M, Z*H)`, threads/threadgroup = `BLOCK_M*BLOCK_N` (1024 at 32×32). Threadgroup mem: `tg_S[BLOCK_M*BLOCK_N]`, `acc[BLOCK_M*head_dim]`, `tg_m/tg_l[BLOCK_M]`, plus per-chunk `tg_Qc[BLOCK_M*Dc]`, `tg_Kc[BLOCK_N*Dc]`, `tg_Vc[BLOCK_N*Dc]`.

- [ ] **Step 1: Write the failing parity test (fp32, non-causal, head_dim=128)**

```python
# tests/test_fa_tiled_template.py
import math, pytest, torch
from triton_metal.codegen._msl_templates import make_flash_attention_kernel_tiled

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

def _ref(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    a = (q * scale) @ k.transpose(-2, -1)
    if causal:
        n = a.shape[-1]
        a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    a = torch.softmax(a, dim=-1)
    return torch.nan_to_num(a, nan=0.0) @ v

@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM", [(1, 1, 64, 128), (1, 2, 96, 128)])
def test_tiled_fa_fp32_noncausal(Z, H, N_CTX, HEAD_DIM):
    BLOCK_M = BLOCK_N = 32
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    src = make_flash_attention_kernel_tiled(HEAD_DIM, BLOCK_M, BLOCK_N, Dc=64,
                                            causal=False, out_dtype="fp32")
    lib = torch.mps.compile_shader(src)
    grid = (N_CTX // BLOCK_M, Z * H, 1)
    tpg = (BLOCK_M * BLOCK_N, 1, 1)
    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    lib.flash_attention(out, q, k, v, Z, H, N_CTX, *s, 1.0 / math.sqrt(HEAD_DIM),
                        grid=grid, threads=tpg)  # exact binding API per compile_shader docs
    ref = _ref(q, k, v, causal=False)
    assert (out - ref).abs().max().item() < 1e-3, (out - ref).abs().max().item()
```

- [ ] **Step 2: Run it; verify it fails** — `pytest tests/test_fa_tiled_template.py -k fp32_noncausal -x -q`. Expected: FAIL (`make_flash_attention_kernel_tiled` not defined / kernel wrong).

- [ ] **Step 3: Implement `make_flash_attention_kernel_tiled` (fp32, non-causal)**

Adapt `make_flash_attention_kernel` (`_msl_templates.py:1614`) to: (a) general strided addressing using the 16 strides (replace the collapsed `head_offset + r*D + c` with `base + z*sz + h*sh + row*srow + col*scol`); (b) **head-dim chunking** — replace the single `tg_K[BC*D]/tg_V[BC*D]/tg_Q[BR*D]` staging with a `for (dc = 0; dc < D; dc += Dc)` loop that stages only `[BLOCK,Dc]` slices when computing `S` (`S[i,j] += Σ_{c<Dc} Qc[i,c]*Kc[j,c]` accumulated across chunks), keeps `acc[BLOCK_M*D]` resident, and accumulates `acc[i,d] += Σ_j P[i,j]*V[j,d]` per chunk. Online-softmax (`tg_m`, `tg_l`, `alpha` rescale) is unchanged from the existing template. Threads = `BLOCK_M*BLOCK_N`; strided loops `for (i=lid; i<N; i+=tpg)` as in the existing template. Budget check (32×32, D=128, Dc=64, fp32): S 4KB + acc 16KB + Qc 8KB + Kc 8KB + Vc 8KB + m/l ≈ 44KB — **if >32KB, reduce `Dc` to 32** (Qc/Kc/Vc → 4KB each, total ≈ 32KB; if still tight, stage V per-chunk only). Iterate `Dc` until the test passes within budget.

- [ ] **Step 4: Run until it passes** — `pytest tests/test_fa_tiled_template.py -k fp32_noncausal -x -q`. Expected: PASS (max err < 1e-3). If OOR, lower `Dc`. If wrong numbers, check the chunked-`S` accumulation and per-chunk `acc` update.

- [ ] **Step 5: Add a golden-structure test** (cheap, no GPU)

```python
def test_tiled_fa_emits_chunk_loop():
    src = make_flash_attention_kernel_tiled(128, 32, 32, Dc=64, causal=False, out_dtype="fp32")
    assert "device const float* Q" in src and "device float* Out" in src
    assert "for (uint dc = 0" in src   # head-dim chunk loop present
    assert "threadgroup float acc" in src
```

- [ ] **Step 6: Commit** — `git add triton_metal/codegen/_msl_templates.py tests/test_fa_tiled_template.py && git commit -m "feat(flash-attn): head-dim-tiled FA2 MSL template (fp32, non-causal)"`

---

## Task 2: FA pattern detection + param extraction (refuse-on-ambiguity)

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (add `_detect_flash_attention`)
- Test: `tests/test_fa_detect.py` (create)

**Interfaces:**
- Produces: `GenericLowerer._detect_flash_attention(self) -> dict | None` — returns `None` if not an FA pattern; a dict `{q,k,v,out (arg indices), strides:{q:[4],k:[4],v:[4],o:[4]}, Z,H,N_CTX (arg indices or constexpr), block_m, block_n, head_dim, causal:bool, scale, out_dtype}` if fully resolved; raises `MetalNonRecoverableError` if it *looks* like FA (≥2 dot + exp + max) but any field can't be resolved unambiguously.

- [ ] **Step 1: Write the failing detection test** (uses the real `_flash_attn_fwd` IR)

```python
# tests/test_fa_detect.py — build the IRGraph for _flash_attn_fwd at head_dim=128,
# BLOCK=32, assert _detect_flash_attention returns a dict with head_dim==128,
# block_m==block_n==32, causal in {True,False}, and 4 distinct pointer roles.
# (Construct the lowerer the same way tests/test_fast_matmul_detect.py builds it.)
```
Write it concretely following `tests/test_fast_matmul_detect.py`'s harness (compile the kernel to TTGIR, build the `GenericLowerer`, call the detector).

- [ ] **Step 2: Run; verify it fails** — `pytest tests/test_fa_detect.py -x -q`. Expected: FAIL (method missing).

- [ ] **Step 3: Implement `_detect_flash_attention`** — reuse `_resolve_dot_ptr_roles` to map the two dots' operands to Q,K (dot 1) and V (dot 2), and the store target to Out. For each pointer, walk its `addptr` chain (same tracing the existing prescan/`_extract_shape` uses) to collect the 4 stride args and confirm a `[BLOCK,HEAD_DIM]` access. Read `BLOCK_M/BLOCK_N/HEAD_DIM/IS_CAUSAL` from constexprs, `scale`, and the element dtype (→ `out_dtype`). Return the dict; if `≥2 dot + exp + max` holds but ANY of {4 roles, 16 strides, the constexprs} is unresolved → `raise MetalNonRecoverableError("FlashAttention recognized but <field> could not be resolved; refusing rather than guess")`.

- [ ] **Step 4: Run until pass** — `pytest tests/test_fa_detect.py -x -q`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(flash-attn): FA pattern detection + param extraction (refuse-on-ambiguity)"`

---

## Task 3: Route detected FA → emit template (fp32, non-causal end-to-end)

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (the FA prescan in `lower()`; add `_lower_flash_attention_template`)
- Modify: `triton_metal/__init__.py` (`CODEGEN_VERSION`)
- Test: `tests/test_flash_attention.py`

**Interfaces:**
- Consumes: `_detect_flash_attention()` (Task 2), `make_flash_attention_kernel_tiled()` (Task 1).
- Produces: `_lower_flash_attention_template(info: dict) -> str` — builds the MSL via the template, binding buffers in the kernel's actual arg order (mirror `_lower_matmul_softmax_template`'s binding).

- [ ] **Step 1: Write the failing end-to-end test** — in `tests/test_flash_attention.py`, add `head_dim=128 @ BLOCK=32` to a new parametrized `test_non_causal_large_head` (fp32), asserting `(out-ref).abs().max() < 0.01` (same bar as the existing `test_non_causal`).

- [ ] **Step 2: Run; verify it fails** — `pytest tests/test_flash_attention.py -k large_head -x -q`. Expected: currently FAILS via the head_dim>64 refusal (`MetalNonRecoverableError`).

- [ ] **Step 3: Wire routing in `lower()`** — at the FA prescan: call `info = self._detect_flash_attention()`. If `info` and `info["block_m"]==info["block_n"]==32 and info["head_dim"]==128`, `return self._lower_flash_attention_template(info)`. Keep the `maxdim>64` refusal for head_dim>128 and any non-routed head_dim>64; keep the `mindim<32` refusal. Implement `_lower_flash_attention_template` to call `make_flash_attention_kernel_tiled(head_dim, 32, 32, Dc, causal=info["causal"], out_dtype="fp32")` and bind args in the kernel's order. Bump `CODEGEN_VERSION` to `2026.06.17.2`.

- [ ] **Step 4: Run until pass** — clear caches (`find ~/.cache/triton_metal ~/.triton/cache -type f -delete`), then `pytest tests/test_flash_attention.py -k large_head -x -q`. Expected: PASS.

- [ ] **Step 5: Regression check** — `pytest tests/test_flash_attention.py -q` (existing 17 still green). Expected: PASS.

- [ ] **Step 6: Commit** — `git commit -am "feat(flash-attn): route head_dim=128 @ BLOCK=32 to tiled template (fp32, non-causal)"`

---

## Task 4: fp16 support (cast epilogue)

**Files:** Modify `_msl_templates.py` (fp16 branch of the template), `generic_lowerer.py` (pass `out_dtype` from `info`), `tests/test_flash_attention.py`.

- [ ] **Step 1: Failing test** — parametrize `test_non_causal_large_head` over `dtype ∈ {fp32, fp16}`; for fp16 use `dtype=torch.float16` q/k/v/out and tolerance `(out-ref).abs().max() < 0.05` (fp16 output rounding). Run; verify fp16 case fails.

- [ ] **Step 2: Implement fp16 branch** — in `make_flash_attention_kernel_tiled`, when `out_dtype=="fp16"`: declare `Q/K/V/Out` as `device const half*`/`device half*`, promote to `float` on load, keep all compute in fp32, and cast on the final store `Out[...] = half(acc[i,d] / tg_l[i])` (mirror `make_simdgroup_matmul_kernel_fast`'s fp16-out epilogue). Detector (Task 2) already returns `out_dtype` from the IR; pass it through `_lower_flash_attention_template`.

- [ ] **Step 3: Run until pass** — clear caches; `pytest tests/test_flash_attention.py -k large_head -q`. Expected: PASS (fp32 + fp16).

- [ ] **Step 4: Commit** — `git commit -am "feat(flash-attn): fp16 in/out for tiled large-head_dim FA (fp32 accumulate + cast epilogue)"`

---

## Task 5: Causal support

**Files:** Modify `_msl_templates.py` (causal mask in the chunked kernel), `tests/test_flash_attention.py`.

- [ ] **Step 1: Failing test** — add `test_causal_large_head` (head_dim=128 @ BLOCK=32, fp32 + fp16) using `_ref(..., causal=True)`, tolerances 0.01/0.05. Run; verify it fails (causal path not yet emitted).

- [ ] **Step 2: Implement causal mask** — port the existing template's causal block (`_msl_templates.py:1656-1665`): `S[i,j] = (kv_pos <= q_pos && kv_pos < N_CTX) ? dot*scale : -INFINITY`, with `q_pos`/`kv_pos` derived from the strided global indices. Only emitted when `causal=True`.

- [ ] **Step 3: Run until pass** — clear caches; `pytest tests/test_flash_attention.py -k large_head -q`. Expected: PASS (causal + non-causal × fp32 + fp16).

- [ ] **Step 4: Commit** — `git commit -am "feat(flash-attn): causal masking for tiled large-head_dim FA"`

---

## Task 6: Integrity — near-miss refuses, never silent

**Files:** `tests/test_flash_attention.py`.

- [ ] **Step 1: Write the refusal tests** — assert `MetalNonRecoverableError` for: (a) `head_dim=256 @ BLOCK=32` (above the routed cap), (b) the existing `head_dim=128 @ BLOCK<32` (`test_small_block_refuses` already covers BLOCK<32 for hd 32/64 — extend to 128), (c) a hand-built FA-shaped kernel whose Out stride can't be resolved (asserts the detector raises rather than routes). For (c), follow `tests/test_refusal_coverage.py`'s style.

- [ ] **Step 2: Run** — `pytest tests/test_flash_attention.py -k "refus or small_block" -q`. Expected: PASS (all refuse loudly).

- [ ] **Step 3: Commit** — `git commit -am "test(flash-attn): large-head_dim route-or-refuse-never-silent coverage"`

---

## Task 7: Docs + full verification

**Files:** `docs/SUPPORTED_OPS.md`, `README.md`, `docs/audits/2026-06-16-phase5-readiness-audit.md`.

- [ ] **Step 1: Update SUPPORTED_OPS.md** — FA row + refusal-catalog #19: head_dim=128 @ BLOCK=32 (fp32+fp16, causal+non-causal) now **supported**; head_dim>128, other blocks, BLOCK<32 still refused.

- [ ] **Step 2: Update README** — Status + Attention section/table: head_dim 32/64 (generic) and **128** (tiled template) at BLOCK=32.

- [ ] **Step 3: Full project suite** — `find ~/.cache/triton_metal ~/.triton/cache -type f -delete; PYTHONPATH=<worktree> python3 -m pytest tests/ -q`. Expected: all pass (≥724 + new FA tests, 0 failed).

- [ ] **Step 4: test_core ratchet** — `python3 scripts/run_upstream_tests.py --test-file test_core.py --timeout 3600`. Expected: **5,559 passed / 0 failed** (unchanged — FA pattern absent from test_core).

- [ ] **Step 5: Commit** — `git commit -am "docs(flash-attn): document head_dim=128 @ BLOCK=32 support + refresh matrices"`

---

## Self-Review

**Spec coverage:** every spec section maps to a task — detection/routing (T2,T3), tiled template + memory (T1), dtype (T4), causal (T5), testing/integrity (T1,T3,T6), docs/verify (T7), staging order preserved. ✓
**Placeholder scan:** the MSL bodies and the `_detect_fa` harness reference exact precedents by file:line and define exact interfaces/ABI; the parity tests are concrete; `Dc` has a default + a fallback rule. The one inherently-iterative part (tuning the MSL until the parity test passes) is bounded by a concrete test + budget math. ✓
**Type consistency:** `make_flash_attention_kernel_tiled(head_dim,BLOCK_M,BLOCK_N,Dc,causal,out_dtype)`, `_detect_flash_attention()->dict|None`, `_lower_flash_attention_template(info)->str` used consistently across T1–T5. ✓
