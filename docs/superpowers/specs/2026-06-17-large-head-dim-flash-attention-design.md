# Large-head_dim FlashAttention-2 (head_dim=128 @ BLOCK=32) — design

**Date:** 2026-06-17
**Status:** design approved; pending implementation plan
**Author:** triton-metal (Apple Metal backend)

## Summary

Make FlashAttention compute **correctly at `HEAD_DIM=128`, `BLOCK_M=BLOCK_N=32`** on the
Apple Metal backend, for **fp32 and fp16**, **causal and non-causal**. Today this config
fails with `OutOfResources` (a *loud* failure — the generic lowering stages
`Q/K/V/S/acc` tiles in threadgroup memory and a naive `[32,128]` layout blows the 32 KB
budget at ~68 KB). The fix routes the recognized FlashAttention pattern to a new,
**head-dim-tiled** FA2 MSL template instead of refusing.

This stays **FlashAttention-2**. FA3/FA4 are NVIDIA Hopper/Blackwell hardware co-designs
(WGMMA, TMA, TMEM, async `tcgen05`, warp-specialization, FP8) with no Apple-GPU analog;
the portable contribution is the FA2 *algorithm*, and head_dim tiling extends its tile
coverage. The online softmax runs over the KV/sequence dimension (`BLOCK_N`), **not** over
head_dim, so tiling the head dimension is numerically free — the FA2 math is unchanged.

## Goals / success criteria

- `HEAD_DIM=128`, `BLOCK_M=BLOCK_N=32`, causal + non-causal, fp32 + fp16 (fp16 in /
  fp32 accumulate / fp16 out) computes correctly vs a PyTorch reference at tight
  tolerance (matching the existing FA suite's bar).
- **Integrity preserved throughout:** the routing either emits a correct kernel or
  refuses loudly (`MetalNonRecoverableError`). It never emits a guessed/approximate
  kernel. Every currently-refused config stays refused.
- No regression: existing FA suite (head_dim 32/64 @ BLOCK=32), full project suite, and
  the `test_core` ratchet (5,559/0) unchanged.

## Non-goals (explicit future work)

- Block sizes other than 32 (`BLOCK<32` stays a refused silent-wrong; `BLOCK>32` stays
  refused). Larger blocks for throughput are future work.
- `head_dim` other than {32, 64 (generic), 128 (this template)}; e.g. 96/256 later.
- Performance optimization / autotuning. This is **correctness-first**; perf is a
  follow-up (and Apple FA will not match MLX hand-tuned kernels regardless).
- fp8 / bf16-input matmul (no Apple matrix HW / separate scope).

## Approach (C: fresh tiled template + prescan routing)

The existing `make_flash_attention_kernel` (`_msl_templates.py:1614`) is a simplified
demo — fp32-only, a collapsed `[n_heads*seq_len, head_dim]` layout, 6 buffers
(`Q,K,V,O,seq_len,scale`) — **incompatible** with a real `@triton.jit` FA kernel's ABI
(4 pointers + 16 strides + `Z/H/N_CTX` + `BLOCK_M/N`,`HEAD_DIM`,`IS_CAUSAL` constexprs).
So we build a fresh template using general strided addressing, and route to it via the
prescan (the mechanism the fast-matmul and `matmul_softmax` features already use).

### Component 1 — Detection + safe routing (the crux, and the main risk)

In `GenericLowerer.lower()`, at the FA prescan that currently refuses `head_dim>64`:
instead **attempt to route** the recognized FA pattern; **refuse on any ambiguity**.

- Recognize structurally (extends the fast-matmul `_resolve_dot_ptr_roles` precedent):
  ≥2 `tt.dot` + `exp` + `max`; identify **Q, K** (dot operands), **V** (second-dot
  operand), **Out** (store target).
- For each of Q/K/V/Out, trace the `addptr` chain to extract its **4 strides** and
  confirm the `[BLOCK, HEAD_DIM]` access shape.
- Extract `BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `IS_CAUSAL` (constexprs), `N_CTX`, and the
  `qk_scale`.
- **Routing gate:** emit the template only for `BLOCK_M==BLOCK_N==32 && HEAD_DIM==128`
  (this increment). Anything that does not resolve **unambiguously and completely** →
  `MetalNonRecoverableError`. `BLOCK<32` (silent-wrong) and `head_dim>128` keep their
  existing refusals. head_dim ≤ 64 @ BLOCK=32 continues on the existing (working)
  generic path — untouched.

This component carries the only real correctness risk (a mis-extracted stride would be a
silent-wrong). It is mitigated by: (a) refuse-on-any-ambiguity, (b) reusing the proven
role-resolution code, (c) a dedicated test that a near-miss kernel shape refuses rather
than routes.

### Component 2 — Tiled MSL template + memory strategy

A new template emitting FA2 with the **head dimension tiled** so threadgroup memory stays
under 32 KB. Budget at `BLOCK=32, D=128, fp32` naively ≈ 68 KB; with `Dc`-chunking:

- `S = QKᵀ`: accumulate `S[bm,bn] += Σ_chunk Q[:, d:d+Dc] · K[:, d:d+Dc]` over `D/Dc`
  chunks; only a `Dc`-wide slice of Q and K is staged at once (`Dc=64` → 8 KB each).
- Softmax on `S[32,32]` (4 KB) — standard FA2 online update (`m`, `l`, rescale `alpha`),
  unchanged.
- `acc[BM,D]` (`[32,128]` fp32 = 16 KB) persists across KV blocks; `P@V` accumulates,
  optionally per D-chunk.
- Peak staging stays comfortably < 32 KB.
- **General strided addressing** using the extracted strides (handles the real
  `[Z,H,N_CTX,head_dim]` layout — not the demo's collapsed layout). Grid mirrors the
  Triton kernel: `(N_CTX // BLOCK_M, Z*H)`.
- `Dc=64` chosen as the default chunk (divides 128, keeps Q/K slices at 8 KB). The
  template is parameterized on `Dc` so it can be tuned later.

### Component 3 — dtype handling

- **fp32 accumulate always** (numerical correctness of the softmax + PV).
- `out_dtype` parameter on the template: `fp32` → `device float*` buffers; `fp16` →
  `device half*` buffers, read-and-promote to `float` on load, and a **cast epilogue** on
  the final `acc/l` store (mirrors the shipped fast-matmul fp16-out epilogue).
- The detector reads the operand/output element types from the IR to select `out_dtype`.

### Component 4 — Testing + integrity

- New parametrized tests in `tests/test_flash_attention.py`:
  `HEAD_DIM=128 @ BLOCK=32 × {fp32, fp16} × {causal, non-causal}` vs a PyTorch reference,
  tight tolerance (fp32 ~1e-4..1e-3; fp16 a looser but real bound).
- Keep all existing refusals green: `test_small_block_refuses` (BLOCK<32),
  head_dim>128 refusal, and a new test that a structurally-near-miss FA kernel
  (e.g. an unresolvable stride) **refuses** rather than silently routes.
- Bump `CODEGEN_VERSION`; clear caches in verification.
- Confirm no regression: FA suite, full project suite, `test_core` ratchet (5,559/0).
- Update `SUPPORTED_OPS.md` (FA row + refusal catalog) and `README` to reflect the new
  supported config.

## Staging (for the implementation plan)

1. **Template, standalone-correct:** fp32, non-causal, `head_dim=128 @ BLOCK=32`, verified
   against torch by direct invocation (before any routing).
2. **Routing + extraction:** detect the canonical FA shape, extract params, emit the
   template for the gated config; refuse-on-ambiguity. Verify via the real `@triton.jit`
   kernel.
3. **fp16 + causal:** add the cast epilogue + causal mask; extend tests.
4. **Full verify + docs:** FA suite, project suite, ratchet, docs, CODEGEN bump.

Each stage is independently verifiable and leaves the tree green (integrity contract
holds at every stage — out-of-scope configs refuse).

## Risks

- **Param extraction (Component 1)** is the primary risk: a mis-identified stride is a
  silent-wrong. Mitigation: refuse-on-any-ambiguity + reuse proven role resolution + an
  explicit near-miss-refuses test.
- **Threadgroup budget** for fp32 `acc[32,128]` (16 KB) + per-chunk staging is tight but
  fits; validated empirically in stage 1 before routing.
- **fp16 numerics:** fp32 accumulate keeps error bounded; the test tolerance reflects
  fp16 output rounding, not algorithmic error.
