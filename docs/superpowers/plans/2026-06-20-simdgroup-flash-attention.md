# simdgroup-MMA FlashAttention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar-FMA head_dim=128 FlashAttention with a `simdgroup_matrix` MMA kernel (~7.4× fp32 / ~8.5× fp16 over scalar; correct), preserving the never-silent-wrong contract via a compile-time contiguity gate, an in-kernel masked tail for runtime N_CTX, and a differential simd==scalar test gate.

**Architecture:** New MSL template `make_flash_attention_kernel_simdgroup` in `_msl_templates.py` (validated in spike `/tmp/fa_v3.py`+`fa_fp16.py`), routed from `_lower_flash_attention_template` when the FA is head_dim=128, block 32×32, fp32/fp16, AND contiguous-innermost-stride. The scalar `make_flash_attention_kernel_tiled` stays as the differential test oracle and the fallback for every case the simd kernel doesn't handle.

**Tech Stack:** Python f-string MSL codegen; Metal `simdgroup_float8x8`/`simdgroup_half8x8` + `simdgroup_load(transpose=)`/`simdgroup_multiply_accumulate`; `torch.mps.compile_shader` test harness; pytest.

## Global Constraints

- Run everything from the worktree: `/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread`.
- Clear caches before any correctness/perf check: `rm -rf ~/.cache/triton_msl ~/.triton/cache` (and the inductor dir for torch.compile tests).
- Prime directive: NEVER silent-wrong. Any case the simd kernel can't prove correct → fall back to `make_flash_attention_kernel_tiled` (handles general strides) or refuse loudly.
- Commits are local-only; do NOT push or open a PR without explicit confirmation. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Buffer ABI of the simd template MUST match `make_flash_attention_kernel_tiled` exactly (Q,K,V,Out @ buffers 0..3; 16 strides @ 4..19; Z,H,N_CTX @ 20..22; scale baked, no scale arg) so `_lower_flash_attention_template`'s existing `arg_decls`/`bindings` machinery binds it unchanged.
- The simd kernel uses **256 threads/threadgroup** (8 SIMD groups), q-tile BLOCK_M=32, internal kv-tile BN=64, head_dim=128. The scalar template uses 1024 threads — so routing to simd MUST set `self.effective_block_size = 256` (not `block_m*block_n`).
- Test tolerances: fp32 ≤ 1e-3 vs torch; fp16 ≤ 5e-2 vs torch (fp16 in / fp32 accumulate gives ~4e-5 in practice).

---

### Task 1: simd FA template — fp32 + fp16, non-causal, aligned

**Files:**
- Modify: `triton_msl/codegen/_msl_templates.py` (add `make_flash_attention_kernel_simdgroup` near `make_flash_attention_kernel_tiled`, ~line 1816)
- Test: `tests/test_fa_simdgroup_template.py` (create)

**Interfaces:**
- Produces: `make_flash_attention_kernel_simdgroup(head_dim=128, BLOCK_M=32, BLOCK_N=64, causal=False, out_dtype="fp32", arg_decls=None, bindings=None, kernel_name="flash_attention", scale=None) -> str` — returns an MSL kernel string. Same ABI/standalone-canonical-form as `make_flash_attention_kernel_tiled`. `out_dtype` ∈ {"fp32","f32","fp16","f16"}. Requires `BLOCK_M==32`, `BLOCK_N==64`, `head_dim % 8 == 0`. This task implements the ALIGNED path (N_CTX % BN == 0, full q-blocks); Task 2 adds boundary handling.

- [ ] **Step 1: Write the failing test** (standalone parity vs torch, fp32 + fp16, non-causal)

```python
# tests/test_fa_simdgroup_template.py
"""Standalone parity tests for the simdgroup-MMA FA template vs a torch reference."""
import math
import pytest
import torch

from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _ref(q, k, v, causal=False):
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(qf.shape[-1])
    a = (qf * scale) @ kf.transpose(-2, -1)
    if causal:
        n = a.shape[-1]
        a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    a = torch.softmax(a, dim=-1)
    return torch.nan_to_num(a, nan=0.0) @ vf


def _launch(lib, name, q, k, v, out):
    Z, H, N_CTX, _ = q.shape
    n_q_blocks = (N_CTX + 31) // 32          # ceil(N_CTX / BLOCK_M=32)
    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    getattr(lib, name)(q, k, v, out, *s, Z, H, N_CTX,
                       threads=(n_q_blocks * 256, Z * H), group_size=(256, 1))


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX", [(1, 1, 64), (1, 2, 128), (1, 8, 256)])
def test_simd_fa_fp32_noncausal(Z, H, N_CTX):
    HEAD_DIM = 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=False, out_dtype="fp32")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out - _ref(q, k, v)).abs().max().item() < 1e-3


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX", [(1, 2, 128), (1, 8, 256)])
def test_simd_fa_fp16_noncausal(Z, H, N_CTX):
    HEAD_DIM = 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    out = torch.empty(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float16)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=False, out_dtype="fp16")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out.float() - _ref(q, k, v)).abs().max().item() < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_template.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'make_flash_attention_kernel_simdgroup'`.

- [ ] **Step 3: Implement the template**

Add this function to `triton_msl/codegen/_msl_templates.py` (immediately before `make_flash_attention_kernel_tiled`). It is the validated spike kernel (`/tmp/fa_v3.py` for fp32, `/tmp/fa_fp16.py` for fp16), merged with an `out_dtype` switch. fp16: half Q/K/V/Out, half P (`tgP`); fp32: float throughout (no `tgP` — P stored in `tg_S`). Accumulator/scores/softmax always fp32.

```python
def make_flash_attention_kernel_simdgroup(head_dim=128, BLOCK_M=32, BLOCK_N=64,
                                          causal=False, out_dtype="fp32",
                                          arg_decls=None, bindings=None,
                                          kernel_name="flash_attention", scale=None):
    """simdgroup_matrix FlashAttention-2 (fp32/fp16, causal/non-causal, head_dim=128).

    Device-direct simdgroup MMA: QK^T via transpose-load, register-resident O
    accumulator with diag-matrix-MMA alpha-rescale + 1/l-normalize, Q staged once,
    V loaded once-per-(col-tile,k) and prefetched. 256 threads / 8 SIMD groups,
    q-tile BLOCK_M=32, kv-tile BLOCK_N=64. ALIGNED ONLY (N_CTX % BN == 0; partial
    q-block handled by the Q-staging zero-pad); Task 2 adds the partial-kv tail.
    Same buffer ABI as make_flash_attention_kernel_tiled.
    """
    import math as _math
    D, BM, BN, NT = head_dim, BLOCK_M, BLOCK_N, 256
    if not (BM == 32 and BN == 64 and D % 8 == 0):
        raise ValueError(f"simd FA requires BLOCK_M=32, BLOCK_N=64, head_dim%8==0 "
                         f"(got {BM},{BN},{D})")
    n_groups = NT // 32                 # 8
    TPG = (D // 8) // n_groups           # O col-tiles per group (=2 for D=128)
    SCALE = float(scale) if scale is not None else 1.0 / _math.sqrt(float(D))
    if out_dtype in ("fp16", "f16"):
        elem, store_cast = "half", lambda e: f"half({e})"
        p_decl, p_store, p_load_t = "half", "half", "simdgroup_half8x8"
        kv_frag = "simdgroup_half8x8"
    elif out_dtype in ("fp32", "f32"):
        elem, store_cast = "float", lambda e: e
        p_decl, p_store, p_load_t = "float", "float", "simdgroup_float8x8"
        kv_frag = "simdgroup_float8x8"
    else:
        raise ValueError(f"out_dtype must be fp32/f32/fp16/f16 (got {out_dtype!r})")
    # Causal mask: kv positions after the query position are -inf before exp.
    guard = "(kv_row <= q_row)" if causal else "true"

    _LOGICAL = ["q_sz", "q_sh", "q_sm", "q_sk", "k_sz", "k_sh", "k_sn", "k_sk",
                "v_sz", "v_sh", "v_sn", "v_sk", "o_sz", "o_sh", "o_sm", "o_sk",
                "Z", "H", "N_CTX"]
    if (arg_decls is None) != (bindings is None):
        raise ValueError("arg_decls and bindings must be provided together")
    if arg_decls is None:
        arg_decls = [f"    device const {elem}* Q [[buffer(0)]]",
                     f"    device const {elem}* K [[buffer(1)]]",
                     f"    device const {elem}* V [[buffer(2)]]",
                     f"    device {elem}* Out [[buffer(3)]]"]
        for i, nm in enumerate(_LOGICAL):
            arg_decls.append(f"    constant uint& {nm} [[buffer({4 + i})]]")
        bindings = {nm: nm for nm in _LOGICAL}
    sig = ",\n".join(arg_decls)
    bind_lines = "\n".join(f"    const uint {nm} = {bindings[nm]};" for nm in _LOGICAL)
    # P operand for the P@V MMA: fp16 uses a half tgP buffer; fp32 reuses tg_S.
    if elem == "half":
        p_buffers = f"    threadgroup half tgP[{BM} * {BN}];"
        p_write = "tgP[r*BN+cj] = half(p);"
        p_src = "tgP"
    else:
        p_buffers = ""
        p_write = "tg_S[r*BN+cj] = p;"
        p_src = "tg_S"

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// simdgroup-MMA FlashAttention-2 ({elem} in/out, fp32 compute, {"causal" if causal else "non-causal"}).
// 256 threads / 8 SIMD groups. Register-resident O; diag-MMA alpha-rescale.
kernel void {kernel_name}(
{sig},
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {{
    const uint BM = {BM}u, BN = {BN}u, D = {D}u, NT = {NT}u, TPG = {TPG}u;
    const float scale = {SCALE!r}f;
{bind_lines}
    uint q_block = pid3.x, zh = pid3.y;
    uint z = zh / H, h = zh % H;
    uint q_start = q_block * BM;
    uint q_base = z*q_sz+h*q_sh, k_base = z*k_sz+h*k_sh, v_base = z*v_sz+h*v_sh, o_base = z*o_sz+h*o_sh;

    threadgroup {elem} tgQ[{BM} * {D}];     // Q staged once (zero-pads OOB q-rows)
    threadgroup float  tg_S[{BM} * {BN}];   // raw scores (fp32 for exp range)
{p_buffers}
    threadgroup float  tg_m[{BM}], tg_l[{BM}], tg_alpha[{BM}];
    threadgroup float  adiag[4 * 64];

    simdgroup_float8x8 o[4][TPG];
    for (uint rb=0u;rb<4u;rb++) for (uint t=0u;t<TPG;t++) o[rb][t]=simdgroup_float8x8(0.0f);

    for (uint i = lid; i < BM*D; i += NT) {{
        uint qr = q_start + i/D;
        tgQ[i] = (qr < N_CTX) ? Q[q_base + qr*q_sm + (i%D)*q_sk] : {elem}(0);
    }}
    if (lid < BM) {{ tg_m[lid]=-INFINITY; tg_l[lid]=0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_kv = N_CTX / BN;             // ALIGNED: N_CTX % BN == 0 (Task 2: + tail)
    for (uint kv_block = 0u; kv_block < n_kv; kv_block++) {{
        uint kv_start = kv_block * BN;
        simdgroup_float8x8 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        {kv_frag} qf, kf;
        for (uint kc = 0u; kc < D; kc += 8u) {{
            simdgroup_load(kf, K + k_base + (kv_start + sgitg*8u)*k_sn + kc*k_sk, k_sn, 0, true);
            simdgroup_load(qf, tgQ + 0u*D + kc, D);  simdgroup_multiply_accumulate(s0, qf, kf, s0);
            simdgroup_load(qf, tgQ + 8u*D + kc, D);  simdgroup_multiply_accumulate(s1, qf, kf, s1);
            simdgroup_load(qf, tgQ + 16u*D + kc, D); simdgroup_multiply_accumulate(s2, qf, kf, s2);
            simdgroup_load(qf, tgQ + 24u*D + kc, D); simdgroup_multiply_accumulate(s3, qf, kf, s3);
        }}
        simdgroup_store(s0, tg_S + 0u*BN + sgitg*8u, BN);
        simdgroup_store(s1, tg_S + 8u*BN + sgitg*8u, BN);
        simdgroup_store(s2, tg_S + 16u*BN + sgitg*8u, BN);
        simdgroup_store(s3, tg_S + 24u*BN + sgitg*8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < BM) {{
            uint r = lid; uint q_row = q_start + r;
            float m_prev=tg_m[r], l_prev=tg_l[r], m_new=m_prev;
            for (uint cj=0u;cj<BN;cj++) {{
                uint kv_row = kv_start + cj;
                float s = {guard} ? (tg_S[r*BN+cj]*scale) : -INFINITY;
                tg_S[r*BN+cj]=s; m_new=max(m_new,s);
            }}
            float alpha=exp(m_prev-m_new); float l_new=l_prev*alpha;
            for (uint cj=0u;cj<BN;cj++) {{
                uint kv_row = kv_start + cj;
                float p = {guard} ? exp(tg_S[r*BN+cj]-m_new) : 0.0f;
                {p_write} l_new+=p;
            }}
            tg_m[r]=m_new; tg_l[r]=l_new; tg_alpha[r]=alpha;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=lid;i<4u*64u;i+=NT) adiag[i]=0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < BM) {{ uint rb=lid/8u, ii=lid%8u; adiag[rb*64u+ii*8u+ii]=tg_alpha[lid]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ad0, ad1, ad2, ad3, tmp;
        simdgroup_load(ad0, adiag + 0u*64u, 8); simdgroup_load(ad1, adiag + 1u*64u, 8);
        simdgroup_load(ad2, adiag + 2u*64u, 8); simdgroup_load(ad3, adiag + 3u*64u, 8);
        {p_load_t} pf, vf, vfs[{BN}/8];
        for (uint t=0u;t<TPG;t++) {{
            uint ct = sgitg + t*{n_groups}u;
            tmp=simdgroup_float8x8(0.0f); simdgroup_multiply_accumulate(tmp, ad0, o[0][t], tmp); o[0][t]=tmp;
            tmp=simdgroup_float8x8(0.0f); simdgroup_multiply_accumulate(tmp, ad1, o[1][t], tmp); o[1][t]=tmp;
            tmp=simdgroup_float8x8(0.0f); simdgroup_multiply_accumulate(tmp, ad2, o[2][t], tmp); o[2][t]=tmp;
            tmp=simdgroup_float8x8(0.0f); simdgroup_multiply_accumulate(tmp, ad3, o[3][t], tmp); o[3][t]=tmp;
            for (uint kk=0u;kk<BN;kk+=8u)
                simdgroup_load(vfs[kk/8u], V + v_base + (kv_start + kk)*v_sn + (ct*8u)*v_sk, v_sn);
            for (uint kk=0u;kk<BN;kk+=8u) {{
                vf = vfs[kk/8u];
                simdgroup_load(pf, {p_src} + 0u*BN + kk, BN);  simdgroup_multiply_accumulate(o[0][t], pf, vf, o[0][t]);
                simdgroup_load(pf, {p_src} + 8u*BN + kk, BN);  simdgroup_multiply_accumulate(o[1][t], pf, vf, o[1][t]);
                simdgroup_load(pf, {p_src} + 16u*BN + kk, BN); simdgroup_multiply_accumulate(o[2][t], pf, vf, o[2][t]);
                simdgroup_load(pf, {p_src} + 24u*BN + kk, BN); simdgroup_multiply_accumulate(o[3][t], pf, vf, o[3][t]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint i=lid;i<4u*64u;i+=NT) adiag[i]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < BM) {{ uint rb=lid/8u, ii=lid%8u; float l=tg_l[lid]; adiag[rb*64u+ii*8u+ii]=(l>0.0f)?(1.0f/l):0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 ld, on;
    for (uint rb=0u;rb<4u;rb++) {{
        simdgroup_load(ld, adiag + rb*64u, 8);
        for (uint t=0u;t<TPG;t++) {{
            uint ct = sgitg + t*{n_groups}u;
            on=simdgroup_float8x8(0.0f);
            simdgroup_multiply_accumulate(on, ld, o[rb][t], on);
            simdgroup_store(on, Out + o_base + (q_start + rb*8u)*o_sm + (ct*8u)*o_sk, o_sm);
        }}
    }}
}}
"""
```

Note for fp16: `simdgroup_store(on, Out, o_sm)` stores a `simdgroup_float8x8` to a `half*` — Metal requires matching types. If the fp16 standalone test FAILS to compile here, change the final store to normalize into a float threadgroup scratch then cast: replace the final `simdgroup_store(on, Out + ...)` block with a store of `on` to a `threadgroup float onbuf[8*8]` (one per group via `adiag`-style reuse) followed by an element-wise `Out[...] = half(onbuf[...])` loop. Keep the float-accumulator path; only the final write casts. (The matmul template stores float→float; we need half out, hence the cast epilogue.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache; PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_template.py -q -p no:cacheprovider`
Expected: PASS (5 tests: 3 fp32 + 2 fp16). If a fp16 compile error appears, apply the cast-epilogue note in Step 3.

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/_msl_templates.py tests/test_fa_simdgroup_template.py
git commit -m "feat(fa): simdgroup-MMA FlashAttention template (fp32+fp16, non-causal, aligned)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Boundary handling — partial kv-block masked tail + causal

**Files:**
- Modify: `triton_msl/codegen/_msl_templates.py` (`make_flash_attention_kernel_simdgroup`)
- Test: `tests/test_fa_simdgroup_template.py` (add cases)

**Interfaces:**
- Consumes: `make_flash_attention_kernel_simdgroup` from Task 1.
- Produces: same signature; now correct for ANY N_CTX (not just multiples of 64) and for `causal=True`. Adds a per-kv-block branch: full blocks use the Task-1 device-direct MMA; the final partial block (`kv_start + BN > N_CTX`) stages K/V into a threadgroup buffer with `kv_row < N_CTX` zero-pad and MMAs from threadgroup; the softmax `kv_row < N_CTX` mask (already added below) zeroes padded columns.

- [ ] **Step 1: Write the failing tests** (unaligned N_CTX + causal)

```python
# append to tests/test_fa_simdgroup_template.py
@requires_mps
@pytest.mark.parametrize("N_CTX", [96, 100, 192, 200])   # not multiples of 64
def test_simd_fa_fp32_unaligned(N_CTX):
    HEAD_DIM, Z, H = 128, 1, 2
    torch.manual_seed(1)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=False, out_dtype="fp32")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out - _ref(q, k, v)).abs().max().item() < 1e-3


@requires_mps
@pytest.mark.parametrize("Z,H,N_CTX", [(1, 2, 128), (1, 4, 192)])
def test_simd_fa_fp32_causal(Z, H, N_CTX):
    HEAD_DIM = 128
    torch.manual_seed(2)
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    src = make_flash_attention_kernel_simdgroup(HEAD_DIM, 32, 64, causal=True, out_dtype="fp32")
    lib = torch.mps.compile_shader(src)
    _launch(lib, "flash_attention", q, k, v, out)
    torch.mps.synchronize()
    assert (out - _ref(q, k, v, causal=True)).abs().max().item() < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache; PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_template.py -k "unaligned or causal" -q -p no:cacheprovider`
Expected: FAIL — unaligned cases give wrong results (last kv-block reads OOB; loop `n_kv = N_CTX/BN` floor-drops the tail). Causal `test_simd_fa_fp32_causal` at N_CTX=128 may pass (aligned) but N_CTX=192 fails (192%64==0 actually — both aligned; causal correctness still needs the guard already in Task 1, so causal may already pass — if so, that confirms causal; keep the test).

- [ ] **Step 3: Add the partial-kv-block masked path**

In `make_flash_attention_kernel_simdgroup`, (a) change the kv-loop bound to `ceil` and (b) branch each block on fullness. Replace `uint n_kv = N_CTX / BN;` and the `for (uint kv_block...)` header region with:

```c
    threadgroup {elem} tgKV[{BN} * {Dc_tail}];   // partial-block staging (Dc_tail-wide chunks)
    uint n_kv = (N_CTX + BN - 1u) / BN;           // ceil — include the partial tail
    for (uint kv_block = 0u; kv_block < n_kv; kv_block++) {{
        uint kv_start = kv_block * BN;
        bool kv_full = (kv_start + BN <= N_CTX);
```

And add `Dc_tail = 32` as a Python local in the generator (`Dc_tail = 32`, with `head_dim % Dc_tail == 0`). For the SCORE phase, wrap the existing device-direct K-load path in `if (kv_full) { ...existing kf load from K device... } else { ...staged kf... }`. The staged-else loads K through `tgKV` per Dc_tail chunk with zero-pad:

```c
            if (kv_full) {{
                simdgroup_load(kf, K + k_base + (kv_start + sgitg*8u)*k_sn + kc*k_sk, k_sn, 0, true);
            }} else {{
                // stage 8 kv-rows x 8 head-cols (this group's slice) with OOB zero-pad
                for (uint e = lid; e < {BN}*8u; e += NT) {{
                    uint rr = e / 8u, cc = e % 8u; uint kvr = kv_start + rr;
                    tgKV[rr*8u + cc] = (kvr < N_CTX) ? K[k_base + kvr*k_sn + (kc+cc)*k_sk] : {elem}(0);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(kf, tgKV + (sgitg*8u)*8u, 8, 0, true);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
```

(The staged buffer here is sized `BN*8` per kc-step; set `Dc_tail=8` to match — adjust the `tgKV[{BN}*8]` declaration accordingly. Budget: tgQ 16KB (fp32) + tg_S 8KB + tgKV BN*8*4=2KB + adiag 1KB ≈ 27KB < 32KB.) Do the analogous `if (kv_full) {...device V...} else {...staged V...}` for the V load in the P@V phase. The softmax `kv_row < N_CTX` mask must apply for partial blocks; extend the causal/`guard` expression to AND with `(kv_row < N_CTX)`:

```python
    guard = "((kv_row < N_CTX) && (kv_row <= q_row))" if causal else "(kv_row < N_CTX)"
```

- [ ] **Step 4: Run to verify pass**

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache; PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_template.py -q -p no:cacheprovider`
Expected: PASS (all fp32/fp16, aligned + unaligned + causal). If the staged-tail path mis-computes, dump the partial block's `tgKV` and compare to a hand gather; the most common bug is the `simdgroup_load(..., 8, 0, true)` stride (must be 8 = the staged tile's row width).

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/_msl_templates.py tests/test_fa_simdgroup_template.py
git commit -m "feat(fa): simd FA correct for runtime N_CTX (masked tail) + causal

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Routing — contiguity gate + dtype, scalar fallback

**Files:**
- Modify: `triton_msl/codegen/generic_lowerer.py` (`_lower_flash_attention_template`, the `make_flash_attention_kernel_tiled(...)` call site ~line 4956, and the `effective_block_size`/`import` lines ~line 4842 & ~line 4976)
- Test: `tests/test_fa_simdgroup_routing.py` (create)

**Interfaces:**
- Consumes: `make_flash_attention_kernel_simdgroup` (Tasks 1–2).
- Produces: `_lower_flash_attention_template` emits the simd kernel when eligible, else the scalar one; sets `self.effective_block_size = 256` for simd, `block_m*block_n` for scalar.

- [ ] **Step 1: Write the failing test** (routing picks simd for contiguous; scalar otherwise)

```python
# tests/test_fa_simdgroup_routing.py
"""Routing: simd FA template chosen for contiguous head_dim=128 FA; scalar otherwise."""
import platform
import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Metal only")


def _emit(causal=False, out_dtype="f32", contiguous=True):
    """Drive _lower_flash_attention_template with a synthetic eligible `info`/graph.
    Returns the emitted MSL string."""
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    # Build the minimal info + graph the routing reads. Reuse the helper the
    # detector tests use; if none exists, construct info dict directly and call
    # the routing branch. (Implementation detail: see _lower_flash_attention_template.)
    raise NotImplementedError  # replaced in Step 3 with the real harness


def test_simd_chosen_for_contiguous_fp32():
    msl = _emit(contiguous=True, out_dtype="f32")
    assert "metal_simdgroup_matrix" in msl and "simdgroup_multiply_accumulate" in msl


def test_scalar_fallback_for_noncontiguous():
    msl = _emit(contiguous=False, out_dtype="f32")
    assert "simdgroup_multiply_accumulate" not in msl  # the scalar tiled template
```

Note: the synthetic-`info` harness depends on `_lower_flash_attention_template`'s internals. If constructing a full graph is impractical, instead assert at the unit level by extracting the eligibility predicate into a tiny pure helper (next step) and testing THAT, plus one end-to-end routed test in Task 5.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_routing.py -q -p no:cacheprovider`
Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Add the eligibility helper + routing branch**

In `generic_lowerer.py`, add a module-level pure helper:

```python
def _simd_fa_eligible(info):
    """True if the detected FA can use the simdgroup template: head_dim=128,
    block 32x32, fp32/fp16, AND contiguous innermost (head-dim) stride for all of
    Q/K/V/Out. info["strides"][role] is [z,h,m,k]; index 3 (k) == "c1" means the
    innermost stride folded to 1 (contiguous). Non-contiguous -> scalar template
    (handles general strides) -> never silent-wrong."""
    if not (info.get("head_dim") == 128 and info.get("block_m") == 32
            and info.get("block_n") == 32 and info.get("out_dtype") in ("f32", "f16")):
        return False
    st = info.get("strides", {})
    return all(st.get(r, [None]*4)[3] == "c1" for r in ("q", "k", "v", "o"))
```

Then in `_lower_flash_attention_template`, replace the `msl = make_flash_attention_kernel_tiled(...)` call with a branch:

```python
        from triton_msl.codegen._msl_templates import (
            make_flash_attention_kernel_tiled,
            make_flash_attention_kernel_simdgroup,
        )
        if _simd_fa_eligible(info):
            msl = make_flash_attention_kernel_simdgroup(
                head_dim, 32, 64, causal=info["causal"], out_dtype=info["out_dtype"],
                arg_decls=arg_decls, bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name), scale=info["scale"])
            self.effective_block_size = 256
        else:
            msl = make_flash_attention_kernel_tiled(
                head_dim, block_m, block_n, Dc=64, causal=info["causal"],
                out_dtype=info["out_dtype"], arg_decls=arg_decls, bindings=bindings,
                kernel_name=_sanitize_msl_name(self.graph.func_name), scale=info["scale"])
            self.effective_block_size = block_m * block_n
        self._used_pid_axes = {0, 1}
        self._prescan_stores()
        return msl
```

Update the test's `_emit` to construct a synthetic `info` and call `_simd_fa_eligible` directly (rename the two tests to `test_eligible_contiguous_fp32` / `test_ineligible_noncontiguous`, asserting `_simd_fa_eligible(info) is True/False`), since the predicate is the routing decision:

```python
def _info(contiguous=True, out_dtype="f32"):
    c = "c1" if contiguous else 5   # non-c1 innermost stride = a runtime arg index
    return {"head_dim": 128, "block_m": 32, "block_n": 32, "out_dtype": out_dtype,
            "causal": False, "scale": 0.0883,
            "strides": {r: ["c1", "c1", 2, c] for r in ("q", "k", "v", "o")}}

def test_eligible_contiguous_fp32():
    from triton_msl.codegen.generic_lowerer import _simd_fa_eligible
    assert _simd_fa_eligible(_info(contiguous=True)) is True

def test_ineligible_noncontiguous():
    from triton_msl.codegen.generic_lowerer import _simd_fa_eligible
    assert _simd_fa_eligible(_info(contiguous=False)) is False
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_routing.py -q -p no:cacheprovider`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/generic_lowerer.py tests/test_fa_simdgroup_routing.py
git commit -m "feat(fa): route contiguous head_dim=128 FA to the simdgroup template

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Differential gate (simd == scalar) + routed end-to-end + full-suite regression

**Files:**
- Test: `tests/test_fa_simdgroup_diff.py` (create)
- Modify: `docs/SUPPORTED_OPS.md` / `README.md` perf note (honest claim: simd FA ~7.4× fp32 / ~8.5× fp16 over scalar at head_dim=128; NOT MFA-competitive)

**Interfaces:**
- Consumes: both templates + routing (Tasks 1–3).

- [ ] **Step 1: Write the differential test** (simd output == scalar output, both vs torch)

```python
# tests/test_fa_simdgroup_diff.py
"""Differential gate: the simd FA template must match the scalar template (the
validated oracle) AND the torch reference, across dtype x causal x alignment."""
import math
import platform
import pytest
import torch

from triton_msl.codegen._msl_templates import (
    make_flash_attention_kernel_simdgroup, make_flash_attention_kernel_tiled)

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()
         and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader")


def _ref(q, k, v, causal):
    qf, kf, vf = q.float(), k.float(), v.float()
    sc = 1.0 / math.sqrt(qf.shape[-1])
    a = (qf*sc) @ kf.transpose(-2, -1)
    if causal:
        n = a.shape[-1]; a = a.masked_fill(torch.tril(torch.ones(n, n, device=a.device)) == 0, float("-inf"))
    return torch.nan_to_num(torch.softmax(a, -1), nan=0.0) @ vf


def _run(src, name, q, k, v, out, threads_pg):
    lib = torch.mps.compile_shader(src)
    Z, H, N, _ = q.shape
    nqb = (N + 31)//32
    s = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    getattr(lib, name)(q, k, v, out, *s, Z, H, N, threads=(nqb*threads_pg, Z*H), group_size=(threads_pg, 1))
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [128, 100, 192])
def test_simd_matches_scalar_and_torch(dt, causal, N):
    HD, Z, H = 128, 1, 2
    od = "f16" if dt == torch.float16 else "f32"
    tol = 5e-2 if dt == torch.float16 else 1e-3
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    k = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    v = torch.randn(Z, H, N, HD, device="mps", dtype=dt)
    ref = _ref(q, k, v, causal)
    o_sd = torch.empty_like(q); o_sc = torch.empty_like(q)
    _run(make_flash_attention_kernel_simdgroup(HD, 32, 64, causal=causal, out_dtype=od),
         "flash_attention", q, k, v, o_sd, 256)
    _run(make_flash_attention_kernel_tiled(HD, 32, 32, Dc=64, causal=causal, out_dtype=od),
         "flash_attention", q, k, v, o_sc, 1024)
    assert (o_sd.float() - ref).abs().max().item() < tol     # simd vs torch
    assert (o_sd.float() - o_sc.float()).abs().max().item() < tol  # simd vs scalar oracle
```

- [ ] **Step 2: Run to verify it passes** (this is the integration gate, not a fail-first unit)

Run: `rm -rf ~/.cache/triton_msl ~/.triton/cache; PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_simdgroup_diff.py -q -p no:cacheprovider`
Expected: PASS (12 cases). Any FAIL is a real correctness bug in Tasks 1–2 — fix there, do not loosen the tolerance.

- [ ] **Step 3: Run the routed @triton.jit FA end-to-end + the full suite**

Run:
```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache /var/folders/*/T/torchinductor_* 2>/dev/null
PYTHONPATH=$(pwd) python3.14 -m pytest tests/test_fa_tiled_template.py tests/test_flash_attention*.py -q -p no:cacheprovider
PYTHONPATH=$(pwd) python3.14 -m pytest tests/ -q -p no:cacheprovider --deselect "tests/test_fast_matmul_perf.py::test_fast_matmul_throughput[dtype1]"
```
Expected: the existing FA tests still pass (now routed through simd for the eligible head_dim=128 cases); full suite green (any pre-existing transformer-convergence flake passes in isolation — re-run it alone 3× to confirm it's the known flake, not a regression).

- [ ] **Step 4: Update the honest perf claim**

In `README.md` (perf section) and `docs/SUPPORTED_OPS.md` FA row, state: "head_dim=128 FlashAttention uses Apple `simdgroup_matrix` MMA: ~7.4× (fp32) / ~8.5× (fp16) faster than the prior scalar path, ~8.2/9.6 TFLOP/s (~70–80% of the in-repo matmul-template peak). Not competitive with Apple metal-flash-attention / MLX in absolute terms." Do NOT overstate.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fa_simdgroup_diff.py README.md docs/SUPPORTED_OPS.md
git commit -m "test(fa): differential simd==scalar gate + honest perf claim

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:**
- Kernel structure (register-O, diag-MMA rescale, transpose-load, Q-stage, V-reuse+prefetch) → Task 1. ✓
- fp32 + fp16 → Task 1 (out_dtype switch + cast epilogue). ✓
- causal + non-causal → Task 2 (guard expr). ✓
- Runtime-N_CTX boundary (partial q-block via Q-stage zero-pad; partial kv-block masked tail) → Task 1 (q zero-pad) + Task 2 (kv tail). ✓
- Contiguity gate + scalar fallback → Task 3 (`_simd_fa_eligible`). ✓
- Routing in `_lower_flash_attention_template`; `effective_block_size=256` → Task 3. ✓
- Differential simd==scalar test gate + routed e2e + full-suite regression → Task 4. ✓
- Scalar template kept as oracle/fallback → Tasks 3–4 (used in both). ✓
- Honest non-goal / perf claim → Task 4 Step 4. ✓
- Budget/compile check → exercised by every compile in Tasks 1–2 (32KB enforced by the pipeline-state creation; a too-large layout fails compile loudly). ✓

**Placeholder scan:** Task 3 Step 1 deliberately starts as a `NotImplementedError` stub that Step 3 replaces with the real `_simd_fa_eligible` unit test — flagged inline, not a latent placeholder. No "TBD"/"add error handling"/"similar to Task N" left.

**Type consistency:** `make_flash_attention_kernel_simdgroup(head_dim, BLOCK_M, BLOCK_N, causal, out_dtype, arg_decls, bindings, kernel_name, scale)` — identical across Tasks 1–4. `_simd_fa_eligible(info)` and `effective_block_size=256` consistent in Task 3 + tests. ABI (Q,K,V,Out@0..3, 16 strides, Z,H,N_CTX, baked scale) matches `make_flash_attention_kernel_tiled` throughout.

**Known open detail to verify during execution:** the fp16 final `simdgroup_store` to a `half*` (Task 1 Step 3 note) — Metal may require the float→half cast-epilogue scratch; the fp16 test in Task 1 Step 4 surfaces this immediately.
