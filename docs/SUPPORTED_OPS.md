# Supported ops & dtypes — triton-msl

> What lowers, what refuses, what the hardware can't do. **The integrity contract:** a
> kernel we can lower runs correctly; a kernel we cannot is **refused loudly**
> (`MetalNonRecoverableError`) — never silently wrong. So this matrix has three states:
> **✓ supported** (lowers + runs correctly), **✗ refused** (fails loudly with an actionable
> message; never silent-wrong), **— hardware-impossible** (Apple GPUs lack the capability).
> Generated from the codegen refusal catalog (24 guard sites), the recognized template
> families, and the upstream `test_core` skip rationale. Pair with `CHANGELOG.md` for the
> current conformance count (regenerate via `scripts/run_upstream_tests.py`).

## Dtypes

| dtype | status | notes |
|---|---|---|
| `float32` | ✓ | full support (compute + the zero-copy fast matmul) |
| `float16` | ✓ | full support, incl. fast-matmul **input and output** (float accumulation for precision) |
| `bfloat16` | ✓ (general) / ⚠ (matmul) | elementwise, reductions, atomics OK; **not a fast-matmul input** (no `simdgroup_bfloat8x8`) — bf16 matmul falls back to the generic float-compute path |
| `int8/16/32` | ✓ | arithmetic, compare, reductions; int8/int4 matmul via dedicated templates |
| `int64` | ✓ / ⚠ | supported, but: `scf.for` with **i64 loop bounds is refused** (induction var must be ≤32-bit); some i64 reduce combine ops refused |
| `uint8/16/32/64` | ✓ | as int |
| `float8` (e4m3/e5m2) | ✗ refused | no Apple matrix hardware for microscaling/fp8 (`tt.dot_scaled`, fp8 dot) |
| `float64` | — impossible | Apple GPUs have no fp64 units |

## Op coverage by category

| category | status | detail |
|---|---|---|
| pointer / `tt.load` / `tt.store` | ✓ | incl. masked, broadcast, multi-dim; n>1-per-thread covered by the MEPT register-array path (see refusal #13/#14 for the uncovered edge) |
| elementwise (`arith.*`, `math.*`, `tt.*` unary/binary) | ✓ | add/mul/sub/div, cmp (f/i), select, exp/log/sqrt/sin/cos, cast, etc. |
| reductions (`tt.reduce`) | ✓ / ⚠ | sum/max/min/argmax/argmin/xor; in-loop reduce over a tile larger than the threadgroup is **refused** unless register-array-covered (#15) |
| `tt.dot` / matmul | ✓ | generic + the zero-copy fast simdgroup path (fp16/fp32 in, fp16/fp32 out); see matmul refusals #1/#2/#8/#9/#10 |
| `tt.dot_scaled` (microscaling) | ✗ refused | no Apple HW |
| control flow `scf.for` / `scf.if` / `scf.while` | ✓ | structured control flow; i64 loop bounds refused (#3); **unstructured** `cf.cond_br` / early-return-inside-conditional refused |
| atomics (`tt.atomic_rmw` / `tt.atomic_cas`) | ✓ / ⚠ | add/max/min/cas, incl. fp16/bf16 add; some 16-bit-float rmw ops refused (#5); n>1-per-thread atomic scatter refused (#13) |
| `tt.reshape` / `tt.broadcast` / `tt.expand_dims` | ✓ | rank-changing reshape may defeat the register-array spine (then a dependent store can hit the n>1 refusal) |
| `tt.trans` | ✓ (rank ≤ 2) / ✗ (rank ≥ 3 non-identity) | rank-≥3 transpose with a non-identity permutation is refused (#12) |
| `tt.cat` / `tt.join` | ✓ (rank ≤ 1) / ✗ (rank ≥ 2) | rank-≥2 cat/join refused; `tt.join` result feeding `tt.dot` refused |
| `tt.gather` | ✓ (1D; 2D axis 0 + same-shape axis 1) / ✗ (otherwise) | 1D `out[i]=src[idx[i]]`; **2D** via full-tile shared staging: axis=0 `out[i,j]=src[idx[i,j],j]` (incl. ragged row counts) and same-shape axis=1 `out[i,j]=src[i,idx[i,j]]`, for tiles fitting a 1024-thread threadgroup. **Refused loudly**: tiles > 1024 elems (one-thread-per-element staging), ragged axis=1, register-array operands. (Triton's own frontend also asserts `isWarpLocal()` on larger gather layouts.) |
| `tt.dot` inside a `noinline` device function | ✗ refused | not lowered through device-function calls |
| FlashAttention | ✓ (`BLOCK_M=BLOCK_N=32`, head_dim ∈ {32, 64, 128}) / ✗ (otherwise) | at **BLOCK_M = BLOCK_N = 32**: head_dim 32/64 (generic lowering, fp32) and **head_dim 128** routed to an Apple **`simdgroup_matrix` MMA kernel** (fp32 + fp16, causal + non-causal, any N_CTX) — **~7.4× (fp32) / ~8.5× (fp16) faster than the prior scalar path, ~8.2/9.6 TFLOP/s (~70–80% of the in-repo matmul-template peak). NOT competitive with Apple metal-flash-attention / MLX in absolute terms.** Contiguity gate: non-contiguous innermost stride falls back to the scalar template. **Refused loudly** outside supported configs: `head_dim > 128`, **`BLOCK_M`/`BLOCK_N` ≠ 32** (the `<32` small-block case silently mis-computed for *any* head_dim — a hole the old head_dim>64 guard missed, closed 2026-06-17), and **bf16** matmul inputs (rejected at the Triton frontend / not routed). Larger blocks and head_dim > 128 are roadmap |

## Framework integration — `torch.compile`

| capability | status | detail |
|---|---|---|
| `torch.compile(model, backend="inductor")` on `"mps"` | ✓ | routes through `triton_msl.inductor.register_metal_triton_backend()` → inductor `TritonScheduling` → triton-msl → MSL. Verified on elementwise, linear, conv, norms (layer/group/instance/batch), pooling, embedding, softmax/log-softmax, residual blocks, transformer encoders, multi-layer GPT, LSTM, and HF GPT-2 (cosine > 0.98). |
| dynamic shapes (`torch.compile(..., dynamic=True)`) | ✓ | symbolic dims flow to the lowerer; a **single compiled graph** serves variable sequence lengths (no per-shape recompile). |
| **training** (forward + backward) | ✓ | `torch.compile`d models train through AOTAutograd: the backward graph is ordinary Triton kernels (matmul→matmul, embedding scatter-add, softmax/layernorm/attention backwards) lowered by triton-msl. MLP / CNN / transformer (w/ embedding) converge and match eager gradients (`tests/test_training.py`). Optimizer step runs eager (or compile it separately). |
| compile parallelism | single-process (enforced) | the backend pins inductor to `compile_threads=1` + `autotune_in_subproc=False`: Metal/PyObjC is **not fork-safe**, so a forked compile worker crashes (and a crash mid-write can corrupt the on-disk cache → silent-wrong). This is a correctness requirement, not a perf tweak. |
| op inductor can't lower to a triton kernel | falls back loudly / to eager | inductor raises `InductorError` (loud) rather than emitting wrong values; conv/matmul may use aten extern kernels (correct, not routed through our MMA path). |

> Note: torch 2.10+ ships a *native* MPS inductor backend (`MetalScheduling`); registering
> triton-msl's backend takes priority and routes `torch.compile` through **our** kernels.

## Loud-refusal catalog (raises `MetalNonRecoverableError` — never silent-wrong)

Each is a case where the compiler recognizes the kernel but cannot lower it correctly, so
it refuses with an actionable message instead of emitting wrong numbers. (Most were once
silent-wrong producers, closed by the integrity prescan — see `CHANGELOG.md`.)

1. **K-loop matmul tiling output across programs with M/N baked as constexpr** — the true
   output strides can't be derived (`test_dot_mulbroadcasted`). Fix: pass M/N/K as runtime args.
2. **matmul template needs runtime M/N/K scalar args** but the kernel bakes dims as constexpr.
3. **`scf.for` with 64-bit loop bounds** — induction variable must be ≤32-bit.
4. **`tt.dot` operands not in a supported layout / shape mismatch.**
5. **`atomic_rmw` on 16-bit float for unsupported ops** (subset of rmw ops).
6. **Unresolved value (`UNKNOWN_<id>`)** — a value defined outside a runtime-bound loop used
   inside it when BLOCK exceeds the threadgroup (multi-element-per-thread edge).
7. **i64 reduce with an unsupported combine op.**
8. **K-loop matmul BLOCK_M/N/K not a multiple of 8** (simdgroup fragments are 8-deep).
9. **K-loop matmul exceeding the threadgroup-memory budget.**
10. **matmul with an unsupported fused epilogue** on the dot result.
11. **MEPT iter-arg array/yield width mismatch** (register-array spine integrity).
12. **rank-≥3 `tt.trans` with a non-identity permutation.**
13. **n>1-per-thread atomic scatter** (BLOCK > num_threads, uncovered) — fix: `num_warps = BLOCK/32`.
14. **n>1-per-thread tensor store** (BLOCK > num_threads, uncovered) — fix: `num_warps = BLOCK/32`.
15. **In-loop reduction over a tile larger than the threadgroup**, uncovered by the register array.
16. **Unlowerable kernel** — refuses rather than fall back to the heuristic legacy text
    parser (which has produced silent-wrongs); set `TRITON_MSL_LEGACY=1` to opt in for debugging.
17. **`tt.dot` operand shape mismatch** / other unsupported dot shapes.
18. **Unstructured kernel-level control flow** (`cf.cond_br`, early-return inside a conditional).
19. **FlashAttention outside the supported tiles** — the attention lowering (≥2 `tt.dot` +
    `exp` + `max`) supports `BLOCK_M = BLOCK_N = 32` at head_dim 32/64 (generic) and head_dim
    128 (routed to the `simdgroup_matrix` MMA kernel for contiguous strides, or the scalar
    tiled template otherwise; both fp32 + fp16, causal + non-causal). The prescan still
    refuses loudly: **(a)** head_dim > 128 (max dot tile dim > 128), **(b)** `BLOCK_M`/`BLOCK_N`
    < 32 (min dot tile dim < 32) — which silently mis-computed (rows past the first → garbage)
    for *any* head_dim incl. 32/64, a hole the old head_dim>64-only guard missed (closed
    2026-06-17), and **(c)** any FA-shaped kernel whose pointer/stride/scale params can't be
    resolved unambiguously (refuse-on-ambiguity — never a guessed kernel). bf16 matmul inputs
    are rejected at the Triton frontend. Larger blocks / head_dim > 128 are future work.

(Plus `tt.dot_scaled`, rank-≥2 `tt.cat`/`tt.join`, `tt.dot` in a noinline callee, and
`tt.join`→`tt.dot`, listed by category above.)

## Recognized fused-pattern templates

When the IR matches a known pattern, a hand-written MSL template is emitted instead of the
generic op-by-op lowering. ~65 families, including: elementwise/activations (`silu`, `gelu`,
`swiglu`, `clamp`, `where`, `dropout`), norms (`layer_norm`, `rms_norm`, `group_norm`,
`instance_norm`, `batch_norm`, fused residual+norm), reductions/softmax (`softmax`,
`online_softmax`, `cross_entropy`, `cumsum`, `variance`, `top_k`/`top_p`, `bitonic_sort`),
matmul (`simdgroup_matmul` + the fast variant, `int8`/`int4` matmul, `fused_mlp`,
`fused_linear`), the **attention family** (`flash_attention`, `causal`, `gqa`,
`paged`/`multi_head_paged`, `kv_cache`, `batched_kv_decode`, `fp16_kv`, `sliding_window`,
`rope_attention`, `repeat_kv`), memory/shape (`gather`, `scatter`, `index_select`,
`transpose`, `concat`, `split`, `embedding`, `rope`), atomics (`atomic_add`, `atomic_max`),
and conv/pool (`conv2d`, `max_pool2d`, `avg_pool2d`). Unmatched kernels use the generic
lowering (which is correct for any supported op, just less specialized).

## How to read this

- **✓ supported** — the kernel lowers to MSL and runs correctly on the Apple GPU.
- **✗ refused** — the compiler raises `MetalNonRecoverableError` with a clear message; the
  kernel does **not** run and does **not** produce wrong numbers. Many refusals include a
  fix (e.g. `num_warps = BLOCK/32`, or pass dims as runtime args).
- **— hardware-impossible** — Apple GPUs lack the unit (fp64, fp8 matrix); no software path.

This is alpha: coverage grows over time. The invariant that does **not** change is the
integrity contract — unsupported ⇒ loud refusal, never silent-wrong.
