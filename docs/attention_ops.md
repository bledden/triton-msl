# Frontier attention ops (KDA + FlashAttention backward)

Two attention capabilities the standard `@triton.jit` → simdgroup-FlashAttention path does
not cover, exposed as direct Metal ops. Both are validated against ground-truth references
(`tests/test_kda.py`, `tests/test_fa_backward.py`) and dispatched through `compile_shader`.

## KDA — Kimi Delta Attention / gated DeltaNet (`triton_msl.kda`)

Linear/delta-rule attention: a per-key-dimension forget gate combined with the delta rule's
fast-weight correction. This is *not* softmax attention, and its chunked (parallel-prefill)
form needs a UT-transform triangular solve, so it cannot be expressed as one `@triton.jit`
kernel — it ships as a direct op rather than a detector/route.

```python
from triton_msl.kda import kda_attention, kda_decode_step

# Prefill (chunked MMA; historical ~7-12x vs naive is unqualified on this candidate):
out = kda_attention(q, k, v, a, beta)          # q,k,v,a: [ZH,T,64]; beta: [ZH,T]; mps

# Decode (one autoregressive step, state updated in place):
S = torch.zeros(ZH, 64, 64, device="mps")
o = kda_decode_step(q_t, k_t, v_t, a_t, beta_t, S)   # q_t..: [ZH,64]; beta_t: [ZH]
```

- `a` is the per-key-dim gate in `(0, 1)`; `beta` is the delta step (typically a sigmoid).
- float32 or float16 I/O (accumulate and state are always fp32); output dtype matches `q`.
- **Constraints (enforced with a loud `ValueError`):** head dim `== 64`, `T % 8 == 0`.
- **Head dim is fixed at 64 by design:** the recurrent state is `d_k × d_v`, so `d = 128`
  would need a 64 KB threadgroup state, over Metal's 32 KB limit. A `d = 128` variant would
  require a device-resident state (memory-bound, much slower) or a low-rank-state redesign;
  `d = 64` is the sweet spot for the on-chip fast path.

## FlashAttention backward — trainable attention (`triton_msl.fa_backward`)

The forward FlashAttention path is inference-only. `flash_attention` wraps a
`torch.autograd.Function` whose backward dispatches tiled, MMA-optimized Metal dQ/dK/dV
kernels (the FA-2 backward), so an attention call inside a training loop gets Metal gradients.

```python
from triton_msl.fa_backward import flash_attention

o = flash_attention(q, k, v, causal=True)   # q,k,v: [..., N, 64] mps; folds leading dims
o.backward(dO)                              # dQ/dK/dV computed on Metal
```

- head dim `== 64`, `N % 16 == 0`, full and causal.
- The forward computes `O` in float32 with explicit torch matmul and softmax, so it
  materializes its scores. Each step (matmul, scale, causal fill, softmax) makes a new
  `[ZH, N, N]` float32 tensor from the previous one, so the peak is about two such tensors
  (about 8 GiB at ZH = 64, N = 4,096), plus an N×N boolean mask when causal. It does not
  call PyTorch SDPA: on MPS, torch 2.12.1 keeps the first `scale`
  it sees for a shape and ignores later ones, which returned a wrong `O`, dQ and dK whenever
  the scale changed within a process. A separate dedicated Metal flash kernel
  (`make_fa_logsumexp_kernel`, tiled/online) computes the log-sum-exp needed by backward
  without materializing its N×N score matrix.


## Biased / triangle attention (trifast's shape) — forward and backward templates

The `@triton.jit` kernels of the trifast / AlphaFold triangle-attention family are recognised and
routed to hand-written Metal templates without source changes: the forward
(`scores = q·kᵀ·scale + bias`, a boolean column mask selecting a score sentinel, `exp2` online
softmax, `lse` stored) and the three backward kernels (dK / dV, dQ + delta, dbias). Each route
*proves* the source against the template's algebra before emitting anything — which pointer plays
which role, along which axis every broadcast lies, the whole downstream gradient graph by
orientation, the exact loop recurrences, the reduce combiner's returned value — and refuses by name
on any difference. Configurations, dtypes and the refusals are listed in `SUPPORTED_OPS.md`
(the two "biased / triangle attention" rows and catalog entry 21).

Narrow dtypes: the templates compute in fp32 and **replay the source's rounding points** rather
than approximate them — the scaled operand (`x * tl.full([1], scale, dtype)`), `P.to(dtype)` where
the source rounds it (per kv block in the forward, before the dV dot in the backward),
`dS.to(dtype)` before the dK / dQ dots, and the output casts. The one rounding no template can
replay is a reduction *accumulated* in the narrow dtype (trifast's unmodified `_bwd_q` computes
`delta = tl.sum(o * do)` that way): it refuses, and the message names the one-line change
(`tl.sum((o * do).to(tl.float32), 1)`) that makes it computable.
