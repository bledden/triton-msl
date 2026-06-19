# Fused matmul epilogue (#158) — design

> Make `tl.dot` followed by a pointwise/broadcast epilogue (bias, activation,
> scale, clamp, chains) actually COMPUTE the epilogue, instead of refusing.
> Before #157 these were silently dropped (returned bare `A@B`); #157 made them
> refuse loudly. This lifts the refusal to real support for the elementwise/
> broadcast case. Softmax keeps its own path (it has a reduce).

## Approach (chosen)

Reuse the generic op-by-op lowerer's already-factored per-op emitters
(`_emit_binary`, `_emit_unary`, `_emit_builtin_binary`, `_emit_cast`, …, all
keyed off `self.values[ssa_id]`) to lower the post-dot subgraph, driven from the
matmul→softmax staged template. The generic "1 scalar per thread" model maps
cleanly to "1 element per loop iteration," so the emitters work unchanged once
the dot result is seeded.

## Components

### 1. Detection — `_detect_matmul_epilogue`
Mirror `_detect_matmul_softmax`'s scaffolding (single `tt.dot`; M/N/K from
operand shapes; `M_BLOCK*N*4 <= 32 KiB` tg-fits cap; ptr-role resolution). Then
BFS the dot result's consumers to the `tt.store` (reusing the #157 traversal),
matching ONLY if every op on the path is in an allowlist of ops the generic
emitter already handles:
- `arith` binary/unary (mulf/addf/subf/divf/maximumf/minimumf/…),
- `math.*` (exp/sqrt/erf/…),
- casts (truncf/extf/sitofp/…),
- `tt.splat`, `tt.broadcast`, `tt.expand_dims`, `ttg.convert_layout`,
- `tt.load` of an extra input (bias) + `arith.constant`.

Any reduce/scan/unsupported op → NO match (softmax has its own earlier path;
everything else falls through to the #157 refusal — integrity preserved).
Returns: M/N/K, ptr_args, the topologically-ordered epilogue ops, bias input
arg(s), and the store SSA.

### 2. Template — `_lower_matmul_epilogue_template`
Factor the matmul→softmax template's shared prologue (matmul + spill to `tg_C`)
into a helper; reuse it. Replace the hardcoded softmax with a cooperative
per-element loop:
```
for (uint i = tiitg; i < M_BLOCK*N; i += 128) {
    uint row = i / N, col = i % N;
    // seed: self.values[dot_ssa.id] = "tg_C[i]"
    // for each epilogue op in topo order: call the matching _emit_* method
    //   (reads self.values for operands, writes self.values[op.id], emits MSL)
    C[(mstrip+row)*N + col] = (out_type) <store-value SSA's MSL var>;
}
```

### 3. Broadcast indexing (main integration risk)
A bias `(N,)` enters via `tt.load`→`tt.broadcast`→`arith.addf`. In the
per-element loop the bias value is `bias_ptr[col]` (or `[row]` for `(M,1)`).
Special-case the epilogue's `tt.load`/`broadcast`/`expand_dims` to index by the
loop's `col`/`row` (a small contained shim that seeds `self.values` for those
SSAs), NOT a change to the generic lowerer's addressing model.

### 4. Routing (in `lower()`)
`_detect_matmul_softmax` (reduce) → **`_detect_matmul_epilogue` (new)** →
`_detect_simple_dot` (pure; its #157 epilogue-refusal stays as the catch-all for
unsupported epilogues). No behavior change to existing paths: softmax still
matches softmax; pure matmul still matches simple_dot; only the previously-
refused pointwise-epilogue case now matches the new detector.

## Error handling / boundaries
- Oversized N (`M_BLOCK*N*4 > 32 KiB`): detector returns None → #157 refusal
  (loud), as today.
- Any non-allowlisted op in the epilogue: no match → #157 refusal (loud).
- Single-threadgroup staged template (same size envelope as matmul→softmax):
  correct but not peak-perf — acceptable since this case refuses entirely today.

## Testing
- Correctness vs numpy: matmul + {scale `*c`, shift `+c`, relu `max(x,0)`,
  clamp, gelu-approx via `x*0.5*(1+erf(...))`, chained ops}, and matmul + bias
  `(N,)` broadcast, and matmul + bias + relu (a full linear layer).
- Integrity: an unsupported epilogue (e.g. a row-reduce that isn't softmax)
  still REFUSES (no silent-wrong); oversized N refuses cleanly.
- Regression: pure matmul still simple_dot; matmul→softmax still softmax.
- Full `test_core` sweep 4326/0 (fresh `~/.cache/triton_msl`).

## Implementation outcome (2026-06-07)

Implemented as designed. Two notes vs the spec:

1. The factored generic emitters (`_emit_binary` etc.) turned out to be
   entangled with the lowerer's MEPT register-array / shared-memory / layout
   machinery — driving them from a per-element loop was high-risk. So the
   epilogue loop reuses the op->MSL DISPATCH MAPPING (same operators/functions
   the generic dispatch maps each op to) applied per element, and reuses
   `_lower_constant` for literals. Same general coverage, far lower risk.

2. Triton FUSES a trailing `acc + bias` into the dot's accumulator operand
   (`tt.dot %a, %b, %bias_broadcast`), so the bias is the dot's 3rd operand,
   NOT a post-dot op. The detector now also inspects `dot.operand_ids[2]`: a
   zero constant is the plain init (ignored); a broadcast-of-load is a bias
   (col for (N,), row for (M,1)) added back in the epilogue seed
   `tg_C[i] + bias[col]`; anything else refuses.

Verified: tests/test_matmul_epilogue.py (scale, relu, bias+relu linear layer,
chained scale->relu->clamp, and an unsupported reduce epilogue still refuses) —
all pass; project suite 591/0; full test_core sweep pending.
