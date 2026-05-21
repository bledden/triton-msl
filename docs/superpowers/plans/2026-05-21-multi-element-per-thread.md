# Multi-element-per-thread refactor (Phase 4 proper)

> Multi-week scope. Mostly affects `triton_metal/codegen/generic_lowerer.py` and
> the C++ MLIR-to-LLVM conversion passes. Lands a per-thread *register array*
> programming model that's the prerequisite for several deferred wins:
> FA HEAD_DIM=64 via the C++ direct path, `test_chained_reductions`,
> `test_dot_mulbroadcasted`, and any kernel where `sizePerThread > 1` would
> otherwise force a wrap-loop that conflicts with cooperative ops.

## Status of foundations (already landed)

- `triton_metal/codegen/_linear_layout.py` — XOR-basis position math for
  `#ttg.linear` / `#ttg.blocked` layouts.
- `IRGraph.mod_text` is plumbed through so layout aliases can be resolved.
- `_lower_convert_layout` raises `MetalNotImplementedError` for unhandled
  multi-element `#linear` sources instead of silently producing wrong output.
- Pattern detectors handle specific cases without the architectural rewrite:
  - `_detect_transpose_via_reshape` (test_trans_reshape)
  - `_detect_matmul_softmax` (test_dot softmax-epilogue)
  - `_detect_simple_dot` (with `_resolve_dot_ptr_roles`, `scf_iters` extraction)

## What's blocked

`make_llir` wraps oversized kernels with a `for (_wlid = lid; _wlid < total;
_wlid += 1024)` loop. This works for elementwise + per-row reduce kernels but
breaks for kernels that:

1. Have **`scf.for` loop-carried state larger than 1024 elements** (FA's
   `acc`/`m_i`/`l_i` at HEAD_DIM=64 → tile = 2048). The phi node carries one
   scalar per thread; each of the 1024 threads would need to carry 2 scalars.
2. Have **`tt.dot` whose operand tile exceeds 1024 elements** — the populate
   phase fills only positions 0..1023; `simdgroup_matrix_load` reads garbage
   from 1024..2047.
3. Need **layout-aware `ttg.convert_layout` between two multi-element-per-
   thread tensors** — the current 1-element-per-thread shuffle can't move
   the extra elements.

All three share the same root: per-thread state is a single scalar SSA value;
the model has no notion of "thread `lid` holds elements `[lid, lid + 1024,
lid + 2048, ...]`."

## Target model

For every tensor SSA value `%v : tensor<NxT, #layout>`:
- Compute `n_per_thread = N / (num_warps * 32)` (after counting register basis
  vectors in `#layout`).
- Represent `%v` in MSL as `T v[n_per_thread]` instead of `T v`.
- All elementwise ops loop over the array.
- `tt.reduce`: per-thread reduce of the local array, then cross-thread SIMD
  reduce.
- `tt.dot`: when an operand's `dot_op<{...}>` layout has multiple registers,
  pass an `acc[k]` matching the dot-op shape.
- `ttg.convert_layout`: shared-memory shuffle. Each thread writes its
  `n_per_thread_src` elements at positions
  `src_layout.position(reg_i, lane, warp)` for `reg_i in 0..n_per_thread_src`;
  barrier; reads `n_per_thread_dst` elements from
  `dst_layout.position(reg_j, lane, warp)`.
- `scf.for` iter args: per-thread arrays become multi-result phi nodes (one
  scalar per element).
- `tt.load` / `tt.store`: gather/scatter `n_per_thread` elements per thread.

## Implementation phases

### 4a. Track elements-per-thread (~1 day)

Add `self.env_n_elems: dict[int, int]` to `GenericLowerer`. Populate it
when each tensor-producing op runs:
- From tt.make_range / tt.splat / tt.broadcast / tt.expand_dims / tt.reshape:
  recompute from the op's result `type_str` using `LinearLayout` /
  `blocked_to_linear`.
- For other ops: inherit from operand 0.
- Default 1 for back-compat with the current single-scalar emission.

Verify by adding a debug print and running the full test_core sweep — every
tensor SSA should get a tracked n_elems value with no regressions.

### 4b. Refactor `_var` / `_emit_passthrough` to optionally array-store (~3 days)

Add `_var_array(name, exprs: list[str], ty)` that emits `T name[len(exprs)]`
plus `name[i] = exprs[i]` assignments. The plain `_var` becomes a special
case (`len(exprs) == 1`).

Update `env` to map SSA id → either a scalar var name *or* an array var name
+ length. All ops that consume an SSA value need to know how to read the
right form.

This is the biggest single chunk. Most op handlers (`_lower_arith_*`,
`_lower_math_*`, `_emit_binary`, `_emit_unary`, etc.) need an "if my operand
is an array, emit a `for` loop" branch.

### 4c. tt.load / tt.store array gathers (~1 day)

Currently `tt.load` reads one element per thread at `ptr[offsets[lid]]`.
For an array tensor result with `n_per_thread = 4`:
```c
T v[4];
for (uint i = 0; i < 4; i++) {
    uint pos = layout.position(i, lane, warp);  // pos in 1024-elem buffer
    v[i] = (mask[i]) ? ptr[offsets[pos]] : other;
}
```
The position formula comes from the result tensor's layout.

`tt.store` is symmetric.

### 4d. Convert_layout shuffle (~1 day)

```c
threadgroup T shuffle_buf[total_elems];
// Write phase
for (uint i = 0; i < n_per_thread_src; i++) {
    shuffle_buf[src_layout.position(i, lane, warp)] = v[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Read phase
T w[n_per_thread_dst];
for (uint j = 0; j < n_per_thread_dst; j++) {
    w[j] = shuffle_buf[dst_layout.position(j, lane, warp)];
}
```

The XOR-basis position math is already implemented in
`_linear_layout.LinearLayout.msl_position_expr`.

### 4e. Reduce + scan over arrays (~2 days)

`tt.reduce` with an array operand becomes a per-thread fold followed by the
existing SIMD/threadgroup reduce. For `axis=` reductions on multi-dim tensors,
the per-thread fold respects the layout's register-basis dimensions.

### 4f. C++ side: ReduceOpConversion + DotOpConversion + ConvertLayoutOp (~5 days)

Mirror the same changes in the C++ MLIR-to-LLVM passes:
- `ReduceOpConversion`: per-thread array reduce → SIMD reduce.
- `DotOpConversion`: support `simdgroup_matrix` ops on tiles where each
  thread holds multiple elements (use the existing per-tile loop structure
  but lift the `for tk` body so accumulation across multiple per-thread
  scalars is preserved).
- `ConvertLayoutOp`: emit the shuffle directly in LLVM IR.
- Remove the `_inject_wrapping_loop` workaround entirely.

### 4g. Pattern-detector deprecation (~1 day)

Once 4a–4f land, the targeted detectors become unnecessary. Remove
`_detect_transpose_via_reshape`, `_detect_matmul_softmax`, etc. — the
generic per-op lowerer should handle these patterns correctly.

Confirm by re-running `test_trans_reshape` / `test_dot` softmax-epilogue /
`test_dot_multidim` against the new lowerer.

## Risk mitigation

- Land 4a behind a feature flag (`TRITON_METAL_MEPT=1`). All existing tests
  must continue to pass with the flag off.
- Add a correctness harness: a small set of kernels with known-good outputs
  (matmul, softmax, layer_norm, FA, sort) gated on each phase.
- Per-phase regression: run the full `test_core.py` sweep after every commit.
- Keep the pattern detectors in place until 4g; they're the safety net.

## Test coverage targets

Specific tests that should turn green at each phase:

- 4a: no test impact (infrastructure only); regression bar: existing
  4325/0 in test_core.
- 4d: `test_chained_reductions` (currently 0 passed, 2 skip-listed) —
  the convert_layout shuffle is the missing piece.
- 4e: `test_dot_mulbroadcasted` (currently 0 passed) — the constexpr
  K-loop matmul stride was the surface bug, but the root cause is the
  multi-element-per-thread reduction.
- 4f: FA HEAD_DIM=64 via C++ direct path. Currently routes to MSL
  (`_has_complex_ops` returns True for `tt.dot` + wrap-loop). After 4f,
  removable.

## Out of scope

- A *new* layout system. We keep `#ttg.linear` / `#ttg.blocked` as-is.
- C++ rewrite of `ElementwiseOpToLLVM.cpp` (already handles per-thread
  scalars and is unrelated to MEPT).
- AOT compilation for MEPT kernels. Existing AOT path is unchanged.
