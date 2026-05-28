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
- **Phase 4a (DONE)** — `env_n_elems` dict + `_track_n_elems` /
  `_parse_blocked_field` helpers populate elements-per-thread for every
  tensor SSA that flows through `_propagate_shape_from_type`. Locked in
  with unit tests in `tests/test_generic_lowerer.py`
  (`test_track_n_elems_*`, `test_parse_blocked_field_*`). Direct
  `env_shapes[]` writes (e.g. tt.make_range, splat) don't yet populate
  `env_n_elems` — 4b call sites must consult both.
- **Phase 4b scaffolding (DONE)** — `TRITON_METAL_MEPT` env flag,
  `env_array` map, `_var_array(prefix, exprs, ty)` emitter, and
  `_lookup_array(ssa_id) -> (name, n, ty)` reader. No op handler is
  wired yet; flag-on default-route preserves byte-identical MSL output
  (regression-tested by `test_mept_flag_on_preserves_existing_behavior`).
- **Phase 4b consumer-side integration (DONE 2026-05-28)** — array
  path wired into every elementwise emit helper:
  - `_emit_passthrough` (book-keeping forward)
  - `_emit_cast`, `_emit_uitofp`, `_emit_int_cast`
  - `_emit_unary`, `_emit_binary` (symmetric + broadcast via
    `_emit_binary_mept` / `_mept_binary_dispatch`)
  - `_emit_builtin_binary`, `_emit_nan_propagating_minmax`
  - `_lower_math` for the unary_map ops (exp/log/sqrt/abs/sin/cos/tanh/
    floor/ceil/round + variants), `math.absi`, `math.fma`,
    `math.powf` / `math.copysign` / `math.atan2`,
    `math.roundeven` / `math.trunc`
  27 unit tests. All flag-gated; flag-off byte-identical. `test_core`
  sweep still 4325/0/5017 — no regressions from any of the wiring.
  Still scalar-only: `math.erf` / `log1p` / `expm1` (multi-statement
  chains; need their own array template). The consumer side is now
  effectively complete for the elementwise op family.

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

### 4a. Track elements-per-thread (~1 day) — **DONE**

Status: landed. `self.env_n_elems` + `_track_n_elems` populate from the
result `type_str` via `LinearLayout` / `blocked_to_linear`. The current
plumbing covers ops that route through `_propagate_shape_from_type`;
direct `env_shapes[]` writes (46 call sites: 32 in `generic_lowerer.py`,
8 in `_lowerer_reduce.py`, 5 in `_lowerer_control.py`, 1 in
`_device_func_lowerer.py`) bypass it and need
backfill before 4b consumers can rely on the dict being complete. The
4b backfill option: replace direct writes with helper calls; the safer
option: have consumers (e.g. `_emit_binary` MEPT branch) call
`_track_n_elems` defensively on operand SSA ids before consulting
`env_n_elems`.

### 4b. Refactor `_var` / `_emit_passthrough` to optionally array-store (~3 days)

**Scaffolding landed**: `TRITON_METAL_MEPT` flag, `env_array` map,
`_var_array`, `_lookup_array`. Remaining work is integrating each op
handler. Suggested order of integration (least → most surface):

1. `_emit_passthrough` — pure book-keeping, no MSL emission needed
   beyond propagating the `env_array` entry.
2. `_emit_cast` — single-operand, single-result; wrap the existing
   `static_cast<T>(a)` in a per-element loop.
3. `_emit_unary` — same shape as cast.
4. `_emit_binary` — two operands; both array form, or one array + one
   broadcasted scalar (already handled via shape-elementwise propagation).
5. tt.load / tt.store gather/scatter — see Phase 4c.

For each handler, the wrap pattern is:
```
if self.mept_enabled:
    n_a = self.env_n_elems.get(op_id_a, 1)
    n_b = self.env_n_elems.get(op_id_b, 1)
    n = max(n_a, n_b)
    if n > 1:
        # array form via _var_array
        ...
        return
# else: scalar form (existing code unchanged)
```

Add `_var_array(name, exprs: list[str], ty)` that emits `T name[len(exprs)]`
plus `name[i] = exprs[i]` assignments. The plain `_var` becomes a special
case (`len(exprs) == 1`).

Update `env` to map SSA id → either a scalar var name *or* an array var name
+ length. All ops that consume an SSA value need to know how to read the
right form.

This is the biggest single chunk. Most op handlers (`_lower_arith_*`,
`_lower_math_*`, `_emit_binary`, `_emit_unary`, etc.) need an "if my operand
is an array, emit a `for` loop" branch.

### 4c. tt.load / tt.store array gathers (~1 day) — **PARTIALLY DONE**

Status: simplest case landed for the **contiguous 1D layout**:

- `_lower_make_range` (commit 46f5362): when `env_n_elems[ssa.id] > 1`
  and MEPT is on, emits `idx[N]` via `_var_array` with
  `idx[i] = start + lid*N + i`. Defers more elaborate layouts (multi-
  dim, interleaved warps, non-default order) to scalar fallback.
- `_lower_addptr` (commit 04f35fa): when offset operand has `env_array`,
  records `env_ptr_array[ssa.id] = (base_ptr, offset_array, n)`.
  Handles array+array, scalar+array, bare-ptr+array parent forms.
- `_lower_load` (commit 04f35fa): when ptr has `env_ptr_array`, emits
  per-position `val[i] = static_cast<T>(base[off[i]])` into a fresh
  `env_array`. Mask / "other" / FP8 paths fall back to scalar.
- `_lower_store` (commit 04f35fa): when ptr has `env_ptr_array` and
  the value has `env_array` of matching length, emits per-position
  writes. Mask in array path is a follow-up.

A synthetic round-trip test
(`test_mept_round_trip_load_op_store`) exercises
`make_range→addptr→load→unary→addptr→store` end-to-end inside one
`GenericLowerer` with the flag on and verifies the full array trail.

What's left for full 4c:
- ~~Mask / "other" in the array load path~~ — landed (b8531db).
- ~~Mask in the array store path~~ — landed (b8531db).
- ~~LinearLayout-aware position math~~ — landed (2ff262c).
- ~~FP8 in array load~~ — landed (ccaa077): two-array form for raw
  uchar gather + per-position float conversion.
- ~~Real TTGIR-driven exercise~~ — landed (4fd92cb): the Triton JIT
  runtime adds `tt.divisibility=16` to pointer args automatically,
  which causes the existing coalesce pass to emit
  `sizePerThread=[4]` on default Apple configs (4 warps × 32 lanes).
  Bridging two final wires (`_track_n_elems` from `_lower_make_range`,
  and `_mept_binary_dispatch` from `_lower_cmpi`) lets a stock
  vector_add through the JIT path produce the full MEPT array form
  end-to-end. Max diff vs PyTorch: 0.000000.

Phase 4c is effectively complete. Remaining for full Phase 4:
- 4e: array reductions / scans (covered by existing `_lowerer_reduce.py`
  for scalar; needs an array-aware variant for the per-thread fold).
- 4f: C++ side mirror changes (`ReduceOpConversion` /
  `DotOpConversion` / `ConvertLayoutOp` — these only matter for the
  `TRITON_METAL_USE_CPP=1` path).
- 4g: pattern-detector deprecation (only after 4f).
- Performance optimization: wrap-loop is redundant inside MEPT tiles;
  could be elided to avoid 4× wasted work on already-masked positions.

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
