# Design Spec: C++ Path Cooperative Ops (Shared Memory + Matmul)

**Status:** Draft — awaiting user review
**Date:** 2026-04-14
**Author:** Blake Ledden (via subagent brainstorming)

## Summary

Extend the triton-msl C++ MLIR pass layer to handle all cooperative Triton ops — TTG shared memory operations, `tt.dot` with simdgroup MMA, shared memory aliasing, and vectorized access via `#shared` encoding. After this work, the C++ path handles every Triton kernel that doesn't require M5-specific hardware features, eliminating MSL fallback for the project test suite.

## Goal

Make the C++ metallib path the default for all non-M5 kernels, reaching parity with the Python/MSL path.

## Non-Goals (M5-blocked only)

- Real async DMA via `simdgroup_async_copy` (Metal 4, M5 hardware)
- M5 MPP tensor ops

Pre-M5 synchronous equivalents are **in** scope.

## Current State

- **C++ metallib path** handles elementwise, reductions, softmax, SCF loops, type casts, float scalar args, wrapping loop for >1024 elements.
- **Missing:** `ttg.local_alloc`, `ttg.local_load`, `ttg.local_store`, `ttg.memdesc_subview`, `ttg.memdesc_trans`, `ttg.async_copy_global_to_local`, `ttg.async_wait`, `ttg.local_dealloc`, `tt.dot`, shared-memory aliasing, vectorized shared-memory access.
- **Result:** Matmul kernels, FlashAttention, cumsum, layer_norm, tiled reductions all fall back to MSL.

## Architecture

### TTG Shared Memory → LLVM addrspace(3)

Triton's `!ttg.memdesc<shape, elem, #encoding>` maps to LLVM `ptr addrspace(3)` (Metal threadgroup memory). Shape/stride info from the MLIR type is consumed at pattern-match time.

```
Triton TTGIR                       LLVM IR (Metal AIR)
────────────────────────────────   ──────────────────────────────────────
%shared = ttg.local_alloc        → @__tg_shared_N = internal addrspace(3)
         : !ttg.memdesc<SxT>        global [S x T] undef
%val = ttg.local_load %m         → load T, ptr addrspace(3) %m
ttg.local_store %v, %m           → store T %v, ptr addrspace(3) %m
%sub = ttg.memdesc_subview       → getelementptr T, ptr addrspace(3) %m
%t = ttg.memdesc_trans           → <same ptr, view-only; encoding tracks order>
ttg.async_copy_global_to_local   → for-loop: thread-strided global→shared copy
ttg.async_wait                   → air.wg.barrier
ttg.local_dealloc                → no-op
```

### `tt.dot` → simdgroup MMA tiling

Apple GPUs expose 8×8 MMA via AIR intrinsics. Larger tiles tile over 8×8 blocks with register accumulation:

```
tt.dot %a, %b, %c : tensor<MxK>, tensor<KxN>, tensor<MxN>
  ↓
for mi in 0..M/8:
  for ni in 0..N/8:
    acc = air.simdgroup_load_indirect_matrix(%c, mi*8, ni*8)
    for ki in 0..K/8:
      a_tile = air.simdgroup_load_indirect_matrix(%a, mi*8, ki*8)
      b_tile = air.simdgroup_load_indirect_matrix(%b, ki*8, ni*8)
      acc = air.simdgroup_matrix_multiply_accumulate(a_tile, b_tile, acc)
    air.simdgroup_store_indirect_matrix(%d, acc, mi*8, ni*8)
```

Memory flow: global → `ttg.local_alloc` → simdgroup registers (via `simdgroup_load`) → MMA accumulate → back through threadgroup → global.

K-loop (`scf.for` wrapping `tt.dot`) is handled by existing SCF→CF lowering — no special treatment.

### Shared Memory Aliasing

Kernels with multiple allocations where live ranges don't overlap share backing memory.

**Algorithm:**
1. Collect all `ttg.local_alloc` ops and compute liveness (first use → last use in linearized IR)
2. Build interference graph: edge between two allocs if live ranges overlap
3. Color graph using largest-first heuristic
4. Allocations with same color reference the same `addrspace(3)` global (largest size wins; smaller allocs take a sub-range)
5. Emit single global per color; fix up `ttg.local_alloc` result SSA values to GEPs into the shared global

**Target:** FlashAttention HEAD_DIM=64 fits in 32KB after aliasing.

### Vectorized Shared Memory Access (`#shared` encoding)

TTG shared memory has encoding attributes:
```
#shared = #ttg.shared<{vec=4, perPhase=2, maxPhase=8, order=[1,0]}>
```

Lowering uses these to:
- **vec=4:** Emit `<4 x T>` vector loads/stores instead of scalar where alignment permits
- **perPhase/maxPhase:** XOR-based swizzle on column index: `col ^ ((row / perPhase) & (maxPhase - 1))`
- **order=[1,0]:** Column-major strides in GEP computation

## Op Lowerings (Detailed)

### 1. `ttg.local_alloc`

Two forms: uninitialized (just reserves memory) and initialized (stores a value).

**C++ pattern:**
```cpp
class LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
    LogicalResult matchAndRewrite(
        triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto memdescTy = op.getType();
        auto shape = memdescTy.getShape();
        auto elemTy = memdescTy.getElementType();
        auto bytes = computeBytes(shape, elemTy);

        // Track per-module cumulative threadgroup bytes
        auto &total = getTotalTgBytes(op->getParentOfType<ModuleOp>());
        if (total + bytes > 32768)
            return op->emitError("threadgroup memory budget exceeded");
        total += alignUp(bytes, 16);

        // Create unique global
        std::string name = "__tg_shared_" + std::to_string(nextCounter());
        auto globalOp = createAddrSpace3Global(module, name, shape, elemTy);
        Value basePtr = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);

        // Initialized form: per-thread store
        if (op->getNumOperands() == 1) {
            Value initVal = adaptor.getOperands()[0];
            Value lid = getThreadPositionInThreadgroup(rewriter, loc);
            Value elemPtr = rewriter.create<LLVM::GEPOp>(loc, ...);
            rewriter.create<LLVM::StoreOp>(loc, initVal, elemPtr);
        }
        rewriter.replaceOp(op, basePtr);
        return success();
    }
};
```

### 2. `ttg.local_load`

```cpp
// ttg.local_load %m : memdesc<SxT> -> T   (per-thread model: scalar)
// → %v = load T, ptr addrspace(3) %m
```

Per-thread model: each thread loads the element at its `lid` index unless a `memdesc_subview` has already narrowed the pointer.

### 3. `ttg.local_store`

Symmetric to `local_load`:
```cpp
// ttg.local_store %v, %m : T, memdesc<SxT>
// → store T %v, ptr addrspace(3) %m
```

### 4. `ttg.memdesc_subview`

```cpp
// %sub = ttg.memdesc_subview %m[%i, %j] : memdesc<MxNxT> -> memdesc<NxT>
// → %off = mul %i, N
//   %sub = getelementptr T, ptr addrspace(3) %m, i32 %off
```

Result type's shape preserves the slice dimensions for downstream consumers.

### 5. `ttg.memdesc_trans`

No data movement. Returns the same pointer; the result type's `order` attribute signals transposed access pattern to the next consumer.

```cpp
// %t = ttg.memdesc_trans %m {order=[1,0]} : memdesc<MxNxT> -> memdesc<NxMxT>
// → %t = %m  (pass-through)
```

When `tt.dot` consumes a transposed operand, its lowering reads the `order` attr and adjusts MMA load patterns accordingly.

### 6. `ttg.async_copy_global_to_local`

Synchronous per-thread cooperative copy (pre-M5):
```cpp
// ttg.async_copy_global_to_local %src, %dst, %mask
// → for (i = lid; i < N; i += block_size) {
//     if (mask[i]) dst[i] = src[i];
//   }
```

Emitted as `scf.for` at the pattern level, which the SCF→CF pass lowers to branches.

### 7. `ttg.async_wait`

```cpp
// ttg.async_wait {num=0}
// → call void @air.wg.barrier(i32 2, i32 1)
```

### 8. `ttg.local_dealloc`

```cpp
// ttg.local_dealloc %m
// → rewriter.eraseOp(op)  (Metal threadgroup is function-scoped)
```

### 9. `tt.dot`

```cpp
class DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
    LogicalResult matchAndRewrite(
        triton::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto aTy = cast<RankedTensorType>(op.getA().getType());
        auto bTy = cast<RankedTensorType>(op.getB().getType());
        auto M = aTy.getShape()[0], K = aTy.getShape()[1];
        auto N = bTy.getShape()[1];
        auto elemTy = aTy.getElementType();

        // Must be multiples of 8 for simdgroup MMA
        if (M % 8 || K % 8 || N % 8)
            return op->emitError("tt.dot requires 8-multiples");

        // Generate tile loop (scf.for wrapping simdgroup ops)
        Value aPtr = adaptor.getA();
        Value bPtr = adaptor.getB();
        Value cPtr = adaptor.getC();

        // emit nested scf.for loops and air.simdgroup_* intrinsic calls
        // (details in implementation)

        rewriter.replaceOp(op, resultValue);
        return success();
    }
};
```

**AIR intrinsics used:**
- `air.simdgroup_load_indirect_matrix_8x8_f16/f32`
- `air.simdgroup_store_indirect_matrix_8x8_f16/f32`
- `air.simdgroup_matrix_multiply_accumulate_8x8_{f16,f32}.{f16,f32}.{f16,f32}`

**Operand types:**
- f16 × f16 → f32 (most common for ML)
- f32 × f32 → f32
- bf16 × bf16 → f32 (M2+)

### Shared Memory Aliasing Pass

Runs after `ttg.local_alloc` lowering, before the final LLVM IR emission:

```cpp
class SharedMemAliasingPass : public Pass {
    void runOnOperation() override {
        ModuleOp module = getOperation();
        for (auto func : module.getOps<LLVMFuncOp>()) {
            // 1. Find all __tg_shared_* globals used by this function
            auto allocs = collectTgGlobals(func);
            // 2. Liveness analysis: first use idx, last use idx
            auto liveness = computeLiveness(func, allocs);
            // 3. Build interference graph
            auto graph = buildInterferenceGraph(allocs, liveness);
            // 4. Color graph
            auto colors = greedyColor(graph);
            // 5. Emit consolidated globals and rewrite GEPs
            rewriteToConsolidatedGlobals(func, allocs, colors);
        }
    }
};
```

## TritonGPU Dialect Linking

This plan requires linking against the TritonGPU dialect (reversing the current "allowUnregisteredDialects" approach). Reasoning:

- `tt.dot` needs type-safe access to operand shapes via `RankedTensorType` — straightforward today
- `ttg.memdesc` types need structured access to shape/encoding/order attributes for `subview`, `trans`, and `local_load`/`store` offset computation
- Pattern matching via `OpConversionPattern<ttg::LocalAllocOp>` gives compile-time safety over string matching
- Shared memory aliasing pass needs to walk TTG ops by C++ type

**Build change:** Add `libTritonGPUIR.a` (or the TTG objects) to the `_triton_msl_cpp` and `triton_msl_plugin` CMake targets. If TTG symbols aren't exported from `libtriton.so` on macOS, link the static archive directly from `$TRITON_BUILDPATH/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/*.o`.

**Risk:** Duplicate MLIR dialect registration if `libtriton.so` also contains TTG. Mitigation: verify TTG is not in `libtriton.so`'s exported symbols before linking; if it is, use link-time symbol visibility controls.

### Type Conversion

```cpp
typeConverter.addConversion([&](triton::gpu::MemDescType mdt) -> Type {
    return LLVM::LLVMPointerType::get(mdt.getContext(), /*addrspace=*/3);
});
```

Patterns use proper C++ types:
```cpp
class LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> { ... };
```

## AIR Metadata for Threadgroup Buffers

Each `addrspace(3)` global needs an `!air.threadgroup_buffer` metadata entry:
```llvm
!threadgroup_bufs = !{
  !{ptr addrspace(3) @__tg_shared_0, !"air.threadgroup_buffer",
    !"air.location_index", i32 0, i32 0,
    !"air.arg_type_size", i32 1024,
    !"air.arg_type_align_size", i32 16}
}
```

The bridge scans `addrspace(3)` globals post-codegen and appends this metadata.

## Compiler.py Integration

### Allowlist Update

Add to `_has_complex_ops` allowlist:
```python
'ttg.local_alloc', 'ttg.local_load', 'ttg.local_store',
'ttg.local_dealloc',
'ttg.memdesc_subview', 'ttg.memdesc_trans',
'ttg.async_copy_global_to_local', 'ttg.async_wait',
'tt.dot',
```

### `_strip_ttg_annotations` Changes

Currently strips TTG ops entirely. Reverse for the ops we now lower — keep them in the MLIR passed to the C++ pipeline. Only strip encoding annotations (`#blocked`, `#slice`) which remain unregistered.

### `_try_generate_matmul_mlir` Removal

The scalar-matmul fallback bypass becomes unnecessary for kernels using `tt.dot` — they flow through real lowering. Remove the bypass or gate it on "C++ path not available".

## Testing

### Unit Tests (`tests/test_cpp_backend.py`)

- `test_local_alloc_basic` — allocate, store lid, load, verify
- `test_local_alloc_initialized` — initialized form
- `test_memdesc_subview` — 2D allocation, slice, read/write
- `test_memdesc_trans` — transposed view + MMA consumer correctness
- `test_async_copy_sync_loop` — global→shared copy + barrier correctness
- `test_32kb_budget` — alloc > 32KB fails cleanly (C++ returns error, MSL handles it)
- `test_dot_8x8_f16` — smallest tile, single 8×8 MMA
- `test_dot_32x32x16_f16` — multi-tile, tiling logic
- `test_dot_with_k_loop` — scf.for wrapping tt.dot
- `test_aliasing_reuse` — two non-overlapping allocs share backing memory
- `test_vectorized_load_store` — `#shared<{vec=4}>` emits vector ops

### Integration Tests

- `test_cpp_tiled_reduction` — reduction via shared memory (not AIR intrinsic path)
- `test_cpp_cumsum` — shared-memory prefix scan
- `test_cpp_layer_norm` — 2-pass (mean/var) via shared memory
- `test_cpp_matmul_32x32` — minimum viable matmul through C++
- `test_cpp_matmul_128x128` — production-sized matmul
- `test_cpp_flash_attention_head32` — FlashAttention HEAD_DIM=32
- `test_cpp_flash_attention_head64` — FlashAttention HEAD_DIM=64 (requires aliasing)

### Upstream Audit

Final task of implementation plan:
1. Clear all caches
2. Run upstream suite with `TRITON_MSL_USE_CPP=1`
3. Count `make_metallib_from_llir` invocations vs `make_metallib(` invocations in debug output
4. Compare pass counts: baseline 4,312 — verify ≥ 4,312 and ideally higher (TTG ops previously forcing MSL fallback now compile correctly through C++)
5. Update `project_status.md`

## Risks

1. **TritonGPU dialect linking** — linking `libTritonGPUIR.a` into our pybind11 module may cause duplicate MLIR registrations if libtriton.so also contains TTG (it doesn't on macOS currently, but could change). Mitigation: verify exported symbols before linking; use visibility controls if needed.

2. **AIR threadgroup metadata format** — undocumented. Derive from MSL-compiled metallib disassembly. May vary across Metal versions.

3. **Simdgroup MMA intrinsic names** — AIR intrinsics are documented but spellings vary by type. Validate via disassembly of known-good kernels.

4. **Aliasing correctness bugs** — graph coloring bugs cause data corruption. Mitigation: extensive unit tests with overlapping-vs-non-overlapping allocations, comparison against MSL output.

5. **Encoding attribute semantics** — `#shared` encoding has subtleties (swizzle phases, vectorization alignment). If we get it wrong, memory corruption or wrong results. Mitigation: cross-check against MSL-generated MSL source for known kernels.

6. **32KB budget cutoff** — some kernels need more. Aliasing reduces this, but occasional kernels still exceed. Graceful MSL fallback covers these.

## Success Criteria

- [ ] All 8+ TTG shared-memory ops lower through C++
- [ ] `tt.dot` lowers to simdgroup MMA (f16, bf16, f32)
- [ ] Shared memory aliasing reuses backing buffers
- [ ] Vectorized shared memory access via `#shared` encoding
- [ ] All unit tests pass (11+ new tests)
- [ ] All integration tests pass with zero/near-zero numerical error
- [ ] 11/11 FlashAttention tests pass via C++ path (both HEAD_DIM=32 and 64)
- [ ] Project test suite: 472+ pass, zero MSL fallbacks in debug output
- [ ] Upstream test count: 4,312+ pass, measurable C++ coverage increase
- [ ] `project_status.md` updated
