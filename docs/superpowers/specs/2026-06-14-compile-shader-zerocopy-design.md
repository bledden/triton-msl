# Zero-copy MPS execution via `torch.mps.compile_shader` — design (2026-06-14)

> Eliminate the per-launch host round-trip for torch MPS tensors by routing
> triton-metal's emitted MSL through PyTorch's own `torch.mps.compile_shader`,
> which dispatches against MPS tensors zero-copy. **Confirmed 10.2× on
> vector_add@16M (28 → 281 GB/s, 5% → 52% of the 546 GB/s peak).** Pure Python,
> no native shim; the existing driver remains the fallback. Phase 4 (perf), the
> #1 memory-bandwidth lever.

## Problem (profiled + confirmed)

`driver.py:605` host-round-trips every torch **MPS** tensor per launch:
`torch.mps.synchronize(); cpu_tensor = arg.cpu()` (GPU→host) then
`ctypes.memmove(dst, src, nbytes)` (host→Metal buffer), and a symmetric copy-back
for outputs. For vector_add@16M that is **201 MB through the host per launch**,
measured at **7.3 ms → 28 GB/s, flat across all BLOCK/num_warps** — proving the
bottleneck is the data movement, not the kernel codegen. (The non-MPS path right
below already zero-copies via `make_buffer_from_ptr`, but MPS `data_ptr()` is a
GPU virtual address — not CPU-dereferenceable, not page-aligned — so it can't be
wrapped, and PyTorch's underlying `MTLBuffer` is not reachable from pure Python.)

## Approach (confirmed by probes)

`torch.mps.compile_shader(msl_source)` compiles a Metal compute library; its
functions dispatch against MPS tensors **zero-copy** (PyTorch owns the buffers
and the MPS stream). Verified it supports the full triton-metal execution model:

- **Buffers** — MPS tensors bound positionally in `[[buffer(i)]]` order, zero-copy.
- **Scalars** — `constant T&` args (int, float) passed positionally.
- **Grid/threadgroup** — `lib.fn(*args, threads=<total>, group_size=<tg>)`, with
  `[[threadgroup_position_in_grid]]` (pid) and `[[thread_position_in_threadgroup]]`
  (lid). `threads`/`group_size` accept ints (1-D) or tuples (2-D/3-D — verified).
- **Threadgroup/shared memory + `threadgroup_barrier` + `simd_*`** — verified a
  shared-memory `simd_sum` reduction is correct.
- **Output write-back** — the kernel writes the MPS output tensor directly; after
  the dispatch PyTorch's stream makes it visible. No copy-back.

Rejected alternatives: (A) a native `getMTLBufferStorage` C++ shim to bind
PyTorch's buffer in our driver — real zero-copy but a fragile, version-dependent
native build the project shelved; (C) merely speeding up the copy — bounded, can't
beat zero-copy.

## Architecture

A new execution path selected in the driver launch when **all tensor args are MPS
tensors** and `torch.mps.compile_shader` is available; otherwise the existing
host-round-trip driver runs unchanged (purely additive — never a regression).

Components (small, single-responsibility):

1. **`CompileShaderRuntime`** (new, e.g. `triton_metal/backend/compile_shader_runtime.py`):
   - `available()` — `hasattr(torch.mps, "compile_shader")` (newer torch only).
   - `get_library(msl_source)` — `torch.mps.compile_shader(msl)`, **cached** keyed
     on the MSL string (compile once per kernel, reuse across launches).
   - `dispatch(lib, kernel_name, tensor_args, scalar_args, threads, group_size)` —
     `getattr(lib, kernel_name)(*tensor_args, *scalar_args, threads=threads,
     group_size=group_size)`.
2. **Driver integration** (`driver.py launch` / the launcher that owns the MSL):
   - Eligibility check: every tensor arg `is_mps` and the runtime is available and
     this kernel hasn't been marked unsupported.
   - **Grid translation:** the existing path dispatches `dispatchThreadgroups(grid,
     tg_size)` where `grid` = threadgroup counts and `tg_size` = threadgroup size
     (= `num_warps*32` from `kernel_metadata`). For `compile_shader`, `group_size =
     tg_size` and `threads = (grid.x*tg.x, grid.y*tg.y, grid.z*tg.z)` (total
     threads per dimension). Collapse to 1-D `int` when y=z=1 (the common case);
     pass a tuple for multi-dim grids.
   - **Arg marshaling:** pass the original torch MPS tensors (NOT copied) in
     buffer-index order, then the scalar args in order, matching the kernel
     signature the existing driver builds. Reuse the existing arg-ordering /
     output-index logic (`output_indices` from metadata) — outputs are just MPS
     tensors written in place.
   - **Kernel name:** the MSL's `kernel void <name>` — already known to the
     compiler (the function/entry-point name used for the metallib pipeline).
3. **Fallback:** any failure in eligibility, compile, or dispatch → run the
   existing driver path. Mark a kernel (by MSL hash) unsupported on first failure
   so we don't retry the slow-then-fail path every launch.

## Data flow

Triton launch → compiler emits MSL (unchanged) → driver: all-MPS-args? →
`CompileShaderRuntime.get_library(msl)` (cached) → `dispatch(...)` → result in the
MPS output tensor (zero-copy) → returns to torch. Non-MPS or unsupported →
existing driver (copy path).

## Error handling / integrity (prime directive: never silent-wrong)

- The `compile_shader` path must produce results **identical to tolerance** to the
  existing driver for every kernel. This is gated by the full-suite correctness
  run below before the path is enabled by default.
- Any exception (compile/dispatch/availability) falls back to the existing driver
  — correct, just slower. Never wrong output.
- A flag `TRITON_METAL_COMPILE_SHADER` (default-on once verified; `=0` escape
  hatch) toggles the path, mirroring `TRITON_METAL_MEPT` — so a regression can be
  bisected/disabled without a code change.
- Mixed args (some MPS, some CPU): only route through `compile_shader` when ALL
  tensor args are MPS; otherwise fall back (the copy path already handles mixed).

## Testing / validation (correctness FIRST, then perf)

1. **Full-suite parity (the gate):** the upstream `test_core` ratchet (both MEPT
   flags) and the project suite must pass with the `compile_shader` path **on**,
   matching the existing path. 0 failed. This is the prime-directive gate — no perf
   claim before it is green.
2. **Real kernels:** FlashAttention (11/11), the relay@256+num_warps=8 (vs torch
   reference), reductions/softmax, atomics — correct to tolerance through the
   `compile_shader` path.
3. **Parity harness:** a test that runs a representative set of kernels through
   BOTH paths and asserts identical (tolerance) outputs.
4. **Perf gate (after correctness):** vector_add / elementwise / softmax / reduce
   re-benched; assert the memory-bound class is materially above the 28 GB/s host-
   round-trip floor (vector_add ≳ 250 GB/s). Record in `reports/perf_baseline.json`.
5. **Fallback paths:** non-MPS tensors and a deliberately-unsupported kernel still
   run correctly via the existing driver; the unsupported-mark prevents retry.

## Open items the plan resolves

- Exact grid↔(threads, group_size) translation incl. 2-D/3-D Triton grids and the
  `needs_2d_grid` metadata flag; confirm against the existing
  `dispatchThreadgroups` call.
- Full scalar-type coverage (the driver currently marshals int/float/bool/i64/…):
  confirm each maps to a `constant T&` arg `compile_shader` accepts; fall back if
  a type isn't supported.
- The precise integration point that still has the **MSL source** (the launcher may
  only hold the compiled pipeline_state; thread the MSL/kernel-name through, or
  select the path earlier where `asm["msl"]` is available).
- Lib cache lifetime/eviction (keyed on MSL hash; bounded like the metallib cache).
- `compile_shader` availability across the supported torch versions + the
  graceful fallback when absent.

## Out of scope

- Vectorized-load / further codegen BW tuning (52% → higher) — a *separate*
  follow-up; this design removes the copy bottleneck, which is the 10× lever.
- Non-MPS (CPU-tensor) zero-copy — already handled by the existing page-aligned
  path; unchanged.
- A native `MTLBuffer` shim (rejected).
