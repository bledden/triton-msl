# Using generated Metal kernels with your own runtime

This guide is for integrations that compile with triton-msl, then load the MSL or
metallib in another runtime, such as a Julia/XLA custom call or a custom PyObjC
driver. It describes the current implementation, not a stable AOT export API.
Pin the compiler revision and qualify your adapter against that revision.

**A compiled library is not a complete launch contract.** Checks performed by
`MetalLauncher` do not run when another driver submits the kernel. Keep the
generated source, binary, specialization and metadata together. If your adapter
does not implement a required contract, reject that kernel before submission.

## What must travel with the kernel

Keep the following as one versioned artifact, not independently interchangeable
cache entries:

- The triton-msl and Triton revisions, toolchain/runtime information, compile
  options, target and source/specialization identity.
- The emitted MSL, metallib bytes and their hashes, plus the exact entry point.
- The specialized runtime signature, including scalar widths and argument order.
  Constexpr and equal-to-one specialization can remove arguments. The original
  Python function's parameter list is not a reliable Metal binding table.
- The full compiler metadata, including `block_size`, `needs_2d_grid`,
  `binary_route`, `launch_contract`, `device_assert` and any template or bounds
  descriptors. Missing information is not proof that no checks are needed.

These are internal fields, not a portable serialization schema. The sealed
`launch_contract` describes producer metadata; it neither authenticates a foreign
pipeline nor proves that a caller supplied valid buffers. Do not transplant a
contract or cached binary across compiler versions to bypass recompilation.

### Initialize device dependencies before capturing origin

The live execution identity includes the loaded framework images. Creating a
Metal device can load driver images and change that identity. Initialize the API,
device and queue before capturing compilation origin, then compile new products
under that origin. Keep checking the live identity when loading and submitting.
If later initialization changes a dependency, recompile after initialization;
never rewrite an old product's origin to make it appear current.

The finite external-loader qualification on M4 Max observed this ordering in both
private native-helper and helper-free installations. Its two source-defined
products passed through real submission, wait, status, readback and close without
a further identity change. A separately accepted window established the same
ordering for a reconstructed grouped fp16 matmul at A[64,32] by B[32,64], including
full-grid and partial-grid publication in both installed modes. It does not cover
the reporter's exact kernel or general fp16 rounding. That evidence does not guarantee stability for other
libraries, kernels or initialization sequences. Accepted cases and remaining
limits are recorded in
[`engineering/external-loader-coverage-20261008.json`](../engineering/external-loader-coverage-20261008.json).

## Binding and dispatch responsibilities

| Area | What the external runtime must preserve |
| --- | --- |
| Buffer ABI | Bind the buffers and scalar representations required by the **emitted entry point**, including synthetic arguments. Preserve declaration widths, signedness and offsets. |
| Argument packing | For the generic emitter's overflow ABI, bind pointer buffers first and one packed scalar buffer after them. Do not use one buffer per original argument. Details below. |
| Geometry | Reproduce the selected route's grid mapping and threads per group. A Triton grid counts programs, not individual Metal threads. |
| Storage | Preserve view offsets, runtime strides, shared backing allocations and aliasing. Check the physical allocation, not just logical tensor dimensions. |
| Tail/batched bounds | Implement the applicable pre-dispatch bounds proof or decline the kernel. Those driver checks are not embedded allocation guards. |
| Device assertions | Bind a zeroed per-invocation status buffer, wait, read it and invalidate failed results before exposing them. |
| Replacement routes | Implement the descriptor's complete argument/layout transformation or decline it. A JIT signature does not necessarily match a replacement kernel. |
| Submission | Propagate execution errors. Do not retry another route after a dispatch may have begun. Keep buffers and pipelines alive through completion. |

### Scalars and the 31-slot limit

Ordinary arguments use the emitted `[[buffer(i)]]` declarations. Do not assume
every integer is 32-bit or every floating scalar is a host double. Read the
declaration and specialized signature together; reject unrepresentable values.

When the generic emitter has more than 31 arguments, it can pack supported
scalars into one `constant uint* _packed_scalars` buffer:

1. Each pointer keeps its own slot, in pointer-relative source order.
2. The packed scalar buffer occupies the next slot.
3. Scalars occupy 32-bit words in scalar-relative source order. Float32 values
   use their bit representation, signed and unsigned integers use their 32-bit
   pattern, and booleans 0 or 1.

This path supports the emitter's `int`, `uint`, `float` and `bool` scalar forms.
It refuses unsupported scalar widths and cases where the pointer buffers plus
the packed buffer cannot fit. This is scalar-buffer packing, not permission to
pack arbitrary Metal pointers. Template signatures and assertion-bearing kernels
have their own restrictions; inspect the actual emitted signature.

In 0.3.0, the Python MPS `compile_shader` route implements this overflow packing.
The host-copy route refuses an over-limit binding before any dispatch instead of
implementing the packed ABI. An external metallib consumer needs its own matching
implementation.

### Grid and threadgroup geometry

For the generic host route, `host_launch_geometry` currently maps a program grid
`(gx, gy, gz)` to `(gx, gy, gz)` groups when `needs_2d_grid` is true, otherwise
to `(gx * gy * gz, 1, 1)`. Its threads per group are
`(min(block_size, 1024), 1, 1)`. This describes that route, not a universal recipe.
Check the chosen pipeline's actual limits as well.

Replacement matmul, attention and quantized kernels can use different geometry
and ABI. In particular, the MLA route prepares concatenated inputs. Its original
JIT arguments cannot simply be bound to the replacement entry point. A metallib
may also contain staged and direct matmul functions; choosing the direct function
requires the corresponding launch conditions, not merely finding its name.

MSL and metallib are not always two representations of the same lowering route.
With optional C++ compilation enabled, `binary_route="cpp"` identifies how the
metallib was produced, while an MSL source is still generated separately. It does
not prove which route was dispatched. Keep C++ opt-in and its coverage limits
separate from the default MSL path.

### Allocation checks and K tails

An unmasked K loop can read an entire final block even when logical K is not a
block multiple. The `matmul_full_k_storage_v1` descriptor records the obligation
checked by the Python launcher: source-width arithmetic, current dimensions and
strides, offsets and allocation reach. Exact-K allocations may be too small.
Providing padded storage makes an access addressable, but does not itself make
the padding numerically neutral. Use source masks when the intended computation
excludes the tail, or initialize and account for every unmasked value.

The separate `batched_dot_host_bounds_v1` descriptor also requires allocation
checks. Preserve offsets and aliases when copying or wrapping storage. Giving two
overlapping views independent dense buffers can change the program even if each
copy looks correct. The Python driver refuses certain float64 alias groups
because separate float32 conversions cannot preserve their shared storage.
It does not provide native float64 Metal arithmetic.

These descriptors cover specific proven routes, not a general memory-safety
certificate for arbitrary Triton source. A foreign driver must retain its own
buffer ownership and lifetime checks.

### Device assertions and result visibility

If `device_assert` is present, bind a zero-initialized 32-bit status buffer at
`buffer_index`. A nonzero status indexes `messages` starting at one. Zero means
no recorded assertion; an invalid status is an error, not a valid result.

The Python boundary waits and checks immediately. To retain that behavior, check
completion and status before returning successful results or copying them back.
Do not reuse a status buffer while another invocation can still write or observe
it, including nested calls. Do not replay a failed dispatch through another
backend. A custom asynchronous design changes this contract and must describe
its own error and result-visibility boundary.

## Qualifying an adapter

Start with a small kernel using the actual compiler and dispatch route. Compare
results against an independent reference and record the executed entry point,
binary hash, buffer binding table, grid and threads per group. Include a masked
tail, an offset/aliased view where supported, representative scalar types and
an assertion success/failure pair. Test unsupported contracts as pre-dispatch
refusals. Do not execute an out-of-bounds shader to test a missing host guard.

For triangle-attention/custom-call integrations, retain each forward/backward
kernel's own specialization, geometry and ABI. A dimension-only cache key cannot
distinguish an old compiler, changed masks or a new argument-packing scheme.
Recompile in a fresh cache when comparing the original and edited kernel sources.
Neither this guide nor the repository's focused kernel tests certify a complete
protein-folding application or a third-party driver.

For a useful report, share the smallest redistributable kernel, compile signature
and constants, launch/binding description, versions and exact diagnostic. Synthetic
inputs are sufficient; model weights, private sequences, credentials and full
application code are not required. Include whether triton-msl dispatched the
kernel or only produced its source/binary. Python launch-overhead measurements do
not describe a custom driver's execution loop; compiler cold-start costs may
still matter when it uses the compilation path.

## Implementation and test references

- ABI: [`msl_emitter.py`](../triton_msl/codegen/msl_emitter.py),
  [`_launch_signature.py`](../triton_msl/backend/_launch_signature.py), and
  [`test_issue4_trifast.py`](../tests/test_issue4_trifast.py).
- Metadata and geometry: [`_launch_contract.py`](../triton_msl/backend/_launch_contract.py),
  [`_generic_launch.py`](../triton_msl/backend/_generic_launch.py).
- Bounds/storage: [`driver.py`](../triton_msl/backend/driver.py),
  [`test_matmul_tail_storage_contract.py`](../tests/test_matmul_tail_storage_contract.py),
  [`test_runtime_storage_pipeline_scalar.py`](../tests/test_runtime_storage_pipeline_scalar.py).
- Assertions/submission: [`_device_assert.py`](../triton_msl/backend/_device_assert.py),
  [`test_dispatch_submission_boundary.py`](../tests/test_dispatch_submission_boundary.py).
- C++ route accounting: [`test_cpp_route_contract.py`](../tests/test_cpp_route_contract.py).

See also [usage](USAGE.md) and [the release limitations](RELEASE_CANDIDATE_LIMITATIONS.md).
