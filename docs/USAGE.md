# Usage and installation

[Back to the project](https://github.com/bledden/triton-msl/blob/main/README.md) · [0.4.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.4.0.md) · [0.3.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.3.0.md)

These examples document the launch interfaces, not universal support or performance.
Historical throughput figures below are not requalified for 0.3.0 or 0.4.0.

Compiling MSL or metallib for Julia, XLA custom calls or your own Metal driver?
Read the [external-driver contract](EXTERNAL_DRIVER.md). The compiled library
does not carry the Python launcher's allocation, argument and assertion checks.

## Installation details

### Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later. 0.3.0 was validated through macOS 26.6 (Tahoe).
  0.4.0 was tested on macOS 26.7 with Xcode 27.2; older macOS releases and
  toolchains were not retested.
- Xcode Command Line Tools: `xcode-select --install`
  - On **macOS 26 (Tahoe) / Xcode 26**, the Metal shader compiler ships as a
    separate on-demand component. If `xcrun metal --version` reports a missing
    Metal Toolchain, install it once with
    `sudo xcodebuild -downloadComponent MetalToolchain`.
- Python 3.10+. 0.4.0 testing used CPython 3.14.4; Python 3.10 to 3.13 were not run.
- PyTorch 2.12+ (2.5+ for the zero-copy MPS fast path; `torch.compile` is developed + tested against 2.12) and Triton 3.7.0

### Install

```bash
pip install triton-msl

# Triton is required but installed separately. There is no official macOS wheel,
# so build the revision used by the documented 3.7.0 wheel:
pip install git+https://github.com/triton-lang/triton.git@4da2e268dcccf090370e7bb7b59940e7567bf0ce
```

The release bundle includes a native `triton-msl` wheel for **GIL-enabled CPython
3.14, macOS 15+, arm64**, a pure Python wheel, and a source archive. The native
wheel contains three CPU host helpers: `_validation_native`, `_packed_native`, and
`_binder_native`; the pure wheel contains none. Missing helpers may fall back individually;
a discoverable broken helper fails loudly. These are not GPU kernel binaries or the optional
C++ compiler route. Native measurements do not describe Python-only installations.
Source builds support validation/packed helpers on GIL-enabled CPython 3.13/3.14;
the binder is 3.14-only. No runtime-qualified CPython 3.13 native wheel is included.
Earlier build/import checks do not establish Torch/Triton runtime coverage. Building helpers
requires Xcode Command Line Tools and does not replace Triton's separate requirements.

### Experimental C++ compiler path

The optional C++ compiler remains default-off and is distinct from the three
host helpers in the release wheel. Its NVIDIA-style load/store cache and eviction
hints do not select an Apple cache policy. They are accepted but ignored on this
experimental path, and must not be used to establish synchronization or memory
ordering. Load volatility is a separate semantic requirement, not a cache hint.

With a rebuilt optional C++ helper, `TRITON_MSL_USE_CPP=1
TRITON_MSL_CPP_SCALAR_CFG=1` additionally opts into bounded LLVM scalar cleanup.
The second switch is default-off and participates in the cache policy. A positive
pointer-translation proof runs before the Metal typed-pointer adapter. An
unproved shape selects the unoptimized native lowering unchanged; Metal compiler
errors are not the eligibility gate. Unsupported byte-offset and chained array
GEPs therefore retain the accepted native route. This does not widen the C++
family admission rules.

Compilation metadata records `cpp_scalar_cfg_route` (`disabled`, `optimized`, or
`fallback`), `cpp_scalar_cfg_reason`, and hashes of the selected opaque IR and
typed IR. Opt-in attempts also record both candidate hashes; successful C++
binary production adds `cpp_scalar_cfg_metallib_sha256`. These are compilation
records; `binary_route` remains the selected binary route and the records alone
do not establish GPU execution or timing.

If you're on the exact platform tuple **Python 3.14 / macOS 15+ / Apple Silicon (M1–M5)**, an unofficial prebuilt Triton wheel is attached to the [GitHub releases](https://github.com/bledden/triton-msl/releases/tag/triton-wheel-3.7.0-cp314-macos-arm64) so you can skip the build:

```bash
pip install https://github.com/bledden/triton-msl/releases/download/triton-wheel-3.7.0-cp314-macos-arm64/triton-3.7.0+git4da2e268-cp314-cp314-macosx_15_0_arm64.whl
```

### Start-up identity record

Before the first kernel, triton-msl computes the identity of the Metal compiler,
the SDK and the Python frameworks it depends on. That identity enters every
kernel cache key. In 0.3.0, every new Python process recomputed it from full
file bytes.

0.4.0 keeps that full computation for the first process, whenever anything it
checks has changed, and on request. Otherwise a later process validates a small
per-user record and reuses the recorded identity. The record holds the identity the full computation
produced, so cache keys are the same in both modes unless a change listed
below goes unseen. On one M4 Max, the first `@triton.jit` call of a later process
with compiled kernels cached took a median 7.03 s with the record switched off and
0.96 s with it in use (native wheel, one workload, 24 processes each, toolchain
files already in the operating system's file cache, other work running on the
machine); see the
[0.4.0 guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.4.0.md#performance) for the table and its limits.

| Setting | Effect |
|---------|--------|
| default | The record is used |
| `TRITON_MSL_IDENTITY_RECORD=0` | Full computation in every process; no record is read or written |
| `TRITON_MSL_IDENTITY_RECORD=refresh` | Recompute and rewrite the record |
| `TRITON_MSL_IDENTITY_DIR=<path>` | Keep records in `<path>` instead of `~/.cache/triton_msl/identity-v1-uid<uid>/` |

Records are private to the user: the directory is mode 0700 and the files 0600.
Inspect or manage them with:

```bash
python -m triton_msl.tools.identity status
python -m triton_msl.tools.identity refresh
python -m triton_msl.tools.identity clear
python -m triton_msl.tools.identity verify   # runs the full computation and compares
```

`refresh` and `verify` reach the framework record of the command's own process
only, because that record depends on the native libraries a process has loaded.
For an application, use `clear`, or run the application once with
`TRITON_MSL_IDENTITY_RECORD=refresh`.

The record is used only where file metadata is on APFS and the record folder
carries no access control list. Otherwise the full computation runs.

The record relies on file metadata, so it does not see what only the full
computation sees:

- a byte change that leaves a file's size, times and inode unchanged (for example
  a write through a memory mapping that was held open across the change);
- a change to an SDK file other than the SDK's version files;
- a privileged change to file times or to the raw device;
- a record altered by the same user.

If your setup can change in these ways, set `TRITON_MSL_IDENTITY_RECORD=0`. The
full contract is in
[`docs/CACHE_CONTRACTS.md`](https://github.com/bledden/triton-msl/blob/main/docs/CACHE_CONTRACTS.md).

## Quick Start

### @triton.jit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cpu")
y = torch.randn(n, device="cpu")
out = torch.empty(n, device="cpu")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)
print(f"Max error: {(out - (x + y)).abs().max():.2e}")
```

Run a fuller, runnable version, vector-add, fused-softmax, and a tiled matmul, each
on a non-multiple shape (so every load/store boundary mask is exercised) and verified
against a NumPy reference, with [`examples/local_triton_dev.py`](https://github.com/bledden/triton-msl/blob/main/examples/local_triton_dev.py):

```bash
python examples/local_triton_dev.py
```

It's the same `@triton.jit` source you'd run on a CUDA GPU: develop and verify locally
on your Mac, then ship the identical kernels to NVIDIA.

### torch.compile

```python
import torch
import triton_msl.inductor
triton_msl.inductor.register_metal_triton_backend()

model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
).to("mps").eval()

# Registration selects Triton for Inductor's MPS device; there is no
# separate torch.compile backend named "metal".
compiled = torch.compile(model, backend="inductor")
x = torch.randn(32, 256, device="mps")
with torch.no_grad():
    out = compiled(x)
```

**Performance and coverage.** This enables Inductor's supported fused kernels on MPS;
it does not promise a speedup over eager MPS. Historical model coverage included transformer
blocks, a small GPT, a ViT, ResNets and LSTMs. See the [0.3.0 results](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.3.0.md) for the measured regressions and the [0.4.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.4.0.md) for the current validation scope.

**Coverage.** Transformer/attention, RNN, **CNN (incl. BatchNorm)**, normalization, and the
reductions (sum, product, mean, max/min incl. NaN-propagating, var/std, argmax/argmin, softmax,
logsumexp, cumsum/cumprod), including small under-filling reductions **and a 2-D reduction
fused with a 2-D scan in one kernel** (e.g. `x.sum(1) + x.cumprod(1)[:,-1]`), compile and match
eager. Anything genuinely beyond the hardware (a reduction/scan tile exceeding Metal's 1024
threads/threadgroup) is **refused loudly rather than mis-computed** (`MetalNonRecoverableError`,
never silent-wrong); the rest of the graph is unaffected.

### MLX

```python
import mlx.core as mx
import triton
import triton.language as tl
from triton_msl.mlx import triton_call

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = mx.random.normal((n,))
y = mx.random.normal((n,))
out = mx.zeros((n,))
results = triton_call(add_kernel, x, y, out, n, grid=(4,), BLOCK=256)
```

The MLX route re-launches the *same* emitted MSL through `mx.fast.metal_kernel`, so it inherits
every kernel-level guarantee of the torch path. Its own contract is narrower and enforced
(`MetalNonRecoverableError` before any dispatch): one kernel per launch with no template dispatch
descriptor (attention / matmul templates take the torch.mps path); outputs are **fresh** arrays, so a
kernel that reads or atomically updates an output pointer refuses; array dtypes must be in the
signature map and integer arguments must fit int32. One thing the route cannot check with MLX 0.32
(`mx.array` exposes neither strides nor contiguity): MLX hands the kernel a **row-contiguous copy** of
every input, so the stride arguments you pass must describe the row-contiguous layout of the array's
shape, not the strides of a transposed or sliced view.

### MPS tensors: zero-copy

Eligible `@triton.jit` kernels run **zero-copy** on `torch` MPS tensors: the driver
dispatches the emitted Metal through `torch.mps.compile_shader`, avoiding the host
round-trip (on by default, no code change):

```python
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty(n, device="mps")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)  # runs on the GPU, no copy
```

Generic 2-D kernels can use this route with a recognized full-K allocation
descriptor after its bounds check succeeds for the current invocation. The
descriptor does not itself force a host round trip. Tensor layout, argument ABI,
geometry and ownership checks still apply. An unmasked partial K requires enough
backing storage for the source's entire final block; otherwise the launch refuses
before submission. Offset or gappy transposes retain the host route. A failure
after possible submission never retries the kernel through another route.

### Matmul (`tl.dot`)

Strided operands, including a transposed B, use source-width address expressions.
A bounded 32×32 fp32 tile stages operands in 8,448 bytes of shared memory. When
the compiler supports matrix instructions, the SIMD width is 32 and the effective
K is divisible by 32, it uses fp32 matrix fragments without converting inputs to
half precision. Other reductions use the staged scalar loop in K order. Other
tile/type cases and exceptional index ranges retain the original scalar path.
Inline matmul, fused matmul epilogues and constant-dot templates use positional
compiler-owned argument names, so source names such as `m`, `n`, `k` or `c`
cannot be shadowed by their generated loops. Source names remain available for validation; launch metadata retains source
argument positions. This does not claim name hygiene for every other template. Matrix reduction can associate differently; numerical
acceptance is unchanged. This optimization does not bypass launch checks or
establish an ordering for overlapping reads/writes across independent programs.

A K-loop matmul with an fp32 output, unit inner strides and `M`, `N`, `K`
arguments also carries a second entry point in the same shader; with a transposed
B the operands must be fp32 as well. The launcher
selects it only when M and N are multiples of the tile, K is a multiple of 8, the
output shares its storage with no other pointer argument and the output holds a
whole tile for every program in the grid; every other launch uses the ordinary
entry. Where the source masks its store by an extent, a program whose tile starts
outside that extent stores nothing. In that entry each 32-lane SIMD group
computes whole 32×32 sub-tiles with matrix fragments read in place, a transposed B
through the transposing fragment load, so no operand is staged. A tile of
32·p × 32·q runs at `32 * min(p * q, 4)` threads per group. A contiguous tile
that does not split into 32×32 sub-tiles keeps the earlier four-strip entry at
128 threads. Both entries re-prove the SIMD layout at run time and fall back to
the scalar loop at the same width when it does not hold. A single program whose
output is one of its operands keeps the ordinary entry, which loads its whole tile
before it stores; overlapping reads and writes across programs remain unordered.

The ordinary and second entries derive output clips from the source store mask,
not from scalar names. An unmasked axis stores every launched tile in full, even
when an argument named `M` or `N` is a stride or a different dimension. Source
load masks remain separate: an unmasked output can contain zeros or nonfinite
products from masked operand contributions. Flattened and grouped grids retain
their own source divisors. Grid domains with unproved nonpositive or wrapping
divisors refuse before dispatch.

When full source tiles require a grid address plan, both entries validate operand
and output backing allocations and source address widths before dispatch, including
aligned K. An incomplete flattened or grouped grid uses its exact tile union for
the output proof. The check can add host work per launch; its cost and the integrated
shader performance require fresh qualification and are not established by earlier
matmul timings.

Eligible contiguous fp16/fp32 matmuls (K%8) on MPS tensors take a direct simdgroup-matrix path, dispatched
zero-copy. A deterministic, occupancy-gated tile selector picks the coarsest blocking the
shape's M and N alignment allow: **M%32 + N%32** runs the `(4,4)` tile at ~11–12 TFLOP/s on
M4 Max; **M%8 or N%8** (not %32) runs a finer rescue tile (lower, but still far above the
generic path). **M%8 ≠ 0 or N%8 ≠ 0** (e.g. an odd vocab/hidden dim) falls to the ~2.4 TFLOP/s
generic path: a partial simdgroup strip can't be masked without writing past the matrix.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b))
        a += BK * sak; b += BK * sbk
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc.to(tl.float16))

M = N = K = 2048
A = torch.randn(M, K, device="mps", dtype=torch.float16)
B = torch.randn(K, N, device="mps", dtype=torch.float16)
C = torch.empty(M, N, device="mps", dtype=torch.float16)
matmul_kernel[(M // 64, N // 64)](
    A, B, C, M, N, K,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    BM=64, BN=64, BK=32)
```

### Unsupported kernels and explicit refusals

The intended behavior for a kernel whose lowering cannot be proved is an explicit
`MetalNonRecoverableError`. This is a design contract, not a proof that no defects remain. For example, a pid-tiled matmul that bakes its M/N
dims as `constexpr` (so the true output strides can't be recovered) is refused:

```python
from triton_msl.errors import MetalNonRecoverableError

@triton.jit
def matmul_baked_dims(a_ptr, b_ptr, c_ptr, K,
                      M: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * K + offk[None, :])
    b = b_ptr + (offk[:, None] * BN + offn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b)); a += BK; b += BK * BN
    tl.store(c_ptr + (offm[:, None] * BN + offn[None, :]), acc)

try:
    matmul_baked_dims[(2, 2)](A, B, C, K, M=64, N=64, BM=32, BN=32, BK=32)
except MetalNonRecoverableError as e:
    print("refused (not silent-wrong):", e)
```

See [`docs/SUPPORTED_OPS.md`](https://github.com/bledden/triton-msl/blob/main/docs/SUPPORTED_OPS.md) for the full op/dtype support
matrix and the loud-refusal catalog.

### FlashAttention

See [the 0.3.0 attention guide and measured comparisons](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.3.0.md#flashattention) and [the attention operation guide](https://github.com/bledden/triton-msl/blob/main/docs/attention_ops.md).

### Frontier attention: linear/delta and a trainable backward

Two attention capabilities beyond the softmax-FA path ship as **direct Metal ops** (neither
is expressible as a single `@triton.jit` kernel), documented in
[`docs/attention_ops.md`](https://github.com/bledden/triton-msl/blob/main/docs/attention_ops.md):

- **KDA (Kimi Delta Attention / gated DeltaNet)** — linear/delta-rule attention with a
  per-key-dimension forget gate, the direction the newest models are taking. Chunked MMA
  prefill + recurrent decode, fp16/fp32, validated against a recurrent reference
  (`triton_msl.kda`).
- **A FlashAttention backward pass** — `triton_msl.fa_backward.flash_attention` is a
  `torch.autograd.Function` whose dQ/dK/dV run on Metal (tiled FA-2 backward, MMA, causal +
  full, fp16/fp32). The forward FA was inference-only; this makes attention **trainable** on
  the GPU.

### Quantized inference (int8 / int4)

Weight-only int8 and int4 matmuls **auto-route** to dedicated dequantizing kernels, in both
the natural `[K, N]` layout and the GPTQ-style `[N, K]`, dispatched zero-copy through
`compile_shader`. The path is **fail-closed**: a shape or dtype the fast kernel can't handle
is **refused** (`MetalNonRecoverableError`), never silent-wrong.

The historical decode comparison reported an int8 weight-only GEMV at about **3.7×**
fp32 decode while moving a quarter of the bytes; this speed ratio and roofline claim are
**unqualified on the current candidate**. Prefill (GEMM) is a memory-footprint
win rather than a speed one — fp32 MPS BLAS still runs the prefill matmul faster. int4 adds
per-group decode with zero points, and the skinny/deep matmul shapes that were
occupancy-starved gained a deterministic two-pass split-K.

## Tuning flags

The first four flags are default-on and can be disabled for diagnosis.
The legacy parser and half accumulation are separate, default-off opt-ins:

| Flag | Effect when disabled |
|------|----------------------|
| `TRITON_MSL_COMPILE_SHADER=0` | Use the host-copy driver instead of the zero-copy `compile_shader` dispatch |
| `TRITON_MSL_FAST_MATMUL=0` | Use the generic matmul instead of the fast simdgroup-matrix path |
| `TRITON_MSL_MATMUL_AUTOTUNE=0` | Pin matmul tile selection to the fixed `(4,4)` blocking (M%32≠0 / N%32≠0 shapes drop to the generic path instead of the finer M%8/N%8 rescue tiles) |
| `TRITON_MSL_MEPT=0` | Disable the multi-element-per-thread register-array model |
| `TRITON_MSL_LEGACY=1` | Opt **in** to the heuristic legacy text parser (off by default, it can be silent-wrong) |
| `TRITON_MSL_FA_HALF_ACCUM=1` | Opt **in** to fp16 (half) MMA accumulators in FlashAttention. Off by default; fp16 kernels only, a no-op for fp32. Accuracy depends on the inputs, shape and accumulation order; no universal maximum-error bound is provided. Validate it for your workload before enabling it. The historically reported ~4% speedup has not been revalidated on this tree. |
| `TRITON_MSL_LAUNCH_MEMO=0` | Run every launch check on every launch. By default a launcher remembers a check that passed (the matmul grid storage proof and the fast matmul output-overlap check) under every value the check reads: grid, scalar payloads, and per role tensor the storage, its size, the view offset, element width and data pointer. A later launch with the same values skips the check; any other value, a non-exact tensor or scalar type, or a launch with an enter hook takes the check. A refusal is never remembered. |
| `TRITON_MSL_LAUNCH_MEMO_VERIFY=1` | Test mode for the memo: on every remembered launch the check runs too, at the same point, and a disagreement raises `LaunchMemoDisagreement` instead of launching. |
| `TRITON_MSL_LAUNCH_MEMO_DIAG=1` | Count memo hits, misses, bypasses and evictions per kernel and check (`triton_msl.backend._launch_memo.diagnostics()`). Off by default so the counting does not cost time. All three memo flags are read once, when the backend is imported. |


### Experimental host storage extent borrowing

`TRITON_MSL_HOST_EXTENT=1` opts CPU tensors on the synchronous host launch route into
borrowing a page-rounded backing span when the native helper proves sufficient default
malloc-zone backing and one covering read-write VM region. The default is off; unproved
storage uses the existing whole-storage mirror. Existing exact-page borrowing is unchanged.
Small, unaligned and overflowing spans issue no allocator or VM queries during launch.
The selected policy is recorded as `false`, `true`, or `"unavailable"` in `HOST_EXTENT`
in the execution stamp; absent/older helpers and failed helper self-tests are unavailable.

At helper load, a bounded self-test checks a positive allocation and two page-aligned
interior pointers. If it fails, borrowing is disabled for that helper instance. This is
an observation of the current allocator, not a proof for every pointer or allocator.
Inspect `triton_msl.backend._cache_contract._validation_native.host_extent_selftest_reason`
when the helper is present: `selftest_page_size` means unsupported page geometry,
`selftest_allocation` means the test allocation failed, `selftest_positive` means the
positive allocation did not prove the required rounded extent, and `selftest_interior`
means an interior pointer did not produce the required refusal. A conforming allocator
that reports only the exact requested size can fail `selftest_positive`; this conservatively
disables the opt-in. `selftest_forced_failure` is reserved for a test helper build.
An absent/older helper may have no reason attribute.

The mechanism assumes a conforming default malloc zone and that nonzero `malloc_size`
identifies a block base. Custom array/tensor allocators delegating to that zone are admitted
on the same terms; a malloc zone is not the identity of the outer tensor allocator.

The owner is retained and checked against the launch's storage snapshot before and after
the query; a mismatch refuses the launch before copying or dispatch. A fresh owner on a
later call is assessed afresh. Concurrent resize/free is outside the valid-storage contract,
as with the existing exact-page and mirror paths. No address, decision or buffer is cached
across launches by this mechanism. Synchronous completion and alias grouping are unchanged.

Rounding never moves the pointer. Bytes between the storage length and the rounded length
may contain allocator-private data. Correct, bounded kernels must not access those bytes;
an out-of-bounds write there would hit allocator-owned data rather than mirror scratch.
This flag does not provide bounds checking for arbitrary kernels.

For diagnostics, `triton_msl.backend._host_extent.last_launch()` returns a copied list of
host preparation decisions in the current execution context, or `None` when no opt-in
host decisions were recorded. Records include route, reason, logical storage bytes and
borrowed bytes; they do not certify successful GPU execution. The trace is also attached
as `host_extent` to available launch metadata for the exit hook. Failed wrap/allocation,
validation and device-assert exceptions remain visible to the caller.

After a device assertion, borrowed outputs (including the existing exact-page path) may
contain partial writes already performed by the GPU. Mirrored output groups are not
copied back when that assertion raises, so those groups remain untouched. The successful
assert-before-store control does not establish rollback for other kernels.

### Host dispatch failures with uncertain completion

The synchronous host launcher retains its Metal bindings, pooled backing mappings and
borrowed storage owners before calling the runtime. If submission or waiting raises, the
original exception propagates without replay or mirror copy-back. Those resources remain
owned independently of the exception traceback and caller variables.

Before another host launch, failed records are inspected. A failure before any commit
attempt can be retired; after a commit attempt, only the matching command buffer's
`Completed` or `Error` status proves that GPU access has ended. A missing command witness,
a nonterminal status or a status-query error retains the record and refuses the new host
launch with `triton_msl.errors.PostSubmitError`. This refusal is new in 0.4.0. Failed records are discarded after that proof; their buffers are not recycled
into the pool and their results are never copied back later. Normal successful launches
retain the existing pool cleanup and output behavior. This protection also applies with
host extent borrowing off; it does not change the separate `compile_shader` route or
supply storage ownership for callers of the raw `MetalUtils.launch` API.

`triton_msl.backend._host_dispatch.reap_completed()` performs the same nonblocking check
and returns the number of unresolved failed records. Zero is not a global GPU-idle barrier
and says nothing about other active work. If completion cannot be observed, the resources
stay retained until worker teardown; there is no timeout or force-clear operation. A new
host launch is refused while any failed record remains unresolved. Already-active launches
keep their separate ownership records.

Retention prevents garbage collection from releasing backing memory; it cannot prevent
explicit storage resize, external free, or CPU mutation. Treat all allocations bound to
an uncertain failed launch as still in use until its completion is established. Borrowed
outputs may contain partial writes and failed results remain invalid even after retirement.
