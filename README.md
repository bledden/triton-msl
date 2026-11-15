# triton-msl

Write standard [Triton](https://github.com/triton-lang/triton) kernels and run them
on your Mac's GPU. triton-msl compiles `@triton.jit` kernels to Metal for Apple Silicon.

```text
@triton.jit → Triton IR → Metal Shading Language → Apple GPU
```

Develop kernel logic locally, then validate and tune it on your target GPU.
The tested sources also run through Triton's NVIDIA and AMD backends; support,
numerical precision and performance depend on the backend.

**Status: alpha.** The goal is to preserve the source computation or refuse an
unsupported kernel explicitly. This is not a universal correctness guarantee.

## What's new in 0.4.0

0.4.0 runs on the Xcode 27 toolchain, which 0.3.0 refuses, and fixes defects found
in 0.3.0. These include an out-of-bounds read on the opt-in C++ route and a buffer
lifetime gap after a failed host launch. It adds a `triton-msl-export` command for
external Metal drivers, a direct route for full-storage dense transposes and
default-off attention fragment compositions. A later Python process can reuse a
validated per-user record of the compiler, SDK and framework identities instead of
recomputing them from file bytes.

No general speedup is claimed. The 0.4.0 release build has not been timed against
0.3.0; an October comparison of a development build against 0.3.0 resolved none of
36 cases as faster or slower. The per-launch costs that 0.3.0 added over 0.2.0
remain. [Read the 0.4.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.4.0.md)
for compatibility, fixes, performance and limitations.

## What's new in 0.3.0

This update fixes silent-wrong results, expands attention and matmul coverage,
implements retained device assertions, and tightens unsupported-case handling.
It includes bounded recoveries for grouped/flat-grid matmul, separate-length and
biased attention, single-query decode, and matmul epilogues.

**There are performance tradeoffs.** The measured warm comparisons retain 13
native-wheel and 14 pure-wheel workloads more than 10% slower than the previous
main. Cold specialization is also slower: one formatted-candidate observation
put the first add at 9.48 s versus 0.63 s. These are scoped measurements, not a
prediction for every model.

[Read the 0.3.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.3.0.md)
for the improvements, validation results, performance comparisons and limitations.
See [GitHub releases](https://github.com/bledden/triton-msl/releases) for published
artifacts and their hashes.

## Install

You need an Apple Silicon Mac, macOS 14+, Python 3.10+, Xcode Command Line Tools,
PyTorch and Triton. The current integration is developed against PyTorch 2.12
and Triton 3.7.0.

```bash
xcode-select --install
pip install torch triton-msl

# Triton is installed separately; use the documented 3.7.0 wheel's source revision:
pip install git+https://github.com/triton-lang/triton.git@4da2e268dcccf090370e7bb7b59940e7567bf0ce
```

`pip install triton-msl` installs the version currently published on PyPI,
which may lag main. For source development, see
[CONTRIBUTING.md](https://github.com/bledden/triton-msl/blob/main/CONTRIBUTING.md).

The native wheel targets **GIL-enabled CPython 3.14, macOS 15+, arm64**.
Other supported setups use the Python fallback, with different latency.
See [installation details](https://github.com/bledden/triton-msl/blob/main/docs/USAGE.md#installation-details)
for the Metal compiler component, optional host helpers and the matching
unofficial Triton wheel. M1 compatibility is a target; 0.4.0 was tested on one
M4 Max, not on M1 hardware.

## Quick start

Save this as a Python file and run it on your Mac:

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


n = 1024
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
torch.mps.synchronize()
print((out - (x + y)).abs().max().item())
```

For `torch.compile`, register the MPS integration and use PyTorch's
`inductor` backend—not a backend named `metal`.
[More examples](https://github.com/bledden/triton-msl/blob/main/docs/USAGE.md)
cover registration, MLX, matmul, attention and tuning flags.

## Capabilities and limits

- Elementwise kernels, reductions, scans, tensor operations and supported atomics.
- Strided matmul, bounded epilogues, and weight-only int8/int4 routes.
- FlashAttention, bounded decode/biased-attention and MLA routes.
- PyTorch Inductor integration; a narrower MLX launch interface.
- Direct Metal operations for KDA and FlashAttention backward.

Support depends on the source, dtype, layout and tile size. FP64 GPU arithmetic,
microscaling and unproved source contracts are not generally available.
The optional C++ route is unaudited and off by default. The reported full
AlphaFold/trifast workflow is not validated end-to-end.

Native unchanged-state validation skips Python observation callbacks by design.
Set `TRITON_MSL_IDENTITY_FAST_PATH=0` if those callbacks are required; see the
[observation contract](https://github.com/bledden/triton-msl/blob/main/docs/RELEASE_CANDIDATE_LIMITATIONS.md#validation-observation-boundary-and-opt-out).
Argument checks and immediate device-assertion observation remain enabled.

See the [support matrix](https://github.com/bledden/triton-msl/blob/main/docs/SUPPORTED_OPS.md)
and [capability register](https://github.com/bledden/triton-msl/blob/main/docs/CAPABILITY_AND_REFUSAL_CENSUS.md)
for specific envelopes. Cross-vendor tests establish parity only for their named
sources and precision settings—not universal byte-identical results.

## Documentation

- [0.4.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.4.0.md)
  — changes, performance, validation and known limitations.
  [0.3.0 release guide](https://github.com/bledden/triton-msl/blob/main/docs/releases/0.3.0.md).
- [Usage](https://github.com/bledden/triton-msl/blob/main/docs/USAGE.md)
  · [Architecture](https://github.com/bledden/triton-msl/blob/main/docs/ARCHITECTURE.md)
  · [Portability](https://github.com/bledden/triton-msl/blob/main/PORTABILITY.md)
- [Changelog](https://github.com/bledden/triton-msl/blob/main/CHANGELOG.md)
  · [Contributing](https://github.com/bledden/triton-msl/blob/main/CONTRIBUTING.md)
  · [Citing](https://github.com/bledden/triton-msl/blob/main/CITING.md)
  · [References](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md)

[MIT license](https://github.com/bledden/triton-msl/blob/main/LICENSE). The vendored
`applegpu` decoder in `triton_msl/_vendor/applegpu` is BSD-3-Clause.
