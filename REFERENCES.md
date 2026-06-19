# References

This document collects the papers, projects, and specifications that
`triton-msl` builds on. The codebase and documentation cite back to entries
here using `[N]` numeric references.

If you cite `triton-msl` in academic work, see
[`CITING.md`](CITING.md) for a suggested BibTeX entry.

---

## Triton compiler

[1] Philippe Tillet, H.T. Kung, David Cox. **"Triton: An Intermediate
Language and Compiler for Tiled Neural Network Computations."** *Proceedings
of the 3rd ACM SIGPLAN International Workshop on Machine Learning and
Programming Languages (MAPL '19)*, June 2019, pages 10–19.
DOI: `10.1145/3315508.3329973`. The original Triton paper; introduces the
tile-based programming model and the compiler stack this backend plugs into.

[2] **OpenAI/Triton repository.** <https://github.com/triton-lang/triton>.
The actively-developed Triton compiler and language; `triton-msl` is a
third-party Metal backend for it. Aligned to release tag `3.7.0` at time of
writing.

[3] **Triton third-party backend plugin architecture
(`TRITON_EXT_ENABLED`).** Upstream design discussion and tracking PR:
<https://github.com/triton-lang/triton/pull/9783>. `triton-msl` is
structured to be compatible with this plugin loading mechanism.

## FlashAttention

[4] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré.
**"FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness."** *Advances in Neural Information Processing Systems
(NeurIPS) 35*, 2022. arXiv: `2205.14135`. The original tiled-attention
algorithm; the IO-aware tiling and online-softmax accumulation pattern this
backend's attention path implements.

[5] Tri Dao. **"FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning."** 2023. arXiv: `2307.08691`. Improvements over [4]
in work partitioning; informs the K-loop staging in this backend's FA
HEAD_DIM=32 path.

## Online softmax

[6] Maxim Milakov, Natalia Gimelshein. **"Online normalizer calculation for
softmax."** 2018. arXiv: `1805.02867`. The single-pass max+sum trick used
in `_lower_softmax_template`, `_lower_layer_norm_template`, and the FA
attention accumulator. Cited inline in `docs/ARCHITECTURE.md` where those
templates are described.

## Apple Silicon / MLX / MPS

[7] **MLX: An array framework for Apple silicon.** Apple Machine Learning
Research. <https://github.com/ml-explore/mlx>. Used by `triton-msl/mlx/`
as a zero-copy kernel dispatch surface (`mx.fast.metal_kernel`). The MLX
hand-tuned kernels are also the comparison baseline for this backend's
performance harness (see WS0 component C6).

[8] **Metal Shading Language Specification.** Apple, current version.
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>.
The source language `triton-msl` emits. Notable intrinsics used:
`simdgroup_multiply_accumulate` (the hardware MMA), `simdgroup_load`/`store`,
`threadgroup` memory address space, `simd_*` reductions.

[9] **Apple GPU performance counter APIs** (`MTLCounterSampleBuffer`,
`MTLCommandBuffer.GPUStartTime`/`GPUEndTime`, Xcode GPU capture, Instruments
Metal System Trace). Apple Developer Documentation, current version. Used
by the WS0/C6 hardware-profiling harness to define "optimal bounds" as
"limiting counter saturated."

## AGX ISA (reverse-engineered)

[10] **Asahi Linux project / Mesa AGX driver.** Alyssa Rosenzweig and
contributors. <https://asahilinux.org>; AGX compiler in
<https://gitlab.freedesktop.org/mesa/mesa>. Open-source reverse-engineered
toolchain for the Apple GPU. `triton-msl` uses the disassembly capability
read-only (for the WS0/C6 harness); the WS3 experimental research track
considers using its assembler/emission capability.

[11] **`applegpu`.** Dougall Johnson. <https://github.com/dougallj/applegpu>.
Apple GPU ISA disassembler and reverse-engineering documentation.
Companion to [10]; the primary disassembly entry point used by the
hardware harness in WS0/C6.

## PyTorch / Inductor

[12] Jason Ansel, et al. **"PyTorch 2: Faster Machine Learning Through
Dynamic Python Bytecode Transformation and Graph Compilation."**
*Proceedings of the 29th ACM International Conference on Architectural
Support for Programming Languages and Operating Systems (ASPLOS '24)*,
April 2024. The TorchDynamo / TorchInductor architecture this backend
integrates with via `triton_msl/inductor/`. Note: as of PyTorch 2.x at
time of writing, `torch.compile`/Dynamo does not support Python 3.14
(PyTorch's own platform guard); the project's test suite gates the
torch.compile suites accordingly. This dependency will lift when PyTorch
ships 3.14 Dynamo support.

## Hardware reference (Apple M4 Max, used in benchmarks)

[13] **Apple M4 Max GPU specifications.** Apple, current product
documentation. The performance comparisons in `README.md`, the perf
baseline in `reports/perf_baseline.json`, and the roofline analysis in the
WS0/C6 harness all reference: 40 GPU cores; 128 ALUs/core; SIMD width 32;
max threads per threadgroup 1024; 32 KB threadgroup memory; 546 GB/s memory
bandwidth; 128 GB unified memory; no FP64, no FP8 (Apple GPU hardware
limitations); supported types FP32, FP16, BF16, INT8–INT32.

## Notes on URL stability

URLs for arXiv papers, conference proceedings, and major open-source
projects (Triton, MLX, Asahi, Mesa, `applegpu`) are stable. Apple
developer documentation links are stable for the *current* spec version;
older versions remain accessible via the Apple Developer archive. If a
link rots, the author/title/venue/year tuple is sufficient to relocate
the source.
