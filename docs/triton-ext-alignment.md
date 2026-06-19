# triton-ext Alignment Status

## Current Architecture

triton-msl uses two compilation paths:
1. **Python MSL path** (primary): TTGIR -> Python walker/lowerer -> MSL text -> xcrun metal -> metallib
2. **C++ LLVM IR path** (expanding): TTGIR -> C++ MLIR passes -> LLVM IR -> xcrun metal -> metallib

## triton-ext Plugin Model

triton-ext expects C++ shared libraries that:
1. Register MLIR passes via `PassRegistration<>`
2. Build via CMake against LLVM/MLIR/Triton headers
3. Load via `TRITON_PASS_PLUGIN_PATH` environment variable
4. Export pass pipeline functions callable from Python

## What We Have (aligned)

- `_triton_msl_cpp.cpython-*.so` -- pybind11 module with:
  - `register_metal_passes()` -- registers `convert-triton-msl-to-llvm` pass
  - `run_to_llvm(mlir_text)` -- full pipeline: parse -> SCF->CF -> Metal->LLVM -> export
- C++ MLIR patterns in `ElementwiseOpToLLVM.cpp`:
  - 16+ Triton op patterns (load, store, reduce, broadcast, reshape, etc.)
  - Custom type converter (tensor<NxT> -> T for per-thread model)
  - AIR intrinsic mapping (simd_sum, wg.barrier, etc.)
- CMake build linking against libtriton.so (shared MLIR symbols)

## What's Needed for triton-ext

1. **Pass plugin interface**: Export passes as loadable `.so` (not pybind11 module)
   - Add: `extern "C" void registerTritonMSLPasses()`
   - Build: separate `.so` target without pybind11 dependency

2. **Python hook integration**: Use `add_stages_inspection_hook` instead of
   monkey-patching `add_stages()`
   - Currently: `TRITON_MSL_USE_CPP=1` env var + overriding stages in add_stages
   - Target: register as triton-ext plugin that inserts passes automatically

3. **TableGen op definitions**: If we define custom Metal ops
   - Currently: no custom ops (we lower to standard LLVM dialect)
   - Future: Metal-specific ops for simdgroup MMA, threadgroup memory

## Recommendation

Ship as pip-installable Python backend (current approach) for production use.
Maintain C++ pass library as optional accelerator and future triton-ext foundation.
Port to triton-ext plugin when the ecosystem stabilizes and we need upstream merge.
