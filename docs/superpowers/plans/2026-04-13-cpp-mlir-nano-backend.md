# C++ MLIR Nano Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a minimal C++ MLIR backend for triton-msl that compiles a vector_add kernel from TTGIR → LLVM IR → MSL → metallib, proving the architecture for eventual triton-ext submission.

**Architecture:** The nano backend adds a `make_llir` stage to the existing Python compilation pipeline. A C++ shared library implements MLIR passes that lower TritonGPU dialect ops to LLVM IR with Metal-specific intrinsics. The existing `make_msl` and `make_metallib` stages are replaced with `make_llir` → `make_msl_from_llvm` (LLVM IR → MSL via xcrun) → `make_metallib`. The Python codegen remains as a fallback for ops not yet handled by the C++ passes.

**Tech Stack:** C++17, MLIR/LLVM (from Triton's build), CMake, pybind11, Metal SDK (xcrun)

---

## Scope

This plan covers ONLY the nano backend scaffold:
- CMake build system that links against Triton's LLVM/MLIR
- One MLIR pass: `ConvertTritonMSLToLLVM` for basic ops (get_program_id, make_range, load, store, add)
- Python integration: load the shared library and register the pass
- One test: vector_add kernel compiles and produces correct results through the C++ path

This is NOT the full C++ backend. It's the foundation that proves:
1. We can build C++ passes against Triton's MLIR
2. We can lower TTGIR to LLVM IR for Metal
3. We can go from LLVM IR to metallib
4. The Python and C++ paths can coexist

## File Structure

```
triton_msl/
├── csrc/                          # NEW: C++ source directory
│   ├── CMakeLists.txt             # Build configuration
│   ├── include/
│   │   └── triton_msl/
│   │       └── Conversion/
│   │           └── TritonMSLToLLVM.h    # Pass declaration
│   └── lib/
│       └── Conversion/
│           ├── CMakeLists.txt
│           ├── TritonMSLToLLVM.cpp      # Main pass: TTGIR → LLVM IR
│           └── ElementwiseOpToLLVM.cpp    # Elementwise op patterns
├── backend/
│   └── compiler.py                # MODIFIED: add make_llir stage
└── tests/
    └── test_cpp_backend.py        # NEW: C++ backend tests
```

## Prerequisites

Before starting, verify:
1. Triton is built from source at `/Users/bledden/Documents/triton` with LLVM/MLIR headers available
2. `xcrun -sdk macosx metal --version` works (Metal compiler available)
3. CMake 3.20+ installed (`brew install cmake` if needed)

---

### Task 1: CMake Build Scaffold

**Files:**
- Create: `triton_msl/csrc/CMakeLists.txt`
- Create: `triton_msl/csrc/lib/Conversion/CMakeLists.txt`

This task sets up the build system that compiles our C++ passes against Triton's MLIR/LLVM libraries.

- [ ] **Step 1: Find Triton's LLVM/MLIR install location**

```bash
source .venv/bin/activate
python3 -c "
import triton
import os
triton_dir = os.path.dirname(triton.__file__)
print(f'Triton dir: {triton_dir}')
# Check for LLVM headers
llvm_dir = os.path.join(os.path.dirname(os.path.dirname(triton_dir)), 'build')
print(f'Build dir: {llvm_dir}')
for root, dirs, files in os.walk(llvm_dir):
    for f in files:
        if f == 'LLVMConfig.cmake':
            print(f'LLVM cmake: {os.path.join(root, f)}')
            break
    for f in files:
        if f == 'MLIRConfig.cmake':
            print(f'MLIR cmake: {os.path.join(root, f)}')
            break
"
```

Record the LLVM and MLIR cmake config paths for the CMakeLists.txt.

- [ ] **Step 2: Create top-level CMakeLists.txt**

Create `triton_msl/csrc/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(triton_msl_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Triton's LLVM/MLIR build
# Users must set TRITON_BUILD_DIR to the Triton build directory
if(NOT DEFINED TRITON_BUILD_DIR)
    # Default: sibling directory
    set(TRITON_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../triton/build" CACHE PATH "Triton build directory")
endif()

# Find LLVM and MLIR from Triton's build
find_package(LLVM REQUIRED CONFIG
    PATHS "${TRITON_BUILD_DIR}/_deps/llvm-build/lib/cmake/llvm"
    NO_DEFAULT_PATH)
find_package(MLIR REQUIRED CONFIG
    PATHS "${TRITON_BUILD_DIR}/_deps/llvm-build/lib/cmake/mlir"
    NO_DEFAULT_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Triton headers
set(TRITON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../triton" CACHE PATH "Triton source directory")
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${TRITON_SRC_DIR}/include
    ${TRITON_BUILD_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(lib/Conversion)
```

- [ ] **Step 3: Create Conversion CMakeLists.txt**

Create `triton_msl/csrc/lib/Conversion/CMakeLists.txt`:

```cmake
add_mlir_library(TritonMSLToLLVM
    TritonMSLToLLVM.cpp
    ElementwiseOpToLLVM.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)
```

- [ ] **Step 4: Verify build scaffold compiles (with empty source files)**

Create placeholder source files:
```bash
mkdir -p triton_msl/csrc/include/triton_msl/Conversion
mkdir -p triton_msl/csrc/lib/Conversion

# Placeholder header
cat > triton_msl/csrc/include/triton_msl/Conversion/TritonMSLToLLVM.h << 'EOF'
#ifndef TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H
#define TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H

namespace mlir {
class Pass;
namespace triton_msl {
// Will be defined in Task 2
} // namespace triton_msl
} // namespace mlir

#endif
EOF

# Placeholder source
cat > triton_msl/csrc/lib/Conversion/TritonMSLToLLVM.cpp << 'EOF'
#include "triton_msl/Conversion/TritonMSLToLLVM.h"
// Placeholder — will be implemented in Task 2
EOF

cat > triton_msl/csrc/lib/Conversion/ElementwiseOpToLLVM.cpp << 'EOF'
// Placeholder — will be implemented in Task 3
EOF
```

Try building:
```bash
cd triton_msl/csrc
mkdir -p build && cd build
TRITON_BUILD_DIR=/Users/bledden/Documents/triton/build/cmake.macosx-15.0-arm64-cpython-3.14 \
cmake .. -DTRITON_BUILD_DIR=$TRITON_BUILD_DIR -DTRITON_SRC_DIR=/Users/bledden/Documents/triton
make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Expected: Build succeeds (may have warnings but no errors).

- [ ] **Step 5: Commit**

```bash
git add triton_msl/csrc/
git commit -m "feat: CMake scaffold for C++ MLIR backend (Phase 5F)

Sets up build system for C++ MLIR passes that lower TritonGPU IR
to LLVM IR for Metal. Links against Triton's LLVM/MLIR libraries.
Placeholder source files compile successfully."
```

---

### Task 2: Minimal TTGIR → LLVM IR Pass

**Files:**
- Modify: `triton_msl/csrc/include/triton_msl/Conversion/TritonMSLToLLVM.h`
- Modify: `triton_msl/csrc/lib/Conversion/TritonMSLToLLVM.cpp`

Implement the `ConvertTritonMSLToLLVM` pass that handles the minimum ops for vector_add:
- `tt.get_program_id` → LLVM call to Metal's threadgroup position
- `tt.make_range` → LLVM integer sequence
- `tt.addptr` → LLVM GEP
- `tt.load` → LLVM load
- `tt.store` → LLVM store
- `arith.addf` → LLVM fadd (handled by existing MLIR lowering)

This task creates the MLIR pass structure. The actual op lowering patterns go in Task 3.

- [ ] **Step 1: Implement pass registration header**

Update `triton_msl/csrc/include/triton_msl/Conversion/TritonMSLToLLVM.h`:

```cpp
#ifndef TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H
#define TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton_msl {

/// Create a pass that converts TritonGPU operations to LLVM IR
/// suitable for Metal GPU compilation.
std::unique_ptr<Pass> createConvertTritonMSLToLLVMPass();

/// Register the pass with MLIR's pass infrastructure.
void registerTritonMSLToLLVMPass();

} // namespace triton_msl
} // namespace mlir

#endif
```

- [ ] **Step 2: Implement pass skeleton**

Update `triton_msl/csrc/lib/Conversion/TritonMSLToLLVM.cpp`:

```cpp
#include "triton_msl/Conversion/TritonMSLToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton_msl {

namespace {

class ConvertTritonMSLToLLVMPass
    : public PassWrapper<ConvertTritonMSLToLLVMPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "convert-triton-msl-to-llvm"; }
  StringRef getDescription() const override {
    return "Convert TritonGPU operations to LLVM IR for Apple Metal GPUs";
  }

  void runOnOperation() override {
    auto module = getOperation();
    // TODO: Task 3 adds conversion patterns here
    // For now, just mark the pass as successful
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTritonMSLToLLVMPass() {
  return std::make_unique<ConvertTritonMSLToLLVMPass>();
}

void registerTritonMSLToLLVMPass() {
  PassRegistration<ConvertTritonMSLToLLVMPass>();
}

} // namespace triton_msl
} // namespace mlir
```

- [ ] **Step 3: Build and verify**

```bash
cd triton_msl/csrc/build
cmake .. && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Expected: Compiles successfully, produces `libTritonMSLToLLVM.a` or `.dylib`.

- [ ] **Step 4: Commit**

```bash
git add triton_msl/csrc/
git commit -m "feat: ConvertTritonMSLToLLVM pass skeleton

MLIR pass that will lower TritonGPU ops to LLVM IR for Metal.
Currently a no-op skeleton — conversion patterns added next."
```

---

### Task 3: Elementwise Op Lowering Patterns

**Files:**
- Modify: `triton_msl/csrc/lib/Conversion/ElementwiseOpToLLVM.cpp`
- Modify: `triton_msl/csrc/lib/Conversion/TritonMSLToLLVM.cpp`

Implement MLIR conversion patterns for the ops needed by vector_add:
- `tt.get_program_id` → Metal threadgroup position builtin
- `tt.make_range` → computed integer sequence based on thread ID
- `tt.splat` → scalar broadcast (no-op in per-thread model)
- `arith.cmpi` (for mask) → LLVM icmp
- `tt.addptr` → pointer arithmetic
- `tt.load` / `tt.store` → LLVM load/store with mask

These patterns follow the same structure as NVIDIA's `TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp` but target Metal's execution model.

- [ ] **Step 1: Implement get_program_id and make_range patterns**

In `ElementwiseOpToLLVM.cpp`, implement:

```cpp
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton_msl {

// Pattern: tt.get_program_id → Metal threadgroup position
// In Metal, this maps to a kernel argument [[threadgroup_position_in_grid]]
// At LLVM IR level, we'll use a function argument placeholder that the
// MSL-from-LLVM step resolves.
class GetProgramIdOpConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::GetProgramIdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // For the nano backend, emit a call to an external function
    // that the MSL generation step will resolve to the Metal builtin.
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();

    // Create an external function declaration for the Metal builtin
    auto axis = op.getAxisAsInt();
    std::string funcName = "__metal_get_program_id_" + std::to_string(axis);

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {});
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    }

    auto call = rewriter.create<LLVM::CallOp>(loc, func, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// Pattern: tt.make_range → lid + offset computation
// make_range {start=0, end=N} produces [0, 1, ..., N-1]
// In per-thread model: each thread's value is its thread index (lid)
class MakeRangeOpConversion
    : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::MakeRangeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();

    // Get thread_position_in_threadgroup (lid)
    std::string funcName = "__metal_get_local_id";
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {});
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    }

    auto lid = rewriter.create<LLVM::CallOp>(loc, func, ValueRange{});
    auto start = op.getStart();
    if (start != 0) {
      auto startConst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(start));
      auto added = rewriter.create<LLVM::AddOp>(loc, i32Ty,
          lid.getResult(), startConst);
      rewriter.replaceOp(op, added.getResult());
    } else {
      rewriter.replaceOp(op, lid.getResult());
    }
    return success();
  }
};

void populateElementwiseOpToLLVMPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<GetProgramIdOpConversion, MakeRangeOpConversion>(
      typeConverter, patterns.getContext());
}

} // namespace triton_msl
} // namespace mlir
```

- [ ] **Step 2: Wire patterns into the pass**

Update `TritonMSLToLLVM.cpp` to register and apply patterns:

```cpp
// Add to runOnOperation():
void runOnOperation() override {
    auto module = getOperation();
    auto context = &getContext();

    // Set up type converter
    LLVMTypeConverter typeConverter(context);

    // Collect patterns
    RewritePatternSet patterns(context);
    populateElementwiseOpToLLVMPatterns(patterns, typeConverter);

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<triton::TritonDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
}
```

Add the extern declaration at the top of `TritonMSLToLLVM.cpp`:
```cpp
namespace mlir {
namespace triton_msl {
void populateElementwiseOpToLLVMPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);
} // namespace triton_msl
} // namespace mlir
```

- [ ] **Step 3: Build and verify compilation**

```bash
cd triton_msl/csrc/build
cmake .. && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Expected: Compiles. (Runtime testing comes in Task 5.)

- [ ] **Step 4: Commit**

```bash
git add triton_msl/csrc/
git commit -m "feat: elementwise TTGIR → LLVM IR patterns for Metal

Implements get_program_id, make_range conversion patterns that lower
Triton ops to LLVM IR with Metal-specific external function calls.
These placeholders are resolved during MSL generation."
```

---

### Task 4: Python Integration (Load Shared Library + Register Pass)

**Files:**
- Modify: `triton_msl/backend/compiler.py`
- Create: `triton_msl/csrc/python_bindings.cpp`

Wire the C++ shared library into the Python compilation pipeline. Add a `make_llir` stage that runs the MLIR pass, then converts LLVM IR to MSL.

- [ ] **Step 1: Add pybind11 bindings**

Create `triton_msl/csrc/python_bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include "triton_msl/Conversion/TritonMSLToLLVM.h"

namespace py = pybind11;

PYBIND11_MODULE(_triton_msl_cpp, m) {
    m.doc() = "C++ MLIR passes for triton-msl";

    m.def("register_metal_passes", []() {
        mlir::triton_msl::registerTritonMSLToLLVMPass();
    }, "Register Metal MLIR passes");
}
```

Update `triton_msl/csrc/CMakeLists.txt` to build the pybind11 module:

```cmake
# Add after existing content:
find_package(pybind11 REQUIRED)

pybind11_add_module(_triton_msl_cpp
    python_bindings.cpp
    lib/Conversion/TritonMSLToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
)
target_include_directories(_triton_msl_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${TRITON_SRC_DIR}/include
    ${TRITON_BUILD_DIR}/include
)
target_link_libraries(_triton_msl_cpp PRIVATE
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)
```

- [ ] **Step 2: Build pybind11 module**

```bash
cd triton_msl/csrc/build
cmake .. && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
# Copy the .so to the package
cp _triton_msl_cpp*.so ../../triton_msl/
```

- [ ] **Step 3: Add make_llir to compiler.py**

In `triton_msl/backend/compiler.py`, add a `make_llir` stage (optional, gated on whether the C++ module is available):

```python
# In add_stages, optionally insert make_llir before make_msl:
def add_stages(self, stages, options, language=None):
    from triton.compiler.compiler import Language

    if language == Language.GLUON:
        stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
    else:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)

    # Optional C++ LLVM IR lowering (when built)
    if self._has_cpp_passes():
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)

    stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
    stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

@staticmethod
def _has_cpp_passes():
    try:
        import triton_msl._triton_msl_cpp
        return True
    except ImportError:
        return False

@staticmethod
def make_llir(mod, metadata, options):
    """Lower TTGIR to LLVM IR using C++ MLIR passes."""
    from triton._C.libtriton import ir, passes
    import triton_msl._triton_msl_cpp as cpp

    cpp.register_metal_passes()

    pm = ir.pass_manager(mod.context)
    # Add standard lowering passes
    passes.convert.add_scf_to_cf(pm)
    # Add our Metal-specific pass
    pm.add_pass("convert-triton-msl-to-llvm")
    pm.run(mod, "make_llir")
    return mod
```

- [ ] **Step 4: Commit**

```bash
git add triton_msl/csrc/ triton_msl/backend/compiler.py
git commit -m "feat: Python-C++ integration for MLIR Metal passes

pybind11 module exposes register_metal_passes(). compiler.py
optionally inserts make_llir stage when C++ module is available.
Falls back to Python-only path when not built."
```

---

### Task 5: End-to-End Test (Vector Add Through C++ Path)

**Files:**
- Create: `tests/test_cpp_backend.py`

- [ ] **Step 1: Write test**

```python
"""Test the C++ MLIR backend path for basic kernels."""
import pytest
import torch

try:
    import triton_msl._triton_msl_cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

requires_cpp = pytest.mark.skipif(not _HAS_CPP, reason="C++ backend not built")

@requires_cpp
def test_vector_add_cpp_path():
    """Vector add compiles through C++ MLIR passes."""
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

    assert (out - (x + y)).abs().max().item() < 1e-5
```

- [ ] **Step 2: Run test**

```bash
source .venv/bin/activate
python -m pytest tests/test_cpp_backend.py -v --timeout=60
```

Expected: PASS if C++ module is built, SKIP otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cpp_backend.py
git commit -m "test: vector_add through C++ MLIR backend path"
```

---

## What This Plan Does NOT Cover

- Load/store with complex addressing (strided, 2D)
- Reductions (tt.reduce)
- Matrix multiply (tt.dot, simdgroup MMA)
- LLVM IR → metallib without xcrun (the metal-ir-pipeline from PR #48)
- torch.compile integration via C++ path
- FlashAttention through C++ path

These are future tasks that build on this foundation. Each will be its own plan.

## Success Criteria

- `cmake .. && make` builds the C++ shared library without errors
- `import triton_msl._triton_msl_cpp` works in Python
- `test_vector_add_cpp_path` passes (correct numerical results)
- Existing 464 Python-path tests continue to pass (C++ path is additive, not replacement)
