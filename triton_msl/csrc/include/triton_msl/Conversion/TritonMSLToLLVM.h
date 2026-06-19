#ifndef TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H
#define TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace llvm { class Module; }

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
class RewritePatternSet;
namespace triton_msl {

/// Create a pass that converts TritonGPU operations to LLVM IR
/// suitable for Metal GPU compilation.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMSLToLLVMPass();

/// Register all Metal conversion passes with MLIR's pass infrastructure.
void registerTritonMSLToLLVMPasses();

void populateSharedMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

void resetSharedMemoryCounter();

/// Coalesce addrspace(3) globals with non-overlapping live ranges via
/// greedy graph coloring. Runs on the LLVM Module after MLIR -> LLVM IR
/// translation.
void aliasSharedMemoryGlobals(llvm::Module &mod);

} // namespace triton_msl
} // namespace mlir

#endif // TRITON_MSL_CONVERSION_TRITONMSLTOLLVM_H
