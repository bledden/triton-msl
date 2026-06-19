// Bridge between the pybind11 module (RTTI-enabled) and MLIR code (no-RTTI).
// This file is compiled with -fno-rtti to match LLVM/MLIR conventions.

#include "triton_msl/Conversion/TritonMSLToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>
#include <sstream>

extern "C" void triton_msl_register_passes() {
    mlir::triton_msl::registerTritonMSLToLLVMPasses();
}

// triton-ext compatible plugin entry point.
// This allows loading our passes via TRITON_PASS_PLUGIN_PATH without
// the pybind11 module overhead.
extern "C" void tritonMetalRegisterPasses(void) {
    mlir::triton_msl::registerTritonMSLToLLVMPasses();
}

// Descriptor for an implicit Metal kernel argument (thread position, grid
// size, etc.) that gets its value from a Metal AIR intrinsic.
struct AIRImplicitArg {
    std::string airMetadata;  // e.g. "air.threadgroup_position_in_grid"
    std::string argName;      // e.g. "pid"
};

// ---------------------------------------------------------------------------
// Add AIR metadata to the LLVM IR module so Metal's compiler can consume it.
//
// This metadata tells the Metal compiler:
// - Which function is a kernel (!air.kernel)
// - How to bind arguments to Metal buffer indices (!air.buffer, etc.)
// - The AIR version and language version
// - Thread position intrinsics (threadgroup_position_in_grid, etc.)
// ---------------------------------------------------------------------------
// numExplicitArgs: number of args from the Triton kernel (before implicit args).
// If 0, all args are treated as explicit (no thread position args).
// implicitArgs: ordered list of implicit arg descriptors (pid, lid, etc.)
static void addAIRMetadata(llvm::Module &mod, llvm::Function &kernelFn,
                           unsigned numExplicitArgs,
                           const llvm::SmallVectorImpl<AIRImplicitArg> &implicitArgs) {
    auto &ctx = mod.getContext();
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);

    // Module flags
    mod.addModuleFlag(llvm::Module::Warning, "wchar_size",
                      llvm::ConstantInt::get(i32Ty, 4));
    mod.addModuleFlag(llvm::Module::Max, "frame-pointer",
                      llvm::ConstantInt::get(i32Ty, 2));
    mod.addModuleFlag(llvm::Module::Max, "air.max_device_buffers",
                      llvm::ConstantInt::get(i32Ty, 31));
    mod.addModuleFlag(llvm::Module::Max, "air.max_constant_buffers",
                      llvm::ConstantInt::get(i32Ty, 31));
    mod.addModuleFlag(llvm::Module::Max, "air.max_threadgroup_buffers",
                      llvm::ConstantInt::get(i32Ty, 31));
    mod.addModuleFlag(llvm::Module::Max, "air.max_textures",
                      llvm::ConstantInt::get(i32Ty, 128));
    mod.addModuleFlag(llvm::Module::Max, "air.max_read_write_textures",
                      llvm::ConstantInt::get(i32Ty, 8));
    mod.addModuleFlag(llvm::Module::Max, "air.max_samplers",
                      llvm::ConstantInt::get(i32Ty, 16));

    // Analyze kernel function signature to build argument metadata.
    // Our lowered kernel has the form:
    //   void kernel(ptr %arg0, ptr %arg1, ..., i32 %argN, i32 %pid, i32 %lid)
    // The pointer args map to device buffers, i32 args to scalars/constant
    // buffers, and pid/lid are implicit thread position args.
    unsigned totalArgs = kernelFn.arg_size();
    // If numExplicitArgs not specified, assume all args are explicit
    unsigned explicitArgs = numExplicitArgs > 0 ? numExplicitArgs : totalArgs;

    llvm::SmallVector<llvm::Metadata *, 8> argMDs;
    unsigned bufferIndex = 0;

    for (unsigned i = 0; i < explicitArgs; ++i) {
        llvm::Argument &arg = *kernelFn.getArg(i);
        llvm::Type *argTy = arg.getType();

        llvm::SmallVector<llvm::Metadata *, 16> fields;
        // Arg position
        fields.push_back(llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(i32Ty, i)));

        if (argTy->isPointerTy()) {
            unsigned addrSpace = argTy->getPointerAddressSpace();

            if (addrSpace == 2) {
                // Constant buffer argument (addrspace 2) — scalar passed as buffer.
                fields.push_back(llvm::MDString::get(ctx, "air.buffer"));
                fields.push_back(llvm::MDString::get(ctx, "air.buffer_size"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 4))); // i32 = 4 bytes
                fields.push_back(llvm::MDString::get(ctx, "air.location_index"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, bufferIndex)));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 1)));
                fields.push_back(llvm::MDString::get(ctx, "air.read"));
                fields.push_back(llvm::MDString::get(ctx, "air.address_space"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 2))); // constant address space

                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 4)));
                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 4)));
                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
                fields.push_back(llvm::MDString::get(ctx, "int"));
                fields.push_back(llvm::MDString::get(ctx, "air.arg_name"));
                fields.push_back(llvm::MDString::get(ctx, ("arg" + std::to_string(i)).c_str()));
            } else {
                // Device buffer argument (addrspace 1).
                // Mark as read_write since we can't reliably determine usage
                // through addrspacecast/GEP chains in the nano backend.
                fields.push_back(llvm::MDString::get(ctx, "air.buffer"));
                fields.push_back(llvm::MDString::get(ctx, "air.location_index"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, bufferIndex)));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 1))); // array length
                fields.push_back(llvm::MDString::get(ctx, "air.read_write"));

                fields.push_back(llvm::MDString::get(ctx, "air.address_space"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 1))); // device address space

                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 4))); // float = 4 bytes
                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
                fields.push_back(llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32Ty, 4)));
                fields.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
                fields.push_back(llvm::MDString::get(ctx, "float"));
                fields.push_back(llvm::MDString::get(ctx, "air.arg_name"));
                fields.push_back(llvm::MDString::get(ctx, ("arg" + std::to_string(i)).c_str()));
            }

            bufferIndex++;
        } else if (argTy->isIntegerTy()) {
            // Scalar integer argument — passed as constant buffer
            unsigned bitWidth = argTy->getIntegerBitWidth();
            unsigned byteWidth = (bitWidth + 7) / 8;

            fields.push_back(llvm::MDString::get(ctx, "air.buffer"));
            fields.push_back(llvm::MDString::get(ctx, "air.buffer_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));
            fields.push_back(llvm::MDString::get(ctx, "air.location_index"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, bufferIndex)));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, 1)));
            fields.push_back(llvm::MDString::get(ctx, "air.read"));
            fields.push_back(llvm::MDString::get(ctx, "air.address_space"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, 2))); // constant address space

            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));
            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));

            std::string typeName = (bitWidth == 32) ? "int" : "long";
            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
            fields.push_back(llvm::MDString::get(ctx, typeName));
            fields.push_back(llvm::MDString::get(ctx, "air.arg_name"));
            fields.push_back(llvm::MDString::get(ctx, ("arg" + std::to_string(i)).c_str()));

            bufferIndex++;
        } else if (argTy->isFloatingPointTy()) {
            // Scalar float argument — passed as constant buffer
            unsigned byteWidth = argTy->isFloatTy() ? 4
                               : argTy->isHalfTy()  ? 2
                               : argTy->isDoubleTy() ? 8 : 4;
            std::string typeName = argTy->isFloatTy() ? "float"
                                 : argTy->isHalfTy()  ? "half"
                                 : "float";

            fields.push_back(llvm::MDString::get(ctx, "air.buffer"));
            fields.push_back(llvm::MDString::get(ctx, "air.buffer_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));
            fields.push_back(llvm::MDString::get(ctx, "air.location_index"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, bufferIndex)));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, 1)));
            fields.push_back(llvm::MDString::get(ctx, "air.read"));
            fields.push_back(llvm::MDString::get(ctx, "air.address_space"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, 2))); // constant address space

            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));
            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
            fields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32Ty, byteWidth)));
            fields.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
            fields.push_back(llvm::MDString::get(ctx, typeName));
            fields.push_back(llvm::MDString::get(ctx, "air.arg_name"));
            fields.push_back(llvm::MDString::get(ctx, ("arg" + std::to_string(i)).c_str()));

            bufferIndex++;
        }

        argMDs.push_back(llvm::MDNode::get(ctx, fields));
    }

    // Add implicit thread position / grid size arguments.
    // These are already in the function signature starting at index
    // explicitArgs, in the order given by the implicitArgs vector.
    for (unsigned i = 0; i < implicitArgs.size(); ++i) {
        unsigned argIdx = explicitArgs + i;
        const auto &ia = implicitArgs[i];
        llvm::Argument &arg = *kernelFn.getArg(argIdx);
        bool isVec = arg.getType()->isVectorTy();

        llvm::SmallVector<llvm::Metadata *, 6> fields;
        fields.push_back(llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(i32Ty, argIdx)));
        fields.push_back(llvm::MDString::get(ctx, ia.airMetadata));
        fields.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
        fields.push_back(llvm::MDString::get(ctx, isVec ? "uint3" : "uint"));
        fields.push_back(llvm::MDString::get(ctx, "air.arg_name"));
        fields.push_back(llvm::MDString::get(ctx, ia.argName));
        argMDs.push_back(llvm::MDNode::get(ctx, fields));
    }

    // Scan for addrspace(3) globals (threadgroup buffers) and add metadata.
    // Metal's compiler uses this to validate shared memory allocations.
    //
    // NOTE: Empirically, MSL-compiled metallibs do NOT emit !air.threadgroup_buffers
    // named metadata — the Metal compiler handles addrspace(3) globals implicitly
    // when they are referenced by the kernel function. We only emit this metadata
    // if addrspace(3) globals are present; if Metal rejects the format, the empty
    // case (no globals) won't trigger any emission at all.
    llvm::SmallVector<llvm::Metadata *, 4> tgBufferMDs;
    unsigned tgLocIdx = 0;
    for (auto &global : mod.globals()) {
      if (global.getAddressSpace() != 3) continue;

      // Skip internal globals not related to TTG shared memory
      if (!global.getName().starts_with("__tg_shared_") &&
          !global.getName().starts_with("__reduce_shared_")) continue;

      // Compute size in bytes
      auto *valTy = global.getValueType();
      const auto &dl = mod.getDataLayout();
      uint64_t bytes = dl.getTypeAllocSize(valTy);

      llvm::SmallVector<llvm::Metadata *, 8> fields;
      fields.push_back(llvm::ConstantAsMetadata::get(&global));
      fields.push_back(llvm::MDString::get(ctx, "air.threadgroup_buffer"));
      fields.push_back(llvm::MDString::get(ctx, "air.location_index"));
      fields.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, tgLocIdx)));
      fields.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, 0)));
      fields.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
      fields.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, bytes)));
      fields.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
      fields.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, 16)));

      tgBufferMDs.push_back(llvm::MDNode::get(ctx, fields));
      tgLocIdx++;
    }

    if (!tgBufferMDs.empty()) {
      auto *tgBuffersMD = mod.getOrInsertNamedMetadata("air.threadgroup_buffers");
      for (auto *md : tgBufferMDs)
        tgBuffersMD->addOperand(llvm::cast<llvm::MDNode>(md));
    }

    // The kernel metadata references the function directly.
    auto *fnPtrConst = llvm::ConstantExpr::getBitCast(
        &kernelFn, llvm::PointerType::getUnqual(ctx));

    llvm::Metadata *kernelMD[] = {
        llvm::ConstantAsMetadata::get(fnPtrConst),
        llvm::MDNode::get(ctx, {}), // no extra kernel attributes
        llvm::MDNode::get(ctx, argMDs)
    };

    auto *airKernelNode = llvm::MDNode::get(ctx, kernelMD);
    auto *namedMD = mod.getOrInsertNamedMetadata("air.kernel");
    namedMD->addOperand(airKernelNode);

    // air.compile_options
    auto *compileOpts = mod.getOrInsertNamedMetadata("air.compile_options");
    compileOpts->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "air.compile.denorms_disable")
    }));
    compileOpts->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "air.compile.fast_math_enable")
    }));
    compileOpts->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "air.compile.framebuffer_fetch_enable")
    }));

    // llvm.ident
    auto *identMD = mod.getOrInsertNamedMetadata("llvm.ident");
    identMD->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "triton-msl")
    }));

    // air.version
    auto *verMD = mod.getOrInsertNamedMetadata("air.version");
    verMD->addOperand(llvm::MDNode::get(ctx, {
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 2)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 7)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 0))
    }));

    // air.language_version
    auto *langMD = mod.getOrInsertNamedMetadata("air.language_version");
    langMD->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "Metal"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 3)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 2)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, 0))
    }));

    // air.source_file_name
    auto *srcMD = mod.getOrInsertNamedMetadata("air.source_file_name");
    srcMD->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "triton_msl_generated")
    }));
}

// ---------------------------------------------------------------------------
// Rewrite the LLVM IR to use AIR-compatible address spaces and calling
// conventions. Metal AIR uses:
//   addrspace(0) = generic
//   addrspace(1) = device (read/write buffers)
//   addrspace(2) = constant (read-only small buffers)
//   addrspace(3) = threadgroup (shared memory)
// Our MLIR lowering produces addrspace(0) pointers. We need to:
// 1. Change device buffer pointer args to addrspace(1)
// 2. Change scalar args from plain i32 to i32 addrspace(2)* (constant buffer)
// ---------------------------------------------------------------------------

extern "C" const char* triton_msl_run_to_llvm(const char* mlir_text,
                                                 int* out_success) {
    *out_success = 0;

    // Create a fresh context with all needed dialects.
    mlir::MLIRContext mlirCtx;
    mlirCtx.loadDialect<mlir::triton::TritonDialect>();
    mlirCtx.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
    mlirCtx.loadDialect<mlir::LLVM::LLVMDialect>();
    mlirCtx.loadDialect<mlir::arith::ArithDialect>();
    mlirCtx.loadDialect<mlir::scf::SCFDialect>();
    mlirCtx.loadDialect<mlir::cf::ControlFlowDialect>();
    mlirCtx.loadDialect<mlir::math::MathDialect>();
    mlirCtx.loadDialect<mlir::func::FuncDialect>();

    // Register LLVM IR translation interfaces
    mlir::registerBuiltinDialectTranslation(mlirCtx);
    mlir::registerLLVMDialectTranslation(mlirCtx);

    // Parse the input MLIR text.
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &mlirCtx);
    if (!module) {
        static std::string errMsg;
        errMsg = "Failed to parse MLIR module";
        return errMsg.c_str();
    }

    // Build the pass pipeline:
    // 1. SCF -> CF (lower structured control flow)
    // 2. Our Metal-to-LLVM conversion (handles Triton ops)
    // 3. Arith -> LLVM (lower arithmetic ops)
    // 4. Index -> LLVM (lower index ops)
    // 5. ControlFlow -> LLVM (lower cf ops to LLVM branches)
    // 6. Canonicalize
    mlir::PassManager pm(&mlirCtx);

    // SCF -> CF first (lower structured control flow to branches)
    pm.addPass(mlir::createSCFToControlFlowPass());
    // Our Metal-to-LLVM pass includes arith/cf/index -> LLVM patterns
    // using a shared type converter so tensor<NxT> -> T works everywhere.
    pm.addPass(mlir::triton_msl::createConvertTritonMSLToLLVMPass());
    // Clean up any remaining unrealized casts
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Enable IR printing for debug
    if (std::getenv("TRITON_MSL_DEBUG_PASSES")) {
        mlirCtx.disableMultithreading();
        pm.enableIRPrinting();
    }

    if (mlir::failed(pm.run(*module))) {
        // Dump the failed module for debugging
        static std::string errMsg;
        std::string modStr;
        llvm::raw_string_ostream modOs(modStr);
        module->print(modOs);
        errMsg = "MLIR pass pipeline failed (TTGIR -> LLVM dialect).\n"
                 "Module after failure:\n" + modStr;
        return errMsg.c_str();
    }

    // Debug: dump MLIR after passes
    if (std::getenv("TRITON_MSL_DEBUG_PASSES")) {
        std::string dbgStr;
        llvm::raw_string_ostream dbgOs(dbgStr);
        module->print(dbgOs);
        llvm::errs() << "=== MLIR after pass pipeline ===\n" << dbgStr << "\n";
    }

    // Translate MLIR LLVM dialect -> actual LLVM IR
    llvm::LLVMContext llvmCtx;
    auto llvmMod = mlir::translateModuleToLLVMIR(*module, llvmCtx,
                                                  "triton_msl_kernel");
    if (!llvmMod) {
        static std::string errMsg;
        errMsg = "Failed to translate MLIR LLVM dialect to LLVM IR";
        return errMsg.c_str();
    }

    // Set AIR target triple and data layout
    llvmMod->setTargetTriple(llvm::Triple("air64_v27-apple-macosx15.0.0"));
    llvmMod->setDataLayout(
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
        "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64"
        "-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256"
        "-v512:512:512-v1024:1024:1024-n8:16:32");

    // Find the kernel function
    llvm::Function *kernelFn = nullptr;
    for (auto &fn : *llvmMod) {
        if (!fn.isDeclaration()) {
            kernelFn = &fn;
            break;
        }
    }

    if (!kernelFn) {
        static std::string errMsg;
        errMsg = "No kernel function found in LLVM IR";
        return errMsg.c_str();
    }

    // Coalesce addrspace(3) globals with non-overlapping live ranges.
    // This reduces total shared-memory footprint for kernels with multiple
    // phases that each need scratch space but don't coexist.
    mlir::triton_msl::aliasSharedMemoryGlobals(*llvmMod);

    // ---------------------------------------------------------------
    // Replace __metal_* intrinsic calls with extra kernel function
    // parameters. In Metal AIR, thread position / grid size intrinsics
    // are implicit function arguments, not calls.
    //
    // Supported intrinsics:
    //   __metal_get_program_id_{0,1,2} → threadgroup_position_in_grid.{x,y,z}
    //   __metal_get_local_id           → thread_position_in_threadgroup.x
    //   __metal_get_num_programs_{0,1,2} → threadgroups_per_grid.{x,y,z}
    // ---------------------------------------------------------------
    llvm::SmallVector<AIRImplicitArg, 6> airImplicitArgs;
    {
        auto *i32Ty = llvm::Type::getInt32Ty(llvmCtx);

        // Collect all __metal_* calls and determine which intrinsics are used.
        struct CallInfo {
            llvm::CallInst *call;
            std::string fnName;
        };
        llvm::SmallVector<CallInfo, 8> calls;
        // Track which intrinsics are present (by function name).
        std::set<std::string> usedIntrinsics;

        for (auto &bb : *kernelFn) {
            for (auto &inst : bb) {
                if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                    if (auto *callee = callInst->getCalledFunction()) {
                        auto name = callee->getName();
                        if (name.starts_with("__metal_get_")) {
                            calls.push_back({callInst, name.str()});
                            usedIntrinsics.insert(name.str());
                        }
                    }
                }
            }
        }

        if (!calls.empty()) {
            // Create new function type with AIR address spaces:
            // - ptr args → ptr addrspace(1) (device buffer)
            // - i32 args → ptr addrspace(2) (constant buffer, load to get value)
            // Then append implicit thread position args (i32 each).
            auto *devicePtrTy = llvm::PointerType::get(llvmCtx, 1); // addrspace(1)
            auto *constPtrTy = llvm::PointerType::get(llvmCtx, 2);  // addrspace(2)

            llvm::SmallVector<llvm::Type *, 8> newArgTypes;
            llvm::SmallVector<bool, 8> isScalarArg;  // track which args need load
            for (auto &arg : kernelFn->args()) {
                if (arg.getType()->isPointerTy()) {
                    newArgTypes.push_back(devicePtrTy);
                    isScalarArg.push_back(false);
                } else if (arg.getType()->isIntegerTy() ||
                           arg.getType()->isFloatingPointTy()) {
                    // Scalar args (int or float) are passed as constant
                    // buffer pointers in Metal AIR.
                    newArgTypes.push_back(constPtrTy);
                    isScalarArg.push_back(true);
                } else {
                    newArgTypes.push_back(arg.getType());
                    isScalarArg.push_back(false);
                }
            }

            // Build ordered list of implicit args. We always add pid and lid
            // as the canonical Metal kernel interface, plus any additional
            // intrinsics (num_programs) as needed.
            struct ImplicitArgEntry {
                std::string intrinsicName;  // __metal_get_* function name
                std::string argName;        // LLVM arg name
                std::string airMetadata;    // AIR metadata key
                bool isVector;              // true if this is a <3 x i32> arg
            };
            llvm::SmallVector<ImplicitArgEntry, 6> implicitArgs;

            // Determine dimensionality of program_id usage.
            bool needsPidY = usedIntrinsics.count("__metal_get_program_id_1") > 0;
            bool needsPidZ = usedIntrinsics.count("__metal_get_program_id_2") > 0;
            bool needsMultiDimPid = needsPidY || needsPidZ;

            // Thread position in grid: for 1D use scalar, for 2D/3D use <3 x i32>.
            // The __metal_get_program_id_{0,1,2} calls map to extractelement
            // from the vector arg.
            implicitArgs.push_back({
                "__metal_get_program_id_0", "pid",
                "air.threadgroup_position_in_grid",
                needsMultiDimPid // vector if multi-dim
            });

            implicitArgs.push_back({
                "__metal_get_local_id", "lid",
                "air.thread_position_in_threadgroup",
                false
            });

            // Add threadgroups_per_grid if get_num_programs is used
            if (usedIntrinsics.count("__metal_get_num_programs_0")) {
                implicitArgs.push_back({
                    "__metal_get_num_programs_0", "grid_size",
                    "air.threadgroups_per_grid",
                    false
                });
            }

            // Add SIMD group intrinsics for reductions
            if (usedIntrinsics.count("__metal_get_sgitg")) {
                implicitArgs.push_back({
                    "__metal_get_sgitg", "sgitg",
                    "air.simdgroup_index_in_threadgroup",
                    false
                });
            }
            if (usedIntrinsics.count("__metal_get_tiisg")) {
                implicitArgs.push_back({
                    "__metal_get_tiisg", "tiisg",
                    "air.thread_index_in_simdgroup",
                    false
                });
            }

            // Append implicit arg types
            auto *vec3i32Ty = llvm::FixedVectorType::get(i32Ty, 3);
            for (auto &ia : implicitArgs) {
                llvm::Type *argTy = ia.isVector
                    ? static_cast<llvm::Type *>(vec3i32Ty)
                    : static_cast<llvm::Type *>(i32Ty);
                newArgTypes.push_back(argTy);
            }

            auto *newFnTy = llvm::FunctionType::get(
                kernelFn->getReturnType(), newArgTypes, false);

            // Create new function
            auto *newFn = llvm::Function::Create(
                newFnTy, kernelFn->getLinkage(),
                kernelFn->getName() + "_impl", llvmMod.get());

            // Copy attributes
            newFn->setCallingConv(kernelFn->getCallingConv());

            // Set up the value map for cloning.
            llvm::ValueToValueMapTy vmap;
            auto newArgIt = newFn->arg_begin();
            unsigned argIdx = 0;
            for (auto &oldArg : kernelFn->args()) {
                newArgIt->setName(oldArg.getName());
                vmap[&oldArg] = &*newArgIt;
                ++newArgIt;
                ++argIdx;
            }

            // Name and record the implicit args.
            // For multi-dim program_id, the vector arg is stored and
            // extractelement is used for each axis.
            std::map<std::string, llvm::Argument *> implicitArgMap;
            llvm::Argument *pidVectorArg = nullptr;
            for (auto &ia : implicitArgs) {
                llvm::Argument *arg = &*newArgIt;
                arg->setName(ia.argName);
                if (ia.isVector && ia.airMetadata == "air.threadgroup_position_in_grid") {
                    pidVectorArg = arg;
                    // Map axis 0 to extractelement (done later in IR)
                    implicitArgMap[ia.intrinsicName] = arg; // placeholder
                } else {
                    implicitArgMap[ia.intrinsicName] = arg;
                }
                ++newArgIt;
            }

            // Clone the function body
            llvm::SmallVector<llvm::ReturnInst *, 4> returns;
            llvm::CloneFunctionInto(newFn, kernelFn, vmap,
                                    llvm::CloneFunctionChangeType::LocalChangesOnly,
                                    returns);

            // Fix up the cloned function:
            // 1. For ptr args: addrspacecast from addrspace(1) to addrspace(0)
            // 2. For scalar args: load value from addrspace(2) pointer
            // 3. Replace __metal_* calls with implicit args
            auto &entryBB = newFn->getEntryBlock();
            llvm::IRBuilder<> builder(&*entryBB.getFirstInsertionPt());

            newArgIt = newFn->arg_begin();
            argIdx = 0;
            for (auto &oldArg : kernelFn->args()) {
                llvm::Argument *newArg = &*newArgIt;

                if (isScalarArg[argIdx]) {
                    auto *loadedVal = builder.CreateLoad(
                        oldArg.getType(), newArg, newArg->getName() + ".val");
                    for (auto it = newArg->use_begin(); it != newArg->use_end(); ) {
                        auto &use = *it;
                        ++it;
                        if (use.getUser() != loadedVal)
                            use.set(loadedVal);
                    }
                } else if (newArg->getType() != oldArg.getType() &&
                           newArg->getType()->isPointerTy()) {
                    auto *genericPtr = llvm::PointerType::get(llvmCtx, 0);
                    auto *castVal = builder.CreateAddrSpaceCast(
                        newArg, genericPtr, newArg->getName() + ".generic");
                    for (auto it = newArg->use_begin(); it != newArg->use_end(); ) {
                        auto &use = *it;
                        ++it;
                        if (use.getUser() != castVal)
                            use.set(castVal);
                    }
                }

                ++newArgIt;
                ++argIdx;
            }

            // Replace __metal_* calls with corresponding implicit args.
            // For multi-dim program_id, extract the component from the
            // <3 x i32> vector arg.
            for (auto &bb : *newFn) {
                for (auto it = bb.begin(); it != bb.end(); ) {
                    auto *inst = &*it;
                    ++it;
                    if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(inst)) {
                        if (auto *callee = callInst->getCalledFunction()) {
                            auto name = callee->getName().str();

                            // Handle multi-dim program_id via extractelement
                            if (needsMultiDimPid && pidVectorArg &&
                                llvm::StringRef(name).starts_with("__metal_get_program_id_")) {
                                unsigned axis = name.back() - '0';
                                llvm::IRBuilder<> b(callInst);
                                auto *idx = llvm::ConstantInt::get(i32Ty, axis);
                                auto *elem = b.CreateExtractElement(
                                    pidVectorArg, idx, "pid_" + std::to_string(axis));
                                callInst->replaceAllUsesWith(elem);
                                callInst->eraseFromParent();
                                continue;
                            }

                            auto mapIt = implicitArgMap.find(name);
                            if (mapIt != implicitArgMap.end()) {
                                callInst->replaceAllUsesWith(mapIt->second);
                                callInst->eraseFromParent();
                            }
                        }
                    }
                }
            }

            // Replace old function with new one
            newFn->takeName(kernelFn);
            kernelFn->eraseFromParent();
            kernelFn = newFn;

            // Remove declarations of __metal_* functions
            for (auto it = llvmMod->begin(); it != llvmMod->end(); ) {
                auto &fn = *it;
                ++it;
                if (fn.isDeclaration() &&
                    fn.getName().starts_with("__metal_get_")) {
                    fn.eraseFromParent();
                }
            }

            // Export implicit arg descriptors for AIR metadata generation
            for (auto &ia : implicitArgs) {
                airImplicitArgs.push_back({ia.airMetadata, ia.argName});
            }
        }
    }

    // Default implicit args if no __metal_* calls were found (shouldn't
    // happen for real kernels, but be safe).
    if (airImplicitArgs.empty()) {
        airImplicitArgs.push_back({"air.threadgroup_position_in_grid", "pid"});
        airImplicitArgs.push_back({"air.thread_position_in_threadgroup", "lid"});
    }

    // Add AIR metadata for Metal compilation.
    unsigned numImplicit = airImplicitArgs.size();
    unsigned numExplicit = kernelFn->arg_size() >= numImplicit
                           ? kernelFn->arg_size() - numImplicit
                           : kernelFn->arg_size();
    addAIRMetadata(*llvmMod, *kernelFn, numExplicit, airImplicitArgs);

    // Serialize to text
    static std::string result;
    result.clear();
    llvm::raw_string_ostream os(result);
    llvmMod->print(os, nullptr);

    *out_success = 1;
    return result.c_str();
}
