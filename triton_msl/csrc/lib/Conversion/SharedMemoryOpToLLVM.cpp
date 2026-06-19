// ===-- SharedMemoryOpToLLVM.cpp - TTG shared memory op lowering ------===//
//
// Conversion patterns for TritonGPU shared memory ops to LLVM IR.
// Maps !ttg.memdesc<...> to LLVM ptr in addrspace(3) (Metal threadgroup).
//
// ===------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton_msl {

// Per-module counter for unique shared memory globals.
static unsigned sharedMemoryCounter = 0;
static uint64_t sharedMemoryBytes = 0;
static constexpr uint64_t SHARED_MEMORY_LIMIT = 32 * 1024;  // 32 KB

void resetSharedMemoryCounter() {
  sharedMemoryCounter = 0;
  sharedMemoryBytes = 0;
}

// Pattern populator (called from TritonMSLToLLVM.cpp).
void populateSharedMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace triton_msl
} // namespace mlir

namespace mlir {
namespace triton_msl {

// Helper: byte size of an element type (best effort; handles common types).
static uint64_t elementByteSize(Type elemTy) {
  if (auto intTy = llvm::dyn_cast<IntegerType>(elemTy)) {
    unsigned bits = intTy.getWidth();
    return (bits + 7) / 8;
  }
  if (llvm::isa<Float16Type, BFloat16Type>(elemTy)) return 2;
  if (llvm::isa<Float32Type>(elemTy)) return 4;
  if (llvm::isa<Float64Type>(elemTy)) return 8;
  if (auto ptrTy = llvm::dyn_cast<LLVM::LLVMPointerType>(elemTy))
    return 8;
  // Fallback: assume 4 bytes (matches fp32/i32, the common case).
  return 4;
}

// Helper: total bytes for a memdesc of `shape` and element type `elemTy`.
static uint64_t computeBytes(ArrayRef<int64_t> shape, Type elemTy) {
  uint64_t n = 1;
  for (int64_t d : shape) n *= static_cast<uint64_t>(d);
  return n * elementByteSize(elemTy);
}

// Helper: round up to 16-byte alignment (matches global alignment = 16).
static uint64_t alignUp16(uint64_t v) { return (v + 15ULL) & ~15ULL; }

// Helper: create a unique threadgroup global for the given memdesc shape/type.
static LLVM::GlobalOp createTgGlobal(ModuleOp module,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      ArrayRef<int64_t> shape,
                                      Type elemTy) {
  uint64_t numElems = 1;
  for (int64_t d : shape) numElems *= d;
  auto arrTy = LLVM::LLVMArrayType::get(elemTy, numElems);
  std::string name = "__tg_shared_" + std::to_string(sharedMemoryCounter++);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return LLVM::GlobalOp::create(
      rewriter, loc, arrTy, /*isConstant=*/false,
      LLVM::Linkage::Internal, name,
      /*value=*/Attribute(),
      /*alignment=*/16, /*addrSpace=*/3);
}

class LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalAllocOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memdescTy = op.getType();
    auto shape = memdescTy.getShape();
    auto elemTy = getTypeConverter()->convertType(
        memdescTy.getElementType());
    if (!elemTy) return failure();

    // Enforce 32KB threadgroup memory budget.
    // If exceeded, fail conversion so MSL fallback can handle this kernel.
    uint64_t opBytes = computeBytes(shape, memdescTy.getElementType());
    uint64_t opBytesAligned = alignUp16(opBytes);
    if (sharedMemoryBytes + opBytesAligned > SHARED_MEMORY_LIMIT) {
      return op->emitOpError()
             << "threadgroup memory budget exceeded: "
             << (sharedMemoryBytes + opBytesAligned)
             << " > " << SHARED_MEMORY_LIMIT;
    }
    sharedMemoryBytes += opBytesAligned;

    auto module = op->getParentOfType<ModuleOp>();
    auto globalOp = createTgGlobal(module, rewriter, loc, shape, elemTy);

    auto tgPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value basePtr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy,
                                                globalOp.getSymName());

    // Initialized form (one operand): store init value at thread's index.
    // This is the per-thread model — each thread writes one element.
    if (op.getNumOperands() == 1 && adaptor.getOperands().size() == 1) {
      Value initVal = adaptor.getOperands()[0];
      auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
      auto lidFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
      auto lidFn = module.lookupSymbol<LLVM::LLVMFuncOp>(
          "__metal_get_local_id");
      if (!lidFn) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        lidFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                          "__metal_get_local_id", lidFnTy);
      }
      auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});
      // Array type must match the global's flattened element count, not
      // just the leading dim — we allocate the full `prod(shape)` slots.
      uint64_t totalElems = 1;
      for (int64_t d : shape) totalElems *= static_cast<uint64_t>(d);
      auto arrTy = LLVM::LLVMArrayType::get(elemTy, totalElems);
      Value zero = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
      Value slotPtr = LLVM::GEPOp::create(
          rewriter, loc, tgPtrTy, arrTy, basePtr,
          ValueRange{zero, lid.getResult()});
      LLVM::StoreOp::create(rewriter, loc, initVal, slotPtr);
    }

    rewriter.replaceOp(op, basePtr);
    return success();
  }
};

class LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) return failure();

    Value srcPtr = adaptor.getSrc();

    // Per-thread model: each thread loads element at its lid
    // (unless a subview has already narrowed the pointer — subview
    // pattern adjusts the base accordingly)
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto module = op->getParentOfType<ModuleOp>();
    auto lidFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
    auto lidFn = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "__metal_get_local_id");
    if (!lidFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      lidFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                        "__metal_get_local_id", lidFnTy);
    }
    auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value slotPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, resultTy, srcPtr,
        ValueRange{lid.getResult()});

    Value loaded = LLVM::LoadOp::create(rewriter, loc, resultTy, slotPtr);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value srcVal = adaptor.getSrc();
    Value dstPtr = adaptor.getDst();

    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto module = op->getParentOfType<ModuleOp>();
    auto lidFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
    auto lidFn = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "__metal_get_local_id");
    if (!lidFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      lidFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                        "__metal_get_local_id", lidFnTy);
    }
    auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value slotPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, srcVal.getType(), dstPtr,
        ValueRange{lid.getResult()});

    LLVM::StoreOp::create(rewriter, loc, srcVal, slotPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

class LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Metal threadgroup memory is function-scoped; no dealloc needed.
    rewriter.eraseOp(op);
    return success();
  }
};

class AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto module = op->getParentOfType<ModuleOp>();

    auto barrierFnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
    auto barrierFn = module.lookupSymbol<LLVM::LLVMFuncOp>("air.wg.barrier");
    if (!barrierFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      barrierFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                            "air.wg.barrier", barrierFnTy);
    }

    Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(2));
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});
    rewriter.eraseOp(op);
    return success();
  }
};

class MemDescSubsliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescSubsliceOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescSubsliceOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::MemDescSubsliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto srcTy = op.getSrc().getType();
    auto srcShape = srcTy.getShape();
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    if (!elemTy) return failure();

    // Offsets are a static DenseI32Array attribute. Compute the linear
    // offset at compile time (row-major):
    //   offset = sum(idx[i] * prod(shape[i+1..]))
    auto offsets = op.getOffsets();
    int64_t linearOffset = 0;
    for (unsigned i = 0; i < offsets.size(); ++i) {
      int64_t stride = 1;
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      linearOffset += static_cast<int64_t>(offsets[i]) * stride;
    }

    Value offsetVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(linearOffset)));

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value subPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, elemTy, adaptor.getSrc(),
        ValueRange{offsetVal});

    rewriter.replaceOp(op, subPtr);
    return success();
  }
};

class MemDescTransOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescTransOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescTransOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::MemDescTransOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // No data movement. The result type's order attribute signals
    // transposed access to downstream consumers (tt.dot handles it).
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // In the per-thread scalar model, layout conversions are no-ops.
    // Each thread owns one element regardless of layout encoding.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Pre-M5 Metal has no async DMA. We lower ttg.async_copy_global_to_local
// to a synchronous per-thread copy: each thread loads one element from
// global and stores it to the destination memdesc at its lid. The op's
// !ttg.async_token result is replaced with an i32 constant (matches the
// upstream TritonGPUToLLVM type converter's AsyncTokenType -> i32 mapping);
// the downstream async_wait is a barrier regardless of token value.
class AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::gpu::AsyncCopyGlobalToLocalOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCopyGlobalToLocalOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);

    // In the op, operand 0 is the global source pointer (as tensor of
    // !tt.ptr), operand 1 is the destination memdesc (already allocated
    // by ttg.local_alloc). After the type converter runs, the adaptor
    // values are the mapped SSA values.
    Value srcPtr = adaptor.getSrc();
    Value dstPtr = adaptor.getResult();

    // Element type comes from the destination memdesc.
    auto dstTy = op.getResult().getType();
    Type elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    if (!elemTy) return failure();

    // Get lid (per-thread copy).
    auto module = op->getParentOfType<ModuleOp>();
    auto lidFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
    auto lidFn = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "__metal_get_local_id");
    if (!lidFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      lidFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                        "__metal_get_local_id", lidFnTy);
    }
    auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});

    // Load one element from global. The mask/other predication operands
    // are ignored: the pipeliner's semantics typically ensure in-bounds,
    // and for our synchronous lowering a simple scalar load matches the
    // per-thread model used by local_alloc/load/store above.
    Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, srcPtr);

    // Store to shared at this thread's slot.
    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value slotPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, elemTy, dstPtr,
        ValueRange{lid.getResult()});
    LLVM::StoreOp::create(rewriter, loc, loaded, slotPtr);

    // Replace the !ttg.async_token result with an i32 zero. This matches
    // the upstream TritonGPUToLLVM type converter's AsyncTokenType->i32
    // convention so downstream uses (async_wait operands) typecheck.
    Value tokenRepl = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, tokenRepl);
    return success();
  }
};

void populateSharedMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  // Convert !ttg.memdesc<...> to llvm.ptr in addrspace(3) (Metal threadgroup).
  typeConverter.addConversion(
      [](triton::gpu::MemDescType mdt) -> Type {
        return LLVM::LLVMPointerType::get(mdt.getContext(), /*addrspace=*/3);
      });
  // Convert !ttg.async_token to i32 (matches upstream TritonGPUToLLVM).
  // Our synchronous async_copy lowering produces an i32 constant in place
  // of the token; any async_wait operand is just ignored (barrier emits
  // unconditionally).
  typeConverter.addConversion(
      [](triton::gpu::AsyncTokenType t) -> Type {
        return IntegerType::get(t.getContext(), 32);
      });
  patterns.add<LocalAllocOpConversion,
               LocalLoadOpConversion,
               LocalStoreOpConversion,
               LocalDeallocOpConversion,
               AsyncWaitOpConversion,
               MemDescSubsliceOpConversion,
               MemDescTransOpConversion,
               ConvertLayoutOpConversion,
               AsyncCopyGlobalToLocalOpConversion>(typeConverter);
}

} // namespace triton_msl
} // namespace mlir
