// ===-- ElementwiseOpToLLVM.cpp - Triton op → LLVM lowering patterns ------===//
//
// Conversion patterns for Triton IR ops to LLVM IR, targeting the Metal
// per-thread execution model.
//
// Key insight: in our 1D per-thread model each thread handles ONE element,
// so tensor<NxT> degenerates to scalar T. The LLVMTypeConverter is
// configured with a custom conversion for RankedTensorType that strips the
// tensor wrapper, returning just the element type.
//
// Metal builtins (program_id, local_id) are emitted as calls to external
// functions that will be resolved during MSL code generation.
//
// ===---------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cmath>
#include <limits>

namespace mlir {
namespace triton_msl {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get or insert an external function declaration into the parent module.
static LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp module,
                                             ConversionPatternRewriter &rewriter,
                                             StringRef name,
                                             LLVM::LLVMFunctionType fnTy) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), name, fnTy);
}

// ---------------------------------------------------------------------------
// tt.get_program_id → call @__metal_get_program_id_{X,Y,Z}
// ---------------------------------------------------------------------------
struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    // Pick the function name based on axis (X=0, Y=1, Z=2)
    unsigned axis = static_cast<unsigned>(op.getAxis());
    std::string fnName = "__metal_get_program_id_" + std::to_string(axis);

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, fnName, fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.make_range → call @__metal_get_local_id  (+ start offset)
//
// In our per-thread model, make_range {start, end} returns
//     start + thread_position_in_threadgroup
// (one element per thread).
// ---------------------------------------------------------------------------
struct MakeRangeOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeRangeOp> {
  using ConvertOpToLLVMPattern<triton::MakeRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, "__metal_get_local_id", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    Value lid = call.getResult();

    // Check for 2D block decomposition attributes added by the Python
    // text-level stripping phase. In 2D kernels (matmul), the linear
    // thread ID (lid) needs to be decomposed into row/col indices:
    //   metal.dim=1 → row index: lid / col_block_size
    //   metal.dim=0 → col index: lid % col_block_size
    auto dimAttr = op->getAttrOfType<IntegerAttr>("metal.dim");
    auto colBlockAttr = op->getAttrOfType<IntegerAttr>("metal.col_block_size");
    if (dimAttr && colBlockAttr) {
      int64_t dim = dimAttr.getInt();
      int64_t colBlockSize = colBlockAttr.getInt();
      auto colBlockConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(colBlockSize));
      if (dim == 1) {
        // Row index: lid / col_block_size
        lid = LLVM::UDivOp::create(rewriter, loc, i32Ty, lid, colBlockConst);
      } else if (dim == 0) {
        // Col index: lid % col_block_size
        lid = LLVM::URemOp::create(rewriter, loc, i32Ty, lid, colBlockConst);
      }
    }

    // Add the start offset if non-zero
    uint32_t start = op.getStart();
    if (start != 0) {
      auto startConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(start));
      lid = LLVM::AddOp::create(rewriter, loc, i32Ty, lid, startConst);
    }

    rewriter.replaceOp(op, lid);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.splat → passthrough (scalar is already per-thread)
// ---------------------------------------------------------------------------
struct SplatOpConversion
    : public ConvertOpToLLVMPattern<triton::SplatOp> {
  using ConvertOpToLLVMPattern<triton::SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In the per-thread model, splat is a no-op: the scalar value is already
    // the per-thread value. The type converter handles tensor<NxT> → T.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.addptr → LLVM getelementptr
// ---------------------------------------------------------------------------
struct AddPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertOpToLLVMPattern<triton::AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto resultTy = op.getType();

    // Resolve the element type through Triton's PointerType.
    // The result could be tensor<N x !tt.ptr<f32>> or just !tt.ptr<f32>.
    Type ptrElemTy;
    if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy)) {
      auto ttPtrTy = cast<triton::PointerType>(tensorTy.getElementType());
      ptrElemTy = getTypeConverter()->convertType(ttPtrTy.getPointeeType());
    } else {
      auto ttPtrTy = cast<triton::PointerType>(resultTy);
      ptrElemTy = getTypeConverter()->convertType(ttPtrTy.getPointeeType());
    }

    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    Value result = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrElemTy, ptr,
                                        offset);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.load → LLVM load (with optional mask → select)
// ---------------------------------------------------------------------------
struct LoadOpConversion
    : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Determine the scalar element type for the load.
    Type origResultTy = op.getType();
    Type elemTy;
    if (auto tensorTy = dyn_cast<RankedTensorType>(origResultTy))
      elemTy = getTypeConverter()->convertType(tensorTy.getElementType());
    else
      elemTy = getTypeConverter()->convertType(origResultTy);

    Value ptr = adaptor.getPtr();
    Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);

    // If there's a mask, use select: mask ? loaded : other
    Value mask = adaptor.getMask();
    if (mask) {
      Value other = adaptor.getOther();
      if (other) {
        loaded = LLVM::SelectOp::create(rewriter, loc, mask, loaded, other);
      }
      // If mask exists but no other value, we still use the loaded value.
      // In practice, Triton always provides 'other' when there's a mask.
    }

    rewriter.replaceOp(op, loaded);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.store → LLVM store (with optional mask → conditional via select/nop)
//
// For simplicity in the nano backend, we emit an unconditional store.
// A production backend would emit a conditional branch around the store.
// ---------------------------------------------------------------------------
struct StoreOpConversion
    : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();
    Value mask = adaptor.getMask();

    if (mask) {
      // Conditional store: create if/then around the store.
      // For the nano backend, use a simple LLVM conditional branch pattern:
      //   if (mask) store(ptr, value)
      //
      // We implement this with an scf-style block split. However, since we're
      // already in LLVM dialect territory, we use LLVM::CondBrOp.
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *storeBlock =
          rewriter.createBlock(afterBlock);

      // Current block: conditional branch
      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, mask, storeBlock, afterBlock);

      // Store block: do the store, then branch to after
      rewriter.setInsertionPointToStart(storeBlock);
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);

      // Continue at afterBlock
      rewriter.setInsertionPointToStart(afterBlock);
    } else {
      // Unconditional store
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.func → llvm.func
//
// Converts Triton's function op to an LLVM function. This is needed because
// the partial conversion will mark tt.func as illegal. We convert the
// function signature (all types through the type converter) and move the body.
// ---------------------------------------------------------------------------
struct FuncOpConversion
    : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern<triton::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *typeConverter = getTypeConverter();

    // Convert the function signature
    TypeConverter::SignatureConversion signatureConversion(
        op.getNumArguments());
    auto funcType = op.getFunctionType();

    // Convert argument types
    SmallVector<Type> convertedArgTypes;
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      Type converted = typeConverter->convertType(funcType.getInput(i));
      if (!converted)
        return failure();
      convertedArgTypes.push_back(converted);
      signatureConversion.addInputs(i, converted);
    }

    // Convert result types
    SmallVector<Type> convertedResultTypes;
    for (Type resTy : funcType.getResults()) {
      Type converted = typeConverter->convertType(resTy);
      if (!converted)
        return failure();
      convertedResultTypes.push_back(converted);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        convertedResultTypes.empty()
            ? LLVM::LLVMVoidType::get(rewriter.getContext())
            : convertedResultTypes.front(),
        convertedArgTypes);

    auto newFunc = LLVM::LLVMFuncOp::create(rewriter, loc,
                                              op.getName(), llvmFuncType);

    // Copy over any attributes we want to preserve
    if (op->hasAttr("sym_visibility"))
      newFunc->setAttr("sym_visibility", op->getAttr("sym_visibility"));

    // Move the function body
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter,
                                           &signatureConversion)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.broadcast → passthrough (per-thread model: tensor shapes are irrelevant)
// ---------------------------------------------------------------------------
struct BroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<triton::BroadcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In the per-thread model, broadcast is a no-op: the scalar is already
    // the per-thread value. The type converter handles tensor<NxT> → T.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.expand_dims → passthrough (per-thread model: dimension changes are no-ops)
// ---------------------------------------------------------------------------
struct ExpandDimsOpConversion
    : public ConvertOpToLLVMPattern<triton::ExpandDimsOp> {
  using ConvertOpToLLVMPattern<triton::ExpandDimsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.reshape → passthrough (per-thread model: reshape is a no-op on scalars)
// ---------------------------------------------------------------------------
struct ReshapeOpConversion
    : public ConvertOpToLLVMPattern<triton::ReshapeOp> {
  using ConvertOpToLLVMPattern<triton::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.extern_elementwise → LLVM intrinsic for known math functions
//
// Triton's frontend emits tt.extern_elementwise for libdevice calls like
// __nv_tanhf, __nv_sinf, etc. We map these to standard LLVM intrinsics
// which Metal's compiler understands.
// ---------------------------------------------------------------------------
struct ExternElementwiseOpConversion
    : public ConvertOpToLLVMPattern<triton::ExternElementwiseOp> {
  using ConvertOpToLLVMPattern<triton::ExternElementwiseOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto symbol = op.getSymbol();

    // Map NVIDIA libdevice symbols to Metal AIR intrinsic names.
    // Metal's GPU runtime recognizes air.fast_* as built-in math functions.
    // Using these instead of llvm.* intrinsics ensures compatibility at
    // both compilation and metallib linking stages.
    static const llvm::StringMap<llvm::StringRef> nvToAIR = {
        {"__nv_tanhf", "air.fast_tanh.f32"},
        {"__nv_tanf", "air.fast_tan.f32"},
        {"__nv_sinf", "air.fast_sin.f32"},
        {"__nv_cosf", "air.fast_cos.f32"},
        {"__nv_expf", "air.fast_exp.f32"},
        {"__nv_exp2f", "air.fast_exp2.f32"},
        {"__nv_logf", "air.fast_log.f32"},
        {"__nv_log2f", "air.fast_log2.f32"},
        {"__nv_sqrtf", "air.fast_sqrt.f32"},
        {"__nv_fabsf", "air.fast_fabs.f32"},
        {"__nv_floorf", "air.fast_floor.f32"},
        {"__nv_ceilf", "air.fast_ceil.f32"},
        {"__nv_roundf", "air.fast_round.f32"},
        {"__nv_fmaf", "air.fast_fma.f32"},
        {"__nv_powf", "air.fast_pow.f32"},
        {"__nv_fmaxf", "air.fast_fmax.f32"},
        {"__nv_fminf", "air.fast_fmin.f32"},
        {"__nv_copysignf", "air.fast_copysign.f32"},
    };

    auto it = nvToAIR.find(symbol);
    if (it == nvToAIR.end()) {
      return rewriter.notifyMatchFailure(
          op, "unknown extern_elementwise symbol: " + symbol.str());
    }

    // Get the converted result type
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // Build the LLVM intrinsic function type
    SmallVector<Type> argTypes;
    for (auto operand : adaptor.getSrcs())
      argTypes.push_back(operand.getType());

    auto fnTy = LLVM::LLVMFunctionType::get(resultTy, argTypes);

    // Get or insert the intrinsic declaration
    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, it->second, fnTy);

    // Call the intrinsic
    SmallVector<Value> args(adaptor.getSrcs().begin(),
                            adaptor.getSrcs().end());
    auto call = LLVM::CallOp::create(rewriter, loc, fn, args);
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.get_num_programs → call @__metal_get_num_programs_{0,1,2}
// ---------------------------------------------------------------------------
struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    unsigned axis = static_cast<unsigned>(op.getAxis());
    std::string fnName = "__metal_get_num_programs_" + std::to_string(axis);

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, fnName, fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.return → llvm.return
// ---------------------------------------------------------------------------
struct ReturnOpConversion
    : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::ReturnOp::create(rewriter, op->getLoc(), adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern for arith.constant with tensor types → scalar LLVM constant.
// Standard arith-to-LLVM doesn't know our tensor→scalar mapping.
class ArithConstantOpConversion
    : public ConvertOpToLLVMPattern<mlir::arith::ConstantOp> {
public:
  using ConvertOpToLLVMPattern<mlir::arith::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();
    auto value = op.getValue();
    // For dense tensor constants, extract the scalar splat value
    if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(value)) {
      if (denseAttr.isSplat()) {
        auto splatVal = denseAttr.getSplatValue<Attribute>();
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, splatVal);
        return success();
      }
      // Non-splat: use first element (per-thread model)
      auto firstVal = *denseAttr.value_begin<Attribute>();
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, firstVal);
      return success();
    }
    // Scalar constants pass through
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, value);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.reduce → AIR SIMD reduction + threadgroup shared memory
//
// Two-stage reduction:
//   Stage 1: air.simd_{sum,max,min}.f32 reduces across 32 threads in a
//            SIMD group.
//   Stage 2: First thread of each SIMD group writes its partial result to
//            threadgroup shared memory. After a barrier, the first SIMD
//            group loads all partials and does a final SIMD reduction.
//            Then broadcasts the result to all threads via shared memory.
//
// Supports up to 1024 threads (32 SIMD groups).
// ---------------------------------------------------------------------------

/// Classify the combine operation inside a tt.reduce body region.
enum class ReduceCombineKind { Sum, Max, Min, Unknown };

static ReduceCombineKind classifyCombineOp(triton::ReduceOp reduceOp) {
  Region &body = reduceOp.getCombineOp();
  Block &block = body.front();

  // Walk the block looking for the combine op.  The body has two block
  // arguments (the accumulator pair) and ends with tt.reduce.return.
  for (Operation &op : block.without_terminator()) {
    if (isa<arith::AddFOp>(&op))
      return ReduceCombineKind::Sum;
    if (isa<arith::AddIOp>(&op))
      return ReduceCombineKind::Sum;
    if (isa<arith::MaximumFOp>(&op) || isa<arith::MaxNumFOp>(&op))
      return ReduceCombineKind::Max;
    if (isa<arith::MinimumFOp>(&op) || isa<arith::MinNumFOp>(&op))
      return ReduceCombineKind::Min;
    if (isa<arith::MaxSIOp>(&op) || isa<arith::MaxUIOp>(&op))
      return ReduceCombineKind::Max;
    if (isa<arith::MinSIOp>(&op) || isa<arith::MinUIOp>(&op))
      return ReduceCombineKind::Min;
  }
  return ReduceCombineKind::Unknown;
}

/// Return the AIR SIMD intrinsic name for the given combine kind and element
/// type.  Currently only f32 is supported.
static StringRef getAIRSimdIntrinsic(ReduceCombineKind kind) {
  switch (kind) {
  case ReduceCombineKind::Sum: return "air.simd_sum.f32";
  case ReduceCombineKind::Max: return "air.simd_max.f32";
  case ReduceCombineKind::Min: return "air.simd_min.f32";
  default: return "";
  }
}

/// Return the identity value for the given combine kind (used to pad inactive
/// lanes during the cross-SIMD-group reduction).
static float getIdentityValue(ReduceCombineKind kind) {
  switch (kind) {
  case ReduceCombineKind::Sum: return 0.0f;
  case ReduceCombineKind::Max: return -std::numeric_limits<float>::infinity();
  case ReduceCombineKind::Min: return std::numeric_limits<float>::infinity();
  default: return 0.0f;
  }
}

/// Emit a scalar combine op (f32) for the given kind.
static Value emitCombine(ConversionPatternRewriter &rewriter, Location loc,
                          ReduceCombineKind kind, Value acc, Value next) {
  switch (kind) {
  case ReduceCombineKind::Sum:
    return LLVM::FAddOp::create(rewriter, loc, acc, next);
  case ReduceCombineKind::Max:
    return LLVM::MaxNumOp::create(rewriter, loc, acc, next);
  case ReduceCombineKind::Min:
    return LLVM::MinNumOp::create(rewriter, loc, acc, next);
  default:
    return acc;
  }
}

struct ReduceOpConversion
    : public ConvertOpToLLVMPattern<triton::ReduceOp> {
  using ConvertOpToLLVMPattern<triton::ReduceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    // Only handle single-operand, single-result reduces (the common case).
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "multi-operand reduce not supported");

    // Only f32 for now.
    // ReduceOp::getResult() returns a result_range; get the first result's type.
    Type origResultTy = op->getResult(0).getType();
    auto resultTy = getTypeConverter()->convertType(origResultTy);
    if (!resultTy || !resultTy.isF32())
      return rewriter.notifyMatchFailure(op, "only f32 reduce supported");

    // Classify the combine operation.
    auto kind = classifyCombineOp(op);
    if (kind == ReduceCombineKind::Unknown)
      return rewriter.notifyMatchFailure(op, "unknown reduce combine op");

    auto f32Ty = Float32Type::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty  = IntegerType::get(ctx, 1);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto module = op->getParentOfType<ModuleOp>();

    // Get the input value (already a scalar in our per-thread model).
    Value inputVal = adaptor.getSrcs()[0];

    // ---- Shape inspection for 2D axis-aware reduce ----
    //
    // The current SIMD path collapses ALL threadgroup elements to one scalar
    // — correct for 1D reductions but WRONG for 2D tensors. Detect the 2D
    // case (tensor<MxN> with both dims > 1) and dispatch to a per-thread
    // threadgroup-memory scan that respects op.getAxis().
    Type origSrcTy = op.getSrcs()[0].getType();
    auto rankedTy = dyn_cast<RankedTensorType>(origSrcTy);
    bool is2D = false;
    int64_t dimM = 1, dimN = 1;
    int64_t axis = op.getAxis();
    if (rankedTy) {
      auto shape = rankedTy.getShape();
      if (shape.size() == 2 && shape[0] > 1 && shape[1] > 1) {
        is2D = true;
        dimM = shape[0];
        dimN = shape[1];
      }
    }

    // ---- Declare helper functions ----

    // air.simd_{sum,max,min}.f32 (only used on the 1D path below)
    StringRef simdName = getAIRSimdIntrinsic(kind);
    auto simdFnTy = LLVM::LLVMFunctionType::get(f32Ty, {f32Ty});
    auto simdFn = getOrInsertFunction(module, rewriter, simdName, simdFnTy);

    // air.wg.barrier(i32, i32)
    auto barrierFnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
    auto barrierFn = getOrInsertFunction(module, rewriter, "air.wg.barrier",
                                         barrierFnTy);

    // __metal_get_sgitg → simdgroup_index_in_threadgroup
    auto idFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
    auto sgitgFn = getOrInsertFunction(module, rewriter,
                                       "__metal_get_sgitg", idFnTy);
    // __metal_get_tiisg → thread_index_in_simdgroup
    auto tiisgFn = getOrInsertFunction(module, rewriter,
                                       "__metal_get_tiisg", idFnTy);
    // __metal_get_local_id → thread_position_in_threadgroup (linear)
    auto lidFn = getOrInsertFunction(module, rewriter,
                                     "__metal_get_local_id", idFnTy);

    // ============================================================
    // 2D axis-aware reduce path (tensor<MxN> -> tensor<M> or <N>)
    // ============================================================
    if (is2D) {
      int64_t totalElems = dimM * dimN;

      // Allocate shared memory [M*N x f32] addrspace(3).
      static unsigned reduce2DCounter = 0;
      std::string sharedName =
          "__reduce2d_shared_" + std::to_string(reduce2DCounter++);
      auto f32ArrTy = LLVM::LLVMArrayType::get(f32Ty, totalElems);
      auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        LLVM::GlobalOp::create(rewriter, loc, f32ArrTy, /*isConstant=*/false,
            LLVM::Linkage::Internal, sharedName,
            /*value=*/Attribute(),
            /*alignment=*/0, /*addrSpace=*/3);
      }

      // Get linear local id.
      auto lidCall = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});
      Value lid = lidCall.getResult();

      // Store this thread's input scalar to shared[lid].
      auto sharedAddr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy,
                                                  sharedName);
      Value zero = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(0));
      Value slotPtr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32ArrTy,
                                           sharedAddr, ValueRange{zero, lid});
      LLVM::StoreOp::create(rewriter, loc, inputVal, slotPtr);

      // Barrier: wait for all threads to publish their inputs.
      Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(2));
      Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(1));
      LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});

      // Compute the base index + stride for this thread's reduction group.
      //   axis=1: base = (lid/N)*N, stride = 1,  count = N
      //   axis=0: base = lid % N,   stride = N,  count = M
      Value nConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(dimN));
      Value base;
      int64_t stride;
      int64_t count;
      if (axis == 1) {
        Value row = LLVM::UDivOp::create(rewriter, loc, i32Ty, lid, nConst);
        base = LLVM::MulOp::create(rewriter, loc, i32Ty, row, nConst);
        stride = 1;
        count = dimN;
      } else if (axis == 0) {
        base = LLVM::URemOp::create(rewriter, loc, i32Ty, lid, nConst);
        stride = dimN;
        count = dimM;
      } else {
        return rewriter.notifyMatchFailure(op, "2D reduce: axis out of range");
      }

      // Seed accumulator with identity value.
      float identityVal = getIdentityValue(kind);
      Value acc = LLVM::ConstantOp::create(rewriter, loc, f32Ty,
          rewriter.getF32FloatAttr(identityVal));

      // Unrolled per-thread scan of `count` elements.
      // For count=32 (FA critical path) this is 32 scalar ops — Metal
      // will further optimize.
      for (int64_t i = 0; i < count; ++i) {
        Value offset;
        if (stride == 1) {
          Value iConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(i));
          offset = LLVM::AddOp::create(rewriter, loc, i32Ty, base, iConst);
        } else {
          Value iConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(i * stride));
          offset = LLVM::AddOp::create(rewriter, loc, i32Ty, base, iConst);
        }
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32ArrTy,
            sharedAddr, ValueRange{zero, offset});
        Value elem = LLVM::LoadOp::create(rewriter, loc, f32Ty, elemPtr);
        acc = emitCombine(rewriter, loc, kind, acc, elem);
      }

      // No barrier needed after the read-only scan; each thread's `acc`
      // is independently computed from threadgroup memory that no further
      // writers will touch. Threads in the same reduction group all
      // produce the same value, matching the broadcast semantics of the
      // original reduce.
      rewriter.replaceOp(op, acc);
      return success();
    }

    // ---- Get thread indices ----
    auto sgitg = LLVM::CallOp::create(rewriter, loc, sgitgFn, ValueRange{});
    auto tiisg = LLVM::CallOp::create(rewriter, loc, tiisgFn, ValueRange{});
    Value sgitgVal = sgitg.getResult();
    Value tiisgVal = tiisg.getResult();

    // ---- Stage 1: SIMD-level reduction ----
    auto simdResult = LLVM::CallOp::create(rewriter, loc, simdFn,
                                           ValueRange{inputVal});
    Value simdVal = simdResult.getResult();

    // ---- Stage 2: Cross-SIMD-group reduction via threadgroup shared memory ----
    //
    // We declare a threadgroup shared memory global:
    //   @__reduce_shared_N = internal addrspace(3) global [32 x float] zeroinitializer
    //
    // (32 slots = max 32 SIMD groups for 1024 threads.)

    // Create a unique global name for this reduce op.
    static unsigned reduceCounter = 0;
    std::string sharedName = "__reduce_shared_" + std::to_string(reduceCounter++);

    // Declare the threadgroup shared memory global.
    auto f32ArrTy = LLVM::LLVMArrayType::get(f32Ty, 32);
    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3); // addrspace(3)

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      LLVM::GlobalOp::create(rewriter, loc, f32ArrTy, /*isConstant=*/false,
          LLVM::Linkage::Internal, sharedName,
          /*value=*/Attribute(),
          /*alignment=*/0, /*addrSpace=*/3);
    }

    // Get pointer to shared memory base: &shared[0]
    auto sharedAddr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy,
                                                  sharedName);

    // ---- Write partial result: if (tiisg == 0) shared[sgitg] = simd_val ----

    Value zero = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
    Value isFirstLane = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
        LLVM::ICmpPredicate::eq, tiisgVal, zero);

    // Create the conditional branch pattern:
    //   if (tiisg == 0) { store simdVal to shared[sgitg] }
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *afterStore = rewriter.splitBlock(currentBlock,
                                           rewriter.getInsertionPoint());
    auto *storeBlock = rewriter.createBlock(afterStore);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, isFirstLane, storeBlock, afterStore);

    // storeBlock: shared[sgitg] = simdVal
    rewriter.setInsertionPointToStart(storeBlock);
    Value slotPtr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32ArrTy,
                                         sharedAddr, ValueRange{zero, sgitgVal});
    LLVM::StoreOp::create(rewriter, loc, simdVal, slotPtr);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterStore);

    // ---- Barrier: wait for all SIMD groups to finish writing ----
    rewriter.setInsertionPointToStart(afterStore);
    Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(2));
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});

    // ---- Stage 2: First SIMD group loads partials and reduces ----
    // All threads load shared[tiisg] (threads with tiisg >= num_simd_groups
    // get the identity value). Then SIMD reduce again.
    //
    // We don't know num_simd_groups at compile time (it depends on block_size
    // which is a launch parameter). But we know it's <= 32. We use a
    // simpler approach:
    //   partial = shared[tiisg]   (all threads in SIMD group 0 load)
    //   final = air.simd_sum(partial)
    //
    // For threads NOT in SIMD group 0, they still participate in the barrier
    // and will get the final result via a second broadcast.

    // Load partial: all threads in sgitg==0 load shared[tiisg].
    // Threads in other SIMD groups load identity value.
    Value isFirstSG = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
        LLVM::ICmpPredicate::eq, sgitgVal, zero);

    Value partialSlotPtr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy,
        f32ArrTy, sharedAddr, ValueRange{zero, tiisgVal});
    Value partialLoad = LLVM::LoadOp::create(rewriter, loc, f32Ty,
                                              partialSlotPtr);

    // Identity value for inactive threads (other SIMD groups).
    float identityVal = getIdentityValue(kind);
    Value identity = LLVM::ConstantOp::create(rewriter, loc, f32Ty,
        rewriter.getF32FloatAttr(identityVal));

    // Select: use loaded value for first SIMD group, identity for others.
    Value partialVal = LLVM::SelectOp::create(rewriter, loc, isFirstSG,
                                               partialLoad, identity);

    // Final SIMD reduction across the partial results.
    auto finalResult = LLVM::CallOp::create(rewriter, loc, simdFn,
                                            ValueRange{partialVal});
    Value finalVal = finalResult.getResult();

    // ---- Broadcast: write final result to shared[0], barrier, all threads read ----
    Value isThread0 = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
        LLVM::ICmpPredicate::eq, tiisgVal, zero);
    Value isBroadcaster = LLVM::AndOp::create(rewriter, loc, isFirstSG,
                                               isThread0);

    // if (sgitg == 0 && tiisg == 0) shared[0] = finalVal
    auto *afterBroadcastStore = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    auto *broadcastStoreBlock = rewriter.createBlock(afterBroadcastStore);

    rewriter.setInsertionPointToEnd(broadcastStoreBlock->getPrevNode());
    LLVM::CondBrOp::create(rewriter, loc, isBroadcaster,
                            broadcastStoreBlock, afterBroadcastStore);

    rewriter.setInsertionPointToStart(broadcastStoreBlock);
    Value slot0Ptr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32ArrTy,
                                          sharedAddr, ValueRange{zero, zero});
    LLVM::StoreOp::create(rewriter, loc, finalVal, slot0Ptr);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBroadcastStore);

    // Barrier: wait for broadcast write
    rewriter.setInsertionPointToStart(afterBroadcastStore);
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});

    // All threads read the final result from shared[0].
    Value finalPtr = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32ArrTy,
                                          sharedAddr, ValueRange{zero, zero});
    Value broadcastedResult = LLVM::LoadOp::create(rewriter, loc, f32Ty,
                                                    finalPtr);

    rewriter.replaceOp(op, broadcastedResult);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.reduce.return → erased (the body is not inlined; we handle reduce
// as a single atomic pattern above)
// ---------------------------------------------------------------------------
struct ReduceReturnOpConversion
    : public ConvertOpToLLVMPattern<triton::ReduceReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReduceReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The reduce.return is consumed as part of ReduceOpConversion —
    // it should be erased when the parent ReduceOp is replaced.
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void populateTritonMSLToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();

  // Register custom type conversion for Triton's tensor types.
  // In our per-thread model, tensor<NxT> → T (scalar element type).
  typeConverter.addConversion([&typeConverter](RankedTensorType tensorTy) -> Type {
    return typeConverter.convertType(tensorTy.getElementType());
  });

  // Register custom type conversion for Triton's pointer type.
  // tt.ptr<T> → llvm.ptr (opaque pointer)
  typeConverter.addConversion([](triton::PointerType ptrTy) -> Type {
    return LLVM::LLVMPointerType::get(ptrTy.getContext());
  });

  // Add all conversion patterns
  // Higher benefit (10) so our pattern beats arith-to-LLVM's constant pattern
  // for tensor-typed constants (dense<0.0> : tensor<256xf32> → scalar 0.0f)
  patterns.add<ArithConstantOpConversion>(typeConverter, /*benefit=*/10);
  patterns.add<GetProgramIdOpConversion>(typeConverter);
  patterns.add<GetNumProgramsOpConversion>(typeConverter);
  patterns.add<MakeRangeOpConversion>(typeConverter);
  patterns.add<SplatOpConversion>(typeConverter);
  patterns.add<BroadcastOpConversion>(typeConverter);
  patterns.add<ExpandDimsOpConversion>(typeConverter);
  patterns.add<ReshapeOpConversion>(typeConverter);
  patterns.add<ExternElementwiseOpConversion>(typeConverter);
  patterns.add<AddPtrOpConversion>(typeConverter);
  patterns.add<LoadOpConversion>(typeConverter);
  patterns.add<StoreOpConversion>(typeConverter);
  patterns.add<FuncOpConversion>(typeConverter);
  patterns.add<ReturnOpConversion>(typeConverter);
  patterns.add<ReduceOpConversion>(typeConverter);
  patterns.add<ReduceReturnOpConversion>(typeConverter);
}

} // namespace triton_msl
} // namespace mlir
