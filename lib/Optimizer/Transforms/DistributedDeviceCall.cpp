/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/MD5.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cstdint>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_DISTRIBUTEDDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "distributed-device-call"

using namespace mlir;

namespace {
static std::uint32_t fnv1aHash(StringRef name) {
  std::uint32_t hash = 2166136261u;
  for (char c : name) {
    hash ^= static_cast<std::uint8_t>(c);
    hash *= 16777619u;
  }
  return hash;
}

static bool isSupportedRealtimeScalar(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    unsigned width = intTy.getWidth();
    return width == 1 || width == 8 || width == 16 || width == 32 ||
           width == 64;
  }
  return isa<Float32Type, Float64Type>(ty);
}

static std::optional<Type> getSupportedRealtimeArrayElement(Type ty) {
  auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty);
  if (!ptrTy)
    return std::nullopt;
  Type elemTy = ptrTy.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == 1 || width == 8 || width == 32)
      return elemTy;
  }
  if (isa<Float32Type, Float64Type>(elemTy))
    return elemTy;
  return std::nullopt;
}

static bool isSupportedRealtimeArrayLength(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  if (!intTy)
    return false;
  unsigned width = intTy.getWidth();
  return width == 8 || width == 16 || width == 32 || width == 64;
}

static bool isRealtimeFlatArrayPair(ValueRange args, unsigned index) {
  return index + 1 < args.size() &&
         getSupportedRealtimeArrayElement(args[index].getType()) &&
         isSupportedRealtimeArrayLength(args[index + 1].getType());
}

static std::uint64_t realtimeScalarAlignment(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return std::max<std::uint64_t>(1, intTy.getWidth() / 8);
  if (isa<Float32Type>(ty))
    return 4;
  if (isa<Float64Type>(ty))
    return 8;
  return 1;
}

static std::uint64_t realtimeArrayElementSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return std::max<std::uint64_t>(1, intTy.getWidth() / 8);
  if (isa<Float32Type>(ty))
    return 4;
  if (isa<Float64Type>(ty))
    return 8;
  return 1;
}

static LogicalResult validateRealtimeDeviceCall(cudaq::cc::DeviceCallOp op) {
  if (op.getNumResults() > 1)
    return op.emitOpError(
        "realtime device_call lowering supports at most one result");

  auto args = op.getArgs();
  for (unsigned i = 0, e = args.size(); i < e; ++i) {
    auto arg = args[i];
    if (isRealtimeFlatArrayPair(args, i)) {
      ++i;
      continue;
    }
    if (isa<cudaq::cc::PointerType>(arg.getType()))
      return op.emitOpError(
          "realtime device_call lowering does not support raw pointer "
          "arguments");
    if (!isSupportedRealtimeScalar(arg.getType()))
      return op.emitOpError("realtime device_call lowering does not support "
                            "argument type ")
             << arg.getType();
  }

  if (op.getNumResults() == 1) {
    Type resultTy = op.getResult(0).getType();
    if (!isSupportedRealtimeScalar(resultTy)) {
      return op.emitOpError("realtime device_call lowering does not support "
                            "result type ")
             << resultTy;
    }
  }

  return success();
}

static LogicalResult validateRealtimeDeviceCalls(ModuleOp module) {
  WalkResult result = module.walk([](cudaq::cc::DeviceCallOp op) {
    if (failed(validateRealtimeDeviceCall(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

static Value i64Constant(OpBuilder &builder, Location loc, std::int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static Value castIntegerToI64(OpBuilder &builder, Location loc, Value value) {
  auto i64Ty = builder.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  return builder.create<cudaq::cc::CastOp>(loc, i64Ty, value,
                                           cudaq::cc::CastOpMode::Unsigned);
}

static Value castIntegerToI32(OpBuilder &builder, Location loc, Value value) {
  auto i32Ty = builder.getI32Type();
  if (value.getType() == i32Ty)
    return value;
  auto intTy = cast<IntegerType>(value.getType());
  if (intTy.getWidth() > 32)
    return builder.create<arith::TruncIOp>(loc, i32Ty, value);
  return builder.create<arith::ExtUIOp>(loc, i32Ty, value);
}

static Value alignOffsetTo(OpBuilder &builder, Location loc, Value offset,
                           std::uint64_t alignment) {
  if (alignment <= 1)
    return offset;
  auto addend = builder.create<arith::ConstantIntOp>(loc, alignment - 1, 64);
  auto mask = builder.create<arith::ConstantIntOp>(
      loc, -static_cast<std::int64_t>(alignment), 64);
  auto incremented = builder.create<arith::AddIOp>(loc, offset, addend);
  return builder.create<arith::AndIOp>(loc, incremented, mask);
}

static Value bytePtrAt(OpBuilder &builder, Location loc, Value buffer,
                       Value offset) {
  return builder.create<cudaq::cc::ComputePtrOp>(
      loc, cudaq::cc::PointerType::get(builder.getI8Type()), buffer,
      ArrayRef<cudaq::cc::ComputePtrArg>{offset});
}

/// Compute `buffer + offset` as a pointer to `elemTy`.
static Value typedPtrAt(OpBuilder &builder, Location loc, Value buffer,
                        Value offset, Type elemTy) {
  Value bytePtr = bytePtrAt(builder, loc, buffer, offset);
  return builder.create<cudaq::cc::CastOp>(
      loc, cudaq::cc::PointerType::get(elemTy), bytePtr);
}

/// Align `cursor` to `alignment` (power of two), store `length` as an i64
/// length prefix into the request `buffer` at the aligned cursor, and return
/// the new cursor positioned just past the length prefix.
static Value writeLengthPrefix(OpBuilder &builder, Location loc, Value cursor,
                               Value buffer, Value length) {
  auto i64Ty = builder.getI64Type();
  cursor = alignOffsetTo(builder, loc, cursor, sizeof(std::uint64_t));
  Value lenPtr = typedPtrAt(builder, loc, buffer, cursor, i64Ty);
  builder.create<cudaq::cc::StoreOp>(loc, length, lenPtr);
  return builder.create<arith::AddIOp>(
      loc, cursor, i64Constant(builder, loc, sizeof(std::uint64_t)));
}

static Value computeArrayPayloadSize(OpBuilder &builder, Location loc,
                                     Value length, Type elementTy) {
  std::uint64_t elementSize = realtimeArrayElementSize(elementTy);
  if (elementSize == 1)
    return length;
  return builder.create<arith::MulIOp>(loc, length,
                                       i64Constant(builder, loc, elementSize));
}

/// Compute the total request payload size (in bytes) for a realtime
/// device_call with the given `args`.
static Value computeRealtimePayloadSize(OpBuilder &builder, Location loc,
                                        ValueRange args) {
  auto i64Ty = builder.getI64Type();
  Value lenSize = i64Constant(builder, loc, sizeof(std::uint64_t));
  Value size = i64Constant(builder, loc, 0);
  auto addAlignedLength = [&]() {
    size = alignOffsetTo(builder, loc, size, sizeof(std::uint64_t));
    size = builder.create<arith::AddIOp>(loc, size, lenSize);
  };
  for (unsigned i = 0, e = args.size(); i < e; ++i) {
    Value arg = args[i];
    if (isRealtimeFlatArrayPair(args, i)) {
      Type elementTy = *getSupportedRealtimeArrayElement(arg.getType());
      Value arrayLength = castIntegerToI64(builder, loc, args[i + 1]);
      addAlignedLength();
      Value arrayBytes =
          computeArrayPayloadSize(builder, loc, arrayLength, elementTy);
      size = builder.create<arith::AddIOp>(loc, size, arrayBytes);
      ++i;
      continue;
    }
    size = alignOffsetTo(builder, loc, size,
                         realtimeScalarAlignment(arg.getType()));
    Value argSize =
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, arg.getType());
    size = builder.create<arith::AddIOp>(loc, size, argSize);
  }
  return size;
}

static void emitTrapOnFailure(PatternRewriter &rewriter, Location loc,
                              Value status, Value frameHandle = {}) {
  auto zeroStatus = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto failedStatus = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, status, zeroStatus);
  rewriter.create<cudaq::cc::IfOp>(
      loc, TypeRange{}, failedStatus,
      [&](OpBuilder &builder, Location loc, Region &region) {
        region.push_back(new Block());
        auto &body = region.front();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        if (frameHandle)
          builder.create<func::CallOp>(
              loc, TypeRange{},
              cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
              ValueRange{frameHandle});
        auto status64 = builder.create<cudaq::cc::CastOp>(
            loc, builder.getI64Type(), status, cudaq::cc::CastOpMode::Signed);
        builder.create<func::CallOp>(loc, TypeRange{}, cudaq::opt::QISTrap,
                                     ValueRange{status64});
        builder.create<cudaq::cc::ContinueOp>(loc);
      });
}

static void addTrapImplementation(cudaq::cc::DeviceCallOp devcall,
                                  func::FuncOp devFunc,
                                  PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto &entryBlock = *devFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // Error code 2 indicates illegal execution of unreachable host code.
  Value errorCodeTwo =
      rewriter.create<arith::ConstantIntOp>(devcall.getLoc(), 2, 64);
  rewriter.create<func::CallOp>(devcall.getLoc(), TypeRange{},
                                cudaq::opt::QISTrap, ValueRange{errorCodeTwo});

  // Return unreachable values of the declared result types. The values only
  // make the IR well-formed; execution traps before reaching them.
  SmallVector<Value> trapResults;
  for (Type resTy : devFunc.getFunctionType().getResults()) {
    auto nullPtr = rewriter.create<arith::ConstantOp>(
        devcall.getLoc(), rewriter.getZeroAttr(rewriter.getIntegerType(64)));
    auto ptrTy = cudaq::cc::PointerType::get(resTy);
    auto castedNullPtr =
        rewriter.create<cudaq::cc::CastOp>(devcall.getLoc(), ptrTy, nullPtr);
    auto loadedVal =
        rewriter.create<cudaq::cc::LoadOp>(devcall.getLoc(), castedNullPtr);
    trapResults.push_back(loadedVal);
  }

  rewriter.create<func::ReturnOp>(devcall.getLoc(), trapResults);
}

static void setTrapImplementationLinkage(func::FuncOp devFunc,
                                         PatternRewriter &rewriter) {
  devFunc.setPrivate();
  auto weakOdrLinkage = mlir::LLVM::linkage::Linkage::WeakODR;
  auto linkage =
      mlir::LLVM::LinkageAttr::get(rewriter.getContext(), weakOdrLinkage);
  devFunc->setAttr("llvm.linkage", linkage);
}

class QIRVendorDeviceCallPat
    : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
  bool insertTrapImplementation;

public:
  using OpRewritePattern::OpRewritePattern;

  QIRVendorDeviceCallPat(MLIRContext *context, bool insertTrapImpl)
      : OpRewritePattern(context), insertTrapImplementation(insertTrapImpl) {}

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    constexpr const char PassthroughAttr[] = "passthrough";
    constexpr const char QIRVendorAttr[] = "cudaq-fnid";
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);
    if (!devFunc) {
      LLVM_DEBUG(llvm::dbgs() << "cannot find the function " << devFuncName
                              << " in module\n");
      return failure();
    }

    llvm::MD5 hash;
    hash.update(devFuncName);
    llvm::MD5::MD5Result result;
    hash.final(result);
    std::uint32_t callbackCode = result.low();

    if (insertTrapImplementation && devFunc.isDeclaration()) {
      // If `insertTrapImplementation` is enabled (e.g., AOT compilation for
      // remote hardware providers), we want to insert a trap implementation for
      // any unresolved device function (declaration only), so that we can
      // perform AOT compilation without needing the actual device function
      // definitions. This trap function will never be executed as the remote
      // JIT pipeline would not be using the `device_call` functions anyway.
      // Rather, these functions will only be resolved at runtime by the remote
      // provider's runtime library.

      // (1) Add a trap implementation for this device function declaration.
      addTrapImplementation(devcall, devFunc, rewriter);

      // (2) Set this trap function as private and weak_odr linkage, to allow
      // multiple definitions across translation units without linker errors.
      // For example, compiling for a remote hardware provider with the actual
      // device call library linkage (even though unused) should not cause any
      // problems.
      setTrapImplementationLinkage(devFunc, rewriter);

      // (3) Replace the device call with a no-inline call to prevent inlining
      // of the trap function.
      // We use a no-inline call here to ensure that the call to the device
      // function is preserved as a call in the IR (even in the presence of the
      // trap implementation). If the actual implementation is provided at link
      // time, it will be used instead of the trap implementation due to the
      // weak_odr linkage.
      rewriter.replaceOpWithNewOp<cudaq::cc::NoInlineCallOp>(
          devcall, devFunc.getFunctionType().getResults(), devFuncName,
          devcall.getArgs(), ArrayAttr{}, ArrayAttr{});

      return success();
    }

    bool needToAddIt = true;
    SmallVector<Attribute> funcIdAttr;
    if (auto passthruAttr = devFunc->getAttr(PassthroughAttr)) {
      auto arrayAttr = cast<ArrayAttr>(passthruAttr);
      funcIdAttr.append(arrayAttr.begin(), arrayAttr.end());
      for (auto a : arrayAttr) {
        if (auto strArrAttr = dyn_cast<ArrayAttr>(a)) {
          auto strAttr = dyn_cast<StringAttr>(strArrAttr[0]);
          if (!strAttr)
            continue;
          if (strAttr.getValue() == QIRVendorAttr) {
            needToAddIt = false;
            break;
          }
        }
      }
    }
    if (needToAddIt) {
      auto callbackCodeAsStr = std::to_string(callbackCode);
      funcIdAttr.push_back(rewriter.getStrArrayAttr(
          {QIRVendorAttr, rewriter.getStringAttr(callbackCodeAsStr)}));
      devFunc->setAttr(PassthroughAttr, rewriter.getArrayAttr(funcIdAttr));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        devcall, devFunc.getFunctionType().getResults(), devFuncName,
        devcall.getArgs());
    return success();
  }
};

class RealtimeDeviceCallPat : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    // Module-level validation has already been performed by the pass before
    // patterns are applied; assert here to catch programming errors.
    assert(succeeded(validateRealtimeDeviceCall(devcall)) &&
           "realtime device_call should have been validated");

    auto loc = devcall.getLoc();
    auto callee = devcall.getCallee();
    std::uint32_t functionId = fnv1aHash(callee);

    auto i8Ty = rewriter.getI8Type();
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);

    auto args = devcall.getArgs();
    Value requestSize = computeRealtimePayloadSize(rewriter, loc, args);

    Type resultTy = nullptr;
    Value responseCapacity = i64Constant(rewriter, loc, 0);
    if (devcall.getNumResults() == 1) {
      resultTy = devcall.getResult(0).getType();
      responseCapacity =
          rewriter.create<cudaq::cc::SizeOfOp>(loc, i64Ty, resultTy);
    }

    auto functionIdValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<std::int64_t>(functionId), 32);
    Value deviceIdValue =
        devcall.getDevice()
            ? castIntegerToI32(rewriter, loc, devcall.getDevice())
            : rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    Value frameHandleSlot = rewriter.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
    Value requestPayloadSlot =
        rewriter.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
    Value responsePayloadSlot =
        rewriter.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
    Value responseLen = rewriter.create<cudaq::cc::AllocaOp>(loc, i64Ty);
    rewriter.create<cudaq::cc::StoreOp>(loc, i64Constant(rewriter, loc, 0),
                                        responseLen);

    auto acquireStatus = rewriter.create<func::CallOp>(
        loc, i32Ty, cudaq::runtime::deviceCallAcquireRealtimeFrame,
        ValueRange{deviceIdValue, functionIdValue, requestSize,
                   responseCapacity, frameHandleSlot, requestPayloadSlot,
                   responsePayloadSlot});
    emitTrapOnFailure(rewriter, loc, acquireStatus.getResult(0));

    Value frameHandle =
        rewriter.create<cudaq::cc::LoadOp>(loc, frameHandleSlot);
    Value requestPayload =
        rewriter.create<cudaq::cc::LoadOp>(loc, requestPayloadSlot);
    auto payloadArrayPtrTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
    Value requestBuffer = rewriter.create<cudaq::cc::CastOp>(
        loc, payloadArrayPtrTy, requestPayload);
    Value responseBuffer;
    if (resultTy) {
      Value responsePayload =
          rewriter.create<cudaq::cc::LoadOp>(loc, responsePayloadSlot);
      responseBuffer = rewriter.create<cudaq::cc::CastOp>(
          loc, payloadArrayPtrTy, responsePayload);
    }

    Value cursor = i64Constant(rewriter, loc, 0);
    for (unsigned i = 0, e = args.size(); i < e; ++i) {
      auto arg = args[i];
      if (isRealtimeFlatArrayPair(args, i)) {
        Type elementTy = *getSupportedRealtimeArrayElement(arg.getType());
        Value arrayLength = castIntegerToI64(rewriter, loc, args[i + 1]);
        cursor = writeLengthPrefix(rewriter, loc, cursor, requestBuffer,
                                   arrayLength);

        Value dataStart = cursor;
        auto argPtrTy = cast<cudaq::cc::PointerType>(arg.getType());
        auto arrayArgTy = cudaq::cc::PointerType::get(
            cudaq::cc::ArrayType::get(argPtrTy.getElementType()));
        auto arrayArg =
            rewriter.create<cudaq::cc::CastOp>(loc, arrayArgTy, arg);
        std::uint64_t elementSize = realtimeArrayElementSize(elementTy);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, arrayLength,
            [&, elementSize, elementTy](OpBuilder &builder, Location loc,
                                        Region &, Block &block) {
              Value index = block.getArgument(0);
              auto elementPtr = builder.create<cudaq::cc::ComputePtrOp>(
                  loc, arg.getType(), arrayArg,
                  ArrayRef<cudaq::cc::ComputePtrArg>{index});
              auto loadedElement =
                  builder.create<cudaq::cc::LoadOp>(loc, elementPtr);
              Value dstIndexOffset = index;
              if (elementSize != 1)
                dstIndexOffset = builder.create<arith::MulIOp>(
                    loc, index, i64Constant(builder, loc, elementSize));
              Value dstOffset =
                  builder.create<arith::AddIOp>(loc, dataStart, dstIndexOffset);
              if (auto intTy = dyn_cast<IntegerType>(elementTy);
                  intTy && intTy.getWidth() == 1) {
                auto byte = builder.create<cudaq::cc::CastOp>(
                    loc, i8Ty, loadedElement.getResult(),
                    cudaq::cc::CastOpMode::Unsigned);
                Value dstPtr =
                    bytePtrAt(builder, loc, requestBuffer, dstOffset);
                builder.create<cudaq::cc::StoreOp>(loc, byte, dstPtr);
              } else {
                Value dstPtr = typedPtrAt(builder, loc, requestBuffer,
                                          dstOffset, elementTy);
                builder.create<cudaq::cc::StoreOp>(
                    loc, loadedElement.getResult(), dstPtr);
              }
            });
        Value arrayBytes =
            computeArrayPayloadSize(rewriter, loc, arrayLength, elementTy);
        cursor = rewriter.create<arith::AddIOp>(loc, cursor, arrayBytes);
        ++i;
        continue;
      }
      cursor = alignOffsetTo(rewriter, loc, cursor,
                             realtimeScalarAlignment(arg.getType()));
      Value typedArgPtr =
          typedPtrAt(rewriter, loc, requestBuffer, cursor, arg.getType());
      rewriter.create<cudaq::cc::StoreOp>(loc, arg, typedArgPtr);
      Value argSize =
          rewriter.create<cudaq::cc::SizeOfOp>(loc, i64Ty, arg.getType());
      cursor = rewriter.create<arith::AddIOp>(loc, cursor, argSize);
    }

    auto status = rewriter.create<func::CallOp>(
        loc, i32Ty, cudaq::runtime::deviceCallDispatchRealtimeFrame,
        ValueRange{frameHandle, responseLen});
    emitTrapOnFailure(rewriter, loc, status.getResult(0), frameHandle);

    if (!resultTy) {
      rewriter.create<func::CallOp>(
          loc, TypeRange{},
          cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
          ValueRange{frameHandle});
      rewriter.eraseOp(devcall);
      return success();
    }

    auto resultPtr = rewriter.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(resultTy), responseBuffer);
    auto result = rewriter.create<cudaq::cc::LoadOp>(loc, resultPtr);
    rewriter.create<func::CallOp>(
        loc, TypeRange{}, cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
        ValueRange{frameHandle});
    rewriter.replaceOp(devcall, ValueRange{result.getResult()});
    return success();
  }
};

class ResolveDevicePtrOpPat
    : public OpRewritePattern<cudaq::cc::ResolveDevicePtrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ResolveDevicePtrOp resolve,
                                PatternRewriter &rewriter) const override {
    auto loc = resolve.getLoc();
    auto call = func::CallOp::create(
        rewriter, loc,
        TypeRange{cudaq::cc::PointerType::get(rewriter.getI8Type())},
        cudaq::runtime::extractDevPtr, ValueRange{resolve.getDevicePtr()});
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
        resolve, resolve.getResult().getType(), call.getResult(0));
    return success();
  }
};

class DistributedDeviceCallPass
    : public cudaq::opt::impl::DistributedDeviceCallBase<
          DistributedDeviceCallPass> {
public:
  using DistributedDeviceCallBase::DistributedDeviceCallBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleOp module = getOperation();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (realtimeLowering) {
      if (failed(validateRealtimeDeviceCalls(module))) {
        signalPassFailure();
        return;
      }

      for (auto name : {cudaq::runtime::deviceCallAcquireRealtimeFrame,
                        cudaq::runtime::deviceCallDispatchRealtimeFrame,
                        cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame}) {
        if (failed(irBuilder.loadIntrinsic(module, name))) {
          module.emitError(std::string{"could not load "} + name);
          signalPassFailure();
          return;
        }
      }

      if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
        module.emitError("could not load QIR trap function.");
        signalPassFailure();
        return;
      }

      patterns.add<RealtimeDeviceCallPat>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        signalPassFailure();
      return;
    }

    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::extractDevPtr))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::extractDevPtr);
      signalPassFailure();
      return;
    }

    if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
      module.emitError("could not load QIR trap function.");
      signalPassFailure();
      return;
    }

    patterns.add<ResolveDevicePtrOpPat>(ctx);
    patterns.insert<QIRVendorDeviceCallPat>(ctx, insertTrapImplementation);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
    return;
  }
};
} // namespace
