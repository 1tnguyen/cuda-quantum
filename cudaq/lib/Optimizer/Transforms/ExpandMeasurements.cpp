/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

// Only an individual qubit measurement returns a scalar token. Both
// `!quake.measure` (legacy `bool`/`Result*` token) and `!cc.measure_handle`
// (the IR alias of `cudaq::measure_handle`, an `i64` payload) are scalar
// per-qubit measurement results, so neither requires expansion to a register.
template <typename A>
bool usesIndividualQubit(A x) {
  return isa<cudaq::quake::MeasureType, cudaq::cc::MeasureHandleType>(
      x.getType());
}

template <typename A>
bool hasExpandableTarget(A measureOp) {
  return llvm::any_of(measureOp.getTargets(), [](Value target) {
    if (auto size = cudaq::quake::getVeqSize(target))
      return *size != 0;
    return false;
  });
}

// Expand a statically sized veq into explicit ref operands while preserving
// the sequence-level measurement result. This is the early half of staged
// measurement expansion and deliberately creates no classical storage.
template <typename A>
class ExpandMeasurementTargets : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    if (!measureOp.getWires().empty())
      return failure();

    SmallVector<Value> targets;
    for (Value target : measureOp.getTargets()) {
      if (!isa<cudaq::quake::VeqType>(target.getType())) {
        targets.push_back(target);
        continue;
      }

      const std::optional<std::size_t> size = cudaq::quake::getVeqSize(target);
      if (!size || *size == 0)
        return failure();
      for (std::size_t i = 0; i < *size; ++i)
        targets.push_back(cudaq::quake::ExtractRefOp::create(
            rewriter, measureOp.getLoc(), target, i));
    }

    auto replacement =
        A::create(rewriter, measureOp.getLoc(), measureOp->getResultTypes(),
                  targets, measureOp.getRegisterNameAttr());
    for (NamedAttribute attr : measureOp->getAttrs())
      replacement->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(measureOp, replacement->getResults());
    return success();
  }
};

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
//
// Handles both result-type families that the vector form of `quake.mz`/`mx`/
// `my` can carry:
//   - `!cc.sequence<!quake.measure>` -- the legacy form. The only legitimate
//     consumer is `quake.discriminate`, so the rewrite folds the per-element
//     measurements straight into a `cc.sequence_init -> !cc.sequence<i1>`.
//   - `!cc.sequence<!cc.measure_handle>` -- the handle-vector value can have
//     non-discriminate consumers. Those consumers expect a value of the
//     original handle-sequence type, so the rewrite additionally builds a
//     per-element handle buffer and folds it into a `cc.sequence_init ->
//     !cc.sequence<!cc.measure_handle>` that replaces all remaining uses.
template <typename A, bool isLateStage = false>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    auto loc = measureOp.getLoc();
    auto *ctx = rewriter.getContext();

    // The dynamic-legality predicate filters out the scalar forms, so by
    // construction the result type here is `!cc.sequence<X>` for some X.
    auto sequenceResTy =
        dyn_cast<cudaq::cc::SequenceType>(measureOp.getMeasOut().getType());
    auto handleTy = cudaq::cc::MeasureHandleType::get(ctx);
    const bool isHandleResult =
        isa<cudaq::cc::MeasureHandleType>(sequenceResTy.getElementType());

    // Per-element scalar result type tracks the original sequence element
    // type. For handle inputs we measure into `!cc.measure_handle` per
    // qubit.
    Type perElemTy =
        isHandleResult ? static_cast<Type>(handleTy)
                       : static_cast<Type>(cudaq::quake::MeasureType::get(ctx));

    // Classify users so we only allocate the buffers we actually need, and
    // collect the discriminate users at the same time. The legacy
    // `!quake.measure` path has only `quake.discriminate` consumers by
    // construction; the handle path may have either, both, or none.
    SmallVector<cudaq::quake::DiscriminateOp> discUsers;
    bool hasNonDiscUser = false;
    for (auto *u : measureOp.getMeasOut().getUsers()) {
      if (auto d = dyn_cast<cudaq::quake::DiscriminateOp>(u))
        discUsers.push_back(d);
      else
        hasNonDiscUser = true;
    }
    const bool discriminateFeedsOnlyLogOutput = [&] {
      if constexpr (!isLateStage)
        return false;
      const bool allTargetsAreScalar =
          llvm::none_of(measureOp.getTargets(), [](Value target) {
            return isa<cudaq::quake::VeqType>(target.getType());
          });
      return allTargetsAreScalar && !hasNonDiscUser && !discUsers.empty() &&
             llvm::all_of(discUsers, [](cudaq::quake::DiscriminateOp disc) {
               return !disc->use_empty() &&
                      llvm::all_of(disc->getUsers(), [](Operation *user) {
                        return isa<cudaq::quake::LogOutputOp>(user);
                      });
             });
    }();

    // A discriminate used only by log_output can be represented as a
    // first-class fixed array in the late stage. Other sequence consumers
    // still require addressable storage.
    const bool needI1Buf =
        isLateStage ? !discriminateFeedsOnlyLogOutput && !discUsers.empty()
                    : !isHandleResult || !discUsers.empty();
    const bool needHandleBuf = isHandleResult && hasNonDiscUser;

    // 1. Determine the total number of qubits we need to measure. This
    // determines the size of the buffer of bools to create to store the results
    // in.
    unsigned numQubits = 0u;
    for (auto v : measureOp.getTargets())
      if (!isa<cudaq::quake::VeqType>(v.getType()))
        ++numQubits;
    Value totalToRead =
        arith::ConstantIntOp::create(rewriter, loc, numQubits, 64);
    auto i64Ty = rewriter.getI64Type();
    for (auto v : measureOp.getTargets())
      if (isa<cudaq::quake::VeqType>(v.getType())) {
        Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        totalToRead = arith::AddIOp::create(rewriter, loc, totalToRead, vecSz);
      }

    // 2. Create the buffers (one per output kind we actually need).
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value i1Buff;
    if (needI1Buf)
      i1Buff = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, totalToRead);
    Value handleBuff;
    if (needHandleBuf)
      handleBuff =
          cudaq::cc::AllocaOp::create(rewriter, loc, handleTy, totalToRead);

    // Per-element store helper. Each qubit is measured exactly once with
    // `perElemTy`; the resulting value is fanned out to whichever buffers we
    // allocated (i1 for discriminate consumers, handle for non-discriminate
    // consumers).
    SmallVector<Value> loggedBits;
    auto storePerElement = [&](OpBuilder &builder, Location loc, Value meas,
                               Value offset) {
      if (needI1Buf || discriminateFeedsOnlyLogOutput) {
        Value bit =
            cudaq::quake::DiscriminateOp::create(builder, loc, i1Ty, meas);
        if (discriminateFeedsOnlyLogOutput) {
          loggedBits.push_back(bit);
          return;
        }
        auto addr = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(i8Ty), i1Buff, offset);
        auto bitByte = cudaq::cc::CastOp::create(
            builder, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        cudaq::cc::StoreOp::create(builder, loc, bitByte, addr);
      }
      if (needHandleBuf) {
        auto addr = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(handleTy), handleBuff,
            offset);
        cudaq::cc::StoreOp::create(builder, loc, meas, addr);
      }
    };

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers, loop over the entire set of qubits.
    Value buffOff = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    SmallVector<Value> replacementWires;
    for (auto v : measureOp.getTargets()) {
      if (!isa<cudaq::quake::VeqType>(v.getType())) {
        SmallVector<Type> resultTypes{perElemTy};
        if (isa<cudaq::quake::WireType>(v.getType()))
          resultTypes.push_back(v.getType());
        auto meas = A::create(rewriter, loc, resultTypes, v,
                              measureOp.getRegisterNameAttr());
        for (NamedAttribute attr : measureOp->getAttrs())
          meas->setAttr(attr.getName(), attr.getValue());
        storePerElement(rewriter, loc, meas.getMeasOut(), buffOff);
        replacementWires.append(meas.getWires().begin(), meas.getWires().end());
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, one);
      } else {
        Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, vecSz,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value iv = block.getArgument(0);
              Value qv =
                  cudaq::quake::ExtractRefOp::create(builder, loc, v, iv);
              auto meas = A::create(builder, loc, perElemTy, qv);
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              for (NamedAttribute attr : measureOp->getAttrs())
                meas->setAttr(attr.getName(), attr.getValue());
              Value offset = arith::AddIOp::create(builder, loc, iv, buffOff);
              storePerElement(builder, loc, meas.getMeasOut(), offset);
            });
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, vecSz);
      }
    }

    // 4. Replace each `quake.discriminate` consumer. A result that is logged
    // directly becomes a first-class fixed array; all other consumers retain
    // the addressable sequence representation.
    if (discriminateFeedsOnlyLogOutput) {
      auto arrayTy = cudaq::cc::ArrayType::get(ctx, i1Ty, loggedBits.size());
      for (auto disc : discUsers) {
        Value array = cudaq::cc::UndefOp::create(rewriter, loc, arrayTy);
        for (auto [index, bit] : llvm::enumerate(loggedBits))
          array = cudaq::cc::InsertValueOp::create(rewriter, loc, arrayTy,
                                                   array, bit, index);
        SmallVector<Operation *> users{disc->getUsers().begin(),
                                       disc->getUsers().end()};
        for (Operation *user : users) {
          auto log = cast<cudaq::quake::LogOutputOp>(user);
          rewriter.modifyOpInPlace(log, [&] {
            for (OpOperand &operand : log->getOpOperands())
              if (operand.get() == disc.getResult())
                operand.set(array);
          });
        }
        rewriter.eraseOp(disc);
      }
    } else if (needI1Buf) {
      auto sequenceI1Ty = cudaq::cc::SequenceType::get(ctx, i1Ty);
      auto ptrArrI1Ty =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
      for (auto disc : discUsers) {
        auto buffCast =
            cudaq::cc::CastOp::create(rewriter, loc, ptrArrI1Ty, i1Buff);
        rewriter.template replaceOpWithNewOp<cudaq::cc::SequenceInitOp>(
            disc, sequenceI1Ty, buffCast, totalToRead);
      }
    }

    // 5. For the handle path with non-discriminate consumers, build a
    // `cc.sequence_init -> !cc.sequence<!cc.measure_handle>` over the handle
    // buffer and route the original result's remaining users to it via
    // `replaceOp` (one atomic substitution)
    Value replacementVal;
    if (needHandleBuf) {
      auto sequenceHandleTy = cudaq::cc::SequenceType::get(ctx, handleTy);
      auto handleSequence = cudaq::cc::SequenceInitOp::create(
          rewriter, loc, sequenceHandleTy, handleBuff, totalToRead);
      replacementVal = handleSequence.getResult();
    }

    // Step 5 builds a handle-vector replacement exactly when the
    // user-classification scan found a non-discriminate consumer. Without
    // this, `replaceOp` below would feed a null value through to a live
    // user.
    assert((replacementVal != nullptr) == hasNonDiscUser &&
           "handle-vector replacement must exist iff a non-discriminate "
           "consumer was present");

    assert(replacementWires.size() == measureOp.getWires().size() &&
           "expanded measurements must preserve every wire result");
    SmallVector<Value> replacements{replacementVal};
    replacements.append(replacementWires);
    rewriter.replaceOp(measureOp, replacements);
    return success();
  }
};

namespace {
using MxRewrite = ExpandRewritePattern<cudaq::quake::MxOp>;
using MyRewrite = ExpandRewritePattern<cudaq::quake::MyOp>;
using MzRewrite = ExpandRewritePattern<cudaq::quake::MzOp>;
using MxLateRewrite = ExpandRewritePattern<cudaq::quake::MxOp, true>;
using MyLateRewrite = ExpandRewritePattern<cudaq::quake::MyOp, true>;
using MzLateRewrite = ExpandRewritePattern<cudaq::quake::MzOp, true>;
using MxTargetRewrite = ExpandMeasurementTargets<cudaq::quake::MxOp>;
using MyTargetRewrite = ExpandMeasurementTargets<cudaq::quake::MyOp>;
using MzTargetRewrite = ExpandMeasurementTargets<cudaq::quake::MzOp>;

// Expand `quake.discriminate : !cc.sequence<!cc.measure_handle> ->
// !cc.sequence<i1>` when the input handle vector is *not* the direct result
// of a measurement op. The bridge emits this shape for `cudaq::to_bools`
// applied to a handle vector that has crossed an SSA boundary
// (e.g. function argument, kernel return), where the measurement-op
// pattern above cannot reach the underlying `quake.mz/mx/my`. It loops
// over the handle vector, discriminates each element, and rewraps the
// resulting bytes as a `!cc.sequence<i1>`. The direct-from-measurement
// case stays handled by `ExpandRewritePattern` to avoid an extra
// per-element load.
class ExpandSequenceHandleDiscriminate
    : public OpRewritePattern<cudaq::quake::DiscriminateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::DiscriminateOp disc,
                                PatternRewriter &rewriter) const override {
    Value handleVec = disc.getMeasurement();
    auto sequenceTy = dyn_cast<cudaq::cc::SequenceType>(handleVec.getType());
    if (!sequenceTy ||
        !isa<cudaq::cc::MeasureHandleType>(sequenceTy.getElementType()))
      return failure();
    if (handleVec.getDefiningOp<cudaq::quake::MeasurementInterface>())
      return failure();

    auto loc = disc.getLoc();
    auto *ctx = rewriter.getContext();
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    auto i64Ty = rewriter.getI64Type();
    auto handleTy = cudaq::cc::MeasureHandleType::get(ctx);

    Value vecSize =
        cudaq::cc::SequenceSizeOp::create(rewriter, loc, i64Ty, handleVec);
    auto handleArrPtrTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(handleTy));
    Value handleData = cudaq::cc::SequenceDataOp::create(
        rewriter, loc, handleArrPtrTy, handleVec);
    // Output is held in an i8 buffer, then bitcast to `!cc.ptr<!cc.array
    // <i1 x ?>>` for the wrap. This matches the convention used by the
    // measurement-op pattern above (steps 2 + 4) so downstream passes see
    // the same shape regardless of which path produced the i1 vector.
    Value i1Buff = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, vecSize);

    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSize,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value handleAddr = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(handleTy), handleData,
              iv);
          Value handleVal = cudaq::cc::LoadOp::create(builder, loc, handleAddr);
          Value bit = cudaq::quake::DiscriminateOp::create(builder, loc, i1Ty,
                                                           handleVal);
          Value byteAddr = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(i8Ty), i1Buff, iv);
          Value bitByte = cudaq::cc::CastOp::create(
              builder, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
          cudaq::cc::StoreOp::create(builder, loc, bitByte, byteAddr);
        });

    auto sequenceI1Ty = cudaq::cc::SequenceType::get(ctx, i1Ty);
    auto ptrArrI1Ty =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
    Value buffCast =
        cudaq::cc::CastOp::create(rewriter, loc, ptrArrI1Ty, i1Buff);
    rewriter.replaceOpWithNewOp<cudaq::cc::SequenceInitOp>(disc, sequenceI1Ty,
                                                           buffCast, vecSize);
    return success();
  }
};

/// Convert a `quake.reset` with a `veq` argument into a loop over the elements
/// of the `veq` and `quake.reset` on each of them.
class ResetRewrite : public OpRewritePattern<cudaq::quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ResetOp resetOp,
                                PatternRewriter &rewriter) const override {
    auto loc = resetOp.getLoc();
    auto veqArg = resetOp.getTargets();
    auto i64Ty = rewriter.getI64Type();
    Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, veqArg);
    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSz,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value qv =
              cudaq::quake::ExtractRefOp::create(builder, loc, veqArg, iv);
          cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, qv);
        });
    rewriter.eraseOp(resetOp);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::impl::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    if (stage != "all" && stage != "early" && stage != "late") {
      getOperation()->emitOpError("unknown expansion stage '") << stage << "'";
      signalPassFailure();
      return;
    }

    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<cudaq::quake::ResetOp>(
        [](cudaq::quake::ResetOp r) {
          return !isa<cudaq::quake::VeqType>(r.getTargets().getType());
        });

    if (stage == "early") {
      patterns.insert<MxTargetRewrite, MyTargetRewrite, MzTargetRewrite,
                      ResetRewrite>(ctx);
      target.addDynamicallyLegalOp<cudaq::quake::MxOp>(
          [](cudaq::quake::MxOp x) { return !hasExpandableTarget(x); });
      target.addDynamicallyLegalOp<cudaq::quake::MyOp>(
          [](cudaq::quake::MyOp x) { return !hasExpandableTarget(x); });
      target.addDynamicallyLegalOp<cudaq::quake::MzOp>(
          [](cudaq::quake::MzOp x) { return !hasExpandableTarget(x); });
      target.addLegalOp<cudaq::quake::DiscriminateOp>();
    } else {
      if (stage == "late")
        patterns.insert<MxLateRewrite, MyLateRewrite, MzLateRewrite>(ctx);
      else
        patterns.insert<MxRewrite, MyRewrite, MzRewrite>(ctx);
      patterns.insert<ResetRewrite, ExpandSequenceHandleDiscriminate>(ctx);
      target.addDynamicallyLegalOp<cudaq::quake::MxOp>(
          [](cudaq::quake::MxOp x) {
            return usesIndividualQubit(x.getMeasOut());
          });
      target.addDynamicallyLegalOp<cudaq::quake::MyOp>(
          [](cudaq::quake::MyOp x) {
            return usesIndividualQubit(x.getMeasOut());
          });
      target.addDynamicallyLegalOp<cudaq::quake::MzOp>(
          [](cudaq::quake::MzOp x) {
            return usesIndividualQubit(x.getMeasOut());
          });
      target.addDynamicallyLegalOp<cudaq::quake::DiscriminateOp>(
          [](cudaq::quake::DiscriminateOp d) {
            // Scalar discriminate is always legal.
            auto sequenceTy =
                dyn_cast<cudaq::cc::SequenceType>(d.getMeasurement().getType());
            if (!sequenceTy)
              return true;
            // Vector discriminate of legacy `!quake.measure` is folded as
            // a side-effect of the measurement-op rewrite (step 4); leave
            // it legal here so the driver does not look for a standalone
            // pattern.
            if (!isa<cudaq::cc::MeasureHandleType>(sequenceTy.getElementType()))
              return true;
            // Vector discriminate of `!cc.measure_handle` whose source is
            // a measurement op is similarly folded (step 4 again). Only
            // the indirect case needs `ExpandSequenceHandleDiscriminate`.
            return d.getMeasurement()
                       .getDefiningOp<cudaq::quake::MeasurementInterface>() !=
                   nullptr;
          });
    }

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitOpError("could not expand measurements");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createExpandMeasurementsPass() {
  return std::make_unique<ExpandMeasurementsPass>();
}
