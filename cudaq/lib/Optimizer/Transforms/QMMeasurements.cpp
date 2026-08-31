/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <map>

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERQMMEASUREMENTS
#define GEN_PASS_DEF_PACKAGEQMMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

struct PointerUses {
  std::map<std::size_t, cudaq::cc::StoreOp> stores;
  SmallVector<cudaq::cc::SequenceInitOp> sequences;
  SmallVector<Operation *> pointerOps;
};

struct MeasurementElement {
  cudaq::quake::MeasurementInterface measurement;
  cudaq::quake::DiscriminateOp discriminate;
  cudaq::cc::CastOp valueCast;
  cudaq::cc::StoreOp store;
};

static std::optional<std::size_t>
getAllocationSize(cudaq::cc::AllocaOp alloca) {
  if (Value size = alloca.getSeqSize()) {
    const auto constant = cudaq::opt::factory::getIntIfConstant(size);
    if (!constant || *constant < 0)
      return std::nullopt;
    return static_cast<std::size_t>(*constant);
  }

  const auto pointerType = dyn_cast<cudaq::cc::PointerType>(alloca.getType());
  if (!pointerType)
    return std::nullopt;
  const auto arrayType =
      dyn_cast<cudaq::cc::ArrayType>(pointerType.getElementType());
  if (!arrayType || arrayType.isUnknownSize())
    return std::nullopt;
  return static_cast<std::size_t>(arrayType.getSize());
}

static LogicalResult collectPointerUses(Value pointer,
                                        std::optional<std::size_t> index,
                                        PointerUses &uses,
                                        DenseSet<Value> &visited) {
  if (!visited.insert(pointer).second)
    return success();

  for (Operation *user : pointer.getUsers()) {
    if (auto cast = dyn_cast<cudaq::cc::CastOp>(user)) {
      if (cast.getValue() != pointer)
        return failure();
      const auto resultType = dyn_cast<cudaq::cc::PointerType>(cast.getType());
      if (!resultType)
        return failure();

      std::optional<std::size_t> nextIndex = index;
      if (!nextIndex && !isa<cudaq::cc::ArrayType>(resultType.getElementType()))
        nextIndex = 0;
      uses.pointerOps.push_back(user);
      if (failed(
              collectPointerUses(cast.getResult(), nextIndex, uses, visited)))
        return failure();
      continue;
    }

    if (auto compute = dyn_cast<cudaq::cc::ComputePtrOp>(user)) {
      if (compute.getBase() != pointer || compute.getNumIndices() != 1)
        return failure();
      const auto constant = compute.getConstantIndex(0);
      if (!constant || *constant < 0)
        return failure();

      uses.pointerOps.push_back(user);
      if (failed(collectPointerUses(compute.getResult(),
                                    static_cast<std::size_t>(*constant), uses,
                                    visited)))
        return failure();
      continue;
    }

    if (auto store = dyn_cast<cudaq::cc::StoreOp>(user)) {
      if (store.getPtrvalue() != pointer || !index)
        return failure();
      if (!uses.stores.try_emplace(*index, store).second)
        return failure();
      continue;
    }

    if (auto sequence = dyn_cast<cudaq::cc::SequenceInitOp>(user)) {
      if (sequence.getBuffer() != pointer)
        return failure();
      uses.sequences.push_back(sequence);
      continue;
    }

    return failure();
  }
  return success();
}

static cudaq::quake::MeasurementInterface
getScalarWireMeasurement(Value result) {
  auto measurement = result.getDefiningOp<cudaq::quake::MeasurementInterface>();
  if (!measurement || measurement->getResult(0) != result ||
      measurement.getTargets().size() != 1 ||
      measurement.getWires().size() != 1 ||
      !isa<cudaq::quake::WireType>(
          measurement.getTargets().front().getType()) ||
      !isa<cudaq::quake::WireType>(measurement.getWires().front().getType()) ||
      !isa<cudaq::quake::MxOp, cudaq::quake::MyOp, cudaq::quake::MzOp>(
          measurement.getOperation()))
    return {};
  return measurement;
}

static std::optional<MeasurementElement>
matchBitStore(cudaq::cc::StoreOp store) {
  auto valueCast = store.getValue().getDefiningOp<cudaq::cc::CastOp>();
  if (!valueCast || !valueCast->hasOneUse() ||
      !isa<IntegerType>(valueCast.getValue().getType()) ||
      cast<IntegerType>(valueCast.getValue().getType()).getWidth() != 1 ||
      !isa<IntegerType>(valueCast.getType()) ||
      cast<IntegerType>(valueCast.getType()).getWidth() != 8 ||
      !valueCast.getZint().value_or(false))
    return std::nullopt;

  auto discriminate =
      valueCast.getValue().getDefiningOp<cudaq::quake::DiscriminateOp>();
  if (!discriminate || !discriminate->hasOneUse())
    return std::nullopt;

  auto measurement = getScalarWireMeasurement(discriminate.getMeasurement());
  if (!measurement || !measurement->getResult(0).hasOneUse())
    return std::nullopt;

  return MeasurementElement{measurement, discriminate, valueCast, store};
}

static std::optional<MeasurementElement>
matchHandleStore(cudaq::cc::StoreOp store) {
  if (!isa<cudaq::cc::MeasureHandleType>(store.getValue().getType()))
    return std::nullopt;

  auto measurement = getScalarWireMeasurement(store.getValue());
  if (!measurement || !measurement->getResult(0).hasOneUse())
    return std::nullopt;

  return MeasurementElement{measurement, {}, {}, store};
}

static bool haveMatchingMeasurements(ArrayRef<MeasurementElement> elements,
                                     ArrayRef<Operation *> pointerOperations) {
  if (elements.empty())
    return false;

  auto exemplarMeasurement = elements.front().measurement;
  Operation *exemplar = exemplarMeasurement.getOperation();
  Type resultType = exemplarMeasurement->getResult(0).getType();
  Block *block = exemplar->getBlock();
  DenseSet<Value> targets;
  DenseSet<Value> wireResults;
  SmallPtrSet<Operation *, 32> matchedOperations;
  for (Operation *operation : pointerOperations) {
    if (operation->getBlock() != block)
      return false;
    matchedOperations.insert(operation);
  }

  for (MeasurementElement element : elements) {
    Operation *operation = element.measurement.getOperation();
    if (operation->getName() != exemplar->getName() ||
        operation->getAttrDictionary() != exemplar->getAttrDictionary() ||
        operation->getBlock() != block ||
        element.measurement->getResult(0).getType() != resultType)
      return false;

    Value target = element.measurement.getTargets().front();
    Value wire = element.measurement.getWires().front();
    if (!targets.insert(target).second)
      return false;
    wireResults.insert(wire);
    matchedOperations.insert(operation);
    matchedOperations.insert(element.store);
    if (element.discriminate)
      matchedOperations.insert(element.discriminate);
    if (element.valueCast)
      matchedOperations.insert(element.valueCast);

    if (element.store->getBlock() != block ||
        (element.discriminate && element.discriminate->getBlock() != block) ||
        (element.valueCast && element.valueCast->getBlock() != block))
      return false;
  }

  for (Value target : targets)
    if (wireResults.contains(target))
      return false;

  Operation *first = exemplar;
  Operation *last = exemplar;
  for (MeasurementElement element : elements) {
    Operation *operation = element.measurement.getOperation();
    if (operation->isBeforeInBlock(first))
      first = operation;
    if (last->isBeforeInBlock(operation))
      last = operation;
  }

  bool insideGroup = false;
  for (Operation &operation : *block) {
    if (&operation == first)
      insideGroup = true;
    if (insideGroup && !matchedOperations.contains(&operation))
      return false;
    if (&operation == last)
      break;
  }
  return true;
}

static bool sequenceMatches(cudaq::cc::SequenceInitOp sequence,
                            Type elementType, std::size_t size, Block *block) {
  const auto sequenceType =
      dyn_cast<cudaq::cc::SequenceType>(sequence.getType());
  if (!sequenceType || sequenceType.getElementType() != elementType ||
      sequence->getBlock() != block)
    return false;

  if (Value length = sequence.getLength()) {
    const auto constant = cudaq::opt::factory::getIntIfConstant(length);
    if (!constant || *constant < 0 ||
        static_cast<std::size_t>(*constant) != size)
      return false;
  }
  return true;
}

static cudaq::quake::MeasurementInterface
createMeasurementLike(PatternRewriter &rewriter,
                      cudaq::quake::MeasurementInterface exemplar,
                      TypeRange resultTypes, ValueRange targets) {
  Operation *replacement = nullptr;
  if (auto mx = dyn_cast<cudaq::quake::MxOp>(exemplar.getOperation()))
    replacement = cudaq::quake::MxOp::create(rewriter, mx.getLoc(), resultTypes,
                                             targets, mx.getRegisterNameAttr())
                      .getOperation();
  else if (auto my = dyn_cast<cudaq::quake::MyOp>(exemplar.getOperation()))
    replacement = cudaq::quake::MyOp::create(rewriter, my.getLoc(), resultTypes,
                                             targets, my.getRegisterNameAttr())
                      .getOperation();
  else if (auto mz = dyn_cast<cudaq::quake::MzOp>(exemplar.getOperation()))
    replacement = cudaq::quake::MzOp::create(rewriter, mz.getLoc(), resultTypes,
                                             targets, mz.getRegisterNameAttr())
                      .getOperation();
  else
    llvm_unreachable("unsupported measurement operation");

  for (NamedAttribute attribute : exemplar->getAttrs())
    replacement->setAttr(attribute.getName(), attribute.getValue());
  return cast<cudaq::quake::MeasurementInterface>(replacement);
}

static void erasePointerOps(PatternRewriter &rewriter,
                            SmallVector<Operation *> operations) {
  while (!operations.empty()) {
    bool madeProgress = false;
    for (auto iter = operations.begin(); iter != operations.end();) {
      Operation *operation = *iter;
      if (!operation->use_empty()) {
        ++iter;
        continue;
      }
      rewriter.eraseOp(operation);
      iter = operations.erase(iter);
      madeProgress = true;
    }
    assert(madeProgress && "matched pointer operations must form a dead tree");
  }
}

class PackageMeasurementBuffer : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    const auto size = getAllocationSize(alloca);
    if (!size || *size < 2)
      return failure();

    PointerUses uses;
    DenseSet<Value> visited;
    if (failed(collectPointerUses(alloca.getAddress(), std::nullopt, uses,
                                  visited)) ||
        uses.stores.size() != *size)
      return failure();

    const Type bufferElementType = [&] {
      Type elementType = alloca.getElementType();
      if (auto arrayType = dyn_cast<cudaq::cc::ArrayType>(elementType))
        elementType = arrayType.getElementType();
      return elementType;
    }();
    const bool bitBuffer = bufferElementType.isInteger(8);
    const bool handleBuffer =
        isa<cudaq::cc::MeasureHandleType>(bufferElementType);
    if (!bitBuffer && !handleBuffer)
      return failure();

    SmallVector<MeasurementElement> elements;
    elements.reserve(*size);
    for (std::size_t index = 0; index < *size; ++index) {
      auto iter = uses.stores.find(index);
      if (iter == uses.stores.end())
        return failure();
      auto element = bitBuffer ? matchBitStore(iter->second)
                               : matchHandleStore(iter->second);
      if (!element)
        return failure();
      elements.push_back(*element);
    }

    if (!haveMatchingMeasurements(elements, uses.pointerOps))
      return failure();

    Block *block = alloca->getBlock();
    const Type outputElementType =
        bitBuffer ? static_cast<Type>(rewriter.getI1Type())
                  : static_cast<Type>(cudaq::cc::MeasureHandleType::get(
                        rewriter.getContext()));
    if (handleBuffer && uses.sequences.empty())
      return failure();
    if (!llvm::all_of(uses.sequences, [&](auto sequence) {
          return sequenceMatches(sequence, outputElementType, *size, block);
        }))
      return failure();

    Operation *insertionPoint = elements.front().measurement.getOperation();
    for (MeasurementElement element : elements)
      if (insertionPoint->isBeforeInBlock(element.measurement.getOperation()))
        insertionPoint = element.measurement.getOperation();

    SmallVector<Value> targets;
    for (MeasurementElement element : elements)
      targets.push_back(element.measurement.getTargets().front());

    const Type scalarMeasurementType =
        elements.front().measurement->getResult(0).getType();
    SmallVector<Type> resultTypes{cudaq::cc::SequenceType::get(
        rewriter.getContext(), scalarMeasurementType)};
    resultTypes.append(*size,
                       cudaq::quake::WireType::get(rewriter.getContext()));

    rewriter.setInsertionPoint(insertionPoint);
    auto packaged = createMeasurementLike(
        rewriter, elements.front().measurement, resultTypes, targets);

    if (bitBuffer) {
      for (cudaq::cc::SequenceInitOp sequence : uses.sequences) {
        Value discriminate = cudaq::quake::DiscriminateOp::create(
            rewriter, sequence.getLoc(), sequence.getType(),
            packaged->getResult(0));
        rewriter.replaceOp(sequence, discriminate);
      }
    } else {
      for (cudaq::cc::SequenceInitOp sequence : uses.sequences)
        rewriter.replaceOp(sequence, packaged->getResult(0));
    }

    for (auto [element, wire] : llvm::zip(elements, packaged.getWires()))
      element.measurement.getWires().front().replaceAllUsesWith(wire);

    for (MeasurementElement &element : elements)
      rewriter.eraseOp(element.store);
    if (bitBuffer) {
      for (MeasurementElement &element : elements) {
        rewriter.eraseOp(element.valueCast);
        rewriter.eraseOp(element.discriminate);
      }
    }
    for (MeasurementElement &element : elements)
      rewriter.eraseOp(element.measurement.getOperation());

    erasePointerOps(rewriter, std::move(uses.pointerOps));
    rewriter.eraseOp(alloca);
    return success();
  }
};

template <typename MeasurementOp>
class LowerPackagedMeasurement : public OpRewritePattern<MeasurementOp> {
public:
  using OpRewritePattern<MeasurementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MeasurementOp measurement,
                                PatternRewriter &rewriter) const override {
    if (measurement.getTargets().size() < 2 ||
        measurement.getWires().size() != measurement.getTargets().size() ||
        !llvm::all_of(measurement.getTargets(), [](Value target) {
          return isa<cudaq::quake::WireType>(target.getType());
        }))
      return failure();

    const auto sequenceType =
        dyn_cast<cudaq::cc::SequenceType>(measurement->getResult(0).getType());
    if (!sequenceType)
      return failure();

    SmallVector<cudaq::quake::DiscriminateOp> discriminates;
    bool hasNonDiscriminateUser = false;
    for (Operation *user : measurement->getResult(0).getUsers()) {
      if (auto discriminate = dyn_cast<cudaq::quake::DiscriminateOp>(user))
        discriminates.push_back(discriminate);
      else
        hasNonDiscriminateUser = true;
    }

    const bool isHandleResult =
        isa<cudaq::cc::MeasureHandleType>(sequenceType.getElementType());
    if (hasNonDiscriminateUser && !isHandleResult)
      return failure();

    const bool discriminatesFeedOnlyLogs =
        !hasNonDiscriminateUser && !discriminates.empty() &&
        llvm::all_of(discriminates, [](auto discriminate) {
          return !discriminate->use_empty() &&
                 llvm::all_of(discriminate->getUsers(), [](Operation *user) {
                   return isa<cudaq::quake::LogOutputOp>(user);
                 });
        });
    const bool needBitBuffer =
        !discriminatesFeedOnlyLogs && !discriminates.empty();
    const bool needHandleBuffer = isHandleResult && hasNonDiscriminateUser;

    const Location location = measurement.getLoc();
    const auto i1Type = rewriter.getI1Type();
    const auto i8Type = rewriter.getI8Type();
    const auto handleType =
        cudaq::cc::MeasureHandleType::get(rewriter.getContext());
    Value total = arith::ConstantIntOp::create(
        rewriter, location, measurement.getTargets().size(), 64);

    Value bitBuffer;
    if (needBitBuffer)
      bitBuffer =
          cudaq::cc::AllocaOp::create(rewriter, location, i8Type, total);
    Value handleBuffer;
    if (needHandleBuffer)
      handleBuffer =
          cudaq::cc::AllocaOp::create(rewriter, location, handleType, total);

    SmallVector<Value> bits;
    SmallVector<Value> replacementWires;
    for (auto [index, target] : llvm::enumerate(measurement.getTargets())) {
      const Type scalarType = sequenceType.getElementType();
      SmallVector<Type> scalarResults{
          scalarType, cudaq::quake::WireType::get(rewriter.getContext())};
      auto scalar = createMeasurementLike(rewriter, measurement, scalarResults,
                                          ValueRange{target});
      replacementWires.push_back(scalar.getWires().front());

      Value offset = arith::ConstantIntOp::create(
          rewriter, location, static_cast<std::int64_t>(index), 64);
      if (needBitBuffer || discriminatesFeedOnlyLogs) {
        Value bit = cudaq::quake::DiscriminateOp::create(
            rewriter, location, i1Type, scalar->getResult(0));
        if (discriminatesFeedOnlyLogs) {
          bits.push_back(bit);
        } else {
          Value address = cudaq::cc::ComputePtrOp::create(
              rewriter, location, cudaq::cc::PointerType::get(i8Type),
              bitBuffer, offset);
          Value byte = cudaq::cc::CastOp::create(
              rewriter, location, i8Type, bit, cudaq::cc::CastOpMode::Unsigned);
          cudaq::cc::StoreOp::create(rewriter, location, byte, address);
        }
      }
      if (needHandleBuffer) {
        Value address = cudaq::cc::ComputePtrOp::create(
            rewriter, location, cudaq::cc::PointerType::get(handleType),
            handleBuffer, offset);
        cudaq::cc::StoreOp::create(rewriter, location, scalar->getResult(0),
                                   address);
      }
    }

    if (discriminatesFeedOnlyLogs) {
      const auto arrayType =
          cudaq::cc::ArrayType::get(rewriter.getContext(), i1Type, bits.size());
      for (cudaq::quake::DiscriminateOp discriminate : discriminates) {
        Value array = cudaq::cc::UndefOp::create(rewriter, location, arrayType);
        for (auto [index, bit] : llvm::enumerate(bits))
          array = cudaq::cc::InsertValueOp::create(
              rewriter, location, arrayType, array, bit, index);

        SmallVector<Operation *> users{discriminate->getUsers().begin(),
                                       discriminate->getUsers().end()};
        for (Operation *user : users) {
          auto log = cast<cudaq::quake::LogOutputOp>(user);
          rewriter.modifyOpInPlace(log, [&] {
            for (OpOperand &operand : log->getOpOperands())
              if (operand.get() == discriminate.getResult())
                operand.set(array);
          });
        }
        rewriter.eraseOp(discriminate);
      }
    } else if (needBitBuffer) {
      const auto resultType =
          cudaq::cc::SequenceType::get(rewriter.getContext(), i1Type);
      const auto bufferType =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Type));
      for (cudaq::quake::DiscriminateOp discriminate : discriminates) {
        Value castBuffer = cudaq::cc::CastOp::create(rewriter, location,
                                                     bufferType, bitBuffer);
        rewriter.replaceOpWithNewOp<cudaq::cc::SequenceInitOp>(
            discriminate, resultType, castBuffer, total);
      }
    }

    Value replacementMeasurement;
    if (needHandleBuffer) {
      const auto resultType =
          cudaq::cc::SequenceType::get(rewriter.getContext(), handleType);
      replacementMeasurement = cudaq::cc::SequenceInitOp::create(
          rewriter, location, resultType, handleBuffer, total);
    }

    assert(replacementWires.size() == measurement.getWires().size());
    SmallVector<Value> replacements{replacementMeasurement};
    replacements.append(replacementWires);
    rewriter.replaceOp(measurement, replacements);
    return success();
  }
};

class PackageQMMeasurementsPass
    : public cudaq::opt::impl::PackageQMMeasurementsBase<
          PackageQMMeasurementsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<PackageMeasurementBuffer>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class LowerQMMeasurementsPass
    : public cudaq::opt::impl::LowerQMMeasurementsBase<
          LowerQMMeasurementsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<LowerPackagedMeasurement<cudaq::quake::MxOp>,
                    LowerPackagedMeasurement<cudaq::quake::MyOp>,
                    LowerPackagedMeasurement<cudaq::quake::MzOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createPackageQMMeasurementsPass() {
  return std::make_unique<PackageQMMeasurementsPass>();
}

std::unique_ptr<mlir::Pass> cudaq::opt::createLowerQMMeasurementsPass() {
  return std::make_unique<LowerQMMeasurementsPass>();
}
