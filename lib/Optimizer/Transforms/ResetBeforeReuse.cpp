/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Emitter.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUBITRESETBEFOREREUSE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "reset-before-reuse"

using namespace mlir;

namespace {
class QubitResetBeforeReusePass
    : public cudaq::opt::impl::QubitResetBeforeReuseBase<
          QubitResetBeforeReusePass> {
public:
  using QubitResetBeforeReuseBase::QubitResetBeforeReuseBase;
  QubitResetBeforeReusePass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.empty())
      return;
    funcOp.dump();
    OpBuilder builder(funcOp);
    funcOp->walk([&](quake::MzOp mz) {
      bool hasNextUse = false;
      for (Value measuredQubit : mz.getWires()) {
        auto users = measuredQubit.getUsers();
        for (auto *nextOp : users) {
          llvm::outs() << "Next use of " << measuredQubit << " is " << *nextOp
                       << "\n";
          if (nextOp) {
            // If the user is a reset/measure op, nothing to do.
            if (isa<quake::ResetOp>(nextOp) || isa<quake::MzOp>(nextOp)) {
              continue;
            }

            // If this is a dealloc op, nothing to do.
            if (isa<quake::DeallocOp>(nextOp)) {
              continue;
            }

            // TODO: handle the wrap/unwrap case properly as a ref can be
            // unwrapped at multiple places.
            if (isa<quake::WrapOp>(nextOp)) {
              continue;
            }

            hasNextUse = true;
          }
        }
      }
      if (hasNextUse) {
        builder.setInsertionPointAfter(mz);
        auto measuredQubit = mz->getResult(1);
        auto wireTy = quake::WireType::get(builder.getContext());
        auto resetOp = builder.create<quake::ResetOp>(
            mz->getLoc(), TypeRange{wireTy}, measuredQubit);
        auto newWire = resetOp.getResult(0);

        // measuredQubit.replaceAllUsesExcept(newWire, resetOp);
        auto measOut = mz->getResult(0);
        mlir::Value measBit = [&]() {
          for (auto *out : measOut.getUsers()) {
            // A mz may be accompanied by a DiscriminateOp op, find that op.
            if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
              builder.setInsertionPointAfter(disc);
              return disc.getResult();
            }
          }
          // No discriminate exists - create the discriminate Op
          auto discOp = builder.create<quake::DiscriminateOp>(
              mz->getLoc(), builder.getI1Type(), measOut);
          return discOp.getResult();
        }();

        [[maybe_unused]] auto ifOp = builder.create<cudaq::cc::IfOp>(
            mz->getLoc(), TypeRange{wireTy}, measBit, ValueRange{newWire},
            [&](OpBuilder &opBuilder, Location location, Region &region) {
              region.push_back(new Block{});
              auto &bodyBlock = region.front();
              region.addArgument(wireTy, location);
              OpBuilder::InsertionGuard guard(opBuilder);
              opBuilder.setInsertionPointToStart(&bodyBlock);

              auto xOp = opBuilder.create<quake::XOp>(
                  location, TypeRange{wireTy},
                  ValueRange{region.getArgument(0)});

              auto segmentSizes = opBuilder.getDenseI32ArrayAttr({0, 0, 1});
              xOp->setAttr("operand_segment_sizes", segmentSizes);
              opBuilder.create<cudaq::cc::ContinueOp>(location,
                                                      xOp.getResult(0));
            },
            [&](OpBuilder &opBuilder, Location location, Region &region) {
              region.push_back(new Block{});
              auto &bodyBlock = region.front();
              region.addArgument(wireTy, location);
              OpBuilder::InsertionGuard guard(opBuilder);
              opBuilder.setInsertionPointToStart(&bodyBlock);
              opBuilder.create<cudaq::cc::ContinueOp>(location,
                                                      region.getArgument(0));
            });
        newWire = ifOp.getResult(0);
        measuredQubit.replaceAllUsesExcept(newWire, resetOp);
      }
      return WalkResult::advance();
    });

    printf("AFTER:\n");
    funcOp.dump();
  }
};
} // namespace
