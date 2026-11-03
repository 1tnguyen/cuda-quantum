/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Instrument-level validation of the Quake PBC passes.
///
/// Interprets emitted primitive gates and conditional measurement states to
/// check Pauli-product projectors, eigenspace coherence, and every joint
/// outcome effect of the conversion/optimization/lowering pipeline.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <bit>
#include <complex>
#include <gtest/gtest.h>
#include <numbers>
#include <sstream>

using namespace mlir;
namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex I{0, 1};

// A small independent interpreter for the actual emitted measurement gadget.
// It returns an unnormalized conditional state, so it detects both incorrect
// probabilities and destruction of coherence within a parity eigenspace.
double scalar(Value value) {
  if (auto c = value.getDefiningOp<arith::ConstantOp>())
    return cast<FloatAttr>(c.getValue()).getValueAsDouble();
  if (auto neg = value.getDefiningOp<arith::NegFOp>())
    return -scalar(neg.getOperand());
  if (auto mul = value.getDefiningOp<arith::MulFOp>())
    return scalar(mul.getLhs()) * scalar(mul.getRhs());
  ADD_FAILURE() << "Unsupported classical expression";
  return 0;
}
State pauli(const State &state, unsigned x, unsigned z) {
  State result(state.size());
  for (unsigned b = 0; b < state.size(); ++b) {
    Complex phase = std::pow(I, std::popcount(x & z));
    if (std::popcount(z & b) & 1)
      phase = -phase;
    result[b ^ x] = phase * state[b];
  }
  return result;
}
State runGadget(func::FuncOp function, unsigned col, unsigned outcomes) {
  unsigned n = function.getNumArguments();
  State state(1u << n);
  state[col] = 1;
  DenseMap<Value, unsigned> ids;
  for (auto [i, arg] : llvm::enumerate(function.getArguments()))
    ids[arg] = i;
  unsigned measurement = 0;
  for (Operation &op : function.front()) {
    if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
      unsigned target = ids.lookup(gate.getTargets()[0]);
      if (!gate.getControls().empty()) {
        unsigned control = ids.lookup(gate.getControls()[0]);
        EXPECT_TRUE((isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op)));
        for (unsigned b = 0; b < state.size(); ++b)
          if ((b >> control) & 1) {
            if (isa<cudaq::quake::ZOp>(op)) {
              if ((b >> target) & 1)
                state[b] = -state[b];
            } else if (!((b >> target) & 1))
              std::swap(state[b], state[b ^ (1u << target)]);
          }
      } else if (isa<cudaq::quake::RxOp, cudaq::quake::RyOp, cudaq::quake::RzOp,
                     cudaq::quake::ExpPauliOp>(op)) {
        unsigned x = 0, z = 0;
        double angle = scalar(gate.getParameters()[0]);
        if (auto exp = dyn_cast<cudaq::quake::ExpPauliOp>(op)) {
          for (auto [i, c] : llvm::enumerate(*exp.getPauliLiteral())) {
            unsigned bit = 1u << ids.lookup(exp.getTargets()[i]);
            if (c == 'X' || c == 'Y')
              x |= bit;
            if (c == 'Z' || c == 'Y')
              z |= bit;
          }
        } else {
          if (!isa<cudaq::quake::RzOp>(op))
            x = 1u << target;
          if (!isa<cudaq::quake::RxOp>(op))
            z = 1u << target;
          angle *= -0.5;
        }
        if (op.hasAttr("is_adj"))
          angle = -angle;
        auto product = pauli(state, x, z);
        for (unsigned b = 0; b < state.size(); ++b)
          state[b] =
              std::cos(angle) * state[b] + I * std::sin(angle) * product[b];
      } else {
        for (unsigned b = 0; b < state.size(); ++b)
          if (!((b >> target) & 1)) {
            auto c = b | (1u << target);
            auto a = state[b], d = state[c];
            if (isa<cudaq::quake::HOp>(op)) {
              state[b] = (a + d) / std::sqrt(2.0);
              state[c] = (a - d) / std::sqrt(2.0);
            } else if (isa<cudaq::quake::SOp>(op))
              state[c] *= op.hasAttr("is_adj") ? -I : I;
            else if (isa<cudaq::quake::TOp>(op))
              state[c] *= std::exp(I * (op.hasAttr("is_adj") ? -1.0 : 1.0) *
                                   std::numbers::pi / 4.0);
            else if (isa<cudaq::quake::ZOp>(op))
              state[c] = -d;
            else if (isa<cudaq::quake::YOp>(op)) {
              state[b] = -I * d;
              state[c] = I * a;
            } else if (isa<cudaq::quake::XOp>(op))
              std::swap(state[b], state[c]);
            else
              ADD_FAILURE() << "Unexpected primitive in mpp lowering";
          }
      }
      SmallVector<Value> inputs(gate.getControls());
      llvm::append_range(inputs, gate.getTargets());
      for (auto [input, output] : llvm::zip(inputs, gate.getWires()))
        ids[output] = ids.lookup(input);
    } else if (auto m = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
      unsigned x = 0, z = 0;
      bool negative = false;
      if (auto joint = dyn_cast<cudaq::quake::MppOp>(op)) {
        negative = joint.getNegative();
        for (auto [i, c] : llvm::enumerate(joint.getWord())) {
          unsigned bit = 1u << ids.lookup(joint.getTargets()[i]);
          if (c == 'X' || c == 'Y')
            x |= bit;
          if (c == 'Z' || c == 'Y')
            z |= bit;
        }
      } else {
        unsigned bit = 1u << ids.lookup(m.getTargets()[0]);
        if (!isa<cudaq::quake::MzOp>(op))
          x = bit;
        if (!isa<cudaq::quake::MxOp>(op))
          z = bit;
      }
      auto product = pauli(state, x, z);
      bool sign = negative ^ bool((outcomes >> measurement++) & 1);
      for (unsigned b = 0; b < state.size(); ++b)
        state[b] = (state[b] + (sign ? -product[b] : product[b])) / 2.0;
      // Quake mx/my include the basis rotation without restoring it: their
      // output wire is in |0> or |1>, unlike the MPP projector instrument.
      if (isa<cudaq::quake::MxOp, cudaq::quake::MyOp>(op)) {
        unsigned bit = 1u << ids.lookup(m.getTargets()[0]);
        if (isa<cudaq::quake::MyOp>(op))
          for (unsigned b = 0; b < state.size(); ++b)
            if (b & bit)
              state[b] *= -I;
        for (unsigned b = 0; b < state.size(); ++b)
          if (!(b & bit)) {
            auto a = state[b], d = state[b | bit];
            state[b] = (a + d) / std::sqrt(2.0);
            state[b | bit] = (a - d) / std::sqrt(2.0);
          }
      }
      for (auto [input, output] : llvm::zip(m.getTargets(), m.getWires()))
        ids[output] = ids.lookup(input);
    } else
      EXPECT_TRUE((isa<cudaq::quake::SinkOp, func::ReturnOp>(op) ||
                   op.getName().getDialectNamespace() == "arith"));
  }
  return state;
}
} // namespace

TEST(PBCPass, TerminalMeasurementPreservesEveryProjectorEffect) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      cudaq::quake::QuakeDialect, cudaq::cc::CCDialect>();
  context.disableMultithreading();
  for (unsigned n = 1; n <= 3; ++n) {
    unsigned count = n == 1 ? 3 : n == 2 ? 9 : 27;
    for (unsigned encoded = 0; encoded < count; ++encoded)
      for (bool negative : {false, true}) {
        std::string word;
        unsigned x = 0, z = 0, code = encoded;
        for (unsigned i = 0; i < n; ++i) {
          char p = "XYZ"[code % 3];
          code /= 3;
          word += p;
          if (p != 'Z')
            x |= 1u << i;
          if (p != 'X')
            z |= 1u << i;
        }
        std::ostringstream source;
        source << "func.func @test(";
        for (unsigned i = 0; i < n; ++i)
          source << (i ? ", " : "") << "%q" << i << ": !quake.wire";
        source << ") -> !quake.measure {\n %m, %r:" << n << " = quake.mpp ";
        for (unsigned i = 0; i < n; ++i)
          source << (i ? ", " : "") << "%q" << i;
        source << " to \"" << word << "\" : (";
        for (unsigned i = 0; i < n; ++i)
          source << (i ? ", " : "") << "!quake.wire";
        source << ") -> (!quake.measure";
        for (unsigned i = 0; i < n; ++i)
          source << ", !quake.wire";
        source << ") {negative = " << (negative ? "true" : "false") << "}\n";
        for (unsigned i = 0; i < n; ++i)
          source << "quake.sink %r#" << i << " : !quake.wire\n";
        source << "return %m : !quake.measure\n}";
        auto module = parseSourceString<ModuleOp>(source.str(), &context);
        ASSERT_TRUE(module) << source.str();
        PassManager pm(&context);
        pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeLowerPBC());
        ASSERT_TRUE(succeeded(pm.run(*module)));
        auto function = module->lookupSymbol<func::FuncOp>("test");
        for (unsigned col = 0; col < (1u << n); ++col)
          for (bool outcome : {false, true}) {
            State expected(1u << n);
            expected[col] = 0.5;
            Complex phase = std::pow(I, std::popcount(x & z));
            if (bool(std::popcount(z & col) & 1) ^ negative ^ outcome)
              phase = -phase;
            expected[col ^ x] += phase / 2.0;
            auto actual = runGadget(function, col, outcome);
            // All quantum outputs are sunk. Compare K^dagger K, not the
            // discarded post-measurement state, including off-diagonal entries.
            for (unsigned other = 0; other < actual.size(); ++other) {
              auto otherState = runGadget(function, other, outcome);
              Complex effect = 0;
              for (unsigned row = 0; row < actual.size(); ++row)
                effect += std::conj(otherState[row]) * actual[row];
              ASSERT_LT(std::abs(expected[other] - effect), 1e-12)
                  << word << " negative=" << negative << " outcome=" << outcome;
            }
          }
      }
  }
}

TEST(PBCPass, FullPipelinePreservesEveryJointOutcomeEffect) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      cudaq::quake::QuakeDialect, cudaq::cc::CCDialect>();
  context.disableMultithreading();
  // Exercises the MLIR boundary: exponent conventions, exact T annotations,
  // adjoints, dynamic arithmetic and transformed terminal measurement words.
  auto original = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a: !quake.wire, %b: !quake.wire) -> (!quake.measure, !quake.measure) {
      %angle = arith.constant 0.173 : f64
      %h = quake.h %a : (!quake.wire) -> !quake.wire
      %c:2 = quake.x [%h] %b : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %t = quake.t %c#1 : (!quake.wire) -> !quake.wire
      %t2 = quake.t %t : (!quake.wire) -> !quake.wire
      %r = quake.ry<adj> (%angle) %c#0 : (f64, !quake.wire) -> !quake.wire
      %e:2 = quake.exp_pauli<adj> (%angle) %r, %t2 to "YX" : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %m0, %q0 = quake.mx %e#0 : (!quake.wire) -> (!quake.measure, !quake.wire)
      %m1, %q1 = quake.my %e#1 : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %q0 : !quake.wire
      quake.sink %q1 : !quake.wire
      return %m0, %m1 : !quake.measure, !quake.measure
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(original);
  OwningOpRef<ModuleOp> optimized(cast<ModuleOp>(original->clone()));
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeToPBC());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeOptimizePBC());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeLowerPBC());
  ASSERT_TRUE(succeeded(pm.run(*optimized)));
  auto lhs = original->lookupSymbol<func::FuncOp>("test");
  auto rhs = optimized->lookupSymbol<func::FuncOp>("test");
  for (unsigned outcome = 0; outcome < 4; ++outcome) {
    std::vector<State> before, after;
    for (unsigned col = 0; col < 4; ++col) {
      before.push_back(runGadget(lhs, col, outcome));
      after.push_back(runGadget(rhs, col, outcome));
    }
    // Compare K^dagger K, including off-diagonals, for each joint outcome.
    // This proves equal probabilities for every input state, not only |00>.
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; j < 4; ++j) {
        Complex a = 0, b = 0;
        for (unsigned row = 0; row < 4; ++row) {
          a += std::conj(before[i][row]) * before[j][row];
          b += std::conj(after[i][row]) * after[j][row];
        }
        ASSERT_LT(std::abs(a - b), 1e-12) << outcome << " " << i << " " << j;
      }
  }
}

namespace {
constexpr StringLiteral primitiveInput = R"mlir(
  func.func @test(%q: !quake.wire) -> !quake.measure {
    %h = quake.h %q : (!quake.wire) -> !quake.wire
    %t = quake.t %h : (!quake.wire) -> !quake.wire
    %u = quake.t %t : (!quake.wire) -> !quake.wire
    %m, %end = quake.mx %u name "answer" : (!quake.wire) -> (!quake.measure, !quake.wire)
    quake.sink %end : !quake.wire
    return %m : !quake.measure
  }
)mlir";
constexpr StringLiteral pbcInput = R"mlir(
  func.func @test(%q: !quake.wire) -> !quake.measure {
    %angle = arith.constant -0.39269908169872414 : f64
    %t = quake.exp_pauli (%angle) %q to "Z" : (f64, !quake.wire) -> !quake.wire {quake.pbc_angle = array<i64: -1, 8>}
    %u = quake.exp_pauli (%angle) %t to "Z" : (f64, !quake.wire) -> !quake.wire {quake.pbc_angle = array<i64: -1, 8>}
    %m, %end = quake.mpp %u to "X" : (!quake.wire) -> (!quake.measure, !quake.wire) {registerName = "answer"}
    quake.sink %end : !quake.wire
    return %m : !quake.measure
  }
)mlir";

std::string printed(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  return text;
}

/// A clean optional fallback must let the next pipeline pass run. An actual
/// failure must stop the pipeline; it must never masquerade as a successful
/// skip.
struct ContinuationPass
    : PassWrapper<ContinuationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContinuationPass)
  bool *reached;
  explicit ContinuationPass(bool &reached) : reached(&reached) {}
  void runOnOperation() override { *reached = true; }
};

class PBCBoundaryTest : public ::testing::Test {
protected:
  MLIRContext context;
  PBCBoundaryTest() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        cudaq::quake::QuakeDialect, cudaq::cc::CCDialect>();
    context.disableMultithreading();
    context.allowUnregisteredDialects();
  }

  void unchanged(ModuleOp module, std::unique_ptr<Pass> pass, bool success,
                 StringRef diagnostic = {}) {
    std::string before = printed(module), messages;
    SmallVector<Operation *> identities;
    module.walk([&](Operation *op) { identities.push_back(op); });
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
      llvm::raw_string_ostream stream(messages);
      d.print(stream);
      return mlir::success();
    });
    bool continued = false;
    PassManager pm(&context);
    // Exercise the pass's own validation, not only the pass manager verifier.
    pm.enableVerifier(false);
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    pm.addNestedPass<func::FuncOp>(
        std::make_unique<ContinuationPass>(continued));
    EXPECT_EQ(succeeded(pm.run(module)), success) << messages;
    EXPECT_EQ(continued, success);
    EXPECT_EQ(printed(module), before);
    SmallVector<Operation *> after;
    module.walk([&](Operation *op) { after.push_back(op); });
    EXPECT_EQ(after, identities);
    if (!diagnostic.empty())
      EXPECT_NE(messages.find(diagnostic.str()), std::string::npos) << messages;
    else if (success)
      EXPECT_TRUE(messages.empty()) << messages;
  }
};
} // namespace

TEST_F(PBCBoundaryTest, ReusedXYMeasurementFallsBackWithoutMutation) {
  for (StringRef axis : {"x", "y"}) {
    std::string source =
        "func.func @test(%q: !quake.wire) -> (!quake.measure, !quake.measure) "
        "{\n"
        "  %m, %r = quake.m" +
        axis.str() +
        " %q : (!quake.wire) -> (!quake.measure, !quake.wire)\n"
        "  %n, %s = quake.m" +
        axis.str() +
        " %r : (!quake.wire) -> (!quake.measure, !quake.wire)\n"
        "  quake.sink %s : !quake.wire\n"
        "  return %m, %n : !quake.measure, !quake.measure\n}";
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto f = module->lookupSymbol<func::FuncOp>("test");
    for (unsigned outcome = 0; outcome < 4; ++outcome) {
      auto state = runGadget(f, 0, outcome);
      double probability = 0;
      for (auto value : state)
        probability += std::norm(value);
      EXPECT_NEAR(probability, 0.25, 1e-12);
    }
    unchanged(*module, cudaq::opt::createQuakeTryPBC(), true);
    unchanged(*module, cudaq::opt::createQuakeToPBC(), false,
              "PBC requires mx/my results to go directly to sink");
  }
}

TEST_F(PBCBoundaryTest, TerminalReadoutPreservesPhysicalWiresAndJointEffects) {
  // The first case used to read wire 0 twice. The second moves each observable
  // entirely off its source wire, requiring an identity operand in PBC.
  for (bool swap : {false, true}) {
    std::string gates = "  %c:2 = quake.x [%a] %b : (!quake.wire, !quake.wire) "
                        "-> (!quake.wire, !quake.wire)\n";
    if (swap)
      gates += "  %d:2 = quake.x [%c#1] %c#0 : (!quake.wire, !quake.wire) -> "
               "(!quake.wire, !quake.wire)\n"
               "  %e:2 = quake.x [%d#1] %d#0 : (!quake.wire, !quake.wire) -> "
               "(!quake.wire, !quake.wire)\n";
    std::string last = swap ? "%e" : "%c";
    auto module = parseSourceString<ModuleOp>(
        "func.func @test(%a: !quake.wire, %b: !quake.wire) -> (!quake.measure, "
        "!quake.measure) {\n" +
            gates + "  %m, %q1 = quake.mz " + last +
            "#1 name \"right\" : (!quake.wire) -> (!quake.measure, "
            "!quake.wire)\n"
            "  %n, %q0 = quake.mz " +
            last +
            "#0 name \"left\" : (!quake.wire) -> (!quake.measure, "
            "!quake.wire)\n"
            "  quake.sink %q0 : !quake.wire\n"
            "  quake.sink %q1 : !quake.wire\n"
            "  return %m, %n : !quake.measure, !quake.measure\n}",
        &context);
    ASSERT_TRUE(module);
    OwningOpRef<ModuleOp> original(cast<ModuleOp>(module->clone()));
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeToPBC());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeLowerPBC());
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto before = original->lookupSymbol<func::FuncOp>("test");
    auto after = module->lookupSymbol<func::FuncOp>("test");
    for (unsigned outcome = 0; outcome < 4; ++outcome)
      for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j) {
          auto a = runGadget(before, i, outcome),
               b = runGadget(before, j, outcome);
          auto c = runGadget(after, i, outcome),
               d = runGadget(after, j, outcome);
          Complex lhs = 0, rhs = 0;
          for (unsigned k = 0; k < 4; ++k) {
            lhs += std::conj(a[k]) * b[k];
            rhs += std::conj(c[k]) * d[k];
          }
          EXPECT_LT(std::abs(lhs - rhs), 1e-12);
        }
    DenseMap<Value, unsigned> ids;
    for (auto [i, arg] : llvm::enumerate(after.getArguments()))
      ids[arg] = i;
    unsigned count = 0;
    for (Operation &op : after.front()) {
      if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
        EXPECT_EQ(count, 0u) << "gate emitted after measurement";
        SmallVector<Value> inputs(gate.getControls());
        llvm::append_range(inputs, gate.getTargets());
        for (auto [in, out] : llvm::zip(inputs, gate.getWires()))
          ids[out] = ids.lookup(in);
      } else if (auto meas = dyn_cast<cudaq::quake::MzOp>(op)) {
        EXPECT_EQ(ids.lookup(meas.getTargets()[0]), count == 0 ? 1u : 0u);
        EXPECT_EQ(*meas.getRegisterName(), count == 0 ? "right" : "left");
        ids[meas.getWires()[0]] = ids.lookup(meas.getTargets()[0]);
        ++count;
      }
    }
    EXPECT_EQ(count, 2u);
  }
}

TEST_F(PBCBoundaryTest, ConflictingReadoutSlotsFailWithoutMutation) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a: !quake.wire, %b: !quake.wire) -> (!quake.measure, !quake.measure) {
      %m, %r:2 = quake.mpp %a, %b to "ZZ" : (!quake.wire, !quake.wire) -> (!quake.measure, !quake.wire, !quake.wire)
      %n, %s = quake.mpp %r#0 to "Z" : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %s : !quake.wire
      quake.sink %r#1 : !quake.wire
      return %m, %n : !quake.measure, !quake.measure
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  unchanged(*module, cudaq::opt::createQuakeLowerPBC(), false,
            "PBC readouts require unsupported classical remapping");
}

TEST_F(PBCBoundaryTest, JointReadoutHandlesSignedOverlappingAndIdentityAxes) {
  for (auto words : {std::pair{"XX", "YY"},
                     {"XZ", "ZX"},
                     {"YI", "IY"},
                     {"ZI", "IZ"},
                     {"IZ", "ZI"}})
    for (unsigned signs = 0; signs < 4; ++signs) {
      // The second readout belongs to b, so spell its word in b,a order.
      std::string second = words.second;
      std::reverse(second.begin(), second.end());
      auto module = parseSourceString<ModuleOp>(
          "func.func @test(%a: !quake.wire, %b: !quake.wire) -> "
          "(!quake.measure, !quake.measure) {\n"
          "  %m, %r:2 = quake.mpp %a, %b to \"" +
              std::string(words.first) +
              "\" : (!quake.wire, !quake.wire) -> (!quake.measure, "
              "!quake.wire, !quake.wire) {negative = " +
              (signs & 1 ? "true" : "false") +
              "}\n"
              "  %n, %s:2 = quake.mpp %r#1, %r#0 to \"" +
              second +
              "\" : (!quake.wire, !quake.wire) -> (!quake.measure, "
              "!quake.wire, !quake.wire) {negative = " +
              (signs & 2 ? "true" : "false") +
              "}\n"
              "  quake.sink %s#0 : !quake.wire\n"
              "  quake.sink %s#1 : !quake.wire\n"
              "  return %m, %n : !quake.measure, !quake.measure\n}",
          &context);
      ASSERT_TRUE(module);
      OwningOpRef<ModuleOp> original(cast<ModuleOp>(module->clone()));
      PassManager pm(&context);
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeLowerPBC());
      ASSERT_TRUE(succeeded(pm.run(*module)));
      auto before = original->lookupSymbol<func::FuncOp>("test");
      auto after = module->lookupSymbol<func::FuncOp>("test");
      for (unsigned outcome = 0; outcome < 4; ++outcome)
        for (unsigned i = 0; i < 4; ++i)
          for (unsigned j = 0; j < 4; ++j) {
            auto a = runGadget(before, i, outcome),
                 b = runGadget(before, j, outcome);
            auto c = runGadget(after, i, outcome),
                 d = runGadget(after, j, outcome);
            Complex lhs = 0, rhs = 0;
            for (unsigned k = 0; k < 4; ++k) {
              lhs += std::conj(a[k]) * b[k];
              rhs += std::conj(c[k]) * d[k];
            }
            EXPECT_LT(std::abs(lhs - rhs), 1e-12)
                << words.first << " " << words.second << " signs=" << signs;
          }
    }
}

TEST_F(PBCBoundaryTest, UnknownMetadataNeverDisappears) {
  for (StringRef kind :
       {"function", "argument", "result", "gate", "measurement", "sink",
        "classical", "return", "disabled"}) {
    SCOPED_TRACE(kind.str());
    auto module = parseSourceString<ModuleOp>(pbcInput, &context);
    ASSERT_TRUE(module);
    auto f = module->lookupSymbol<func::FuncOp>("test");
    Builder b(&context);
    if (kind == "function")
      f->setAttr("audit.semantic_contract", b.getUnitAttr());
    else if (kind == "argument")
      f.setArgAttr(0, "audit.semantic_contract", b.getUnitAttr());
    else if (kind == "result")
      f.setResultAttr(0, "audit.semantic_contract", b.getUnitAttr());
    else if (kind == "disabled")
      f->setAttr("quake.noOptimization", b.getUnitAttr());
    else
      f.walk([&](Operation *op) {
        if ((kind == "gate" && isa<cudaq::quake::ExpPauliOp>(op)) ||
            (kind == "measurement" && isa<cudaq::quake::MppOp>(op)) ||
            (kind == "sink" && isa<cudaq::quake::SinkOp>(op)) ||
            (kind == "classical" && isa<arith::ConstantOp>(op)) ||
            (kind == "return" && isa<func::ReturnOp>(op)))
          op->setAttr("audit.semantic_contract", b.getUnitAttr());
      });
    unchanged(*module, cudaq::opt::createQuakeTryPBC(), true);
    StringRef expected = kind == "disabled" ? "respects quake.noOptimization"
                         : kind == "argument" || kind == "result"
                             ? "argument/result attributes"
                             : "unsupported PBC attribute";
    unchanged(*module, cudaq::opt::createQuakeToPBC(), false, expected);
    unchanged(*module, cudaq::opt::createQuakeVerifyPBC(), false, expected);
    unchanged(*module, cudaq::opt::createQuakeOptimizePBC(), false, expected);
    unchanged(*module, cudaq::opt::createQuakeLowerPBC(), false, expected);
  }
}

TEST_F(PBCBoundaryTest, LateUnknownOperationRejectsTheWholeFunction) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q: !quake.wire) -> !quake.measure {
      %h = quake.h %q : (!quake.wire) -> !quake.wire
      %u = "audit.unknown"(%h) : (!quake.wire) -> !quake.wire
      %m, %end = quake.mz %u : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %end : !quake.wire
      return %m : !quake.measure
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  cudaq::opt::QuakeTryPBCOptions options;
  options.emitRemarks = true;
  unchanged(*module, cudaq::opt::createQuakeTryPBC(options), true,
            "unsupported operation in PBC");
  unchanged(*module, cudaq::opt::createQuakeToPBC(), false,
            "unsupported operation in PBC");
}

TEST_F(PBCBoundaryTest, EscapingQuantumOutputSkipsCleanly) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q: !quake.wire) -> !quake.wire {
      %h = quake.h %q : (!quake.wire) -> !quake.wire
      return %h : !quake.wire
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  unchanged(*module, cudaq::opt::createQuakeTryPBC(), true);
  unchanged(*module, cudaq::opt::createQuakeToPBC(), false,
            "escaping quantum outputs");
}

TEST_F(PBCBoundaryTest, UnsupportedRegionsAndReferencesSkipCleanly) {
  for (
      StringRef source : {
          R"mlir(func.func @test(%q: !quake.wire, %condition: i1) -> !quake.measure {
             cc.if(%condition) {} else {}
             %m, %end = quake.mz %q : (!quake.wire) -> (!quake.measure, !quake.wire)
             quake.sink %end : !quake.wire
             return %m : !quake.measure
           })mlir",
          R"mlir(func.func @test(%q: !quake.ref) {
             quake.h %q : (!quake.ref) -> ()
             return
           })mlir"}) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    unchanged(*module, cudaq::opt::createQuakeTryPBC(), true);
    unchanged(*module, cudaq::opt::createQuakeToPBC(), false);
  }
}

TEST_F(PBCBoundaryTest, DynamicWidthRoundTripRetainsHighQubitIdentity) {
  for (unsigned width : {65u, 118u, 433u, 1025u}) {
    std::ostringstream source;
    source << "func.func @test(";
    for (unsigned i = 0; i < width; ++i)
      source << (i ? ", " : "") << "%q" << i << ": !quake.wire";
    source << ") -> !quake.measure {\n"
           << "%a = quake.t %q" << width - 1
           << " : (!quake.wire) -> !quake.wire\n"
           << "%b = quake.t %a : (!quake.wire) -> !quake.wire\n"
           << "%m, %end = quake.mx %b : (!quake.wire) -> (!quake.measure, "
              "!quake.wire)\n";
    for (unsigned i = 0; i + 1 < width; ++i)
      source << "quake.sink %q" << i << " : !quake.wire\n";
    source << "quake.sink %end : !quake.wire\nreturn %m : !quake.measure\n}";
    auto module = parseSourceString<ModuleOp>(source.str(), &context);
    ASSERT_TRUE(module);
    cudaq::opt::QuakeTryPBCOptions options;
    options.enableGraphSynthesis = false;
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto f = module->lookupSymbol<func::FuncOp>("test");
    ASSERT_TRUE(f->hasAttr("pbc.t_after")) << width;
    EXPECT_EQ(f->getAttrOfType<IntegerAttr>("pbc.t_after").getInt(), 0);
    unsigned sinks = 0, gates = 0;
    DenseMap<Value, unsigned> ids;
    for (auto [i, arg] : llvm::enumerate(f.getArguments()))
      ids[arg] = i;
    for (Operation &op : f.front()) {
      if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
        ++gates;
        ASSERT_EQ(gate.getTargets().size(), 1u);
        EXPECT_EQ(ids.lookup(gate.getTargets()[0]), width - 1);
        ids[gate.getWires()[0]] = ids.lookup(gate.getTargets()[0]);
      } else if (auto measurement =
                     dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
        EXPECT_EQ(ids.lookup(measurement.getTargets()[0]), width - 1);
        ids[measurement.getWires()[0]] =
            ids.lookup(measurement.getTargets()[0]);
      } else if (isa<cudaq::quake::SinkOp>(op))
        ++sinks;
    }
    EXPECT_EQ(sinks, width);
    EXPECT_GT(gates, 0u);
  }
}

TEST_F(PBCBoundaryTest, LimitsAndDisabledStrategiesLeaveNoStatistics) {
  for (unsigned mode = 0; mode < 5; ++mode) {
    auto module = parseSourceString<ModuleOp>(pbcInput, &context);
    ASSERT_TRUE(module);
    cudaq::opt::QuakeTryPBCOptions optional;
    cudaq::opt::QuakeOptimizePBCOptions strict;
    if (mode == 0)
      optional.maxWork = strict.maxWork = 1;
    else if (mode == 1)
      optional.maxGates = strict.maxGates = 1;
    else if (mode == 2) {
      optional.enableTCount = strict.enableTCount = false;
      optional.enableGraphSynthesis = strict.enableGraphSynthesis = false;
    } else
      // Seven admits this input but rejects the eight-operation output.
      optional.maxOperations = mode == 3 ? 1 : 7;
    unchanged(*module, cudaq::opt::createQuakeTryPBC(optional), true);
    if (mode < 3)
      unchanged(*module, cudaq::opt::createQuakeOptimizePBC(strict), true);
  }
}

TEST_F(PBCBoundaryTest, NoImprovementIsAnIdentityRewrite) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q: !quake.wire) -> !quake.measure {
      %m, %end = quake.mpp %q to "Z" : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %end : !quake.wire
      return %m : !quake.measure
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  unchanged(*module, cudaq::opt::createQuakeTryPBC(), true);
  unchanged(*module, cudaq::opt::createQuakeOptimizePBC(), true);
}

TEST_F(PBCBoundaryTest, InvalidIRIsNeverAQuietSkip) {
  auto module = parseSourceString<ModuleOp>(pbcInput, &context);
  ASSERT_TRUE(module);
  Builder b(&context);
  module->walk([&](cudaq::quake::ExpPauliOp op) {
    op->setAttr("quake.pbc_angle", b.getDenseI64ArrayAttr({1, 0}));
  });
  unchanged(*module, cudaq::opt::createQuakeTryPBC(), false,
            "pbc_angle requires two i64 coefficients");
}

TEST_F(PBCBoundaryTest, InvalidOptionsFailEvenOnIneligibleInput) {
  auto module = parseSourceString<ModuleOp>(primitiveInput, &context);
  ASSERT_TRUE(module);
  module->lookupSymbol<func::FuncOp>("test")->setAttr("audit.unknown",
                                                      UnitAttr::get(&context));
  cudaq::opt::QuakeTryPBCOptions options;
  options.maxWork = 0;
  unchanged(*module, cudaq::opt::createQuakeTryPBC(options), false,
            "requires positive limits");
  options.maxWork = 1;
  options.objective = "unknown";
  unchanged(*module, cudaq::opt::createQuakeTryPBC(options), false,
            "PBC objective must be");
}

TEST_F(PBCBoundaryTest, ExactLoweringFailureDoesNotCommitConversion) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q: !quake.wire) -> !quake.measure {
      %tangle = arith.constant -0.39269908169872414 : f64
      %angle = arith.constant 0.44879895051282759 : f64
      %t = quake.exp_pauli (%tangle) %q to "Z" : (f64, !quake.wire) -> !quake.wire {quake.pbc_angle = array<i64: -1, 8>}
      %u = quake.exp_pauli (%tangle) %t to "Z" : (f64, !quake.wire) -> !quake.wire {quake.pbc_angle = array<i64: -1, 8>}
      %r = quake.exp_pauli (%angle) %u to "X" : (f64, !quake.wire) -> !quake.wire {quake.pbc_angle = array<i64: 1, 7>}
      %m, %end = quake.mpp %r to "Z" : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %end : !quake.wire
      return %m : !quake.measure
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  cudaq::opt::QuakeTryPBCOptions options;
  options.emitRemarks = true;
  unchanged(*module, cudaq::opt::createQuakeTryPBC(options), true,
            "exact-angle contract");
  unchanged(*module, cudaq::opt::createQuakeLowerPBC(), false,
            "explicit allow-approximate-angles");
}

TEST_F(PBCBoundaryTest,
       OptionalRoundTripImprovesAndPreservesEveryOutcomeEffect) {
  auto original = parseSourceString<ModuleOp>(primitiveInput, &context);
  ASSERT_TRUE(original);
  auto function = original->lookupSymbol<func::FuncOp>("test");
  function->setAttr("cudaq-kernel", UnitAttr::get(&context));
  function->setAttr("cudaq-entrypoint", UnitAttr::get(&context));
  OwningOpRef<ModuleOp> optimized(cast<ModuleOp>(original->clone()));
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC());
  ASSERT_TRUE(succeeded(pm.run(*optimized)));
  auto lhs = original->lookupSymbol<func::FuncOp>("test");
  auto rhs = optimized->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(rhs->hasAttr("cudaq-kernel"));
  EXPECT_TRUE(rhs->hasAttr("cudaq-entrypoint"));
  ASSERT_TRUE(rhs->hasAttr("pbc.t_after"));
  EXPECT_EQ(rhs->getAttrOfType<IntegerAttr>("pbc.t_after").getInt(), 0);
  rhs.walk([&](Operation *op) {
    EXPECT_FALSE(
        (isa<cudaq::quake::MppOp, cudaq::quake::ExpPauliOp, cudaq::quake::TOp>(
            op)));
    if (isa<cudaq::quake::MeasurementInterface>(op))
      EXPECT_EQ(op->getAttrOfType<StringAttr>("registerName").getValue(),
                "answer");
  });
  for (unsigned outcome = 0; outcome < 2; ++outcome)
    for (unsigned i = 0; i < 2; ++i)
      for (unsigned j = 0; j < 2; ++j) {
        auto a = runGadget(lhs, i, outcome), b = runGadget(lhs, j, outcome);
        auto c = runGadget(rhs, i, outcome), d = runGadget(rhs, j, outcome);
        Complex before = 0, after = 0;
        for (unsigned row = 0; row < 2; ++row) {
          before += std::conj(a[row]) * b[row];
          after += std::conj(c[row]) * d[row];
        }
        EXPECT_LT(std::abs(before - after), 1e-12);
      }
}

namespace {
llvm::json::Value readTelemetry(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  EXPECT_TRUE(bool(buffer));
  if (!buffer)
    return nullptr;
  auto value = llvm::json::parse((*buffer)->getBuffer());
  EXPECT_TRUE(bool(value));
  if (!value) {
    llvm::consumeError(value.takeError());
    return nullptr;
  }
  return std::move(*value);
}
} // namespace

TEST_F(PBCBoundaryTest, TelemetryReportsCommittedMetricsWithoutChangingIR) {
  auto original = parseSourceString<ModuleOp>(primitiveInput, &context);
  ASSERT_TRUE(original);
  OwningOpRef<ModuleOp> observed(cast<ModuleOp>(original->clone()));
  llvm::SmallString<128> path;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("cudaq-pbc-telemetry", "jsonl", path));
  llvm::FileRemover cleanup(path);
  cudaq::opt::QuakeTryPBCOptions options;
  options.telemetry = "off";
  PassManager baseline(&context);
  baseline.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
  ASSERT_TRUE(succeeded(baseline.run(*original)));
  options.telemetry = path.str().str();
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
  ASSERT_TRUE(succeeded(pm.run(*observed)));
  EXPECT_EQ(printed(*original), printed(*observed));
  auto report = readTelemetry(path);
  auto *object = report.getAsObject();
  ASSERT_TRUE(object);
  EXPECT_EQ(object->getString("outcome"), "committed");
  EXPECT_EQ(object->getInteger("schema_version"), 1);
  EXPECT_GT(object->getInteger("elapsed_ns").value_or(0), 0);
  auto *metrics = object->getObject("mechanical_estimates");
  ASSERT_TRUE(metrics);
  EXPECT_EQ(metrics->getObject("input")->getInteger("t_count"), 2);
  EXPECT_EQ(metrics->getObject("output")->getInteger("t_count"), 0);
  auto *ir = object->getObject("observed_ir");
  ASSERT_TRUE(ir);
  EXPECT_EQ(ir->getObject("input")->getInteger("explicit_t_count"), 2);
  EXPECT_EQ(ir->getObject("output")->getInteger("explicit_t_count"), 0);
  EXPECT_TRUE(object->getBoolean("lineage_complete").value_or(false));
  auto *events = object->getArray("events");
  ASSERT_TRUE(events);
  EXPECT_FALSE(events->empty());
  bool merge = false, graph = false;
  for (const auto &event : *events) {
    auto *e = event.getAsObject();
    ASSERT_TRUE(e);
    auto method = e->getString("method");
    ASSERT_TRUE(method);
    merge |= method->starts_with("merge.");
    graph |= *method == "graph-synthesis";
    EXPECT_GE(e->getInteger("elapsed_ns").value_or(0),
              e->getInteger("verification_ns").value_or(0));
  }
  EXPECT_TRUE(merge);
  EXPECT_TRUE(graph);
}

TEST_F(PBCBoundaryTest, TelemetryReportsSkipsFailuresAndTruncation) {
  for (unsigned mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    auto module = parseSourceString<ModuleOp>(primitiveInput, &context);
    ASSERT_TRUE(module);
    llvm::SmallString<128> path;
    ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("cudaq-pbc-telemetry",
                                                    "jsonl", path));
    llvm::FileRemover cleanup(path);
    cudaq::opt::QuakeTryPBCOptions options;
    options.telemetry = path.str().str();
    if (mode == 0)
      module->lookupSymbol<func::FuncOp>("test")->setAttr(
          "audit.unknown", UnitAttr::get(&context));
    else if (mode == 1)
      options.maxWork = 0;
    else if (mode == 2)
      options.maxWork = 1;
    else
      options.telemetryMaxEvents = 0;
    if (mode < 3)
      unchanged(*module, cudaq::opt::createQuakeTryPBC(options), mode != 1);
    else {
      PassManager pm(&context);
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
      ASSERT_TRUE(succeeded(pm.run(*module)));
    }
    auto report = readTelemetry(path);
    auto *object = report.getAsObject();
    ASSERT_TRUE(object);
    EXPECT_EQ(object->getString("outcome"), mode == 1   ? "failed"
                                            : mode == 3 ? "committed"
                                                        : "skipped");
    if (mode == 3) {
      EXPECT_TRUE(object->getArray("events")->empty());
      EXPECT_GT(object->getInteger("dropped_events").value_or(0), 0);
      EXPECT_FALSE(object->getBoolean("lineage_complete").value_or(true));
      EXPECT_GT(
          object->getObject("portfolio")->getInteger("candidates").value_or(0),
          0);
    }
  }
}

TEST_F(PBCBoundaryTest, TelemetryWriteFailureDoesNotFailOptimization) {
  auto module = parseSourceString<ModuleOp>(primitiveInput, &context);
  ASSERT_TRUE(module);
  cudaq::opt::QuakeTryPBCOptions options;
  // /dev/null is a file, so this cannot accidentally become a writable path.
  options.telemetry = "/dev/null/pbc.jsonl";
  bool warned = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
    warned |= d.getSeverity() == DiagnosticSeverity::Warning;
    return success();
  });
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));
  EXPECT_TRUE(warned);
  EXPECT_TRUE(
      module->lookupSymbol<func::FuncOp>("test")->hasAttr("pbc.t_after"));
}

TEST_F(PBCBoundaryTest, TelemetryDistinguishesCXAndCZAndUnknownTCost) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a: !quake.wire, %b: !quake.wire, %theta: f64) -> (!quake.measure, !quake.measure) {
      %cx:2 = quake.x [%a] %b : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %cz:2 = quake.z [%cx#0] %cx#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %t = quake.t %cz#0 : (!quake.wire) -> !quake.wire
      %u = quake.t %t : (!quake.wire) -> !quake.wire
      %r = quake.rx (%theta) %u : (f64, !quake.wire) -> !quake.wire
      %m0, %q0 = quake.mx %r : (!quake.wire) -> (!quake.measure, !quake.wire)
      %m1, %q1 = quake.mz %cz#1 : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %q0 : !quake.wire
      quake.sink %q1 : !quake.wire
      return %m0, %m1 : !quake.measure, !quake.measure
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  llvm::SmallString<128> path;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("cudaq-pbc-telemetry", "jsonl", path));
  llvm::FileRemover cleanup(path);
  cudaq::opt::QuakeTryPBCOptions options;
  options.telemetry = path.str().str();
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));
  auto report = readTelemetry(path);
  auto *object = report.getAsObject();
  ASSERT_TRUE(object);
  auto *input = object->getObject("observed_ir")->getObject("input");
  EXPECT_EQ(input->getInteger("cx_count"), 1);
  EXPECT_EQ(input->getInteger("cz_count"), 1);
  EXPECT_EQ(
      input->getObject("gate_histogram")->getInteger("quake.x.controls_1"), 1);
  auto *estimate =
      object->getObject("mechanical_estimates")->getObject("input");
  EXPECT_EQ(estimate->getInteger("unsynthesized_rotations"), 1);
  EXPECT_FALSE(estimate->getBoolean("t_count_complete").value_or(true));
  EXPECT_EQ(estimate->getInteger("t_count"), 2);
}

TEST_F(PBCBoundaryTest, TotalGateNetworkPreservesEveryJointOutcomeEffect) {
  auto original = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a: !quake.wire, %b: !quake.wire) -> (!quake.measure, !quake.measure) {
      %h = quake.h %b : (!quake.wire) -> !quake.wire
      %c:2 = quake.x [%a] %h : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %h2 = quake.h %c#1 : (!quake.wire) -> !quake.wire
      %x = quake.x %c#0 : (!quake.wire) -> !quake.wire
      %t = quake.t %x : (!quake.wire) -> !quake.wire
      %x2 = quake.x %t : (!quake.wire) -> !quake.wire
      %m0, %q0 = quake.mx %x2 name "first" : (!quake.wire) -> (!quake.measure, !quake.wire)
      %m1, %q1 = quake.my %h2 name "second" : (!quake.wire) -> (!quake.measure, !quake.wire)
      quake.sink %q0 : !quake.wire
      quake.sink %q1 : !quake.wire
      return %m0, %m1 : !quake.measure, !quake.measure
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(original);
  OwningOpRef<ModuleOp> optimized(cast<ModuleOp>(original->clone()));
  cudaq::opt::QuakeTryPBCOptions options;
  options.objective = "total-gates";
  options.enableTCount = options.enableGraphSynthesis = false;
  options.enableGateCount = true;
  options.preserveCliffordNetwork = true;
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
  ASSERT_TRUE(succeeded(pm.run(*optimized)));
  auto lhs = original->lookupSymbol<func::FuncOp>("test");
  auto rhs = optimized->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(rhs->hasAttr("pbc.t_after"));
  EXPECT_EQ(rhs->getAttrOfType<IntegerAttr>("pbc.t_after").getInt(), 1);
  for (unsigned outcome = 0; outcome < 4; ++outcome)
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; j < 4; ++j) {
        auto a = runGadget(lhs, i, outcome), b = runGadget(lhs, j, outcome);
        auto c = runGadget(rhs, i, outcome), d = runGadget(rhs, j, outcome);
        Complex before = 0, after = 0;
        for (unsigned row = 0; row < 4; ++row) {
          before += std::conj(a[row]) * b[row];
          after += std::conj(c[row]) * d[row];
        }
        EXPECT_LT(std::abs(before - after), 1e-12);
      }
}

TEST_F(PBCBoundaryTest, MeasurementHandleSurvivesStrictAndOptionalRoundTrips) {
  std::string source = primitiveInput.str();
  const std::string oldType = "!quake.measure";
  for (size_t pos = 0;
       (pos = source.find(oldType, pos)) != std::string::npos;) {
    source.replace(pos, oldType.size(), "!cc.measure_handle");
    pos += std::string("!cc.measure_handle").size();
  }
  for (bool optional : {false, true}) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    PassManager pm(&context);
    if (optional) {
      cudaq::opt::QuakeTryPBCOptions options;
      options.preserveCliffordNetwork = true;
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeTryPBC(options));
    } else {
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeToPBC());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeVerifyPBC());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeLowerPBC());
    }
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto function = module->lookupSymbol<func::FuncOp>("test");
    EXPECT_TRUE(
        isa<cudaq::cc::MeasureHandleType>(function.getResultTypes()[0]));
    function.walk([&](cudaq::quake::MeasurementInterface measurement) {
      EXPECT_TRUE(isa<cudaq::cc::MeasureHandleType>(
          measurement->getResult(0).getType()));
    });
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}
