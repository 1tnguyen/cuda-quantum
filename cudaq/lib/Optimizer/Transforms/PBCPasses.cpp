/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief MLIR passes for verified Pauli-based computation in Quake.
///
/// Implements conversion, verification, optimization, and lowering using scalar
/// Quake wires, Pauli rotations, and terminal joint measurements. Imports
/// validate the complete supported function before rewriting; native strategies
/// operate under the explicit measurement and exact-angle contracts.
//===----------------------------------------------------------------------===//

#include "PBCTelemetry.h"
#include "PassDetails.h"
#include "Pauli/TCountInternal.h"
#include "cudaq/Optimizer/Pauli/Optimizer.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <bit>
#include <limits>
#include <numbers>
#include <tuple>

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKESYNTHESIZEPBC
#define GEN_PASS_DEF_QUAKETOPBC
#define GEN_PASS_DEF_QUAKEVERIFYPBC
#define GEN_PASS_DEF_QUAKEOPTIMIZEPBC
#define GEN_PASS_DEF_QUAKELOWERPBC
#define GEN_PASS_DEF_QUAKETRYPBC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;
using cudaq::pauli::QubitSet;
namespace {
// Exact coefficients are stored separately from the floating-point exponent
// operand so a PBC round trip does not lose rational-angle information.
constexpr StringLiteral exactAngleAttr = "quake.pbc_angle";
/// Opaque native angle backed by a scaled MLIR value.
struct Symbol {
  /// Original SSA angle operand, preserved without guessing an exact rational.
  Value value;
  /// Multiplicative conversion from the source convention to the native angle.
  double scale;
};
/// Terminal measurement whose classical identity must survive rewriting.
struct Measurement {
  /// Source measurement operation, used for reconstruction metadata/location.
  Operation *op;
  /// Classical result whose users must receive the replacement outcome.
  Value outcome;
  /// Optional measurement register name preserved on emission.
  StringAttr name;
  /// Physical readout identity. In PBC this is the first MPP operand, which
  /// may carry I when frame absorption moves the observable to other wires.
  unsigned readout;
};
/// Complete validated import and rewrite bookkeeping, built before IR mutation.
struct Plan {
  /// Initial scalar wires, ordered by their native wire indices.
  SmallVector<Value> inputs;
  /// Opaque angle table; native nonzero symbol IDs are one-based indices here.
  SmallVector<Symbol> symbols;
  /// Source operations to erase in reverse order after replacement emission.
  SmallVector<Operation *> erase;
  /// Terminal sink operations paired with the native wire index to reconnect.
  SmallVector<std::pair<Operation *, unsigned>> sinks;
  /// Original measurements in outcome order, aligned with observables.
  SmallVector<Measurement> measurements;
  /// Imported unitary operations in native circuit order.
  std::vector<cudaq::pauli::Gate> gates;
  /// Signed terminal Pauli observables in original measurement order.
  std::vector<cudaq::pauli::Pauli> observables;
  /// Replacement insertion anchor, assigned by a successful read.
  Operation *insertion = nullptr;
};

/// Unsupported valid IR is a clean fallback for the optional pass. Invalid IR
/// is always an error; neither category permits partial conversion.
enum class Eligibility { Supported, Unsupported, Invalid };
/// Import eligibility and deferred diagnostic, independent of any IR rewrite.
/// Callers choose whether unsupported input is a clean skip or a strict error;
/// invalid input has already been diagnosed by the MLIR verifier.
struct ImportResult {
  /// Classification of the inspected input; initially no rejection is recorded.
  Eligibility status = Eligibility::Supported;
  /// Borrowed diagnostic anchor, required for Unsupported and valid until the
  /// caller handles this result. Null is permitted for a supported import.
  Operation *operation = nullptr;
  /// Owned explanation of an unsupported construct; empty when no rejection
  /// was recorded or the verifier already diagnosed invalid IR.
  std::string reason;

  /// Record unsupported input without emitting a diagnostic or changing IR.
  /// Return failure so import helpers can propagate the rejection immediately.
  LogicalResult reject(Operation *op, const Twine &message) {
    status = Eligibility::Unsupported;
    operation = op;
    reason = message.str();
    return failure();
  }
  /// Apply strict handling: diagnose unsupported input and return failure for
  /// either rejection category, without duplicating verifier diagnostics.
  LogicalResult diagnose() const {
    if (status == Eligibility::Supported)
      return success();
    // The MLIR verifier has already diagnosed invalid input.
    if (status == Eligibility::Unsupported)
      operation->emitError(reason);
    return failure();
  }
};

/// Attributes can change semantics even on recognized operations. Permit only
/// attributes whose meaning is handled here, rather than silently copying or
/// discarding an extension the importer does not understand.
LogicalResult checkAttributes(Operation *op, ArrayRef<StringRef> allowed,
                              ImportResult &result) {
  for (NamedAttribute attr : op->getAttrs())
    if (!llvm::is_contained(allowed, attr.getName().getValue()))
      return result.reject(op, "unsupported PBC attribute '" +
                                   attr.getName().getValue() + "'");
  return success();
}

LogicalResult checkFunctionAttributes(func::FuncOp function,
                                      ImportResult &result) {
  if (function->hasAttr("quake.noOptimization"))
    return result.reject(function, "PBC respects quake.noOptimization");
  if (failed(checkAttributes(
          function,
          {"sym_name", "function_type", "sym_visibility", "arg_attrs",
           "res_attrs", "cudaq-entrypoint", "cudaq-kernel", "quake.pbc",
           "pbc.t_before", "pbc.t_after", "pbc.two_qubit_before",
           "pbc.two_qubit_after", "pbc.depth_before", "pbc.depth_after",
           "pbc.work", "pbc.budget_exhaustions"},
          result)))
    return failure();
  // Argument/result contracts also require explicit support. Empty dictionaries
  // carry no additional semantics and are retained with the function.
  for (StringRef name : {"arg_attrs", "res_attrs"})
    if (auto attrs = function->getAttrOfType<ArrayAttr>(name))
      for (Attribute attr : attrs)
        if (!cast<DictionaryAttr>(attr).empty())
          return result.reject(
              function, "PBC does not support argument/result attributes");
  return success();
}

/// Reduce a rational coefficient before checking that it fits the native angle.
/// Callers use widened intermediates for sign and factor-of-two conversions.
std::optional<cudaq::pauli::Angle> rational(__int128 n, __int128 d) {
  __int128 a = n < 0 ? -n : n, b = d;
  while (b) {
    __int128 r = a % b;
    a = b;
    b = r;
  }
  if (a) {
    n /= a;
    d /= a;
  }
  if (n < INT64_MIN || n > INT64_MAX || d <= 0 || d > INT64_MAX)
    return std::nullopt;
  return cudaq::pauli::Angle{int64_t(n), int64_t(d)};
}

/// Intern an SSA angle and its unit conversion; keep adjoint negation separate.
/// Reusing IDs lets native optimizers recognize the same opaque angle without
/// evaluating it or inferring an exact rational from a floating-point value.
cudaq::pauli::Angle symbol(Plan &plan, Value value, double scale,
                           bool adjoint) {
  auto it = llvm::find_if(plan.symbols, [&](Symbol s) {
    return s.value == value && s.scale == scale;
  });
  unsigned index = std::distance(plan.symbols.begin(), it);
  if (it == plan.symbols.end())
    plan.symbols.push_back({value, scale});
  return {0, 1, index + 1, adjoint};
}

/// Import a supported function into a native plan without changing its IR.
/// Strict mode accepts only PBC rotations and joint measurements; conversion
/// mode also accepts the explicitly handled primitive gates and measurements.
/// Both modes check the whole function, including SSA ownership and terminal
/// measurement ordering. A partially built plan must be discarded on failure.
LogicalResult readSupported(func::FuncOp function, bool strict, Plan &plan,
                            ImportResult &result) {
  if (failed(checkFunctionAttributes(function, result)))
    return failure();
  if (function.isExternal())
    return success();
  if (!llvm::hasSingleElement(function.getBody()))
    return result.reject(function,
                         "PBC requires a single block without control flow");
  auto &block = function.front();
  // Native indices stay fixed as each Quake operation produces a new SSA wire.
  // `identities` maps all encountered owners; `current` rejects stale owners.
  DenseMap<Value, unsigned> identities;
  SmallVector<Value> current;
  // Once measurement or disposal starts, no further unitary work is accepted.
  bool terminal = false;
  cudaq::pauli::pbc::CommutingObservables observed;
  // Admit a new logical wire; native Pauli support grows with its index.
  auto add = [&](Value wire) -> LogicalResult {
    if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
      return result.reject(function, "PBC requires exclusive scalar SSA wires");
    identities[wire] = plan.inputs.size();
    plan.inputs.push_back(wire);
    current.push_back(wire);
    return success();
  };
  for (Value arg : block.getArguments()) {
    if (isa<cudaq::quake::WireType>(arg.getType())) {
      if (failed(add(arg)))
        return failure();
    } else if (!arg.getType().isIntOrFloat())
      return result.reject(
          function,
          "PBC accepts only scalar wires and scalar classical inputs");
  }
  // Resolve operands only when they are distinct, live, exclusively used wires.
  auto indices = [&](Operation *op, ValueRange values,
                     SmallVectorImpl<unsigned> &out) -> LogicalResult {
    llvm::SmallDenseSet<unsigned> seen;
    for (Value wire : values) {
      auto it = identities.find(wire);
      if (it == identities.end() || current[it->second] != wire ||
          !wire.hasOneUse() || !seen.insert(it->second).second)
        return result.reject(
            op, "PBC requires distinct current scalar wire owners");
      out.push_back(it->second);
    }
    return success();
  };
  // Transfer ownership in operand order to the operation's wire results.
  auto advance = [&](Operation *op, ArrayRef<unsigned> qs,
                     ValueRange results) -> LogicalResult {
    if (results.size() != qs.size())
      return result.reject(op,
                           "PBC requires one updated wire per quantum operand");
    for (auto [index, wire] : llvm::zip(qs, results)) {
      if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
        return result.reject(op,
                             "PBC requires exclusive scalar SSA wire results");
      current[index] = wire;
      identities[wire] = index;
    }
    return success();
  };
  for (Operation &op : block) {
    SmallVector<StringRef> allowed;
    if (isa<cudaq::quake::OperatorInterface>(op))
      allowed = {"operandSegmentSizes", "is_adj", "negated_qubit_controls"};
    if (isa<cudaq::quake::ExpPauliOp>(op))
      llvm::append_range(allowed,
                         ArrayRef<StringRef>{"pauliLiteral", exactAngleAttr});
    if (isa<cudaq::quake::MeasurementInterface>(op))
      allowed.push_back("registerName");
    if (isa<cudaq::quake::MppOp>(op))
      llvm::append_range(allowed, ArrayRef<StringRef>{"word", "negative"});
    // Registered, pure scalar arithmetic is retained verbatim. Its inherent
    // attributes remain meaningful; unknown extension attributes do not.
    bool arithmetic = op.getName().getDialectNamespace() == "arith" &&
                      op.getRegisteredInfo() && !op.getNumRegions() &&
                      isPure(&op) &&
                      llvm::all_of(op.getOperandTypes(),
                                   [](Type t) { return t.isIntOrFloat(); }) &&
                      llvm::all_of(op.getResultTypes(),
                                   [](Type t) { return t.isIntOrFloat(); });
    if (arithmetic)
      for (StringAttr name : op.getRegisteredInfo()->getAttributeNames())
        allowed.push_back(name.getValue());
    if (failed(checkAttributes(&op, allowed, result)))
      return failure();
    if (isa<cudaq::quake::BorrowWireOp, cudaq::quake::ReturnWireOp>(op))
      return result.reject(
          &op, "PBC cannot expose quantum state through a wire set");
    if (isa<cudaq::quake::NullWireOp>(op)) {
      if (terminal)
        return result.reject(
            &op, "PBC preparation must precede terminal measurements");
      if (failed(add(op.getResult(0))))
        return failure();
      continue;
    }
    if (isa<cudaq::quake::SinkOp>(op)) {
      terminal = true;
      if (!plan.insertion)
        plan.insertion = &op;
      SmallVector<unsigned> qs;
      if (op.getNumOperands() != 1 ||
          failed(indices(&op, op.getOperands(), qs)))
        return failure();
      plan.sinks.emplace_back(&op, qs[0]);
      current[qs[0]] = {};
      continue;
    }
    // Record each joint observable and its classical result separately.
    // Shared support is allowed, but all terminal observables must commute.
    if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
      if (!isa<cudaq::quake::MppOp>(op) &&
          (strict ||
           !isa<cudaq::quake::MxOp, cudaq::quake::MyOp, cudaq::quake::MzOp>(
               op)))
        return result.reject(&op,
                             "PBC form permits only quake.mpp measurements");
      terminal = true;
      if (!plan.insertion)
        plan.insertion = &op;
      SmallVector<unsigned> qs;
      if (failed(indices(&op, meas.getTargets(), qs)) ||
          failed(advance(&op, qs, meas.getWires())))
        return failure();
      if (!isa<cudaq::quake::MeasureType, cudaq::cc::MeasureHandleType>(
              op.getResult(0).getType()))
        return result.reject(
            &op, "PBC requires one scalar measurement outcome per observable");
      // Quake mx/my rotate to Z before measuring and leave a computational
      // basis state, unlike the nondestructive MPP instrument. They are
      // interchangeable only when the measured wire is immediately discarded.
      if (isa<cudaq::quake::MxOp, cudaq::quake::MyOp>(op) &&
          !llvm::all_of(meas.getWires(), [](Value wire) {
            return wire.hasOneUse() &&
                   isa<cudaq::quake::SinkOp>(*wire.getUsers().begin());
          }))
        return result.reject(
            &op, "PBC requires mx/my results to go directly to sink");
      cudaq::pauli::Pauli word;
      if (auto mpp = dyn_cast<cudaq::quake::MppOp>(op)) {
        word.sign = mpp.getNegative();
        for (auto [i, c] : llvm::enumerate(mpp.getWord())) {
          if (c == 'X' || c == 'Y')
            word.x |= QubitSet::single(qs[i], 1);
          if (c == 'Z' || c == 'Y')
            word.z |= QubitSet::single(qs[i], 1);
        }
      } else {
        if (qs.size() != 1)
          return result.reject(&op,
                               "PBC conversion requires scalar measurements");
        if (!isa<cudaq::quake::MzOp>(op))
          word.x = QubitSet::single(qs[0], 1);
        if (!isa<cudaq::quake::MxOp>(op))
          word.z = QubitSet::single(qs[0], 1);
      }
      if (!observed.insert(word))
        return result.reject(&op,
                             "PBC terminal observables must pairwise commute");
      plan.observables.push_back(word);
      plan.measurements.push_back({&op, op.getResult(0),
                                   op.getAttrOfType<StringAttr>("registerName"),
                                   qs.front()});
      plan.erase.push_back(&op);
      continue;
    }
    if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
      if (terminal)
        return result.reject(
            &op, "PBC rotations must precede all measurements and sinks");
      if (strict && !isa<cudaq::quake::ExpPauliOp>(op))
        return result.reject(
            &op, "PBC form permits only quake.exp_pauli unitary operations");
      SmallVector<Value> operands(gate.getControls());
      llvm::append_range(operands, gate.getTargets());
      SmallVector<unsigned> qs;
      if (failed(indices(&op, operands, qs)) ||
          failed(advance(&op, qs, gate.getWires())))
        return failure();
      if (qs.empty())
        return result.reject(&op, "PBC requires nonempty gate support");
      if (auto neg = gate.getNegatedControls();
          neg && llvm::is_contained(*neg, true))
        return result.reject(&op,
                             "PBC conversion does not accept negated controls");
      bool adj = op.hasAttr("is_adj");
      unsigned nc = gate.getControls().size();
      if (nc && (nc != 1 || qs.size() != 2 ||
                 !isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op)))
        return result.reject(
            &op, "PBC conversion only supports Clifford CX/CZ controls");
      if (auto exp = dyn_cast<cudaq::quake::ExpPauliOp>(op)) {
        if (nc || !exp.getPauliLiteral() || exp.getParameters().size() != 1 ||
            !exp.getParameters()[0].getType().isF64())
          return result.reject(&op,
                               "PBC rotations require a literal word and one "
                               "f64 exponent, without controls");
        cudaq::pauli::Pauli word;
        for (auto [i, c] : llvm::enumerate(*exp.getPauliLiteral())) {
          if (c == 'X' || c == 'Y')
            word.x |= QubitSet::single(qs[i], 1);
          if (c == 'Z' || c == 'Y')
            word.z |= QubitSet::single(qs[i], 1);
        }
        if (strict && !word.weight())
          return result.reject(&op,
                               "PBC does not retain identity-only rotations");
        // Quake exp_pauli uses exp(+i exponent P), while the native rotation
        // uses exp(-i angle P/2): angle = -2 * exponent. Only the explicit
        // exact-angle attribute supplies a rational; other operands stay
        // opaque.
        cudaq::pauli::Angle angle;
        if (auto exact = op.getAttrOfType<DenseI64ArrayAttr>(exactAngleAttr)) {
          auto converted =
              rational(-2 * __int128(exact[0]) * (adj ? -1 : 1), exact[1]);
          if (!converted)
            return result.reject(
                &op, "PBC rotation angle exceeds exact integer range");
          angle = *converted;
        } else
          angle = symbol(plan, exp.getParameters()[0], -2, adj);
        plan.gates.push_back(cudaq::pauli::Gate::rotation(word, angle));
      } else if (isa<cudaq::quake::RxOp, cudaq::quake::RyOp, cudaq::quake::RzOp,
                     cudaq::quake::PhaseOp>(op)) {
        if (qs.size() != 1 || gate.getParameters().size() != 1 ||
            !gate.getParameters()[0].getType().isF64())
          return result.reject(&op,
                               "PBC conversion requires scalar f64 rotations");
        auto angle = symbol(plan, gate.getParameters()[0], 1, adj);
        if (isa<cudaq::quake::PhaseOp>(op))
          plan.gates.push_back(cudaq::pauli::Gate::phase(angle));
        else {
          unsigned axis = isa<cudaq::quake::RxOp>(op)   ? 1
                          : isa<cudaq::quake::RyOp>(op) ? 3
                                                        : 2;
          plan.gates.push_back(
              cudaq::pauli::Gate::rotation({QubitSet::single(qs[0], axis & 1),
                                            QubitSet::single(qs[0], axis >> 1)},
                                           angle));
        }
      } else if (isa<cudaq::quake::TOp>(op) && qs.size() == 1) {
        plan.gates.push_back(cudaq::pauli::Gate::rotation(
            {0, QubitSet::single(qs[0], 1)}, {adj ? -1 : 1, 4}));
      } else {
        cudaq::pauli::Kind kind;
        if (isa<cudaq::quake::HOp>(op))
          kind = cudaq::pauli::Kind::H;
        else if (isa<cudaq::quake::SOp>(op))
          kind = adj ? cudaq::pauli::Kind::Sdg : cudaq::pauli::Kind::S;
        else if (isa<cudaq::quake::XOp>(op))
          kind = nc ? cudaq::pauli::Kind::CX : cudaq::pauli::Kind::X;
        else if (isa<cudaq::quake::ZOp>(op))
          kind = nc ? cudaq::pauli::Kind::CZ : cudaq::pauli::Kind::Z;
        else if (isa<cudaq::quake::YOp>(op))
          kind = cudaq::pauli::Kind::Y;
        else if (isa<cudaq::quake::SwapOp>(op))
          kind = cudaq::pauli::Kind::Swap;
        else
          return result.reject(&op, "unsupported gate in PBC conversion");
        bool two = kind == cudaq::pauli::Kind::CX ||
                   kind == cudaq::pauli::Kind::CZ ||
                   kind == cudaq::pauli::Kind::Swap;
        if (qs.size() != (two ? 2u : 1u) || !gate.getParameters().empty())
          return result.reject(&op,
                               "malformed Clifford gate in PBC conversion");
        plan.gates.push_back({kind, qs[0], two ? qs[1] : 0});
      }
      plan.erase.push_back(&op);
      continue;
    }
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      for (Value v : ret.getOperands())
        if (isa<cudaq::quake::WireType, cudaq::quake::RefType,
                cudaq::quake::VeqType>(v.getType()))
          return result.reject(&op,
                               "PBC does not permit escaping quantum outputs");
      continue;
    }
    // These classical operations remain in the block. Measurement users are
    // reconnected during emission; angle expressions retain their SSA identity.
    if (isa<cudaq::quake::DiscriminateOp>(op) || arithmetic)
      continue;
    return result.reject(&op, "unsupported operation in PBC; expected scalar "
                              "straight-line Quake and classical arithmetic");
  }
  if (llvm::any_of(current, [](Value v) { return bool(v); }))
    return result.reject(function,
                         "PBC requires every quantum owner to end at sink");
  if (!plan.insertion)
    plan.insertion = block.getTerminator();
  return success();
}

/// Classify the whole function before choosing strict rejection or optional
/// fallback. Verification errors are never downgraded to unsupported input.
ImportResult inspect(func::FuncOp function, bool strict, Plan &plan) {
  if (failed(mlir::verify(function)))
    return {Eligibility::Invalid, function, {}};
  ImportResult result;
  if (failed(readSupported(function, strict, plan, result)) &&
      result.status == Eligibility::Supported)
    (void)result.reject(function, "unsupported PBC operand structure");
  return result;
}

LogicalResult read(func::FuncOp function, bool strict, Plan &plan) {
  return inspect(function, strict, plan).diagnose();
}

/// Commit only a verified replacement. Cloning keeps all original SSA users,
/// attributes and operation identities untouched on every rejected path.
void commit(func::FuncOp original, func::FuncOp candidate) {
  original->setAttrs(candidate->getAttrs());
  original.getBody().takeBody(candidate.getBody());
}

/// Enumerate nonidentity wire indices in ascending order. The first is the
/// parity pivot used by mechanical lowering; callers require nonempty support.
SmallVector<unsigned> support(const cudaq::pauli::Pauli &word) {
  SmallVector<unsigned> qs;
  auto bits = word.x | word.z;
  llvm::append_range(qs, bits.set_bits());
  return qs;
}
/// Spell a Pauli word in operand order; its overall sign is encoded separately.
std::string wordText(const cudaq::pauli::Pauli &word, ArrayRef<unsigned> qs) {
  std::string text;
  for (auto i : qs)
    text += "IXZY"[word.axis(i)];
  return text;
}

/// Simultaneously map commuting observables to Z on their original readout
/// wires. All Clifford work precedes every measurement, so no uncomputation
/// turns terminal readout into a mid-circuit measurement. Previously fixed Z
/// observables may be controls of later parity CXs but must remain unchanged.
/// Reject dependent/conflicting readouts that would require classical result
/// remapping or extra qubits; callers must leave the original IR untouched.
std::optional<std::vector<cudaq::pauli::Gate>>
terminalReadout(unsigned width,
                const std::vector<cudaq::pauli::Pauli> &observables,
                ArrayRef<Measurement> measurements) {
  std::vector<unsigned> wires;
  for (auto measurement : measurements)
    wires.push_back(measurement.readout);
  auto result =
      cudaq::pauli::graph::synthesizeReadout(width, observables, wires, {});
  if (!result)
    return std::nullopt;
  return std::move(result.gates);
}

/// Cost the exact terminal-readout plan emitted below, including one MZ per
/// source outcome on its designated wire. Z is a cost-only measurement proxy.
cudaq::pauli::tcount::Cost
readoutCost(unsigned width, const std::vector<cudaq::pauli::Gate> &rotations,
            const std::vector<cudaq::pauli::Gate> &readout,
            ArrayRef<Measurement> measurements) {
  auto gates = rotations;
  llvm::append_range(gates, readout);
  for (auto measurement : measurements)
    gates.push_back({cudaq::pauli::Kind::Z, measurement.readout});
  return cudaq::pauli::tcount::cost(width, gates);
}
/// Estimate primitive-circuit cost, including terminal measurement gadgets.
/// Use the same basis/parity construction as emission so moving a Clifford
/// into an observable is not counted as a free entangling-gate reduction.
cudaq::pauli::tcount::Cost
mechanicalCost(unsigned width, const std::vector<cudaq::pauli::Gate> &rotations,
               const std::vector<cudaq::pauli::Pauli> &measurements,
               ArrayRef<Measurement> source = {}) {
  if (!source.empty() && llvm::all_of(source, [](const Measurement &m) {
        return isa<cudaq::quake::MppOp>(m.op);
      }))
    if (auto readout = terminalReadout(width, measurements, source))
      return readoutCost(width, rotations, *readout, source);
  auto gates = rotations;
  for (auto [index, word] : llvm::enumerate(measurements)) {
    auto qs = support(word);
    // Primitive X/Y/Z readout is already one operation in the input. Charge
    // parity gadgets only for actual MPP input or for the candidate lowering.
    if (!source.empty() && !isa<cudaq::quake::MppOp>(source[index].op)) {
      gates.push_back({cudaq::pauli::Kind::Z, qs.front()});
      continue;
    }
    std::vector<cudaq::pauli::Gate> basis;
    for (auto i : qs) {
      if (word.axis(i) == 3)
        basis.push_back({cudaq::pauli::Kind::Sdg, i});
      if (word.axis(i) & 1)
        basis.push_back({cudaq::pauli::Kind::H, i});
    }
    for (auto i : ArrayRef<unsigned>(qs).drop_front())
      basis.push_back({cudaq::pauli::Kind::CX, i, qs.front()});
    if (word.sign)
      basis.push_back({cudaq::pauli::Kind::X, qs.front()});
    llvm::append_range(gates, basis);
    // A Z gate is a COST-ONLY stand-in for one single-qubit measurement:
    // one scheduled operation, no T gates, and no entangling gates.
    gates.push_back({cudaq::pauli::Kind::Z, qs.front()});
    llvm::append_range(gates, cudaq::pauli::tcount::detail::adjoint(basis));
  }
  return cudaq::pauli::tcount::cost(width, gates);
}
/// Materialize a symbolic angle conversion, reusing its SSA value when
/// possible.
Value scaled(IRRewriter &b, Location loc, Symbol symbol, double factor) {
  factor *= symbol.scale;
  if (factor == 1)
    return symbol.value;
  if (factor == -1)
    return arith::NegFOp::create(b, loc, symbol.value);
  auto f = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(factor));
  return arith::MulFOp::create(b, loc, symbol.value, f);
}

/// Emit one primitive with controls preceding targets in `qs`.
/// Replace each affected entry in `wires` with its new SSA owner.
void primitive(IRRewriter &b, Location loc, StringRef name,
               ArrayRef<unsigned> qs, SmallVectorImpl<Value> &wires,
               Value angle = {}, bool adj = false, unsigned controls = 0) {
  OperationState state(loc, ("quake." + name).str());
  if (angle)
    state.addOperands(angle);
  for (auto i : qs)
    state.addOperands(wires[i]);
  state.addTypes(SmallVector<Type>(
      qs.size(), cudaq::quake::WireType::get(b.getContext())));
  state.addAttribute("operandSegmentSizes",
                     b.getDenseI32ArrayAttr({angle ? 1 : 0, int32_t(controls),
                                             int32_t(qs.size() - controls)}));
  if (adj)
    state.addAttribute("is_adj", b.getUnitAttr());
  auto *op = b.create(state);
  for (auto [i, v] : llvm::zip(qs, op->getResults()))
    wires[i] = v;
}

/// Map the unsigned Pauli product onto Z on the first support wire, or undo
/// that mapping. Single-qubit basis changes precede a CX parity network.
/// The caller handles an observable's sign separately. Undoing the mapping
/// after measurement preserves the joint projector, rather than measuring
/// each constituent Pauli independently.
void basis(IRRewriter &b, Location loc, cudaq::pauli::Pauli word,
           ArrayRef<unsigned> qs, SmallVectorImpl<Value> &wires, bool inverse) {
  auto one = [&](unsigned i) {
    auto axis = word.axis(i);
    if (!inverse && axis == 3)
      primitive(b, loc, "s", {i}, wires, {}, true);
    if (axis & 1)
      primitive(b, loc, "h", {i}, wires);
    if (inverse && axis == 3)
      primitive(b, loc, "s", {i}, wires);
  };
  if (!inverse) {
    for (auto i : qs)
      one(i);
    for (auto i : qs.drop_front())
      primitive(b, loc, "x", {i, qs.front()}, wires, {}, false, 1);
  } else {
    for (auto i : llvm::reverse(qs.drop_front()))
      primitive(b, loc, "x", {i, qs.front()}, wires, {}, false, 1);
    for (auto i : llvm::reverse(qs))
      one(i);
  }
}

/// Reconstruct a normalized native result and reconnect measurements and sinks.
/// With `lower` false, retain PBC rotations and joint measurements; otherwise
/// expand rotations into parity gadgets and jointly prepare terminal readouts.
/// Non-Clifford+T rational angles need explicit permission for
/// floating-point lowering. Symbolic angles retain their SSA expressions in
/// either mode.
LogicalResult emit(func::FuncOp function, Plan &plan,
                   const cudaq::pauli::pbc::Result &form, bool lower,
                   bool allowApproximate = false, bool nativeNetwork = false) {
  unsigned width = plan.inputs.size();
  auto validWord = [width](const cudaq::pauli::Pauli &word) {
    return (word.x || word.z) && word.x.fits(width) && word.z.fits(width);
  };
  // Check the native/MLIR boundary independently of strategy success. In
  // particular, zip must not truncate outcomes and symbol IDs must not index
  // outside the imported SSA table. These are internal errors, never skips.
  if (!form || form.measurements.size() != plan.measurements.size() ||
      !cudaq::pauli::tcount::detail::valid(width, form.rotations) ||
      !llvm::all_of(form.rotations,
                    [&](const cudaq::pauli::Gate &gate) {
                      if (nativeNetwork &&
                          gate.kind != cudaq::pauli::Kind::Rotate)
                        return true;
                      return gate.kind == cudaq::pauli::Kind::Rotate &&
                             validWord(gate.pauli) &&
                             (nativeNetwork || !gate.pauli.sign) &&
                             gate.angle.symbol <= plan.symbols.size();
                    }) ||
      !llvm::all_of(form.measurements, validWord))
    return function.emitError("invalid native PBC reconstruction contract");
  cudaq::pauli::pbc::CommutingObservables observed;
  for (auto word : form.measurements)
    if (!observed.insert(word))
      return function.emitError(
          "native PBC reconstruction has noncommuting outcomes");
  auto readout = terminalReadout(width, form.measurements, plan.measurements);
  if (lower && !readout)
    return function.emitError(
        "PBC readouts require unsupported classical remapping");
  // Validate every exact angle before editing the owning function.
  for (auto gate : form.rotations) {
    if (gate.kind != cudaq::pauli::Kind::Rotate)
      continue;
    if (gate.angle.symbol)
      continue;
    auto a = gate.angle;
    if (!rational(-__int128(a.numerator) * (a.negative ? -1 : 1),
                  __int128(a.denominator) * 2))
      return function.emitError("PBC exponent exceeds exact integer range");
    if (lower && !allowApproximate &&
        !cudaq::pauli::tcount::detail::quarters(a))
      return function.emitError("non-Clifford+T rational angle requires "
                                "explicit allow-approximate-angles");
  }
  IRRewriter b(function.getContext());
  b.setInsertionPoint(plan.insertion);
  auto loc = function.getLoc();
  SmallVector<Value> wires(plan.inputs);
  for (auto gate : form.rotations) {
    if (nativeNetwork && gate.kind != cudaq::pauli::Kind::Rotate) {
      using cudaq::pauli::Kind;
      // Scalar phase is unobservable only at this measured-function boundary.
      // The native transformations themselves preserve it exactly.
      if (gate.kind == Kind::Phase)
        continue;
      if (gate.kind == Kind::CX || gate.kind == Kind::CZ)
        primitive(b, loc, gate.kind == Kind::CX ? "x" : "z", {gate.a, gate.b},
                  wires, {}, false, 1);
      else if (gate.kind == Kind::Swap)
        primitive(b, loc, "swap", {gate.a, gate.b}, wires);
      else {
        const char *name = gate.kind == Kind::H   ? "h"
                           : gate.kind == Kind::X ? "x"
                           : gate.kind == Kind::Y ? "y"
                           : gate.kind == Kind::Z ? "z"
                                                  : "s";
        primitive(b, loc, name, {gate.a}, wires, {}, gate.kind == Kind::Sdg);
      }
      continue;
    }
    if (gate.pauli.sign)
      gate.angle = gate.angle.negated();
    auto qs = support(gate.pauli);
    auto a = gate.angle;
    if (lower) {
      basis(b, loc, gate.pauli, qs, wires, false);
      // Emit exact multiples of pi/4 directly as Z/S/T gates, avoiding a
      // floating-point Rz that would need to be synthesized again downstream.
      if (auto k = cudaq::pauli::tcount::detail::quarters(a)) {
        int turns = *k % 8;
        unsigned magnitude = std::abs(turns);
        if (magnitude & 4)
          primitive(b, loc, "z", {qs.front()}, wires);
        if (magnitude & 2)
          primitive(b, loc, "s", {qs.front()}, wires, {}, turns < 0);
        if (magnitude & 1)
          primitive(b, loc, "t", {qs.front()}, wires, {}, turns < 0);
      } else {
        Value angle = a.symbol
                          ? scaled(b, loc, plan.symbols[a.symbol - 1],
                                   a.negative ? -1 : 1)
                          : arith::ConstantOp::create(
                                b, loc,
                                b.getF64FloatAttr(
                                    (double(a.numerator) / a.denominator) *
                                    std::numbers::pi * (a.negative ? -1 : 1)))
                                .getResult();
        primitive(b, loc, "rz", {qs.front()}, wires, angle);
      }
      basis(b, loc, gate.pauli, qs, wires, true);
    } else {
      OperationState state(loc, cudaq::quake::ExpPauliOp::getOperationName());
      Value angle;
      if (a.symbol)
        angle =
            scaled(b, loc, plan.symbols[a.symbol - 1], a.negative ? 0.5 : -0.5);
      else {
        auto exponent =
            *rational(-__int128(a.numerator) * (a.negative ? -1 : 1),
                      __int128(a.denominator) * 2);
        angle = arith::ConstantOp::create(
            b, loc,
            b.getF64FloatAttr(
                (double(exponent.numerator) / exponent.denominator) *
                std::numbers::pi));
        // The operand serves Quake consumers; the attribute preserves the
        // exact coefficient for subsequent PBC imports.
        state.addAttribute(
            exactAngleAttr,
            b.getDenseI64ArrayAttr({exponent.numerator, exponent.denominator}));
      }
      state.addOperands(angle);
      for (auto i : qs)
        state.addOperands(wires[i]);
      state.addTypes(SmallVector<Type>(
          qs.size(), cudaq::quake::WireType::get(b.getContext())));
      state.addAttribute("operandSegmentSizes",
                         b.getDenseI32ArrayAttr({1, 0, int32_t(qs.size()), 0}));
      state.addAttribute("pauliLiteral",
                         b.getStringAttr(wordText(gate.pauli, qs)));
      auto *op = b.create(state);
      for (auto [i, v] : llvm::zip(qs, op->getResults()))
        wires[i] = v;
    }
  }
  // Prepare every parity before the first measurement. No quantum state escapes
  // this function, so restoring the old basis after readout is unnecessary.
  if (lower)
    for (auto gate : *readout) {
      using cudaq::pauli::Kind;
      switch (gate.kind) {
      case Kind::Swap:
        primitive(b, loc, "swap", {gate.a, gate.b}, wires);
        break;
      case Kind::CZ:
        primitive(b, loc, "z", {gate.a, gate.b}, wires, {}, false, 1);
        break;
      case Kind::CX:
        primitive(b, loc, "x", {gate.a, gate.b}, wires, {}, false, 1);
        break;
      case Kind::H:
      case Kind::X:
      case Kind::Sdg:
        primitive(b, loc,
                  gate.kind == Kind::H   ? "h"
                  : gate.kind == Kind::X ? "x"
                                         : "s",
                  {gate.a}, wires, {}, gate.kind == Kind::Sdg);
        break;
      default:
        return function.emitError("invalid terminal-readout Clifford");
      }
    }
  // Preserve both the classical SSA outcome and its physical readout identity.
  for (auto [observable, original] :
       llvm::zip(form.measurements, plan.measurements)) {
    auto qs = support(observable);
    if (!lower) {
      // The first MPP operand carries readout identity through PBC rewrites.
      // Retain it as I if it no longer belongs to the transformed support.
      llvm::erase(qs, original.readout);
      qs.insert(qs.begin(), original.readout);
    }
    OperationState state(original.op->getLoc(),
                         lower ? cudaq::quake::MzOp::getOperationName()
                               : cudaq::quake::MppOp::getOperationName());
    if (lower) {
      state.addOperands(wires[original.readout]);
      state.addTypes({original.outcome.getType(),
                      cudaq::quake::WireType::get(b.getContext())});
    } else {
      for (auto i : qs)
        state.addOperands(wires[i]);
      state.addTypes(original.outcome.getType());
      state.addTypes(SmallVector<Type>(
          qs.size(), cudaq::quake::WireType::get(b.getContext())));
      state.addAttribute("word", b.getStringAttr(wordText(observable, qs)));
      state.addAttribute("negative", b.getBoolAttr(observable.sign));
    }
    if (original.name)
      state.addAttribute("registerName", original.name);
    auto *op = b.create(state);
    original.outcome.replaceAllUsesWith(op->getResult(0));
    if (lower) {
      wires[original.readout] = op->getResult(1);
    } else
      for (auto [i, v] : llvm::zip(qs, op->getResults().drop_front()))
        wires[i] = v;
  }
  // Reconnect terminal owners before erasing the old circuit. Reverse erasure
  // removes consumers before producers in the original SSA wire chain.
  for (auto [sink, index] : plan.sinks)
    sink->setOperand(0, wires[index]);
  for (auto *op : llvm::reverse(plan.erase))
    b.eraseOp(op);
  return success();
}

/// Validate the full PBC contract without rewriting the function.
class QuakeVerifyPBCPass
    : public cudaq::opt::impl::QuakeVerifyPBCBase<QuakeVerifyPBCPass> {
public:
  using QuakeVerifyPBCBase::QuakeVerifyPBCBase;
  void runOnOperation() override {
    Plan plan;
    if (failed(read(getOperation(), true, plan)))
      signalPassFailure();
  }
};
/// Import supported Quake, absorb its Clifford frame, and emit verified PBC.
class QuakeToPBCPass : public cudaq::opt::impl::QuakeToPBCBase<QuakeToPBCPass> {
public:
  using QuakeToPBCBase::QuakeToPBCBase;
  void runOnOperation() override {
    OwningOpRef<func::FuncOp> candidate(getOperation().clone());
    Plan plan;
    if (failed(read(*candidate, false, plan)))
      return signalPassFailure();
    if (plan.inputs.empty())
      return;
    if (assumeMeasureAll && plan.measurements.empty()) {
      // This explicit benchmark contract is opt-in. Insert at each terminal
      // sink, then reimport so ordinary measurement bookkeeping owns the bits.
      OpBuilder b(candidate->getContext());
      auto sinks = plan.sinks;
      llvm::sort(sinks, [](auto a, auto b) { return a.second < b.second; });
      b.setInsertionPoint(plan.sinks.front().first);
      for (auto [sink, index] : sinks) {
        OperationState state(sink->getLoc(),
                             cudaq::quake::MzOp::getOperationName());
        state.addOperands(sink->getOperand(0));
        state.addTypes({cudaq::quake::MeasureType::get(b.getContext()),
                        cudaq::quake::WireType::get(b.getContext())});
        auto *measurement = b.create(state);
        sink->setOperand(0, measurement->getResult(1));
      }
      plan = {};
      if (failed(read(*candidate, false, plan)))
        return signalPassFailure();
    }
    // Push the Clifford frame into rotation axes and terminal observables.
    // This preserves joint outcome probabilities for arbitrary inputs; final
    // quantum states are not outputs of the accepted PBC contract.
    auto result = cudaq::pauli::pbc::normalize(plan.inputs.size(), plan.gates,
                                               plan.observables);
    if (!result) {
      getOperation().emitError(result.error);
      return signalPassFailure();
    }
    Plan replacement;
    if (failed(read(*candidate, false, replacement)) ||
        failed(emit(*candidate, replacement, result, false)))
      return signalPassFailure();
    Plan output;
    if (failed(read(*candidate, true, output)))
      return signalPassFailure();
    commit(getOperation(), *candidate);
  }
};
/// Expand verified PBC rotations and joint measurements into primitive gates.
class QuakeLowerPBCPass
    : public cudaq::opt::impl::QuakeLowerPBCBase<QuakeLowerPBCPass> {
public:
  using QuakeLowerPBCBase::QuakeLowerPBCBase;
  void runOnOperation() override {
    Plan plan;
    if (failed(read(getOperation(), true, plan)))
      return signalPassFailure();
    if (plan.inputs.empty())
      return;
    cudaq::pauli::pbc::Result form{plan.gates, plan.observables, {}};
    OwningOpRef<func::FuncOp> candidate(getOperation().clone());
    Plan replacement;
    if (failed(read(*candidate, true, replacement)) ||
        failed(emit(*candidate, replacement, form, true,
                    allowApproximateAngles)) ||
        failed(mlir::verify(*candidate)))
      return signalPassFailure();
    commit(getOperation(), *candidate);
  }
};

/// Map the public profile to the single native synthesis implementation.
template <typename Pass>
std::string synthesisOptions(const Pass &pass,
                             cudaq::pauli::graph::Options &o) {
  using cudaq::pauli::graph::SynthesisObjective;
  if (pass.objective == "total-gates")
    o.objective = SynthesisObjective::TotalGates;
  else if (pass.objective == "t-count")
    o.objective = SynthesisObjective::TCount;
  else if (pass.objective == "entanglers")
    o.objective = SynthesisObjective::Entanglers;
  else if (pass.objective == "depth")
    o.objective = SynthesisObjective::Depth;
  else if (pass.objective == "t-depth")
    o.objective = SynthesisObjective::TDepth;
  else
    return "PBC objective must be t-count, t-depth, entanglers, depth, or "
           "total-gates";
  o.actionWeight = pass.actionWeight;
  o.readoutWeight = pass.readoutWeight;
  o.nativeCX = o.nativeCZ = o.nativeSwap = false;
  SmallVector<StringRef> names;
  StringRef(pass.nativeEntanglers.getValue()).split(names, ',');
  for (StringRef name : names) {
    name = name.trim();
    if (name == "cx")
      o.nativeCX = true;
    else if (name == "cz")
      o.nativeCZ = true;
    else if (name == "swap")
      o.nativeSwap = true;
    else
      return "synthesis-native-entanglers must contain only cx, cz, swap";
  }
  o.singleQubitCost = pass.singleQubitCost;
  o.tCost = pass.tCost;
  o.cxCost = pass.cxCost;
  o.czCost = pass.czCost;
  o.swapCost = pass.swapCost;
  o.tieSeed = pass.tieSeed;
  o.discrepancySteps = pass.discrepancySteps;
  o.representationSteps = pass.representationSteps;
  o.indexedRepresentationSteps = pass.indexedRepresentationSteps;
  o.jointPhaseCandidates = pass.jointPhaseCandidates;
  o.jointPhasePlannerWork = pass.jointPhasePlannerWork;
  o.jointPhaseMaxWork = pass.jointPhaseMaxWork;
  o.jointPhaseTimeoutMs = pass.jointPhaseTimeoutMs;
  o.jointPhaseWideSearch = pass.jointPhaseWideSearch;
  if (pass.jointPhaseAcceptance == "all-metrics")
    o.jointPhaseAcceptance = cudaq::pauli::graph::JointPhaseAcceptance::AllFive;
  else if (pass.jointPhaseAcceptance == "gates-and-t")
    o.jointPhaseAcceptance = cudaq::pauli::graph::JointPhaseAcceptance::GatesAndT;
  else
    return "synthesis-joint-phase-acceptance must be all-metrics or gates-and-t";
  o.readoutCandidates = pass.readoutCandidates;
  o.commutingCandidates = pass.commutingCandidates;
  o.algebraicCandidates = pass.algebraicCandidates;
  o.legacyWorkLimit = pass.legacyWorkLimit;
  o.legacyPortfolioTimeoutMs = pass.legacyPortfolioTimeoutMs;
  o.pruneUnobserved = pass.pruneUnobserved;
  o.parallelizationCredit = pass.parallelizationCredit;
  o.frontierWeight = pass.frontierWeight;
  o.frameWeight = pass.frameWeight;
  o.lookahead = pass.lookahead;
  o.candidateNodes = pass.candidateNodes;
  o.simplify = pass.simplify;
  o.asymmetric = pass.asymmetric;
  o.maxGates = pass.maxGates;
  o.maxDependencies = pass.maxDependencies;
  o.maxSteps = pass.maxSteps;
  // Empty unitary synthesis validates every numeric control without searching.
  return cudaq::pauli::graph::synthesize(1, {}, o).error;
}

/// Isolated, transactional normalized-PBC-to-native synthesis.
class QuakeSynthesizePBCPass
    : public cudaq::opt::impl::QuakeSynthesizePBCBase<QuakeSynthesizePBCPass> {
public:
  using QuakeSynthesizePBCBase::QuakeSynthesizePBCBase;
  template <typename Pass>
  friend std::string synthesisOptions(const Pass &,
                                      cudaq::pauli::graph::Options &);
  void runOnOperation() override {
    cudaq::pauli::graph::Options options;
    if (auto error = synthesisOptions(*this, options); !error.empty()) {
      getOperation().emitError(error);
      return signalPassFailure();
    }
    Plan plan;
    if (failed(read(getOperation(), true, plan)))
      return signalPassFailure();
    if (plan.inputs.empty())
      return;
    std::vector<unsigned> wires;
    for (auto m : plan.measurements)
      wires.push_back(m.readout);
    cudaq::pauli::SearchBudget budget(maxWork, timeoutMs);
    auto form = cudaq::pauli::graph::synthesizeMeasured(
        plan.inputs.size(), plan.gates, plan.observables, wires, options,
        &budget);
    if (!form) {
      getOperation().emitError(form.error);
      return signalPassFailure();
    }
    OwningOpRef<func::FuncOp> candidate(getOperation().clone());
    Plan replacement;
    if (failed(read(*candidate, true, replacement)) ||
        failed(emit(*candidate, replacement, form, true, false, true)) ||
        failed(mlir::verify(*candidate)))
      return signalPassFailure();
    Builder b(getOperation().getContext());
    (*candidate)
        ->setAttr("pbc.synthesis_work", b.getI64IntegerAttr(budget.work()));
    (*candidate)
        ->setAttr("pbc.synthesis_budget_exhausted",
                  b.getBoolAttr(budget.stopped()));
    commit(getOperation(), *candidate);
  }
};

/// Shared strict/optional driver. Search uses read-only native data. Only the
/// final candidate is cloned, emitted, checked and committed. The optional
/// wrapper includes conversion and lowering in that same transaction.
template <typename Base, bool Optional>
class PBCOptimizationPass : public Base {
  template <typename Pass>
  friend std::string synthesisOptions(const Pass &,
                                      cudaq::pauli::graph::Options &);
  cudaq::pauli::OptimizerOptions nativeOptions() const {
    return {
        .enableTCount = this->enableTCount,
        .enableGraphSynthesis = this->enableGraphSynthesis,
        .rounds = this->portfolioRounds,
        .beamWidth = this->portfolioBeamWidth,
        .maxCandidates = this->maxCandidates,
        .maxWork = this->maxWork,
        .timeoutMs = this->timeoutMs,
        .maxGates = this->maxGates,
        .maxTwoQubitGrowth = this->maxTwoQubitGrowth,
        .maxDepthGrowth = this->maxDepthGrowth,
        .intermediateTwoQubitGrowth = this->intermediateGrowth,
        .tcount = {.merge = this->merge,
                   .reverse = this->reverse,
                   .decode = this->decode,
                   .reuseCliffords = this->reuseCliffords,
                   .rounds = this->rounds,
                   .beamWidth = this->beamWidth,
                   .maxBlockRank = this->maxBlockRank,
                   .maxBlockGates = this->maxBlockGates,
                   .mergeWindow = this->mergeWindow,
                   .maxCandidates = this->maxCandidates,
                   .maxWork = this->maxWork},
        .enableGateCount = this->enableGateCount,
        .gatecount = {.rounds = this->gateCountRounds,
                      .commutationWindow = this->gateCommutationWindow,
                      .propagate = this->pauliPropagation,
                      .phasePlacement = this->phasePlacement,
                      .cliffordWindows = this->cliffordWindows},
        .graphSynthesis = {.parallelizationCredit = this->parallelizationCredit,
                           .frontierWeight = this->frontierWeight,
                           .frameWeight = this->frameWeight,
                           .lookahead = this->lookahead,
                           .candidateNodes = this->candidateNodes,
                           .simplify = this->simplify,
                           .asymmetric = this->asymmetric,
                           .maxDependencies = this->maxDependencies,
                           .maxSteps = this->maxSteps}};
  }

  void skip(func::FuncOp function, const Twine &reason,
            cudaq::opt::PBCTelemetry &report) {
    report.outcome = "skipped";
    report.reason = reason.str();
    this->markAllAnalysesPreserved();
    if constexpr (Optional)
      if (this->emitRemarks)
        function.emitRemark() << "PBC optimization skipped: " << reason;
  }

public:
  using Base::Base;
  void runOnOperation() override {
    auto function = this->getOperation();
    cudaq::opt::PBCTelemetry report(function, this->getArgument(),
                                    this->telemetry, this->telemetryMaxEvents);
    auto options = nativeOptions();
    if (auto error = synthesisOptions(*this, options.graphSynthesis);
        !error.empty()) {
      function.emitError(error);
      return this->signalPassFailure();
    }
    if (this->objective == "t-count")
      options.objective = cudaq::pauli::Objective::TCount;
    else if (this->objective == "t-depth")
      options.objective = cudaq::pauli::Objective::TDepth;
    else if (this->objective == "entanglers")
      options.objective = cudaq::pauli::Objective::Entanglers;
    else if (this->objective == "depth")
      options.objective = cudaq::pauli::Objective::Depth;
    else if (this->objective == "total-gates")
      options.objective = cudaq::pauli::Objective::TotalGates;
    else {
      report.reason = "PBC objective must be t-count, t-depth, entanglers, "
                      "depth, or total-gates";
      function.emitError(report.reason);
      return this->signalPassFailure();
    }
    // Invalid configuration is an error even on an ineligible/empty function.
    auto validation = cudaq::pauli::optimize(1, {}, options);
    if (!validation) {
      report.reason = validation.error;
      function.emitError(validation.error);
      return this->signalPassFailure();
    }
    if (report.enabled()) {
      std::string pipeline;
      llvm::raw_string_ostream stream(pipeline);
      this->printAsTextualPipeline(stream);
      report.configure(options, pipeline);
    }
    if constexpr (Optional) {
      if (!this->maxOperations) {
        report.reason = "PBC max-operations must be positive";
        function.emitError(report.reason);
        return this->signalPassFailure();
      }
      // Verify even oversized input: invalid IR must not escape as a skip.
      if (failed(report.time("input_verification",
                             [&] { return mlir::verify(function); })))
        return this->signalPassFailure();
      uint64_t count = 0;
      auto walk = function.walk([&](Operation *) {
        return ++count > this->maxOperations ? WalkResult::interrupt()
                                             : WalkResult::advance();
      });
      if (walk.wasInterrupted()) {
        skip(function, "function operation limit", report);
        return;
      }
    }
    Plan plan;
    auto eligibility = report.time(
        "import", [&] { return inspect(function, !Optional, plan); });
    if (eligibility.status != Eligibility::Supported) {
      if constexpr (Optional) {
        if (eligibility.status == Eligibility::Unsupported) {
          skip(function, eligibility.reason, report);
          return;
        }
      }
      report.reason =
          eligibility.reason.empty() ? "invalid input IR" : eligibility.reason;
      (void)eligibility.diagnose();
      return this->signalPassFailure();
    }
    report.captureInput();
    if (plan.inputs.empty()) {
      skip(function, "no quantum inputs", report);
      return;
    }
    if (!options.enableTCount && !options.enableGraphSynthesis &&
        !options.enableGateCount) {
      skip(function, "all strategies disabled", report);
      return;
    }
    if (plan.gates.size() > options.maxGates) {
      skip(function, "native gate limit", report);
      return;
    }
    if (!plan.inputs.empty() && report.enabled()) {
      report.inputCost = mechanicalCost(plan.inputs.size(), plan.gates,
                                        plan.observables, plan.measurements);
      report.unresolvedInput = cudaq::pauli::unsynthesizedRotations(plan.gates);
    }
    cudaq::pauli::SearchBudget budget(options.maxWork, options.timeoutMs);
    bool measuredSynthesis = false;
    if constexpr (Optional)
      measuredSynthesis = options.enableGraphSynthesis &&
                          !this->preserveCliffordNetwork &&
                          !plan.observables.empty();
    auto portfolioOptions = options;
    if (measuredSynthesis)
      portfolioOptions.enableGraphSynthesis = false;
    auto result = report.time("search", [&] {
      return cudaq::pauli::optimize(plan.inputs.size(), plan.gates,
                                    portfolioOptions, &budget,
                                    report.collector());
    });
    report.recordStatistics(result.stats);
    if (!result) {
      report.reason = result.error;
      function.emitError(result.error);
      return this->signalPassFailure();
    }
    if (result.stats.skippedSize || budget.stopped()) {
      skip(function, "search size/work/deadline limit", report);
      return;
    }
    auto before = mechanicalCost(plan.inputs.size(), plan.gates,
                                 plan.observables, plan.measurements);
    bool preserveNetwork = false;
    if constexpr (Optional)
      preserveNetwork = this->preserveCliffordNetwork;
    auto normalized = report.time("normalization", [&] {
      if (measuredSynthesis) {
        std::vector<unsigned> wires;
        for (auto m : plan.measurements)
          wires.push_back(m.readout);
        preserveNetwork = true;
        cudaq::pauli::TelemetryScope observation(
            report.collector(), "graph-synthesis", report.trace.winner);
        uint64_t initialWork = budget.work();
        if (report.enabled()) {
          observation.event.before =
              cudaq::pauli::tcount::cost(plan.inputs.size(), result.gates);
          observation.event.unresolvedBefore =
              cudaq::pauli::unsynthesizedRotations(result.gates);
        }
        auto form = cudaq::pauli::graph::synthesizeMeasured(
            plan.inputs.size(), result.gates, plan.observables, wires,
            options.graphSynthesis, &budget);
        if (report.enabled()) {
          observation.event.work = budget.work() - initialWork;
          observation.event.decision = form ? "selected" : "rejected";
          observation.event.reason =
              form ? "verified terminal measurement synthesis" : form.error;
          if (form) {
            observation.event.after =
                cudaq::pauli::tcount::cost(plan.inputs.size(), form.rotations);
            observation.event.unresolvedAfter =
                cudaq::pauli::unsynthesizedRotations(form.rotations);
            report.trace.winner = observation.event.id;
          }
        }
        return form;
      }
      if (preserveNetwork)
        return cudaq::pauli::pbc::Result{result.gates, plan.observables, {}};
      return cudaq::pauli::pbc::normalize(plan.inputs.size(), result.gates,
                                          plan.observables, &budget);
    });
    if (!normalized) {
      if (budget.stopped()) {
        skip(function, "normalization work/deadline limit", report);
        return;
      }
      if constexpr (Optional) {
        if (measuredSynthesis) {
          // Like graph proposals inside the native portfolio, a failed
          // measured-synthesis candidate is declined without committing it.
          // Structural limits need not exhaust the shared work budget.
          // Input and option validation have already succeeded above.
          skip(function, normalized.error, report);
          return;
        }
      }
      report.reason = normalized.error;
      function.emitError(normalized.error);
      return this->signalPassFailure();
    }
    // Lexicographic ordering prioritizes the requested objective. Include the
    // cost of reconstructing every outcome, not just the native rotations.
    auto readout = terminalReadout(plan.inputs.size(), normalized.measurements,
                                   plan.measurements);
    if (!readout) {
      skip(function, "PBC readouts require unsupported classical remapping",
           report);
      return;
    }
    auto after = readoutCost(plan.inputs.size(), normalized.rotations, *readout,
                             plan.measurements);
    if (report.enabled()) {
      report.candidateCost = after;
      report.unresolvedCandidate =
          cudaq::pauli::unsynthesizedRotations(normalized.rotations);
    }
    auto score = [&](const cudaq::pauli::tcount::Cost &c) {
      switch (options.objective) {
      case cudaq::pauli::Objective::TotalGates:
        return std::tuple{c.gates, c.t, c.depth, c.twoQubit, c.tDepth};
      case cudaq::pauli::Objective::TDepth:
        return std::tuple{c.tDepth, c.t, c.twoQubit, c.depth, uint64_t(0)};
      case cudaq::pauli::Objective::Entanglers:
        return std::tuple{c.twoQubit, c.depth, c.t, c.tDepth, uint64_t(0)};
      case cudaq::pauli::Objective::Depth:
        return std::tuple{c.depth, c.twoQubit, c.t, c.tDepth, uint64_t(0)};
      default:
        return std::tuple{c.t, c.twoQubit, c.depth, c.tDepth, uint64_t(0)};
      }
    };
    bool representable =
        llvm::all_of(normalized.rotations, [](const auto &gate) {
          if (gate.kind != cudaq::pauli::Kind::Rotate)
            return true;
          auto a = gate.angle;
          return a.symbol ||
                 (rational(-__int128(a.numerator) * (a.negative ? -1 : 1),
                           __int128(a.denominator) * 2) &&
                  (!Optional || cudaq::pauli::tcount::detail::quarters(a)));
        });
    if (!representable) {
      skip(function,
           "candidate cannot be lowered with the exact-angle contract", report);
      return;
    }
    if ((options.objective == cudaq::pauli::Objective::TotalGates &&
         after.t > before.t) ||
        !(score(after) < score(before)) ||
        after.twoQubit > before.twoQubit + options.maxTwoQubitGrowth ||
        after.depth > before.depth + options.maxDepthGrowth) {
      skip(function, "no admissible improvement", report);
      return;
    }
    if constexpr (Optional) {
      // Bound allocation before constructing parity gadgets and the planned
      // terminal Clifford/readout sequence.
      uint64_t operations =
          1 + function.front().getOperations().size() - plan.erase.size();
      for (const auto &gate : normalized.rotations)
        operations += gate.kind == cudaq::pauli::Kind::Rotate
                          ? 6 * gate.pauli.weight() + 7
                          : 1;
      operations += readout->size() + normalized.measurements.size();
      if (operations > this->maxOperations) {
        skip(function, "candidate operation limit", report);
        return;
      }
    }
    OwningOpRef<func::FuncOp> candidate(
        report.time("clone", [&] { return function.clone(); }));
    Plan replacement;
    if (failed(report.time(
            "candidate_import",
            [&] { return read(*candidate, !Optional, replacement); })) ||
        failed(report.time(Optional ? "lowering" : "pbc_emission", [&] {
          return emit(*candidate, replacement, normalized, Optional, false,
                      preserveNetwork);
        })))
      return this->signalPassFailure();
    // Lowered output contains primitive gates and measurements, so verify it
    // as ordinary Quake rather than requiring the strict PBC operation set.
    if constexpr (Optional) {
      if (failed(report.time("output_verification",
                             [&] { return mlir::verify(*candidate); })))
        return this->signalPassFailure();
    } else {
      Plan output;
      if (failed(report.time("output_verification",
                             [&] { return read(*candidate, true, output); })))
        return this->signalPassFailure();
    }
    Builder b(function.getContext());
    (*candidate)->setAttr("pbc.t_before", b.getI64IntegerAttr(before.t));
    (*candidate)->setAttr("pbc.t_after", b.getI64IntegerAttr(after.t));
    (*candidate)
        ->setAttr("pbc.two_qubit_before", b.getI64IntegerAttr(before.twoQubit));
    (*candidate)
        ->setAttr("pbc.two_qubit_after", b.getI64IntegerAttr(after.twoQubit));
    (*candidate)
        ->setAttr("pbc.depth_before", b.getI64IntegerAttr(before.depth));
    (*candidate)->setAttr("pbc.depth_after", b.getI64IntegerAttr(after.depth));
    (*candidate)->setAttr("pbc.work", b.getI64IntegerAttr(budget.work()));
    (*candidate)
        ->setAttr("pbc.budget_exhaustions",
                  b.getI64IntegerAttr(result.stats.budgetExhaustions));
    commit(function, *candidate);
    report.outcome = "committed";
    report.reason = "verified admissible improvement";
  }
};

/// Strict optimization requires verified PBC input and retains PBC output.
class QuakeOptimizePBCPass
    : public PBCOptimizationPass<
          cudaq::opt::impl::QuakeOptimizePBCBase<QuakeOptimizePBCPass>, false> {
public:
  using PBCOptimizationPass::PBCOptimizationPass;
};

/// Optional conversion, optimization and lowering are one atomic rewrite.
class QuakeTryPBCPass
    : public PBCOptimizationPass<
          cudaq::opt::impl::QuakeTryPBCBase<QuakeTryPBCPass>, true> {
public:
  using PBCOptimizationPass::PBCOptimizationPass;
};
} // namespace
