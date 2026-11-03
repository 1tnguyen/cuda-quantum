/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief JSONL output and emitted-IR metrics for optional PBC telemetry.
///
/// Reports inclusive strategy timing, ancestry and mechanical estimates apart
/// from observed Quake operations. File writes are serialized within a process;
/// use separate destinations for independent compiler processes.
//===----------------------------------------------------------------------===//

#include "PBCTelemetry.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <set>

using namespace mlir;
using namespace cudaq;
namespace {
llvm::json::Value cost(const std::optional<pauli::tcount::Cost> &value,
                       std::optional<uint64_t> unresolved = {}) {
  if (!value)
    return nullptr;
  return llvm::json::Object{
      {"t_count", value->t},
      {"two_qubit_count", value->twoQubit},
      {"depth", value->depth},
      {"total_gates", value->gates},
      {"t_depth", value->tDepth},
      {"unsynthesized_rotations", unresolved ? llvm::json::Value(*unresolved)
                                             : llvm::json::Value(nullptr)},
      {"t_count_complete", unresolved && !*unresolved}};
}

/// Positive values are reductions relative to the same invocation's input.
llvm::json::Value reduction(const std::optional<pauli::tcount::Cost> &before,
                            const std::optional<pauli::tcount::Cost> &after) {
  if (!before || !after)
    return nullptr;
  return llvm::json::Object{
      {"t_count", int64_t(before->t) - int64_t(after->t)},
      {"two_qubit_count", int64_t(before->twoQubit) - int64_t(after->twoQubit)},
      {"depth", int64_t(before->depth) - int64_t(after->depth)},
      {"total_gates", int64_t(before->gates) - int64_t(after->gates)},
      {"t_depth", int64_t(before->tDepth) - int64_t(after->tDepth)}};
}

/// Count the supported scalar-wire IR itself. This does not synthesize Pauli
/// rotations or assign them an invented T/magic-state cost. Depth includes
/// measurements; total_gates excludes them. Classical operations are excluded.
llvm::json::Object irMetrics(func::FuncOp function) {
  llvm::json::Object histogram;
  llvm::DenseMap<Value, std::pair<uint64_t, uint64_t>> layers;
  uint64_t gates = 0, measurements = 0, t = 0, two = 0, clifford = 0,
           rotations = 0, depth = 0, tDepth = 0, cx = 0, cz = 0, swaps = 0;
  if (!function.isExternal())
    for (auto &op : function.front()) {
      auto gate = dyn_cast<quake::OperatorInterface>(op);
      auto measurement = dyn_cast<quake::MeasurementInterface>(op);
      if (!gate && !measurement)
        continue;
      std::string name = op.getName().getStringRef().str();
      if (gate && !gate.getControls().empty())
        name += ".controls_" + std::to_string(gate.getControls().size());
      if (op.hasAttr("is_adj"))
        name += ".adj";
      histogram[name] = histogram.getInteger(name).value_or(0) + 1;
      bool isT = isa<quake::TOp>(op);
      bool isRotation = isa<quake::ExpPauliOp, quake::RxOp, quake::RyOp,
                            quake::RzOp, quake::PhaseOp>(op);
      if (gate) {
        ++gates;
        cx += isa<quake::XOp>(op) && gate.getControls().size() == 1;
        cz += isa<quake::ZOp>(op) && gate.getControls().size() == 1;
        swaps += isa<quake::SwapOp>(op);
        t += isT;
        rotations += isRotation;
        clifford += !isT && !isRotation;
        two += gate.getControls().size() + gate.getTargets().size() == 2;
      } else
        ++measurements;
      uint64_t d = 0, td = 0;
      for (Value operand : op.getOperands())
        if (isa<quake::WireType>(operand.getType())) {
          auto previous = layers.lookup(operand);
          d = std::max(d, previous.first);
          td = std::max(td, previous.second);
        }
      ++d;
      td += isT;
      depth = std::max(depth, d);
      tDepth = std::max(tDepth, td);
      for (Value result : op.getResults())
        if (isa<quake::WireType>(result.getType()))
          layers[result] = {d, td};
    }
  return llvm::json::Object{{"gate_histogram", std::move(histogram)},
                            {"total_gates", gates},
                            {"measurements", measurements},
                            {"cx_count", cx},
                            {"cz_count", cz},
                            {"swap_count", swaps},
                            {"explicit_t_count", t},
                            {"two_qubit_operations", two},
                            {"named_clifford_count", clifford},
                            {"rotation_operations", rotations},
                            {"depth_including_measurements", depth},
                            {"explicit_t_depth", tDepth},
                            {"magic_states", nullptr},
                            {"post_target_synthesis", false}};
}
} // namespace

opt::PBCTelemetry::PBCTelemetry(func::FuncOp function, llvm::StringRef pass,
                                llvm::StringRef requested, unsigned maxEvents)
    : function(function), pass(pass.str()) {
  destination = requested.str();
  if (destination.empty())
    if (const char *environment = std::getenv("CUDAQ_PBC_TELEMETRY"))
      destination = environment;
  if (destination == "off")
    destination.clear();
  trace.maxEvents = maxEvents;
  if (enabled())
    timer.emplace(&elapsedNs);
}

void opt::PBCTelemetry::configure(const pauli::OptimizerOptions &o,
                                  llvm::StringRef pipeline) {
  if (!enabled())
    return;
  configuration = llvm::json::Object{
      {"pipeline", pipeline.str()},
      {"objective",
       std::array<const char *, 5>{"t-count", "t-depth", "entanglers", "depth",
                                   "total-gates"}[unsigned(o.objective)]},
      {"gate_count", o.enableGateCount},
      {"gate_count_options",
       llvm::json::Object{{"rounds", o.gatecount.rounds},
                          {"commutation_window", o.gatecount.commutationWindow},
                          {"pauli_propagation", o.gatecount.propagate},
                          {"phase_placement", o.gatecount.phasePlacement},
                          {"clifford_windows", o.gatecount.cliffordWindows}}},
      {"t_count", o.enableTCount},
      {"graph_synthesis", o.enableGraphSynthesis},
      {"rounds", o.rounds},
      {"beam_width", o.beamWidth},
      {"max_candidates", o.maxCandidates},
      {"max_work", o.maxWork},
      {"timeout_ms", o.timeoutMs},
      {"max_gates", o.maxGates},
      {"max_two_qubit_growth", o.maxTwoQubitGrowth},
      {"max_depth_growth", o.maxDepthGrowth},
      {"intermediate_two_qubit_growth", o.intermediateTwoQubitGrowth},
      {"t_options",
       llvm::json::Object{{"merge", o.tcount.merge},
                          {"reverse", o.tcount.reverse},
                          {"decode", o.tcount.decode},
                          {"reuse_cliffords", o.tcount.reuseCliffords},
                          {"rounds", o.tcount.rounds},
                          {"beam_width", o.tcount.beamWidth},
                          {"max_block_rank", o.tcount.maxBlockRank},
                          {"max_block_gates", o.tcount.maxBlockGates},
                          {"merge_window", o.tcount.mergeWindow},
                          {"max_candidates", o.tcount.maxCandidates},
                          {"max_work", o.tcount.maxWork}}},
      {"graph_options",
       llvm::json::Object{
           {"parallelization_credit", o.graphSynthesis.parallelizationCredit},
           {"frontier_weight", o.graphSynthesis.frontierWeight},
           {"frame_weight", o.graphSynthesis.frameWeight},
           {"lookahead", o.graphSynthesis.lookahead},
           {"candidate_nodes", o.graphSynthesis.candidateNodes},
           {"simplify", o.graphSynthesis.simplify},
           {"asymmetric", o.graphSynthesis.asymmetric},
           {"max_dependencies", o.graphSynthesis.maxDependencies},
           {"max_steps", o.graphSynthesis.maxSteps}}}};
}

void opt::PBCTelemetry::captureInput() {
  if (enabled())
    inputIR = irMetrics(function);
}
void opt::PBCTelemetry::recordStatistics(const pauli::OptimizerStatistics &s) {
  if (enabled())
    statistics =
        llvm::json::Object{{"work", s.work},
                           {"candidates", s.candidates},
                           {"t_candidates", s.tcountCandidates},
                           {"graph_candidates", s.graphSynthesisCandidates},
                           {"gate_count_candidates", s.gateCountCandidates},
                           {"rejected_candidates", s.rejectedCandidates},
                           {"rescued_candidates", s.rescuedCandidates},
                           {"budget_exhaustions", s.budgetExhaustions},
                           {"native_improved", s.changed}};
}

opt::PBCTelemetry::~PBCTelemetry() {
  if (!enabled())
    return;
  timer.reset();
  std::set<uint64_t> visited;
  bool complete = true;
  llvm::json::Array lineage;
  std::function<void(uint64_t)> visit = [&](uint64_t id) {
    if (!id || !visited.insert(id).second)
      return;
    auto it =
        llvm::find_if(trace.events, [&](const auto &e) { return e.id == id; });
    if (it == trace.events.end()) {
      complete = false;
      return;
    }
    visit(it->parent);
    visit(it->selectedChild);
    lineage.push_back(id);
  };
  visit(trace.winner);
  llvm::json::Object methods;
  for (const auto &[name, total] : trace.methods)
    methods[name] =
        llvm::json::Object{{"calls", total.calls},
                           {"elapsed_ns", total.elapsedNs},
                           {"verification_ns", total.verificationNs},
                           {"work", total.work},
                           {"calls_lowering_t", total.lowerT},
                           {"calls_lowering_two_qubit", total.lowerTwoQubit},
                           {"calls_lowering_depth", total.lowerDepth},
                           {"calls_lowering_total_gates", total.lowerGates},
                           {"calls_lowering_t_depth", total.lowerTDepth}};
  llvm::json::Array events;
  for (const auto &e : trace.events) {
    llvm::json::Object counters;
    for (const auto &[key, value] : e.counters)
      counters[key] = value;
    events.push_back(
        llvm::json::Object{{"id", e.id},
                           {"parent", e.parent},
                           {"scope", e.scope},
                           {"selected_child", e.selectedChild},
                           {"method", e.method},
                           {"decision", e.decision},
                           {"reason", e.reason},
                           {"elapsed_ns", e.elapsedNs},
                           {"verification_ns", e.verificationNs},
                           {"work", e.work},
                           {"before", cost(e.before, e.unresolvedBefore)},
                           {"after", cost(e.after, e.unresolvedAfter)},
                           {"reduction", reduction(e.before, e.after)},
                           {"counters", std::move(counters)}});
  }
  llvm::json::Value outputIR = inputIR;
  if (outcome == "committed")
    outputIR = irMetrics(function);
  llvm::json::Object report{
      {"schema_version", 1},
      {"metric_semantics",
       "event costs exclude measurements; pass estimates include mechanical "
       "measurement gadgets; arbitrary rotations have no downstream T "
       "estimate; observed IR counts are before target synthesis"},
      {"function", function.getSymName().str()},
      {"pass", pass},
      {"outcome", outcome},
      {"reason", reason},
      {"elapsed_ns", elapsedNs},
      {"timing_semantics",
       "inclusive; verification is a subset; nested events must not be summed"},
      {"configuration", std::move(configuration)},
      {"stage_ns", std::move(stages)},
      {"portfolio", std::move(statistics)},
      {"events", std::move(events)},
      {"dropped_events", trace.droppedEvents},
      {"max_events", uint64_t(trace.maxEvents)},
      {"method_totals", std::move(methods)},
      {"native_winner", trace.winner},
      {"winner_committed", outcome == "committed"},
      {"winning_candidate_ids", std::move(lineage)},
      {"lineage_complete", complete},
      {"mechanical_estimates",
       llvm::json::Object{
           {"input", cost(inputCost, unresolvedInput)},
           {"committed_reduction",
            reduction(inputCost,
                      outcome == "committed" ? candidateCost : inputCost)},
           {"candidate", cost(candidateCost, unresolvedCandidate)},
           {"output", cost(outcome == "committed" ? candidateCost : inputCost,
                           outcome == "committed" ? unresolvedCandidate
                                                  : unresolvedInput)}}},
      {"observed_ir", llvm::json::Object{{"input", std::move(inputIR)},
                                         {"output", std::move(outputIR)}}}};
  std::string line =
      llvm::formatv("{0}", llvm::json::Value(std::move(report))).str() + "\n";
  static std::mutex outputMutex;
  std::lock_guard<std::mutex> guard(outputMutex);
  if (destination == "-") {
    llvm::errs() << line;
    return;
  }
  std::error_code error;
  llvm::raw_fd_ostream stream(destination, error, llvm::sys::fs::OF_Append);
  if (!error) {
    stream << line;
    stream.flush();
    error = stream.error();
    stream.clear_error();
  }
  if (error)
    function.emitWarning() << "cannot write PBC telemetry to '" << destination
                           << "': " << error.message();
}
