/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Per-function PBC telemetry reporting, separate from compiler IR.
///
/// Owns optional native observations and serializes a bounded JSONL record on
/// every exit, including clean skips and failures. Reporting never edits IR.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/Optimizer.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::opt {
/// Own one function invocation's optional telemetry and report it on scope
/// exit. Observations never participate in optimization or mutate the
/// function's IR.
class PBCTelemetry {
  /// Borrowed function, used for identity, committed-output metrics and
  /// warnings.
  mlir::func::FuncOp function;
  /// Resolved output destination (empty disables, '-' selects stderr) and pass
  /// name identifying this invocation.
  std::string destination, pass;
  /// Effective controls, named-stage nanosecond timings, and portfolio
  /// counters.
  llvm::json::Object configuration, stages, statistics;
  /// Observed input-IR metrics; null until supported input has been captured.
  llvm::json::Value inputIR = nullptr;
  /// Instrumented invocation time, excluding final report construction/output.
  uint64_t elapsedNs = 0;
  /// Optional timer targeting elapsedNs, stopped before report serialization.
  std::optional<pauli::TelemetryTimer> timer;

public:
  /// Bounded native event collector and method summaries for this invocation.
  pauli::Telemetry trace;
  /// Input and normalized-candidate mechanical costs, including measurement
  /// reconstruction. The candidate need not be committed; absent means unknown.
  std::optional<pauli::tcount::Cost> inputCost, candidateCost;
  /// Unsynthesized-rotation counts associated with the input/candidate costs.
  std::optional<uint64_t> unresolvedInput, unresolvedCandidate;
  /// Final pass disposition and explanation. Default to failure so an early
  /// error exit cannot be reported as a successful optimization or clean skip.
  std::string outcome = "failed",
              reason = "validation or reconstruction failed";

  /// Resolve the explicit destination or environment default and start timing
  /// when enabled. The function must remain alive until reporting completes.
  PBCTelemetry(mlir::func::FuncOp, llvm::StringRef pass,
               llvm::StringRef destination, unsigned maxEvents);
  /// Append one JSONL record when enabled; report write failures as warnings.
  ~PBCTelemetry();
  /// Whether destination resolution enabled reporting for this invocation.
  bool enabled() const { return !destination.empty(); }
  /// Supply native strategies with the collector, or null for disabled tracing.
  pauli::Telemetry *collector() { return enabled() ? &trace : nullptr; }
  /// Record validated native controls and the textual MLIR pass configuration.
  void configure(const pauli::OptimizerOptions &, llvm::StringRef pipeline);
  /// Snapshot supported scalar-wire input metrics before any committed rewrite.
  void captureInput();
  /// Copy search accounting independently of the detailed-event retention cap.
  void recordStatistics(const pauli::OptimizerStatistics &);
  /// Measure a named stage and return its result unchanged. Reusing a stage
  /// name replaces its timing; disabled reporting directly invokes the
  /// callable.
  template <typename Callable>
  auto time(llvm::StringRef name, Callable &&callable) {
    if (!enabled())
      return callable();
    uint64_t elapsed = 0;
    // Keep JSON serialization outside the measured stage.
    /// Store the stage duration after the later-declared timer has stopped.
    struct Record {
      /// Borrowed stage map owned by the enclosing telemetry report.
      llvm::json::Object &stages;
      /// Borrowed stage key, valid throughout the callable's execution.
      llvm::StringRef name;
      /// Borrowed accumulator populated by the stage timer before destruction.
      uint64_t &elapsed;
      ~Record() { stages[name] = elapsed; }
    } record{stages, name, elapsed};
    pauli::TelemetryTimer stageTimer(&elapsed);
    return callable();
  }
};
} // namespace cudaq::opt
