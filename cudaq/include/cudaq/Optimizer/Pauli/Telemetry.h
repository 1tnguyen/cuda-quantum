/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Optional, bounded observations of Pauli portfolio exploration.
///
/// Records candidate ancestry and inclusive timing without participating in
/// candidate selection. A null collector avoids clocks and record allocation.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/TCount.h"
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::pauli {
/// Rotations whose eventual Clifford+T synthesis cost is not modeled here.
inline uint64_t unsynthesizedRotations(const std::vector<Gate> &gates) {
  uint64_t count = 0;
  for (const auto &g : gates)
    if (g.kind == Kind::Rotate &&
        (g.angle.symbol ||
         (__int128(g.angle.numerator) * 4) % g.angle.denominator))
      ++count;
  return count;
}

/// Accumulate monotonic elapsed nanoseconds, only when a destination exists.
class TelemetryTimer {
  /// Borrowed nanosecond accumulator; null disables clock reads and updates.
  uint64_t *destination;
  /// Monotonic origin, initialized only when collection is enabled.
  std::chrono::steady_clock::time_point start;

public:
  /// Start measuring into an accumulator that must outlive this timer.
  explicit TelemetryTimer(uint64_t *destination) : destination(destination) {
    if (destination)
      start = std::chrono::steady_clock::now();
  }
  /// Add the elapsed interval to the accumulator without replacing prior time.
  ~TelemetryTimer() {
    if (destination)
      *destination += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  }
  TelemetryTimer(const TelemetryTimer &) = delete;
  TelemetryTimer &operator=(const TelemetryTimer &) = delete;
};

/// One strategy invocation. Parent denotes input-circuit ancestry; scope
/// denotes nesting. selectedChild identifies the winner of a nested portfolio,
/// if any.
struct TelemetryEvent {
  /// Unique invocation ID, input-candidate ID, enclosing invocation ID, and
  /// selected nested invocation ID, respectively. Zero denotes the original
  /// input for parent and the absence of an invocation for the other IDs.
  uint64_t id = 0, parent = 0, scope = 0, selectedChild = 0;
  /// Strategy/variant name, candidate-selection disposition, and explanation.
  /// A disposition describes exploration; it does not imply a final IR commit.
  std::string method, decision = "candidate", reason;
  /// Inclusive elapsed nanoseconds, explicit proof-check nanoseconds (a subset
  /// of elapsed time), and charged work units. Nested times/work may overlap.
  uint64_t elapsedNs = 0, verificationNs = 0, work = 0;
  /// Mechanical unitary costs before/after this invocation, excluding terminal
  /// measurements. Missing values indicate an unavailable cost, not zero cost.
  std::optional<tcount::Cost> before, after;
  /// Rotation counts whose downstream T costs are unknown, aligned with the
  /// before/after costs. An absent count means completeness was not measured.
  std::optional<uint64_t> unresolvedBefore, unresolvedAfter;
  /// Strategy-specific activity counts, including work on discarded candidates.
  std::map<std::string, uint64_t> counters;
};

/// Aggregate exploration effort, retained even when detailed events are capped.
/// Nested method totals overlap; they are not an additive pass-time breakdown.
struct TelemetryAggregate {
  /// Invocation count and sums of inclusive elapsed nanoseconds, proof-check
  /// nanoseconds, and work units across all invocations of one method.
  uint64_t calls = 0, elapsedNs = 0, verificationNs = 0, work = 0;
  /// Numbers of calls that lower T count, entangler count, depth, total gates,
  /// or T depth relative to their own inputs. These are counts of improving
  /// calls, not additive savings or contributions to the committed circuit.
  uint64_t lowerT = 0, lowerTwoQubit = 0, lowerDepth = 0, lowerGates = 0,
           lowerTDepth = 0;
};

/// Per-function collector, owned by the caller. IDs remain unique when records
/// are truncated; missing ancestry is explicitly signaled by droppedEvents.
struct Telemetry {
  /// Detailed-record cap; zero keeps method summaries without retaining events.
  size_t maxEvents = 4096;
  /// Next invocation ID, records omitted by the cap, and native winning ID.
  /// IDs advance even for omitted records. A zero winner denotes the input;
  /// a nonzero native winner can still be rejected by the enclosing MLIR pass.
  uint64_t nextId = 1, droppedEvents = 0, winner = 0;
  /// Retained records in completion order, which can differ from ID order.
  std::vector<TelemetryEvent> events;
  /// Per-method totals, accumulated even when detailed records are omitted.
  std::map<std::string, TelemetryAggregate> methods;

  /// Update a retained record's disposition without affecting search selection.
  /// An ID omitted by the record cap is intentionally ignored.
  void decision(uint64_t id, const std::string &value) {
    for (auto &event : events)
      if (event.id == id) {
        event.decision = value;
        return;
      }
  }
};

/// Stack-local event, safe across nested collector vector reallocations.
class TelemetryScope {
  /// Borrowed collector, which must outlive the scope; null disables recording.
  Telemetry *collector;
  /// Optional inclusive timer targeting this scope's event, not vector storage.
  std::optional<TelemetryTimer> timer;

public:
  /// Mutable observation populated by the strategy before scope completion.
  TelemetryEvent event;
  /// Allocate an ID and start timing only when a collector is supplied.
  TelemetryScope(Telemetry *collector, const char *method, uint64_t parent,
                 uint64_t scope = 0)
      : collector(collector) {
    if (collector) {
      event.id = collector->nextId++;
      event.parent = parent;
      event.scope = scope;
      event.method = method;
      timer.emplace(&event.elapsedNs);
    }
  }
  /// Finish timing, update method totals, and retain or count the omitted
  /// event.
  ~TelemetryScope() {
    timer.reset();
    if (collector) {
      auto &total = collector->methods[event.method];
      ++total.calls;
      total.elapsedNs += event.elapsedNs;
      total.verificationNs += event.verificationNs;
      total.work += event.work;
      if (event.before && event.after) {
        total.lowerT += event.after->t < event.before->t;
        total.lowerTwoQubit += event.after->twoQubit < event.before->twoQubit;
        total.lowerDepth += event.after->depth < event.before->depth;
        total.lowerGates += event.after->gates < event.before->gates;
        total.lowerTDepth += event.after->tDepth < event.before->tDepth;
      }
      if (collector->events.size() < collector->maxEvents)
        collector->events.push_back(std::move(event));
      else
        ++collector->droppedEvents;
    }
  }
  /// Return the proof-time accumulator, or null when instrumentation is off.
  /// The pointer is borrowed and remains valid only while this scope is alive.
  uint64_t *verificationTime() {
    return collector ? &event.verificationNs : nullptr;
  }
};
} // namespace cudaq::pauli
