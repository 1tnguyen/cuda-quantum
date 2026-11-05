/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include <optional>
#include <utility>

namespace cudaq::pauli::graph::detail {

/// Borrowed, immutable whole-program contract. Observable order and physical
/// readout wires are part of the problem: neither may be permuted by a solver.
/// This contract releases the terminal quantum state for arbitrary inputs. It
/// must not be used for an internal region with live quantum outputs.
struct MeasuredProblem {
  unsigned width;
  const std::vector<Gate> &rotations;
  const std::vector<Pauli> &observables;
  const std::vector<unsigned> &readoutWires;
};

enum class SubmissionStatus {
  Verified,
  RejectedByCost,
  ProofBudgetExhausted,
  InvalidProof
};

struct SubmissionResult {
  SubmissionStatus status;
  std::string error;
};

/// Only the checked sink can create this value. The local-channel proof is
/// composed with measured word replay; the final refined word need not pass
/// that restricted word matcher again (T/Tdg identities change its word).
class VerifiedCandidate {
public:
  enum class Proof { MeasuredWordReplay, MeasuredReplayAndLocalChannels };
  const std::vector<Gate> &gates() const { return circuit; }
  Proof proof() const { return checkedProof; }
  std::vector<Gate> takeGates() && { return std::move(circuit); }

private:
  friend class MeasuredCandidateSink;
  VerifiedCandidate(std::vector<Gate> gates, Proof proof)
      : circuit(std::move(gates)), checkedProof(proof) {}
  std::vector<Gate> circuit;
  Proof checkedProof;
};

/// Compatibility adapter for the existing measured selection policy. A solver
/// submits a complete native network including readout. The sink performs the
/// original proof, optional checked local refinement, T guard and ranking in
/// their original order, using the same shared budget and charges.
///
/// The problem and options must outlive this sink. A candidate is certified for
/// this sink's problem only; its proof enum is descriptive, not a transferable
/// certificate for another problem. New representation providers must establish
/// an explicit source-to-alternative proof before composing with this adapter.
class MeasuredCandidateSink {
public:
  MeasuredCandidateSink(MeasuredProblem problem, const Options &options,
                        SearchBudget *budget);
  const std::string &error() const { return inputError; }
  SubmissionResult submit(std::vector<Gate> gates);
  /// Retains the original five-metric guard and consumes the selected value.
  /// Exhaustion after a successful submission does not invalidate that value.
  std::optional<VerifiedCandidate> takeBest();

private:
  MeasuredProblem problem;
  const Options &options;
  SearchBudget *budget;
  std::vector<Pauli> outcomes;
  uint64_t inputTCount = 0;
  std::string inputError;
  std::optional<VerifiedCandidate> best, rawBest, refinedRawBest;
};

} // namespace cudaq::pauli::graph::detail
