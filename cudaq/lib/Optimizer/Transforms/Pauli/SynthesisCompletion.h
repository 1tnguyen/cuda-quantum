/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#pragma once

#include "SynthesisCost.h"
#include "SynthesisRegions.h"
#include "SynthesisRepresentations.h"
#include <functional>
#include <memory>

namespace cudaq::pauli::graph::detail {

enum class CompletionPolicy { AllFive, GatesAndT };
enum class CompletionStatus { Complete, Unsupported, Exhausted, InvalidProof };
enum class CompletionStage { RawMeasured, SinkChannels, Native, Four };
struct CompletionLimits {
  unsigned maxRoutes = 4, maxPlans = 32, maxWidth = 4096;
  unsigned maxGates = 262144, maxNativeCandidates = 16;
  uint64_t maxWork = 5000000000, maxLiveBytes = 1ull << 30;
  uint64_t nativeWork = 100000000, fourWork = 50000000;
  bool enableNative = true, enableFour = true;
  /// Follow one bounded ranked continuation chain. All checked intermediate
  /// words still reach global admission, independently of local chain choice.
  bool rankedContinuation = false;
};
namespace completion_detail {
struct Source;
}
class CompletionSession;

/// Owned, immutable route. An indexed route exists only after independent
/// verification against this session's original normalized measured source.
class CompletionRoute {
public:
  CompletionRoute(const CompletionRoute &) = delete;
  CompletionRoute &operator=(const CompletionRoute &) = delete;
  MeasuredProblem problem() const;
  bool indexed() const { return isIndexed; }

private:
  friend class CompletionSession;
  friend class CertifiedCompletion;
  CompletionRoute() = default;
  std::shared_ptr<const completion_detail::Source> original, source;
  bool isIndexed = false;
};
using CompletionRouteHandle = std::shared_ptr<const CompletionRoute>;

/// Proof-bearing completed word. Metrics/ranking never establish this proof.
/// Global phase is irrelevant only at the terminal measured boundary; native
/// and Four continuation certificates preserve their complete source unitary.
class CertifiedCompletion {
public:
  const std::vector<Gate> &gates() const { return word; }
  const CostVector &metrics() const { return evaluated; }
  const CostRank &rank() const { return score; }
  CompletionStage stage() const { return completionStage; }
  bool indexed() const { return route && route->indexed(); }
  bool matchesSource(const MeasuredProblem &) const;

private:
  friend class CompletionSession;
  CertifiedCompletion() = default;
  CompletionRouteHandle route;
  std::vector<Gate> word;
  CostVector evaluated;
  CostRank score;
  CompletionStage completionStage;
};
struct CompletionResult {
  CompletionStatus status = CompletionStatus::Complete;
  std::optional<CostRank> rank;
  std::string error;
  unsigned published = 0;
  bool cacheHit = false;
  unsigned nativeSources = 0, fourSources = 0;
};
struct CompletionRouteResult {
  CompletionStatus status = CompletionStatus::Complete;
  CompletionRouteHandle route;
  std::string error;
};
using CompletionVisitor = std::function<void(const CertifiedCompletion &)>;

/// The caller supplies costs of its already verified immutable incumbent.
/// This guard is admission policy only; it cannot manufacture a certificate.
bool completionAdmissible(const CostVector &, const CostVector &anchor,
                          CompletionPolicy);

/// Invocation-owned, bounded completion/cache service. Every checked stage is
/// admitted internally against the SAME anchor before local continuation
/// ranking. The optional visitor is observational, never required for safety
/// or final selection. Do not retain its borrowed candidate reference.
///
/// A complete RegionalPlan remains a proposal until this service checks it.
/// Exhaustion retains all prior completed results; InvalidProof is fatal and
/// clears the session's publication/cache. Callers must propagate that status
/// even if an earlier visitor invocation observed a valid candidate.
class CompletionSession {
public:
  CompletionSession(const MeasuredProblem &, const Options &,
                    const CostVector &anchor, const CostRank &anchorRank,
                    const CompletionLimits &, SearchBudget &,
                    CompletionPolicy = CompletionPolicy::AllFive);
  ~CompletionSession();
  CompletionSession(const CompletionSession &) = delete;
  CompletionSession &operator=(const CompletionSession &) = delete;
  CompletionStatus status() const;
  const std::string &error() const;
  CompletionRouteHandle originalRoute() const;
  CompletionRouteResult checkIndexedRoute(const RepresentationAlternative &,
                                          SearchBudget &);
  CompletionResult complete(const CompletionRouteHandle &, const RegionalPlan &,
                            SearchBudget &, const CompletionVisitor & = {});
  /// Freeze the first checked completion's heuristic rank, even if optional
  /// refinement exhausted. Later complete() retries may improve admission but
  /// never rewrite feedback already supplied to a planner.
  CompletionResult score(const CompletionRouteHandle &, const RegionalPlan &,
                         SearchBudget &);
  const CertifiedCompletion *best() const;

private:
  CompletionResult runCompletion(const CompletionRouteHandle &,
                                 const RegionalPlan &, SearchBudget &,
                                 const CompletionVisitor &, bool scoring);
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace cudaq::pauli::graph::detail
