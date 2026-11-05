/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#include "SynthesisCompletion.h"
#include "SynthesisLinearFour.h"
#include "SynthesisNativeRefinement.h"
#include "SynthesisOpen.h"
#include <algorithm>
#include <cmath>

namespace cudaq::pauli::graph::detail {
namespace completion_detail {
using native_refinement_detail::add;
using native_refinement_detail::multiply;
using native_refinement_detail::sameWord;
using native_refinement_detail::slice;
struct Source {
  unsigned width;
  std::vector<Gate> rotations;
  std::vector<Pauli> observables;
  std::vector<unsigned> readouts;
  MeasuredProblem problem() const {
    return {width, rotations, observables, readouts};
  }
  bool matches(const MeasuredProblem &other) const {
    return width == other.width && sameWord(rotations, other.rotations) &&
           observables == other.observables && readouts == other.readoutWires;
  }
};
bool finite(const CostRank &rank) {
  return std::all_of(rank.begin(), rank.end(),
                     [](double value) { return std::isfinite(value); });
}
uint64_t rowBytes(unsigned width) {
  return sizeof(Gate) + sizeof(Pauli) + 80 + 16 * (uint64_t(width) / 64 + 1);
}
uint64_t sourceBytes(const MeasuredProblem &source) {
  return multiply(source.rotations.size() + source.observables.size() +
                      source.readoutWires.size() + 1,
                  rowBytes(source.width));
}
bool boundedRows(const MeasuredProblem &source) {
  for (const auto &gate : source.rotations)
    if (!gate.pauli.x.fits(source.width) || !gate.pauli.z.fits(source.width))
      return false;
  for (const auto &row : source.observables)
    if (!row.x.fits(source.width) || !row.z.fits(source.width))
      return false;
  return true;
}
} // namespace completion_detail
namespace cd = completion_detail;

MeasuredProblem CompletionRoute::problem() const { return source->problem(); }
bool CertifiedCompletion::matchesSource(const MeasuredProblem &source) const {
  return route && route->original && route->original->matches(source);
}
bool completionAdmissible(const CostVector &cost, const CostVector &anchor,
                          CompletionPolicy policy) {
  return cost.gates <= anchor.gates && cost.t <= anchor.t &&
         (policy == CompletionPolicy::GatesAndT ||
          (cost.twoQubit <= anchor.twoQubit && cost.depth <= anchor.depth &&
           cost.tDepth <= anchor.tDepth));
}

struct CompletionSession::Impl {
  struct Entry {
    CompletionRouteHandle route;
    RegionalPlan plan;
    CostRank rank;
    CostRank scorerRank;
    CompletionStatus status;
  };
  Options options;
  CompletionLimits limits;
  CompletionPolicy policy;
  CostVector anchor;
  CostRank anchorRank;
  CompletionStatus status = CompletionStatus::Complete;
  std::string error;
  uint64_t ownedBytes = 0;
  std::vector<CompletionRouteHandle> routes;
  std::vector<Entry> cache;
  std::optional<CertifiedCompletion> best;
  std::shared_ptr<const LinearFourDictionary> dictionary;
  void fatal(std::string message) {
    status = CompletionStatus::InvalidProof;
    error = std::move(message);
    best.reset();
    cache.clear();
  }
};

CompletionSession::CompletionSession(const MeasuredProblem &source,
                                     const Options &options,
                                     const CostVector &anchor,
                                     const CostRank &anchorRank,
                                     const CompletionLimits &limits,
                                     SearchBudget &parent,
                                     CompletionPolicy policy)
    : impl(std::make_unique<Impl>()) {
  auto &p = *impl;
  p.options = options;
  p.options.pruneUnobserved = false;
  p.limits = limits;
  p.policy = policy;
  p.anchor = anchor;
  p.anchorRank = anchorRank;
  p.limits.maxRoutes = std::min(limits.maxRoutes, 8u);
  p.limits.maxPlans = std::min(limits.maxPlans, 32u);
  p.limits.maxNativeCandidates = std::min(limits.maxNativeCandidates, 16u);
  p.options.maxGates = std::min(options.maxGates, limits.maxGates);
  auto unsupported = [&](const char *message) {
    p.status = CompletionStatus::Unsupported;
    p.error = message;
  };
  if (!source.width || source.width > std::min(limits.maxWidth, 4096u) ||
      source.rotations.size() > 32768 ||
      source.rotations.size() > p.options.maxGates ||
      source.observables.size() > 4096 || !p.limits.maxRoutes ||
      !p.limits.maxPlans || !p.options.maxGates ||
      (!options.nativeCX && !options.nativeCZ) ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(SynthesisObjective::TDepth) ||
      !cd::finite(anchorRank) ||
      (policy != CompletionPolicy::AllFive &&
       policy != CompletionPolicy::GatesAndT)) {
    unsupported("invalid completion session limits/anchor");
    return;
  }
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0) {
      unsupported("invalid completion native cost profile");
      return;
    }
  auto work = cd::slice(parent, limits.maxWork);
  if (!work || !work->take(cd::add(
                   cd::sourceBytes(source),
                   cd::multiply(16 * uint64_t(source.observables.size() + 1) *
                                    (source.observables.size() + 1),
                                uint64_t(source.width) / 64 + 1)))) {
    p.status = CompletionStatus::Exhausted;
    return;
  }
  if (!cd::boundedRows(source) || !representation::validProblem(source)) {
    unsupported("completion needs a normalized measured source");
    return;
  }
  // The remaining half covers live completion scratch/providers. A final
  // admitted word has its own worst-case slot, independent of the plan cache.
  p.ownedBytes =
      cd::add(cd::sourceBytes(source),
              cd::multiply(p.options.maxGates, cd::rowBytes(source.width)));
  p.ownedBytes = cd::add(p.ownedBytes, 65536);
  if (p.ownedBytes > limits.maxLiveBytes / 2) {
    p.status = CompletionStatus::Exhausted;
    return;
  }
  auto owned = std::make_shared<cd::Source>(cd::Source{
      source.width, source.rotations, source.observables, source.readoutWires});
  auto route = std::shared_ptr<CompletionRoute>(new CompletionRoute);
  route->source = route->original = std::move(owned);
  p.routes.reserve(p.limits.maxRoutes);
  p.cache.reserve(p.limits.maxPlans);
  p.routes.push_back(std::move(route));
}
CompletionSession::~CompletionSession() = default;
CompletionStatus CompletionSession::status() const { return impl->status; }
const std::string &CompletionSession::error() const { return impl->error; }
CompletionRouteHandle CompletionSession::originalRoute() const {
  return impl->routes.empty() ? nullptr : impl->routes.front();
}
const CertifiedCompletion *CompletionSession::best() const {
  return impl->best ? &*impl->best : nullptr;
}

CompletionRouteResult
CompletionSession::checkIndexedRoute(const RepresentationAlternative &proposal,
                                     SearchBudget &parent) {
  auto &p = *impl;
  if (p.status != CompletionStatus::Complete)
    return {p.status, {}, p.error};
  if (p.routes.size() >= p.limits.maxRoutes)
    return {CompletionStatus::Exhausted, {}, "completion route cap"};
  const auto &root = p.routes.front()->original;
  MeasuredProblem alternate{proposal.width, proposal.rotations,
                            proposal.observables, proposal.readoutWires};
  if (proposal.width != root->width || proposal.rotations.size() > 32768 ||
      proposal.observables.size() > 4096 ||
      proposal.certificate.size() > 4096 || !cd::boundedRows(alternate)) {
    p.fatal("indexed completion route has invalid source/storage binding");
    return {p.status, {}, p.error};
  }
  uint64_t bytes = cd::sourceBytes(alternate);
  auto work = cd::slice(parent, p.limits.maxWork);
  if (bytes > p.limits.maxLiveBytes / 2 - p.ownedBytes || !work ||
      !work->take(bytes))
    return {CompletionStatus::Exhausted, {}, "indexed route storage/work cap"};
  auto checked = verifyRepresentation(root->problem(), proposal, &*work);
  if (checked.status == RepresentationCheckStatus::InvalidProof) {
    p.fatal(checked.error);
    return {p.status, {}, p.error};
  }
  if (!checked)
    return {CompletionStatus::Exhausted, {}, checked.error};
  auto route = std::shared_ptr<CompletionRoute>(new CompletionRoute);
  route->original = root;
  route->source = std::make_shared<cd::Source>(
      cd::Source{proposal.width, proposal.rotations, proposal.observables,
                 proposal.readoutWires});
  route->isIndexed = true;
  p.ownedBytes += bytes;
  p.routes.push_back(route);
  return {CompletionStatus::Complete, std::move(route), {}};
}

CompletionResult CompletionSession::complete(const CompletionRouteHandle &route,
                                             const RegionalPlan &plan,
                                             SearchBudget &parent,
                                             const CompletionVisitor &visitor) {
  return runCompletion(route, plan, parent, visitor, false);
}
CompletionResult CompletionSession::score(const CompletionRouteHandle &route,
                                          const RegionalPlan &plan,
                                          SearchBudget &parent) {
  return runCompletion(route, plan, parent, {}, true);
}
CompletionResult CompletionSession::runCompletion(
    const CompletionRouteHandle &route, const RegionalPlan &plan,
    SearchBudget &parent, const CompletionVisitor &visitor, bool scoring) {
  auto &p = *impl;
  CompletionResult result;
  if (p.status != CompletionStatus::Complete)
    return {p.status, {}, p.error};
  auto fail = [&](std::string error) {
    p.fatal(std::move(error));
    result.status = p.status;
    result.error = p.error;
    result.rank.reset();
    return result;
  };
  auto exhausted = [&](const char *error) {
    result.status = CompletionStatus::Exhausted;
    result.error = error;
    return result;
  };
  if (!route ||
      std::find(p.routes.begin(), p.routes.end(), route) == p.routes.end())
    return fail("completion route belongs to another source/session");
  auto source = route->problem();
  const unsigned width = source.width;
  const uint64_t limbs = uint64_t(width) / 64 + 1;
  const uint64_t rows = cd::rowBytes(width);
  auto work = cd::slice(parent, p.limits.maxWork);
  if (!work)
    return exhausted("completion work reserve");
  if (plan.prefix.size() > p.options.maxGates ||
      plan.observables.size() != source.observables.size())
    return fail("completion plan violates width/output/gate contract");
  if (!work->take(cd::multiply(plan.prefix.size() + plan.observables.size() + 1,
                               limbs)))
    return exhausted("completion input validation reserve");
  for (const auto &gate : plan.prefix)
    if (!gate.pauli.x.fits(width) || !gate.pauli.z.fits(width) ||
        !open_detail::validNative(width, gate, p.options))
      return fail("completion prefix is not a bounded native word");
  for (const auto &axis : plan.observables)
    if (!axis.x.fits(width) || !axis.z.fits(width))
      return fail("completion observable storage exceeds source width");
  std::optional<size_t> retryIndex;
  for (size_t index = 0; index < p.cache.size(); ++index) {
    const auto &old = p.cache[index];
    if (!work->take(cd::multiply(old.plan.prefix.size() + plan.prefix.size() +
                                     plan.observables.size() + 1,
                                 limbs)))
      return exhausted("completion cache comparison reserve");
    if (old.route == route && region_detail::samePlan(old.plan, plan)) {
      if (scoring || old.status == CompletionStatus::Complete)
        return {old.status, scoring ? old.scorerRank : old.rank, {}, 0, true};
      // A small planner scorer may have finished proof but exhausted optional
      // refinement. Permit its outside-callback retry with a larger budget.
      retryIndex = index;
      result.rank = old.rank;
      break;
    }
  }
  uint64_t retained =
      cd::multiply(plan.prefix.size() + plan.observables.size() + 1, rows);
  if (!retryIndex && (p.cache.size() >= p.limits.maxPlans ||
                      retained > p.limits.maxLiveBytes / 2 - p.ownedBytes))
    return exhausted("completion cache storage cap");
  const uint64_t scratchBytes = p.limits.maxLiveBytes / 2;
  const uint64_t dictionaryBytes = p.limits.enableFour ? 64ull << 20 : 0;
  if (scratchBytes <= dictionaryBytes)
    return exhausted("completion continuation scratch cap");
  const uint64_t temporaryBytes =
      p.limits.rankedContinuation && p.limits.enableNative
          ? cd::multiply(8 * uint64_t(p.options.maxGates), rows)
          : 0;
  const bool temporaryFunded = temporaryBytes < scratchBytes - dictionaryBytes;
  // Include simultaneously retained provider alternatives, source copies,
  // native lowering temporaries and vector growth. Delegate algorithm-specific
  // work/table caps to the checked providers; never increase their gate cap.
  const unsigned providerGateCap =
      temporaryFunded
          ? std::min<uint64_t>(
                p.options.maxGates,
                (scratchBytes - dictionaryBytes - temporaryBytes) /
                    cd::multiply(2 * uint64_t(p.limits.maxNativeCandidates) +
                                     (p.limits.rankedContinuation ? 16 : 48),
                                 rows))
          : 0;
  const unsigned rawGateCap = std::min<uint64_t>(
      p.options.maxGates, (scratchBytes - dictionaryBytes) / (12 * rows));
  if (plan.prefix.size() > rawGateCap)
    return exhausted("completion raw scratch cap");
  // Raw/sink publication is prepaid before measured proof; ranked mode uses
  // actual sizes once readout construction completes below.
  uint64_t stageCharge = cd::multiply(
      32 * uint64_t(rawGateCap) + source.observables.size() + width + 1, limbs);
  if (!work->take(cd::add(p.limits.rankedContinuation ? 0 : stageCharge,
                          cd::multiply(source.observables.size() + 1,
                                       (p.limits.rankedContinuation ? 1 : 16) *
                                           (uint64_t(width) * width + 1)))))
    return exhausted("readout/completion publication reserve");
  Options completionOptions = p.options;
  completionOptions.maxGates = rawGateCap;
  auto tail = graph::synthesizeReadout(width, plan.observables,
                                       source.readoutWires, completionOptions);
  if (!tail) {
    if (tail.stats.budgetExhausted)
      return exhausted("completion readout exhausted");
    return fail("completion readout failed: " + tail.error);
  }
  if (tail.gates.size() > rawGateCap - plan.prefix.size())
    return exhausted("completion native word cap");
  if (!work->take(
          cd::multiply(tail.gates.size() + 1,
                       cd::multiply(plan.observables.size() + 1, limbs))))
    return exhausted("readout independent replay reserve");
  auto outcomes = plan.observables;
  for (const auto &gate : tail.gates) {
    if (!region_detail::validClifford(width, gate) ||
        !open_detail::validNative(width, gate, p.options))
      return fail("completion readout emitted invalid native Clifford");
    for (auto &row : outcomes)
      row = tcount::detail::conjugate(row, gate);
  }
  for (unsigned i = 0; i < outcomes.size(); ++i)
    if (outcomes[i] != Pauli{{}, QubitSet::single(source.readoutWires[i])})
      return fail("completion changed signed ordered physical readout");
  if (p.limits.rankedContinuation &&
      !work->take(
          cd::multiply(64 * (tail.gates.size() + 2 * plan.prefix.size() +
                             plan.observables.size() + width + 1),
                       limbs)))
    return exhausted("actual raw publication reserve");
  std::vector<Gate> raw = plan.prefix;
  raw.insert(raw.end(), tail.gates.begin(), tail.gates.end());
  auto proof = cd::slice(*work, p.limits.maxWork);
  if (!proof)
    return exhausted("completion measured proof reserve");
  MeasuredCandidateSink sink(source, p.options, &*proof);
  if (!sink.error().empty()) {
    if (proof->stopped())
      return exhausted("completion sink construction exhausted");
    return fail(sink.error());
  }
  auto submitted = sink.submit(raw);
  if (submitted.status == SubmissionStatus::InvalidProof)
    return fail(submitted.error);
  if (submitted.status == SubmissionStatus::ProofBudgetExhausted)
    return exhausted("completion measured proof exhausted");
  if (submitted.status != SubmissionStatus::Verified) {
    result.status = CompletionStatus::Unsupported;
    result.error = "completion rejected by existing sink T guard";
    return result;
  }
  auto selected = sink.takeBest();
  if (!selected)
    return fail("verified completion sink lost its candidate");

  // Local chain scheduling never establishes final admissibility. Retain the
  // raw-relative G/T envelope only when choosing this single continuation.
  std::vector<Gate> rankedWord;
  CostVector rawCost{};
  CostRank rankedScore{};
  if (p.limits.rankedContinuation) {
    rankedWord = raw;
    rawCost = tcount::cost(width, raw);
    rankedScore = circuitScore(width, raw, p.options);
    if (cd::finite(rankedScore) && (!result.rank || rankedScore < *result.rank))
      result.rank = rankedScore;
  }

  auto offer = [&](const std::vector<Gate> &word, CompletionStage stage) {
    CertifiedCompletion candidate;
    candidate.route = route;
    candidate.word = word;
    candidate.evaluated = tcount::cost(width, word);
    candidate.score = circuitScore(width, word, p.options);
    candidate.completionStage = stage;
    // Admission never depends on local raw-relative ranking, an earlier
    // accepted route, or on the visitor's presence.
    if (cd::finite(candidate.score) &&
        completionAdmissible(candidate.metrics(), p.anchor, p.policy) &&
        candidate.rank() < p.anchorRank &&
        (!p.best || candidate.rank() < p.best->rank()))
      p.best = candidate;
    ++result.published;
    if (visitor)
      visitor(candidate);
    if (p.limits.rankedContinuation) {
      if (cd::finite(candidate.score) &&
          candidate.metrics().gates <= rawCost.gates &&
          candidate.metrics().t <= rawCost.t &&
          candidate.rank() < rankedScore) {
        rankedWord = candidate.gates();
        rankedScore = candidate.rank();
      }
      if (cd::finite(rankedScore) &&
          (!result.rank || rankedScore < *result.rank))
        result.rank = rankedScore;
    } else if (cd::finite(candidate.score) &&
               (!result.rank || candidate.rank() < *result.rank)) {
      result.rank = candidate.rank();
    }
  };
  offer(raw, CompletionStage::RawMeasured);
  offer(selected->gates(), CompletionStage::SinkChannels);
  if (!result.rank) {
    result.status = CompletionStatus::Unsupported;
    result.error = "completion native rank overflow";
    return result;
  }
  // Plan ownership/rank is retained independently of optional continuation.
  if (!retryIndex) {
    p.cache.push_back(
        {route, plan, *result.rank, *result.rank, CompletionStatus::Complete});
    p.ownedBytes += retained;
  }
  const size_t cacheIndex = retryIndex ? *retryIndex : p.cache.size() - 1;
  auto finish = [&]() {
    p.cache[cacheIndex].rank = *result.rank;
    if (!retryIndex)
      p.cache[cacheIndex].scorerRank = *result.rank;
    p.cache[cacheIndex].status = result.status;
    return result;
  };
  auto optionalExhausted = [&]() {
    result.status = CompletionStatus::Exhausted;
  };

  auto publicationCharge = [&](const std::vector<Gate> &word,
                               uint64_t candidateSize) {
    return cd::multiply(32 * (word.size() + candidateSize + width + 1), limbs);
  };
  auto providerSlice =
      [&](uint64_t cap, uint64_t publication) -> std::optional<SearchBudget> {
    if (!p.limits.rankedContinuation) {
      if (!work->take(publication))
        return std::nullopt;
      return cd::slice(*work, cap);
    }
    if (work->remaining() <= publication || !open_detail::hasTime(*work))
      return std::nullopt;
    // Reserve without billing nonexistent max-sized outputs. Providers own
    // their configured child caps inside this remaining-minus-reserve envelope.
    // Local exhaustion cannot consume publication's remaining work allowance.
    return SearchBudget(*work, work->remaining() - publication,
                        open_detail::halfTime(*work));
  };

  // Keep publication/binding outside provider slices. Expanded mode continues
  // both native-source lanes and every native result through Four. Ranked mode
  // chooses one chain while still admitting every checked stage independently.
  auto four = [&](const std::vector<Gate> &word) -> bool {
    if (!p.limits.enableFour || width < 4)
      return true;
    if (!providerGateCap || word.size() > providerGateCap) {
      optionalExhausted();
      return true;
    }
    auto stage = providerSlice(p.limits.fourWork,
                               p.limits.rankedContinuation
                                   ? publicationCharge(word, providerGateCap)
                                   : stageCharge);
    if (!stage) {
      optionalExhausted();
      return true;
    }
    // Providers already reserve a bounded child. Do not halve that configured
    // allowance a second time in the single ranked continuation chain.
    SearchBudget &providerBudget = *stage;
    LinearFourLimits limits;
    limits.maxGates = providerGateCap;
    limits.maxWork = p.limits.fourWork;
    if (!p.dictionary) {
      auto built = buildLinearFourDictionary(p.options, limits, providerBudget);
      if (built.status == LinearFourStatus::InvalidProof) {
        fail(built.error);
        return false;
      }
      p.dictionary = built.dictionary;
      if (!p.dictionary) {
        optionalExhausted();
        return true;
      }
    }
    ++result.fourSources;
    auto refined = optimizeLinearFourWindows(width, word, p.options, limits,
                                             p.dictionary, providerBudget);
    if (refined.status == LinearFourStatus::InvalidProof) {
      fail(refined.error);
      return false;
    }
    uint64_t actualPublication = 0;
    for (const auto &candidate : refined.alternatives)
      actualPublication = cd::add(
          actualPublication, publicationCharge(word, candidate.gates().size()));
    if (p.limits.rankedContinuation && !work->take(actualPublication)) {
      optionalExhausted();
      return true;
    }
    for (const auto &candidate : refined.alternatives) {
      if (!candidate.matchesSource(width, word)) {
        fail("Four completion certificate source mismatch");
        return false;
      }
      offer(candidate.gates(), CompletionStage::Four);
    }
    if (refined.status == LinearFourStatus::Exhausted)
      optionalExhausted();
    return true;
  };
  // Freeze this source before offer() can update the ranked continuation.
  const std::vector<Gate> rankedNativeSource =
      p.limits.rankedContinuation ? rankedWord : std::vector<Gate>{};
  std::vector<const std::vector<Gate> *> nativeSources;
  if (p.limits.rankedContinuation) {
    nativeSources.push_back(&rankedNativeSource);
  } else {
    nativeSources.push_back(&raw);
    if (!cd::sameWord(raw, selected->gates()))
      nativeSources.push_back(&selected->gates());
  }
  for (const auto *word : nativeSources) {
    if (p.limits.enableNative && p.limits.maxNativeCandidates) {
      // Reserve certificate binding/cost/publication independently of backend
      // work. Local backend exhaustion leaves that allowance available; the
      // enclosing session deadline remains authoritative.
      uint64_t publication =
          cd::multiply(p.limits.maxNativeCandidates,
                       p.limits.rankedContinuation
                           ? publicationCharge(*word, providerGateCap)
                           : stageCharge);
      if (!providerGateCap || word->size() > providerGateCap) {
        optionalExhausted();
      } else {
        auto stage = providerSlice(p.limits.nativeWork, publication);
        if (!stage) {
          optionalExhausted();
          continue;
        }
        SearchBudget &providerBudget = *stage;
        NativeRefinementLimits limits;
        limits.maxGates = providerGateCap;
        limits.maxTemporaryGates =
            p.limits.rankedContinuation ? p.options.maxGates : providerGateCap;
        limits.maxCandidates = p.limits.maxNativeCandidates;
        limits.maxWork = p.limits.nativeWork;
        ++result.nativeSources;
        auto refined = refineSynthesizedNative(width, *word, p.options, limits,
                                               providerBudget);
        if (refined.status == NativeRefinementStatus::InvalidProof)
          return fail(refined.error);
        uint64_t actualPublication = 0;
        for (const auto &candidate : refined.alternatives)
          actualPublication =
              cd::add(actualPublication,
                      publicationCharge(*word, candidate.gates().size()));
        if (p.limits.rankedContinuation && !work->take(actualPublication)) {
          optionalExhausted();
          continue;
        }
        for (const auto &candidate : refined.alternatives) {
          if (!candidate.matchesSource(width, *word))
            return fail("native completion certificate source mismatch");
          offer(candidate.gates(), CompletionStage::Native);
        }
        // All native candidates are published BEFORE optional Four work.
        if (!p.limits.rankedContinuation)
          for (const auto &candidate : refined.alternatives)
            if (!four(candidate.gates()))
              return result;
        if (refined.status == NativeRefinementStatus::Exhausted)
          optionalExhausted();
      }
    }
    if (!p.limits.rankedContinuation && !four(*word))
      return result;
  }
  if (p.limits.rankedContinuation) {
    const auto rankedFourSource = rankedWord;
    if (!four(rankedFourSource))
      return result;
  }
  return finish();
}
} // namespace cudaq::pauli::graph::detail
