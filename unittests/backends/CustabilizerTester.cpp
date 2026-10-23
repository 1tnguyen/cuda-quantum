/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "CustabilizerCircuitSimulator.cpp"
#include "StimCircuitSimulator.cpp"
#include "common/AnalysisScope.h"
#include "common/ExecutionContext.h"
#include "nvqir/QIRTypes.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

/// Wrapper to expose protected methods for testing.
class CustabilizerSimulatorTester
    : public nvqir::custabilizer::CustabilizerCircuitSimulator {
public:
  using GateApplicationTask =
      nvqir::CircuitSimulatorBase<double>::GateApplicationTask;

  void applyNamedGate(const std::string &name,
                      const std::vector<std::size_t> &controls,
                      const std::vector<std::size_t> &targets) {
    flushGateQueue();
    GateApplicationTask task(name, {}, controls, targets, {});
    applyGate(task);
  }

  void resetToZero() { setToZeroState(); }

  using nvqir::custabilizer::CustabilizerCircuitSimulator::applyNoise;
  cudaq::ExecutionResult sampleNow(const std::vector<std::size_t> &qubits,
                                   int shots, bool sequential = true) {
    flushGateQueue();
    return sample(qubits, shots, sequential);
  }
  bool hasCpuFrames() const { return bool(sampleSim); }
  bool hasGpuRunner() const { return bool(frameRunner); }
  std::int64_t gpuShots() const { return frameRunner->shots(); }
  std::size_t gpuApplications() const {
    return frameRunner->circuitApplications();
  }
  bool onlineMeasurement(std::size_t qubit) { return measureQubit(qubit); }

  /// @brief Render `recordedCircuit` as a Stim-format string for inspection
  /// from the test body.
  std::string recordedCircuitText() const {
    std::stringstream ss;
    ss << recordedCircuit;
    return ss.str();
  }
};

// QEC declaration ops are wired into the Stim NVQIR backend by the runtime
// adapters in `runtime/nvqir/NVQIR.cpp`. Declared here so the test runner
// picks up the definitions from libnvqir; tests then exercise the full path
// from `extern "C"` symbol → `Result**`-to-`int64_t`-vector conversion →
// dispatch to the active simulator's QEC virtuals.
extern "C" void __quantum__qis__detector(Result **results, std::int64_t count);
extern "C" void __quantum__qis__logical_observable(Result **results,
                                                   std::int64_t count,
                                                   std::int64_t obs_index);
extern "C" void __quantum__qis__pair_detectors(Result **prev_results,
                                               std::int64_t prev_count,
                                               Result **curr_results,
                                               std::int64_t curr_count);
extern "C" void __quantum__rt__clear_result_maps();

/// @brief Encode an `int64_t` chronological measurement index as the
/// `Result*` bit pattern the QIR ABI delivers to the QEC runtime adapter.
/// Mirrors the lowering pattern in `cudaq/test/Optimizer/qir_api_qec.qke`.
static Result *measureIndexAsResultPtr(std::int64_t i) {
  return reinterpret_cast<Result *>(static_cast<std::intptr_t>(i));
}

CUDAQ_TEST(CliffordSemantics, TwoQubitPauliProducts) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();

  sim.applyNamedGate("IX", {}, {q0, q1});
  EXPECT_EQ(false, sim.mz(q0));
  EXPECT_EQ(true, sim.mz(q1));
}

CUDAQ_TEST(CliffordSemantics, IDGateMapping) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  sim.applyNamedGate("ID", {}, {q0});
  EXPECT_EQ(false, sim.mz(q0));
}

CUDAQ_TEST(CliffordSemantics, SetToZeroStatePreservesSimulators) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  for (int i = 0; i < 3; i++) {
    sim.x(q0);
    EXPECT_EQ(true, sim.mz(q0));
    sim.resetToZero();
  }
  EXPECT_EQ(false, sim.mz(q0));
}

CUDAQ_TEST(QecCompatibility, DetectorEmitsRecordReferences) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);

  const std::int64_t indices[] = {0, 1};
  sim.detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos)
      << "recordedCircuit was:\n"
      << text;
}

CUDAQ_TEST(QecCompatibility, LogicalObservableAccumulates) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);
  sim.mz(q2);

  const std::int64_t idsA[] = {0, 1};
  sim.logical_observable(idsA, 2, /*observable_index=*/0);
  const std::int64_t idsB[] = {2};
  sim.logical_observable(idsB, 1, /*observable_index=*/0);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("OBSERVABLE_INCLUDE(0) rec[-1]"), std::string::npos)
      << text;
}

CUDAQ_TEST(QecCompatibility, PairDetectorsEmitsOnePerPair) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  auto q3 = sim.allocateQubit();
  sim.mz(q0); // index 0
  sim.mz(q1); // index 1
  sim.mz(q2); // index 2
  sim.mz(q3); // index 3

  const std::int64_t prev[] = {0, 1};
  const std::int64_t curr[] = {2, 3};
  sim.pair_detectors(prev, curr, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-4] rec[-2]"), std::string::npos) << text;
  EXPECT_NE(text.find("DETECTOR rec[-3] rec[-1]"), std::string::npos) << text;
}

CUDAQ_TEST(QecCompatibility, DetectorThrowsOnOutOfRangeIndex) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1); // num_measurements = 2

  const std::int64_t mixed[] = {0, 9};
  try {
    sim.detector(mixed, 2);
    FAIL() << "expected std::out_of_range";
  } catch (const std::out_of_range &e) {
    const std::string what(e.what());
    EXPECT_NE(what.find("9"), std::string::npos) << what;
    EXPECT_NE(what.find("[0, 2)"), std::string::npos) << what;
  }

  EXPECT_EQ(sim.recordedCircuitText().find("DETECTOR"), std::string::npos)
      << sim.recordedCircuitText();
}

CUDAQ_TEST(QecCompatibility, PairDetectorsAllOrNothingOnOutOfRange) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  auto q3 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);
  sim.mz(q2);
  sim.mz(q3); // num_measurements = 4

  const std::int64_t prev[] = {0, 1};
  const std::int64_t curr[] = {2, 99};
  EXPECT_ANY_THROW(sim.pair_detectors(prev, curr, 2));
  EXPECT_EQ(sim.recordedCircuitText().find("DETECTOR"), std::string::npos)
      << sim.recordedCircuitText();
}

CUDAQ_TEST(QecCompatibility, PairDetectorsAdapterRejectsSizeMismatch) {
  Result *prev[1] = {measureIndexAsResultPtr(0)};
  Result *curr[2] = {measureIndexAsResultPtr(1), measureIndexAsResultPtr(2)};
  EXPECT_ANY_THROW(__quantum__qis__pair_detectors(prev, 1, curr, 2));
}

CUDAQ_TEST(QecCompatibility, AdapterRejectsNegativeCount) {
  Result *results[1] = {measureIndexAsResultPtr(0)};
  EXPECT_ANY_THROW(__quantum__qis__detector(results, /*count=*/-1));
  EXPECT_ANY_THROW(__quantum__qis__logical_observable(results, /*count=*/-1,
                                                      /*obs_index=*/0));
}

CUDAQ_TEST(QecCompatibility, AdapterRejectsNullBufferWithPositiveCount) {
  EXPECT_ANY_THROW(__quantum__qis__detector(nullptr, /*count=*/2));
}

CUDAQ_TEST(QecCompatibility, AdapterRejectsNegativeObservableIndex) {
  Result *results[1] = {measureIndexAsResultPtr(0)};
  EXPECT_ANY_THROW(__quantum__qis__logical_observable(results, /*count=*/1,
                                                      /*obs_index=*/-1));
}

// End-to-end coverage of the NVQIR adapter
CUDAQ_TEST(QecCompatibility, AdapterAcceptsRawChronologicalIndices) {
  // Clear stale handle-to-index maps left by earlier tests in this process
  // so the raw chronological indices below are not remapped.
  __quantum__rt__clear_result_maps();
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);
  cudaq::detail::AnalysisScope scope{"stim_qec_adapter_test", sim, {}};

  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);

  Result *indices[] = {measureIndexAsResultPtr(0), measureIndexAsResultPtr(1)};
  __quantum__qis__detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos) << text;
}

/// @brief RAII guard for the thread-local execution-context pointer used by
/// the simulator. Without this, an `ASSERT_*` failure inside the test would
/// abort before `resetExecutionContext`, leaving the global pointer dangling
/// to a destructed `ExecutionContext` and corrupting downstream tests.
class ScopedExecutionContext {
public:
  explicit ScopedExecutionContext(cudaq::ExecutionContext &ctx) {
    cudaq::detail::setExecutionContext(&ctx);
  }
  ~ScopedExecutionContext() { cudaq::detail::resetExecutionContext(); }
  ScopedExecutionContext(const ScopedExecutionContext &) = delete;
  ScopedExecutionContext &operator=(const ScopedExecutionContext &) = delete;
};

CUDAQ_TEST(QecCompatibility, DetectorFlushesPendingSampleMeasurements) {
  CustabilizerSimulatorTester sim;
  sim.setRandomSeed(42);

  cudaq::ExecutionContext ctx("sample", /*shots=*/1);
  ctx.explicitMeasurements = true;
  sim.configureExecutionContext(ctx);
  ScopedExecutionContext ctxGuard{ctx};

  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0, "h0");
  sim.mz(q1, "h1");

  // Before the QEC call, the deferred `M` op is not yet in the recorded
  // circuit.
  EXPECT_EQ(sim.recordedCircuitText().find("M "), std::string::npos);

  const std::int64_t indices[] = {0, 1};
  sim.detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  const auto mPos = text.find("M ");
  const auto dPos = text.find("DETECTOR");
  ASSERT_NE(mPos, std::string::npos) << text;
  ASSERT_NE(dPos, std::string::npos) << text;
  EXPECT_LT(mPos, dPos) << "M must land before DETECTOR:\n" << text;
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos) << text;
}

CUDAQ_TEST(CustabilizerGPU, SamplesRandomCliffordFramesWithoutCpuFrames) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 129);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.setRandomSeed(42);
  auto q = sim.allocateQubits(2);
  sim.h(q[0]);
  sim.x({q[0]}, q[1]);
  auto result = sim.sampleNow(q, 129);
  EXPECT_FALSE(sim.hasCpuFrames());
  EXPECT_TRUE(sim.hasGpuRunner());
  EXPECT_EQ(result.sequentialData.size(), 129);
  EXPECT_EQ(result.counts.size(), 2);
  EXPECT_GT(result.counts["00"], 20);
  EXPECT_GT(result.counts["11"], 20);
  EXPECT_EQ(result.counts["00"] + result.counts["11"], 129);
}

CUDAQ_TEST(CustabilizerGPU, OptionalSequentialResults) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 65);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubits(65);
  sim.x(q[0]);
  sim.x(q[64]);
  auto result = sim.sampleNow(q, 65, false);
  EXPECT_TRUE(result.sequentialData.empty());
  EXPECT_EQ(result.counts.at("1" + std::string(63, '0') + "1"), 65);
}

CUDAQ_TEST(CustabilizerResults, PackedConversionMatchesScalarReference) {
  // Include partial bytes, words and tiles, nonzero measurement prefixes and
  // unaligned trajectory starts. Padding bits are deliberately randomized.
  std::mt19937_64 random(1997);
  for (std::size_t width :
       {0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 801, 5185}) {
    for (std::size_t totalShots : {1, 31, 32, 33, 63, 64, 65, 129, 257}) {
      const auto words = (totalShots + 31) / 32;
      for (std::size_t prefix : {0, 3}) {
        std::vector<bool> reference(width + prefix);
        for (std::size_t m = 0; m < reference.size(); ++m)
          reference[m] = random() & 1;
        std::vector<std::uint32_t> packed(reference.size() * words);
        for (auto &word : packed)
          word = random();
        for (std::size_t firstShot : {0, 1, 31, 32, 33, 63, 64, 65}) {
          if (firstShot >= totalShots)
            continue;
          const auto shots = totalShots - firstShot;
          cudaq::ExecutionResult expected;
          for (std::size_t s = firstShot; s < totalShots; ++s) {
            std::string bits;
            for (std::size_t m = prefix; m < reference.size(); ++m) {
              const auto flip = (packed[m * words + s / 32] >> (s % 32)) & 1;
              bits.push_back((flip ^ reference[m]) ? '1' : '0');
            }
            ++expected.counts[bits];
            expected.sequentialData.push_back(std::move(bits));
          }
          for (bool sequential : {false, true}) {
            const auto actual = nvqir::custabilizer::makeSampleResult(
                packed, words, reference, prefix, firstShot, shots, sequential);
            ASSERT_EQ(actual.counts, expected.counts)
                << "width=" << width << " shots=" << shots
                << " firstShot=" << firstShot << " prefix=" << prefix;
            if (sequential)
              ASSERT_EQ(actual.sequentialData, expected.sequentialData);
            else
              ASSERT_TRUE(actual.sequentialData.empty());
          }
        }
      }
    }
  }
}

CUDAQ_TEST(CustabilizerGPU, ResetClearsCircuitAndPreservesAllocatedRunner) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 33);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubits(2);
  sim.x(q[0]);
  EXPECT_EQ(sim.sampleNow(q, 33).counts.at("10"), 33);
  sim.resetToZero();
  EXPECT_EQ(sim.sampleNow(q, 33).counts.at("00"), 33);
  EXPECT_FALSE(sim.hasCpuFrames());
  EXPECT_TRUE(sim.hasGpuRunner());
}

CUDAQ_TEST(CustabilizerGPU, DetectorRowsDoNotChangeRawMeasurementLayout) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 37);
  context.explicitMeasurements = true;
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubits(2);
  sim.x(q[0]);
  sim.mz(q[0], "first");
  sim.mz(q[1], "second");
  const std::int64_t indices[] = {0, 1};
  sim.detector(indices, 2);
  auto result = sim.sampleNow({}, 37);
  EXPECT_EQ(result.counts.at("10"), 37);
  for (const auto &shot : result.sequentialData)
    EXPECT_EQ(shot.size(), 2);
}

template <typename Sim>
cudaq::msm_result buildCompatibilityMsm(Sim &sim) {
  cudaq::msm_policy policy;
  policy.dimensions = std::make_pair(1, 3);
  sim.configureExecutionContext(policy);
  auto q = sim.allocateQubit();
  static_cast<nvqir::CircuitSimulator &>(sim).applyNoise(
      cudaq::depolarization_channel(0.3), {q});
  sim.mz(q);
  return static_cast<nvqir::CircuitSimulator &>(sim).finalizeExecutionContext(
      policy);
}

CUDAQ_TEST(CustabilizerCompatibility, MsmMatchesUnchangedStim) {
  nvqir::StimCircuitSimulator reference;
  CustabilizerSimulatorTester sim;
  auto expected = buildCompatibilityMsm(reference);
  auto actual = buildCompatibilityMsm(sim);
  EXPECT_EQ(actual.probabilities, expected.probabilities);
  EXPECT_EQ(actual.probability_error_ids, expected.probability_error_ids);
  EXPECT_EQ(actual.samples.sequential_data(),
            expected.samples.sequential_data());
  EXPECT_FALSE(sim.hasGpuRunner());
}

CUDAQ_TEST(CustabilizerGPU, RejectsInvalidShotsBeforeRecordingMeasurements) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 65);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubit();
  const auto before = sim.recordedCircuitText();
  for (int shots : {-1, 0, 66}) {
    EXPECT_THROW(sim.sampleNow({q}, shots), std::invalid_argument);
    EXPECT_EQ(sim.recordedCircuitText(), before);
  }
  EXPECT_EQ(sim.sampleNow({q}, 33).counts.at("0"), 33);
  EXPECT_EQ(sim.gpuShots(), 33);
}

CUDAQ_TEST(CustabilizerGPU, RejectsOnlineMeasurementWithoutNoisyFrames) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 65);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubit();
  const auto before = sim.recordedCircuitText();
  EXPECT_THROW(sim.onlineMeasurement(q), std::runtime_error);
  EXPECT_EQ(sim.recordedCircuitText(), before);
}

CUDAQ_TEST(CustabilizerGPU, ValidatesAndNormalizesOverlappingTargets) {
  using nvqir::custabilizer::FrameRunner;
  stim::Circuit circuit("R 0 1 2\nX_ERROR(1) 0\nCX 0 1 1 2\nM 0 1 2");
  EXPECT_THROW(FrameRunner::validateParallelTargets(circuit),
               std::invalid_argument);
  FrameRunner runner;
  const auto frames = runner.sample(circuit, 65, 12);
  ASSERT_EQ(frames.size(), 9);
  for (std::size_t m = 0; m < 3; ++m)
    for (std::size_t shot = 0; shot < 65; ++shot)
      EXPECT_EQ((frames[m * 3 + shot / 32] >> (shot % 32)) & 1u, 1u);
}

CUDAQ_TEST(CustabilizerPTSBE, UnevenMasksShareOneCircuitApplication) {
  using namespace cudaq::ptsbe;
  using cudaq::ptsbe::TraceInstructionType;
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("ptsbe-sample", 130);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.allocateQubit();
  PTSBatch batch;
  batch.trace = {{TraceInstructionType::Gate, "h", {0}, {}, {}},
                 {TraceInstructionType::Noise,
                  "depolarization_channel",
                  {0},
                  {},
                  {},
                  cudaq::depolarization_channel(0.3)},
                 {TraceInstructionType::Gate, "h", {0}, {}, {}},
                 {TraceInstructionType::Measurement, "mz", {0}, {}, {}}};
  batch.measureQubits = {0};
  batch.includeSequentialData = true;
  const std::vector<std::size_t> counts{1, 31, 0, 33, 65};
  const std::vector<std::size_t> operators{0, 1, 2, 2, 3};
  for (std::size_t t = 0; t < counts.size(); ++t)
    batch.trajectories.emplace_back(
        t,
        std::vector<cudaq::KrausSelection>{
            {1, {0}, "h", operators[t], operators[t] != 0}},
        0.2, counts[t]);
  const auto results = sim.sampleWithPTSBE(batch);
  ASSERT_EQ(results.size(), counts.size());
  EXPECT_EQ(sim.gpuApplications(), 1);
  EXPECT_EQ(sim.gpuShots(), 130);
  for (std::size_t t = 0; t < counts.size(); ++t) {
    const std::string expected = operators[t] >= 2 ? "1" : "0";
    EXPECT_EQ(results[t].get_total_shots(), counts[t]);
    EXPECT_EQ(results[t].count(expected), counts[t]);
    EXPECT_EQ(results[t].sequential_data(),
              std::vector<std::string>(counts[t], expected));
  }
  batch.includeSequentialData = false;
  const auto again = sim.sampleWithPTSBE(batch);
  EXPECT_EQ(sim.gpuApplications(), 2);
  for (std::size_t t = 0; t < counts.size(); ++t) {
    EXPECT_EQ(again[t].to_map(), results[t].to_map());
    EXPECT_TRUE(again[t].sequential_data().empty());
  }
}

CUDAQ_TEST(CustabilizerPTSBE, TwoQubitPaulisRespectTargetAndMeasurementOrder) {
  using namespace cudaq::ptsbe;
  using cudaq::ptsbe::TraceInstructionType;
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("ptsbe-sample", 136);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.allocateQubits(2);
  PTSBatch batch;
  batch.trace = {{TraceInstructionType::Noise,
                  "depolarization2",
                  {1, 0},
                  {},
                  {},
                  cudaq::depolarization2(0.3)}};
  batch.measureQubits = {0, 1};
  const std::string paulis = "IXYZ";
  for (std::size_t k = 0; k < 16; ++k)
    batch.trajectories.emplace_back(
        k, std::vector<cudaq::KrausSelection>{{0, {1, 0}, "cx", k, k != 0}},
        1. / 16., k + 1);
  const auto results = sim.sampleWithPTSBE(batch);
  ASSERT_EQ(results.size(), 16);
  EXPECT_EQ(sim.gpuApplications(), 1);
  for (std::size_t k = 0; k < 16; ++k) {
    std::string expected;
    for (auto p : {paulis[k % 4], paulis[k / 4]})
      expected += p == 'X' || p == 'Y' ? '1' : '0';
    EXPECT_EQ(results[k].count(expected), k + 1);
  }
}

CUDAQ_TEST(CustabilizerPTSBE,
           RandomMeasurementsAreReproducibleAndZeroShotsStayEmpty) {
  using namespace cudaq::ptsbe;
  using cudaq::ptsbe::TraceInstructionType;
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("ptsbe-sample", 129);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.allocateQubit();
  PTSBatch batch;
  batch.trace = {{TraceInstructionType::Gate, "h", {0}, {}, {}}};
  batch.measureQubits = {0};
  batch.includeSequentialData = true;
  batch.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{}, 1.,
                                  129);
  batch.trajectories.emplace_back(1, std::vector<cudaq::KrausSelection>{}, 0.,
                                  0);
  sim.setRandomSeed(123);
  const auto first = sim.sampleWithPTSBE(batch);
  sim.setRandomSeed(123);
  const auto second = sim.sampleWithPTSBE(batch);
  EXPECT_EQ(first[0].sequential_data(), second[0].sequential_data());
  EXPECT_GT(first[0].count("0"), 20);
  EXPECT_GT(first[0].count("1"), 20);
  EXPECT_EQ(first[1].get_total_shots(), 0);
  batch.trajectories[0].num_shots = 0;
  const auto applications = sim.gpuApplications();
  const auto empty = sim.sampleWithPTSBE(batch);
  ASSERT_EQ(empty.size(), 2);
  EXPECT_EQ(empty[0].get_total_shots(), 0);
  EXPECT_EQ(sim.gpuApplications(), applications);
}

CUDAQ_TEST(CustabilizerPTSBE,
           NonPauliCliffordFallbackUsesEachTrajectoryShotCount) {
  using namespace cudaq::ptsbe;
  using cudaq::ptsbe::TraceInstructionType;
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("ptsbe-sample", 34);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.allocateQubit();
  const double a = std::sqrt(0.5);
  cudaq::kraus_channel channel(
      {cudaq::complex{a, 0.}, {0., 0.}, {0., 0.}, {a, 0.}},
      {cudaq::complex{0.5, 0.}, {0.5, 0.}, {0.5, 0.}, {-0.5, 0.}});
  channel.op_names = {"id", "h"};
  PTSBatch batch;
  batch.trace = {{TraceInstructionType::Noise, "custom", {0}, {}, {}, channel}};
  batch.measureQubits = {0};
  batch.trajectories = {{0, {{0, {0}, "id", 0, false}}, 0.5, 1},
                        {1, {{0, {0}, "id", 1, true}}, 0.5, 33}};
  const auto results = sim.sampleWithPTSBE(batch);
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].count("0"), 1);
  EXPECT_EQ(results[1].get_total_shots(), 33);
  EXPECT_GT(results[1].count("0"), 0);
  EXPECT_GT(results[1].count("1"), 0);
  EXPECT_EQ(sim.gpuApplications(), 2);
  EXPECT_EQ(sim.gpuShots(), 33);
}

CUDAQ_TEST(CustabilizerPTSBE, CustomPauliMatricesDoNotRequireGateNames) {
  using namespace cudaq::ptsbe;
  using cudaq::ptsbe::TraceInstructionType;
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("ptsbe-sample", 65);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  sim.allocateQubit();
  const double a = std::sqrt(0.5);
  // The second unitary is iZ: its global phase is irrelevant to the frame.
  cudaq::kraus_channel channel(
      {cudaq::complex{a, 0.}, {0., 0.}, {0., 0.}, {a, 0.}},
      {cudaq::complex{0., a}, {0., 0.}, {0., 0.}, {0., -a}});
  PTSBatch batch;
  batch.trace = {{TraceInstructionType::Gate, "h", {0}, {}, {}},
                 {TraceInstructionType::Noise, "custom", {0}, {}, {}, channel},
                 {TraceInstructionType::Gate, "h", {0}, {}, {}}};
  batch.measureQubits = {0};
  batch.trajectories = {{0, {{1, {0}, "h", 0, false}}, 0.5, 1},
                        {1, {{1, {0}, "h", 1, true}}, 0.5, 64}};
  const auto results = sim.sampleWithPTSBE(batch);
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].count("0"), 1);
  EXPECT_EQ(results[1].count("1"), 64);
  EXPECT_EQ(sim.gpuApplications(), 1);
}
