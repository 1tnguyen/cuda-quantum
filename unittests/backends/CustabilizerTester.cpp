/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "CustabilizerCircuitSimulator.cpp"
#include "StimCircuitSimulator.h"
#include "common/AnalysisScope.h"
#include "common/ExecutionContext.h"
#include "nvqir/QIRTypes.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>

class CustabilizerFrameRunnerTester : public nvqir::CustabilizerFrameRunner {
public:
  CustabilizerFrameRunnerTester() : CustabilizerFrameRunner(1) {}

  std::vector<std::uint32_t> sample(const stim::Circuit &input,
                                    std::int64_t shots, std::uint64_t seed) {
    clear();
    apply(input);
    return getMeasurementTable(shots, true, seed);
  }
};

/// Wrapper to expose protected methods for testing.
class CustabilizerSimulatorTester : public nvqir::CustabilizerCircuitSimulator {
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

  using nvqir::CustabilizerCircuitSimulator::applyNoise;
  cudaq::ExecutionResult sampleNow(const std::vector<std::size_t> &qubits,
                                   int shots, bool sequential = true) {
    flushGateQueue();
    return sample(qubits, shots, sequential);
  }
  bool hasCpuFrames() const { return bool(sampleSim); }
  bool hasGpuRunner() const { return bool(frameRunner); }
  std::int64_t gpuShots() const { return frameRunner->allocated_shots; }
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
  // Include partial bytes, words and tiles, and nonzero measurement prefixes.
  // Padding bits are deliberately randomized.
  std::mt19937_64 random(1997);
  for (std::size_t width :
       {0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 801, 5185}) {
    for (std::size_t shots : {1, 31, 32, 33, 63, 64, 65, 129, 257}) {
      const auto words = (shots + 31) / 32;
      for (std::size_t prefix : {0, 3}) {
        std::vector<bool> reference(width + prefix);
        for (std::size_t m = 0; m < reference.size(); ++m)
          reference[m] = random() & 1;
        std::vector<std::uint32_t> packed(reference.size() * words);
        for (auto &word : packed)
          word = random();
        cudaq::ExecutionResult expected;
        for (std::size_t s = 0; s < shots; ++s) {
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
              packed, words, reference, prefix, shots, sequential);
          ASSERT_EQ(actual.counts, expected.counts)
              << "width=" << width << " shots=" << shots
              << " prefix=" << prefix;
          if (sequential)
            ASSERT_EQ(actual.sequentialData, expected.sequentialData);
          else
            ASSERT_TRUE(actual.sequentialData.empty());
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

CUDAQ_TEST(CustabilizerGPU, ExecutionEndReleasesRunnerAndAllowsReallocation) {
  CustabilizerSimulatorTester sim;
  for (auto [qubits, shots] : {std::pair{65, 129}, std::pair{1, 33}}) {
    cudaq::ExecutionContext context("sample", shots);
    sim.configureExecutionContext(context);
    ScopedExecutionContext scope(context);
    auto q = sim.allocateQubits(qubits);
    sim.x(q.back());
    EXPECT_EQ(sim.sampleNow({q.back()}, shots).counts.at("1"), shots);
    ASSERT_TRUE(sim.hasGpuRunner());
    EXPECT_EQ(sim.gpuShots(), shots);

    sim.endExecution();
    EXPECT_FALSE(sim.hasGpuRunner());
    EXPECT_FALSE(sim.hasCpuFrames());
  }
}

CUDAQ_TEST(CustabilizerGPU, BatchReusesRunnerUntilFinalExecution) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 33);
  context.totalIterations = 2;
  context.batchIteration = 0;
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubits(2);
  sim.x(q[0]);
  EXPECT_EQ(sim.sampleNow(q, 33).counts.at("10"), 33);
  sim.endExecution();
  ASSERT_TRUE(sim.hasGpuRunner());

  context.batchIteration = 1;
  sim.configureExecutionContext(context);
  q = sim.allocateQubits(2);
  EXPECT_EQ(sim.sampleNow(q, 33).counts.at("00"), 33);
  sim.endExecution();
  EXPECT_FALSE(sim.hasGpuRunner());
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

CUDAQ_TEST(CustabilizerGPU, RejectsMsmPolicies) {
  CustabilizerSimulatorTester sim;
  EXPECT_THROW(sim.configureExecutionContext(cudaq::msm_policy{}),
               std::runtime_error);
  EXPECT_THROW(sim.configureExecutionContext(cudaq::msm_size_policy{}),
               std::runtime_error);
  EXPECT_FALSE(sim.hasCpuFrames());
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

CUDAQ_TEST(CustabilizerGPU, OnlineMeasurementsUseGpuFrames) {
  CustabilizerSimulatorTester sim;
  cudaq::ExecutionContext context("sample", 65);
  sim.configureExecutionContext(context);
  ScopedExecutionContext scope(context);
  auto q = sim.allocateQubit();
  sim.applyNamedGate("H", {}, {q});
  const bool first = sim.onlineMeasurement(q);
  sim.applyNamedGate("Z", {}, {q});
  EXPECT_EQ(sim.onlineMeasurement(q), first);
  EXPECT_TRUE(sim.hasGpuRunner());
  EXPECT_FALSE(sim.hasCpuFrames());
}

CUDAQ_TEST(CustabilizerGPU, CachedSamplesRespectShotCountAndSeed) {
  nvqir::CustabilizerFrameRunner runner(1);
  runner.apply(stim::Circuit("R 0\nX_ERROR(1) 0\nM 0"));
  for (std::int64_t shots : {32, 64, 33, 128, 1}) {
    const auto frames = runner.getMeasurementTable(shots, true, 123);
    for (std::int64_t shot = 0; shot < shots; ++shot)
      EXPECT_EQ((frames[shot / 32] >> (shot % 32)) & 1u, 1u);
  }
  runner.clear();
  runner.apply(stim::Circuit("H 0\nM 0"));
  const auto first = runner.getMeasurementTable(32768, true, 123);
  EXPECT_EQ(runner.getMeasurementTable(32768, true, 123), first);
  EXPECT_NE(runner.getMeasurementTable(32768, true, 456), first);
  EXPECT_EQ(runner.getMeasurementTable(32768, true, 123), first);
}

CUDAQ_TEST(CustabilizerGPU, ReplayPreservesMeasurementsWhenQubitsGrow) {
  // These dimensions cross a cuStabilizer 0.5 RNG layout boundary. Replaying
  // with the same seed must preserve every previously recorded measurement.
  for (const std::string prefix :
       {"H 127\nM 127\n", "R 127\nH 127\nM 127\n",
        "H 127\nM 127\nH 127\nM 127\n", "X_ERROR(0.25) 127\nM 127\n"}) {
    CustabilizerFrameRunnerTester runner;
    const stim::Circuit before(prefix);
    const stim::Circuit grown(prefix + "X_ERROR(0.25) 128\nM 127");
    for (const std::int64_t shots : {1, 31, 32, 33, 65, 129, 32768, 32769}) {
      const auto first = runner.sample(before, shots, 123);
      const auto replay = runner.sample(grown, shots, 123);
      ASSERT_EQ(replay.size(), first.size() + (shots + 31) / 32);
      EXPECT_TRUE(std::equal(first.begin(), first.end(), replay.begin()))
          << prefix << "shots=" << shots;
      const auto words = (shots + 31) / 32;
      EXPECT_TRUE(
          std::equal(first.end() - words, first.end(), replay.end() - words))
          << prefix << "shots=" << shots;
      EXPECT_EQ(runner.sample(before, shots, 123), first);
    }
  }
}

CUDAQ_TEST(CustabilizerGPU, FailedReplacementDoesNotReusePreviousInputKey) {
  CustabilizerFrameRunnerTester runner;
  const stim::Circuit a("R 0\nX_ERROR(1) 0\nM 0");
  const stim::Circuit b("R 0\nM 0");
  const std::vector<std::uint32_t> ones{UINT32_MAX}, zeros{0};
  ASSERT_EQ(runner.sample(a, 32, 123), ones);

  // B compiles before allocating its frame buffers. This request cannot fit in
  // GPU virtual address space; adding 31 to the shot count still fits int64_t.
  EXPECT_THROW(
      runner.sample(b, std::numeric_limits<std::int64_t>::max() - 31, 123),
      std::runtime_error);
  EXPECT_EQ(runner.sample(a, 32, 123), ones);
  EXPECT_EQ(runner.sample(b, 32, 123), zeros);
  EXPECT_EQ(runner.sample(a, 32, 123), ones);
}

CUDAQ_TEST(CustabilizerSerialization, PreservesNoiseProbabilitiesExactly) {
  for (double probability : {0., 1., 0.123456789123456, 0.9999999,
                             std::nextafter(1., 0.), std::nextafter(0.5, 0.),
                             std::nextafter(0.5, 1.), 1.23456789123456e-100}) {
    stim::Circuit circuit;
    circuit.safe_append_u("X_ERROR", {0}, {probability});
    const auto text = nvqir::serializeCircuit(circuit);
    EXPECT_EQ(stim::Circuit(text), circuit) << text;
  }

  // Six significant digits would round these valid disjoint probabilities to
  // 0.333334, 0.333334, 0.333333, whose sum exceeds one.
  stim::Circuit circuit;
  circuit.safe_append_u("PAULI_CHANNEL_1", {0},
                        {0.3333336, 0.3333336, 0.3333328});
  const auto text = nvqir::serializeCircuit(circuit);
  EXPECT_EQ(stim::Circuit(text), circuit) << text;
}

CUDAQ_TEST(CustabilizerGPU, NormalizesOverlappingTargets) {
  using FrameRunner = CustabilizerFrameRunnerTester;
  stim::Circuit circuit("R 0 1 2\nX_ERROR(1) 0\nCX 0 1 1 2\nM 0 1 2");
  FrameRunner runner;
  const auto frames = runner.sample(circuit, 65, 12);
  ASSERT_EQ(frames.size(), 9);
  for (std::size_t m = 0; m < 3; ++m)
    for (std::size_t shot = 0; shot < 65; ++shot)
      EXPECT_EQ((frames[m * 3 + shot / 32] >> (shot % 32)) & 1u, 1u);
}
