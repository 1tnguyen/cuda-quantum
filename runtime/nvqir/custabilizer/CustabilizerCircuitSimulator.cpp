/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CustabilizerErrorHandling.h"
#include "CustabilizerSampleResult.h"
#include "../stim/StimCircuitSimulator.h"
#include <iomanip>
#include <locale>
#include <unordered_set>

using namespace cudaq;

namespace nvqir {
namespace {

/// Preserve double-valued probabilities when cuStabilizer parses circuit text.
inline std::string serializeCircuit(const stim::Circuit &circuit) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(std::numeric_limits<double>::max_digits10)
       << circuit;
  return text.str();
}

/// Device storage is retained across launches and released on every exit path.
template <typename T>
class DeviceBuffer {
  T *data_ = nullptr;
  std::size_t capacity_ = 0;

public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { cudaFree(data_); }

  T *data() const { return data_; }

  void reserve(std::size_t count) {
    if (count <= capacity_)
      return;
    T *next = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&next, count * sizeof(T)));
    cudaFree(data_);
    data_ = next;
    capacity_ = count;
  }
};

// A compiled program borrows the runner's handle and owns its device storage.
struct CustabilizerCircuit {
  custabilizerCircuit_t circuit = nullptr;
  DeviceBuffer<std::uint8_t> buffer_d;

  CustabilizerCircuit(custabilizerHandle_t handle, const std::string &text) {
    std::int64_t buffer_size = 0;
    HANDLE_CUST_ERROR(
        custabilizerCircuitSizeFromString(handle, text.c_str(), &buffer_size));
    buffer_d.reserve(buffer_size);
    HANDLE_CUST_ERROR(custabilizerCreateCircuitFromString(
        handle, text.c_str(), buffer_d.data(), buffer_size, &circuit));
  }

  ~CustabilizerCircuit() {
    if (circuit)
      custabilizerDestroyCircuit(circuit);
  }

  CustabilizerCircuit(const CustabilizerCircuit &) = delete;
  CustabilizerCircuit &operator=(const CustabilizerCircuit &) = delete;
};

struct CustabilizerFrameRunner {
  custabilizerHandle_t handle = nullptr;
  custabilizerFrameSimulator_t frame_simulator = nullptr;
  DeviceBuffer<custabilizerBitInt_t> x_table_d, z_table_d, m_table_d;
  std::int64_t num_qubits = 0;
  std::int64_t allocated_shots = 0;
  std::int64_t allocated_measurements = 0;
  std::int64_t stride_bytes = 0;
  stim::Circuit circuitCache;
  bool cache_executed = false;

  explicit CustabilizerFrameRunner(std::int64_t qubits) : num_qubits(qubits) {
    if (qubits <= 0)
      throw std::invalid_argument("cuStabilizer requires positive qubits.");
    HANDLE_CUST_ERROR(custabilizerCreate(&handle));
  }

  ~CustabilizerFrameRunner() {
    if (frame_simulator)
      custabilizerDestroyFrameSimulator(frame_simulator);
    circuit.reset();
    initialCircuit.reset();
    custabilizerDestroy(handle);
  }

  CustabilizerFrameRunner(const CustabilizerFrameRunner &) = delete;
  CustabilizerFrameRunner &operator=(const CustabilizerFrameRunner &) = delete;

  void apply(const stim::Circuit &next) {
    circuitCache += next;
    circuit_changed = true;
    cache_executed = false;
  }

  void clear() {
    circuitCache.clear();
    circuit_changed = true;
    cache_executed = false;
  }

  std::vector<std::uint32_t>
  getMeasurementTable(std::int64_t num_shots, bool randomize_after_measurement,
                      std::uint64_t seed) {
    if (num_shots <= 0 ||
        num_shots > std::numeric_limits<std::int64_t>::max() - 31)
      throw std::invalid_argument("Invalid cuStabilizer shot count.");
    if (circuit_changed) {
      auto text = serializeCircuit(frameCircuit(circuitCache));
      auto stats = circuitCache.compute_stats();
      if (!circuit || compiled_text != text) {
        // Commit the program and its key together. A later failure must never
        // leave a key naming another program.
        auto next = std::make_unique<CustabilizerCircuit>(handle, text);
        cache_executed = false;
        circuit.swap(next);
        compiled_text.swap(text);
      }
      circuit_stats = stats;
      circuit_changed = false;
    }
    const auto qubits = std::max(
        num_qubits, static_cast<std::int64_t>(circuit_stats.num_qubits));
    const auto measurements =
        static_cast<std::int64_t>(circuit_stats.num_measurements);
    if (!frame_simulator || qubits != num_qubits ||
        num_shots != allocated_shots || measurements != allocated_measurements)
      reallocate(qubits, num_shots, measurements);

    if (!cache_executed || executed_seed != seed ||
        executed_randomization != randomize_after_measurement) {
      cache_executed = false;
      HANDLE_CUDA_ERROR(
          cudaMemset(x_table_d.data(), 0, num_qubits * stride_bytes));
      HANDLE_CUDA_ERROR(
          cudaMemset(z_table_d.data(), 0, num_qubits * stride_bytes));
      HANDLE_CUDA_ERROR(
          cudaMemset(m_table_d.data(), 0, measurements * stride_bytes));
      if (randomize_after_measurement) {
        initializeFrames();
        // cuStabilizer resets randomize Z rows without depending on the total
        // qubit count. Separate execution preserves the main circuit's RNG.
        // XOR with the 64-bit golden-ratio constant to use distinct seeds.
        HANDLE_CUST_ERROR(custabilizerFrameSimulatorApplyCircuit(
            handle, frame_simulator, initialCircuit->circuit, 1,
            seed ^ UINT64_C(0x9e3779b97f4a7c15), x_table_d.data(),
            z_table_d.data(), m_table_d.data(), nullptr));
      }
      HANDLE_CUST_ERROR(custabilizerFrameSimulatorApplyCircuit(
          handle, frame_simulator, circuit->circuit,
          randomize_after_measurement ? 1 : 0, seed, x_table_d.data(),
          z_table_d.data(), m_table_d.data(), nullptr));
      executed_seed = seed;
      executed_randomization = randomize_after_measurement;
      cache_executed = true;
    }
    const auto words = (num_shots + 31) / 32;
    std::vector<std::uint32_t> out(measurements * words);
    if (!out.empty())
      HANDLE_CUDA_ERROR(cudaMemcpy(out.data(), m_table_d.data(),
                                   out.size() * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost));
    return out;
  }

private:
  std::unique_ptr<CustabilizerCircuit> circuit, initialCircuit;
  std::string compiled_text;
  stim::CircuitStats circuit_stats;
  bool circuit_changed = true;
  std::int64_t initial_qubits = 0;
  std::uint64_t executed_seed = 0;
  bool executed_randomization = false;

  // Signs in Clifford conjugation affect the noiseless reference, which is
  // supplied separately. They do not affect Pauli-frame propagation.
  static stim::Circuit frameCircuit(const stim::Circuit &input) {
    stim::Circuit result;
    input.for_each_operation([&](const stim::CircuitInstruction &op) {
      if (op.gate_type == stim::GateType::S_DAG) {
        auto equivalent = op;
        equivalent.gate_type = stim::GateType::S;
        result.safe_append(equivalent);
      } else if (op.gate_type == stim::GateType::CY ||
                 op.gate_type == stim::GateType::SWAP) {
        for (std::size_t i = 0; i < op.targets.size(); i += 2) {
          const auto a = op.targets[i].data, b = op.targets[i + 1].data;
          if (op.gate_type == stim::GateType::CY) {
            result.safe_append_u("S", {b});
            result.safe_append_u("CX", {a, b});
            result.safe_append_u("S", {b});
          } else {
            result.safe_append_u("CX", {a, b});
            result.safe_append_u("CX", {b, a});
            result.safe_append_u("CX", {a, b});
          }
        }
      } else {
        result.safe_append(op);
      }
    });
    // cuStabilizer distributes one instruction's targets across GPU threads.
    // Stim fuses consecutive gates even when they share a qubit (for example
    // CX 0 1 1 2). Split such instructions into disjoint layers. TICK preserves
    // the boundary when cuStabilizer parses the serialized Stim circuit again.
    stim::Circuit parallel;
    result.for_each_operation([&](const stim::CircuitInstruction &op) {
      const auto flags = stim::GATE_DATA[op.gate_type].flags;
      const std::size_t width = flags & stim::GATE_TARGETS_PAIRS ? 2 : 1;
      if (!(flags & (stim::GATE_IS_UNITARY | stim::GATE_IS_NOISY |
                     stim::GATE_IS_RESET | stim::GATE_PRODUCES_RESULTS))) {
        parallel.safe_append(op);
        return;
      }
      std::unordered_set<std::uint32_t> used;
      std::vector<stim::GateTarget> layer;
      auto flush = [&]() {
        auto instruction = op;
        instruction.targets = {layer.data(), layer.data() + layer.size()};
        parallel.safe_append(instruction);
        layer.clear();
        used.clear();
      };
      for (std::size_t i = 0; i < op.targets.size(); i += width) {
        bool overlaps = false;
        for (std::size_t j = 0; j < width; ++j)
          overlaps |= used.count(op.targets[i + j].data) != 0;
        if (overlaps) {
          flush();
          parallel.safe_append_u("TICK", {});
        }
        for (std::size_t j = 0; j < width; ++j) {
          layer.push_back(op.targets[i + j]);
          used.insert(op.targets[i + j].data);
        }
      }
      if (!layer.empty())
        flush();
    });
    return parallel;
  }

  void initializeFrames() {
    if (!initialCircuit || initial_qubits != num_qubits) {
      std::string text = "R";
      for (std::int64_t q = 0; q < num_qubits; ++q)
        text += " " + std::to_string(q);
      text += "\n";
      auto next = std::make_unique<CustabilizerCircuit>(handle, text);
      initialCircuit.swap(next);
      initial_qubits = num_qubits;
    }
  }

  void reallocate(std::int64_t new_qubits, std::int64_t new_shots,
                  std::int64_t new_measurements) {
    cache_executed = false;
    if (frame_simulator) {
      custabilizerDestroyFrameSimulator(frame_simulator);
      frame_simulator = nullptr;
    }
    const auto words = (new_shots + 31) / 32;
    x_table_d.reserve(new_qubits * words);
    z_table_d.reserve(new_qubits * words);
    m_table_d.reserve(new_measurements * words);
    const auto stride = words * sizeof(custabilizerBitInt_t);
    HANDLE_CUST_ERROR(custabilizerCreateFrameSimulator(
        handle, new_qubits, new_shots, new_measurements, stride,
        &frame_simulator));
    num_qubits = new_qubits;
    allocated_shots = new_shots;
    allocated_measurements = new_measurements;
    stride_bytes = stride;
  }
};

/// GPU frame simulation with Stim's noiseless Tableau reference.
class CustabilizerCircuitSimulator : public StimCircuitSimulator {
protected:
  using StimCircuitSimulator::applyNoise;

  std::unique_ptr<CustabilizerFrameRunner> frameRunner;
  std::optional<std::uint64_t> frameSeed;

  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error("The cuStabilizer backend does not support "
                               "initialization of qubits from state data.");
    // Allocate only the noiseless reference; GPU frames supply every shot.
    if (!tableau)
      tableau = std::make_unique<stim::TableauSimulator<W>>(
          std::mt19937_64(randomEngine), /*num_qubits=*/0, /*sign_bias=*/+1);
    if (!frameRunner)
      frameRunner = std::make_unique<CustabilizerFrameRunner>(
          std::max(nQubitsAllocated, qubitCount));
  }

  void deallocateStateImpl() override {
    StimCircuitSimulator::deallocateStateImpl();
    frameRunner.reset();
    frameSeed.reset();
  }

  void applyOpToSims(const std::string &gate_name,
                     const std::vector<std::uint32_t> &targets) override {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    tempCircuit.safe_append_u(gate_name, targets);
    tableau->safe_do_circuit(tempCircuit);
    recordedCircuit.safe_append_u(gate_name, targets);
    frameRunner->apply(tempCircuit);
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::uint32_t> &qubits) override {
    if (auto res = isValidStimNoiseChannel(channel)) {
      if (res->num_targets <= 0 || qubits.size() % res->num_targets != 0)
        throw std::invalid_argument("Invalid cuStabilizer noise targets.");
      stim::Circuit noiseOps;
      noiseOps.safe_append_u(res->stim_name, qubits, channel.parameters);
      recordedCircuit += noiseOps;
      frameRunner->apply(noiseOps);
    }
  }

  void setToZeroState() override {
    flushPendingQubits();
    if (!tableau) {
      deallocateState();
      return;
    }
    recordedCircuit.clear();
    frameRunner->clear();
    const auto qubits =
        std::max(tableau->inv_state.num_qubits, nQubitsAllocated);
    if (qubits > 0) {
      std::vector<std::uint32_t> targets(qubits);
      std::iota(targets.begin(), targets.end(), 0);
      applyOpToSims("R", targets);
    }
    tableau->measurement_record.clear();
    frameSeed.reset();
    num_measurements = 0;
  }

  bool measureQubit(const std::size_t index) override {
    applyOpToSims("M", {static_cast<std::uint32_t>(index)});
    ++num_measurements;
    if (!frameSeed)
      frameSeed = randomEngine();
    const auto flips = frameRunner->getMeasurementTable(1, true, *frameSeed);
    return tableau->measurement_record.storage.back() ^
           bool(flips[num_measurements - 1] & 1u);
  }

public:
  using StimCircuitSimulator::configureExecutionContext;

  CustabilizerCircuitSimulator() { summaryData.name = name(); }

  void configureExecutionContext(const cudaq::msm_policy &) override {
    throw std::runtime_error(
        "MSM mode is not supported by custabilizer backend.");
  }

  void configureExecutionContext(const cudaq::msm_size_policy &) override {
    throw std::runtime_error(
        "MSM mode is not supported by custabilizer backend.");
  }

  void configureExecutionContext(const cudaq::dem_policy &policy) override {
    CircuitSimulator::configureExecutionContext(policy);
  }

  void setRandomSeed(std::size_t seed) override {
    StimCircuitSimulator::setRandomSeed(seed);
    frameSeed.reset();
  }

  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    const auto *executionContext = getExecutionContext();
    if (!executionContext)
      throw std::runtime_error(
          "cuStabilizer sampling requires an execution context.");
    if (shots <= 0 || static_cast<std::size_t>(shots) > getBatchSize())
      throw std::invalid_argument("Invalid cuStabilizer shot count.");
    if (executionContext->explicitMeasurements && qubits.empty() &&
        num_measurements == 0)
      throw std::runtime_error(
          "The sampling option `explicit_measurements` is not supported on a "
          "kernel without any measurement operation.");
    if (!tableau || !frameRunner)
      throw std::runtime_error(
          "custabilizer backend state is not initialized.");

    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    applyOpToSims("M", stimTargetQubits);
    num_measurements += stimTargetQubits.size();
    if (executionContext->explicitMeasurements && !qubits.empty())
      return cudaq::ExecutionResult();

    const auto &reference = tableau->measurement_record.storage;
    if (reference.size() != num_measurements)
      throw std::logic_error(
          "cuStabilizer reference measurement count mismatch.");
    if (!frameSeed)
      frameSeed = randomEngine();
    const auto frames =
        frameRunner->getMeasurementTable(shots, true, *frameSeed);
    const auto words = (shots + 31) / 32;
    if (frames.size() != num_measurements * words)
      throw std::logic_error("cuStabilizer frame measurement count mismatch.");
    const auto first = executionContext->explicitMeasurements
                           ? 0
                           : num_measurements - qubits.size();
    return custabilizer::makeSampleResult(frames, words, reference, first,
                                          shots, includeSequentialData);
  }

  std::string name() const override { return "custabilizer"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CustabilizerCircuitSimulator)
};

} // namespace
} // namespace nvqir

#ifndef CUDAQ_CUSTABILIZER_NO_REGISTRATION
NVQIR_REGISTER_SIMULATOR(nvqir::CustabilizerCircuitSimulator, custabilizer)
#endif
