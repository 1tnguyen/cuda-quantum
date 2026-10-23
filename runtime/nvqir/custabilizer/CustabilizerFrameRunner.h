/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CustabilizerErrorHandling.h"
#include "stim.h"
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace nvqir::custabilizer {
namespace {

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

/// Execute complete, batchable circuits. Classical feedback and MSM are handled
/// by this plugin's private CPU implementation, without loading nvqir-stim.
class FrameRunner {
  custabilizerHandle_t handle_ = nullptr;
  custabilizerCircuit_t circuit_ = nullptr;
  custabilizerFrameSimulator_t simulator_ = nullptr;
  DeviceBuffer<std::uint8_t> circuitStorage_;
  DeviceBuffer<custabilizerBitInt_t> x_, z_, measurements_;
  DeviceBuffer<double> initialProbabilities_;
  stim::Circuit compiledInput_;
  std::string compiledText_;
  bool hasCompiledInput_ = false;
  std::size_t circuitApplications_ = 0;
  std::int64_t qubits_ = 0, shots_ = 0, measurementCount_ = 0;
  std::int64_t detectorCount_ = 0;

  static bool hasParallelTargets(const stim::CircuitInstruction &op) {
    const auto flags = stim::GATE_DATA[op.gate_type].flags;
    return flags & (stim::GATE_IS_UNITARY | stim::GATE_IS_NOISY |
                    stim::GATE_IS_RESET | stim::GATE_PRODUCES_RESULTS);
  }

  // Signs in Clifford conjugation affect the noiseless reference, which is
  // supplied separately. They do not affect Pauli-frame propagation.
  static stim::Circuit frameCircuit(const stim::Circuit &input) {
    stim::Circuit result;
    input.for_each_operation([&](const stim::CircuitInstruction &op) {
      if (op.gate_type == stim::GateType::S_DAG) {
        auto equivalent = op;
        equivalent.gate_type = stim::GateType::S;
        result.safe_append(equivalent);
      } else if (op.gate_type == stim::GateType::Y_ERROR) {
        std::vector<std::uint32_t> targets;
        for (auto target : op.targets)
          targets.push_back(target.data);
        result.safe_append_u("PAULI_CHANNEL_1", targets,
                             {0.0, op.args[0], 0.0});
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
    // the boundary when the SDK parses the serialized Stim circuit again.
    stim::Circuit parallel;
    result.for_each_operation([&](const stim::CircuitInstruction &op) {
      const auto flags = stim::GATE_DATA[op.gate_type].flags;
      const std::size_t width = flags & stim::GATE_TARGETS_PAIRS ? 2 : 1;
      if (!hasParallelTargets(op)) {
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

  std::vector<std::uint32_t> sampleProgram(const std::string &text,
                                           const stim::CircuitStats &stats,
                                           std::int64_t shots,
                                           std::uint64_t seed) {
    if (shots <= 0)
      throw std::invalid_argument("cuStabilizer requires positive shots.");

    const auto numQubits = static_cast<std::int64_t>(stats.num_qubits);
    const auto numMeasurements =
        static_cast<std::int64_t>(stats.num_measurements);
    const auto numDetectors = static_cast<std::int64_t>(stats.num_detectors);
    const auto words = (shots + 31) / 32;
    const auto stride = words * sizeof(custabilizerBitInt_t);

    if (!circuit_ || compiledText_ != text) {
      if (circuit_) {
        custabilizerDestroyCircuit(circuit_);
        circuit_ = nullptr;
      }
      std::int64_t bytes = 0;
      HANDLE_CUST_ERROR(
          custabilizerCircuitSizeFromString(handle_, text.c_str(), &bytes));
      circuitStorage_.reserve(bytes);
      HANDLE_CUST_ERROR(custabilizerCreateCircuitFromString(
          handle_, text.c_str(), circuitStorage_.data(), bytes, &circuit_));
      compiledText_ = text;
    }

    if (!simulator_ || qubits_ != numQubits || shots_ != shots ||
        measurementCount_ != numMeasurements ||
        detectorCount_ != numDetectors) {
      if (simulator_) {
        custabilizerDestroyFrameSimulator(simulator_);
        simulator_ = nullptr;
      }
      x_.reserve(numQubits * words);
      z_.reserve(numQubits * words);
      measurements_.reserve((numMeasurements + numDetectors) * words);
      initialProbabilities_.reserve(numQubits);
      std::vector<double> probabilities(numQubits, 0.5);
      HANDLE_CUDA_ERROR(
          cudaMemcpy(initialProbabilities_.data(), probabilities.data(),
                     numQubits * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_CUST_ERROR(custabilizerCreateFrameSimulator(
          handle_, numQubits, shots, numMeasurements, stride, &simulator_));
      qubits_ = numQubits;
      shots_ = shots;
      measurementCount_ = numMeasurements;
      detectorCount_ = numDetectors;
    }

    HANDLE_CUDA_ERROR(cudaMemset(x_.data(), 0, numQubits * stride));
    HANDLE_CUDA_ERROR(cudaMemset(measurements_.data(), 0,
                                 (numMeasurements + numDetectors) * stride));
    // A zero X frame and a random Z frame represent |0>. Keep measurement
    // randomization enabled for both ordinary and masked PTSBE execution.
    HANDLE_CUST_ERROR(custabilizerSampleProbArray(
        handle_, words * 32, numQubits, initialProbabilities_.data(),
        seed ^ UINT64_C(0x9e3779b97f4a7c15), z_.data(), nullptr));
    HANDLE_CUST_ERROR(custabilizerFrameSimulatorApplyCircuit(
        handle_, simulator_, circuit_, 1, seed, x_.data(), z_.data(),
        measurements_.data(), nullptr));
    ++circuitApplications_;

    std::vector<std::uint32_t> result(numMeasurements * words);
    HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), measurements_.data(),
                                 result.size() * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost));
    return result;
  }

public:
  std::int64_t shots() const { return shots_; }
  std::size_t circuitApplications() const { return circuitApplications_; }

  struct Segment {
    stim::Circuit circuit;
    // A negative end means all shots. Other ranges are half-open and may
    // start or end inside a packed word.
    std::int64_t begin = 0, end = -1;
  };

  static void validateParallelTargets(const stim::Circuit &circuit) {
    circuit.for_each_operation([](const stim::CircuitInstruction &op) {
      if (!hasParallelTargets(op))
        return;
      std::unordered_set<std::uint32_t> used;
      for (auto target : op.targets)
        if (target.has_qubit_value() &&
            !used.insert(target.qubit_value()).second)
          throw std::invalid_argument("cuStabilizer GPU instruction contains "
                                      "overlapping qubit targets.");
    });
  }

  FrameRunner() { HANDLE_CUST_ERROR(custabilizerCreate(&handle_)); }
  FrameRunner(const FrameRunner &) = delete;
  FrameRunner &operator=(const FrameRunner &) = delete;

  ~FrameRunner() {
    if (simulator_)
      custabilizerDestroyFrameSimulator(simulator_);
    if (circuit_)
      custabilizerDestroyCircuit(circuit_);
    custabilizerDestroy(handle_);
  }

  std::vector<std::uint32_t> sample(const stim::Circuit &input,
                                    std::int64_t shots, std::uint64_t seed) {
    if (circuit_ && hasCompiledInput_ && compiledInput_ == input)
      return sampleProgram(compiledText_, input.compute_stats(), shots, seed);
    auto normalized = frameCircuit(input);
    validateParallelTargets(normalized);
    auto result =
        sampleProgram(normalized.str(), input.compute_stats(), shots, seed);
    compiledInput_ = input;
    hasCompiledInput_ = true;
    return result;
  }

  std::vector<std::uint32_t> sampleMasked(const std::vector<Segment> &segments,
                                          std::int64_t shots,
                                          std::uint64_t seed) {
    if (shots <= 0)
      throw std::invalid_argument("cuStabilizer requires positive shots.");
    std::string text;
    stim::CircuitStats stats;
    bool masked = false;
    for (const auto &segment : segments) {
      auto normalized = frameCircuit(segment.circuit);
      validateParallelTargets(normalized);
      if (segment.end >= 0) {
        if (segment.begin < 0 || segment.begin >= segment.end ||
            segment.end > shots)
          throw std::invalid_argument("Invalid cuStabilizer PTSBE shot mask.");
        text += "SHOT_MASK_SET(" + std::to_string(segment.begin) + "," +
                std::to_string(segment.end) + ")\n";
        masked = true;
      } else if (masked) {
        text += "SHOT_MASK_SET(0," + std::to_string(shots) + ")\n";
        masked = false;
      }
      auto part = normalized.compute_stats();
      stats.num_qubits = std::max(stats.num_qubits, part.num_qubits);
      stats.num_measurements += part.num_measurements;
      stats.num_detectors += part.num_detectors;
      text += normalized.str() + "\nTICK\n";
    }
    compiledInput_.clear();
    hasCompiledInput_ = false;
    return sampleProgram(text, stats, shots, seed);
  }
};

} // namespace
} // namespace nvqir::custabilizer
