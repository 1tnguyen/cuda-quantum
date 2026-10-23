/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CustabilizerFrameRunner.h"
#include "CustabilizerSampleResult.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "nvqir/CircuitSimulator.h"
#include "stim.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>
#include <limits>
#include <numeric>

using namespace cudaq;

namespace nvqir::custabilizer {
namespace {

/// @brief Collection of information about a noise type in Stim
struct CliffordNoiseType {
  /// The name of the error mechanism in Stim
  std::string stim_name;
  /// Whether the error mechanism flips X; one per error mechanism per target
  std::vector<bool> flips_x;
  /// Whether the error mechanism flips Z; one per error mechanism per target
  std::vector<bool> flips_z;
  /// One probability per error mechanism
  std::vector<double> params;
  /// The number of targets for the noise type
  int num_targets = 1;
};

using CustabilizerSimulatorBase = nvqir::CircuitSimulatorBase<double>;

/// @brief GPU Clifford sampling with the same CUDA-Q semantics as Stim.
/// The compatibility implementation is deliberately private to this backend:
/// it was copied from the unchanged Stim adapter at 9695b1aaf408 so introducing
/// cuStabilizer does not refactor or change the release's existing Stim plugin.
/// CPU frames remain available for MSM and measurement-dependent execution.
class CustabilizerCircuitSimulator : public CustabilizerSimulatorBase,
                                     public cudaq::ptsbe::BatchSimulator {
protected:
  // Follow Stim naming convention (W) for bit width (required for templates).
  static constexpr std::size_t W = stim::MAX_BITWORD_WIDTH;

  /// @brief Number of measurements performed so far.
  std::size_t num_measurements = 0;

  /// @brief Top-level random engine. Stim simulator RNGs are based off of this
  /// engine.
  std::mt19937_64 randomEngine;

  /// @brief Stim Tableau simulator (noiseless)
  std::unique_ptr<stim::TableauSimulator<W>> tableau;

  /// @brief Stim Frame/Flip simulator (used to generate multiple shots)
  std::unique_ptr<stim::FrameSimulator<W>> sampleSim;

  /// @brief Error counter for MSM generation. This is only used for "msm" and
  /// "msm_size" policies.
  std::size_t msm_err_count = 0;

  /// @brief Error ID counter for MSM generation. This is only used for "msm"
  /// and "msm_size" execution contexts.
  std::size_t msm_id_counter = 0;

  /// @brief Whether or not the execution context name is "msm" (value is cached
  /// for speed)
  bool is_msm_mode = false;

  std::optional<cudaq::msm_dimensions> activeMsmDimensions;

  /// @brief Probabilities of each error mechanism, accumulated during MSM
  /// execution. Only used for "msm" policy.
  std::vector<double> msmProbabilities;

  /// @brief Error IDs corresponding to each entry in @c msmProbabilities.
  std::vector<std::size_t> msmProbErrId;

  /// @brief Accumulated Stim circuit covering gates, measurements, noise,
  /// detectors, and observables. Reset alongside `num_measurements` in
  /// `deallocateStateImpl` and `setToZeroState` so successive kernel
  /// executions on a reused simulator start from a clean slate. Stim-side
  /// internal only; not part of any public CUDA-Q API.
  stim::Circuit recordedCircuit;

  std::unique_ptr<FrameRunner> frameRunner;
  std::optional<std::uint64_t> frameSeed;

  bool useGpuFrames() const {
    auto *context = cudaq::getExecutionContext();
    return !is_msm_mode && context &&
           (context->name == "sample" || context->name == "ptsbe-sample") &&
           !context->hasConditionalsOnMeasureResults;
  }

  // Recognize a tensor product of Paulis up to a global phase. Inspect the
  // matrix rather than trusting channel op_names, including custom channels.
  static std::optional<std::string>
  pauliProduct(const GateApplicationTask &task) {
    const auto n = task.targets.size();
    if (n >= std::numeric_limits<std::size_t>::digits / 2)
      return std::nullopt;
    const std::size_t dim = std::size_t{1} << n;
    const auto &matrix = task.matrix;
    if (matrix.size() != dim * dim)
      return std::nullopt;
    constexpr double tolerance = 1e-10;
    std::size_t xMask = 0, zMask = 0;
    while (xMask < dim && std::abs(matrix[xMask]) < tolerance)
      ++xMask;
    if (xMask == dim || std::abs(std::abs(matrix[xMask]) - 1.) > tolerance)
      return std::nullopt;
    const auto phase = matrix[xMask];
    for (std::size_t bit = 0; bit < n; ++bit) {
      const auto row = std::size_t{1} << bit;
      const auto value = matrix[row * dim + (row ^ xMask)];
      if (std::abs(value + phase) < tolerance)
        zMask |= row;
      else if (std::abs(value - phase) > tolerance)
        return std::nullopt;
    }
    for (std::size_t row = 0; row < dim; ++row)
      for (std::size_t col = 0; col < dim; ++col) {
        const auto expected =
            col == (row ^ xMask)
                ? ((__builtin_popcountll(row & zMask) & 1) ? -phase : phase)
                : std::complex<double>{0., 0.};
        if (std::abs(matrix[row * dim + col] - expected) > tolerance)
          return std::nullopt;
      }
    std::string product(n, 'I');
    for (std::size_t q = 0; q < n; ++q) {
      const auto bit = std::size_t{1} << (n - q - 1);
      product[q] = "IXZY"[bool(xMask & bit) + 2 * bool(zMask & bit)];
    }
    return product;
  }

  std::optional<CliffordNoiseType>
  isValidStimNoiseChannel(const kraus_channel &channel) const {

    // Check the old way first
    switch (channel.noise_type) {
    case cudaq::noise_model_type::bit_flip_channel:
    case cudaq::noise_model_type::x_error:
      return CliffordNoiseType{.stim_name = "X_ERROR",
                               .flips_x = {true},
                               .flips_z = {false},
                               .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::y_error:
      return CliffordNoiseType{.stim_name = "Y_ERROR",
                               .flips_x = {true},
                               .flips_z = {true},
                               .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::phase_flip_channel:
    case cudaq::noise_model_type::z_error:
      return CliffordNoiseType{.stim_name = "Z_ERROR",
                               .flips_x = {false},
                               .flips_z = {true},
                               .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::depolarization_channel:
    case cudaq::noise_model_type::depolarization1:
      return CliffordNoiseType{
          .stim_name = "DEPOLARIZE1",
          .flips_x = {true, true, false},
          .flips_z = {false, true, true},
          .params = std::vector<double>(3, channel.parameters[0] / 3.0)};
    case cudaq::noise_model_type::depolarization2: {
      constexpr bool x_err[4] = {false, true, true, false}; // X errors for IXYZ
      constexpr bool z_err[4] = {false, false, true, true}; // Z errors for IXYZ
      CliffordNoiseType ret{.stim_name = "DEPOLARIZE2", .num_targets = 2};
      ret.params = std::vector<double>(15, channel.parameters[0] / 15.0);
      // Generate the entries for p/15: IX, IY, IZ, XI, XX, XY, XZ, YI, YX,
      // YY, YZ, ZI, ZX, ZY, ZZ
      for (int q1_err = 0; q1_err < 4; q1_err++) {   // qubit 1 loop
        for (int q2_err = 0; q2_err < 4; q2_err++) { // qubit 2 loop
          if (q1_err == 0 && q2_err == 0)            // skip II
            continue;
          // Push back the values for the two qubits, for both x and z errors
          ret.flips_x.insert(ret.flips_x.end(), {x_err[q1_err], x_err[q2_err]});
          ret.flips_z.insert(ret.flips_z.end(), {z_err[q1_err], z_err[q2_err]});
        }
      }
      return ret;
    }
    case cudaq::noise_model_type::pauli1: {
      // Either X error, Y error, or Z error happens, each with its own
      // probability that is specified in the 3 channel parameters.
      static_assert(cudaq::pauli1::num_parameters == 3);
      assert(channel.parameters.size() == cudaq::pauli1::num_parameters);
      return CliffordNoiseType{.stim_name = "PAULI_CHANNEL_1",
                               .flips_x = {true, true, false},
                               .flips_z = {false, true, true},
                               .params = channel.parameters};
    }
    case cudaq::noise_model_type::pauli2: {
      static_assert(cudaq::pauli2::num_parameters == 15);
      assert(channel.parameters.size() == cudaq::pauli2::num_parameters);
      CliffordNoiseType ret{.stim_name = "PAULI_CHANNEL_2", .num_targets = 2};
      // Generate the entries for: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ,
      // ZI, ZX, ZY, ZZ
      std::vector<bool> x_err{false, true, true, false}; // X errors for IXYZ
      std::vector<bool> z_err{false, false, true, true}; // Z errors for IXYZ
      for (int q1_err = 0; q1_err < 4; q1_err++) {       // qubit 1 loop
        for (int q2_err = 0; q2_err < 4; q2_err++) {     // qubit 2 loop
          if (q1_err == 0 && q2_err == 0)                // skip II
            continue;
          // Push back the values for the two qubits, for both x and z errors
          ret.flips_x.insert(ret.flips_x.end(), {x_err[q1_err], x_err[q2_err]});
          ret.flips_z.insert(ret.flips_z.end(), {z_err[q1_err], z_err[q2_err]});
        }
      }
      ret.params = channel.parameters;
      return ret;
    }
    case cudaq::noise_model_type::amplitude_damping_channel:
    case cudaq::noise_model_type::amplitude_damping:
    case cudaq::noise_model_type::phase_damping:
    case cudaq::noise_model_type::unknown:
      return std::nullopt;
    }

    return std::nullopt;
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Get the batch size to use for the Stim simulator.
  std::size_t getBatchSize() {
    // Default to single shot
    std::size_t batch_size = 1;
    auto *executionContext = getExecutionContext();
    if (executionContext &&
        (executionContext->name == "sample" ||
         executionContext->name == "ptsbe-sample") &&
        !executionContext->hasConditionalsOnMeasureResults)
      batch_size = executionContext->shots;
    else if (is_msm_mode)
      batch_size = activeMsmDimensions.value_or(std::make_pair(1, 1)).second;
    return batch_size;
  }

  /// @brief Return the number of rows and columns needed for a Parity Check
  /// Matrix
  std::optional<std::pair<std::size_t, std::size_t>> generateMSMSize() {
    return std::make_pair(num_measurements, msm_err_count);
  }

  cudaq::msm_result generateMSM() {
    const auto num_cols = getBatchSize();
    stim::simd_bit_table<W> msmSample = sampleSim->m_record.storage;
    // Disabled because it's too verbose, but left here as comments for
    // reference.
    // CUDAQ_INFO("msmSample is {} {}\n{}", msm_err_count, num_cols,
    //             msmSample.str(num_measurements, num_cols).c_str());

    // Now it's msmSample[error_mechanism_index][measure_idx]
    msmSample = msmSample.transposed();
    CountsDictionary counts;
    std::vector<std::string> sequentialData;
    sequentialData.reserve(num_cols);
    for (std::size_t shot = 0; shot < num_cols; shot++) {
      std::string aShot(num_measurements, '0');
      for (std::size_t b = 0; b < num_measurements; b++)
        aShot[b] = msmSample[shot][b] ? '1' : '0';
      counts[aShot]++;
      sequentialData.push_back(std::move(aShot));
    }
    ExecutionResult execResult(counts);
    execResult.sequentialData = std::move(sequentialData);

    cudaq::msm_result result;
    result.samples = std::move(execResult);
    result.probabilities = std::move(msmProbabilities);
    result.probability_error_ids = std::move(msmProbErrId);
    return result;
  }

  /// @brief Finalize the execution context, ensuring the simulator is left in a
  /// clean state even if finalization throws.
  void finalizeExecutionContext(const cudaq::other_policies &policy,
                                cudaq::ExecutionContext &context) override {
    try {
      CustabilizerSimulatorBase::finalizeExecutionContext(policy, context);
    } catch (...) {
      endExecution();
      throw;
    }
  }

  cudaq::msm_dimensions
  finalizeExecutionContext(const cudaq::msm_size_policy &policy) override {
    if (nQubitsAllocated == 0)
      return {};
    finalizeExecutionContextImpl();
    auto dimensions = generateMSMSize();
    if (!dimensions)
      throw std::runtime_error("MSM size analysis not supported.");
    return *dimensions;
  }

  cudaq::msm_result
  finalizeExecutionContext(const cudaq::msm_policy &policy) override {
    if (nQubitsAllocated == 0)
      return {};
    finalizeExecutionContextImpl();
    return generateMSM();
  }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error("The Stim simulator does not support "
                               "initialization of qubits from state data.");

    if (!tableau) {
      CUDAQ_INFO("Creating new cuStabilizer reference tableau");
      // CPU compatibility frames retain Stim's seed splitting. The GPU RNG
      // must not depend on whether its noiseless reference was just allocated
      // or reused, so reference allocation does not advance the GPU seed.
      if (!useGpuFrames())
        randomEngine.discard(
            std::uniform_int_distribution<int>(1, 30)(randomEngine));
      tableau = std::make_unique<stim::TableauSimulator<W>>(
          std::mt19937_64(randomEngine), /*num_qubits=*/0,
          /*sign_bias=*/useGpuFrames() ? +1 : 0);
    }
    if (!useGpuFrames() && !sampleSim) {
      std::size_t anticipated_num_measurements = 0;
      std::size_t num_msm_cols = 0;
      if (is_msm_mode) {
        auto dims = activeMsmDimensions.value_or(std::make_pair(1, 1));
        anticipated_num_measurements = dims.first;
        num_msm_cols = dims.second;
        msmProbabilities.clear();
        msmProbabilities.reserve(num_msm_cols);
        msmProbErrId.clear();
        msmProbErrId.reserve(num_msm_cols);
      }

      // If possible, provide a non-empty stim::CircuitStats in order to avoid
      // reallocations during execution.
      stim::CircuitStats circuit_stats;
      circuit_stats.num_measurements = anticipated_num_measurements;

      auto batch_size = getBatchSize();
      CUDAQ_INFO("Creating new Stim frame simulator with batch size {}",
                 batch_size);
      // Bump the randomEngine before cloning and giving to the sample
      // simulator.
      randomEngine.discard(
          std::uniform_int_distribution<int>(1, 30)(randomEngine));
      sampleSim = std::make_unique<stim::FrameSimulator<W>>(
          circuit_stats, stim::FrameSimulatorMode::STORE_MEASUREMENTS_TO_MEMORY,
          batch_size, std::mt19937_64(randomEngine));
      if (is_msm_mode) {
        sampleSim->guarantee_anticommutation_via_frame_randomization = false;
      }
      sampleSim->reset_all();
      msm_err_count = 0;
      msm_id_counter = 0;
    }
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    tableau.reset();
    // Update the randomEngine so that future invocations will use the updated
    // RNG state.
    if (sampleSim)
      randomEngine = std::move(sampleSim->rng);
    sampleSim.reset();
    num_measurements = 0;
    msm_err_count = 0;
    msm_id_counter = 0;
    is_msm_mode = false;
    activeMsmDimensions = std::nullopt;
    msmProbabilities.clear();
    msmProbErrId.clear();
    recordedCircuit.clear();
    frameSeed.reset();
  }

  /// @brief Apply operation to all Stim simulators and append it to the
  /// persistent `recordedCircuit` log.
  void applyOpToSims(const std::string &gate_name,
                     const std::vector<uint32_t> &targets) {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    CUDAQ_INFO("Calling applyOpToSims {} - {}", gate_name, targets);
    tempCircuit.safe_append_u(gate_name, targets);
    if (tableau)
      tableau->safe_do_circuit(tempCircuit);
    if (sampleSim)
      sampleSim->safe_do_circuit(tempCircuit);
    recordedCircuit.safe_append_u(gate_name, targets);
  }

  /// @brief Apply the noise channel on \p qubits
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override {
    auto executionContext = getExecutionContext();

    // Do nothing if no execution context
    if (!executionContext)
      return;

    // Do nothing if no noise model
    if (!getNoiseModel())
      return;

    // Get the name as a string
    std::string gName(gateName);

    // Cast size_t to uint32_t
    std::vector<std::uint32_t> stimTargets;
    stimTargets.reserve(controls.size() + targets.size());
    for (auto q : controls)
      stimTargets.push_back(static_cast<std::uint32_t>(q));
    for (auto q : targets)
      stimTargets.push_back(static_cast<std::uint32_t>(q));

    // Get the Kraus channels specified for this gate and qubits
    auto krausChannels =
        getNoiseModel()->get_channels(gName, targets, controls, params);

    // If none, do nothing
    if (krausChannels.empty())
      return;

    CUDAQ_INFO("Applying {} kraus channels to qubits {}", krausChannels.size(),
               stimTargets);

    for (auto &channel : krausChannels)
      applyNoise(channel, stimTargets);
  }

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override {
    switch (type) {
    case cudaq::noise_model_type::bit_flip_channel:
    case cudaq::noise_model_type::x_error:
    case cudaq::noise_model_type::y_error:
    case cudaq::noise_model_type::phase_flip_channel:
    case cudaq::noise_model_type::z_error:
    case cudaq::noise_model_type::depolarization_channel:
    case cudaq::noise_model_type::depolarization1:
    case cudaq::noise_model_type::depolarization2:
    case cudaq::noise_model_type::pauli1:
    case cudaq::noise_model_type::pauli2:
      return true;
    default:
      return false;
    }
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &qubits) override {
    flushGateQueue();
    std::vector<std::uint32_t> stimTargets(qubits.begin(), qubits.end());
    applyNoise(channel, stimTargets);
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::uint32_t> &qubits) {
    CUDAQ_INFO("[stim] apply kraus channel {}, is_msm_mode = {}",
               channel.get_type_name(), is_msm_mode);

    // If we have a valid operation, apply it
    if (auto res = isValidStimNoiseChannel(channel)) {
      // A channel acting on `num_targets` qubits is broadcast independently
      // across each consecutive group of `num_targets` qubits, matching Stim's
      // multi-target semantics.
      const std::size_t num_targets = res->num_targets;
      if (num_targets == 0 || qubits.size() % num_targets != 0)
        throw std::runtime_error(fmt::format(
            "Stim noise channel '{}' expects a positive multiple of {} target "
            "qubit(s) but was applied to {} qubit(s).",
            res->stim_name, num_targets, qubits.size()));
      const std::size_t num_groups = qubits.size() / num_targets;
      const std::size_t num_mechanisms = res->params.size();

      if (is_msm_mode) {
        // If the noise operation is the first operation done to a qubit, the
        // x_table and z_table may not be sized for the qubits. If that is the
        // case, then we simply perform a reset on the qubit to essentially
        // allocate it, which ensures the tables are resized to the correct
        // size.
        auto max_qubit = *std::max_element(qubits.begin(), qubits.end());
        if (sampleSim->num_qubits < max_qubit + 1)
          applyOpToSims("R", std::vector<std::uint32_t>{max_qubit});

        // Apply the errors found in res directly into sampleSim, as if they
        // definitely happened, 1 mechanism at a time. (For example, a
        // depolarization channel will manifest as 3 possible error mechanisms:
        // an X error, Y error, or Z error.) Each broadcast group is an
        // independent error source with its own error id.
        for (std::size_t g = 0; g < num_groups; g++) {
          for (std::size_t m = 0; m < num_mechanisms; m++) {
            // In this mode, the "shot" is an alias for the MSM error count.
            std::size_t shot = msm_err_count;
            if (msm_err_count < sampleSim->batch_size) {
              for (std::size_t t = 0; t < num_targets; t++) {
                auto q = qubits[g * num_targets + t];
                auto flip_ix = m * num_targets + t;
                sampleSim->x_table[q][shot] ^= res->flips_x[flip_ix];
                sampleSim->z_table[q][shot] ^= res->flips_z[flip_ix];
              }
              msmProbabilities.push_back(res->params[m]);
              msmProbErrId.push_back(msm_id_counter);
              msm_err_count++;
            }
          }
          msm_id_counter++;
        }
      } else {
        stim::Circuit noiseOps;
        noiseOps.safe_append_u(res.value().stim_name, qubits,
                               channel.parameters);
        // Only apply the noise operations to the sample simulator (not the
        // Tableau simulator).
        if (sampleSim)
          sampleSim->safe_do_circuit(noiseOps);
        recordedCircuit.safe_append_u(res.value().stim_name, qubits,
                                      channel.parameters);

        // Count one mechanism per group.
        msm_err_count += num_groups * num_mechanisms;
      }
    }
  }

  /// @brief Check if gateName is a two-qubit Pauli product (e.g. "IX", "ZZ").
  static bool isTwoQubitPauliProduct(const std::string &gateName) {
    if (gateName.size() != 2)
      return false;
    static const std::string paulis = "IXYZ";
    return paulis.find(gateName[0]) != std::string::npos &&
           paulis.find(gateName[1]) != std::string::npos;
  }

  static bool isApproxAngle(double value, double target) {
    constexpr double tolerance = 1e-12;
    return std::abs(value - target) < tolerance;
  }

  void applyGate(const GateApplicationTask &task) override {
    std::string gateName(task.operationName);
    std::transform(gateName.begin(), gateName.end(), gateName.begin(),
                   ::toupper);

    // Two-qubit Pauli product gates (e.g. "IX", "XY", "ZZ") decompose into
    // independent single-qubit gates on each target qubit.
    if (isTwoQubitPauliProduct(gateName)) {
      if (!task.controls.empty() || task.targets.size() != 2)
        CUDAQ_ERROR("Two-qubit Pauli product gate {} requires exactly 2 "
                    "targets and no controls, got {} targets and {} controls.",
                    task.operationName, task.targets.size(),
                    task.controls.size());
      for (int i = 0; i < 2; i++) {
        if (gateName[i] != 'I')
          applyOpToSims(std::string(1, gateName[i]),
                        {static_cast<std::uint32_t>(task.targets[i])});
      }
      return;
    }

    // These CUDA-Q rotation gates have the same name as Stim "reset" gates.
    // Stim is a Clifford simulator, so it doesn't actually support rotational
    // gates. Throw exceptions if they are encountered here.
    // TODO - consider adding support for specific rotations (e.g. pi/2).
    if (gateName == "RX" || gateName == "RY" || gateName == "RZ")
      CUDAQ_ERROR("Gate not supported by Stim simulator: {}. Note that Stim "
                  "can only simulate Clifford gates.",
                  task.operationName);
    else if (gateName == "R1") {
      if (task.parameters.size() != 1)
        CUDAQ_ERROR("Gate not supported by Stim simulator: {}. Note that Stim "
                    "can only simulate Clifford gates.",
                    task.operationName);

      auto angle = task.parameters.front();
      if (isApproxAngle(angle, M_PI_2))
        gateName = "S";
      else if (isApproxAngle(angle, -M_PI_2))
        gateName = "S_DAG";
      else if (isApproxAngle(angle, M_PI) || isApproxAngle(angle, -M_PI))
        gateName = "Z";
      else
        CUDAQ_ERROR("Gate not supported by Stim simulator: {}({}). Note that "
                    "Stim can only simulate Clifford gates.",
                    task.operationName, angle);
    } else if (gateName == "SDG")
      gateName = "S_DAG";
    else if (gateName == "ID")
      gateName = "I";

    std::vector<std::uint32_t> stimTargets;
    if (task.controls.size() > 1)
      CUDAQ_ERROR("Gates with >1 controls not supported by Stim simulator");
    if (task.controls.size() >= 1)
      gateName = "C" + gateName;
    for (auto c : task.controls)
      stimTargets.push_back(c);
    for (auto t : task.targets)
      stimTargets.push_back(t);
    try {
      applyOpToSims(gateName, stimTargets);
    } catch (...) {
      CUDAQ_ERROR("Gate not supported by Stim simulator: {}. Note that Stim "
                  "can only simulate Clifford gates.",
                  task.operationName);
    }
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
    // Without this the reset below would drop queued allocations.
    flushPendingQubits();
    if (!tableau) {
      deallocateState();
      return;
    }

    // Reset all qubits to |0> and clear measurement records, preserving
    // the allocated simulators for reuse (required by the PTSBE
    // per-trajectory loop which calls setToZeroState between trajectories).
    // `recordedCircuit` is cleared before the reset `R` ops so the next
    // trajectory's circuit starts with its own resets, not stacked on top
    // of the previous trajectory's gates and detectors.
    recordedCircuit.clear();
    // Include newly allocated qubits before their first gate, so reset/replay
    // produces the same GPU program on first use and reuse.
    auto nq = std::max(tableau->inv_state.num_qubits, nQubitsAllocated);
    if (nq > 0) {
      std::vector<std::uint32_t> allQubits(nq);
      std::iota(allQubits.begin(), allQubits.end(), 0);
      applyOpToSims("R", allQubits);
    }
    tableau->measurement_record.clear();
    if (sampleSim)
      sampleSim->m_record.clear();
    frameSeed.reset();
    num_measurements = 0;
  }

  /// @brief Override the calculateStateDim because this is not a state vector
  /// simulator.
  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return 0;
  }

  /// @brief Measure the qubit and return the result.
  bool measureQubit(const std::size_t index) override {
    if (!tableau || !sampleSim)
      throw std::runtime_error(
          "cuStabilizer online measurement requires CPU frames; deferred GPU "
          "sampling cannot supply an online measurement result.");
    // Perform measurement
    applyOpToSims(
        "M", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
    num_measurements++;

    // Get the tableau bit that was just generated.
    const std::vector<bool> &v = tableau->measurement_record.storage;
    const bool tableauBit = *v.crbegin();

    // Get the mid-circuit sample to be XOR-ed with tableauBit.
    bool sampleSimBit =
        sampleSim->m_record.storage[num_measurements - 1][/*shot=*/0];

    // Calculate the result.
    bool result = tableauBit ^ sampleSimBit;

    return result;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

public:
  using CustabilizerSimulatorBase::configureExecutionContext;
  CustabilizerCircuitSimulator() : randomEngine(std::random_device{}()) {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
    // Set supportsBufferedSample = true to tell the base class that this
    // simulator knows how to buffer the results across multiple sample()
    // invocations.
    supportsBufferedSample = true;
  }
  virtual ~CustabilizerCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    randomEngine = std::mt19937_64(seed);
    frameSeed.reset();
  }

  bool canHandleObserve() override { return false; }

  /// Share the Clifford circuit across trajectories. Only the selected Pauli
  /// faults are masked; terminal measurement and frame randomization apply to
  /// every shot. Non-Pauli Clifford trajectories retain the generic path.
  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const cudaq::ptsbe::PTSBatch &batch) override {
    using namespace cudaq::ptsbe;
    std::vector<cudaq::sample_result> results(batch.trajectories.size());
    if (batch.trajectories.empty() || batch.measureQubits.empty())
      return results;
    if (!useGpuFrames())
      return cudaq::ptsbe::detail::samplePTSBEGeneric<double>(*this, batch);

    struct Fault {
      stim::Circuit circuit;
      std::int64_t begin, end;
    };
    std::vector<std::vector<Fault>> faults(batch.trace.size());
    std::size_t totalShots = 0;
    for (const auto &trajectory : batch.trajectories) {
      if (trajectory.num_shots >
          std::numeric_limits<std::int64_t>::max() - 31 - totalShots)
        throw std::invalid_argument("cuStabilizer PTSBE shot count overflows.");
      const auto begin = totalShots;
      totalShots += trajectory.num_shots;
      if (trajectory.num_shots == 0)
        continue;
      if (!trajectory.isOrdered())
        throw std::invalid_argument(
            "cuStabilizer PTSBE selections are unordered.");
      for (const auto &selection : trajectory.kraus_selections) {
        if (selection.circuit_location >= batch.trace.size())
          throw std::invalid_argument(
              "cuStabilizer PTSBE noise location is out of range.");
        auto task = cudaq::ptsbe::detail::krausSelectionToTask<double>(
            selection, batch.trace[selection.circuit_location]);
        auto product = pauliProduct(task);
        if (!product)
          return cudaq::ptsbe::detail::samplePTSBEGeneric<double>(*this, batch);
        stim::Circuit errors;
        for (std::size_t q = 0; q < product->size(); ++q)
          if ((*product)[q] != 'I')
            errors.safe_append_u(std::string(1, (*product)[q]) + "_ERROR",
                                 {static_cast<std::uint32_t>(task.targets[q])},
                                 {1.});
        if (!errors.operations.empty())
          faults[selection.circuit_location].push_back(
              {std::move(errors), static_cast<std::int64_t>(begin),
               static_cast<std::int64_t>(totalShots)});
      }
    }
    if (totalShots == 0)
      return results;

    flushPendingQubits();
    setToZeroState();
    std::vector<FrameRunner::Segment> program;
    auto appendCommon = [&]() {
      if (!recordedCircuit.operations.empty()) {
        program.push_back({std::move(recordedCircuit)});
        recordedCircuit.clear();
      }
    };
    for (std::size_t i = 0; i < batch.trace.size(); ++i) {
      if (batch.trace[i].type == cudaq::ptsbe::TraceInstructionType::Gate)
        applyGate(cudaq::ptsbe::detail::convertToSimulatorTask<double>(
            batch.trace[i]));
      if (!faults[i].empty()) {
        appendCommon();
        for (auto &fault : faults[i])
          program.push_back({std::move(fault.circuit), fault.begin, fault.end});
      }
    }
    std::vector<std::uint32_t> targets(batch.measureQubits.begin(),
                                       batch.measureQubits.end());
    applyOpToSims("M", targets);
    num_measurements += targets.size();
    appendCommon();
    if (!frameRunner)
      frameRunner = std::make_unique<FrameRunner>();
    auto flips = frameRunner->sampleMasked(program, totalShots, randomEngine());
    const auto words = (totalShots + 31) / 32;
    const auto &reference = tableau->measurement_record.storage;
    if (reference.size() != targets.size() ||
        flips.size() != targets.size() * words)
      throw std::logic_error(
          "cuStabilizer PTSBE measurement layout is invalid.");
    std::size_t offset = 0;
    for (std::size_t t = 0; t < batch.trajectories.size(); ++t) {
      const auto shots = batch.trajectories[t].num_shots;
      auto result = makeSampleResult(flips, words, reference, 0, offset, shots,
                                     batch.includeSequentialData);
      offset += shots;
      results[t] = cudaq::sample_result(std::move(result));
    }
    return results;
  }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    flushAnySamplingTasks();
    applyOpToSims(
        "R", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
  }

  /// @brief Sample the multi-qubit state. If \p qubits is empty and
  /// explicitMeasurements is set, this returns all previously saved
  /// measurements.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    auto executionContext = getExecutionContext();

    if (!executionContext)
      throw std::runtime_error(
          "cuStabilizer sampling requires an execution context.");
    if (shots <= 0 || static_cast<std::size_t>(shots) > getBatchSize())
      throw std::invalid_argument(
          "cuStabilizer sample shots must be positive and no greater than "
          "the execution-context batch size.");

    if (executionContext->explicitMeasurements && qubits.empty() &&
        num_measurements == 0)
      throw std::runtime_error(
          "The sampling option `explicit_measurements` is not supported on a "
          "kernel without any measurement operation.");

    bool populateResult = [&]() {
      if (executionContext->explicitMeasurements)
        return qubits.empty();
      return true;
    }();
    if (!tableau || (!useGpuFrames() && !sampleSim))
      throw std::runtime_error("cuStabilizer state is not initialized. "
                               "Cannot sample from uninitialized state.");
    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    applyOpToSims("M", stimTargetQubits);
    num_measurements += stimTargetQubits.size();

    if (!populateResult)
      return cudaq::ExecutionResult();

    // CUDA-Q consumers (including CUDA-QX) require the chronological raw
    // measurements, corrected by the noiseless reference, not detector bits.
    if (useGpuFrames()) {
      if (!frameRunner)
        frameRunner = std::make_unique<FrameRunner>();
      if (!frameSeed)
        frameSeed = randomEngine();
      auto flips = frameRunner->sample(recordedCircuit, shots, *frameSeed);
      const auto wordsPerRow = (shots + 31) / 32;
      const auto &reference = tableau->measurement_record.storage;
      if (reference.size() != num_measurements ||
          flips.size() != num_measurements * wordsPerRow)
        throw std::logic_error(
            "cuStabilizer measurement layout does not match the reference.");
      const auto first = executionContext->explicitMeasurements
                             ? 0
                             : num_measurements - qubits.size();
      return makeSampleResult(flips, wordsPerRow, reference, first, 0, shots,
                              includeSequentialData);
    }

    // Generate a reference sample
    const std::vector<bool> &v = tableau->measurement_record.storage;
    stim::simd_bits<W> ref(v.size());
    for (size_t k = 0; k < v.size(); k++)
      ref[k] ^= v[k];

    // Now XOR results on a per-shot basis
    stim::simd_bit_table<W> sample = sampleSim->m_record.storage;
    auto nShots = sampleSim->batch_size;

    // This is a slightly modified version of `sample_batch_measurements`, where
    // we already have the `sample` from the frame simulator. It also places the
    // `sample` in a layout amenable to the order of the loops below (shot
    // major).
    sample = sample.transposed();
    if (ref.not_zero())
      for (size_t s = 0; s < nShots; s++)
        sample[s].word_range_ref(0, ref.num_simd_words) ^= ref;

    size_t bits_per_sample = num_measurements;
    std::vector<std::string> sequentialData;
    // Only retain the final "qubits.size()" measurements. All other
    // measurements were mid-circuit measurements that have been previously
    // accounted for and saved.
    assert(bits_per_sample >= qubits.size());
    std::size_t first_bit_to_save = executionContext->explicitMeasurements
                                        ? 0
                                        : bits_per_sample - qubits.size();
    CountsDictionary counts;
    if (includeSequentialData)
      sequentialData.reserve(shots);
    for (std::size_t shot = 0; shot < shots; shot++) {
      std::string aShot(bits_per_sample - first_bit_to_save, '0');
      for (std::size_t b = first_bit_to_save; b < bits_per_sample; b++)
        aShot[b - first_bit_to_save] = sample[shot][b] ? '1' : '0';
      counts[aShot]++;
      if (includeSequentialData)
        sequentialData.push_back(std::move(aShot));
    }
    ExecutionResult result;
    result.counts = std::move(counts);
    if (includeSequentialData)
      result.sequentialData = std::move(sequentialData);
    return result;
  }

  /// @brief Translate chronological measurement indices into Stim record-
  /// reference targets (`lookback | TARGET_RECORD_BIT`). Throws
  /// `std::out_of_range` if any index lies outside `[0, num_measurements)`
  /// so a broken upstream lowering surfaces before an invalid record
  /// reference is appended to the sampling circuit.
  std::vector<std::uint32_t>
  measurementIndicesToRecordTargets(const std::int64_t *indices,
                                    std::size_t count) const {
    std::vector<std::uint32_t> targets;
    targets.reserve(count);
    for (std::size_t i = 0; i < count; i++) {
      const auto idx = indices[i];
      if (idx < 0 || static_cast<std::size_t>(idx) >= num_measurements)
        throw std::out_of_range(
            "QEC: measurement index " + std::to_string(idx) +
            " is out of range [0, " + std::to_string(num_measurements) +
            "); the lowering must produce chronological indices into the "
            "current kernel's measurement record");
      auto lookback = static_cast<std::uint32_t>(num_measurements -
                                                 static_cast<std::size_t>(idx));
      targets.push_back(lookback | stim::TARGET_RECORD_BIT);
    }
    return targets;
  }

  void detector(const std::int64_t *indices, std::size_t count) override {
    // Commit any deferred sample `M` ops so subsequent `DETECTOR rec[-N]`
    // references resolve. In `cudaq::sample` + `explicitMeasurements` mode
    // `mz()` defers the `M` op to flush time; without this nudge a
    // `qec.detector(handle)` immediately following an `mz` would otherwise
    // reference a measurement that has not reached the sampling circuit.
    if (!sampleQubits.empty())
      flushAnySamplingTasks(/*force=*/false);
    auto targets = measurementIndicesToRecordTargets(indices, count);
    if (!targets.empty())
      recordedCircuit.safe_append_u("DETECTOR", targets);
  }

  void logical_observable(const std::int64_t *indices, std::size_t count,
                          std::size_t observable_index) override {
    if (!sampleQubits.empty())
      flushAnySamplingTasks(/*force=*/false);
    auto targets = measurementIndicesToRecordTargets(indices, count);
    if (!targets.empty())
      recordedCircuit.safe_append_ua("OBSERVABLE_INCLUDE", targets,
                                     static_cast<double>(observable_index));
  }

  void pair_detectors(const std::int64_t *prev, const std::int64_t *curr,
                      std::size_t count) override {
    if (!sampleQubits.empty())
      flushAnySamplingTasks(/*force=*/false);
    std::vector<std::vector<std::uint32_t>> all_targets;
    all_targets.reserve(count);
    for (std::size_t i = 0; i < count; i++) {
      const std::int64_t pair[2] = {prev[i], curr[i]};
      all_targets.push_back(measurementIndicesToRecordTargets(pair, 2));
    }
    for (const auto &targets : all_targets)
      if (!targets.empty())
        recordedCircuit.safe_append_u("DETECTOR", targets);
  }

  /// @brief Return the chronological index of the most-recent `mz`.
  std::int64_t getMeasureIndex() const override {
    auto pending = static_cast<std::int64_t>(sampleQubits.size());
    auto committed = static_cast<std::int64_t>(num_measurements);
    auto total = committed + pending;
    if (total == 0)
      return std::numeric_limits<std::int64_t>::max();
    return total - 1;
  }

  bool isStateVectorSimulator() const override { return false; }

  std::string name() const override { return "custabilizer"; }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &) override {
    throw std::runtime_error(
        "Simulation data not available for the stim simulator backend.");
  }

  void configureExecutionContext(const cudaq::msm_policy &policy) override {
    CustabilizerSimulatorBase::configureExecutionContextImpl(policy);
    is_msm_mode = true;
    activeMsmDimensions = policy.dimensions;
  }

  void
  configureExecutionContext(const cudaq::msm_size_policy &policy) override {
    CustabilizerSimulatorBase::configureExecutionContextImpl(policy);
  }

  NVQIR_SIMULATOR_CLONE_IMPL(CustabilizerCircuitSimulator)
};

} // namespace
} // namespace nvqir::custabilizer

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::custabilizer::CustabilizerCircuitSimulator,
                         custabilizer)
#endif
