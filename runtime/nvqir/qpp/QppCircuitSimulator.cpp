/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"

#include <bit>
#include <iostream>
#include <qpp.h>
#include <set>
#include <span>

using namespace cudaq;

namespace nvqir {

/// @brief QppState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Circuit Simulator.
template <typename StateType>
struct QppState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  StateType state;
  // Default ctor: create an empty state
  QppState() = default;
  // Copy ctor
  QppState(const QppState &other) : state(other.state) {}
  // Data move ctor
  QppState(StateType &&data) : state(std::move(data)) {}
  QppState(const std::vector<std::size_t> &shape,
           const std::vector<std::complex<double>> &data) {
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      if (shape.size() != 1)
        throw std::runtime_error(
            "QppState must be created from data with 1D shape.");

      state = Eigen::Map<StateType>(
          const_cast<std::complex<double> *>(data.data()), shape[0]);
    } else {
      if (shape.size() != 2)
        throw std::runtime_error("QppState (density matrix) must be created "
                                 "from data with 2D shape.");

      state = Eigen::Map<StateType>(
          const_cast<std::complex<double> *>(data.data()), shape[0], shape[1]);
    }
  }

  std::size_t getNumQubits() const override {
    return std::is_same_v<StateType, qpp::ket> ? std::log2(state.size())
                                               : std::log2(state.rows());
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      if (other.getNumTensors() != 1 ||
          (other.getTensor().extents != getTensor().extents))
        throw std::runtime_error(
            "[qpp-state] overlap error - other state "
            "dimension not equal to this state dimension.");

      std::span<std::complex<double>> otherState(
          reinterpret_cast<std::complex<double> *>(other.getTensor().data),
          other.getTensor().extents[0]);
      return std::inner_product(
                 state.begin(), state.end(), otherState.begin(),
                 complex{0., 0.}, [](auto a, auto b) { return a + b; },
                 [](auto a, auto b) { return std::abs(a * std::conj(b)); })
          .real();
    } else {
      if (other.getNumTensors() != 1 ||
          (other.getTensor().extents != getTensor().extents))
        throw std::runtime_error(
            "[qpp-dm-state] overlap error - other state "
            "dimension not equal to this state dimension.");
      // Create rho and sigma matrices
      Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
          state.data(), getTensor().extents[0], getTensor().extents[1]);
      Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
          reinterpret_cast<std::complex<double> *>(other.getTensor().data),
          other.getTensor().extents[0], other.getTensor().extents[1]);

      // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
      // sqrt(det(rho)*det(sigma))
      auto detprod = rho.determinant() * sigma.determinant();
      return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
    }
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(fmt::format(
          "[qpp-state] getAmplitude with an invalid number of bits in the "
          "basis state: expected {}, provided {}.",
          getNumQubits(), basisState.size()));
    if (std::any_of(basisState.begin(), basisState.end(),
                    [](int x) { return x != 0 && x != 1; }))
      throw std::runtime_error(
          "[qpp-state] getAmplitude with an invalid basis state: only "
          "qubit state (0 or 1) is supported.");

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    // Returns the element at index for state vector or the diagonal element for
    // density matrix.
    return std::is_same_v<StateType, qpp::ket> ? state(idx) : state(idx, idx);
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");
    return std::is_same_v<StateType, qpp::ket>
               ? Tensor{reinterpret_cast<void *>(
                            const_cast<std::complex<double> *>(state.data())),
                        std::vector<std::size_t>{
                            static_cast<std::size_t>(state.size())},
                        getPrecision()}
               : Tensor{reinterpret_cast<void *>(
                            const_cast<std::complex<double> *>(state.data())),
                        std::vector<std::size_t>{
                            static_cast<std::size_t>(state.rows()),
                            static_cast<std::size_t>(state.cols())},
                        getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  // /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");

    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      if (indices.size() != 1)
        throw std::runtime_error("[qpp-state] invalid element extraction. "
                                 "Expect one index for a ket state.");
      return state(indices[0]);
    }

    if (indices.size() != 2)
      throw std::runtime_error("[qpp-state] invalid element extraction. Expect "
                               "two indices for a density matrix state.");

    return state(indices[0], indices[1]);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    if (std::is_same_v<StateType, qpp::ket>)
      return std::make_unique<QppState<qpp::ket>>(Eigen::Map<qpp::ket>(
          reinterpret_cast<std::complex<double> *>(ptr), size));
    else
      return std::make_unique<QppState<qpp::cmat>>(
          Eigen::Map<qpp::cmat>(reinterpret_cast<std::complex<double> *>(ptr),
                                std::sqrt(size), std::sqrt(size)));
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    StateType k;
    state = k;
  }
};

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  /// The simulation state representation (holding a qpp::ket or qpp::cmat).
  // Note: The simulator needs to be able to accept and operate on a
  // pre-existing state (cudaq::SimulationState subtype), hence it doesn't
  // directly hold qpp::ket/qpp::cmat by value.
  std::unique_ptr<QppState<StateType>> simState =
      std::make_unique<QppState<StateType>>();
  bool ownsStateVector = true;
  /// @brief Convert internal qubit index to Q++ qubit index.
  ///
  /// In Q++, qubits are indexed from left to right, and thus q0 is the leftmost
  /// qubit. Internally, in CUDA Quantum, qubits are index from right to left,
  /// hence q0 is the rightmost qubit. Example:
  /// ```
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 3  2  1  0 : CUDA Quantum indices
  /// ```
  std::size_t convertQubitIndex(std::size_t qubitIndex) {
    assert(stateDimension > 0 && "The state is empty, and thus has no qubits");
    return std::log2(stateDimension) - qubitIndex - 1;
  }

  /// @brief Compute the expectation value <Z...Z> over the given qubit indices.
  double calculateExpectationValue(const std::vector<std::size_t> &qubits) {
    std::size_t bitmask = 0;
    for (auto q : qubits)
      bitmask |= (1ULL << q);
    const auto hasEvenParity = [&bitmask](std::size_t x) -> bool {
      return std::popcount(x & bitmask) % 2 == 0;
    };

    std::vector<double> result;
    auto &state = simState->state;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      result.resize(stateDimension);
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (std::size_t i = 0; i < stateDimension; ++i)
        result[i] = (hasEvenParity(i) ? 1.0 : -1.0) * std::norm(state[i]);
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
      result.resize(state.rows());
#ifdef CUDAQ_HAS_OPENMP
#pragma omp parallel for
#endif
      for (Eigen::Index i = 0; i < state.rows(); ++i)
        result[i] = hasEvenParity(i) ? diag(i).real() : -diag(i).real();
    }

    // Accumulate outside the for loop to ensure repeatability
    return std::accumulate(result.begin(), result.end(), 0.0);
  }

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nTargets) {
    auto nRows = (1UL << nTargets);
    assert(data.size() == nRows * nRows &&
           "Invalid number of gate matrix elements passed to toQppMatrix");

    // we represent row major, they represent column major
    return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>(
        const_cast<std::complex<double> *>(data.data()), nRows, nRows);
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    auto &state = simState->state;
    if (qubitCount == 0)
      return;

    auto *stateData = reinterpret_cast<std::complex<double> *>(
        const_cast<void *>(stateDataIn));

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      if (stateData == nullptr) {
        state = qpp::ket::Zero(stateDimension);
        state(0) = 1.0;
      } else
        state = qpp::ket::Map(stateData, stateDimension);
      return;
    }
    // If we are resizing an existing, allocate
    // a zero state on a n qubit, and Kron-prod
    // that with the existing state.
    if (stateData == nullptr) {
      qpp::ket zero_state = qpp::ket::Zero((1UL << qubitCount));
      zero_state(0) = 1.0;
      state = qpp::kron(zero_state, state);
    } else {
      qpp::ket initState = qpp::ket::Map(stateData, (1UL << qubitCount));
      state = qpp::kron(initState, state);
    }
    return;
  }

  void addQubitsToState(cudaq::SimulationState *initState,
                        AllocatorFlag flag) override {
    // Check if it is the state of this Simulator
    auto *statePtr = dynamic_cast<QppState<StateType> *>(initState);
    if (!statePtr)
      throw std::runtime_error("Incompatible initial state provided.");
    if (simState->state.size() == 0) {
      switch (flag) {
      case nvqir::CircuitSimulator::AllocatorFlag::OwnershipTransfer: {
        // This is a move: recapture the pointer as a unique pointer and mark
        // that we own it.
        simState = std::unique_ptr<QppState<StateType>>(statePtr);
        ownsStateVector = true;
        break;
      }
      case nvqir::CircuitSimulator::AllocatorFlag::ConstReference: {
        // This is a const ref -> perform a copy.
        simState->state = statePtr->state;
        ownsStateVector = true;
        break;
      }
      case nvqir::CircuitSimulator::AllocatorFlag::Reference: {
        // This is a ref (non-const) -> wrap it yet marking that we don't own it
        // for proper handling.
        simState = std::unique_ptr<QppState<StateType>>(statePtr);
        ownsStateVector = false;
        break;
      }
      default:
        __builtin_unreachable();
      }
    } else {
      simState->state = qpp::kron(statePtr->state, simState->state);
      if (flag == nvqir::CircuitSimulator::AllocatorFlag::OwnershipTransfer) {
        // The state is sent on by a move, delete the input state.
        statePtr->destroyState();
        delete statePtr;
      }
    }
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    // We don't own this state, just release it.
    if (!ownsStateVector)
      simState.release();

    simState = std::make_unique<QppState<StateType>>();
    ownsStateVector = true;
  }

  void applyGate(const GateApplicationTask &task) override {
    auto &state = simState->state;
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    // First, convert all of the qubit indices to big endian.
    std::vector<std::size_t> controls;
    for (auto index : task.controls) {
      controls.push_back(convertQubitIndex(index));
    }
    std::vector<std::size_t> targets;
    for (auto index : task.targets) {
      targets.push_back(convertQubitIndex(index));
    }

    if (controls.empty()) {
      state = qpp::apply(state, matrix, targets);
      return;
    }
    state = qpp::applyCTRL(state, matrix, controls, targets);
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
    auto &state = simState->state;
    state = qpp::ket::Zero(stateDimension);
    state(0) = 1.0;
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    auto &state = simState->state;
    const auto qubitIdx = convertQubitIndex(index);
    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple =
        qpp::measure(state, qpp::cmat::Identity(2, 2), {qubitIdx},
                     /*qudit dimension=*/2, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.size());
    } else {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.rows(),
                                          collapsed_state.cols());
    }
    cudaq::info("Measured qubit {} -> {}", qubitIdx, measurement_result);
    return measurement_result == 1 ? true : false;
  }

public:
  QppCircuitSimulator() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }
  virtual ~QppCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    qpp::RandomDevices::get_instance().get_prng().seed(seed);
  }

  bool canHandleObserve() override {
    // Do not compute <H> from matrix if shots based sampling requested
    if (executionContext &&
        executionContext->shots != static_cast<std::size_t>(-1)) {
      return false;
    }

    return !shouldObserveFromSampling();
  }

  cudaq::observe_result observe(const cudaq::spin_op &op) override {

    flushGateQueue();
    auto &state = simState->state;
    // The op is on the following target bits.
    std::vector<std::size_t> targets;
    op.for_each_term([&](cudaq::spin_op &term) {
      term.for_each_pauli(
          [&](cudaq::pauli p, std::size_t idx) { targets.push_back(idx); });
    });

    std::sort(targets.begin(), targets.end());
    const auto last_iter = std::unique(targets.begin(), targets.end());
    targets.erase(last_iter, targets.end());

    // Get the matrix as an Eigen matrix
    auto matrix = op.to_matrix();
    qpp::cmat asEigen =
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>>(
            matrix.data(), matrix.rows(), matrix.cols());

    // Compute the expected value
    double ee = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      qpp::ket k = qpp::apply(state, asEigen, targets, 2);
      ee = state.dot(k).real();
    } else {
      ee = qpp::apply(asEigen, state, targets).trace().real();
    }

    return cudaq::observe_result(ee, op,
                                 cudaq::sample_result(cudaq::ExecutionResult(
                                     {}, op.to_string(false), ee)));
  }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    auto &state = simState->state;
    const auto qubitIdx = convertQubitIndex(index);
    state = qpp::reset(state, {qubitIdx});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    auto &state = simState->state;
    if (shots < 1) {
      double expectationValue = calculateExpectationValue(qubits);
      cudaq::info("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    std::vector<std::size_t> measuredBits;
    for (auto index : qubits) {
      measuredBits.push_back(convertQubitIndex(index));
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstream;
    cudaq::ExecutionResult counts;

    // Expectation value from the counts
    double expVal = 0.0;
    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstream << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      auto bitstring = bitstream.str();
      counts.appendResult(bitstring, count);
      auto par = cudaq::sample_result::has_even_parity(bitstring);
      auto p = count / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
      // Reset the state.
      bitstream.str("");
      bitstream.clear();
    }

    counts.expectationValue = expVal;
    return counts;
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    if (!ownsStateVector) {
      // Create a copy
      return std::make_unique<QppState<StateType>>(*simState);
    }
    auto currentState = std::move(simState);
    simState = std::make_unique<QppState<StateType>>();
    return currentState;
  }

  bool isStateVectorSimulator() const override {
    return std::is_same_v<StateType, qpp::ket>;
  }

  /// @brief Primarily used for testing.
  auto getStateVector() {
    flushGateQueue();
    return simState->state;
  }
  std::string name() const override { return "qpp"; }
  NVQIR_SIMULATOR_CLONE_IMPL(QppCircuitSimulator<StateType>)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::QppCircuitSimulator<qpp::ket>, qpp)
#endif
