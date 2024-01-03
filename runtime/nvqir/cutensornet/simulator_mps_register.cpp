/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "simulator_cutensornet.h"
#include <charconv>
namespace nvqir {

class SimulatorMPS : public SimulatorTensorNetBase {
  // Default max bond dim
  int64_t m_maxBond = 64;
  // Default absolute cutoff
  double m_absCutoff = 1e-5;
  // Default relative cutoff
  double m_relCutoff = 1e-5;
  std::vector<void *> m_mpsTensors_d;

public:
  SimulatorMPS() : SimulatorTensorNetBase() {
    if (auto *maxBondEnvVar = std::getenv("CUDAQ_MPS_MAX_BOND")) {
      const std::string maxBondStr(maxBondEnvVar);
      int maxBond;
      auto [ptr, ec] = std::from_chars(
          maxBondStr.data(), maxBondStr.data() + maxBondStr.size(), maxBond);
      if (ec != std::errc{} || maxBond < 1)
        throw std::runtime_error("Invalid CUDAQ_MPS_MAX_BOND setting. Expected "
                                 "a positive number. Got: " +
                                 maxBondStr);

      m_maxBond = maxBond;
      cudaq::info("Setting MPS max bond dimension to {}.", m_maxBond);
    }
    // Cutoff values
    if (auto *absCutoffEnvVar = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) {
      const std::string absCutoffStr(absCutoffEnvVar);
      double absCutoff;
      auto [ptr, ec] =
          std::from_chars(absCutoffStr.data(),
                          absCutoffStr.data() + absCutoffStr.size(), absCutoff);
      if (ec != std::errc{} || absCutoff <= 0.0 || absCutoff >= 1.0)
        throw std::runtime_error(
            "Invalid CUDAQ_MPS_ABS_CUTOFF setting. Expected "
            "a number in range (0.0, 1.0). Got: " +
            absCutoffStr);

      m_absCutoff = absCutoff;
      cudaq::info("Setting MPS absolute cutoff to {}.", m_absCutoff);
    }
    if (auto *relCutoffEnvVar = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) {
      const std::string relCutoffStr(relCutoffEnvVar);
      double relCutoff;
      auto [ptr, ec] =
          std::from_chars(relCutoffStr.data(),
                          relCutoffStr.data() + relCutoffStr.size(), relCutoff);
      if (ec != std::errc{} || relCutoff <= 0.0 || relCutoff >= 1.0)
        throw std::runtime_error(
            "Invalid CUDAQ_MPS_RELATIVE_CUTOFF setting. Expected "
            "a number in range (0.0, 1.0). Got: " +
            relCutoffStr);

      m_relCutoff = relCutoff;
      cudaq::info("Setting MPS relative cutoff to {}.", m_relCutoff);
    }
  }

  virtual void prepareQubitTensorState() override {
    LOG_API_TIME();
    // Clean up previously factorized MPS tensors
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor));
    }
    m_mpsTensors_d.clear();
    // Factorize the state:
    if (m_state->getNumQubits() > 1)
      m_mpsTensors_d =
          m_state->factorizeMPS(m_maxBond, m_absCutoff, m_relCutoff);
  }

  virtual void applyGate(const GateApplicationTask &task) override {
    // Check that we don't apply gates on 3+ qubits (not supported in MPS)
    if (task.controls.size() + task.targets.size() > 2) {
      const std::string gateDesc = task.operationName +
                                   containerToString(task.controls) +
                                   containerToString(task.targets);
      throw std::runtime_error(
          "MPS simulator: Internal error - Gates on 3 or more qubits were not "
          "properly handled. Encountered: " +
          gateDesc);
    }
    SimulatorTensorNetBase::applyGate(task);
  }

  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  virtual std::string name() const override { return "tensornet-mps"; }

  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor));
    }
    m_mpsTensors_d.clear();
  }

  void phaseCcx(std::size_t control1, std::size_t control2,
                std::size_t target) {
    // Ref: https://arxiv.org/pdf/1210.0974.pdf#page=2
    h({}, target);
    x({target}, control1);
    x({control1}, control2);
    t({}, control2);
    tdg({}, control1);
    t({}, target);
    x({target}, control1);
    x({control1}, control2);
    tdg({}, control2);
    x({target}, control2);
    h({}, target);
  }

  /// Collects the list of control qubits into auxiliarly qubits.
  void collectControls(const std::vector<std::size_t> &ctls,
                       const std::vector<std::size_t> &aux, int adjustment) {
    for (int i = 0; i <= static_cast<int>(ctls.size()) - 2; i += 2)
      phaseCcx(ctls[i], ctls[i + 1], aux[i / 2]);
    for (int i = 0; i <= (static_cast<int>(ctls.size() / 2) - 2 - adjustment);
         ++i)
      phaseCcx(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
  }

  /// Adjustment for uneven number of original control qubits: the second to
  /// last auxiliary will be collected into the last auxiliary.
  void adjustForSingleControl(const std::vector<std::size_t> &ctls,
                              const std::vector<std::size_t> &aux) {
    if (ctls.size() % 2 != 0)
      phaseCcx(ctls.back(), aux[ctls.size() - 3], aux[ctls.size() - 2]);
  }

  void ccz(std::size_t control1, std::size_t control2, std::size_t target) {
    // Ref: https://arxiv.org/pdf/1206.0758v3.pdf#page=15
    tdg({}, control1);
    tdg({}, control2);
    x({target}, control1);
    t({}, control1);
    x({control2}, target);
    x({control2}, control1);
    t({}, target);
    tdg({}, control1);
    x({control2}, target);
    x({target}, control1);
    tdg({}, target);
    t({}, control1);
    x({control2}, control1);
  }

  // Rotation basis mapping
  void mapPauli(std::size_t qubit, cudaq::pauli from, cudaq::pauli to) {
    if (from == to)
      return;
    if ((from == cudaq::pauli::Z && to == cudaq::pauli::X) ||
        (from == cudaq::pauli::X && to == cudaq::pauli::Z)) {
      h({}, qubit);
    } else if (from == cudaq::pauli::Z && to == cudaq::pauli::Y) {
      h({}, qubit);
      s({}, qubit);
      h({}, qubit);
    } else if (from == cudaq::pauli::Y && to == cudaq::pauli::Z) {
      h({}, qubit);
      sdg({}, qubit);
      h({}, qubit);
    } else if (from == cudaq::pauli::Y && to == cudaq::pauli::X) {
      s({}, qubit);
    } else if (from == cudaq::pauli::X && to == cudaq::pauli::Y) {
      sdg({}, qubit);
    } else {
      throw std::runtime_error("Unsupported Pauli mapping");
    }
  }

  void ccnot(std::size_t control1, std::size_t control2, std::size_t target) {
    mapPauli(target, cudaq::pauli::Z, cudaq::pauli::X);
    ccz(control1, control2, target);
    mapPauli(target, cudaq::pauli::X, cudaq::pauli::Z);
  }

  void ccy(std::size_t control1, std::size_t control2, std::size_t target) {
    mapPauli(target, cudaq::pauli::Z, cudaq::pauli::Y);
    ccz(control1, control2, target);
    mapPauli(target, cudaq::pauli::Y, cudaq::pauli::Z);
  }

  void cch(std::size_t control1, std::size_t control2, std::size_t target) {
    s({}, target);
    h({}, target);
    t({}, target);
    ccnot(control1, control2, target);
    t({}, target);
    h({}, target);
    s({}, target);
  }

  /// @brief  Override general X gate with multi-control decomposition.
  virtual void x(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::x(controls, qubitIdx);
    if (controls.size() == 2) {
      return ccnot(controls[0], controls[1], qubitIdx);
    } else {
      auto aux = allocateQubits(controls.size() - 2);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      if (controls.size() % 2 != 0)
        ccnot(controls.back(), aux[controls.size() - 3], qubitIdx);
      else
        ccnot(aux[controls.size() - 3], aux[controls.size() - 4], qubitIdx);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general Y gate with multi-control decomposition.
  virtual void y(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::y(controls, qubitIdx);
    if (controls.size() == 2) {
      return ccy(controls[0], controls[1], qubitIdx);
    } else {
      auto aux = allocateQubits(controls.size() - 2);
      collectControls(controls, aux, 1 - (controls.size() % 2));

      if (controls.size() % 2 != 0)
        ccy(controls.back(), aux[controls.size() - 3], qubitIdx);
      else
        ccy(aux[controls.size() - 3], aux[controls.size() - 4], qubitIdx);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general Z gate with multi-control decomposition.
  virtual void z(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::z(controls, qubitIdx);
    if (controls.size() == 2) {
      return ccz(controls[0], controls[1], qubitIdx);
    } else {
      auto aux = allocateQubits(controls.size() - 2);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      if (controls.size() % 2 != 0)
        ccz(controls.back(), aux[controls.size() - 3], qubitIdx);
      else
        ccz(aux[controls.size() - 3], aux[controls.size() - 4], qubitIdx);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general H gate with multi-control decomposition.
  virtual void h(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::h(controls, qubitIdx);
    if (controls.size() == 2) {
      return cch(controls[0], controls[1], qubitIdx);
    } else {
      auto aux = allocateQubits(controls.size() - 2);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      if (controls.size() % 2 != 0)
        cch(controls.back(), aux[controls.size() - 3], qubitIdx);
      else
        cch(aux[controls.size() - 3], aux[controls.size() - 4], qubitIdx);
      collectControls(controls, aux, 1 - (controls.size() % 2));
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general RX gate with multi-control decomposition.
  virtual void rx(const double angle, const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::rx(angle, controls, qubitIdx);
    else {
      mapPauli(qubitIdx, cudaq::pauli::Z, cudaq::pauli::X);
      rz(angle, controls, qubitIdx);
      mapPauli(qubitIdx, cudaq::pauli::X, cudaq::pauli::Z);
    }
  }

  /// @brief  Override general RY gate with multi-control decomposition.
  virtual void ry(const double angle, const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::ry(angle, controls, qubitIdx);
    else {
      mapPauli(qubitIdx, cudaq::pauli::Z, cudaq::pauli::Y);
      rz(angle, controls, qubitIdx);
      mapPauli(qubitIdx, cudaq::pauli::Y, cudaq::pauli::Z);
    }
  }

  /// @brief  Override general RZ gate with multi-control decomposition.
  virtual void rz(const double angle, const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::rz(angle, controls, qubitIdx);
    else {
      auto aux = allocateQubits(controls.size() - 1);
      collectControls(controls, aux, 0);
      adjustForSingleControl(controls, aux);
      SimulatorTensorNetBase::rz(angle, {aux[controls.size() - 2]}, qubitIdx);
      adjustForSingleControl(controls, aux);
      collectControls(controls, aux, 0);
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general R1 gate with multi-control decomposition.
  virtual void r1(const double angle, const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::r1(angle, controls, qubitIdx);
    else {
      auto aux = allocateQubits(controls.size() - 1);
      collectControls(controls, aux, 0);
      adjustForSingleControl(controls, aux);
      SimulatorTensorNetBase::r1(angle, {aux[controls.size() - 2]}, qubitIdx);
      adjustForSingleControl(controls, aux);
      collectControls(controls, aux, 0);
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general U1 gate
  /// Delegate to R1 with multi-control decomposition support.
  virtual void u1(const double angle, const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    return r1(angle, controls, qubitIdx);
  }

  /// @brief  Override general U3 gate with multi-control decomposition.
  virtual void u3(const double theta, const double phi, const double lambda,
                  const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) override {
    if (controls.size() <= 1)
      return SimulatorTensorNetBase::u3(theta, phi, lambda, controls, qubitIdx);
    else {
      auto aux = allocateQubits(controls.size() - 1);
      collectControls(controls, aux, 0);
      adjustForSingleControl(controls, aux);
      SimulatorTensorNetBase::u3(theta, phi, lambda, {aux[controls.size() - 2]},
                                 qubitIdx);
      adjustForSingleControl(controls, aux);
      collectControls(controls, aux, 0);
      for (const auto &qId : aux) {
        tracker.returnIndex(qId);
        --nQubitsAllocated;
      }
    }
  }

  /// @brief  Override general T gate
  /// Delegate to R1 with multi-control decomposition support.
  virtual void t(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    return r1(M_PI_4, controls, qubitIdx);
  }
  /// @brief  Override general Tdg gate
  /// Delegate to R1 with multi-control decomposition support.
  virtual void tdg(const std::vector<std::size_t> &controls,
                   const std::size_t qubitIdx) override {
    return r1(-M_PI_4, controls, qubitIdx);
  }
  /// @brief  Override general S gate
  /// Delegate to R1 with multi-control decomposition support.
  virtual void s(const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    return r1(M_PI_2, controls, qubitIdx);
  }
  /// @brief  Override general Sdg gate
  /// Delegate to R1 with multi-control decomposition support.
  virtual void sdg(const std::vector<std::size_t> &controls,
                   const std::size_t qubitIdx) override {
    return r1(-M_PI_2, controls, qubitIdx);
  }
  /// @brief  Override general swap gate with multi-control decomposition.
  virtual void swap(const std::vector<std::size_t> &controls,
                    const std::size_t qubitIdx1,
                    const std::size_t qubitIdx2) override {
    if (controls.empty()) {
      return SimulatorTensorNetBase::swap(controls, qubitIdx1, qubitIdx2);
    } else {
      x({qubitIdx1}, qubitIdx2);
      auto ctls = controls;
      ctls.emplace_back(qubitIdx2);
      x(ctls, qubitIdx1);
      x({qubitIdx1}, qubitIdx2);
    }
  }
};
} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
