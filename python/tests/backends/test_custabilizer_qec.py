# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q contracts used by downstream QEC libraries; no CUDA-QX dependency."""

from typing import List

import cudaq
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def backend():
    if not cudaq.has_target("custabilizer"):
        pytest.skip("cuStabilizer plugin was not built")
    cudaq.set_target("custabilizer")
    yield
    cudaq.reset_target()


@cudaq.kernel
def parity_checks(data: cudaq.qview, ancillas: cudaq.qview,
                  x_basis: bool) -> List[cudaq.measure_handle]:
    if x_basis:
        h(ancillas)
        for i in range(2):
            cx(ancillas[i], data[i])
            cx(ancillas[i], data[i + 1])
        h(ancillas)
    else:
        for i in range(2):
            cx(data[i], ancillas[i])
            cx(data[i + 1], ancillas[i])
    results = mz(ancillas)
    reset(ancillas)
    return results


@cudaq.kernel
def repeated_checks(x_basis: bool, logical_one: bool, fault_qubit: int,
                    probability: float):
    data = cudaq.qvector(3)
    ancillas = cudaq.qvector(2)
    if logical_one:
        x(data)
    if x_basis:
        h(data)
    previous = parity_checks(data, ancillas, x_basis)
    cudaq.detector(previous[0])
    cudaq.detector(previous[1])
    if fault_qubit >= 0:
        if x_basis:
            cudaq.apply_noise(cudaq.ZError, probability, data[fault_qubit])
        else:
            cudaq.apply_noise(cudaq.XError, probability, data[fault_qubit])
    for _ in range(2):
        current = parity_checks(data, ancillas, x_basis)
        cudaq.detectors(previous, current)
        previous = current
    if x_basis:
        h(data)
    final = mz(data)
    cudaq.detector(previous[0], final[0], final[1])
    cudaq.detector(previous[1], final[1], final[2])
    cudaq.logical_observable(final[0])


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize("logical_one", [False, True])
@pytest.mark.parametrize("shots", [1, 31, 32, 33, 129])
def test_repeated_checks_raw_measurement_layout(x_basis, logical_one, shots):
    """Reset/reuse and QEC annotations preserve chronological measurements."""
    expected_bits = "000000" + ("111" if logical_one else "000")
    for target in ["stim", "custabilizer", "stim"]:
        cudaq.set_target(target)
        result = cudaq.sample(repeated_checks,
                              x_basis,
                              logical_one,
                              -1,
                              0.0,
                              shots_count=shots,
                              explicit_measurements=True)
        assert result.get_sequential_data() == [expected_bits] * shots


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize("fault_qubit", [0, 1, 2])
def test_faults_preserve_measurement_history(x_basis, fault_qubit):
    """A persistent fault affects subsequent checks and final data bits."""
    parities = [int(fault_qubit in (0, 1)), int(fault_qubit in (1, 2))]
    data = [int(i == fault_qubit) for i in range(3)]
    expected_bits = "00" + "".join(map(str, parities * 2 + data))
    for target in ["stim", "custabilizer", "stim"]:
        cudaq.set_target(target)
        result = cudaq.sample(repeated_checks,
                              x_basis,
                              False,
                              fault_qubit,
                              1.0,
                              noise_model=cudaq.NoiseModel(),
                              shots_count=65,
                              explicit_measurements=True)
        assert result.get_sequential_data() == [expected_bits] * 65


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
def test_noisy_samples_preserve_fault_correlations(x_basis):
    """The check history and final logical bit share the same sampled fault."""
    probability, shots = 0.125, 4097
    for target in ["stim", "custabilizer", "stim"]:
        cudaq.set_target(target)
        cudaq.set_random_seed(987)
        result = cudaq.sample(repeated_checks,
                              x_basis,
                              False,
                              0,
                              probability,
                              noise_model=cudaq.NoiseModel(),
                              shots_count=shots,
                              explicit_measurements=True)
        assert set(result) == {"000000000", "001010100"}
        measurements = np.array(
            [list(map(int, bits)) for bits in result.get_sequential_data()],
            dtype=np.uint8)
        assert measurements.shape == (shots, 9)
        first_check_change = measurements[:, 0] ^ measurements[:, 2]
        logical_bits = measurements[:, 6]
        np.testing.assert_array_equal(first_check_change, logical_bits)
        assert np.mean(logical_bits) == pytest.approx(
            probability,
            abs=6 * np.sqrt(probability * (1 - probability) / shots))


def test_observable_annotations_preserve_raw_measurements():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        m = mz(q)
        cudaq.detector(m[0], m[0])
        cudaq.detector(m[0], m[1], m[1])
        cudaq.logical_observable(m[0], observable_index=2)
        cudaq.logical_observable(m[1], m[2], observable_index=2)

    for target in ["stim", "custabilizer", "stim"]:
        cudaq.set_target(target)
        result = cudaq.sample(kernel,
                              shots_count=33,
                              explicit_measurements=True)
        assert result.get_sequential_data() == ["101"] * 33


def test_noisy_conditional_feedback():

    @cudaq.kernel
    def feedback() -> List[bool]:
        q = cudaq.qvector(2)
        cudaq.apply_noise(cudaq.XError, 1.0, q[0])
        outcome = mz(q[0])
        reset(q[0])
        if outcome:
            x(q[1])
        return mz(q)

    for target in ["stim", "custabilizer", "stim"]:
        cudaq.set_target(target)
        results = cudaq.run(feedback,
                            noise_model=cudaq.NoiseModel(),
                            shots_count=33)
        assert len(results) == 33
        assert all(list(bits) == [False, True] for bits in results)
