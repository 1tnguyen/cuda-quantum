# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from typing import List
import pytest

import cudaq
import numpy as np


@pytest.fixture(autouse=True)
def setTarget():
    if not cudaq.has_target('custabilizer'):
        pytest.skip("cuStabilizer plugin was not built")
    cudaq.set_target('custabilizer')
    yield
    cudaq.reset_target()


def test_custabilizer_non_clifford():
    """The backend preserves Stim's rejection of non-Clifford gates."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        rx(0.1, qubits[0])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert 'Gate not supported by Stim simulator' in repr(e)


def test_custabilizer_toffoli_gates():
    """The backend preserves Stim's rejection of multi-control gates."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        cx(qubits[0:9], qubits[9])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert 'Gates with >1 controls not supported by Stim simulator' in repr(e)


def test_custabilizer_sample():
    """Test basic sampling with large qubit count (custabilizer GPU backend)."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(250)
        h(qubits[0])
        for i in range(1, 250):
            cx(qubits[i - 1], qubits[i])
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert len(counts) == 2
    assert '0' * 250 in counts
    assert '1' * 250 in counts


def test_custabilizer_all_mz_types():
    """Test different measurement basis types."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        mx(qubits)
        my(qubits)
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert len(counts) > 1


def test_custabilizer_state_preparation():
    """Custabilizer does not support state initialization."""

    @cudaq.kernel
    def kernel(vec: List[complex]):
        qubits = cudaq.qvector(vec)

    with pytest.raises(RuntimeError) as e:
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)
    assert 'Simulation data not available' in repr(e)


def test_custabilizer_state_preparation_builder():
    """Custabilizer does not support state initialization via builder."""
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    with pytest.raises(RuntimeError) as e:
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)
    assert 'Simulation data not available' in repr(e)


def test_custabilizer_bell_state():
    """Test Bell state preparation and measurement."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, shots_count=1000)
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts
    assert np.isclose(counts.probability('00'), 0.5, atol=0.2)
    assert np.isclose(counts.probability('11'), 0.5, atol=0.2)


def test_custabilizer_mid_circuit_measurement():
    """Test mid-circuit measurement with conditionals (exercises measureQubit)."""

    @cudaq.kernel
    def kernel() -> List[bool]:
        q = cudaq.qvector(2)
        h(q[0])
        # Mid-circuit measurement that triggers measureQubit call
        result = mz(q[0])
        if result:
            x(q[1])
        return mz(q)

    cudaq.set_random_seed(13)
    results = cudaq.run(kernel, shots_count=1000)
    # Should see 00 and 11 (when q[0] is 0 or 1, q[1] follows)
    assert len(results) == 1000
    assert all(a == b for a, b in results)
    assert np.isclose(np.mean([a for a, _ in results]), 0.5, atol=0.2)


def test_custabilizer_ghz_state():
    """Test GHZ state with varying qubit counts."""

    @cudaq.kernel
    def kernel(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            cx(qubits[i], qubits[i + 1])
        mz(qubits)

    for n in [3, 5, 10]:
        counts = cudaq.sample(kernel, n, shots_count=100)
        assert len(counts) == 2
        assert '0' * n in counts
        assert '1' * n in counts


def test_custabilizer_noise_support():
    """Test that noise models work with custabilizer."""
    noise = cudaq.NoiseModel()
    # Add bit flip noise on X gates
    noise.add_channel('x', [0], cudaq.BitFlipChannel(0.1))

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(5)
        x(q)
        mz(q)

    # Without noise, should always get 11111
    counts_no_noise = cudaq.sample(kernel)
    assert counts_no_noise['11111'] == 1000

    # With noise, should see some bit flips
    counts_with_noise = cudaq.sample(kernel,
                                     noise_model=noise,
                                     shots_count=1000)
    # Should have multiple outcomes due to noise
    assert len(counts_with_noise) > 1


def test_custabilizer_explicit_measurements():
    """Test explicit measurements flag."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[0])
        mz(q[0])
        mz(q[1])
        mz(q[2])

    counts = cudaq.sample(kernel)
    assert counts['100'] == 1000
    counts = cudaq.sample(kernel, explicit_measurements=True)
    assert counts['100'] == 1000


def test_custabilizer_register_names():
    """Test named measurement registers."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        mz(q[0], register_name="first")
        cx(q[0], q[1])
        mz(q, register_name="both")

    result = cudaq.sample(kernel, shots_count=100)
    assert 'first' in result.register_names
    assert 'both' in result.register_names
    first_reg = result.get_register_counts('first')
    assert '0' in first_reg or '1' in first_reg


@pytest.mark.parametrize("shots", [1, 31, 32, 33, 65, 129])
def test_custabilizer_matches_stim_measurement_layout(shots):

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[2])
        first = mz(q[2])
        reset(q[2])
        x(q[0])
        mz(q[1])
        mz(q[0])
        mz(q[2])

    outputs = []
    for target in ['stim', 'custabilizer', 'stim']:
        cudaq.set_target(target)
        result = cudaq.sample(kernel,
                              shots_count=shots,
                              explicit_measurements=True)
        outputs.append(result.get_sequential_data())
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[1]) == shots
    assert outputs[1] == ['1010'] * shots


def test_custabilizer_named_register_correlations():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        mz(q[0], register_name="first")
        mz(q[1], register_name="second")

    for target in ['stim', 'custabilizer', 'stim']:
        cudaq.set_target(target)
        result = cudaq.sample(kernel, shots_count=1025)
        first = result.get_sequential_data('first')
        second = result.get_sequential_data('second')
        assert first == second
        assert set(first) == {'0', '1'}


def test_custabilizer_seed_reproducibility_and_reallocation():

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        h(q[0])
        for i in range(1, n):
            cx(q[i - 1], q[i])
        mz(q)

    for n, shots in [(3, 65), (7, 129), (2, 31), (3, 65)]:
        cudaq.set_random_seed(1234)
        first = cudaq.sample(kernel, n, shots_count=shots)
        cudaq.set_random_seed(1234)
        second = cudaq.sample(kernel, n, shots_count=shots)
        assert first.get_sequential_data() == second.get_sequential_data()
        assert set(first.get_sequential_data()) == {'0' * n, '1' * n}


def test_custabilizer_clifford_gate_phase_parity():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        h(q[0])
        s(q[0])
        s.adj(q[0])
        h(q[0])
        x(q[1])
        y.ctrl(q[1], q[2])
        swap(q[0], q[1])
        mz(q)

    for target in ['stim', 'custabilizer', 'stim']:
        cudaq.set_target(target)
        result = cudaq.sample(kernel, shots_count=65)
        assert result.get_sequential_data() == ['101'] * 65


def test_custabilizer_deterministic_pauli_noise():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q)
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel('x', cudaq.BitFlipChannel(1.0))
    result = cudaq.sample(kernel, shots_count=65, noise_model=noise)
    assert result.get_sequential_data() == ['00'] * 65


@pytest.mark.parametrize("channel,expected", [
    (cudaq.XError(1.0), '0'),
    (cudaq.YError(1.0), '1'),
    (cudaq.ZError(1.0), '1'),
    (cudaq.PhaseFlipChannel(1.0), '1'),
    (cudaq.Pauli1([1.0, 0.0, 0.0]), '0'),
    (cudaq.Pauli1([0.0, 1.0, 0.0]), '1'),
    (cudaq.Pauli1([0.0, 0.0, 1.0]), '1'),
])
def test_custabilizer_single_qubit_pauli_channels(channel, expected):

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        x(q)
        h(q)
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], channel)
    result = cudaq.sample(kernel, noise_model=noise, shots_count=65)
    assert result.get_sequential_data() == [expected] * 65


@pytest.mark.parametrize(
    "paulis", [a + b for a in 'IXYZ' for b in 'IXYZ' if a + b != 'II'])
def test_custabilizer_two_qubit_pauli_channels(paulis):

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cx(q[0], q[1])
        mz(q)

    probabilities = [0.0] * 15
    probabilities[4 * 'IXYZ'.index(paulis[0]) + 'IXYZ'.index(paulis[1]) -
                  1] = 1.0
    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0, 1], cudaq.Pauli2(probabilities))
    expected = ''.join('1' if p in 'XY' else '0' for p in paulis)
    result = cudaq.sample(kernel, noise_model=noise, shots_count=65)
    assert result.get_sequential_data() == [expected] * 65


def test_custabilizer_depolarizing_statistics_match_stim():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cx(q[0], q[1])
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0, 1], cudaq.Depolarization2(0.3))
    for target in ['stim', 'custabilizer', 'stim']:
        cudaq.set_target(target)
        cudaq.set_random_seed(42)
        result = cudaq.sample(kernel, noise_model=noise, shots_count=20001)
        for outcome, expected in [('00', 0.76), ('01', 0.08), ('10', 0.08),
                                  ('11', 0.08)]:
            assert abs(result.probability(outcome) - expected) < 0.02


def test_custabilizer_qec_measurement_contract():

    @cudaq.kernel
    def memory(rounds: int):
        q = cudaq.qvector(2)
        previous = mz(q[1])
        for _ in range(rounds):
            cx(q[0], q[1])
            current = mz(q[1])
            cudaq.detector(current, previous)
            previous = current
            reset(q[1])
        final = mz(q[0])
        cudaq.logical_observable(final)

    outputs = []
    for target in ['stim', 'custabilizer', 'stim']:
        cudaq.set_target(target)
        samples = cudaq.sample(memory,
                               3,
                               shots_count=65,
                               explicit_measurements=True)
        outputs.append(samples.get_sequential_data())
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[1] == ['00000'] * 65


@pytest.mark.parametrize("force_generic", [False, True])
def test_custabilizer_ptsbe_public_api(monkeypatch, force_generic):
    monkeypatch.setenv("CUDAQ_PTSBE_FORCE_GENERIC",
                       "1" if force_generic else "0")

    @cudaq.kernel
    def memory():
        q = cudaq.qubit()
        x(q)
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.BitFlipChannel(0.5))
    cudaq.set_random_seed(123)
    result = cudaq.ptsbe.sample(
        memory,
        noise_model=noise,
        shots_count=4097,
        sampling_strategy=cudaq.ptsbe.ExhaustiveSamplingStrategy(),
        include_sequential_data=True)
    assert sum(result.count(bits) for bits in result) == 4097
    assert len(result.get_sequential_data()) == 4097
    assert result.probability("0") == pytest.approx(0.5,
                                                    abs=6 *
                                                    np.sqrt(0.25 / 4097))


def test_custabilizer_sdk_is_preloaded():
    assert any(
        path.endswith("/libcustabilizer.so.0")
        for path in os.environ["CUDAQ_DYNLIBS"].split(":"))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
