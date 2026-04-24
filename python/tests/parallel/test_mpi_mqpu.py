# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest
from cudaq import spin
import numpy as np

skipIfUnsupported = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-mqpu')),
    reason="nvidia-mqpu backend not available or mpi not found")


@pytest.fixture(scope='session', autouse=True)
def mpi_init_finalize():
    cudaq.mpi.initialize()
    yield
    cudaq.mpi.finalize()


@pytest.fixture(autouse=True)
def set_up_target():
    cudaq.set_target('nvidia-mqpu')
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def check_mpi(entity):
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    if numQpus == 0:
        pytest.skip("No QPUs available for target, skipping MPI test")
    else:
        print(
            f"Target: {target}, NumQPUs: {numQpus}, MPI Ranks: {cudaq.mpi.num_ranks()}"
        )
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    # Confirmed expectation value for this system when `theta=0.59`.
    want_expectation_value = -1.7487948611472093

    # Get the `cudaq.ObserveResult` back from `cudaq.observe()`.
    # No shots provided.
    result_no_shots = cudaq.observe(entity,
                                    hamiltonian,
                                    0.59,
                                    execution=cudaq.parallel.mpi)
    expectation_value_no_shots = result_no_shots.expectation()
    assert np.isclose(want_expectation_value, expectation_value_no_shots)

    sub_term_expectation_sum = 0.0
    for sub_term in hamiltonian:
        sub_term_expectation_sum += (
            sub_term.evaluate_coefficient() *
            result_no_shots.expectation(sub_term=sub_term))
    assert np.isclose(expectation_value_no_shots, sub_term_expectation_sum.real)

    # Test all gather
    numRanks = cudaq.mpi.num_ranks()
    local = [1.0]
    globalList = cudaq.mpi.all_gather(numRanks, local)
    assert len(globalList) == numRanks


def check_mpi_many_terms(entity):
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    if numQpus == 0:
        pytest.skip("No QPUs available for target, skipping MPI test")

    hamiltonian = (0.37 - 0.91 * spin.x(0) + 0.23 * spin.y(0) -
                   0.41 * spin.z(0) + 0.17 * spin.x(1) - 0.33 * spin.y(1) +
                   0.29 * spin.z(1) - 0.14 * spin.x(2) + 0.27 * spin.y(2) -
                   0.35 * spin.z(2) + 0.12 * spin.x(0) * spin.x(1) -
                   0.19 * spin.x(0) * spin.y(1) + 0.21 * spin.x(0) * spin.z(1) +
                   0.16 * spin.y(0) * spin.x(1) - 0.28 * spin.y(0) * spin.z(1) +
                   0.32 * spin.z(0) * spin.x(1) - 0.11 * spin.z(0) * spin.y(1) +
                   0.26 * spin.x(1) * spin.x(2) - 0.22 * spin.y(1) * spin.y(2) +
                   0.18 * spin.z(1) * spin.z(2) -
                   0.24 * spin.x(0) * spin.y(1) * spin.z(2) +
                   0.13 * spin.y(0) * spin.z(1) * spin.x(2) -
                   0.31 * spin.z(0) * spin.x(1) * spin.y(2) +
                   0.09 * spin.x(0) * spin.x(1) * spin.x(2))

    serial_result = cudaq.observe(entity, hamiltonian, 0.59)
    mpi_result = cudaq.observe(entity,
                               hamiltonian,
                               0.59,
                               execution=cudaq.parallel.mpi)

    assert np.isclose(serial_result.expectation(), mpi_result.expectation())

    sub_term_expectation_sum = 0.0
    for sub_term in hamiltonian:
        sub_term_expectation_sum += (sub_term.evaluate_coefficient() *
                                     mpi_result.expectation(sub_term=sub_term))
    assert np.isclose(mpi_result.expectation(), sub_term_expectation_sum.real)


@skipIfUnsupported
def testMPI():

    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    check_mpi(kernel)


@skipIfUnsupported
def testMPI_kernel():

    @cudaq.kernel
    def kernel(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_mpi(kernel)


@skipIfUnsupported
def testMPI_kernel_many_terms():

    @cudaq.kernel
    def kernel(theta: float):
        qreg = cudaq.qvector(3)
        x(qreg[0])
        ry(theta, qreg[1])
        rz(theta / 2.0, qreg[2])
        x.ctrl(qreg[1], qreg[0])
        y.ctrl(qreg[2], qreg[1])
        z.ctrl(qreg[0], qreg[2])

    check_mpi_many_terms(kernel)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
