# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from multiprocessing import Process

import cudaq
import pytest
from network_utils import check_server_connection
from utils.mock_qpu import quantum_machines

pytestmark = pytest.mark.xdist_group("quantum_machines_mock_execution")

port = 62548


@pytest.fixture(scope="module", autouse=True)
def start_mock_server():
    os.environ["QUANTUM_MACHINES_API_KEY"] = "00000000000000000000000000000000"
    process = Process(target=quantum_machines.start_server, args=(port,))
    process.start()

    if not check_server_connection(port):
        process.terminate()
        pytest.exit("Mock server did not start in time.", returncode=1)

    yield

    process.terminate()
    process.join()


@pytest.fixture(autouse=True)
def configure_target():
    cudaq.set_target("quantum_machines",
                     url=f"http://localhost:{port}",
                     qubit_mapping_mode="backend",
                     executor="sim")
    yield
    cudaq.reset_target()


def test_histogram_is_simulated():

    @cudaq.kernel
    def always_one():
        qubit = cudaq.qubit()
        x(qubit)

    shots = 16
    counts = cudaq.sample(always_one, shots_count=shots)
    assert dict(counts.items()) == {"1": shots}


def test_output_log_contains_simulated_return_values():
    cudaq.set_target("quantum_machines",
                     url=f"http://localhost:{port}",
                     qubit_mapping_mode="backend",
                     executor="qpu")

    @cudaq.kernel
    def multiply(lhs: int, rhs: int) -> int:
        return lhs * rhs

    assert cudaq.run(multiply, 6, 7, shots_count=2) == [42, 42]
