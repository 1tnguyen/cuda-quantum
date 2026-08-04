# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import sys

cudaq.set_target("quake_fake")


@cudaq.kernel
def taken_early_return_expected_1_unwind_expected_1_if_expected_0_cfg() -> int:
    qvector = cudaq.qvector(3)

    x(qvector[0])
    measured = mz(qvector[0])
    if measured:
        x.ctrl(qvector[0], qvector[1])
        return mz(qvector[1])
    return 2


@cudaq.kernel
def fallthrough_return_expected_1_unwind_expected_1_if_expected_0_cfg() -> int:
    qvector = cudaq.qvector(3)

    measured = mz(qvector[0])
    if measured:
        x(qvector[1])
        return mz(qvector[1])
    return 2


@cudaq.kernel
def nested_early_return_expected_2_unwinds_expected_2_ifs_expected_0_cfg(
) -> int:
    qvector = cudaq.qvector(2)

    x(qvector[0])
    if mz(qvector[0]):
        if mz(qvector[1]):
            return 3
        x(qvector[1])
        return mz(qvector[1])
    return 2


@cudaq.kernel
def loop_early_return_expected_1_unwind_expected_1_if_expected_1_loop_expected_0_cfg(
) -> int:
    q = cudaq.qubit()

    for i in range(3):
        x(q)
        if i == 1:
            return mz(q)
    return 2


@cudaq.kernel
def wire_index_callee(q: cudaq.qview):
    for i in range(3):
        x(q[i])


@cudaq.kernel
def wire_index_caller_expected_0_loops_expected_0_unwinds_expected_0_ifs_expected_0_cfg_expected_0_qkernel_calls(
) -> int:
    q = cudaq.qvector(3)
    wire_index_callee(q)
    return int(mz(q[0])) + 2 * int(mz(q[1])) + 4 * int(mz(q[2]))


@cudaq.kernel
def fixed_wire_callee(q: cudaq.qview):
    for i in range(2):
        x(q[0])


@cudaq.kernel
def fixed_wire_caller_expected_1_loop_expected_0_unwinds_expected_0_ifs_expected_0_cfg_expected_0_qkernel_calls(
) -> int:
    q = cudaq.qvector(3)
    fixed_wire_callee(q)
    return int(mz(q[0]))


@cudaq.kernel
def callee_with_early_return(q: cudaq.qview) -> int:
    if mz(q[0]):
        return 1
    return 2


@cudaq.kernel
def taken_callee_return_expected_0_loops_expected_1_unwind_expected_1_if_expected_0_cfg_expected_0_qkernel_calls(
) -> int:
    q = cudaq.qvector(3)
    x(q[0])
    value = callee_with_early_return(q)
    x(q[2])
    return value + 4 * int(mz(q[2]))


@cudaq.kernel
def untaken_callee_return_expected_0_loops_expected_1_unwind_expected_1_if_expected_0_cfg_expected_0_qkernel_calls(
) -> int:
    q = cudaq.qvector(3)
    value = callee_with_early_return(q)
    x(q[2])
    return value + 4 * int(mz(q[2]))


def check_results(kernel, expected):
    results = cudaq.run(kernel, shots_count=5)
    assert len(results) == 5
    for result in results:
        assert result == expected, f"expected {expected}, got {result}"


try:
    check_results(
        taken_early_return_expected_1_unwind_expected_1_if_expected_0_cfg, 1)
    check_results(
        fallthrough_return_expected_1_unwind_expected_1_if_expected_0_cfg, 2)
    check_results(
        nested_early_return_expected_2_unwinds_expected_2_ifs_expected_0_cfg, 1)
    check_results(
        loop_early_return_expected_1_unwind_expected_1_if_expected_1_loop_expected_0_cfg,
        0)
    check_results(
        wire_index_caller_expected_0_loops_expected_0_unwinds_expected_0_ifs_expected_0_cfg_expected_0_qkernel_calls,
        7)
    check_results(
        fixed_wire_caller_expected_1_loop_expected_0_unwinds_expected_0_ifs_expected_0_cfg_expected_0_qkernel_calls,
        0)
    check_results(
        taken_callee_return_expected_0_loops_expected_1_unwind_expected_1_if_expected_0_cfg_expected_0_qkernel_calls,
        5)
    check_results(
        untaken_callee_return_expected_0_loops_expected_1_unwind_expected_1_if_expected_0_cfg_expected_0_qkernel_calls,
        6)
except Exception as error:
    print(error)
    sys.exit(1)
