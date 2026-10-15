# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ python3 %s | FileCheck %s

import cudaq

cudaq.set_target("qpp-cpu")


@cudaq.kernel
def memory(num_rounds: int):
    q = cudaq.qubit()
    x(q)
    for _ in range(num_rounds):
        cudaq.idle(q)
    mz(q)


@cudaq.kernel
def broadcast():
    qs = cudaq.qvector(3)
    x(qs)
    idle(qs)
    mz(qs)


print(memory)
print(broadcast)
assert cudaq.sample(memory, 3, shots_count=16)["1"] == 16
assert cudaq.sample(broadcast, shots_count=16)["111"] == 16


@cudaq.kernel
def invalid_rounds():
    q = cudaq.qubit()
    cudaq.idle(q, rounds=2)


try:
    invalid_rounds.compile()
except RuntimeError as error:
    assert "idle does not accept keyword arguments" in str(error)
else:
    raise AssertionError("idle accepted a rounds argument")

# CHECK-LABEL: func.func @__nvqpp__mlirgen__memory
# CHECK: cc.loop
# CHECK: quake.idle {{.*}} : (!quake.ref) -> ()
# CHECK: quake.mz
# CHECK-LABEL: func.func @__nvqpp__mlirgen__broadcast
# CHECK: cc.loop
# CHECK: quake.extract_ref
# CHECK: quake.idle {{.*}} : (!quake.ref) -> ()
# CHECK: quake.mz
