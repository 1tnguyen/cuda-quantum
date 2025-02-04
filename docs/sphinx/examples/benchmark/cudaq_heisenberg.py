import cudaq
from cudaq import operators, spin, Schedule, RungeKuttaIntegrator

import numpy as np
import cupy as cp
import os
from cudaq.util.timing_helper import PerfTrace
import argparse

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=15)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--discards", type=int, default=10)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--J_x", type=float, default=2.5)
parser.add_argument("--J_y", type=float, default=1.75)
parser.add_argument("--J_z", type=float, default=3.25)
parser.add_argument("--h", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dtype", type=str, default="complex128")
params = parser.parse_args()


N = params.L
dimensions = {}
for i in range(N):
    dimensions[i] = 2


# Heisenberg model spin coupling strength
Jx = params.J_x 
Jy = params.J_y
Jz = params.J_y
h = params.h
# Construct the Hamiltonian
H = h * spin.z(0) 
for i in range(N - 1):
    H += h * spin.z(i + 1)

for i in range(N - 1):
    H += Jx * spin.x(i) * spin.x(i + 1)
    H += Jy * spin.y(i) * spin.y(i + 1)
    H += Jz * spin.z(i) * spin.z(i + 1)

steps = np.linspace(0.0, 0.1, 10)
schedule = Schedule(steps, ["time"])

# Prepare the initial state vector
psi0_ = cp.zeros(2**N, dtype=cp.complex128)
psi0_[:] = 1/np.sqrt(2**N)
psi0 = cudaq.State.from_data(psi0_)

# Run the simulation
evolution_result = cudaq.evolve(H,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[],
                                collapse_operators=[],
                                store_intermediate_results=False,
                                integrator=RungeKuttaIntegrator())



PerfTrace.dump()