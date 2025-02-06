import cudaq
from cudaq import operators, spin, Schedule, RungeKuttaIntegrator

import numpy as np
import cupy as cp
import time
from cudaq.util.timing_helper import PerfTrace
import argparse

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=10)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--J_x", type=float, default=2.5)
parser.add_argument("--J_y", type=float, default=1.75)
parser.add_argument("--J_z", type=float, default=3.25)
parser.add_argument("--h", type=float, default=1.0)
parser.add_argument("--dtype", type=str, default="complex128")
params = parser.parse_args()

n_repeats = params.repeats
N = params.L
dimensions = {}
for i in range(N):
    dimensions[i] = 2


# Heisenberg model spin coupling strength
Jx = params.J_x 
Jy = params.J_y
Jz = params.J_y
h = params.h
gamma = params.gamma
# Construct the Hamiltonian
H = h * spin.z(0) 
for i in range(N - 1):
    H += h * spin.z(i + 1)

for i in range(N - 1):
    H += Jx * spin.x(i) * spin.x(i + 1)
    H += Jy * spin.y(i) * spin.y(i + 1)
    H += Jz * spin.z(i) * spin.z(i + 1)

c_ops = [gamma * spin.z(i) for i in range(N)]

steps = np.linspace(0.0, 0.1, 10)
schedule = Schedule(steps, ["time"])

# Initial state vector
psi0 = cudaq.operator.InitialState.ZERO

# Run the simulation
time_data = []
for run in range(n_repeats):
    start = time.time()
    evolution_result = cudaq.evolve(H,
                                    dimensions,
                                    schedule,
                                    psi0,
                                    observables=[],
                                    collapse_operators=c_ops,
                                    store_intermediate_results=False,
                                    integrator=RungeKuttaIntegrator(order=1))
    end = time.time()
    print(f"N = {N}. Run #{run}: elapsed time: {end -start}")
    time_data.append(end -start)

print(f"Summary: average = {np.mean(time_data)}; stddev = {np.std(time_data)}")
# PerfTrace.dump()