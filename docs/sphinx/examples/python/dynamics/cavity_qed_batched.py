import cudaq
from cudaq import operators, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrate a simulation of cavity quantum electrodynamics (interaction between light confined in a reflective cavity and atoms)

# System dimensions: atom (2-level system) and cavity (10-level system)
dimensions = {0: 2, 1: 10}

# Alias for commonly used operators
# Cavity operators
a = operators.annihilate(1)
a_dag = operators.create(1)

# Atom operators
sm = operators.annihilate(0)
sm_dag = operators.create(0)

# Defining the Hamiltonian for the system: self-energy terms and cavity-atom interaction term.
# This is the so-called Jaynes-Cummings model:
# https://en.wikipedia.org/wiki/Jaynes%E2%80%93Cummings_model
hamiltonian = 2 * np.pi * operators.number(1) + 2 * np.pi * operators.number(
    0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)

# Initial state of the system
# Atom in ground state
qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)

# Cavity in a state which has 5 photons initially
cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
cavity_state[5][5] = 1.0
rho0_1 = cudaq.State.from_data(cp.kron(qubit_state, cavity_state))

cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
cavity_state[8][8] = 1.0
rho0_2 = cudaq.State.from_data(cp.kron(qubit_state, cavity_state))


steps = np.linspace(0, 10, 201)
schedule = Schedule(steps, ["time"])

evolution_results = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    [rho0_1],
    observables=[operators.number(1), operators.number(0)],
    collapse_operators=[np.sqrt(0.1) * a],
    store_intermediate_results=True,
    integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
results_1 = [
    get_result(0, evolution_results[0]),
    get_result(1, evolution_results[0])
]
results_2 = [
    get_result(0, evolution_results[1]),
    get_result(1, evolution_results[1])
]

fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, results_1[0])
plt.plot(steps, results_1[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
plt.title("Initial photon count = 5")

plt.subplot(1, 2, 2)
plt.plot(steps, results_2[0])
plt.plot(steps, results_2[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
plt.title("Initial photon count = 8")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('cavity_qed.png', dpi=fig.dpi)
