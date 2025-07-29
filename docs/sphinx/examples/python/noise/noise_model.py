import cudaq
import numpy as np

# A simple physical device noise model based on characteristics of a quantum device + gate benchmarking.
# Qubit properties (arbitrary values for example purposes)
T1 = 10
T2 = 1

def amplitude_damping_model(gate_time):
    """
    Compute amplitude damping noise based on T1 time
    """
    error_probability = 1 - np.exp(-gate_time / T1)
    return cudaq.AmplitudeDampingChannel(error_probability)


def dephasing_model(gate_time):
    """
    Compute dephasing noise based on T2 time
    """
    error_probability = 1 - np.exp(-gate_time / T2)
    return cudaq.PhaseDamping(error_probability)

def amplitude_damping_model_2q(gate_time):
    """
    Compute 2-qubit amplitude damping noise based on T1 time
    """
    error_probability = 1 - np.exp(-gate_time / T1)
    kraus_0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - error_probability)]],
                       dtype=np.complex128)
    kraus_1 = np.array([[0.0, np.sqrt(error_probability)], [0.0, 0.0]],
                       dtype=np.complex128)
    k_00 = np.kron(kraus_0, kraus_0)
    k_01 = np.kron(kraus_0, kraus_1)
    k_10 = np.kron(kraus_1, kraus_0)
    k_11 = np.kron(kraus_1, kraus_1)
    kraus_operators = [k_00, k_01, k_10, k_11]
    return cudaq.KrausChannel(kraus_operators)


def dephasing_model_2q(gate_time):
    """
    Compute 2-qubit dephasing noise based on T2 time
    """
    error_probability = 1 - np.exp(-gate_time / T2)
    kraus_0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - error_probability)]],
                       dtype=np.complex128)
    kraus_1 = np.array([[0.0, 0.0], [0.0, np.sqrt(error_probability)]],
                       dtype=np.complex128)
    k_00 = np.kron(kraus_0, kraus_0)
    k_01 = np.kron(kraus_0, kraus_1)
    k_10 = np.kron(kraus_1, kraus_0)
    k_11 = np.kron(kraus_1, kraus_1)
    kraus_operators = [k_00, k_01, k_10, k_11]
    return cudaq.KrausChannel(kraus_operators)

# Example of gate time (arbitrary values for example purposes)
gate_time_1q = 0.001 
gate_time_2q = 0.005

# Gate error probabilities as depolarization noise
error_probability_1q = 0.001 # (99.9% fidelity) 
error_probability_2q = 0.005 # (99.5% fidelity)

# Some additional depolarization noise model
depol_1q = cudaq.Depolarization1(error_probability_1q)
depol_2q = cudaq.Depolarization2(error_probability_2q)

# Add noise channels to our noise model.
noise_model = cudaq.NoiseModel()

for gate in ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]:
    # Add amplitude damping noise for single qubit gates
    noise_model.add_all_qubit_channel(gate, amplitude_damping_model(gate_time_1q))
    # Add dephasing noise for single qubit gates
    noise_model.add_all_qubit_channel(gate, dephasing_model(gate_time_1q))
    # Add depolarization noise for single qubit gates
    noise_model.add_all_qubit_channel(gate, depol_1q)

for gate in ["cx", "cz"]:
    # Add amplitude damping noise for two qubit gates
    noise_model.add_all_qubit_channel(gate, amplitude_damping_model_2q(gate_time_2q))
    # Add dephasing noise for two qubit gates
    noise_model.add_all_qubit_channel(gate, dephasing_model_2q(gate_time_2q))
    # Add depolarization noise for two qubit gates
    noise_model.add_all_qubit_channel(gate, depol_2q)   


