from noise_model import *
cudaq.set_target("density-matrix-cpu") # 'nvidia' for trajectory simulation on GPU 

# Param
num_qubits = 8
num_iterations = 1 # Number of Grover iterations (rounds of amplification)

# Change the secret string to whatever you prefer
secret_string = np.random.randint(2, size=(num_qubits,))
print(f"secret bitstring to find = {secret_string}")

# Define an 'oracle' kernel so that kernel|state_to_find> = -|state_to_find> but fixes 
# all other computational basis states.
@cudaq.kernel
def oracle(qubits: cudaq.qview, secret_string: list[int]):
    for index, bit in enumerate(secret_string):
        if bit == 0:
            x(qubits[index])
    # Last qubit is the auxiliary qubit.
    # FIXME: decompose multi-qubit controlled gates so that noise model can be applied.
    z.ctrl(qubits.front(len(qubits) - 1), qubits.back())
    for index, bit in enumerate(secret_string):
        if bit == 0:
            x(qubits[index])
            
# Reflection about the all zero state
@cudaq.kernel
def all_zero_reflection(qubits: cudaq.qview):
    num_qubits = len(qubits)
    x(qubits)    
    # FIXME: decompose multi-qubit controlled gates so that noise model can be applied.
    z.ctrl(qubits[0:num_qubits-1], qubits[num_qubits-1])
    x(qubits)

# Reflection about the equal superposition state
# Wrap the all_zero_reflection kernel with hadamard gates applied to the n qubits
@cudaq.kernel
def reflection_about_xi(qubits : cudaq.qview):
    h(qubits)
    all_zero_reflection(qubits)
    h(qubits)

# Define the Grover diffusion operator

@cudaq.kernel
def diffusion_operator(qubits: cudaq.qview):
    # Apply Hadamard gates
    h(qubits)
    # Apply rotation about the all zero state
    all_zero_reflection(qubits)
    # Apply Hadamard gates
    h(qubits)


# Apply the Grover diffusion operation to the equal superposition state
@cudaq.kernel
def grover(num_qubits: int, num_iterations:int, secret_string: list[int]):
    qubits = cudaq.qvector(num_qubits)
    
    # Initialize qubits in an equal superposition state
    h(qubits)
    for _ in range(num_iterations):
        # Apply the oracle
        oracle(qubits, secret_string)
        # Apply the diffusion operator
        diffusion_operator(qubits)
    
    # Measure all qubits, except the auxillary qubit
    mz(qubits)

# Sample 
sample_result = cudaq.sample(grover, num_qubits, num_iterations, secret_string, shots_count = 5000, noise_model=noise_model)
print(f"Result of the Grover's algorithm with {num_iterations} iterations:")
sample_result.dump()
print(f"Most probable measured state = {sample_result.most_probable()}")
