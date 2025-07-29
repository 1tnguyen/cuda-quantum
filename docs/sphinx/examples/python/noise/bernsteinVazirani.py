
from noise_model import *
cudaq.set_target("density-matrix-cpu") # 'nvidia' for trajectory simulation on GPU 

# Parameters for the Bernstein-Vazirani algorithm
qubit_count = 10  

# Change the secret string to whatever you prefer
secret_string = np.random.randint(2, size=(qubit_count,))
print(f"secret bitstring to find = {secret_string}")

@cudaq.kernel
def oracle(register: cudaq.qview, auxiliary_qubit: cudaq.qubit,
           secret_string: list[int]):

    for index, bit in enumerate(secret_string):
        if bit == 1:
            x.ctrl(register[index], auxiliary_qubit)

@cudaq.kernel
def bernstein_vazirani(secret_string: list[int]):

    qubits = cudaq.qvector(len(secret_string))  # register of size n
    auxiliary_qubit = cudaq.qubit()  # auxiliary qubit

    # Prepare the auxillary qubit.
    x(auxiliary_qubit)
    h(auxiliary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxiliary_qubit, secret_string)

    # Apply another set of Hadamards to the register.
    h(qubits)

    mz(qubits)  # measures only the main register


result = cudaq.sample(bernstein_vazirani, secret_string, noise_model=noise_model)
print("Result of the Bernstein-Vazirani algorithm:")
result.dump()
print(f"Most probable measured state = {result.most_probable()}")
print(
    f"Were we successful? {''.join([str(i) for i in secret_string]) == result.most_probable()}"
)