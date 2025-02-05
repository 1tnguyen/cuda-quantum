import numpy as np
import time
import argparse
import qutip

# Adapt from https://github.com/qutip/qutip-benchmark/blob/master/qutip_benchmark/benchmarks/bench_solvers.py

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=10)
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

# initial state
state_list = [qutip.basis(2, 0)] * (N)
psi0 = qutip.tensor(state_list)

# Setup operators for individual qubits
sx_list, sy_list, sz_list = [], [], []
for i in range(N):
    op_list = [qutip.qeye(2)] * N
    op_list[i] = qutip.sigmax()
    sx_list.append(qutip.tensor(op_list))
    op_list[i] = qutip.sigmay()
    sy_list.append(qutip.tensor(op_list))
    op_list[i] = qutip.sigmaz()
    sz_list.append(qutip.tensor(op_list))


# Heisenberg model spin coupling strength
Jx = params.J_x 
Jy = params.J_y
Jz = params.J_y
h = params.h
gamma = params.gamma
# Construct the Hamiltonian
H = h * sz_list[0] 
for i in range(N - 1):
    H += h * sz_list[i + 1]

for i in range(N - 1):
    H += Jx * sx_list[i] * sx_list[i + 1]
    H += Jy * sy_list[i] * sy_list[i + 1]
    H += Jz * sz_list[i] * sz_list[i + 1]

# collapse operators
c_ops = [gamma * sz_list[i] for i in range(N)]

# Time schedule
t_final = 0.1
n_steps = 10
times = np.linspace(0., t_final, n_steps)

integrator = qutip.solver.integrator.IntegratorVern7
integrator.method = "euler" #"rk4" # "euler" 

# Run 
start = time.time()
result = qutip.mesolve(H, psi0, times, c_ops, e_ops = [], options = {"method": "vern7"})
end = time.time()
print(f"N = {N}, elapsed time: {end -start}")
