# %%
# !pip install sympy pandas==2.2.1 scikit-learn==1.4.1.post1 pytket==1.26.0 pytket-cutensornet==0.6.0
# !echo cuda-quantum | sudo -S apt-get install -y cuda-toolkit-11.8 && python -m pip install cupy
# !pip install cuquantum-python==23.10.0

import cudaq 
import numpy as np 
import os 
import numpy as np
import time

import numpy as np


truncation_error = 1e-16 #Truncation error for SVD for tket
loglevel = 30      #loglevel: Set to 10 for debug mode. Defaults to 30 (quiet).


cudaq.set_random_seed(2)



# `CUDAQ_MPS_MAX_BOND=X`: The maximum number of singular values to keep (fixed extent truncation). Default: 64.
# `CUDAQ_MPS_ABS_CUTOFF=X`: The cutoff for the largest singular value during truncation. Eigenvalues that are smaller will be trimmed out. Default: 1e-5.
# `CUDAQ_MPS_RELATIVE_CUTOFF=X`: The cutoff for the maximal singular value relative to the largest eigenvalue. Eigenvalues that are smaller than this fraction of the largest singular value will be trimmed out. Default: 1e-5

os.environ["CUDAQ_MPS_MAX_BOND"] = "128"
os.environ["CUDAQ_MPS_ABS_CUTOFF"] = "1e-16"
os.environ["CUDAQ_MPS_RELATIVE_CUTOFF"] = "1e-5"
cudaq.set_target('tensornet-mps')


gpuname = 'a100'

num_qubits = 50  #should equal the number of features to be encoded 
num_features = num_qubits
reps = 2 
gamma = 1

runs = 1   #average over how many runs 

cudaq_simulation_time  = {}
cudaq_overlap_time = {}

tket_simulation_time  = {}
tket_overlap_time = {}


def entanglement_graph(nq, nn):
    """
    Function to produce the edgelist/entanglement map for a circuit ansatz

    Args:
        nq (int): Number of qubits/features.
        nn (int): Number of nearest neighbors for linear entanglement map.

    Returns:
        A list of pairs of qubits that should have a Rxx acting between them.
    """
    map = []
    for d in range(1, nn+1):  # For all distances from 1 to nn
        busy = set()  # Collect the right qubits of pairs on the first layer for this distance
        # Apply each gate between qubit i and its i+d (if it fits). Do so in two layers.
        for i in range(nq):
            if i not in busy and i+d < nq:  # All of these gates can be applied in one layer
                map.append((i, i+d))
                busy.add(i+d)
        # Apply the other half of the gates on distance d; those whose left qubit is in `busy`
        for i in busy:
            if i+d < nq:
                map.append((i, i+d))

    # map = [list(t) for t in map]
    print("Map size: ", len(map))    
    map = [item for sublist in map for item in sublist]
    # print("Map after: ", map)    
    return map


feature_values = [0.9839110785506365,
 0.9512731960259875,
 0.9171572738840177,
 0.9677085778393889,
 0.0,
 0.0,
 0.0,
 0.0,
 0.9839110785506364,
 0.9512731960259875,
 0.0,
 0.9512731960259874,
 1.0322914221417925,
 1.0160889214305449,
 0.0,
 0.0,
 1.0487268039551938,
 0.9512731960259875,
 0.0,
 0.9839110785506364,
 1.048726803955194,
 1.048726803955194,
 0.8801913294999058,
 0.7112928479387994,
 0.0,
 0.7863628589166145,
 0.9999999999905906,
 0.9999999999905906,
 0.8801913294999058,
 0.7112928479387994,
 0.0,
 0.7863628589166145,
 0.9999999999905906,
 0.9999999999905906,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.9839110785506368,
 0.9839110785506368,
 0.0,
 0.0,
 0.0,
 0.0]

for nearest_neighbors in [12]:
    print('nearest_neighbours', nearest_neighbors)

    entanglement_map = entanglement_graph(nq=num_features, nn=nearest_neighbors)
    # print(entanglement_map)
    @cudaq.kernel
    def kernel(qubit_count: int, params: np.ndarray):
        
        qubits = cudaq.qvector(qubit_count)
        
        h(qubits)
        
        for _ in range(reps):
            
            for i in range(qubit_count):
                
                exponent = (2/np.pi)*gamma*params[i]
                rz(np.pi*exponent, qubits[i])
                        
                        
            for i in range(0, len(entanglement_map), 2): 
                
                # self.ansatz_circ.XXPhase(exponent, q0, q1)
                
                q0 = entanglement_map[i]
                q1 = entanglement_map[i+1]
            
                exponent = gamma * gamma * (1 - params[q0]) * (1 - params[q1])
                
                # xxphase(theta) = Rxx(theta) = exp_pauli((-i * theta / 2)*XX)
                # theta = Pi * exponent
                exp_pauli(-np.pi * exponent / 2.0, [qubits[q0], qubits[q1]], 'XX')
               
                
    kernel.compile()



    st = []
    ot = []

    for _ in range(runs): 
        #MPS simulation 
        t0 = time.time()
        state = cudaq.get_state(kernel, num_qubits, feature_values)
        t1 = time.time()
        st.append(t1-t0)
        
        #Overlap calculation 
        t0 = time.time()
        overlap = state.overlap(state)
        t1 = time.time()
        ot.append(t1-t0)
        
    cudaq_simulation_time[nearest_neighbors] = st
    cudaq_overlap_time[nearest_neighbors] = ot
    print("Simulation time:", cudaq_simulation_time)
    print("Overlap time:", cudaq_overlap_time)
   

