import numpy as np 
import time
import numpy as np
from sympy import Symbol
from cupy.cuda.runtime import getDeviceCount

from pytket import Circuit, OpType
from pytket import Circuit
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle, SimulationAlgorithm, Config, simulate

#pytket configurations
config = Config(chi = 128, value_of_zero = 1e-16)


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
        
    map = [item for sublist in map for item in sublist]

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

class KernelStateAnsatz:
    """Class that creates and stores a symbolic ansatz circuit and can be used to
    produce instances of the circuit U(x)|0> for given parameters.

    Attributes:
        ansatz_circ: The symbolic circuit to be used as ansatz.
        feature_symbol_list: The list of symbols in the circuit, each corresponding to a feature.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int,
        gamma: float,
        entanglement_map: list[tuple[int, int]]
    ):
        """Generate the ansatz circuit and store it. The circuit has as many symbols as qubits, which
        is also the same number of features in the data set. Multiple gates will use the same symbols;
        particularly, 1-qubit gates acting on qubit `i` all use the same symbol, and two qubit gates
        acting qubits `(i,j)` will use the symbols for feature `i` and feature `j`.

        Args:
            num_qubits: number of qubits is the number of features to be encoded.
            reps: number of times to repeat the layer of unitaries.
            gamma: hyper parameter in unitary to be optimized for overfitting.
            entanglement_map: pairs of qubits to be entangled together in the ansatz,
                for now limit entanglement only to two body terms
            hadamard_init: whether a layer of H gates should be applied to all qubits
                at the beginning of the circuit.
        """

        self.one_q_symbol_list = []
        self.two_q_symbol_list = []

        self.ansatz_circ = Circuit(num_qubits)
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_qubits)]

        for i in range(num_qubits):
                self.ansatz_circ.H(i)

        for _ in range(reps):
            
            for i in range(num_qubits):
                
                exponent = (2/np.pi)*gamma*self.feature_symbol_list[i]
                self.ansatz_circ.Rz(exponent, i)
                
            
                
            for i in range(0, len(entanglement_map), 2): 
                    
                q0 = entanglement_map[i]
                q1 = entanglement_map[i+1]
                
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                
                exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)
        
           

        # Apply routing by adding SWAPs eagerly just before the XXPhase gates
        qubit_pos = {q: p for p, q in enumerate(self.ansatz_circ.qubits)}
                
        routed_circ = Circuit(self.ansatz_circ.n_qubits)  # The new circuit

        for cmd in self.ansatz_circ.get_commands():
            # Add it directly to the circuit if it's not an Rxx (aka XXPhase) gate
            if cmd.op.type != OpType.XXPhase:
                routed_circ.add_gate(cmd.op, cmd.qubits)
            # If it is Rxx, add SWAPs as necessary
            else:
                q0 = qubit_pos[cmd.qubits[0]]
                q1 = qubit_pos[cmd.qubits[1]]
                (q0, q1) = (min(q0,q1), max(q0,q1))
                # Add SWAP gates
                for q in range(q0, q1-1):
                    routed_circ.SWAP(q,q+1)
                # Apply XXPhase gate
                routed_circ.add_gate(cmd.op, [q1-1,q1])
                # Apply SWAP gates on the opposite order to return qubit to position
                for q in reversed(range(q0, q1-1)):
                    routed_circ.SWAP(q,q+1)

                self.ansatz_circ = routed_circ


    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        """Produce the circuit with its symbols being replaced by the given values.
        """
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)

        return the_circuit


for nearest_neighbors in [12]:
    
    print('nearest_neighbors', nearest_neighbors)
            
    entanglement_map = entanglement_graph(nq=num_features, nn=nearest_neighbors)

    circuit = KernelStateAnsatz(num_qubits, reps, gamma, entanglement_map)

    circuit = circuit.circuit_for_data(feature_values[:num_qubits])

    device_id = 0

    st = []
    ot = []

    for _ in range(runs): 

        with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
            
            t0 = time.time()
            mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, config)
            t1 = time.time()
            st.append(t1-t0)
            
            #Overlap calculation 
            t0 = time.time()
            overlap = mps.vdot(mps)
            t1 = time.time()
            ot.append(t1-t0)
        
                
    tket_simulation_time[nearest_neighbors] = st
    tket_overlap_time[nearest_neighbors] = ot
    print("Simulation time (qubit count -> runtime) [secs]:", tket_simulation_time)