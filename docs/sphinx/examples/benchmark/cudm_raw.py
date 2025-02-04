"""Heisenberg model in cuSuperOp."""

import os
import sys
import numpy as np
import cupy as cp
import qutip as qt
import cuquantum.densitymat as cuso
from cuquantum.densitymat import (
    Operator,
    OperatorTerm,
    DenseOperator,
    tensor_product,
    DenseMixedState,
    WorkStream
)
from mpi4py import MPI
from functools import reduce
from operator import add
from typing import Union, Sequence, Callable
from numbers import Number
import argparse

class GPUTimer:

    def __init__(self, unit: str = 'ms') -> None:
        import cupy as cp
        assert unit in ['ms', 's']

        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()
        self.unit = unit

    def __enter__(self) -> "GPUTimer":
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        self.end_event.record()
        self.end_event.synchronize()

    @property
    def duration(self) -> float:
        """Return the duration in milliseconds."""
        import cupy as cp
        duration_ms = cp.cuda.get_elapsed_time(self.start_event, self.end_event)
        if self.unit == 'ms':
            return duration_ms
        elif self.unit == 's':
            return duration_ms / 1e3

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=15)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--discards", type=int, default=10)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--J_x", type=float, default=2.5)
parser.add_argument("--J_y", type=float, default=1.75)
parser.add_argument("--J_z", type=float, default=3.25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dtype", type=str, default="complex128")
params = parser.parse_args()

device_id = 7

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
cp.cuda.Device(device_id).use()

num_qubits = params.L
dims = (2,)*num_qubits
if rank==0:
    print(f"Running density-matrix Heisenberg example for {num_qubits} spins on {world_size} GPUs.")
    print("Parameters:")
    print(params)
# Elementary operators
xx = np.tensordot(qt.sigmax().full(),qt.sigmax().full(), axes=0).transpose(0,2,1,3)
yy = np.tensordot(qt.sigmay().full(),qt.sigmay().full(), axes=0).transpose(0,2,1,3)
zz = np.tensordot(qt.sigmaz().full(),qt.sigmaz().full(), axes=0).transpose(0,2,1,3)
h = params.J_x * xx + params.J_y * yy + params.J_z * zz
sigma_m = qt.sigmam().full()
one_sided_term_first_site = np.tensordot(sigma_m.T.conjugate()@sigma_m, np.eye(2),axes=0).transpose(0,2,1,3)
one_sided_term_second_site = np.tensordot(np.eye(2),sigma_m.T.conjugate()@sigma_m,axes=0).transpose(0,2,1,3)
dis_left_boundary = DenseOperator(-0.5*params.gamma*(one_sided_term_first_site+0.5*one_sided_term_second_site))
dis_right_boundary = DenseOperator(-0.5*params.gamma*(0.5*one_sided_term_first_site+one_sided_term_second_site))
dis_bulk = DenseOperator(-0.5*params.gamma*(0.5*one_sided_term_first_site+0.5*one_sided_term_second_site))
sigma_m_ = DenseOperator(sigma_m)
two_sided_dissipators = reduce(add,(tensor_product((sigma_m_, (i,),(False,)), (sigma_m_.dag(), (i,),(True,)), coeff= params.gamma) for i in range(num_qubits)))
# Hamiltonian
hket=DenseOperator(-1j*h)
hbra=DenseOperator(1j*h)
left_boundary_ket = hket + dis_left_boundary
right_boundary_ket = hket + dis_right_boundary
bulk_ket = hket + dis_bulk
left_boundary_bra = hbra + dis_left_boundary
right_boundary_bra = hbra + dis_right_boundary
bulk_bra = hbra + dis_bulk
terms_acting_on_ket = OperatorTerm(dtype=params.dtype)
terms_acting_on_ket+=tensor_product((left_boundary_ket, (0,1)))
terms_acting_on_ket+=tensor_product((right_boundary_ket, ( num_qubits - 2, num_qubits - 1)))
terms_acting_on_ket+=reduce(add,(tensor_product((bulk_ket, (i,i+1))) for i in range(1, num_qubits-2)))
terms_acting_on_bra = OperatorTerm(dtype=params.dtype)
terms_acting_on_bra+=tensor_product((left_boundary_bra, (0,1), (True,True)))
terms_acting_on_bra+=tensor_product((right_boundary_bra, ( num_qubits - 2, num_qubits - 1), (True, True)))
terms_acting_on_bra+=reduce(add,(tensor_product((bulk_bra, (i,i+1), (True, True))) for i in range(1, num_qubits-2)))

liouvillian = Operator(dims, (terms_acting_on_bra,), (terms_acting_on_ket,), (two_sided_dissipators,))

# Library context
stream = cp.cuda.Stream()
context = cuso.WorkStream(device_id=device_id)
# context = cuso.WorkStream(device_id=rank % NUM_DEVICES)
# context.set_communicator(comm, provider="MPI")

# Initial and final states

rho0 = cuso.DenseMixedState(context, dims, 1, params.dtype)
rho0.allocate_storage()
rho0.storage[:] = 1/np.prod(dims)
rho1 = rho0.clone(cp.zeros_like(rho0.storage))
liouvillian.prepare_action(context, rho0)

# Liouvillian action
runtimes = []
for i in range(params.repeats):
    rho1.inplace_scale(0.0)
    with GPUTimer() as timer:
        liouvillian.compute_action(0.0, [], rho0, rho1)

    if rank==0:
        print(f"Runtime in {i}th run is {timer.duration:.6f} ms")
        runtimes.append(timer.duration)
if rank==0:
    print(f"Average runtime in cuSuperOp is {np.average(runtimes[params.discards:]):.6f} ms")