# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Sequence, Mapping, List, Optional

from .cudm_helpers import CuDensityMatOpConversion, constructLiouvillian
from ..runtime.observe import observe
from .schedule import Schedule
from .expressions import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .cudm_helpers import cudm, CudmStateType
from .cudm_state import CuDensityMatState, as_cudm_state
from .helpers import InitialState, InitialStateArgT
from .integrator import BaseIntegrator
from .integrators.builtin_integrators import RungeKuttaIntegrator, cuDensityMatTimeStepper
import cupy
import math
from ..util.timing_helper import ScopeTimer


# Master-equation solver using `CuDensityMatState`
def evolve_dynamics(
        hamiltonian: Operator,
        dimensions: Mapping[int, int],
        schedule: Schedule,
        initial_state: InitialStateArgT | Sequence[InitialStateArgT],
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results=False,
        integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult:
    if cudm is None:
        raise ImportError(
            "[dynamics target] Failed to import cuquantum density module. Please check your installation."
        )

    # Reset the schedule
    schedule.reset()
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))

    # Check that the integrator can support distributed state if this is a distributed simulation.
    if cudaq_runtime.mpi.is_initialized() and cudaq_runtime.mpi.num_ranks(
    ) > 1 and integrator is not None and not integrator.support_distributed_state(
    ):
        raise ValueError(
            f"Integrator {type(integrator).__name__} does not support distributed state."
        )

    has_collapse_operators = len(collapse_operators) > 0
    batch_size = 1
    if isinstance(initial_state, InitialState):
        initial_state = CuDensityMatState.create_initial_state(
            initial_state, hilbert_space_dims, has_collapse_operators)
    elif isinstance(initial_state, Sequence):
        batch_size = len(initial_state)
        initial_state = CuDensityMatState.create_batched_initial_state(
            initial_state, hilbert_space_dims, has_collapse_operators)
    else:
        with ScopeTimer("evolve.as_cudm_state") as timer:
            initial_state = as_cudm_state(initial_state)

    if not isinstance(initial_state, CuDensityMatState):
        raise ValueError("Unknown type")

    if not initial_state.is_initialized():
        with ScopeTimer("evolve.init_state") as timer:
            initial_state.init_state(hilbert_space_dims)

    is_density_matrix = initial_state.is_density_matrix()
    me_solve = False
    if not is_density_matrix:
        if len(collapse_operators) == 0:
            me_solve = False
        else:
            with ScopeTimer("evolve.initial_state.to_dm") as timer:
                initial_state = initial_state.to_dm()
            me_solve = True
    else:
        # Always solve the master equation if the input is a density matrix
        me_solve = True

    with ScopeTimer("evolve.hamiltonian._evaluate") as timer:
        ham_term = hamiltonian._evaluate(
            CuDensityMatOpConversion(dimensions, schedule))
    linblad_terms = []
    for c_op in collapse_operators:
        with ScopeTimer("evolve.collapse_operators._evaluate") as timer:
            linblad_terms.append(
                c_op._evaluate(CuDensityMatOpConversion(dimensions, schedule)))

    with ScopeTimer("evolve.constructLiouvillian") as timer:
        liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                           linblad_terms, me_solve)

    initial_state = initial_state.get_impl()
    cudm_ctx = initial_state._ctx
    stepper = cuDensityMatTimeStepper(liouvillian, cudm_ctx)
    if integrator is None:
        integrator = RungeKuttaIntegrator(stepper)
    else:
        integrator.set_system(dimensions, schedule, hamiltonian,
                              collapse_operators)
    expectation_op = [
        cudm.Operator(
            hilbert_space_dims,
            (observable._evaluate(CuDensityMatOpConversion(dimensions)), 1.0))
        for observable in observables
    ]
    integrator.set_state(initial_state, schedule._steps[0])
    exp_vals = [ [] for _ in range(batch_size) ]
    intermediate_states = [ [] for _ in range(batch_size) ]
    for step_idx, parameters in enumerate(schedule):
        if step_idx > 0:
            with ScopeTimer("evolve.integrator.integrate") as timer:
                integrator.integrate(schedule.current_step)
        # If we store intermediate values, compute them for each step.
        # Otherwise, just for the last step.
        if store_intermediate_results or step_idx == (len(schedule) - 1):
            step_exp_vals = [[] for _ in range(batch_size)]
            for obs_idx, obs in enumerate(expectation_op):
                _, state = integrator.get_state()
                with ScopeTimer("evolve.prepare_expectation") as timer:
                    obs.prepare_expectation(cudm_ctx, state)
                with ScopeTimer("evolve.compute_expectation") as timer:
                    exp_val = obs.compute_expectation(schedule.current_step, (),
                                                      state)
                for batch_id in range(batch_size):
                    step_exp_vals[batch_id].append(float(cupy.real(exp_val[batch_id])))
            for batch_id in range(batch_size):
                exp_vals[batch_id].append(step_exp_vals[batch_id])
        if store_intermediate_results:
            _, state = integrator.get_state()
            state_length = (int)(state.storage.size / batch_size)
            print("Shape:", state.storage.shape)
            if batch_size == 1:
                if is_density_matrix:
                    dimension = int(math.sqrt(state_length))
                    with ScopeTimer("evolve.intermediate_states.append") as timer:
                        intermediate_states[0].append(
                            cudaq_runtime.State.from_data(
                                state.storage.reshape((dimension, dimension))))
                else:
                    dimension = state_length
                    with ScopeTimer("evolve.intermediate_states.append") as timer:
                        intermediate_states[0].append(
                            cudaq_runtime.State.from_data(
                                state.storage.reshape((dimension,))))
            else:
                single_state = cupy.zeros((state_length,), dtype="complex128", order="F")
                size_bytes = single_state.nbytes
                for batch_id in range(batch_size):
                    cupy.cuda.runtime.memcpy(single_state.data.ptr, state.storage.data.ptr + size_bytes * batch_id, size_bytes, cupy.cuda.runtime.memcpyDeviceToDevice)
                    if is_density_matrix:
                        dimension = int(math.sqrt(state_length))
                        with ScopeTimer("evolve.intermediate_states.append") as timer:
                            intermediate_states[batch_id].append(
                                cudaq_runtime.State.from_data(
                                    single_state.reshape((dimension, dimension))))
                    else:
                        dimension = state_length
                        with ScopeTimer("evolve.intermediate_states.append") as timer:
                            intermediate_states[batch_id].append(
                                cudaq_runtime.State.from_data(
                                    single_state.reshape((dimension,))))

    if store_intermediate_results:
        if batch_size == 1:
            return cudaq_runtime.EvolveResult(intermediate_states[0], exp_vals[0])
        else:
            return [cudaq_runtime.EvolveResult(intermediate_state, exp_val_result) for exp_val_result, intermediate_state in zip(exp_vals, intermediate_states)]
    else:
        _, state = integrator.get_state()
        
        if batch_size == 1:
            state_length = state.storage.size

            if is_density_matrix:
                dimension = int(math.sqrt(state_length))
                with ScopeTimer("evolve.final_state") as timer:
                    final_state = cudaq_runtime.State.from_data(
                        state.storage.reshape((dimension, dimension)))
            else:
                dimension = state_length
                with ScopeTimer("evolve.final_state") as timer:
                    final_state = cudaq_runtime.State.from_data(
                        state.storage.reshape((dimension,)))
                return cudaq_runtime.EvolveResult(final_state, exp_vals[0][-1])
        else:
            state_length = state.storage.size / batch_size
            single_state = cupy.zeros((state_length,), dtype="complex128", order="F")
            size_bytes = single_state.nbytes
            final_states = []
            for batch_id in range(batch_size):
                cupy.cuda.runtime.memcpy(single_state.data.ptr, state.storage.ctypes.data + size_bytes * batch_id, size_bytes, cupy.cuda.runtime.memcpyDeviceToDevice)
                if is_density_matrix:
                    dimension = int(math.sqrt(state_length))
                    with ScopeTimer("evolve.intermediate_states.append") as timer:
                        final_states.append(
                            cudaq_runtime.State.from_data(
                                single_state.reshape((dimension, dimension))))
                else:
                    dimension = state_length
                    with ScopeTimer("evolve.intermediate_states.append") as timer:
                        final_states.append(
                            cudaq_runtime.State.from_data(
                                single_state.reshape((dimension,))))
            return [cudaq_runtime.EvolveResult(final_state, exp_val_result[-1]) for final_state, exp_val_result in zip(final_state, exp_vals)]
