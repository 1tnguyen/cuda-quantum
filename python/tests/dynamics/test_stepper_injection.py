# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Tests for custom stepper injection through integrator constructors and
the full cudaq.evolve() call path.
"""
import os, math, pytest
import numpy as np
import cudaq

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
has_gpu = cudaq.num_available_gpus() > 0

has_dynamics = True
try:
    from cudaq.dynamics import nvqir_dynamics_bindings as bindings
except ImportError:
    has_dynamics = False

has_scipy = True
try:
    from scipy.integrate import ode
except ImportError:
    has_scipy = False

skip_no_dynamics = pytest.mark.skipif(not has_dynamics,
                                      reason="dynamics bindings unavailable")
skip_no_scipy = pytest.mark.skipif(not has_scipy,
                                   reason="scipy unavailable")
skip_no_gpu = pytest.mark.skipif(not has_gpu, reason="GPU required")

# ---------------------------------------------------------------------------
# Imports guarded by availability
# ---------------------------------------------------------------------------
if has_dynamics and has_scipy:
    from cudaq.dynamics.integrator import BaseTimeStepper, BaseIntegrator
    from cudaq.dynamics.integrators.scipy_integrators import ScipyZvodeIntegrator
    from cudaq.dynamics.integrators.builtin_integrators import cuDensityMatTimeStepper
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
    from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator

class _Sentinel:
    """Lightweight stand-in for a real stepper object."""
    pass


@skip_no_dynamics
@skip_no_scipy
class TestScipyZvodeConstructor:

    def test_stepper_positional(self):
        sentinel = _Sentinel()
        integ = ScipyZvodeIntegrator(sentinel)
        assert integ.stepper is sentinel

    def test_stepper_keyword(self):
        sentinel = _Sentinel()
        integ = ScipyZvodeIntegrator(stepper=sentinel)
        assert integ.stepper is sentinel

    def test_no_stepper_defaults_to_none(self):
        integ = ScipyZvodeIntegrator()
        assert integ.stepper is None


@skip_no_dynamics
@skip_no_scipy
class TestSetSystemPreservesStepper:

    @pytest.fixture(autouse=True)
    def _target(self):
        cudaq.set_target("dynamics")
        yield
        cudaq.reset_target()

    def _minimal_set_system(self, integ):
        """Call set_system with the minimal valid arguments."""
        from cudaq import operators
        from cudaq.dynamics import Schedule
        dimensions = {0: 2}
        schedule = Schedule(np.linspace(0.0, 1.0, 11), ["t"])
        hamiltonian = 1.0 * operators.number(0)
        collapse_operators = []
        integ.set_system(dimensions, schedule, hamiltonian,
                         collapse_operators)

    def test_injected_stepper_survives_set_system(self):
        sentinel = _Sentinel()
        integ = ScipyZvodeIntegrator(stepper=sentinel)
        self._minimal_set_system(integ)
        assert integ.stepper is sentinel

    def test_default_none_stepper_stays_none(self):
        integ = ScipyZvodeIntegrator()
        self._minimal_set_system(integ)
        assert integ.stepper is None


# ===================================================================
# Group B — Integration test: custom stepper through cudaq.evolve()
# ===================================================================


@skip_no_gpu
@skip_no_dynamics
@skip_no_scipy
class TestCustomStepperThroughEvolve:

    @pytest.fixture(autouse=True)
    def _target(self):
        cudaq.set_target("dynamics")
        yield
        cudaq.reset_target()

    def test_custom_stepper_produces_correct_results(self):
        """
        Inject a tracking wrapper stepper into ScipyZvodeIntegrator and
        run cudaq.evolve().  Verify the custom stepper is actually called
        and produces results matching the default code path.
        """
        import cupy as cp
        from cudaq.operators.boson import annihilate, number
        from cudaq.dynamics import Schedule

        # ---- system parameters (same as TestCavityModel) ----
        N = 10
        decay_rate = 0.1
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        collapse_ops = [np.sqrt(decay_rate) * annihilate(0)]

        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)

        # ---- baseline: default stepper (no injection) ----
        baseline_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[hamiltonian],
            collapse_operators=collapse_ops,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=ScipyZvodeIntegrator())
        baseline_exp = [
            ev[0].expectation()
            for ev in baseline_result.expectation_values()
        ]

        # ---- build the same stepper the default path would build ----
        schedule2 = Schedule(steps, ["t"])
        native_schedule = bindings.Schedule(schedule2._steps,
                                            list(schedule2._parameters))

        # Pure state vector of length N → not a density matrix
        is_density = False

        # Convert to MatrixOperator (same conversion evolve_dynamics does)
        mat_hamiltonian = MatrixOperator(hamiltonian)
        mat_collapse_ops = [MatrixOperator(op) for op in collapse_ops]

        inner_stepper = cuDensityMatTimeStepper(native_schedule,
                                                mat_hamiltonian,
                                                mat_collapse_ops, [N],
                                                is_density)

        # ---- tracking wrapper ----
        class TrackingTimeStepper(BaseTimeStepper[cudaq_runtime.State]):
            def __init__(self, delegate):
                self.delegate = delegate
                self.call_count = 0

            def compute(self, state, t: float):
                self.call_count += 1
                return self.delegate.compute(state, t)

        tracker = TrackingTimeStepper(inner_stepper)

        # ---- evolve with injected stepper ----
        psi0_fresh2 = cudaq.State.from_data(psi0_)
        custom_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            Schedule(steps, ["t"]),
            psi0_fresh2,
            observables=[hamiltonian],
            collapse_operators=collapse_ops,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=ScipyZvodeIntegrator(stepper=tracker))
        custom_exp = [
            ev[0].expectation()
            for ev in custom_result.expectation_values()
        ]

        # ---- assertions ----
        assert tracker.call_count > 0, \
            "Custom stepper was never called during integration"
        np.testing.assert_allclose(baseline_exp, custom_exp, rtol=1e-5,
                                   err_msg="Custom stepper results diverge "
                                           "from default stepper")
