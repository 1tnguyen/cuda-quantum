
#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cmath>
#include <map>
#include <vector>

int main() {
  // Define components of the annealing Hamiltonian
  auto H_init = 1.0 * cudaq::spin_op::x(0) + 2.0 * cudaq::spin_op::x(1) +
                3.0 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
  auto H_inter = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
                 0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
  auto H_final = 0.5 * cudaq::spin_op::z(0) + 1.5 * cudaq::spin_op::z(1) +
                 2.5 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
  // Overall evolution time schedule
  constexpr double tmax = 10.0;
  const std::vector<double> steps = cudaq::linspace(0.0, tmax, 11);
  cudaq::schedule schedule(steps, {"t"});

  // Piecewise linear field schedule: linearly decrease the contribution of
  // H_init and linearly increase the contribution of H_inter, then linearly
  // decrease H_inter and linearly increase H_final
  constexpr double mid_time = tmax / 2.0;
  auto H_init_field =
      cudaq::annealing_hamiltonian::create_piecewise_linear_field(
          schedule, {{0.0, 1.0}, {mid_time, 0.0}});
  auto H_inter_field =
      cudaq::annealing_hamiltonian::create_piecewise_linear_field(
          schedule, {{0.0, 0.0}, {mid_time, 1.0}, {tmax, 0.0}});
  auto H_final_field =
      cudaq::annealing_hamiltonian::create_piecewise_linear_field(
          schedule, {{mid_time, 0.0}, {tmax, 1.0}});

  // Annealing Hamiltonian
  cudaq::annealing_hamiltonian hamiltonian(
      {H_init, H_inter, H_final}, {H_init_field, H_inter_field, H_final_field});

  // Observables
  auto observable1 = cudaq::spin_op::z(0) + cudaq::spin_op::z(1);
  auto observable2 = cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
  const std::vector<std::complex<double>> initial_state = {0.5, 0.5, 0.5, 0.5};

  // Evolve the system
  auto result = cudaq::evolve(hamiltonian, schedule, {observable1, observable2},
                              initial_state);

  return 0;
}
