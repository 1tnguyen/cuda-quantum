// Compile and run with:
// ```
// nvq++ noise_depolarization.cpp --target density-matrix-cpu -o dyn.x
// && ./dyn.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>

// CUDA-Q supports several different models of noise. In this
// case, we will examine the modeling of depolarization noise. This
// depolarization will result in the qubit state decaying into a mix
// of the basis states, |0> and |1>, with a user-provided probability.

int main() {

  // We will begin by defining an empty noise model that we will add
  // our depolarization channel to.
  cudaq::noise_model noise;

  // We define a depolarization channel setting the probability of the
  // qubit state being scrambled to `0.75` (25% of probability each for X, Y, or
  // Z noise). The remaining 25% is for no noise, i.e., identity (I) operation.
  // Note: This depolarizing channel (equal probability for I, X, Y, and Z)
  // produced a so-called maximally-mixed state, which have a flat distribution
  // on all measurement bases.
  cudaq::depolarization_channel depolarization(0.75);
  // We will apply the channel to any Y-gate on qubit 0. In other words,
  // for each Y-gate on our qubit, the qubit will have a `1.0`
  // probability of decaying into a maximally-mixed state.
  noise.add_channel<cudaq::types::y>({0}, depolarization);

  // Our kernel will apply a Y-gate to qubit 0.
  // This will bring the qubit to the |1> state, where it will undergo an equal
  // mixture of I, X, Y, Z depolarizing noise.
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    y(q);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // With noise, the measurements should be a roughly 50/50
  // mix between the |0> and |1> states.
  auto noisy_counts = cudaq::sample(kernel);
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  // Without noise, the qubit should still be in the |1> state.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel);
  noiseless_counts.dump();
}
