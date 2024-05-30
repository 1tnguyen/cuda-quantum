// Compile: g++ /home/cuda-quantum/test_mps.cpp
// -I/usr/local/cuda-11.8/targets/x86_64-linux/include
// -I/opt/nvidia/cuquantum/include -I/home/cuda-quantum/runtime -lcutensornet
// -L/opt/nvidia/cuquantum/lib -lcudart
// -L/usr/local/cuda-11.8/targets/x86_64-linux/lib

#include <cassert>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include <cuda_runtime.h>
#include <cutensornet.h>
// Returns a list of pairs of qubits that should have a Rxx acting between them.
// nq (int): Number of qubits/features.
// nn (int): Number of nearest neighbors for linear entanglement map.
std::vector<int> entanglement_graph(int nq, int nn) {
  std::vector<int> map;
  // For all distances from 1 to nn
  for (int d = 1; d < nn + 1; ++d) {
    std::set<int> busy; // Collect the right qubits of pairs on the first layer
                        // for this distance
    // Apply each gate between qubit i and its i+d (if it fits). Do so in two
    // layers.
    for (int i = 0; i < nq; ++i) {
      if (busy.find(i) == busy.end() && i + d < nq) {
        // All of these gates can be applied in one layer
        map.emplace_back(i);
        map.emplace_back(i + d);
        busy.emplace(i + d);
      }
    }
    // Apply the other half of the gates on distance d; those whose left qubit
    // is in `busy`
    for (const auto i : busy) {
      if ((i + d) < nq) {
        map.emplace_back(i);
        map.emplace_back(i + d);
      }
    }
  }
  return map;
}

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  };

#define HANDLE_CUTN_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSORNET_STATUS_SUCCESS) {                                   \
      printf("cuTensorNet error %s in line %d\n",                              \
             cutensornetGetErrorString(err), __LINE__);                        \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  };
static constexpr std::complex<double> im = std::complex<double>(0, 1.);
static constexpr std::size_t fp64size = sizeof(double);
void *allocateRz(double angle) {
  const std::vector<std::complex<double>> mat{std::exp(-im * angle / 2.0), 0, 0,
                                              std::exp(im * angle / 2.0)};
  void *d_gate = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, mat.data(), 4 * (2 * fp64size),
                               cudaMemcpyHostToDevice));
  return d_gate;
}

void *allocateRx(double angle) {
  const std::vector<std::complex<double>> mat{
      {std::cos(angle / 2.0), 0.},
      {0., -1.0 * std::sin(angle / 2.0)},
      {0, -1.0 * std::sin(angle / 2.0)},
      {std::cos(angle / 2.0), 0.}};
  void *d_gate = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, mat.data(), 4 * (2 * fp64size),
                               cudaMemcpyHostToDevice));
  return d_gate;
}

void *allocateXXPhase(double alpha) {
  const std::vector<std::complex<double>> mat{
      {std::cos(M_PI_2 * alpha), 0.}, 0., 0., { 0., -std::sin(M_PI_2 * alpha)},
      0., {std::cos(M_PI_2 * alpha), 0.}, { 0., -std::sin(M_PI_2 * alpha)}, 0.,
      0., { 0., -std::sin(M_PI_2 * alpha)}, {std::cos(M_PI_2 * alpha), 0.}, 0.,
      {0., -std::sin(M_PI_2 * alpha)}, 0., 0., {std::cos(M_PI_2 * alpha), 0.}};
  void *d_gate = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, mat.size() * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, mat.data(),  mat.size() * (2 * fp64size),
                               cudaMemcpyHostToDevice));
  return d_gate;
}

int main() {
  constexpr bool verbose = false;
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  const std::vector<double> feature_values = {0.9839110785506365,
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
                                              0.0};

  const int nearest_neighbors = 12;
  const int num_qubits =
      50; // #should equal the number of features to be encoded
  const int num_features = num_qubits;
  const auto entanglement_map =
      entanglement_graph(num_features, nearest_neighbors);

  const std::vector<int64_t> qubitDims(num_qubits, 2); // qubit size
  std::cout << "Quantum circuit: " << num_qubits << " qubits\n";
  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";
  // Define necessary quantum gate tensors in Host memory
  const double invsq2 = 1.0 / std::sqrt(2.0);
  //  Hadamard gate
  const std::vector<std::complex<double>> h_gateH{
      {invsq2, 0.0}, {invsq2, 0.0}, {invsq2, 0.0}, {-invsq2, 0.0}};
  //  CX gate
  const std::vector<std::complex<double>> h_gateCX{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
  void *d_gateH{nullptr}, *d_gateCX{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size),
                               cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size),
                               cudaMemcpyHostToDevice));

  std::vector<void *> gateMatrixToCleanUp{d_gateH, d_gateCX};
  /// BOND DIMENSION settings
  const int64_t maxExtent = 128;
  std::vector<std::vector<int64_t>> extents;
  std::vector<int64_t *> extentsPtr(num_qubits);
  std::vector<void *> d_mpsTensors(num_qubits, nullptr);
  for (int32_t i = 0; i < num_qubits; i++) {
    if (i == 0) { // left boundary MPS tensor
      extents.push_back({2, maxExtent});
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    } else if (i == num_qubits - 1) { // right boundary MPS tensor
      extents.push_back({maxExtent, 2});
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    } else { // middle MPS tensors
      extents.push_back({maxExtent, 2, maxExtent});
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mpsTensors[i],
                                   2 * maxExtent * maxExtent * 2 * fp64size));
    }
    extentsPtr[i] = extents[i].data();
  }
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize =
      (freeSize - (freeSize % 4096)) /
      2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize
            << " bytes of scratch memory on GPU: "
            << "[" << d_scratch << ":"
            << (void *)(((char *)(d_scratch)) + scratchSize) << ")\n";
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      cutnHandle, CUTENSORNET_STATE_PURITY_PURE, num_qubits, qubitDims.data(),
      CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Construct the quantum circuit state (apply quantum gates)
  int64_t id;

  // h(qubits)
  for (int qId = 0; qId < num_qubits; ++qId) {
    if (verbose)
      std::cout << "Apply H @ q" << qId << "\n";
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
        cutnHandle, quantumState, 1, std::vector<int32_t>{{qId}}.data(),
        d_gateH, nullptr, 1, 0, 1, &id));
  }

  const int reps = 2;
  const double gamma = 1.0;
  for (int iReps = 0; iReps < reps; ++iReps) {
    for (int qId = 0; qId < num_qubits; ++qId) {
      const double exponent = (2.0 / M_PI) * gamma * feature_values[qId];
      void *d_gateRz = allocateRz(M_PI * exponent);
      gateMatrixToCleanUp.emplace_back(d_gateRz);
      if (verbose)
        std::cout << "Apply Rz(" << M_PI * exponent << ")  @ q" << qId << "\n";
      HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
          cutnHandle, quantumState, 1, std::vector<int32_t>{{qId}}.data(),
          d_gateRz, nullptr, 1, 0, 1, &id));
    }

    for (std::size_t i = 0; i < entanglement_map.size(); i += 2) {
      const auto q0 = entanglement_map[i];
      const auto q1 = entanglement_map[i + 1];
      // XXPhase(exponent, q0, q1)
      const double exponent =
          gamma * gamma * (1 - feature_values[q0]) * (1 - feature_values[q1]);
      void *d_gateXXPhase = allocateXXPhase(exponent);
      gateMatrixToCleanUp.emplace_back(d_gateXXPhase);
      if (verbose)
        std::cout << "Apply XXPhase(" << exponent << ")  @ q" << q0 << ", q"
                  << q1 << "\n";
      HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
          cutnHandle, quantumState, 2, std::vector<int32_t>{{q0, q1}}.data(),
          d_gateXXPhase, nullptr, 1, 0, 1, &id));
    }
  }

  std::cout << "Applied quantum gates\n";

  // Specify the final target MPS representation (use default fortran strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
      cutnHandle, quantumState, CUTENSORNET_BOUNDARY_CONDITION_OPEN,
      extentsPtr.data(), /*strides=*/nullptr));
  cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
  constexpr double absCutoff = 1e-16;
  constexpr double relCutoff = 1e-5;
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      cutnHandle, quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, &algo,
      sizeof(algo)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      cutnHandle, quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF,
      &absCutoff, sizeof(absCutoff)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      cutnHandle, quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF,
      &relCutoff, sizeof(relCutoff)));
  std::cout << "Configured the MPS computation\n";
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  HANDLE_CUTN_ERROR(cutensornetStatePrepare(cutnHandle, quantumState,
                                            scratchSize, workDesc, 0x0));
  std::cout << "Prepared the computation of the quantum circuit state\n";
  double flops{0.0};
  HANDLE_CUTN_ERROR(cutensornetStateGetInfo(cutnHandle, quantumState,
                                            CUTENSORNET_STATE_INFO_FLOPS,
                                            &flops, sizeof(flops)));
  if (flops > 0.0) {
    std::cout << "Total flop count = " << (flops / 1e9) << " GFlop\n";
  } else if (flops < 0.0) {
    std::cout << "ERROR: Negative Flop count!\n";
    std::abort();
  }

  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  std::cout << "Scratch GPU workspace size (bytes) for MPS computation = "
            << worksize << std::endl;
  if (worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  } else {
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer for MPS computation\n";

  // Execute MPS computation
  auto t1 = high_resolution_clock::now();
  HANDLE_CUTN_ERROR(cutensornetStateCompute(
      cutnHandle, quantumState, workDesc, extentsPtr.data(),
      /*strides=*/nullptr, d_mpsTensors.data(), 0));
  auto t2 = high_resolution_clock::now();

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  std::cout << "***************************************************************"
               "**********\n";
  std::cout << "cutensornetStateCompute elapsed time: " << ms_int.count()
            << "ms\n";
  std::cout << "***************************************************************"
               "**********\n";
  std::cout << "Computed MPS factorization\n";
  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  for (int32_t i = 0; i < num_qubits; i++) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpsTensors[i]));
  }
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));

  for (void *ptr : gateMatrixToCleanUp)
    HANDLE_CUDA_ERROR(cudaFree(ptr));
  std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";
  return 0;
}