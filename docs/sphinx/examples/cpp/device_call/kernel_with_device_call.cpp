#include "cudaq.h"
#include <iostream>


__qpu__ int random_bit(int a, int b) {
  cudaq::qubit q;
  h(q);
  int result = cudaq::device_call<int>(/*device_id*/ 0, "add_op", a, b);
  return mz(q) + result;
}

int main() {
  int a = 2;
  int b = 3;
  auto results = cudaq::run(100, random_bit, a, b);
  for (const auto &res : results) {
    std::cout << "Result: " << res << "\n";
  }
  return 0;
}
