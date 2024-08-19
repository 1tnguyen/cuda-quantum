# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .utils import __createArgumentSet
from typing import List

def run(kernel, *args, shots_count=1, noise_model=None) -> List[cudaq_runtime.RunResult]:
  """Launch an execution of a kernel  specified number of circuit executions (`shots_count`). 
Each argument in `arguments` provided can be a list or `ndarray` of arguments  
of the specified kernel argument type, and in this case, the `sample` 
functionality will be broadcasted over all argument sets and a list of 
`sample_result` instances will be returned.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For 
    example, if the kernel takes two `float` values as input, the `sample` call 
    should be structured as `cudaq.sample(kernel, firstFloat, secondFloat)`. For 
    broadcasting of the `sample` function, the arguments should be structured as a 
    `list` or `ndarray` of argument values of the specified kernel argument type.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 1000. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
    to add noise to the kernel execution on the simulator. Defaults to
    an empty noise model.

Returns:
 `list[SampleResult]`: A list of such results."""

  if noise_model != None:
    cudaq_runtime.set_noise(noise_model)

  results = []
  
  for i in range(shots_count):
    results.append(cudaq_runtime.RunResult(kernel(*args)))
          

  if noise_model != None:
    cudaq_runtime.unset_noise()
  
  return results
