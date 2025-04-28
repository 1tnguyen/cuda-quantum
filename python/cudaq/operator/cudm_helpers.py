# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy
from typing import Any, Mapping, List, Sequence, Union
from numbers import Number
from .expressions import ElementaryOperator, ScalarOperator
from .manipulation import OperatorArithmetics
import logging
from .schedule import Schedule
import warnings

cudm = None
CudmStateType = None
try:
    # Suppress deprecation warnings on `cuquantum` import.
    # FIXME: remove this after `cuquantum` no longer warns on import.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from cuquantum import densitymat as cudm
        from cuquantum.densitymat.callbacks import Callback as CallbackCoefficient
        from cuquantum.densitymat.callbacks import CPUCallback
    CudmStateType = Union[cudm.DensePureState, cudm.DenseMixedState]
    CudmOperator = cudm.Operator
    CudmOperatorTerm = cudm.OperatorTerm
    CudmWorkStream = cudm.WorkStream
except ImportError:
    cudm = None
    CudmOperator = Any
    CudmOperatorTerm = Any
    CudmWorkStream = Any
    CallbackCoefficient = Any

logger = logging.getLogger(__name__)
