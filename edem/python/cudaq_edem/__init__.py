# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared EDEM data structures and operations from cudaq-edem.

Early development: the API, ABI, and packed format are unstable and subject
to rapid, potentially incompatible changes during v0.x development.

The packed representation and C API allow compilers and runtime components
to exchange and evaluate EDEMs.
"""

from pathlib import Path

from . import _core
from ._core import (
    ABI_VERSION,
    FORMAT_VERSION,
    CodePresentation,
    DEM,
    EDEM,
    Endpoint,
    EndpointDirection,
    FaultDistribution,
    FaultGroup,
    GroupKind,
    MatrixBlock,
    PortKind,
    SparseMatrix,
    compose_chain,
    coalesce_dem_columns,
    factorize_dem,
    project_DEM,
)

__version__ = "@CUDAQ_EDEM_VERSION@"


def tensor_product(occurrences):
    """Combine independent EDEMs in parallel, in occurrence order.

    Code boundaries and free ports are concatenated, retaining each model's
    internal response matrices and fault laws. There are no cross-model fault
    correlations or connections. Repeated models receive distinct fault ports.
    The caller wires these ordered boundaries to the next serial layer.
    Code presentation names identify types and can occur on multiple wires.
    """
    occurrences = tuple(occurrences)
    if not occurrences or any(
            not isinstance(value, EDEM) for value in occurrences):
        raise TypeError("tensor product requires one or more EDEM values")
    if len(occurrences) == 1:
        return occurrences[0]
    codes, names = [], {}
    inputs, outputs = [], []
    faults, detectors, observables, successes = [], [], [], []
    blocks = []
    boundary_kinds = (PortKind.VIRTUAL, PortKind.SYNDROME, PortKind.LOGICAL)
    for value in occurrences:
        code_ids = []
        for code in value.codes:
            signature = (code.syndrome_count, code.virtual_outcome_count,
                         code.logical_qubit_count)
            if code.name in names:
                index, previous = names[code.name]
                if signature != previous:
                    raise ValueError("conflicting code presentations: " +
                                     code.name)
            else:
                index = len(codes)
                names[code.name] = index, signature
                codes.append(code)
            code_ids.append(index)

        def endpoint(port, incoming):
            if port.kind in boundary_kinds:
                offset = len(inputs) if incoming else len(outputs)
            elif incoming and port.kind == PortKind.FAULT:
                offset = len(faults)
            elif not incoming and port.kind == PortKind.DETECTOR:
                offset = len(detectors)
            elif not incoming and port.kind == PortKind.OBSERVABLE:
                offset = len(observables)
            elif not incoming and port.kind == PortKind.SUCCESS:
                offset = len(successes)
            else:
                raise ValueError("unsupported EDEM endpoint direction or kind")
            return Endpoint(port.kind, offset + port.index)

        blocks.extend(
            MatrixBlock(endpoint(block.output, False),
                        endpoint(block.input, True), block.matrix)
            for block in value.blocks)
        inputs.extend(code_ids[index] for index in value.inputs)
        outputs.extend(code_ids[index] for index in value.outputs)
        faults.extend(value.faults)
        detectors.extend(value.detectors)
        observables.extend(value.observables)
        successes.extend(value.successes)
    return EDEM(codes=codes,
                inputs=inputs,
                outputs=outputs,
                faults=faults,
                detectors=detectors,
                observables=observables,
                successes=successes,
                blocks=blocks)


def get_cmake_dir() -> str:
    """Return the installed cudaq-edem CMake package directory."""
    return str((Path(__file__).resolve().parent /
                "@CUDAQ_EDEM_PYTHON_CMAKE_DIR@").resolve())


__all__ = [
    "ABI_VERSION",
    "FORMAT_VERSION",
    "CodePresentation",
    "DEM",
    "EDEM",
    "Endpoint",
    "EndpointDirection",
    "FaultDistribution",
    "FaultGroup",
    "GroupKind",
    "MatrixBlock",
    "PortKind",
    "SparseMatrix",
    "compose_chain",
    "tensor_product",
    "coalesce_dem_columns",
    "factorize_dem",
    "get_cmake_dir",
    "project_DEM",
]
