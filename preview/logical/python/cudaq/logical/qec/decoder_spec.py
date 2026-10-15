# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Immutable decoder binding for explicitly authored QEC operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cudaq.logical._core.immutable import freeze_mapping


@dataclass(frozen=True, slots=True)
class DecoderSpec:
    """Name and execution metadata for QEC decoder operations.

    Syndrome inputs follow measurement order. Configure the decoder to consume
    that layout during setup; submission does not permute the measured bits.

    This declaration does not instantiate a decoder or generate its model.
    """

    name: str
    implementation: str = "mwpm"
    inputs: tuple[str, ...] = ("detectors",)
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("DecoderSpec.name must be a nonempty string")
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError(
                "DecoderSpec.implementation must be a nonempty string")
        if isinstance(self.inputs, str):
            raise TypeError(
                "DecoderSpec.inputs must be a sequence of input names")
        inputs = tuple(self.inputs)
        if not inputs or any(
                not isinstance(value, str) or not value for value in inputs):
            raise ValueError("DecoderSpec.inputs must contain nonempty strings")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "options", freeze_mapping(self.options))


__all__ = ["DecoderSpec"]
