# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Portable P2 metadata for QEC decoder bindings."""

from __future__ import annotations

import json
from collections.abc import Mapping

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.qec.decoder_spec import DecoderSpec


def _json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("decoder options must be JSON serializable")


def _json(value) -> str:
    return json.dumps(dict(value or {}),
                      default=_json_default,
                      separators=(",", ":"),
                      sort_keys=True)


def decoder_attributes(decoder, context):
    """Return portable attributes identifying a decoder at a QEC operation."""

    spec = decoder
    name = spec.name if isinstance(spec, DecoderSpec) else spec
    if not isinstance(name, str) or not name:
        raise TypeError("decoder= must be a DecoderSpec or name")
    attributes = {
        "decoder": mlir_ir.FlatSymbolRefAttr.get(name, context=context),
    }
    if not isinstance(spec, DecoderSpec):
        return attributes

    attributes.update({
        "decoder_implementation":
            mlir_ir.StringAttr.get(spec.implementation, context=context),
        "decoder_inputs":
            mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=context)
                    for value in spec.inputs
                ],
                context=context,
            ),
        "decoder_options_json":
            mlir_ir.StringAttr.get(_json(spec.options), context=context),
    })
    return attributes


__all__ = ["decoder_attributes"]
