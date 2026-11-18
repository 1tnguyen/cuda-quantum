# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Portable runtime bindings attached to decoder operations."""

import cudaq.mlir.ir as mlir_ir
from cudaq.logical.qec.decoder_spec import DecoderSpec


def decoder_attributes(decoder, context):
    if not isinstance(decoder, DecoderSpec):
        raise TypeError("decoder= must be a DecoderSpec")
    attributes = {
        "decoder":
            mlir_ir.FlatSymbolRefAttr.get(decoder.name, context=context),
        "decoder_session":
            mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=context),
                decoder.session_id),
    }
    for field in ("reset", "submit", "correction"):
        attributes["decoder_" + field] = mlir_ir.FlatSymbolRefAttr.get(
            getattr(decoder, field), context=context)
    return attributes


__all__ = ["decoder_attributes"]
