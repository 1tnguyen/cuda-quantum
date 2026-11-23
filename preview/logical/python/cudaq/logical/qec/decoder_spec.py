# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Target-independent bindings for a whole-shot packed-record decoder."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecoderSpec:
    """Bind a decoder session to three external device-call symbols.

    All arguments and results are unsigned 64-bit words. Both interfaces
    reset(session) at shot entry and submit up to 64 records per batch,
    least-significant bit first.

    With correction, submit(session, count, bits, tag) appends raw bits and
    correction(session, 1, 0) supplies the flip for the gadget's local parity.

    With logical_readout, submit(session, count, bits, tag, mapping) tags the
    last batch of the final measurement with mapping 1 (FINAL), all others
    with 0 (BULK). Each unconditional terminal gadget owns one observable in
    composed EDEM order. At first use, logical_readout(session, index, 1, 0)
    returns that observable's corrected measurement. Its raw measurement map
    in the decoder must match the gadget's parity_indices, before inversion.
    The configured graph must have released the observable by that first use.

    The application supplies callback implementations and decoder configuration.
    Logical does not select a decoder algorithm or an execution backend.
    """

    name: str
    reset: str
    submit: str
    correction: str | None = None
    session_id: int = 0
    num_observables: int = 1
    logical_readout: str | None = None

    def __post_init__(self):
        if (isinstance(self.num_observables, bool) or
                not isinstance(self.num_observables, int) or
                not 1 <= self.num_observables < 2**63):
            raise ValueError(
                "DecoderSpec.num_observables must be a positive int64")
        if (self.correction is None) == (self.logical_readout is None):
            raise ValueError(
                "specify exactly one of correction and logical_readout")
        if self.correction is not None and self.num_observables != 1:
            raise ValueError("multiple observables require logical_readout")
        getter = "correction" if self.logical_readout is None else "logical_readout"
        for field in ("name", "reset", "submit", getter):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"DecoderSpec.{field} must be a nonempty string")
        if len({self.reset, self.submit, getattr(self, getter)}) != 3:
            raise ValueError("decoder callbacks require distinct symbols")
        if (isinstance(self.session_id, bool) or
                not isinstance(self.session_id, int) or
                not 0 <= self.session_id < 2**63):
            raise ValueError("decoder session_id must be a nonnegative int64")


__all__ = ["DecoderSpec"]
