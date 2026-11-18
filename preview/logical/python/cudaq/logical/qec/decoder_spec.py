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

    All arguments and the correction result use unsigned 64-bit words:
    ``reset(session)``, ``submit(session, count, packed_bits, tag)``, and
    ``correction(session, count, reset) -> packed_bits``. Submission packs
    consecutive measurement bits least-significant bit first, in batches of
    at most 64. Whole-shot decoding uses tag zero and requests one observable
    correction without resetting the session. Reset runs once at shot entry.

    The application supplies callback implementations and decoder configuration.
    Logical does not select a decoder algorithm or an execution backend.
    """

    name: str
    reset: str
    submit: str
    correction: str
    session_id: int = 0

    def __post_init__(self):
        for field in ("name", "reset", "submit", "correction"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"DecoderSpec.{field} must be a nonempty string")
        if len({self.reset, self.submit, self.correction}) != 3:
            raise ValueError("decoder callbacks require distinct symbols")
        if (isinstance(self.session_id, bool) or
                not isinstance(self.session_id, int) or
                not 0 <= self.session_id < 2**63):
            raise ValueError("decoder session_id must be a nonnegative int64")


__all__ = ["DecoderSpec"]
