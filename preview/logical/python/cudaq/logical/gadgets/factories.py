# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small reusable realization factories, exported as ordinary Python values.

This module is a library, not a registry.  Calling a factory returns a normal
``GadgetDefinition`` that the importing module can expose to QLX's private
module linker alongside research-local realizations.
"""

from __future__ import annotations

import hashlib

from cudaq.logical.programs.decorators import objective
from cudaq.logical.ops._impl import (
    cx,
    cz,
    discard,
    decode_bit,
    mz,
    extract_syndrome,
    h,
    reset,
    s,
    sdg,
    x,
    y,
    z,
)
from cudaq.logical.ops._impl import measure_x as logical_measure_x
from cudaq.logical.ops._impl import measure_z as logical_measure_z
from cudaq.logical.ops._impl import (
    mpp,
    prepare,
)
from cudaq.logical.types.values import logical_qubit
from cudaq.logical.codes import (
    Code,
    Encoding,
)
from cudaq.logical.algebra.pauli import (
    PauliGroupElement,
    PauliProduct,
    X,
    Z,
)
from cudaq.logical.gadgets.definition import gadget
from cudaq.logical.gadgets.interface import patch
from cudaq.logical.types.semantic import (
    plus,
    zero,
)
from ..std import LogicalInstrumentRef
from ..std import idle as idle_objective
from ..std import prepare_plus as prepare_plus_objective
from ..std import prepare_zero as prepare_zero_objective
from ..std import x as x_objective
from ..std import z as z_objective


@objective(name="qlx_standard_measure_z")
def _measure_z_intent(q: logical_qubit) -> bool:
    return logical_measure_z(q)


@objective(name="qlx_standard_measure_x")
def _measure_x_intent(q: logical_qubit) -> bool:
    return logical_measure_x(q)


def _encoding(value):
    if isinstance(value, Encoding):
        return value
    if isinstance(value, Code):
        return value.default_encoding
    raise TypeError(
        "logical measurement realization requires a Code or Encoding")


def _measurement(
    value,
    *,
    basis: str,
    logical: int,
    preserve_block: bool,
    name: str | None,
    decoder=None,
):
    encoding = _encoding(value)
    code = encoding.code
    if not isinstance(logical, int) or isinstance(logical, bool):
        raise TypeError("logical measurement port must be an int")
    if not 0 <= logical < code.k:
        raise ValueError(f"code {code.name} has no protected logical {logical}")
    objective = _measure_z_intent if basis == "z" else _measure_x_intent
    pauli = Z if basis == "z" else X

    if decoder is not None:
        if preserve_block:
            raise ValueError(
                "decoder-backed readout destructively measures all data qubits")
        representative = (code.logical_z_basis.rows[logical] if basis == "z"
                          else code.logical_x_basis.rows[logical])
        support = representative[
            code.n:] if basis == "z" else representative[:code.n]
        other = representative[:code.
                               n] if basis == "z" else representative[code.n:]
        if any(other):
            raise ValueError(
                "decoder-backed readout requires a uniform X or Z logical representative"
            )
        indices = tuple(index for index, bit in enumerate(support) if bit)

    def realization(block):
        if decoder is None:
            block, outcome = mpp(pauli(block[logical]))
        else:
            if basis == "x":
                block = h(block.data)
            block, bits = mz(block.data, record="terminal_data")
            outcome = decode_bit(bits, decoder=decoder, parity_indices=indices)
        if preserve_block:
            return block, outcome
        discard(block)
        return outcome

    suffix = "_preserve_block" if preserve_block else ""
    if decoder is not None:
        suffix += "_decoded"
    realization.__name__ = (name or
                            f"{code.name}_measure_{basis}{logical}{suffix}")
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": tuple[patch[encoding], bool] if preserve_block else bool,
    }

    def analysis_profile(definition):
        from .profiles import GadgetProfile
        from .records import StructuredRecord
        from .semantics import Detector, LogicalObservable

        if decoder is None:
            # This realization measures one logical product, not all data.
            return GadgetProfile(definition,
                                 boundary={
                                     definition.outputs.only().syndrome:
                                         definition.inputs.only().syndrome
                                 } if preserve_block else None,
                                 observables=(LogicalObservable(
                                     definition.records.product(0)),))
        data = StructuredRecord(definition, "terminal_data",
                                data_count=code.n).data
        detectors = []
        for i, row in enumerate(encoding.profile.effective_stabilizers.rows):
            support, other = (row[code.n:],
                              row[:code.n]) if basis == "z" else (row[:code.n],
                                                                  row[code.n:])
            if not any(other):
                check = tuple(j for j, bit in enumerate(support) if bit)
                detectors.append(
                    Detector(definition.inputs.only().syndrome[i] ^
                             data.parity(check)))
        return GadgetProfile(
            definition,
            detectors=tuple(detectors),
            # Local observable zero describes this gadget's sole Boolean result.
            # The compiler binds each invocation to the experiment's results.
            observables=(LogicalObservable(data.parity(indices)),))

    return gadget(realization,
                  implements=objective,
                  name=realization.__name__,
                  analysis_profile=analysis_profile)


def measure_z(
    value,
    *,
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
    decoder=None,
):
    """Return logical-Z readout, optionally using a terminal decoder.

    With ``decoder``, measure all data qubits in their existing order and
    return ``decode_bit`` with the selected logical representative's parity.
    Without it, retain the ideal logical-product measurement realization.
    """
    return _measurement(
        value,
        basis="z",
        logical=logical,
        preserve_block=preserve_block,
        name=name,
        decoder=decoder,
    )


def measure_x(
    value,
    *,
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
    decoder=None,
):
    """Return logical-X readout, optionally using a terminal decoder.

    Decoder-backed readout rotates all data qubits to X before measurement;
    ``decoder`` must be configured for this terminal measurement basis.
    """
    return _measurement(
        value,
        basis="x",
        logical=logical,
        preserve_block=preserve_block,
        name=name,
        decoder=decoder,
    )


def logical_measure(
    value,
    *,
    basis="z",
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
    decoder=None,
):
    """Return a destructive logical measurement realization for one basis.

    ``basis`` accepts ``"x"``/``"z"`` (or any object whose ``value``/``name``
    normalizes to one of those). This is the single-call spelling of the
    :func:`measure_x`/:func:`measure_z` pair.
    """
    text = getattr(basis, "value", None) or getattr(basis, "name",
                                                    None) or basis
    text = str(text).lower()
    if text not in ("x", "z"):
        raise ValueError("logical_measure basis must be 'x' or 'z'")
    return _measurement(
        value,
        basis=text,
        logical=logical,
        preserve_block=preserve_block,
        name=name,
        decoder=decoder,
    )


def logical_pauli(
    value,
    *,
    basis: str,
    logical: int = 0,
    name: str | None = None,
):
    """Return the canonical encoded logical-X or logical-Z realization."""

    encoding = _encoding(value)
    code = encoding.code
    basis = str(basis).lower()
    if basis not in {"x", "z"}:
        raise ValueError("logical_pauli basis must be 'x' or 'z'")
    if not isinstance(logical, int) or isinstance(logical, bool):
        raise TypeError("logical Pauli port must be an int")
    if not 0 <= logical < code.k:
        raise ValueError(f"code {code.name} has no protected logical {logical}")
    representative = (code.logical_x_basis.rows[logical]
                      if basis == "x" else code.logical_z_basis.rows[logical])
    support = (representative[:code.n]
               if basis == "x" else representative[code.n:])
    indices = tuple(index for index, bit in enumerate(support) if bit)

    def realization(block):
        action = x if basis == "x" else z
        return action(block.frame[indices])

    realization.__name__ = name or f"{code.name}_logical_{basis}{logical}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(
        realization,
        implements=x_objective if basis == "x" else z_objective,
        name=realization.__name__,
        metadata={
            "logical": logical,
            "pauli": basis.upper(),
            "representative": indices,
        },
    )


def css_memory_round(value,
                     *,
                     decoder=None,
                     record: str | None = None,
                     name: str | None = None):
    """Return a code-specific logical memory round.

    One syndrome-extraction pass over the encoded block: the gadget consumes
    and returns the same linear patch owner while its check records update the
    syndrome history. It implements the standard ``cudaq.logical.logical.idle``
    objective, so QEC selection binds it to explicit ``cudaq.logical.idle`` memory
    workloads placed on the code.

    With ``decoder=`` the gadget explicitly submits each extracted round in
    measurement order. Configure the decoder to consume that order during setup.
    Omitting the decoder retains extraction-only behavior. Decoder reset and
    correction consumption are separate operations.
    """
    encoding = _encoding(value)
    code = encoding.code

    def realization(block):
        if record is None:
            block, syndrome = extract_syndrome(block)
        else:
            block, syndrome = extract_syndrome(block, record=record)
        if decoder is not None:
            from cudaq.logical.ops import submit_syndrome
            submit_syndrome(syndrome, decoder=decoder)
        return block

    realization.__name__ = name or f"{code.name}_memory_round"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }

    def analysis_profile(definition):
        from .profiles import GadgetProfile
        from .records import StructuredRecord
        from .semantics import Detector, OutputSyndromeAssignment

        measured = tuple(
            tuple(int(i in support)
                  for i in range(code.n)) + (0,) * code.n
            for support in code.hx) + tuple(
                (0,) * code.n + tuple(int(i in support)
                                      for i in range(code.n))
                for support in code.hz)
        checks = StructuredRecord(definition,
                                  record or "syndrome0",
                                  check_count=len(measured)).checks
        rows = encoding.profile.effective_stabilizers.rows
        if any(row not in measured for row in rows):
            raise ValueError(
                "CSS memory default profile requires each effective stabilizer "
                "to be a measured hx/hz check; supply an explicit profile")
        ordered = tuple(checks[measured.index(row)] for row in rows)
        return GadgetProfile(
            definition,
            detectors=tuple(
                Detector(definition.inputs.only().syndrome[i] ^ bit)
                for i, bit in enumerate(ordered)),
            output_syndromes=tuple(
                OutputSyndromeAssignment(definition.outputs.only(), i, bit)
                for i, bit in enumerate(ordered)))

    return gadget(realization,
                  implements=idle_objective,
                  name=realization.__name__,
                  analysis_profile=analysis_profile)


def _preparation(value, *, state, objective, name, preparation):
    encoding = _encoding(value)
    code = encoding.code
    if preparation not in (None, "product", "encoded"):
        raise ValueError("preparation must be 'product' or 'encoded'")

    if preparation == "encoded":
        pauli = Z if state == zero else X
        realization = _stabilizer_preparation_body(
            encoding, tuple(pauli(i) for i in range(code.k)))
    else:

        def realization(block):
            if preparation is None:
                return prepare(block, state=state)
            block = reset(block.data)
            return h(block.data) if state == plus else block

    def analysis_profile(definition):
        from .profiles import GadgetProfile
        from .semantics import OutputSyndromeAssignment

        return GadgetProfile(
            definition,
            output_syndromes=tuple(
                OutputSyndromeAssignment(definition.outputs.only(), i, False)
                for i, row in enumerate(
                    encoding.profile.effective_stabilizers.rows)
                if preparation == "encoded" or
                not any(row[:code.n] if state == zero else row[code.n:])))

    suffix = "" if preparation is None else f"_{preparation}"
    realization.__name__ = name or f"{code.name}_prepare_{state.name}{suffix}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(
        realization,
        implements=objective,
        name=realization.__name__,
        analysis_profile=analysis_profile if preparation is not None else None)


def prepare_zero(value,
                 *,
                 preparation: str | None = None,
                 name: str | None = None):
    """Return zero preparation with an explicit physical realization.

    ``preparation="product"`` resets the data carriers to zero; subsequent
    syndrome rounds establish the remaining stabilizers. ``"encoded"`` uses
    the existing stabilizer-state synthesis to prepare logical zero directly.
    Each explicit choice supplies its matching default analysis profile.
    Omitting the choice preserves legacy context-dependent lowering without
    supplying a default analysis profile.
    """
    return _preparation(value,
                        state=zero,
                        objective=prepare_zero_objective,
                        name=name,
                        preparation=preparation)


def prepare_plus(value,
                 *,
                 preparation: str | None = None,
                 name: str | None = None):
    """Return plus preparation; choices have the same meaning as prepare_zero.

    Product preparation resets the data and applies H to each carrier. Encoded
    preparation synthesizes the code stabilizers and all logical X eigenvalues.
    """
    return _preparation(value,
                        state=plus,
                        objective=prepare_plus_objective,
                        name=name,
                        preparation=preparation)


def _stim_pauli(element: PauliGroupElement, *, width: int):
    """Convert one phase-complete Pauli-group element to a Stim Pauli product."""

    try:
        import stim
    except ImportError as exc:
        raise ImportError("stabilizer-state preparation requires stim") from exc
    if element.arity > width:
        raise ValueError("logical stabilizer exceeds the encoded logical width")
    x_mask = element.x_mask
    z_mask = element.z_mask
    paulis = []
    for index in range(width):
        x_bit = bool(x_mask & (1 << index))
        z_bit = bool(z_mask & (1 << index))
        paulis.append(
            "Y" if x_bit and z_bit else "X" if x_bit else "Z" if z_bit else "I")
    # Stim's Hermitian Y is iXZ. Remove that local convention from the
    # phase-complete i^p X^x Z^z representation to recover the overall sign.
    residual_phase = (element.phase_exponent_mod_4 -
                      (x_mask & z_mask).bit_count()) % 4
    if residual_phase not in {0, 2}:
        raise ValueError("stabilizer-state generator must be Hermitian")
    result = stim.PauliString("".join(paulis))
    return -result if residual_phase == 2 else result


def _positive_row_pauli(code: Code, row) -> PauliGroupElement:
    """Interpret one phase-free symplectic row as a positive Hermitian Pauli."""

    x_mask = sum(int(bit) << index for index, bit in enumerate(row[:code.n]))
    z_mask = sum(int(bit) << index for index, bit in enumerate(row[code.n:]))
    return PauliGroupElement(
        code.n,
        x_mask,
        z_mask,
        (x_mask & z_mask).bit_count() % 4,
    )


def _lift_logical_stabilizer(code: Code, value: PauliProduct):
    """Lift one formal logical Pauli through the code's chosen representatives."""

    if not isinstance(value, PauliProduct):
        raise TypeError(
            "logical_stabilizers entries must be formal qlx Pauli products")
    logical = PauliGroupElement.from_product(value)
    if logical.arity > code.k:
        raise ValueError(
            f"logical stabilizer addresses port {logical.arity - 1}, but "
            f"code {code.name!r} has only {code.k} protected logical port(s)")
    physical = PauliGroupElement(
        code.n,
        0,
        0,
        logical.phase_exponent_mod_4,
    )

    # PauliGroupElement uses the canonical i^p X^x Z^z order. Preserve that
    # order while replacing every formal logical X/Z by the code's physical
    # representative, including the phase induced where representatives cross.
    for logical_index, row in enumerate(code.logical_x_basis.rows):
        if logical.x_mask & (1 << logical_index):
            physical = physical.multiply(_positive_row_pauli(code, row))
    for logical_index, row in enumerate(code.logical_z_basis.rows):
        if logical.z_mask & (1 << logical_index):
            physical = physical.multiply(_positive_row_pauli(code, row))
    return physical


def _stabilizer_preparation_circuit(
    encoding: Encoding,
    logical_stabilizers: tuple[PauliProduct, ...],
):
    try:
        import stim
    except ImportError as exc:
        raise ImportError("stabilizer-state preparation requires stim") from exc
    code = encoding.code
    if code.r:
        raise NotImplementedError(
            "stabilizer-state preparation currently requires a stabilizer code; "
            "subsystem-code gauge-state policy must be explicit")
    if len(logical_stabilizers) != code.k:
        raise ValueError(
            "logical_stabilizers must provide exactly one independent "
            f"generator per protected logical port ({code.k} required)")
    physical_generators = [
        _stim_pauli(
            _positive_row_pauli(code, row),
            width=code.n,
        ) for row in code.stabilizer_basis.rows
    ]
    physical_generators.extend(
        _stim_pauli(
            _lift_logical_stabilizer(code, stabilizer),
            width=code.n,
        ) for stabilizer in logical_stabilizers)
    try:
        tableau = stim.Tableau.from_stabilizers(physical_generators)
    except ValueError as exc:
        raise ValueError(
            "logical_stabilizers must be independent, mutually commuting, and "
            "compatible with the code stabilizers") from exc
    return tableau.to_circuit(method="elimination")


def _stabilizer_preparation_body(encoding, logical_stabilizers):
    circuit = _stabilizer_preparation_circuit(
        encoding,
        logical_stabilizers,
    )
    instructions = tuple((
        instruction.name,
        tuple(target.value for target in instruction.targets_copy()),
    ) for instruction in circuit)
    supported = {"H", "S", "S_DAG", "X", "Y", "Z", "CX", "CZ"}
    unsupported = tuple(instruction for instruction, _targets in instructions
                        if instruction not in supported)
    if unsupported:
        raise NotImplementedError(
            "stabilizer tableau synthesis emitted unsupported instruction "
            f"{unsupported[0]!r}")

    def realization(block):
        block = reset(block.data)
        unary = {
            "H": h,
            "S": s,
            "S_DAG": sdg,
            "X": x,
            "Y": y,
            "Z": z,
        }
        for instruction, targets in instructions:
            if instruction in unary:
                for target in targets:
                    block = unary[instruction](block.data[target])
                continue
            binary = cx if instruction == "CX" else cz
            for offset in range(0, len(targets), 2):
                block = binary(
                    block.data[targets[offset]],
                    block.data[targets[offset + 1]],
                )
        return block

    return realization


def stabilizer_preparation(
    value,
    *,
    logical_stabilizers,
    name: str | None = None,
):
    """Return an explicit encoded stabilizer-state preparation gadget.

    ``logical_stabilizers`` contains exactly ``code.k`` independent, commuting
    formal Pauli products over the code's protected logical-port indices. For
    example, ``(cudaq.logical.Y(0),)`` selects encoded ``|+Y>`` for a one-logical-qubit
    code, while ``(cudaq.logical.X(0) @ cudaq.logical.X(1), cudaq.logical.Z(0) @ cudaq.logical.Z(1))`` selects an
    encoded Bell state for a two-logical-qubit code.

    The returned ordinary :class:`~cudaq.logical.GadgetDefinition` contains the concrete
    reset and Clifford operations produced by tableau elimination. It is not a
    device-side opaque preparation instruction and does not claim a
    fault-tolerant preparation protocol by itself.
    """

    encoding = _encoding(value)
    code = encoding.code
    logical_stabilizers = tuple(logical_stabilizers)
    realization = _stabilizer_preparation_body(encoding, logical_stabilizers)

    state_key = ";".join(f"{stabilizer.sign}:" +
                         ",".join(f"{factor.pauli}{factor.operand}"
                                  for factor in stabilizer.factors)
                         for stabilizer in logical_stabilizers)
    digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()[:10]
    realization.__name__ = name or f"{code.name}_prepare_stabilizer_{digest}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(
        realization,
        implements=LogicalInstrumentRef(
            f"prepare_stabilizer_{digest}",
            0,
            1,
        ),
        name=realization.__name__,
        metadata={
            "preparation_kind": "stabilizer_state",
            "logical_stabilizers": state_key,
        },
    )


__all__ = [
    "css_memory_round",
    "logical_measure",
    "logical_pauli",
    "measure_x",
    "measure_z",
    "prepare_plus",
    "prepare_zero",
    "stabilizer_preparation",
]
