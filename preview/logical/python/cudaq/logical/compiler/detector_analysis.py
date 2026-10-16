# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compose gadget parity contracts into experiment-wide measurement parities.

This is the static analysis between P3 lowering and backend DEM generation:

* Gadget authors define ``fabric.detector``, ``fabric.observable``, and
  ``fabric.output_syndrome`` rules in their analysis profiles. These describe
  XORs of gadget-local measurement records and incoming syndrome components.
* P3 lowering retains the selected profiles, record projections, patch boundary
  bindings, and bindings from local observables to physical call results.
* This module walks the physical graph, numbers dynamic measurement occurrences,
  and substitutes incoming syndrome terms with the preceding gadget's outgoing
  parities. The result is expressed entirely in experiment measurement columns.
* ``qec.decoder_config.generate_dem`` passes the graph, these rows, and the
  user's noise model to a backend. That backend computes fault probabilities
  and the DEM. The configuration adapter then uses the submission layout
  collected here to arrange decoder matrix columns at setup time.

For example, suppose a profile explicitly declares an outgoing syndrome
``s_out = m`` and a detector ``d = s_in XOR m``. If preparation declares
``s_out = 0``, two invocations measuring columns 0 and 1 produce detector rows
``(0,)`` and ``(0, 1)``. If preparation leaves that syndrome unknown, the first
row retains an unknown term and is omitted; the second still becomes ``(0, 1)``.
This rule is an illustrative gadget contract, not a rule inferred by the
compiler. Code geometry, stabilizer ordering, and preparation semantics must
come from the selected profiles.

Parities describe fault-induced flips relative to the noiseless reference, not
absolute Boolean outcomes. XOR is set symmetric difference over GF(2): repeated
measurement terms cancel. Unknown boundary components are symbolic terms and
can also cancel; an unresolved term prevents a detector from being emitted.
Constant offsets do not change fault-flip parities. The backend is responsible
for quantum execution and reference outcomes; this analysis does not prove the
quantum validity of a gadget's declared detector rules.

The execution IR stays folded and is not modified. Analysis nevertheless visits
every occurrence of a fixed loop and produces flattened measurement/parity
lists; it is not a compact, symbolic composition of probabilistic DEM blocks.
It neither simulates quantum gates nor invokes a decoder, and does not add
runtime measurement shuffling.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType

from cudaq.logical.qec.decoder_spec import DecoderSpec


def _text(attr):
    return str(attr.value)


@dataclass(frozen=True)
class DetectorAnalysis:
    """Immutable analysis results shared by the DEM backend and config adapter.

    Every integer in a parity row or submission list indexes ``records``.
    Parity rows are sorted XOR supports, not lists of measured bit values.

    Attributes:
        records: Chronological measurements, identified by ``(record_id,
            occurrence)``. The zero-based occurrence distinguishes executions
            of the same static record in a loop; template aliases are resolved.
        detectors: Closed detector rows in traversal/profile declaration order.
            Rows with unresolved incoming syndrome terms are omitted.
        observables: Closed parities in program return order. A gadget's local
            observable is bound to its call result, so a discarded readout does
            not automatically become an experiment observable.
        submissions: Decoder symbol to measurement columns in executable operand
            order, concatenated across submission sites. Includes terminal data
            measurements consumed by ``phys.decode_bit``; no sorting is applied.
        decoders: Decoder symbol to its consistent, declared ``DecoderSpec``.
        readouts: Decoder symbol to the raw terminal logical parity selected by
            ``phys.decode_bit.parity_indices``. These are not decoded correction
            values. The config adapter checks this parity against the observable.
    """

    records: tuple[tuple[str, int], ...]
    detectors: tuple[tuple[int, ...], ...]
    observables: tuple[tuple[int, ...], ...]
    submissions: Mapping[str, tuple[int, ...]]
    decoders: Mapping[str, DecoderSpec]
    readouts: Mapping[str, tuple[int, ...]]


def analyze_detectors(build):
    """Analyze a verified P3 Build without changing its execution graph.

    Each physical call must name an explicit retained profile and bind its input
    and output patch ports. ``phys.record_projection`` connects profile-local
    record names to physical record IDs; ``profile_result_indices`` connects
    local observable ordinals to call results. Missing contracts are errors,
    rather than an invitation to infer a code-specific detector rule.

    Fixed ``cflow.repeat`` (including nested loops) and shared call templates are
    supported. Dynamic branches, while loops, retries, and postselection need
    additional model/acceptance contracts and are rejected. Results must be
    known Boolean parities, with at least one detector and one returned result.
    Decoder sites must consume raw measurements and close at most one lifetime
    per symbol. The CUDA-QX config adapter imposes its own narrower constraint
    of one decoder and one observable; that restriction is not imposed here.

    Returns:
        DetectorAnalysis with global measurement columns and decoder input maps.

    Raises:
        ValueError: Invalid IR or missing/inconsistent analysis bindings.
        NotImplementedError: A construct has no supported static model contract.
    """
    if not build.module.operation.verify():
        raise ValueError("detector analysis requires verified P3 IR")
    symbols = {
        _text(op.attributes["sym_name"]): op
        for view in build.module.body.operations
        if "sym_name" in (op := view.operation).attributes
    }
    graph = symbols.get(build.root.symbol)
    if graph is None or graph.name != "phys.graph":
        raise ValueError("detector analysis requires a P3 physical graph")
    # Shared call templates refer to a canonical body by event ID. Indexing the
    # body does not execute it; walk() visits it for each dynamic invocation.
    events = {}

    def index_events(op):
        if "event_id" in op.attributes:
            events[_text(op.attributes["event_id"])] = op
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    index_events(child.operation)

    index_events(graph)
    # (gadget instance, stable profile record) -> physical static record ID.
    # Dynamic occurrence selection happens later, inside the invocation scope.
    projection = {}
    for op in symbols.values():
        if op.name != "phys.record_projection" or _text(
                op.attributes["graph"]) != build.root.symbol:
            continue
        for entry in op.attributes["entries"]:
            key = (_text(entry["instance"]), _text(entry["source_record"]))
            physical = _text(entry["physical_record"])
            if key in projection and projection[key] != physical:
                raise NotImplementedError(
                    "analysis requires one physical record template per source record"
                )
            projection[key] = physical

    records, detectors = [], []
    # counts persists across invocations; measured resolves only records visible
    # in the current call. boundaries carries syndrome expressions across calls:
    # opaque patch identity -> ({syndrome index: parity}, unknown epoch).
    # A missing syndrome entry means unknown, never implicitly zero. Each output
    # boundary gets a fresh epoch so unrelated unknown values cannot cancel.
    counts, measured, boundaries = {}, {}, {}
    submissions, decoders, readouts = {}, {}, {}
    unknown_epoch = 0

    def parity(declaration, instance, incoming, aliases):
        """Substitute one profile row into global columns and unknown tokens.

        Record names resolve through the instance projection, template aliases,
        and this invocation's measurements. Incoming syndrome terms resolve
        through the boundary snapshot captured before entering the gadget.
        Repeated terms, including the same symbolic unknown, cancel under XOR.
        """
        support = set()
        for source in declaration.attributes.get("records", ()):
            try:
                record = projection[(instance, _text(source))]
                index = measured[aliases.get(record, record)]
            except KeyError as exc:
                raise ValueError(
                    "profile record has no measurement in this invocation"
                ) from exc
            support.symmetric_difference_update((index,))
        for term in declaration.attributes.get("input_syndromes", ()):
            port, index = int(term["port_index"]), int(term["index"])
            if not 0 <= port < len(incoming):
                raise ValueError(
                    "profile input syndrome has no physical patch binding")
            rows, epoch = incoming[port]
            support.symmetric_difference_update(
                rows.get(index, (f"unknown:{epoch}:{index}",)))
        # Constant offsets change the reference outcome, not the fault parity
        # used by a DEM. Quantum execution supplies that reference to its backend.
        return frozenset(support)

    def profile(call, incoming, outputs, aliases):
        """Apply a call's contract after its physical body has been visited.

        Detectors append global rows. Observables replace the abstract value of
        their bound call result, allowing SSA uses/returns to select experiment
        observables. Output syndromes replace each outgoing patch boundary;
        unspecified components become unknown instead of retaining stale rows.
        """
        nonlocal unknown_epoch
        instance = _text(call.attributes["instance"])
        contract = symbols[_text(call.attributes["profile"])]
        outgoing = {}
        for view in contract.regions[0].blocks[0].operations:
            op = view.operation
            if op.name in {"fabric.profile_end", "fabric.success"}:
                if op.name == "fabric.success":
                    raise NotImplementedError(
                        "DEM generation with postselection needs an explicit acceptance policy"
                    )
                continue
            if op.name not in {
                    "fabric.detector", "fabric.observable",
                    "fabric.output_syndrome"
            }:
                raise NotImplementedError(
                    f"detector analysis does not support {op.name}")
            row = parity(op, instance, incoming, aliases)
            known = all(isinstance(index, int) for index in row)
            if op.name == "fabric.detector":
                if known:
                    detectors.append(tuple(sorted(row)))
            elif op.name == "fabric.observable":
                if not known:
                    raise ValueError(
                        "observable depends on an unknown incoming syndrome")
                indices = call.attributes.get("profile_result_indices")
                ordinal = int(op.attributes["idx"])
                if indices is None or not 0 <= ordinal < len(indices):
                    raise ValueError(
                        "profile observable has no gadget result binding")
                index = int(indices[ordinal])
                if not 0 <= index < len(outputs) or not (
                        str(call.results[index].type) == "i1" or
                        str(call.results[index].type).startswith(
                            "!phys.record<")):
                    raise ValueError(
                        "profile observable binding must name a Boolean result")
                outputs[index] = row
            else:
                outgoing.setdefault(int(op.attributes["port_index"]),
                                    {})[int(op.attributes["index"])] = row
        for port, patch in enumerate(call.attributes["profile_output_patches"]):
            unknown_epoch += 1
            boundaries[str(patch)] = (outgoing.get(port, {}), unknown_epoch)

    def decoder_site(op, values):
        """Record the executable decoder input contract, without decoding.

        Submission order comes directly from physical operands. A terminal
        decode_bit both contributes its input measurements and closes the
        symbol's lifetime. Its raw parity is recorded separately for the config
        adapter; no correction value can be computed by this static analysis.
        """
        name = _text(op.attributes["decoder"])
        if "decoder_implementation" not in op.attributes:
            raise ValueError(
                f"decoder @{name} needs a DecoderSpec to generate its configuration"
            )
        spec = DecoderSpec(
            name,
            implementation=_text(op.attributes["decoder_implementation"]),
            inputs=tuple(_text(v) for v in op.attributes["decoder_inputs"]),
            options=json.loads(_text(op.attributes["decoder_options_json"]))
            if "decoder_options_json" in op.attributes else None)
        if spec.inputs != ("measurements",):
            raise ValueError(
                "CUDA-QX configuration requires raw measurement decoder inputs")
        if name in decoders and decoders[name] != spec:
            raise ValueError(f"inconsistent DecoderSpec for @{name}")
        decoders[name] = spec
        inputs = [values.get(v) for v in op.operands]
        if any(type(value) is not int for value in inputs):
            raise ValueError(
                "decoder submissions must reference physical measurement records"
            )
        if name in readouts:
            raise NotImplementedError(
                "one closed decoder lifetime per symbol is required")
        submissions.setdefault(name, []).extend(inputs)
        if op.name == "phys.decode_bit":
            indices = tuple(int(i) for i in op.attributes["parity_indices"])
            row = set()
            for index in indices:
                row.symmetric_difference_update((inputs[index],))
            readouts[name] = tuple(sorted(row))

    def walk(block, args=(), enclosing=None, aliases=None):
        """Interpret SSA values as measurement provenance, not quantum state.

        The value map uses int for a raw measurement column, frozenset for a
        Boolean fault parity, and None for unmodeled values such as quantum
        state. Boolean constants have empty fault parity. A block returns the
        abstract values of its terminator operands; repeats carry these values
        between iterations and calls expose them to their enclosing block.

        aliases maps shared-body record IDs to this invocation's IDs. Quantum
        gates are visited but not simulated: the DEM backend executes the same
        graph to determine how faults affect its measurements.
        """
        nonlocal measured
        aliases = aliases or {}
        values = dict(enclosing or {})
        values.update(zip(block.arguments, args))
        for view in block.operations:
            op = view.operation
            args = [values.get(value) for value in op.operands]
            outputs = [None] * len(op.results)
            if op.name in {"phys.return", "phys.yield", "cflow.yield"}:
                return args
            if op.name in {"phys.call", "phys.call_template"}:
                if "profile" not in op.attributes:
                    raise ValueError(
                        "DEM generation requires an explicit analysis profile for every physical call"
                    )
                contract = symbols[_text(op.attributes["profile"])]
                for side in ("input", "output"):
                    key = f"profile_{side}_patches"
                    if key not in op.attributes or len(
                            op.attributes[key]) != len(
                                contract.attributes[f"{side}s"]):
                        raise ValueError(
                            "profiled call is missing its exact physical patch boundary"
                        )
                # Patch bindings are opaque identities (reference labels or
                # native physical resource arrays), not inferred geometry.
                # Capture the old boundary before the body/profile replaces it.
                incoming = []
                for patch in op.attributes["profile_input_patches"]:
                    key = str(patch)
                    incoming.append(boundaries.setdefault(key, ({}, key)))
                # Records must come from this invocation, never a preceding
                # iteration with the same static record ID.
                previous = measured
                measured = {}
                body_owner = op
                local_aliases = dict(aliases)
                if op.name == "phys.call_template":
                    body_owner = events[_text(op.attributes["template_event"])]
                    for alias in op.attributes.get("record_aliases", ()):
                        target = _text(alias["alias"])
                        local_aliases[_text(alias["template"])] = aliases.get(
                            target, target)
                    # Native templates may omit state-only SSA boundaries;
                    # explicit profile patch bindings still carry syndromes.
                    if "state_boundary_elided" in op.attributes:
                        args = [None] * len(
                            body_owner.regions[0].blocks[0].arguments)
                outputs = walk(body_owner.regions[0].blocks[0], args, values,
                               local_aliases)
                if "state_boundary_elided" in op.attributes:
                    outputs = []
                profile(op, incoming, outputs, aliases)
                previous.update(measured)
                measured = previous
            elif op.name == "cflow.repeat":
                for _ in range(int(op.attributes["count"])):
                    args = walk(op.regions[0].blocks[0], args, values, aliases)
                outputs = args
            elif op.name in {"phys.measure", "phys.measure_product"}:
                record = _text(op.attributes["record_id"])
                record = aliases.get(record, record)
                occurrence = counts.get(record, 0)
                counts[record] = occurrence + 1
                measured[record] = len(records)
                records.append((record, occurrence))
                outputs[-1] = len(records) - 1
            elif op.name in {"phys.submit_syndrome", "phys.decode_bit"}:
                decoder_site(op, values)
                if op.name == "phys.decode_bit":
                    # Model the uncorrected observable for DEM generation. The
                    # execution backend still performs decoding and correction.
                    outputs[0] = frozenset(readouts[_text(
                        op.attributes["decoder"])])
            elif op.name == "phys.condition":
                if isinstance(args[0], frozenset):
                    outputs[0] = args[0]
                elif type(args[0]) is int:
                    outputs[0] = frozenset((args[0],))
                else:
                    raise ValueError(
                        "condition requires a physical measurement record")
            elif op.name == "phys.xor":
                if any(not isinstance(value, frozenset) for value in args):
                    raise ValueError(
                        "observable XOR requires known Boolean parities")
                outputs[0] = args[0] ^ args[1]
            elif op.name == "arith.constant" and str(
                    op.results[0].type) == "i1":
                # A constant changes the reference value, not its fault parity.
                outputs[0] = frozenset()
            elif op.name in {
                    "phys.acquire", "phys.release", "phys.prepare",
                    "phys.reset", "phys.apply", "phys.delay", "phys.tick",
                    "phys.barrier", "arith.constant"
            }:
                pass
            else:
                raise NotImplementedError(
                    f"static detector analysis does not support {op.name}")
            if len(outputs) != len(op.results):
                raise ValueError(
                    "physical analysis encountered an invalid call boundary")
            values.update(zip(op.results, outputs))
        raise ValueError("physical analysis block has no return")

    # Return position, rather than a profile-local observable index, defines
    # experiment observable numbering. Reject any unresolved returned parity.
    returned = walk(graph.regions[0].blocks[0])
    if not detectors or not returned:
        raise ValueError(
            "DEM generation requires detectors and returned observables")
    if any(not isinstance(row, frozenset) or any(
            type(index) is not int for index in row) for row in returned):
        raise ValueError(
            "experiment results require known Boolean observable parities")
    return DetectorAnalysis(
        tuple(records), tuple(detectors),
        tuple(tuple(sorted(row)) for row in returned),
        MappingProxyType({
            key: tuple(value) for key, value in submissions.items()
        }), MappingProxyType(decoders), MappingProxyType(readouts))
