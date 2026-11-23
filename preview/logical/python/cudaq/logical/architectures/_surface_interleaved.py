# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Rotated-surface gadgets with the interleaved N/Z gate schedule."""

from __future__ import annotations

from dataclasses import dataclass
from cudaq.logical import codes, gadgets, ops, std, types
from cudaq.logical.architecture.geometry import rotated_surface_lattice

# Displacements from each check ancilla, one interaction per carrier per layer.
# The final two interactions put X hooks horizontally and Z hooks vertically,
# transverse to the vertical logical X and horizontal logical Z strings.
_X_ORDER = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_Z_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True, slots=True)
class SurfaceInterleavedNZDefinitions:
    """Reusable preparations, SEC and destructive readouts for one distance.

    Preparations and SEC each include one extraction round. A data-only readout
    can follow an explicit SEC. Coordinates cover data, X ancillas,
    then Z ancillas, in the compiler's physical carrier order.
    """

    distance: int
    code: codes.Code
    prepare_zero: gadgets.GadgetDefinition
    syndrome_extraction: gadgets.GadgetDefinition
    measure_z_with_syndrome: gadgets.GadgetDefinition
    coordinates: tuple[tuple[float, float], ...]
    prepare_plus: gadgets.GadgetDefinition | None = None
    measure_z_data: gadgets.GadgetDefinition | None = None
    measure_x_data: gadgets.GadgetDefinition | None = None


def build_definitions(distance, *, decoder=None):
    from cudaq.logical.compiler import compile, project_clifford

    code = codes.Surface[distance]
    encoding = code.default_encoding
    lattice = rotated_surface_lattice(distance)
    data = {point: i for i, point in enumerate(lattice.coordinates)}
    xs = tuple(face for face in lattice.faces if face.sublattice == 1)
    zs = tuple(face for face in lattice.faces if face.sublattice == 0)
    ancillas = sorted(
        [(face.coordinate, "sx", i) for i, face in enumerate(xs)] +
        [(face.coordinate, "sz", i) for i, face in enumerate(zs)],
        key=lambda item: (item[0][1], item[0][0]))
    z_boundary_order = sorted(range(len(zs)), key=lambda i: zs[i].coordinate)

    def define(function, name, action, result, *, rounds=1):
        function.__name__ = function.__qualname__ = (
            f"surface_d{distance}_circuit_{name}")
        function.__annotations__ = {
            "block": types.patch[encoding],
            "return": result
        }
        return gadgets.gadget(function,
                              implements=action,
                              metadata={"syndrome_rounds": rounds})

    def extract(block):
        block = ops.h(block.sx)
        for x_delta, z_delta in zip(_X_ORDER, _Z_ORDER):
            for faces, delta, x_check in ((xs, x_delta, True), (zs, z_delta,
                                                                False)):
                for i, face in enumerate(faces):
                    point = (face.coordinate[0] + delta[0],
                             face.coordinate[1] + delta[1])
                    if point not in data:
                        continue
                    if x_check:
                        block = ops.cx(block.sx[(i,)],
                                       block.data[(data[point],)])
                    else:
                        block = ops.cx(block.data[(data[point],)],
                                       block.sz[(i,)])
        block = ops.h(block.sx)
        # Resets belong to the next gadget that uses each ancilla. Only data
        # errors cross the encoded interface between extraction rounds.
        measurements = []
        for _, partition, i in ancillas:
            block, bits = ops.mz(getattr(block, partition)[(i,)],
                                 record=f"{partition}{i}")
            measurements.append(bits)
        if decoder is not None:
            # One submission per extraction, preserving the measurement order
            # used by the detector contract below.
            ops.submit_syndrome(measurements, decoder=decoder)
        return block

    extraction = define(extract, "extract", std.idle, types.patch[encoding])

    def syndrome_round(block):
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        return extraction(block)

    sec = define(syndrome_round, "sec", std.idle, types.patch[encoding])

    def prepare_with_checks(block):
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        # Keep preparation's checks on this patch. Sharing SEC's nested
        # extraction here can bind its data operands to a sibling patch.
        return extract(block)

    def prepare(block):
        return prepare_with_checks(ops.reset(block.data))

    def prepare_plus(block):
        block = ops.reset(block.data)
        return prepare_with_checks(ops.h(block.data))

    prep = define(prepare, "prepare_zero", std.prepare_zero,
                  types.patch[encoding])
    plus = define(prepare_plus, "prepare_plus", std.prepare_plus,
                  types.patch[encoding])

    def measure_data(block, logical_support):
        if decoder is not None:
            block, bits = ops.mz(block.data, record="terminal_data")
            result = ops.decode_bit(bits,
                                    decoder=decoder,
                                    parity_indices=tuple(logical_support))
            ops.discard(block)
            return result
        logical = []
        for i in range(code.n):
            block, outcome = ops.mz(block.data[(i,)], record=f"data{i}")
            if i in logical_support:
                logical.append(outcome)
        ops.discard(block)
        return ops.parity(*logical)

    def measure_z_data(block):
        return measure_data(block, code.lz[0])

    def measure_x_data(block):
        return measure_data(ops.h(block.data), code.lx[0])

    def readout(block):
        block = sec(block)
        # Match the reference circuit's terminal reset fault sites. With no
        # subsequent ancilla interactions, those faults have zero response.
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        return measure_z_data(block)

    mz = define(readout, "measure_z", std.measure_z, bool)
    data_mz = define(measure_z_data,
                     "measure_z_data",
                     std.measure_z,
                     bool,
                     rounds=0)
    data_mx = define(measure_x_data,
                     "measure_x_data",
                     std.measure_x,
                     bool,
                     rounds=0)
    sites = []

    def contract(gadget, kind):
        projection = project_clifford(compile(gadget), gadget=gadget.name)
        port, = projection.interface.inputs or projection.interface.outputs
        graph = gadgets.GadgetContractBuilder(gadget.interface)
        # Each carrier is measured at most once in these single-round gadgets.
        measured = {
            row.carriers[0]: gadget.record(row.record)
            for row in projection.interface.measurements
        }

        def close_checks(previous, checks, order):
            for i in order:
                parity = previous[i]
                for q in checks[i]:
                    parity ^= measured[port.data_carriers[q]]
                graph.detector(parity)

        if kind in ("data_z", "data_x"):
            incoming = gadget.inputs.only().syndrome
            if kind == "data_z":
                close_checks([incoming[len(xs) + i] for i in range(len(zs))],
                             code.hz, z_boundary_order)
            else:
                close_checks([incoming[i] for i in range(len(xs))], code.hx,
                             range(len(xs)))
            return graph.freeze()
        # Extraction begins with H on the X ancillas, after the prep/reset.
        entry = next(op for op in projection.instructions
                     if op.kind == "H" and op.carriers == port.carriers("sx"))
        sites.append((gadget, projection, entry.site_id, port.data_carriers))
        records = {
            (partition, i): measured[q] for partition in ("sx", "sz")
            for i, q in enumerate(port.carriers(partition))
        }
        if kind == "prep":
            for i in z_boundary_order:
                graph.detector(records["sz", i])
        elif kind == "plus":
            for i in range(len(xs)):
                graph.detector(records["sx", i])
        else:
            incoming = gadget.inputs.only().syndrome
            for _, partition, i in ancillas:
                index = i if partition == "sx" else len(xs) + i
                graph.detector(incoming[index] ^ records[partition, i])
        if kind == "readout":
            close_checks([records["sz", i] for i in range(len(zs))], code.hz,
                         z_boundary_order)
        else:
            graph.bind(
                gadget.outputs.only().syndrome,
                tuple(records[partition, i]
                      for partition, faces in (("sx", xs), ("sz", zs))
                      for i in range(len(faces))))
        return graph.freeze()

    definitions = SurfaceInterleavedNZDefinitions(
        distance,
        code,
        prep,
        sec,
        mz,
        lattice.coordinates + tuple(face.coordinate for face in xs) +
        tuple(face.coordinate for face in zs),
        prepare_plus=plus,
        measure_z_data=data_mz,
        measure_x_data=data_mx)
    contracts = (contract(prep, "prep"), contract(plus, "plus"),
                 contract(sec, "sec"), contract(mz, "readout"),
                 contract(data_mz, "data_z"), contract(data_mx, "data_x"))
    return definitions, contracts, sites
