# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""Shared execution of layered surface-code experiments with a whole-shot decoder."""

import cudaq
import cudaq.logical as cql
import cudaq_edem as edem
import cudaq_qec as qec


def surface_device(code, gadgets, n):
    architecture = cql.devices.QECArchitecture('surface',
                                               code.default_encoding,
                                               link_roots=gadgets)
    builder = cql.devices.DeviceBuilder('surface_device', source_module=None)
    # Each logical slot needs one physical block for its data and check ancillas.
    region = builder.qec.bind(builder.logical.add_compute(capacity=n),
                              architecture=architecture)
    builder.physical.bind(
        region,
        to=builder.physical.add_qubits(
            n * code.block.size,
            native_actions=cql.architecture.physical_actions.clifford_set(),
            native_instruments=(cql.architecture.physical_instruments.MZ,)))
    return builder.build()


def measurement_maps(layers, gadgets):
    """Contract detector maps and collect the terminal layer's readout rows.

    QLX's detector_record_indices currently accepts a single patch chain.
    CUDA-QX needs these absolute measurement indices to form the detectors
    from the raw bits that the gadgets submit in layer order.
    """
    # Look up each named record's position within one execution of its gadget.
    # Repeated SEC calls use the same lookup at different positions in the shot.
    manifests = {}
    for gadget in gadgets:
        projection = cql.compiler.project_clifford(cql.compile(gadget.contract))
        manifests[gadget] = projection.record_indices, len(
            projection.interface.measurements)

    detectors = []
    offset = 0
    live = []
    for layer in layers:
        # Each live patch carries a table from syndrome bit to the earlier
        # measurement indices whose XOR gives that bit. Consume the previous
        # layer's tables from left to right and build the next layer's tables.
        inputs = iter(live)
        live = []
        for gadget in layer:
            # A bare wire carries its virtual syndromes through a CX on other patches.
            if gadget is None:
                live.append(next(inputs))
                continue
            contract = gadget.contract
            indices, count = manifests[gadget]
            # Each prep starts a wire. A one-patch SEC consumes one table,
            # while the CX takes both tables in control, target order.
            incoming = {port: next(inputs) for port in gadget.inputs}

            def resolve(parity):
                if parity.constant:
                    raise ValueError('detector rows require zero offset')
                support = set()
                # offset counts all measurements from earlier gadget calls.
                # Adding it turns a gadget-local index into a shot-wide index.
                for record in parity.records:
                    support ^= {offset + indices[record.name]}
                # Substitute the measurements behind each incoming syndrome.
                # Set XOR cancels any measurement that appears twice.
                for bit in parity.input_syndromes:
                    support ^= incoming[bit.endpoint][bit.index]
                return support

            # Each row tells CUDA-QX which raw bits to XOR for one detector.
            detectors.extend(
                sorted(resolve(d.parity)) for d in contract.detectors)
            # Carry the resolved output syndromes to the next layer. For example,
            # the CX target's outgoing Z check combines both incoming Z checks.
            # A destructive readout ends its wire and has no output table.
            for port in gadget.outputs:
                live.append({
                    row.index: resolve(row.parity)
                    for row in contract.output_syndromes
                    if row.endpoint == port
                })
            offset += count
    # CUDA-QX reads the last layer as its own measurement frame. Each sink
    # declares one observable. Offset its local support by preceding sinks'
    # widths so row i selects only the raw bits from patch i's MZ or MX.
    readouts = []
    terminal_width = 0
    for measure in layers[-1]:
        indices, count = manifests[measure]
        observable, = measure.contract.observables
        readouts.append(
            sorted(terminal_width + indices[r.name]
                   for r in observable.parity.records))
        terminal_width += count
    return detectors, readouts, terminal_width


def decoder_configuration(layers, gadgets, model, observable_groups=None):
    # Convert the composed EDEM to independent errors for CUDA-QX, merging
    # columns that flip the same detectors and logical observables.
    independent, _ = edem.coalesce_dem_columns(
        edem.factorize_dem(edem.project_DEM(model)))
    detectors, readouts, terminal_width = measurement_maps(layers, gadgets)
    # Coalescing produces one fault port and one row block per output kind.
    # H and O use the same columns as the independent fault probabilities.
    rows = {
        edem.PortKind.DETECTOR: [[] for _ in range(sum(independent.detectors))],
        edem.PortKind.OBSERVABLE:
            [[] for _ in range(sum(independent.observables))]
    }
    for block in independent.blocks:
        for row, fault in block.matrix.entries:
            rows[block.output.kind][row].append(fault)
    if observable_groups is not None:
        # Map fault effects into the same logical frame as the returned bits.
        # A fault affecting both terms of a parity cancels from that parity.
        raw_rows = rows[edem.PortKind.OBSERVABLE]
        mapped = []
        for group in observable_groups:
            faults = set()
            for index in group:
                faults.symmetric_difference_update(raw_rows[index])
            mapped.append(sorted(faults))
        rows[edem.PortKind.OBSERVABLE] = mapped
    fault_port, = independent.faults
    distribution, = fault_port.groups

    # The readout node skips the bulk stream and reads the last terminal_width
    # bits at FINAL. Its template has no faults: the decode node owns H, O and
    # the priors. Each m2o row names one patch's raw observable in MZ/MX order.
    empty_fault_rows = [-1] * len(readouts)
    return f"""
decoders:
  - id: 0
    block_size: {fault_port.size}
    syndrome_size: {len(detectors)}
    error_rate_vec: {distribution.probabilities}
    H_sparse: {qec.d_sparse(rows[edem.PortKind.DETECTOR])}
    O_sparse: {qec.d_sparse(rows[edem.PortKind.OBSERVABLE])}
    D_sparse: {qec.d_sparse(detectors)}
    task_graph:
      nodes:
        - id: decode
          type: single_window
          inner:
            type: nv-qldpc-decoder
            use_sparsity: true
            # Enable order-zero ordered-statistics decoding when BP fails.
            use_osd: true
            osd_method: 1
        - id: readout
          type: logical_readout
          dem_chunks:
            seam: {{from: next, to: prev}}
            phases_supply_D_sparse: true
            connections: [{{from: bulk, to: bulk}}, {{from: bulk, to: terminal}}]
            phases:
              - name: bulk
                spec: {{num_faults: 0, error_rates: [], O_sparse: {empty_fault_rows}, num_measurements: 1}}
              - name: terminal
                spec:
                  num_faults: 0
                  error_rates: []
                  O_sparse: {empty_fault_rows}
                  num_measurements: {terminal_width}
                  m2o_sparse: {qec.d_sparse(readouts)}
        - {{id: corrected, type: xor}}
      edges:
        - {{from: measurements, to: decode}}
        - {{from: measurements, to: readout}}
        - {{from: decode, to: output.corrections}}
        - {{from: decode, to: corrected.a}}
        - {{from: readout, to: corrected.b}}
        - {{from: corrected, to: output.logical}}
syndrome_mapping: [{{id: 0, type: bulk}}, {{id: 1, type: final}}]
"""


def run_decoded(protocol,
                code,
                n,
                layers,
                gadgets,
                model,
                p,
                shots,
                *,
                observable_groups=None):
    configuration = decoder_configuration(layers, gadgets, model,
                                          observable_groups)
    physical = cql.compile(protocol,
                           pipeline=cql.compiler.pipelines.physical(),
                           device=surface_device(code, gadgets, n))
    cudaq.set_target('stim')
    qec.qecrt.load_quantinuum_realtime_decoding()
    shot_noise = cudaq.NoiseModel()
    # CUDA-Q represents CX as X with one control. Use the same operand noise
    # probability here as in the gadget EDEMs used to configure the decoder.
    shot_noise.add_all_qubit_channel('x',
                                     qec.TwoQubitDepolarization(p),
                                     num_controls=1)
    try:
        if qec.configure_decoders_from_str(configuration) != 0:
            raise RuntimeError('CUDA-QX decoder configuration failed')
        return cudaq.run(physical, shots_count=shots, noise_model=shot_noise)
    finally:
        qec.finalize_decoders()
