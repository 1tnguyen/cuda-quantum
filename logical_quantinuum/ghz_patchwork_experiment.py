# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""Logical P2 GHZ preparation by adjacent surface-code ZZ measurements.

    patch   prep + initial SEC    rung 0     idle      rung 1     idle      readout
    q0      |+>, d rounds          ZZ01      [SEC]^r   [SEC]^d    [SEC]^r   M
    q1      |+>, d rounds          ZZ01      [SEC]^r    ZZ12      [SEC]^r   M
    q2      |+>, d rounds         [SEC]^d    [SEC]^r    ZZ12      [SEC]^r   M

ZZ01 and ZZ12 each denote one joint measurement shared by two rows. In Z,
the three reported bits include the measured Pauli frame:
    z0, z1 ^ m01, z2 ^ m01 ^ m12
CUDA-QX predicts the fault flips for each of these three expressions.
As in Patchwork Appendix B (arXiv:2610.22112), a measurement layer runs d
merged-check rounds while spectators run d memory rounds. Here the patches
touch along a straight seam. Each ZZ gadget restores the independent
boundary checks before returning. Every rung is followed by r ordinary SEC
rounds on every patch, including the last rung before readout.

This example uses a fixed seam and a CUDA-QX whole-shot decoder. The paper
also describes routing corridors and partitioning the DEM for staged decoding.
"""

import argparse
from collections import Counter

import cudaq.logical as cql
import cudaq_edem as edem
from cudaq.logical.architectures import surface

if __package__:
    from .surface_experiment import run_decoded
else:
    from surface_experiment import run_decoded


def ghz_layers(memory, seams, n, d, rounds, basis):
    # Each lane is a sequence on one patch (or the ZZ pair). Lanes tensor
    # together; layers compose in time. Preparation includes one SEC round.
    sec = memory.syndrome_extraction
    layers = [((memory.prepare_plus,),) * n]
    layers.extend([((sec,),) * n] * (d - 1))
    for i in range(n - 1):
        spectators = (sec,) * d
        layers.append((spectators,) * i + ((seams[i % 2],),) + (spectators,) *
                      (n - i - 2))
        layers.extend([((sec,),) * n] * rounds)
    readout = memory.measure_z_data if basis == 'Z' else memory.measure_x_data
    layers.append(((readout,),) * n)
    return layers


def record_layers(layers):
    # CUDA-QX indexes the submitted records in P2 call order. For its record
    # map only, visit parallel lanes left to right, preserving each SEC call.
    n = len(layers[0])
    yield tuple(lane[0] for lane in layers[0])
    for layer in layers[1:-1]:
        start = 0
        for lane in layer:
            width = len(lane[0].inputs)
            for gadget in lane:
                yield (None,) * start + (
                    gadget,) + (None,) * (n - start - width)
            start += width
    yield tuple(lane[0] for lane in layers[-1])


def ghz_protocol(layers, code, basis):
    n = len(layers[0])

    @cql.protocol
    def ghz() -> tuple[(bool,) * n]:
        patches = [
            prep(cql.ops.allocate_patch(code, region='compute'))
            for (prep,) in layers[0]
        ]
        joint = []
        for layer in layers[1:-1]:
            i = 0
            for lane in layer:
                for gadget in lane:
                    if len(gadget.inputs) == 2:
                        patches[i], patches[i + 1], bit = gadget(
                            patches[i], patches[i + 1])
                        joint.append(bit)
                    else:
                        patches[i] = gadget(patches[i])
                i += len(lane[0].inputs)

        bits = [
            measure(patch) for (measure,), patch in zip(layers[-1], patches)
        ]
        if basis == 'Z':
            # The decoder corrects fault flips in the frame-adjusted observables.
            # The measured random ZZ branch still supplies the raw X frame.
            prefix = joint[0]
            for i in range(1, n):
                if i > 1:
                    prefix = cql.ops.xor(prefix, joint[i - 1])
                bits[i] = cql.ops.xor(bits[i], prefix)
        return tuple(bits)

    return ghz


def run(n, distance, rounds, basis, p, shots):
    decoder = cql.qec.DecoderSpec(
        'ghz_surgery_decoder',
        reset='reset_decoder_ui64',
        submit='enqueue_syndromes_mapped_ui64',
        logical_readout='get_logical_measurements_ui64',
        num_observables=n)
    memory = surface.interleaved_nz_definitions(distance, decoder=decoder)
    # Build each seam orientation once. The P2 rungs and their EDEMs share it.
    seams = [
        zz_surgery(memory.code, distance, bottom=bool(i), decoder=decoder)
        for i in range(min(2, n - 1))
    ]
    layers = ghz_layers(memory, seams, n, distance, rounds, basis)
    gadgets = tuple(
        dict.fromkeys(g for layer in layers for lane in layer for g in lane))
    noise = cql.compiler.IndependentCXDepolarizingNoise(probability=p)
    models = {g: cql.analysis.edem_from_gadget(g, noise=noise) for g in gadgets}

    # Compose the spectator's d extractions, tensor with the two-patch seam
    # and other spectators, then connect the open patch ports across layers.
    lanes = {
        lane: edem.compose_chain([models[g] for g in lane])
        for lane in dict.fromkeys(lane for layer in layers for lane in layer)
    }
    model = edem.compose_chain([
        edem.tensor_product([lanes[lane] for lane in layer]) for layer in layers
    ])

    # EDEM observables are the n-1 ZZ results followed by n terminal bits.
    # Returned Z_i includes ZZ_0 through ZZ_(i-1). An X frame commutes with MX.
    groups = [(*range(i), n - 1 + i) if basis == 'Z' else (n - 1 + i,)
              for i in range(n)]
    return run_decoded(ghz_protocol(layers, memory.code, basis),
                       memory.code,
                       n,
                       tuple(record_layers(layers)),
                       gadgets,
                       model,
                       p,
                       shots,
                       observable_groups=groups)


@cql.action
def measure_zz(
    a: cql.types.logical_qubit, b: cql.types.logical_qubit
) -> tuple[cql.types.logical_qubit, cql.types.logical_qubit, bool]:
    return cql.mpp(cql.types.Z(a) @ cql.types.Z(b))


def zz_surgery(code, distance, *, bottom, decoder):
    """Build a straight ZZ seam between two mirror-oriented surface patches.

    Column j on each patch faces column j on the other. Two facing weight-two
    X boundary checks form one weight-four X check. The new Z checks tile
    the seam and their product measures the logical ZZ on the facing edges.
    """
    patch = cql.patch[code]
    edge = tuple(
        range(code.n - distance, code.n) if bottom else range(distance))
    checks = (*code.hx, *code.hz)
    edge_sites = set(edge)
    boundary = [
        i for i, row in enumerate(code.hx) if edge_sites.issuperset(row)
    ]
    # A seam Z check occupies the gap between two consecutive X faces.
    cuts = {min(code.hx[i]) for i in boundary}
    spans = []
    start = 0
    for j, site in enumerate(edge):
        if site in cuts or j == distance - 1:
            spans.append(edge[start:j + 1])
            start = j + 1

    # Each support is a tuple of (patch index, local data index) pairs.
    # Names identify the same physical records for execution and the EDEM.
    merged = {}
    split_checks = {}
    for i, row in enumerate(checks):
        is_x = i < len(code.hx)
        pauli = 'X' if is_x else 'Z'
        part, index = ('sx', i) if is_x else ('sz', i - len(code.hx))
        if i in boundary:
            support = [(p, q) for p in (0, 1) for q in row]
            merged[f'join{i}'] = pauli, (0, part, index), support
        # The final CX pair in a bulk Z check runs along a column. A Z
        # ancilla fault then spreads transverse to the horizontal logical Z.
        if not is_x:
            row = sorted(row, key=lambda q: (q % distance, q // distance))
        for p in (0, 1):
            check = pauli, (p, part, index), [(p, q) for q in row]
            if i in boundary:
                split_checks[f'split{p}c{i}'] = check
            else:
                merged[f'p{p}c{i}'] = check
    for j, span in enumerate(spans):
        # Borrow an adjacent boundary ancilla on the other patch for this
        # seam check. It is reset before reuse, including at the seam end.
        index = min(
            boundary,
            key=lambda i: abs(sum(code.hx[i]) / 2 - sum(span) / len(span)))
        support = [(p, q) for q in span for p in (0, 1)]
        merged[f'seam{j}'] = 'Z', (1, 'sx', index), support

    # On the bottom edge, multiplying every Z stabilizer on both patches
    # moves the measured Z strings to the code's canonical top-row strings.
    outcome_names = [f'seam{j}' for j in range(len(spans))]
    if bottom:
        outcome_names.extend(f'p{p}c{i}' for p in (0, 1)
                             for i in range(len(code.hx), len(checks)))

    def contract(g):
        graph = cql.gadgets.GadgetContractBuilder(g.interface)
        inputs = g.inputs.blocks
        previous = {}
        for i in range(len(checks)):
            if i in boundary:
                previous[
                    f'join{i}'] = inputs[0].syndrome[i] ^ inputs[1].syndrome[i]
            else:
                for p in (0, 1):
                    previous[f'p{p}c{i}'] = inputs[p].syndrome[i]

        def record(name, check):
            _, (_, part, _), _ = check
            return g.record(f'{name}.{part}0')

        for t in range(distance):
            current = {
                name: record(f'r{t}_{name}', check)
                for name, check in merged.items()
            }
            # First-round seam outcomes project onto a random ZZ sector.
            # From the second round onward their differences are detectors.
            for name, prior in previous.items():
                graph.detector(prior ^ current[name])
            previous = current
        split = {
            (p, i): record(f'split{p}c{i}', split_checks[f'split{p}c{i}'])
            for p in (0, 1) for i in boundary
        }
        for i in boundary:
            graph.detector(split[0, i] ^ split[1, i] ^ previous[f'join{i}'])
        for p, output in enumerate(g.outputs.blocks):
            graph.bind(
                output.syndrome,
                tuple(split[p, i] if i in boundary else previous[f'p{p}c{i}']
                      for i in range(len(checks))))
        # The gadget's returned bit supplies its logical observable.
        return graph.freeze()

    @cql.gadget(implements=measure_zz,
                contract=contract,
                name=f'surface_d{distance}_zz_{"bottom" if bottom else "top"}')
    def surgery(a: patch, b: patch) -> tuple[patch, patch, bool]:
        blocks = [a, b]

        def extract(check, record):
            pauli, (owner, part, index), support = check

            def ancilla():
                return getattr(blocks[owner], part)[(index,)]

            blocks[owner] = cql.ops.reset(ancilla())
            if pauli == 'X':
                blocks[owner] = cql.ops.h(ancilla())
            for p, q in support:
                data = blocks[p].data[(q,)]
                if pauli == 'X':
                    control, target = owner, p
                    result = cql.ops.cx(ancilla(), data)
                else:
                    control, target = p, owner
                    result = cql.ops.cx(data, ancilla())
                if control == target:
                    blocks[p] = result
                else:
                    blocks[control], blocks[target] = result
            if pauli == 'X':
                blocks[owner] = cql.ops.h(ancilla())
            blocks[owner], bit = cql.ops.mz(ancilla(), record=record)
            return bit

        for t in range(distance):
            bits = {
                name: extract(check, f'r{t}_{name}')
                for name, check in merged.items()
            }
            cql.ops.submit_syndrome(list(bits.values()), decoder=decoder)
        zz = cql.ops.parity(*(bits[name] for name in outcome_names))

        # Split by remeasuring the independent X boundary checks. All other
        # checks still have their last merged-round values at this boundary.
        split = [extract(check, name) for name, check in split_checks.items()]
        cql.ops.submit_syndrome(split, decoder=decoder)
        return blocks[0], blocks[1], zz

    return surgery


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', type=int, default=3)
    parser.add_argument('--distance', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--basis', choices=('Z', 'X'), default='Z')
    parser.add_argument('--p', type=float, default=.001)
    parser.add_argument('--shots', type=int, default=1000)
    args = parser.parse_args()
    if args.qubits < 2 or args.distance < 3 or args.distance % 2 == 0:
        parser.error('require at least two patches and odd distance >= 3')
    if args.rounds < 1 or args.shots < 1 or not 0 < args.p <= .75:
        parser.error('require rounds and shots >= 1 and 0 < p <= .75')
    results = run(args.qubits, args.distance, args.rounds, args.basis, args.p,
                  args.shots)
    counts = Counter(''.join(str(int(b)) for b in bits) for bits in results)
    if args.basis == 'Z':
        errors = sum(
            count for bits, count in counts.items() if len(set(bits)) > 1)
    else:
        errors = sum(
            count for bits, count in counts.items() if bits.count('1') % 2)
    print(
        f'G{args.qubits} d={args.distance} r={args.rounds} basis={args.basis} '
        f'p={args.p:g} GHZ parity failures={errors}/{len(results)}')
    print(dict(sorted(counts.items())))
