# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""Encoded GHZ ladder from Sections 6.2 and 6.3 of arXiv:2609.13087.

With three patches and r syndrome-extraction rounds per rung the circuit is

    q0    |+> -- @ -- [SEC]^r ------- [SEC]^r -- M
                 |
    q1    |0> -- X -- [SEC]^r -- @ -- [SEC]^r -- M
                                 |
    q2    |0> ------- [SEC]^r -- X -- [SEC]^r -- M

Each [SEC]^r column repeats the full layer r times (--rounds), with one
syndrome extraction on each of the n patches (--qubits) per round.
In the Z basis each adjacent pair must agree. In the X basis the XOR of
all n logical outcomes must be zero. Shots use the Bell example's independent
CX-operand depolarization. The separate Table 3 comparison uses p=0.01
and checks detector and observable flip probabilities for three fault models.

Rotated-surface G3 and G5 comparisons reproduced with Stim 1.16.0:

    Distance         Rounds per rung    Matching DEMs
    3, 5, 7          2, 10, 30          108/108
    11, 15, 21, 31   2, 10               96/96
    3, 5             100                 24/24
"""

import argparse

import cudaq.logical as cql
import cudaq_edem as edem
from cudaq.logical.architectures import surface

if __package__:
    from .surface_experiment import run_decoded
else:
    from surface_experiment import run_decoded


def ghz_layers(memory, cx, n, rounds, basis):
    layers = [(memory.prepare_plus,) + (memory.prepare_zero,) * (n - 1)]
    extraction = (memory.syndrome_extraction,) * n
    for control in range(n - 1):
        # For G3 the first rung is (cx, None): CX uses q0 and q1, and
        # q2 passes through. The following extraction covers all three.
        layers.append((None,) * control + (cx,) + (None,) * (n - control - 2))
        layers.extend([extraction] * rounds)
    readout = memory.measure_z_data if basis == 'Z' else memory.measure_x_data
    layers.append((readout,) * n)
    return layers


def ghz_protocol(layers, code):
    n = len(layers[0])

    @cql.protocol
    def ghz() -> tuple[(bool,) * n]:
        patches = [
            prepare(cql.ops.allocate_patch(code, region='compute'))
            for prepare in layers[0]
        ]
        for layer in layers[1:-1]:
            # Walk the patches in wire order. A bare wire advances one patch
            # without a gate call. A CX consumes and returns two patches.
            i = 0
            for gadget in layer:
                if gadget is None:
                    i += 1
                elif len(gadget.inputs) == 2:
                    patches[i], patches[i + 1] = gadget(patches[i],
                                                        patches[i + 1])
                    i += 2
                else:
                    patches[i] = gadget(patches[i])
                    i += 1
        return tuple(
            measure(patch) for measure, patch in zip(layers[-1], patches))

    return ghz


def compose_ghz(layers, models):
    # All patches use the code presentation of the first preparation.
    presentation, = models[layers[0][0]].codes
    blocks = []
    for kind, width in ((edem.PortKind.VIRTUAL,
                         presentation.virtual_outcome_count),
                        (edem.PortKind.SYNDROME, presentation.syndrome_count),
                        (edem.PortKind.LOGICAL,
                         2 * presentation.logical_qubit_count)):
        port = edem.Endpoint(kind)
        # Each diagonal entry copies one input bit to the same output bit.
        # The logical Pauli frame has an X and a Z bit per encoded qubit.
        identity = edem.SparseMatrix(width, width,
                                     [(i, i) for i in range(width)])
        blocks.append(edem.MatrixBlock(port, port, identity))
    # One-patch identity EDEM. Both [0] entries refer to codes[0], so the
    # input and output use the same code presentation. The identity blocks
    # carry existing errors through the rung without adding faults or detectors.
    wire = edem.EDEM(codes=[presentation],
                     inputs=[0],
                     outputs=[0],
                     blocks=blocks)
    layer_models = {}
    for layer in layers:
        if layer not in layer_models:
            # For the first G3 rung this tensors the q0-q1 CX EDEM with
            # the identity EDEM on q2, in that order.
            layer_models[layer] = edem.tensor_product(
                [wire if g is None else models[g] for g in layer])
    # Connect each layer's output ports to the next layer's input ports.
    return edem.compose_chain([layer_models[layer] for layer in layers])


def main(args):
    n = args.qubits
    # The shared decoder returns one corrected measurement per terminal patch.
    decoder = cql.qec.DecoderSpec(
        'ghz_decoder',
        reset='reset_decoder_ui64',
        submit='enqueue_syndromes_mapped_ui64',
        logical_readout='get_logical_measurements_ui64',
        num_observables=n)
    memory = surface.interleaved_nz_definitions(args.distance, decoder=decoder)
    cx = surface.definitions(args.distance).transversal_cx
    layers = ghz_layers(memory, cx, n, args.rounds, args.basis)
    # Derive each distinct gadget's EDEM once and reuse it at every occurrence.
    gadgets = tuple(
        dict.fromkeys(g for layer in layers for g in layer if g is not None))
    noise = cql.compiler.IndependentCXDepolarizingNoise(probability=args.p)
    models = {g: cql.analysis.edem_from_gadget(g, noise=noise) for g in gadgets}
    model = compose_ghz(layers, models)
    results = run_decoded(ghz_protocol(layers, memory.code), memory.code, n,
                          layers, gadgets, model, args.p, args.shots)
    # Count each failed shot once. Z checks every adjacent pair for agreement.
    # X checks that the parity of all corrected readouts is even.
    if args.basis == 'Z':
        errors = sum(
            any(a != b for a, b in zip(bits, bits[1:])) for bits in results)
    else:
        errors = sum(sum(bits) % 2 for bits in results)
    print(f'G{n} d={args.distance} r={args.rounds} basis={args.basis} '
          f'p={args.p:g} GHZ parity failures={errors}/{len(results)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', type=int, default=3)
    parser.add_argument('--distance', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--basis', choices=('Z', 'X'), default='Z')
    parser.add_argument('--p', type=float, default=.001)
    parser.add_argument('--shots', type=int, default=1000)
    args = parser.parse_args()
    if args.qubits < 2:
        parser.error('require at least two GHZ patches')
    if args.distance < 3 or args.distance % 2 == 0 or args.rounds < 1:
        parser.error('require odd distance >= 3 and rounds >= 1')
    if not 0 < args.p <= .75 or args.shots < 1:
        parser.error('require 0 < p <= .75 and shots >= 1')
    main(args)
