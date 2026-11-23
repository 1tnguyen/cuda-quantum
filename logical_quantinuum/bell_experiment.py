# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""Surface-code Bell experiment from Fig. 6 of arXiv:2609.13087.

Time runs from left to right as in Fig. 6(c).

    control  Prep(+)[1] --V-- SEC[3] --V-- @ --V-- SEC[5] --V-- MZ[7]
                                           |
    target   Prep(0)[2] --V-- SEC[4] --V-- X --V-- SEC[6] --V-- MZ[8]

SEC is a syndrome-extraction round. Gadget k supplies measurement record Mk
and detector group Dk. The virtual syndromes V connect each gadget's detector
map to its successor on that patch. Contracting those wires gives Fig. 6(d),
a map from M1,...,M8 to D1,...,D8. The transversal CX mixes the incoming
virtual syndromes without measuring either patch.

Each MZ contributes one logical observable. CUDA-QX returns corrected
observable 0 for the control MZ and 1 for the target MZ after their records
arrive. Their XOR is the deterministic Bell parity (Sec. 2.6, Eqs. 17-18).
The simulated noise is independent depolarization on each CX operand.
"""

import argparse

import cudaq.logical as cql
import cudaq_edem as edem
from cudaq.logical.architectures import surface

if __package__:
    from .surface_experiment import run_decoded
else:
    from surface_experiment import run_decoded


def bell_layers(memory, gates):
    """Each tuple is one EDEM layer, ordered from the left patch to the right."""
    return (
        (memory.prepare_plus, memory.prepare_zero),
        (memory.syndrome_extraction, memory.syndrome_extraction),
        (gates.transversal_cx,),
        (memory.syndrome_extraction, memory.syndrome_extraction),
        (memory.measure_z_data, memory.measure_z_data),
    )


def bell_protocol(layers, code):
    # Execute the same gadget occurrences and wire order used to build the EDEM.
    prep, before_cx, (cx,), after_cx, mz = layers

    @cql.protocol
    def bell() -> tuple[bool, bool]:
        left = prep[0](cql.ops.allocate_patch(code, region='compute'))
        right = prep[1](cql.ops.allocate_patch(code, region='compute'))
        left, right = before_cx[0](left), before_cx[1](right)
        left, right = cx(left, right)
        left, right = after_cx[0](left), after_cx[1](right)
        return mz[0](left), mz[1](right)

    return bell


def main(args):
    decoder = cql.qec.DecoderSpec(
        'bell_decoder',
        reset='reset_decoder_ui64',
        submit='enqueue_syndromes_mapped_ui64',
        logical_readout='get_logical_measurements_ui64',
        num_observables=2)
    memory = surface.interleaved_nz_definitions(args.distance, decoder=decoder)
    layers = bell_layers(memory, surface.definitions(args.distance))
    gadgets = tuple(dict.fromkeys(g for layer in layers for g in layer))
    # Tensor the independent maps on each layer and connect the patch ports
    # across layers. The CX is one two-patch map.
    noise = cql.compiler.IndependentCXDepolarizingNoise(probability=args.p)
    models = {g: cql.analysis.edem_from_gadget(g, noise=noise) for g in gadgets}
    model = edem.compose_chain(
        [edem.tensor_product([models[g] for g in layer]) for layer in layers])
    results = run_decoded(bell_protocol(layers, memory.code), memory.code, 2,
                          layers, gadgets, model, args.p, args.shots)
    # Each MZ returns its own corrected bit. Unequal bits fail the Bell ZZ check.
    errors = sum(left ^ right for left, right in results)
    print(
        f'd={args.distance} p={args.p:g} Bell parity errors={errors}/{len(results)}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance',
                        type=int,
                        choices=(3, 5, 7, 9, 11),
                        default=3)
    parser.add_argument('--p', type=float, default=.001)
    parser.add_argument('--shots', type=int, default=10000)
    args = parser.parse_args()
    if args.shots < 1 or not 0 < args.p <= .75:
        parser.error('require shots >= 1 and 0 < p <= .75')
    main(args)
