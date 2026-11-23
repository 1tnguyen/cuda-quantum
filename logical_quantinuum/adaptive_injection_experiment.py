# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""Select and compose a P2 gadget's EDEM after a decoded measurement (EDEM §7.1).

    Inputs: data D in |+>, resources A and B in T|+>.
    First injection                   Decoded m       Second injection
    MPP(Z_D Z_A), MX(A)  ---------------- 0 --------> MPP(X_D Z_B), MX(B)
                                    '---- 1 --------> MPP(Y_D Z_B), MX(B)

These inject successive Z- and X-axis pi/8 Pauli rotations (Litinski, "A
Game of Surface Codes", Fig. 7). We track the first injection's Clifford
correction, so it changes the physical axis of the second measurement.
Each joint parity is read three times. Only the three parity *records* can flip,
independently with probability p. The MX read and magic inputs are ideal.

This CPU state-vector example uses three bare data carriers in one P2 block.
It illustrates the adaptive order of Pauli measurements at gadget level.
A surface-code implementation would also need encoded magic states and
lattice-surgery realizations of the three products.

    python3 logical_quantinuum/adaptive_injection_experiment.py --shots 128
"""

import argparse
from collections import Counter

import cudaq
import cudaq.logical as cql
import cudaq_edem as edem
import cudaq_qec as qec
import numpy as np
from cudaq.logical import types as t

# Keep D, A and B on the same open EDEM boundary through both injections.
# After its MX, a resource stays in its measured state and is never reused.
code = cql.codes.CSSCode(name='injection_register',
                         n=3,
                         k=3,
                         d=1,
                         lx=((0,), (1,), (2,)),
                         lz=((0,), (1,), (2,)))
patch = cql.patch[code]


def injection(axis, resource):

    @cql.action(name=f'inject_{axis.lower()}')
    def inject(
        d: t.logical_qubit, a: t.logical_qubit
    ) -> tuple[t.logical_qubit, t.logical_qubit, bool, bool]:
        d, a, parity = cql.mpp(getattr(t, axis)(d) @ t.Z(a))
        a, mx = cql.mpp(t.X(a))
        return d, a, parity, mx

    def contract(g):
        graph = cql.gadgets.GadgetContractBuilder(g.interface)
        r = [g.record(f'p{i}.outcome') for i in range(3)]
        # The first eigenvalue is random. Changes between repeats are checks.
        graph.detector(r[0] ^ r[1])
        graph.detector(r[1] ^ r[2])
        return graph.freeze()

    @cql.gadget(implements=inject,
                name=f'measure_{axis.lower()}z',
                logical_ports={
                    'd': 'block.q0',
                    'a': f'block.q{resource}'
                },
                contract=contract)
    def measure(block: patch) -> tuple[patch, bool, bool]:
        for i in range(3):
            block, parity = cql.ops.measure_pauli(block.data[(0, resource)],
                                                  paulis=axis + 'Z',
                                                  record=f'p{i}')
        block, mx = cql.ops.measure_pauli(block.data[resource],
                                          paulis='X',
                                          record='mx')
        return block, cql.ops.parity(parity), cql.ops.parity(mx)

    return measure


def measurement_kernel(gadget):
    """Execute the P2 product measurements on a resumable CUDA-Q CPU state."""
    circuit = cql.compiler.project_clifford(cql.compile(gadget.contract))
    kernel, state = cudaq.make_kernel(cudaq.State)
    q = kernel.qalloc(state)
    for op in circuit.instructions:
        if op.kind != 'MPP' or op.inverted:
            raise ValueError(
                'this example executes positive Pauli measurements')
        # Give each physical record a meter. Basis change, collect its parity
        # into the meter, then restore the data basis for the next measurement.
        meter = q[circuit.interface.carrier_count + op.measurement_start]
        kernel.reset(meter)
        for i, pauli in op.paulis:
            if pauli == 'Y':
                kernel.sdg(q[i])
            if pauli != 'Z':
                kernel.h(q[i])
            kernel.cx(q[i], meter)
            if pauli != 'Z':
                kernel.h(q[i])
            if pauli == 'Y':
                kernel.s(q[i])
        kernel.mz(meter)
    return kernel


def measure_and_decode(state, kernel, model, p, rng):
    state = cudaq.get_state(kernel, state)
    # Each meter has collapsed, so every occupied basis state has identical
    # bits 3..6. The data amplitudes are carried intact into the next gadget.
    basis = int(np.argmax(np.abs(np.asarray(state))))
    records = np.array([(basis >> (3 + i)) & 1 for i in range(4)],
                       dtype=np.uint8)
    records[:3] ^= rng.random(3) < p
    r0, r1, raw, mx = records

    # Only classical record faults are present. Their effects end in their
    # own stage, so the newest H and O blocks of the prefix suffice to decode.
    stage = len(model.faults) - 1
    h = np.zeros((2, 3), dtype=np.uint8)
    o = np.zeros((2, 3), dtype=np.uint8)
    for block in model.blocks:
        if block.input.kind != edem.PortKind.FAULT or block.input.index != stage:
            continue
        matrix = {
            edem.PortKind.DETECTOR: h,
            edem.PortKind.OBSERVABLE: o
        }[block.output.kind]
        for row, col in block.matrix.entries:
            matrix[row, col] = 1
    decoder = qec.get_decoder('single_error_lut', h)
    faults = decoder.decode([r0 ^ r1, r1 ^ raw]).result
    correction = (o @ np.rint(faults).astype(np.uint8)) % 2
    return state, int(raw ^ correction[0]), int(mx ^ correction[1]), int(raw)


def run(shots, p, seed):
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    prep = cudaq.make_kernel()
    q = prep.qalloc(7)  # D, A, B and four meters for the physical P2 records.
    for i in range(3):
        prep.h(q[i])
    for i in (1, 2):
        prep.t(q[i])
    initial = cudaq.get_state(prep)

    # Precompute the three candidate gadgets and their individual fault maps.
    kernels, models = {}, {}
    for axis, resource in [('Z', 1), ('X', 2), ('Y', 2)]:
        gadget = injection(axis, resource)
        faults = cql.analysis.FaultSet()
        for i in range(3):
            faults = faults.record_flip(gadget.record(f'p{i}.outcome'), p)
        models[axis] = cql.analysis.edem_from_gadget(gadget, faults=faults)
        kernels[axis] = measurement_kernel(gadget)

    branches = Counter()
    rerouted = 0
    for _ in range(shots):
        state, m, a, raw = measure_and_decode(initial, kernels['Z'],
                                              models['Z'], p, rng)
        # The correction taking physical D into its ideal frame is Z^a S^m.
        # Its conjugation sends logical X to (-1)^(a xor m) times X or Y.
        axis = 'Y' if m else 'X'
        # The EDEMs differ even on the same three wires. For example, an
        # incoming X_D reverses Y_D Z_B and commutes with X_D Z_B.
        prefix = edem.compose_chain([models['Z'], models[axis]])
        _, parity, mx, _ = measure_and_decode(state, kernels[axis], prefix, p,
                                              rng)
        if not branches[axis]:
            print(f'raw ZZ={raw}, decoded ZZ={m}, MX(A)={a} -> {axis}Z; '
                  f'signed second parity={parity ^ a ^ m}, MX(B)={mx}')
        if raw != m and not rerouted:
            print(f'First rerouted shot: raw ZZ={raw}, decoded ZZ={m}'
                  f' -> executed {axis}Z')
        branches[axis] += 1
        rerouted += raw != m
    print(f'ZZ -> XZ: {branches["X"]} shots; ZZ -> YZ: {branches["Y"]} shots')
    print(
        f'Decoding changed the branch relative to raw ZZ in {rerouted} shots.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', type=int, default=128)
    parser.add_argument('--p', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.shots < 1 or not 0 <= args.p < .5 or args.seed < 0:
        parser.error('require shots >= 1, 0 <= p < 0.5 and a nonnegative seed')
    run(args.shots, args.p, args.seed)
