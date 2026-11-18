# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.       #
# ============================================================================ #
"""P0 surface memory -> composed DEM and ordinary CUDA-Q execution.

This file contains all example glue; it imports no other example modules.
The two paths independently lower the same P0 kernel with the same device.
CUDA-QX configures nv-qldpc from the model and corrects each shot in the kernel.
All generated artifacts default to build/logical-memory.
"""

import argparse
import json
from pathlib import Path

import cudaq
import cudaq.logical as cql
import cudaq_edem as edem
import cudaq_qec as qec
from cudaq.logical.architectures import surface


def memory_device(distance, *, decoder=None):
    """Specify how the abstract P0 memory program is physically realized.

    P0 says "prepare zero, idle, measure Z" on a logical qubit. It does not
    choose a code, a stabilizer-extraction circuit, or physical qubits. This
    device supplies those choices to placement and QEC/physical lowering:

        P1: one logical compute slot
        P2: a distance-d surface-code block and selected gadget implementations
        P3: physical qubits, gate processes, and measurement instruments

    Analysis and execution receive this same device. Consequently their
    independent P2 lowerings select the same circuits. Analysis follows the
    selected calls and uses their attached detector contracts to derive the
    error model. The device supplies implementations; the P0 kernel
    supplies their order and repetition count.

    This is a compilation resource model for the local example. Selecting the
    CUDA-Q execution target (Stim here), configuring noise, and configuring a
    decoder are separate choices. It is not a calibrated Quantinuum device.
    """
    # Returns a SurfaceInterleavedNZDefinitions bundle for this distance:
    #   prepare_zero: prepare logical zero, including one extraction round.
    #   syndrome_extraction: reset check ancillas and extract the stabilizers.
    #   measure_z_with_syndrome: extract once, then measure all data qubits
    #                           and return their logical-Z parity.
    # The bundle also contains code, distance, and physical-qubit coordinates.
    # These three gadgets are the surface-code subset needed for this memory
    # experiment. Each has a concrete interleaved N/Z circuit and an attached
    # detector contract describing measurement parities and syndrome boundaries;
    # EDEM derivation uses both the circuit and that contract.
    # With decoder=, these stock gadgets explicitly submit check measurements
    # and use decode_bit for terminal readout. Their contracts describe the
    # same physical circuit and its pre-correction observable for DEM analysis.
    recipe = surface.interleaved_nz_definitions(distance, decoder=decoder)

    # Make these implementations available to QEC selection. The encoding
    # identifies how one logical qubit is represented by the code block.
    # Each gadget declares the logical action it implements:
    #   prepare_zero             -> P0 prepare_zero
    #   syndrome_extraction      -> P0 idle (one extraction per idle round)
    #   measure_z_with_syndrome  -> P0 measure_z
    # link_roots is a set of available definitions, not an execution sequence;
    # the compiler selects calls according to the P0 program. Their nested
    # helper gadgets are linked as dependencies. The public Build inspection
    # API exposes the selected outer calls in program order.
    # Preparation and readout each include an extraction, so this recipe adds
    # two extraction rounds to those explicitly requested by P0 idle calls.
    architecture = cql.devices.QECArchitecture(
        "memory_nz",
        recipe.code.default_encoding,
        link_roots=(recipe.prepare_zero, recipe.syndrome_extraction,
                    recipe.measure_z_with_syndrome))
    builder = cql.devices.DeviceBuilder("memory_device", source_module=None)

    # Reserve capacity for one live LOGICAL qubit, not one physical qubit.
    # Bind that logical compute region to the surface-code realization above.
    # This tells placement where q can live and tells QEC lowering which code
    # and implementation bundle apply to operations on q.
    region = builder.qec.bind(builder.logical.add_compute(capacity=1),
                              architecture=architecture)

    # Back the encoded block with d*d data qubits and d*d-1 check ancillas.
    # Ancillas are reset and reused across rounds; the physical resource count
    # therefore does not grow with num_rounds. Typed action/instrument objects
    # provide the process definitions used by physical and Quake lowering.
    # The Clifford set covers this recipe's H/CX gates; MZ is the physical
    # single-qubit readout used for both syndrome and terminal measurements.
    builder.physical.bind(
        region,
        to=builder.physical.add_qubits(
            recipe.code.block.size,
            native_actions=cql.architecture.physical_actions.clifford_set(),
            native_instruments=(cql.architecture.physical_instruments.MZ,)))

    # Freeze the connected logical, QEC, and physical layers into the device
    # manifest passed to both compilation paths.
    return builder.build()


# CUDA-QX surface-memory reference settings for the nv-qldpc relay decoder.
RELAY_OPTIONS = {
    "use_sparsity": True,
    "max_iterations": 50,
    "bp_method": 3,
    "composition": 1,
    "gamma0": 0.0,
    "clip_value": 200.0,
    "repeatable": True,
    "srelay_config": {
        "pre_iter": 5,
        "num_sets": 10,
        "stopping_criterion": "All",
        "stop_nconv": 1
    },
    "gamma_dist": [0.1, 0.2],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7, 9), default=3)
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=12,
        help="total extractions, including preparation and readout (>=distance)"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.001,
        help="depolarization probability for the DEM and simulated shots")
    parser.add_argument(
        "--noiseless",
        action="store_true",
        help="disable shot noise while retaining the decoder model")
    parser.add_argument("--target",
                        choices=("stim", "quantinuum"),
                        default="stim")
    parser.add_argument("--emulate", action="store_true")
    parser.add_argument("--machine", default="Helios-1SC")
    parser.add_argument("--project")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        type=Path,
                        default=Path("build/logical-memory"))
    args = parser.parse_args(argv)
    if args.num_rounds < args.distance or args.shots < 1 or not 0 < args.p <= 0.75:
        parser.error(
            "require rounds >= distance, shots >= 1, and 0 < p <= 0.75")

    if args.target == "quantinuum" and not args.emulate and not args.project:
        parser.error("remote Quantinuum execution requires --project")
    return args


# Total extraction rounds, including one in preparation and one in readout.
num_rounds = 12
if __name__ == "__main__":
    args = parse_args()
    num_rounds = args.num_rounds


@cql.program
def memory() -> bool:
    q = cql.prepare_zero()
    for _ in range(num_rounds - 2):
        q = cql.idle(q, rounds=1)
    return cql.measure_z(q)


if __name__ == "__main__":
    # Bind Logical's generic packed-record interface to CUDA-QX callbacks.
    # These names belong to the runtime library loaded below. The same
    # cc.device_call IR is used locally and on Quantinuum; CUDA-Q owns the
    # target-specific lowering. Decoder choice and configuration stay here.
    decoder = cql.qec.DecoderSpec("memory_decoder",
                                  reset="reset_decoder_ui64",
                                  submit="enqueue_syndromes_ui64",
                                  correction="get_corrections_ui64",
                                  session_id=0)
    device = memory_device(args.distance, decoder=decoder)
    # Analysis: independently compile the same P0 kernel to its selected P2
    # gadgets. Our Python range loop expands during authoring, so calls() gives
    # every occurrence in order, including repeated calls to the SEC gadget.
    # It is a call-site query, not a runtime trace of adaptive control flow.
    p2 = cql.compile(memory,
                     pipeline=cql.compiler.pipelines.qec(),
                     device=device)
    selected = p2.definitions[p2.root.symbol].calls()
    available = {
        gadget.name: gadget
        for gadget in device.logical_to_qec[0].architecture.link_roots
    }
    contracts = tuple(available[symbol].contract for symbol in selected)
    # Match the CUDA-QX shot noise below: each CX operand independently receives
    # I with probability 1-p or X/Y/Z with probability p/3 each. There is no H,
    # reset, or measurement noise. Joint two-qubit depolarization would describe
    # a different channel, even with the same p.
    model = edem.project_DEM(
        cql.analysis.derive_edem_chain(
            contracts,
            noise=cql.compiler.IndependentCXDepolarizingNoise(
                probability=args.p)))

    # CUDA-QX consumes an independent-error DEM plus measurement-to-detector
    # rows. Factorization preserves the noise law exactly; coalescing combines
    # equivalent columns. Logical resolves the detector rows from the same
    # contracts, and CUDA-QX supplies their sparse transport representation.
    independent, _ = edem.coalesce_dem_columns(edem.factorize_dem(model))
    config = qec.decoder_config()
    config.id = 0  # Matches DecoderSpec.session_id above.
    config.type = "nv-qldpc-decoder"
    config.stim_dem = str(cql.analysis.to_stim_dem(independent))
    config.D_sparse = qec.d_sparse(
        cql.compiler.detector_record_indices(contracts))
    config.decoder_custom_args = RELAY_OPTIONS
    configuration = qec.multi_decoder_config()
    configuration.decoders = [config]

    # Independently lower the same P0 program through the official path.
    physical = cql.compile(memory,
                           pipeline=cql.compiler.pipelines.physical(),
                           device=device)
    # The selected gadgets carry fabric.submit_syndrome/decode_bit through
    # P2 -> P3. Stock Build.to_quake lowers them to the CUDA-QX runtime calls;
    # cudaq.run invokes to_quake automatically. No execution wrapper is needed.
    if args.target == "quantinuum":
        remote_options = {"project": args.project} if not args.emulate else {}
        cudaq.set_target("quantinuum",
                         emulate=args.emulate,
                         machine=args.machine,
                         extra_payload_provider="decoder",
                         **remote_options)
    else:
        cudaq.set_target("stim")
    # Load CUDA-QX's existing packed-uint64 callbacks, also used by local
    # execution. Their local implementation forwards to the configured server.
    qec.qecrt.load_quantinuum_realtime_decoding()
    cudaq.set_random_seed(args.seed)
    shot_noise = cudaq.NoiseModel()
    if not args.noiseless:
        shot_noise.add_all_qubit_channel("x",
                                         qec.TwoQubitDepolarization(args.p),
                                         num_controls=1)
    options = {"noise_model": shot_noise} if args.target == "stim" else {}
    # Configure before launch and always release worker/decoder resources.
    try:
        if qec.configure_decoders(configuration) != 0:
            raise RuntimeError("CUDA-QX decoder configuration failed")
        results = cudaq.run(physical, shots_count=args.shots, **options)
    finally:
        qec.finalize_decoders()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "memory.edemb").write_bytes(model.to_bytes())
    (args.output / "p2.mlir").write_text(p2.to_mlir())
    (args.output / "p3.mlir").write_text(physical.to_mlir())
    (args.output / "quake.mlir").write_text(str(physical.to_quake()))
    (args.output / "decoder.yaml").write_text(configuration.to_yaml_str(200))
    summary = {
        "shots": len(results),
        "remaining_logical_errors": sum(bool(value) for value in results)
    }
    (args.output /
     "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"d={args.distance}, total extraction rounds={args.num_rounds}")
    print("Selected P2 occurrences: " + " -> ".join(selected))
    print(summary)
    print(f"Artifacts: {args.output}")
