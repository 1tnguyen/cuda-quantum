# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile a CUDA-Q memory kernel and generate a CUDA-QX decoder configuration.

Requires CUDA-QX and an external analysis backend providing dem_from_kernel
for Logical P3. P3-to-QIR translation is deliberately not supplied here.

Example (once a backend is provided by the caller):
  python memory.py --analysis-backend my_backend:stim --distance 3 --rounds 12

The analysis backend is a module attribute, independent of the backend later
used for execution. The generated config can be installed locally using
qec.configure_decoders(config), or serialized for a remote decoder payload.
"""
import argparse
import importlib
from pathlib import Path

import cudaq
import cudaq.logical as cql
import cudaq_qec as qec


@cudaq.kernel
def memory(num_rounds: int) -> bool:
    q = cudaq.qubit()
    for _ in range(num_rounds):
        idle(q)
    return mz(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-backend",
                        required=True,
                        metavar="MODULE:OBJECT")
    parser.add_argument("--distance", type=int, choices=(3, 5), default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--p-cx", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path("decoder.yaml"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not 0 <= args.p_cx <= 1:
        parser.error("--p-cx must be a probability")
    module, attribute = args.analysis_backend.split(":", 1)
    analysis_target = getattr(importlib.import_module(module), attribute)

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x",
                                qec.TwoQubitDepolarization(args.p_cx),
                                num_controls=1)
    decoder = cql.qec.DecoderSpec("memory_decoder",
                                  implementation="nv-qldpc-decoder",
                                  inputs=("measurements",),
                                  options={
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
                                  })
    code = cql.codes.Surface[args.distance]
    carriers = cql.architecture.ResourceClass(
        "qubit",
        code.block.size,
        native_actions=("h", "cx"),
        native_instruments=(cql.architecture.physical_instruments.MZ,))
    builder = cql.devices.DeviceBuilder(
        "memory_device",
        physical=cql.architecture.PhysicalMachine(
            "memory_machine", resource_classes={"carriers": carriers}))
    compute = builder.logical.add_compute(capacity=1)
    region = builder.qec.bind(
        compute,
        architecture=cql.devices.QECArchitecture(
            "memory_architecture",
            code.default_encoding,
            link_roots=(cql.gadgets.prepare_zero(code, preparation="product"),
                        cql.gadgets.css_memory_round(code, decoder=decoder),
                        cql.gadgets.measure_z(code, decoder=decoder))))
    builder.physical.bind(region, to=builder.physical.carriers)
    device = builder.build()

    p0 = cql.compiler.import_cudaq(memory, args.rounds)
    p1 = cql.compiler.place(p0, device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    p3 = cql.compile(p2,
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)
    model = cql.qec.generate_dem(p3,
                                 target=analysis_target,
                                 noise_model=noise,
                                 decompose_errors=False)
    config = cql.qec.generate_decoder_config(p3, model=model)
    args.output.write_text(config.to_yaml_str(200))
    print(
        f"Wrote {args.output}; execution decoder IDs: {dict(model.decoder_ids)}"
    )
