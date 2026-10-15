// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | qlx-opt | FileCheck %s --check-prefix=P2
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device})' | qlx-opt | FileCheck %s --check-prefix=P3

// Exercise decoder dataflow and detached analysis together. The decoder name
// is an external binding; this test does not need a decoder runtime or DEM.
module attributes {qlx.profiles = ["p2n"]} {
  phys.action @x {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22x\22,\22parameters\22:{}}"
  }
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, k = 1 : i64, n = 2 : i64, r = 0 : i64,
    partitions = {data = 2 : i64},
    lx = [array<i64: 0>], lz = [array<i64: 0>]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.gadget @readout(%patch: !fabric.patch<@code>)
      -> (!fabric.patch<@code>, i1) {
    %measured, %bits = fabric.mz %patch data [0, 1] {record = "data"}
      : !fabric.patch<@code> -> tensor<2xi1>
    fabric.submit_syndrome %bits {
      decoder = @external_decoder, round = 0 : i64,
      decoder_erasure_semantics = "none"
    } : tensor<2xi1>
    %parity = fabric.parity %bits : (tensor<2xi1>) -> i1
    fabric.submit_syndrome scalars(%parity : i1) {
      decoder = @external_decoder, round = 1 : i64
    }
    %decoded = fabric.decode_bit %bits {
      decoder = @external_decoder, parity_indices = array<i64: 1>,
      invert = true
    } : tensor<2xi1> -> i1
    %corrected = cflow.if %decoded -> !fabric.patch<@code> {
      %flipped = fabric.x %measured data : !fabric.patch<@code>
      cflow.yield %flipped : !fabric.patch<@code>
    } else {
      cflow.yield %measured : !fabric.patch<@code>
    }
    fabric.return %corrected, %decoded : !fabric.patch<@code>, i1
  }
  fabric.gadget_profile @analysis for @readout {
    fabric.detector() {records = ["readout.data.data0", "readout.data.data1"]}
    // Deliberately declare observables out of idx order.
    fabric.observable() {idx = 1 : i64, records = ["readout.data.data1"]}
    fabric.observable() {idx = 0 : i64, records = ["readout.data.data0"]}
  }
  fabric.protocol @p2 : () -> i1 {
    %patch = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %out, %decoded = fabric.call @readout(%patch) {profile = @analysis}
      : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    fabric.dealloc %out : !fabric.patch<@code>
    fabric.protocol_return %decoded : i1
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = [@x]
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@qubits]
    }
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["qubits"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical
  }
}

// P2: fabric.submit_syndrome %{{.*}} {decoder = @external_decoder, decoder_erasure_semantics = "none", round = 0 : i64} : tensor<2xi1>
// P2: fabric.submit_syndrome scalars(%{{.*}} : i1) {decoder = @external_decoder, round = 1 : i64}
// P2: fabric.decode_bit

// P3-LABEL: phys.graph @p2_physical
// P3: %[[CALL:.*]]:3 = "phys.call"
// P3: %{{.*}}, %[[M0:.*]] = phys.measure
// P3-SAME: record_id = "p2_physical.data.0"
// P3: %{{.*}}, %[[M1:.*]] = phys.measure
// P3-SAME: record_id = "p2_physical.data.1"
// P3: phys.submit_syndrome %[[M0]], %[[M1]]
// P3-SAME: decoder = @external_decoder
// P3-SAME: decoder_erasure_semantics = "none"
// P3-SAME: event_id = "submit_syndrome{{[0-9]+}}"
// P3-SAME: round = 0 : i64
// P3: %[[PARITY:.*]] = phys.xor
// P3: phys.submit_syndrome %[[PARITY]]
// P3-SAME: round = 1 : i64
// P3: %[[DECODED:.*]] = phys.decode_bit %[[M0]], %[[M1]]
// P3-SAME: decoder = @external_decoder
// P3-SAME: invert = true
// P3-SAME: parity_indices = array<i64: 1>
// P3: cflow.if %[[DECODED]]
// P3: phys.apply @x
// P3: phys.yield %{{.*}}, %{{.*}}, %[[DECODED]]
// P3: phys.return %[[CALL]]#2
// P3: phys.detector_sidecar
// P3-SAME: records = ["p2_physical.data.0", "p2_physical.data.1"]
// P3-SAME: source_profile = @analysis
// P3: phys.observable_sidecar
// P3-SAME: idx = 0 : i64
// P3-SAME: records = ["p2_physical.data.0"]
// P3-SAME: source_row = 0 : i64
// P3: phys.observable_sidecar
// P3-SAME: idx = 1 : i64
// P3-SAME: records = ["p2_physical.data.1"]
// P3-SAME: source_row = 1 : i64
// P3: phys.record_projection
