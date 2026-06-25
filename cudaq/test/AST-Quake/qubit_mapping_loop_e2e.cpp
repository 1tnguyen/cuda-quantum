/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// End-to-end from C++: a repeating loop is kept rolled in value semantics
// (keep-value-semantic-loops), then mapped to a star topology. The loop stays
// rolled; the non-adjacent CNOT is routed inside the body and the layout is
// restored at the back-edge.

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --factor-quantum-alloc --memtoreg --cc-loop-unroll=keep-value-semantic-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | cudaq-opt --cc-loop-unroll --canonicalize > %t.orig.mlir
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --factor-quantum-alloc --memtoreg --cc-loop-unroll=keep-value-semantic-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | cudaq-opt '--qubit-mapping=device=star(3)' | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --factor-quantum-alloc --memtoreg --cc-loop-unroll=keep-value-semantic-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | cudaq-opt '--qubit-mapping=device=star(3)' | cudaq-opt --cc-loop-unroll --canonicalize | CircuitCheck --up-to-mapping %t.orig.mlir
// clang-format on

#include <cudaq.h>

__qpu__ bool kernel() {
  cudaq::qubit a, b, c;
  for (int r = 0; r < 4; r++) {
    x<cudaq::ctrl>(a, b); // adjacent on star (center a)
    x<cudaq::ctrl>(b, c); // b-c not adjacent -> routed inside the loop body
  }
  return mz(a) ^ mz(b) ^ mz(c);
}

// CHECK-LABEL:   quake.wire_set @mapped_wireset[3] adjacency sparse<{{\[\[}}0, 2], [0, 1], [1, 0], [2, 0]], true> : tensor<3x3xi1> attributes {sym_visibility = "private"}
// CHECK:         quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernelv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", mapping_v2p = [0, 1, 2], no_this} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[BORROW_WIRE_0:.*]] = quake.borrow_wire @mapped_wireset[0] : !quake.wire
// CHECK:           %[[BORROW_WIRE_1:.*]] = quake.borrow_wire @mapped_wireset[1] : !quake.wire
// CHECK:           %[[BORROW_WIRE_2:.*]] = quake.borrow_wire @mapped_wireset[2] : !quake.wire
// CHECK:           %[[LOOP_0:.*]]:4 = cc.loop while ((%[[VAL_0:.*]] = %[[BORROW_WIRE_0]], %[[VAL_1:.*]] = %[[BORROW_WIRE_1]], %[[VAL_2:.*]] = %[[BORROW_WIRE_2]], %[[VAL_3:.*]] = %[[CONSTANT_2]]) -> (!quake.wire, !quake.wire, !quake.wire, i32)) {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_3]], %[[CONSTANT_1]] : i32
// CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !quake.wire, !quake.wire, !quake.wire, i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.wire, %[[VAL_5:.*]]: !quake.wire, %[[VAL_6:.*]]: !quake.wire, %[[VAL_7:.*]]: i32):
// CHECK:             %[[X_0:.*]]:2 = quake.x {{\[}}%[[VAL_4]]] %[[VAL_5]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:             %[[SWAP_0:.*]]:2 = quake.swap %[[X_0]]#1, %[[X_0]]#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:             %[[X_1:.*]]:2 = quake.x {{\[}}%[[SWAP_0]]#1] %[[VAL_6]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:             %[[SWAP_1:.*]]:2 = quake.swap %[[SWAP_0]]#0, %[[X_1]]#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:             cc.continue %[[SWAP_1]]#1, %[[SWAP_1]]#0, %[[X_1]]#1, %[[VAL_7]] : !quake.wire, !quake.wire, !quake.wire, i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: !quake.wire, %[[VAL_9:.*]]: !quake.wire, %[[VAL_10:.*]]: !quake.wire, %[[VAL_11:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_11]], %[[CONSTANT_0]] : i32
// CHECK:             cc.continue %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[ADDI_0]] : !quake.wire, !quake.wire, !quake.wire, i32
// CHECK:           }
// CHECK:           %[[VAL_12:.*]], %[[MZ_0:.*]] = quake.mz %[[VAL_13:.*]]#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_12]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_0:.*]] = cc.cast unsigned %[[DISCRIMINATE_0]] : (i1) -> i32
// CHECK:           %[[VAL_14:.*]], %[[MZ_1:.*]] = quake.mz %[[VAL_13]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[VAL_14]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_1:.*]] = cc.cast unsigned %[[DISCRIMINATE_1]] : (i1) -> i32
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[CAST_0]], %[[CAST_1]] : i32
// CHECK:           %[[VAL_15:.*]], %[[MZ_2:.*]] = quake.mz %[[VAL_13]]#2 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_2:.*]] = quake.discriminate %[[VAL_15]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_2:.*]] = cc.cast unsigned %[[DISCRIMINATE_2]] : (i1) -> i32
// CHECK:           %[[XORI_1:.*]] = arith.xori %[[XORI_0]], %[[CAST_2]] : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi ne, %[[XORI_1]], %[[CONSTANT_2]] : i32
// CHECK:           return %[[CMPI_1]] : i1
// CHECK:         }
// CHECK-LABEL:   func.func @_Z6kernelv() -> i1 attributes {no_this} {
// CHECK:           %[[UNDEF_0:.*]] = cc.undef i1
// CHECK:           return %[[UNDEF_0]] : i1
// CHECK:         }
