/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Starting from C++, mimic a value-semantics target pipeline: convert to wires,
// selectively unroll only loops that block wire conversion, then assign wire-set
// indices so the explicit quake.borrow_wire values are visible.
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --factor-quantum-alloc --memtoreg --cc-loop-unroll=keep-value-semantic-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | FileCheck %s
// clang-format on

#include <cudaq.h>

// `repeat` applies the same gates to the same qubits each iteration (a clean
// wire-iter-arg loop after mem2reg) so it stays rolled, carrying its qubits as
// quake.borrow_wire loop iter-args. `q[i]` indexes by the induction variable
// (reference semantics after mem2reg) so it is unrolled.
__qpu__ bool kernel() {
  cudaq::qubit a, b;
  for (int r = 0; r < 5; r++) {
    h(a);
    x<cudaq::ctrl>(a, b);
  }
  cudaq::qvector q(3);
  for (int i = 0; i < 3; i++) {
    h(q[i]);
  }
  return mz(a) ^ mz(b) ^ mz(q[0]) ^ mz(q[1]) ^ mz(q[2]);
}

// CHECK-LABEL:   quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernelv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[BORROW_WIRE_0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[BORROW_WIRE_1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[BORROW_WIRE_2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK:           %[[BORROW_WIRE_3:.*]] = quake.borrow_wire @wires[3] : !quake.wire
// CHECK:           %[[BORROW_WIRE_4:.*]] = quake.borrow_wire @wires[4] : !quake.wire
// CHECK:           %[[LOOP_0:.*]]:3 = cc.loop while ((%[[VAL_0:.*]] = %[[BORROW_WIRE_3]], %[[VAL_1:.*]] = %[[BORROW_WIRE_4]], %[[VAL_2:.*]] = %[[CONSTANT_0]]) -> (!quake.wire, !quake.wire, i32)) {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_2]], %[[CONSTANT_1]] : i32
// CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : !quake.wire, !quake.wire, i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !quake.wire, %[[VAL_4:.*]]: !quake.wire, %[[VAL_5:.*]]: i32):
// CHECK:             %[[H_0:.*]] = quake.h %[[VAL_3]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[X_0:.*]]:2 = quake.x {{\[}}%[[H_0]]] %[[VAL_4]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:             cc.continue %[[X_0]]#0, %[[X_0]]#1, %[[VAL_5]] : !quake.wire, !quake.wire, i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_6:.*]]: !quake.wire, %[[VAL_7:.*]]: !quake.wire, %[[VAL_8:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_2]] : i32
// CHECK:             cc.continue %[[VAL_6]], %[[VAL_7]], %[[ADDI_0]] : !quake.wire, !quake.wire, i32
// CHECK:           }
// CHECK:           %[[H_1:.*]] = quake.h %[[BORROW_WIRE_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_2:.*]] = quake.h %[[BORROW_WIRE_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_3:.*]] = quake.h %[[BORROW_WIRE_2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_9:.*]], %[[MZ_0:.*]] = quake.mz %[[VAL_10:.*]]#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_9]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_0:.*]] = cc.cast unsigned %[[DISCRIMINATE_0]] : (i1) -> i32
// CHECK:           %[[VAL_11:.*]], %[[MZ_1:.*]] = quake.mz %[[VAL_10]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[VAL_11]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_1:.*]] = cc.cast unsigned %[[DISCRIMINATE_1]] : (i1) -> i32
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[CAST_0]], %[[CAST_1]] : i32
// CHECK:           %[[VAL_12:.*]], %[[MZ_2:.*]] = quake.mz %[[H_1]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_2:.*]] = quake.discriminate %[[VAL_12]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_2:.*]] = cc.cast unsigned %[[DISCRIMINATE_2]] : (i1) -> i32
// CHECK:           %[[XORI_1:.*]] = arith.xori %[[XORI_0]], %[[CAST_2]] : i32
// CHECK:           %[[VAL_13:.*]], %[[MZ_3:.*]] = quake.mz %[[H_2]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_3:.*]] = quake.discriminate %[[VAL_13]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_3:.*]] = cc.cast unsigned %[[DISCRIMINATE_3]] : (i1) -> i32
// CHECK:           %[[XORI_2:.*]] = arith.xori %[[XORI_1]], %[[CAST_3]] : i32
// CHECK:           %[[VAL_14:.*]], %[[MZ_4:.*]] = quake.mz %[[H_3]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_4:.*]] = quake.discriminate %[[VAL_14]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_4:.*]] = cc.cast unsigned %[[DISCRIMINATE_4]] : (i1) -> i32
// CHECK:           %[[XORI_3:.*]] = arith.xori %[[XORI_2]], %[[CAST_4]] : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi ne, %[[XORI_3]], %[[CONSTANT_0]] : i32
// CHECK:           quake.return_wire %[[MZ_2]] : !quake.wire
// CHECK:           quake.return_wire %[[MZ_3]] : !quake.wire
// CHECK:           quake.return_wire %[[MZ_4]] : !quake.wire
// CHECK:           return %[[CMPI_1]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @_Z6kernelv() -> i1 attributes {no_this} {
// CHECK:           %[[UNDEF_0:.*]] = cc.undef i1
// CHECK:           return %[[UNDEF_0]] : i1
// CHECK:         }
