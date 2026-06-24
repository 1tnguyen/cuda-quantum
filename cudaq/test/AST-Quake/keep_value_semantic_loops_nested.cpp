/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Starting from C++, mimic a value-semantics target pipeline and check that the
// two inner index-dependent loops are unrolled while the purely-repeating outer
// loop stays rolled.
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --factor-quantum-alloc --memtoreg --cc-loop-unroll=keep-value-semantic-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | FileCheck %s
// clang-format on

#include <cudaq.h>

// The outer `r` loop is purely repeating (r is not used to index qubits), so it
// stays rolled, carrying its qubits as quake.borrow_wire loop iter-args. The
// middle `i` loop and the innermost `j` loop both index qubits by their
// induction variable, so they are unrolled. After unrolling, the outer loop body
// is a flat, constant-index gate sequence that mem2reg threads as wires.
__qpu__ bool kernel() {
  cudaq::qvector q(3);
  for (int r = 0; r < 4; r++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        h(q[j]);
      }
      rx(0.7, q[i]);
    }
  }
  return mz(q[0]) ^ mz(q[1]) ^ mz(q[2]);
}

// CHECK-LABEL:   quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernelv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.69999999999999996 : f64
// CHECK:           %[[BORROW_WIRE_0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[BORROW_WIRE_1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[BORROW_WIRE_2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK:           %[[LOOP_0:.*]]:4 = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_0]], %[[VAL_1:.*]] = %[[BORROW_WIRE_0]], %[[VAL_2:.*]] = %[[BORROW_WIRE_1]], %[[VAL_3:.*]] = %[[BORROW_WIRE_2]]) -> (i32, !quake.wire, !quake.wire, !quake.wire)) {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_1]] : i32
// CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, !quake.wire, !quake.wire, !quake.wire)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: !quake.wire, %[[VAL_6:.*]]: !quake.wire, %[[VAL_7:.*]]: !quake.wire):
// CHECK:             %[[H_0:.*]] = quake.h %[[VAL_5]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_1:.*]] = quake.h %[[VAL_6]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_2:.*]] = quake.h %[[VAL_7]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[RX_0:.*]] = quake.rx (%[[CONSTANT_3]]) %[[H_0]] : (f64, !quake.wire) -> !quake.wire
// CHECK:             %[[H_3:.*]] = quake.h %[[RX_0]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_4:.*]] = quake.h %[[H_1]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_5:.*]] = quake.h %[[H_2]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[RX_1:.*]] = quake.rx (%[[CONSTANT_3]]) %[[H_4]] : (f64, !quake.wire) -> !quake.wire
// CHECK:             %[[H_6:.*]] = quake.h %[[H_3]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_7:.*]] = quake.h %[[RX_1]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[H_8:.*]] = quake.h %[[H_5]] : (!quake.wire) -> !quake.wire
// CHECK:             %[[RX_2:.*]] = quake.rx (%[[CONSTANT_3]]) %[[H_8]] : (f64, !quake.wire) -> !quake.wire
// CHECK:             cc.continue %[[VAL_4]], %[[H_6]], %[[H_7]], %[[RX_2]] : i32, !quake.wire, !quake.wire, !quake.wire
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: !quake.wire, %[[VAL_10:.*]]: !quake.wire, %[[VAL_11:.*]]: !quake.wire):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_2]] : i32
// CHECK:             cc.continue %[[ADDI_0]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : i32, !quake.wire, !quake.wire, !quake.wire
// CHECK:           }
// CHECK:           %[[VAL_12:.*]], %[[MZ_0:.*]] = quake.mz %[[VAL_13:.*]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_12]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_0:.*]] = cc.cast unsigned %[[DISCRIMINATE_0]] : (i1) -> i32
// CHECK:           %[[VAL_14:.*]], %[[MZ_1:.*]] = quake.mz %[[VAL_13]]#2 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[VAL_14]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_1:.*]] = cc.cast unsigned %[[DISCRIMINATE_1]] : (i1) -> i32
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[CAST_0]], %[[CAST_1]] : i32
// CHECK:           %[[VAL_15:.*]], %[[MZ_2:.*]] = quake.mz %[[VAL_13]]#3 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[DISCRIMINATE_2:.*]] = quake.discriminate %[[VAL_15]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CAST_2:.*]] = cc.cast unsigned %[[DISCRIMINATE_2]] : (i1) -> i32
// CHECK:           %[[XORI_1:.*]] = arith.xori %[[XORI_0]], %[[CAST_2]] : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi ne, %[[XORI_1]], %[[CONSTANT_0]] : i32
// CHECK:           quake.return_wire %[[MZ_0]] : !quake.wire
// CHECK:           quake.return_wire %[[MZ_1]] : !quake.wire
// CHECK:           quake.return_wire %[[MZ_2]] : !quake.wire
// CHECK:           return %[[CMPI_1]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @_Z6kernelv() -> i1 attributes {no_this} {
// CHECK:           %[[UNDEF_0:.*]] = cc.undef i1
// CHECK:           return %[[UNDEF_0]] : i1
// CHECK:         }
