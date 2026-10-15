// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake. --implicit-check-not=qlx.resource_request --implicit-check-not=qlx.consume_resource

module {
  func.func @__nvqpp__mlirgen__idle_loop() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q0 = quake.null_wire
    %loop:2 = cc.loop while (
        (%w = %q0, %iv = %c0) -> (!quake.wire, i32)) {
      // CUDA-Q's normalized counted-loop form may retain a forward signed
      // comparison rather than rewriting it to `ne`.
      %go = arith.cmpi slt, %iv, %c5 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      %t = quake.idle %w : (!quake.wire) -> !quake.wire
      cc.continue %t, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %n = arith.addi %iv, %c1 : i32
      cc.continue %w, %n : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @idle_loop
// CHECK: cflow.repeat
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[NEXT:.*]] = qlx.idle %{{.*}} rounds %[[ONE]]
// CHECK: cflow.yield %[[NEXT]]
