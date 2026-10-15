// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics


module {
  // expected-error @+1 {{logical observable indices must be unique and contiguous from zero}}
  fabric.gadget_profile @noncontiguous_observables for @g {
    fabric.observable() {idx = 1 : i64, records = ["g.m0"]}
  }
}

// -----

module {
  fabric.gadget @g() {
  }
  fabric.gadget_profile @missing_record for @g {
    // expected-error @+1 {{is not an exact output of a typed record-producing operation}}
    fabric.detector() {records = ["g.m0"]}
  }
}

// -----

module {
  fabric.gadget @duplicate_scalar() {
    %record = "arith.constant"() {record = "m0", value = false} : () -> i1
  }
  fabric.gadget_profile @duplicate_scalar_profile for @duplicate_scalar {
    // expected-error @+1 {{scalar profile record support must be duplicate-free}}
    fabric.detector() {
      records = ["duplicate_scalar.m0", "duplicate_scalar.m0"]
    }
  }
}

// -----

module {
  fabric.gadget @duplicate_input_syndrome() {}
  fabric.gadget_profile @duplicate_input_profile for @duplicate_input_syndrome {
    // expected-error @+1 {{scalar profile input-syndrome support must be duplicate-free}}
    fabric.detector() {
      input_syndromes = [
        {index = 0 : i64, port_index = 0 : i64},
        {index = 0 : i64, port_index = 0 : i64}
      ]
    }
  }
}

// -----

module {
  fabric.gadget_spec @spec for @objective
      : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1) {
    encodings = [], record_schema = ["a.outcome", "b.outcome"],
    outcome_map = {
      records = ["a.outcome", "b.outcome"],
      rows = dense<[[1, 1]]> : tensor<1x2xi1>,
      constants = array<i64: 0>, input_syndromes = [[]],
      roles = [["observable"]]
    }
  }
  fabric.gadget @g(%patch: !fabric.patch<@code>)
      -> (!fabric.patch<@code>, i1) {
    %p0, %a = fabric.measure_product %patch {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "a"
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    %p1, %b = fabric.measure_product %p0 {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "X", record = "b"
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    %outcome = fabric.xor %a, %b : i1
    fabric.return %p1, %outcome : !fabric.patch<@code>, i1
  } {spec = @spec}
  // expected-error @+1 {{profile observable row 0 disagrees with linked GadgetSpec outcome_map row 0}}
  fabric.gadget_profile @contradictory_observable for @g {
    fabric.observable() {
      idx = 0 : i64, records = ["g.b.outcome", "g.a.outcome"]
    }
  }
}
