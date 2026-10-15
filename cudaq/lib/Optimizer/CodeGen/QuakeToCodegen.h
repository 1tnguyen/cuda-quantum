/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace mlir {
class Operation;
class ConversionTarget;
class RewritePatternSet;
} // namespace mlir

namespace cudaq::codegen {
/// Drop logical idle markers at the physical code-generation boundary,
/// forwarding wire results. Do not call this during shared optimization or
/// logical import.
void eraseIdleOperations(mlir::Operation *);
void populateQuakeToCodegenPatterns(mlir::RewritePatternSet &);
void setQuakeToCodegenLegality(mlir::ConversionTarget &);
} // namespace cudaq::codegen
