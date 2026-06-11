#!/bin/bash
# ============================================================================#
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/hsb_test_cpu.sh" --bridge-binary hsb_bridge_cpu_bridge "$@"
