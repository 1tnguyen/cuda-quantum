/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

// Must include <array> before opnic SDK headers for nvcc compatibility
// (SDK's signature.hpp uses std::array without including the header)
#include <array>

#include <opnic/stream/packet_definitions.hpp>
#include <opnic/stream/value.hpp>

//==============================================================================
// OPNIC Packet Definitions
//
// These packet structures must be identical to the QUA-side definitions.
// The QM_DECLARE_PACKET macro generates serialize/deserialize methods and
// the static `pattern` string used for OPX handshake signature validation.
//
// In the zero-copy direct path, we don't create IncomingStream/OutgoingStream
// objects, but we still need these packet definitions for:
//   1. Calculating allocation_size (packet stride in OPNIC buffers)
//   2. Generating sync patterns for OPX handshake
//   3. Documenting the wire format
//
// The header region is byte-identical to the dispatcher's `RPCHeader` /
// `RPCResponse` structs in `dispatch_kernel_launch.h` (24 B = 6 uint32_t
// words), so the bridge can `reinterpret_cast` slot bytes directly to those
// structs.  `request_id` and `ptp_timestamp` are real fields on the wire
// (incrementing per call on the QUA side, echoed by the host) -- the bridge
// no longer synthesizes zeros for them.
//
// Wire format (serialized uint32_t array):
//   RPCInputPacket:  [magic:1][function_id:1][arg_len:1][request_id:1]
//                    [ptp_timestamp:2][data:1] = 7 words
//   RPCOutputPacket: [magic:1][status:1][result_len:1][request_id:1]
//                    [ptp_timestamp:2][result:1] = 7 words
//==============================================================================

struct RPCInputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> function_id;
  qm::Value<int, 1> arg_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, 1> data;

  QM_DECLARE_PACKET(RPCInputPacket, magic, function_id, arg_len, request_id,
                    ptp_timestamp, data);
};

struct RPCOutputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> status;
  qm::Value<int, 1> result_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, 1> result;

  QM_DECLARE_PACKET(RPCOutputPacket, magic, status, result_len, request_id,
                    ptp_timestamp, result);
};
