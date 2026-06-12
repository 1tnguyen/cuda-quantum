/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file decoder_packets.h
/// @brief OPNIC wire packets for the realtime decoder server's DATA stream.
///
/// The decoder server now uses a single stream for all decode RPCs
/// (reset_decoder / enqueue_syndromes / get_corrections). Decoder
/// configuration happens host-side before any QUA program runs, so there is
/// no control stream and no configure-chunk packets.
///
/// The 24-byte packet header is byte-identical to RPCHeader / RPCResponse
/// (6 × uint32_t), so slots dispatch directly with no per-word transcoding.
///
/// The largest decode frame is enqueue: header + 4 × u64 = 24 + 32 = 56 B.
/// A 64 B slot (one cache line) holds it with a little headroom.
///
/// These structures must match the QUA-side packet definitions so the OPX
/// handshake signature validates. The QUA program declares one data stream.

// Must include <array> before opnic SDK headers for nvcc compatibility
// (SDK's signature.hpp uses std::array without including the header).
#include <array>

#include <opnic/stream/packet_definitions.hpp>
#include <opnic/stream/value.hpp>

/// Data-stream payload: 10 words (40 B) holds the largest decode frame
/// (enqueue = 4 × u64 = 32 B) with headroom; total slot = 6 + 10 = 16
/// words = 64 B.
inline constexpr int DECODER_DATA_PAYLOAD_WORDS = 10;

//==============================================================================
// DATA stream: decode verbs.
//   DataInputPacket:  [magic][function_id][arg_len][request_id][ptp:2][args:10]
//   DataOutputPacket: [magic][status][result_len][request_id][ptp:2][result:10]
//==============================================================================

struct DataInputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> function_id;
  qm::Value<int, 1> arg_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, DECODER_DATA_PAYLOAD_WORDS> args; ///< ui64-packed decode args

  QM_DECLARE_PACKET(DataInputPacket, magic, function_id, arg_len, request_id,
                    ptp_timestamp, args);
};

struct DataOutputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> status;
  qm::Value<int, 1> result_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, DECODER_DATA_PAYLOAD_WORDS> result; ///< ui64 corrections

  QM_DECLARE_PACKET(DataOutputPacket, magic, status, result_len, request_id,
                    ptp_timestamp, result);
};
