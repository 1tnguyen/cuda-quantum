/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file decoder_packets.h
/// @brief OPNIC wire packets for the realtime decoder server's TWO streams.
///
/// The decoder server separates a latency-critical **data plane** from a bulky
/// **control plane**, each on its own OPNIC stream with its own packet size
/// (see decoder_server_cpu.cpp for how both are serviced by one poll thread):
///
///   DATA stream  -- the hot path. Carries the decode verbs
///                   (reset_decoder / enqueue_syndromes / get_corrections),
///                   whose arguments are packed `uint64_t` words (the
///                   `device_call` ui64 ABI the shim consumes). The largest
///                   decode frame is enqueue: header + 4 x u64 = 24 + 32 = 56 B.
///                   We use a 64 B slot (one cache line) -- the smallest clean
///                   size that holds that frame with a little headroom, so the
///                   data path stays minimal.
///
///   CONTROL stream -- rare, off the hot path. Carries `configure_decoder`,
///                   whose payload is chunked as:
///                   [chunk metadata][<= CUDAQX_RT_CONFIGURE_CHUNK_BYTES YAML].
///                   We observed QUA/OPNIC external-stream packets fail around
///                   the 4 KiB boundary on this setup. The current 1024 byte
///                   chunk payload is just a conservative value we picked; the
///                   protocol only requires that each chunk fit the transport.
///                   The shim reassembles the full YAML before calling QEC.
///
/// Splitting the two means every decode RPC DMAs only 64 B. The 24 B header of
/// every packet is byte-identical to the
/// dispatcher's `RPCHeader` / `RPCResponse` (6 x uint32_t), so a slot can be
/// dispatched directly with no per-word transcoding.
///
/// These structures must match the QUA-side packet definitions for the OPX
/// handshake signature to validate (`QM_DECLARE_PACKET` derives the `pattern`
/// from the field names/widths). The QUA program declares two streams to match:
/// a small data stream and a chunk-sized control stream.

// Must include <array> before opnic SDK headers for nvcc compatibility
// (SDK's signature.hpp uses std::array without including the header).
#include <array>

#include <opnic/stream/packet_definitions.hpp>
#include <opnic/stream/value.hpp>

// Shim contract: configure chunks are reassembled in the configure handler.
#include "cudaqx_decoder_hostcall.h"

//==============================================================================
// Wire word counts. The 6-word (24 B) header mirrors RPCHeader/RPCResponse.
//==============================================================================

/// Data-stream payload: 10 words (40 B) holds the largest decode frame (enqueue
/// = 4 x u64 = 32 B) with headroom; total slot = 6 + 10 = 16 words = 64 B.
inline constexpr int DECODER_DATA_PAYLOAD_WORDS = 10;

/// Control-stream payload: one configure chunk header plus up to
/// CUDAQX_RT_CONFIGURE_CHUNK_BYTES YAML bytes. With the current arbitrary
/// 1024 B chunk size this is 262 words, comfortably below the 4096 B packet
/// boundary that failed on real OPNIC/QUA.
inline constexpr int DECODER_CONTROL_PAYLOAD_WORDS =
    (CUDAQX_RT_CONFIGURE_CHUNK_FRAME_BYTES + sizeof(int) - 1) / sizeof(int);

//==============================================================================
// DATA stream (hot path): decode verbs.
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
  qm::Value<int, DECODER_DATA_PAYLOAD_WORDS> result; ///< ui64-packed corrections

  QM_DECLARE_PACKET(DataOutputPacket, magic, status, result_len, request_id,
                    ptp_timestamp, result);
};

//==============================================================================
// CONTROL stream (off hot path): chunked configure_decoder.
//   ControlInputPacket:  header + [config:262]  (chunk metadata + YAML bytes)
//   ControlOutputPacket: header + [config:262]  (only the 24 B status is used)
// Both directions share the footprint so the slot strides match and the handler
// can rewrite the (copied) request slot in place as the RPCResponse.
//==============================================================================

struct ControlInputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> function_id;
  qm::Value<int, 1> arg_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, DECODER_CONTROL_PAYLOAD_WORDS> config; ///< configure chunk

  QM_DECLARE_PACKET(ControlInputPacket, magic, function_id, arg_len, request_id,
                    ptp_timestamp, config);
};

struct ControlOutputPacket {
  qm::Value<int, 1> magic;
  qm::Value<int, 1> status;
  qm::Value<int, 1> result_len;
  qm::Value<int, 1> request_id;
  qm::Value<int, 2> ptp_timestamp;
  qm::Value<int, DECODER_CONTROL_PAYLOAD_WORDS> config;

  QM_DECLARE_PACKET(ControlOutputPacket, magic, status, result_len, request_id,
                    ptp_timestamp, config);
};
