/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file cudaqx_decoder_hostcall.h
/// @brief Out-of-tree shim that presents the CUDA-QX realtime decoders as
///        CUDA-Q realtime HOST_CALL handlers.
///
/// This is a thin adapter built *on top of the cuda-qx wheel* (it links
/// `libcudaq-qec-realtime-decoding.so` and is otherwise standalone). It exposes
/// the realtime decode verbs -- reset / enqueue / get -- plus a `configure_decoder`
/// control verb as `CUDAQ_DISPATCH_HOST_CALL` entries that a realtime server app
/// installs into its dispatcher function table. The decode verbs forward to the
/// bank's decoder-agnostic host API, so it works for *every* decoder type the
/// wheel ships (nv-qldpc and others), not just one.
///
/// The server is configured entirely over the wire: send a `configure_decoder`
/// RPC (one chunk frame per payload) to (re)build the bank at any time. There
/// is no host-side config call -- an always-on server starts empty and is
/// configured (and live-reconfigured) by the network layer.
///
/// Usage (server app):
/// @code{.c}
///   cudaq_function_entry_t e[CUDAQX_RT_FUNCTION_COUNT];
///   uint32_t n = 0;
///   cudaqx_rt_get_function_table(e, CUDAQX_RT_FUNCTION_COUNT, &n);
///   // merge e[0..n) into the dispatcher's function table; size ring slots to
///   // at least CUDAQX_RT_FRAME_SIZE.
///   // ... serve device_call RPCs on CUDAQ_DISPATCH_PATH_HOST; the first
///   //     configure_decoder RPC builds the bank, decode RPCs follow ...
///   cudaqx_rt_finalize();
/// @endcode
///
/// Requires at runtime: the cuda-qx wheel libs (libcudaq-qec-realtime-decoding,
/// libcudaq-qec, decoder plugins) and a CUDA-Q realtime runtime that dispatches
/// HOST_CALL entries (cuda-quantum PR #4652).

#pragma once

#include <stdint.h>

// cudaq_function_entry_t / cudaq_host_rpc_fn_t (the realtime dispatch ABI).
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#if defined(CUDAQX_RT_BUILD) && !defined(_WIN32)
#define CUDAQX_RT_API __attribute__((visibility("default")))
#else
#define CUDAQX_RT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Status codes. `CUDAQX_RT_OK` is zero; any other value is a failure.
enum {
  CUDAQX_RT_OK = 0,
  CUDAQX_RT_ERR_INVALID_ARG = 1,
  CUDAQX_RT_ERR_CONFIG = 2,
  CUDAQX_RT_ERR_RUNTIME = 3
};

/// Function IDs the handlers register under -- `cudaq::realtime::fnv1a_hash` of
/// the realtime decoder wire function names emitted by `device_call`, so the
/// dispatcher routes incoming RPCs to them by id.
enum {
  CUDAQX_RT_RESET_DECODER_FUNCTION_ID = 0x8a7d3ef8,     // reset_decoder_ui64
  CUDAQX_RT_ENQUEUE_SYNDROMES_FUNCTION_ID = 0xee9c576f, // enqueue_syndromes_ui64
  CUDAQX_RT_GET_CORRECTIONS_FUNCTION_ID = 0xe138287e,   // get_corrections_ui64
  // Shim-defined CONTROL verb (not a qec decode wire function): (re)build the
  // decoder bank from chunked YAML frames carried in RPC payloads. Lets an
  // always-on server be reconfigured over the network without restarting.
  // fnv1a_hash("configure_decoder").
  CUDAQX_RT_CONFIGURE_DECODER_FUNCTION_ID = 0x00a0a48c,
  CUDAQX_RT_FUNCTION_COUNT = 4
};

/// Config-frame sizing. `configure_decoder` accepts exactly one chunk frame per
/// RPC payload:
///
///   [magic][version][total_bytes][offset_bytes][chunk_bytes][flags][bytes...]
///
/// The function id remains `CUDAQX_RT_CONFIGURE_DECODER_FUNCTION_ID`; chunking
/// is a payload ABI detail owned by the configure handler. The handler
/// reassembles chunks and calls cuda-qec only when the final chunk arrives.
/// `magic` is the uint32 value `0x43585143`, which appears on the little-endian
/// wire as bytes `43 51 58 43` (`"CQXC"`). The shim uses it as a cheap frame
/// marker before interpreting the payload as the chunked configure ABI.
///
/// We observed the real QUA external-stream path fail around the 4 KiB packet
/// size boundary on this setup. The 1024 byte chunk size below is only a
/// conservative value we picked to stay comfortably under that boundary; it is
/// not a semantic decoder limit. The assembled YAML is separately capped by
/// `CUDAQX_RT_CONFIGURE_MAX_BYTES`.
enum {
  CUDAQX_RT_RPC_HEADER_BYTES = 24,              // sizeof(RPCHeader)/RPCResponse
  CUDAQX_RT_CONFIGURE_MAX_BYTES = 256 * 1024,   // max assembled YAML payload
  CUDAQX_RT_CONFIGURE_CHUNK_BYTES = 1024,       // max YAML bytes per chunk
  CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES = 24,  // 6 x uint32_t
  CUDAQX_RT_CONFIGURE_CHUNK_FRAME_BYTES =
      CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES + CUDAQX_RT_CONFIGURE_CHUNK_BYTES,
  CUDAQX_RT_FRAME_SIZE =
      CUDAQX_RT_RPC_HEADER_BYTES + CUDAQX_RT_CONFIGURE_CHUNK_FRAME_BYTES,
  CUDAQX_RT_CONFIGURE_CHUNK_MAGIC = 0x43585143, // bytes "CQXC" on LE wire
  CUDAQX_RT_CONFIGURE_CHUNK_VERSION = 1,
  CUDAQX_RT_CONFIGURE_CHUNK_BEGIN = 1,
  CUDAQX_RT_CONFIGURE_CHUNK_END = 2
};

/// Fill `entries[0..CUDAQX_RT_FUNCTION_COUNT)` with the HOST_CALL function
/// table: the three decode verbs (reset / enqueue / get) plus the
/// `configure_decoder` control verb. Each entry has `dispatch_mode =
/// CUDAQ_DISPATCH_HOST_CALL`, a `handler.host_fn`, the matching `function_id`,
/// and a minimal `schema`. `capacity` must be >= CUDAQX_RT_FUNCTION_COUNT.
///
/// `configure_decoder` carries one chunked payload frame. Frames are assembled
/// inside the shim and only the final chunk rebuilds the bank via
/// `config::finalize_decoders` + `config::configure_decoders_from_str`. Because
/// HOST_CALL handlers run synchronously on the single dispatcher thread, a
/// reconfigure is serialized against decode RPCs.
CUDAQX_RT_API int cudaqx_rt_get_function_table(cudaq_function_entry_t *entries,
                                               uint32_t capacity,
                                               uint32_t *count);

/// Tear down the decoder bank. Forwards to `config::finalize_decoders`.
CUDAQX_RT_API void cudaqx_rt_finalize(void);

#ifdef __cplusplus
}
#endif
