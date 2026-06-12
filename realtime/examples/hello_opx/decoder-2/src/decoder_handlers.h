/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file decoder_handlers.h
/// @brief In-process CUDA-QX realtime decoder HOST_CALL handlers.
///
/// This is an in-process adapter (no dlopen) that presents the three CUDA-QX
/// realtime decode verbs as `CUDAQ_DISPATCH_HOST_CALL` entries. Unlike the
/// out-of-tree `cudaqx_decoder_hostcall` shim, the decoder bank is configured
/// once before serving via `decoder_configure_from_file`, which calls the
/// CUDA-QX config API directly from the host application. The decoder bank
/// persists across QUA phases; `decoder_finalize` tears it down at exit.
///
/// Usage:
/// @code
///   if (decoder_configure_from_file("/path/to/config.yml") != DECODER_OK)
///     return 1;
///   cudaq_function_entry_t entries[DECODER_FUNCTION_COUNT]{};
///   uint32_t n = 0;
///   build_decoder_function_table(entries, DECODER_FUNCTION_COUNT, &n);
///   // ... serve HOST_CALL RPCs on the data stream ...
///   decoder_finalize();
/// @endcode

#pragma once

#include <stdint.h>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

/// Status codes. `DECODER_OK` is zero; any other value is a failure.
enum {
  DECODER_OK = 0,
  DECODER_ERR_INVALID_ARG = 1,
  DECODER_ERR_CONFIG = 2,
  DECODER_ERR_RUNTIME = 3
};

/// Function IDs the handlers register under -- `cudaq::realtime::fnv1a_hash`
/// of the realtime decoder wire function names emitted by `device_call`.
enum {
  DECODER_RESET_FUNCTION_ID = 0x8a7d3ef8,     // reset_decoder_ui64
  DECODER_ENQUEUE_FUNCTION_ID = 0xee9c576f,   // enqueue_syndromes_ui64
  DECODER_GET_CORRECTIONS_FUNCTION_ID = 0xe138287e, // get_corrections_ui64
  DECODER_FUNCTION_COUNT = 3
};

/// End-of-batch marker the QUA program sends (fire-and-forget) when its data
/// phase finishes. It is a dedicated non-zero id -- NOT function_id 0, which
/// the library generic host loop reserves for "OPX shutdown". The server
/// registers a phase-done handler under this id whose only job is to set the
/// loop's break flag, so the loop returns and the server can reconnect for the
/// next QUA program without unloading the decoder bank. Value is
/// `fnv1a_hash("phase_complete")`.
enum { DECODER_PHASE_DONE_FUNCTION_ID = 0x3265ada2 };

#ifdef __cplusplus
extern "C" {
#endif

/// Configure the decoder bank from a YAML config file. Calls
/// `cudaq::qec::decoding::config::finalize_decoders` (drops any previous bank),
/// then `configure_decoders_from_str`. Must be called once before serving.
/// Returns `DECODER_OK` on success, `DECODER_ERR_CONFIG` on failure.
int decoder_configure_from_file(const char *yaml_path);

/// Fill `entries[0..DECODER_FUNCTION_COUNT)` with the three HOST_CALL
/// function-table entries (reset / enqueue / get_corrections).
/// `capacity` must be >= DECODER_FUNCTION_COUNT.
int build_decoder_function_table(cudaq_function_entry_t *entries,
                                 uint32_t capacity, uint32_t *count);

/// Tear down the decoder bank. Forwards to
/// `cudaq::qec::decoding::config::finalize_decoders`.
void decoder_finalize(void);

#ifdef __cplusplus
}
#endif
