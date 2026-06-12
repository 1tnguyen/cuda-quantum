/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Out-of-tree HOST_CALL shim over the cuda-qx decoder bank. Pure C++ -- no CUDA
// kernels, no LLVM, no graph capture. It links the cuda-qx wheel's
// libcudaq-qec-realtime-decoding.so. The decode verbs (reset/enqueue/get)
// forward to the decoder-agnostic host API; the configure_decoder verb
// (re)builds the bank from chunked YAML payloads so an always-on server can be
// configured/reconfigured over the wire. Works for any decoder the wheel ships.

#include "cudaqx_decoder_hostcall.h"

// Public cuda-qx header (shipped in the wheel): the config front-end that builds
// the decoder bank. Pulls cudaqx::heterogeneous_map; needs C++20.
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"

// Realtime RPC wire format: RPCHeader / RPCResponse / magics.
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// The three "hot" realtime functions are exported (visibility("default")) from
// libcudaq-qec-realtime-decoding.so, but their declaring header
// (lib/realtime/realtime_decoding.h) is lib-internal and not shipped in the
// wheel. Re-declare the stable signatures so we can call them; they resolve at
// link time against the wheel's .so.
namespace cudaq::qec::decoding::host {
void reset_decoder(std::size_t decoder_id);
void enqueue_syndromes(std::size_t decoder_id, std::uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag);
void get_corrections(std::size_t decoder_id, std::uint8_t *corrections,
                     std::uint64_t correction_length, bool reset);
} // namespace cudaq::qec::decoding::host

using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace {

constexpr std::uint64_t kMaxBits = 64; // ui64 wire: <= 64 syndrome / obs bits

struct ConfigureAssembly {
  std::vector<std::uint8_t> bytes;
  std::uint32_t total = 0;
  std::uint32_t received = 0;
  bool active = false;
};

ConfigureAssembly g_config_assembly;

struct TrtRawValidator {
  std::uint64_t syndrome_size = 0;
  std::string golden_path;
  std::unique_ptr<cudaq::qec::decoder> decoder;
  std::unordered_map<std::uint64_t, double> expected_by_syndrome;
  double tolerance = 1.0e-4;
  std::uint64_t checks = 0;
  std::uint64_t passed = 0;
  std::uint64_t failed = 0;
  std::uint64_t missing = 0;
};

std::unordered_map<std::size_t, TrtRawValidator> g_trt_raw_validators;

// Read the i-th uint64 argument (8-byte packed, matching the device_call ui64
// ABI). Sets ok=false if it would read past arg_len.
std::uint64_t read_u64(const std::uint8_t *args, std::uint32_t arg_len,
                       std::uint32_t index, bool &ok) {
  const std::uint32_t offset = index * sizeof(std::uint64_t);
  if (offset + sizeof(std::uint64_t) > arg_len) {
    ok = false;
    return 0;
  }
  std::uint64_t value = 0;
  std::memcpy(&value, args + offset, sizeof(std::uint64_t));
  return value;
}

// Validate the request header and expose the argument region; returns nullptr
// if the slot is malformed. request_id/ptp are captured so the caller can echo
// them into the response that overwrites the (overlapping) header.
const std::uint8_t *request_args(void *slot, std::size_t slot_size,
                                 std::uint32_t &arg_len,
                                 std::uint32_t &request_id,
                                 std::uint64_t &ptp_timestamp) {
  if (slot_size < sizeof(RPCHeader))
    return nullptr;
  const auto *hdr = static_cast<const RPCHeader *>(slot);
  if (hdr->magic != RPC_MAGIC_REQUEST)
    return nullptr;
  arg_len = hdr->arg_len;
  request_id = hdr->request_id;
  ptp_timestamp = hdr->ptp_timestamp;
  const std::uint32_t max_args =
      static_cast<std::uint32_t>(slot_size - sizeof(RPCHeader));
  if (arg_len > max_args)
    arg_len = max_args;
  return static_cast<const std::uint8_t *>(slot) + sizeof(RPCHeader);
}

void write_response(void *slot, std::uint32_t request_id,
                    std::uint64_t ptp_timestamp, std::int32_t status,
                    std::uint32_t result_len) {
  auto *resp = static_cast<RPCResponse *>(slot);
  resp->magic = RPC_MAGIC_RESPONSE;
  resp->status = status;
  resp->result_len = status == 0 ? result_len : 0;
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp_timestamp;
}

std::uint32_t read_u32_le(const std::uint8_t *p) {
  std::uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::uint64_t pack_bits_lsb(const std::vector<std::uint8_t> &bits) {
  std::uint64_t packed = 0;
  for (std::size_t i = 0; i < bits.size() && i < kMaxBits; ++i)
    if (bits[i])
      packed |= (1ULL << i);
  return packed;
}

void print_trt_raw_validation_summary() {
  for (const auto &[decoder_id, validator] : g_trt_raw_validators) {
    std::printf("[TRT raw validation] decoder=%zu checks=%llu passed=%llu "
                "failed=%llu missing=%llu tolerance=%.3g golden=%s\n",
                decoder_id,
                static_cast<unsigned long long>(validator.checks),
                static_cast<unsigned long long>(validator.passed),
                static_cast<unsigned long long>(validator.failed),
                static_cast<unsigned long long>(validator.missing),
                validator.tolerance, validator.golden_path.c_str());
  }
}

bool load_trt_golden_file(const std::string &path, TrtRawValidator &validator) {
  std::ifstream in(path);
  if (!in)
    return false;

  validator.golden_path = path;
  validator.expected_by_syndrome.clear();
  std::string line;
  while (std::getline(in, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos)
      line.resize(comment);
    std::istringstream is(line);
    std::string key;
    double expected = 0.0;
    if (!(is >> key >> expected))
      continue;
    std::size_t parsed = 0;
    unsigned long long packed = 0;
    try {
      packed = std::stoull(key, &parsed, 0);
    } catch (...) {
      return false;
    }
    if (parsed != key.size() ||
        packed > std::numeric_limits<std::uint64_t>::max())
      return false;
    validator.expected_by_syndrome[static_cast<std::uint64_t>(packed)] =
        expected;
  }
  return !validator.expected_by_syndrome.empty();
}

bool configure_trt_raw_validators(
    const cudaq::qec::decoding::config::multi_decoder_config &config) {
  g_trt_raw_validators.clear();
  const char *golden_path = std::getenv("CUDAQX_TRT_GOLDEN_PATH");
  if (!golden_path || !*golden_path)
    return true;

  const char *tol_env = std::getenv("CUDAQX_TRT_GOLDEN_TOLERANCE");
  double tolerance = 1.0e-4;
  if (tol_env && *tol_env)
    tolerance = std::strtod(tol_env, nullptr);
  if (!(tolerance > 0.0))
    tolerance = 1.0e-4;

  for (const auto &decoder_config : config.decoders) {
    if (decoder_config.type != "trt_decoder")
      continue;
    TrtRawValidator validator;
    validator.syndrome_size = decoder_config.syndrome_size;
    validator.tolerance = tolerance;
    if (!load_trt_golden_file(golden_path, validator)) {
      std::printf("[TRT raw validation] decoder=%lld failed: could not load "
                  "golden file %s\n",
                  static_cast<long long>(decoder_config.id), golden_path);
      return false;
    }
    try {
      auto pcm = cudaq::qec::pcm_from_sparse_vec(
          decoder_config.H_sparse, decoder_config.syndrome_size,
          decoder_config.block_size);
      validator.decoder = cudaq::qec::get_decoder(
          decoder_config.type, pcm,
          decoder_config.decoder_custom_args_to_heterogeneous_map());
      std::vector<cudaq::qec::float_t> zero(decoder_config.syndrome_size, 0.0);
      validator.decoder->decode(zero);
    } catch (const std::exception &e) {
      std::printf("[TRT raw validation] decoder=%lld failed: %s\n",
                  static_cast<long long>(decoder_config.id), e.what());
      return false;
    } catch (...) {
      std::printf("[TRT raw validation] decoder=%lld failed: unknown exception\n",
                  static_cast<long long>(decoder_config.id));
      return false;
    }
    g_trt_raw_validators[static_cast<std::size_t>(decoder_config.id)] =
        std::move(validator);
    std::printf("[TRT raw validation] decoder=%lld enabled: %zu golden "
                "syndromes, tolerance=%.3g, file=%s\n",
                static_cast<long long>(decoder_config.id),
                g_trt_raw_validators[static_cast<std::size_t>(decoder_config.id)]
                    .expected_by_syndrome.size(),
                tolerance, golden_path);
  }
  return true;
}

std::int32_t validate_trt_raw_output(std::size_t decoder_id,
                                     const std::vector<std::uint8_t> &bits) {
  auto it = g_trt_raw_validators.find(decoder_id);
  if (it == g_trt_raw_validators.end())
    return CUDAQX_RT_OK;

  auto &validator = it->second;
  ++validator.checks;
  const std::uint64_t packed = pack_bits_lsb(bits);
  auto expected_it = validator.expected_by_syndrome.find(packed);
  if (expected_it == validator.expected_by_syndrome.end()) {
    ++validator.missing;
    std::printf("[TRT raw validation] decoder=%zu missing golden syndrome "
                "0x%llx\n",
                decoder_id, static_cast<unsigned long long>(packed));
    return CUDAQX_RT_ERR_RUNTIME;
  }

  if (bits.size() != validator.syndrome_size || !validator.decoder) {
    ++validator.failed;
    std::printf("[TRT raw validation] decoder=%zu invalid state: bits=%zu "
                "expected_bits=%llu validator_ready=%c\n",
                decoder_id, bits.size(),
                static_cast<unsigned long long>(validator.syndrome_size),
                validator.decoder ? 'Y' : 'N');
    return CUDAQX_RT_ERR_RUNTIME;
  }

  std::vector<cudaq::qec::float_t> soft(bits.size(), 0.0);
  for (std::size_t i = 0; i < bits.size(); ++i)
    soft[i] = static_cast<cudaq::qec::float_t>(bits[i] ? 1.0 : 0.0);

  try {
    const auto result = validator.decoder->decode(soft);
    if (result.result.empty())
      throw std::runtime_error("empty TRT raw decoder result");
    const double actual = static_cast<double>(result.result.front());
    const double expected = expected_it->second;
    const double error = std::abs(actual - expected);
    if (error > validator.tolerance) {
      ++validator.failed;
      std::printf("[TRT raw validation] decoder=%zu mismatch syndrome=0x%llx "
                  "actual=%.9f expected=%.9f error=%.9f tolerance=%.3g\n",
                  decoder_id, static_cast<unsigned long long>(packed), actual,
                  expected, error, validator.tolerance);
      return CUDAQX_RT_ERR_RUNTIME;
    }
    ++validator.passed;
  } catch (const std::exception &e) {
    ++validator.failed;
    std::printf("[TRT raw validation] decoder=%zu exception syndrome=0x%llx: "
                "%s\n",
                decoder_id, static_cast<unsigned long long>(packed), e.what());
    return CUDAQX_RT_ERR_RUNTIME;
  } catch (...) {
    ++validator.failed;
    std::printf("[TRT raw validation] decoder=%zu unknown exception "
                "syndrome=0x%llx\n",
                decoder_id, static_cast<unsigned long long>(packed));
    return CUDAQX_RT_ERR_RUNTIME;
  }

  return CUDAQX_RT_OK;
}

std::int32_t configure_assembled_yaml(const std::uint8_t *bytes,
                                      std::uint32_t len) {
  try {
    const std::string yaml(reinterpret_cast<const char *>(bytes), len);
    const auto parsed_config =
        cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    cudaq::qec::decoding::config::finalize_decoders(); // drop existing bank
    g_trt_raw_validators.clear();
    const auto status =
        cudaq::qec::decoding::config::configure_decoders_from_str(yaml.c_str());
    if (status != 0)
      return CUDAQX_RT_ERR_CONFIG;
    if (!configure_trt_raw_validators(parsed_config)) {
      cudaq::qec::decoding::config::finalize_decoders();
      g_trt_raw_validators.clear();
      return CUDAQX_RT_ERR_CONFIG;
    }
    return CUDAQX_RT_OK;
  } catch (...) {
    g_trt_raw_validators.clear();
    return CUDAQX_RT_ERR_CONFIG;
  }
}

std::int32_t configure_chunk(const std::uint8_t *args, std::uint32_t arg_len) {
  if (arg_len < CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES)
    return CUDAQX_RT_ERR_CONFIG;

  const std::uint32_t magic = read_u32_le(args + 0);
  const std::uint32_t version = read_u32_le(args + 4);
  const std::uint32_t total = read_u32_le(args + 8);
  const std::uint32_t offset = read_u32_le(args + 12);
  const std::uint32_t chunk_bytes = read_u32_le(args + 16);
  const std::uint32_t flags = read_u32_le(args + 20);

  const bool begin = (flags & CUDAQX_RT_CONFIGURE_CHUNK_BEGIN) != 0;
  const bool end = (flags & CUDAQX_RT_CONFIGURE_CHUNK_END) != 0;
  const std::uint32_t payload_len =
      arg_len - CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES;

  // The configure protocol is intentionally stricter than a generic byte
  // stream. Chunks must be delivered in order: BEGIN creates the assembly
  // buffer, each subsequent offset must match `received`, and END is the only
  // point where CUDA-QX sees the full YAML. This lets QUA split a large decoder
  // config into OPNIC-sized packets while the decoder bank changes atomically
  // from the server's point of view.
  const bool valid_header =
      magic == CUDAQX_RT_CONFIGURE_CHUNK_MAGIC &&
      version == CUDAQX_RT_CONFIGURE_CHUNK_VERSION && total > 0 &&
      total <= CUDAQX_RT_CONFIGURE_MAX_BYTES &&
      chunk_bytes <= CUDAQX_RT_CONFIGURE_CHUNK_BYTES &&
      chunk_bytes <= payload_len && offset <= total &&
      chunk_bytes <= total - offset && (begin || g_config_assembly.active);
  if (!valid_header) {
    g_config_assembly = ConfigureAssembly{};
    return CUDAQX_RT_ERR_CONFIG;
  }

  if (begin) {
    g_config_assembly = ConfigureAssembly{};
    g_config_assembly.bytes.assign(total, 0);
    g_config_assembly.total = total;
    g_config_assembly.active = true;
  }

  if (!g_config_assembly.active || g_config_assembly.total != total ||
      g_config_assembly.received != offset) {
    g_config_assembly = ConfigureAssembly{};
    return CUDAQX_RT_ERR_CONFIG;
  }

  const std::uint8_t *chunk =
      args + CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES;
  std::memcpy(g_config_assembly.bytes.data() + offset, chunk, chunk_bytes);
  g_config_assembly.received = offset + chunk_bytes;

  if (!end)
    return CUDAQX_RT_OK;

  if (g_config_assembly.received != total) {
    g_config_assembly = ConfigureAssembly{};
    return CUDAQX_RT_ERR_CONFIG;
  }

  const std::int32_t status = configure_assembled_yaml(
      g_config_assembly.bytes.data(), g_config_assembly.total);
  g_config_assembly = ConfigureAssembly{};
  return status;
}

//===----------------------------------------------------------------------===//
// HOST_CALL handlers (cudaq_host_rpc_fn_t: void(void* slot, size_t slot_size)).
// ui64 wire arg layout (8-byte packed, LSB-first bits), matching the realtime
// decoder wire functions:
//   reset_decoder_ui64(decoder_id)
//   enqueue_syndromes_ui64(decoder_id, syndrome_size, syndrome, tag)
//   get_corrections_ui64(decoder_id, return_size, reset) -> uint64
//===----------------------------------------------------------------------===//

void reset_handler(void *slot, std::size_t slot_size) {
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  std::int32_t status = CUDAQX_RT_ERR_INVALID_ARG;
  if (ok) {
    try {
      cudaq::qec::decoding::host::reset_decoder(static_cast<std::size_t>(decoder_id));
      status = CUDAQX_RT_OK;
    } catch (...) {
      status = CUDAQX_RT_ERR_RUNTIME;
    }
  }
  write_response(slot, request_id, ptp, status, 0);
}

void enqueue_syndromes_handler(void *slot, std::size_t slot_size) {
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  const std::uint64_t syndrome_size = read_u64(args, arg_len, 1, ok);
  const std::uint64_t syndrome = read_u64(args, arg_len, 2, ok);
  const std::uint64_t tag = read_u64(args, arg_len, 3, ok);

  std::int32_t status = CUDAQX_RT_ERR_INVALID_ARG;
  if (ok && syndrome_size <= kMaxBits) {
    // Unpack the ui64 syndrome (LSB-first) into one byte per measurement.
    std::vector<std::uint8_t> bits(static_cast<std::size_t>(syndrome_size));
    for (std::uint64_t i = 0; i < syndrome_size; ++i)
      bits[i] = static_cast<std::uint8_t>((syndrome >> i) & 0x1ULL);
    try {
      const auto validation_status =
          validate_trt_raw_output(static_cast<std::size_t>(decoder_id), bits);
      if (validation_status != CUDAQX_RT_OK) {
        write_response(slot, request_id, ptp, validation_status, 0);
        return;
      }
      cudaq::qec::decoding::host::enqueue_syndromes(static_cast<std::size_t>(decoder_id), bits.data(),
                              syndrome_size, tag);
      status = CUDAQX_RT_OK;
    } catch (...) {
      status = CUDAQX_RT_ERR_RUNTIME;
    }
  }
  write_response(slot, request_id, ptp, status, 0);
}

void get_corrections_handler(void *slot, std::size_t slot_size) {
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  const std::uint64_t return_size = read_u64(args, arg_len, 1, ok);
  const std::uint64_t reset = read_u64(args, arg_len, 2, ok);

  if (!ok || return_size == 0 || return_size > kMaxBits ||
      slot_size < sizeof(RPCResponse) + sizeof(std::uint64_t)) {
    write_response(slot, request_id, ptp, CUDAQX_RT_ERR_INVALID_ARG, 0);
    return;
  }

  std::vector<std::uint8_t> corr(static_cast<std::size_t>(return_size), 0);
  try {
    cudaq::qec::decoding::host::get_corrections(static_cast<std::size_t>(decoder_id), corr.data(),
                          return_size, reset != 0);
  } catch (...) {
    write_response(slot, request_id, ptp, CUDAQX_RT_ERR_RUNTIME, 0);
    return;
  }

  // Pack corrections LSB-first into the uint64 result (the ui64 ABI return).
  std::uint64_t result = 0;
  for (std::uint64_t i = 0; i < return_size; ++i)
    if (corr[i])
      result |= (1ULL << i);

  write_response(slot, request_id, ptp, CUDAQX_RT_OK,
                 static_cast<std::uint32_t>(sizeof(std::uint64_t)));
  std::memcpy(static_cast<std::uint8_t *>(slot) + sizeof(RPCResponse), &result,
              sizeof(result));
}

// configure_decoder(chunk): CONTROL verb. Every payload must be one chunk frame:
//   [magic][version][total_bytes][offset_bytes][chunk_bytes][flags][bytes...]
// The handler accumulates those frames until END arrives. Only then do we
// rebuild the decoder bank. Runs on the dispatcher thread, so config chunks and
// decode RPCs are serialized without extra locking.
void configure_handler(void *slot, std::size_t slot_size) {
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  std::int32_t status = CUDAQX_RT_ERR_CONFIG;
  if (arg_len >= CUDAQX_RT_CONFIGURE_CHUNK_HEADER_BYTES &&
      read_u32_le(args) == CUDAQX_RT_CONFIGURE_CHUNK_MAGIC)
    status = configure_chunk(args, arg_len);
  else
    g_config_assembly = ConfigureAssembly{};
  write_response(slot, request_id, ptp, status, 0);
}

void fill_entry(cudaq_function_entry_t &e, cudaq_host_rpc_fn_t fn,
                std::uint32_t function_id, std::uint8_t num_args,
                std::uint8_t num_results,
                std::uint8_t arg_type_id = CUDAQ_TYPE_INT64) {
  e = cudaq_function_entry_t{};
  e.handler.host_fn = fn;
  e.function_id = function_id;
  e.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  e.schema.num_args = num_args;
  e.schema.num_results = num_results;
  for (std::uint8_t i = 0; i < num_args && i < 8; ++i)
    e.schema.args[i].type_id = arg_type_id;
  for (std::uint8_t i = 0; i < num_results && i < 4; ++i)
    e.schema.results[i].type_id = CUDAQ_TYPE_INT64;
}

} // namespace

extern "C" {

int cudaqx_rt_get_function_table(cudaq_function_entry_t *entries,
                                 uint32_t capacity, uint32_t *count) {
  if (!entries || !count || capacity < CUDAQX_RT_FUNCTION_COUNT)
    return CUDAQX_RT_ERR_INVALID_ARG;
  fill_entry(entries[0], &reset_handler, CUDAQX_RT_RESET_DECODER_FUNCTION_ID,
             /*num_args=*/1, /*num_results=*/0);
  fill_entry(entries[1], &enqueue_syndromes_handler,
             CUDAQX_RT_ENQUEUE_SYNDROMES_FUNCTION_ID, /*num_args=*/4,
             /*num_results=*/0);
  fill_entry(entries[2], &get_corrections_handler,
             CUDAQX_RT_GET_CORRECTIONS_FUNCTION_ID,
             /*num_args=*/3, /*num_results=*/1);
  // configure_decoder: 1 byte-array arg carrying one configure chunk frame.
  fill_entry(entries[3], &configure_handler,
             CUDAQX_RT_CONFIGURE_DECODER_FUNCTION_ID, /*num_args=*/1,
             /*num_results=*/0, /*arg_type_id=*/CUDAQ_TYPE_ARRAY_UINT8);
  *count = CUDAQX_RT_FUNCTION_COUNT;
  return CUDAQX_RT_OK;
}

void cudaqx_rt_finalize(void) {
  try {
    print_trt_raw_validation_summary();
    g_trt_raw_validators.clear();
    cudaq::qec::decoding::config::finalize_decoders();
  } catch (...) {
  }
}

} // extern "C"
