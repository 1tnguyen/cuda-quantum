/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// In-process HOST_CALL handlers for the CUDA-QX realtime decoder bank.
// The three decode verbs (reset / enqueue / get) forward to the
// decoder-agnostic host API. Configuration is done once at startup via
// decoder_configure_from_file, which reads the YAML and calls the CUDA-QX
// config API directly -- no chunked-RPC protocol, no control stream.

#include "decoder_handlers.h"

// Public CUDA-QX headers (installed at ~/.cudaqx/include).
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"

// Realtime RPC wire format: RPCHeader / RPCResponse / magics.
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <chrono>
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

// The three host decode functions are exported from
// libcudaq-qec-realtime-decoding.so but their declaring header is
// lib-internal (not installed). Re-declare the stable signatures.
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

// ---------------------------------------------------------------------------
// TRT raw probability validator (optional; enabled by CUDAQX_TRT_GOLDEN_PATH)
// ---------------------------------------------------------------------------

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

std::uint64_t pack_bits_lsb(const std::vector<std::uint8_t> &bits) {
  std::uint64_t packed = 0;
  for (std::size_t i = 0; i < bits.size() && i < kMaxBits; ++i)
    if (bits[i])
      packed |= (1ULL << i);
  return packed;
}

void print_trt_raw_validation_summary() {
  for (const auto &[decoder_id, v] : g_trt_raw_validators) {
    std::printf("[TRT raw validation] decoder=%zu checks=%llu passed=%llu "
                "failed=%llu missing=%llu tolerance=%.3g golden=%s\n",
                decoder_id,
                static_cast<unsigned long long>(v.checks),
                static_cast<unsigned long long>(v.passed),
                static_cast<unsigned long long>(v.failed),
                static_cast<unsigned long long>(v.missing),
                v.tolerance, v.golden_path.c_str());
  }
}

bool load_trt_golden_file(const std::string &path, TrtRawValidator &v) {
  std::ifstream in(path);
  if (!in)
    return false;
  v.golden_path = path;
  v.expected_by_syndrome.clear();
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
    v.expected_by_syndrome[static_cast<std::uint64_t>(packed)] = expected;
  }
  return !v.expected_by_syndrome.empty();
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

  for (const auto &dc : config.decoders) {
    if (dc.type != "trt_decoder")
      continue;
    TrtRawValidator v;
    v.syndrome_size = dc.syndrome_size;
    v.tolerance = tolerance;
    if (!load_trt_golden_file(golden_path, v)) {
      std::printf("[TRT raw validation] decoder=%lld failed: could not load "
                  "golden file %s\n",
                  static_cast<long long>(dc.id), golden_path);
      return false;
    }
    try {
      auto pcm = cudaq::qec::pcm_from_sparse_vec(
          dc.H_sparse, dc.syndrome_size, dc.block_size);
      v.decoder = cudaq::qec::get_decoder(
          dc.type, pcm, dc.decoder_custom_args_to_heterogeneous_map());
      std::vector<cudaq::qec::float_t> zero(dc.syndrome_size, 0.0);
      v.decoder->decode(zero);
    } catch (const std::exception &e) {
      std::printf("[TRT raw validation] decoder=%lld failed: %s\n",
                  static_cast<long long>(dc.id), e.what());
      return false;
    } catch (...) {
      std::printf("[TRT raw validation] decoder=%lld failed: unknown "
                  "exception\n",
                  static_cast<long long>(dc.id));
      return false;
    }
    g_trt_raw_validators[static_cast<std::size_t>(dc.id)] = std::move(v);
    std::printf("[TRT raw validation] decoder=%lld enabled: %zu golden "
                "syndromes, tolerance=%.3g, file=%s\n",
                static_cast<long long>(dc.id),
                g_trt_raw_validators[static_cast<std::size_t>(dc.id)]
                    .expected_by_syndrome.size(),
                tolerance, golden_path);
  }
  return true;
}

std::int32_t validate_trt_raw_output(std::size_t decoder_id,
                                     const std::vector<std::uint8_t> &bits) {
  auto it = g_trt_raw_validators.find(decoder_id);
  if (it == g_trt_raw_validators.end())
    return DECODER_OK;

  auto &v = it->second;
  ++v.checks;
  const std::uint64_t packed = pack_bits_lsb(bits);
  auto expected_it = v.expected_by_syndrome.find(packed);
  if (expected_it == v.expected_by_syndrome.end()) {
    ++v.missing;
    std::printf("[TRT raw validation] decoder=%zu missing golden syndrome "
                "0x%llx\n",
                decoder_id, static_cast<unsigned long long>(packed));
    return DECODER_ERR_RUNTIME;
  }

  if (bits.size() != v.syndrome_size || !v.decoder) {
    ++v.failed;
    std::printf("[TRT raw validation] decoder=%zu invalid state: bits=%zu "
                "expected_bits=%llu validator_ready=%c\n",
                decoder_id, bits.size(),
                static_cast<unsigned long long>(v.syndrome_size),
                v.decoder ? 'Y' : 'N');
    return DECODER_ERR_RUNTIME;
  }

  std::vector<cudaq::qec::float_t> soft(bits.size(), 0.0);
  for (std::size_t i = 0; i < bits.size(); ++i)
    soft[i] = static_cast<cudaq::qec::float_t>(bits[i] ? 1.0 : 0.0);

  try {
    const auto result = v.decoder->decode(soft);
    if (result.result.empty())
      throw std::runtime_error("empty TRT raw decoder result");
    const double actual = static_cast<double>(result.result.front());
    const double expected = expected_it->second;
    const double error = std::abs(actual - expected);
    if (error > v.tolerance) {
      ++v.failed;
      std::printf("[TRT raw validation] decoder=%zu mismatch syndrome=0x%llx "
                  "actual=%.9f expected=%.9f error=%.9f tolerance=%.3g\n",
                  decoder_id, static_cast<unsigned long long>(packed), actual,
                  expected, error, v.tolerance);
      return DECODER_ERR_RUNTIME;
    }
    ++v.passed;
  } catch (const std::exception &e) {
    ++v.failed;
    std::printf("[TRT raw validation] decoder=%zu exception syndrome=0x%llx: "
                "%s\n",
                decoder_id, static_cast<unsigned long long>(packed), e.what());
    return DECODER_ERR_RUNTIME;
  } catch (...) {
    ++v.failed;
    std::printf("[TRT raw validation] decoder=%zu unknown exception "
                "syndrome=0x%llx\n",
                decoder_id, static_cast<unsigned long long>(packed));
    return DECODER_ERR_RUNTIME;
  }
  return DECODER_OK;
}

// ---------------------------------------------------------------------------
// Shared RPC slot helpers
// ---------------------------------------------------------------------------

std::uint64_t read_u64(const std::uint8_t *args, std::uint32_t arg_len,
                       std::uint32_t index, bool &ok) {
  const std::uint32_t offset = index * sizeof(std::uint64_t);
  if (offset + sizeof(std::uint64_t) > arg_len) {
    ok = false;
    return 0;
  }
  std::uint64_t value = 0;
  std::memcpy(&value, args + offset, sizeof(value));
  return value;
}

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

// Handler self-timing. The shared OPNIC host loop dispatches via
// `host_dispatch_rpc`, which (unlike our former bespoke loop) does not time
// the handler. So each decode handler times its own work and reports the
// elapsed nanoseconds back through the response's `ptp_timestamp` field --
// the low-overhead timing channel QUA reads as the per-RPC host processing
// time.
inline std::uint64_t ns_since(
    const std::chrono::steady_clock::time_point &t0) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t0)
          .count());
}

// ---------------------------------------------------------------------------
// HOST_CALL handlers
// ---------------------------------------------------------------------------

void reset_handler(void *slot, std::size_t slot_size) {
  const auto t0 = std::chrono::steady_clock::now();
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  (void)ptp;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  std::int32_t status = DECODER_ERR_INVALID_ARG;
  if (ok) {
    try {
      cudaq::qec::decoding::host::reset_decoder(
          static_cast<std::size_t>(decoder_id));
      status = DECODER_OK;
    } catch (...) {
      status = DECODER_ERR_RUNTIME;
    }
  }
  write_response(slot, request_id, ns_since(t0), status, 0);
}

void enqueue_syndromes_handler(void *slot, std::size_t slot_size) {
  const auto t0 = std::chrono::steady_clock::now();
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  (void)ptp;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  const std::uint64_t syndrome_size = read_u64(args, arg_len, 1, ok);
  const std::uint64_t syndrome = read_u64(args, arg_len, 2, ok);
  const std::uint64_t tag = read_u64(args, arg_len, 3, ok);

  std::int32_t status = DECODER_ERR_INVALID_ARG;
  if (ok && syndrome_size <= kMaxBits) {
    std::vector<std::uint8_t> bits(static_cast<std::size_t>(syndrome_size));
    for (std::uint64_t i = 0; i < syndrome_size; ++i)
      bits[i] = static_cast<std::uint8_t>((syndrome >> i) & 0x1ULL);
    try {
      const auto vs =
          validate_trt_raw_output(static_cast<std::size_t>(decoder_id), bits);
      if (vs != DECODER_OK) {
        write_response(slot, request_id, ns_since(t0), vs, 0);
        return;
      }
      cudaq::qec::decoding::host::enqueue_syndromes(
          static_cast<std::size_t>(decoder_id), bits.data(), syndrome_size,
          tag);
      status = DECODER_OK;
    } catch (...) {
      status = DECODER_ERR_RUNTIME;
    }
  }
  write_response(slot, request_id, ns_since(t0), status, 0);
}

void get_corrections_handler(void *slot, std::size_t slot_size) {
  const auto t0 = std::chrono::steady_clock::now();
  std::uint32_t arg_len = 0, request_id = 0;
  std::uint64_t ptp = 0;
  const std::uint8_t *args =
      request_args(slot, slot_size, arg_len, request_id, ptp);
  if (!args)
    return;
  (void)ptp;
  bool ok = true;
  const std::uint64_t decoder_id = read_u64(args, arg_len, 0, ok);
  const std::uint64_t return_size = read_u64(args, arg_len, 1, ok);
  const std::uint64_t reset = read_u64(args, arg_len, 2, ok);

  if (!ok || return_size == 0 || return_size > kMaxBits ||
      slot_size < sizeof(RPCResponse) + sizeof(std::uint64_t)) {
    write_response(slot, request_id, ns_since(t0), DECODER_ERR_INVALID_ARG, 0);
    return;
  }

  std::vector<std::uint8_t> corr(static_cast<std::size_t>(return_size), 0);
  try {
    cudaq::qec::decoding::host::get_corrections(
        static_cast<std::size_t>(decoder_id), corr.data(), return_size,
        reset != 0);
  } catch (...) {
    write_response(slot, request_id, ns_since(t0), DECODER_ERR_RUNTIME, 0);
    return;
  }

  std::uint64_t result = 0;
  for (std::uint64_t i = 0; i < return_size; ++i)
    if (corr[i])
      result |= (1ULL << i);

  write_response(slot, request_id, ns_since(t0), DECODER_OK,
                 static_cast<std::uint32_t>(sizeof(std::uint64_t)));
  std::memcpy(static_cast<std::uint8_t *>(slot) + sizeof(RPCResponse), &result,
              sizeof(result));
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

// ---------------------------------------------------------------------------
// YAML assembly + CUDA-QX configure call (shared by configure_from_file)
// ---------------------------------------------------------------------------

std::int32_t configure_from_yaml_str(const std::string &yaml) {
  const auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  cudaq::qec::decoding::config::finalize_decoders();
  g_trt_raw_validators.clear();
  const int status =
      cudaq::qec::decoding::config::configure_decoders_from_str(yaml.c_str());
  if (status != 0)
    return DECODER_ERR_CONFIG;
  if (!configure_trt_raw_validators(parsed)) {
    cudaq::qec::decoding::config::finalize_decoders();
    g_trt_raw_validators.clear();
    return DECODER_ERR_CONFIG;
  }
  return DECODER_OK;
}

} // namespace

extern "C" {

int decoder_configure_from_file(const char *yaml_path) {
  if (!yaml_path || !*yaml_path)
    return DECODER_ERR_INVALID_ARG;
  std::ifstream in(yaml_path);
  if (!in) {
    std::fprintf(stderr, "ERROR: cannot open config file '%s'\n", yaml_path);
    return DECODER_ERR_CONFIG;
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  if (buf.str().empty()) {
    std::fprintf(stderr, "ERROR: config file '%s' is empty\n", yaml_path);
    return DECODER_ERR_CONFIG;
  }
  try {
    return configure_from_yaml_str(buf.str());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: configure_from_file('%s'): %s\n", yaml_path,
                 e.what());
    return DECODER_ERR_CONFIG;
  } catch (...) {
    std::fprintf(stderr, "ERROR: configure_from_file('%s'): unknown error\n",
                 yaml_path);
    return DECODER_ERR_CONFIG;
  }
}

int build_decoder_function_table(cudaq_function_entry_t *entries,
                                 uint32_t capacity, uint32_t *count) {
  if (!entries || !count || capacity < DECODER_FUNCTION_COUNT)
    return DECODER_ERR_INVALID_ARG;
  fill_entry(entries[0], &reset_handler, DECODER_RESET_FUNCTION_ID,
             /*num_args=*/1, /*num_results=*/0);
  fill_entry(entries[1], &enqueue_syndromes_handler, DECODER_ENQUEUE_FUNCTION_ID,
             /*num_args=*/4, /*num_results=*/0);
  fill_entry(entries[2], &get_corrections_handler,
             DECODER_GET_CORRECTIONS_FUNCTION_ID,
             /*num_args=*/3, /*num_results=*/1);
  *count = DECODER_FUNCTION_COUNT;
  return DECODER_OK;
}

void decoder_finalize(void) {
  try {
    print_trt_raw_validation_summary();
    g_trt_raw_validators.clear();
    cudaq::qec::decoding::config::finalize_decoders();
  } catch (...) {
  }
}

} // extern "C"
