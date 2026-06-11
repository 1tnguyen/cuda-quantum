/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file opnic_ring_shim.hpp
/// @brief Internal CPU bridge adapter: OPNIC pi/doorbell transport to the
///        library flag-ring contract for the 3-thread CUDAQ_EXEC_HOST path.
///
/// Not part of the public bridge API.  The CPU OPNIC bridge owns an instance,
/// starts it from `launch`, and exposes the ring view via
/// `get_transport_context(RING_BUFFER, ...)`.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h" // RPCHeader
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h" // CUDAQ_REALTIME_CPU_RELAX

#include "opnic_bridge_cpu.hpp" // opnic_cpu_transport_ctx

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

class OpnicRingShim {
public:
  OpnicRingShim(const opnic_cpu_transport_ctx &ctx, volatile int *shutdown_flag)
      : ctx_(ctx), shutdown_(shutdown_flag),
        num_slots_(static_cast<std::uint32_t>(ctx.buf_count)),
        rx_flags_(num_slots_, 0), tx_flags_(num_slots_, 0) {}

  cudaq_ringbuffer_t ring() {
    cudaq_ringbuffer_t rb{};
    rb.rx_flags_host = rx_flags_.data();
    rb.tx_flags_host = tx_flags_.data();
    rb.rx_data_host = reinterpret_cast<std::uint8_t *>(ctx_.rx_buffer);
    rb.tx_data_host = reinterpret_cast<std::uint8_t *>(ctx_.tx_buffer);
    rb.rx_stride_sz = ctx_.rx_alloc_size;
    rb.tx_stride_sz = ctx_.tx_alloc_size;
    return rb;
  }

  std::uint32_t num_slots() const { return num_slots_; }
  std::uint32_t slot_size() const {
    return static_cast<std::uint32_t>(ctx_.rx_alloc_size < ctx_.tx_alloc_size
                                          ? ctx_.rx_alloc_size
                                          : ctx_.tx_alloc_size);
  }

  void start() {
    rx_thread_ = std::thread([this] { rx_loop(); });
    tx_thread_ = std::thread([this] { tx_loop(); });
  }

  void join() {
    if (rx_thread_.joinable())
      rx_thread_.join();
    if (tx_thread_.joinable())
      tx_thread_.join();
  }

private:
  std::uint8_t *rx_slot_addr(std::uint32_t slot) {
    return reinterpret_cast<std::uint8_t *>(ctx_.rx_buffer) +
           slot * ctx_.rx_alloc_size;
  }

  void rx_loop() {
    std::uint32_t consumer = 0;
    while (*shutdown_ == 0) {
      while (static_cast<std::int32_t>(*ctx_.pi_ptr - consumer) < 1) {
        if (*shutdown_ != 0)
          return;
        CUDAQ_REALTIME_CPU_RELAX();
      }
      const std::uint32_t slot = consumer % num_slots_;
      while (std::atomic_ref<std::uint64_t>(rx_flags_[slot])
                 .load(std::memory_order_acquire) != 0) {
        if (*shutdown_ != 0)
          return;
        CUDAQ_REALTIME_CPU_RELAX();
      }
      auto *hdr = reinterpret_cast<const cudaq::realtime::RPCHeader *>(
          rx_slot_addr(slot));
      if (hdr->function_id == 0) {
        *shutdown_ = 1;
        return;
      }
      std::atomic_ref<std::uint64_t>(rx_flags_[slot])
          .store(reinterpret_cast<std::uint64_t>(rx_slot_addr(slot)),
                 std::memory_order_release);
      consumer++;
    }
  }

  void tx_loop() {
    std::uint32_t out = 0;
    while (*shutdown_ == 0) {
      const std::uint32_t slot = out % num_slots_;
      const std::uint64_t v = std::atomic_ref<std::uint64_t>(tx_flags_[slot])
                                  .load(std::memory_order_acquire);
      if (v == 0) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      *ctx_.doorbell_ptr = 1;
      std::atomic_ref<std::uint64_t>(tx_flags_[slot])
          .store(0, std::memory_order_release);
      out++;
    }
  }

  opnic_cpu_transport_ctx ctx_;
  volatile int *shutdown_;
  std::uint32_t num_slots_;
  std::vector<std::uint64_t> rx_flags_;
  std::vector<std::uint64_t> tx_flags_;
  std::thread rx_thread_;
  std::thread tx_thread_;
};
