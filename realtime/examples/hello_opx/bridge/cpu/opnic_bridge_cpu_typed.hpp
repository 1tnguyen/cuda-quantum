/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "opnic_bridge_cpu.hpp"
#include "opnic_direct_cpu.hpp"
#include "opnic_ring_shim.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

/// @brief CPU bridge context for a caller-selected OPNIC packet layout.
template <typename IncomingPacket, typename OutgoingPacket>
class OpnicBridgeCpuTypedContext final : public OpnicBridgeCpuContextBase {
public:
  using OpnicDirectCpu =
      OpnicDirectCpuContext<IncomingPacket, OutgoingPacket>;

  explicit OpnicBridgeCpuTypedContext(const OpnicBridgeCpuConfig &cfg)
      : config_(cfg) {
    opnic_direct_ = std::make_unique<OpnicDirectCpu>(
        cfg.input_stream_id, cfg.output_stream_id, cfg.buffer_count,
        cfg.force_reset);
    opnic_streams_ = opnic_direct_->get_streams();

    transport_ctx_.rx_buffer = opnic_streams_.rx_buffer;
    transport_ctx_.pi_ptr = opnic_streams_.rx_pi_ptr;
    transport_ctx_.rx_alloc_size = opnic_streams_.rx_allocation_size;
    transport_ctx_.buf_count = opnic_streams_.buffer_count;
    transport_ctx_.tx_buffer = opnic_streams_.tx_buffer;
    transport_ctx_.doorbell_ptr = opnic_streams_.tx_doorbell_ptr;
    transport_ctx_.tx_alloc_size = opnic_streams_.tx_allocation_size;

    if (!cfg.unified) {
      if (!cfg.shutdown_flag)
        throw std::invalid_argument(
            "OpnicBridgeCpuConfig::shutdown_flag required for ring path");
      ring_shim_ =
          std::make_unique<OpnicRingShim>(transport_ctx_, cfg.shutdown_flag);
    }
  }

  ~OpnicBridgeCpuTypedContext() override {
    if (ring_shim_)
      ring_shim_->join();
  }

  cudaq_status_t
  get_transport_context(cudaq_realtime_transport_context_t context_type,
                        void *out_context) override {
    if (!out_context)
      return CUDAQ_ERR_INVALID_ARG;

    if (context_type == UNIFIED) {
      *static_cast<opnic_cpu_transport_ctx *>(out_context) = transport_ctx_;
      return CUDAQ_OK;
    }
    if (context_type == RING_BUFFER) {
      if (config_.unified) {
        std::cerr << "ERROR: CPU OPNIC bridge RING_BUFFER context unavailable "
                     "in unified mode"
                  << std::endl;
        return CUDAQ_ERR_INVALID_ARG;
      }
      if (!ring_shim_) {
        std::cerr << "ERROR: CPU OPNIC bridge ring shim not initialized"
                  << std::endl;
        return CUDAQ_ERR_INTERNAL;
      }
      *static_cast<cudaq_ringbuffer_t *>(out_context) = ring_shim_->ring();
      return CUDAQ_OK;
    }

    std::cerr << "ERROR: Unsupported CPU OPNIC bridge transport context type"
              << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }

  cudaq_status_t connect() override {
    printf("[HOST] Waiting for OPX handshake, start the qua program\n");
    opnic_direct_->sync();
    printf("[HOST] OPX<->OPNIC sync is complete (CPU path). Packets can now be "
           "sent/received\n");
    return CUDAQ_OK;
  }

  cudaq_status_t launch() override {
    if (config_.unified)
      return CUDAQ_OK;
    if (!ring_shim_)
      return CUDAQ_ERR_INTERNAL;
    ring_shim_->start();
    return CUDAQ_OK;
  }

  cudaq_status_t disconnect() override {
    if (config_.unified)
      return CUDAQ_OK;
    if (ring_shim_)
      ring_shim_->join();
    return CUDAQ_OK;
  }

  cudaq_status_t get_host_dataplane(cudaq_host_dataplane_t *out) override {
    if (!out)
      return CUDAQ_ERR_INVALID_ARG;
    if (!config_.unified)
      return CUDAQ_ERR_INVALID_ARG;
    out->host_ctx = this;
    out->rx_acquire = &host_rx_acquire;
    out->tx_acquire = &host_tx_acquire;
    out->tx_commit = &host_tx_commit;
    return CUDAQ_OK;
  }

private:
  static cudaq_rx_status_t host_rx_acquire(void *host_ctx, void **out_rx_slot,
                                           size_t *out_slot_size) {
    auto *ctx = static_cast<OpnicBridgeCpuTypedContext *>(host_ctx);
    const opnic_cpu_transport_ctx &t = ctx->transport_ctx_;

    if (static_cast<std::int32_t>(*t.pi_ptr - ctx->consumer_index_) < 1)
      return CUDAQ_RX_EMPTY;
    const std::uint32_t slot =
        ctx->consumer_index_ % static_cast<std::uint32_t>(t.buf_count);
    *out_rx_slot = reinterpret_cast<std::uint8_t *>(t.rx_buffer) +
                   slot * t.rx_alloc_size;
    *out_slot_size = t.rx_alloc_size;
    ctx->consumer_index_++;
    return CUDAQ_RX_READY;
  }

  static cudaq_status_t host_tx_acquire(void *host_ctx, void **out_tx_slot,
                                        size_t *out_slot_size) {
    auto *ctx = static_cast<OpnicBridgeCpuTypedContext *>(host_ctx);
    const opnic_cpu_transport_ctx &t = ctx->transport_ctx_;

    const std::uint32_t slot =
        ctx->out_index_ % static_cast<std::uint32_t>(t.buf_count);
    *out_tx_slot = reinterpret_cast<std::uint8_t *>(t.tx_buffer) +
                   slot * t.tx_alloc_size;
    *out_slot_size = t.tx_alloc_size;
    return CUDAQ_OK;
  }

  static cudaq_status_t host_tx_commit(void *host_ctx) {
    auto *ctx = static_cast<OpnicBridgeCpuTypedContext *>(host_ctx);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *ctx->transport_ctx_.doorbell_ptr = 1;
    ctx->out_index_++;
    return CUDAQ_OK;
  }

  OpnicBridgeCpuConfig config_;
  std::unique_ptr<OpnicDirectCpu> opnic_direct_;
  OpnicDirectCpuStreams opnic_streams_{};
  opnic_cpu_transport_ctx transport_ctx_{};
  std::unique_ptr<OpnicRingShim> ring_shim_;
  std::uint32_t consumer_index_ = 0;
  std::uint32_t out_index_ = 0;
};

/// @brief Construct a CPU OPNIC bridge context for a caller-selected packet pair.
template <typename IncomingPacket, typename OutgoingPacket>
cudaq_realtime_bridge_handle_t
opnic_bridge_cpu_create_context_for_packets(const OpnicBridgeCpuConfig *cfg) {
  if (!cfg)
    return nullptr;
  try {
    return new OpnicBridgeCpuTypedContext<IncomingPacket, OutgoingPacket>(*cfg);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to create OpnicBridgeCpuContext: " << e.what()
              << std::endl;
    return nullptr;
  }
}
