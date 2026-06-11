/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <format>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <cuda_runtime.h>

#include <qm/opnic_driver.h>  // For OPNIC_IOCTL_RESET_STREAMS
#include <opnic/stream/configurators/context.hpp>
#include <opnic/stream/configurators/driver_configurator.hpp>
#include <opnic/stream/configurators/sync.hpp>
#include <opnic/utils/gpu_aligned_memory.hpp>
#include <opnic/utils/memory.hpp>

#include "opnic_type.h"

/// @brief Direct OPNIC stream pointers for zero-copy GPU access.
/// Contains raw GPU-accessible pointers to OPNIC hardware buffers,
/// bypassing the IncomingStream/OutgoingStream abstractions.
struct OpnicDirectStreams {
  std::uint32_t* rx_buffer;             ///< OPNIC incoming DMA target (RDMA GPU mem)
  volatile std::uint32_t* rx_pi_ptr;    ///< Producer index (RDMA GPU mem)
  std::uint32_t* tx_buffer;             ///< OPNIC outgoing DMA source (RDMA GPU mem)
  std::uint64_t* tx_doorbell_ptr;       ///< mmap'd doorbell (cudaHostRegistered)
  std::uint64_t* tx_status_ptr;         ///< mmap'd status (cudaHostRegistered)
  std::size_t rx_allocation_size;       ///< Per-packet stride in rx_buffer
  std::size_t tx_allocation_size;       ///< Per-packet stride in tx_buffer
  std::size_t buffer_count;             ///< Number of packet slots

  std::uint64_t* tx_doorbell_base_ptr;  ///< Base address for cleanup
  std::size_t tx_status_offset;         ///< Status offset for cleanup
};

/// @brief Context for direct OPNIC streams, managing lifetime of resources.
/// NOTE: OPNIC driver requires SEPARATE file descriptors for each stream.
///
/// Templated on the two packet types so call sites pick their own wire
/// layout. Both types must be `QM_DECLARE_PACKET`-declared and expose
/// `::size`, `::allocation_size`, and `::pattern` (the OPNIC SDK sync
/// signature is built from `pattern`).
///
/// The original (non-templated) name is preserved as a default-templated
/// alias below so existing call sites (`hello_opx_gpu`) keep compiling
/// unchanged.
template <typename IncomingPacket, typename OutgoingPacket>
class OpnicDirectContextT {
public:
  OpnicDirectContextT(std::uint16_t in_stream_id, std::uint16_t out_stream_id,
                      std::size_t buffer_count, bool force_reset = true)
      : in_stream_id_(in_stream_id), out_stream_id_(out_stream_id),
        buffer_count_(buffer_count) {

    // Create SEPARATE driver contexts for each stream (required by OPNIC driver)
    printf("[OPNIC] Finding available driver for RX stream...\n");
    rx_driver_context_ = qm::stream::find_available_driver();
    if (!rx_driver_context_ || !rx_driver_context_->is_valid()) {
      throw std::runtime_error("No OPNIC driver found for RX. Is the driver loaded?");
    }
    printf("[OPNIC] RX driver found (fd=%d)\n", rx_driver_context_->get_fd());

    printf("[OPNIC] Finding available driver for TX stream...\n");
    tx_driver_context_ = qm::stream::find_available_driver();
    if (!tx_driver_context_ || !tx_driver_context_->is_valid()) {
      throw std::runtime_error("No OPNIC driver found for TX. Is the driver loaded?");
    }
    printf("[OPNIC] TX driver found (fd=%d)\n", tx_driver_context_->get_fd());

    // Force explicit reset using RX context
    if (force_reset) {
      reset_streams();
    }

    // Create SEPARATE configurators for each stream
    printf("[OPNIC] Creating RX DriverConfigurator...\n");
    rx_configurator_ = std::make_unique<qm::stream::DriverConfigurator>(*rx_driver_context_);
    printf("[OPNIC] RX DriverConfigurator created\n");

    printf("[OPNIC] Creating TX DriverConfigurator...\n");
    tx_configurator_ = std::make_unique<qm::stream::DriverConfigurator>(*tx_driver_context_);
    printf("[OPNIC] TX DriverConfigurator created\n");

    setup_incoming_stream();
    setup_outgoing_stream();
  }

  ~OpnicDirectContextT() {
    cleanup();
  }

  OpnicDirectContextT(const OpnicDirectContextT&) = delete;
  OpnicDirectContextT& operator=(const OpnicDirectContextT&) = delete;

  /// @brief Get the direct stream pointers for kernel use.
  OpnicDirectStreams get_streams() const {
    return streams_;
  }

  /// @brief Perform OPX handshake synchronization.
  void sync() {
    // Use RX configurator for sync (either works, they share driver state)
    qm::sync(rx_configurator_.get());
  }

  /// @brief Force reset all OPNIC streams at driver level.
  /// Bypasses SDK's std::call_once to ensure clean state.
  void reset_streams() {
    int fd = rx_driver_context_->get_fd();
    printf("[OPNIC] Forcing stream reset via ioctl (fd=%d)...\n", fd);
    auto res = ioctl(fd, OPNIC_IOCTL_RESET_STREAMS, nullptr);
    if (res == -1) {
      throw std::runtime_error("Failed to reset OPNIC streams: errno=" + std::to_string(errno));
    }
    printf("[OPNIC] Stream reset successful\n");
  }

private:
  void setup_incoming_stream() {
    printf("[OPNIC] Setting up incoming stream %u...\n", in_stream_id_);
    constexpr std::size_t packet_size = IncomingPacket::size;
    constexpr std::size_t allocation_size = IncomingPacket::allocation_size;
    const std::size_t buffer_size = allocation_size * buffer_count_;

    printf("[OPNIC] Allocating GPU memory for incoming stream (buffer=%zu, pi=4)...\n", buffer_size);
    rx_buffer_mem_ = std::make_unique<qm::internal::GpuAlignedMemory>(buffer_size);
    rx_pi_mem_ = std::make_unique<qm::internal::GpuAlignedMemory>(sizeof(std::uint32_t));
    printf("[OPNIC] GPU memory allocated: buffer=0x%llx, pi=0x%llx\n", 
           rx_buffer_mem_->get(), rx_pi_mem_->get());

    cudaMemset(reinterpret_cast<void*>(rx_pi_mem_->get()), 0, sizeof(std::uint32_t));

    printf("[OPNIC] Registering sync pattern for incoming stream %u...\n", in_stream_id_);
    qm::stream::sync::register_incoming_stream(
        in_stream_id_,
        std::format("q2h{}_{}", in_stream_id_, IncomingPacket::pattern));

    printf("[OPNIC] Calling setup_incoming_stream ioctl for stream %u...\n", in_stream_id_);
    auto params = rx_configurator_->setup_incoming_stream(
        in_stream_id_,
        qm::StreamType::GPU,
        reinterpret_cast<std::uint32_t*>(rx_pi_mem_->get()),
        reinterpret_cast<std::uint32_t*>(rx_buffer_mem_->get()),
        buffer_size,
        packet_size,
        allocation_size);

    streams_.rx_buffer = params.buffer;
    streams_.rx_pi_ptr = params.pi_ptr;
    streams_.rx_allocation_size = allocation_size;
    streams_.buffer_count = buffer_count_;
    printf("[OPNIC] Incoming stream %u setup complete\n", in_stream_id_);
  }

  void setup_outgoing_stream() {
    printf("[OPNIC] Setting up outgoing stream %u...\n", out_stream_id_);
    constexpr std::size_t packet_size = OutgoingPacket::size;
    constexpr std::size_t allocation_size = OutgoingPacket::allocation_size;
    const std::size_t buffer_size = allocation_size * buffer_count_;

    printf("[OPNIC] Allocating GPU memory for outgoing stream (buffer=%zu)...\n", buffer_size);
    tx_buffer_mem_ = std::make_unique<qm::internal::GpuAlignedMemory>(buffer_size);
    printf("[OPNIC] GPU memory allocated for outgoing: buffer=0x%llx\n", tx_buffer_mem_->get());

    printf("[OPNIC] Registering sync pattern for outgoing stream %u...\n", out_stream_id_);
    qm::stream::sync::register_outgoing_stream(
        out_stream_id_,
        std::format("h2q{}_{}", out_stream_id_, OutgoingPacket::pattern));

    printf("[OPNIC] Calling setup_outgoing_stream ioctl for stream %u...\n", out_stream_id_);
    auto params = tx_configurator_->setup_outgoing_stream(
        out_stream_id_,
        qm::StreamType::GPU,
        reinterpret_cast<std::uint32_t*>(tx_buffer_mem_->get()),
        buffer_size,
        packet_size,
        allocation_size);

    auto register_ret = cudaHostRegister(
        params.status_ptr,
        params.status_offset + sizeof(std::uint64_t),
        cudaHostRegisterIoMemory);
    if (register_ret != cudaSuccess) {
      munmap(params.status_ptr, params.status_offset + sizeof(std::uint64_t));
      munmap(params.doorbell_ptr, sizeof(std::uint64_t));
      throw std::runtime_error("Failed to cudaHostRegister status pointer");
    }
    status_registered_ = true;

    register_ret = cudaHostRegister(
        params.doorbell_ptr,
        sizeof(std::uint64_t),
        cudaHostRegisterIoMemory);
    if (register_ret != cudaSuccess) {
      cudaHostUnregister(params.status_ptr);
      munmap(params.status_ptr, params.status_offset + sizeof(std::uint64_t));
      munmap(params.doorbell_ptr, sizeof(std::uint64_t));
      throw std::runtime_error("Failed to cudaHostRegister doorbell pointer");
    }
    doorbell_registered_ = true;

    streams_.tx_buffer = params.buffer;
    streams_.tx_doorbell_ptr = params.doorbell_ptr;
    streams_.tx_status_ptr = reinterpret_cast<std::uint64_t*>(
        reinterpret_cast<std::uintptr_t>(params.status_ptr) + params.status_offset);
    streams_.tx_allocation_size = allocation_size;
    streams_.tx_doorbell_base_ptr = params.doorbell_base_ptr;
    streams_.tx_status_offset = params.status_offset;

    outgoing_params_ = params;
  }

  void cleanup() {
    if (doorbell_registered_ && outgoing_params_.doorbell_ptr) {
      cudaHostUnregister(outgoing_params_.doorbell_ptr);
    }
    if (status_registered_ && outgoing_params_.status_ptr) {
      cudaHostUnregister(outgoing_params_.status_ptr);
    }
    if (outgoing_params_.status_ptr) {
      munmap(outgoing_params_.status_ptr,
             outgoing_params_.status_offset + sizeof(std::uint64_t));
    }
    if (outgoing_params_.doorbell_base_ptr) {
      munmap(outgoing_params_.doorbell_base_ptr,
             outgoing_params_.doorbell_ptr - outgoing_params_.doorbell_base_ptr);
    }

    qm::stream::sync::unregister_incoming_stream(in_stream_id_);
    qm::stream::sync::unregister_outgoing_stream(out_stream_id_);
  }

  std::uint16_t in_stream_id_;
  std::uint16_t out_stream_id_;
  std::size_t buffer_count_;

  // OPNIC driver requires separate file descriptors per stream
  std::shared_ptr<qm::stream::DriverFileContext> rx_driver_context_;
  std::shared_ptr<qm::stream::DriverFileContext> tx_driver_context_;
  std::unique_ptr<qm::stream::DriverConfigurator> rx_configurator_;
  std::unique_ptr<qm::stream::DriverConfigurator> tx_configurator_;

  std::unique_ptr<qm::internal::GpuAlignedMemory> rx_buffer_mem_;
  std::unique_ptr<qm::internal::GpuAlignedMemory> rx_pi_mem_;
  std::unique_ptr<qm::internal::GpuAlignedMemory> tx_buffer_mem_;

  qm::stream::OutgoingStreamParams outgoing_params_{};
  bool status_registered_ = false;
  bool doorbell_registered_ = false;

  OpnicDirectStreams streams_{};
};

/// @brief Default alias for the original hello_opx wire shape.
/// Preserves the pre-template name so `gpu/opnic_bridge.cu` and any other
/// callers that hard-code the RPC packet pair compile unchanged.
using OpnicDirectContext = OpnicDirectContextT<RPCInputPacket, RPCOutputPacket>;
