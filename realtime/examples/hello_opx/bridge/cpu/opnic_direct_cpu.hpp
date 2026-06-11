/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <new>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h> // getpagesize()

// Must include <array> before opnic SDK headers for nvcc compatibility
// (SDK's signature.hpp uses std::array without including the header).
#include <array>

#include <qm/opnic_driver.h> // For OPNIC_IOCTL_RESET_STREAMS
#include <opnic/stream/configurators/context.hpp>
#include <opnic/stream/configurators/driver_configurator.hpp>
#include <opnic/stream/configurators/sync.hpp>

/// @brief Host-resident OPNIC stream pointers for the CPU data path.
struct OpnicDirectCpuStreams {
  std::uint32_t *rx_buffer;          ///< OPNIC incoming DMA target (host mem)
  volatile std::uint32_t *rx_pi_ptr; ///< Producer index            (host mem)
  std::uint32_t *tx_buffer;          ///< OPNIC outgoing DMA source (host mem)
  std::uint64_t *tx_doorbell_ptr;    ///< mmap'd doorbell PCIe BAR
  std::uint64_t *tx_status_ptr;      ///< mmap'd status word (raw, post-offset)
  std::size_t rx_allocation_size;    ///< Per-packet stride in rx_buffer
  std::size_t tx_allocation_size;    ///< Per-packet stride in tx_buffer
  std::size_t buffer_count;          ///< Number of packet slots

  std::uint64_t *tx_doorbell_base_ptr; ///< Base address for cleanup
  std::size_t tx_status_offset;        ///< Status offset for cleanup
};

/// @brief Aligned, owning host buffer wrapper.
///
/// IMPORTANT: Alignment and the rounded allocation size MUST be the kernel
/// page size (`getpagesize()`). On systems with 64 KiB pages
/// (e.g. NVIDIA ARM64 GH `linux ...-64k`), the OPNIC kernel driver pins the
/// enclosing 64 KiB page via:
/// ```c++
///   pin_user_pages_fast(user_addr & PAGE_MASK, nr_pages = PAGE_ALIGN(size) >> PAGE_SHIFT)
/// ```
/// and describes it to the FPGA as a single SG entry starting at offset 0. If
/// our pointer is not aligned to the page size, say it is only 4 KiB aligned,
/// it falls at a non-zero offset within the pinned 64 KiB page and the FPGA's
/// DMA writes land at the page base (somewhere else inside our buffer), so the
/// host sees `pi == 0` and an all-zero RX buffer forever.
class HostAlignedMemory {
public:
  explicit HostAlignedMemory(std::size_t size) : size_(size) {
    const std::size_t page_size = static_cast<std::size_t>(getpagesize());
    std::size_t rounded = ((size + page_size - 1) / page_size) * page_size;
    void *p = std::aligned_alloc(page_size, rounded);
    if (!p)
      throw std::bad_alloc();
    std::memset(p, 0, rounded);
    ptr_ = p;
  }

  ~HostAlignedMemory() {
    if (ptr_)
      std::free(ptr_);
  }

  HostAlignedMemory(const HostAlignedMemory &) = delete;
  HostAlignedMemory &operator=(const HostAlignedMemory &) = delete;

  std::uintptr_t get() const { return reinterpret_cast<std::uintptr_t>(ptr_); }
  std::size_t size() const { return size_; }

private:
  void *ptr_ = nullptr;
  std::size_t size_ = 0;
};

/// @brief CPU-side context for direct OPNIC streams.
///
/// Templated on the two packet types so call sites pick their own wire
/// layout. Both types must be `QM_DECLARE_PACKET`-declared and expose
/// `::size`, `::allocation_size`, and `::pattern` (used to build the OPNIC
/// SDK sync signatures).
///
/// NOTE: The OPNIC driver requires separate file descriptors for the incoming
/// and outgoing streams.
template <typename IncomingPacket, typename OutgoingPacket>
class OpnicDirectCpuContext {
public:
  /// @param in_stream_id    OPNIC stream id for q2h (OPX -> host).
  /// @param out_stream_id   OPNIC stream id for h2q (host -> OPX).
  /// @param buffer_count    Number of ring slots per stream.
  /// @param force_reset     If true, issue OPNIC_IOCTL_RESET_STREAMS first.
  OpnicDirectCpuContext(std::uint16_t in_stream_id, std::uint16_t out_stream_id,
                        std::size_t buffer_count = 1024,
                        bool force_reset = true)
      : in_stream_id_(in_stream_id), out_stream_id_(out_stream_id),
        buffer_count_(buffer_count) {

    printf("[OPNIC-CPU] Finding available driver for RX stream...\n");
    rx_driver_context_ = qm::stream::find_available_driver();
    if (!rx_driver_context_ || !rx_driver_context_->is_valid())
      throw std::runtime_error("No OPNIC driver found for RX. Is the driver loaded?");
    printf("[OPNIC-CPU] RX driver found (fd=%d)\n", rx_driver_context_->get_fd());

    printf("[OPNIC-CPU] Finding available driver for TX stream...\n");
    tx_driver_context_ = qm::stream::find_available_driver();
    if (!tx_driver_context_ || !tx_driver_context_->is_valid())
      throw std::runtime_error("No OPNIC driver found for TX. Is the driver loaded?");
    printf("[OPNIC-CPU] TX driver found (fd=%d)\n", tx_driver_context_->get_fd());

    if (force_reset)
      reset_streams();

    rx_configurator_ = std::make_unique<qm::stream::DriverConfigurator>(*rx_driver_context_);
    tx_configurator_ = std::make_unique<qm::stream::DriverConfigurator>(*tx_driver_context_);

    setup_incoming_stream();
    setup_outgoing_stream();
  }

  ~OpnicDirectCpuContext() { cleanup(); }

  OpnicDirectCpuContext(const OpnicDirectCpuContext &) = delete;
  OpnicDirectCpuContext &operator=(const OpnicDirectCpuContext &) = delete;

  /// @brief Get the direct stream pointers for the unified host loop.
  OpnicDirectCpuStreams get_streams() const { return streams_; }

  /// @brief Perform OPX handshake synchronization.
  void sync() { qm::sync(rx_configurator_.get()); }

  /// @brief Force reset all OPNIC streams at driver level.
  void reset_streams() {
    int fd = rx_driver_context_->get_fd();
    printf("[OPNIC-CPU] Forcing stream reset via ioctl (fd=%d)...\n", fd);
    auto res = ioctl(fd, OPNIC_IOCTL_RESET_STREAMS, nullptr);
    if (res == -1)
      throw std::runtime_error("Failed to reset OPNIC streams: errno=" +
                               std::to_string(errno));
    printf("[OPNIC-CPU] Stream reset successful\n");
  }

private:
  void setup_incoming_stream() {
    printf("[OPNIC-CPU] Setting up incoming stream %u...\n", in_stream_id_);
    constexpr std::size_t packet_size = IncomingPacket::size;
    constexpr std::size_t allocation_size = IncomingPacket::allocation_size;
    const std::size_t buffer_size = allocation_size * buffer_count_;

    rx_buffer_mem_ = std::make_unique<HostAlignedMemory>(buffer_size);
    rx_pi_mem_ = std::make_unique<HostAlignedMemory>(sizeof(std::uint32_t));

    // The aligned-host allocator already zeroes the memory; the producer
    // index must start at 0.

    qm::stream::sync::register_incoming_stream(
        in_stream_id_,
        std::format("q2h{}_{}", in_stream_id_, IncomingPacket::pattern));

    auto params = rx_configurator_->setup_incoming_stream(
        in_stream_id_,
        qm::StreamType::CPU,
        reinterpret_cast<std::uint32_t *>(rx_pi_mem_->get()),
        reinterpret_cast<std::uint32_t *>(rx_buffer_mem_->get()),
        buffer_size,
        packet_size,
        allocation_size);

    streams_.rx_buffer = params.buffer;
    streams_.rx_pi_ptr = params.pi_ptr;
    streams_.rx_allocation_size = allocation_size;
    streams_.buffer_count = buffer_count_;
    printf("[OPNIC-CPU] Incoming stream %u setup complete\n", in_stream_id_);
  }

  void setup_outgoing_stream() {
    printf("[OPNIC-CPU] Setting up outgoing stream %u...\n", out_stream_id_);
    constexpr std::size_t packet_size = OutgoingPacket::size;
    constexpr std::size_t allocation_size = OutgoingPacket::allocation_size;
    const std::size_t buffer_size = allocation_size * buffer_count_;

    tx_buffer_mem_ = std::make_unique<HostAlignedMemory>(buffer_size);

    qm::stream::sync::register_outgoing_stream(
        out_stream_id_,
        std::format("h2q{}_{}", out_stream_id_, OutgoingPacket::pattern));

    auto params = tx_configurator_->setup_outgoing_stream(
        out_stream_id_,
        qm::StreamType::CPU,
        reinterpret_cast<std::uint32_t *>(tx_buffer_mem_->get()),
        buffer_size,
        packet_size,
        allocation_size);

    streams_.tx_buffer = params.buffer;
    streams_.tx_doorbell_ptr = params.doorbell_ptr;
    streams_.tx_status_ptr = reinterpret_cast<std::uint64_t *>(
        reinterpret_cast<std::uintptr_t>(params.status_ptr) +
        params.status_offset);
    streams_.tx_allocation_size = allocation_size;
    streams_.tx_doorbell_base_ptr = params.doorbell_base_ptr;
    streams_.tx_status_offset = params.status_offset;

    outgoing_params_ = params;
    printf("[OPNIC-CPU] Outgoing stream %u setup complete\n", out_stream_id_);
  }

  void cleanup() {
    if (outgoing_params_.status_ptr)
      munmap(outgoing_params_.status_ptr,
             outgoing_params_.status_offset + sizeof(std::uint64_t));
    if (outgoing_params_.doorbell_base_ptr)
      munmap(outgoing_params_.doorbell_base_ptr,
             outgoing_params_.doorbell_ptr - outgoing_params_.doorbell_base_ptr);

    qm::stream::sync::unregister_incoming_stream(in_stream_id_);
    qm::stream::sync::unregister_outgoing_stream(out_stream_id_);
  }

  std::uint16_t in_stream_id_;
  std::uint16_t out_stream_id_;
  std::size_t buffer_count_;

  std::shared_ptr<qm::stream::DriverFileContext> rx_driver_context_;
  std::shared_ptr<qm::stream::DriverFileContext> tx_driver_context_;
  std::unique_ptr<qm::stream::DriverConfigurator> rx_configurator_;
  std::unique_ptr<qm::stream::DriverConfigurator> tx_configurator_;

  std::unique_ptr<HostAlignedMemory> rx_buffer_mem_;
  std::unique_ptr<HostAlignedMemory> rx_pi_mem_;
  std::unique_ptr<HostAlignedMemory> tx_buffer_mem_;

  qm::stream::OutgoingStreamParams outgoing_params_{};

  OpnicDirectCpuStreams streams_{};
};
