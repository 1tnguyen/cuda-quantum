/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "opnic_type.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>
#include <cstdint>

#define DEBUG_PRINT 0  // set to 1 for debug prints (adds latency)
#if DEBUG_PRINT
  #define DPRINTF(...) printf(__VA_ARGS__)
#else
  #define DPRINTF(...) ((void)0)
#endif

// Ring buffer definitions. RING_BUFFER_PAGE_SIZE must be at least
// sizeof(RPCHeader) + max(arg_len, result_len) for the application's wire
// packet shape; the hello_opx example needs 32 B (24 B header + 8 B payload)
// but qec_stim's 3-verb protocol needs 64 B (24 B + 40 B arg region). Both
// are guarded so call sites can pick the right size via a compile flag.
#ifndef RING_BUFFER_PAGE_SIZE
#define RING_BUFFER_PAGE_SIZE 32
#endif
#ifndef RING_BUFFER_NUM_PAGES
#define RING_BUFFER_NUM_PAGES 1024
#endif

//==============================================================================
// Direct OPNIC RX Kernel (Zero-Copy Path)
//
// Eliminates 3 copies from the original SDK path:
//   - No deserialize() copy to packet_cache
//   - No get() copy to local RPCInputPacket
//   - Reads directly from OPNIC DMA buffer
//
// Single remaining copy: OPNIC buffer -> dispatch ring buffer (format translation)
//
// `static` (internal linkage): this kernel is launched only by the GPU bridge
// (opnic_bridge.cu).  Internal linkage lets the header be included by multiple
// translation units without the device linker seeing duplicate definitions.
//==============================================================================
static __global__ void opnic_rx_kernel_direct(
    std::uint32_t* opnic_rx_buffer,
    volatile std::uint32_t* pi_ptr,
    std::size_t opnic_alloc_size,
    std::size_t opnic_buf_count,
    volatile std::uint64_t* rx_flags,
    std::uint8_t* rx_data,
    volatile int* shutdown_flag) {

  std::uint32_t consumer_index = 0;
  std::uint32_t ring_slot = 0;

  DPRINTF("[RX-DIRECT] Kernel started, persistent mode\n");
  DPRINTF("[RX-DIRECT] pi_ptr=%p, opnic_rx_buffer=%p, alloc_size=%lu, buf_count=%lu\n",
          pi_ptr, opnic_rx_buffer, opnic_alloc_size, opnic_buf_count);
  DPRINTF("[RX-DIRECT] Initial pi_ptr value: %u\n", *pi_ptr);

  while (!*shutdown_flag) {
    // Poll OPNIC producer index directly (replaces wait_for_packets + deserialize + get)
    while (static_cast<std::int32_t>(*pi_ptr - consumer_index) < 1) {
      if (*shutdown_flag) {
        DPRINTF("[RX-DIRECT] Shutdown during poll, exiting\n");
        return;
      }
    }

    DPRINTF("[RX-DIRECT] Packet detected! pi=%u, consumer=%u\n", *pi_ptr, consumer_index);

    // Read raw packet from OPNIC DMA buffer (zero-copy from hardware)
    std::uint32_t opnic_slot = consumer_index % opnic_buf_count;
    std::uint32_t* pkt = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(opnic_rx_buffer) + opnic_slot * opnic_alloc_size);

    DPRINTF("[RX-DIRECT] Packet at slot %u: magic=0x%08x, func_id=%u, arg_len=%u\n",
           opnic_slot, pkt[0], pkt[1], pkt[2]);

    // Check for shutdown packet (function_id == 0)
    std::uint32_t function_id = pkt[1];
    if (function_id == 0) {
      DPRINTF("[RX-DIRECT] Shutdown packet received via OPX. Setting shutdown flag.\n");
      *shutdown_flag = 1;
      __threadfence_system();
      break;
    }

    // Write RPCHeader + payload to dispatch ring buffer (single copy, format
    // translation). Wire layout (uint32 words), byte-identical to RPCHeader:
    //   [0]=magic, [1]=function_id, [2]=arg_len, [3]=request_id,
    //   [4..5]=ptp_timestamp (lo,hi), [6..]=data
    std::uint8_t* rx_slot = rx_data + ring_slot * RING_BUFFER_PAGE_SIZE;
    cudaq::realtime::RPCHeader* hdr = reinterpret_cast<cudaq::realtime::RPCHeader*>(rx_slot);
    hdr->magic = pkt[0];
    hdr->function_id = pkt[1];
    hdr->arg_len = pkt[2];
    hdr->request_id = pkt[3];
    hdr->ptp_timestamp = static_cast<std::uint64_t>(pkt[4]) |
                         (static_cast<std::uint64_t>(pkt[5]) << 32);

    // Copy payload data
    std::int32_t* payload = reinterpret_cast<std::int32_t*>(rx_slot + sizeof(cudaq::realtime::RPCHeader));
    std::uint32_t nwords = pkt[2] / sizeof(std::int32_t);
    for (std::uint32_t i = 0; i < nwords; i++) {
      payload[i] = static_cast<std::int32_t>(pkt[6 + i]);
    }

    // Signal the dispatcher that a request is ready
    // Fence ensures data writes are visible before flag is set (dispatcher runs on different SM)
    __threadfence();
    rx_flags[ring_slot] = reinterpret_cast<std::uint64_t>(rx_slot);

    ring_slot = (ring_slot + 1) % RING_BUFFER_NUM_PAGES;
    consumer_index++;
  }

  DPRINTF("[RX-DIRECT] Shutdown signal received, exiting\n");
}

//==============================================================================
// Direct OPNIC TX Kernel (Zero-Copy Path)
//
// Eliminates 2 copies from the original SDK path:
//   - No intermediate RPCOutputPacket allocation
//   - No serialize() copy to SDK data_buffer
//   - No atomicCAS lock overhead
//
// Single remaining copy: dispatch ring buffer -> OPNIC buffer (format translation)
//==============================================================================
static __global__ void opnic_tx_kernel_direct(
    std::uint32_t* opnic_tx_buffer,
    std::uint64_t* doorbell_ptr,
    std::size_t opnic_alloc_size,
    std::size_t opnic_buf_count,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* tx_data,
    volatile int* shutdown_flag) {

  std::uint32_t ring_slot = 0;
  std::uint32_t out_index = 0;

  DPRINTF("[TX-DIRECT] Kernel started, persistent mode\n");
  DPRINTF("[TX-DIRECT] doorbell_ptr=%p, opnic_tx_buffer=%p\n", doorbell_ptr, opnic_tx_buffer);

  while (!*shutdown_flag) {
    // Poll with volatile read only - no fence needed for polling
    if (tx_flags[ring_slot] != 0) {
      DPRINTF("[TX-DIRECT] Response detected at slot %u, flags=0x%llx\n", 
             ring_slot, (unsigned long long)tx_flags[ring_slot]);
      std::uint8_t* tx_slot = tx_data + ring_slot * RING_BUFFER_PAGE_SIZE;
      const cudaq::realtime::RPCResponse* resp =
          reinterpret_cast<const cudaq::realtime::RPCResponse*>(tx_slot);

      // Validate response before sending
      if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE &&
          resp->status == 0 && resp->result_len >= sizeof(std::int32_t)) {

        // Write directly to OPNIC outgoing buffer (no intermediate packet)
        std::uint32_t opnic_slot = out_index % opnic_buf_count;
        std::uint32_t* out = reinterpret_cast<std::uint32_t*>(
            reinterpret_cast<std::uint8_t*>(opnic_tx_buffer) + opnic_slot * opnic_alloc_size);

        // Wire layout matches RPCResponse (6 header words + result):
        //   [0]=magic, [1]=status, [2]=result_len, [3]=request_id,
        //   [4..5]=ptp_timestamp (lo,hi), [6..]=result
        out[0] = resp->magic;
        out[1] = static_cast<std::uint32_t>(resp->status);
        out[2] = resp->result_len;
        out[3] = resp->request_id;
        out[4] = static_cast<std::uint32_t>(resp->ptp_timestamp & 0xFFFFFFFFu);
        out[5] = static_cast<std::uint32_t>(resp->ptp_timestamp >> 32);

        // Copy result payload
        const std::int32_t* result = reinterpret_cast<const std::int32_t*>(
            tx_slot + sizeof(cudaq::realtime::RPCResponse));
        std::uint32_t nwords = resp->result_len / sizeof(std::int32_t);
        for (std::uint32_t i = 0; i < nwords; i++) {
          out[6 + i] = static_cast<std::uint32_t>(result[i]);
        }

        // Ring doorbell to send packet (requires __threadfence_system for PCIe)
        __threadfence_system();
        *doorbell_ptr = 1;

        DPRINTF("[TX-DIRECT] Doorbell rung for packet %u\n", out_index);
        out_index++;
      }

      __threadfence();
      tx_flags[ring_slot] = 0;
      ring_slot = (ring_slot + 1) % RING_BUFFER_NUM_PAGES;
    }
  }

  DPRINTF("[TX-DIRECT] Shutdown signal received, exiting\n");
}