/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace cudaq_internal::device_call {

enum class SocketWaitStatus { Ready, Timeout, Error };

class TcpSocket {
public:
  static constexpr int InvalidSocket = -1;

  TcpSocket() = default;
  explicit TcpSocket(int socketFd);
  ~TcpSocket();

  TcpSocket(const TcpSocket &) = delete;
  TcpSocket &operator=(const TcpSocket &) = delete;
  TcpSocket(TcpSocket &&other) noexcept;
  TcpSocket &operator=(TcpSocket &&other) noexcept;

  bool isValid() const { return socketFd != InvalidSocket; }
  int native() const { return socketFd; }
  int release() noexcept;
  void close() noexcept;

  std::optional<std::uint16_t> listen(const std::string &host,
                                      std::uint16_t port, int backlog);
  bool connect(const std::string &host, std::uint16_t port,
               std::uint64_t timeoutMs);
  TcpSocket accept() const;

  void setTimeout(std::uint64_t timeoutMs);
  bool setNoDelay();
  bool setReuseAddress();

  std::ptrdiff_t writeSome(const void *buffer, std::size_t bytes);
  bool writeExact(const void *buffer, std::size_t bytes);

  template <typename StopPredicate>
  bool writeExact(const void *buffer, std::size_t bytes,
                  StopPredicate shouldStop) {
    const auto *data = static_cast<const std::uint8_t *>(buffer);
    std::size_t offset = 0;
    while (offset < bytes && !shouldStop()) {
      const std::ptrdiff_t written = writeSome(data + offset, bytes - offset);
      if (written <= 0)
        return false;
      offset += static_cast<std::size_t>(written);
    }
    return offset == bytes;
  }

  std::ptrdiff_t readSome(void *buffer, std::size_t bytes);
  bool readExact(void *buffer, std::size_t bytes);

  template <typename StopPredicate>
  bool readExact(void *buffer, std::size_t bytes, StopPredicate shouldStop) {
    auto *data = static_cast<std::uint8_t *>(buffer);
    std::size_t offset = 0;
    while (offset < bytes && !shouldStop()) {
      const std::ptrdiff_t read = readSome(data + offset, bytes - offset);
      if (read == 0)
        return false;
      if (read < 0)
        return false;
      offset += static_cast<std::size_t>(read);
    }
    return offset == bytes;
  }

  SocketWaitStatus
  waitForReadable(std::chrono::milliseconds pollInterval) const;

  template <typename StopPredicate>
  bool waitForReadable(StopPredicate shouldStop,
                       std::chrono::milliseconds pollInterval =
                           std::chrono::milliseconds(100)) const {
    while (!shouldStop()) {
      const SocketWaitStatus status = waitForReadable(pollInterval);
      if (status == SocketWaitStatus::Ready)
        return true;
      if (status == SocketWaitStatus::Timeout)
        continue;
      return false;
    }
    return false;
  }

  template <typename StopPredicate>
  bool readExactPolling(void *buffer, std::size_t bytes,
                        StopPredicate shouldStop) {
    auto *data = static_cast<std::uint8_t *>(buffer);
    std::size_t offset = 0;
    while (offset < bytes && !shouldStop()) {
      if (!waitForReadable(shouldStop))
        return false;
      const std::ptrdiff_t read = readSome(data + offset, bytes - offset);
      if (read == 0)
        return false;
      if (read < 0)
        return false;
      offset += static_cast<std::size_t>(read);
    }
    return offset == bytes;
  }

private:
  int socketFd = InvalidSocket;
};

std::uint32_t hostToNetwork32(std::uint32_t value);

std::uint32_t networkToHost32(std::uint32_t value);

bool writeLengthPrefixedFrame(TcpSocket &socket, const void *frame,
                              std::size_t frameLen);

template <typename StopPredicate>
bool writeLengthPrefixedFrame(TcpSocket &socket, const void *frame,
                              std::size_t frameLen, StopPredicate shouldStop) {
  if (frameLen > std::numeric_limits<std::uint32_t>::max())
    return false;
  const std::uint32_t networkLen =
      hostToNetwork32(static_cast<std::uint32_t>(frameLen));
  return socket.writeExact(&networkLen, sizeof(networkLen), shouldStop) &&
         socket.writeExact(frame, frameLen, shouldStop);
}

bool readLengthPrefixedFrame(TcpSocket &socket, void *frame,
                             std::size_t frameCapacity,
                             std::uint32_t &frameLen);

template <typename StopPredicate>
bool readLengthPrefixedFrame(TcpSocket &socket,
                             std::vector<std::uint8_t> &frame,
                             std::uint64_t maxFrameLen,
                             StopPredicate shouldStop) {
  std::uint32_t networkLen = 0;
  if (!socket.readExact(&networkLen, sizeof(networkLen), shouldStop))
    return false;
  const std::uint32_t frameLen = networkToHost32(networkLen);
  if (frameLen > maxFrameLen)
    return false;
  frame.assign(frameLen, 0);
  return socket.readExact(frame.data(), frame.size(), shouldStop);
}

template <typename StopPredicate>
bool readLengthPrefixedFramePolling(TcpSocket &socket,
                                    std::vector<std::uint8_t> &frame,
                                    std::uint64_t maxFrameLen,
                                    StopPredicate shouldStop) {
  std::uint32_t networkLen = 0;
  if (!socket.readExactPolling(&networkLen, sizeof(networkLen), shouldStop))
    return false;
  const std::uint32_t frameLen = networkToHost32(networkLen);
  if (frameLen > maxFrameLen)
    return false;
  frame.assign(frameLen, 0);
  return socket.readExactPolling(frame.data(), frame.size(), shouldStop);
}

} // namespace cudaq_internal::device_call
