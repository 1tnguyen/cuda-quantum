/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SocketUtils.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Process.h"

#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <utility>

namespace {

using cudaq_internal::device_call::networkToHost16;
using cudaq_internal::device_call::TcpSocket;

int noSignalSendFlags() {
#ifdef MSG_NOSIGNAL
  return MSG_NOSIGNAL;
#else
  return 0;
#endif
}

void configureNoSigPipe(int fd) {
#if !defined(MSG_NOSIGNAL) && defined(SO_NOSIGPIPE)
  int one = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
  (void)fd;
#endif
}

void closeSocket(int &fd) {
  if (fd < 0)
    return;
  (void)llvm::sys::Process::SafelyCloseFileDescriptor(fd);
  fd = TcpSocket::InvalidSocket;
}

struct AddressInfo {
  addrinfo *head = nullptr;

  AddressInfo() = default;

  ~AddressInfo() {
    if (head)
      ::freeaddrinfo(head);
  }

  AddressInfo(const AddressInfo &) = delete;
  AddressInfo &operator=(const AddressInfo &) = delete;
};

bool getAddressInfo(const std::string &host, std::uint16_t port, int flags,
                    AddressInfo &addresses) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = flags;

  const std::string portText = std::to_string(port);
  const char *hostName = host.empty() || host == "*" ? nullptr : host.c_str();
  if (::getaddrinfo(hostName, portText.c_str(), &hints, &addresses.head) != 0) {
    errno = EINVAL;
    return false;
  }
  return true;
}

std::uint16_t portFromAddress(const sockaddr *address) {
  if (!address)
    return 0;
  if (address->sa_family == AF_INET) {
    const auto *addr = reinterpret_cast<const sockaddr_in *>(address);
    return networkToHost16(addr->sin_port);
  }
  if (address->sa_family == AF_INET6) {
    const auto *addr = reinterpret_cast<const sockaddr_in6 *>(address);
    return networkToHost16(addr->sin6_port);
  }
  return 0;
}

std::uint16_t getBoundPort(int fd) {
  sockaddr_storage storage{};
  socklen_t length = sizeof(storage);
  if (::getsockname(fd, reinterpret_cast<sockaddr *>(&storage), &length) != 0)
    return 0;
  return portFromAddress(reinterpret_cast<const sockaddr *>(&storage));
}

} // namespace

namespace cudaq_internal::device_call {

TcpSocket::TcpSocket(int socketFd) : socketFd(socketFd) {}

TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket &&other) noexcept
    : socketFd(std::exchange(other.socketFd, InvalidSocket)) {}

TcpSocket &TcpSocket::operator=(TcpSocket &&other) noexcept {
  if (this != &other) {
    close();
    socketFd = std::exchange(other.socketFd, InvalidSocket);
  }
  return *this;
}

int TcpSocket::release() noexcept {
  return std::exchange(socketFd, InvalidSocket);
}

void TcpSocket::close() noexcept { closeSocket(socketFd); }

std::optional<std::uint16_t>
TcpSocket::listen(const std::string &host, std::uint16_t port, int backlog) {
  close();

  AddressInfo addresses;
  if (!getAddressInfo(host, port, AI_PASSIVE, addresses))
    return std::nullopt;

  int lastErrno = 0;
  for (addrinfo *addr = addresses.head; addr; addr = addr->ai_next) {
    TcpSocket candidate(
        ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol));
    if (!candidate.isValid()) {
      lastErrno = errno;
      continue;
    }

    if (!candidate.setReuseAddress()) {
      lastErrno = errno;
      continue;
    }

    if (::bind(candidate.native(), addr->ai_addr, addr->ai_addrlen) != 0 ||
        ::listen(candidate.native(), backlog) != 0) {
      lastErrno = errno;
      continue;
    }

    std::uint16_t boundPort = getBoundPort(candidate.native());
    if (boundPort == 0) {
      lastErrno = errno;
      continue;
    }

    socketFd = candidate.release();
    return boundPort;
  }

  errno = lastErrno ? lastErrno : EINVAL;
  return std::nullopt;
}

bool TcpSocket::connect(const std::string &host, std::uint16_t port,
                        std::uint64_t timeoutMs) {
  close();
  if (host.empty() || host == "*") {
    errno = EINVAL;
    return false;
  }

  AddressInfo addresses;
  if (!getAddressInfo(host, port, 0, addresses))
    return false;

  int lastErrno = 0;
  for (addrinfo *addr = addresses.head; addr; addr = addr->ai_next) {
    TcpSocket candidate(
        ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol));
    if (!candidate.isValid()) {
      lastErrno = errno;
      continue;
    }

    candidate.setTimeout(timeoutMs);
    if (llvm::sys::RetryAfterSignal(-1, [&] {
          return ::connect(candidate.native(), addr->ai_addr, addr->ai_addrlen);
        }) != 0) {
      lastErrno = errno;
      continue;
    }

    if (!candidate.setNoDelay()) {
      lastErrno = errno;
      continue;
    }

    socketFd = candidate.release();
    return true;
  }

  errno = lastErrno ? lastErrno : EINVAL;
  return false;
}

TcpSocket TcpSocket::accept() const {
  if (!isValid())
    return {};

  TcpSocket client(llvm::sys::RetryAfterSignal(
      -1, [&] { return ::accept(socketFd, nullptr, nullptr); }));
  if (client.isValid())
    (void)client.setNoDelay();
  return client;
}

void TcpSocket::setTimeout(std::uint64_t timeoutMs) {
  if (!isValid())
    return;
  configureNoSigPipe(socketFd);
  timeval timeout{};
  timeout.tv_sec = static_cast<long>(timeoutMs / 1000);
  timeout.tv_usec = static_cast<long>((timeoutMs % 1000) * 1000);
  (void)::setsockopt(socketFd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout));
  (void)::setsockopt(socketFd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                     sizeof(timeout));
}

bool TcpSocket::setNoDelay() {
  if (!isValid())
    return false;
  int one = 1;
  return ::setsockopt(socketFd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) ==
         0;
}

bool TcpSocket::setReuseAddress() {
  if (!isValid())
    return false;
  int one = 1;
  return ::setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) ==
         0;
}

std::ptrdiff_t TcpSocket::writeSome(const void *buffer, std::size_t bytes) {
  if (!isValid()) {
    errno = EBADF;
    return -1;
  }

  configureNoSigPipe(socketFd);
  const auto written =
      llvm::sys::RetryAfterSignal(static_cast<ssize_t>(-1), [&] {
        return ::send(socketFd, buffer, bytes, noSignalSendFlags());
      });
  return static_cast<std::ptrdiff_t>(written);
}

bool TcpSocket::writeExact(const void *buffer, std::size_t bytes) {
  const auto *data = static_cast<const std::uint8_t *>(buffer);
  std::size_t offset = 0;
  while (offset < bytes) {
    const std::ptrdiff_t written = writeSome(data + offset, bytes - offset);
    if (written <= 0)
      return false;
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

std::ptrdiff_t TcpSocket::readSome(void *buffer, std::size_t bytes) {
  if (!isValid()) {
    errno = EBADF;
    return -1;
  }

  const auto read = llvm::sys::RetryAfterSignal(static_cast<ssize_t>(-1), [&] {
    return ::recv(socketFd, buffer, bytes, 0);
  });
  return static_cast<std::ptrdiff_t>(read);
}

bool TcpSocket::readExact(void *buffer, std::size_t bytes) {
  auto *data = static_cast<std::uint8_t *>(buffer);
  std::size_t offset = 0;
  while (offset < bytes) {
    const std::ptrdiff_t read = readSome(data + offset, bytes - offset);
    if (read == 0)
      return false;
    if (read < 0)
      return false;
    offset += static_cast<std::size_t>(read);
  }
  return true;
}

SocketWaitStatus
TcpSocket::waitForReadable(std::chrono::milliseconds interval) const {
  if (!isValid()) {
    errno = EBADF;
    return SocketWaitStatus::Error;
  }

  pollfd pfd{};
  pfd.fd = socketFd;
  pfd.events = POLLIN;
  const int timeoutMs = static_cast<int>(interval.count());
  const int rc = llvm::sys::RetryAfterSignal(
      -1, [&] { return ::poll(&pfd, 1, timeoutMs); });
  if (rc == 0)
    return SocketWaitStatus::Timeout;
  if (rc < 0)
    return SocketWaitStatus::Error;
  if ((pfd.revents & POLLIN) != 0)
    return SocketWaitStatus::Ready;
  return SocketWaitStatus::Error;
}

std::uint32_t hostToNetwork32(std::uint32_t value) {
  return llvm::support::endian::byte_swap(value, llvm::endianness::big);
}

std::uint32_t networkToHost32(std::uint32_t value) {
  return llvm::support::endian::byte_swap(value, llvm::endianness::big);
}

std::uint16_t hostToNetwork16(std::uint16_t value) {
  return llvm::support::endian::byte_swap(value, llvm::endianness::big);
}

std::uint16_t networkToHost16(std::uint16_t value) {
  return llvm::support::endian::byte_swap(value, llvm::endianness::big);
}

bool writeLengthPrefixedFrame(TcpSocket &socket, const void *frame,
                              std::size_t frameLen) {
  if (frameLen > std::numeric_limits<std::uint32_t>::max())
    return false;
  const std::uint32_t networkLen =
      hostToNetwork32(static_cast<std::uint32_t>(frameLen));
  return socket.writeExact(&networkLen, sizeof(networkLen)) &&
         socket.writeExact(frame, frameLen);
}

bool readLengthPrefixedFrame(TcpSocket &socket,
                             std::vector<std::uint8_t> &frame,
                             std::uint64_t maxFrameLen) {
  std::uint32_t networkLen = 0;
  if (!socket.readExact(&networkLen, sizeof(networkLen)))
    return false;
  const std::uint32_t frameLen = networkToHost32(networkLen);
  if (frameLen > maxFrameLen)
    return false;
  frame.assign(frameLen, 0);
  return socket.readExact(frame.data(), frame.size());
}

bool readLengthPrefixedFrame(TcpSocket &socket, void *frame,
                             std::size_t frameCapacity,
                             std::uint32_t &frameLen) {
  std::uint32_t networkLen = 0;
  if (!socket.readExact(&networkLen, sizeof(networkLen)))
    return false;
  frameLen = networkToHost32(networkLen);
  if (frameLen > frameCapacity)
    return false;
  return socket.readExact(frame, frameLen);
}

} // namespace cudaq_internal::device_call
