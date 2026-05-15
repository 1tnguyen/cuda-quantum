/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Test-only helper for building small CUDA device-call service shims. This
// generates the service factory consumed by the runtime boundary, but the macro
// surface itself is not part of the device_call ABI or runtime API.

#if !defined(__CUDACC__)

// This header may be included while compiling the host-side CUDA-Q application
// or a natural library header with a non-CUDA compiler. In that mode the
// registration macros intentionally disappear: only the CUDA shim translation
// unit is expected to emit realtime table setup code.
#define CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(name)
#define CUDAQ_DEVICE_CALL_EXPORT(function)
#define CUDAQ_DEVICE_CALL_LIBRARY_END()

#else

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq_internal/device_call/DeviceCallService.h"

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>

namespace cudaq_internal::device_call::detail {

constexpr std::int32_t InvalidPayload = 101;
constexpr std::int32_t ResultTooSmall = 102;

// Canonical payload storage for the first prototype ABI. References and
// cv-qualification are erased before packing/unpacking so the wire format uses
// value semantics for plain scalar arguments.
template <typename T>
using scalar_storage_t = std::remove_cvref_t<T>;

// --- Concepts ----------------------------------------------------------------
//
// The initial device_call ABI is deliberately narrow. It supports scalar
// integer and floating-point arguments/results, plus void results. Flat array
// arguments use a pointer immediately followed by an integral element count.

template <typename T>
concept SupportedScalar =
    !std::is_reference_v<T> && !std::is_pointer_v<scalar_storage_t<T>> &&
    ((std::is_integral_v<scalar_storage_t<T>> &&
      sizeof(scalar_storage_t<T>) <= sizeof(std::uint64_t)) ||
     std::is_same_v<scalar_storage_t<T>, float> ||
     std::is_same_v<scalar_storage_t<T>, double>);

template <typename T>
concept SupportedArrayElement =
    !std::is_reference_v<T> && !std::is_pointer_v<scalar_storage_t<T>> &&
    (std::is_same_v<scalar_storage_t<T>, bool> ||
     (std::is_integral_v<scalar_storage_t<T>> &&
      (sizeof(scalar_storage_t<T>) == 1 || sizeof(scalar_storage_t<T>) == 4)) ||
     std::is_same_v<scalar_storage_t<T>, float> ||
     std::is_same_v<scalar_storage_t<T>, double>);

template <typename T>
struct is_supported_array_pointer : std::false_type {};

template <typename T>
struct is_supported_array_pointer<T *>
    : std::bool_constant<SupportedArrayElement<T>> {};

template <typename T>
struct is_supported_array_pointer<const T *>
    : std::bool_constant<SupportedArrayElement<T>> {};

template <typename T>
concept SupportedArrayPointer =
    is_supported_array_pointer<scalar_storage_t<T>>::value;

template <typename T>
struct array_element;

template <typename T>
struct array_element<T *> {
  using type = T;
};

template <typename T>
struct array_element<const T *> {
  using type = T;
};

template <typename T>
using array_element_t = typename array_element<scalar_storage_t<T>>::type;

template <typename T>
concept SupportedArrayLength =
    SupportedScalar<T> && std::is_integral_v<scalar_storage_t<T>> &&
    !std::is_same_v<scalar_storage_t<T>, bool>;

// Pack predicate: a sequence of scalar arguments where any supported pointer
// must be immediately followed by an integral element count. Expressed as a
// recursive metafunction because the array-pointer-then-length pairing is not a
// fold expression.
template <typename... Args>
struct supported_arguments;

template <>
struct supported_arguments<> : std::true_type {};

template <typename First>
struct supported_arguments<First> : std::bool_constant<SupportedScalar<First>> {
};

template <typename First, typename Second, typename... Rest>
struct supported_arguments<First, Second, Rest...>
    : std::bool_constant<SupportedArrayPointer<First>
                             ? (SupportedArrayLength<Second>
                                    &&supported_arguments<Rest...>::value)
                             : (SupportedScalar<First> &&supported_arguments<
                                   Second, Rest...>::value)> {};

template <typename... Args>
concept SupportedArguments = supported_arguments<Args...>::value;

template <typename... Args>
struct payload_argument_count;

template <>
struct payload_argument_count<> : std::integral_constant<std::uint8_t, 0> {};

template <typename First>
struct payload_argument_count<First> : std::integral_constant<std::uint8_t, 1> {
};

template <typename First, typename Second, typename... Rest>
struct payload_argument_count<First, Second, Rest...>
    : std::integral_constant<
          std::uint8_t,
          SupportedArrayPointer<First>
              ? static_cast<std::uint8_t>(
                    1 + payload_argument_count<Rest...>::value)
              : static_cast<std::uint8_t>(
                    1 + payload_argument_count<Second, Rest...>::value)> {};

// --- Compile-time helpers ----------------------------------------------------

template <typename T>
__host__ __device__ constexpr std::uint64_t scalarAlignment() {
  constexpr auto size = sizeof(scalar_storage_t<T>);
  if constexpr (size >= sizeof(std::uint64_t))
    return sizeof(std::uint64_t);
  else if constexpr (size >= sizeof(std::uint32_t))
    return sizeof(std::uint32_t);
  else if constexpr (size >= sizeof(std::uint16_t))
    return sizeof(std::uint16_t);
  else
    return 1;
}

template <typename T>
struct dependent_false : std::false_type {};

// Function ids are derived from the source-level function name used by
// cudaq::device_call. The same hash must be usable in host and device code
// because table initialization runs in a CUDA kernel.
__host__ __device__ constexpr std::uint32_t fnv1aHash(const char *name) {
  std::uint32_t hash = 2166136261u;
  for (; *name; ++name) {
    hash ^= static_cast<std::uint8_t>(*name);
    hash *= 16777619u;
  }
  return hash;
}

template <typename T>
  requires SupportedScalar<T>
__host__ __device__ constexpr std::uint8_t payloadTypeId() {
  using U = scalar_storage_t<T>;
  if constexpr (std::is_integral_v<U> && sizeof(U) == 1)
    return CUDAQ_TYPE_UINT8;
  else if constexpr (std::is_integral_v<U> && sizeof(U) <= 4)
    return CUDAQ_TYPE_INT32;
  else if constexpr (std::is_integral_v<U> && sizeof(U) == 8)
    return CUDAQ_TYPE_INT64;
  else if constexpr (std::is_same_v<U, float>)
    return CUDAQ_TYPE_FLOAT32;
  else if constexpr (std::is_same_v<U, double>)
    return CUDAQ_TYPE_FLOAT64;
  else
    static_assert(dependent_false<U>::value,
                  "unsupported cudaq device_call scalar type");
}

template <typename T>
  requires SupportedArrayElement<T>
__host__ __device__ constexpr std::uint8_t payloadArrayTypeId() {
  using U = scalar_storage_t<T>;
  if constexpr (std::is_same_v<U, bool> ||
                (std::is_integral_v<U> && sizeof(U) == 1))
    return CUDAQ_TYPE_ARRAY_UINT8;
  else if constexpr (std::is_integral_v<U> && sizeof(U) == 4)
    return CUDAQ_TYPE_ARRAY_INT32;
  else if constexpr (std::is_same_v<U, float>)
    return CUDAQ_TYPE_ARRAY_FLOAT32;
  else if constexpr (std::is_same_v<U, double>)
    return CUDAQ_TYPE_ARRAY_FLOAT64;
  else
    static_assert(dependent_false<U>::value,
                  "unsupported cudaq device_call dynamic array element type");
}

// --- Decode helpers ----------------------------------------------------------

template <typename T>
__device__ scalar_storage_t<T> loadScalar(const std::uint8_t *bytes,
                                          std::uint64_t offset) {
  scalar_storage_t<T> value;
  std::memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

template <typename T>
__device__ void storeScalar(void *bytes, const T &value) {
  std::memcpy(bytes, &value, sizeof(scalar_storage_t<T>));
}

template <typename T>
__device__ void zeroObject(T &object) {
  std::memset(&object, 0, sizeof(T));
}

// Aligns `offset` upward to `alignment` (a power of two) and validates that
// the result still lies within `argLen`.
__device__ inline bool alignOffset(std::uint64_t &offset,
                                   std::uint64_t alignment,
                                   std::uint64_t argLen) {
  if (alignment <= 1)
    return offset <= argLen;
  std::uint64_t addend = alignment - 1;
  if (offset > ~std::uint64_t{0} - addend)
    return false;
  offset = (offset + addend) & ~addend;
  return offset <= argLen;
}

// Decodes a single scalar of payload type `T` from `args`, advancing `offset`.
// Returns 0 on success or `InvalidPayload` on misalignment / overflow.
template <typename T>
__device__ inline std::int32_t
decodeScalar(const std::uint8_t *args, std::uint64_t argLen,
             std::uint64_t &offset, scalar_storage_t<T> &out) {
  using S = scalar_storage_t<T>;
  if (!alignOffset(offset, scalarAlignment<T>(), argLen) ||
      sizeof(S) > argLen - offset)
    return InvalidPayload;
  out = loadScalar<T>(args, offset);
  offset += sizeof(S);
  return 0;
}

// --- Schema fillers ----------------------------------------------------------

template <typename T>
  requires SupportedArrayElement<T>
__device__ void fillArrayTypeDesc(cudaq_type_desc_t &desc) {
  desc.type_id = payloadArrayTypeId<T>();
  desc.reserved[0] = 0;
  desc.reserved[1] = 0;
  desc.reserved[2] = 0;
  desc.size_bytes = 0;
  desc.num_elements = 0;
}

template <typename T>
__device__ void fillTypeDesc(cudaq_type_desc_t &desc) {
  using U = scalar_storage_t<T>;
  if constexpr (SupportedArrayPointer<T>) {
    fillArrayTypeDesc<array_element_t<T>>(desc);
  } else {
    desc.reserved[0] = 0;
    desc.reserved[1] = 0;
    desc.reserved[2] = 0;
    desc.type_id = payloadTypeId<U>();
    desc.size_bytes = sizeof(U);
    desc.num_elements = 1;
  }
}

template <std::size_t SchemaIndex>
__device__ void fillArgumentSchema(cudaq_handler_schema_t &) {}

template <std::size_t SchemaIndex, typename First>
__device__ void fillArgumentSchema(cudaq_handler_schema_t &schema) {
  fillTypeDesc<First>(schema.args[SchemaIndex]);
}

template <std::size_t SchemaIndex, typename First, typename Second,
          typename... Rest>
__device__ void fillArgumentSchema(cudaq_handler_schema_t &schema) {
  fillTypeDesc<First>(schema.args[SchemaIndex]);
  if constexpr (SupportedArrayPointer<First>)
    fillArgumentSchema<SchemaIndex + 1, Rest...>(schema);
  else
    fillArgumentSchema<SchemaIndex + 1, Second, Rest...>(schema);
}

// Common entry-header initialization shared by both wrapper kinds.
__device__ inline void fillEntryHeader(cudaq_function_entry_t &entry,
                                       void *fnPtr, const char *functionName) {
  zeroObject(entry);
  entry.handler.device_fn_ptr = fnPtr;
  entry.function_id = fnv1aHash(functionName);
  entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
}

// --- Adapter wrappers --------------------------------------------------------

// Adapter from the realtime handler ABI to the user's natural device function.
// `Fn` is a function pointer non-type template parameter (C++20 `auto`), so
// the wrapper is fully typed and can infer argument/result schemas from the
// C++ signature.
//
// Important build assumption: this wrapper directly calls Fn. Therefore the
// shim translation unit must have the natural CUDA implementation available in
// the same CUDA device module, e.g. by including a .cuh implementation body.
// A device function that exists only in an already-linked CUDA shared library
// cannot be resolved by this direct-call wrapper.
template <auto Fn>
struct DeviceCallWrapper;

template <typename R, typename... Args, R (*Fn)(Args...)>
  requires(sizeof...(Args) <= 8) && SupportedArguments<Args...> &&
          (std::is_void_v<R> || SupportedScalar<R>)
struct DeviceCallWrapper<Fn> {
  __device__ static std::int32_t call(const void *input, void *output,
                                      std::uint32_t argLen,
                                      std::uint32_t maxResultLen,
                                      std::uint32_t *resultLen) {
    // The realtime payload is exactly the canonical device-call argument
    // buffer. Schema matching is a future manager responsibility established
    // during initialization/registration.
    if (!resultLen || (argLen > 0 && !input))
      return InvalidPayload;

    auto *args = static_cast<const std::uint8_t *>(input);
    return decodeAndInvoke<0>(args, argLen, 0, output, maxResultLen, resultLen);
  }

private:
  template <typename... Decoded>
  __device__ static std::int32_t
  invokeDecoded(void *output, std::uint32_t maxResultLen,
                std::uint32_t *resultLen, Decoded... decoded) {
    // The realtime ABI always reports an explicit result length. Void
    // functions produce no bytes; scalar functions store one packed result.
    if constexpr (std::is_void_v<R>) {
      Fn(decoded...);
      *resultLen = 0;
      return 0;
    } else {
      if (!output || maxResultLen < sizeof(scalar_storage_t<R>))
        return ResultTooSmall;
      R result = Fn(decoded...);
      storeScalar(output, result);
      *resultLen = sizeof(scalar_storage_t<R>);
      return 0;
    }
  }

  template <std::size_t Index, typename... Decoded>
  __device__ static std::int32_t
  decodeAndInvoke(const std::uint8_t *args, std::uint64_t argLen,
                  std::uint64_t offset, void *output,
                  std::uint32_t maxResultLen, std::uint32_t *resultLen,
                  Decoded... decoded) {
    if constexpr (Index == sizeof...(Args)) {
      if (offset != argLen)
        return InvalidPayload;
      return invokeDecoded(output, maxResultLen, resultLen, decoded...);
    } else {
      using Arg = std::tuple_element_t<Index, std::tuple<Args...>>;
      if constexpr (SupportedArrayPointer<Arg>) {
        static_assert(Index + 1 < sizeof...(Args),
                      "cudaq device_call array pointer argument requires a "
                      "following element count");
        using LengthArg = std::tuple_element_t<Index + 1, std::tuple<Args...>>;
        static_assert(SupportedArrayLength<LengthArg>,
                      "cudaq device_call array pointer argument requires a "
                      "following integral element count");
        using Element = scalar_storage_t<array_element_t<Arg>>;
        std::uint64_t elementCount = 0;
        if (auto rc =
                decodeScalar<std::uint64_t>(args, argLen, offset, elementCount))
          return rc;
        if (elementCount > (argLen - offset) / sizeof(Element))
          return InvalidPayload;
        const Element *constElements =
            reinterpret_cast<const Element *>(args + offset);
        offset += elementCount * sizeof(Element);
        using Pointer = scalar_storage_t<Arg>;
        Pointer elements = nullptr;
        if constexpr (std::is_const_v<std::remove_pointer_t<Pointer>>)
          elements = constElements;
        else
          elements = const_cast<Element *>(constElements);
        return decodeAndInvoke<Index + 2>(
            args, argLen, offset, output, maxResultLen, resultLen, decoded...,
            elements, static_cast<scalar_storage_t<LengthArg>>(elementCount));
      } else {
        scalar_storage_t<Arg> value;
        if (auto rc = decodeScalar<Arg>(args, argLen, offset, value))
          return rc;
        return decodeAndInvoke<Index + 1>(args, argLen, offset, output,
                                          maxResultLen, resultLen, decoded...,
                                          value);
      }
    }
  }
};

// --- Entry fillers ----------------------------------------------------------

template <auto Fn>
struct DeviceCallEntry;

// Populate a realtime function-table entry from the natural C++ signature.
// This runs in a CUDA initialization kernel so taking the wrapper's device
// function address happens in device code, matching the realtime dispatch
// contract.
template <typename R, typename... Args, R (*Fn)(Args...)>
struct DeviceCallEntry<Fn> {
  __device__ static void fill(cudaq_function_entry_t &entry,
                              const char *functionName) {
    fillEntryHeader(entry,
                    reinterpret_cast<void *>(&DeviceCallWrapper<Fn>::call),
                    functionName);
    entry.schema.num_args = payload_argument_count<Args...>::value;
    entry.schema.num_results = std::is_void_v<R> ? 0 : 1;
    fillArgumentSchema<0, Args...>(entry.schema);
    if constexpr (!std::is_void_v<R>)
      fillTypeDesc<R>(entry.schema.results[0]);
  }
};

template <auto Fn>
__device__ void fillEntry(cudaq_function_entry_t &entry,
                          const char *functionName) {
  DeviceCallEntry<Fn>::fill(entry, functionName);
}

inline cudaError_t reportCudaError(cudaError_t err, const char *expr) {
  if (err == cudaSuccess)
    return cudaSuccess;
  std::fprintf(stderr, "%s failed: %s\n", expr, cudaGetErrorString(err));
  return err;
}

// --- Generated service traits -----------------------------------------------
//
// Common host-side machinery for a generated realtime device_call service.
// Parameterized by the per-TU table-init kernel and entry count, so each
// translation unit gets its own instantiation (and its own dispatch stream).
template <auto InitTableKernel, std::uint32_t Count>
struct GeneratedDeviceCallService {
  static inline cudaStream_t dispatchStream = nullptr;

  static cudaError_t getDispatchStream(cudaStream_t *stream) {
    if (!stream)
      return cudaErrorInvalidValue;
    if (!dispatchStream) {
      cudaError_t err =
          cudaStreamCreateWithFlags(&dispatchStream, cudaStreamNonBlocking);
      if (err != cudaSuccess)
        return err;
    }
    *stream = dispatchStream;
    return cudaSuccess;
  }

  static void launchDispatchKernel(
      volatile std::uint64_t *rxFlags, volatile std::uint64_t *txFlags,
      std::uint8_t *rxData, std::uint8_t *txData, std::size_t rxStrideSize,
      std::size_t txStrideSize, cudaq_function_entry_t *functionTable,
      std::size_t functionCount, volatile int *shutdownFlag,
      std::uint64_t *stats, std::size_t numSlots, std::uint32_t numBlocks,
      std::uint32_t threadsPerBlock, cudaStream_t /*stream*/) {
    cudaStream_t launchStream = nullptr;
    if (reportCudaError(getDispatchStream(&launchStream),
                        "getDispatchStream") != cudaSuccess)
      return;
    cudaq_launch_dispatch_kernel_regular(
        rxFlags, txFlags, rxData, txData, rxStrideSize, txStrideSize,
        functionTable, functionCount, shutdownFlag, stats, numSlots, numBlocks,
        threadsPerBlock, launchStream);
    reportCudaError(cudaGetLastError(), "cudaq_launch_dispatch_kernel_regular");
  }

  static cudaError_t synchronizeDispatchKernel() {
    if (!dispatchStream)
      return cudaSuccess;
    cudaError_t syncErr = cudaStreamSynchronize(dispatchStream);
    cudaError_t destroyErr = cudaStreamDestroy(dispatchStream);
    dispatchStream = nullptr;
    return syncErr != cudaSuccess ? syncErr : destroyErr;
  }

  static int populateTable(cudaq_function_entry_t *entries,
                           std::uint32_t capacity, cudaStream_t stream) {
    if (!entries || capacity < Count)
      return 1;
    InitTableKernel<<<1, 1, 0, stream>>>(entries);
    return reportCudaError(cudaGetLastError(),
                           "device_call_init_table launch") != cudaSuccess;
  }

  // Service vtable thunks.
  static int serviceCreate(const void *, std::size_t, void **handle) {
    if (handle)
      *handle = nullptr;
    return 0;
  }
  static int serviceDestroy(void *) { return 0; }
  static std::uint32_t serviceGetCount(void *) { return Count; }
  static int serviceStart(void *) { return 0; }
  static int serviceStop(void *) {
    return reportCudaError(synchronizeDispatchKernel(),
                           "synchronizeDispatchKernel") != cudaSuccess;
  }
  static int servicePopulateTable(void *, cudaq_function_entry_t *entries,
                                  std::uint32_t capacity, cudaStream_t stream) {
    return populateTable(entries, capacity, stream);
  }
  static cudaq_dispatch_launch_fn_t serviceGetLaunch(void *) {
    return &launchDispatchKernel;
  }
  static cudaq_device_call_dispatch_synchronize_fn_t
  serviceGetSynchronize(void *) {
    return &synchronizeDispatchKernel;
  }

  static int fillService(cudaq_realtime_device_call_service *out) {
    if (!out)
      return 1;
    std::memset(out, 0, sizeof(*out));
    out->create = &serviceCreate;
    out->destroy = &serviceDestroy;
    out->get_function_count = &serviceGetCount;
    out->populate_table = &servicePopulateTable;
    out->get_device_dispatch_launch = &serviceGetLaunch;
    out->get_device_dispatch_synchronize = &serviceGetSynchronize;
    out->start = &serviceStart;
    out->stop = &serviceStop;
    return 0;
  }
};

} // namespace cudaq_internal::device_call::detail

// --- Registration macros -----------------------------------------------------

#define CUDAQ_DEVICE_CALL_CONCAT(a, b) CUDAQ_DEVICE_CALL_CONCAT_INNER(a, b)
#define CUDAQ_DEVICE_CALL_CONCAT_INNER(a, b) a##b

#define CUDAQ_DEVICE_CALL_SERVICE_FACTORY_NAME(name)                           \
  CUDAQ_DEVICE_CALL_CONCAT(cudaq_realtime_get_service_, name)

// Begin one device-call registration table. The macro opens a private
// namespace and a CUDA kernel body; each CUDAQ_DEVICE_CALL_EXPORT invocation
// appends one entry to that kernel. Use exactly one BEGIN/END pair per shim
// translation unit in this prototype.
#define CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(name)                                  \
  namespace {                                                                  \
  int __cudaq_device_call_fill_service(                                        \
      cudaq_realtime_device_call_service *out);                                \
  }                                                                            \
  extern "C" int CUDAQ_DEVICE_CALL_SERVICE_FACTORY_NAME(name)(                 \
      cudaq_realtime_device_call_service * out) {                              \
    return __cudaq_device_call_fill_service(out);                              \
  }                                                                            \
  extern "C" int cudaq_realtime_get_service(                                   \
      cudaq_realtime_device_call_service *out) {                               \
    return CUDAQ_DEVICE_CALL_SERVICE_FACTORY_NAME(name)(out);                  \
  }                                                                            \
  namespace {                                                                  \
  enum { __cudaq_device_call_counter_begin = __COUNTER__ };                    \
  __global__ void                                                              \
  __cudaq_device_call_init_table(cudaq_function_entry_t *entries) {            \
    if (threadIdx.x != 0 || blockIdx.x != 0)                                   \
      return;

// Register one natural device function. The function name string is the
// logical dispatch key, and the function pointer is the source of truth for
// the scalar argument/result schema.
#define CUDAQ_DEVICE_CALL_EXPORT(function)                                     \
  ::cudaq_internal::device_call::detail::fillEntry<&function>(                 \
      entries[__COUNTER__ - __cudaq_device_call_counter_begin - 1],            \
      #function);

// Finish the table and export the service factory used by cudaq::realtime.
// The CUDA-Q runtime owns function table allocation and dispatch session
// lifecycle.
#define CUDAQ_DEVICE_CALL_LIBRARY_END()                                        \
  } /* close __cudaq_device_call_init_table */                                 \
  constexpr std::uint32_t __cudaq_device_call_count =                          \
      static_cast<std::uint32_t>(__COUNTER__ -                                 \
                                 __cudaq_device_call_counter_begin - 1);       \
  using __cudaq_device_call_service_traits =                                   \
      ::cudaq_internal::device_call::detail::GeneratedDeviceCallService<       \
          &__cudaq_device_call_init_table, __cudaq_device_call_count>;         \
  int __cudaq_device_call_fill_service(                                        \
      cudaq_realtime_device_call_service *out) {                               \
    return __cudaq_device_call_service_traits::fillService(out);               \
  }                                                                            \
  } /* close anonymous namespace */

#endif
