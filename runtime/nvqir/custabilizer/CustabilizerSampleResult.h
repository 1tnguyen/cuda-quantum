/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include "stim/mem/simd_util.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>

namespace nvqir::custabilizer {
namespace {

/// Convert packed GPU measurement rows into CUDA-Q's chronological shot
/// strings. Work on 64 shots at a time so each input word supplies all of its
/// bits, and expand eight adjacent result bits per copy instead of one strided
/// load per character. The working set is bounded by 64 output strings.
inline cudaq::ExecutionResult
makeSampleResult(std::span<const std::uint32_t> measurements,
                 std::size_t wordsPerRow, const std::vector<bool> &reference,
                 std::size_t firstMeasurement, std::size_t firstShot,
                 std::size_t shots, bool includeSequentialData) {
  static constexpr auto byteStrings = [] {
    std::array<std::array<char, 8>, 256> table{};
    for (std::size_t value = 0; value < table.size(); ++value)
      for (std::size_t bit = 0; bit < 8; ++bit)
        table[value][bit] = '0' + ((value >> bit) & 1);
    return table;
  }();

  cudaq::ExecutionResult result;
  if (includeSequentialData)
    result.sequentialData.reserve(shots);
  const auto bitsPerShot = reference.size() - firstMeasurement;
  auto save = [&](std::string bits) {
    // Build the result in place: ExecutionResult(CountsDictionary) copies its
    // by-value argument again, which duplicates every distinct long bitstring.
    auto [entry, inserted] =
        includeSequentialData ? result.counts.try_emplace(bits, 0)
                              : result.counts.try_emplace(std::move(bits), 0);
    ++entry->second;
    if (includeSequentialData)
      result.sequentialData.push_back(std::move(bits));
  };

  // Avoid a full transpose for short results and tiny batches.
  if (bitsPerShot < 8 || shots < 32) {
    for (std::size_t s = firstShot; s < firstShot + shots; ++s) {
      std::string bits(bitsPerShot, '0');
      for (std::size_t b = 0; b < bitsPerShot; ++b) {
        const auto m = firstMeasurement + b;
        bits[b] =
            '0' + (((measurements[m * wordsPerRow + s / 32] >> (s % 32)) & 1u) ^
                   reference[m]);
      }
      save(std::move(bits));
    }
    return result;
  }

  for (std::size_t begin = 0; begin < shots; begin += 64) {
    const auto count = std::min(std::size_t{64}, shots - begin);
    const auto shot = firstShot + begin;
    const auto word = shot / 32;
    const auto shift = shot % 32;
    std::array<std::string, 64> strings;
    for (std::size_t s = 0; s < count; ++s)
      strings[s].assign(bitsPerShot, '0');

    for (std::size_t bit = 0; bit < bitsPerShot; bit += 64) {
      const auto width = std::min(std::size_t{64}, bitsPerShot - bit);
      std::array<std::uint64_t, 64> tile{};
      for (std::size_t b = 0; b < width; ++b) {
        const auto m = firstMeasurement + bit + b;
        const auto *row = measurements.data() + m * wordsPerRow;
        std::uint64_t value = row[word];
        if (word + 1 < wordsPerRow)
          value |= std::uint64_t{row[word + 1]} << 32;
        if (shift) {
          value >>= shift;
          if (word + 2 < wordsPerRow)
            value |= std::uint64_t{row[word + 2]} << (64 - shift);
        }
        tile[b] = value ^ (reference[m] ? UINT64_MAX : 0);
      }
      stim::inplace_transpose_64x64(tile.data(), 1);
      for (std::size_t s = 0; s < count; ++s) {
        auto *destination = strings[s].data() + bit;
        std::size_t b = 0;
        for (; b + 8 <= width; b += 8)
          std::memcpy(destination + b,
                      byteStrings[(tile[s] >> b) & 0xff].data(), 8);
        if (b < width)
          std::memcpy(destination + b,
                      byteStrings[(tile[s] >> b) & 0xff].data(), width - b);
      }
    }
    for (std::size_t s = 0; s < count; ++s)
      save(std::move(strings[s]));
  }
  return result;
}

} // namespace
} // namespace nvqir::custabilizer
