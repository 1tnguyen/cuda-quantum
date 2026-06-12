#!/usr/bin/env python3
"""Shared QUA packet helpers for the realtime decoder OPNIC example."""

from __future__ import annotations

from pathlib import Path

from qm import DictQuaConfig
from qm.qua import QuaArray, qua_struct


RPC_MAGIC_REQUEST = 0x43555152
RPC_MAGIC_RESPONSE = 0x43555153
CUDAQX_RT_OK = 0

RESET_DECODER_ID = -1971503368      # 0x8A7D3EF8 as signed int32
ENQUEUE_SYNDROMES_ID = -291743889   # 0xEE9C576F as signed int32
GET_CORRECTIONS_ID = -516413314     # 0xE138287E as signed int32
CONFIGURE_DECODER_ID = 0x00A0A48C

DATA_STREAM_ID = 1
CONTROL_STREAM_ID = 2
DATA_PAYLOAD_WORDS = 10
# Configure chunk frame marker. As a uint32, 0x43585143 is emitted by QUA on the
# little-endian control stream as bytes 43 51 58 43 ("CQXC"). The shim checks
# this before treating the payload as [version,total,offset,chunk_bytes,flags].
CONFIG_CHUNK_MAGIC = 0x43585143
CONFIG_CHUNK_VERSION = 1
# Configure packets are intentionally chunked. In testing, the real QUA/OPNIC
# external-stream path failed around the 4 KiB packet size boundary. 1024 bytes
# is a conservative value we picked to stay well below that boundary; it is not
# a decoder-level limit.
CONFIG_CHUNK_BYTES = 1024
CONFIG_CHUNK_HEADER_WORDS = 6
CONFIG_CHUNK_HEADER_BYTES = 4 * CONFIG_CHUNK_HEADER_WORDS
CONFIG_CHUNK_BEGIN = 1
CONFIG_CHUNK_END = 2
CONTROL_PAYLOAD_WORDS = CONFIG_CHUNK_HEADER_WORDS + (CONFIG_CHUNK_BYTES // 4)
MAX_CHUNK_BITS = 64
DECODER_ID = 0
NUM_OBSERVABLES = 1


@qua_struct
class DecoderDataRequestPacket:
    magic: QuaArray[int, 1]
    function_id: QuaArray[int, 1]
    arg_len: QuaArray[int, 1]
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]
    args: QuaArray[int, DATA_PAYLOAD_WORDS]


@qua_struct
class DecoderDataResponsePacket:
    magic: QuaArray[int, 1]
    status: QuaArray[int, 1]
    result_len: QuaArray[int, 1]
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]
    result: QuaArray[int, DATA_PAYLOAD_WORDS]


@qua_struct
class DecoderControlRequestPacket:
    magic: QuaArray[int, 1]
    function_id: QuaArray[int, 1]
    arg_len: QuaArray[int, 1]
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]
    config: QuaArray[int, CONTROL_PAYLOAD_WORDS]


@qua_struct
class DecoderControlResponsePacket:
    magic: QuaArray[int, 1]
    status: QuaArray[int, 1]
    result_len: QuaArray[int, 1]
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]
    config: QuaArray[int, CONTROL_PAYLOAD_WORDS]


def to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v if v < 0x80000000 else v - 0x100000000


def split_u64(v: int) -> list[int]:
    return [to_i32(v), to_i32(v >> 32)]


def u64_words(values: list[int]) -> list[int]:
    words: list[int] = []
    for value in values:
        words.extend(split_u64(value))
    return words


def pack_config_words(config: bytes) -> list[int]:
    padded = config + b"\0" * ((4 - (len(config) % 4)) % 4)
    return [
        to_i32(int.from_bytes(padded[i : i + 4], "little"))
        for i in range(0, len(padded), 4)
    ]


def make_config_chunks(config: bytes) -> list[tuple[int, int, list[int]]]:
    """Build in-order configure frames consumed by the shim.

    The shim only calls CUDA-QX after receiving the END chunk, so a config phase
    either installs the fully assembled YAML or returns a configure error.
    """
    chunks: list[tuple[int, int, list[int]]] = []
    total = len(config)
    for offset in range(0, total, CONFIG_CHUNK_BYTES):
        chunk = config[offset : offset + CONFIG_CHUNK_BYTES]
        flags = 0
        if offset == 0:
            flags |= CONFIG_CHUNK_BEGIN
        if offset + len(chunk) == total:
            flags |= CONFIG_CHUNK_END
        words = [
            to_i32(CONFIG_CHUNK_MAGIC),
            CONFIG_CHUNK_VERSION,
            total,
            offset,
            len(chunk),
            flags,
        ] + pack_config_words(chunk)
        if len(words) > CONTROL_PAYLOAD_WORDS:
            raise RuntimeError(
                f"config chunk needs {len(words)} words, "
                f"but control packet has {CONTROL_PAYLOAD_WORDS}"
            )
        chunks.append((offset, len(chunk), words))
    if not chunks:
        raise RuntimeError("empty decoder config")
    return chunks


def load_dataset(path: Path) -> tuple[list[list[int]], list[int]]:
    shots: list[list[int]] = []
    corrections: list[int] = []
    cur: list[int] = []
    in_shot = False
    in_corr = False

    def flush() -> None:
        nonlocal cur
        if in_shot and cur:
            shots.append(cur)
        cur = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("SHOT_BITS"):
            flush()
            _, bit_string = line.split(maxsplit=1)
            if not bit_string or any(ch not in "01" for ch in bit_string):
                raise RuntimeError(f"invalid SHOT_BITS row in {path}: {line}")
            shots.append([int(ch) for ch in bit_string])
            in_shot = False
            in_corr = False
            continue
        if line.startswith("CORRECTION "):
            _, value = line.split(maxsplit=1)
            corrections.append(int(value))
            continue
        if line.startswith("SHOT_START"):
            flush()
            in_shot = True
            in_corr = False
            continue
        if line == "CORRECTIONS_START":
            flush()
            in_shot = False
            in_corr = True
            continue
        if line == "CORRECTIONS_END":
            break
        if not line or line.startswith("NUM_"):
            continue
        try:
            bit = int(line)
        except ValueError:
            continue
        if in_shot:
            cur.append(bit)
        elif in_corr:
            corrections.append(bit)
    flush()
    if not shots:
        raise RuntimeError(f"no shots found in {path}")
    return shots, corrections


def syndrome_chunks(bits: list[int]) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    for off in range(0, len(bits), MAX_CHUNK_BITS):
        chunk = bits[off : off + MAX_CHUNK_BITS]
        packed = 0
        for idx, bit in enumerate(chunk):
            if bit:
                packed |= 1 << idx
        chunks.append((len(chunk), packed))
    return chunks


def make_config(opx_ip: str, opx_port: int, fem_index: int) -> DictQuaConfig:
    return {
        "controllers": {
            "con1": {
                "fems": {
                    fem_index: {
                        "analog_inputs": {},
                        "analog_outputs": {1: {"offset": 0.0}},
                        "digital_inputs": {},
                        "digital_outputs": {},
                        "type": "LF",
                    },
                }
            }
        },
        "digital_waveforms": {},
        "elements": {
            "qe1": {
                "intermediate_frequency": 1e6,
                "operations": {"analog": "qe1_analog_pulse"},
                "singleInput": {"port": ("con1", fem_index, 1)},
            }
        },
        "integration_weights": {},
        "mixers": {},
        "oscillators": {},
        "pulses": {
            "qe1_analog_pulse": {
                "length": 2000,
                "operation": "control",
                "waveforms": {"single": "qe1_analog_pulse_wf_I"},
            }
        },
        "waveforms": {"qe1_analog_pulse_wf_I": {"sample": 0.5, "type": "constant"}},
    }
