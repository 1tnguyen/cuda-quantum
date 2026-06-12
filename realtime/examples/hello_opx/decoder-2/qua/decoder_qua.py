#!/usr/bin/env python3
"""Shared QUA packet helpers for the realtime decoder-2 OPNIC example.

Only the data stream is used here; decoder configuration is done host-side
from a YAML file before any QUA program runs.
"""

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

# End-of-batch marker. Dedicated non-zero id (fnv1a_hash("phase_complete")),
# distinct from function_id 0 which the host's shared loop reserves for
# "OPX shutdown". The host registers a phase-done handler under this id that
# ends the serve loop so it can reconnect for the next QUA program.
PHASE_DONE_ID = 845524386           # 0x3265ADA2 as signed int32

DATA_STREAM_ID = 1
DATA_PAYLOAD_WORDS = 10
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
