#!/usr/bin/env python3
"""Send one chunked decoder config over the control OPNIC stream."""

from __future__ import annotations

import argparse
from pathlib import Path

from qm import QuantumMachinesManager
from qm.qua import *
import numpy as np

from decoder_qua import (
    CONFIGURE_DECODER_ID,
    CONFIG_CHUNK_BYTES,
    CONFIG_CHUNK_HEADER_BYTES,
    CONTROL_STREAM_ID,
    CUDAQX_RT_OK,
    DATA_STREAM_ID,
    RPC_MAGIC_REQUEST,
    RPC_MAGIC_RESPONSE,
    DecoderControlRequestPacket,
    DecoderControlResponsePacket,
    DecoderDataRequestPacket,
    DecoderDataResponsePacket,
    make_config,
    make_config_chunks,
)
from decoder_specs import (
    DEFAULT_TRT_ONNX_PATH,
    available_decoder_tests,
    get_decoder_test,
    is_git_lfs_pointer,
    load_config_bytes,
)


def timestamp_delta_ns(end: np.ndarray, start: np.ndarray) -> np.ndarray:
    # QUA timestamps are carried through signed 32-bit stream values. Compute
    # per-RPC deltas modulo 2^32 so a long config phase near the wrap point does
    # not add a spurious 4.29 s to the round-trip/transport split.
    return (end.astype(np.int64) - start.astype(np.int64)) % (1 << 32)


def format_duration_ns(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f} us"
    return f"{value:.0f} ns"


def build_program(config_chunks: list[tuple[int, int, list[int]]]):
    request_id = 1

    with program() as prog:
        # Declare both streams so the program signature matches the standing
        # decoder server, even though this phase only uses the control stream.
        declare_struct(DecoderDataRequestPacket)
        declare_struct(DecoderDataResponsePacket)
        declare_external_stream(
            DecoderDataRequestPacket, DATA_STREAM_ID, QuaStreamDirection.OUTGOING
        )
        declare_external_stream(
            DecoderDataResponsePacket, DATA_STREAM_ID, QuaStreamDirection.INCOMING
        )

        ctrl_req = declare_struct(DecoderControlRequestPacket)
        ctrl_resp = declare_struct(DecoderControlResponsePacket)
        ctrl_out = declare_external_stream(
            DecoderControlRequestPacket,
            CONTROL_STREAM_ID,
            QuaStreamDirection.OUTGOING,
        )
        ctrl_in = declare_external_stream(
            DecoderControlResponsePacket,
            CONTROL_STREAM_ID,
            QuaStreamDirection.INCOMING,
        )

        start = declare_stream()
        end = declare_stream()
        valid = declare_stream()
        host_processing = declare_stream()
        marker = declare(int)
        host_ns = declare(int)
        ok = declare(bool)

        for _, chunk_len, chunk_words in config_chunks:
            assign(ctrl_req.magic[0], RPC_MAGIC_REQUEST)
            assign(ctrl_req.function_id[0], CONFIGURE_DECODER_ID)
            assign(ctrl_req.arg_len[0], CONFIG_CHUNK_HEADER_BYTES + chunk_len)
            assign(ctrl_req.request_id[0], request_id)
            assign(ctrl_req.ptp_timestamp[0], 0)
            assign(ctrl_req.ptp_timestamp[1], 0)
            for idx, word in enumerate(chunk_words):
                assign(ctrl_req.config[idx], word)

            assign(marker, request_id)
            save(marker, start)
            send_to_external_stream(ctrl_out, ctrl_req)
            receive_from_external_stream(ctrl_in, ctrl_resp)
            save(marker, end)
            assign(host_ns, ctrl_resp.ptp_timestamp[0])
            save(host_ns, host_processing)
            assign(
                ok,
                (ctrl_resp.magic[0] == RPC_MAGIC_RESPONSE)
                & (ctrl_resp.status[0] == CUDAQX_RT_OK)
                & (ctrl_resp.request_id[0] == request_id),
            )
            save(ok, valid)
            request_id += 1

        # Mark the config phase complete. The host consumes function_id == 0
        # locally, recreates OPNIC contexts, and then syncs with the next QUA
        # program while keeping the shim/decoder bank alive.
        assign(ctrl_req.magic[0], RPC_MAGIC_REQUEST)
        assign(ctrl_req.function_id[0], 0)
        assign(ctrl_req.arg_len[0], 0)
        assign(ctrl_req.request_id[0], request_id)
        assign(ctrl_req.ptp_timestamp[0], 0)
        assign(ctrl_req.ptp_timestamp[1], 0)
        send_to_external_stream(ctrl_out, ctrl_req)

        with stream_processing():
            start.timestamps().save_all("config_ts_start")
            end.timestamps().save_all("config_ts_end")
            host_processing.save_all("config_host_processing_ns")
            valid.save_all("config_valid")

    return prog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default="multi-lut",
        help=f"decoder test to configure ({available_decoder_tests()})",
    )
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--trt-onnx-path", type=Path, default=DEFAULT_TRT_ONNX_PATH)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--opx-ip", default="10.137.129.5")
    parser.add_argument("--opx-port", type=int, default=9510)
    parser.add_argument("--fem-index", type=int, default=5)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_decoder_test(args.test)
    config_path = args.config_path or spec.config_path(args.data_dir)
    if spec.requires_trt and not args.trt_onnx_path.is_file():
        print(f"ERROR: TRT ONNX model not found: {args.trt_onnx_path}")
        print("Run ../../cudaqx_decoder_hostcall/build.sh or pass --trt-onnx-path.")
        return 2
    if spec.requires_trt and is_git_lfs_pointer(args.trt_onnx_path):
        print(f"ERROR: TRT ONNX model is still a Git LFS pointer: {args.trt_onnx_path}")
        print("Run ../../cudaqx_decoder_hostcall/build.sh to fetch the real ONNX blob.")
        return 2

    config_bytes = load_config_bytes(config_path, args.trt_onnx_path)
    config_chunks = make_config_chunks(config_bytes)

    print("== Decoder Config QUA ================================")
    print(f"Test: {spec.name} ({spec.label})")
    print(
        f"Config: {config_path} "
        f"({len(config_bytes)} B, {len(config_chunks)} chunks, "
        f"{CONFIG_CHUNK_BYTES} B max/chunk)"
    )
    print(f"Streams: data={DATA_STREAM_ID}, control={CONTROL_STREAM_ID}")

    prog = build_program(config_chunks)
    if args.build_only:
        print("Build-only: QUA program constructed")
        return 0

    qmm = QuantumMachinesManager(host=args.opx_ip, port=args.opx_port)
    qm = qmm.open_qm(make_config(args.opx_ip, args.opx_port, args.fem_index))
    job = qm.execute(prog)
    job.result_handles.wait_for_all_values()

    config_start = job.result_handles.get("config_ts_start").fetch_all(
        flat_struct=True
    )
    config_end = job.result_handles.get("config_ts_end").fetch_all(
        flat_struct=True
    )
    valid = job.result_handles.get("config_valid").fetch_all(flat_struct=True)
    ok = int(np.count_nonzero(valid))
    total_ns = 0.0
    if len(config_start) and len(config_end):
        total_ns = float(
            timestamp_delta_ns(
                np.asarray([config_end[-1]]), np.asarray([config_start[0]])
            )[0]
        )

    print(
        f"Configure complete: valid={ok}/{len(valid)}, "
        f"total_config_time={format_duration_ns(total_ns)}"
    )
    return 0 if ok == len(valid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
