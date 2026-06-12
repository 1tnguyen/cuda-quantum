#!/usr/bin/env python3
"""Run decoder data playback against the currently configured standing server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qm import QuantumMachinesManager
from qm.qua import *
import numpy as np

from decoder_qua import (
    CUDAQX_RT_OK,
    DATA_STREAM_ID,
    ENQUEUE_SYNDROMES_ID,
    GET_CORRECTIONS_ID,
    PHASE_DONE_ID,
    RESET_DECODER_ID,
    RPC_MAGIC_REQUEST,
    RPC_MAGIC_RESPONSE,
    DecoderDataRequestPacket,
    DecoderDataResponsePacket,
    load_dataset,
    make_config,
    split_u64,
    syndrome_chunks,
)
from decoder_specs import available_decoder_tests, get_decoder_test

DATA_SUMMARY_PREFIX = "DECODER_DATA_SUMMARY_JSON: "


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    body = arr[1:] if len(arr) > 1 else arr
    return {
        "first": float(arr[0]),
        "min": float(np.min(body)),
        "max": float(np.max(body)),
        "avg": float(np.mean(body)),
        "stddev": float(np.std(body)),
        "p95": float(np.percentile(body, 95)),
        "p99": float(np.percentile(body, 99)),
    }


def combined_stats_line(
    name: str,
    total: np.ndarray,
    transport: np.ndarray,
    decoder: np.ndarray,
    total_label: str,
) -> None:
    values = np.asarray(total).ravel()
    if len(values) == 0:
        print(f"{name}: no samples")
        return
    print(f"== {name} =====================================")
    print(f"Samples: {len(values)}")
    totals = _stats(total)
    transports = _stats(transport)
    decoders = _stats(decoder)
    rows = [
        ("First", "first"),
        ("Min", "min"),
        ("Max", "max"),
        ("Avg", "avg"),
        ("Stddev", "stddev"),
        ("95th percentile", "p95"),
        ("99th percentile", "p99"),
    ]
    for label, key in rows:
        print(
            f"{label + ':':<17}"
            f"{transports[key]:.2f} ns (transport); "
            f"{decoders[key]:.2f} ns (decoder); "
            f"{totals[key]:.2f} ns ({total_label})"
        )


def timestamp_delta_ns(end: np.ndarray, start: np.ndarray) -> np.ndarray:
    # QUA timestamps are carried through signed 32-bit stream values. Compute
    # deltas modulo 2^32 so a phase near the wrap point does not inflate the
    # transport/dispatch remainder by one full counter period.
    return (end.astype(np.int64) - start.astype(np.int64)) % (1 << 32)


def _float_values(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).ravel()]


def _valid_count(values: np.ndarray) -> dict[str, int]:
    arr = np.asarray(values).ravel()
    return {"ok": int(np.count_nonzero(arr)), "total": int(len(arr))}


def data_summary_json(
    test_name: str,
    start_shot: int,
    shot_count: int,
    zero_valid: np.ndarray,
    data_rpc_valid: np.ndarray,
    shot_valid: np.ndarray,
    data_rpc_latency: np.ndarray,
    data_rpc_host_processing: np.ndarray,
    data_rpc_transport: np.ndarray,
    shot_latency: np.ndarray,
    shot_host_processing: np.ndarray,
    shot_transport: np.ndarray,
) -> str:
    summary = {
        "test": test_name,
        "start_shot": int(start_shot),
        "shots": int(shot_count),
        "validations": {
            "zero_syndrome": _valid_count(zero_valid),
            "data_rpc": _valid_count(data_rpc_valid),
            "shot_corrections": _valid_count(shot_valid),
        },
        "metrics_ns": {
            "data_rpc_round_trip_latency": _float_values(data_rpc_latency),
            "data_rpc_decoder_processing_time": _float_values(
                data_rpc_host_processing
            ),
            "data_rpc_transport_dispatch_remainder": _float_values(
                data_rpc_transport
            ),
            "shot_pipeline_latency": _float_values(shot_latency),
            "shot_decoder_processing_time": _float_values(shot_host_processing),
            "shot_transport_dispatch_remainder": _float_values(shot_transport),
        },
    }
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def _human_bytes(nbytes: int) -> str:
    """Human-readable byte size (B / KB / MB, decimal)."""
    if nbytes >= 1_000_000:
        return f"{nbytes / 1e6:.2f} MB"
    if nbytes >= 1_000:
        return f"{nbytes / 1e3:.2f} KB"
    return f"{nbytes} B"


def bake_arrays(
    shots: list[list[int]],
    corrections: list[int],
    zero_correction: int,
):
    """Flatten the dataset into the constant arrays the QUA program bakes into
    OPX data memory.

    The whole experiment fits in OPX data memory, so instead of unrolling one
    explicit RPC per call (which blows up the *program* size and forces phase
    batching), we bake the syndrome words into `declare(int, value=...)` arrays
    and iterate over them with QUA `for_` loops -- the program size becomes
    independent of the shot count.

    Returns
    -------
    packed_lo, packed_hi : list[int]
        Per-(shot, chunk) syndrome words, LSB-first (low 32 then high 32),
        length `total_shots * nchunks`; indexed by `shot * nchunks + chunk`.
        Shot index 0 is the all-zero warm-up shot.
    nbits_flat : list[int]
        Per-chunk detector count, shot-invariant, length `nchunks`.
    expected_flat : list[int]
        Per-shot expected correction, length `total_shots`
        (`[zero_correction] + corrections`).
    nchunks, total_shots : int
    """
    bit_len = len(shots[0])
    for idx, bits in enumerate(shots):
        if len(bits) != bit_len:
            raise RuntimeError(
                "baked single-phase program requires every shot to have the "
                f"same bit length; shot {idx} has {len(bits)} bits, expected "
                f"{bit_len}. Use the batched runner for ragged datasets."
            )

    base_chunks = syndrome_chunks(shots[0])
    nchunks = len(base_chunks)
    nbits_flat = [nbits for nbits, _ in base_chunks]

    packed_lo: list[int] = []
    packed_hi: list[int] = []
    # Warm-up shot 0: all-zero syndromes.
    for _ in range(nchunks):
        packed_lo.append(0)
        packed_hi.append(0)
    for bits in shots:
        for _, packed in syndrome_chunks(bits):
            lo, hi = split_u64(packed)
            packed_lo.append(lo)
            packed_hi.append(hi)

    expected_flat = [zero_correction] + [
        corrections[idx] if idx < len(corrections) else 0
        for idx in range(len(shots))
    ]
    total_shots = len(shots) + 1
    return packed_lo, packed_hi, nbits_flat, expected_flat, nchunks, total_shots


def build_program(
    shots: list[list[int]],
    corrections: list[int],
    phase_done: bool,
    decoder_id: int,
    num_observables: int,
    zero_correction: int,
):
    (packed_lo, packed_hi, nbits_flat, expected_flat,
     nchunks, total_shots) = bake_arrays(shots, corrections, zero_correction)

    # Baked immediate arrays in OPX data memory (QUA int = 4 bytes). Only the
    # syndrome words scale with shots; nbits/expected are tiny. Printed to
    # stderr so the stdout summary line stays clean for run_decoder_sequence.
    _arr_sizes = [
        ("packed_lo", len(packed_lo)),
        ("packed_hi", len(packed_hi)),
        ("nbits", len(nbits_flat)),
        ("expected", len(expected_flat)),
    ]
    _arr_total = sum(n for _, n in _arr_sizes)
    print("[QUA] Baked array data memory (int32, 4 B each):", file=sys.stderr)
    for _name, _n in _arr_sizes:
        print(f"         {_name:<12} {_n:>8,} ints  "
              f"{_human_bytes(4 * _n):>10}", file=sys.stderr)
    print(f"         {'TOTAL':<12} {_arr_total:>8,} ints  "
          f"{_human_bytes(4 * _arr_total):>10}", file=sys.stderr)

    with program() as prog:
        # Single data stream: decoder configuration is host-side, no control
        # stream needed.
        data_req = declare_struct(DecoderDataRequestPacket)
        data_resp = declare_struct(DecoderDataResponsePacket)
        data_out = declare_external_stream(
            DecoderDataRequestPacket, DATA_STREAM_ID, QuaStreamDirection.OUTGOING
        )
        data_in = declare_external_stream(
            DecoderDataResponsePacket, DATA_STREAM_ID, QuaStreamDirection.INCOMING
        )

        # Baked syndrome data: program size is now independent of shot count.
        packed_lo_arr = declare(int, value=packed_lo)
        packed_hi_arr = declare(int, value=packed_hi)
        nbits_arr = declare(int, value=nbits_flat)
        expected_arr = declare(int, value=expected_flat)

        data_rpc_start = declare_stream()
        data_rpc_end = declare_stream()
        data_rpc_host_processing = declare_stream()
        data_rpc_valid = declare_stream()
        shot_start = declare_stream()
        shot_end = declare_stream()
        shot_host_processing = declare_stream()
        shot_valid = declare_stream()
        zero_valid = declare_stream()

        marker = declare(int)
        rpc_host_ns = declare(int)
        shot_host_ns = declare(int)
        rpc_ok = declare(bool)
        shot_ok = declare(bool)
        req_id = declare(int, value=1)
        shot = declare(int)
        chunk = declare(int)
        flat = declare(int)
        expected_v = declare(int)

        def emit_rpc(function_id, arg_len, fill_args, expected_var=None,
                     update_shot=True):
            """Emit one decoder RPC. `fill_args` is a callable that assigns the
            `data_req.args[...]` words from QUA arrays/scalars (so the call can
            sit inside a QUA `for_` loop). `expected_var`, when given, is a QUA
            scalar the get_corrections result is validated against."""
            assign(data_req.magic[0], RPC_MAGIC_REQUEST)
            assign(data_req.function_id[0], function_id)
            assign(data_req.arg_len[0], arg_len)
            assign(data_req.request_id[0], req_id)
            assign(data_req.ptp_timestamp[0], 0)
            assign(data_req.ptp_timestamp[1], 0)
            fill_args()
            assign(marker, req_id)
            save(marker, data_rpc_start)
            send_to_external_stream(data_out, data_req)
            receive_from_external_stream(data_in, data_resp)
            save(marker, data_rpc_end)
            assign(rpc_host_ns, data_resp.ptp_timestamp[0])
            save(rpc_host_ns, data_rpc_host_processing)
            assign(
                rpc_ok,
                (data_resp.magic[0] == RPC_MAGIC_RESPONSE)
                & (data_resp.status[0] == CUDAQX_RT_OK)
                & (data_resp.request_id[0] == req_id),
            )
            if expected_var is not None:
                assign(
                    rpc_ok,
                    rpc_ok
                    & (data_resp.result_len[0] == 8)
                    & (data_resp.result[0] == expected_var)
                    & (data_resp.result[1] == 0),
                )
            save(rpc_ok, data_rpc_valid)
            if update_shot:
                assign(shot_host_ns, shot_host_ns + rpc_host_ns)
                assign(shot_ok, shot_ok & rpc_ok)
            assign(req_id, req_id + 1)

        def fill_reset():
            assign(data_req.args[0], decoder_id)
            assign(data_req.args[1], 0)

        def fill_enqueue():
            assign(data_req.args[0], decoder_id)
            assign(data_req.args[1], 0)
            assign(data_req.args[2], nbits_arr[chunk])
            assign(data_req.args[3], 0)
            assign(data_req.args[4], packed_lo_arr[flat])
            assign(data_req.args[5], packed_hi_arr[flat])
            assign(data_req.args[6], 0)
            assign(data_req.args[7], 0)

        def fill_get():
            assign(data_req.args[0], decoder_id)
            assign(data_req.args[1], 0)
            assign(data_req.args[2], num_observables)
            assign(data_req.args[3], 0)
            assign(data_req.args[4], 1)
            assign(data_req.args[5], 0)

        # One QUA loop over every shot (shot 0 is the all-zero warm-up). The
        # body is emitted once but runs `total_shots` times on the OPX, so the
        # program size does not grow with the shot count.
        with for_(shot, 0, shot < total_shots, shot + 1):
            assign(shot_ok, True)
            assign(shot_host_ns, 0)
            assign(marker, req_id)
            save(marker, shot_start)

            # reset_decoder -- u64_words([decoder_id]) => 8 bytes.
            emit_rpc(RESET_DECODER_ID, 8, fill_reset)

            # enqueue_syndromes per chunk -- u64_words([decoder_id, nbits,
            # packed, 0]) => 32 bytes.
            with for_(chunk, 0, chunk < nchunks, chunk + 1):
                assign(flat, shot * nchunks + chunk)
                emit_rpc(ENQUEUE_SYNDROMES_ID, 32, fill_enqueue)

            # get_corrections -- u64_words([decoder_id, num_observables, 1])
            # => 24 bytes; validate against the baked expected correction.
            assign(expected_v, expected_arr[shot])
            emit_rpc(GET_CORRECTIONS_ID, 24, fill_get, expected_var=expected_v)

            save(marker, shot_end)
            save(shot_host_ns, shot_host_processing)
            # Route shot 0 (warm-up) to zero_valid, real shots to shot_valid.
            with if_(shot == 0):
                save(shot_ok, zero_valid)
            with else_():
                save(shot_ok, shot_valid)

        if phase_done:
            # Mark this data phase complete with the dedicated PHASE_DONE_ID
            # (not function_id 0, which the host loop reserves for shutdown).
            # The host's phase-done handler ends the serve loop, recreates
            # OPNIC contexts, and syncs with the next QUA program without
            # unloading the decoder bank.
            assign(data_req.magic[0], RPC_MAGIC_REQUEST)
            assign(data_req.function_id[0], PHASE_DONE_ID)
            assign(data_req.arg_len[0], 0)
            assign(data_req.request_id[0], req_id)
            assign(data_req.ptp_timestamp[0], 0)
            assign(data_req.ptp_timestamp[1], 0)
            send_to_external_stream(data_out, data_req)

        with stream_processing():
            data_rpc_start.timestamps().save_all("data_rpc_ts_start")
            data_rpc_end.timestamps().save_all("data_rpc_ts_end")
            data_rpc_host_processing.save_all("data_rpc_host_processing_ns")
            data_rpc_valid.save_all("data_rpc_valid")
            shot_start.timestamps().save_all("shot_ts_start")
            shot_end.timestamps().save_all("shot_ts_end")
            shot_host_processing.save_all("shot_host_processing_ns")
            shot_valid.save_all("shot_valid")
            zero_valid.save_all("zero_valid")

    return prog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default="multi-lut",
        help=f"decoder test to play back ({available_decoder_tests()})",
    )
    parser.add_argument("--syndromes-path", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument("--opx-ip", default="10.137.129.5")
    parser.add_argument("--opx-port", type=int, default=9510)
    parser.add_argument("--fem-index", type=int, default=5)
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--start-shot", type=int, default=0)
    parser.add_argument("--phase-done", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="emit one machine-readable summary line for run_decoder_sequence.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_decoder_test(args.test)
    syndromes_path = args.syndromes_path or spec.syndromes_path(args.data_dir)
    shots, corrections = load_dataset(syndromes_path)
    if args.start_shot:
        shots = shots[args.start_shot :]
        corrections = corrections[args.start_shot :]
    if args.max_shots > 0:
        shots = shots[: args.max_shots]
        corrections = corrections[: args.max_shots]
    if not shots:
        raise RuntimeError("no shots selected")

    print("== Decoder Data QUA ==================================")
    print(f"Test: {spec.name} ({spec.label})")
    print(f"Syndromes: {syndromes_path}")
    print(f"Shots: {len(shots)} x {len(shots[0])} bits")
    print(f"Start shot: {args.start_shot}")
    print(f"Corrections: {len(corrections)}, ones={sum(corrections)}")
    print(
        f"Decoder id: {spec.decoder_id}, observables: {spec.num_observables}"
    )
    print(f"Stream: data={DATA_STREAM_ID}")
    print(f"Phase done packet: {args.phase_done}")

    prog = build_program(
        shots,
        corrections,
        args.phase_done,
        spec.decoder_id,
        spec.num_observables,
        spec.zero_correction,
    )
    if args.build_only:
        print("Build-only: QUA program constructed")
        return 0

    qmm = QuantumMachinesManager(host=args.opx_ip, port=args.opx_port)
    qm = qmm.open_qm(make_config(args.opx_ip, args.opx_port, args.fem_index))
    job = qm.execute(prog)
    job.result_handles.wait_for_all_values()

    data_rpc_latency = timestamp_delta_ns(
        job.result_handles.get("data_rpc_ts_end").fetch_all(flat_struct=True),
        job.result_handles.get("data_rpc_ts_start").fetch_all(flat_struct=True),
    )
    data_rpc_host_processing = job.result_handles.get(
        "data_rpc_host_processing_ns"
    ).fetch_all(flat_struct=True)
    data_rpc_transport = (
        data_rpc_latency.astype(float) - data_rpc_host_processing.astype(float)
    )
    data_rpc_valid = job.result_handles.get("data_rpc_valid").fetch_all(
        flat_struct=True
    )
    shot_latency = timestamp_delta_ns(
        job.result_handles.get("shot_ts_end").fetch_all(flat_struct=True),
        job.result_handles.get("shot_ts_start").fetch_all(flat_struct=True),
    )
    shot_host_processing = job.result_handles.get(
        "shot_host_processing_ns"
    ).fetch_all(flat_struct=True)
    shot_transport = shot_latency.astype(float) - shot_host_processing.astype(float)
    zero_valid = job.result_handles.get("zero_valid").fetch_all(flat_struct=True)
    shot_valid = job.result_handles.get("shot_valid").fetch_all(flat_struct=True)

    data_ok = int(np.count_nonzero(data_rpc_valid))
    zero_ok = int(np.count_nonzero(zero_valid))
    shot_ok = int(np.count_nonzero(shot_valid))

    print("== Decoder Data Validation ===========================")
    print(f"Zero-syndrome valid: {zero_ok} / {len(zero_valid)}")
    print(f"Data RPC valid: {data_ok} / {len(data_rpc_valid)}")
    print(f"Shot corrections valid: {shot_ok} / {len(shot_valid)}")
    combined_stats_line(
        "Data RPC latency",
        data_rpc_latency,
        data_rpc_transport,
        data_rpc_host_processing,
        "round-trip",
    )
    combined_stats_line(
        "Shot latency",
        shot_latency,
        shot_transport,
        shot_host_processing,
        "pipeline",
    )
    print("======================================================")
    if args.summary_json:
        print(
            DATA_SUMMARY_PREFIX
            + data_summary_json(
                spec.name,
                args.start_shot,
                len(shots),
                zero_valid,
                data_rpc_valid,
                shot_valid,
                data_rpc_latency,
                data_rpc_host_processing,
                data_rpc_transport,
                shot_latency,
                shot_host_processing,
                shot_transport,
            )
        )

    return 0 if (
        zero_ok == len(zero_valid)
        and data_ok == len(data_rpc_valid)
        and shot_ok == len(shot_valid)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
