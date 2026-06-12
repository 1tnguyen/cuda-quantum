#!/usr/bin/env python3
"""Run a per-decoder sequence: start one server per decoder, run data batches.

For each selected decoder the runner:
  1. Writes the decoder's YAML config (with ${TRT_ONNX_PATH} substituted) to
     a temp file in the output directory.
  2. Starts a fresh `decoder-2_cpu` with `--config=<resolved.yml>`.
  3. Waits for the server to print its OPNIC handshake marker.
  4. Runs `decoder_data_qua.py` in batches for that decoder.
  5. Waits for the server to exit cleanly, then moves on to the next decoder.

No shim, no venv, no QUA config phase -- the decoder bank is configured
entirely host-side before any QUA program runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from decoder_specs import (
    DEFAULT_TRT_GOLDEN_PATH,
    DEFAULT_TRT_ONNX_PATH,
    DecoderTestSpec,
    available_decoder_tests,
    expand_decoder_tests,
    is_git_lfs_pointer,
    load_config_bytes,
)


HERE = Path(__file__).resolve().parent
DATA_QUA = HERE / "decoder_data_qua.py"
HOST_READY_MARKER = "Waiting for OPX handshake"

DEFAULT_BUILD = Path("/workspaces/cuda-quantum/build-realtime")
DEFAULT_SERVER = DEFAULT_BUILD / "examples/hello_opx/decoder-2/decoder-2_cpu"

# Runtime library paths for the local CUDA-QX and CUDA-Q installs.
CUDAQX_DIR = Path(os.environ.get("CUDAQX_INSTALL_DIR", Path.home() / ".cudaqx"))
CUDAQ_DIR = Path(os.environ.get("CUDAQ_INSTALL_DIR", "/usr/local/cudaq"))

DATA_SUMMARY_PREFIX = "DECODER_DATA_SUMMARY_JSON: "


def server_build_library_path(server: Path) -> Optional[Path]:
    try:
        return server.resolve().parents[3] / "lib"
    except IndexError:
        return None


def build_ld_library_path(server: Optional[Path] = None) -> str:
    """Build LD_LIBRARY_PATH from the local installs."""
    paths = []
    if server is not None:
        server_lib = server_build_library_path(server)
        if server_lib is not None:
            paths.append(server_lib)
    paths.extend([
        # Prefer the build tree that produced decoder-2_cpu over any stale
        # libcudaq-realtime.so elsewhere on LD_LIBRARY_PATH (e.g. realtime/build/lib).
        DEFAULT_BUILD / "lib",
        CUDAQX_DIR / "lib",
        CUDAQX_DIR / "lib" / "decoder-plugins",
        CUDAQ_DIR / "lib",
    ])
    existing = [str(p) for p in paths if p.is_dir()]
    env_val = os.environ.get("LD_LIBRARY_PATH", "")
    if env_val:
        existing.append(env_val)
    return ":".join(existing)


class HostProcess:
    def __init__(self, cmd: list[str], env: dict[str, str], log_path: Path):
        self.cmd = cmd
        self.env = env
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self.ready = threading.Event()
        self.tail: list[str] = []
        self._logf = None
        self._thread = None

    def start(self) -> None:
        self._logf = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._logf is not None:
                self._logf.write(line)
                self._logf.flush()
            self.tail.append(line.rstrip("\n"))
            del self.tail[:-120]
            if HOST_READY_MARKER in line:
                self.ready.set()
        self.ready.set()

    def wait_ready(self, timeout: float) -> bool:
        return self.ready.wait(timeout=timeout)

    def poll(self) -> Optional[int]:
        return self.proc.poll() if self.proc is not None else None

    def wait_exit(self, timeout: float) -> Optional[int]:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def close(self) -> None:
        if self._logf is not None:
            self._logf.close()
            self._logf = None

    def stats(self) -> Optional[int]:
        try:
            lines = self.log_path.read_text().splitlines()
        except OSError:
            lines = self.tail
        for line in reversed(lines):
            m = re.search(r"decode RPCs:\s*(\d+)", line)
            if m:
                return int(m.group(1))
        return None

    def trt_raw_validation_summaries(self) -> list[str]:
        try:
            lines = self.log_path.read_text().splitlines()
        except OSError:
            lines = self.tail
        return [line for line in lines if line.startswith("[TRT raw validation]")]


def count_shots(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines()
               if line.startswith("SHOT_START") or line.startswith("SHOT_BITS "))


class SelectedJob:
    def __init__(
        self,
        spec: DecoderTestSpec,
        total_shots: int,
        start: int,
        requested: int,
        num_batches: int,
        batch_size: int,
    ):
        self.spec = spec
        self.total_shots = total_shots
        self.start = start
        self.requested = requested
        self.num_batches = num_batches
        self.batch_size = batch_size


def resolve_batch_plan(requested: int, batch_size_arg: int) -> tuple[int, int]:
    """Return (effective_batch_size, number_of_batches)."""
    if requested <= 0:
        raise ValueError("requested must be positive")
    if batch_size_arg < 0:
        raise ValueError("batch_size_arg must be non-negative")
    batch_size = requested if batch_size_arg == 0 else batch_size_arg
    return batch_size, (requested + batch_size - 1) // batch_size


def parse_data_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(DATA_SUMMARY_PREFIX):
            payload = line[len(DATA_SUMMARY_PREFIX):]
            return json.loads(payload)
    raise ValueError("decoder data QUA output did not contain a summary line")


def aggregate_data_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("cannot aggregate an empty summary list")
    test_name = summaries[0]["test"]
    aggregate = {
        "test": test_name,
        "data_programs": len(summaries),
        "shots": 0,
        "validations": {
            "zero_syndrome": {"ok": 0, "total": 0},
            "data_rpc": {"ok": 0, "total": 0},
            "shot_corrections": {"ok": 0, "total": 0},
        },
        "metrics_ns": {},
    }
    metric_names = summaries[0].get("metrics_ns", {}).keys()
    for name in metric_names:
        aggregate["metrics_ns"][name] = []

    for summary in summaries:
        if summary.get("test") != test_name:
            raise ValueError("cannot aggregate summaries from different tests")
        aggregate["shots"] += int(summary.get("shots", 0))
        for name, counts in summary.get("validations", {}).items():
            dst = aggregate["validations"].setdefault(
                name, {"ok": 0, "total": 0}
            )
            dst["ok"] += int(counts.get("ok", 0))
            dst["total"] += int(counts.get("total", 0))
        for name, values in summary.get("metrics_ns", {}).items():
            aggregate["metrics_ns"].setdefault(name, []).extend(
                float(v) for v in values
            )
    return aggregate


def _stats(values: list[float]) -> dict[str, float]:
    body = values[1:] if len(values) > 1 else values
    body_sorted = sorted(body)

    def percentile(p: float) -> float:
        if len(body_sorted) == 1:
            return body_sorted[0]
        rank = (len(body_sorted) - 1) * p / 100.0
        lo = int(rank)
        hi = min(lo + 1, len(body_sorted) - 1)
        frac = rank - lo
        return body_sorted[lo] * (1.0 - frac) + body_sorted[hi] * frac

    avg = sum(body) / len(body)
    variance = sum((v - avg) ** 2 for v in body) / len(body)
    return {
        "first": values[0],
        "min": min(body),
        "max": max(body),
        "avg": avg,
        "stddev": variance ** 0.5,
        "p95": percentile(95),
        "p99": percentile(99),
    }


def combined_latency_stats_lines(
    name: str,
    total: list[float],
    transport: list[float],
    decoder: list[float],
    total_label: str,
) -> list[str]:
    if not total:
        return [f"{name}: no samples"]
    totals = _stats(total)
    transports = _stats(transport)
    decoders = _stats(decoder)
    lines = [
        f"== {name} =====================================",
        f"Samples: {len(total)}",
    ]
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
        lines.append(
            f"{label + ':':<17}"
            f"{transports[key]:.2f} ns (transport); "
            f"{decoders[key]:.2f} ns (decoder); "
            f"{totals[key]:.2f} ns ({total_label})"
        )
    return lines


def format_batch_result(summary: dict) -> str:
    """Concise one-line result for a single data batch."""
    validations = summary.get("validations", {})
    shot = validations.get("shot_corrections", {})
    data_rpc = validations.get("data_rpc", {})
    latencies = summary.get("metrics_ns", {}).get("shot_pipeline_latency", [])
    parts = [
        f"shots={summary.get('shots', 0)}",
        f"shot_ok={shot.get('ok', 0)}/{shot.get('total', 0)}",
        f"rpc_ok={data_rpc.get('ok', 0)}/{data_rpc.get('total', 0)}",
    ]
    if latencies:
        body = latencies[1:] if len(latencies) > 1 else latencies
        avg = sum(body) / len(body)
        parts.append(f"avg_shot={avg / 1000.0:.2f}us")
    return ", ".join(parts)


def print_decoder_data_summary(summary: dict) -> None:
    title = f"== {summary['test']} Decoder Data Summary "
    print(title + "=" * max(0, 54 - len(title)))
    print(f"Data QUA programs: {summary['data_programs']}")
    print(f"Shots: {summary['shots']}")
    validations = summary["validations"]
    print(
        "Zero-syndrome valid: "
        f"{validations['zero_syndrome']['ok']} / "
        f"{validations['zero_syndrome']['total']}"
    )
    print(
        "Data RPC valid: "
        f"{validations['data_rpc']['ok']} / {validations['data_rpc']['total']}"
    )
    print(
        "Shot corrections valid: "
        f"{validations['shot_corrections']['ok']} / "
        f"{validations['shot_corrections']['total']}"
    )
    metrics = summary["metrics_ns"]
    for line in combined_latency_stats_lines(
        "Data RPC latency",
        metrics.get("data_rpc_round_trip_latency", []),
        metrics.get("data_rpc_transport_dispatch_remainder", []),
        metrics.get("data_rpc_decoder_processing_time", []),
        "round-trip",
    ):
        print(line)
    for line in combined_latency_stats_lines(
        "Shot latency",
        metrics.get("shot_pipeline_latency", []),
        metrics.get("shot_transport_dispatch_remainder", []),
        metrics.get("shot_decoder_processing_time", []),
        "pipeline",
    ):
        print(line)
    print("======================================================")


def run_job(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout: float,
    echo_command: bool = True,
    echo_output: bool = True,
) -> tuple[int, str]:
    command_line = " ".join(cmd)
    if echo_command:
        print(f"[run] qua: {command_line}")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
        )
        out = proc.stdout or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        out = stdout + stderr + f"\n[run] QUA timeout after {timeout}s\n"
        rc = 124
    log_path.write_text(out)
    if echo_output or rc != 0:
        if not echo_command and rc != 0:
            print(f"[run] qua: {command_line}")
        print(out, end="")
    return rc, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-stream", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--qua-timeout", type=float, default=180.0)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--host-exit-timeout", type=float, default=30.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="shots per QUA program. 0 (default) bakes every requested shot "
        "into a single program (one host phase), since the baked data fits "
        "easily in OPX data memory. Set >0 to split into multiple phases.",
    )
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--start-shot", type=int, default=0)
    parser.add_argument(
        "--tests",
        default=",".join(("multi-lut", "nvqldpc", "trt")),
        help=f"comma-separated decoder tests to run ({available_decoder_tests()})",
    )
    parser.add_argument("--trt-onnx-path", type=Path, default=DEFAULT_TRT_ONNX_PATH)
    parser.add_argument(
        "--trt-golden-path",
        type=Path,
        default=DEFAULT_TRT_GOLDEN_PATH,
        help="golden raw TRT output table consumed by the shim validator",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE.parent / "data",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--opx-ip", default="10.137.129.5")
    parser.add_argument("--opx-port", type=int, default=9510)
    parser.add_argument("--fem-index", type=int, default=5)
    parser.add_argument(
        "--verbose-batches",
        action="store_true",
        help="echo each internal data QUA program instead of only logging it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 0:
        print("ERROR: --batch-size must be >= 0", file=sys.stderr)
        return 2
    try:
        test_specs = expand_decoder_tests(args.tests)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not args.server.exists():
        print(f"ERROR: server not found: {args.server}", file=sys.stderr)
        return 2
    if not DATA_QUA.exists():
        print(f"ERROR: data QUA program not found: {DATA_QUA}", file=sys.stderr)
        return 2

    if any(spec.requires_trt for spec in test_specs):
        if not args.trt_onnx_path.is_file():
            print(f"ERROR: TRT ONNX model not found: {args.trt_onnx_path}",
                  file=sys.stderr)
            print(
                "Set --trt-onnx-path or CUDAQX_TRT_ONNX_PATH to the path of "
                "surface_code_decoder.onnx (available at "
                "/workspaces/cudaqx/assets/tests/surface_code_decoder.onnx).",
                file=sys.stderr,
            )
            return 2
        if is_git_lfs_pointer(args.trt_onnx_path):
            print(f"ERROR: TRT ONNX model is still a Git LFS pointer: "
                  f"{args.trt_onnx_path}", file=sys.stderr)
            return 2
        if not args.trt_golden_path.is_file():
            print(f"ERROR: TRT golden output table not found: "
                  f"{args.trt_golden_path}", file=sys.stderr)
            return 2

    jobs: list[SelectedJob] = []
    for spec in test_specs:
        syndromes_path = spec.syndromes_path(args.data_dir)
        if not syndromes_path.exists():
            print(f"ERROR: syndrome data not found for {spec.name}: "
                  f"{syndromes_path}", file=sys.stderr)
            return 2
        config_path = spec.config_path(args.data_dir)
        if not config_path.exists():
            print(f"ERROR: config not found for {spec.name}: {config_path}",
                  file=sys.stderr)
            return 2
        total_shots = count_shots(syndromes_path)
        start = args.start_shot
        available = max(0, total_shots - start)
        requested = (
            available if args.max_shots == 0 else min(args.max_shots, available)
        )
        if requested <= 0:
            print(f"ERROR: no shots selected for {spec.name}", file=sys.stderr)
            return 2
        # batch_size 0 => bake every requested shot into a single QUA program
        # (one host phase). The baked syndrome data fits easily in OPX data
        # memory, so phase batching is unnecessary by default.
        batch_size, num_batches = resolve_batch_plan(requested, args.batch_size)
        jobs.append(
            SelectedJob(
                spec, total_shots, start, requested, num_batches, batch_size
            )
        )

    ts = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = args.out_dir or Path("/tmp") / f"decoder-2_sequence_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = build_ld_library_path(args.server)
    base_env.setdefault("CUDAQ_LOG_LEVEL", "error")

    stdbuf = shutil.which("stdbuf")

    print(
        "[run] tests: "
        + ", ".join(
            f"{job.spec.name}(shots={job.requested}/{job.total_shots}, "
            f"batches={job.num_batches})"
            for job in jobs
        )
    )

    overall_rc = 0

    for job in jobs:
        spec = job.spec
        print(f"\n[run] ===== {spec.name}: starting server =====")

        # Resolve the YAML config: substitute ${TRT_ONNX_PATH}.
        resolved_config = out_dir / f"config_{spec.name}.yml"
        resolved_config.write_bytes(
            load_config_bytes(spec.config_path(args.data_dir), args.trt_onnx_path)
        )

        # Build server environment: TRT golden path if needed.
        server_env = base_env.copy()
        if spec.requires_trt:
            server_env["CUDAQX_TRT_GOLDEN_PATH"] = str(args.trt_golden_path)

        host_cmd = ([stdbuf, "-oL", "-eL"] if stdbuf else []) + [
            str(args.server),
            f"--config={resolved_config}",
            f"--data-stream={args.data_stream}",
            f"--timeout={args.server_timeout}",
            f"--max-phases={job.num_batches}",
        ]
        host = HostProcess(
            host_cmd, server_env,
            out_dir / f"decoder.{spec.name}.host.log",
        )
        print(f"[run] host: {' '.join(host_cmd)}", flush=True)
        host.start()

        try:
            print(
                f"[run] {spec.name}: configuring decoder bank and waiting for "
                f"OPNIC handshake (up to {args.ready_timeout:.0f}s)...",
                flush=True,
            )
            if not host.wait_ready(args.ready_timeout):
                print(f"[run] ERROR: host ({spec.name}) did not reach OPNIC handshake")
                print("\n".join(host.tail[-20:]))
                overall_rc = 1
                continue
            if host.poll() is not None:
                print(f"[run] ERROR: host ({spec.name}) exited before QUA started")
                print("\n".join(host.tail[-20:]))
                overall_rc = 1
                continue
            print(f"[run] {spec.name}: server ready", flush=True)

            common_qua_args = [
                f"--opx-ip={args.opx_ip}",
                f"--opx-port={args.opx_port}",
                f"--fem-index={args.fem_index}",
                f"--data-dir={args.data_dir}",
            ]

            done = 0
            batch_index = 0
            data_summaries: list[dict] = []
            print(
                f"[run] === {spec.name}: data "
                f"({job.num_batches} internal QUA program"
                f"{'' if job.num_batches == 1 else 's'}) ==="
            )
            while done < job.requested:
                n = min(job.batch_size, job.requested - done)
                batch_start = job.start + done
                batch_cmd = [
                    args.python,
                    str(DATA_QUA),
                    *common_qua_args,
                    f"--test={spec.name}",
                    f"--start-shot={batch_start}",
                    f"--max-shots={n}",
                    "--phase-done",
                    "--summary-json",
                ]
                # Print progress before launching the batch so a long-running
                # job is visibly making progress (not stuck). --verbose-batches
                # additionally echoes the full QUA program output.
                if args.verbose_batches:
                    print(f"[run] --- {spec.name}: data batch {batch_index} ---")
                else:
                    started = dt.datetime.now()
                    print(
                        f"[run] {spec.name}: batch {batch_index + 1}/"
                        f"{job.num_batches} (shots {batch_start}-"
                        f"{batch_start + n - 1}) running...",
                        flush=True,
                    )
                rc, out = run_job(
                    batch_cmd,
                    os.environ.copy(),
                    out_dir / f"decoder.{spec.name}.data.{batch_index:03d}.qua.log",
                    args.qua_timeout,
                    echo_command=args.verbose_batches,
                    echo_output=args.verbose_batches,
                )
                if rc != 0:
                    print(
                        f"[run] ERROR: {spec.name} data QUA batch {batch_index} "
                        f"(start={batch_start}, n={n}) failed rc={rc}",
                        flush=True,
                    )
                    overall_rc = 1
                    break
                try:
                    summary = parse_data_summary(out)
                    data_summaries.append(summary)
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"[run] ERROR: {spec.name} data QUA batch {batch_index} "
                        f"did not produce a usable summary: {e}",
                        flush=True,
                    )
                    overall_rc = 1
                    break
                if not args.verbose_batches:
                    elapsed = (dt.datetime.now() - started).total_seconds()
                    print(
                        f"[run] {spec.name}: batch {batch_index + 1}/"
                        f"{job.num_batches} done in {elapsed:.1f}s "
                        f"({format_batch_result(summary)})",
                        flush=True,
                    )
                done += n
                batch_index += 1

            host_rc = host.wait_exit(args.host_exit_timeout)
            decode_rpcs = host.stats()
            print(
                f"[run] host_exit={host_rc} decode_rpcs={decode_rpcs}"
            )
            for line in host.trt_raw_validation_summaries():
                print(f"[run] {line}")

            if data_summaries:
                print_decoder_data_summary(
                    aggregate_data_summaries(data_summaries)
                )

            if host_rc != 0:
                overall_rc = 1

        finally:
            host.terminate()
            host.close()

    print(f"\n[run] logs: {out_dir}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
