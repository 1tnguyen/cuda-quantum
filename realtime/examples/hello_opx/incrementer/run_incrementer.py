#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Run the incrementer example across its dispatch variants and print a single,
human-readable round-trip-latency comparison:

    CPU 3-thread       (incrementer_cpu)           CUDAQ_EXEC_HOST
    CPU unified        (incrementer_cpu --unified)   CUDAQ_EXEC_HOST_UNIFIED (generic loop)
    GPU 3-kernel       (incrementer_gpu)           CUDAQ_EXEC_GPU_PERSISTENT
    GPU unified        (incrementer_gpu --unified)   CUDAQ_EXEC_GPU_UNIFIED (generic loop)

For each variant the orchestrator:
  1. spawns the host binary (under `sudo` unless --no-sudo / already root),
     tees its stdout to a log, and waits for the OPNIC handshake line;
  2. runs the QUA program (`incrementer_qua.py`), which drives the OPX,
     measures per-call round-trip latency, and prints a stats block;
  3. parses the QUA stats + the host's "Total RPCs dispatched" count.

It then renders one comparison table to the terminal (and optionally JSON).

Hardware: the host needs root to mmap the OPNIC PCIe BARs, and the QUA side
needs a reachable OPX (configured inside incrementer_qua.py). This script only
orchestrates and reports; it does not require either to be present to import.

Example
-------
    sudo -v && python3 run_incrementer.py
    python3 run_incrementer.py --configs cpu_unified,gpu_unified --json out.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
QUA_PROGRAM = HERE / "incrementer_qua.py"
HOST_READY_MARKER = "Waiting for OPX"

# sizeof(RPCHeader) == sizeof(RPCResponse), per rpc_wire_format.h
# (CUDAQ_RPC_HEADER_SIZE). The on-wire RPC packet is this header plus the
# payload; the payload size dominates the transfer cost and thus latency.
RPC_HEADER_BYTES = 24

# (key, label, host kind, extra host args, executor name)
CONFIGS = [
    ("cpu_3thread", "CPU 3-thread", "cpu", [], "CUDAQ_EXEC_HOST"),
    ("cpu_unified", "CPU unified", "cpu", ["--unified"],
     "CUDAQ_EXEC_HOST_UNIFIED (generic loop)"),
    ("gpu_3kernel", "GPU 3-kernel", "gpu", [], "CUDAQ_EXEC_GPU_PERSISTENT"),
    ("gpu_unified", "GPU unified", "gpu", ["--unified"],
     "CUDAQ_EXEC_GPU_UNIFIED (generic loop)"),
]
CONFIG_KEYS = [c[0] for c in CONFIGS]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class Result:
    key: str
    label: str
    executor: str
    ok: bool = False
    error: Optional[str] = None
    # Latency stats (ns), parsed from the QUA report (first call discarded).
    valid_ok: Optional[int] = None
    valid_total: Optional[int] = None
    first_ns: Optional[float] = None
    min_ns: Optional[float] = None
    avg_ns: Optional[float] = None
    p95_ns: Optional[float] = None
    p99_ns: Optional[float] = None
    max_ns: Optional[float] = None
    stddev_ns: Optional[float] = None
    host_rpcs: Optional[int] = None


# ---------------------------------------------------------------------------
# Binary / repo discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Optional[Path]:
    cur = start
    for _ in range(12):
        if (cur / "build").is_dir() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def default_build_dirs(repo: Optional[Path]) -> list[Path]:
    """Candidate build trees, in priority order."""
    cands: list[Path] = []
    if repo is not None:
        cands += [repo / "build", repo / "realtime" / "build"]
    # The realtime standalone build, relative to this file.
    cands.append(HERE.parents[2] / "build")  # .../realtime/build
    seen, out = set(), []
    for c in cands:
        c = c.resolve()
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def locate_binary(name: str, build_dirs: list[Path]) -> Optional[Path]:
    rel = Path("examples") / "hello_opx" / "incrementer" / name
    for bd in build_dirs:
        cand = bd / rel
        if cand.exists():
            return cand
    return None


def runtime_lib_dirs(build_dirs: list[Path]) -> list[str]:
    """Directories that must be on LD_LIBRARY_PATH to run the host binaries."""
    dirs: list[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            dirs.append(resolved)

    for bd in build_dirs:
        lib = bd / "lib"
        if (lib / "libcudaq-realtime.so").exists():
            add(lib)

    for prefix in (Path("/usr/local/lib"),
                     Path.home() / ".cudaq_realtime" / "lib"):
        if (prefix / "libcudaq-realtime.so").exists():
            add(prefix)

    return dirs


def prepend_ld_library_path(lib_dirs: list[str]) -> None:
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = lib_dirs + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


# ---------------------------------------------------------------------------
# Host process (spawn + tee + ready event)
# ---------------------------------------------------------------------------

class HostProcess:
    def __init__(self, cmd: list, log_path: Path):
        self.cmd = cmd
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logf = None
        self.tail: list[str] = []  # last lines, for diagnostics

    def start(self) -> None:
        self._logf = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            if self._logf:
                self._logf.write(line)
                self._logf.flush()
            self.tail.append(line.rstrip("\n"))
            del self.tail[:-40]
            if HOST_READY_MARKER in line:
                self.ready.set()
        self.ready.set()  # stream closed -> unblock waiters

    def wait_ready(self, timeout: float) -> bool:
        return self.ready.wait(timeout=timeout)

    def poll(self):
        return self.proc.poll() if self.proc else None

    def wait_exit(self, timeout: float) -> Optional[int]:
        if not self.proc:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def close(self) -> None:
        if self._logf:
            self._logf.close()
            self._logf = None

    def host_rpcs(self) -> Optional[int]:
        for line in reversed(self.tail):
            m = re.search(r"Total RPCs dispatched:\s*(\d+)", line)
            if m:
                return int(m.group(1))
        # Also scan the full log (tail may have rotated).
        try:
            for line in reversed(self.log_path.read_text().splitlines()):
                m = re.search(r"Total RPCs dispatched:\s*(\d+)", line)
                if m:
                    return int(m.group(1))
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# QUA latency parsing
# ---------------------------------------------------------------------------

_FLOAT = r"([-+]?\d+(?:\.\d+)?)"
_QUA_PATTERNS = {
    "first_ns": re.compile(rf"First:\s*{_FLOAT}\s*ns"),
    "min_ns": re.compile(rf"Min:\s*{_FLOAT}\s*ns"),
    "max_ns": re.compile(rf"Max:\s*{_FLOAT}\s*ns"),
    "avg_ns": re.compile(rf"Avg:\s*{_FLOAT}\s*ns"),
    "stddev_ns": re.compile(rf"Stddev:\s*{_FLOAT}\s*ns"),
    "p95_ns": re.compile(rf"95th percentile:\s*{_FLOAT}\s*ns"),
    "p99_ns": re.compile(rf"99th percentile:\s*{_FLOAT}\s*ns"),
}
_VALID_RE = re.compile(r"Valid:\s*(\d+)\s*/\s*(\d+)")


def rpc_payload_bytes(qua_path: Path, default: int = 4) -> int:
    """RPC payload size (bytes), read from the QUA program's `arg_len`
    assignment so the reported packet size tracks the actual stimulus."""
    try:
        m = re.search(r"arg_len\[0\]\s*,\s*(\d+)", qua_path.read_text())
        return int(m.group(1)) if m else default
    except OSError:
        return default


def parse_qua_output(text: str, res: Result) -> None:
    for field_name, rx in _QUA_PATTERNS.items():
        m = rx.search(text)
        if m:
            setattr(res, field_name, float(m.group(1)))
    m = _VALID_RE.search(text)
    if m:
        res.valid_ok = int(m.group(1))
        res.valid_total = int(m.group(2))


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------

def build_host_cmd(sudo_prefix: list, binary: Path, extra: list,
                   timeout_sec: int) -> list:
    stdbuf = shutil.which("stdbuf")
    buf = [stdbuf, "-oL", "-eL"] if stdbuf else []
    cmd = list(sudo_prefix) + buf + [str(binary)] + list(extra)
    # Only the CPU host accepts --timeout; the GPU host runs until shutdown.
    if binary.name.endswith("_cpu"):
        cmd += [f"--timeout={timeout_sec}"]
    return cmd


def run_config(cfg, binary: Path, args, out_dir: Path,
               sudo_prefix: list) -> Result:
    key, label, _kind, extra, executor = cfg
    res = Result(key=key, label=label, executor=executor)

    host_cmd = build_host_cmd(sudo_prefix, binary, extra, args.host_timeout)
    host = HostProcess(host_cmd, out_dir / f"{key}.host.log")
    print(f"[run] {label}: {' '.join(host_cmd)}")
    host.start()

    try:
        if not host.wait_ready(args.ready_timeout):
            if host.poll() is not None:
                res.error = "host exited before OPNIC handshake"
            else:
                res.error = (f"host did not reach '{HOST_READY_MARKER}' "
                             f"within {args.ready_timeout}s")
            return res
        if host.poll() is not None:
            res.error = "host exited right after becoming ready"
            return res

        # Drive the OPX from the QUA side.  Bounded by --qua-timeout so a host
        # that never services RPCs (e.g. a broken transport path) fails this
        # variant instead of hanging the whole comparison: subprocess.run kills
        # the QUA child on timeout and re-raises TimeoutExpired.
        qua_log = out_dir / f"{key}.qua.log"
        qua_cmd = [args.python, str(QUA_PROGRAM)]
        print(f"[run] {label}: qua -> {' '.join(qua_cmd)}")
        qua_out = ""
        qua_rc = None
        try:
            qua = subprocess.run(qua_cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 timeout=args.qua_timeout)
            qua_out = qua.stdout or ""
            qua_rc = qua.returncode
        except subprocess.TimeoutExpired as e:
            partial = e.stdout
            if isinstance(partial, (bytes, bytearray)):
                partial = partial.decode(errors="replace")
            qua_out = partial or ""
            res.error = (f"QUA driver did not finish within "
                         f"{args.qua_timeout:g}s (host likely not servicing "
                         f"RPCs; see {qua_log})")
        qua_log.write_text(qua_out)
        if qua_rc is not None and qua_rc != 0 and res.error is None:
            res.error = f"QUA program exited {qua_rc} (see {qua_log})"
        parse_qua_output(qua_out, res)

        host.wait_exit(timeout=args.host_timeout)
        res.host_rpcs = host.host_rpcs()

        # Consider it a success if we got latency numbers back.
        if res.avg_ns is not None and res.error is None:
            res.ok = True
        elif res.avg_ns is not None and qua_rc == 0:
            res.ok = True
    finally:
        host.terminate()
        host.close()
    return res


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class Style:
    """ANSI styling, auto-disabled when stdout is not a TTY or --no-color."""
    def __init__(self, enabled: bool):
        self.on = enabled
    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s
    def bold(self, s): return self._w("1", s)
    def dim(self, s): return self._w("2", s)
    def green(self, s): return self._w("1;32", s)
    def red(self, s): return self._w("31", s)
    def cyan(self, s): return self._w("36", s)


def _fmt_ns(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:,.1f}"


def render_table(results: list[Result], st: Style) -> str:
    # Column spec: (header, accessor -> str, raw value for "best", lower_is_better)
    def valid_str(r: Result) -> str:
        if r.valid_ok is None:
            return "-"
        return f"{r.valid_ok}/{r.valid_total}"

    cols = [
        ("Variant", lambda r: r.label, None, None),
        ("Valid", valid_str, None, None),
        ("Min", lambda r: _fmt_ns(r.min_ns), lambda r: r.min_ns, True),
        ("Avg", lambda r: _fmt_ns(r.avg_ns), lambda r: r.avg_ns, True),
        ("p95", lambda r: _fmt_ns(r.p95_ns), lambda r: r.p95_ns, True),
        ("p99", lambda r: _fmt_ns(r.p99_ns), lambda r: r.p99_ns, True),
        ("Max", lambda r: _fmt_ns(r.max_ns), lambda r: r.max_ns, True),
        ("Stddev", lambda r: _fmt_ns(r.stddev_ns), lambda r: r.stddev_ns, True),
        ("RPCs", lambda r: ("-" if r.host_rpcs is None else f"{r.host_rpcs:,}"),
         None, None),
    ]

    # Best (lowest) value per metric column, for highlighting.
    best = {}
    for ci, (_h, _f, raw, lower) in enumerate(cols):
        if not lower:
            continue
        vals = [(raw(r)) for r in results if r.ok and raw(r) is not None]
        if vals:
            best[ci] = min(vals)

    # Raw (unstyled) cell text, to measure widths.
    raw_rows = []
    for r in results:
        raw_rows.append([f(r) for (_h, f, _raw, _l) in cols])
    headers = [h for (h, _f, _r, _l) in cols]
    widths = [len(h) for h in headers]
    for row in raw_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def hrule(left, mid, right, fill="\u2500"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def fmt_row(cells, styler=None, highlight_best=False, res=None):
        out = []
        for i, cell in enumerate(cells):
            pad = " " * (widths[i] - len(cell))
            # First column left-aligned, the rest right-aligned.
            body = (cell + pad) if i == 0 else (pad + cell)
            shown = body
            if styler:
                shown = styler(body)
            elif highlight_best and res is not None:
                raw = cols[i][2]
                if (i in best and raw is not None and res.ok
                        and raw(res) is not None and raw(res) == best[i]):
                    shown = st.green(st.bold(body))
            out.append(" " + shown + " ")
        return "\u2502" + "\u2502".join(out) + "\u2502"

    lines = []
    lines.append(hrule("\u250c", "\u252c", "\u2510"))
    lines.append(fmt_row(headers, styler=st.bold))
    lines.append(hrule("\u251c", "\u253c", "\u2524"))
    for r, raw in zip(results, raw_rows):
        if not r.ok:
            # Render the variant name + a spanning error note.
            name = raw[0]
            pad = " " * (widths[0] - len(name))
            note = st.red(f"FAILED: {r.error or 'no latency data'}")
            cell0 = "\u2502 " + st.bold(name + pad) + " \u2502 "
            lines.append(cell0 + note)
        else:
            lines.append(fmt_row(raw, highlight_best=True, res=r))
    lines.append(hrule("\u2514", "\u2534", "\u2518"))
    return "\n".join(lines)


def print_report(args, results: list[Result], out_dir: Path,
                 st: Style, payload_bytes: int) -> None:
    title = "incrementer round-trip latency comparison"
    bar = "\u2550" * max(len(title), 60)
    pkt = RPC_HEADER_BYTES + payload_bytes
    print()
    print(st.cyan(st.bold(bar)))
    print(st.cyan(st.bold(title)))
    print(st.cyan(st.bold(bar)))
    print(f"{st.dim('OPX-driven calls:')}  {args.iterations} per variant "
          f"(first discarded)")
    print(f"{st.dim('RPC packet:')}        {pkt} B each way "
          f"({RPC_HEADER_BYTES} B header + {payload_bytes} B payload)")
    print(f"{st.dim('Latencies:')}         nanoseconds, round-trip "
          f"(OPX send -> response)")
    print(f"{st.dim('Logs:')}              {out_dir}/")
    print()
    print(render_table(results, st))
    print()

    # Headline: fastest by average among successful runs.
    ok = [r for r in results if r.ok and r.avg_ns is not None]
    if ok:
        best = min(ok, key=lambda r: r.avg_ns)
        worst = max(ok, key=lambda r: r.avg_ns)
        line = (f"Fastest average: {st.green(st.bold(best.label))} "
                f"({best.avg_ns:,.1f} ns)")
        if worst.avg_ns and best.avg_ns and best is not worst:
            line += f"  \u2014  {worst.avg_ns / best.avg_ns:.2f}x faster than "\
                    f"{worst.label} ({worst.avg_ns:,.1f} ns)"
        print(line)
    failed = [r for r in results if not r.ok]
    if failed:
        print(st.red(f"{len(failed)} variant(s) did not produce results: "
                     + ", ".join(r.label for r in failed)))
    print()


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def acquire_sudo(no_sudo: bool) -> list:
    if no_sudo or os.geteuid() == 0:
        return []
    try:
        subprocess.run(["sudo", "-v"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"ERROR: sudo authentication failed ({e}); re-run with "
                 f"--no-sudo if the host binaries already have the needed "
                 f"capabilities, or run this script as root.")
    # -E preserves LD_LIBRARY_PATH (and friends) for dynamically linked host
    # binaries such as incrementer_cpu / incrementer_gpu.
    return ["sudo", "-E", "-n"]


def parse_cli(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="all",
                   help="Comma-separated subset of "
                        f"{{{','.join(CONFIG_KEYS)}}} or 'all' (default).")
    p.add_argument("--build-dir", type=Path, default=None,
                   help="Build tree holding the incrementer binaries "
                        "(default: auto-detect <repo>/build).")
    p.add_argument("--cpu-binary", type=Path, default=None,
                   help="Override path to incrementer_cpu.")
    p.add_argument("--gpu-binary", type=Path, default=None,
                   help="Override path to incrementer_gpu.")
    p.add_argument("--iterations", type=int, default=1024,
                   help="Calls per variant (informational; matches the value "
                        "baked into incrementer_qua.py).")
    p.add_argument("--ready-timeout", type=float, default=120.0,
                   help="Seconds to wait for the host's OPNIC handshake "
                        "(default 120).")
    p.add_argument("--host-timeout", type=float, default=60.0,
                   help="Seconds to wait for the host to exit after QUA "
                        "finishes (default 60).")
    p.add_argument("--qua-timeout", type=float, default=90.0,
                   help="Seconds to wait for the QUA driver to finish before "
                        "giving up on a variant (default 90). Prevents one "
                        "non-responsive config (e.g. a host that never "
                        "services RPCs) from hanging the whole comparison.")
    p.add_argument("--settle-sec", type=float, default=2.0,
                   help="Pause between variants to let OPNIC streams reset "
                        "(default 2).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Log directory. Default /tmp/incrementer_run_<ts>.")
    p.add_argument("--json", type=Path, default=None,
                   help="Also write the comparison as JSON to this path.")
    p.add_argument("--no-sudo", action="store_true",
                   help="Don't validate/prepend sudo for the host spawn.")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colors in the report.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter for incrementer_qua.py (default: the one "
                        "running this script).")
    args = p.parse_args(argv)

    if args.configs.strip().lower() == "all":
        args.selected = CONFIG_KEYS
    else:
        sel = [c.strip() for c in args.configs.split(",") if c.strip()]
        bad = [c for c in sel if c not in CONFIG_KEYS]
        if bad:
            sys.exit(f"ERROR: unknown config(s): {', '.join(bad)}; "
                     f"choose from {', '.join(CONFIG_KEYS)}.")
        args.selected = sel
    return args


def main(argv=None) -> int:
    args = parse_cli(argv)
    st = Style(enabled=(not args.no_color) and sys.stdout.isatty())

    if not QUA_PROGRAM.exists():
        sys.exit(f"ERROR: QUA program not found: {QUA_PROGRAM}")

    repo = find_repo_root(HERE)
    build_dirs = ([args.build_dir] if args.build_dir
                  else default_build_dirs(repo))

    cpu_bin = args.cpu_binary or locate_binary("incrementer_cpu", build_dirs)
    gpu_bin = args.gpu_binary or locate_binary("incrementer_gpu", build_dirs)

    # Work directory for logs.
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = Path("/tmp") / f"incrementer_run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate availability of the binaries we need for the selection.
    need_cpu = any(c[2] == "cpu" for c in CONFIGS if c[0] in args.selected)
    need_gpu = any(c[2] == "gpu" for c in CONFIGS if c[0] in args.selected)
    if need_cpu and (cpu_bin is None or not Path(cpu_bin).exists()):
        sys.exit("ERROR: incrementer_cpu not found. Build it first "
                 "(cmake --build <build> --target incrementer_cpu) or pass "
                 "--cpu-binary / --build-dir.")
    if need_gpu and (gpu_bin is None or not Path(gpu_bin).exists()):
        sys.exit("ERROR: incrementer_gpu not found. Build it first "
                 "(cmake --build <build> --target incrementer_gpu) or pass "
                 "--gpu-binary / --build-dir.")

    payload_bytes = rpc_payload_bytes(QUA_PROGRAM)

    lib_dirs = runtime_lib_dirs(build_dirs)
    if lib_dirs:
        prepend_ld_library_path(lib_dirs)
        print(f"[run] LD_LIBRARY_PATH includes: {':'.join(lib_dirs)}")

    sudo_prefix = acquire_sudo(args.no_sudo)

    results: list[Result] = []
    selected_cfgs = [c for c in CONFIGS if c[0] in args.selected]
    for i, cfg in enumerate(selected_cfgs):
        binary = Path(cpu_bin if cfg[2] == "cpu" else gpu_bin)
        res = run_config(cfg, binary, args, out_dir, sudo_prefix)
        results.append(res)
        if i + 1 < len(selected_cfgs) and args.settle_sec > 0:
            time.sleep(args.settle_sec)

    print_report(args, results, out_dir, st, payload_bytes)

    if args.json is not None:
        payload = {
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "iterations": args.iterations,
            "rpc_header_bytes": RPC_HEADER_BYTES,
            "rpc_payload_bytes": payload_bytes,
            "rpc_packet_bytes": RPC_HEADER_BYTES + payload_bytes,
            "out_dir": str(out_dir),
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"[run] JSON report: {args.json}")

    # Persist a copy alongside the logs too.
    (out_dir / "comparison.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2))

    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
