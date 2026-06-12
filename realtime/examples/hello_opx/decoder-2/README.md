# decoder-2

`decoder-2` is a CPU OPNIC realtime server for CUDA-QX QEC decoder RPCs.
It links the CUDA-QX decoder library directly (no `dlopen`, no separate build
script, no Python wheel or venv), configures the decoder bank host-side from a
YAML file before any QUA program runs, and serves decode RPCs on a single data
stream.

## How it differs from the `decoder` example

| aspect | `decoder` | `decoder-2` |
|---|---|---|
| handler loading | `dlopen` out-of-tree shim | compiled into the binary |
| decoder config | QUA sends chunked YAML over a control stream | host reads `--config=<yaml>` at startup |
| OPNIC streams | data + control | data only |
| build dependencies | build.sh, pip wheel, GitHub checkout | `~/.cudaqx` + `/usr/local/cudaq` |

## Directory layout

```text
decoder-2/
  CMakeLists.txt
  README.md
  src/
    decoder_server_cpu.cpp   # main: host-side configure + single-stream serve
    decoder_handlers.h       # in-process function-table API (3 decode verbs)
    decoder_handlers.cpp     # reset/enqueue/get handlers + TRT golden validator
    decoder_packets.h        # data-stream OPNIC packets
  qua/
    decoder_qua.py           # data-stream packet helpers
    decoder_specs.py         # decoder catalog; default ONNX/golden paths
    decoder_data_qua.py      # data playback (data stream only)
    run_decoder_sequence.py  # per-decoder server restart with --config
  data/
    config_multi_err_lut.yml
    config_nvqldpc_relay.yml
    config_trt_surface_code.yml
    syndromes_multi_err_lut.txt
    syndromes_nvqldpc_relay.txt
    syndromes_trt_surface_code.txt
    trt_surface_code_golden.txt
```

## Build prerequisites

CUDA-QX must be installed at `~/.cudaqx` and CUDA-Q at `/usr/local/cudaq`.
Both are present in the standard development container.

The TRT ONNX model is available at:

```text
/workspaces/cudaqx/assets/tests/surface_code_decoder.onnx
```

TensorRT (`libnvinfer.so.10`, `libnvonnxparsers.so.10`) must be on the system
library path (installed by default in the development container).

## Build

Build inside the `build-realtime` tree:

```bash
cmake -S /workspaces/cuda-quantum/realtime \
      -B /workspaces/cuda-quantum/build-realtime
cmake --build /workspaces/cuda-quantum/build-realtime \
      --target decoder_2_cpu -j$(nproc)
```

The binary is placed at:

```text
/workspaces/cuda-quantum/build-realtime/examples/hello_opx/decoder-2/decoder-2_cpu
```

Override the CUDA-QX or CUDA-Q install paths if they differ:

```bash
cmake -S /workspaces/cuda-quantum/realtime \
      -B /workspaces/cuda-quantum/build-realtime \
      -DCUDAQX_INSTALL_DIR=/path/to/cudaqx \
      -DCUDAQ_INSTALL_DIR=/path/to/cudaq
```

## Streams

| stream | default id | carries | slot size |
|---|---|---|---|
| data | 1 | `reset_decoder`, `enqueue_syndromes`, `get_corrections` | 64 B |

The QUA program declares only the data stream. There is no control stream.

Serving uses the CPU OPNIC bridge host data-plane and the library generic host
loop rather than a bespoke poll loop. At the end of each batch the QUA program
sends a dedicated phase-done marker (`function_id = fnv1a("phase_complete")`,
a non-zero id distinct from `function_id 0`, which the shared loop reserves for
shutdown). The server's phase-done handler ends the loop so it can reconnect
for the next QUA program while keeping the decoder bank loaded.

## Run the decoder handoff test

From `qua/`:

```bash
sudo -E python3 run_decoder_sequence.py \
  --server=/workspaces/cuda-quantum/build-realtime/examples/hello_opx/decoder-2/decoder-2_cpu \
  --server-timeout=900 \
  --qua-timeout=180
```

This runs all three decoders in sequence. For each decoder the runner:

1. Writes the decoder's YAML config (with `${TRT_ONNX_PATH}` substituted) to a
   temp file.
2. Starts a fresh `decoder-2_cpu --config=<resolved.yml>`. The server
   configures the bank host-side, then waits for the first QUA program.
3. Runs `decoder_data_qua.py` once on the data stream. By default every
   requested shot is baked into a single QUA program (one host phase): the
   syndrome data fits easily in OPX data memory, and the program iterates the
   baked arrays with QUA `for_` loops instead of unrolling one RPC per call.
   Pass `--batch-size=N` to split into multiple phases instead.
4. Waits for the server to exit, then moves to the next decoder.

| test name | decoder | dataset |
|---|---|---|
| `multi-lut` | `multi_error_lut` | full multi-error LUT dataset |
| `nvqldpc` | `nv-qldpc-decoder` | CUDA-QX realtime relay dataset |
| `trt` | `trt_decoder` | surface-code ONNX with TRT golden validation |

Run a subset with `--tests`:

```bash
sudo -E python3 run_decoder_sequence.py \
  --tests=multi-lut,nvqldpc
```

Quick smoke test across all three decoders:

```bash
sudo -E python3 run_decoder_sequence.py \
  --max-shots=3 --batch-size=3
```

### TRT ONNX path

By default the runner uses:

```text
/workspaces/cudaqx/assets/tests/surface_code_decoder.onnx
```

Override with `--trt-onnx-path` or `CUDAQX_TRT_ONNX_PATH`.

### TRT golden output validation

The shim performs an extra raw `decode()` on each TRT syndrome to compare
TensorRT probabilities against `data/trt_surface_code_golden.txt`. Override the
golden file with `--trt-golden-path` or `CUDAQX_TRT_GOLDEN_PATH`. Set
`CUDAQX_TRT_GOLDEN_PATH` to an empty string to disable validation.

## Long-running server mode

Launch the server directly without `--max-phases` (or `--max-phases=0`) for a
standing service:

```bash
sudo -E /path/to/decoder-2_cpu \
  --config=data/config_multi_err_lut.yml \
  --data-stream=1 \
  --timeout=0
```

Run data batches independently:

```bash
python3 qua/decoder_data_qua.py --test=multi-lut --start-shot=0 --max-shots=10 --phase-done
```

To switch decoder type, restart the server with a different `--config`.

## Server options

```text
--config=<path>     YAML decoder config file (required)
--data-stream=N     OPNIC stream id for decode RPCs (default 1)
--data-buffers=N    data-ring slots (default 1024)
--max-phases=N      exit after N QUA phases (default 0: unlimited)
--timeout=N         auto-shutdown after N seconds (default 60)
```

## Timing

The QUA summaries split timing into three parts:

| field | meaning |
|---|---|
| round-trip latency | QUA timestamp before send through timestamp after receive |
| decoder processing time | host-measured handler duration, returned in ptp_timestamp |
| transport/dispatch remainder | round-trip minus decoder processing time |
