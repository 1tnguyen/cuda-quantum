# decoder_server_cpu

`decoder_server_cpu` is a real OPNIC server for CUDA-QX realtime QEC decoder
RPCs. It loads `libcudaqx-decoder-hostcall.so` at runtime, receives decoder
configuration from QUA over a control stream, and receives decode requests over a
separate data stream.

## Streams

| stream | default id | carries | slot profile |
|---|---:|---|---|
| data | 1 | `reset_decoder`, `enqueue_syndromes`, `get_corrections` | 64 B packets |
| control | 2 | chunked `configure_decoder` | 1048 B payload plus RPC header |

The server services both streams on one poll thread. That serializes
configuration and decode calls against the shared CUDA-QX decoder bank without a
lock on the data path.

The control payload is one configure chunk frame:

```text
[magic][version][total_bytes][offset_bytes][chunk_bytes][flags][yaml bytes...]
```

`magic` is `CONFIG_CHUNK_MAGIC` (`0x43585143`). It is sent on the little-endian
control stream as bytes `43 51 58 43`, the ASCII marker `"CQXC"`. The shim uses
it to recognize the configure chunk ABI before reading the version, total size,
offset, chunk length, and flags fields.

The QUA helpers currently send at most 1024 YAML bytes per chunk. That number is
only a conservative value we picked after observing the real QUA/OPNIC
external-stream path fail around the 4 KiB packet size boundary. It is not a
semantic decoder limit; the hard rule is that each chunk frame must fit the
transport. The shim assembles chunks in order and calls CUDA-QX only after the
final chunk arrives.

Each QUA program sends a `function_id == 0` packet when its phase is complete.
The server then recreates the OPNIC contexts and synchronizes with the next QUA
program while keeping the loaded shim and decoder bank alive. This packet is a
host/QUA phase marker, not a decoder RPC and not a server shutdown. It allows a
single long-running server to accept config, data playback, reconfig, and more
data playback as separate QUA jobs.

## Build Prerequisites

Build the shim first:

```bash
cd ../../cudaqx_decoder_hostcall
./build.sh
source /tmp/cudaqx-decoder-hostcall-build/shim-env.sh
```

The shim build also fetches CUDA-QX's real TensorRT
`surface_code_decoder.onnx` model with Git LFS and exports
`CUDAQX_TRT_ONNX_PATH` plus `CUDAQX_TRT_GOLDEN_PATH`. On aarch64/SBSA
containers it installs the TensorRT runtime/parser apt packages when they are
missing and the script is running as root.

Build `hello_opx` normally with real OPNIC enabled. The sequence runner assumes
the server is at:

```text
/tmp/cudaq-realtime-hello-opx-build/examples/hello_opx/decoder/decoder_server_cpu
```

Override that with `--server` if your build directory is different.

## Run the Decoder Handoff Test

From this directory:

```bash
sudo -E python3 run_decoder_sequence.py \
  --server=/path/to/decoder_server_cpu \
  --shim="${CUDAQX_SHIM_LIBRARY}" \
  --venv="${CUDAQX_SHIM_VENV}" \
  --trt-onnx-path="${CUDAQX_TRT_ONNX_PATH}" \
  --trt-golden-path="${CUDAQX_TRT_GOLDEN_PATH}" \
  --batch-size=10 \
  --server-timeout=900 \
  --qua-timeout=180
```

By default, the runner uses one long-running `decoder_server_cpu` process and
runs all three decoder jobs in sequence:

| test name | decoder | dataset |
|---|---|---|
| `multi-lut` | `multi_error_lut` | full multi-error LUT dataset |
| `nvqldpc` | `nv-qldpc-decoder` | cuda-qx realtime relay dataset |
| `trt` | `trt_decoder` | CUDA-QX `surface_code_decoder.onnx` with checked-in golden raw outputs |

The runner:

1. Starts one `decoder_server_cpu` process.
2. For each selected test, runs `decoder_config_qua.py` to send that decoder's
   YAML in 1024-byte chunks on the control stream.
3. Runs `decoder_data_qua.py` repeatedly to play back that decoder's syndrome
   dataset in batches on the data stream.
4. Keeps the same server process alive across config/data/config transitions,
   validating handoff between decoder jobs.
5. Validates every OPNIC response and every expected correction.
6. For `trt`, validates the decoder's raw TensorRT probability output against
   `data/trt_surface_code_golden.txt` before CUDA-QX reduces it to observable
   correction bits.

The raw TRT validation is enabled by `CUDAQX_TRT_GOLDEN_PATH` in the server
environment. It intentionally runs an extra raw `decode()` on each TRT syndrome
so it can compare TensorRT probabilities to the golden table. That gives a real
model validation without PyTorch at runtime, but the reported TRT decoder
processing time includes this validation overhead. Unset `CUDAQX_TRT_GOLDEN_PATH`
for latency-only TRT runs.

Select a subset with `--tests`:

```bash
sudo -E python3 run_decoder_sequence.py \
  --server=/path/to/decoder_server_cpu \
  --shim="${CUDAQX_SHIM_LIBRARY}" \
  --venv="${CUDAQX_SHIM_VENV}" \
  --tests=multi-lut,nvqldpc
```

`--max-shots` and `--start-shot` apply to each selected dataset. For a short
handoff check across all decoders, use:

```bash
sudo -E python3 run_decoder_sequence.py \
  --server=/path/to/decoder_server_cpu \
  --shim="${CUDAQX_SHIM_LIBRARY}" \
  --venv="${CUDAQX_SHIM_VENV}" \
  --trt-onnx-path="${CUDAQX_TRT_ONNX_PATH}" \
  --trt-golden-path="${CUDAQX_TRT_GOLDEN_PATH}" \
  --max-shots=3 \
  --batch-size=3
```

The QUA summaries split timing into three parts:

| field | meaning |
|---|---|
| round-trip latency | QUA timestamp before send through timestamp after receive |
| decoder processing time | host-measured shim/CUDA-QX handler duration, returned in the response timing field |
| transport/dispatch remainder | round-trip minus decoder processing; includes OPX/OPNIC transport, host polling, packet copy, and doorbell overhead |

The bundled configs and datasets are in `data/`:

```text
data/config_multi_err_lut.yml
data/syndromes_multi_err_lut.txt
data/config_nvqldpc_relay.yml
data/syndromes_nvqldpc_relay.txt
data/config_trt_surface_code.yml
data/syndromes_trt_surface_code.txt
data/trt_surface_code_golden.txt
```

## Long-Running Server Mode

`run_decoder_sequence.py` passes `--max-phases` so the test exits cleanly after
config plus all data batches. For a standing server, launch the server directly
without `--max-phases` or with `--max-phases=0`:

```bash
sudo -E /path/to/decoder_server_cpu \
  --shim="${CUDAQX_SHIM_LIBRARY}" \
  --data-stream=1 \
  --control-stream=2 \
  --timeout=0
```

You can then run config and data QUA programs independently:

```bash
python3 decoder_config_qua.py --test=multi-lut
python3 decoder_data_qua.py --test=multi-lut --start-shot=0 --max-shots=10 --phase-done

python3 decoder_config_qua.py --test=nvqldpc
python3 decoder_data_qua.py --test=nvqldpc --start-shot=0 --max-shots=10 --phase-done

python3 decoder_config_qua.py --test=trt --trt-onnx-path="${CUDAQX_TRT_ONNX_PATH}"
python3 decoder_data_qua.py --test=trt --phase-done
```

Run another config phase later to switch decoder type or decoder parameters; the
shim finalizes the old decoder bank and installs the new one after the final
config chunk arrives.
