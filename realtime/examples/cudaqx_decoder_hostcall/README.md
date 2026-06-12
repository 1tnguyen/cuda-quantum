# cudaqx-decoder-hostcall

This directory builds a small out-of-tree shared library that exposes CUDA-QX
realtime QEC decoders as CUDA-Q realtime `HOST_CALL` handlers. The OPNIC decoder
server loads this library with `dlopen` and serves the handlers over two real
OPNIC streams.

The build is intentionally wheel-based: it creates a Python venv, installs the
CUDA-QX QEC wheel, uses the CUDA-Q libraries pulled by that wheel, and fetches
the few headers that the wheel does not currently install.

## ABI

The shim exports:

| symbol | purpose |
|---|---|
| `cudaqx_rt_get_function_table(entries, capacity, count)` | returns four `HOST_CALL` entries |
| `cudaqx_rt_finalize()` | tears down the decoder bank |

The function table contains:

| function id | verb |
|---|---|
| `fnv1a_hash("reset_decoder_ui64")` | reset decoder state |
| `fnv1a_hash("enqueue_syndromes_ui64")` | enqueue up to 64 syndrome bits |
| `fnv1a_hash("get_corrections_ui64")` | return up to 64 correction bits |
| `fnv1a_hash("configure_decoder")` | configure or reconfigure the decoder bank |

`configure_decoder` accepts one chunked payload frame per RPC:

```text
[magic][version][total_bytes][offset_bytes][chunk_bytes][flags][yaml bytes...]
```

`magic` is `CONFIG_CHUNK_MAGIC` / `CUDAQX_RT_CONFIGURE_CHUNK_MAGIC`:
`0x43585143`. On the little-endian OPNIC payload it is carried as bytes
`43 51 58 43`, the ASCII marker `"CQXC"`. The configure handler checks this
marker before it interprets the rest of the payload as a chunk frame; a packet
without the marker is rejected as a configure error.

`CUDAQX_RT_CONFIGURE_CHUNK_BYTES` is currently 1024. The shim assembles the full
config and only calls CUDA-QX after it receives the chunk with
`CUDAQX_RT_CONFIGURE_CHUNK_END`.

The 1024-byte chunk size is not a decoder requirement. We picked it
conservatively after seeing the real QUA/OPNIC external-stream path fail around
the 4 KiB packet size boundary. Larger chunks may work on another QUA/OPNIC
stack, but each control packet must stay below the transport's practical
message limit.

## Build

From this directory:

```bash
./build.sh
```

Defaults:

| variable | default |
|---|---|
| `VENV` | `/tmp/cudaqx-shim-venv` |
| `BUILD_DIR` | `/tmp/cudaqx-decoder-hostcall-build` |
| `CUDAQX_REF` | `0.6.0` |
| `CUDAQX_PIP_PACKAGE` | `cudaq-qec-cu12==${CUDAQX_REF}` |
| `CUDAQ_PIP_PACKAGE` | `cuda-quantum-cu12==0.14.2` |
| `CUDAQX_PIP_NO_DEPS` | `1` |
| `CUDAQX_ENABLE_TRT` | `1` |
| `TRT_ONNX_PATH` | `${BUILD_DIR}/surface_code_decoder.onnx` |
| `TRT_GOLDEN_PATH` | hello_opx decoder `data/trt_surface_code_golden.txt` |
| `HEADER_ROOT` | `/tmp/cudaqx-github-${CUDAQX_REF}-headers` |
| `CUDAQ_REALTIME_INCLUDE_DIR` | this repo's `realtime/include` |

The script:

1. Creates or updates the venv.
2. Installs only the two wheels needed by the shim by default:
   `cuda-quantum-cu12` and `cudaq-qec-cu12`, both with `pip --no-deps`.
   The shim links against libraries shipped in those wheels; the larger
   transitive Python/GPU dependency set pulled by `cudaq-qec-cu12` is not needed
   for this OPNIC host-call build. Set `CUDAQX_PIP_NO_DEPS=0` to restore pip's
   normal dependency resolution.
3. Prepares TensorRT for the `trt_decoder` handoff test when
   `CUDAQX_ENABLE_TRT=1`.
   - On x86_64 it first tries the wheel extra
     `cudaq-qec-cu12[trt-decoder]`.
   - On aarch64/SBSA, where that pip extra does not currently provide matching
     TensorRT library wheels, it installs `libnvinfer10` and
     `libnvonnxparsers10` from the CUDA apt repository when running as root.
4. Installs `git-lfs` when needed and fetches CUDA-QX's real
   `assets/tests/surface_code_decoder.onnx` LFS object into `${TRT_ONNX_PATH}`.
   The script rejects the file if it is still a Git LFS pointer.
5. Sparse-checks out `libs/core/include`, `libs/qec/include`, and the TensorRT
   ONNX asset from
   `https://github.com/NVIDIA/cudaqx.git` into `/tmp`.
6. Configures and builds this CMake project.
7. Writes `${BUILD_DIR}/shim-env.sh` with the shim path, TensorRT model path,
   TensorRT golden-output table path, and wheel `LD_LIBRARY_PATH`.
8. Runs `ldd` on the produced library.

Expected output:

```text
/tmp/cudaqx-decoder-hostcall-build/libcudaqx-decoder-hostcall.so
/tmp/cudaqx-decoder-hostcall-build/shim-env.sh
```

Use the env file before launching the decoder server:

```bash
source /tmp/cudaqx-decoder-hostcall-build/shim-env.sh
sudo -E /path/to/decoder_server_cpu --shim="${CUDAQX_SHIM_LIBRARY}"
```

## Requirements

- Python with `venv` and `pip`
- CMake
- A C++20 compiler; `build.sh` prefers `g++-13` when present
- CUDA toolkit headers discoverable by CMake
- Network access for `pip install`, the sparse GitHub checkout, and the Git LFS
  ONNX download, unless the venv, `HEADER_ROOT`, and `${TRT_ONNX_PATH}` already
  exist
- For the TensorRT decoder test, either pip-provided TensorRT libraries or apt
  packages that provide `libnvinfer.so.10` and `libnvonnxparser.so.10`. Set
  `CUDAQX_ENABLE_TRT=0` to skip preparing the TensorRT runtime/model.

The build does not require building CUDA-QX or CUDA-Q from source.
