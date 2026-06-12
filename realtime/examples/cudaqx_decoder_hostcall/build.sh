#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)

PYTHON=${PYTHON:-python3}
CUDAQX_REF=${CUDAQX_REF:-0.6.0}
CUDAQX_PIP_PACKAGE=${CUDAQX_PIP_PACKAGE:-cudaq-qec-cu12==${CUDAQX_REF}}
CUDAQ_PIP_PACKAGE=${CUDAQ_PIP_PACKAGE:-cuda-quantum-cu12==0.14.2}
CUDAQX_PIP_NO_DEPS=${CUDAQX_PIP_NO_DEPS:-1}
CUDAQX_TRT_PIP_PACKAGE=${CUDAQX_TRT_PIP_PACKAGE:-cudaq-qec-cu12[trt-decoder]==${CUDAQX_REF}}
CUDAQX_ENABLE_TRT=${CUDAQX_ENABLE_TRT:-1}
VENV=${VENV:-/tmp/cudaqx-shim-venv}
BUILD_DIR=${BUILD_DIR:-/tmp/cudaqx-decoder-hostcall-build}
HEADER_SRC=${HEADER_SRC:-/tmp/cudaqx-github-${CUDAQX_REF}-headers-src}
HEADER_ROOT=${HEADER_ROOT:-/tmp/cudaqx-github-${CUDAQX_REF}-headers}
CUDAQ_REALTIME_INCLUDE_DIR=${CUDAQ_REALTIME_INCLUDE_DIR:-${REPO_ROOT}/realtime/include}
TRT_ONNX_REPO_PATH=${TRT_ONNX_REPO_PATH:-assets/tests/surface_code_decoder.onnx}
TRT_ONNX_PATH=${TRT_ONNX_PATH:-${BUILD_DIR}/surface_code_decoder.onnx}
TRT_GOLDEN_PATH=${TRT_GOLDEN_PATH:-${REPO_ROOT}/realtime/examples/hello_opx/decoder/data/trt_surface_code_golden.txt}
TRT_APT_PACKAGES=${TRT_APT_PACKAGES:-"libnvinfer10=10.13.0.35-1+cuda12.9 libnvonnxparsers10=10.13.0.35-1+cuda12.9"}

if [ -z "${CMAKE_CXX_COMPILER:-}" ]; then
  if command -v g++-13 >/dev/null 2>&1; then
    CMAKE_CXX_COMPILER=$(command -v g++-13)
  else
    CMAKE_CXX_COMPILER=$(command -v g++)
  fi
fi

log() {
  printf '[shim-build] %s\n' "$*"
}

create_venv() {
  if "${PYTHON}" -m venv "${VENV}"; then
    return
  fi

  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    printf 'ERROR: failed to create venv %s. Install python venv support first.\n' \
      "${VENV}" >&2
    exit 1
  fi

  local py_short
  py_short=$("${PYTHON}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  log "installing python${py_short}-venv with apt"
  apt-get update
  apt-get install -y "python${py_short}-venv" || apt-get install -y python3-venv
  "${PYTHON}" -m venv "${VENV}"
}

need_header_fetch() {
  [ ! -f "${HEADER_ROOT}/include/cuda-qx/core/heterogeneous_map.h" ] ||
    [ ! -f "${HEADER_ROOT}/include/cudaq/qec/realtime/decoding_config.h" ]
}

ensure_cudaqx_source() {
  if [ -d "${HEADER_SRC}/.git" ]; then
    return
  fi
  log "cloning cudaqx ${CUDAQX_REF} source into ${HEADER_SRC}"
  rm -rf "${HEADER_SRC}"
  git clone \
    --filter=blob:none \
    --depth 1 \
    --branch "${CUDAQX_REF}" \
    --no-checkout \
    https://github.com/NVIDIA/cudaqx.git \
    "${HEADER_SRC}"
  git -C "${HEADER_SRC}" sparse-checkout init --cone
}

checkout_cudaqx_paths() {
  ensure_cudaqx_source
  git -C "${HEADER_SRC}" sparse-checkout set \
    libs/core/include \
    libs/qec/include \
    "$(dirname "${TRT_ONNX_REPO_PATH}")"
  git -C "${HEADER_SRC}" checkout "${CUDAQX_REF}"
}

fetch_headers() {
  log "fetching cudaqx ${CUDAQX_REF} headers into ${HEADER_ROOT}"
  rm -rf "${HEADER_ROOT}"
  checkout_cudaqx_paths
  mkdir -p "${HEADER_ROOT}/include"
  cp -a "${HEADER_SRC}/libs/core/include/." "${HEADER_ROOT}/include/"
  cp -a "${HEADER_SRC}/libs/qec/include/." "${HEADER_ROOT}/include/"
}

install_git_lfs() {
  if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
    return
  fi
  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    printf 'ERROR: git-lfs is required to fetch %s. Install git-lfs first.\n' \
      "${TRT_ONNX_REPO_PATH}" >&2
    exit 1
  fi
  log "installing git-lfs with apt"
  apt-get update
  apt-get install -y git-lfs
}

is_lfs_pointer() {
  [ -f "$1" ] &&
    head -c 64 "$1" | grep -q '^version https://git-lfs.github.com/spec/v1'
}

fetch_trt_onnx() {
  if [ "${CUDAQX_ENABLE_TRT}" != "1" ]; then
    return
  fi

  if [ -f "${TRT_ONNX_PATH}" ] && ! is_lfs_pointer "${TRT_ONNX_PATH}"; then
    log "using existing real TRT ONNX model at ${TRT_ONNX_PATH}"
    return
  fi

  install_git_lfs
  checkout_cudaqx_paths
  git -C "${HEADER_SRC}" lfs install --local
  log "fetching CUDA-QX TRT ONNX LFS asset ${TRT_ONNX_REPO_PATH}"
  git -C "${HEADER_SRC}" lfs pull \
    --include="${TRT_ONNX_REPO_PATH}" \
    --exclude=""

  local source_path="${HEADER_SRC}/${TRT_ONNX_REPO_PATH}"
  if [ ! -f "${source_path}" ] || is_lfs_pointer "${source_path}"; then
    printf 'ERROR: failed to materialize real ONNX model at %s\n' \
      "${source_path}" >&2
    printf '       The file is missing or still a Git LFS pointer.\n' >&2
    exit 1
  fi

  mkdir -p "$(dirname "${TRT_ONNX_PATH}")"
  cp -f "${source_path}" "${TRT_ONNX_PATH}"
  if is_lfs_pointer "${TRT_ONNX_PATH}"; then
    printf 'ERROR: copied TRT ONNX is still a Git LFS pointer: %s\n' \
      "${TRT_ONNX_PATH}" >&2
    exit 1
  fi
  log "installed real TRT ONNX model at ${TRT_ONNX_PATH} ($(wc -c <"${TRT_ONNX_PATH}") bytes)"
}

trt_runtime_available() {
  ldconfig -p 2>/dev/null | grep -q 'libnvinfer\.so\.10' &&
    ldconfig -p 2>/dev/null | grep -q 'libnvonnxparser\.so\.10'
}

install_trt_runtime() {
  if [ "${CUDAQX_ENABLE_TRT}" != "1" ]; then
    return
  fi

  if trt_runtime_available; then
    log "TensorRT runtime libraries found"
    return
  fi

  # The TensorRT pip extra currently works on x86_64. On aarch64/SBSA the
  # wheel-stub package does not provide a matching library wheel, so use the
  # CUDA apt repository when the container is root-capable.
  if [ "$(uname -m)" = "x86_64" ]; then
    log "installing TensorRT pip extra ${CUDAQX_TRT_PIP_PACKAGE}"
    if "${VENV}/bin/python" -m pip install "${CUDAQX_TRT_PIP_PACKAGE}"; then
      return
    fi
    log "TensorRT pip extra failed; trying apt runtime packages"
  fi

  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    log "WARNING: TensorRT runtime libraries not found"
    log "WARNING: install libnvinfer10 and libnvonnxparsers10, or run with CUDAQX_ENABLE_TRT=0"
    return
  fi

  log "installing TensorRT apt runtime packages: ${TRT_APT_PACKAGES}"
  apt-get update
  if ! apt-get install -y ${TRT_APT_PACKAGES}; then
    log "pinned TensorRT package install failed; trying unpinned packages"
    apt-get install -y libnvinfer10 libnvonnxparsers10
  fi
}

runtime_library_path() {
  local dirs=()
  local dir
  for dir in \
    "${SITE_PACKAGES}/cudaq_qec/lib" \
    "${SITE_PACKAGES}/cudaq_qec/lib/decoder-plugins" \
    "${SITE_PACKAGES}/lib" \
    "${SITE_PACKAGES}/tensorrt" \
    "${SITE_PACKAGES}/tensorrt_libs" \
    "${SITE_PACKAGES}"/cuda_quantum*.libs \
    "${SITE_PACKAGES}/cuquantum/lib" \
    "${SITE_PACKAGES}/cutensor/lib" \
    "${SITE_PACKAGES}/nvidia/tensorrt/lib" \
    "${SITE_PACKAGES}/nvidia/tensorrt_libs/lib" \
    "${SITE_PACKAGES}"/nvidia/*/lib; do
    [ -d "${dir}" ] && dirs+=("${dir}")
  done
  local IFS=:
  printf '%s' "${dirs[*]}"
}

log "creating/updating venv ${VENV}"
create_venv
"${VENV}/bin/python" -m pip install --upgrade pip
if [ "${CUDAQX_PIP_NO_DEPS}" = "1" ]; then
  # The shim links against the libraries shipped in these two wheels. The
  # transitive Python/GPU dependencies pulled by cudaq-qec are not needed for
  # building or running the host-call shim in the OPNIC container, and they make
  # a fresh setup much slower. Set CUDAQX_PIP_NO_DEPS=0 to restore pip's normal
  # dependency resolution.
  "${VENV}/bin/python" -m pip install --no-deps \
    "${CUDAQ_PIP_PACKAGE}" \
    "${CUDAQX_PIP_PACKAGE}"
else
  "${VENV}/bin/python" -m pip install "${CUDAQX_PIP_PACKAGE}"
fi
install_trt_runtime
fetch_trt_onnx

SITE_PACKAGES=$("${VENV}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ ! -d "${SITE_PACKAGES}/cudaq_qec/lib" ]; then
  printf 'ERROR: cudaq_qec libraries not found under %s\n' "${SITE_PACKAGES}" >&2
  exit 1
fi

if need_header_fetch; then
  fetch_headers
else
  log "using existing cudaqx headers at ${HEADER_ROOT}"
fi

log "configuring ${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCUDAQX_INSTALL_DIR="${SITE_PACKAGES}" \
  -DCUDAQX_SOURCE_INCLUDE_DIR="${HEADER_ROOT}/include" \
  -DCUDAQ_REALTIME_INCLUDE_DIR="${CUDAQ_REALTIME_INCLUDE_DIR}"

log "building shim"
cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_LEVEL:-2}"

TARGET="${BUILD_DIR}/libcudaqx-decoder-hostcall.so"
LD_PATH=$(runtime_library_path)
ENV_FILE="${BUILD_DIR}/shim-env.sh"
cat >"${ENV_FILE}" <<EOF
export CUDAQX_SHIM_VENV="${VENV}"
export CUDAQX_SHIM_LIBRARY="${TARGET}"
export CUDAQX_TRT_ONNX_PATH="${TRT_ONNX_PATH}"
export CUDAQX_TRT_GOLDEN_PATH="${TRT_GOLDEN_PATH}"
export LD_LIBRARY_PATH="${LD_PATH}:\${LD_LIBRARY_PATH:-}"
EOF

log "built ${TARGET}"
log "wrote ${ENV_FILE}"
if command -v ldd >/dev/null 2>&1; then
  log "checking dynamic dependencies"
  LD_LIBRARY_PATH="${LD_PATH}:${LD_LIBRARY_PATH:-}" ldd "${TARGET}"
fi
