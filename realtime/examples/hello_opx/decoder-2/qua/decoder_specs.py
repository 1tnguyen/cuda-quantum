#!/usr/bin/env python3
"""Decoder job catalog for the realtime decoder-2 OPNIC example."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


# Local CUDA-QX source tree (checked out at /workspaces/cudaqx).  The TRT ONNX
# model is available there via Git LFS without a separate download step.
DEFAULT_TRT_ONNX_PATH = Path(
    os.environ.get(
        "CUDAQX_TRT_ONNX_PATH",
        "/workspaces/cudaqx/assets/tests/surface_code_decoder.onnx",
    )
)
DEFAULT_TRT_GOLDEN_PATH = Path(
    os.environ.get(
        "CUDAQX_TRT_GOLDEN_PATH",
        Path(__file__).resolve().parent.parent / "data" / "trt_surface_code_golden.txt",
    )
)


@dataclass(frozen=True)
class DecoderTestSpec:
    name: str
    label: str
    config_file: str
    syndromes_file: str
    decoder_id: int = 0
    num_observables: int = 1
    zero_correction: int = 0
    requires_trt: bool = False
    raw_golden_file: str | None = None

    def config_path(self, data_dir: Path) -> Path:
        return data_dir / self.config_file

    def syndromes_path(self, data_dir: Path) -> Path:
        return data_dir / self.syndromes_file


DECODER_TESTS = {
    "multi-lut": DecoderTestSpec(
        name="multi-lut",
        label="multi_error_lut",
        config_file="config_multi_err_lut.yml",
        syndromes_file="syndromes_multi_err_lut.txt",
    ),
    "nvqldpc": DecoderTestSpec(
        name="nvqldpc",
        label="nv-qldpc-decoder relay",
        config_file="config_nvqldpc_relay.yml",
        syndromes_file="syndromes_nvqldpc_relay.txt",
    ),
    "trt": DecoderTestSpec(
        name="trt",
        label="trt_decoder surface-code ONNX",
        config_file="config_trt_surface_code.yml",
        syndromes_file="syndromes_trt_surface_code.txt",
        zero_correction=1,
        requires_trt=True,
        raw_golden_file="trt_surface_code_golden.txt",
    ),
}

DEFAULT_DECODER_TEST_NAMES = ("multi-lut", "nvqldpc", "trt")

DECODER_TEST_ALIASES = {
    "all": "all",
    "multi_lut": "multi-lut",
    "multi-error-lut": "multi-lut",
    "multi_error_lut": "multi-lut",
    "nv-qldpc": "nvqldpc",
    "nv-qldpc-decoder": "nvqldpc",
    "tensor-rt": "trt",
    "tensorrt": "trt",
    "trt_decoder": "trt",
}


def available_decoder_tests() -> str:
    return ", ".join(DEFAULT_DECODER_TEST_NAMES)


def normalize_decoder_test_name(name: str) -> str:
    key = name.strip()
    normalized = DECODER_TEST_ALIASES.get(key, key)
    if normalized == "all" or normalized in DECODER_TESTS:
        return normalized
    raise ValueError(
        f"unknown decoder test '{name}'. Available tests: {available_decoder_tests()}"
    )


def get_decoder_test(name: str) -> DecoderTestSpec:
    normalized = normalize_decoder_test_name(name)
    if normalized == "all":
        raise ValueError("'all' selects more than one decoder test")
    return DECODER_TESTS[normalized]


def expand_decoder_tests(value: str | None) -> list[DecoderTestSpec]:
    if value is None or not value.strip() or value.strip() == "all":
        names = list(DEFAULT_DECODER_TEST_NAMES)
    else:
        names = []
        for raw in value.split(","):
            if not raw.strip():
                continue
            normalized = normalize_decoder_test_name(raw)
            if normalized == "all":
                names.extend(DEFAULT_DECODER_TEST_NAMES)
            else:
                names.append(normalized)

    specs: list[DecoderTestSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        specs.append(DECODER_TESTS[name])
    if not specs:
        raise ValueError(
            f"no decoder tests selected. Available tests: {available_decoder_tests()}"
        )
    return specs


def load_config_bytes(config_path: Path, trt_onnx_path: Path) -> bytes:
    text = config_path.read_text()
    return text.replace("${TRT_ONNX_PATH}", str(trt_onnx_path)).encode()


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False
