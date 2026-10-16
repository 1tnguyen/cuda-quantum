# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""DEM characterization and setup-time adaptation to CUDA-QX configuration."""
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from cudaq.logical.compiler.detector_analysis import analyze_detectors, DetectorAnalysis


def _fingerprint(build):
    return sha256(build.to_mlir().encode()).hexdigest()


@dataclass(frozen=True)
class DecoderModel:
    """Backend DEM and measurement maps tied to one compiled experiment."""

    dem: str
    measurement_to_detectors: tuple[tuple[int, ...], ...]
    measurement_to_observables: tuple[tuple[int, ...], ...]
    analysis: DetectorAnalysis
    source_sha256: str

    @property
    def decoder_ids(self):
        """Stable IDs shared by the execution backend and CUDA-QX config."""
        return MappingProxyType({
            name: i for i, name in enumerate(sorted(self.analysis.decoders))
        })


def generate_dem(build, *, target, noise_model=None, **options):
    """Characterize a P3 experiment through an explicit analysis backend.

    ``target.dem_from_kernel(build, analysis=..., noise_model=..., **options)``
    must lower the physical graph and the composed detector/observable rows,
    returning a CUDA-Q ``DEMResult`` with measurement matrices. The target is
    independent of the execution backend; no simulator or QIR translator is
    selected implicitly. ``noise_model`` is the user's CUDA-Q NoiseModel and
    is forwarded unchanged.
    """
    from cudaq import NoiseModel
    if noise_model is not None and not isinstance(noise_model, NoiseModel):
        raise TypeError("noise_model must be a cudaq.NoiseModel or None")
    method = getattr(target, "dem_from_kernel", None)
    if not callable(method):
        raise TypeError("target must provide dem_from_kernel for a P3 Build")
    if options.get("return_measurement_matrices") is False:
        raise ValueError("decoder configuration requires measurement matrices")
    options["return_measurement_matrices"] = True
    analysis = analyze_detectors(build)
    result = method(build,
                    analysis=analysis,
                    noise_model=noise_model,
                    **options)
    m2d = tuple(tuple(row) for row in result.m2d)
    m2o = tuple(tuple(row) for row in result.m2o)
    if result.num_measurements != len(analysis.records):
        raise ValueError("DEM backend changed the physical measurement count")
    # Ordering is part of the backend contract, not a decoder convention.
    # Refuse to bind a model from a different circuit/annotation order.
    if m2d != analysis.detectors or m2o != analysis.observables:
        raise ValueError(
            "DEM backend measurement maps differ from the selected detector profile"
        )
    if result.num_detectors != len(m2d) or result.num_observables != len(m2o):
        raise ValueError("DEM backend returned inconsistent matrix dimensions")
    return DecoderModel(result.dem, m2d, m2o, analysis, _fingerprint(build))


def generate_decoder_config(build, *, model):
    """Return an existing CUDA-QX multi_decoder_config, without installing it.

    The initial adapter supports one closed decoder lifetime and one terminal
    observable. It refuses multi-decoder models until their detector/observable
    ownership is explicit. It does not infer a global decoder partition.
    D_sparse columns are arranged once, at setup, in physical submission order.
    Execution can pack each submitted bundle directly, without a permutation.
    """
    if not isinstance(model, DecoderModel):
        raise TypeError("model must be returned by generate_dem")
    if _fingerprint(build) != model.source_sha256:
        raise ValueError(
            "decoder model belongs to a different compiled experiment")
    analysis = model.analysis
    if len(analysis.decoders) != 1 or len(analysis.observables) != 1:
        raise NotImplementedError(
            "configuration requires one explicitly closed decoder and one observable"
        )
    name, = analysis.decoders
    spec = analysis.decoders[name]
    if analysis.readouts.get(name) != analysis.observables[0]:
        raise ValueError(
            "terminal decode_bit parity does not match the model observable")
    submitted = analysis.submissions[name]
    if len(set(submitted)) != len(submitted):
        raise ValueError(
            "a physical measurement is submitted more than once to the decoder")
    positions = {
        measurement: index for index, measurement in enumerate(submitted)
    }
    try:
        rows = [
            sorted(positions[index]
                   for index in row)
            for row in model.measurement_to_detectors
        ]
    except KeyError as exc:
        raise ValueError(
            "a detector references a measurement not submitted to its decoder"
        ) from exc

    import cudaq_qec as qec
    dem = qec.dem_from_stim_text(model.dem)
    if dem.num_observables() != len(model.measurement_to_observables):
        raise ValueError(
            "DEM observable count differs from its measurement map")
    if dem.num_detectors() != len(rows):
        raise ValueError("DEM detector count differs from its measurement map")
    config = qec.decoder_config()
    config.id = model.decoder_ids[name]
    config.type = spec.implementation
    config.block_size = dem.num_error_mechanisms()
    config.syndrome_size = dem.num_detectors()
    config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
    config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
    config.D_sparse = qec.d_sparse(rows)
    config.error_rate_vec = list(dem.error_rates)

    # DecoderSpec recursively freezes options. Materialize ordinary dictionaries
    # for CUDA-QX's schema-backed configuration binding.
    def thaw(value):
        from collections.abc import Mapping
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [thaw(item) for item in value]
        return value

    config.decoder_custom_args = thaw(spec.options or {})
    config.validate_custom_args()
    result = qec.multi_decoder_config()
    result.decoders = [config]
    return result
