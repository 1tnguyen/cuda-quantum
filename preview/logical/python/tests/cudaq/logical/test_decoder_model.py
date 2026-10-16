# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Selected analysis profiles, counted composition, and decoder model binding."""
from dataclasses import replace

import cudaq
import cudaq.logical as cql
import pytest
from cudaq.logical.compiler.detector_analysis import analyze_detectors


def memory_build(distance=3,
                 rounds=3,
                 *,
                 unrolled=False,
                 profiles=None,
                 observable_support=None,
                 duplicate=False,
                 decoder=None,
                 kernel=None,
                 code=None,
                 preparation="product",
                 program=None,
                 capacity=1,
                 decoded=True,
                 round_override=None):
    code = code or cql.codes.Surface[distance]
    decoder = decoder or cql.qec.DecoderSpec("memory_decoder",
                                             implementation="nv-qldpc-decoder",
                                             inputs=("measurements",),
                                             options={"max_iterations": 50})
    g = cql.gadgets
    preparation = g.prepare_zero(
        code, preparation=preparation if profiles is not False else None)
    round_gadget = round_override or g.css_memory_round(
        code, decoder=decoder if decoded else None)
    readout = g.measure_z(code, decoder=decoder if decoded else None)
    roots = [preparation, round_gadget, readout]
    nx = len(code.hx)
    if profiles or observable_support is not None or duplicate:
        prepared = g.GadgetProfile(
            preparation,
            output_syndromes=tuple(
                g.OutputSyndromeAssignment(preparation.outputs.only(), nx +
                                           i, False)
                for i in range(len(code.hz))))
        checks = round_gadget.records.syndrome(0).checks
        round_profile = g.GadgetProfile(
            round_gadget,
            detectors=(g.Detector(round_gadget.inputs.only().syndrome ^
                                  checks),),
            boundary={round_gadget.outputs.only().syndrome: checks})
        data = g.StructuredRecord(readout, "terminal_data",
                                  data_count=code.n).data
        measured = g.GadgetProfile(
            readout,
            detectors=tuple(
                g.Detector(readout.inputs.only().syndrome[nx + i] ^
                           data.parity(support))
                for i, support in enumerate(code.hz)),
            observables=(g.LogicalObservable(
                data.parity(code.lz[0] if observable_support is
                            None else observable_support)),))
        roots.extend((prepared, round_profile, measured))
        if duplicate:
            roots.append(
                replace(round_profile,
                        output_syndromes=(),
                        name="alternative_round_profile"))
    carriers = cql.architecture.ResourceClass(
        "qubit",
        code.block.size * capacity,
        native_actions=("h", "cx"),
        native_instruments=(cql.architecture.physical_instruments.MZ,) +
        (() if decoded else (cql.architecture.physical_instruments.MPP,)))
    builder = cql.devices.DeviceBuilder(
        "memory_device",
        physical=cql.architecture.PhysicalMachine(
            "memory_machine", resource_classes={"carriers": carriers}))
    compute = builder.logical.add_compute(capacity=capacity)
    region = builder.qec.bind(compute,
                              architecture=cql.devices.QECArchitecture(
                                  "memory_architecture",
                                  code.default_encoding,
                                  link_roots=tuple(roots)))
    builder.physical.bind(region, to=builder.physical.carriers)
    device = builder.build()

    @cql.program
    def memory() -> bool:
        q = cql.prepare_zero()
        if unrolled:
            for _ in range(rounds):
                q = cql.idle(q, rounds=1)
        else:
            q = cql.idle(q, rounds=rounds)
        return cql.measure_z(q)

    p0 = cql.compile(program or
                     memory) if kernel is None else cql.compiler.import_cudaq(
                         kernel, rounds)
    p1 = cql.compiler.place(p0, device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    p3 = cql.compile(p2,
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)
    return p3


@pytest.mark.parametrize("distance,rounds", [(3, 1), (3, 3), (5, 5)])
def test_counted_profiles_match_unrolled(distance, rounds):
    counted = memory_build(distance, rounds)
    expanded = memory_build(distance, rounds, unrolled=True)
    a, b = analyze_detectors(counted), analyze_detectors(expanded)
    code = cql.codes.Surface[distance]
    checks = len(code.hx) + len(code.hz)
    assert len(a.records) == rounds * checks + code.n
    assert len(a.detectors) == (rounds - 1) * checks + 2 * len(code.hz)
    assert a.detectors == b.detectors
    assert a.observables == b.observables
    assert a.submissions == b.submissions
    assert a.submissions["memory_decoder"] == tuple(range(len(a.records)))
    # Initial X syndromes are unknown, not incorrectly set to zero.
    assert a.detectors[:len(code.hz)] == tuple(
        (i,) for i in range(len(code.hx), checks))
    assert a.readouts["memory_decoder"] == a.observables[0]
    replay = cql.compiler.Build.replay(counted.serialize())
    assert analyze_detectors(replay) == a
    assert counted.module.operation.verify()


def test_missing_and_ambiguous_profiles_fail_closed():
    with pytest.raises(ValueError, match="explicit analysis profile"):
        analyze_detectors(memory_build(profiles=False))
    with pytest.raises(ValueError, match="multiple linked analysis profiles"):
        memory_build(duplicate=True)


class RecordingBackend:

    def __init__(self):
        self.noise = None
        self.options = None

    def dem_from_kernel(self, build, *, analysis, noise_model, **options):
        self.noise, self.options = noise_model, options
        # This unit fixture checks the provider contract. Real QIR/Stim and
        # CUDA-QX integration are validated separately with an actual backend.
        return cudaq.DEMResult("error(0.01) D0 L0", analysis.detectors,
                               analysis.observables, len(analysis.detectors),
                               len(analysis.observables), len(analysis.records))


def test_noise_and_analysis_options_are_explicit():
    build = memory_build()
    backend = RecordingBackend()
    noise = cudaq.NoiseModel()
    model = cql.qec.generate_dem(build,
                                 target=backend,
                                 noise_model=noise,
                                 decompose_errors=False)
    assert backend.noise is noise
    assert backend.options == {
        "decompose_errors": False,
        "return_measurement_matrices": True
    }
    assert model.decoder_ids == {"memory_decoder": 0}
    with pytest.raises(ValueError, match="requires measurement matrices"):
        cql.qec.generate_dem(build,
                             target=backend,
                             return_measurement_matrices=False)
    with pytest.raises(TypeError, match="NoiseModel"):
        cql.qec.generate_dem(build,
                             target=backend,
                             noise_model="hardcoded_noise")
    with pytest.raises(ValueError, match="different compiled experiment"):
        cql.qec.generate_decoder_config(memory_build(rounds=4), model=model)


def test_terminal_observable_must_match_decode_bit():
    build = memory_build(observable_support=(0,))
    model = cql.qec.generate_dem(build, target=RecordingBackend())
    with pytest.raises(ValueError, match="does not match the model observable"):
        cql.qec.generate_decoder_config(build, model=model)


def test_backend_cannot_change_measurement_contract():

    class WrongBackend(RecordingBackend):

        def dem_from_kernel(self, build, *, analysis, **kwargs):
            return cudaq.DEMResult("", analysis.detectors[::-1],
                                   analysis.observables,
                                   len(analysis.detectors), 1,
                                   len(analysis.records))

    with pytest.raises(ValueError, match="measurement maps differ"):
        cql.qec.generate_dem(memory_build(), target=WrongBackend())


def test_cudaq_kernel_returning_decoded_memory_bit():

    @cudaq.kernel
    def memory(num_rounds: int) -> bool:
        q = cudaq.qubit()
        for _ in range(num_rounds):
            idle(q)
        return mz(q)

    actual = analyze_detectors(memory_build(kernel=memory))
    expected = analyze_detectors(memory_build())
    assert actual.detectors == expected.detectors
    assert actual.observables == expected.observables
    assert actual.submissions == expected.submissions


def test_composition_is_not_specific_to_surface_codes():
    code = cql.codes.Steane
    analysis = analyze_detectors(memory_build(code=code, rounds=2))
    checks = len(code.hx) + len(code.hz)
    assert len(analysis.records) == 2 * checks + code.n
    assert len(analysis.detectors) == checks + 2 * len(code.hz)
    assert analysis.readouts["memory_decoder"] == analysis.observables[0]


def test_builtin_profiles_match_explicit_contracts():
    default = analyze_detectors(memory_build())
    explicit = analyze_detectors(memory_build(profiles=True))
    assert default == explicit


def test_explicit_preparation_changes_initial_boundary():
    product = analyze_detectors(memory_build(preparation="product"))
    encoded = analyze_detectors(memory_build(preparation="encoded"))
    code = cql.codes.Surface[3]
    checks = len(code.hx) + len(code.hz)
    assert len(encoded.detectors) == len(product.detectors) + len(code.hx)
    assert encoded.detectors[:checks] == tuple((i,) for i in range(checks))
    assert encoded.observables == product.observables


def test_preparation_choice_is_explicit_and_validated():
    g = cql.gadgets
    code = cql.codes.Surface[3]
    assert g.prepare_zero(code).analysis_profile is None
    with pytest.raises(ValueError, match="preparation must be"):
        g.prepare_zero(code, preparation="automatic")
    zero = g.prepare_zero(code, preparation="product")
    plus = g.prepare_plus(code, preparation="product")
    assert [row.index for row in zero.analysis_profile.output_syndromes
           ] == list(range(len(code.hx),
                           len(code.hx) + len(code.hz)))
    assert [row.index for row in plus.analysis_profile.output_syndromes
           ] == list(range(len(code.hx)))
    assert zero.analysis_profile is zero.analysis_profile
    for factory in (g.prepare_zero, g.prepare_plus):
        encoded = factory(code, preparation="encoded")
        assert encoded.analysis_profile.boundary_complete
        text = str(cql.compile(encoded).module)
        assert "fabric.prep_z" not in text and "fabric.prep_x" not in text


def test_default_profile_factory_must_describe_its_gadget():
    g = cql.gadgets
    other = g.prepare_zero(cql.codes.Surface[3], preparation="product")

    @cql.gadget(implements=cql.logical.idle,
                analysis_profile=lambda definition: other.analysis_profile)
    def bad(
        block: cql.patch[cql.codes.Surface[3]]
    ) -> cql.patch[cql.codes.Surface[3]]:
        return block

    with pytest.raises(ValueError, match="for its gadget"):
        bad.analysis_profile


@pytest.mark.parametrize("reverse", [False, True])
def test_local_observables_follow_program_result_order(reverse):

    @cql.program
    def two_results() -> tuple[bool, bool]:
        left = cql.prepare_zero()
        right = cql.prepare_zero()
        left = cql.idle(left, rounds=2)
        right = cql.idle(right, rounds=3)
        a = cql.measure_z(left)
        b = cql.measure_z(right)
        return (b, a) if reverse else (a, b)

    analysis = analyze_detectors(
        memory_build(program=two_results, capacity=2, decoded=False))
    # Each readout gadget declares local observable 0. They must remain two
    # separate experiment observables, ordered by returned SSA values.
    a, b = len(analysis.records) - 2, len(analysis.records) - 1
    assert analysis.observables == ((b,), (
        a,)) if reverse else analysis.observables == ((a,), (b,))


def test_unreturned_readout_does_not_become_an_observable():

    @cql.program
    def discard_result() -> bool:
        left = cql.prepare_zero()
        right = cql.prepare_zero()
        left = cql.idle(left, rounds=2)
        right = cql.idle(right, rounds=2)
        cql.measure_z(left)
        return cql.measure_z(right)

    analysis = analyze_detectors(
        memory_build(program=discard_result, capacity=2, decoded=False))
    assert analysis.observables == ((len(analysis.records) - 1,),)


def test_named_round_records_belong_to_default_profile():
    g = cql.gadgets
    round_gadget = g.css_memory_round(cql.codes.Surface[3], record="checks")
    profile = round_gadget.analysis_profile
    assert all(
        record.name.startswith("checks.")
        for detector in profile.detectors
        for record in detector.parity.records)
    assert cql.compile(profile).module.operation.verify()


@pytest.mark.parametrize("unrolled", [False, True])
def test_native_profile_boundaries_and_shared_templates(unrolled):
    code = cql.codes.Repetition
    g = cql.gadgets

    def profile(definition):
        records = tuple(
            definition.record(f"checks.sz{i}") for i in range(len(code.hz)))
        return g.GadgetProfile(
            definition,
            # Only the first check is harvested here. The other check records
            # still need projection/aliases because the boundary exports them.
            detectors=(g.Detector(definition.inputs.only().syndrome[0] ^
                                  records[0]),),
            output_syndromes=tuple(
                g.OutputSyndromeAssignment(definition.outputs.only(), i, record)
                for i, record in enumerate(records)))

    @cql.gadget(implements=cql.logical.idle, analysis_profile=profile)
    def repetition_round(block: cql.patch[code]) -> cql.patch[code]:
        block = cql.reset(block.sz)
        for check, support in enumerate(code.hz):
            for data in support:
                block = cql.cx(block.data[data], block.sz[check])
        block, _ = cql.mz(block.sz, record="checks")
        return block

    build = memory_build(code=code,
                         rounds=3,
                         unrolled=unrolled,
                         decoded=False,
                         round_override=repetition_round)
    assert any(item.producer == "fabric-to-phys@0.3" for item in build.evidence)
    analysis = analyze_detectors(build)
    width = len(code.hz)
    assert analysis.detectors == ((0,), (0, width), (width, 2 * width))
    assert analysis.observables == ((3 * width,),)
    text = build.to_mlir()
    assert "repetition_round.checks.sz1" in text
    assert "profile_result_indices" in text
    assert ("phys.call_template" in text) == unrolled

    # Removing the only consumer of an exported record must invalidate its
    # projection; boundary retention must not permit unauthenticated records.
    import cudaq.mlir.ir as ir
    module = build._fresh_module()
    with module.context:
        for definition in module.body.operations:
            if definition.operation.name != "fabric.gadget_profile":
                continue
            for view in definition.operation.regions[0].blocks[0].operations:
                op = view.operation
                if (op.name == "fabric.output_syndrome" and
                        int(op.attributes["index"]) == 1 and
                        "records" in op.attributes):
                    del op.attributes["records"]
                    op.attributes["constant"] = ir.BoolAttr.get(False)
        with pytest.raises(ir.MLIRError, match="not authenticated"):
            module.operation.verify()
