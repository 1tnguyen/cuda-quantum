# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Explicit decoder authoring and record ordering through P0 to P3."""

import json

import pytest
import cudaq.logical as cql


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _ops(build, name):
    return [op for op in _walk(build.module.operation) if op.name == name]


def _device(code, decoder):
    carriers = cql.architecture.ResourceClass(
        "qubit",
        code.block.size,
        native_actions=("h", "cx"),
        native_instruments=(cql.architecture.physical_instruments.MZ,
                            cql.architecture.physical_instruments.MPP))
    physical = cql.architecture.PhysicalMachine(
        "memory_machine", resource_classes={"carriers": carriers})
    builder = cql.devices.DeviceBuilder("memory_device", physical=physical)
    compute = builder.logical.add_compute(capacity=1)
    architecture = cql.devices.QECArchitecture(
        "memory_architecture",
        code.default_encoding,
        link_roots=(cql.gadgets.prepare_zero(code),
                    cql.gadgets.css_memory_round(code, decoder=decoder),
                    cql.gadgets.logical_measure(code,
                                                basis="z",
                                                decoder=decoder)))
    region = builder.qec.bind(compute, architecture=architecture)
    builder.physical.bind(region, to=builder.physical.carriers)
    return builder.build()


@pytest.mark.parametrize("distance,rounds", [(3, 1), (5, 1), (3, 3), (5, 3)])
@pytest.mark.parametrize("unrolled", [False, True])
def test_memory_decoder_dataflow(distance, rounds, unrolled):
    code = cql.codes.Surface[distance]
    nx, nz = len(code.hx), len(code.hz)
    decoder = cql.qec.DecoderSpec(
        "memory_decoder",
        implementation="nv-qldpc",
        inputs=("measurements",),
        options={"config": "pre_generated_decoder_config"})
    device = _device(code, decoder)

    @cql.program
    def memory() -> bool:
        q = cql.prepare_zero()
        if unrolled:
            for _ in range(rounds):
                q = cql.idle(q, rounds=1)
        else:
            q = cql.idle(q, rounds=rounds)
        return cql.measure_z(q)

    p0 = cql.compile(memory)
    p1 = cql.compiler.place(p0, device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    # The public pipeline uses its reference projector for syndrome extraction;
    # native submission dataflow is covered independently by the IR tests.
    p3 = cql.compile(p2,
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)

    for build in (p0, p1, p2, p3):
        assert build.module.operation.verify()
    authored_submit, = _ops(p2, "fabric.submit_syndrome")
    assert authored_submit.operands[
        0].owner.operation.name == "fabric.read_syndrome_ancillas"
    assert "input_order" not in authored_submit.attributes
    submits = _ops(p3, "phys.submit_syndrome")
    decoded, = _ops(p3, "phys.decode_bit")
    measurements = _ops(p3, "phys.measure")
    static_rounds = rounds if unrolled else 1
    assert len(submits) == static_rounds
    assert len(measurements) == static_rounds * (nx + nz) + code.n
    graph, = _ops(p3, "phys.graph")
    repeats = [op for op in _walk(graph) if op.name == "cflow.repeat"]
    if not unrolled and rounds > 1:
        repeat, = repeats
        assert repeat.attributes["count"].value == rounds
        assert submits[0] in list(_walk(repeat))
    else:
        assert not repeats
    records = [op.results[1] for op in measurements]
    for index, submit in enumerate(submits):
        measured = records[index * (nx + nz):(index + 1) * (nx + nz)]
        assert list(submit.operands) == measured
        assert "input_order" not in submit.attributes
    for op in [*submits, decoded]:
        assert str(op.attributes["decoder"]) == "@memory_decoder"
        assert op.attributes["decoder_implementation"].value == "nv-qldpc"
        assert "pre_generated_decoder_config" in op.attributes[
            "decoder_options_json"].value
    assert len({op.attributes["record_id"].value for op in measurements
               }) == len(measurements)
    assert list(decoded.operands) == records[-code.n:]
    assert tuple(decoded.attributes["parity_indices"]) == tuple(code.lz[0])
    returned, = _ops(p3, "phys.return")
    call = decoded.parent
    assert call.name == "phys.call"
    yielded = list(call.regions[0].blocks[0].operations)[-1].operation
    assert yielded.name == "phys.yield"
    assert list(yielded.operands) == [decoded.result]
    assert list(returned.operands) == [call.result]
    assert not _ops(p3, "phys.measure_product")
    assert not _ops(p3,
                    "phys.xor")  # raw parity/correction belongs to decode_bit
    assert cql.compiler.Build.replay(p3.serialize()).module.operation.verify()


@pytest.mark.parametrize("code", [cql.codes.Surface[3], cql.codes.Steane])
def test_round_submission_is_opt_in(code):
    baseline = cql.compile(cql.gadgets.css_memory_round(code))
    assert not _ops(baseline, "fabric.submit_syndrome")
    bound = cql.compile(
        cql.gadgets.css_memory_round(code, decoder="external_decoder"))
    submit, = _ops(bound, "fabric.submit_syndrome")
    assert "input_order" not in submit.attributes
    assert str(submit.attributes["decoder"]) == "@external_decoder"
    assert not _ops(bound, "fabric.decode_bit")


@pytest.mark.parametrize("scalar", [False, True])
def test_explicit_submission_authoring(scalar):
    code = cql.codes.Surface[3]
    decoder = cql.qec.DecoderSpec("decoder",
                                  inputs=("measurements",),
                                  options={"config": {
                                      "path": "decoder.json"
                                  }})

    @cql.gadget(device=_device(code, None))
    def submissions() -> None:
        block = cql.allocate_patch(code)
        block = cql.reset(block.data)
        block, bits = cql.mz(block.data)
        submitted = cql.ops.parity(bits) if scalar else bits
        cql.ops.submit_syndrome(submitted, decoder=decoder)
        cql.ops.submit_syndrome(submitted, decoder=decoder)
        cql.discard(block)

    build = cql.compile(submissions)
    submits = _ops(build, "fabric.submit_syndrome")
    assert [op.attributes["round"].value for op in submits] == [0, 1]
    assert submits[0].operands[0] == submits[1].operands[0]
    assert json.loads(submits[0].attributes["decoder_options_json"].value) == {
        "config": {
            "path": "decoder.json"
        }
    }


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("scalar", [False, True])
def test_terminal_decode_reference_projection(distance, scalar):
    code = cql.codes.Surface[distance]
    device = _device(code, None)
    decoder = cql.qec.DecoderSpec("decoder", inputs=("measurements",))
    indices = (0,) if scalar else tuple(code.lz[0])

    @cql.gadget(device=device)
    def readout() -> bool:
        block = cql.allocate_patch(code)
        block = cql.reset(block.data)
        # Exercise the reference projector used by the CSS memory path.
        block, syndrome = cql.extract_syndrome(block)
        cql.ops.submit_syndrome(syndrome, decoder=decoder)
        block, bits = cql.mz(block.data)
        data = cql.ops.parity(bits) if scalar else bits
        result = cql.ops.decode_bit(data,
                                    decoder=decoder,
                                    parity_indices=None if scalar else indices,
                                    invert=True)
        cql.discard(block)
        return result

    p2 = cql.compile(readout)
    decoded_p2, = _ops(p2, "fabric.decode_bit")
    # A decoded value must never acquire affine raw-record provenance.
    assert "outcome_map" not in _ops(p2, "fabric.gadget_spec")[0].attributes
    p3 = cql.compile(p2,
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)
    assert p3.module.operation.verify()
    decoded, = _ops(p3, "phys.decode_bit")
    assert decoded.attributes["decoder"] == decoded_p2.attributes["decoder"]
    assert decoded.attributes["decoder_inputs"] == decoded_p2.attributes[
        "decoder_inputs"]
    assert tuple(decoded.attributes["parity_indices"]) == indices
    assert decoded.attributes["invert"].value
    final_records = [op.results[1] for op in _ops(p3, "phys.measure")][-code.n:]
    if scalar:
        conditions = _ops(p3, "phys.condition")
        assert [op.operands[0] for op in conditions] == final_records
        xors = _ops(p3, "phys.xor")
        assert len(xors) == code.n - 1
        assert list(xors[0].operands) == [op.result for op in conditions[:2]]
        for index, xor in enumerate(xors[1:], start=1):
            assert list(xor.operands) == [
                xors[index - 1].result, conditions[index + 1].result
            ]
        assert list(decoded.operands) == [xors[-1].result]
    else:
        assert list(decoded.operands) == final_records
    returned, = _ops(p3, "phys.return")
    assert list(returned.operands) == [decoded.result]
    assert cql.compiler.Build.replay(p3.serialize()).module.operation.verify()


@pytest.mark.parametrize("bundles", [(1,), (3,), (1, 2)])
def test_reference_parity_bundle_projection(bundles):
    code = cql.codes.Surface[3]
    device = _device(code, None)

    @cql.gadget(device=device)
    def readout() -> bool:
        block = cql.allocate_patch(code)
        block = cql.reset(block.data)
        block, syndrome = cql.extract_syndrome(block)
        measurements = []
        start = 0
        for size in bundles:
            block, bits = cql.mz(block.data[tuple(range(start, start + size))])
            measurements.append(bits)
            start += size
        result = cql.ops.parity(*measurements)
        cql.discard(block)
        return result

    p3 = cql.compile(cql.compile(readout),
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)
    assert p3.module.operation.verify()
    records = [op.results[1] for op in _ops(p3, "phys.measure")][-sum(bundles):]
    conditions = _ops(p3, "phys.condition")
    assert [op.operands[0] for op in conditions] == records
    xors = _ops(p3, "phys.xor")
    assert len(xors) == sum(bundles) - 1
    returned, = _ops(p3, "phys.return")
    assert returned.operands[0] == (xors[-1].result
                                    if xors else conditions[0].result)


@pytest.mark.parametrize("indices", [(-1,), (9,), (True,), (0.5,)])
def test_decode_rejects_invalid_parity_indices(indices):
    code = cql.codes.Surface[3]

    @cql.gadget(device=_device(code, None))
    def readout() -> bool:
        block = cql.allocate_patch(code)
        block, bits = cql.mz(block.data)
        result = cql.ops.decode_bit(bits,
                                    decoder="decoder",
                                    parity_indices=indices)
        cql.discard(block)
        return result

    with pytest.raises(ValueError, match="parity_indices must index"):
        cql.compile(readout)


@pytest.mark.parametrize(
    "code", [cql.codes.Surface[3], cql.codes.Surface[5], cql.codes.Steane])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_decoded_readout_is_opt_in(code, basis):
    ideal = cql.gadgets.logical_measure(code, basis=basis)
    baseline = cql.compile(ideal)
    assert not _ops(baseline, "fabric.decode_bit")
    assert len(_ops(baseline, "fabric.measure_product")) == 1
    assert ideal.outcome_map is not None
    assert ideal.decoder_results is None
    decoded = cql.gadgets.logical_measure(code, basis=basis, decoder="decoder")
    build = cql.compile(decoded)
    assert decoded.outcome_map is None
    assert decoded.decoder_results == ("decoder",)
    spec, = _ops(build, "fabric.gadget_spec")
    assert "outcome_map" not in spec.attributes
    assert str(spec.attributes["decoder_results"]) == "[@decoder]"
    terminal, = _ops(build, "fabric.mz")
    decode, = _ops(build, "fabric.decode_bit")
    assert decode.operands[0] == terminal.results[1]
    assert tuple(decode.attributes["parity_indices"]) == tuple(
        code.lx[0] if basis == "x" else code.lz[0])
    assert len(_ops(build, "fabric.h")) == int(basis == "x")
    assert cql.compiler.Build.replay(
        build.serialize()).module.operation.verify()
    # Repeated materialization must preserve both contract and SSA semantics.
    assert cql.compile(decoded).to_mlir() == build.to_mlir()


@pytest.mark.parametrize("basis", ["x", "z"])
def test_decoded_readout_rejects_preserved_block(basis):
    with pytest.raises(ValueError, match="destructively measures"):
        cql.gadgets.logical_measure(cql.codes.Surface[3],
                                    basis=basis,
                                    decoder="decoder",
                                    preserve_block=True)


def test_explicit_decoder_result_contract():
    code = cql.codes.Surface[3]
    spec = cql.gadgets.GadgetSpec(implements=cql.logical.measure_z,
                                  ports=(cql.gadgets.Port.input(
                                      "block",
                                      encoding=code.default_encoding,
                                      measured=True),),
                                  decoder_results=("decoder",))

    @cql.gadget(spec=spec)
    def readout(block: cql.patch[code]) -> bool:
        block, bits = cql.mz(block.data)
        result = cql.ops.decode_bit(bits,
                                    decoder="decoder",
                                    parity_indices=code.lz[0])
        cql.discard(block)
        return result

    build = cql.compile(readout)
    assert build.module.operation.verify()
    assert readout.outcome_map is None
    assert readout.decoder_results == ("decoder",)


@pytest.mark.parametrize(
    "mutation", ["wrong_decoder", "missing_result", "wrong_kind", "raw_return"])
def test_native_decoder_result_contract_rejects_invalid_ir(mutation):
    from cudaq.mlir import ir
    build = cql.compile(
        cql.gadgets.measure_z(cql.codes.Surface[3], decoder="decoder"))
    spec, = _ops(build, "fabric.gadget_spec")
    decoded, = _ops(build, "fabric.decode_bit")
    with build.module.context:
        if mutation == "wrong_decoder":
            spec.attributes["decoder_results"] = ir.ArrayAttr.get(
                [ir.FlatSymbolRefAttr.get("other")])
        elif mutation == "missing_result":
            spec.attributes["decoder_results"] = ir.ArrayAttr.get([])
        elif mutation == "wrong_kind":
            spec.attributes["decoder_results"] = ir.ArrayAttr.get(
                [ir.StringAttr.get("decoder")])
        else:
            returned, = _ops(build, "fabric.return")
            with ir.InsertionPoint(returned), ir.Location.unknown():
                raw = ir.Operation.create(
                    "fabric.parity",
                    operands=[decoded.operands[0]],
                    results=[ir.IntegerType.get_signless(1)])
            returned.operands[0] = raw.result
        with pytest.raises(ir.MLIRError, match="decoder_results"):
            build.module.operation.verify()


def test_decoder_result_cannot_claim_an_affine_outcome_map():
    from cudaq.mlir import ir
    ideal = cql.compile(cql.gadgets.measure_z(cql.codes.Surface[3]))
    decoded = cql.compile(
        cql.gadgets.measure_z(cql.codes.Surface[3], decoder="decoder"))
    ideal_spec, = _ops(ideal, "fabric.gadget_spec")
    spec, = _ops(decoded, "fabric.gadget_spec")
    with decoded.module.context:
        spec.attributes["outcome_map"] = ir.Attribute.parse(
            str(ideal_spec.attributes["outcome_map"]))
        with pytest.raises(ir.MLIRError, match="mutually exclusive"):
            decoded.module.operation.verify()


@pytest.mark.parametrize("distance", [3, 5])
def test_decoded_readout_native_projection(distance):
    code = cql.codes.Surface[distance]
    device = _device(code, "decoder")

    @cql.program
    def readout() -> bool:
        q = cql.prepare_zero()
        return cql.measure_z(q)

    placed = cql.compiler.place(cql.compile(readout), device=device)
    p2 = cql.compile(placed,
                     pipeline=cql.compiler.pipelines.qec(),
                     device=device)
    p3 = cql.compile(p2,
                     pipeline=cql.compiler.pipelines.physical(),
                     device=device)
    assert p3.module.operation.verify()
    assert any(item.producer == "fabric-to-phys@0.3" for item in p3.evidence)
    decoded, = _ops(p3, "phys.decode_bit")
    assert len(decoded.operands) == code.n
    assert tuple(decoded.attributes["parity_indices"]) == tuple(code.lz[0])


def test_affine_provenance_still_rejects_decoder_result():
    from cudaq.mlir import ir
    build = cql.compile(
        cql.gadgets.measure_z(cql.codes.Surface[3], decoder="decoder"))
    spec, = _ops(build, "fabric.gadget_spec")
    with build.module.context:
        del spec.attributes["decoder_results"]
        spec.attributes["outcome_map"] = ir.Attribute.parse(
            '{records = ["terminal_data.data0", "terminal_data.data1", "terminal_data.data2"], '
            'rows = dense<true> : tensor<1x3xi1>, constants = array<i64: 0>}')
        with pytest.raises(ir.MLIRError,
                           match="affine typed-record provenance"):
            build.module.operation.verify()
