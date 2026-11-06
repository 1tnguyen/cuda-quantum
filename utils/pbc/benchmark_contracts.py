#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Benchmark hold synthesis and compare release with a pre-change compiler.

Inputs are the original circuit files in the existing fifteen-case synthesis
snapshot, never a release-normalized circuit or a preserve-network output.
Requires Qiskit 2.6.0 and MQT QCEC 3.11.0. Runs one compiler/checker at a time.
"""

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--before-compiler", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["all"])
    parser.add_argument("--release-cases", nargs="+", default=["adder4", "qft_n002", "new_0"])
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--proof-timeout", type=int, default=30)
    parser.add_argument("--hold-profile", choices=["checkpoints", "legacy"],
                        default="checkpoints",
                        help="fixed hold checkpoint profile or earlier global comparison profile")
    parser.add_argument("--verify-only", action="store_true",
                        help="retry unresolved proofs on saved outputs without resynthesis")
    args = parser.parse_args()
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Operator
    from mqt import qcec
    import numpy as np

    snapshot = args.snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    spec = importlib.util.spec_from_file_location("pbc_counts", snapshot / "snapshot/tools/check.py")
    counts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counts)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.verify_only)
    compilers = {"before": args.before_compiler.resolve(), "after": args.compiler.resolve()}
    options = {
        "objective": "total-gates", "max-gates": "262144",
        "max-dependencies": "64000000", "max-work": "1000000000000",
        "timeout-ms": str(args.timeout_ms), "synthesis-native-entanglers": "cx,cz,swap",
        "synthesis-simplify": "false", "synthesis-prune-unobserved": "false",
        "synthesis-discrepancy-steps": "22", "synthesis-commuting-candidates": "8",
        "synthesis-algebraic-candidates": "4", "synthesis-joint-phase-candidates": "4",
        "synthesis-joint-phase-planner-work": "3000000000",
        "synthesis-joint-phase-max-work": "100000000000",
        "synthesis-joint-phase-timeout-ms": str(args.timeout_ms),
        "synthesis-joint-phase-wide-search": "true",
        "synthesis-joint-phase-acceptance": "gates-and-t",
    }
    hold_options = dict(options)
    if args.hold_profile == "checkpoints":
        hold_options.update({
            "synthesis-hold-checkpoints": "true",
            "synthesis-simplify": "true", "synthesis-candidate-nodes": "1",
            "synthesis-lookahead": "16", "synthesis-commuting-candidates": "0",
            "synthesis-joint-phase-candidates": "0",
            "synthesis-joint-phase-wide-search": "false",
        })
    else:
        hold_options["synthesis-hold-checkpoints"] = "false"
    records = {
        "compiler_sha256": {k: sha(v) for k, v in compilers.items()},
        "runner_sha256": sha(__file__),
        "versions": {name: importlib.metadata.version(name) for name in ["qiskit", "mqt.qcec"]},
        "options": options, "hold_options": hold_options,
        "hold_profile": args.hold_profile,
        "source_manifest_sha256": sha(snapshot / "manifest.json"),
        "address_space_bytes": 8 << 30, "proof_timeout_seconds": args.proof_timeout,
        "cases": [],
    }
    if args.verify_only:
        expected = records
        records = json.loads((output / "results.json").read_text())
        assert records["compiler_sha256"] == expected["compiler_sha256"]
        assert records["source_manifest_sha256"] == expected["source_manifest_sha256"]
        records["proof_retry_runner_sha256"] = sha(__file__)

    def save():
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")

    def invoke(row, stage, argv, seconds):
        command = ["prlimit", "--as=" + str(8 << 30), "--core=0", "--", *map(str, argv)]
        start = time.monotonic()
        record = {"stage": stage, "argv": command}
        row["commands"].append(record)
        with (row["folder"] / (stage + ".log")).open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds)
                record["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                record["timeout"] = True
                record["returncode"] = 124
        record["seconds"] = time.monotonic() - start
        if record["returncode"]:
            raise RuntimeError(stage + " failed: " + str(record["returncode"]))

    def circuit(path):
        parsed = counts.read_ir(path)
        qc = qasm2.loads(counts.qasm(parsed), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        # Include actual emitted scalar phases in the independent dense checks.
        constants = {}
        phases = []
        for line in path.read_text().splitlines():
            match = re.search(r"(%[\w.]+) = arith.constant ([^ :]+).*: f64", line)
            if match:
                constants[match[1]] = float(match[2])
            match = re.search(r"quake.phase(<adj>)? \((%[\w.]+)\)", line)
            if match:
                phases.append(constants[match[2]] * (-1 if match[1] else 1))
        qc.global_phase = math.fsum(phases)
        return parsed, qc

    def verify(row, source, destination, hold):
        a, ca = circuit(source)
        b, cb = circuit(destination)
        assert a["n"] == b["n"]
        if not a["measurements"] and not hold:
            a["measurements"] = [[i, None] for i in range(a["n"])]
        assert a["measurements"] == b["measurements"], "readout identities changed"
        row["metrics"] = b["metrics"]
        if hold:
            row["proof_contract"] = "unitary (QCEC); exact scalar additionally checked by compiler and small dense oracle"
            if a["n"] <= 8:
                error = float(np.max(np.abs(Operator(ca).data - Operator(cb).data)))
                row["dense_full_phase_max_error"] = error
                assert error < 1e-10, "full-phase dense unitary mismatch"
        else:
            row["proof_contract"] = "ordered joint measurement probabilities for arbitrary inputs"

            def oracle(parsed, qc):
                result = QuantumCircuit(parsed["n"] + len(parsed["measurements"]))
                result.compose(qc, range(parsed["n"]), inplace=True)
                for index, (wire, _) in enumerate(parsed["measurements"]):
                    result.cx(wire, parsed["n"] + index)
                result.compose(qc.inverse(), range(parsed["n"]), inplace=True)
                return result

            ca, cb = oracle(a, ca), oracle(b, cb)
        result = qcec.verify(
            ca, cb, run_simulation_checker=False, run_construction_checker=False,
            run_alternating_checker=a["n"] <= 15, run_zx_checker=a["n"] > 15,
            nthreads=1, timeout=args.proof_timeout,
        ).json()
        row["verification"] = result
        return result["equivalence"] in {"equivalent", "equivalent_up_to_global_phase"}, b

    requested = set(args.cases)
    selected = [c for c in manifest["cases"] if "all" in requested or c["name"] in requested]
    assert selected and ("all" in requested or requested == {c["name"] for c in selected})
    for case in selected:
        name = case["name"]
        source = snapshot / case["inputs"]["original"]["path"]
        assert sha(source) == case["inputs"]["original"]["sha256"]
        if args.verify_only:
            entry = next((c for c in records["cases"] if c["name"] == name), None)
            if entry is None:
                continue
            for label, row in entry["runs"].items():
                destination = Path(row["folder"]) / "output.qke"
                if row["status"] == "verified" or not destination.exists():
                    continue
                if "output_sha256" in row:
                    assert row["output_sha256"] == sha(destination)
                row.setdefault("proof_attempts", []).append({
                    key: row[key] for key in ["status", "error", "verification",
                                              "dense_full_phase_max_error"] if key in row})
                row.pop("error", None)
                try:
                    checked, _ = verify(row, source, destination, label.startswith("hold-"))
                    row["status"] = "verified" if checked else "unverified"
                except Exception as error:
                    row["status"] = "failed"
                    row["error"] = str(error)
                row["output_sha256"] = sha(destination)
                save()
                print(name, label, "proof retry", row["status"], flush=True)
            continue
        entry = {"name": name, "case_id": case["case_id"], "width": case["width"], "input_sha256": sha(source), "runs": {}}
        records["cases"].append(entry)
        modes = [("hold", "after")]
        if name in args.release_cases:
            modes += [("release", "before"), ("release", "after")]
        release_words = {}
        for semantics, version in modes:
            label = semantics + "-" + version
            folder = output / name / label
            folder.mkdir(parents=True)
            shutil.copy2(source, folder / "input.qke")
            row = {"status": "running", "commands": [], "folder": folder}
            entry["runs"][label] = row
            # Store a JSON-compatible path while retaining Path locally.
            try:
                binary = compilers[version]
                # As in the existing report, remove only compiler-generated
                # implicit-output annotations before scalar-wire conversion.
                # They are not gates or explicit measurement operations.
                prepared = folder / "prepared.qke"
                lines = source.read_text().splitlines(keepends=True)
                retained = [line for line in lines if not re.match(
                    r"\s*quake\.evince\b.*\{compilerGenerated\}\s*$", line)]
                prepared.write_text("".join(retained))
                row["removed_compiler_generated_evince"] = len(lines) - len(retained)
                row["prepared_input_sha256"] = sha(prepared)
                normalization = ""
                if version == "after":
                    normalization = "semantics=" + semantics
                    if semantics == "hold":
                        normalization += " local-frames=" + str(
                            args.hold_profile == "checkpoints").lower()
                if semantics == "release":
                    normalization += " assume-measure-all=true"
                pipeline = "builtin.module(convert-to-linear-values,func.func(quake-to-pbc{" + normalization.strip() + "}))"
                invoke(row, "normalize", [binary, prepared, "--pass-pipeline=" + pipeline,
                       "--mlir-disable-threading", "-o", folder / "normalized.qke"], 60)
                opts = dict(hold_options if semantics == "hold" else options)
                if version == "after":
                    opts["semantics"] = semantics
                if semantics == "release":
                    opts.update({"synthesis-readout-candidates": "8", "synthesis-representation-steps": "32",
                                 "synthesis-indexed-representation-steps": "64"})
                invoke(row, "synthesize", [binary, folder / "normalized.qke", "--quake-synthesize-pbc=" +
                       " ".join(k + "=" + v for k, v in opts.items()), "--mlir-disable-threading",
                       "-o", folder / "synthesized.qke"], args.timeout_ms / 1000 + 30)
                row["budget_exhausted"] = "pbc.synthesis_budget_exhausted = true" in (folder / "synthesized.qke").read_text()
                invoke(row, "regtomem", [binary, folder / "synthesized.qke", "--regtomem",
                       "--mlir-disable-threading", "-o", folder / "output.qke"], 30)
                checked, parsed = verify(row, source, folder / "output.qke", semantics == "hold")
                row["status"] = "verified" if checked else "unverified"
                row["output_sha256"] = sha(folder / "output.qke")
                if semantics == "release":
                    release_words[version] = parsed
            except Exception as error:
                row["status"] = "failed"
                row["error"] = str(error)
            row["folder"] = str(folder)
            save()
            print(name, label, row["status"], row.get("metrics", row.get("error")), flush=True)
        if len(release_words) == 2:
            entry["release_identical_gate_word_and_readouts"] = (
                release_words["before"] == release_words["after"])
        save()
    assert records["compiler_sha256"] == {k: sha(v) for k, v in compilers.items()}


if __name__ == "__main__":
    main()
