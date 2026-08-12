# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ctypes
import multiprocessing
import re
import threading

import cudaq
from cudaq.kernel.utils import getMLIRContext
from cudaq.mlir.dialects import llvm as mlir_llvm
from cudaq.mlir.ir import Module
from cudaq.mlir.passmanager import PassManager
from llvmlite import binding as llvm

SERVER_EXECUTION_PIPELINE = (
    "builtin.module("
    "canonicalize,distributed-device-call,cse,"
    "func.func("
    "memtoreg,canonicalize,cc-loop-normalize,"
    "cc-loop-unroll{maximum-iterations=1024 "
    "signal-failure-if-any-loop-cannot-be-completely-unrolled=true "
    "allow-early-exit=true},"
    "canonicalize"
    "),"
    "canonicalize,cse,return-to-output-log,symbol-dce,lower-to-cfg,"
    "func.func(stack-frame-prealloc,combine-quantum-alloc,canonicalize,cse),"
    "symbol-dce,"
    "lower-wireset-to-profile-qir{convert-to=qir-adaptive},"
    "lower-to-cfg,symbol-dce,cc-to-llvm"
    ")")

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
_target = llvm.Target.from_default_triple()
_target_machine = _target.create_target_machine()
_backing_module = llvm.parse_assembly("")
_engine = llvm.create_mcjit_compiler(_backing_module, _target_machine)
_engine_lock = threading.Lock()


def _verify_module(module, stage):
    if not module.operation.verify():
        raise RuntimeError(f"MLIR verification failed for {stage} module.")


def _lower_quake_to_llvm(quake_ir):
    context = getMLIRContext()
    module = Module.parse(quake_ir, context=context)
    _verify_module(module, "submitted")

    pass_manager = PassManager.parse(SERVER_EXECUTION_PIPELINE, context=context)
    try:
        pass_manager.run(module.operation)
    except Exception as error:
        raise RuntimeError(
            "Failed to lower the submitted Quake module for execution: "
            f"{error}\n{module}") from error

    _verify_module(module, "server-lowered")
    return mlir_llvm.translate_module_to_llvmir(module.operation)


def _get_kernel_function(module):
    definitions = [
        function for function in module.functions if not function.is_declaration
    ]
    for function in definitions:
        if any("entry_point" in str(attribute)
               for attribute in function.attributes):
            return function
    return definitions[0] if definitions else None


def _get_required_count(function, *attribute_names):
    for attribute in function.attributes:
        attribute_text = str(attribute)
        for name in attribute_names:
            match = re.search(rf'{re.escape(name)}"?="?(\d+)', attribute_text)
            if match is not None:
                return int(match.group(1))
    return 0


def _create_output_log(num_qubits, num_results, shots, kernel):
    records = [
        "HEADER\tschema_id\tlabeled\n",
        "HEADER\tschema_version\t1.0\n",
        "START\n",
        "METADATA\tentry_point\n",
        "METADATA\tqir_profiles\tadaptive_profile\n",
        f"METADATA\trequired_num_qubits\t{num_qubits}\n",
        f"METADATA\trequired_num_results\t{num_results}\n",
    ]
    for shot in range(shots):
        if shot:
            records.append("START\n")
        records.append(cudaq.testing.runKernel(num_qubits, kernel))
        records.append("END\t0\n")
    return "".join(records)


def execute_quake(quake_ir, shots, output_format):
    """Execute value-semantics Quake and return QM-compatible results."""
    if shots < 1:
        raise ValueError("The shot count must be positive.")
    if output_format not in ("histogram", "qir-raw"):
        raise ValueError(f"Unsupported output format: {output_format}")

    with _engine_lock:
        llvm_ir = _lower_quake_to_llvm(quake_ir)
        module = llvm.module.parse_assembly(llvm_ir)
        module.verify()

        function = _get_kernel_function(module)
        if function is None:
            raise RuntimeError("Could not find the submitted kernel function.")

        num_qubits = _get_required_count(function, "required_num_qubits",
                                         "requiredQubits")
        num_results = _get_required_count(function, "required_num_results",
                                          "requiredResults")

        _engine.add_module(module)
        try:
            _engine.finalize_object()
            _engine.run_static_constructors()
            function_pointer = _engine.get_function_address(function.name)
            if function_pointer == 0:
                raise RuntimeError(
                    f"Could not JIT compile kernel function {function.name}.")
            kernel = ctypes.CFUNCTYPE(None)(function_pointer)

            if output_format == "qir-raw":
                return _create_output_log(num_qubits, num_results, shots,
                                          kernel)

            counts = cudaq.testing.sampleKernel(num_qubits, shots, kernel)
            return {bits: int(count) for bits, count in counts.items()}
        finally:
            _engine.remove_module(module)


def _execute_quake_worker(connection, quake_ir, shots, output_format):
    try:
        connection.send((True, execute_quake(quake_ir, shots, output_format)))
    except BaseException as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def execute_quake_isolated(quake_ir, shots, output_format, timeout=120):
    """Execute one job in a disposable process to isolate native JIT state."""
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(target=_execute_quake_worker,
                              args=(send_connection, quake_ir, shots,
                                    output_format))
    process.start()
    send_connection.close()

    try:
        if not receive_connection.poll(timeout):
            process.terminate()
            process.join()
            raise RuntimeError(
                f"Quake simulation timed out after {timeout} seconds.")
        try:
            succeeded, result = receive_connection.recv()
        except EOFError as error:
            process.join()
            raise RuntimeError(
                "Quake simulation worker exited without a result "
                f"(exit code {process.exitcode}).") from error
    finally:
        receive_connection.close()

    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Quake simulation worker did not shut down.")
    if process.exitcode != 0:
        raise RuntimeError(
            f"Quake simulation worker exited with code {process.exitcode}.")
    if not succeeded:
        raise RuntimeError(result)
    return result
