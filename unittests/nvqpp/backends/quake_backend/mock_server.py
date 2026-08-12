# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from fastapi import FastAPI, HTTPException, Request
import uvicorn, uuid, base64, sys, re
from llvmlite import binding as llvm
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir.ir import Module
from cudaq.kernel.utils import getMLIRContext
from cudaq.mlir.dialects import func
from utils.mock_qpu.quake_execution import execute_quake

# Define the REST Server App
app = FastAPI()

# Keep track of Job Ids to their Names
createdJobs = {}


def verifyValueSemanticsPayload(decoded_payload):
    required_tokens = ["quake.wire_set", "quake.borrow_wire"]
    for token in required_tokens:
        if token not in decoded_payload:
            raise RuntimeError(
                f"Remote payload is missing `{token}`. The server must receive"
                " value-semantics MLIR with an assigned wireset.")

    forbidden_tokens = [
        "quake.alloca",
        "quake.extract_ref",
        "quake.subveq",
        "quake.concat",
        "quake.relax_size",
        "quake.unwrap",
        "quake.wrap",
        "!quake.ref",
        "!quake.veq",
    ]
    for token in forbidden_tokens:
        if token in decoded_payload:
            raise RuntimeError(
                f"Remote payload still contains reference-semantics token"
                f" `{token}`. The server must receive wireset MLIR.")


def verifyExpectedLoopCount(decoded_payload, entry_func_name):
    match = re.search(r"expected_(\d+)_loops?", entry_func_name)
    if not match:
        return

    expectedCount = int(match.group(1))
    actualCount = len(re.findall(r"\bcc\.loop\b", decoded_payload))
    if actualCount != expectedCount:
        raise RuntimeError(
            "Remote payload preserved an unexpected number of `cc.loop` ops "
            f"for `{entry_func_name}`: expected {expectedCount}, got "
            f"{actualCount}.")


def verifyModule(module, stage):
    if not module.operation.verify():
        raise RuntimeError(f"MLIR verification failed for {stage} module.")


@app.post("/job")
async def postJob(request: Request):
    global createdJobs
    payload = await request.json()
    # Decode base64
    decoded_payload = base64.b64decode(payload["ir"]).decode('utf-8')
    # Verify that the input MLIR does not contain actual `malloc` or `memcpy`
    # calls. Match `@malloc` or `@llvm.memcpy` as function references (calls or
    # declarations).
    if re.search(r'@malloc\b', decoded_payload) or \
       re.search(r'@(llvm\.)?memcpy\b', decoded_payload):
        raise RuntimeError(
            "Input MLIR contains malloc or memcpy calls. These should have been"
            " eliminated by the eliminate-dead-heap-copy pass.")

    verifyValueSemanticsPayload(decoded_payload)

    ctx = getMLIRContext()
    recovered_mod = Module.parse(decoded_payload, context=ctx)
    verifyModule(recovered_mod, "submitted")
    pm = PassManager.parse(
        "builtin.module(canonicalize,distributed-device-call,cse)", context=ctx)
    try:
        pm.run(recovered_mod.operation)
    except Exception as e:
        raise RuntimeError(
            f"Failed to run pass manager on the recovered module: {e}")

    entry_func_name = ""
    for op in recovered_mod.body.operations:
        if isinstance(op, func.FuncOp):
            for attr in op.attributes:
                if attr == "cudaq-entrypoint":
                    entry_func_name = op.name.value
                    break
    if not entry_func_name:
        raise RuntimeError(
            "Remote payload is missing a `cudaq-entrypoint` function.")
    verifyExpectedLoopCount(decoded_payload, entry_func_name)

    # Job ID
    newId = str(uuid.uuid4())
    createdJobs[newId] = execute_quake(decoded_payload, payload["shots"],
                                       "qir-raw")
    # Job "created", return the id
    return ({"id": newId}, 201)


@app.get("/job/{jobId}")
async def getJob(jobId: str):
    if jobId not in createdJobs:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job_output = createdJobs[jobId]
    if job_output.startswith("FAILURE:"):
        res = ({"status": "error", "message": job_output}, 200)
    else:
        res = ({"status": "done", "qir_output": job_output}, 201)
    return res


def startServer(port):
    print("Server Started")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="debug")


if __name__ == '__main__':
    args = sys.argv[1:]

    # Load the device library if provided as an argument
    if len(args) == 2 and args[0] == '-device-lib':
        print(f"Loading device library from {args[1]}")
        llvm.load_library_permanently(args[1])

    print("Server Starting")
    startServer(62453)
