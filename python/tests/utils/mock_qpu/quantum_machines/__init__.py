# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse
from typing import Union
import base64
import binascii
import importlib.util
import uuid
import cudaq
from pydantic import BaseModel
import logging

from ..quake_execution import execute_quake_isolated

_quake_to_qua_ast_type = None
_qubit_mapping_mode_type = None
_cudaq_frontend_import_error = None

if importlib.util.find_spec("cudaq_frontend") is not None:
    try:
        from cudaq_frontend.translate.quake_to_qua_ast import QuakeToQuaAst
        from cudaq_frontend.utils import QubitMappingMode

        _quake_to_qua_ast_type = QuakeToQuaAst
        _qubit_mapping_mode_type = QubitMappingMode
    except Exception as error:
        _cudaq_frontend_import_error = error

# Define the REST Server App
app = FastAPI()


class Input(BaseModel):
    format: str
    data: str


# Jobs look like the following type
class Job(BaseModel):
    shots: int
    content: str
    executor: str
    qubit_mapping_mode: str = None
    api_key: str = None
    source: str = "oq2"
    output_format: str = "histogram"


createdJobs = {}


def _decode_content(content: str):
    try:
        return base64.b64decode(content, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return content


def _create_response(job: Job, job_id):
    if job.source != "quake":
        raise ValueError(f"Unsupported program source: {job.source}")

    results = execute_quake_isolated(_decode_content(job.content), job.shots,
                                     job.output_format)
    return {"id": job_id, "results": results, "status": "Done"}


def _validate_quake_ir(job: Job):
    if job.source != "quake":
        return

    if _cudaq_frontend_import_error is not None:
        raise HTTPException(
            status_code=500,
            detail=("cudaq_frontend is installed but could not be imported: "
                    f"{_cudaq_frontend_import_error}"))

    if _quake_to_qua_ast_type is None:
        return

    try:
        quake_ir = _decode_content(job.content)

        translator = _quake_to_qua_ast_type(
            qubit_mapping=_qubit_mapping_mode_type.backend,
            repetitions=job.shots)
        translator.translate(quake_ir)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Quake to QUA translation failed: {error}") from error


@app.get("/v1/config/qubits", response_class=PlainTextResponse)
async def get_qubit_config():
    return ("Number of nodes: 5\n"
            "0 --> {1, 2, 3, 4}\n"
            "1 --> {0, 2, 3, 4}\n"
            "2 --> {0, 1, 3, 4}\n"
            "3 --> {0, 1, 2, 4}\n"
            "4 --> {0, 1, 2, 3}\n")


@app.post("/v1/execute")
async def post_execute_job(job: Job,
                           token: Union[str,
                                        None] = Header(alias="Authorization",
                                                       default=None)):
    logging.info("In /v1/execute. code: %s", job)
    _validate_quake_ir(job)
    jobID = uuid.uuid4()
    try:
        response = _create_response(job, jobID)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Quake simulation failed: {error}") from error
    createdJobs[str(jobID)] = response
    logging.info("In /v1/execute. response: %s", response)
    return response


@app.get("/v1/results/{id}")
async def get_results(id: str):
    response = createdJobs.get(id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {id}")
    logging.info("In /v1/results/%s. returning job results: %s", id, response)
    assert response
    return response


def start_server(port):
    import uvicorn
    cudaq.reset_target()
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")
