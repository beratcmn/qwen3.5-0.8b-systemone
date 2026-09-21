from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .contract import SystemOneRequest, SystemOneResponse
from .engine import ContextLimitError, EngineBusyError, InferenceError, get_engine

MAX_BODY_BYTES = 24 * 1024 * 1024

app = FastAPI(title="Qwen3.5 System One", version="0.1.0")
CONNECT_FOUR_HTML = Path(__file__).with_name("connect_four.html")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


async def _request_model(request: Request) -> SystemOneRequest:
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_BODY_BYTES:
                raise HTTPException(
                    status_code=413, detail="request body exceeds 24 MiB"
                )
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail="invalid Content-Length header"
            ) from error
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body exceeds 24 MiB")
    try:
        payload = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid number: {value}")
            ),
        )
        return SystemOneRequest.model_validate(payload)
    except ValidationError as error:
        raise HTTPException(
            status_code=422, detail=error.errors(include_url=False)
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready() -> dict[str, bool]:
    return {"ready": get_engine().loaded}


@app.get("/demos/connect-four", include_in_schema=False)
def connect_four() -> FileResponse:
    return FileResponse(CONNECT_FOUR_HTML)


@app.post(
    "/v1/systemone",
    response_model=SystemOneResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": SystemOneRequest.model_json_schema()}
            },
        }
    },
)
async def system_one(request: Request) -> SystemOneResponse:
    payload = await _request_model(request)
    try:
        return await run_in_threadpool(get_engine().evaluate, payload)
    except EngineBusyError as error:
        raise HTTPException(status_code=529, detail=str(error)) from error
    except ContextLimitError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except InferenceError as error:
        raise HTTPException(status_code=529, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
