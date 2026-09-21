from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contract import SystemOneRequest, SystemOneResponse
from .engine import ContextLimitError, EngineBusyError, InferenceError, get_engine

MAX_BODY_BYTES = 24 * 1024 * 1024

app = FastAPI(title="Qwen3.5 System One", version="0.1.0")
CONNECT_FOUR_HTML = Path(__file__).with_name("connect_four.html")
DINO_HTML = Path(__file__).with_name("dino.html")
DINO_LOCK = threading.Lock()
DINO_HISTORY: deque[dict[str, Any]] = deque(maxlen=80)
DINO_STATE: dict[str, Any] = {
    "revision": 0,
    "session_id": None,
    "status": "idle",
    "error": None,
    "latest": None,
}

DinoAction = Literal["wait", "jump", "duck", "restart"]


class DinoTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(min_length=1, max_length=64)
    sequence: int = Field(ge=0)
    captured_at: float = Field(ge=0)
    status: Literal["calibrating", "running", "paused", "stopped", "error"]
    frame_count: Literal[1, 2] = 1
    frame_base64: str | None = Field(default=None, max_length=750_000)
    model_action: DinoAction | None = None
    executed_action: DinoAction | None = None
    probabilities: dict[DinoAction, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, max_length=500)


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


@app.get("/demos/dino", include_in_schema=False)
def dino() -> FileResponse:
    return FileResponse(DINO_HTML)


@app.get("/v1/dino/telemetry", include_in_schema=False)
def dino_telemetry() -> dict[str, Any]:
    with DINO_LOCK:
        return {**DINO_STATE, "history": list(DINO_HISTORY)}


@app.post("/v1/dino/telemetry", include_in_schema=False)
def update_dino_telemetry(update: DinoTelemetry) -> dict[str, bool]:
    item = update.model_dump(mode="json")
    with DINO_LOCK:
        if update.session_id != DINO_STATE["session_id"]:
            DINO_HISTORY.clear()
            DINO_STATE["session_id"] = update.session_id
            DINO_STATE["latest"] = None
        DINO_STATE["revision"] += 1
        DINO_STATE["status"] = update.status
        DINO_STATE["error"] = update.error
        if update.model_action is not None:
            DINO_STATE["latest"] = item
            DINO_HISTORY.append(
                {key: value for key, value in item.items() if key != "frame_base64"}
            )
    return {"ok": True}


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
