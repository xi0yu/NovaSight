from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from novasight.config import parse_runtime_config
from novasight.config.schema import runtime_config_schema
from novasight.executors import ExecutorRegistry
from novasight.hardware import create_hardware_box
from novasight.runtime.pipeline import RuntimePipeline
from novasight.runtime.status import StatusHub


router = APIRouter(tags=["runtime"])


@router.get("/api/config")
def get_config(request: Request) -> dict[str, Any]:
    return asdict(request.app.state.runtime.config_store.snapshot())


@router.get("/api/config/schema")
def get_config_schema(request: Request) -> dict[str, Any]:
    return runtime_config_schema(request.app.state.runtime.config_store.snapshot())


@router.put("/api/config")
async def put_config(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        config = parse_runtime_config(payload)
        _apply_config(request, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "config": asdict(config),
        "schema": runtime_config_schema(config),
        "restart_required": bool(request.app.state.runtime.running),
    }


@router.get("/api/license")
def get_license(request: Request) -> dict[str, Any]:
    return request.app.state.license.asdict()


@router.put("/api/license")
async def put_license(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        status = request.app.state.license.save(str(payload.get("key", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(status)


@router.delete("/api/license")
def delete_license(request: Request) -> dict[str, Any]:
    return asdict(request.app.state.license.clear())


@router.post("/api/runtime/start")
def start_runtime(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    if runtime.pipeline is None:
        runtime.pipeline = RuntimePipeline(
            capture=request.app.state.capture,
            runtime=runtime,
        )
    runtime.pipeline.start()
    return runtime.pipeline.status()


@router.post("/api/runtime/stop")
def stop_runtime(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    if runtime.pipeline is not None:
        runtime.pipeline.stop()
        return runtime.pipeline.status()
    runtime.running = False
    return {"running": False}


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    hub = StatusHub(websocket.app.state.runtime)
    queue = await hub.subscribe()
    pump = asyncio.create_task(hub.pump_forever())
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)
        pump.cancel()


def _apply_config(request: Request, config) -> None:
    app = request.app
    app.state.config = config
    app.state.capture.config = config.capture
    app.state.executors = ExecutorRegistry.from_config(config)
    app.state.hardware = create_hardware_box(config)
    app.state.runtime.executors = app.state.executors
    app.state.runtime.update_config(config)
