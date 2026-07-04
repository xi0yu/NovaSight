from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from novasight.config import parse_runtime_config
from novasight.config.schema import runtime_config_schema
from novasight.runtime.reconfigurator import RuntimeReconfigurator
from novasight.runtime.pipeline import RuntimePipeline
from novasight.runtime.status import StatusHub


router = APIRouter(tags=["runtime"])
logger = logging.getLogger("novasight.api.runtime")


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
        report = RuntimeReconfigurator(request.app).apply(config)
    except ValueError as exc:
        logger.warning("runtime config update rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("runtime config updated restart_required=%s", bool(request.app.state.runtime.running))
    return report.asdict()


@router.get("/api/license")
def get_license(request: Request) -> dict[str, Any]:
    return request.app.state.license.asdict()


@router.post("/api/license/activate")
async def activate_license(request: Request) -> dict[str, Any]:
    return await put_license(request)


@router.put("/api/license")
async def put_license(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        status = request.app.state.license.save(str(payload.get("key", "")))
    except ValueError as exc:
        logger.warning("license activation rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "license activated tier=%s fingerprint=%s expires_at=%s",
        status.tier,
        status.fingerprint,
        status.expires_at,
    )
    return asdict(status)


@router.delete("/api/license")
def delete_license(request: Request) -> dict[str, Any]:
    logger.info("license cleared")
    return asdict(request.app.state.license.clear())


@router.post("/api/runtime/start")
def start_runtime(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    if runtime.pipeline is None:
        runtime.pipeline = RuntimePipeline(
            capture=request.app.state.capture,
            runtime=runtime,
        )
    try:
        runtime.pipeline.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("runtime pipeline started")
    return runtime.pipeline.status()


@router.post("/api/runtime/stop")
def stop_runtime(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    if runtime.pipeline is not None:
        runtime.pipeline.stop()
        logger.info("runtime pipeline stopped")
        return runtime.pipeline.status()
    runtime.running = False
    logger.info("runtime pipeline stop requested while idle")
    return {"running": False}


@router.post("/api/runtime/local-trigger")
async def update_local_trigger(request: Request) -> dict[str, Any]:
    payload = await request.json()
    active = bool(payload.get("active", False))
    raw_bindings = payload.get("bindings", [])
    bindings = [str(item) for item in raw_bindings[:2]] if isinstance(raw_bindings, list) else []
    return request.app.state.runtime.update_local_trigger(active=active, bindings=bindings)


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    status = websocket.app.state.license.status()
    if not status.configured or not status.valid:
        await websocket.close(code=4401, reason="license required")
        return
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
