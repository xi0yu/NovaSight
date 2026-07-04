from __future__ import annotations

import asyncio
import logging
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
        _apply_config(request, config)
    except ValueError as exc:
        logger.warning("runtime config update rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("runtime config updated restart_required=%s", bool(request.app.state.runtime.running))
    return {
        "config": asdict(config),
        "schema": runtime_config_schema(config),
        "restart_required": bool(request.app.state.runtime.running),
    }


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


def _apply_config(request: Request, config) -> None:
    app = request.app
    previous_config = getattr(app.state, "config", None)
    previous_roi_size = (
        getattr(getattr(previous_config, "roi", None), "size", None)
        if previous_config is not None
        else None
    )
    app.state.config = config
    app.state.capture.config = config.capture
    app.state.capture.roi_size = config.roi.size
    app.state.executors = ExecutorRegistry.from_config(config)
    app.state.hardware = create_hardware_box(config)
    app.state.inference.configure(
        confidence_threshold=config.inference.confidence_threshold,
        nms_threshold=config.inference.nms_threshold,
    )
    app.state.runtime.executors = app.state.executors
    app.state.runtime.hardware = app.state.hardware
    app.state.runtime.update_config(config)
    if previous_roi_size is not None and previous_roi_size != config.roi.size:
        _reconfigure_live_capture_for_roi(app)


def _reconfigure_live_capture_for_roi(app) -> None:
    capture = app.state.capture
    session = getattr(capture, "session", None)
    state = getattr(capture, "state", None)
    profile = getattr(state, "profile", None)
    if (
        profile is None
        or getattr(state, "available", False) is not True
        or (session is not None and getattr(session, "running", False) is not True)
    ):
        return
    if getattr(profile, "preference", "") == "image":
        return
    logger.info(
        "capture roi changed; rebuilding live capture pipeline device=%s roi_size=%s",
        profile.device,
        app.state.config.roi.size,
    )
    new_state = capture.configure(
        profile.device,
        preference="manual",
        pixel_format=profile.pixel_format,
        width=profile.width,
        height=profile.height,
        fps=profile.fps,
    )
    config_error = getattr(capture, "last_config_error", None)
    if config_error is not None:
        raise ValueError(
            f"runtime config applied but live capture ROI rebuild failed: {config_error.last_error}"
        )
    if getattr(new_state, "available", False) is not True:
        raise ValueError(
            f"runtime config applied but live capture ROI rebuild failed: {new_state.last_error}"
        )
