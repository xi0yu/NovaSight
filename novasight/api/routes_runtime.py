from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from novasight.config import parse_runtime_config
from novasight.config.schema import runtime_config_schema
from novasight.runtime.reconfigurator import RuntimeReconfigurator
from novasight.runtime.pipeline_factory import create_runtime_pipeline


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
    return await post_config(request)


@router.post("/api/config")
async def post_config(request: Request) -> dict[str, Any]:
    payload = await _request_json(request)
    field_update = isinstance(payload, dict) and {
        "section",
        "key",
        "value",
    }.issubset(payload)
    try:
        reconfigurator = RuntimeReconfigurator(request.app)
        if field_update:
            report = await run_in_threadpool(
                reconfigurator.apply_field,
                str(payload["section"]),
                str(payload["key"]),
                payload["value"],
            )
        else:
            config = _config_from_payload(payload)
            report = await run_in_threadpool(reconfigurator.apply, config)
    except ValueError as exc:
        logger.warning("runtime config update rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "runtime config updated restart_required=%s",
        bool(request.app.state.runtime.running),
    )
    return report.asdict(include_schema=not field_update)


async def _request_json(request: Request) -> Any:
    try:
        return await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc


def _config_from_payload(payload: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        raise ValueError("runtime config update payload must be a mapping")
    return parse_runtime_config(payload)


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
    power_supervisor = getattr(request.app.state, "runtime_power", None)
    if power_supervisor is not None:
        try:
            power_status = power_supervisor.request_start()
        except Exception as exc:
            return _runtime_start_failure(runtime, exc)
        if power_status.get("running") is not True:
            runtime.fatal_error = None
            return {
                "running": False,
                "accepted": True,
                "failed": False,
                "standby": True,
                "power_saving": power_status,
            }
        pipeline = getattr(runtime, "pipeline", None)
        pipeline_status = pipeline.status() if pipeline is not None else {"running": True}
        runtime.fatal_error = None
        return {
            **pipeline_status,
            "accepted": True,
            "failed": False,
            "standby": False,
            "power_saving": power_status,
        }
    try:
        pipeline_status = _start_runtime_pipeline_core(request.app)
    except Exception as exc:
        logger.exception("runtime pipeline start failed")
        return _runtime_start_failure(runtime, exc)
    return {
        **pipeline_status,
        "accepted": True,
        "failed": False,
    }


def _start_runtime_pipeline_core(app: Any) -> dict[str, Any]:
    runtime = app.state.runtime
    if runtime.pipeline is None:
        runtime.pipeline = create_runtime_pipeline(
            capture=app.state.capture,
            runtime=runtime,
        )
    runtime.pipeline.start()
    runtime.fatal_error = None
    logger.info("runtime pipeline started")
    return dict(runtime.pipeline.status())


def _runtime_start_failure(runtime: Any, exc: Exception) -> dict[str, Any]:
    logger.error("runtime pipeline start failed: %s", exc)
    _clear_failed_runtime_pipeline(runtime)
    failure = {
        "type": "RUNTIME_START_FAILED",
        "thread": "api.runtime.start",
        "message": str(exc),
        "crash_log": "",
    }
    runtime.fatal_error = failure
    return {
        "running": False,
        "accepted": False,
        "failed": True,
        "recoverable": True,
        "last_error": str(exc),
        "fatal_error": failure,
    }


def _stop_existing_runtime_pipeline(runtime: Any) -> None:
    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception as stop_exc:
        logger.warning("runtime pipeline cleanup before restart failed: %s", stop_exc)
    runtime.pipeline = None
    runtime.running = False


def _clear_failed_runtime_pipeline(runtime: Any) -> None:
    _stop_existing_runtime_pipeline(runtime)


def _stop_runtime_pipeline_core(runtime: Any, reason: str) -> None:
    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is not None:
        pipeline.stop()
        runtime.pipeline = None
        runtime.running = False
        logger.info("runtime pipeline stopped reason=%s", reason)
        return
    reset_session = getattr(runtime, "reset_runtime_session", None)
    if callable(reset_session):
        reset_session(str(reason))
    else:
        runtime.running = False
    logger.info("runtime pipeline stop requested while idle reason=%s", reason)


@router.post("/api/runtime/stop")
def stop_runtime(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    power_supervisor = getattr(request.app.state, "runtime_power", None)
    if power_supervisor is not None:
        power_status = power_supervisor.request_stop()
    else:
        _stop_runtime_pipeline_core(runtime, "USER_STOP")
        power_status = None
    payload = asdict(runtime.state())
    if power_status is not None:
        payload["power_saving"] = power_status
    return payload


@router.get("/api/runtime/power")
def get_runtime_power(request: Request) -> dict[str, Any]:
    supervisor = getattr(request.app.state, "runtime_power", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="runtime power supervisor unavailable")
    return {"power_saving": supervisor.status()}


@router.post("/api/runtime/presence/heartbeat")
async def post_runtime_presence_heartbeat(request: Request) -> dict[str, Any]:
    supervisor = getattr(request.app.state, "runtime_power", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="runtime power supervisor unavailable")
    payload = await _request_json(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="heartbeat payload must be a mapping")
    try:
        status = await run_in_threadpool(
            supervisor.heartbeat,
            str(payload.get("host_id") or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("runtime auto-resume after host heartbeat failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "power_saving": status}


@router.post("/api/runtime/calibration/fingerprint")
async def post_runtime_calibration_fingerprint(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a mapping")
    try:
        status = request.app.state.runtime.update_external_sensitivity_fingerprint(
            str(payload.get("fingerprint", "")),
            source=str(payload.get("source", "external")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return status


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    status = websocket.app.state.license.status()
    if not status.configured or not status.valid:
        await websocket.close(code=4401, reason="license required")
        return
    await websocket.accept()
    hub = websocket.app.state.status_hub
    queue = await hub.subscribe()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_text(payload)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)
