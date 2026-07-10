from __future__ import annotations

import logging
import math
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger("novasight.api.executors")


class DiagnosticMoveRequest(BaseModel):
    dx: int = 1
    dy: int = 0
    repeat: int = Field(default=1, ge=1, le=200)
    interval_ms: int = Field(default=0, ge=0, le=50)
    move_ms: int = Field(default=12, ge=0, le=5000)
    move_kind: str | None = None
    ctrl_x1: int | None = None
    ctrl_y1: int | None = None
    ctrl_x2: int | None = None
    ctrl_y2: int | None = None


class DiagnosticCircleRequest(BaseModel):
    radius: int = Field(default=8, ge=1, le=80)
    steps: int = Field(default=32, ge=8, le=128)
    interval_ms: int = Field(default=8, ge=0, le=50)


@router.get("/api/executors")
def list_executors(request: Request) -> dict[str, Any]:
    return request.app.state.executors.status()


@router.post("/api/executors/kmnet/connect")
def connect_kmnet(request: Request) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    connect_async = getattr(executor, "connect_async", None)
    if callable(connect_async):
        return connect_async()
    connect = getattr(executor, "connect", None)
    if not callable(connect):
        raise HTTPException(status_code=400, detail="kmNet executor does not support connect")
    return connect()


@router.post("/api/executors/kmnet/disconnect")
def disconnect_kmnet(request: Request) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    disconnect = getattr(executor, "disconnect", None)
    if not callable(disconnect):
        raise HTTPException(status_code=400, detail="kmNet executor does not support disconnect")
    scheduler = getattr(request.app.state.executors, "scheduler", None)
    clear = getattr(scheduler, "clear", None)
    if callable(clear):
        clear("KMNET_MANUAL_DISCONNECT")
    return disconnect()


@router.get("/api/executors/kmnet/buttons")
def buttons_kmnet(request: Request) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    read = getattr(executor, "read_buttons", None)
    if not callable(read):
        raise HTTPException(status_code=400, detail="kmNet executor does not support button read")
    buttons = read()
    logger.info("kmNet diagnostic buttons %s", buttons)
    return buttons


@router.post("/api/executors/kmnet/diagnostic-move")
def diagnostic_move_kmnet(
    request: Request,
    payload: DiagnosticMoveRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        raise HTTPException(status_code=400, detail="kmNet executor does not support diagnostic move")
    if payload.repeat == 1 and payload.interval_ms == 0:
        result = move(
            payload.dx,
            payload.dy,
            move_kind=payload.move_kind,
            move_ms=payload.move_ms,
            bezier_ctrl=_bezier_ctrl_from_payload(payload),
        )
        logger.info(
            "kmNet diagnostic move sent dx=%s dy=%s kind=%s move_ms=%s result=%s",
            payload.dx,
            payload.dy,
            payload.move_kind,
            payload.move_ms,
            result,
        )
        return {
            **_execution_result_payload(result),
            "queued": False,
            "steps_requested": 1,
            "steps_sent": 1 if bool(getattr(result, "sent", False)) else 0,
            "status": executor.status() if callable(getattr(executor, "status", None)) else {},
        }
    background_tasks.add_task(_run_diagnostic_move, executor, payload)
    logger.info(
        "kmNet diagnostic move queued dx=%s dy=%s repeat=%s interval_ms=%s kind=%s",
        payload.dx,
        payload.dy,
        payload.repeat,
        payload.interval_ms,
        payload.move_kind,
    )
    return {
        "sent": True,
        "queued": True,
        "message": "queued diagnostic move",
        "steps_requested": payload.repeat,
        "steps_sent": 0,
        "move_ms": payload.move_ms,
        "failed": None,
        "status": executor.status() if callable(getattr(executor, "status", None)) else {},
    }


def _run_diagnostic_move(executor: Any, payload: DiagnosticMoveRequest) -> None:
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        logger.warning("kmNet diagnostic move dropped: executor has no diagnostic_move")
        return
    result: Any | None = None
    sent = 0
    failed: dict[str, Any] | None = None
    for index in range(payload.repeat):
        result = move(
            payload.dx,
            payload.dy,
            move_kind=payload.move_kind,
            move_ms=payload.move_ms,
            bezier_ctrl=_bezier_ctrl_from_payload(payload),
        )
        if bool(getattr(result, "sent", False)):
            sent += 1
        else:
            failed = {
                "step": index + 1,
                "message": str(getattr(result, "message", "")),
            }
            break
        if payload.interval_ms > 0 and index < payload.repeat - 1:
            time.sleep(payload.interval_ms / 1000)
    if failed is None and sent > 0:
        logger.info(
            "kmNet diagnostic move finished dx=%s dy=%s repeat=%s kind=%s",
            payload.dx,
            payload.dy,
            sent,
            payload.move_kind,
        )
    else:
        logger.warning(
            "kmNet diagnostic move failed dx=%s dy=%s repeat=%s message=%s failed=%s kind=%s",
            payload.dx,
            payload.dy,
            payload.repeat,
            str(getattr(result, "message", "")) if result is not None else "",
            failed,
            payload.move_kind,
        )


def _bezier_ctrl_from_payload(payload: DiagnosticMoveRequest) -> tuple[int, int, int, int] | None:
    if payload.ctrl_x1 is None or payload.ctrl_y1 is None or payload.ctrl_x2 is None or payload.ctrl_y2 is None:
        return None
    return (
        int(payload.ctrl_x1),
        int(payload.ctrl_y1),
        int(payload.ctrl_x2),
        int(payload.ctrl_y2),
    )


@router.post("/api/executors/kmnet/diagnostic-circle")
def diagnostic_circle_kmnet(
    request: Request,
    payload: DiagnosticCircleRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        raise HTTPException(status_code=400, detail="kmNet executor does not support diagnostic circle")
    background_tasks.add_task(_run_diagnostic_circle, executor, payload)
    logger.info(
        "kmNet diagnostic circle queued steps=%s radius=%s interval_ms=%s",
        payload.steps,
        payload.radius,
        payload.interval_ms,
    )
    return {
        "sent": True,
        "queued": True,
        "message": "queued diagnostic circle",
        "steps_requested": payload.steps,
        "steps_sent": 0,
        "radius": payload.radius,
        "interval_ms": payload.interval_ms,
        "failed": None,
        "status": executor.status() if callable(getattr(executor, "status", None)) else {},
    }


def _run_diagnostic_circle(executor: Any, payload: DiagnosticCircleRequest) -> None:
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        logger.warning("kmNet diagnostic circle dropped: executor has no diagnostic_move")
        return
    previous_x = payload.radius
    previous_y = 0
    sent = 0
    failed: dict[str, Any] | None = None
    for index in range(1, payload.steps + 1):
        angle = (math.tau * index) / payload.steps
        x = int(round(math.cos(angle) * payload.radius))
        y = int(round(math.sin(angle) * payload.radius))
        dx = x - previous_x
        dy = y - previous_y
        previous_x = x
        previous_y = y
        if dx == 0 and dy == 0:
            continue

        result = move(dx, dy)
        if bool(getattr(result, "sent", False)):
            sent += 1
        else:
            failed = {
                "step": index,
                "dx": dx,
                "dy": dy,
                "message": str(getattr(result, "message", "")),
            }
            break
        if payload.interval_ms > 0:
            time.sleep(payload.interval_ms / 1000)

    status = executor.status() if callable(getattr(executor, "status", None)) else {}
    if failed is None and sent > 0:
        logger.info("kmNet diagnostic circle finished steps=%s radius=%s status=%s", sent, payload.radius, status)
    else:
        logger.warning("kmNet diagnostic circle failed sent=%s failed=%s", sent, failed)


def _kmnet_executor(request: Request) -> Any:
    executor = getattr(request.app.state.executors, "executors", {}).get("kmnet")
    if executor is None:
        raise HTTPException(status_code=404, detail="kmNet executor not registered")
    return executor


def _execution_result_payload(result: Any) -> dict[str, Any]:
    intent = getattr(result, "intent", None)
    return {
        "executor_id": getattr(result, "executor_id", "unknown"),
        "sent": bool(getattr(result, "sent", False)),
        "message": str(getattr(result, "message", "")),
        "metadata": getattr(result, "metadata", None) or {},
        "intent": {
            "dx": getattr(intent, "dx", 0),
            "dy": getattr(intent, "dy", 0),
            "accepted": getattr(intent, "accepted", False),
            "clipped": getattr(intent, "clipped", False),
            "reason": getattr(intent, "reason", ""),
        },
    }
