from __future__ import annotations

import logging
import math
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger("novasight.api.executors")


class DiagnosticMoveRequest(BaseModel):
    dx: int = 1
    dy: int = 0


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
    return disconnect()


@router.post("/api/executors/kmnet/diagnostic-move")
def diagnostic_move_kmnet(request: Request, payload: DiagnosticMoveRequest) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        raise HTTPException(status_code=400, detail="kmNet executor does not support diagnostic move")
    result = move(payload.dx, payload.dy)
    response = _execution_result_payload(result)
    response["status"] = executor.status() if callable(getattr(executor, "status", None)) else {}
    if response["sent"]:
        logger.info("kmNet diagnostic move sent dx=%s dy=%s", payload.dx, payload.dy)
    else:
        logger.warning(
            "kmNet diagnostic move rejected dx=%s dy=%s message=%s",
            payload.dx,
            payload.dy,
            response.get("message"),
        )
    return response


@router.post("/api/executors/kmnet/diagnostic-circle")
def diagnostic_circle_kmnet(request: Request, payload: DiagnosticCircleRequest) -> dict[str, Any]:
    executor = _kmnet_executor(request)
    move = getattr(executor, "diagnostic_move", None)
    if not callable(move):
        raise HTTPException(status_code=400, detail="kmNet executor does not support diagnostic circle")

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
        logger.info("kmNet diagnostic circle sent steps=%s radius=%s", sent, payload.radius)
    else:
        logger.warning("kmNet diagnostic circle failed sent=%s failed=%s", sent, failed)
    return {
        "sent": failed is None and sent > 0,
        "steps_requested": payload.steps,
        "steps_sent": sent,
        "radius": payload.radius,
        "interval_ms": payload.interval_ms,
        "failed": failed,
        "status": status,
    }


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
        "intent": {
            "dx": getattr(intent, "dx", 0),
            "dy": getattr(intent, "dy", 0),
            "accepted": getattr(intent, "accepted", False),
            "clipped": getattr(intent, "clipped", False),
            "reason": getattr(intent, "reason", ""),
        },
    }
