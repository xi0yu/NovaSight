from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class DiagnosticMoveRequest(BaseModel):
    dx: int = 1
    dy: int = 0


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
    return _execution_result_payload(result)


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
