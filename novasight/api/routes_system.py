from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from novasight.system import collect_system_info


router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("")
@router.get("/")
def system_summary(request: Request) -> dict[str, Any]:
    payload = collect_system_info().to_dict()
    payload["instance_lock"] = _instance_lock_status(request)
    return payload


@router.get("/status")
def system_status(request: Request) -> dict[str, Any]:
    return system_summary(request)


def _instance_lock_status(request: Request) -> dict[str, Any]:
    instance_lock = getattr(request.app.state, "instance_lock", None)
    status = getattr(instance_lock, "status", None)
    if callable(status):
        return dict(status())
    return {
        "configured": False,
        "path": "",
        "acquired": False,
        "pid": None,
    }


__all__ = ["router", "system_status", "system_summary"]
