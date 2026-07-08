from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from novasight.system import collect_system_info


router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("")
@router.get("/")
def system_summary() -> dict[str, Any]:
    return collect_system_info().to_dict()


@router.get("/status")
def system_status() -> dict[str, Any]:
    return system_summary()


__all__ = ["router", "system_status", "system_summary"]
