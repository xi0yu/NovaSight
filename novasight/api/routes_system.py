from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from novasight.system import collect_system_info


router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/status")
def system_status() -> dict[str, Any]:
    return collect_system_info().to_dict()


__all__ = ["router", "system_status"]
