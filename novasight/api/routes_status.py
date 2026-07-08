from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1", tags=["status"])


@router.get("/status")
def get_status(request: Request) -> dict[str, Any]:
    return asdict(request.app.state.runtime.state())
