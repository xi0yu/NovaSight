from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/executors")
def list_executors(request: Request) -> dict[str, Any]:
    return request.app.state.executors.status()
