from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@router.get("/api/runtime/state")
def runtime_state(request: Request) -> dict[str, Any]:
    return asdict(request.app.state.runtime.state())
