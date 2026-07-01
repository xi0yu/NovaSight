from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from pydantic import BaseModel

from novasight.capture import query_capabilities


router = APIRouter(prefix="/api/capture", tags=["capture"])


class CaptureSelectRequest(BaseModel):
    device: str = "/dev/video0"


@router.get("/capabilities")
def capabilities(device: str = "/dev/video0") -> dict:
    return asdict(query_capabilities(device))


@router.get("/state")
def state(request: Request) -> dict:
    return asdict(request.app.state.capture.state)


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest) -> dict:
    state = request.app.state.capture.configure(payload.device)
    return asdict(state)
