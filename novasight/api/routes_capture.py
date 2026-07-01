from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


router = APIRouter(prefix="/api/capture", tags=["capture"])


class CaptureSelectRequest(BaseModel):
    device: str
    preference: str | None = None
    pixel_format: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None


@router.get("/capabilities")
def capabilities(request: Request, device: str = "/dev/video0") -> dict:
    return asdict(request.app.state.capture.capabilities(device))


@router.get("/state")
def state(request: Request) -> dict:
    return asdict(request.app.state.capture.state)


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest):
    config = request.app.state.capture.config
    for field in ("preference", "pixel_format", "width", "height", "fps"):
        value = getattr(payload, field)
        if value is not None:
            setattr(config, field, value)

    state = request.app.state.capture.configure(payload.device)
    body = asdict(state)
    if state.available is False:
        return JSONResponse(status_code=400, content=body)
    return body
