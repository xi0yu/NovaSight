from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator


router = APIRouter(prefix="/api/capture", tags=["capture"])


class CaptureSelectRequest(BaseModel):
    device: str
    preference: str | None = None
    pixel_format: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None

    @field_validator("device")
    @classmethod
    def device_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("device must not be blank")
        return stripped


@router.get("/capabilities")
def capabilities(request: Request, device: str = "/dev/video0") -> dict:
    return asdict(request.app.state.capture.capabilities(device))


@router.get("/state")
def state(request: Request) -> dict:
    return asdict(request.app.state.capture.state)


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest):
    capture = request.app.state.capture
    state = capture.configure(
        payload.device,
        preference=payload.preference,
        pixel_format=payload.pixel_format,
        width=payload.width,
        height=payload.height,
        fps=payload.fps,
    )
    config_error = getattr(capture, "last_config_error", None)
    if config_error is not None:
        return JSONResponse(
            status_code=400,
            content=asdict(config_error),
        )
    body = asdict(state)
    if state.available is False:
        return JSONResponse(status_code=400, content=body)
    return body
