from __future__ import annotations

import time
from dataclasses import asdict
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from novasight.capture.preview import render_preview_frame


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


@router.get("/stream.mjpg")
def stream(request: Request):
    capture = request.app.state.capture
    if capture.source is None:
        state = capture.configure(capture.config.device)
        config_error = getattr(capture, "last_config_error", None)
        if config_error is not None:
            return JSONResponse(status_code=503, content=asdict(config_error))
        if state.available is False:
            return JSONResponse(status_code=503, content=asdict(state))
    return StreamingResponse(
        _mjpeg_frames(capture, runtime=getattr(request.app.state, "runtime", None)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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


def _mjpeg_frames(capture, *, runtime=None) -> Iterator[bytes]:
    import cv2

    empty_reads = 0
    max_empty_reads = 50
    while True:
        try:
            frame = capture.read_frame()
        except Exception:
            break
        if frame is None:
            empty_reads += 1
            if empty_reads >= max_empty_reads:
                capture.mark_unavailable(f"capture stream produced {max_empty_reads} empty reads")
                break
            if capture.empty_read_sleep_s > 0:
                time.sleep(capture.empty_read_sleep_s)
            continue
        empty_reads = 0
        preview = render_preview_frame(frame, runtime=runtime)
        ok, encoded = cv2.imencode(".jpg", preview)
        if not ok:
            capture.state.frames_dropped += 1
            capture.state.last_error = "capture stream jpeg encode failed"
            continue
        payload = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
            + payload
            + b"\r\n"
        )
