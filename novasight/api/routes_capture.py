from __future__ import annotations

import logging
import time
from io import BytesIO
from dataclasses import asdict
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from novasight.capture.preview import render_preview_frame


router = APIRouter(prefix="/api/capture", tags=["capture"])
logger = logging.getLogger("novasight.api.capture")
PREVIEW_FPS_CHOICES = (15, 30, 60)


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
    session = getattr(capture, "session", None)
    if capture.source is None or (session is not None and not session.running):
        return JSONResponse(
            status_code=503,
            content={"message": "采集未启动，无法打开预览。"},
        )
    preview_fps = _normalize_preview_fps(
        getattr(getattr(request.app.state, "config", None), "limits", None)
        and request.app.state.config.limits.stream_fps
    )
    capture.state.preview_target_fps = preview_fps
    return StreamingResponse(
        _mjpeg_frames(
            capture,
            runtime=getattr(request.app.state, "runtime", None),
            preview_fps=preview_fps,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest):
    capture = request.app.state.capture
    logger.info(
        "capture select requested device=%s preference=%s pixel_format=%s size=%sx%s fps=%s",
        payload.device,
        payload.preference,
        payload.pixel_format,
        payload.width,
        payload.height,
        payload.fps,
    )
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
        logger.warning(
            "capture select rejected device=%s error=%s",
            payload.device,
            config_error.last_error,
        )
        return JSONResponse(
            status_code=400,
            content=asdict(config_error),
        )
    body = asdict(state)
    if state.available is False:
        logger.warning(
            "capture select unavailable device=%s error=%s",
            payload.device,
            state.last_error,
        )
        return JSONResponse(status_code=400, content=body)
    logger.info(
        "capture select applied device=%s backend=%s profile=%s",
        state.device,
        state.backend,
        asdict(state.profile) if state.profile else None,
    )
    return body


@router.post("/stop")
def stop(request: Request) -> dict:
    state = request.app.state.capture.stop("capture stopped by user")
    return asdict(state)


def _normalize_preview_fps(value: int | None) -> int:
    if value in PREVIEW_FPS_CHOICES:
        return int(value)
    return 30


def _mjpeg_frames(
    capture,
    *,
    runtime=None,
    preview_fps: int = 30,
    max_frames: int | None = None,
) -> Iterator[bytes]:
    attempts = 0
    last_frame_id = 0
    preview_fps = _normalize_preview_fps(preview_fps)
    interval_s = 1.0 / preview_fps
    capture.state.preview_target_fps = preview_fps
    while True:
        if max_frames is not None and attempts >= max_frames:
            break
        attempts += 1
        frame = capture.wait_preview_frame(
            after_frame_id=last_frame_id,
            timeout_s=interval_s,
        )
        if frame is None:
            capture.record_preview_drop(target_fps=preview_fps)
            time.sleep(interval_s)
            continue
        last_frame_id = frame.frame_id
        preview = render_preview_frame(frame, runtime=runtime)
        payload = _encode_jpeg(preview)
        if payload is None:
            capture.record_preview_drop(target_fps=preview_fps)
            capture.state.last_error = "capture stream jpeg encode failed"
            continue
        capture.record_preview_output(frame, target_fps=preview_fps)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
            + payload
            + b"\r\n"
        )
        time.sleep(interval_s)


def _encode_jpeg(image) -> bytes | None:
    from PIL import Image

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    elif hasattr(image, "shape"):
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = Image.fromarray(np.ascontiguousarray(arr[:, :, :3][:, :, ::-1]), mode="RGB")
    else:
        return None
    output = BytesIO()
    rgb.save(output, format="JPEG", quality=80)
    return output.getvalue()
