from __future__ import annotations

import logging
import time
from io import BytesIO
from dataclasses import asdict
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from novasight.capture.preview import render_preview_frame
from novasight.config import save_runtime_config
from novasight.runtime.reconfigurator import RuntimeReconfigurator
from novasight.roi import normalize_roi_size


router = APIRouter(prefix="/api/capture", tags=["capture"])
logger = logging.getLogger("novasight.api.capture")
PREVIEW_FPS_CHOICES = (5, 10, 15, 30)


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


class CaptureCapabilitiesRequest(BaseModel):
    device: str = "/dev/video0"

    @field_validator("device")
    @classmethod
    def device_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("device must not be blank")
        return stripped


class ImageSourceRequest(BaseModel):
    path: str
    fps: int = 30

    @field_validator("path")
    @classmethod
    def path_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("path must not be blank")
        return stripped


class PreviewStateRequest(BaseModel):
    enabled: bool


@router.get("/capabilities")
def capabilities(request: Request, device: str = "/dev/video0") -> dict:
    return asdict(request.app.state.capture.capabilities(device))


@router.post("/capabilities")
def post_capabilities(payload: CaptureCapabilitiesRequest, request: Request) -> dict:
    return asdict(request.app.state.capture.capabilities(payload.device))


@router.get("/state")
def state(request: Request) -> dict:
    return asdict(request.app.state.capture.state)


@router.get("/stream.mjpg")
def stream(request: Request, fps: int | None = None):
    capture = request.app.state.capture
    config = getattr(request.app.state, "config", None)
    consumers = getattr(config, "consumers", None)
    if consumers is not None and getattr(consumers, "preview", True) is not True:
        return JSONResponse(
            status_code=503,
            content={"message": "预览消费者已关闭，主链路保持运行。"},
        )
    deepstream_backend = _active_deepstream_preview_backend(request)
    if deepstream_backend is not None:
        if not deepstream_backend.running:
            return JSONResponse(
                status_code=503,
                content={"message": "DeepStream 主链未运行，无法打开预览。"},
            )
        preview_status = deepstream_backend.status()
        if preview_status.get("preview_enabled") is not True:
            return JSONResponse(
                status_code=503,
                content={
                    "message": str(
                        preview_status.get("preview_reason")
                        or "DeepStream 硬件预览未启用。"
                    )
                },
            )
        if preview_status.get("preview_active") is not True:
            return JSONResponse(
                status_code=503,
                content={"message": "实时预览已暂停，推理与控制继续运行。"},
            )
        preview_fps = _normalize_preview_fps(
            fps
            if fps is not None
            else getattr(config, "limits", None) and config.limits.stream_fps
        )
        return StreamingResponse(
            _deepstream_mjpeg_frames(
                deepstream_backend,
                preview_fps=preview_fps,
                config_getter=lambda: getattr(request.app.state, "config", None),
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    session = getattr(capture, "session", None)
    if (
        capture.source is None
        or capture.state.available is False
        or (session is not None and not session.running)
    ):
        return JSONResponse(
            status_code=503,
            content={"message": "采集未启动，无法打开预览。"},
        )
    preview_fps = _normalize_preview_fps(
        fps
        if fps is not None
        else getattr(config, "limits", None) and config.limits.stream_fps
    )
    roi_size = _normalize_roi_size(
        getattr(config, "roi", None)
        and config.roi.size
    )
    capture.state.preview_target_fps = preview_fps
    return StreamingResponse(
        _mjpeg_frames(
            capture,
            runtime=getattr(request.app.state, "runtime", None),
            preview_fps=preview_fps,
            roi_size=roi_size,
            config_getter=lambda: getattr(request.app.state, "config", None),
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/preview")
def set_preview_state(payload: PreviewStateRequest, request: Request) -> dict:
    backend = _active_deepstream_preview_backend(request)
    if backend is None:
        raise HTTPException(
            status_code=409,
            detail="当前运行链不支持动态硬件预览控制",
        )
    try:
        return backend.set_preview_active(payload.enabled)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest):
    logger.info(
        "capture select requested device=%s preference=%s pixel_format=%s size=%sx%s fps=%s",
        payload.device,
        payload.preference,
        payload.pixel_format,
        payload.width,
        payload.height,
        payload.fps,
    )
    report = RuntimeReconfigurator(request.app).select_capture(
        device=payload.device,
        preference=payload.preference,
        pixel_format=payload.pixel_format,
        width=payload.width,
        height=payload.height,
        fps=payload.fps,
    )
    if not report.applied:
        capture = report.capture or {}
        logger.warning(
            "capture select rejected device=%s error=%s",
            payload.device,
            report.message,
        )
        return JSONResponse(
            status_code=400,
            content={
                **capture,
                "report": report.asdict(),
            },
        )
    logger.info(
        "capture select applied device=%s backend=%s profile=%s",
        (report.capture or {}).get("device"),
        (report.capture or {}).get("backend"),
        (report.capture or {}).get("profile"),
    )
    return {
        **(report.capture or {}),
        "report": report.asdict(),
    }


@router.post("/image")
def image_source(request: Request, payload: ImageSourceRequest):
    capture = request.app.state.capture
    fps = payload.fps if payload.fps in {1, 5, 15, 30, 60} else 30
    try:
        state = capture.configure_image(payload.path, fps=fps)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "available": False,
                "device": payload.path,
                "last_error": str(exc),
            },
        )
    config = getattr(request.app.state, "config", None)
    if config is not None:
        config.source.default = "image"
        config.source.image_path = payload.path
        config.source.image_fps = fps
        capture.roi_size = config.roi.size
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is not None:
            runtime.update_config(config)
        config_path = getattr(request.app.state, "config_path", None)
        if config_path is not None:
            save_runtime_config(config, config_path)
    body = asdict(state)
    if state.available is False:
        return JSONResponse(status_code=400, content=body)
    return body


@router.post("/stop")
def stop(request: Request) -> dict:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        _clear_runtime_pipeline(runtime, "capture stopped by user")
    state = request.app.state.capture.stop("capture stopped by user")
    config = getattr(request.app.state, "config", None)
    if config is not None:
        config.source.default = "null"
        if runtime is not None:
            runtime.update_config(config)
        config_path = getattr(request.app.state, "config_path", None)
        if config_path is not None:
            save_runtime_config(config, config_path)
    return asdict(state)


def _normalize_preview_fps(value: int | None) -> int:
    if value in PREVIEW_FPS_CHOICES:
        return int(value)
    return 30


def _normalize_roi_size(value: int | None) -> int:
    try:
        return normalize_roi_size(value or 640)
    except ValueError:
        return 640


def _clear_runtime_pipeline(runtime, reason: str) -> None:
    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception as exc:
            logger.warning("runtime pipeline stop after %s failed: %s", reason, exc)
    runtime.pipeline = None
    runtime.running = False
    logger.info("runtime pipeline cleared after %s", reason)


def _active_deepstream_preview_backend(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    pipeline = getattr(runtime, "pipeline", None)
    backend = getattr(pipeline, "backend", None)
    if getattr(backend, "backend_id", "") != "deepstream_nvinfer":
        return None
    if not callable(getattr(backend, "wait_preview_jpeg", None)):
        return None
    return backend


def _deepstream_mjpeg_frames(
    backend,
    *,
    preview_fps: int = 30,
    config_getter=None,
    max_frames: int | None = None,
    max_attempts: int | None = None,
) -> Iterator[bytes]:
    attempts = 0
    emitted = 0
    last_sequence: int | None = None
    preview_fps = _normalize_preview_fps(preview_fps)
    timeout_s = 1.0 / preview_fps
    emit_interval_s = 1.0 / preview_fps
    next_emit_at = 0.0
    acquire = getattr(backend, "acquire_preview_consumer", None)
    release = getattr(backend, "release_preview_consumer", None)
    if callable(acquire):
        acquire()
    try:
        while True:
            if config_getter is not None:
                config = config_getter()
                consumers = getattr(config, "consumers", None)
                if consumers is not None and getattr(consumers, "preview", True) is not True:
                    break
            if max_frames is not None and emitted >= max_frames:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            if not backend.running:
                break
            if getattr(backend, "preview_active", True) is False:
                break
            attempts += 1
            result = backend.wait_preview_jpeg(
                after_sequence=last_sequence,
                timeout_s=timeout_s,
            )
            if result is None:
                continue
            sequence, payload = result
            last_sequence = int(sequence)
            now = time.monotonic()
            if now < next_emit_at:
                continue
            next_emit_at = now + emit_interval_s
            emitted += 1
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
                + b"\r\n"
            )
    finally:
        if callable(release):
            release()


def _mjpeg_frames(
    capture,
    *,
    runtime=None,
    preview_fps: int = 30,
    roi_size: int = 640,
    config_getter=None,
    max_frames: int | None = None,
    max_attempts: int | None = None,
) -> Iterator[bytes]:
    attempts = 0
    emitted = 0
    last_frame_id = 0
    preview_fps = _normalize_preview_fps(preview_fps)
    roi_size = _normalize_roi_size(roi_size)
    interval_s = 1.0 / preview_fps
    capture.state.preview_target_fps = preview_fps
    while True:
        if config_getter is not None:
            config = config_getter()
            consumers = getattr(config, "consumers", None)
            if consumers is not None and getattr(consumers, "preview", True) is not True:
                break
        if max_frames is not None and emitted >= max_frames:
            break
        if max_attempts is not None and attempts >= max_attempts:
            break
        attempts += 1
        frame = capture.wait_preview_frame(
            after_frame_id=last_frame_id,
            timeout_s=interval_s,
        )
        if frame is None and last_frame_id > 0:
            latest = capture.wait_preview_frame(after_frame_id=None, timeout_s=0.0)
            if latest is not None and int(latest.frame_id) <= int(last_frame_id):
                last_frame_id = 0
                frame = latest
        if frame is None:
            capture.record_preview_drop(target_fps=preview_fps)
            time.sleep(interval_s)
            continue
        last_frame_id = frame.frame_id
        preview = render_preview_frame(
            frame,
            runtime=runtime,
            roi_size=roi_size,
        )
        payload = _encode_jpeg(preview)
        if payload is None:
            capture.record_preview_drop(target_fps=preview_fps)
            capture.state.preview_available = False
            capture.state.preview_reason = (
                "NVMM preview branch has not produced a CPU snapshot"
                if frame.image is None
                else "capture stream jpeg encode failed"
            )
            continue
        capture.state.preview_available = True
        capture.state.preview_reason = ""
        capture.record_preview_output(frame, target_fps=preview_fps)
        emitted += 1
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
