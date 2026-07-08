from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from novasight.runtime.reconfigurator import RuntimeReconfigurator

router = APIRouter(prefix="/api/v1/capture", tags=["capture"])
logger = logging.getLogger("novasight.api.capture.v1")


class CaptureConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
def get_capture_capabilities(request: Request, device: str = "/dev/video0") -> dict[str, Any]:
    return asdict(request.app.state.capture.capabilities(device))


@router.put("/config")
async def put_capture_config(request: Request, payload: CaptureConfigRequest) -> dict[str, Any]:
    return _apply_capture_config(request, payload)


@router.post("/config")
async def post_capture_config(request: Request, payload: CaptureConfigRequest) -> dict[str, Any]:
    return _apply_capture_config(request, payload)


def _apply_capture_config(request: Request, payload: CaptureConfigRequest) -> dict[str, Any]:
    try:
        report = RuntimeReconfigurator(request.app).select_capture(
            device=payload.device,
            preference=payload.preference,
            pixel_format=payload.pixel_format,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
        )
    except ValueError as exc:
        logger.warning("capture config rejected device=%s error=%s", payload.device, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    body = {
        "capture": report.capture or {},
        "config": {
            "source": report.config.get("source", {}),
            "capture": report.config.get("capture", {}),
            "roi": report.config.get("roi", {}),
            "inference": report.config.get("inference", {}),
        },
        "applied": report.applied,
        "rolled_back": report.rolled_back,
        "restart_required": report.restart_required,
        "sections": [asdict(section) for section in report.sections],
        "message": report.message,
    }
    if not report.applied:
        return JSONResponse(status_code=400, content=body)
    logger.info("capture config applied device=%s", payload.device)
    return body
