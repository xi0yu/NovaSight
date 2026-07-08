from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from novasight.config import parse_runtime_config
from novasight.runtime.reconfigurator import RuntimeReconfigurator

router = APIRouter(prefix="/api/v1/control", tags=["control"])
logger = logging.getLogger("novasight.api.control")

_CONTROL_SECTIONS = {"calibration", "control"}


@router.get("")
@router.get("/")
def get_control_config(request: Request) -> dict[str, Any]:
    snapshot = request.app.state.runtime.config_store.snapshot()
    runtime_state = request.app.state.runtime.state()
    return {
        "calibration": asdict(snapshot.calibration),
        "control": asdict(snapshot.control),
        "config_version": request.app.state.runtime.config_store.version,
        "runtime": {
            "running": runtime_state.running,
            "vision": runtime_state.vision,
            "fatal_error": runtime_state.fatal_error,
        },
    }


@router.put("")
@router.put("/")
async def put_control_config(request: Request) -> dict[str, Any]:
    return await _apply_control_config(request)


@router.post("")
@router.post("/")
async def post_control_config(request: Request) -> dict[str, Any]:
    return await _apply_control_config(request)


async def _apply_control_config(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="control config payload must be a mapping")
    unknown = sorted(set(payload) - _CONTROL_SECTIONS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown control config section(s): {', '.join(unknown)}",
        )
    if not any(section in payload for section in _CONTROL_SECTIONS):
        raise HTTPException(
            status_code=400,
            detail="payload must include calibration or control",
        )
    current = asdict(request.app.state.runtime.config_store.snapshot())
    for section in _CONTROL_SECTIONS:
        if section not in payload:
            continue
        section_payload = payload[section]
        if not isinstance(section_payload, dict):
            raise HTTPException(
                status_code=400,
                detail=f"control config section '{section}' must be a mapping",
            )
        current_section = current.get(section)
        if not isinstance(current_section, dict):
            raise HTTPException(
                status_code=500,
                detail=f"runtime config section '{section}' is unavailable",
            )
        current_section.update(section_payload)
    try:
        config = parse_runtime_config(current)
        report = RuntimeReconfigurator(request.app).apply(config)
    except ValueError as exc:
        logger.warning("control config update rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "control config updated sections=%s restart_required=%s",
        ",".join(section for section in _CONTROL_SECTIONS if section in payload),
        report.restart_required,
    )
    return {
        "calibration": report.config["calibration"],
        "control": report.config["control"],
        "applied": report.applied,
        "restart_required": report.restart_required,
        "sections": [asdict(section) for section in report.sections],
        "message": report.message,
    }
