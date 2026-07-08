from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time
from typing import Any


def get_telemetry_summary(app_state: Any) -> dict[str, Any]:
    runtime = getattr(app_state, "runtime", None)
    state = _runtime_state(runtime)
    statistics = dict(state.get("statistics", {})) if isinstance(state.get("statistics"), dict) else {}
    return {
        "generated_ts_ns": time.monotonic_ns(),
        "runtime": {
            "running": bool(state.get("running", False)),
            "source": state.get("source", ""),
            "fatal_error": state.get("fatal_error"),
        },
        "capture": _capture_summary(state.get("capture")),
        "inference": _mapping(state.get("inference")),
        "pipeline": _mapping(state.get("pipeline")),
        "executor": _mapping(state.get("executor")),
        "statistics": statistics,
        "control": _mapping(getattr(runtime, "last_control", None)),
        "target": _mapping(getattr(runtime, "last_target", None)),
        "execution": _mapping(getattr(runtime, "last_execution", None)),
        "active_model": _mapping(state.get("active_model")),
    }


def _runtime_state(runtime: Any) -> dict[str, Any]:
    state_fn = getattr(runtime, "state", None)
    if not callable(state_fn):
        return {}
    try:
        state = state_fn()
    except Exception as exc:
        return {"fatal_error": {"message": f"runtime state failed: {exc}"}}
    if is_dataclass(state):
        return asdict(state)
    return dict(state) if isinstance(state, dict) else {}


def _capture_summary(value: Any) -> dict[str, Any]:
    capture = _mapping(value)
    return {
        "device": capture.get("device", ""),
        "available": bool(capture.get("available", False)),
        "profile": _mapping(capture.get("profile")),
        "statistics": _mapping(capture.get("statistics")),
        "reason": capture.get("reason", ""),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["get_telemetry_summary"]
