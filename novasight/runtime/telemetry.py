from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time
from typing import Any


_CONTROL_TIMING_FIELDS = (
    "frame_id",
    "target_id",
    "capture_ts_ns",
    "inference_end_ts_ns",
    "control_now_ts_ns",
    "measurement_dt_ms",
    "frame_age_ms",
    "configured_actuation_delay_s",
    "actuation_delay_source",
    "prediction_horizon_ms",
)


def get_telemetry_summary(app_state: Any) -> dict[str, Any]:
    runtime = getattr(app_state, "runtime", None)
    state = _runtime_state(runtime)
    # The browser consumes this endpoint as RuntimeState. Keep the complete
    # REST state contract at the top level and add lightweight control trace
    # fields alongside it instead of replacing the state with a partial shape.
    summary = dict(state)
    summary.update(
        {
            "generated_ts_ns": time.monotonic_ns(),
            "runtime": {
                "running": bool(state.get("running", False)),
                "source": state.get("source", ""),
                "fatal_error": state.get("fatal_error"),
            },
            "control_timing": _control_timing_summary(runtime),
            "control": _mapping(getattr(runtime, "last_control", None)),
            "target": _mapping(getattr(runtime, "last_target", None)),
            "execution": _mapping(getattr(runtime, "last_execution", None)),
        }
    )
    return summary


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


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if isinstance(value, dict) else {}


def _control_timing_summary(runtime: Any) -> dict[str, Any]:
    timing = _mapping(getattr(runtime, "last_control_timing", None))
    if timing:
        return timing
    inference = _mapping(getattr(runtime, "last_inference_status", None))
    return {key: inference[key] for key in _CONTROL_TIMING_FIELDS if key in inference}


__all__ = ["get_telemetry_summary"]
