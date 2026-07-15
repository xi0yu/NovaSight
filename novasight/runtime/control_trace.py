from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CONTROL_TRACE_SCHEMA_NAME = "novasight.control_trace"
CONTROL_TRACE_SCHEMA_VERSION = 4
MONOTONIC_CLOCK_DOMAIN = "monotonic"
UNKNOWN_REASON_DEVICE_FEEDBACK = "device_feedback_unavailable"

CONTROL_TRACE_FIELD_UNITS: dict[str, str] = {
    "detection.capture_ts": "ns",
    "detection.publish_ts": "ns",
    "detection.input_age": "ms",
    "detection.inference": "ms",
    "detection.result_age": "ms",
    "tracker.state_ts": "ns",
    "tracker.position": "px",
    "tracker.velocity": "px/s",
    "tracker.position_sigma": "px",
    "control.control_now": "ns",
    "control.prediction_horizon": "ms",
    "control.error_px": "px",
    "control.predicted_error_px": "px",
    "control.error_rad": "rad",
    "control.error_rate_rad_s": "rad/s",
    "control.p": "rad",
    "control.d": "rad",
    "control.prediction_delta_px": "px",
    "control.prediction_delta_rad": "rad",
    "control.prediction_velocity_term": "rad",
    "counts.planned": "counts",
    "counts.theoretical": "counts",
    "counts.mode_limited": "counts",
    "counts.deadzone_limited": "counts",
    "counts.slew_limited": "counts",
    "counts.feasible": "counts",
    "counts.queued": "counts",
    "counts.sent": "counts",
    "counts.estimated_applied": "counts",
    "counts.unobserved": "counts",
    "scheduler.pending_age": "ms",
    "scheduler.created_ts": "ns",
    "scheduler.expires_ts": "ns",
    "device.send_start_ts": "ns",
    "device.send_end_ts": "ns",
    "device.scheduler_send_delay": "us",
    "correlation.capture_ts": "ns",
    "algorithm.measurement_dt": "ms",
    "algorithm.aim": "px",
    "algorithm.error_measured": "px",
    "algorithm.error_real": "px",
    "algorithm.error_control": "px",
    "algorithm.velocity_x": "px/s",
    "algorithm.robust_velocity_x": "px/ms",
    "algorithm.velocity_spread": "px/ms",
    "algorithm.prediction_horizon": "ms",
    "algorithm.prediction_offset_x": "px",
    "algorithm.full_error_counts": "counts",
    "algorithm.float_demand": "counts",
    "algorithm.integer_command": "counts",
    "algorithm.quantizer_residual": "counts",
}


class ControlTraceJsonlRecorder:
    def __init__(self, path: str | Path, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.flush = bool(flush)
        self._lock = threading.Lock()
        self._file: Any | None = None

    def record_control_trace(self, record: Mapping[str, Any]) -> None:
        self.record(record)

    def record(self, record: Mapping[str, Any]) -> None:
        payload = serialize_control_trace(record)
        with self._lock:
            file_obj = self._ensure_file()
            file_obj.write(payload)
            file_obj.write("\n")
            if self.flush:
                file_obj.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
            self._file = None

    def _ensure_file(self) -> Any:
        if self._file is not None:
            return self._file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        return self._file


def serialize_control_trace(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_control_trace_record(
    *,
    control: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    inference: Mapping[str, Any] | None,
    execution: Mapping[str, Any] | None,
    scheduler_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    control_payload = _dict(control)
    target_payload = _dict(target)
    inference_payload = _dict(inference)
    execution_payload = _dict(execution)
    execution_metadata = _dict(execution_payload.get("metadata"))
    pipeline = _dict(control_payload.get("pipeline"))
    algorithm_velocity_unit = _text(pipeline.get("velocity_unit"))
    mouse = _first_dict(
        control_payload.get("mouse_observation"),
        target_payload.get("mouse_observation"),
    )
    track = _dict(control_payload.get("track_diagnostics"))
    estimate = _selected_track_estimate(track)
    scheduler = _scheduler_metadata(execution_payload, scheduler_status)
    generation = _first_int(
        inference_payload.get("generation"),
        inference_payload.get("detection_batch_generation"),
        control_payload.get("trajectory_generation"),
        control_payload.get("frame_id"),
        inference_payload.get("frame_id"),
    )
    frame_id = _first_int(control_payload.get("frame_id"), target_payload.get("frame_id"), inference_payload.get("frame_id"))
    capture_ts = _first_int(
        control_payload.get("capture_ts_ns"),
        target_payload.get("capture_ts_ns"),
        inference_payload.get("capture_ts_ns"),
        inference_payload.get("detection_batch_capture_ts_ns"),
    )
    command_id = _first_int(scheduler.get("command_id"), _dict(execution_payload.get("metadata")).get("command_id"))
    correlation_id = _correlation_id(generation=generation, frame_id=frame_id, capture_ts_ns=capture_ts)
    prediction_delta_px = _axis_pair(
        _subtract(mouse.get("predicted_x_px"), mouse.get("observed_x_px")),
        _subtract(mouse.get("predicted_y_px"), mouse.get("observed_y_px")),
    )
    prediction_delta_rad = _axis_pair(
        _subtract(pipeline.get("predicted_error_x_rad"), pipeline.get("observed_error_x_rad")),
        _subtract(pipeline.get("predicted_error_y_rad"), pipeline.get("observed_error_y_rad")),
    )
    effective_kp = _axis_pair(
        _effective_gain(pipeline.get("p_x_rad"), pipeline.get("predicted_error_x_rad")),
        _effective_gain(pipeline.get("p_y_rad"), pipeline.get("predicted_error_y_rad")),
    )
    prediction_velocity_term = _axis_pair(
        _multiply(prediction_delta_rad["x"], effective_kp["x"]),
        _multiply(prediction_delta_rad["y"], effective_kp["y"]),
    )
    sent = bool(execution_payload.get("sent")) if execution_payload else False
    sent_counts = _axis_pair(
        execution_payload.get("output_dx") if sent else None,
        execution_payload.get("output_dy") if sent else None,
    )
    device_clock_domain = _text(execution_metadata.get("device_send_clock_domain")) or MONOTONIC_CLOCK_DOMAIN
    return {
        "schema": {
            "name": CONTROL_TRACE_SCHEMA_NAME,
            "version": CONTROL_TRACE_SCHEMA_VERSION,
        },
        "field_units": dict(CONTROL_TRACE_FIELD_UNITS),
        "correlation": {
            "id": correlation_id,
            "detection_generation": generation,
            "frame_id": frame_id,
            "capture_ts": _timestamp(capture_ts, _clock_domain(inference_payload)),
            "trajectory_generation": _first_int(
                control_payload.get("trajectory_generation"),
                scheduler.get("trajectory_generation"),
                scheduler.get("pending_trajectory_generation"),
            ),
            "command_id": command_id,
        },
        "detection": {
            "generation": generation,
            "frame_id": frame_id,
            "source_sequence": _first_int(inference_payload.get("source_sequence")),
            "capture_ts": _timestamp(capture_ts, _clock_domain(inference_payload)),
            "publish_ts": _timestamp(_first_int(inference_payload.get("publish_ts_ns")), _clock_domain(inference_payload)),
            "input_age_ms": _optional_number(inference_payload.get("input_age_ms")),
            "inference_ms": _optional_number(inference_payload.get("inference_ms")),
            "result_age_ms": _optional_number(inference_payload.get("result_age_ms")),
            "is_stale": _optional_bool(inference_payload.get("is_stale")),
        },
        "tracker": {
            "track_id": _first_int(target_payload.get("track_id"), estimate.get("track_id"), track.get("selected_track_id")),
            "state": _text(control_payload.get("selector_state") or track.get("selected_track_state") or track.get("selection_state")),
            "state_ts": _timestamp(_first_int(estimate.get("state_ts_ns")), MONOTONIC_CLOCK_DOMAIN),
            "position_px": _axis_pair(estimate.get("x"), estimate.get("y")),
            "velocity_px_s": _axis_pair(estimate.get("vx"), estimate.get("vy")),
            "position_sigma_px": _optional_number(estimate.get("position_sigma_px")),
            "cov_trace": _optional_number(estimate.get("cov_trace")),
            "nis": _optional_number(estimate.get("nis")),
            "prediction_confidence": _optional_number(estimate.get("prediction_confidence")),
            "identity_confidence": _first_number(target_payload.get("identity_confidence"), track.get("selected_identity_confidence")),
            "missing_ms": _optional_number(track.get("selected_missing_ms")),
        },
        "control": {
            "mode": _text(pipeline.get("control_mode")),
            "control_now": _timestamp(
                _first_int(control_payload.get("control_now_ts_ns"), control_payload.get("control_start_ts_ns")),
                MONOTONIC_CLOCK_DOMAIN,
            ),
            "prediction_horizon_ms": _first_number(
                _multiply(mouse.get("prediction_horizon_s"), 1000.0),
                pipeline.get("prediction_horizon_ms"),
                inference_payload.get("prediction_horizon_ms"),
            ),
            "prediction_confidence": _first_number(
                pipeline.get("prediction_confidence"),
                mouse.get("prediction_confidence"),
                estimate.get("prediction_confidence"),
            ),
            "prediction_delta_px": prediction_delta_px,
            "prediction_delta_rad": prediction_delta_rad,
            "prediction_velocity_term_rad": prediction_velocity_term,
            "error_px": _axis_pair(pipeline.get("observed_error_x_px"), pipeline.get("observed_error_y_px")),
            "predicted_error_px": _axis_pair(pipeline.get("predicted_error_x_px"), pipeline.get("predicted_error_y_px")),
            "error_rad": _axis_pair(pipeline.get("observed_error_x_rad"), pipeline.get("observed_error_y_rad")),
            "predicted_error_rad": _axis_pair(pipeline.get("predicted_error_x_rad"), pipeline.get("predicted_error_y_rad")),
            "error_rate_rad_s": _axis_pair(pipeline.get("d_ema_x_rad_s"), pipeline.get("d_ema_y_rad_s")),
            "error_rate_raw_rad_s": _axis_pair(pipeline.get("d_raw_x_rad_s"), pipeline.get("d_raw_y_rad_s")),
            "p_rad": _axis_pair(pipeline.get("p_x_rad"), pipeline.get("p_y_rad")),
            "d_rad": _axis_pair(pipeline.get("d_x_rad"), pipeline.get("d_y_rad")),
            "u_rad": _axis_pair(pipeline.get("limited_output_x_rad"), pipeline.get("limited_output_y_rad")),
            "velocity_confidence": _optional_number(mouse.get("velocity_confidence")),
            "control_allowed": _optional_bool(control_payload.get("control_allowed")),
            "will_emit": _optional_bool(control_payload.get("will_emit")),
            "reason": _text(
                control_payload.get("no_send_reason")
                or control_payload.get("reason")
                or control_payload.get("selection_reason")
            ),
        },
        "algorithm_decision": {
            "algorithm_id": _text(pipeline.get("algorithm") or pipeline.get("algorithm_id")),
            "class_id": _first_int(pipeline.get("class_id"), target_payload.get("cls")),
            "effective_aim_y_ratio": _first_number(
                pipeline.get("effective_aim_y_ratio"),
                control_payload.get("aim_y_ratio"),
                target_payload.get("aim_y_ratio"),
            ),
            "phase": _text(pipeline.get("mode") or pipeline.get("control_mode")),
            "measurement_dt_ms": _first_number(
                pipeline.get("measurement_dt_ms"),
                _multiply(pipeline.get("measurement_dt_s"), 1000.0),
            ),
            "aim_px": _axis_pair(pipeline.get("aim_x"), pipeline.get("aim_y")),
            "bbox": {
                "x1": _optional_number(pipeline.get("bbox_x1")),
                "y1": _optional_number(pipeline.get("bbox_y1")),
                "x2": _optional_number(pipeline.get("bbox_x2")),
                "y2": _optional_number(pipeline.get("bbox_y2")),
                "width": _optional_number(pipeline.get("bbox_width")),
                "height": _optional_number(pipeline.get("bbox_height")),
            },
            "error_real_px": _axis_pair(
                pipeline.get("error_real_x"),
                pipeline.get("error_real_y"),
            ),
            "error_measured_px": _axis_pair(
                _first_number(pipeline.get("error_meas_x"), pipeline.get("error_real_x")),
                _first_number(pipeline.get("error_meas_y"), pipeline.get("error_real_y")),
            ),
            "error_control_px": _axis_pair(
                pipeline.get("error_control_x"),
                pipeline.get("error_control_y"),
            ),
            "estimator": {
                "velocity_x_px_s": (
                    None
                    if algorithm_velocity_unit == "px/ms"
                    else _optional_number(pipeline.get("estimated_velocity_x"))
                ),
                "innovation_x": _optional_number(pipeline.get("innovation_x")),
                "normalized_innovation_x": _optional_number(
                    pipeline.get("normalized_innovation_x")
                ),
                "motion_confidence": _optional_number(pipeline.get("motion_confidence")),
                "direction_quality": _optional_number(pipeline.get("direction_quality")),
                "accepted": _optional_bool(pipeline.get("estimator_accepted")),
                "reset": _optional_bool(pipeline.get("estimator_reset")),
            },
            "robust_velocity": {
                "unit": algorithm_velocity_unit,
                "history_position_count": _first_int(
                    pipeline.get("history_position_count")
                ),
                "segments_px_ms": [
                    _optional_number(pipeline.get("velocity_1")),
                    _optional_number(pipeline.get("velocity_2")),
                    _optional_number(pipeline.get("velocity_3")),
                ],
                "median_px_ms": _optional_number(pipeline.get("median_velocity")),
                "filtered_px_ms": _optional_number(pipeline.get("filtered_velocity")),
                "spread_px_ms": _optional_number(pipeline.get("velocity_spread")),
                "history_quality": _optional_number(pipeline.get("history_quality")),
                "spread_quality": _optional_number(pipeline.get("spread_quality")),
                "trend_quality": _optional_number(pipeline.get("trend_quality")),
                "detection_quality": _optional_number(pipeline.get("detection_quality")),
                "track_quality": _optional_number(pipeline.get("track_quality")),
                "motion_confidence": _optional_number(pipeline.get("motion_confidence")),
            },
            "prediction": {
                "reference_dt_ms": _optional_number(pipeline.get("reference_dt_ms")),
                "lead_frames": _optional_number(pipeline.get("prediction_lead_frames")),
                "horizon_ms": _first_number(
                    pipeline.get("prediction_horizon_ms"),
                    _multiply(pipeline.get("prediction_horizon_s"), 1000.0),
                ),
                "raw_offset_x": _optional_number(pipeline.get("prediction_raw_offset_x")),
                "weighted_offset_x": _optional_number(
                    pipeline.get("prediction_weighted_offset_x")
                ),
                "weight": _optional_number(pipeline.get("prediction_weight")),
                "confidence": _optional_number(pipeline.get("prediction_confidence")),
                "allowed_cap_x": _optional_number(
                    pipeline.get("prediction_allowed_cap_x")
                ),
                "safe_offset_x": _optional_number(
                    pipeline.get("prediction_safe_offset_x")
                ),
                "allowed": _optional_bool(pipeline.get("prediction_allowed")),
                "crossing_limited": _optional_bool(
                    pipeline.get("prediction_crossing_limited")
                ),
            },
            "recoil": {
                "enabled": _optional_bool(pipeline.get("recoil_enabled")),
                "active": _optional_bool(pipeline.get("recoil_active")),
                "left_hold_ms": _optional_number(pipeline.get("recoil_left_hold_ms")),
                "ramp": _optional_number(pipeline.get("recoil_ramp")),
                "y_counts": _optional_number(pipeline.get("recoil_y_counts_float")),
            },
            "full_error_counts": _axis_pair(
                pipeline.get("full_error_counts_x"),
                pipeline.get("full_error_counts_y"),
            ),
            "float_demand": _axis_pair(
                pipeline.get("float_demand_x"),
                pipeline.get("float_demand_y"),
            ),
            "integer_command": _axis_pair(
                pipeline.get("integer_command_x"),
                pipeline.get("integer_command_y"),
            ),
            "quantizer_residual": _axis_pair(
                pipeline.get("quantizer_residual_x"),
                pipeline.get("quantizer_residual_y"),
            ),
            "overzero_detected": {
                "x": _optional_bool(pipeline.get("overzero_detected_x")),
                "y": _optional_bool(pipeline.get("overzero_detected_y")),
            },
            "will_emit": _optional_bool(pipeline.get("will_emit")),
            "block_reason": _text(pipeline.get("block_reason")),
            "executor_success": _optional_bool(pipeline.get("executor_success")),
            "executor_block_reason": _text(pipeline.get("executor_block_reason")),
            "delivery_mode": _text(pipeline.get("delivery_mode")),
            "scheduler_used": _optional_bool(pipeline.get("scheduler_used")),
        },
        "counts": {
            "planned_counts": _axis_pair(control_payload.get("dx"), control_payload.get("dy")),
            "theoretical_counts": _axis_pair(
                _first_number(
                    pipeline.get("theoretical_counts_x_float"),
                    pipeline.get("full_error_counts_x"),
                ),
                _first_number(
                    pipeline.get("theoretical_counts_y_float"),
                    pipeline.get("full_error_counts_y"),
                ),
            ),
            "mode_limited_counts": _axis_pair(
                _first_number(
                    pipeline.get("mode_limited_counts_x_float"),
                    pipeline.get("float_demand_x"),
                ),
                _first_number(
                    pipeline.get("mode_limited_counts_y_float"),
                    pipeline.get("float_demand_y"),
                ),
            ),
            "deadzone_limited_counts": _axis_pair(pipeline.get("deadzone_limited_counts_x_float"), pipeline.get("deadzone_limited_counts_y_float")),
            "slew_limited_counts": _axis_pair(pipeline.get("slew_limited_counts_x_float"), pipeline.get("slew_limited_counts_y_float")),
            "feasible_counts": _axis_pair(pipeline.get("feasible_counts_x_float"), pipeline.get("feasible_counts_y_float")),
            "queued_counts": _axis_pair(scheduler.get("pending_dx"), scheduler.get("pending_dy")),
            "sent_counts": sent_counts,
            "estimated_applied_counts": _unknown_axis_pair(UNKNOWN_REASON_DEVICE_FEEDBACK),
            "unobserved_counts": _unknown_axis_pair(UNKNOWN_REASON_DEVICE_FEEDBACK),
            "residual_counts": _axis_pair(
                _first_number(
                    pipeline.get("residual_x_counts"),
                    pipeline.get("quantizer_residual_x"),
                ),
                _first_number(
                    pipeline.get("residual_y_counts"),
                    pipeline.get("quantizer_residual_y"),
                ),
            ),
            "final_counts": _axis_pair(
                _first_number(
                    pipeline.get("final_dx"),
                    pipeline.get("integer_command_x"),
                ),
                _first_number(
                    pipeline.get("final_dy"),
                    pipeline.get("integer_command_y"),
                ),
            ),
        },
        "scheduler": {
            "used": _optional_bool(pipeline.get("scheduler_used")),
            "delivery_mode": _text(pipeline.get("delivery_mode")),
            "generation": _first_int(
                scheduler.get("trajectory_generation"),
                scheduler.get("pending_trajectory_generation"),
                control_payload.get("trajectory_generation"),
            ),
            "command_id": command_id,
            "plan_id": _first_int(scheduler.get("plan_id"), command_id),
            "command_status": _text(scheduler.get("command_status")),
            "step_index": _first_int(scheduler.get("step_index")),
            "step_count": _first_int(scheduler.get("step_count")),
            "pending_age_ms": _optional_number(scheduler.get("pending_age_ms")),
            "pending_steps": _first_int(scheduler.get("pending_steps")),
            "cancel_reason": _text(scheduler.get("cancel_reason") or scheduler.get("last_cancel_reason")),
            "created_ts": _timestamp(_first_int(scheduler.get("created_ts_ns"), scheduler.get("pending_created_ts_ns")), MONOTONIC_CLOCK_DOMAIN),
            "expires_ts": _timestamp(_first_int(scheduler.get("expires_ts_ns"), scheduler.get("pending_expires_ts_ns")), MONOTONIC_CLOCK_DOMAIN),
        },
        "device": {
            "executor_id": _text(execution_payload.get("executor_id")),
            "send_start_ts": _timestamp(
                _first_int(execution_payload.get("device_send_start_ts_ns"), execution_metadata.get("device_send_start_ts_ns")),
                device_clock_domain,
            ),
            "send_end_ts": _timestamp(
                _first_int(execution_payload.get("device_send_end_ts_ns"), execution_metadata.get("device_send_end_ts_ns")),
                device_clock_domain,
            ),
            "sent": _optional_bool(execution_payload.get("sent")),
            "scheduler_send_delay_us": _first_number(
                execution_payload.get("scheduler_send_delay_us"),
                execution_metadata.get("scheduler_send_delay_us"),
            ),
            "message": _text(execution_payload.get("message")),
            "driver_counts": _axis_pair(control_payload.get("driver_dx"), control_payload.get("driver_dy")),
        },
    }


def _scheduler_metadata(
    execution: Mapping[str, Any],
    scheduler_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = _dict(execution.get("metadata"))
    scheduler = _dict(metadata.get("scheduler"))
    result = dict(scheduler or {})
    nested_execution = _dict(result.get("execution"))
    if nested_execution:
        result.update(nested_execution)
    for key in (
        "command_id",
        "plan_id",
        "trajectory_generation",
        "created_ts_ns",
        "expires_ts_ns",
        "command_status",
        "cancel_reason",
        "pending_dx",
        "pending_dy",
        "pending_steps",
        "step_index",
        "step_count",
        "scheduled_ts_ns",
        "cancelled_remaining_dx",
        "cancelled_remaining_dy",
    ):
        if key in metadata and key not in result:
            result[key] = metadata[key]
    for key, value in _dict(scheduler_status).items():
        result.setdefault(key, value)
    return result


def _timestamp(value: int | None, clock_domain: str) -> dict[str, Any]:
    return {
        "value": value,
        "clock_domain": str(clock_domain or MONOTONIC_CLOCK_DOMAIN),
        "unit": "ns",
    }


def _axis_pair(x: Any, y: Any) -> dict[str, float | None]:
    return {
        "x": _optional_number(x),
        "y": _optional_number(y),
    }


def _unknown_axis_pair(reason: str) -> dict[str, Any]:
    return {
        "x": None,
        "y": None,
        "status": "unknown",
        "reason": reason,
    }


def _prediction_delta_rad(
    delta_px: Mapping[str, Any],
    focal_px: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        "x": _atan_ratio(delta_px.get("x"), focal_px.get("x")),
        "y": _atan_ratio(delta_px.get("y"), focal_px.get("y")),
    }


def _atan_ratio(numerator: Any, denominator: Any) -> float | None:
    import math

    if not isinstance(numerator, (int, float)) or isinstance(numerator, bool):
        return None
    if not isinstance(denominator, (int, float)) or isinstance(denominator, bool):
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return math.atan(float(numerator) / float(denominator))


def _effective_gain(p_component: Any, error_rad: Any) -> float | None:
    if not isinstance(p_component, (int, float)) or isinstance(p_component, bool):
        return None
    if not isinstance(error_rad, (int, float)) or isinstance(error_rad, bool):
        return None
    if abs(float(error_rad)) <= 1e-12:
        return None
    return float(p_component) / float(error_rad)


def _subtract(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return None
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return None
    return float(left) - float(right)


def _selected_track_estimate(track: Mapping[str, Any]) -> dict[str, Any]:
    selected_id = _first_int(track.get("selected_track_id"))
    tracks = track.get("tracks")
    if selected_id is None or not isinstance(tracks, list):
        return {}
    for item in tracks:
        payload = _dict(item)
        if _first_int(payload.get("track_id")) == selected_id:
            return _dict(payload.get("estimate"))
    return {}


def _multiply(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return None
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return None
    return float(left) * float(right)


def _correlation_id(
    *,
    generation: int | None,
    frame_id: int | None,
    capture_ts_ns: int | None,
) -> str:
    generation_part = "unknown" if generation is None else str(generation)
    frame_part = "unknown" if frame_id is None else str(frame_id)
    capture_part = "unknown" if capture_ts_ns is None else str(capture_ts_ns)
    return f"control:{generation_part}:{frame_part}:{capture_part}"


def _clock_domain(payload: Mapping[str, Any]) -> str:
    value = payload.get("clock_domain")
    return str(value).strip() if value else MONOTONIC_CLOCK_DOMAIN


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        payload = _dict(value)
        if payload:
            return payload
    return {}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_number(value)
        if parsed is not None:
            return parsed
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value)


__all__ = [
    "CONTROL_TRACE_FIELD_UNITS",
    "CONTROL_TRACE_SCHEMA_NAME",
    "CONTROL_TRACE_SCHEMA_VERSION",
    "ControlTraceJsonlRecorder",
    "build_control_trace_record",
    "serialize_control_trace",
]
