from __future__ import annotations

import csv
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from novasight.runtime.control_trace import (
    CONTROL_TRACE_FIELD_UNITS,
    CONTROL_TRACE_SCHEMA_NAME,
    CONTROL_TRACE_SCHEMA_VERSION,
    ControlTraceJsonlRecorder,
    build_control_trace_record,
    serialize_control_trace,
)


CONTROL_FRAME_FIELDS = [
    "frame_id",
    "capture_ts_ns",
    "control_now_ts_ns",
    "measurement_dt_ms",
    "frame_age_ms",
    "prediction_horizon_ms",
    "inference_latency_ms",
    "control_width_px",
    "control_height_px",
    "candidate_count",
    "track_id",
    "track_state",
    "control_mode",
    "target_confidence",
    "kalman_x_px",
    "kalman_y_px",
    "kalman_vx_px_s",
    "kalman_vy_px_s",
    "observed_aim_x_px",
    "observed_aim_y_px",
    "predicted_aim_x_px",
    "predicted_aim_y_px",
    "base_prediction_confidence",
    "velocity_confidence",
    "prediction_confidence",
    "prediction_applied",
    "observed_error_x_px",
    "observed_error_y_px",
    "predicted_error_x_px",
    "predicted_error_y_px",
    "observed_error_x_rad",
    "observed_error_y_rad",
    "predicted_error_x_rad",
    "predicted_error_y_rad",
    "d_raw_x_rad_s",
    "d_raw_y_rad_s",
    "d_ema_x_rad_s",
    "d_ema_y_rad_s",
    "requested_output_x_rad",
    "requested_output_y_rad",
    "limited_output_x_rad",
    "limited_output_y_rad",
    "theoretical_counts_x_float",
    "theoretical_counts_y_float",
    "mode_limited_counts_x_float",
    "mode_limited_counts_y_float",
    "deadzone_limited_counts_x_float",
    "deadzone_limited_counts_y_float",
    "slew_limited_counts_x_float",
    "slew_limited_counts_y_float",
    "feasible_counts_x_float",
    "feasible_counts_y_float",
    "residual_x_counts",
    "residual_y_counts",
    "budget_clamped_x",
    "budget_clamped_y",
    "planned_x_counts",
    "planned_y_counts",
    "executed_counts_last_20ms_x",
    "executed_counts_last_20ms_y",
    "executed_counts_last_40ms_x",
    "executed_counts_last_40ms_y",
    "executed_counts_last_60ms_x",
    "executed_counts_last_60ms_y",
    "driver_x_counts",
    "driver_y_counts",
    "device_send_start_ts_ns",
    "device_send_end_ts_ns",
    "scheduler_send_delay_us",
    "command_id",
    "plan_id",
    "step_index",
    "step_count",
    "expires_ts_ns",
    "command_status",
    "cancel_reason",
    "cancelled_remaining_dx",
    "cancelled_remaining_dy",
    "global_state",
    "reason_code",
    "config_version",
]


class ControlFrameCsvRecorder:
    def __init__(self, path: str | Path, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.flush = bool(flush)
        self._lock = threading.Lock()
        self._file: Any | None = None
        self._writer: csv.DictWriter | None = None

    def record_control_frame(self, record: Mapping[str, Any]) -> None:
        self.record(record)

    def record(self, record: Mapping[str, Any]) -> None:
        row = {field: _parquet_value(record.get(field, "")) for field in CONTROL_FRAME_FIELDS}
        with self._lock:
            writer = self._ensure_writer()
            writer.writerow(row)
            if self.flush and self._file is not None:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
            self._file = None
            self._writer = None

    def _ensure_writer(self) -> csv.DictWriter:
        if self._writer is not None:
            return self._writer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CONTROL_FRAME_FIELDS)
        if write_header:
            self._writer.writeheader()
        return self._writer


class ControlFrameParquetRecorder:
    def __init__(self, path: str | Path) -> None:
        if not self.available():
            raise RuntimeError(
                "pyarrow is required for production Parquet control-frame recording"
            )
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._closed = False

    @classmethod
    def available(cls) -> bool:
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except Exception:
            return False
        return True

    def record_control_frame(self, record: Mapping[str, Any]) -> None:
        self.record(record)

    def record(self, record: Mapping[str, Any]) -> None:
        row = {field: _csv_value(record.get(field, "")) for field in CONTROL_FRAME_FIELDS}
        with self._lock:
            if self._closed:
                raise RuntimeError("control-frame Parquet recorder is closed")
            self._rows.append(row)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            rows = list(self._rows)
            self._rows.clear()
            self._closed = True
        self._write_rows(rows)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(
                "pyarrow is required for production Parquet control-frame recording"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([(field, pa.string()) for field in CONTROL_FRAME_FIELDS])
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, self.path)


def build_control_frame_record(
    *,
    control: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    inference: Mapping[str, Any] | None,
    execution: Mapping[str, Any] | None,
    config: Any,
    scheduler_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    control_payload = _dict(control)
    target_payload = _dict(target)
    inference_payload = _dict(inference)
    execution_payload = _dict(execution)
    if not _same_frame_payload(control_payload, target_payload):
        target_payload = {}
    pipeline = _dict(control_payload.get("pipeline"))
    mouse = _first_dict(
        control_payload.get("mouse_observation"),
        target_payload.get("mouse_observation"),
    )
    candidate = _dict(control_payload.get("candidate_filter"))
    track = _dict(control_payload.get("track_diagnostics"))
    selected_track = _selected_track(track)
    scheduler = _scheduler_metadata(execution_payload, scheduler_status)
    calibration = getattr(config, "calibration", None)
    return {
        "frame_id": _first_number(control_payload.get("frame_id"), target_payload.get("frame_id"), inference_payload.get("frame_id")),
        "capture_ts_ns": _first_number(control_payload.get("capture_ts_ns"), target_payload.get("capture_ts_ns"), inference_payload.get("capture_ts_ns")),
        "control_now_ts_ns": _number(control_payload.get("control_now_ts_ns")),
        "measurement_dt_ms": _none_if_missing(_number(pipeline.get("measurement_dt_s")), 1000.0, lambda value, scale: value * scale),
        "frame_age_ms": _first_number(control_payload.get("frame_age_ms"), inference_payload.get("frame_age_ms")),
        "prediction_horizon_ms": _none_if_missing(_number(mouse.get("prediction_horizon_s")), 1000.0, lambda value, scale: value * scale),
        "inference_latency_ms": _number(inference_payload.get("detection_batch_inference_latency_ms")),
        "control_width_px": _number(mouse.get("control_width_px")),
        "control_height_px": _number(mouse.get("control_height_px")),
        "candidate_count": _first_number(candidate.get("filtered_candidates"), control_payload.get("candidates"), inference_payload.get("mapped_detections")),
        "track_id": _first_number(track.get("selected_track_id"), target_payload.get("track_id"), mouse.get("target_id")),
        "track_state": _text(track.get("selected_track_state") or track.get("selection_state")),
        "control_mode": _text(pipeline.get("control_mode")),
        "target_confidence": _first_number(mouse.get("target_confidence"), target_payload.get("score")),
        "kalman_x_px": _number(mouse.get("kalman_x_px")),
        "kalman_y_px": _number(mouse.get("kalman_y_px")),
        "kalman_vx_px_s": _number(mouse.get("kalman_vx_px_s")),
        "kalman_vy_px_s": _number(mouse.get("kalman_vy_px_s")),
        "observed_aim_x_px": _number(mouse.get("observed_x_px")),
        "observed_aim_y_px": _number(mouse.get("observed_y_px")),
        "predicted_aim_x_px": _number(mouse.get("predicted_x_px")),
        "predicted_aim_y_px": _number(mouse.get("predicted_y_px")),
        "base_prediction_confidence": _number(mouse.get("base_prediction_confidence")),
        "velocity_confidence": _number(mouse.get("velocity_confidence")),
        "prediction_confidence": _number(mouse.get("prediction_confidence")),
        "prediction_applied": bool(
            _number(mouse.get("prediction_confidence"))
            and (
                _number(mouse.get("predicted_x_px")) != _number(mouse.get("observed_x_px"))
                or _number(mouse.get("predicted_y_px")) != _number(mouse.get("observed_y_px"))
            )
        ),
        "observed_error_x_px": _number(pipeline.get("observed_error_x_px")),
        "observed_error_y_px": _number(pipeline.get("observed_error_y_px")),
        "predicted_error_x_px": _number(pipeline.get("predicted_error_x_px")),
        "predicted_error_y_px": _number(pipeline.get("predicted_error_y_px")),
        "observed_error_x_rad": _number(pipeline.get("observed_error_x_rad")),
        "observed_error_y_rad": _number(pipeline.get("observed_error_y_rad")),
        "predicted_error_x_rad": _number(pipeline.get("predicted_error_x_rad")),
        "predicted_error_y_rad": _number(pipeline.get("predicted_error_y_rad")),
        "d_raw_x_rad_s": _number(pipeline.get("d_raw_x_rad_s")),
        "d_raw_y_rad_s": _number(pipeline.get("d_raw_y_rad_s")),
        "d_ema_x_rad_s": _number(pipeline.get("d_ema_x_rad_s")),
        "d_ema_y_rad_s": _number(pipeline.get("d_ema_y_rad_s")),
        "requested_output_x_rad": _number(pipeline.get("requested_output_x_rad")),
        "requested_output_y_rad": _number(pipeline.get("requested_output_y_rad")),
        "limited_output_x_rad": _number(pipeline.get("limited_output_x_rad")),
        "limited_output_y_rad": _number(pipeline.get("limited_output_y_rad")),
        "theoretical_counts_x_float": _number(pipeline.get("theoretical_counts_x_float")),
        "theoretical_counts_y_float": _number(pipeline.get("theoretical_counts_y_float")),
        "mode_limited_counts_x_float": _number(pipeline.get("mode_limited_counts_x_float")),
        "mode_limited_counts_y_float": _number(pipeline.get("mode_limited_counts_y_float")),
        "deadzone_limited_counts_x_float": _number(pipeline.get("deadzone_limited_counts_x_float")),
        "deadzone_limited_counts_y_float": _number(pipeline.get("deadzone_limited_counts_y_float")),
        "slew_limited_counts_x_float": _number(pipeline.get("slew_limited_counts_x_float")),
        "slew_limited_counts_y_float": _number(pipeline.get("slew_limited_counts_y_float")),
        "feasible_counts_x_float": _number(pipeline.get("feasible_counts_x_float")),
        "feasible_counts_y_float": _number(pipeline.get("feasible_counts_y_float")),
        "residual_x_counts": _number(pipeline.get("residual_x_counts")),
        "residual_y_counts": _number(pipeline.get("residual_y_counts")),
        "budget_clamped_x": _bool(pipeline.get("budget_clamped_x")),
        "budget_clamped_y": _bool(pipeline.get("budget_clamped_y")),
        "planned_x_counts": _first_number(pipeline.get("final_dx"), control_payload.get("dx")),
        "planned_y_counts": _first_number(pipeline.get("final_dy"), control_payload.get("dy")),
        "executed_counts_last_20ms_x": _number(mouse.get("executed_counts_last_20ms_x")),
        "executed_counts_last_20ms_y": _number(mouse.get("executed_counts_last_20ms_y")),
        "executed_counts_last_40ms_x": _number(mouse.get("executed_counts_last_40ms_x")),
        "executed_counts_last_40ms_y": _number(mouse.get("executed_counts_last_40ms_y")),
        "executed_counts_last_60ms_x": _number(mouse.get("executed_counts_last_60ms_x")),
        "executed_counts_last_60ms_y": _number(mouse.get("executed_counts_last_60ms_y")),
        "driver_x_counts": _first_number(control_payload.get("driver_dx"), _driver_counts(control_payload).get("dx")),
        "driver_y_counts": _first_number(control_payload.get("driver_dy"), _driver_counts(control_payload).get("dy")),
        "device_send_start_ts_ns": _number(execution_payload.get("device_send_start_ts_ns")),
        "device_send_end_ts_ns": _number(execution_payload.get("device_send_end_ts_ns")),
        "scheduler_send_delay_us": _number(execution_payload.get("scheduler_send_delay_us")),
        "command_id": _number(scheduler.get("command_id")),
        "plan_id": _first_number(scheduler.get("plan_id"), scheduler.get("command_id")),
        "step_index": _number(scheduler.get("step_index")),
        "step_count": _number(scheduler.get("step_count")),
        "expires_ts_ns": _number(scheduler.get("expires_ts_ns")),
        "command_status": _text(scheduler.get("command_status") or ("sent" if execution_payload.get("sent") else "")),
        "cancel_reason": _text(scheduler.get("cancel_reason")),
        "cancelled_remaining_dx": _number(scheduler.get("cancelled_remaining_dx")),
        "cancelled_remaining_dy": _number(scheduler.get("cancelled_remaining_dy")),
        "global_state": _global_state(control_payload, execution_payload),
        "reason_code": _text(control_payload.get("reason") or control_payload.get("selection_reason") or execution_payload.get("message")),
        "config_version": _number(getattr(calibration, "profile_version", None)),
    }


def _scheduler_metadata(
    execution: Mapping[str, Any],
    scheduler_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = _dict(execution.get("metadata"))
    nested = _dict(metadata.get("scheduler"))
    if nested:
        result = dict(nested)
        nested_execution = _dict(nested.get("execution"))
        if nested_execution:
            result.update(nested_execution)
        return result
    direct_keys = {
        "command_id",
        "plan_id",
        "step_index",
        "step_count",
        "expires_ts_ns",
        "command_status",
        "cancel_reason",
    }
    if any(key in metadata for key in direct_keys):
        return dict(metadata)
    return dict(scheduler_status or {})


def _selected_track(track: Mapping[str, Any]) -> dict[str, Any]:
    selected_id = track.get("selected_track_id")
    tracks = track.get("tracks")
    if not isinstance(tracks, list):
        return {}
    for item in tracks:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("track_id")) == int(selected_id):
                return item
        except (TypeError, ValueError):
            continue
    return {}


def _same_frame_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not left or not right:
        return True
    left_frame = left.get("frame_id")
    right_frame = right.get("frame_id")
    if not isinstance(left_frame, (int, float)) or not isinstance(right_frame, (int, float)):
        return True
    if isinstance(left_frame, bool) or isinstance(right_frame, bool):
        return True
    return int(left_frame) == int(right_frame)


def _driver_counts(control: Mapping[str, Any]) -> dict[str, Any]:
    return _dict(control.get("driver_counts"))


def _global_state(control: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    explicit = control.get("global_state")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if execution.get("sent") is True:
        return "sent"
    if control.get("control_allowed") is False:
        return "control_blocked"
    if control.get("will_emit") is False:
        return "not_emitted"
    if execution:
        return "execution_blocked"
    return str(control.get("selector_state") or "unknown")


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        payload = _dict(value)
        if payload:
            return payload
    return {}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_number(*values: Any) -> Any:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, list):
            return len(value)
    return ""


def _number(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return ""


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _bool(value: Any) -> Any:
    return bool(value) if isinstance(value, bool) else ""


def _divide(numerator: Any, denominator: Any) -> Any:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return ""
    if isinstance(numerator, bool) or isinstance(denominator, bool) or abs(float(denominator)) < 1e-12:
        return ""
    return float(numerator) / float(denominator)


def _none_if_missing(left: Any, right: Any, fn: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not isinstance(left, bool) and not isinstance(right, bool):
            return fn(float(left), float(right))
    return ""


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _parquet_value(value: Any) -> str:
    return str(_csv_value(value))


__all__ = [
    "CONTROL_FRAME_FIELDS",
    "CONTROL_TRACE_FIELD_UNITS",
    "CONTROL_TRACE_SCHEMA_NAME",
    "CONTROL_TRACE_SCHEMA_VERSION",
    "ControlFrameCsvRecorder",
    "ControlFrameParquetRecorder",
    "ControlTraceJsonlRecorder",
    "build_control_frame_record",
    "build_control_trace_record",
    "serialize_control_trace",
]
