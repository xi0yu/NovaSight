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
    "measurement_age_ms",
    "inference_latency_ms",
    "candidate_count",
    "selected_class_id",
    "selected_confidence",
    "selected_quality_score",
    "track_id",
    "track_state",
    "continuity_score",
    "identity_confidence",
    "missing_ms",
    "x",
    "y",
    "vx",
    "vy",
    "cov_trace",
    "position_sigma_px",
    "nis",
    "predicted",
    "prediction_confidence",
    "raw_aim_x",
    "raw_aim_y",
    "smoothed_aim_x",
    "smoothed_aim_y",
    "anchor_jump_norm",
    "comp_x",
    "comp_y",
    "delta_x",
    "delta_y",
    "velocity_confidence",
    "comp_applied",
    "control_width_px",
    "control_height_px",
    "fov_x_rad",
    "fov_y_rad",
    "focal_x_px",
    "focal_y_px",
    "error_x_px",
    "error_y_px",
    "error_x_rad",
    "error_y_rad",
    "dt_s",
    "zone",
    "kp_x_user",
    "kd_x_user",
    "kp_x_effective",
    "kd_x_effective",
    "derivative_x_raw",
    "derivative_x_ema",
    "u_x_rad",
    "u_y_rad",
    "counts_per_360_x",
    "counts_per_360_y",
    "raw_counts_x",
    "raw_counts_y",
    "residual_x_before",
    "residual_y_before",
    "residual_x_after",
    "residual_y_after",
    "final_output_x_counts",
    "final_output_y_counts",
    "driver_x_counts",
    "driver_y_counts",
    "command_id",
    "expires_ts_ns",
    "sent_x",
    "sent_y",
    "command_status",
    "cancel_reason",
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
    angular = _dict(pipeline.get("angular_controller"))
    aim = _first_dict(
        control_payload.get("aim_point"),
        target_payload.get("aim_point"),
        pipeline.get("aim_point"),
    )
    comp = _first_dict(
        control_payload.get("compensated_target"),
        target_payload.get("compensated_target"),
        pipeline.get("compensated_target"),
    )
    candidate = _dict(control_payload.get("candidate_filter"))
    track = _dict(control_payload.get("track_diagnostics"))
    selected_track = _selected_track(track)
    estimate = _first_dict(
        control_payload.get("estimated_target_state"),
        target_payload.get("estimated_target_state"),
        pipeline.get("estimated_target_state"),
        selected_track.get("estimate") if selected_track else None,
    )
    scheduler = _scheduler_metadata(execution_payload, scheduler_status)
    calibration = getattr(config, "calibration", None)
    control_config = getattr(config, "control", None)
    raw_counts_x = _number(pipeline.get("raw_dx_counts"))
    raw_counts_y = _number(pipeline.get("raw_dy_counts"))
    accum_x = _number(pipeline.get("accum_x_counts"))
    accum_y = _number(pipeline.get("accum_y_counts"))
    residual_x_after = _number(pipeline.get("residual_x_counts"))
    residual_y_after = _number(pipeline.get("residual_y_counts"))
    derivative_x_ema = _number(angular.get("derivative_x_rad_s"))
    error_x_rad = _number(pipeline.get("error_x_rad"))
    p_x = _number(pipeline.get("p_x"))
    d_x = _number(pipeline.get("d_x"))
    kp_x_user = _number(getattr(control_config, "experimental_angle_kp_x", None))
    kd_x_user = _number(getattr(control_config, "experimental_angle_kd", None))
    return {
        "frame_id": _first_number(control_payload.get("frame_id"), target_payload.get("frame_id"), inference_payload.get("frame_id")),
        "capture_ts_ns": _first_number(control_payload.get("capture_ts_ns"), target_payload.get("capture_ts_ns"), inference_payload.get("capture_ts_ns")),
        "measurement_age_ms": _first_number(comp.get("measurement_age_ms"), control_payload.get("frame_age_ms"), inference_payload.get("frame_age_ms")),
        "inference_latency_ms": _number(inference_payload.get("detection_batch_inference_latency_ms")),
        "candidate_count": _first_number(candidate.get("filtered_candidates"), control_payload.get("candidates"), inference_payload.get("mapped_detections")),
        "selected_class_id": _first_number(target_payload.get("class_id"), _dict(candidate.get("selected")).get("cls")),
        "selected_confidence": _first_number(target_payload.get("score"), _dict(candidate.get("selected")).get("score")),
        "selected_quality_score": _first_number(control_payload.get("quality_score"), target_payload.get("quality_score"), _dict(candidate.get("selected")).get("quality_score")),
        "track_id": _first_number(track.get("selected_track_id"), target_payload.get("track_id"), estimate.get("track_id")),
        "track_state": _text(track.get("selected_track_state") or track.get("selection_state")),
        "continuity_score": _number(track.get("selected_continuity_score")),
        "identity_confidence": _first_number(track.get("selected_identity_confidence"), selected_track.get("identity_confidence") if selected_track else None),
        "missing_ms": _first_number(track.get("selected_missing_ms"), selected_track.get("missing_ms") if selected_track else None),
        "x": _number(estimate.get("x")),
        "y": _number(estimate.get("y")),
        "vx": _number(estimate.get("vx")),
        "vy": _number(estimate.get("vy")),
        "cov_trace": _number(estimate.get("cov_trace")),
        "position_sigma_px": _number(estimate.get("position_sigma_px")),
        "nis": _first_number(estimate.get("nis"), track.get("selected_mahalanobis")),
        "predicted": _bool(estimate.get("predicted")),
        "prediction_confidence": _first_number(estimate.get("prediction_confidence"), comp.get("prediction_confidence")),
        "raw_aim_x": _first_number(aim.get("raw_x"), comp.get("raw_x"), pipeline.get("raw_aim_x")),
        "raw_aim_y": _first_number(aim.get("raw_y"), comp.get("raw_y"), pipeline.get("raw_aim_y")),
        "smoothed_aim_x": _first_number(aim.get("smoothed_x"), comp.get("smoothed_x"), pipeline.get("aim_x")),
        "smoothed_aim_y": _first_number(aim.get("smoothed_y"), comp.get("smoothed_y"), pipeline.get("aim_y")),
        "anchor_jump_norm": _number(aim.get("anchor_jump_norm")),
        "comp_x": _first_number(pipeline.get("comp_x"), comp.get("control_x")),
        "comp_y": _first_number(pipeline.get("comp_y"), comp.get("control_y")),
        "delta_x": _number(comp.get("delta_x")),
        "delta_y": _number(comp.get("delta_y")),
        "velocity_confidence": _number(comp.get("prediction_confidence")),
        "comp_applied": _bool(comp.get("applied")),
        "control_width_px": _number(pipeline.get("capture_width")),
        "control_height_px": _number(pipeline.get("capture_height")),
        "fov_x_rad": _number(pipeline.get("fov_x_rad")),
        "fov_y_rad": _number(pipeline.get("fov_y_rad")),
        "focal_x_px": _number(pipeline.get("focal_x")),
        "focal_y_px": _number(pipeline.get("focal_y")),
        "error_x_px": _number(pipeline.get("error_x_px")),
        "error_y_px": _number(pipeline.get("error_y_px")),
        "error_x_rad": error_x_rad,
        "error_y_rad": _number(pipeline.get("error_y_rad")),
        "dt_s": _first_number(pipeline.get("dt"), angular.get("dt")),
        "zone": _text(angular.get("zone")),
        "kp_x_user": kp_x_user,
        "kd_x_user": kd_x_user,
        "kp_x_effective": _divide(p_x, error_x_rad),
        "kd_x_effective": _divide(d_x, derivative_x_ema),
        "derivative_x_raw": _number(angular.get("derivative_x_raw_rad_s")),
        "derivative_x_ema": derivative_x_ema,
        "u_x_rad": _number(pipeline.get("out_x_rad")),
        "u_y_rad": _number(pipeline.get("out_y_rad")),
        "counts_per_360_x": _first_number(pipeline.get("counts_per_360_x"), getattr(calibration, "counts_per_360_x", None)),
        "counts_per_360_y": _first_number(pipeline.get("counts_per_360_y"), getattr(calibration, "counts_per_360_y", None)),
        "raw_counts_x": raw_counts_x,
        "raw_counts_y": raw_counts_y,
        "residual_x_before": _none_if_missing(accum_x, raw_counts_x, lambda a, b: a - b),
        "residual_y_before": _none_if_missing(accum_y, raw_counts_y, lambda a, b: a - b),
        "residual_x_after": residual_x_after,
        "residual_y_after": residual_y_after,
        "final_output_x_counts": _first_number(pipeline.get("final_dx"), control_payload.get("dx")),
        "final_output_y_counts": _first_number(pipeline.get("final_dy"), control_payload.get("dy")),
        "driver_x_counts": _first_number(control_payload.get("driver_dx"), _driver_counts(control_payload).get("dx")),
        "driver_y_counts": _first_number(control_payload.get("driver_dy"), _driver_counts(control_payload).get("dy")),
        "command_id": _number(scheduler.get("command_id")),
        "expires_ts_ns": _number(scheduler.get("expires_ts_ns")),
        "sent_x": _number(execution_payload.get("output_dx")),
        "sent_y": _number(execution_payload.get("output_dy")),
        "command_status": _text(scheduler.get("command_status") or ("sent" if execution_payload.get("sent") else "")),
        "cancel_reason": _text(scheduler.get("cancel_reason")),
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
    direct_keys = {"command_id", "expires_ts_ns", "command_status", "cancel_reason"}
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
