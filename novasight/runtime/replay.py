from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import quantiles
from typing import Any

from .recorder import CONTROL_FRAME_FIELDS


@dataclass(frozen=True)
class ReplayInjection:
    fixed_latency_ms: float = 0.0
    miss_frame_ids: frozenset[int] = field(default_factory=frozenset)
    lost_ranges: tuple[tuple[int, int], ...] = ()
    track_switches: dict[int, int] = field(default_factory=dict)
    device_fail_frame_ids: frozenset[int] = field(default_factory=frozenset)
    stale_after_ms: float | None = None


@dataclass(frozen=True)
class ControlFrameReplayEvent:
    index: int
    frame_id: int | None
    original_capture_ts_ns: int
    replay_offset_ns: int
    delta_ns: int
    record: dict[str, Any]
    injections: tuple[str, ...] = ()
    stale: bool = False


@dataclass(frozen=True)
class ReplayMetrics:
    frame_count: int
    duration_ms: float
    sent_count: int
    predicted_count: int
    missing_count: int
    switch_count: int
    device_failure_count: int
    stale_count: int
    timestamp_error_count: int
    p95_abs_error_x_rad: float
    p95_abs_error_y_rad: float
    p95_measurement_age_ms: float
    p95_inference_latency_ms: float
    overshoot_count: int
    max_final_counts: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "frame_count": self.frame_count,
            "duration_ms": self.duration_ms,
            "sent_count": self.sent_count,
            "predicted_count": self.predicted_count,
            "missing_count": self.missing_count,
            "switch_count": self.switch_count,
            "device_failure_count": self.device_failure_count,
            "stale_count": self.stale_count,
            "timestamp_error_count": self.timestamp_error_count,
            "p95_abs_error_x_rad": self.p95_abs_error_x_rad,
            "p95_abs_error_y_rad": self.p95_abs_error_y_rad,
            "p95_measurement_age_ms": self.p95_measurement_age_ms,
            "p95_inference_latency_ms": self.p95_inference_latency_ms,
            "overshoot_count": self.overshoot_count,
            "max_final_counts": self.max_final_counts,
        }


@dataclass(frozen=True)
class ReplayResult:
    events: tuple[ControlFrameReplayEvent, ...]
    metrics: ReplayMetrics
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayComparison:
    baseline: ReplayMetrics
    variant: ReplayMetrics
    delta: dict[str, float | int]


@dataclass(frozen=True)
class ReplayAcceptanceGate:
    max_p95_abs_error_x_rad: float | None = None
    max_p95_abs_error_y_rad: float | None = None
    max_p95_measurement_age_ms: float | None = None
    max_p95_inference_latency_ms: float | None = None
    max_overshoot_count: int | None = None
    max_timestamp_error_count: int = 0
    min_predicted_count: int = 0
    max_predicted_count: int | None = None
    min_missing_count: int = 0
    min_switch_count: int = 0
    min_device_failure_count: int = 0
    required_reason_codes: frozenset[str] = field(default_factory=frozenset)
    required_cancel_reasons: frozenset[str] = field(default_factory=frozenset)
    required_config_versions: frozenset[int] = field(default_factory=frozenset)
    required_control_shapes: frozenset[tuple[int, int]] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ReplayAcceptanceCase:
    name: str
    records: tuple[dict[str, Any], ...]
    gate: ReplayAcceptanceGate
    injection: ReplayInjection = field(default_factory=ReplayInjection)


@dataclass(frozen=True)
class ReplayAcceptanceReport:
    case_name: str
    result: ReplayResult
    gate: ReplayAcceptanceGate
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class ReplayAcceptanceSuiteReport:
    reports: tuple[ReplayAcceptanceReport, ...]

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cases": {
                report.case_name: {
                    "passed": report.passed,
                    "failures": list(report.failures),
                    "metrics": report.result.metrics.as_dict(),
                }
                for report in self.reports
            },
        }


class ControlFrameReplay:
    def load_csv(self, path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError("control-frame CSV has no header")
            missing = [field for field in CONTROL_FRAME_FIELDS if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"control-frame CSV missing required field(s): {', '.join(missing)}")
            return [_normalize_record(row) for row in reader]

    def run(
        self,
        records: list[dict[str, Any]],
        *,
        injection: ReplayInjection | None = None,
    ) -> ReplayResult:
        injection = injection or ReplayInjection()
        events: list[ControlFrameReplayEvent] = []
        violations: list[str] = []
        first_ts: int | None = None
        previous_ts: int | None = None

        for index, source_record in enumerate(records):
            record = _normalize_record(source_record)
            capture_ts_ns = _required_int(record, "capture_ts_ns", index=index)
            frame_id = _optional_int(record.get("frame_id"))
            if first_ts is None:
                first_ts = capture_ts_ns
            if previous_ts is not None and capture_ts_ns < previous_ts:
                violations.append(f"timestamp_decreased:index={index}")
            delta_ns = 0 if previous_ts is None else capture_ts_ns - previous_ts
            previous_ts = capture_ts_ns

            injections = _apply_injections(record, injection)
            measurement_age = _optional_float(record.get("frame_age_ms"))
            stale = False
            if injection.stale_after_ms is not None and measurement_age is not None:
                stale = measurement_age > float(injection.stale_after_ms)
                if stale and "stale_measurement" not in injections:
                    injections.append("stale_measurement")
                    if _text(record.get("global_state")) != "cooldown":
                        record["global_state"] = "control_blocked"
                    record["reason_code"] = "STALE_MEASUREMENT"

            events.append(
                ControlFrameReplayEvent(
                    index=index,
                    frame_id=frame_id,
                    original_capture_ts_ns=capture_ts_ns,
                    replay_offset_ns=capture_ts_ns - first_ts,
                    delta_ns=delta_ns,
                    record=record,
                    injections=tuple(injections),
                    stale=stale,
                )
            )

        metrics = _build_metrics(events, timestamp_error_count=len(violations))
        return ReplayResult(events=tuple(events), metrics=metrics, violations=tuple(violations))


def compare_replay_metrics(baseline: ReplayMetrics, variant: ReplayMetrics) -> ReplayComparison:
    baseline_dict = baseline.as_dict()
    variant_dict = variant.as_dict()
    delta: dict[str, float | int] = {}
    for key, baseline_value in baseline_dict.items():
        variant_value = variant_dict[key]
        if isinstance(baseline_value, int) and isinstance(variant_value, int):
            delta[key] = variant_value - baseline_value
        else:
            delta[key] = float(variant_value) - float(baseline_value)
    return ReplayComparison(baseline=baseline, variant=variant, delta=delta)


def build_default_replay_acceptance_cases() -> tuple[ReplayAcceptanceCase, ...]:
    return (
        ReplayAcceptanceCase(
            name="static_target",
            records=tuple(
                _acceptance_record(index, error_x_rad=error)
                for index, error in enumerate((0.030, 0.020, 0.012, 0.006, 0.002), start=1)
            ),
            gate=ReplayAcceptanceGate(
                max_p95_abs_error_x_rad=0.028,
                max_overshoot_count=0,
            ),
        ),
        ReplayAcceptanceCase(
            name="constant_velocity",
            records=tuple(
                _acceptance_record(
                    index,
                    error_x_rad=error,
                    vx=240.0,
                    prediction_confidence=0.8,
                )
                for index, error in enumerate((0.040, 0.032, 0.024, 0.016, 0.008), start=1)
            ),
            gate=ReplayAcceptanceGate(
                max_p95_abs_error_x_rad=0.039,
                max_overshoot_count=0,
            ),
        ),
        ReplayAcceptanceCase(
            name="sudden_turn",
            records=tuple(
                _acceptance_record(index, error_x_rad=error, vx=vx)
                for index, (error, vx) in enumerate(
                    ((0.035, 180.0), (0.018, 120.0), (-0.012, -220.0), (-0.006, -140.0), (-0.002, -60.0)),
                    start=1,
                )
            ),
            gate=ReplayAcceptanceGate(
                max_p95_abs_error_x_rad=0.032,
                max_overshoot_count=1,
            ),
        ),
        ReplayAcceptanceCase(
            name="single_frame_miss",
            records=tuple(
                _acceptance_record(index, error_x_rad=error)
                for index, error in enumerate((0.020, 0.014, 0.010, 0.006), start=1)
            ),
            injection=ReplayInjection(miss_frame_ids=frozenset({2})),
            gate=ReplayAcceptanceGate(
                max_predicted_count=0,
                min_missing_count=1,
                required_reason_codes=frozenset({"SINGLE_FRAME_MISS_INJECTED"}),
            ),
        ),
        ReplayAcceptanceCase(
            name="long_lost",
            records=tuple(
                _acceptance_record(index, error_x_rad=error)
                for index, error in enumerate((0.025, 0.020, 0.018, 0.016, 0.014), start=1)
            ),
            injection=ReplayInjection(lost_ranges=((3, 5),)),
            gate=ReplayAcceptanceGate(
                min_missing_count=3,
                required_cancel_reasons=frozenset({"TARGET_UNAVAILABLE"}),
                required_reason_codes=frozenset({"LONG_LOST_INJECTED"}),
            ),
        ),
        ReplayAcceptanceCase(
            name="track_switch",
            records=tuple(
                _acceptance_record(index, error_x_rad=error, track_id=1)
                for index, error in enumerate((0.018, 0.012, 0.010, 0.008), start=1)
            ),
            injection=ReplayInjection(track_switches={3: 22}),
            gate=ReplayAcceptanceGate(
                min_switch_count=1,
                required_reason_codes=frozenset({"TRACK_SWITCH_INJECTED"}),
            ),
        ),
        ReplayAcceptanceCase(
            name="device_failure",
            records=tuple(
                _acceptance_record(index, error_x_rad=error)
                for index, error in enumerate((0.018, 0.012, 0.010), start=1)
            ),
            injection=ReplayInjection(device_fail_frame_ids=frozenset({2})),
            gate=ReplayAcceptanceGate(
                min_device_failure_count=1,
                required_cancel_reasons=frozenset({"DEVICE_ERROR"}),
                required_reason_codes=frozenset({"DEVICE_FAILURE_INJECTED"}),
            ),
        ),
        ReplayAcceptanceCase(
            name="aspect_ratio_switch",
            records=(
                _acceptance_record(1, error_x_rad=0.018, control_width=640, control_height=640),
                _acceptance_record(2, error_x_rad=0.014, control_width=1920, control_height=1080),
                _acceptance_record(3, error_x_rad=0.010, control_width=1920, control_height=1080),
            ),
            gate=ReplayAcceptanceGate(
                required_control_shapes=frozenset({(640, 640), (1920, 1080)}),
                max_p95_abs_error_x_rad=0.018,
            ),
        ),
        ReplayAcceptanceCase(
            name="calibration_change",
            records=(
                _acceptance_record(1, error_x_rad=0.018, config_version=1),
                _acceptance_record(
                    2,
                    error_x_rad=0.014,
                    config_version=2,
                    command_status="canceled",
                    cancel_reason="CALIBRATION_PROFILE_CHANGED",
                    reason_code="CALIBRATION_PROFILE_CHANGED",
                ),
                _acceptance_record(3, error_x_rad=0.010, config_version=2),
            ),
            gate=ReplayAcceptanceGate(
                required_config_versions=frozenset({1, 2}),
                required_cancel_reasons=frozenset({"CALIBRATION_PROFILE_CHANGED"}),
                required_reason_codes=frozenset({"CALIBRATION_PROFILE_CHANGED"}),
            ),
        ),
    )


def run_replay_acceptance(
    cases: tuple[ReplayAcceptanceCase, ...] | None = None,
) -> ReplayAcceptanceSuiteReport:
    replay = ControlFrameReplay()
    reports: list[ReplayAcceptanceReport] = []
    for case in cases or build_default_replay_acceptance_cases():
        result = replay.run([dict(record) for record in case.records], injection=case.injection)
        failures = _evaluate_gate(result, case.gate)
        reports.append(
            ReplayAcceptanceReport(
                case_name=case.name,
                result=result,
                gate=case.gate,
                failures=tuple(failures),
            )
        )
    return ReplayAcceptanceSuiteReport(reports=tuple(reports))


def _evaluate_gate(result: ReplayResult, gate: ReplayAcceptanceGate) -> list[str]:
    failures: list[str] = []
    metrics = result.metrics
    _check_max(
        failures,
        "p95_abs_error_x_rad",
        metrics.p95_abs_error_x_rad,
        gate.max_p95_abs_error_x_rad,
    )
    _check_max(
        failures,
        "p95_abs_error_y_rad",
        metrics.p95_abs_error_y_rad,
        gate.max_p95_abs_error_y_rad,
    )
    _check_max(
        failures,
        "p95_measurement_age_ms",
        metrics.p95_measurement_age_ms,
        gate.max_p95_measurement_age_ms,
    )
    _check_max(
        failures,
        "p95_inference_latency_ms",
        metrics.p95_inference_latency_ms,
        gate.max_p95_inference_latency_ms,
    )
    _check_max(failures, "overshoot_count", metrics.overshoot_count, gate.max_overshoot_count)
    if metrics.timestamp_error_count > gate.max_timestamp_error_count:
        failures.append(
            f"timestamp_error_count={metrics.timestamp_error_count} > {gate.max_timestamp_error_count}"
        )
    if metrics.predicted_count < gate.min_predicted_count:
        failures.append(f"predicted_count={metrics.predicted_count} < {gate.min_predicted_count}")
    if gate.max_predicted_count is not None and metrics.predicted_count > gate.max_predicted_count:
        failures.append(f"predicted_count={metrics.predicted_count} > {gate.max_predicted_count}")
    if metrics.missing_count < gate.min_missing_count:
        failures.append(f"missing_count={metrics.missing_count} < {gate.min_missing_count}")
    if metrics.switch_count < gate.min_switch_count:
        failures.append(f"switch_count={metrics.switch_count} < {gate.min_switch_count}")
    if metrics.device_failure_count < gate.min_device_failure_count:
        failures.append(
            f"device_failure_count={metrics.device_failure_count} < {gate.min_device_failure_count}"
        )

    reason_codes = {_text(event.record.get("reason_code")) for event in result.events}
    cancel_reasons = {_text(event.record.get("cancel_reason")) for event in result.events}
    config_versions = {
        version
        for version in (_optional_int(event.record.get("config_version")) for event in result.events)
        if version is not None
    }
    control_shapes = {
        (width, height)
        for width, height in (
            (
                _optional_int(event.record.get("control_width_px")),
                _optional_int(event.record.get("control_height_px")),
            )
            for event in result.events
        )
        if width is not None and height is not None
    }
    for required in sorted(gate.required_reason_codes):
        if required not in reason_codes:
            failures.append(f"missing_reason_code={required}")
    for required in sorted(gate.required_cancel_reasons):
        if required not in cancel_reasons:
            failures.append(f"missing_cancel_reason={required}")
    for required in sorted(gate.required_config_versions):
        if required not in config_versions:
            failures.append(f"missing_config_version={required}")
    for required in sorted(gate.required_control_shapes):
        if required not in control_shapes:
            failures.append(f"missing_control_shape={required[0]}x{required[1]}")
    if result.violations:
        failures.extend(result.violations)
    return failures


def _check_max(
    failures: list[str],
    name: str,
    value: float | int,
    maximum: float | int | None,
) -> None:
    if maximum is not None and value > maximum:
        failures.append(f"{name}={value} > {maximum}")


def _acceptance_record(
    frame_id: int,
    *,
    error_x_rad: float,
    error_y_rad: float = 0.0,
    track_id: int = 1,
    vx: float = 0.0,
    vy: float = 0.0,
    prediction_confidence: float = 0.9,
    control_width: int = 640,
    control_height: int = 640,
    config_version: int = 1,
    command_status: str = "sent",
    cancel_reason: str = "",
    reason_code: str = "",
) -> dict[str, Any]:
    record = {field: "" for field in CONTROL_FRAME_FIELDS}
    count_x = round(error_x_rad * 9980.0 / 6.283185307179586)
    count_y = round(error_y_rad * 9980.0 / 6.283185307179586)
    capture_ts_ns = 1_000_000_000 + (frame_id - 1) * 10_000_000
    record.update(
        {
            "frame_id": frame_id,
            "capture_ts_ns": capture_ts_ns,
            "control_now_ts_ns": capture_ts_ns + 8_000_000,
            "measurement_dt_ms": 10.0,
            "frame_age_ms": 8.0,
            "prediction_horizon_ms": 12.0,
            "inference_latency_ms": 3.0,
            "control_width_px": control_width,
            "control_height_px": control_height,
            "candidate_count": 1,
            "track_id": track_id,
            "track_state": "ACTIVE",
            "control_mode": "calibrated_angular",
            "target_confidence": 0.92,
            "kalman_x_px": control_width * 0.5 + error_x_rad * 1000.0,
            "kalman_y_px": control_height * 0.5 + error_y_rad * 1000.0,
            "kalman_vx_px_s": vx,
            "kalman_vy_px_s": vy,
            "observed_aim_x_px": control_width * 0.5 + error_x_rad * 1000.0,
            "observed_aim_y_px": control_height * 0.5 + error_y_rad * 1000.0,
            "predicted_aim_x_px": control_width * 0.5 + error_x_rad * 1000.0 + vx * 0.012,
            "predicted_aim_y_px": control_height * 0.5 + error_y_rad * 1000.0 + vy * 0.012,
            "base_prediction_confidence": prediction_confidence,
            "velocity_confidence": prediction_confidence,
            "prediction_confidence": prediction_confidence,
            "prediction_applied": abs(vx) > 0 or abs(vy) > 0,
            "observed_error_x_px": error_x_rad * 1000.0,
            "observed_error_y_px": error_y_rad * 1000.0,
            "predicted_error_x_px": error_x_rad * 1000.0 + vx * 0.012,
            "predicted_error_y_px": error_y_rad * 1000.0 + vy * 0.012,
            "observed_error_x_rad": error_x_rad,
            "observed_error_y_rad": error_y_rad,
            "predicted_error_x_rad": error_x_rad + vx * 0.000012,
            "predicted_error_y_rad": error_y_rad + vy * 0.000012,
            "d_raw_x_rad_s": 0.0,
            "d_raw_y_rad_s": 0.0,
            "d_ema_x_rad_s": 0.0,
            "d_ema_y_rad_s": 0.0,
            "requested_output_x_rad": error_x_rad * 0.35,
            "requested_output_y_rad": error_y_rad * 0.35,
            "limited_output_x_rad": error_x_rad * 0.35,
            "limited_output_y_rad": error_y_rad * 0.35,
            "theoretical_counts_x_float": error_x_rad * 9980.0 / 6.283185307179586,
            "theoretical_counts_y_float": error_y_rad * 9980.0 / 6.283185307179586,
            "mode_limited_counts_x_float": float(count_x),
            "mode_limited_counts_y_float": float(count_y),
            "deadzone_limited_counts_x_float": float(count_x),
            "deadzone_limited_counts_y_float": float(count_y),
            "slew_limited_counts_x_float": float(count_x),
            "slew_limited_counts_y_float": float(count_y),
            "feasible_counts_x_float": float(count_x),
            "feasible_counts_y_float": float(count_y),
            "residual_x_counts": 0.0,
            "residual_y_counts": 0.0,
            "budget_clamped_x": False,
            "budget_clamped_y": False,
            "planned_x_counts": count_x,
            "planned_y_counts": count_y,
            "driver_x_counts": count_x if command_status == "sent" else "",
            "driver_y_counts": count_y if command_status == "sent" else "",
            "device_send_start_ts_ns": capture_ts_ns + 10_000_000,
            "device_send_end_ts_ns": capture_ts_ns + 10_100_000,
            "command_id": frame_id,
            "expires_ts_ns": capture_ts_ns + 30_000_000,
            "command_status": command_status,
            "cancel_reason": cancel_reason,
            "global_state": "sent" if command_status == "sent" else "control_blocked",
            "reason_code": reason_code,
            "config_version": config_version,
        }
    )
    return record


def _apply_injections(record: dict[str, Any], injection: ReplayInjection) -> list[str]:
    frame_id = _optional_int(record.get("frame_id"))
    injections: list[str] = []
    if injection.fixed_latency_ms:
        current = _optional_float(record.get("frame_age_ms")) or 0.0
        record["frame_age_ms"] = current + float(injection.fixed_latency_ms)
        injections.append("fixed_latency")
    if frame_id is not None and frame_id in injection.miss_frame_ids:
        _apply_single_miss(record)
        injections.append("single_frame_miss")
    if frame_id is not None and _in_lost_range(frame_id, injection.lost_ranges):
        _apply_long_lost(record)
        injections.append("long_lost")
    if frame_id is not None and frame_id in injection.track_switches:
        record["track_id"] = injection.track_switches[frame_id]
        record["track_state"] = "SWITCH_COMMITTED"
        record["reason_code"] = "TRACK_SWITCH_INJECTED"
        injections.append("track_switch")
    if frame_id is not None and frame_id in injection.device_fail_frame_ids:
        _apply_device_failure(record)
        injections.append("device_failure")
    return injections


def _apply_single_miss(record: dict[str, Any]) -> None:
    _apply_long_lost(record)
    record["reason_code"] = "SINGLE_FRAME_MISS_INJECTED"


def _apply_long_lost(record: dict[str, Any]) -> None:
    for field in (
        "candidate_count",
        "target_confidence",
        "observed_aim_x_px",
        "observed_aim_y_px",
        "predicted_aim_x_px",
        "predicted_aim_y_px",
        "planned_x_counts",
        "planned_y_counts",
        "driver_x_counts",
        "driver_y_counts",
    ):
        record[field] = "" if field != "candidate_count" else 0
    record["track_state"] = "LOST"
    record["prediction_applied"] = False
    record["command_status"] = "canceled"
    record["cancel_reason"] = "TARGET_UNAVAILABLE"
    record["global_state"] = "control_blocked"
    record["reason_code"] = "LONG_LOST_INJECTED"


def _apply_device_failure(record: dict[str, Any]) -> None:
    record["command_status"] = "device_error"
    record["cancel_reason"] = "DEVICE_ERROR"
    record["global_state"] = "cooldown"
    record["driver_x_counts"] = ""
    record["driver_y_counts"] = ""
    record["reason_code"] = "DEVICE_FAILURE_INJECTED"


def _build_metrics(
    events: list[ControlFrameReplayEvent],
    *,
    timestamp_error_count: int,
) -> ReplayMetrics:
    records = [event.record for event in events]
    error_x = [abs(value) for value in _numbers(records, "observed_error_x_rad")]
    error_y = [abs(value) for value in _numbers(records, "observed_error_y_rad")]
    measurement_age = _numbers(records, "frame_age_ms")
    inference_latency = _numbers(records, "inference_latency_ms")
    final_counts = [
        max(abs(x), abs(y))
        for x, y in zip(
            _numbers(records, "planned_x_counts"),
            _numbers(records, "planned_y_counts"),
            strict=False,
        )
    ]
    duration_ms = 0.0
    if events:
        duration_ms = max(0.0, events[-1].replay_offset_ns / 1e6)
    return ReplayMetrics(
        frame_count=len(events),
        duration_ms=duration_ms,
        sent_count=sum(1 for record in records if _text(record.get("command_status")) == "sent"),
        predicted_count=sum(1 for record in records if _truthy(record.get("prediction_applied"))),
        missing_count=sum(1 for record in records if _is_missing_record(record)),
        switch_count=sum(1 for record in records if _text(record.get("track_state")) == "SWITCH_COMMITTED"),
        device_failure_count=sum(1 for record in records if _text(record.get("command_status")) == "device_error"),
        stale_count=sum(1 for event in events if event.stale),
        timestamp_error_count=timestamp_error_count,
        p95_abs_error_x_rad=_p95(error_x),
        p95_abs_error_y_rad=_p95(error_y),
        p95_measurement_age_ms=_p95(measurement_age),
        p95_inference_latency_ms=_p95(inference_latency),
        overshoot_count=_overshoot_count(_numbers(records, "observed_error_x_rad"))
        + _overshoot_count(_numbers(records, "observed_error_y_rad")),
        max_final_counts=max(final_counts) if final_counts else 0.0,
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field, "") for field in CONTROL_FRAME_FIELDS}


def _required_int(record: dict[str, Any], field: str, *, index: int) -> int:
    value = _optional_int(record.get(field))
    if value is None:
        raise ValueError(f"control-frame record index={index} missing integer {field}")
    return value


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _numbers(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _optional_float(record.get(field))
        if value is not None:
            values.append(value)
    return values


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return float(values[0])
    return float(quantiles(values, n=100, method="inclusive")[94])


def _overshoot_count(values: list[float]) -> int:
    count = 0
    previous_sign = 0
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign and previous_sign and sign != previous_sign:
            count += 1
        if sign:
            previous_sign = sign
    return count


def _in_lost_range(frame_id: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= frame_id <= end for start, end in ranges)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_missing_record(record: dict[str, Any]) -> bool:
    state = _text(record.get("track_state"))
    if state == "LOST":
        return True
    candidate_count = _optional_int(record.get("candidate_count"))
    return candidate_count == 0
