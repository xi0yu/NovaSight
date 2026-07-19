from __future__ import annotations

import json
from types import SimpleNamespace

from novasight.config import RuntimeConfig
from novasight.runtime import RuntimeService
from novasight.runtime.control_trace import (
    CONTROL_TRACE_SCHEMA_VERSION,
    ControlTraceJsonlRecorder,
    build_control_trace_record,
    serialize_control_trace,
)


def _trace_payload() -> dict:
    return build_control_trace_record(
        control={
            "frame_id": 42,
            "capture_ts_ns": 1_000_000_000,
            "control_now_ts_ns": 1_050_000_000,
            "trajectory_generation": 420,
            "selector_state": "tracking",
            "control_allowed": True,
            "will_emit": True,
            "dx": 5,
            "dy": -2,
            "mouse_observation": {
                "observed_x_px": 512.0,
                "observed_y_px": 496.0,
                "predicted_x_px": 515.0,
                "predicted_y_px": 495.0,
                "prediction_horizon_s": 0.024,
                "prediction_confidence": 0.75,
                "velocity_confidence": 0.8,
            },
            "pipeline": {
                "control_mode": "calibrated_angular",
                "observed_error_x_px": 12.0,
                "observed_error_y_px": -4.0,
                "predicted_error_x_px": 15.0,
                "predicted_error_y_px": -5.0,
                "observed_error_x_rad": 0.012,
                "observed_error_y_rad": -0.004,
                "predicted_error_x_rad": 0.015,
                "predicted_error_y_rad": -0.005,
                "focal_x_px": 1000.0,
                "focal_y_px": 1000.0,
                "p_x_rad": 0.0042,
                "p_y_rad": -0.0014,
                "d_x_rad": 0.0005,
                "d_y_rad": -0.0002,
                "limited_output_x_rad": 0.0047,
                "limited_output_y_rad": -0.0016,
                "theoretical_counts_x_float": 7.4,
                "theoretical_counts_y_float": -2.1,
                "mode_limited_counts_x_float": 7.0,
                "mode_limited_counts_y_float": -2.0,
                "deadzone_limited_counts_x_float": 7.0,
                "deadzone_limited_counts_y_float": -2.0,
                "slew_limited_counts_x_float": 5.4,
                "slew_limited_counts_y_float": -2.0,
                "feasible_counts_x_float": 5.4,
                "feasible_counts_y_float": -2.0,
                "residual_x_counts": 0.4,
                "residual_y_counts": -0.1,
                "final_dx": 5,
                "final_dy": -2,
                "d_ema_x_rad_s": 0.18,
                "d_ema_y_rad_s": -0.07,
                "d_raw_x_rad_s": 0.20,
                "d_raw_y_rad_s": -0.08,
            },
        },
        target={"frame_id": 42, "track_id": 7},
        inference={
            "generation": 420,
            "frame_id": 42,
            "source_sequence": 420,
            "capture_ts_ns": 1_000_000_000,
            "publish_ts_ns": 1_048_000_000,
            "clock_domain": "monotonic",
            "input_age_ms": 8.0,
            "inference_ms": 12.5,
            "result_age_ms": 48.0,
            "is_stale": False,
        },
        execution={
            "executor_id": "kmnet",
            "sent": True,
            "message": "sent",
            "output_dx": 5.0,
            "output_dy": -2.0,
            "metadata": {
                "device_send_start_ts_ns": 1_051_000_000,
                "device_send_end_ts_ns": 1_052_000_000,
                "device_send_clock_domain": "monotonic",
                "scheduler": {
                    "command_id": 11,
                    "trajectory_generation": 420,
                    "pending_dx": 0.0,
                    "pending_dy": 0.0,
                    "pending_steps": 0,
                    "pending_age_ms": 0.0,
                    "command_status": "ready",
                    "cancel_reason": "",
                    "created_ts_ns": 1_050_500_000,
                    "expires_ts_ns": 1_085_500_000,
                },
            },
        },
        scheduler_status={
            "pending_dx": 0.0,
            "pending_dy": 0.0,
            "pending_steps": 0,
            "pending_age_ms": 0.0,
        },
    )


def test_control_trace_prefers_explicit_no_send_reason() -> None:
    trace = build_control_trace_record(
        control={
            "will_emit": False,
            "no_send_reason": "CONTROL_OUTPUT_ZERO",
            "reason": "mouse_control",
        },
        target={},
        inference={},
        execution={},
    )

    assert trace["control"]["reason"] == "CONTROL_OUTPUT_ZERO"


def test_control_trace_preserves_dual_phase_decision_and_direct_delivery() -> None:
    trace = build_control_trace_record(
        control={
            "frame_id": 8,
            "trajectory_generation": 8,
            "capture_ts_ns": 1_000_000_000,
            "control_now_ts_ns": 1_008_000_000,
            "dx": 12,
            "dy": 1,
            "pipeline": {
                "algorithm": "dual_phase_atan_robust_predictive_v2",
                "class_id": 1,
                "active_class_profile": "default",
                "effective_aim_role": "head",
                "effective_aim_y_ratio": 0.35,
                "mode": "far",
                "measurement_dt_ms": 8.3,
                "aim_x": 400.0,
                "aim_y": 320.0,
                "error_real_x": 40.0,
                "error_real_y": 0.0,
                "error_control_x": 42.0,
                "error_control_y": 0.0,
                "estimated_velocity_x": 240.0,
                "innovation_x": 0.8,
                "normalized_innovation_x": 0.2,
                "motion_confidence": 0.9,
                "prediction_horizon_ms": 13.0,
                "prediction_raw_offset_x": 3.12,
                "prediction_weight": 0.27,
                "prediction_allowed_cap_x": 8.0,
                "prediction_safe_offset_x": 2.0,
                "prediction_allowed": True,
                "prediction_crossing_limited": False,
                "full_error_counts_x": 70.0,
                "full_error_counts_y": 0.0,
                "float_demand_x": 12.4,
                "float_demand_y": 1.1,
                "integer_command_x": 12,
                "integer_command_y": 1,
                "quantizer_residual_x": 0.4,
                "quantizer_residual_y": 0.1,
                "overzero_detected_x": False,
                "overzero_detected_y": False,
                "will_emit": True,
                "block_reason": "",
                "delivery_mode": "single_command_per_observation",
                "scheduler_used": False,
            },
        },
        target={"track_id": 3},
        inference={"generation": 8, "frame_id": 8, "capture_ts_ns": 1_000_000_000},
        execution={"sent": True, "output_dx": 12, "output_dy": 1},
    )

    decision = trace["algorithm_decision"]
    assert decision["algorithm_id"] == "dual_phase_atan_robust_predictive_v2"
    assert decision["class_id"] == 1
    assert decision["class_profile"] == "default"
    assert decision["aim_role"] == "head"
    assert decision["effective_aim_y_ratio"] == 0.35
    assert decision["phase"] == "far"
    assert decision["error_real_px"] == {"x": 40.0, "y": 0.0}
    assert decision["error_control_px"] == {"x": 42.0, "y": 0.0}
    assert decision["prediction"]["safe_offset_x"] == 2.0
    assert decision["integer_command"] == {"x": 12.0, "y": 1.0}
    assert decision["delivery_mode"] == "single_command_per_observation"
    assert decision["scheduler_used"] is False
    assert trace["scheduler"]["used"] is False


def test_control_trace_records_robust_v2_velocity_in_px_per_ms() -> None:
    trace = build_control_trace_record(
        control={
            "frame_id": 9,
            "trajectory_generation": 9,
            "capture_ts_ns": 1_000_000_000,
            "pipeline": {
                "algorithm": "dual_phase_atan_robust_predictive_v2",
                "mode": "far",
                "error_meas_x": 40.0,
                "error_meas_y": 1.0,
                "error_ctrl_x": 42.0,
                "error_ctrl_y": 1.0,
                "error_real_x": 40.0,
                "error_real_y": 1.0,
                "error_control_x": 42.0,
                "error_control_y": 1.0,
                "velocity_unit": "px/ms",
                "history_position_count": 4,
                "velocity_1": 0.4,
                "velocity_2": 0.5,
                "velocity_3": 0.6,
                "median_velocity": 0.5,
                "filtered_velocity": 0.45,
                "velocity_spread": 0.1,
                "motion_confidence": 0.8,
                "reference_dt_ms": 8.33,
                "prediction_lead_frames": 1.5,
                "prediction_raw_offset_x": 5.0,
                "prediction_weighted_offset_x": 6.0,
                "prediction_safe_offset_x": 2.0,
                "recoil_enabled": True,
                "recoil_active": True,
                "recoil_mode": "independent_target_relative_rate",
                "recoil_state": "ACTIVE",
                "recoil_base_rate_counts_s": 250.0,
                "recoil_fast_add_rate_counts_s": 25.0,
                "recoil_position_gate": 1.0,
                "recoil_final_rate_counts_s": 275.0,
                "recoil_requested_counts_y": 1.1,
                "recoil_emitted_counts_y": 1,
                "recoil_residual_counts_y": 0.25,
                "recoil_error_y_norm": 0.1,
                "recoil_observation_age_ms": 7.0,
                "recoil_source_generation": 9,
                "recoil_block_reason": "",
                "executor_success": True,
            },
        },
        target={"track_id": 3},
        inference={"generation": 9, "frame_id": 9, "capture_ts_ns": 1_000_000_000},
        execution={"sent": True, "output_dx": 12, "output_dy": 1},
    )

    decision = trace["algorithm_decision"]
    assert decision["error_measured_px"] == {"x": 40.0, "y": 1.0}
    assert decision["estimator"]["velocity_x_px_s"] is None
    assert decision["robust_velocity"]["unit"] == "px/ms"
    assert decision["robust_velocity"]["segments_px_ms"] == [0.4, 0.5, 0.6]
    assert decision["robust_velocity"]["median_px_ms"] == 0.5
    assert decision["prediction"]["reference_dt_ms"] == 8.33
    assert decision["prediction"]["lead_frames"] == 1.5
    assert decision["prediction"]["weighted_offset_x"] == 6.0
    assert decision["recoil"]["active"] is True
    assert decision["recoil"]["mode"] == "independent_target_relative_rate"
    assert decision["recoil"]["state"] == "ACTIVE"
    assert decision["recoil"]["final_rate_counts_s"] == 275.0
    assert decision["recoil"]["emitted_counts_y"] == 1.0
    assert decision["recoil"]["residual_counts_y"] == 0.25
    assert decision["recoil"]["error_y_norm"] == 0.1
    assert decision["executor_success"] is True


def test_control_trace_schema_serializes_units_and_correlation() -> None:
    trace = _trace_payload()

    assert trace["schema"]["version"] == CONTROL_TRACE_SCHEMA_VERSION
    assert trace["field_units"]["control.error_rad"] == "rad"
    assert trace["field_units"]["counts.sent"] == "counts"
    assert trace["correlation"] == {
        "id": "control:420:42:1000000000",
        "detection_generation": 420,
        "frame_id": 42,
        "capture_ts": {
            "value": 1_000_000_000,
            "clock_domain": "monotonic",
            "unit": "ns",
        },
        "trajectory_generation": 420,
        "command_id": 11,
    }
    assert trace["detection"]["capture_ts"] == {
        "value": 1_000_000_000,
        "clock_domain": "monotonic",
        "unit": "ns",
    }
    assert trace["control"]["control_now"]["clock_domain"] == "monotonic"
    assert trace["control"]["error_px"] == {"x": 12.0, "y": -4.0}
    assert trace["control"]["predicted_error_px"] == {"x": 15.0, "y": -5.0}
    assert trace["control"]["mode"] == "calibrated_angular"
    assert trace["control"]["error_rad"] == {"x": 0.012, "y": -0.004}
    assert trace["control"]["error_rate_rad_s"] == {"x": 0.18, "y": -0.07}
    assert trace["control"]["p_rad"] == {"x": 0.0042, "y": -0.0014}
    assert trace["control"]["d_rad"] == {"x": 0.0005, "y": -0.0002}
    assert trace["control"]["prediction_horizon_ms"] == 24.0
    assert trace["counts"]["planned_counts"] == {"x": 5.0, "y": -2.0}
    assert trace["counts"]["theoretical_counts"] == {"x": 7.4, "y": -2.1}
    assert trace["counts"]["slew_limited_counts"] == {"x": 5.4, "y": -2.0}
    assert trace["counts"]["feasible_counts"] == {"x": 5.4, "y": -2.0}
    assert trace["counts"]["queued_counts"] == {"x": 0.0, "y": 0.0}
    assert trace["counts"]["sent_counts"] == {"x": 5.0, "y": -2.0}
    assert trace["counts"]["estimated_applied_counts"]["status"] == "unknown"
    assert trace["counts"]["estimated_applied_counts"]["x"] is None
    assert trace["counts"]["unobserved_counts"]["status"] == "unknown"
    assert trace["scheduler"]["generation"] == 420
    assert trace["scheduler"]["pending_age_ms"] == 0.0
    assert trace["device"]["send_start_ts"]["value"] == 1_051_000_000
    assert trace["device"]["sent"] is True

    serialized = serialize_control_trace(trace)
    assert json.loads(serialized)["correlation"]["id"] == "control:420:42:1000000000"


def test_control_trace_jsonl_recorder_writes_one_json_record(tmp_path) -> None:
    path = tmp_path / "control_trace.jsonl"
    recorder = ControlTraceJsonlRecorder(path)

    recorder.record_control_trace(_trace_payload())
    recorder.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema"]["version"] == CONTROL_TRACE_SCHEMA_VERSION
    assert payload["correlation"]["id"] == "control:420:42:1000000000"


class _TraceOnlyRecorder:
    def __init__(self) -> None:
        self.traces: list[dict] = []

    def record_control_trace(self, record: dict) -> None:
        self.traces.append(record)


def test_runtime_records_control_trace_without_frame_recording_side_effect() -> None:
    config = RuntimeConfig()
    config.consumers.recording = True
    recorder = _TraceOnlyRecorder()
    service = RuntimeService(
        config,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            scheduler=SimpleNamespace(
                status=lambda: {
                    "pending_dx": 1.0,
                    "pending_dy": -1.0,
                    "pending_steps": 1,
                    "pending_age_ms": 3.0,
                    "pending_trajectory_generation": 420,
                }
            ),
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
        ),
        recorder=recorder,
    )
    service.last_control = _trace_payload()["control"] | {
        "frame_id": 42,
        "capture_ts_ns": 1_000_000_000,
        "control_now_ts_ns": 1_050_000_000,
        "trajectory_generation": 420,
        "dx": 5,
        "dy": -2,
        "will_emit": True,
    }
    service.last_target = {"frame_id": 42, "track_id": 7}
    service.last_inference_status = {
        "generation": 420,
        "frame_id": 42,
        "capture_ts_ns": 1_000_000_000,
        "clock_domain": "monotonic",
        "publish_ts_ns": 1_048_000_000,
    }
    service.last_execution = {
        "executor_id": "noop",
        "sent": False,
        "message": "not sent",
        "output_dx": 5.0,
        "output_dy": -2.0,
        "metadata": {},
    }

    service._record_control_frame()

    assert len(recorder.traces) == 1
    trace = recorder.traces[0]
    assert trace["correlation"]["id"] == "control:420:42:1000000000"
    assert trace["counts"]["sent_counts"] == {"x": None, "y": None}
    assert trace["scheduler"]["pending_age_ms"] == 3.0
