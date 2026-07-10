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
                "observed_error_x_px": 12.0,
                "observed_error_y_px": -4.0,
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
                "counts_x_float": 7.4,
                "counts_y_float": -2.1,
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
    assert trace["control"]["error_rad"] == {"x": 0.012, "y": -0.004}
    assert trace["control"]["error_rate_rad_s"] == {"x": 0.18, "y": -0.07}
    assert trace["control"]["p_rad"] == {"x": 0.0042, "y": -0.0014}
    assert trace["control"]["d_rad"] == {"x": 0.0005, "y": -0.0002}
    assert trace["control"]["prediction_horizon_ms"] == 24.0
    assert trace["counts"]["planned_counts"] == {"x": 5.0, "y": -2.0}
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
