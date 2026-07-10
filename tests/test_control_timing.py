from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from novasight.config import RuntimeConfig
from novasight.contracts import Detection, DetectionBatch
from novasight.runtime.aim import (
    AimPointState,
    EstimatedTargetState,
    LatencyCompensationConfig,
    LatencyCompensator,
)
from novasight.runtime.control_timing import ControlTimingModel
from novasight.runtime.service import RuntimeService
from novasight.runtime.telemetry import get_telemetry_summary


def test_control_timing_uses_capture_delta_and_separates_frame_age() -> None:
    timing = ControlTimingModel()

    first = timing.observe(
        frame_id=10,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=1_004_000_000,
        control_now_ts_ns=1_012_000_000,
        configured_extra_prediction_delay_ms=4.0,
    )
    second = timing.observe(
        frame_id=11,
        target_id=7,
        capture_ts_ns=1_020_000_000,
        inference_end_ts_ns=1_025_000_000,
        control_now_ts_ns=1_033_000_000,
        configured_extra_prediction_delay_ms=4.0,
    )

    assert first.measurement_dt_s is None
    assert second.measurement_dt_s == pytest.approx(0.020)
    assert second.frame_age_s == pytest.approx(0.013)
    assert second.prediction_horizon_s == pytest.approx(0.017)
    assert second.as_telemetry() == {
        "frame_id": 11,
        "target_id": 7,
        "capture_ts_ns": 1_020_000_000,
        "inference_end_ts_ns": 1_025_000_000,
        "control_now_ts_ns": 1_033_000_000,
        "measurement_dt_ms": pytest.approx(20.0),
        "frame_age_ms": pytest.approx(13.0),
        "configured_extra_prediction_delay_ms": pytest.approx(4.0),
        "extra_prediction_delay_source": "configured_estimate",
        "prediction_horizon_ms": pytest.approx(17.0),
    }


def test_control_timing_target_switch_starts_a_new_measurement_sequence() -> None:
    timing = ControlTimingModel()
    timing.observe(
        frame_id=10,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=1_004_000_000,
        control_now_ts_ns=1_012_000_000,
        configured_extra_prediction_delay_ms=4.0,
    )

    switched = timing.observe(
        frame_id=11,
        target_id=8,
        capture_ts_ns=1_020_000_000,
        inference_end_ts_ns=1_025_000_000,
        control_now_ts_ns=1_033_000_000,
        configured_extra_prediction_delay_ms=4.0,
    )

    assert switched.target_id == 8
    assert switched.measurement_dt_s is None


def test_detection_batch_observation_publishes_complete_control_timing(caplog) -> None:
    caplog.set_level("DEBUG", logger="novasight.runtime.service")
    config = RuntimeConfig()
    config.control.fov_ratio = 1.0
    config.control.tracker_confirm_frames = 1
    config.control.configured_extra_prediction_delay_ms = 4.0
    config.runtime.freshness_threshold_ms = 1_000.0
    config.control.latency_reject_if_age_exceeds_ms = 1_000.0
    service = RuntimeService(
        config,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _config: None,
            execute=lambda _intent: pytest.fail("observation timing must not emit control"),
        ),
    )
    base_ns = time.monotonic_ns() - 25_000_000

    for frame_id, capture_ts_ns in ((10, base_ns), (11, base_ns + 10_000_000)):
        service.process_detection_batch(
            DetectionBatch(
                frame_id=frame_id,
                generation=frame_id,
                capture_ts_ns=capture_ts_ns,
                inference_start_ts_ns=capture_ts_ns + 2_000_000,
                inference_end_ts_ns=capture_ts_ns + 4_000_000,
                detections=[
                    Detection(cls=0, score=0.9, x1=220, y1=180, x2=260, y2=300)
                ],
                classes=["target"],
                coordinate_space="roi",
            ),
            width=480,
            height=480,
        )

    timing = service.last_control_timing
    assert timing["frame_id"] == 11
    assert timing["target_id"] is not None
    assert timing["capture_ts_ns"] == base_ns + 10_000_000
    assert timing["inference_end_ts_ns"] == base_ns + 14_000_000
    assert timing["control_now_ts_ns"] >= timing["capture_ts_ns"]
    assert timing["measurement_dt_ms"] == pytest.approx(10.0)
    assert timing["frame_age_ms"] >= 0.0
    assert timing["configured_extra_prediction_delay_ms"] == pytest.approx(4.0)
    assert timing["extra_prediction_delay_source"] == "configured_estimate"
    assert timing["prediction_horizon_ms"] == pytest.approx(timing["frame_age_ms"] + 4.0)
    assert service.last_inference_status["measurement_dt_ms"] == pytest.approx(10.0)
    assert service.last_inference_status["prediction_horizon_ms"] == pytest.approx(
        timing["prediction_horizon_ms"]
    )
    assert "control_timing event=detection_batch frame=11" in caplog.text
    assert "control_timing event=target_observation frame=11" in caplog.text


def test_telemetry_summary_exposes_control_timing_snapshot() -> None:
    timing = {
        "frame_id": 11,
        "target_id": 7,
        "capture_ts_ns": 1_020_000_000,
        "inference_end_ts_ns": 1_025_000_000,
        "control_now_ts_ns": 1_033_000_000,
        "measurement_dt_ms": 20.0,
        "frame_age_ms": 13.0,
        "configured_extra_prediction_delay_ms": 4.0,
        "extra_prediction_delay_source": "configured_estimate",
        "prediction_horizon_ms": 17.0,
    }
    runtime = SimpleNamespace(
        last_control_timing=timing,
        last_control=None,
        last_target=None,
        last_execution=None,
        state=lambda: {"running": True, "statistics": {}},
    )

    summary = get_telemetry_summary(SimpleNamespace(runtime=runtime))

    assert summary["control_timing"] == timing


def test_telemetry_summary_keeps_rejected_batch_time_chain_visible() -> None:
    rejected_batch_timing = {
        "frame_id": 12,
        "target_id": None,
        "capture_ts_ns": 1_020_000_000,
        "inference_end_ts_ns": 1_025_000_000,
        "control_now_ts_ns": 1_100_000_000,
        "measurement_dt_ms": None,
        "frame_age_ms": 80.0,
        "configured_extra_prediction_delay_ms": 4.0,
        "extra_prediction_delay_source": "configured_estimate",
        "prediction_horizon_ms": 84.0,
    }
    runtime = SimpleNamespace(
        last_control_timing={},
        last_inference_status={**rejected_batch_timing, "stale_rejected": True},
        last_control=None,
        last_target=None,
        last_execution=None,
        state=lambda: {"running": True, "statistics": {}},
    )

    summary = get_telemetry_summary(SimpleNamespace(runtime=runtime))

    assert summary["control_timing"] == rejected_batch_timing


def test_latency_compensator_uses_extra_prediction_delay_as_additional_lead() -> None:
    aim = AimPointState(
        track_id=7,
        state_ts_ns=10_000_000,
        raw_x=100.0,
        raw_y=100.0,
        smoothed_x=100.0,
        smoothed_y=100.0,
        center_x=100.0,
        center_y=100.0,
        anchor_jump_norm=0.0,
        aim_confidence=1.0,
    )
    estimate = EstimatedTargetState(
        track_id=7,
        state_ts_ns=10_000_000,
        capture_ts_ns=8_000_000,
        x=100.0,
        y=100.0,
        vx=1_000.0,
        vy=0.0,
        valid=True,
        prediction_confidence=1.0,
        velocity_measurements=3,
    )

    result = LatencyCompensator().compensate(
        aim=aim,
        estimate=estimate,
        source_frame_id=11,
        compute_ts_ns=20_000_000,
        roi_offset_x=0.0,
        roi_offset_y=0.0,
        roi_width=640.0,
        roi_height=640.0,
        config=LatencyCompensationConfig(
            scale=1.0,
            max_compensation_ms=100.0,
            reject_if_age_exceeds_ms=100.0,
            max_compensation_px=100.0,
            min_velocity_px_s=0.0,
            max_velocity_px_s=2_000.0,
            min_velocity_measurements=2,
            min_velocity_confidence=0.0,
            extra_prediction_delay_ms=5.0,
        ),
    )

    assert result.compensation_ms == pytest.approx(15.0)
    assert result.delta_x == pytest.approx(15.0)
