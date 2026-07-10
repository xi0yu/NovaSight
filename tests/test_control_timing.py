from __future__ import annotations

from types import SimpleNamespace

import pytest

from novasight.runtime.control_timing import ControlTimingModel
from novasight.runtime.telemetry import get_telemetry_summary


def test_control_timing_uses_capture_delta_and_separates_frame_age() -> None:
    timing = ControlTimingModel()

    first = timing.observe(
        frame_id=10,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=1_004_000_000,
        control_now_ts_ns=1_012_000_000,
        configured_actuation_delay_s=0.004,
    )
    second = timing.observe(
        frame_id=11,
        target_id=7,
        capture_ts_ns=1_020_000_000,
        inference_end_ts_ns=1_025_000_000,
        control_now_ts_ns=1_033_000_000,
        configured_actuation_delay_s=0.004,
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
        "configured_actuation_delay_s": pytest.approx(0.004),
        "actuation_delay_source": "configured_estimate",
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
        configured_actuation_delay_s=0.004,
    )

    switched = timing.observe(
        frame_id=11,
        target_id=8,
        capture_ts_ns=1_020_000_000,
        inference_end_ts_ns=1_025_000_000,
        control_now_ts_ns=1_033_000_000,
        configured_actuation_delay_s=0.004,
    )

    assert switched.target_id == 8
    assert switched.measurement_dt_s is None


def test_control_timing_does_not_advance_on_duplicate_capture_timestamp() -> None:
    timing = ControlTimingModel()
    timing.observe(
        frame_id=10,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=None,
        control_now_ts_ns=1_010_000_000,
        configured_actuation_delay_s=0.004,
    )

    duplicate = timing.observe(
        frame_id=11,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=None,
        control_now_ts_ns=1_012_000_000,
        configured_actuation_delay_s=0.004,
    )
    next_observation = timing.observe(
        frame_id=12,
        target_id=7,
        capture_ts_ns=1_020_000_000,
        inference_end_ts_ns=None,
        control_now_ts_ns=1_030_000_000,
        configured_actuation_delay_s=0.004,
    )

    assert duplicate.measurement_dt_s is None
    assert next_observation.measurement_dt_s == pytest.approx(0.020)


def test_control_timing_clamps_negative_age_and_delay() -> None:
    snapshot = ControlTimingModel().observe(
        frame_id=1,
        target_id=7,
        capture_ts_ns=1_000_000_000,
        inference_end_ts_ns=None,
        control_now_ts_ns=999_000_000,
        configured_actuation_delay_s=-0.010,
    )

    assert snapshot.frame_age_s == 0.0
    assert snapshot.configured_actuation_delay_s == 0.0
    assert snapshot.prediction_horizon_s == 0.0


def test_telemetry_summary_exposes_control_timing_snapshot() -> None:
    timing = {
        "frame_id": 11,
        "target_id": 7,
        "capture_ts_ns": 1_020_000_000,
        "inference_end_ts_ns": 1_025_000_000,
        "control_now_ts_ns": 1_033_000_000,
        "measurement_dt_ms": 20.0,
        "frame_age_ms": 13.0,
        "configured_actuation_delay_s": 0.004,
        "actuation_delay_source": "configured_estimate",
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
