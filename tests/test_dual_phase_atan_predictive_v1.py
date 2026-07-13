from __future__ import annotations

from dataclasses import fields, replace
import math

import pytest

from novasight.control.algorithms.dual_phase_atan_predictive_v1 import (
    ALGORITHM_ID,
    AtanPhaseConfig,
    ControlDecision,
    ControlMode,
    DualPhaseAtanPredictiveV1Algorithm,
    DualPhaseAtanPredictiveV1Config,
    DualPhaseAtanPredictiveV1Observation,
    EstimatorConfig,
    ModeSelectorConfig,
    PredictionConfig,
    ProjectionConfig,
)


def _observation(
    *,
    generation: int,
    error_x: float,
    error_y: float = 0.0,
    frame_id: int | None = None,
    target_id: int = 7,
    capture_ts_ns: int | None = None,
    inference_end_ts_ns: int | None = None,
    control_now_ns: int | None = None,
    trigger_active: bool = True,
    target_valid: bool = True,
    detection_confidence: float = 0.95,
    track_confidence: float = 0.90,
    bbox_height: float = 20.0,
) -> DualPhaseAtanPredictiveV1Observation:
    capture_ns = (
        capture_ts_ns if capture_ts_ns is not None else 1_000_000_000 + generation * 10_000_000
    )
    now_ns = control_now_ns if control_now_ns is not None else capture_ns + 8_000_000
    crosshair_x = 160.0
    crosshair_y = 160.0
    return DualPhaseAtanPredictiveV1Observation(
        generation=generation,
        frame_id=generation if frame_id is None else frame_id,
        target_id=target_id,
        capture_ts_ns=capture_ns,
        inference_end_ts_ns=(
            capture_ns + 4_000_000 if inference_end_ts_ns is None else inference_end_ts_ns
        ),
        control_now_ns=now_ns,
        aim_x=crosshair_x + error_x,
        aim_y=crosshair_y + error_y,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        bbox_x1=crosshair_x - 10.0,
        bbox_y1=crosshair_y - bbox_height * 0.22,
        bbox_x2=crosshair_x + 10.0,
        bbox_y2=crosshair_y + bbox_height * 0.78,
        observation_width=320,
        observation_height=320,
        roi_left=640,
        roi_top=220,
        roi_width=640,
        roi_height=640,
        source_width=1920,
        source_height=1080,
        detection_confidence=detection_confidence,
        track_confidence=track_confidence,
        trigger_active=trigger_active,
        target_valid=target_valid,
    )


def test_real_error_drives_two_phase_hysteresis() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())

    far = algorithm.calculate(_observation(generation=1, error_x=30.0))
    entered_near = algorithm.calculate(_observation(generation=2, error_x=10.0))
    held_near = algorithm.calculate(_observation(generation=3, error_x=15.0))
    returned_far = algorithm.calculate(_observation(generation=4, error_x=19.0))

    assert ALGORITHM_ID == "dual_phase_atan_predictive_v1"
    assert far.telemetry["mode"] == ControlMode.FAR.value
    assert entered_near.telemetry["mode"] == ControlMode.NEAR.value
    assert held_near.telemetry["mode"] == ControlMode.NEAR.value
    assert returned_far.telemetry["mode"] == ControlMode.FAR.value


def test_projection_and_atan_gain_use_frozen_kp_independent_of_capture_dt() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    config = replace(
        defaults,
        projection=ProjectionConfig(fov_x_deg=90.0, counts_per_360=360.0),
        mode=ModeSelectorConfig(
            near_enter_min_px=0.01,
            near_exit_min_px=0.02,
            near_enter_bbox_h_ratio=0.0,
            near_exit_bbox_h_ratio=0.0,
        ),
        far=AtanPhaseConfig(
            kp=0.25,
            atan_scale_counts=256.0,
            max_counts_per_update=140.0,
        ),
        prediction=replace(defaults.prediction, enabled_x=False),
    )
    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)

    first = algorithm.calculate(
        _observation(generation=1, error_x=30.0, capture_ts_ns=1_000_000_000)
    )
    second = algorithm.calculate(
        _observation(generation=2, error_x=30.0, capture_ts_ns=1_020_000_000)
    )

    source_error_x = 30.0 * 640.0 / 320.0
    focal_x = 1920.0 * 0.5 / math.tan(math.radians(90.0) * 0.5)
    full_counts_x = math.atan(source_error_x / focal_x) * 360.0 / math.tau
    expected_first_demand = 0.25 * 256.0 * math.atan(full_counts_x / 256.0)

    assert first.telemetry["measurement_dt_s"] == pytest.approx(0.0)
    assert first.telemetry["kp"] == pytest.approx(0.25)
    assert first.telemetry["full_error_counts_x"] == pytest.approx(full_counts_x)
    assert first.telemetry["float_demand_x"] == pytest.approx(expected_first_demand)
    assert second.telemetry["measurement_dt_s"] == pytest.approx(0.020)
    assert second.telemetry["kp"] == pytest.approx(0.25)
    assert second.telemetry["float_demand_x"] == pytest.approx(expected_first_demand)


def test_subcount_demand_has_no_deadzone_and_accumulates_with_truncation() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    phase = AtanPhaseConfig(
        kp=0.25,
        atan_scale_counts=256.0,
        max_counts_per_update=140.0,
    )
    config = replace(
        defaults,
        projection=ProjectionConfig(fov_x_deg=90.0, counts_per_360=36_000.0),
        mode=ModeSelectorConfig(
            near_enter_min_px=0.01,
            near_exit_min_px=0.02,
            near_enter_bbox_h_ratio=0.0,
            near_exit_bbox_h_ratio=0.0,
        ),
        far=phase,
        prediction=replace(defaults.prediction, enabled_x=False),
    )
    target_demand = 0.4
    full_counts = phase.atan_scale_counts * math.tan(
        target_demand / (phase.kp * phase.atan_scale_counts)
    )
    theta = full_counts * math.tau / config.projection.counts_per_360
    focal_x = 1920.0 * 0.5 / math.tan(math.radians(90.0) * 0.5)
    source_error = focal_x * math.tan(theta)
    observation_error = source_error * 320.0 / 640.0
    assert observation_error > 0.0

    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)
    decisions = [
        algorithm.calculate(_observation(generation=index, error_x=observation_error))
        for index in range(1, 5)
    ]

    assert [decision.dx for decision in decisions] == [0, 0, 1, 0]
    assert all(decision.telemetry["float_demand_x"] == pytest.approx(0.4) for decision in decisions)
    assert decisions[-1].telemetry["quantizer_residual_x"] == pytest.approx(0.6)
    assert decisions[2].emit_allowed is True
    assert decisions[0].emit_allowed is False


def test_variable_dt_estimator_adds_only_bounded_confidence_weighted_x_prediction() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    config = replace(
        defaults,
        estimator=EstimatorConfig(
            measurement_std_px=0.5,
            acceleration_std_px_s2=50.0,
            min_dt_s=0.003,
            reset_dt_s=0.080,
            innovation_soft_gate_sigma=3.0,
            innovation_hard_gate_sigma=8.0,
            hard_outlier_reset_count=2,
            max_velocity_px_s=3000.0,
            warmup_updates=2,
        ),
        prediction=PredictionConfig(
            enabled_x=True,
            enabled_y=False,
            actuation_delay_s=0.005,
            max_horizon_s=0.035,
            far_weight=0.50,
            near_weight=0.12,
            far_abs_cap_px=4.0,
            near_abs_cap_px=2.0,
            far_base_cap_px=1.0,
            near_base_cap_px=0.5,
            far_relative_cap=0.10,
            near_relative_cap=0.15,
            near_cross_allow_px=0.5,
            high_confidence_cross_threshold=0.85,
            overzero_cooldown_frames=2,
        ),
    )
    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)

    decisions = []
    for generation, error_x in enumerate((40.0, 42.0, 44.0, 46.0, 48.0), start=1):
        capture_ns = 1_000_000_000 + generation * 10_000_000
        decisions.append(
            algorithm.calculate(
                _observation(
                    generation=generation,
                    error_x=error_x,
                    error_y=float(generation),
                    capture_ts_ns=capture_ns,
                    control_now_ns=capture_ns + 10_000_000,
                    track_confidence=0.80,
                )
            )
        )

    decision = decisions[-1]
    telemetry = decision.telemetry
    assert telemetry["measurement_dt_s"] == pytest.approx(0.010)
    assert telemetry["estimated_velocity_x"] > 0.0
    assert telemetry["motion_confidence"] > 0.0
    assert telemetry["prediction_horizon_s"] == pytest.approx(0.015)
    assert telemetry["prediction_weight"] == pytest.approx(
        config.prediction.far_weight * telemetry["motion_confidence"] * 0.80
    )
    assert 0.0 < telemetry["prediction_safe_offset_x"] <= 4.0
    assert telemetry["error_control_x"] > telemetry["error_real_x"]
    assert telemetry["error_control_y"] == pytest.approx(telemetry["error_real_y"])


def test_near_low_confidence_prediction_cannot_reverse_real_error_direction() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    config = replace(
        defaults,
        estimator=replace(
            defaults.estimator,
            measurement_std_px=0.5,
            acceleration_std_px_s2=200.0,
            innovation_hard_gate_sigma=100.0,
            warmup_updates=2,
        ),
        prediction=replace(
            defaults.prediction,
            actuation_delay_s=0.015,
            max_horizon_s=0.035,
            near_weight=1.0,
            near_abs_cap_px=8.0,
            near_base_cap_px=8.0,
            near_relative_cap=0.0,
            high_confidence_cross_threshold=0.85,
        ),
    )
    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)

    decision = None
    for generation, error_x in enumerate((52.0, 42.0, 32.0, 22.0, 12.0, 2.0), start=1):
        capture_ns = 1_000_000_000 + generation * 10_000_000
        decision = algorithm.calculate(
            _observation(
                generation=generation,
                error_x=error_x,
                capture_ts_ns=capture_ns,
                control_now_ns=capture_ns + 20_000_000,
                track_confidence=0.20,
            )
        )

    assert decision is not None
    telemetry = decision.telemetry
    assert telemetry["mode"] == ControlMode.NEAR.value
    assert telemetry["prediction_raw_offset_x"] < 0.0
    assert telemetry["prediction_crossing_limited"] is True
    assert telemetry["error_real_x"] == pytest.approx(2.0)
    assert telemetry["error_control_x"] == pytest.approx(0.0)
    assert telemetry["prediction_safe_offset_x"] == pytest.approx(-2.0)


def test_invalid_or_non_monotonic_observation_blocks_whole_decision_and_cancels() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())
    first = _observation(
        generation=10,
        error_x=30.0,
        capture_ts_ns=1_000_000_000,
        control_now_ns=1_008_000_000,
    )
    algorithm.calculate(first)

    invalid_observations = [
        replace(
            first,
            capture_ts_ns=1_010_000_000,
            inference_end_ts_ns=1_014_000_000,
            control_now_ns=1_018_000_000,
        ),
        replace(
            first,
            generation=11,
            capture_ts_ns=1_010_000_000,
            inference_end_ts_ns=1_014_000_000,
            control_now_ns=1_018_000_000,
        ),
        replace(first, generation=11, frame_id=11),
        _observation(
            generation=11,
            error_x=30.0,
            capture_ts_ns=1_010_000_000,
            control_now_ns=1_070_000_000,
        ),
        _observation(
            generation=11,
            error_x=30.0,
            capture_ts_ns=1_010_000_000,
            control_now_ns=1_009_000_000,
        ),
    ]
    expected_reasons = [
        "NON_MONOTONIC_OBSERVATION",
        "NON_MONOTONIC_OBSERVATION",
        "NON_MONOTONIC_OBSERVATION",
        "STALE_OBSERVATION",
        "TIMESTAMP_DOMAIN_INVALID",
    ]

    blocked = [algorithm.calculate(value) for value in invalid_observations]

    assert [decision.block_reason for decision in blocked] == expected_reasons
    assert all(decision.emit_allowed is False for decision in blocked)
    assert all((decision.dx, decision.dy) == (0, 0) for decision in blocked)
    accepted = algorithm.calculate(
        _observation(
            generation=11,
            error_x=30.0,
            capture_ts_ns=1_010_000_000,
            control_now_ns=1_018_000_000,
        )
    )
    assert accepted.block_reason == ""
    assert accepted.telemetry["measurement_dt_s"] == pytest.approx(0.010)


def test_inactive_trigger_cancels_without_banking_fractional_counts() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    phase = AtanPhaseConfig(
        kp=0.25,
        atan_scale_counts=256.0,
        max_counts_per_update=140.0,
    )
    config = replace(
        defaults,
        projection=ProjectionConfig(fov_x_deg=90.0, counts_per_360=36_000.0),
        mode=ModeSelectorConfig(
            near_enter_min_px=0.01,
            near_exit_min_px=0.02,
            near_enter_bbox_h_ratio=0.0,
            near_exit_bbox_h_ratio=0.0,
        ),
        far=phase,
        prediction=replace(defaults.prediction, enabled_x=False),
    )
    desired_demand = 0.4
    full_counts = phase.atan_scale_counts * math.tan(
        desired_demand / (phase.kp * phase.atan_scale_counts)
    )
    theta = full_counts * math.tau / config.projection.counts_per_360
    focal_x = 1920.0 * 0.5 / math.tan(math.radians(90.0) * 0.5)
    observation_error = focal_x * math.tan(theta) * 320.0 / 640.0
    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)

    inactive = [
        algorithm.calculate(
            _observation(
                generation=generation,
                error_x=observation_error,
                trigger_active=False,
            )
        )
        for generation in range(1, 7)
    ]
    first_active = algorithm.calculate(
        _observation(generation=7, error_x=observation_error, trigger_active=True)
    )

    assert all(decision.block_reason == "TRIGGER_INACTIVE" for decision in inactive)
    assert all(decision.emit_allowed is False for decision in inactive)
    assert all((decision.dx, decision.dy) == (0, 0) for decision in inactive)
    assert all(decision.telemetry["quantizer_residual_x"] == 0.0 for decision in inactive)
    assert first_active.dx == 0
    assert first_active.telemetry["quantizer_residual_x"] == pytest.approx(0.4)


def test_every_real_sign_cross_clears_residual_and_cools_prediction() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())

    positive = algorithm.calculate(_observation(generation=1, error_x=2.0))
    near_center_positive = algorithm.calculate(_observation(generation=2, error_x=0.5))
    crossed = algorithm.calculate(_observation(generation=3, error_x=-0.5))
    cooldown_second = algorithm.calculate(_observation(generation=4, error_x=-0.5))
    cooldown_finished = algorithm.calculate(_observation(generation=5, error_x=-0.5))

    assert isinstance(positive.dx, int)
    assert isinstance(positive.dy, int)
    assert {item.name for item in fields(ControlDecision)} == {
        "dx",
        "dy",
        "emit_allowed",
        "block_reason",
        "telemetry",
    }
    assert near_center_positive.telemetry["overzero_detected_x"] is False
    assert crossed.telemetry["overzero_detected_x"] is True
    assert crossed.telemetry["overzero_cleared_residual_x"] > 0.0
    assert crossed.telemetry["residual_direction_reset_x"] is False
    assert crossed.telemetry["prediction_allowed"] is False
    assert crossed.telemetry["prediction_cooldown_frames_remaining"] == 1
    assert cooldown_second.telemetry["prediction_allowed"] is False
    assert cooldown_second.telemetry["prediction_cooldown_frames_remaining"] == 0
    assert cooldown_finished.telemetry["prediction_allowed"] is True


def test_target_switch_fully_resets_mode_estimator_prediction_and_overzero_history() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())
    for generation, error_x in enumerate((40.0, 42.0, 44.0, 46.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    switched = algorithm.calculate(_observation(generation=5, target_id=8, error_x=-40.0))

    assert switched.telemetry["mode"] == ControlMode.FAR.value
    assert switched.telemetry["measurement_dt_s"] == pytest.approx(0.0)
    assert switched.telemetry["estimator_reset"] is True
    assert switched.telemetry["estimated_velocity_x"] == 0.0
    assert switched.telemetry["motion_confidence"] == 0.0
    assert switched.telemetry["prediction_safe_offset_x"] == 0.0
    assert switched.telemetry["overzero_detected_x"] is False
    assert switched.telemetry["prediction_cooldown_frames_remaining"] == 0


def test_target_switch_cannot_bypass_global_observation_monotonicity() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())
    algorithm.calculate(_observation(generation=5, target_id=7, error_x=30.0))

    regressed = algorithm.calculate(
        _observation(
            generation=4,
            frame_id=4,
            target_id=8,
            error_x=30.0,
            capture_ts_ns=1_030_000_000,
            control_now_ns=1_038_000_000,
        )
    )

    assert regressed.block_reason == "NON_MONOTONIC_OBSERVATION"
    assert regressed.emit_allowed is False


def test_inference_timestamp_must_share_the_monotonic_control_domain() -> None:
    algorithm = DualPhaseAtanPredictiveV1Algorithm(DualPhaseAtanPredictiveV1Config())
    capture_ns = 1_000_000_000

    before_capture = algorithm.calculate(
        _observation(
            generation=1,
            error_x=30.0,
            inference_end_ts_ns=capture_ns - 1,
            capture_ts_ns=capture_ns,
            control_now_ns=capture_ns + 8_000_000,
        )
    )
    after_control = algorithm.calculate(
        _observation(
            generation=1,
            error_x=30.0,
            inference_end_ts_ns=capture_ns + 9_000_000,
            capture_ts_ns=capture_ns,
            control_now_ns=capture_ns + 8_000_000,
        )
    )

    assert before_capture.block_reason == "TIMESTAMP_DOMAIN_INVALID"
    assert after_control.block_reason == "TIMESTAMP_DOMAIN_INVALID"


def test_single_hard_innovation_cannot_create_prediction_and_second_resets_estimator() -> None:
    defaults = DualPhaseAtanPredictiveV1Config()
    config = replace(
        defaults,
        estimator=replace(
            defaults.estimator,
            measurement_std_px=0.5,
            acceleration_std_px_s2=25.0,
            innovation_soft_gate_sigma=1.0,
            innovation_hard_gate_sigma=2.0,
            hard_outlier_reset_count=2,
            warmup_updates=2,
        ),
    )
    algorithm = DualPhaseAtanPredictiveV1Algorithm(config)
    for generation, error_x in enumerate((40.0, 41.0, 42.0, 43.0, 44.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    first_outlier = algorithm.calculate(_observation(generation=6, error_x=200.0))
    second_outlier = algorithm.calculate(_observation(generation=7, error_x=202.0))

    assert first_outlier.telemetry["estimator_accepted"] is False
    assert first_outlier.telemetry["estimator_reset"] is False
    assert first_outlier.telemetry["motion_confidence"] == 0.0
    assert first_outlier.telemetry["prediction_safe_offset_x"] == 0.0
    assert second_outlier.telemetry["estimator_accepted"] is False
    assert second_outlier.telemetry["estimator_reset"] is True
    assert second_outlier.telemetry["estimated_velocity_x"] == 0.0
