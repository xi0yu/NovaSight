from __future__ import annotations

from dataclasses import replace
import math
from statistics import mean

import pytest

from novasight.control.algorithms.dual_phase_atan_robust_predictive_v2 import (
    ALGORITHM_ID,
    AtanControllerConfig,
    AtanModeConfig,
    ControlMode,
    DualPhaseAtanRobustPredictiveV2Algorithm,
    DualPhaseAtanRobustPredictiveV2Config,
    DualPhaseAtanRobustPredictiveV2Observation,
    ModeSelectorConfig,
    PredictionConfig,
    PredictionModeConfig,
    ProjectionConfig,
    RobustVelocityEstimator,
    VelocityConfig,
)


def _observation(
    *,
    generation: int,
    error_x: float,
    error_y: float = 0.0,
    frame_id: int | None = None,
    target_id: int = 7,
    capture_ts_ns: int | None = None,
    control_now_ns: int | None = None,
    trigger_active: bool = True,
    target_valid: bool = True,
    detection_confidence: float = 0.95,
    track_confidence: float = 0.90,
    track_rebuilt: bool = False,
    left_trigger_active: bool = False,
    left_trigger_hold_ms: float = 0.0,
    measurement_dt_ms: float | None = 10.0,
) -> DualPhaseAtanRobustPredictiveV2Observation:
    capture_ns = (
        capture_ts_ns if capture_ts_ns is not None else 1_000_000_000 + generation * 10_000_000
    )
    now_ns = control_now_ns if control_now_ns is not None else capture_ns + 8_000_000
    crosshair_x = 160.0
    crosshair_y = 160.0
    return DualPhaseAtanRobustPredictiveV2Observation(
        generation=generation,
        frame_id=generation if frame_id is None else frame_id,
        target_id=target_id,
        capture_ts_ns=capture_ns,
        inference_end_ts_ns=capture_ns + 4_000_000,
        control_now_ns=now_ns,
        aim_x=crosshair_x + error_x,
        aim_y=crosshair_y + error_y,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        bbox_x1=crosshair_x + error_x - 10.0,
        bbox_y1=crosshair_y + error_y - 4.4,
        bbox_x2=crosshair_x + error_x + 10.0,
        bbox_y2=crosshair_y + error_y + 15.6,
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
        left_trigger_active=left_trigger_active,
        left_trigger_hold_ms=left_trigger_hold_ms,
        measurement_dt_ms=measurement_dt_ms,
        track_rebuilt=track_rebuilt,
    )


def test_four_positions_use_three_segment_median_and_px_per_ms() -> None:
    estimator = RobustVelocityEstimator(VelocityConfig())
    estimates = [
        estimator.update(
            target_id=1,
            aim_x=position,
            capture_ts_ns=1_000_000_000 + index * 10_000_000,
            detection_confidence=1.0,
        )
        for index, position in enumerate((100.0, 102.0, 160.0, 106.0))
    ]

    assert estimates[:3] == [None, None, None]
    estimate = estimates[3]
    assert estimate is not None
    assert estimate.raw_velocities == pytest.approx((0.2, 5.8, -5.4))
    assert estimate.median_velocity == pytest.approx(0.2)
    assert estimate.filtered_velocity == pytest.approx(0.2)
    assert estimate.spread == pytest.approx(5.6)
    assert estimate.motion_confidence < 0.05


def test_variable_capture_intervals_preserve_px_per_ms_velocity() -> None:
    estimator = RobustVelocityEstimator(VelocityConfig())
    estimate = None
    for elapsed_ms in (0.0, 8.0, 20.0, 29.0):
        estimate = estimator.update(
            target_id=1,
            aim_x=100.0 + 0.5 * elapsed_ms,
            capture_ts_ns=1_000_000_000 + int(elapsed_ms * 1_000_000),
            detection_confidence=1.0,
        )

    assert estimate is not None
    assert estimate.raw_velocities == pytest.approx((0.5, 0.5, 0.5))
    assert estimate.median_velocity == pytest.approx(0.5)
    assert estimate.filtered_velocity == pytest.approx(0.5)
    assert estimate.reference_dt_ms == pytest.approx((8.0 + 12.0 + 9.0) / 3.0)


def test_stationary_samples_decay_ema_without_forced_zero() -> None:
    config = VelocityConfig(smoothing_frames=3.0)
    estimator = RobustVelocityEstimator(config)
    positions = (100.0, 105.0, 110.0, 115.0, 115.0, 115.0, 115.0)
    estimates = []
    for index, position in enumerate(positions):
        estimate = estimator.update(
            target_id=1,
            aim_x=position,
            capture_ts_ns=1_000_000_000 + index * 10_000_000,
            detection_confidence=1.0,
        )
        if estimate is not None:
            estimates.append(estimate)

    assert estimates[0].filtered_velocity == pytest.approx(0.5)
    assert estimates[-1].median_velocity == 0.0
    # Once two of the three rolling segments are zero, the median is zero;
    # this sequence therefore applies two 10 ms EMA decay updates.
    expected = 0.5 * math.exp(-2.0 / 3.0)
    assert estimates[-1].filtered_velocity == pytest.approx(expected)
    assert estimates[-1].filtered_velocity > 0.0


def test_sustained_new_speed_recovers_confidence_after_transition() -> None:
    estimator = RobustVelocityEstimator(VelocityConfig())
    positions = (100.0, 102.0, 104.0, 106.0, 108.0, 118.0, 128.0, 138.0, 148.0)
    estimates = []
    for index, position in enumerate(positions):
        estimate = estimator.update(
            target_id=1,
            aim_x=position,
            capture_ts_ns=1_000_000_000 + index * 10_000_000,
            detection_confidence=1.0,
        )
        if estimate is not None:
            estimates.append(estimate)

    transition = min(estimates[2:4], key=lambda item: item.motion_confidence)
    recovered = estimates[-1]
    assert transition.motion_confidence < recovered.motion_confidence
    assert recovered.median_velocity == pytest.approx(1.0)
    assert recovered.spread == pytest.approx(0.0)


def test_reversal_changes_ema_direction_gradually() -> None:
    estimator = RobustVelocityEstimator(VelocityConfig(smoothing_frames=3.0))
    positions = (100.0, 105.0, 110.0, 115.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0)
    estimates = []
    for index, position in enumerate(positions):
        estimate = estimator.update(
            target_id=1,
            aim_x=position,
            capture_ts_ns=1_000_000_000 + index * 10_000_000,
            detection_confidence=1.0,
        )
        if estimate is not None:
            estimates.append(estimate)

    first_negative_median = next(item for item in estimates if item.median_velocity < 0.0)
    assert first_negative_median.filtered_velocity > -0.5
    assert estimates[-1].filtered_velocity < 0.0


def test_prediction_uses_average_dt_times_lead_frames_and_cannot_bypass_cap() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    config = replace(
        defaults,
        prediction=PredictionConfig(
            enabled=True,
            lead_frames=2.0,
            far=PredictionModeConfig(
                absolute_cap_px=3.0,
                base_cap_px=0.0,
                relative_cap=0.05,
            ),
            near=defaults.prediction.near,
        ),
    )
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(config)

    decisions = [
        algorithm.calculate(_observation(generation=index, error_x=error_x))
        for index, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1)
    ]
    decision = decisions[-1]

    assert decision.telemetry["filtered_velocity"] == pytest.approx(0.4)
    assert decision.telemetry["reference_dt_ms"] == pytest.approx(10.0)
    assert decision.telemetry["prediction_lead_frames"] == pytest.approx(2.0)
    assert decision.telemetry["prediction_raw_offset_x"] == pytest.approx(8.0)
    assert decision.telemetry["prediction_weighted_offset_x"] > 2.6
    assert decision.telemetry["prediction_allowed_cap_x"] == pytest.approx(2.6)
    assert decision.telemetry["prediction_safe_offset_x"] == pytest.approx(2.6)
    assert decision.telemetry["error_ctrl_x"] == pytest.approx(54.6)


def test_default_atan_path_skips_prediction_and_velocity_history() -> None:
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        DualPhaseAtanRobustPredictiveV2Config()
    )
    decision = None
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        decision = algorithm.calculate(_observation(generation=generation, error_x=error_x))

    assert decision is not None
    assert decision.telemetry["history_position_count"] == 0
    assert decision.telemetry["filtered_velocity"] == 0.0
    assert decision.telemetry["prediction_lead_frames"] == 0.0
    assert decision.telemetry["prediction_allowed"] is False
    assert decision.telemetry["prediction_safe_offset_x"] == 0.0
    assert decision.telemetry["error_ctrl_x"] == decision.telemetry["error_meas_x"]
    assert decision.telemetry["error_ctrl_y"] == decision.telemetry["error_meas_y"]


def test_track_identity_confidence_scales_prediction_to_zero() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    )
    decision = None
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        decision = algorithm.calculate(
            _observation(
                generation=generation,
                error_x=error_x,
                track_confidence=0.0,
            )
        )

    assert decision is not None
    assert decision.telemetry["filtered_velocity"] == pytest.approx(0.4)
    assert decision.telemetry["track_quality"] == 0.0
    assert decision.telemetry["motion_confidence"] == 0.0
    assert decision.telemetry["prediction_safe_offset_x"] == 0.0
    assert decision.telemetry["error_ctrl_x"] == decision.telemetry["error_meas_x"]


def test_real_measurement_error_drives_single_threshold_far_near_selection() -> None:
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(DualPhaseAtanRobustPredictiveV2Config())

    far = algorithm.calculate(_observation(generation=1, error_x=30.0))
    entered = algorithm.calculate(_observation(generation=2, error_x=10.0))
    returned_far = algorithm.calculate(_observation(generation=3, error_x=15.0))
    returned_near = algorithm.calculate(_observation(generation=4, error_x=11.0))

    assert ALGORITHM_ID == "dual_phase_atan_robust_predictive_v2"
    assert far.telemetry["mode"] == ControlMode.FAR.value
    assert entered.telemetry["mode"] == ControlMode.NEAR.value
    assert returned_far.telemetry["mode"] == ControlMode.FAR.value
    assert returned_near.telemetry["mode"] == ControlMode.NEAR.value
    assert returned_far.telemetry["near_threshold_px"] == 12.0


def test_defaults_preserve_the_verified_python_control_profile() -> None:
    config = DualPhaseAtanRobustPredictiveV2Config()

    assert config.prediction.enabled is False
    assert config.velocity.smoothing_frames == 3.0
    assert config.prediction.lead_frames == 1.0
    assert config.prediction.far.absolute_cap_px == 10.0
    assert config.prediction.near.absolute_cap_px == 3.0
    assert config.atan.scale_counts == 256.0
    assert config.atan.far.kp == 0.45
    assert config.atan.near.kp == 0.22
    assert config.atan.far.max_counts_per_update == 127.0
    assert config.atan.near.max_counts_per_update == 72.0


def test_far_controller_can_use_kmnet_counts_above_legacy_hid8_limit() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(
            defaults,
            atan=AtanControllerConfig(
                scale_counts=1024.0,
                far=AtanModeConfig(kp=0.90, max_counts_per_update=600.0),
                near=defaults.atan.near,
            ),
        )
    )

    decision = algorithm.calculate(_observation(generation=1, error_x=80.0))

    assert decision.dx > 127
    assert decision.dx <= 600


def test_atan_only_path_ignores_humanized_profile() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    axis_limit = 12.0
    profile = {
        "timing": {"fitts_a_ms": 0.0, "fitts_b_ms": 1.0},
        "progress_curve": [0.0, 1.0],
        "runtime_parameters": {"micro_bypass_px": 0.0},
    }
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(
            defaults,
            mode=ModeSelectorConfig(near_threshold_px=0.01),
            atan=replace(
                defaults.atan,
                far=AtanModeConfig(kp=1.0, max_counts_per_update=axis_limit),
            ),
            humanized_profile=profile,
        )
    )
    algorithm.calculate(
        _observation(
            generation=1,
            error_x=150.0,
            error_y=150.0,
        )
    )
    decision = algorithm.calculate(
        _observation(
            generation=2,
            error_x=150.0,
            error_y=150.0,
        )
    )

    assert decision.telemetry["humanized_motion_enabled"] is False
    assert decision.telemetry["humanized_motion_reason"] == "atan_only_production_path"
    assert abs(decision.dx) <= axis_limit
    assert abs(decision.dy) <= axis_limit
    assert abs(decision.telemetry["float_demand_x"]) <= axis_limit
    assert abs(decision.telemetry["float_demand_y"]) <= axis_limit


def test_projection_atan_and_control_atan_are_separate_unit_steps() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    far = AtanModeConfig(kp=0.25, max_counts_per_update=127.0)
    config = replace(
        defaults,
        projection=ProjectionConfig(fov_x_deg=90.0, counts_per_360=360.0),
        mode=ModeSelectorConfig(near_threshold_px=0.01),
        prediction=replace(defaults.prediction, lead_frames=0.0),
        atan=AtanControllerConfig(
            scale_counts=256.0,
            far=far,
            near=defaults.atan.near,
        ),
    )
    decision = DualPhaseAtanRobustPredictiveV2Algorithm(config).calculate(
        _observation(generation=1, error_x=30.0)
    )

    source_error_x = 30.0 * 640.0 / 320.0
    focal_x = 1920.0 * 0.5 / math.tan(math.radians(90.0) * 0.5)
    full_counts_x = math.atan(source_error_x / focal_x) * 360.0 / math.tau
    expected_demand = 0.25 * 256.0 * math.atan(full_counts_x / 256.0)

    assert decision.telemetry["full_error_counts_x"] == pytest.approx(full_counts_x)
    assert decision.telemetry["float_demand_x"] == pytest.approx(expected_demand)


def test_target_switch_clears_velocity_and_fractional_counts() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    predictive = replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(predictive)
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    switched = algorithm.calculate(_observation(generation=5, target_id=8, error_x=-40.0))
    fresh = DualPhaseAtanRobustPredictiveV2Algorithm(
        predictive
    ).calculate(_observation(generation=5, target_id=8, error_x=-40.0))

    assert switched.telemetry["mode"] == ControlMode.FAR.value
    assert switched.telemetry["history_position_count"] == 1
    assert switched.telemetry["motion_confidence"] == 0.0
    assert switched.telemetry["filtered_velocity"] == 0.0
    assert switched.telemetry["prediction_safe_offset_x"] == 0.0
    assert switched.dx == fresh.dx
    assert switched.dy == fresh.dy
    assert switched.telemetry["quantizer_residual_x"] == pytest.approx(
        fresh.telemetry["quantizer_residual_x"]
    )


def test_history_gap_restarts_window_from_current_measurement() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    )
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    capture_ns = 1_000_000_000 + 5 * 10_000_000 + 90_000_000
    after_gap = algorithm.calculate(
        _observation(
            generation=5,
            error_x=56.0,
            capture_ts_ns=capture_ns,
            control_now_ns=capture_ns + 8_000_000,
        )
    )

    assert after_gap.telemetry["history_position_count"] == 1
    assert after_gap.telemetry["motion_confidence"] == 0.0
    assert after_gap.telemetry["prediction_safe_offset_x"] == 0.0


def test_coordinate_space_change_resets_same_target_history() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    )
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    changed_geometry = replace(
        _observation(generation=5, error_x=56.0),
        roi_left=600,
    )
    decision = algorithm.calculate(changed_geometry)

    assert decision.telemetry["history_position_count"] == 1
    assert decision.telemetry["motion_confidence"] == 0.0
    assert decision.telemetry["prediction_safe_offset_x"] == 0.0


def test_tracker_rebuild_resets_same_id_history() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    )
    for generation, error_x in enumerate((40.0, 44.0, 48.0, 52.0), start=1):
        algorithm.calculate(_observation(generation=generation, error_x=error_x))

    rebuilt = algorithm.calculate(
        _observation(
            generation=5,
            error_x=56.0,
            track_rebuilt=True,
        )
    )

    assert rebuilt.telemetry["track_rebuilt"] is True
    assert rebuilt.telemetry["history_position_count"] == 1
    assert rebuilt.telemetry["motion_confidence"] == 0.0
    assert rebuilt.telemetry["prediction_safe_offset_x"] == 0.0


def test_capture_timestamp_regression_uses_feedback_and_starts_a_new_epoch() -> None:
    defaults = DualPhaseAtanRobustPredictiveV2Config()
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        replace(defaults, prediction=replace(defaults.prediction, enabled=True))
    )
    algorithm.calculate(
        _observation(
            generation=1,
            error_x=40.0,
            capture_ts_ns=1_010_000_000,
        )
    )

    regressed = algorithm.calculate(
        _observation(
            generation=2,
            error_x=44.0,
            capture_ts_ns=1_005_000_000,
        )
    )
    recovered = algorithm.calculate(
        _observation(
            generation=3,
            error_x=48.0,
            capture_ts_ns=1_015_000_000,
        )
    )

    assert regressed.block_reason == ""
    assert regressed.emit_allowed is True
    assert regressed.telemetry["capture_timestamp_discontinuity"] is True
    assert regressed.telemetry["history_position_count"] == 0
    assert regressed.telemetry["prediction_safe_offset_x"] == 0.0
    assert regressed.telemetry["error_ctrl_x"] == regressed.telemetry["error_meas_x"]
    assert recovered.block_reason == ""
    assert recovered.telemetry["capture_timestamp_discontinuity"] is False
    assert recovered.telemetry["history_position_count"] == 1


def test_stale_observation_cannot_emit_and_clears_fractional_residual() -> None:
    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(DualPhaseAtanRobustPredictiveV2Config())
    algorithm.calculate(_observation(generation=1, error_x=0.2))
    capture_ns = 1_020_000_000
    stale = algorithm.calculate(
        _observation(
            generation=2,
            error_x=0.2,
            capture_ts_ns=capture_ns,
            control_now_ns=capture_ns + 56_000_000,
        )
    )

    assert stale.block_reason == "STALE_OBSERVATION"
    assert stale.emit_allowed is False
    assert stale.dx == 0
    assert stale.dy == 0
    assert stale.telemetry["quantizer_residual_x"] == 0.0


def test_frame_lead_prediction_reduces_closed_loop_lag_against_feedback_baseline() -> None:
    def simulate(lead_frames: float) -> float:
        defaults = DualPhaseAtanRobustPredictiveV2Config()
        algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
            replace(
                defaults,
                prediction=replace(
                    defaults.prediction,
                    enabled=lead_frames > 0.0,
                    lead_frames=lead_frames,
                ),
            )
        )
        error_x = 20.0
        settled_errors: list[float] = []
        # Linearized local projection: one device count rotates the image by
        # roughly 0.23 observation pixels for this test geometry/calibration.
        observation_px_per_count = 0.23
        for generation in range(1, 181):
            error_x += 1.0
            decision = algorithm.calculate(_observation(generation=generation, error_x=error_x))
            error_x -= decision.dx * observation_px_per_count
            if generation > 30:
                settled_errors.append(abs(error_x))
        return mean(settled_errors)

    feedback_error = simulate(0.0)
    predictive_error = simulate(1.0)

    assert predictive_error < feedback_error
