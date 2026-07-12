from __future__ import annotations

import math

import pytest

from novasight.contracts import Track
from novasight.control import (
    CALIBRATED_ANGULAR,
    UNIVERSAL_SATURATED,
    CalibratedAngularController,
    CalibratedAngularControllerConfig,
    ControllerFactory,
    MouseController,
    MouseControllerConfig,
    MouseObservation,
    RawAimPointProjector,
    SharedOutputConfig,
    TTBOX_PID_ATAN,
    TtboxPidAtanController,
    TtboxPidAtanControllerConfig,
    UniversalSaturatedController,
    UniversalSaturatedControllerConfig,
    normalize_aim_y_ratio,
    target_motion_estimate_from_debug,
)
from novasight.coordinates import CoordinateTransform


def _config(
    *,
    mode: str = CALIBRATED_ANGULAR,
    calibrated: dict[str, float] | None = None,
    universal: dict[str, float] | None = None,
    ttbox: dict[str, float | bool] | None = None,
    shared: dict[str, float | int | bool] | None = None,
) -> MouseControllerConfig:
    calibrated_values = {
        "fov_x_deg": 90.0,
        "counts_per_360_x": 1000.0,
        "counts_per_360_y": 1000.0,
        "kp_x": 1.0,
        "kp_y": 1.0,
        "kd_x": 0.0,
        "kd_y": 0.0,
        "d_ema_alpha": 1.0,
        "max_angle_step_x_rad": 10.0,
        "max_angle_step_y_rad": 10.0,
    }
    universal_values = {
        "response_scale_x_px": 100.0,
        "response_scale_y_px": 100.0,
        "max_step_x_counts": 100.0,
        "max_step_y_counts": 100.0,
    }
    ttbox_values: dict[str, float | bool] = {
        "response_scale_x_px": 256.0,
        "response_scale_y_px": 256.0,
        "per_frame_gain_x": 0.1,
        "per_frame_gain_y": 0.1,
        "normalize_to_dt": True,
        "nominal_dt_s": 0.016,
        "max_step_x_counts": 50.0,
        "max_step_y_counts": 40.0,
    }
    shared_values: dict[str, float | int | bool] = {
        "deadzone_x_px": 0.0,
        "deadzone_y_px": 0.0,
        "max_count_slew_x": 1000.0,
        "max_count_slew_y": 1000.0,
        "invert_y": False,
        "max_budget_counts_x": 1000,
        "max_budget_counts_y": 1000,
        "min_effective_counts_x": 1,
        "min_effective_counts_y": 1,
    }
    calibrated_values.update(calibrated or {})
    universal_values.update(universal or {})
    ttbox_values.update(ttbox or {})
    shared_values.update(shared or {})
    return MouseControllerConfig(
        mode=mode,
        calibrated_angular=CalibratedAngularControllerConfig(**calibrated_values),
        universal_saturated=UniversalSaturatedControllerConfig(**universal_values),
        ttbox_pid_atan=TtboxPidAtanControllerConfig(**ttbox_values),
        shared=SharedOutputConfig(**shared_values),
    )


def _observation(
    *,
    frame_id: int,
    target_id: int = 7,
    observed_x: float = 120.0,
    observed_y: float = 100.0,
    predicted_x: float | None = None,
    predicted_y: float | None = None,
    measurement_dt_s: float | None = 0.1,
) -> MouseObservation:
    return MouseObservation(
        frame_id=frame_id,
        target_id=target_id,
        capture_ts_ns=frame_id * 100_000_000,
        control_now_ts_ns=frame_id * 100_000_000 + 10_000_000,
        measurement_dt_s=measurement_dt_s,
        control_width_px=200.0,
        control_height_px=200.0,
        observed_x_px=observed_x,
        observed_y_px=observed_y,
        predicted_x_px=observed_x if predicted_x is None else predicted_x,
        predicted_y_px=observed_y if predicted_y is None else predicted_y,
        prediction_horizon_s=0.02,
        target_confidence=0.9,
        prediction_confidence=0.8,
    )


def test_raw_aim_uses_bbox_center_x_and_rounded_y_ratio_in_full_control_space() -> None:
    track = Track(track_id=7, cls=0, score=0.9, x=10, y=20, w=40, h=100)
    transform = CoordinateTransform(
        model_width=200,
        model_height=200,
        roi_x=300,
        roi_y=100,
        roi_width=200,
        roi_height=200,
        capture_width=800,
        capture_height=600,
    )

    aim = RawAimPointProjector().project(
        track=track,
        frame_id=11,
        capture_ts_ns=1_000_000,
        y_ratio=0.224,
        coordinate_transform=transform,
        control_width_px=800,
        control_height_px=600,
        source_geometry_trusted=True,
    )

    assert aim.valid is True
    assert aim.y_ratio == 0.22
    assert aim.aim_roi_x_px == pytest.approx(30.0)
    assert aim.aim_roi_y_px == pytest.approx(42.0)
    assert aim.aim_control_x_px == pytest.approx(330.0)
    assert aim.aim_control_y_px == pytest.approx(142.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.225, 0.23), (2.0, 1.0)],
)
def test_aim_y_ratio_is_clamped_and_rounded(value: float, expected: float) -> None:
    assert normalize_aim_y_ratio(value) == expected


def test_raw_aim_rejects_untrusted_projection_geometry() -> None:
    track = Track(track_id=7, cls=0, score=0.9, x=10, y=20, w=40, h=100)

    aim = RawAimPointProjector().project(
        track=track,
        frame_id=11,
        capture_ts_ns=1_000_000,
        y_ratio=0.22,
        coordinate_transform=None,
        control_width_px=800,
        control_height_px=600,
        source_geometry_trusted=False,
    )

    assert aim.valid is False
    assert aim.invalid_reason == "SOURCE_GEOMETRY_UNTRUSTED"


def test_missing_tracker_estimate_falls_back_to_the_same_filtered_aim_point() -> None:
    track = Track(
        track_id=7,
        cls=0,
        score=0.9,
        x=10,
        y=20,
        w=40,
        h=100,
        observed_aim_px=(30.0, 42.0),
        filtered_aim_px=(31.5, 43.5),
    )

    estimate = target_motion_estimate_from_debug(
        track=track,
        capture_ts_ns=1_000_000,
        tracker_debug={},
    )

    assert (estimate.x, estimate.y) == pytest.approx(track.filtered_aim_px)
    assert estimate.y != pytest.approx(track.cy)


def test_mouse_controller_uses_predicted_error_for_p() -> None:
    controller = MouseController(_config(calibrated={"kd_x": 0.0, "kp_y": 0.0}))

    command = controller.calculate(
        _observation(frame_id=1, observed_x=120.0, predicted_x=160.0)
    )

    assert command.dx > 0
    assert command.debug["p_x_rad"] == pytest.approx(math.atan(0.6))
    assert command.debug["observed_error_x_rad"] == pytest.approx(math.atan(0.2))
    assert command.debug["predicted_error_x_rad"] == pytest.approx(math.atan(0.6))


def test_mouse_controller_derivative_uses_observed_error_not_predicted_error() -> None:
    controller = MouseController(
        _config(calibrated={"kp_x": 0.0, "kp_y": 0.0, "kd_x": 1.0})
    )
    controller.calculate(_observation(frame_id=1, observed_x=120.0, predicted_x=120.0))

    command = controller.calculate(
        _observation(frame_id=2, observed_x=120.0, predicted_x=180.0)
    )

    assert command.debug["d_raw_x_rad_s"] == pytest.approx(0.0)
    assert command.debug["d_x_rad"] == pytest.approx(0.0)
    assert command.dx == 0


def test_mouse_controller_derivative_damps_error_approaching_center() -> None:
    controller = MouseController(_config(calibrated={"kd_x": 0.1, "kp_y": 0.0}))
    controller.calculate(_observation(frame_id=1, observed_x=160.0))

    command = controller.calculate(_observation(frame_id=2, observed_x=140.0))

    assert command.debug["d_raw_x_rad_s"] < 0.0
    assert command.debug["d_x_rad"] < 0.0
    assert command.debug["requested_output_x_rad"] < command.debug["p_x_rad"]


def test_mouse_controller_target_switch_resets_derivative_and_residual() -> None:
    controller = MouseController(
        _config(calibrated={"kp_x": 0.0, "kp_y": 0.0, "kd_x": 1.0})
    )
    controller.calculate(_observation(frame_id=1, target_id=7, observed_x=150.0))
    controller.state.residual_x_counts = 0.75

    command = controller.calculate(
        _observation(frame_id=2, target_id=8, observed_x=120.0)
    )

    assert command.debug["d_raw_x_rad_s"] == pytest.approx(0.0)
    assert controller.state.residual_x_counts == pytest.approx(0.0)


def test_mouse_controller_rejects_duplicate_observation() -> None:
    controller = MouseController(_config())
    observation = _observation(frame_id=1)
    controller.calculate(observation)

    duplicate = controller.calculate(observation)

    assert duplicate.dx == 0
    assert duplicate.dy == 0
    assert duplicate.reason == "DUPLICATE_OBSERVATION"


def test_mouse_controller_fractional_counts_are_not_permanently_lost() -> None:
    desired_rad = 0.4 * math.tau / 1000.0
    observed_x = 100.0 + math.tan(desired_rad) * 100.0
    controller = MouseController(_config(calibrated={"kp_y": 0.0}))

    outputs = [
        controller.calculate(
            _observation(
                frame_id=frame_id,
                observed_x=observed_x,
                measurement_dt_s=None,
            )
        ).dx
        for frame_id in (1, 2, 3)
    ]

    assert outputs == [0, 1, 0]
    assert controller.state.residual_x_counts == pytest.approx(0.2, abs=1e-9)


def test_subminimum_device_counts_accumulate_until_device_can_move() -> None:
    desired_counts = 2.0
    response_scale = 80.0
    max_counts = 50.0
    error_px = response_scale * math.tan(desired_counts * math.pi / (2.0 * max_counts))
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            universal={
                "response_scale_x_px": response_scale,
                "max_step_x_counts": max_counts,
            },
            shared={"min_effective_counts_x": 16},
        )
    )

    error = error_px
    outputs = []
    errors = []
    for frame_id in range(1, 9):
        errors.append(error)
        output = controller.calculate(
            _observation(frame_id=frame_id, observed_x=100.0 + error)
        ).dx
        outputs.append(output)
        error -= output

    assert outputs == [0] * 7 + [16]
    assert errors == pytest.approx([error_px] * 8)
    assert controller.state.residual_x_counts == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("observed_x", "expected_dx", "expected_residual"),
    [
        (105.0, 1, -0.4033631306),
        (95.0, -1, 0.4033631306),
    ],
)
def test_universal_saturated_quantizes_first_action_without_losing_residual(
    observed_x: float,
    expected_dx: int,
    expected_residual: float,
) -> None:
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            universal={
                "response_scale_x_px": 160.0,
                "response_scale_y_px": 120.0,
                "max_step_x_counts": 30.0,
                "max_step_y_counts": 24.0,
            },
        )
    )

    command = controller.calculate(
        _observation(frame_id=1, observed_x=observed_x, observed_y=100.0)
    )

    assert abs(command.debug["theoretical_counts_x_float"]) == pytest.approx(0.5966368694)
    assert command.dx == expected_dx
    assert controller.state.residual_x_counts == pytest.approx(expected_residual)


def test_mouse_controller_inverts_y_only_in_count_mapping() -> None:
    controller = MouseController(
        _config(calibrated={"kp_x": 0.0}, shared={"invert_y": True})
    )

    command = controller.calculate(_observation(frame_id=1, observed_y=150.0))

    assert command.debug["limited_output_y_rad"] > 0.0
    assert command.dy < 0


def test_mouse_controller_distinguishes_theoretical_and_limited_counts() -> None:
    controller = MouseController(
        _config(
            calibrated={"kp_y": 0.0, "max_angle_step_x_rad": 0.1},
        )
    )

    command = controller.calculate(
        _observation(frame_id=1, observed_x=180.0, predicted_x=180.0)
    )

    assert command.debug["theoretical_counts_x_float"] > command.debug["mode_limited_counts_x_float"]
    assert command.debug["mode_limited_counts_x_float"] == pytest.approx(
        command.debug["feasible_counts_x_float"]
    )


def test_controller_factory_creates_only_requested_mode() -> None:
    calibrated = ControllerFactory.create(_config(mode=CALIBRATED_ANGULAR))
    universal = ControllerFactory.create(_config(mode=UNIVERSAL_SATURATED))

    assert isinstance(calibrated, CalibratedAngularController)
    assert not isinstance(calibrated, UniversalSaturatedController)
    assert isinstance(universal, UniversalSaturatedController)
    assert not isinstance(universal, CalibratedAngularController)


def test_universal_saturated_zero_sign_and_bound() -> None:
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            universal={
                "response_scale_x_px": 20.0,
                "response_scale_y_px": 20.0,
                "max_step_x_counts": 30.0,
                "max_step_y_counts": 24.0,
            },
        )
    )

    zero = controller.calculate(_observation(frame_id=1, observed_x=100, observed_y=100))
    positive = controller.calculate(_observation(frame_id=2, observed_x=10_000, observed_y=10_000))
    negative = controller.calculate(_observation(frame_id=3, observed_x=-10_000, observed_y=-10_000))

    assert (zero.dx, zero.dy) == (0, 0)
    assert 0 < positive.dx <= 30
    assert 0 < positive.dy <= 24
    assert -30 <= negative.dx < 0
    assert -24 <= negative.dy < 0
    assert "d_raw_x_rad_s" not in positive.debug


def test_universal_response_scale_controls_near_center_gain() -> None:
    fast = MouseController(
        _config(mode=UNIVERSAL_SATURATED, universal={"response_scale_x_px": 20.0})
    )
    soft = MouseController(
        _config(mode=UNIVERSAL_SATURATED, universal={"response_scale_x_px": 200.0})
    )

    fast_command = fast.calculate(_observation(frame_id=1, observed_x=110, observed_y=100))
    soft_command = soft.calculate(_observation(frame_id=1, observed_x=110, observed_y=100))

    assert fast_command.dx > soft_command.dx > 0


def test_shared_count_slew_limits_adjacent_observation_change() -> None:
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            shared={"max_count_slew_x": 3.0},
        )
    )
    controller.calculate(_observation(frame_id=1, observed_x=100, observed_y=100))

    command = controller.calculate(_observation(frame_id=2, observed_x=10_000, observed_y=100))

    assert command.debug["directed_counts_x_float"] > 3.0
    assert command.debug["slew_limited_counts_x_float"] == pytest.approx(3.0)
    assert command.dx == 3


def test_arrival_hold_cannot_be_undone_by_slew_decay() -> None:
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            universal={"response_scale_y_px": 20.0, "max_step_y_counts": 100.0},
            shared={"deadzone_y_px": 4.0, "max_count_slew_y": 8.0},
        )
    )
    first = controller.calculate(_observation(frame_id=1, observed_y=200.0))
    assert first.dy > 8

    arrived = controller.calculate(_observation(frame_id=2, observed_y=100.0))

    assert arrived.debug["deadzone_limited_counts_y_float"] == pytest.approx(0.0)
    assert arrived.debug["slew_limited_counts_y_float"] == pytest.approx(0.0)
    assert arrived.dy == 0


def test_settled_axis_ignores_one_frame_outside_exit_threshold() -> None:
    controller = MouseController(
        _config(
            mode=UNIVERSAL_SATURATED,
            shared={"deadzone_y_px": 4.0},
        )
    )
    controller.calculate(_observation(frame_id=1, observed_y=100.0))
    settled = controller.calculate(_observation(frame_id=2, observed_y=100.0))
    assert settled.debug["arrival_settled_y"] is True

    noisy = controller.calculate(_observation(frame_id=3, observed_y=107.0))

    assert noisy.debug["arrival_settled_y"] is True
    assert noisy.debug["arrival_departure_candidate_y_frames"] == 1
    assert noisy.dy == 0

    confirmed_departure = controller.calculate(_observation(frame_id=4, observed_y=107.0))

    assert confirmed_departure.debug["arrival_settled_y"] is False
    assert confirmed_departure.debug["arrival_departure_candidate_y_frames"] == 0
    assert confirmed_departure.dy > 0


def test_ttbox_pid_atan_factory_and_mode_string() -> None:
    controller = ControllerFactory.create(_config(mode=TTBOX_PID_ATAN))

    assert isinstance(controller, TtboxPidAtanController)
    assert controller.mode == TTBOX_PID_ATAN
    assert not isinstance(controller, UniversalSaturatedController)


def test_ttbox_pid_atan_zero_sign_and_bound() -> None:
    controller = MouseController(
        _config(
            mode=TTBOX_PID_ATAN,
            ttbox={
                "response_scale_x_px": 100.0,
                "response_scale_y_px": 100.0,
                "per_frame_gain_x": 0.5,
                "per_frame_gain_y": 0.5,
                "max_step_x_counts": 30.0,
                "max_step_y_counts": 24.0,
                "normalize_to_dt": False,
            },
        )
    )

    zero = controller.calculate(_observation(frame_id=1, observed_x=100, observed_y=100))
    positive = controller.calculate(_observation(frame_id=2, observed_x=10_000, observed_y=10_000))
    negative = controller.calculate(_observation(frame_id=3, observed_x=-10_000, observed_y=-10_000))

    assert (zero.dx, zero.dy) == (0, 0)
    assert 0 < positive.dx <= 30
    assert 0 < positive.dy <= 24
    assert -30 <= negative.dx < 0
    assert -24 <= negative.dy < 0


def test_ttbox_pid_atan_near_center_gain_matches_gain_x_scale_formula() -> None:
    controller = MouseController(
        _config(
            mode=TTBOX_PID_ATAN,
            ttbox={
                "response_scale_x_px": 200.0,
                "response_scale_y_px": 200.0,
                "per_frame_gain_x": 0.5,
                "per_frame_gain_y": 0.5,
                "max_step_x_counts": 1000.0,
                "max_step_y_counts": 1000.0,
                "normalize_to_dt": False,
            },
        )
    )

    command = controller.calculate(_observation(frame_id=1, observed_x=110, observed_y=100))

    expected = 0.5 * 200.0 * math.atan(10.0 / 200.0)
    assert command.debug["theoretical_counts_x_float"] == pytest.approx(expected)
    assert command.dx == pytest.approx(_round_trip(expected))


def _round_trip(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))

def test_ttbox_pid_atan_response_scale_controls_near_center_gain() -> None:
    fast = MouseController(
        _config(
            mode=TTBOX_PID_ATAN,
            ttbox={
                "response_scale_x_px": 50.0,
                "per_frame_gain_x": 0.5,
                "normalize_to_dt": False,
            },
        )
    )
    soft = MouseController(
        _config(
            mode=TTBOX_PID_ATAN,
            ttbox={
                "response_scale_x_px": 400.0,
                "per_frame_gain_x": 0.5,
                "normalize_to_dt": False,
            },
        )
    )
    fast_cmd = fast.calculate(_observation(frame_id=1, observed_x=125, observed_y=100))
    soft_cmd = soft.calculate(_observation(frame_id=1, observed_x=125, observed_y=100))

    assert soft_cmd.debug["theoretical_counts_x_float"] > fast_cmd.debug["theoretical_counts_x_float"] > 0
    assert fast_cmd.debug["response_scale_x_px"] == 50.0
    assert soft_cmd.debug["response_scale_x_px"] == 400.0


def test_ttbox_pid_atan_normalize_to_dt_scales_with_frame_dt() -> None:
    controller = MouseController(
        _config(
            mode=TTBOX_PID_ATAN,
            ttbox={
                "response_scale_x_px": 200.0,
                "per_frame_gain_x": 1.0,
                "per_frame_gain_y": 1.0,
                "normalize_to_dt": True,
                "nominal_dt_s": 0.016,
                "max_step_x_counts": 1000.0,
                "max_step_y_counts": 1000.0,
            },
        )
    )

    slow = controller.calculate(
        _observation(frame_id=1, observed_x=110, observed_y=100, measurement_dt_s=0.016)
    )
    controller.calculate(_observation(frame_id=2, observed_x=110, observed_y=100, measurement_dt_s=0.008))
    fast_dt = controller.calculate(_observation(frame_id=3, observed_x=110, observed_y=100, measurement_dt_s=0.008))

    expected_fast = 1.0 * 0.5 * 200.0 * math.atan(10.0 / 200.0)
    expected_slow = 1.0 * 1.0 * 200.0 * math.atan(10.0 / 200.0)
    assert fast_dt.debug["applied_gain_x"] == pytest.approx(0.5)
    assert slow.debug["applied_gain_x"] == pytest.approx(1.0)
    assert fast_dt.debug["theoretical_counts_x_float"] == pytest.approx(expected_fast)
    assert slow.debug["theoretical_counts_x_float"] == pytest.approx(expected_slow)


def test_ttbox_pid_atan_mode_constant_in_export() -> None:
    from novasight.control import CONTROL_MODES

    assert TTBOX_PID_ATAN in CONTROL_MODES
