from __future__ import annotations

import math

import pytest

from novasight.contracts import Track
from novasight.control import (
    MouseController,
    MouseControllerConfig,
    MouseObservation,
    RawAimPointProjector,
    normalize_aim_y_ratio,
)
from novasight.coordinates import CoordinateTransform


def _config(**overrides: float | int | bool) -> MouseControllerConfig:
    values: dict[str, float | int | bool] = {
        "fov_x_deg": 90.0,
        "counts_per_360_x": 1000.0,
        "counts_per_360_y": 1000.0,
        "invert_y": False,
        "kp_x": 1.0,
        "kp_y": 1.0,
        "kd_x": 0.0,
        "kd_y": 0.0,
        "d_ema_alpha": 1.0,
        "deadzone_px_x": 0.0,
        "deadzone_px_y": 0.0,
        "max_output_rad_x": 10.0,
        "max_output_rad_y": 10.0,
        "max_output_rate_rad_s_x": 1000.0,
        "max_output_rate_rad_s_y": 1000.0,
        "max_budget_counts_x": 1000,
        "max_budget_counts_y": 1000,
    }
    values.update(overrides)
    return MouseControllerConfig(**values)


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


def test_mouse_controller_uses_predicted_error_for_p() -> None:
    controller = MouseController(_config(kd_x=0.0, kp_y=0.0))

    command = controller.calculate(
        _observation(frame_id=1, observed_x=120.0, predicted_x=160.0)
    )

    assert command.dx > 0
    assert command.debug["p_x_rad"] == pytest.approx(math.atan(0.6))
    assert command.debug["observed_error_x_rad"] == pytest.approx(math.atan(0.2))
    assert command.debug["predicted_error_x_rad"] == pytest.approx(math.atan(0.6))


def test_mouse_controller_derivative_uses_observed_error_not_predicted_error() -> None:
    controller = MouseController(_config(kp_x=0.0, kp_y=0.0, kd_x=1.0))
    controller.calculate(_observation(frame_id=1, observed_x=120.0, predicted_x=120.0))

    command = controller.calculate(
        _observation(frame_id=2, observed_x=120.0, predicted_x=180.0)
    )

    assert command.debug["d_raw_x_rad_s"] == pytest.approx(0.0)
    assert command.debug["d_x_rad"] == pytest.approx(0.0)
    assert command.dx == 0


def test_mouse_controller_derivative_damps_error_approaching_center() -> None:
    controller = MouseController(_config(kd_x=0.1, kp_y=0.0))
    controller.calculate(_observation(frame_id=1, observed_x=160.0))

    command = controller.calculate(_observation(frame_id=2, observed_x=140.0))

    assert command.debug["d_raw_x_rad_s"] < 0.0
    assert command.debug["d_x_rad"] < 0.0
    assert command.debug["requested_output_x_rad"] < command.debug["p_x_rad"]


def test_mouse_controller_target_switch_resets_derivative_and_residual() -> None:
    controller = MouseController(_config(kp_x=0.0, kp_y=0.0, kd_x=1.0))
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
    controller = MouseController(_config(kp_y=0.0))

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

    assert outputs == [0, 0, 1]
    assert controller.state.residual_x_counts == pytest.approx(0.2, abs=1e-9)


def test_mouse_controller_inverts_y_only_in_count_mapping() -> None:
    controller = MouseController(_config(kp_x=0.0, invert_y=True))

    command = controller.calculate(_observation(frame_id=1, observed_y=150.0))

    assert command.debug["limited_output_y_rad"] > 0.0
    assert command.dy < 0
