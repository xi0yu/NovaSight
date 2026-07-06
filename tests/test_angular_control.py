import math

from novasight.contracts import Detection
from novasight.control import (
    AngularErrorMapper,
    AngularPDConfig,
    AngularPDController,
    CalibrationProfile,
    ExperimentalAnglePidStrategy,
)
from novasight.control.angular import AngularErrorState
from novasight.hardware import BoxInputState


def test_mapper_uses_horizontal_fov_and_full_control_geometry() -> None:
    mapper = AngularErrorMapper(CalibrationProfile(fov_x_deg=105, counts_per_360_x=9980))

    err = mapper.map(
        comp_x=1060,
        comp_y=540,
        control_width_px=1920,
        control_height_px=1080,
        dt_s=1 / 120,
        source_frame_id=10,
    )

    focal_x = 1920 / (2 * math.tan(math.radians(105) / 2))
    expected = math.atan(100 / focal_x)
    assert err.control_allowed is True
    assert err.error_x_px == 100
    assert err.error_y_px == 0
    assert err.error_x_rad == expected
    assert err.focal_x_px == focal_x


def test_mapper_rejects_missing_control_geometry_instead_of_roi_fallback() -> None:
    mapper = AngularErrorMapper(CalibrationProfile())

    err = mapper.map(
        comp_x=160,
        comp_y=160,
        control_width_px=0,
        control_height_px=0,
        dt_s=1 / 120,
    )

    assert err.control_allowed is False
    assert err.invalid_reason == "CONTROL_GEOMETRY_INVALID"


def test_angular_controller_accumulates_fractional_counts() -> None:
    controller = AngularPDController(
        AngularPDConfig(kp_x=1.0, kp_y=0.0, max_step_counts=80),
        CalibrationProfile(counts_per_360_x=2 * math.pi, counts_per_360_y=2 * math.pi),
    )

    err = AngularErrorState(
        source_frame_id=1,
        track_id=None,
        control_width_px=100,
        control_height_px=100,
        center_x_px=50,
        center_y_px=50,
        comp_x=50,
        comp_y=50,
        error_x_px=0,
        error_y_px=0,
        fov_x_rad=1,
        fov_y_rad=1,
        focal_x_px=1,
        focal_y_px=1,
        error_x_rad=0.4,
        error_y_rad=0.0,
        dt_s=1 / 120,
    )

    assert controller.update(err).dx == 0
    assert controller.update(err).dx == 0
    third = controller.update(err)
    assert third.dx == 1
    assert 0 < third.residual_x_counts < 1


def test_experimental_strategy_rejects_missing_full_geometry() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False)
    target = Detection(cls=0, score=0.9, x=150, y=150, w=20, h=20)

    command = strategy.calculate(
        target,
        (160, 160),
        BoxInputState(left=True, raw={"roi_width": 320, "roi_height": 320}),
    )

    assert command.dx == 0
    assert command.dy == 0
    assert command.reason == "CONTROL_GEOMETRY_INVALID"


def test_experimental_strategy_maps_roi_target_to_full_control_coordinates() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False, kp_x=0.35, kp_y=0.35)
    target = Detection(cls=0, score=0.9, x=160, y=160, w=20, h=20)

    command = strategy.calculate(
        target,
        (160, 160),
        BoxInputState(
            left=True,
            raw={
                "roi_width": 320,
                "roi_height": 320,
                "capture_width": 1920,
                "capture_height": 1080,
                "roi_offset_x": 800,
                "roi_offset_y": 380,
            },
        ),
    )

    assert command.dx > 0
    assert command.dy > 0
    assert command.debug["comp_x"] == 970
    assert command.debug["comp_y"] == 550
    assert command.debug["capture_size_source"] == "frame_metadata"
