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


def _compensated_target_raw(
    *,
    roi_width: int = 320,
    roi_height: int = 320,
    capture_width: int = 1920,
    capture_height: int = 1080,
    roi_offset_x: int = 800,
    roi_offset_y: int = 380,
    roi_x: float = 170,
    roi_y: float = 170,
    control_allowed: bool = True,
) -> dict:
    return {
        "roi_width": roi_width,
        "roi_height": roi_height,
        "capture_width": capture_width,
        "capture_height": capture_height,
        "roi_offset_x": roi_offset_x,
        "roi_offset_y": roi_offset_y,
        "compensated_target": {
            "track_id": 1,
            "source_frame_id": 10,
            "capture_ts_ns": 1_000_000,
            "state_ts_ns": 1_000_000,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "control_x": roi_offset_x + roi_x,
            "control_y": roi_offset_y + roi_y,
            "raw_x": roi_x,
            "raw_y": roi_y,
            "smoothed_x": roi_x,
            "smoothed_y": roi_y,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "applied": False,
            "reason": "TEST",
            "measurement_age_ms": 0.0,
            "compensation_ms": 0.0,
            "prediction_confidence": 1.0,
            "predicted_source": False,
            "control_allowed": control_allowed,
        },
    }


def _angular_error(
    *,
    source_frame_id: int = 1,
    track_id: int | None = 1,
    error_x_rad: float = 0.0,
    error_y_rad: float = 0.0,
    dt_s: float = 1 / 120,
    control_width_px: float = 100,
    control_height_px: float = 100,
    control_allowed: bool = True,
    invalid_reason: str = "",
) -> AngularErrorState:
    return AngularErrorState(
        source_frame_id=source_frame_id,
        track_id=track_id,
        control_width_px=control_width_px,
        control_height_px=control_height_px,
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
        error_x_rad=error_x_rad,
        error_y_rad=error_y_rad,
        dt_s=dt_s,
        control_allowed=control_allowed,
        invalid_reason=invalid_reason,
    )


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
        AngularPDConfig(kp_x=1.0, kp_y=0.0, max_control_angle_rad=10.0, max_step_counts=80),
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


def test_angular_controller_derivative_uses_seconds() -> None:
    fast = AngularPDController(
        AngularPDConfig(
            kp_x=0.0,
            kp_y=0.0,
            kd_x_s=1.0,
            far_kd_scale=1.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )
    slow = AngularPDController(
        AngularPDConfig(
            kp_x=0.0,
            kp_y=0.0,
            kd_x_s=1.0,
            far_kd_scale=1.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )

    assert math.isclose(fast.update(_angular_error(error_x_rad=0.1, dt_s=0.01)).d_x_rad, 0.0)
    assert math.isclose(fast.update(_angular_error(error_x_rad=0.2, dt_s=0.01)).d_x_rad, 10.0)
    assert math.isclose(slow.update(_angular_error(error_x_rad=0.1, dt_s=0.02)).d_x_rad, 0.0)
    assert math.isclose(slow.update(_angular_error(error_x_rad=0.2, dt_s=0.02)).d_x_rad, 5.0)


def test_angular_controller_resets_derivative_on_switch_committed_track_change() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=0.0,
            kp_y=0.0,
            kd_x_s=1.0,
            far_kd_scale=1.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )

    controller.update(_angular_error(track_id=1, error_x_rad=0.1, dt_s=0.01))
    assert math.isclose(
        controller.update(_angular_error(track_id=1, error_x_rad=0.2, dt_s=0.01)).d_x_rad,
        10.0,
    )
    switched = controller.update(_angular_error(track_id=2, error_x_rad=-0.2, dt_s=0.01))

    assert math.isclose(switched.d_x_rad, 0.0)
    assert switched.debug["derivative_reset_reason"] == "SWITCH_COMMITTED"
    assert controller.memory.active_track_id == 2


def test_angular_controller_resets_state_on_calibration_change() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=0.0,
            kd_x_s=1.0,
            max_control_angle_rad=10.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(profile_id="old", counts_per_360_x=2 * math.pi),
    )
    first = controller.update(_angular_error(track_id=1, error_x_rad=0.4, dt_s=0.01))
    assert math.isclose(first.residual_x_counts, 0.4)

    controller.calibration = CalibrationProfile(profile_id="new", counts_per_360_x=2 * math.pi)
    after_change = controller.update(_angular_error(track_id=1, error_x_rad=0.0, dt_s=0.01))

    assert math.isclose(after_change.d_x_rad, 0.0)
    assert math.isclose(after_change.residual_x_counts, 0.0)
    assert after_change.debug["derivative_reset_reason"] == "CALIBRATION_CHANGED"


def test_angular_controller_resets_state_on_control_geometry_change() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=0.0,
            kd_x_s=1.0,
            max_control_angle_rad=10.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )
    first = controller.update(
        _angular_error(track_id=1, error_x_rad=0.4, dt_s=0.01, control_width_px=100)
    )
    assert math.isclose(first.residual_x_counts, 0.4)

    after_change = controller.update(
        _angular_error(track_id=1, error_x_rad=0.0, dt_s=0.01, control_width_px=200)
    )

    assert math.isclose(after_change.d_x_rad, 0.0)
    assert math.isclose(after_change.residual_x_counts, 0.0)
    assert after_change.debug["derivative_reset_reason"] == "CONTROL_GEOMETRY_CHANGED"


def test_angular_controller_resets_state_when_control_disallowed() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=0.0,
            kd_x_s=1.0,
            max_control_angle_rad=10.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )
    controller.update(_angular_error(track_id=1, error_x_rad=0.4, dt_s=0.01))
    assert controller.memory.initialized is True

    blocked = controller.update(
        _angular_error(
            track_id=1,
            error_x_rad=0.5,
            dt_s=0.01,
            control_allowed=False,
            invalid_reason="CONTROL_DT_INVALID",
        )
    )

    assert blocked.dx == 0
    assert blocked.dy == 0
    assert blocked.debug["invalid_reason"] == "CONTROL_DT_INVALID"
    assert controller.memory.initialized is False
    assert controller.memory.residual_x_counts == 0.0


def test_angular_controller_derivative_ema_can_hot_update() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=0.0,
            kp_y=0.0,
            kd_x_s=1.0,
            derivative_ema_alpha=1.0,
            far_kd_scale=1.0,
            dt_min_s=0.001,
            dt_max_s=1.0,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi),
    )
    controller.update(_angular_error(track_id=1, error_x_rad=0.1, dt_s=0.01))
    second = controller.update(_angular_error(track_id=1, error_x_rad=0.2, dt_s=0.01))
    assert math.isclose(second.d_x_rad, 10.0)

    controller.config = AngularPDConfig(
        kp_x=0.0,
        kp_y=0.0,
        kd_x_s=1.0,
        derivative_ema_alpha=0.5,
        far_kd_scale=1.0,
        dt_min_s=0.001,
        dt_max_s=1.0,
    )
    after_update = controller.update(_angular_error(track_id=1, error_x_rad=0.4, dt_s=0.01))

    assert math.isclose(after_update.d_x_rad, 15.0)
    assert after_update.debug["derivative_ema_alpha"] == 0.5


def test_angular_controller_applies_angular_vector_clamp_before_counts() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=1.0,
            far_kp_scale=1.0,
            max_control_angle_rad=0.5,
            max_step_counts=100,
            max_counts_delta_x=100,
            max_counts_delta_y=100,
        ),
        CalibrationProfile(counts_per_360_x=2 * math.pi, counts_per_360_y=2 * math.pi),
    )

    out = controller.update(_angular_error(error_x_rad=1.0, error_y_rad=0.0))

    assert out.dx == 0
    assert math.isclose(out.residual_x_counts, 0.5)
    assert math.isclose(out.out_x_rad, 0.5)
    assert out.debug["angular_vector_clipped"] is True


def test_angular_controller_applies_counts_vector_clamp_after_residual() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=1.0,
            far_kp_scale=1.0,
            max_control_angle_rad=10.0,
            max_step_counts=10,
            max_counts_delta_x=100,
            max_counts_delta_y=100,
        ),
        CalibrationProfile(counts_per_360_x=200 * math.pi, counts_per_360_y=200 * math.pi),
    )

    out = controller.update(_angular_error(error_x_rad=0.1, error_y_rad=0.1))

    assert math.hypot(out.dx, out.dy) <= 10
    assert out.debug["counts_vector_clipped"] is True


def test_angular_controller_applies_slew_rate_limit_after_counts_clamp() -> None:
    controller = AngularPDController(
        AngularPDConfig(
            kp_x=1.0,
            kp_y=0.0,
            far_kp_scale=1.0,
            max_control_angle_rad=10.0,
            max_step_counts=100,
            max_counts_delta_x=5,
            max_counts_delta_y=5,
        ),
        CalibrationProfile(counts_per_360_x=200 * math.pi),
    )

    first = controller.update(_angular_error(track_id=1, error_x_rad=0.1))
    second = controller.update(_angular_error(track_id=1, error_x_rad=-0.1))

    assert first.dx == 5
    assert second.dx == 0
    assert second.debug["slew_rate_limited"] is True


def test_experimental_strategy_rejects_missing_full_geometry() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False)
    target = Detection(cls=0, score=0.9, x=150, y=150, w=20, h=20)

    command = strategy.calculate(
        target,
        (160, 160),
        BoxInputState(
            left=True,
            raw=_compensated_target_raw(
                capture_width=0,
                capture_height=0,
                roi_offset_x=0,
                roi_offset_y=0,
                roi_x=160,
                roi_y=160,
            ),
        ),
    )

    assert command.dx == 0
    assert command.dy == 0
    assert command.reason == "CONTROL_GEOMETRY_INVALID"


def test_experimental_strategy_rejects_untrusted_roi_geometry_without_runtime_config() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False)
    target = Detection(cls=0, score=0.9, x=150, y=150, w=20, h=20)
    raw = _compensated_target_raw(capture_width=320, capture_height=320)
    raw["capture_geometry_trusted"] = False
    raw["capture_geometry_source"] = "missing_source_geometry"

    command = strategy.calculate(target, (160, 160), BoxInputState(left=True, raw=raw))

    assert command.dx == 0
    assert command.dy == 0
    assert command.reason == "CONTROL_GEOMETRY_INVALID"
    assert command.debug["capture_size_source"] == "untrusted_frame_metadata"


def test_experimental_strategy_uses_runtime_config_when_frame_geometry_untrusted() -> None:
    strategy = ExperimentalAnglePidStrategy(
        kalman_enabled=False,
        capture_width=1920,
        capture_height=1080,
    )
    target = Detection(cls=0, score=0.9, x=150, y=150, w=20, h=20)
    raw = _compensated_target_raw(capture_width=320, capture_height=320)
    raw["capture_geometry_trusted"] = False
    raw["capture_geometry_source"] = "missing_source_geometry"

    command = strategy.calculate(target, (160, 160), BoxInputState(left=True, raw=raw))

    assert command.reason.startswith("experimental angle pid")
    assert command.debug["capture_width"] == 1920
    assert command.debug["capture_height"] == 1080
    assert command.debug["capture_size_source"] == "runtime_config"


def test_experimental_strategy_requires_compensated_target_contract() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False)
    target = Detection(cls=0, score=0.9, x=150, y=150, w=20, h=20)

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

    assert command.dx == 0
    assert command.dy == 0
    assert command.reason == "COMPENSATED_TARGET_REQUIRED"
    assert command.debug["missing_contract"] == "compensated_target"


def test_experimental_strategy_maps_roi_target_to_full_control_coordinates() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False, kp_x=0.35, kp_y=0.35)
    target = Detection(cls=0, score=0.9, x=160, y=160, w=20, h=20)

    command = strategy.calculate(
        target,
        (160, 160),
        BoxInputState(
            left=True,
            raw=_compensated_target_raw(),
        ),
    )

    assert command.dx > 0
    assert command.dy > 0
    assert command.debug["comp_x"] == 970
    assert command.debug["comp_y"] == 550
    assert command.debug["capture_size_source"] == "frame_metadata"
    assert command.debug["error_px"] == {
        "x": command.debug["error_x_px"],
        "y": command.debug["error_y_px"],
    }
    assert command.debug["error_rad"] == {
        "x": command.debug["error_x_rad"],
        "y": command.debug["error_y_rad"],
    }
    assert command.debug["u_rad"] == {
        "x": command.debug["out_x_rad"],
        "y": command.debug["out_y_rad"],
    }
    assert command.debug["raw_counts"] == {
        "dx": command.debug["raw_dx_counts"],
        "dy": command.debug["raw_dy_counts"],
    }
    assert command.debug["final_counts"] == {"dx": command.dx, "dy": command.dy}


def test_experimental_strategy_control_error_is_independent_of_roi_size() -> None:
    target = Detection(cls=0, score=0.9, x=160, y=160, w=20, h=20)
    small_roi = _compensated_target_raw(
        roi_width=320,
        roi_height=320,
        roi_offset_x=800,
        roi_offset_y=380,
        roi_x=260,
        roi_y=160,
    )
    large_roi = _compensated_target_raw(
        roi_width=640,
        roi_height=640,
        roi_offset_x=640,
        roi_offset_y=220,
        roi_x=420,
        roi_y=320,
    )

    small = ExperimentalAnglePidStrategy(kalman_enabled=False).calculate(
        target,
        (160, 160),
        BoxInputState(left=True, raw=small_roi),
    )
    large = ExperimentalAnglePidStrategy(kalman_enabled=False).calculate(
        target,
        (320, 320),
        BoxInputState(left=True, raw=large_roi),
    )

    assert small.debug["comp_x"] == large.debug["comp_x"] == 1060
    assert small.debug["comp_y"] == large.debug["comp_y"] == 540
    assert math.isclose(small.debug["error_x_rad"], large.debug["error_x_rad"])
    assert math.isclose(small.debug["error_y_rad"], large.debug["error_y_rad"])


def test_experimental_strategy_uses_calibration_profile_for_counts_and_axis() -> None:
    strategy = ExperimentalAnglePidStrategy(
        kalman_enabled=False,
        kp_x=1.0,
        kp_y=1.0,
        calibration_profile_id="arena",
        calibration_profile_version=7,
        counts_per_360_x=9980,
        counts_per_360_y=4990,
        axis_sign_x=-1,
        axis_sign_y=1,
    )
    target = Detection(cls=0, score=0.9, x=160, y=160, w=20, h=20)

    command = strategy.calculate(
        target,
        (160, 160),
        BoxInputState(
            left=True,
            raw=_compensated_target_raw(),
        ),
    )

    assert command.dx < 0
    assert command.dy > 0
    assert command.debug["calibration_profile_id"] == "arena"
    assert command.debug["calibration_profile_version"] == 7
    assert command.debug["counts_per_360_x"] == 9980
    assert command.debug["counts_per_360_y"] == 4990
    assert command.debug["axis_sign_x"] == -1
