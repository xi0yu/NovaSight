from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from novasight.control.angular import (
    AngularErrorMapper,
    AngularPDConfig,
    AngularPDController,
    CalibrationProfile,
)
from novasight.hardware import BoxInputState
from novasight.contracts import ControlIntent, Detection, Track


Target = Detection | Track
MOVE_KINDS = {"raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"}


@dataclass(frozen=True)
class MoveCommand:
    dx: float
    dy: float
    confidence: float
    reason: str
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0
    bezier_ctrl: tuple[int, int, int, int] | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class IControlStrategy(Protocol):
    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        ...


class ExperimentalAnglePidStrategy:
    def __init__(
        self,
        *,
        kp_x: float = 0.35,
        kp_y: float = 0.24,
        ki: float = 0.0,
        kd: float = 0.0,
        integral_limit: float = 0.0,
        speed: float = 1.0,
        smooth_factor: float = 0.0,
        deadzone_px: float = 0.0,
        derivative_filter: float = 1.0,
        near_error_deg: float = 0.35,
        far_error_deg: float = 2.50,
        near_kp_scale: float = 0.35,
        middle_kp_scale: float = 0.70,
        far_kp_scale: float = 1.00,
        near_kd_scale: float = 1.00,
        middle_kd_scale: float = 0.80,
        far_kd_scale: float = 0.50,
        prediction_gain_min: float = 0.35,
        prediction_d_gain_min: float = 0.25,
        max_control_angle_deg: float = 3.0,
        calibration_profile_id: str = "default",
        calibration_profile_version: int = 1,
        fov_semantics: str = "horizontal",
        fov_x_deg: float = 105.0,
        counts_per_360_x: float = 9980.0,
        counts_per_360_y: float = 9980.0,
        axis_sign_x: float = 1.0,
        axis_sign_y: float = 1.0,
        game_sensitivity_fingerprint: str = "unverified-default",
        projection_profile: str = "fixed_horizontal_fov",
        max_step_counts: float = 80.0,
        max_counts_delta_x: float = 35.0,
        max_counts_delta_y: float = 35.0,
        control_hz: float = 60.0,
        kalman_enabled: bool = True,
        kalman_process_noise: float = 2.0,
        kalman_measurement_noise: float = 16.0,
        hungarian_enabled: bool = True,
        matching_distance_px: float = 140.0,
        max_extrapolate_frames: int = 3,
        target_filter_enabled: bool = True,
        target_filter_min_score: float = 0.0,
        target_filter_fov_ratio: float = 1.0,
        target_filter_same_class: bool = False,
        prediction_lead_ms: float = 0.0,
        extrapolate_confidence_decay: float = 1.0,
        magnet_enabled: bool = False,
        magnet_radius_px: float = 120.0,
        magnet_strength: float = 0.25,
        magnet_curve: float = 1.0,
        magnet_deadzone_px: float = 0.0,
        magnet_max_counts: float = 20.0,
        capture_width: float = 0.0,
        capture_height: float = 0.0,
        move_kind: str = "raw",
        move_ms: int = 0,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.speed = max(0.0, speed)
        self.deadzone_px = max(0.0, deadzone_px)
        self.fov_x_deg = max(1.0, min(179.0, fov_x_deg))
        self.counts_per_360_x = max(1.0, counts_per_360_x)
        self.counts_per_360_y = max(1.0, counts_per_360_y)
        self.max_step_counts = max(1.0, max_step_counts)
        self.control_hz = max(1.0, control_hz)
        self.axis_sign_x = -1.0 if axis_sign_x < 0 else 1.0
        self.axis_sign_y = -1.0 if axis_sign_y < 0 else 1.0
        self.magnet_enabled = bool(magnet_enabled)
        self.magnet_radius_px = max(1.0, magnet_radius_px)
        self.magnet_strength = max(0.0, magnet_strength)
        self.magnet_curve = max(0.1, magnet_curve)
        self.magnet_deadzone_px = max(0.0, magnet_deadzone_px)
        self.magnet_max_counts = max(0.0, magnet_max_counts)
        self.capture_width = max(0.0, capture_width)
        self.capture_height = max(0.0, capture_height)
        self.move_kind = move_kind if move_kind in MOVE_KINDS else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self.calibration = CalibrationProfile(
            profile_id=calibration_profile_id,
            profile_version=calibration_profile_version,
            fov_semantics=fov_semantics,
            fov_x_deg=self.fov_x_deg,
            counts_per_360_x=self.counts_per_360_x,
            counts_per_360_y=self.counts_per_360_y,
            axis_sign_x=self.axis_sign_x,
            axis_sign_y=self.axis_sign_y,
            game_sensitivity_fingerprint=game_sensitivity_fingerprint,
            projection_profile=projection_profile,
        ).normalized()
        self.error_mapper = AngularErrorMapper(self.calibration)
        self.angular_controller = AngularPDController(
            AngularPDConfig(
                kp_x=max(0.0, kp_x) * self.speed,
                kp_y=max(0.0, kp_y) * self.speed,
                kd_x_s=kd,
                kd_y_s=kd,
                derivative_ema_alpha=derivative_filter,
                near_error_rad=math.radians(max(0.0, near_error_deg)),
                far_error_rad=math.radians(max(0.0, far_error_deg)),
                near_kp_scale=max(0.0, near_kp_scale),
                middle_kp_scale=max(0.0, middle_kp_scale),
                far_kp_scale=max(0.0, far_kp_scale),
                near_kd_scale=max(0.0, near_kd_scale),
                middle_kd_scale=max(0.0, middle_kd_scale),
                far_kd_scale=max(0.0, far_kd_scale),
                prediction_gain_min=max(0.0, min(1.0, prediction_gain_min)),
                prediction_d_gain_min=max(0.0, min(1.0, prediction_d_gain_min)),
                max_control_angle_rad=math.radians(max(0.001, max_control_angle_deg)),
                max_step_counts=self.max_step_counts,
                max_counts_delta_x=max(0.0, max_counts_delta_x),
                max_counts_delta_y=max(0.0, max_counts_delta_y),
                dt_min_s=1.0 / 240.0,
                dt_max_s=1.0 / 15.0,
            ),
            self.calibration,
        )
        del (
            smooth_factor,
            kalman_enabled,
            kalman_process_noise,
            kalman_measurement_noise,
            hungarian_enabled,
            matching_distance_px,
            max_extrapolate_frames,
            target_filter_enabled,
            target_filter_min_score,
            target_filter_fov_ratio,
            target_filter_same_class,
            prediction_lead_ms,
            extrapolate_confidence_decay,
        )

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        raw = box_input.raw or {}
        roi_width = _positive_number(raw.get("roi_width"), current_pos[0] * 2.0)
        roi_height = _positive_number(raw.get("roi_height"), current_pos[1] * 2.0)
        capture_width, capture_height, capture_source = self._capture_dimensions(raw, roi_width, roi_height)
        roi_offset_x = _number(raw.get("roi_offset_x"), 0.0)
        roi_offset_y = _number(raw.get("roi_offset_y"), 0.0)
        raw_aim_x = (float(target.x1) + float(target.x2)) * 0.5
        raw_aim_y = (float(target.y1) + float(target.y2)) * 0.5
        external_target = raw.get("compensated_target")
        external_compensated_target = external_target if isinstance(external_target, dict) else None
        if external_compensated_target is None:
            self.angular_controller.reset()
            return MoveCommand(
                dx=0,
                dy=0,
                confidence=0.0,
                reason="COMPENSATED_TARGET_REQUIRED",
                move_kind=self.move_kind,
                move_ms=self.move_ms,
                trace_ms=self.trace_ms,
                debug={
                    "stage": "experimental_angle_pid",
                    "algorithm": "experimental_angle_pid",
                    "unit_pipeline": "compensated_control_px_to_angle_rad_to_counts",
                    "control_allowed": False,
                    "missing_contract": "compensated_target",
                    "raw_aim_x": raw_aim_x,
                    "raw_aim_y": raw_aim_y,
                    "roi_width": roi_width,
                    "roi_height": roi_height,
                    "capture_width": capture_width,
                    "capture_height": capture_height,
                    "capture_size_source": capture_source,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )
        aim_x = _number(external_compensated_target.get("roi_x"), raw_aim_x)
        aim_y = _number(external_compensated_target.get("roi_y"), raw_aim_y)
        comp_x = _number(external_compensated_target.get("control_x"), roi_offset_x + aim_x)
        comp_y = _number(external_compensated_target.get("control_y"), roi_offset_y + aim_y)
        tracker_debug = {
            "enabled": False,
            "used": True,
            "external_compensated_target": True,
            "new_observation": not bool(external_compensated_target.get("predicted_source", False)),
            "dt": 1.0 / self.control_hz,
            "prediction_confidence": _number(external_compensated_target.get("prediction_confidence"), 1.0),
            "reason": str(external_compensated_target.get("reason") or ""),
        }
        if not bool(external_compensated_target.get("control_allowed", True)):
            self.angular_controller.reset()
            return MoveCommand(
                dx=0,
                dy=0,
                confidence=0.0,
                reason=str(external_compensated_target.get("reason") or "compensated target disallowed"),
                move_kind=self.move_kind,
                move_ms=self.move_ms,
                trace_ms=self.trace_ms,
                debug={
                    "stage": "experimental_angle_pid",
                    "algorithm": "experimental_angle_pid",
                    "unit_pipeline": "compensated_control_px_to_angle_rad_to_counts",
                    "control_allowed": False,
                    "tracker": tracker_debug,
                    "aim_point": raw.get("aim_point"),
                    "compensated_target": external_compensated_target,
                    "latency_compensation": raw.get("latency_compensation"),
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )
        roi_center_x = roi_width * 0.5
        roi_center_y = roi_height * 0.5
        dt = self._debug_dt(tracker_debug)
        source_frame_id = self._raw_frame_id(raw)
        raw_target_key = raw.get("target_key")
        track_id = _stable_int(raw_target_key) if raw_target_key is not None else None
        angular_error = self.error_mapper.map(
            comp_x=comp_x,
            comp_y=comp_y,
            control_width_px=capture_width,
            control_height_px=capture_height,
            dt_s=dt,
            source_frame_id=source_frame_id,
            track_id=track_id,
            predicted_source=bool(tracker_debug.get("new_observation") is False),
            prediction_confidence=float(
                tracker_debug.get("prediction_confidence")
                or tracker_debug.get("extrapolate_confidence_decay")
                or 1.0
            ),
        )
        if not angular_error.control_allowed:
            self.angular_controller.reset()
            return MoveCommand(
                dx=0,
                dy=0,
                confidence=0.0,
                reason=angular_error.invalid_reason or "angular control invalid",
                move_kind=self.move_kind,
                move_ms=self.move_ms,
                trace_ms=self.trace_ms,
                debug={
                    "stage": "experimental_angle_pid",
                    "algorithm": "experimental_angle_pid",
                    "unit_pipeline": "compensated_control_px_to_angle_rad_to_counts",
                    "control_allowed": False,
                    "tracker": tracker_debug,
                    "aim_point": raw.get("aim_point"),
                    "compensated_target": external_compensated_target,
                    "latency_compensation": raw.get("latency_compensation"),
                    "roi_width": roi_width,
                    "roi_height": roi_height,
                    "capture_width": capture_width,
                    "capture_height": capture_height,
                    "capture_size_source": capture_source,
                    "invalid_reason": angular_error.invalid_reason,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )
        deadzone_rad = (
            math.atan(self.deadzone_px / max(1e-6, min(angular_error.focal_x_px, angular_error.focal_y_px)))
            if self.deadzone_px > 0
            else 0.0
        )
        if deadzone_rad != self.angular_controller.config.deadzone_rad:
            self.angular_controller.config = replace(self.angular_controller.config, deadzone_rad=deadzone_rad)
        raw_tracker_decay = (
            tracker_debug.get("prediction_confidence")
            if tracker_debug.get("prediction_confidence") is not None
            else tracker_debug.get("extrapolate_confidence_decay")
        )
        tracker_confidence_decay = (
            max(0.0, min(1.0, float(raw_tracker_decay)))
            if isinstance(raw_tracker_decay, (int, float)) and not isinstance(raw_tracker_decay, bool)
            else 1.0
        )
        output = self.angular_controller.update(angular_error)
        dx_i = output.dx
        dy_i = output.dy
        bezier_ctrl = _bezier_ctrl(dx_i, dy_i, self.bezier_curvature) if self.move_kind in {"bezier", "enc_bezier"} else None
        return MoveCommand(
            dx=dx_i,
            dy=dy_i,
            confidence=float(target.score),
            reason=f"experimental angle pid px={angular_error.error_norm_px:.1f}",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                "stage": "experimental_angle_pid",
                "algorithm": "experimental_angle_pid",
                "unit_pipeline": "compensated_control_px_to_angle_rad_to_counts",
                "control_allowed": True,
                "coordinate_y": "image_down_positive_until_calibration_axis_sign",
                "raw_aim_x": raw_aim_x,
                "raw_aim_y": raw_aim_y,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "aim_point": raw.get("aim_point"),
                "compensated_target": external_compensated_target,
                "latency_compensation": raw.get("latency_compensation"),
                "comp_x": comp_x,
                "comp_y": comp_y,
                "roi_offset_x": roi_offset_x,
                "roi_offset_y": roi_offset_y,
                "tracker": tracker_debug,
                "roi_center_x": roi_center_x,
                "roi_center_y": roi_center_y,
                "roi_width": roi_width,
                "roi_height": roi_height,
                "capture_width": capture_width,
                "capture_height": capture_height,
                "capture_size_source": capture_source,
                "error_x_px": angular_error.error_x_px,
                "error_y_px": angular_error.error_y_px,
                "error_px": {"x": angular_error.error_x_px, "y": angular_error.error_y_px},
                "error_y_image_down_px": angular_error.error_y_px,
                "fov_x_deg": self.fov_x_deg,
                "fov_x_rad": angular_error.fov_x_rad,
                "fov_y_rad": angular_error.fov_y_rad,
                "focal_x": angular_error.focal_x_px,
                "focal_y": angular_error.focal_y_px,
                "error_x_rad": angular_error.error_x_rad,
                "error_y_rad": angular_error.error_y_rad,
                "error_rad": {"x": angular_error.error_x_rad, "y": angular_error.error_y_rad},
                "error_x_deg": math.degrees(angular_error.error_x_rad),
                "error_y_deg": math.degrees(angular_error.error_y_rad),
                "out_x_rad": output.out_x_rad,
                "out_y_rad": output.out_y_rad,
                "u_rad": {"x": output.out_x_rad, "y": output.out_y_rad},
                "calibration_profile_id": self.calibration.profile_id,
                "calibration_profile_version": self.calibration.profile_version,
                "projection_profile": self.calibration.projection_profile,
                "game_sensitivity_fingerprint": self.calibration.game_sensitivity_fingerprint,
                "counts_per_360_x": self.calibration.counts_per_360_x,
                "counts_per_360_y": self.calibration.counts_per_360_y,
                "counts_per_rad_x": output.counts_per_rad_x,
                "counts_per_rad_y": output.counts_per_rad_y,
                "speed": self.speed,
                "deadzone_px": self.deadzone_px,
                "deadzone_rad": deadzone_rad,
                "in_deadzone": abs(angular_error.error_x_rad) <= deadzone_rad and abs(angular_error.error_y_rad) <= deadzone_rad,
                "pid_dx_counts": output.raw_x_counts,
                "pid_dy_counts": output.raw_y_counts,
                "magnet": {"enabled": self.magnet_enabled, "applied": False, "reason": "disabled by angular control contract"},
                "magnet_dx_counts": 0.0,
                "magnet_dy_counts": 0.0,
                "tracker_confidence_decay": tracker_confidence_decay,
                "raw_dx_counts": output.raw_x_counts,
                "raw_dy_counts": output.raw_y_counts,
                "raw_counts": {"dx": output.raw_x_counts, "dy": output.raw_y_counts},
                "accum_x_counts": output.accum_x_counts,
                "accum_y_counts": output.accum_y_counts,
                "residual_x_counts": output.residual_x_counts,
                "residual_y_counts": output.residual_y_counts,
                "residual_counts": {"x": output.residual_x_counts, "y": output.residual_y_counts},
                "angular_controller": output.debug,
                "axis_sign_x": self.calibration.axis_sign_x,
                "axis_sign_y": self.calibration.axis_sign_y,
                "max_step_counts": self.max_step_counts,
                "dt": dt,
                "control_hz": self.control_hz,
                "p_x": output.p_x_rad,
                "p_y": output.p_y_rad,
                "i_x": 0.0,
                "i_y": 0.0,
                "d_x": output.d_x_rad,
                "d_y": output.d_y_rad,
                "final_dx": dx_i,
                "final_dy": dy_i,
                "final_counts": {"dx": dx_i, "dy": dy_i},
            },
        )

    def observe(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> dict[str, Any]:
        raw = box_input.raw or {}
        external_target = raw.get("compensated_target")
        if isinstance(external_target, dict):
            return {
                "enabled": False,
                "used": True,
                "external_compensated_target": True,
                "new_observation": not bool(external_target.get("predicted_source", False)),
                "prediction_confidence": _number(external_target.get("prediction_confidence"), 1.0),
                "reason": str(external_target.get("reason") or ""),
                "aim_point": raw.get("aim_point"),
                "compensated_target": external_target,
                "latency_compensation": raw.get("latency_compensation"),
            }
        return {
            "enabled": False,
            "used": False,
            "external_compensated_target": False,
            "reason": "COMPENSATED_TARGET_REQUIRED",
            "missing_contract": "compensated_target",
        }

    def reset(self) -> None:
        self.angular_controller.reset()

    def _capture_dimensions(self, raw: dict[str, object], roi_width: float, roi_height: float) -> tuple[float, float, str]:
        raw_capture_width = _positive_number(raw.get("capture_width"), 0.0)
        raw_capture_height = _positive_number(raw.get("capture_height"), 0.0)
        raw_trusted = raw.get("capture_geometry_trusted")
        if raw_capture_width > 0 and raw_capture_height > 0 and raw_trusted is not False:
            source = raw.get("capture_geometry_source")
            source_label = str(source).strip() if source is not None else ""
            return raw_capture_width, raw_capture_height, source_label or "frame_metadata"
        if self.capture_width > 0 and self.capture_height > 0:
            return self.capture_width, self.capture_height, "runtime_config"
        if raw_trusted is False:
            return 0.0, 0.0, "untrusted_frame_metadata"
        return 0.0, 0.0, "missing_control_geometry"

    def _debug_dt(self, debug: dict[str, Any]) -> float:
        value = debug.get("dt")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return self._clamp_dt(float(value))
        return 1.0 / self.control_hz

    def _raw_frame_id(self, raw: dict[str, object]) -> int | None:
        value = raw.get("frame_id")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        return None

    @staticmethod
    def _clamp_dt(dt: float) -> float:
        return max(1.0 / 240.0, min(1.0 / 15.0, float(dt)))


def _bezier_ctrl(dx: int, dy: int, curvature: float) -> tuple[int, int, int, int]:
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (0, 0, 0, 0)
    nx = -dy / dist
    ny = dx / dist
    bow = curvature * dist
    return (
        int(round(dx / 3.0 + nx * bow)),
        int(round(dy / 3.0 + ny * bow)),
        int(round(dx * 2.0 / 3.0 + nx * bow)),
        int(round(dy * 2.0 / 3.0 + ny * bow)),
    )


def _positive_number(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return max(0.0, float(fallback))


def _number(value: object, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return float(fallback)


def _stable_int(value: object) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        for part in value.split(":"):
            try:
                return int(part)
            except ValueError:
                continue
    return None
