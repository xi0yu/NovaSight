from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from novasight.hardware import BoxInputState
from novasight.contracts import ControlIntent, Detection, Track


Target = Detection | Track


def aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, aim_ratio)) / 100.0
    return (float(target.x) + float(target.w) / 2.0, float(target.y) + float(target.h) * ratio)


def aim_point_y_ratio(target: Target, aim_y_ratio: float) -> tuple[float, float]:
    return aim_point(target, aim_y_ratio)


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


class PIDStrategy:
    def __init__(
        self,
        *,
        kp: float = 0.35,
        ki: float = 0.0,
        kd: float = 0.04,
        kp_x: float | None = None,
        kp_y: float | None = None,
        integral_limit: float = 250.0,
        move_limit: float | None = None,
        move_limit_x: float | None = None,
        move_limit_y: float | None = None,
        derivative_alpha: float = 0.35,
        prediction_factor: float = 0.1,
        prediction_stationary_px: float = 1.5,
        prediction_moving_px: float = 12.0,
        aim_ratio: float = 40.0,
        near_px: float = 24.0,
        near_speed: float = 0.16,
        far_speed: float = 0.42,
        deadzone_counts: int = 1,
        counts_per_revolution_x: float = 4096.0,
        counts_per_revolution_y: float = 4096.0,
        move_kind: str = "raw",
        move_ms: int = 0,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.kp_x = kp if kp_x is None else kp_x
        self.kp_y = kp if kp_y is None else kp_y
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.move_limit = None if move_limit is None else abs(move_limit)
        self.move_limit_x = (
            self.move_limit if move_limit_x is None else abs(move_limit_x)
        )
        self.move_limit_y = (
            self.move_limit if move_limit_y is None else abs(move_limit_y)
        )
        self.derivative_alpha = max(0.0, min(1.0, derivative_alpha))
        self.prediction_factor = max(0.0, prediction_factor)
        self.prediction_stationary_px = max(0.0, prediction_stationary_px)
        self.prediction_moving_px = max(
            self.prediction_stationary_px + 1.0,
            prediction_moving_px,
        )
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self.near_px = max(0.0, near_px)
        self.near_speed = max(0.0, near_speed)
        self.far_speed = max(0.0, far_speed)
        self.deadzone_counts = max(0, int(deadzone_counts))
        self.counts_per_revolution_x = max(1.0, counts_per_revolution_x)
        self.counts_per_revolution_y = max(1.0, counts_per_revolution_y)
        self.move_kind = move_kind if move_kind in {"raw", "auto", "bezier"} else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._ix = 0.0
        self._iy = 0.0
        self._last_error: tuple[float, float] | None = None
        self._last_derivative = (0.0, 0.0)
        self._last_center: tuple[float, float] | None = None
        self._last_s: float | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            self._reset_motion_state()
            return MoveCommand(
                0,
                0,
                0,
                "hardware trigger inactive",
                debug={"stage": "trigger", "coordinate_y": "cartesian_up_positive"},
            )
        aim_center = aim_point(target, self.aim_ratio)
        predicted_center, prediction_weight = self._predict_center(aim_center)
        ex_px = predicted_center[0] - current_pos[0]
        ey_px = current_pos[1] - predicted_center[1]
        err_px = math.hypot(ex_px, ey_px)
        fov_counts_x, fov_counts_y = self._pixel_error_to_counts(ex_px, ey_px)
        speed = self._speed_for(err_px, min(current_pos[0], current_pos[1]) * 2)
        ex = fov_counts_x * speed
        ey = fov_counts_y * speed
        now_s = time.monotonic()
        dt = now_s - self._last_s if self._last_s is not None else 1.0 / 60.0
        self._last_s = now_s
        if not (0.001 < dt < 0.5):
            dt = 1.0 / 60.0
        self._ix = max(-self.integral_limit, min(self.integral_limit, self._ix + ex * dt))
        self._iy = max(-self.integral_limit, min(self.integral_limit, self._iy + ey * dt))
        if self._last_error is None:
            raw_dx = raw_dy = 0.0
        else:
            raw_dx = (ex - self._last_error[0]) / dt
            raw_dy = (ey - self._last_error[1]) / dt
        dx_d = self.derivative_alpha * raw_dx + (1 - self.derivative_alpha) * self._last_derivative[0]
        dy_d = self.derivative_alpha * raw_dy + (1 - self.derivative_alpha) * self._last_derivative[1]
        self._last_error = (ex, ey)
        self._last_derivative = (dx_d, dy_d)
        px = self.kp_x * ex
        py = self.kp_y * ey
        ix = self.ki * self._ix
        iy = self.ki * self._iy
        dx_term = self.kd * dx_d
        dy_term = self.kd * dy_d
        dx = px + ix + dx_term
        dy = py + iy + dy_term
        debug = {
            "stage": "pid_counts_pipeline",
            "coordinate_y": "cartesian_up_positive",
            "raw_px_x": ex_px,
            "raw_px_y": ey_px,
            "aim_ratio": self.aim_ratio,
            "aim_x": aim_center[0],
            "aim_y": aim_center[1],
            "predicted_x": predicted_center[0],
            "predicted_y": predicted_center[1],
            "fov_counts_x": fov_counts_x,
            "fov_counts_y": fov_counts_y,
            "speed": speed,
            "work_counts_x": ex,
            "work_counts_y": ey,
            "dt": dt,
            "p_x": px,
            "p_y": py,
            "i_x": ix,
            "i_y": iy,
            "d_x": dx_term,
            "d_y": dy_term,
        }
        if abs(ex) <= self.deadzone_counts and abs(ey) <= self.deadzone_counts:
            return MoveCommand(
                0,
                0,
                target.score,
                "pid deadzone",
                debug={**debug, "final_dx": 0.0, "final_dy": 0.0},
            )
        dx = self._minimum_effective_step(dx, ex)
        dy = self._minimum_effective_step(dy, ey)
        dx = self._clamp_axis(dx, self.move_limit_x)
        dy = self._clamp_axis(dy, self.move_limit_y)
        bezier_ctrl = _bezier_ctrl(int(round(dx)), int(round(dy)), self.bezier_curvature) if self.move_kind == "bezier" else None
        debug["final_dx"] = dx
        debug["final_dy"] = dy
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason=(
                f"pid counts strategy px={err_px:.1f}"
                if prediction_weight <= 0
                else f"pid counts strategy px={err_px:.1f} prediction={prediction_weight:.2f}"
            ),
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug=debug,
        )

    @staticmethod
    def _clamp_axis(value: float, limit: float | None) -> float:
        if limit is None:
            return value
        return max(-limit, min(limit, value))

    def _predict_center(self, center: tuple[float, float]) -> tuple[tuple[float, float], float]:
        if self._last_center is None or self.prediction_factor <= 0:
            self._last_center = center
            return center, 0.0

        vx = center[0] - self._last_center[0]
        vy = center[1] - self._last_center[1]
        self._last_center = center

        speed = math.hypot(vx, vy)
        if speed <= self.prediction_stationary_px:
            return center, 0.0
        span = self.prediction_moving_px - self.prediction_stationary_px
        motion_weight = min(1.0, (speed - self.prediction_stationary_px) / span)
        lead = self.prediction_factor * motion_weight
        return (center[0] + vx * lead, center[1] + vy * lead), motion_weight

    def _reset_motion_state(self) -> None:
        self._ix = 0.0
        self._iy = 0.0
        self._last_error = None
        self._last_derivative = (0.0, 0.0)
        self._last_center = None
        self._last_s = None

    def _pixel_error_to_counts(self, ex_px: float, ey_px: float) -> tuple[float, float]:
        rx = self.counts_per_revolution_x / (2.0 * math.pi)
        ry = self.counts_per_revolution_y / (2.0 * math.pi)
        counts_x = math.atan2(ex_px, rx) * rx
        counts_y = math.atan2(ey_px, math.sqrt(ex_px * ex_px + rx * rx)) * ry
        return counts_x, counts_y

    def _speed_for(self, err_px: float, fov_radius: float) -> float:
        if self.near_px <= 0:
            return self.far_speed
        if err_px <= self.near_px:
            return self.near_speed
        span = max(1.0, fov_radius - self.near_px)
        ratio = min(1.0, (err_px - self.near_px) / span)
        return self.near_speed + (self.far_speed - self.near_speed) * ratio

    @staticmethod
    def _minimum_effective_step(value: float, error_counts: float) -> float:
        if round(value) != 0 or abs(error_counts) < 1.0:
            return value
        return 1.0 if error_counts > 0 else -1.0


class StraightStrategy:
    VELOCITY_CLAMP_COUNTS_PER_SEC = 3000.0
    OVERSHOOT_GUARD_COUNTS = 2.0
    OVERSHOOT_VELOCITY_DAMP = 0.1
    MOTION_FLOOR_COUNTS_PER_SEC = 60.0
    FRAME_DT_FALLBACK = 1.0 / 60.0
    FRAME_DT_MIN = 0.001
    FRAME_DT_MAX = 0.35
    PREDICT_DT_MIN = 0.001
    PREDICT_DT_MAX = 0.05

    def __init__(
        self,
        *,
        aim_y_ratio: float = 40.0,
        fov_deg: float = 105.0,
        counts_per_revolution: float = 9980.0,
        kp_x: float = 0.3,
        kp_y: float = 0.3,
        first_frame_gain: float = 1.0,
        first_frame_max_step: float = 200.0,
        refine_max_step: float = 80.0,
        in_deadzone_px: float = 8.0,
        jump_threshold_px: float = 40.0,
        pred_gain: float = 0.0,
        pred_consistency_frames: int = 3,
        gain_y: float = 1.0,
        move_kind: str = "raw",
        move_ms: int = 12,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.aim_y_ratio = max(0.0, min(100.0, aim_y_ratio))
        self.fov_deg = max(1.0, min(179.0, fov_deg))
        self.counts_per_revolution = max(100.0, counts_per_revolution)
        self.kp_x = max(0.0, kp_x)
        self.kp_y = max(0.0, kp_y)
        self.first_frame_gain = max(0.0, first_frame_gain)
        self.first_frame_max_step = max(1.0, first_frame_max_step)
        self.refine_max_step = max(1.0, refine_max_step)
        self.in_deadzone_px = max(0.0, in_deadzone_px)
        self.jump_threshold_px = max(1.0, jump_threshold_px)
        self.pred_gain = max(0.0, pred_gain)
        self.pred_consistency_frames = max(1, int(pred_consistency_frames))
        self.gain_y = max(0.0, gain_y)
        self.move_kind = move_kind if move_kind in {"raw", "auto", "bezier"} else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._last_error_px: tuple[float, float] | None = None
        self._last_error_counts = (0.0, 0.0)
        self._last_output_counts = (0.0, 0.0)
        self._last_s: float | None = None
        self._has_previous_sample = False
        self._consistency_x = 0
        self._consistency_y = 0
        self._prev_vel_sign_x = 0
        self._prev_vel_sign_y = 0

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            self._reset()
            return MoveCommand(
                0,
                0,
                0,
                "hardware trigger inactive",
                debug={"stage": "straight_trigger", "coordinate_y": "cartesian_up_positive"},
            )

        aim_x, aim_y = aim_point_y_ratio(target, self.aim_y_ratio)
        dx_px = aim_x - current_pos[0]
        dy_px = current_pos[1] - aim_y
        now_s = time.monotonic()
        dt_s = self._frame_dt(now_s)
        acquire = self._is_acquire(dx_px, dy_px)
        projection = self._project_pixels_to_counts(dx_px, dy_px, current_pos[0] * 2.0)
        err_cx = projection["err_cx"]
        err_cy = projection["err_cy"]
        velocity_x, velocity_y = self._estimate_velocity_counts(err_cx, err_cy, dt_s)
        motion_coef_x, motion_coef_y = self._motion_coef(velocity_x, velocity_y)
        pred_dt = max(self.PREDICT_DT_MIN, min(self.PREDICT_DT_MAX, dt_s))
        pred_scale_x = max(0.0, min(1.0, abs(dx_px) / max(1.0, float(target.w) * 0.5)))
        pred_scale_y = max(0.0, min(1.0, abs(dy_px) / max(1.0, float(target.h) * 0.5)))
        lead_x = velocity_x * pred_dt * pred_scale_x * motion_coef_x * self.pred_gain
        lead_y = velocity_y * pred_dt * pred_scale_y * motion_coef_y * self.pred_gain
        fused_x = err_cx + lead_x
        fused_y = err_cy + lead_y
        if acquire:
            out_x = self._clamp(fused_x * self.first_frame_gain, self.first_frame_max_step)
            out_y = self._clamp(fused_y * self.first_frame_gain * self.gain_y, self.first_frame_max_step)
            stage = "straight_acquire"
        else:
            out_x = self._clamp(self.kp_x * err_cx + lead_x, self.refine_max_step)
            out_y = self._clamp(self.kp_y * err_cy + lead_y, self.refine_max_step)
            stage = "straight_refine"
        if abs(dx_px) < self.in_deadzone_px:
            out_x = 0.0
        if abs(dy_px) < self.in_deadzone_px:
            out_y = 0.0
        if out_x == 0.0 and out_y == 0.0:
            stage = "straight_hold"
        self._last_error_px = (dx_px, dy_px)
        if self._was_sent(box_input):
            self._last_output_counts = (out_x, out_y)
        else:
            self._last_output_counts = (0.0, 0.0)
        dx = int(round(out_x))
        dy = int(round(out_y))
        bezier_ctrl = _bezier_ctrl(dx, dy, self.bezier_curvature) if self.move_kind == "bezier" else None
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason=f"straight atan {stage} px={math.hypot(dx_px, dy_px):.1f}",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                **projection,
                "stage": stage,
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": dx_px,
                "raw_px_y": dy_px,
                "aim_ratio": self.aim_y_ratio,
                "aim_y_ratio": self.aim_y_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "dt": dt_s,
                "vel_x": velocity_x,
                "vel_y": velocity_y,
                "motion_coef_x": motion_coef_x,
                "motion_coef_y": motion_coef_y,
                "pred_scale_x": pred_scale_x,
                "pred_scale_y": pred_scale_y,
                "lead_x": lead_x,
                "lead_y": lead_y,
                "fused_x": fused_x,
                "fused_y": fused_y,
                "p_x": self.kp_x * err_cx,
                "p_y": self.kp_y * err_cy,
                "i_x": 0.0,
                "i_y": 0.0,
                "d_x": 0.0,
                "d_y": 0.0,
                "fov_counts_x": err_cx,
                "fov_counts_y": err_cy,
                "c360": self.counts_per_revolution,
                "fov_deg": self.fov_deg,
                "final_dx": dx,
                "final_dy": dy,
            },
        )

    def _frame_dt(self, now_s: float) -> float:
        dt_s = now_s - self._last_s if self._last_s is not None else self.FRAME_DT_FALLBACK
        self._last_s = now_s
        if not (self.FRAME_DT_MIN < dt_s < self.FRAME_DT_MAX):
            self._has_previous_sample = False
            return self.FRAME_DT_FALLBACK
        return dt_s

    def _is_acquire(self, dx_px: float, dy_px: float) -> bool:
        if self._last_error_px is None:
            self._has_previous_sample = False
            return True
        jumped = math.hypot(dx_px - self._last_error_px[0], dy_px - self._last_error_px[1])
        if jumped > self.jump_threshold_px:
            self._has_previous_sample = False
            return True
        return False

    def _project_pixels_to_counts(self, error_pixels_x: float, error_pixels_y: float, frame_width: float) -> dict[str, float]:
        focal = (frame_width * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5) if frame_width > 0 else 0.0
        counts_per_degree = self.counts_per_revolution / 360.0
        yaw_deg = math.degrees(math.atan(error_pixels_x / focal)) if focal > 0 else 0.0
        pitch_deg = math.degrees(math.atan(error_pixels_y / focal)) if focal > 0 else 0.0
        return {
            "focal": focal,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "counts_per_degree": counts_per_degree,
            "err_cx": yaw_deg * counts_per_degree,
            "err_cy": pitch_deg * counts_per_degree,
        }

    def _estimate_velocity_counts(self, error_counts_x: float, error_counts_y: float, dt_s: float) -> tuple[float, float]:
        if not self._has_previous_sample:
            self._last_error_counts = (error_counts_x, error_counts_y)
            self._has_previous_sample = True
            return 0.0, 0.0
        dt = max(self.PREDICT_DT_MIN, min(self.PREDICT_DT_MAX, dt_s))
        velocity_x = self._clamp(
            (error_counts_x - self._last_error_counts[0] + self._last_output_counts[0]) / dt,
            self.VELOCITY_CLAMP_COUNTS_PER_SEC,
        )
        velocity_y = self._clamp(
            (error_counts_y - self._last_error_counts[1] + self._last_output_counts[1]) / dt,
            self.VELOCITY_CLAMP_COUNTS_PER_SEC,
        )
        if abs(error_counts_x) > self.OVERSHOOT_GUARD_COUNTS and velocity_x * error_counts_x < 0.0:
            velocity_x *= self.OVERSHOOT_VELOCITY_DAMP
        if abs(error_counts_y) > self.OVERSHOOT_GUARD_COUNTS and velocity_y * error_counts_y < 0.0:
            velocity_y *= self.OVERSHOOT_VELOCITY_DAMP
        self._last_error_counts = (error_counts_x, error_counts_y)
        return velocity_x, velocity_y

    def _motion_coef(self, velocity_x: float, velocity_y: float) -> tuple[float, float]:
        self._consistency_x, self._prev_vel_sign_x, coef_x = self._axis_motion_coef(
            velocity_x,
            self._consistency_x,
            self._prev_vel_sign_x,
        )
        self._consistency_y, self._prev_vel_sign_y, coef_y = self._axis_motion_coef(
            velocity_y,
            self._consistency_y,
            self._prev_vel_sign_y,
        )
        return coef_x, coef_y

    def _axis_motion_coef(self, velocity: float, count: int, previous_sign: int) -> tuple[int, int, float]:
        if velocity > self.MOTION_FLOOR_COUNTS_PER_SEC:
            sign = 1
        elif velocity < -self.MOTION_FLOOR_COUNTS_PER_SEC:
            sign = -1
        else:
            sign = 0
        if sign == 0:
            count = 0
        elif sign == previous_sign:
            count += 1
        else:
            count = 1
        coef = max(0.0, min(1.0, count / float(self.pred_consistency_frames)))
        return count, sign, coef

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _was_sent(box_input: BoxInputState) -> bool:
        return str(box_input.raw.get("mode", "")) != "telemetry_control_calculation"

    def _reset(self) -> None:
        self._last_error_px = None
        self._last_error_counts = (0.0, 0.0)
        self._last_output_counts = (0.0, 0.0)
        self._last_s = None
        self._has_previous_sample = False
        self._consistency_x = 0
        self._consistency_y = 0
        self._prev_vel_sign_x = 0
        self._prev_vel_sign_y = 0

    def _reset_motion_state(self) -> None:
        self._reset()


class PredictiveStrategy:
    def __init__(self, *, lead_factor: float = 0.25, aim_ratio: float = 40.0) -> None:
        self.lead_factor = lead_factor
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self._last_center: tuple[float, float] | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            return MoveCommand(0, 0, 0, "hardware trigger inactive")
        center = aim_point(target, self.aim_ratio)
        if self._last_center is None:
            vx = vy = 0.0
        else:
            vx = center[0] - self._last_center[0]
            vy = center[1] - self._last_center[1]
        self._last_center = center
        predicted = (center[0] + vx * self.lead_factor, center[1] + vy * self.lead_factor)
        return MoveCommand(
            dx=predicted[0] - current_pos[0],
            dy=current_pos[1] - predicted[1],
            confidence=target.score,
            reason="predictive strategy",
            debug={
                "stage": "predictive",
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": predicted[0] - current_pos[0],
                "raw_px_y": current_pos[1] - predicted[1],
                "aim_ratio": self.aim_ratio,
                "aim_x": center[0],
                "aim_y": center[1],
                "final_dx": predicted[0] - current_pos[0],
                "final_dy": current_pos[1] - predicted[1],
            },
        )


class ProportionalStrategy:
    def __init__(
        self,
        *,
        fov_ratio: float = 0.28,
        aim_ratio: float = 40.0,
        near_px: float = 24.0,
        near_speed: float = 0.16,
        far_speed: float = 0.42,
        ema_alpha: float = 0.45,
        deadzone_counts: int = 1,
        counts_per_revolution_x: float = 4096.0,
        counts_per_revolution_y: float = 4096.0,
        move_kind: str = "raw",
        move_ms: int = 12,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.fov_ratio = max(0.0, min(1.0, fov_ratio))
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self.near_px = max(0.0, near_px)
        self.near_speed = max(0.0, near_speed)
        self.far_speed = max(0.0, far_speed)
        self.ema_alpha = max(0.0, min(1.0, ema_alpha))
        self.deadzone_counts = max(0, int(deadzone_counts))
        self.counts_per_revolution_x = max(1.0, counts_per_revolution_x)
        self.counts_per_revolution_y = max(1.0, counts_per_revolution_y)
        self.move_kind = move_kind if move_kind in {"raw", "auto", "bezier"} else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._ema_x = 0.0
        self._ema_y = 0.0

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            self._ema_x = 0.0
            self._ema_y = 0.0
            return MoveCommand(0, 0, 0, "hardware trigger inactive")

        aim_x, aim_y = aim_point(target, self.aim_ratio)
        ex = aim_x - current_pos[0]
        ey = current_pos[1] - aim_y
        err = math.hypot(ex, ey)
        radius = min(current_pos[0], current_pos[1]) * 2 * self.fov_ratio
        if radius > 0 and err > radius:
            self._ema_x = 0.0
            self._ema_y = 0.0
            return MoveCommand(
                0,
                0,
                target.score,
                "target outside proportional fov",
                debug={
                    "stage": "proportional",
                    "coordinate_y": "cartesian_up_positive",
                    "raw_px_x": ex,
                    "raw_px_y": ey,
                    "aim_ratio": self.aim_ratio,
                    "aim_x": aim_x,
                    "aim_y": aim_y,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )

        speed = self._speed_for(err, radius)
        rx = self.counts_per_revolution_x / (2 * math.pi)
        ry = self.counts_per_revolution_y / (2 * math.pi)
        counts_x = rx * math.atan((ex * speed) / rx)
        counts_y = ry * math.atan((ey * speed) / ry)
        self._ema_x = self.ema_alpha * counts_x + (1 - self.ema_alpha) * self._ema_x
        self._ema_y = self.ema_alpha * counts_y + (1 - self.ema_alpha) * self._ema_y
        dx = int(round(self._ema_x))
        dy = int(round(self._ema_y))
        if abs(dx) < self.deadzone_counts and abs(dy) < self.deadzone_counts:
            return MoveCommand(
                0,
                0,
                target.score,
                "proportional deadzone",
                debug={
                    "stage": "proportional",
                    "coordinate_y": "cartesian_up_positive",
                    "raw_px_x": ex,
                    "raw_px_y": ey,
                    "aim_ratio": self.aim_ratio,
                    "aim_x": aim_x,
                    "aim_y": aim_y,
                    "fov_counts_x": counts_x,
                    "fov_counts_y": counts_y,
                    "speed": speed,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )

        bezier_ctrl = _bezier_ctrl(dx, dy, self.bezier_curvature) if self.move_kind == "bezier" else None
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason="proportional strategy",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                "stage": "proportional",
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": ex,
                "raw_px_y": ey,
                "aim_ratio": self.aim_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "fov_counts_x": counts_x,
                "fov_counts_y": counts_y,
                "speed": speed,
                "final_dx": dx,
                "final_dy": dy,
            },
        )

    def _speed_for(self, err_px: float, fov_radius: float) -> float:
        if err_px <= self.near_px:
            return self.near_speed
        ref = fov_radius if fov_radius > 0 else max(self.near_px * 5.0, self.near_px + 1.0)
        span = max(1.0, ref - self.near_px)
        ratio = min(1.0, (err_px - self.near_px) / span)
        return self.near_speed + (self.far_speed - self.near_speed) * ratio


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


class ControlCommandCoalescer:
    def __init__(self, *, min_interval_s: float = 0.001) -> None:
        self.min_interval_s = min_interval_s
        self._last_emit_s = 0.0
        self._pending_dx = 0.0
        self._pending_dy = 0.0

    def push(self, intent: ControlIntent, now_s: float | None = None) -> ControlIntent | None:
        now = time.monotonic() if now_s is None else now_s
        self._pending_dx += intent.dx
        self._pending_dy += intent.dy
        if now - self._last_emit_s < self.min_interval_s:
            return None
        merged = ControlIntent(
            dx=self._pending_dx,
            dy=self._pending_dy,
            action=intent.action,
            confidence=intent.confidence,
            reason="coalesced control command",
            source_id=intent.source_id,
            move_kind=intent.move_kind,
            move_ms=intent.move_ms,
            trace_ms=intent.trace_ms,
            bezier_ctrl=intent.bezier_ctrl,
        )
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_emit_s = now
        return merged
