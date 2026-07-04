from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from novasight.hardware import BoxInputState
from novasight.contracts import ControlIntent, Detection, Track


Target = Detection | Track


def _aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, aim_ratio)) / 100.0
    return (float(target.x) + float(target.w) / 2.0, float(target.y) + float(target.h) * ratio)


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
        aim_center = _aim_point(target, self.aim_ratio)
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
        center = _aim_point(target, self.aim_ratio)
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

        aim_x, aim_y = _aim_point(target, self.aim_ratio)
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
        )
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_emit_s = now
        return merged
