from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from novasight.hardware import BoxInputState
from novasight.plugins import ControlIntent, Detection, Track


Target = Detection | Track


@dataclass(frozen=True)
class MoveCommand:
    dx: float
    dy: float
    confidence: float
    reason: str


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
        ki_x: float | None = None,
        ki_y: float | None = None,
        kd_x: float | None = None,
        kd_y: float | None = None,
        integral_limit: float = 250.0,
        integral_limit_x: float | None = None,
        integral_limit_y: float | None = None,
        output_limit_x: float | None = None,
        output_limit_y: float | None = None,
        derivative_alpha: float = 0.35,
    ) -> None:
        self.kp_x = kp if kp_x is None else kp_x
        self.kp_y = kp if kp_y is None else kp_y
        self.ki_x = ki if ki_x is None else ki_x
        self.ki_y = ki if ki_y is None else ki_y
        self.kd_x = kd if kd_x is None else kd_x
        self.kd_y = kd if kd_y is None else kd_y
        self.integral_limit_x = abs(integral_limit if integral_limit_x is None else integral_limit_x)
        self.integral_limit_y = abs(integral_limit if integral_limit_y is None else integral_limit_y)
        self.output_limit_x = None if output_limit_x is None else abs(output_limit_x)
        self.output_limit_y = None if output_limit_y is None else abs(output_limit_y)
        self.derivative_alpha = max(0.0, min(1.0, derivative_alpha))
        self._ix = 0.0
        self._iy = 0.0
        self._last_error: tuple[float, float] | None = None
        self._last_derivative = (0.0, 0.0)

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            return MoveCommand(0, 0, 0, "hardware trigger inactive")
        ex = target.cx - current_pos[0]
        ey = target.cy - current_pos[1]
        self._ix = max(-self.integral_limit_x, min(self.integral_limit_x, self._ix + ex))
        self._iy = max(-self.integral_limit_y, min(self.integral_limit_y, self._iy + ey))
        if self._last_error is None:
            raw_dx = raw_dy = 0.0
        else:
            raw_dx = ex - self._last_error[0]
            raw_dy = ey - self._last_error[1]
        dx_d = self.derivative_alpha * raw_dx + (1 - self.derivative_alpha) * self._last_derivative[0]
        dy_d = self.derivative_alpha * raw_dy + (1 - self.derivative_alpha) * self._last_derivative[1]
        self._last_error = (ex, ey)
        self._last_derivative = (dx_d, dy_d)
        dx = self.kp_x * ex + self.ki_x * self._ix + self.kd_x * dx_d
        dy = self.kp_y * ey + self.ki_y * self._iy + self.kd_y * dy_d
        return MoveCommand(
            dx=self._clamp_axis(dx, self.output_limit_x),
            dy=self._clamp_axis(dy, self.output_limit_y),
            confidence=target.score,
            reason="pid strategy",
        )

    @staticmethod
    def _clamp_axis(value: float, limit: float | None) -> float:
        if limit is None:
            return value
        return max(-limit, min(limit, value))


class PredictiveStrategy:
    def __init__(self, *, lead_factor: float = 0.25) -> None:
        self.lead_factor = lead_factor
        self._last_center: tuple[float, float] | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            return MoveCommand(0, 0, 0, "hardware trigger inactive")
        center = (target.cx, target.cy)
        if self._last_center is None:
            vx = vy = 0.0
        else:
            vx = center[0] - self._last_center[0]
            vy = center[1] - self._last_center[1]
        self._last_center = center
        predicted = (center[0] + vx * self.lead_factor, center[1] + vy * self.lead_factor)
        return MoveCommand(
            dx=predicted[0] - current_pos[0],
            dy=predicted[1] - current_pos[1],
            confidence=target.score,
            reason="predictive strategy",
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
            plugin_id=intent.plugin_id,
        )
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_emit_s = now
        return merged
