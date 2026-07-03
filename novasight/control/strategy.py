from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from novasight.hardware import BoxInputState
from novasight.contracts import ControlIntent, Detection, Track


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
        integral_limit: float = 250.0,
        move_limit: float | None = None,
        derivative_alpha: float = 0.35,
    ) -> None:
        self.kp_x = kp if kp_x is None else kp_x
        self.kp_y = kp if kp_y is None else kp_y
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.move_limit = None if move_limit is None else abs(move_limit)
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
        self._ix = max(-self.integral_limit, min(self.integral_limit, self._ix + ex))
        self._iy = max(-self.integral_limit, min(self.integral_limit, self._iy + ey))
        if self._last_error is None:
            raw_dx = raw_dy = 0.0
        else:
            raw_dx = ex - self._last_error[0]
            raw_dy = ey - self._last_error[1]
        dx_d = self.derivative_alpha * raw_dx + (1 - self.derivative_alpha) * self._last_derivative[0]
        dy_d = self.derivative_alpha * raw_dy + (1 - self.derivative_alpha) * self._last_derivative[1]
        self._last_error = (ex, ey)
        self._last_derivative = (dx_d, dy_d)
        dx = self.kp_x * ex + self.ki * self._ix + self.kd * dx_d
        dy = self.kp_y * ey + self.ki * self._iy + self.kd * dy_d
        return MoveCommand(
            dx=self._clamp_axis(dx, self.move_limit),
            dy=self._clamp_axis(dy, self.move_limit),
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
            source_id=intent.source_id,
        )
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_emit_s = now
        return merged
