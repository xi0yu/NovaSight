from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


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


@dataclass(frozen=True, slots=True)
class MouseObservation:
    frame_id: int
    target_id: int
    capture_ts_ns: int
    control_now_ts_ns: int
    measurement_dt_s: float | None
    control_width_px: float
    control_height_px: float
    observed_x_px: float
    observed_y_px: float
    predicted_x_px: float
    predicted_y_px: float
    prediction_horizon_s: float
    target_confidence: float
    prediction_confidence: float
    observed_valid: bool = True
    valid: bool = True
    invalid_reason: str = ""


@dataclass(frozen=True, slots=True)
class MouseControllerConfig:
    fov_x_deg: float
    counts_per_360_x: float
    counts_per_360_y: float
    invert_y: bool
    kp_x: float
    kp_y: float
    kd_x: float
    kd_y: float
    d_ema_alpha: float
    deadzone_px_x: float
    deadzone_px_y: float
    max_output_rad_x: float
    max_output_rad_y: float
    max_output_rate_rad_s_x: float
    max_output_rate_rad_s_y: float
    max_budget_counts_x: int
    max_budget_counts_y: int


@dataclass(slots=True)
class MouseControllerState:
    target_id: int | None = None
    frame_id: int | None = None
    capture_ts_ns: int | None = None
    observed_error_x_rad: float = 0.0
    observed_error_y_rad: float = 0.0
    observed_history_valid: bool = False
    d_ema_x_rad_s: float = 0.0
    d_ema_y_rad_s: float = 0.0
    output_x_rad: float = 0.0
    output_y_rad: float = 0.0
    residual_x_counts: float = 0.0
    residual_y_counts: float = 0.0

    def reset(self) -> None:
        self.target_id = None
        self.frame_id = None
        self.capture_ts_ns = None
        self.observed_error_x_rad = 0.0
        self.observed_error_y_rad = 0.0
        self.observed_history_valid = False
        self.d_ema_x_rad_s = 0.0
        self.d_ema_y_rad_s = 0.0
        self.output_x_rad = 0.0
        self.output_y_rad = 0.0
        self.residual_x_counts = 0.0
        self.residual_y_counts = 0.0


class MouseController:
    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        self.state = MouseControllerState()

    def reset(self) -> None:
        self.state.reset()

    def calculate(self, observation: MouseObservation) -> MoveCommand:
        if not isinstance(observation, MouseObservation):
            self.reset()
            return self._zero("MOUSE_OBSERVATION_REQUIRED")
        if not observation.valid:
            self.reset()
            return self._zero(observation.invalid_reason or "MOUSE_OBSERVATION_INVALID")
        if self.state.frame_id == observation.frame_id:
            return self._zero("DUPLICATE_OBSERVATION")

        geometry = _projection_geometry(
            observation.control_width_px,
            observation.control_height_px,
            self.config.fov_x_deg,
        )
        if geometry is None:
            self.reset()
            return self._zero("CONTROL_PROJECTION_INVALID")
        center_x, center_y, focal_x, focal_y = geometry
        observed_error_x_px = observation.observed_x_px - center_x
        observed_error_y_px = observation.observed_y_px - center_y
        predicted_error_x_px = observation.predicted_x_px - center_x
        predicted_error_y_px = observation.predicted_y_px - center_y
        observed_error_x_rad = math.atan(observed_error_x_px / focal_x)
        observed_error_y_rad = math.atan(observed_error_y_px / focal_y)
        predicted_error_x_rad = math.atan(predicted_error_x_px / focal_x)
        predicted_error_y_rad = math.atan(predicted_error_y_px / focal_y)

        switched = self.state.target_id is not None and self.state.target_id != observation.target_id
        if switched:
            self.state.reset()
        dt = observation.measurement_dt_s
        dt_valid = dt is not None and math.isfinite(dt) and 0.0 < dt <= 0.2
        if (
            switched
            or self.state.target_id is None
            or not self.state.observed_history_valid
            or not observation.observed_valid
            or not dt_valid
        ):
            d_raw_x = 0.0
            d_raw_y = 0.0
            self.state.d_ema_x_rad_s = 0.0
            self.state.d_ema_y_rad_s = 0.0
        else:
            assert dt is not None
            d_raw_x = (observed_error_x_rad - self.state.observed_error_x_rad) / dt
            d_raw_y = (observed_error_y_rad - self.state.observed_error_y_rad) / dt
            alpha = _clamp(self.config.d_ema_alpha, 0.01, 1.0)
            self.state.d_ema_x_rad_s = alpha * d_raw_x + (1.0 - alpha) * self.state.d_ema_x_rad_s
            self.state.d_ema_y_rad_s = alpha * d_raw_y + (1.0 - alpha) * self.state.d_ema_y_rad_s

        x_dead = abs(observed_error_x_px) <= max(0.0, self.config.deadzone_px_x)
        y_dead = abs(observed_error_y_px) <= max(0.0, self.config.deadzone_px_y)
        p_x_rad = self.config.kp_x * predicted_error_x_rad
        p_y_rad = self.config.kp_y * predicted_error_y_rad
        d_x_rad = self.config.kd_x * self.state.d_ema_x_rad_s
        d_y_rad = self.config.kd_y * self.state.d_ema_y_rad_s
        requested_x_rad = 0.0 if x_dead else p_x_rad + d_x_rad
        requested_y_rad = 0.0 if y_dead else p_y_rad + d_y_rad
        limited_x_rad = _clamp_axis(requested_x_rad, self.config.max_output_rad_x)
        limited_y_rad = _clamp_axis(requested_y_rad, self.config.max_output_rad_y)

        if self.state.target_id == observation.target_id and dt_valid:
            assert dt is not None
            limited_x_rad = _rate_limit(
                limited_x_rad,
                self.state.output_x_rad,
                self.config.max_output_rate_rad_s_x,
                dt,
            )
            limited_y_rad = _rate_limit(
                limited_y_rad,
                self.state.output_y_rad,
                self.config.max_output_rate_rad_s_y,
                dt,
            )

        counts_x_float = limited_x_rad * self.config.counts_per_360_x / math.tau
        counts_y_float = limited_y_rad * self.config.counts_per_360_y / math.tau
        if self.config.invert_y:
            counts_y_float = -counts_y_float
        total_x = counts_x_float + self.state.residual_x_counts
        total_y = counts_y_float + self.state.residual_y_counts
        requested_counts_x = math.trunc(total_x)
        requested_counts_y = math.trunc(total_y)
        self.state.residual_x_counts = total_x - requested_counts_x
        self.state.residual_y_counts = total_y - requested_counts_y
        counts_x = int(_clamp_axis(requested_counts_x, self.config.max_budget_counts_x))
        counts_y = int(_clamp_axis(requested_counts_y, self.config.max_budget_counts_y))

        self.state.target_id = observation.target_id
        self.state.frame_id = observation.frame_id
        self.state.capture_ts_ns = observation.capture_ts_ns
        if observation.observed_valid:
            self.state.observed_error_x_rad = observed_error_x_rad
            self.state.observed_error_y_rad = observed_error_y_rad
            self.state.observed_history_valid = True
        else:
            self.state.observed_history_valid = False
        self.state.output_x_rad = limited_x_rad
        self.state.output_y_rad = limited_y_rad

        return MoveCommand(
            dx=counts_x,
            dy=counts_y,
            confidence=observation.target_confidence,
            reason="mouse_control",
            debug={
                "algorithm": "mouse_control",
                "control_allowed": True,
                "frame_id": observation.frame_id,
                "target_id": observation.target_id,
                "measurement_dt_s": dt,
                "prediction_horizon_s": observation.prediction_horizon_s,
                "target_confidence": observation.target_confidence,
                "prediction_confidence": observation.prediction_confidence,
                "observed_x_px": observation.observed_x_px,
                "observed_y_px": observation.observed_y_px,
                "predicted_x_px": observation.predicted_x_px,
                "predicted_y_px": observation.predicted_y_px,
                "observed_error_x_px": observed_error_x_px,
                "observed_error_y_px": observed_error_y_px,
                "observed_error_x_rad": observed_error_x_rad,
                "observed_error_y_rad": observed_error_y_rad,
                "predicted_error_x_rad": predicted_error_x_rad,
                "predicted_error_y_rad": predicted_error_y_rad,
                "focal_x_px": focal_x,
                "focal_y_px": focal_y,
                "p_x_rad": p_x_rad,
                "p_y_rad": p_y_rad,
                "d_x_rad": d_x_rad,
                "d_y_rad": d_y_rad,
                "d_raw_x_rad_s": d_raw_x,
                "d_raw_y_rad_s": d_raw_y,
                "d_ema_x_rad_s": self.state.d_ema_x_rad_s,
                "d_ema_y_rad_s": self.state.d_ema_y_rad_s,
                "requested_output_x_rad": requested_x_rad,
                "requested_output_y_rad": requested_y_rad,
                "limited_output_x_rad": limited_x_rad,
                "limited_output_y_rad": limited_y_rad,
                "counts_x_float": counts_x_float,
                "counts_y_float": counts_y_float,
                "requested_counts_x": requested_counts_x,
                "requested_counts_y": requested_counts_y,
                "budget_clamped_x": counts_x != requested_counts_x,
                "budget_clamped_y": counts_y != requested_counts_y,
                "residual_x_counts": self.state.residual_x_counts,
                "residual_y_counts": self.state.residual_y_counts,
                "final_dx": counts_x,
                "final_dy": counts_y,
            },
        )

    @staticmethod
    def _zero(reason: str) -> MoveCommand:
        return MoveCommand(
            dx=0,
            dy=0,
            confidence=0.0,
            reason=reason,
            debug={
                "algorithm": "mouse_control",
                "control_allowed": False,
                "final_dx": 0,
                "final_dy": 0,
                "invalid_reason": reason,
            },
        )


def _projection_geometry(
    width: float,
    height: float,
    fov_x_deg: float,
) -> tuple[float, float, float, float] | None:
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
        return None
    if not math.isfinite(fov_x_deg) or not 0.0 < fov_x_deg < 180.0:
        return None
    fov_x_rad = math.radians(fov_x_deg)
    focal = width / (2.0 * math.tan(fov_x_rad * 0.5))
    if not math.isfinite(focal) or focal <= 0.0:
        return None
    return width * 0.5, height * 0.5, focal, focal


def _rate_limit(requested: float, previous: float, rate_rad_s: float, dt_s: float) -> float:
    max_delta = max(0.0, float(rate_rad_s)) * max(0.0, float(dt_s))
    return previous + _clamp(requested - previous, -max_delta, max_delta)


def _clamp_axis(value: float | int, limit: float | int) -> float:
    bound = max(0.0, float(limit))
    return _clamp(float(value), -bound, bound)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
