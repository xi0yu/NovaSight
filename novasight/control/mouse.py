from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol


CALIBRATED_ANGULAR = "calibrated_angular"
UNIVERSAL_SATURATED = "universal_saturated"
CONTROL_MODES = frozenset({CALIBRATED_ANGULAR, UNIVERSAL_SATURATED})


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
class Vec2:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ControllerInput:
    predicted_error_px: Vec2
    observed_error_px: Vec2
    control_size_px: Vec2
    dt_s: float | None
    observed_history_valid: bool
    observed_valid: bool


@dataclass(frozen=True, slots=True)
class ControllerComputation:
    counts: Vec2
    debug: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CalibratedAngularControllerConfig:
    fov_x_deg: float
    counts_per_360_x: float
    counts_per_360_y: float
    kp_x: float
    kp_y: float
    kd_x: float
    kd_y: float
    d_ema_alpha: float
    max_angle_step_x_rad: float
    max_angle_step_y_rad: float


@dataclass(frozen=True, slots=True)
class UniversalSaturatedControllerConfig:
    response_scale_x_px: float
    response_scale_y_px: float
    max_step_x_counts: float
    max_step_y_counts: float


@dataclass(frozen=True, slots=True)
class SharedOutputConfig:
    deadzone_x_px: float
    deadzone_y_px: float
    max_count_slew_x: float
    max_count_slew_y: float
    invert_y: bool
    max_budget_counts_x: int
    max_budget_counts_y: int
    min_effective_counts_x: int
    min_effective_counts_y: int


@dataclass(frozen=True, slots=True)
class MouseControllerConfig:
    mode: str
    calibrated_angular: CalibratedAngularControllerConfig
    universal_saturated: UniversalSaturatedControllerConfig
    shared: SharedOutputConfig


class ControlController(Protocol):
    mode: str

    def reset(self) -> None:
        ...

    def compute_counts(self, value: ControllerInput) -> ControllerComputation:
        ...


@dataclass(slots=True)
class _CalibratedAngularState:
    previous_observed_error_rad: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    derivative_ema_rad_s: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    history_valid: bool = False

    def reset(self) -> None:
        self.previous_observed_error_rad = Vec2(0.0, 0.0)
        self.derivative_ema_rad_s = Vec2(0.0, 0.0)
        self.history_valid = False


class CalibratedAngularController:
    mode = CALIBRATED_ANGULAR

    def __init__(self, config: CalibratedAngularControllerConfig) -> None:
        self.config = config
        self.state = _CalibratedAngularState()

    def reset(self) -> None:
        self.state.reset()

    def compute_counts(self, value: ControllerInput) -> ControllerComputation:
        geometry = _projection_geometry(
            value.control_size_px.x,
            value.control_size_px.y,
            self.config.fov_x_deg,
        )
        if geometry is None:
            raise ValueError("CONTROL_PROJECTION_INVALID")
        focal_x, focal_y = geometry
        observed_rad = Vec2(
            math.atan(value.observed_error_px.x / focal_x),
            math.atan(value.observed_error_px.y / focal_y),
        )
        predicted_rad = Vec2(
            math.atan(value.predicted_error_px.x / focal_x),
            math.atan(value.predicted_error_px.y / focal_y),
        )
        dt_valid = _valid_measurement_dt(value.dt_s)
        if (
            not self.state.history_valid
            or not value.observed_history_valid
            or not value.observed_valid
            or not dt_valid
        ):
            d_raw = Vec2(0.0, 0.0)
            self.state.derivative_ema_rad_s = Vec2(0.0, 0.0)
        else:
            assert value.dt_s is not None
            d_raw = Vec2(
                (observed_rad.x - self.state.previous_observed_error_rad.x) / value.dt_s,
                (observed_rad.y - self.state.previous_observed_error_rad.y) / value.dt_s,
            )
            alpha = _clamp(self.config.d_ema_alpha, 0.01, 1.0)
            self.state.derivative_ema_rad_s = Vec2(
                alpha * d_raw.x + (1.0 - alpha) * self.state.derivative_ema_rad_s.x,
                alpha * d_raw.y + (1.0 - alpha) * self.state.derivative_ema_rad_s.y,
            )

        p_rad = Vec2(
            self.config.kp_x * predicted_rad.x,
            self.config.kp_y * predicted_rad.y,
        )
        d_rad = Vec2(
            self.config.kd_x * self.state.derivative_ema_rad_s.x,
            self.config.kd_y * self.state.derivative_ema_rad_s.y,
        )
        requested_rad = Vec2(p_rad.x + d_rad.x, p_rad.y + d_rad.y)
        limited_rad = Vec2(
            _clamp_axis(requested_rad.x, self.config.max_angle_step_x_rad),
            _clamp_axis(requested_rad.y, self.config.max_angle_step_y_rad),
        )
        theoretical_counts = Vec2(
            requested_rad.x * self.config.counts_per_360_x / math.tau,
            requested_rad.y * self.config.counts_per_360_y / math.tau,
        )
        limited_counts = Vec2(
            limited_rad.x * self.config.counts_per_360_x / math.tau,
            limited_rad.y * self.config.counts_per_360_y / math.tau,
        )

        if value.observed_valid:
            self.state.previous_observed_error_rad = observed_rad
            self.state.history_valid = True
        else:
            self.state.history_valid = False

        return ControllerComputation(
            counts=limited_counts,
            debug={
                "focal_x_px": focal_x,
                "focal_y_px": focal_y,
                "observed_error_x_rad": observed_rad.x,
                "observed_error_y_rad": observed_rad.y,
                "predicted_error_x_rad": predicted_rad.x,
                "predicted_error_y_rad": predicted_rad.y,
                "p_x_rad": p_rad.x,
                "p_y_rad": p_rad.y,
                "d_x_rad": d_rad.x,
                "d_y_rad": d_rad.y,
                "d_raw_x_rad_s": d_raw.x,
                "d_raw_y_rad_s": d_raw.y,
                "d_ema_x_rad_s": self.state.derivative_ema_rad_s.x,
                "d_ema_y_rad_s": self.state.derivative_ema_rad_s.y,
                "requested_output_x_rad": requested_rad.x,
                "requested_output_y_rad": requested_rad.y,
                "limited_output_x_rad": limited_rad.x,
                "limited_output_y_rad": limited_rad.y,
                "theoretical_counts_x_float": theoretical_counts.x,
                "theoretical_counts_y_float": theoretical_counts.y,
                "mode_limited_counts_x_float": limited_counts.x,
                "mode_limited_counts_y_float": limited_counts.y,
            },
        )


class UniversalSaturatedController:
    mode = UNIVERSAL_SATURATED

    def __init__(self, config: UniversalSaturatedControllerConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return None

    def compute_counts(self, value: ControllerInput) -> ControllerComputation:
        counts = Vec2(
            _saturated_axis(
                value.predicted_error_px.x,
                self.config.response_scale_x_px,
                self.config.max_step_x_counts,
            ),
            _saturated_axis(
                value.predicted_error_px.y,
                self.config.response_scale_y_px,
                self.config.max_step_y_counts,
            ),
        )
        return ControllerComputation(
            counts=counts,
            debug={
                "response_scale_x_px": self.config.response_scale_x_px,
                "response_scale_y_px": self.config.response_scale_y_px,
                "max_step_x_counts": self.config.max_step_x_counts,
                "max_step_y_counts": self.config.max_step_y_counts,
                "theoretical_counts_x_float": counts.x,
                "theoretical_counts_y_float": counts.y,
                "mode_limited_counts_x_float": counts.x,
                "mode_limited_counts_y_float": counts.y,
            },
        )


class ControllerFactory:
    @staticmethod
    def create(config: MouseControllerConfig) -> ControlController:
        if config.mode == CALIBRATED_ANGULAR:
            return CalibratedAngularController(config.calibrated_angular)
        if config.mode == UNIVERSAL_SATURATED:
            return UniversalSaturatedController(config.universal_saturated)
        raise ValueError(f"unsupported mouse control mode: {config.mode}")


@dataclass(slots=True)
class MouseControllerState:
    target_id: int | None = None
    frame_id: int | None = None
    residual_x_counts: float = 0.0
    residual_y_counts: float = 0.0
    previous_counts: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    output_history_valid: bool = False

    def reset(self) -> None:
        self.target_id = None
        self.frame_id = None
        self.residual_x_counts = 0.0
        self.residual_y_counts = 0.0
        self.previous_counts = Vec2(0.0, 0.0)
        self.output_history_valid = False


class MouseController:
    """Single production entry that owns exactly one mode controller."""

    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        self.controller = ControllerFactory.create(config)
        self.state = MouseControllerState()

    @property
    def mode(self) -> str:
        return self.controller.mode

    def reset(self) -> None:
        self.controller.reset()
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
        if not _valid_control_space(observation.control_width_px, observation.control_height_px):
            self.reset()
            return self._zero("CONTROL_PROJECTION_INVALID")

        switched = self.state.target_id is not None and self.state.target_id != observation.target_id
        if switched:
            self.reset()

        center_x = observation.control_width_px * 0.5
        center_y = observation.control_height_px * 0.5
        observed_error = Vec2(
            observation.observed_x_px - center_x,
            observation.observed_y_px - center_y,
        )
        predicted_error = Vec2(
            observation.predicted_x_px - center_x,
            observation.predicted_y_px - center_y,
        )
        try:
            computation = self.controller.compute_counts(
                ControllerInput(
                    predicted_error_px=predicted_error,
                    observed_error_px=observed_error,
                    control_size_px=Vec2(
                        observation.control_width_px,
                        observation.control_height_px,
                    ),
                    dt_s=observation.measurement_dt_s,
                    observed_history_valid=not switched and self.state.target_id is not None,
                    observed_valid=observation.observed_valid,
                )
            )
        except ValueError as exc:
            self.reset()
            return self._zero(str(exc))

        shared = self.config.shared
        deadzone_limited = Vec2(
            0.0 if abs(observed_error.x) <= shared.deadzone_x_px else computation.counts.x,
            0.0 if abs(observed_error.y) <= shared.deadzone_y_px else computation.counts.y,
        )
        directed_counts = Vec2(
            deadzone_limited.x,
            -deadzone_limited.y if shared.invert_y else deadzone_limited.y,
        )
        slew_limited = directed_counts
        if self.state.output_history_valid:
            slew_limited = Vec2(
                _slew_limit(directed_counts.x, self.state.previous_counts.x, shared.max_count_slew_x),
                _slew_limit(directed_counts.y, self.state.previous_counts.y, shared.max_count_slew_y),
            )
        feasible_counts = Vec2(
            _clamp_axis(slew_limited.x, shared.max_budget_counts_x),
            _clamp_axis(slew_limited.y, shared.max_budget_counts_y),
        )

        counts_x, residual_x = _quantize_effective_counts(
            feasible_counts.x,
            self.state.residual_x_counts,
            shared.min_effective_counts_x,
        )
        counts_y, residual_y = _quantize_effective_counts(
            feasible_counts.y,
            self.state.residual_y_counts,
            shared.min_effective_counts_y,
        )
        self.state.residual_x_counts = residual_x
        self.state.residual_y_counts = residual_y
        self.state.target_id = observation.target_id
        self.state.frame_id = observation.frame_id
        self.state.previous_counts = feasible_counts
        self.state.output_history_valid = True

        debug = {
            "algorithm": self.mode,
            "control_mode": self.mode,
            "control_allowed": True,
            "frame_id": observation.frame_id,
            "target_id": observation.target_id,
            "measurement_dt_s": observation.measurement_dt_s,
            "prediction_horizon_s": observation.prediction_horizon_s,
            "target_confidence": observation.target_confidence,
            "prediction_confidence": observation.prediction_confidence,
            "observed_x_px": observation.observed_x_px,
            "observed_y_px": observation.observed_y_px,
            "predicted_x_px": observation.predicted_x_px,
            "predicted_y_px": observation.predicted_y_px,
            "observed_error_x_px": observed_error.x,
            "observed_error_y_px": observed_error.y,
            "predicted_error_x_px": predicted_error.x,
            "predicted_error_y_px": predicted_error.y,
            **computation.debug,
            "deadzone_limited_counts_x_float": deadzone_limited.x,
            "deadzone_limited_counts_y_float": deadzone_limited.y,
            "directed_counts_x_float": directed_counts.x,
            "directed_counts_y_float": directed_counts.y,
            "slew_limited_counts_x_float": slew_limited.x,
            "slew_limited_counts_y_float": slew_limited.y,
            "feasible_counts_x_float": feasible_counts.x,
            "feasible_counts_y_float": feasible_counts.y,
            "budget_clamped_x": feasible_counts.x != slew_limited.x,
            "budget_clamped_y": feasible_counts.y != slew_limited.y,
            "residual_x_counts": self.state.residual_x_counts,
            "residual_y_counts": self.state.residual_y_counts,
            "min_effective_counts_x": shared.min_effective_counts_x,
            "min_effective_counts_y": shared.min_effective_counts_y,
            "count_quantization": "minimum_effective_accumulator",
            "final_dx": counts_x,
            "final_dy": counts_y,
        }
        return MoveCommand(
            dx=counts_x,
            dy=counts_y,
            confidence=observation.target_confidence,
            reason="mouse_control",
            debug=debug,
        )

    def _zero(self, reason: str) -> MoveCommand:
        return MoveCommand(
            dx=0,
            dy=0,
            confidence=0.0,
            reason=reason,
            debug={
                "algorithm": self.mode,
                "control_mode": self.mode,
                "control_allowed": False,
                "final_dx": 0,
                "final_dy": 0,
                "invalid_reason": reason,
            },
        )


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _quantize_effective_counts(
    requested: float,
    residual: float,
    minimum_effective_counts: int,
) -> tuple[int, float]:
    if not math.isfinite(requested) or not math.isfinite(residual):
        return 0, 0.0
    if requested == 0.0:
        return 0, 0.0
    minimum = max(1, int(minimum_effective_counts))
    total = requested + residual
    quantized = _round_half_away_from_zero(total)
    if minimum == 1:
        return quantized, total - quantized
    if abs(quantized) < minimum:
        return 0, total
    requested_quantized = _round_half_away_from_zero(requested)
    if abs(requested_quantized) < minimum:
        emitted = minimum if total > 0.0 else -minimum
        return emitted, total - emitted
    return quantized, total - quantized


def _projection_geometry(width: float, height: float, fov_x_deg: float) -> tuple[float, float] | None:
    if not _valid_control_space(width, height):
        return None
    if not math.isfinite(fov_x_deg) or not 0.0 < fov_x_deg < 180.0:
        return None
    fov_x_rad = math.radians(fov_x_deg)
    focal_x = width / (2.0 * math.tan(fov_x_rad * 0.5))
    fov_y_rad = 2.0 * math.atan(math.tan(fov_x_rad * 0.5) * height / width)
    focal_y = height / (2.0 * math.tan(fov_y_rad * 0.5))
    if not all(math.isfinite(value) and value > 0.0 for value in (focal_x, focal_y)):
        return None
    return focal_x, focal_y


def _valid_control_space(width: float, height: float) -> bool:
    return all(math.isfinite(value) and value > 0.0 for value in (width, height))


def _valid_measurement_dt(dt_s: float | None) -> bool:
    return dt_s is not None and math.isfinite(dt_s) and 0.0 < dt_s <= 0.2


def _saturated_axis(error_px: float, response_scale_px: float, max_counts: float) -> float:
    return max_counts * (2.0 / math.pi) * math.atan(error_px / response_scale_px)


def _slew_limit(requested: float, previous: float, max_slew: float) -> float:
    limit = max(0.0, float(max_slew))
    return previous + _clamp(requested - previous, -limit, limit)


def _clamp_axis(value: float | int, limit: float | int) -> float:
    bound = max(0.0, float(limit))
    return _clamp(float(value), -bound, bound)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
